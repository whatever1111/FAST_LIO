// Tests for engulf_release.hpp: the released velocity decision and the additive covariance inflation.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <limits>

#include "engulf_release.hpp"

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kDeg = M_PI / 180.0;

Eigen::Matrix3d yawRotation(double yaw_rad)
{
  return Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

fast_lio::ReleaseVelocityParams legParams()
{
  fast_lio::ReleaseVelocityParams p;
  p.source = fast_lio::ReleaseVelocitySource::kLeg;
  p.leg_max_age_s = 0.5;
  p.leg_sigma = 0.3;
  p.keep_sigma = 0.5;
  p.zero_sigma = 0.0;
  p.max_speed = 3.0;
  return p;
}
}  // namespace

TEST(ReleaseVelocity, ZeroSourceReproducesTheLegacyDiscard)
{
  fast_lio::ReleaseVelocityParams p;  // defaults: kZero, zero_sigma 0
  const auto d = fast_lio::releaseVelocity(Eigen::Vector3d(0.7, 0.1, 0.0), Eigen::Matrix3d::Identity(),
                                           Eigen::Vector3d(0.8, 0.0, 0.0), 0.05, p);
  EXPECT_EQ(d.used, fast_lio::ReleaseVelocitySource::kZero);
  EXPECT_DOUBLE_EQ(d.velocity.norm(), 0.0);
  EXPECT_DOUBLE_EQ(d.sigma, 0.0);
}

TEST(ReleaseVelocity, ZeroSourceCanStillWriteAnHonestSigma)
{
  fast_lio::ReleaseVelocityParams p;
  p.zero_sigma = 0.5;
  const auto d = fast_lio::releaseVelocity(Eigen::Vector3d(0.7, 0.1, 0.0), Eigen::Matrix3d::Identity(),
                                           Eigen::Vector3d(0.8, 0.0, 0.0), 0.05, p);
  EXPECT_EQ(d.used, fast_lio::ReleaseVelocitySource::kZero);
  EXPECT_DOUBLE_EQ(d.sigma, 0.5);
}

TEST(ReleaseVelocity, FreshLegSampleIsRotatedIntoTheWorld)
{
  const Eigen::Matrix3d R = yawRotation(90.0 * kDeg);  // body x -> world y
  const auto d = fast_lio::releaseVelocity(Eigen::Vector3d(0.7, 0.1, 0.0), R, Eigen::Vector3d(0.8, 0.0, 0.0), 0.1,
                                           legParams());
  EXPECT_EQ(d.used, fast_lio::ReleaseVelocitySource::kLeg);
  EXPECT_NEAR(d.velocity.x(), 0.0, 1e-12);
  EXPECT_NEAR(d.velocity.y(), 0.8, 1e-12);
  EXPECT_NEAR(d.velocity.z(), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(d.sigma, 0.3);
}

TEST(ReleaseVelocity, StaleOrMissingOrNaNLegFallsBackToTheEstimate)
{
  const Eigen::Vector3d est(0.7, 0.1, 0.0);
  const auto stale = fast_lio::releaseVelocity(est, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.8, 0.0, 0.0), 0.9,
                                               legParams());
  EXPECT_EQ(stale.used, fast_lio::ReleaseVelocitySource::kKeep);
  EXPECT_TRUE(stale.velocity.isApprox(est));
  EXPECT_DOUBLE_EQ(stale.sigma, 0.5);
  const auto none = fast_lio::releaseVelocity(est, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), -1.0,
                                              legParams());
  EXPECT_EQ(none.used, fast_lio::ReleaseVelocitySource::kKeep);
  const auto nan = fast_lio::releaseVelocity(est, Eigen::Matrix3d::Identity(), Eigen::Vector3d(kNaN, 0.0, 0.0), 0.1,
                                             legParams());
  EXPECT_EQ(nan.used, fast_lio::ReleaseVelocitySource::kKeep);
  const auto nan_age = fast_lio::releaseVelocity(est, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.8, 0.0, 0.0),
                                                 kNaN, legParams());
  EXPECT_EQ(nan_age.used, fast_lio::ReleaseVelocitySource::kKeep);
}

TEST(ReleaseVelocity, RunawayOrNonFiniteEstimateIsDiscarded)
{
  fast_lio::ReleaseVelocityParams p = legParams();
  p.source = fast_lio::ReleaseVelocitySource::kKeep;
  const auto runaway = fast_lio::releaseVelocity(Eigen::Vector3d(4.0, 0.0, 0.0), Eigen::Matrix3d::Identity(),
                                                 Eigen::Vector3d::Zero(), -1.0, p);
  EXPECT_EQ(runaway.used, fast_lio::ReleaseVelocitySource::kZero);
  EXPECT_DOUBLE_EQ(runaway.velocity.norm(), 0.0);
  const auto inf = fast_lio::releaseVelocity(Eigen::Vector3d(kInf, 0.0, 0.0), Eigen::Matrix3d::Identity(),
                                             Eigen::Vector3d::Zero(), -1.0, p);
  EXPECT_EQ(inf.used, fast_lio::ReleaseVelocitySource::kZero);
  // a leg sample still wins over a runaway estimate
  const auto leg = fast_lio::releaseVelocity(Eigen::Vector3d(4.0, 0.0, 0.0), Eigen::Matrix3d::Identity(),
                                             Eigen::Vector3d(0.6, 0.0, 0.0), 0.1, legParams());
  EXPECT_EQ(leg.used, fast_lio::ReleaseVelocitySource::kLeg);
}

TEST(ReleaseSigma, ClampsBetweenFloorAndCeilingAndTreatsBadDurationsAsZero)
{
  fast_lio::ReleaseInflationParams p;
  p.pos_sigma_min = 0.1;
  p.pos_sigma_rate = 0.2;
  p.pos_sigma_max = 1.0;
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(0.0, p), 0.1);
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(2.5, p), 0.6);
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(100.0, p), 1.0);
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(-3.0, p), 0.1);
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(kNaN, p), 0.1);
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(kInf, p), 0.1);  // non-finite bookkeeping counts as zero, the floor applies
  p.rp_sigma_min_deg = 0.5;
  p.rp_sigma_rate_deg = 0.3;
  p.rp_sigma_max_deg = 3.0;
  EXPECT_NEAR(fast_lio::releaseRollPitchSigmaRad(2.0, p), 1.1 * kDeg, 1e-12);
  EXPECT_NEAR(fast_lio::releaseRollPitchSigmaRad(50.0, p), 3.0 * kDeg, 1e-12);
  // a ceiling below the floor collapses to the floor instead of inverting the clamp
  p.pos_sigma_max = 0.05;
  EXPECT_DOUBLE_EQ(fast_lio::releasePositionSigma(2.0, p), 0.1);
}

TEST(ReleaseInflation, DisabledOrBadInputLeavesPAlone)
{
  Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Identity() * 1e-4;
  const Eigen::Matrix<double, 23, 23> P0 = P;
  fast_lio::ReleaseInflationParams p;  // disabled
  EXPECT_FALSE(fast_lio::inflateReleaseCovariance(P, 2.0, Eigen::Vector3d::UnitZ(), p));
  EXPECT_TRUE(P.isApprox(P0));
  p.enabled = true;
  EXPECT_FALSE(fast_lio::inflateReleaseCovariance(P, 2.0, Eigen::Vector3d(kNaN, 0.0, 1.0), p));
  EXPECT_FALSE(fast_lio::inflateReleaseCovariance(P, 2.0, Eigen::Vector3d::Zero(), p));
  EXPECT_FALSE(fast_lio::inflateReleaseCovariance(P, 2.0, Eigen::Vector3d::UnitZ(), p, 21, 3));  // pos block past the end
  EXPECT_TRUE(P.isApprox(P0));
}

TEST(ReleaseInflation, AddsPositionAndRollPitchVarianceAndLeavesYaw)
{
  Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Zero();
  fast_lio::ReleaseInflationParams p;
  p.enabled = true;
  p.pos_sigma_min = 0.1;
  p.pos_sigma_rate = 0.2;
  p.pos_sigma_max = 1.0;
  p.rp_sigma_min_deg = 0.5;
  p.rp_sigma_rate_deg = 0.3;
  p.rp_sigma_max_deg = 3.0;
  ASSERT_TRUE(fast_lio::inflateReleaseCovariance(P, 2.5, Eigen::Vector3d::UnitZ(), p));
  const double sp2 = 0.6 * 0.6;
  const double sr2 = std::pow(1.25 * kDeg, 2);
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(P(k, k), sp2, 1e-15);
  }
  EXPECT_NEAR(P(3, 3), sr2, 1e-15);
  EXPECT_NEAR(P(4, 4), sr2, 1e-15);
  EXPECT_NEAR(P(5, 5), 0.0, 1e-15);  // yaw untouched
  EXPECT_NEAR(P(3, 4), 0.0, 1e-15);
  EXPECT_NEAR(P(12, 12), 0.0, 1e-15);  // velocity block untouched
  EXPECT_TRUE(P.isApprox(P.transpose()));
}

TEST(ReleaseInflation, TiltedVerticalKeepsTheYawDirectionQuiet)
{
  Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Zero();
  fast_lio::ReleaseInflationParams p;
  p.enabled = true;
  p.yaw_sigma_deg = 0.0;
  const Eigen::Vector3d u = Eigen::Vector3d(0.3, -0.2, 0.9).normalized();
  ASSERT_TRUE(fast_lio::inflateReleaseCovariance(P, 0.0, u * 0.9, p));  // a not-quite-unit vertical is normalised
  const Eigen::Matrix3d R = P.block(3, 3, 3, 3);
  EXPECT_NEAR((R * u).norm(), 0.0, 1e-15);  // no variance along the body vertical
  const Eigen::Vector3d h = u.cross(Eigen::Vector3d::UnitX()).normalized();
  EXPECT_NEAR(h.dot(R * h), std::pow(0.5 * kDeg, 2), 1e-15);  // full floor variance in a horizontal direction
  EXPECT_TRUE(R.isApprox(R.transpose()));
}

TEST(ReleaseInflation, InflationIsAdditiveAndKeepsCrossTerms)
{
  Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Zero();
  P(0, 0) = 1e-4;
  P(0, 4) = 2e-6;
  P(4, 0) = 2e-6;
  P(4, 4) = 1e-6;
  fast_lio::ReleaseInflationParams p;
  p.enabled = true;
  ASSERT_TRUE(fast_lio::inflateReleaseCovariance(P, 1.0, Eigen::Vector3d::UnitZ(), p));
  EXPECT_NEAR(P(0, 0), 1e-4 + 0.3 * 0.3, 1e-15);
  EXPECT_NEAR(P(4, 4), 1e-6 + std::pow(0.8 * kDeg, 2), 1e-15);
  EXPECT_NEAR(P(0, 4), 2e-6, 1e-15);
  EXPECT_NEAR(P(4, 0), 2e-6, 1e-15);
}

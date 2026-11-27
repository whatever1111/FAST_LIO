// Tests for engulf_release.hpp: the released velocity decision.
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

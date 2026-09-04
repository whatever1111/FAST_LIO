#include <cmath>
#include <gtest/gtest.h>

#include "parked_hold.hpp"

using fast_lio::parkedHoldMeasurement;
using fast_lio::ParkedHoldParams;
using fast_lio::parkedHoldShouldApply;
using fast_lio::ParkedHoldState;
using fast_lio::updateParkedHold;

namespace
{
ParkedHoldParams enabledParams()
{
  ParkedHoldParams p;
  p.enabled = true;
  return p;
}

Eigen::Matrix3d expMap(const Eigen::Vector3d & w)
{
  const double angle = w.norm();
  if (angle < 1e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(angle, w / angle).toRotationMatrix();
}
}  // namespace

// ---------------------------------------------------------------- hysteresis

TEST(ParkedHold, EngagesOnlyAfterTheStreak)
{
  ParkedHoldState s;
  const ParkedHoldParams p = enabledParams();  // engage_scans = 3
  EXPECT_FALSE(updateParkedHold(&s, true, p).engaged);
  EXPECT_FALSE(updateParkedHold(&s, true, p).engaged);
  const auto third = updateParkedHold(&s, true, p);
  EXPECT_TRUE(third.engaged);
  EXPECT_TRUE(third.just_engaged);
  EXPECT_FALSE(updateParkedHold(&s, true, p).just_engaged);
}

TEST(ParkedHold, OneMovingScanReleasesAndRestartsTheStreak)
{
  ParkedHoldState s;
  const ParkedHoldParams p = enabledParams();
  for (int i = 0; i < 5; ++i) {
    updateParkedHold(&s, true, p);
  }
  const auto released = updateParkedHold(&s, false, p);
  EXPECT_FALSE(released.engaged);
  EXPECT_TRUE(released.just_released);
  EXPECT_EQ(s.parked_scans, 0);
  // The streak has to be earned again, not resumed.
  EXPECT_FALSE(updateParkedHold(&s, true, p).engaged);
}

TEST(ParkedHold, DisabledNeverEngages)
{
  ParkedHoldState s;
  ParkedHoldParams p;  // enabled = false
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(updateParkedHold(&s, true, p).engaged);
  }
  EXPECT_FALSE(s.active);
}

// --------------------------------------------------------------- measurement

TEST(ParkedHold, MatchingStateProducesNoInnovation)
{
  const Eigen::Vector3d pos(1.0, -2.0, 0.3);
  const Eigen::Matrix3d rot = expMap(Eigen::Vector3d(0.1, -0.2, 0.7));
  const auto m = parkedHoldMeasurement(pos, rot, pos, rot, Eigen::Vector3d::Zero(), enabledParams());
  EXPECT_NEAR(m.residual.norm(), 0.0, 1e-12);
}

TEST(ParkedHold, PositionAndVelocityRowsAreTheObviousOnes)
{
  const Eigen::Vector3d anchor(1.0, 2.0, 3.0);
  const Eigen::Vector3d est(1.5, 1.0, 3.25);
  const Eigen::Vector3d vel(0.4, -0.1, 0.05);
  const auto m =
    parkedHoldMeasurement(anchor, Eigen::Matrix3d::Identity(), est, Eigen::Matrix3d::Identity(), vel, enabledParams());
  EXPECT_TRUE(m.residual.segment<3>(0).isApprox(anchor - est));
  EXPECT_TRUE(m.residual.segment<3>(6).isApprox(-vel));
  // Extra parens: the preprocessor counts the comma in block<3, 3> as an argument separator.
  EXPECT_TRUE((m.H.block<3, 3>(0, 0).isApprox(Eigen::Matrix3d::Identity())));
  EXPECT_TRUE((m.H.block<3, 3>(6, 12).isApprox(Eigen::Matrix3d::Identity())));
}

// The sign convention is the whole risk in this file: the ESIKF applies
// x.boxplus(K * residual) with a RIGHT perturbation, so a unit-gain update must
// land exactly on the anchor. Get this backwards and the prior doubles the
// error it was meant to remove.
TEST(ParkedHold, AttitudeResidualCarriesTheEstimateOntoTheAnchor)
{
  const Eigen::Matrix3d est = expMap(Eigen::Vector3d(0.05, -0.02, 0.3));
  const Eigen::Matrix3d anchor = expMap(Eigen::Vector3d(0.05, -0.02, 0.32));
  const auto m = parkedHoldMeasurement(
    Eigen::Vector3d::Zero(), anchor, Eigen::Vector3d::Zero(), est, Eigen::Vector3d::Zero(), enabledParams());
  const Eigen::Matrix3d applied = est * expMap(m.residual.segment<3>(3));
  EXPECT_TRUE(applied.isApprox(anchor, 1e-9));
}

TEST(ParkedHold, AttitudeResidualRoundTripsAtLargeAngles)
{
  const Eigen::Matrix3d est = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d anchor = expMap(Eigen::Vector3d(0.0, 0.0, 2.9));  // ~166 deg
  const auto m = parkedHoldMeasurement(
    Eigen::Vector3d::Zero(), anchor, Eigen::Vector3d::Zero(), est, Eigen::Vector3d::Zero(), enabledParams());
  EXPECT_TRUE((est * expMap(m.residual.segment<3>(3))).isApprox(anchor, 1e-9));
}

TEST(ParkedHold, HTouchesOnlyPositionAttitudeAndVelocity)
{
  const auto m = parkedHoldMeasurement(Eigen::Vector3d::Zero(),
                                       Eigen::Matrix3d::Identity(),
                                       Eigen::Vector3d::Zero(),
                                       Eigen::Matrix3d::Identity(),
                                       Eigen::Vector3d::Zero(),
                                       enabledParams());
  for (int col = 0; col < 23; ++col) {
    const bool observed = (col < 6) || (col >= 12 && col < 15);
    EXPECT_EQ(m.H.col(col).norm() > 0.0, observed) << "column " << col;
  }
}

TEST(ParkedHold, NoiseEntersAsVariance)
{
  ParkedHoldParams p = enabledParams();
  p.pos_noise = 0.04;
  p.rot_noise = 0.01;
  p.vel_noise = 0.2;
  const auto m = parkedHoldMeasurement(Eigen::Vector3d::Zero(),
                                       Eigen::Matrix3d::Identity(),
                                       Eigen::Vector3d::Zero(),
                                       Eigen::Matrix3d::Identity(),
                                       Eigen::Vector3d::Zero(),
                                       p);
  EXPECT_NEAR(m.R_diag(0), 0.04 * 0.04, 1e-15);
  EXPECT_NEAR(m.R_diag(3), 0.01 * 0.01, 1e-15);
  EXPECT_NEAR(m.R_diag(6), 0.2 * 0.2, 1e-15);
}

// ------------------------------------------------------------------ deadband

TEST(ParkedHold, DeadbandHoldsFireOnlyOnRealDeviation)
{
  const ParkedHoldParams p = enabledParams();  // 3 sigma of 0.05 m = 0.15 m
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  EXPECT_FALSE(parkedHoldShouldApply(Eigen::Vector3d::Zero(), I, Eigen::Vector3d(0.10, 0.0, 0.0), I, p));
  EXPECT_TRUE(parkedHoldShouldApply(Eigen::Vector3d::Zero(), I, Eigen::Vector3d(0.20, 0.0, 0.0), I, p));
}

TEST(ParkedHold, DeadbandAlsoWatchesAttitude)
{
  const ParkedHoldParams p = enabledParams();  // 3 sigma of 0.02 rad = 0.06 rad
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d o = Eigen::Vector3d::Zero();
  EXPECT_FALSE(parkedHoldShouldApply(o, expMap(Eigen::Vector3d(0.0, 0.0, 0.04)), o, I, p));
  EXPECT_TRUE(parkedHoldShouldApply(o, expMap(Eigen::Vector3d(0.0, 0.0, 0.09)), o, I, p));
}

TEST(ParkedHold, ZeroDeadbandAppliesEveryScan)
{
  ParkedHoldParams p = enabledParams();
  p.deadband_sigma = 0.0;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  EXPECT_TRUE(parkedHoldShouldApply(Eigen::Vector3d::Zero(), I, Eigen::Vector3d::Zero(), I, p));
}

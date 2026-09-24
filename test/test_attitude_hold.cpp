#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <limits>

#include "attitude_hold.hpp"

using fast_lio::AttitudeHoldInputs;
using fast_lio::AttitudeHoldParams;
using fast_lio::holdAttitudeThisScan;
using fast_lio::projectRotationRowToYaw;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

AttitudeHoldParams enabled()
{
  AttitudeHoldParams p;
  p.enabled = true;
  return p;
}

// A locked front end that just collapsed into the near field: the case the hold is for.
AttitudeHoldInputs engulfed()
{
  AttitudeHoldInputs in;
  in.engulf_latched = true;
  in.effct_prev = 500;
  in.hold_age_s = 0.5;
  in.scan_gap_s = 0.1;
  in.far_frac = 0.1;
  in.z_weak_prev = 1.0;
  return in;
}
}  // namespace

// Off by default: nothing holds, whatever the scene says.
TEST(AttitudeHold, DisabledNeverHolds)
{
  const AttitudeHoldParams p;
  EXPECT_FALSE(holdAttitudeThisScan(p, engulfed()));
}

TEST(AttitudeHold, EngulfmentOnALockedFrontEndHolds)
{
  EXPECT_TRUE(holdAttitudeThisScan(enabled(), engulfed()));
  AttitudeHoldInputs in = engulfed();
  in.engulf_latched = false;
  EXPECT_FALSE(holdAttitudeThisScan(enabled(), in));  // far_frac_max 0: the near-field rule is off
}

// m20 0825: the re-anchor verification at start-up froze the map on scan 1 while the
// init attitude was still wrong; a hold there is a deadlock. The hold is keyed on the
// engulfment latch, never on the map being frozen.
TEST(AttitudeHold, AFrozenMapAloneIsNotAReasonToHold)
{
  AttitudeHoldInputs in = engulfed();
  in.engulf_latched = false;
  in.far_frac = 0.95;
  EXPECT_FALSE(holdAttitudeThisScan(enabled(), in));
}

// A starved scan, a long hold, or a scan hole each release the hold: the first two
// mean the attitude being held is no longer trustworthy, the third that the gyro
// propagation across the gap is not either.
TEST(AttitudeHold, StarvationAgeAndHolesRelease)
{
  const AttitudeHoldParams p = enabled();
  AttitudeHoldInputs in = engulfed();
  in.effct_prev = p.min_effct - 1;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
  in = engulfed();
  in.hold_age_s = p.max_hold_s + 0.1;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
  in = engulfed();
  in.scan_gap_s = p.max_scan_gap_s + 0.1;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
  in = engulfed();
  in.hold_age_s = kNaN;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
  in = engulfed();
  in.scan_gap_s = kNaN;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
}

// The optional near-field rule needs both a low far-field share on this scan and
// no horizontal surface on the previous one, and obeys the same release conditions.
TEST(AttitudeHold, NearFieldRuleNeedsBothConditionsAndAThreshold)
{
  AttitudeHoldParams p = enabled();
  AttitudeHoldInputs in = engulfed();
  in.engulf_latched = false;
  in.far_frac = 0.2;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));  // far_frac_max 0: rule off
  p.far_frac_max = 0.5;
  EXPECT_TRUE(holdAttitudeThisScan(p, in));
  in.z_weak_prev = 0.5;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));  // horizontal surface was seen
  in.z_weak_prev = 1.0;
  in.far_frac = 0.8;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));  // far field is healthy
  in.far_frac = kNaN;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));
  in.far_frac = 0.2;
  in.effct_prev = 10;
  EXPECT_FALSE(holdAttitudeThisScan(p, in));  // starved: released like the latch case
}

// The projection keeps exactly the yaw component of a row and nothing else.
TEST(AttitudeHold, ProjectionKeepsOnlyTheRotationAboutTheVertical)
{
  const Eigen::Vector3d up(0.0, 0.0, 1.0);
  const Eigen::Vector3d row(0.3, -0.2, 0.7);
  const Eigen::Vector3d kept = projectRotationRowToYaw(row, up);
  EXPECT_NEAR(kept.x(), 0.0, 1e-15);
  EXPECT_NEAR(kept.y(), 0.0, 1e-15);
  EXPECT_NEAR(kept.z(), 0.7, 1e-15);
  // a row with no yaw content vanishes; a pure-yaw row is unchanged
  EXPECT_NEAR(projectRotationRowToYaw(Eigen::Vector3d(0.3, -0.2, 0.0), up).norm(), 0.0, 1e-15);
  EXPECT_NEAR((projectRotationRowToYaw(Eigen::Vector3d(0.0, 0.0, 0.7), up) - Eigen::Vector3d(0.0, 0.0, 0.7)).norm(), 0.0,
              1e-15);
  // a tilted body: the vertical is not the body z; the kept component is along that vertical
  const Eigen::Vector3d tilted = Eigen::Vector3d(0.1, 0.0, 1.0).normalized();
  const Eigen::Vector3d kept2 = projectRotationRowToYaw(row, tilted);
  EXPECT_NEAR(kept2.cross(tilted).norm(), 0.0, 1e-15);
  EXPECT_NEAR(kept2.dot(tilted), row.dot(tilted), 1e-15);
}

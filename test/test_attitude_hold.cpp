#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <limits>

#include "attitude_hold.hpp"

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
}  // namespace

// Off by default: nothing holds, whatever the scene says.
TEST(AttitudeHold, DisabledNeverHolds)
{
  const AttitudeHoldParams p;
  EXPECT_FALSE(holdAttitudeThisScan(p, true, true, 0.0, 1.0));
}

// The m20 doorway: the guard froze the map on the previous scan, or the
// engulfment latch is still set — either alone holds roll/pitch.
TEST(AttitudeHold, FrozenMapOrEngulfmentHolds)
{
  const AttitudeHoldParams p = enabled();
  EXPECT_TRUE(holdAttitudeThisScan(p, true, false, 0.95, 0.1));
  EXPECT_TRUE(holdAttitudeThisScan(p, false, true, 0.95, 0.1));
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, 0.95, 0.1));
}

// The optional near-field rule needs both a low far-field share on this scan and
// no horizontal surface on the previous one; it is off while far_frac_max is 0.
TEST(AttitudeHold, NearFieldRuleNeedsBothConditionsAndAThreshold)
{
  AttitudeHoldParams p = enabled();
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, 0.2, 1.0));  // far_frac_max 0: rule off
  p.far_frac_max = 0.5;
  EXPECT_TRUE(holdAttitudeThisScan(p, false, false, 0.2, 1.0));
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, 0.2, 0.5));  // horizontal surface was seen
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, 0.8, 1.0));  // far field is healthy
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, kNaN, 1.0));
  EXPECT_FALSE(holdAttitudeThisScan(p, false, false, 0.2, kNaN));
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

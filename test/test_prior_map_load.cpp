#include "prior_map_load.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using fast_lio::computePriorMapBounds;
using fast_lio::kInitialPoseSize;
using fast_lio::priorMapFitsCapacity;
using fast_lio::resolveInitialPose;

namespace
{
struct P
{
  float x, y, z;
};

// A local ENU field map: tens of metres across, centred near the origin.
std::vector<P> localMap()
{
  return {{-10.f, -20.f, -1.f}, {30.f, 40.f, 2.f}, {0.f, 0.f, 0.f}};
}

constexpr double kDegToRad = M_PI / 180.0;
constexpr double kTol = 1e-9;
}  // namespace

TEST(PriorMapBounds, EmptyCloudIsNotValid)
{
  const std::vector<P> empty;
  const auto b = computePriorMapBounds(empty.begin(), empty.end());
  EXPECT_FALSE(b.valid);
  EXPECT_EQ(b.finite_points, 0u);
  EXPECT_FALSE(b.likely_global_frame);
}

TEST(PriorMapBounds, AllNonFinitePointsIsNotValid)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<P> pts = {{nan, 0.f, 0.f}, {0.f, inf, 0.f}, {0.f, 0.f, -inf}};
  const auto b = computePriorMapBounds(pts.begin(), pts.end());
  EXPECT_FALSE(b.valid);
  EXPECT_EQ(b.finite_points, 0u);
}

TEST(PriorMapBounds, NonFinitePointsAreSkippedNotFatal)
{
  auto pts = localMap();
  pts.push_back({std::numeric_limits<float>::quiet_NaN(), 1.f, 1.f});
  const auto b = computePriorMapBounds(pts.begin(), pts.end());
  ASSERT_TRUE(b.valid);
  EXPECT_EQ(b.finite_points, 3u);  // the NaN point did not widen the box
  EXPECT_NEAR(b.xmin, -10.0, 1e-6);
  EXPECT_NEAR(b.xmax, 30.0, 1e-6);
  EXPECT_NEAR(b.ymin, -20.0, 1e-6);
  EXPECT_NEAR(b.ymax, 40.0, 1e-6);
  EXPECT_NEAR(b.zmin, -1.0, 1e-6);
  EXPECT_NEAR(b.zmax, 2.0, 1e-6);
}

TEST(PriorMapBounds, LocalMapIsNotFlaggedGlobal)
{
  const auto pts = localMap();
  const auto b = computePriorMapBounds(pts.begin(), pts.end());
  ASSERT_TRUE(b.valid);
  EXPECT_FALSE(b.likely_global_frame);
  EXPECT_LT(b.center_to_origin_m, 100.0);
}

TEST(PriorMapBounds, UtmOffsetIsFlaggedGlobal)
{
  // A UTM easting/northing map: the same 40 m extent, sitting 6e5 / 3.5e6 m out.
  const std::vector<P> pts = {{669000.f, 3560000.f, 10.f}, {669040.f, 3560030.f, 12.f}};
  const auto b = computePriorMapBounds(pts.begin(), pts.end());
  ASSERT_TRUE(b.valid);
  EXPECT_TRUE(b.likely_global_frame);
  EXPECT_GT(b.center_to_origin_m, 3.0e6);
}

TEST(PriorMapBounds, SinglePointGivesADegenerateButValidBox)
{
  const std::vector<P> pts = {{5.f, -5.f, 1.f}};
  const auto b = computePriorMapBounds(pts.begin(), pts.end());
  ASSERT_TRUE(b.valid);
  EXPECT_NEAR(b.xmin, b.xmax, 1e-6);
  EXPECT_NEAR(b.center_to_origin_m, std::sqrt(25.0 + 25.0 + 1.0), 1e-6);
}

TEST(InitialPose, ShortVectorIsRejected)
{
  for (std::size_t n = 0; n < kInitialPoseSize; ++n) {
    const std::vector<double> v(n, 1.0);
    EXPECT_FALSE(resolveInitialPose(v, false, 0.0, 0.0).valid) << "size " << n;
  }
}

TEST(InitialPose, NonFiniteComponentIsRejected)
{
  std::vector<double> v = {1.0, 2.0, 3.0, 0.0, 0.0, 90.0};
  v[4] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(resolveInitialPose(v, true, 0.0, 0.0).valid);
}

TEST(InitialPose, DefaultKeepsGravityRollPitchAndTakesYaw)
{
  // The vector asks for 30° roll; gravity says 2°. Default must keep gravity's.
  const std::vector<double> v = {1.0, 2.0, 3.0, 30.0, -20.0, 90.0};
  const double grav_roll = 2.0 * kDegToRad;
  const double grav_pitch = -1.0 * kDegToRad;
  const auto out = resolveInitialPose(v, false, grav_roll, grav_pitch);
  ASSERT_TRUE(out.valid);
  EXPECT_TRUE(out.kept_gravity_rp);
  EXPECT_NEAR(out.applied_roll_rad, grav_roll, kTol);
  EXPECT_NEAR(out.applied_pitch_rad, grav_pitch, kTol);
  EXPECT_NEAR(out.applied_yaw_rad, 90.0 * kDegToRad, kTol);
  EXPECT_NEAR(out.position.x(), 1.0, kTol);
  EXPECT_NEAR(out.position.z(), 3.0, kTol);
}

TEST(InitialPose, OverrideTakesRollPitchFromTheVector)
{
  const std::vector<double> v = {0.0, 0.0, 0.0, 30.0, -20.0, 45.0};
  const auto out = resolveInitialPose(v, true, 2.0 * kDegToRad, -1.0 * kDegToRad);
  ASSERT_TRUE(out.valid);
  EXPECT_FALSE(out.kept_gravity_rp);
  EXPECT_NEAR(out.applied_roll_rad, 30.0 * kDegToRad, kTol);
  EXPECT_NEAR(out.applied_pitch_rad, -20.0 * kDegToRad, kTol);
}

TEST(InitialPose, YawOnlyPoseRotatesTheXAxisIntoY)
{
  const std::vector<double> v = {0.0, 0.0, 0.0, 0.0, 0.0, 90.0};
  const auto out = resolveInitialPose(v, false, 0.0, 0.0);
  ASSERT_TRUE(out.valid);
  const Eigen::Vector3d x_world = out.orientation * Eigen::Vector3d::UnitX();
  EXPECT_NEAR(x_world.x(), 0.0, 1e-9);
  EXPECT_NEAR(x_world.y(), 1.0, 1e-9);
  EXPECT_NEAR(x_world.z(), 0.0, 1e-9);
}

TEST(InitialPose, IdentityVectorGivesIdentityRotation)
{
  const std::vector<double> v(kInitialPoseSize, 0.0);
  const auto out = resolveInitialPose(v, true, 0.0, 0.0);
  ASSERT_TRUE(out.valid);
  EXPECT_NEAR(out.orientation.angularDistance(Eigen::Quaterniond::Identity()), 0.0, 1e-12);
}

TEST(InitialPose, CompositionIsYawTimesPitchTimesRoll)
{
  // Guards the intrinsic Z-Y-X order the filter's own state expects: composing the same
  // angles the other way round gives a different attitude, and a wrong one here silently
  // tilts every body-to-world transform of the prior map.
  const std::vector<double> v = {0.0, 0.0, 0.0, 10.0, 20.0, 30.0};
  const auto out = resolveInitialPose(v, true, 0.0, 0.0);
  ASSERT_TRUE(out.valid);
  const Eigen::Quaterniond expected(Eigen::AngleAxisd(30.0 * kDegToRad, Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(20.0 * kDegToRad, Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(10.0 * kDegToRad, Eigen::Vector3d::UnitX()));
  EXPECT_NEAR(out.orientation.angularDistance(expected), 0.0, 1e-12);
}

TEST(PriorMapCapacity, FitsWhenUnderOrAtCapacityAndWhenUnbounded)
{
  EXPECT_TRUE(priorMapFitsCapacity(100u, 1000u));
  EXPECT_TRUE(priorMapFitsCapacity(1000u, 1000u));
  EXPECT_FALSE(priorMapFitsCapacity(1001u, 1000u));
  EXPECT_TRUE(priorMapFitsCapacity(9999999u, 0u));  // 0 = unbounded
}

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "gravity_align_kinematics.hpp"

using fast_lio::kLegEdgeMinSamples;
using fast_lio::kLegEdgeStandstillSpeed;
using fast_lio::leanOfAcceleration;
using fast_lio::legEdgeValid;
using fast_lio::legEdgeVariance;
using fast_lio::legEdgeVelocity;
using fast_lio::LevellingWindowScan;
using fast_lio::LinearAccelerationTerm;
using fast_lio::linearAccelerationTerm;

namespace
{
constexpr double kG = 9.81;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

LevellingWindowScan scan(double t_start, double t_end, const Eigen::Vector3d & v_start, const Eigen::Vector3d & v_end)
{
  LevellingWindowScan s;
  s.t_start = t_start;
  s.t_end = t_end;
  s.v_start = v_start;
  s.v_end = v_end;
  return s;
}
}  // namespace

// ── The case this header exists for ──────────────────────────────────────────
// m20 0826 131 s: the dog walks into a doorway at 0.86 m/s and stops inside the
// 1 s window. The window mean of the specific force then leans 5° forward, which
// the prior took as roll/pitch. The term recovers the −0.86 m/s² that explains it.
TEST(GravityAlignKinematics, DoorwayStopIsRecoveredFromTheVelocityEdges)
{
  const auto oldest = scan(100.0, 100.1, {0.86, 0.0, 0.0}, {0.80, 0.0, 0.0});
  const auto newest = scan(100.9, 101.0, {0.05, 0.0, 0.0}, {0.0, 0.0, 0.0});
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, false, kG, 0.0);
  ASSERT_TRUE(t.valid);
  EXPECT_NEAR(t.span, 1.0, 1e-12);
  EXPECT_NEAR(t.accel.x(), -0.86, 1e-12);
  EXPECT_NEAR(t.accel.y(), 0.0, 1e-12);
  EXPECT_NEAR(t.accel.z(), 0.0, 1e-12);
  EXPECT_NEAR(leanOfAcceleration(t.accel, kG) * 180.0 / M_PI, 5.01, 0.01);
  EXPECT_DOUBLE_EQ(t.sigma, 0.0);  // scale 0: the sigma term is off
}

// Only the two edges matter: the middle scans' velocities are irrelevant to the
// window mean of dv/dt (fundamental theorem of calculus), and a constant body
// velocity through a turn contributes nothing — that part is the ω×v term.
TEST(GravityAlignKinematics, ConstantBodyVelocityThroughATurnIsZero)
{
  const Eigen::Vector3d v(1.2, 0.0, 0.0);
  const auto oldest = scan(0.0, 0.1, v, v);
  const auto newest = scan(0.9, 1.0, v, v);
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, false, kG, 1.0);
  ASSERT_TRUE(t.valid);
  EXPECT_NEAR(t.accel.norm(), 0.0, 1e-15);
}

// A velocity that was overwritten inside the window has no derivative. Any of the
// three places the flag can arrive from must withhold the term.
TEST(GravityAlignKinematics, AVelocityOverwriteInsideTheWindowWithholdsTheTerm)
{
  const auto oldest = scan(0.0, 0.1, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0});
  const auto newest = scan(0.9, 1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  EXPECT_TRUE(linearAccelerationTerm(oldest, newest, false, kG, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(oldest, newest, true, kG, 1.0).valid);
  auto o = oldest;
  o.velocity_overwritten = true;
  EXPECT_FALSE(linearAccelerationTerm(o, newest, false, kG, 1.0).valid);
  auto n = newest;
  n.velocity_overwritten = true;
  EXPECT_FALSE(linearAccelerationTerm(oldest, n, false, kG, 1.0).valid);
}

TEST(GravityAlignKinematics, SigmaComesFromTheEdgeCovariancesOverTheSpan)
{
  auto oldest = scan(0.0, 0.1, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  auto newest = scan(0.9, 1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  oldest.var_start = 0.01;  // (0.1 m/s)² per edge
  newest.var_end = 0.01;
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, false, kG, 1.0);
  ASSERT_TRUE(t.valid);
  EXPECT_NEAR(t.sigma, std::sqrt(0.02) / kG, 1e-12);  // 0.0144 rad ≈ 0.83°
  // the scale multiplies, a longer span divides
  EXPECT_NEAR(linearAccelerationTerm(oldest, newest, false, kG, 2.0).sigma, 2.0 * std::sqrt(0.02) / kG, 1e-12);
  auto longer = newest;
  longer.t_end = 2.0;
  EXPECT_NEAR(linearAccelerationTerm(oldest, longer, false, kG, 1.0).sigma, std::sqrt(0.02) / (2.0 * kG), 1e-12);
  // a negative covariance trace (numerical) counts as zero, never as imaginary
  oldest.var_start = -1e-9;
  EXPECT_NEAR(linearAccelerationTerm(oldest, newest, false, kG, 1.0).sigma, std::sqrt(0.01) / kG, 1e-12);
}

TEST(GravityAlignKinematics, DegenerateInputsAreInvalidNotWild)
{
  const auto ok_old = scan(0.0, 0.1, {0.3, 0.0, 0.0}, {0.3, 0.0, 0.0});
  const auto ok_new = scan(0.9, 1.0, {0.4, 0.0, 0.0}, {0.4, 0.0, 0.0});
  EXPECT_TRUE(linearAccelerationTerm(ok_old, ok_new, false, kG, 1.0).valid);
  // zero or negative span (clock went backwards, or the same scan passed twice)
  EXPECT_FALSE(linearAccelerationTerm(ok_new, ok_old, false, kG, 1.0).valid);
  auto same = ok_old;
  same.t_end = same.t_start;
  EXPECT_FALSE(linearAccelerationTerm(same, same, false, kG, 1.0).valid);
  // non-finite velocity, gravity, covariance or scale
  auto nan_v = ok_new;
  nan_v.v_end = Eigen::Vector3d(kNaN, 0.0, 0.0);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, nan_v, false, kG, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, false, kNaN, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, false, 0.0, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, false, kG, kNaN).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, false, kG, -1.0).valid);
  auto nan_var = ok_old;
  nan_var.var_start = kNaN;
  EXPECT_FALSE(linearAccelerationTerm(nan_var, ok_new, false, kG, 1.0).valid);
  // an invalid term carries no acceleration at all
  const LinearAccelerationTerm t = linearAccelerationTerm(ok_new, ok_old, false, kG, 1.0);
  EXPECT_DOUBLE_EQ(t.accel.norm(), 0.0);
  EXPECT_DOUBLE_EQ(t.sigma, 0.0);
}

TEST(GravityAlignKinematics, LeanOfAccelerationUsesTheHorizontalPartOnly)
{
  EXPECT_NEAR(leanOfAcceleration({kG, 0.0, 0.0}, kG), M_PI / 4.0, 1e-12);
  EXPECT_NEAR(leanOfAcceleration({0.0, 0.0, 3.0}, kG), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(leanOfAcceleration({1.0, 0.0, 0.0}, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(leanOfAcceleration({kNaN, 0.0, 0.0}, kG), 0.0);
}

// ── Leg-odometry edges ───────────────────────────────────────────────────────
// m20: the legs measure the dog centre, the IMU sits 0.254 m ahead and 0.279 m
// above it (FP_POI→FP_VRTK). A body pitching at ω_y moves the IMU point along x
// and z relative to the centre; the edge velocity must carry that.
TEST(GravityAlignKinematics, LegEdgeVelocityCarriesTheLeverArm)
{
  const Eigen::Vector3d r(0.2538, 0.0, 0.2793);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d v0 = legEdgeVelocity(1.0, I, Eigen::Vector3d::Zero(), r);
  EXPECT_NEAR((v0 - Eigen::Vector3d(1.0, 0.0, 0.0)).norm(), 0.0, 1e-15);
  const Eigen::Vector3d v1 = legEdgeVelocity(1.0, I, Eigen::Vector3d(0.0, 0.5, 0.0), r);
  // ω×r with ω = 0.5 about y: (0.5·0.2793, 0, −0.5·0.2538)
  EXPECT_NEAR(v1.x(), 1.0 + 0.5 * 0.2793, 1e-12);
  EXPECT_NEAR(v1.y(), 0.0, 1e-12);
  EXPECT_NEAR(v1.z(), -0.5 * 0.2538, 1e-12);
  // a yawed robot→IMU mounting rotates the forward speed into the IMU axes
  Eigen::Matrix3d Rz;
  Rz << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Eigen::Vector3d v2 = legEdgeVelocity(0.8, Rz, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  EXPECT_NEAR(v2.x(), 0.0, 1e-12);
  EXPECT_NEAR(v2.y(), 0.8, 1e-12);
}

// The doorway case: the legs read ~0 while the IMU says the body moves — that is
// the bridge's "cannot measure", not a measurement. Parked and reading 0 is fine.
TEST(GravityAlignKinematics, LegEdgeIsInvalidWhenZeroOnAMovingBody)
{
  EXPECT_TRUE(legEdgeValid(10, 0.86, false));
  EXPECT_TRUE(legEdgeValid(10, 0.0, true));
  EXPECT_FALSE(legEdgeValid(10, 0.0, false));
  EXPECT_FALSE(legEdgeValid(10, 0.5 * kLegEdgeStandstillSpeed, false));
  EXPECT_FALSE(legEdgeValid(kLegEdgeMinSamples - 1, 0.86, false));
  EXPECT_TRUE(legEdgeValid(kLegEdgeMinSamples, 0.86, false));
  EXPECT_FALSE(legEdgeValid(10, kNaN, false));
  EXPECT_FALSE(legEdgeValid(10, kNaN, true));
}

TEST(GravityAlignKinematics, LegEdgeVarianceIsExactZeroOnlyWhenParked)
{
  EXPECT_DOUBLE_EQ(legEdgeVariance(0.0, true, 0.15), 0.0);
  EXPECT_DOUBLE_EQ(legEdgeVariance(0.0, false, 0.15), 0.15 * 0.15);
  EXPECT_DOUBLE_EQ(legEdgeVariance(1.0, true, 0.15), 0.15 * 0.15);
  EXPECT_DOUBLE_EQ(legEdgeVariance(1.0, false, 0.15), 0.15 * 0.15);
}

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

#include "gravity_align_kinematics.hpp"
#include "gravity_levelled_insertion.hpp"

using fast_lio::ImuTiltLegEdge;
using fast_lio::ImuTiltReference;
using fast_lio::ImuTiltReferenceParams;
using fast_lio::ImuTiltWindowRejection;
using fast_lio::kLegEdgeMinSamples;
using fast_lio::kLegEdgeStandstillSpeed;
using fast_lio::leanOfAcceleration;
using fast_lio::legEdgeValid;
using fast_lio::legEdgeVariance;
using fast_lio::legEdgeVelocity;
using fast_lio::levellingSigma;
using fast_lio::LevellingSigmaPolicy;
using fast_lio::LevellingWindowScan;
using fast_lio::LinearAccelerationTerm;
using fast_lio::linearAccelerationTerm;

namespace
{
constexpr double kG = 9.81;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kImuHz = 200;
constexpr double kDt = 1.0 / kImuHz;
const Eigen::Vector3d kDown(0.0, 0.0, -1.0);
const Eigen::Vector3d kAccel(0.0, 0.0, kG);
const Eigen::Vector3d kZero = Eigen::Vector3d::Zero();

ImuTiltReferenceParams tiltParams()
{
  ImuTiltReferenceParams params;
  params.window_s = 1.0;
  params.gyr_cov = 0.0015636343949698187;
  params.b_gyr_cov = 3.564031869636761e-05;
  params.noise_static = 0.008;
  params.noise_moving = 0.045;
  params.motion_init_sigma = 8.0 * M_PI / 180.0;
  params.gravity_raw = kG;
  params.mag_ref = kG;
  params.vel_sigma_scale = 1.0;
  return params;
}

ImuTiltLegEdge legEdge(double speed, bool valid = true)
{
  return {Eigen::Vector3d(speed, 0.0, 0.0), 0.15 * 0.15, valid, true};
}

double angle(const Eigen::Vector3d & lhs, const Eigen::Vector3d & rhs)
{
  return std::atan2(lhs.cross(rhs).norm(), lhs.dot(rhs));
}

void expectSameRotation(const Eigen::Quaterniond & expected, const Eigen::Quaterniond & used)
{
  EXPECT_EQ(std::memcmp(expected.coeffs().data(), used.coeffs().data(), sizeof(double) * 4), 0);
}

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

TEST(ImuTiltReference, LevelStationaryFor100SecondsPreservesQuaternionBytes)
{
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, tiltParams());
  ASSERT_TRUE(tilt.initialised());
  tilt.closeWindow({}, true, 0.0);
  for (int i = 1; i <= 100 * kImuHz; ++i) {
    tilt.propagate(kZero, kZero, kAccel, kDt);
    if (i % kImuHz == 0)
      tilt.closeWindow({}, true, i * kDt);
  }
  const Eigen::Vector3d down = tilt.downBodyAt(100.0);
  EXPECT_LE(angle(down, kDown), 1e-12);
  Eigen::Quaterniond rot(Eigen::AngleAxisd(0.34, Eigen::Vector3d::UnitZ()));
  rot.x() = -0.0;
  expectSameRotation(rot, fast_lio::levelledInsertionRotation(rot, down));
}

TEST(ImuTiltReference, PurePitchRotatesTheEntireForceSumIntoTheEndFrame)
{
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, tiltParams());
  tilt.closeWindow(legEdge(0.0), true, 0.0);
  const double rate = 10.0 * M_PI / 180.0;
  const Eigen::Vector3d gyro = rate * Eigen::Vector3d::UnitY();
  for (int i = 1; i <= kImuHz; ++i) {
    const double t = i * kDt;
    const Eigen::AngleAxisd world_to_body(-rate * t, Eigen::Vector3d::UnitY());
    tilt.propagate(gyro, gyro, world_to_body * kAccel, kDt);
    EXPECT_LE(angle(tilt.downBodyAt(t), world_to_body * kDown), 1e-6) << "sample " << i;
  }
  tilt.closeWindow(legEdge(0.0), false, 1.0);
  ASSERT_TRUE(tilt.diagnostics().updated);
  EXPECT_LE(tilt.diagnostics().residual_angle, 1e-6);
  const Eigen::AngleAxisd one_period_ahead(-rate * (1.0 + kDt), Eigen::Vector3d::UnitY());
  EXPECT_LE(angle(tilt.downBodyAt(1.0 + 5.0 * kDt), one_period_ahead * kDown), 1e-6);
}

TEST(ImuTiltReference, SustainedAccelerationUsesBothLegEdgesOrWithholdsWindow)
{
  const Eigen::Vector3d accel = kAccel + Eigen::Vector3d(0.5, 0.0, 0.0);
  for (int invalid_edge : {-1, 0, 1}) {
    SCOPED_TRACE(invalid_edge);
    auto params = tiltParams();
    params.window_s = 2.0;
    ImuTiltReference tilt;
    tilt.initialise(kAccel, kZero, true, params);
    tilt.closeWindow(legEdge(0.4, invalid_edge != 0), false, 0.0);
    for (int i = 0; i < 2 * kImuHz; ++i)
      tilt.propagate(kZero, kZero, accel, kDt);
    tilt.closeWindow(legEdge(1.4, invalid_edge != 1), false, 2.0);
    const auto & diag = tilt.diagnostics();
    if (invalid_edge == -1) {
      ASSERT_TRUE(diag.updated);
      EXPECT_LE(diag.residual_angle, 1e-6);
    } else {
      EXPECT_FALSE(diag.updated);
      EXPECT_EQ(diag.rejection, ImuTiltWindowRejection::kLegEdge);
    }
    EXPECT_LE(angle(tilt.downBodyAt(2.0), kDown), 1e-6);
  }
}

TEST(ImuTiltReference, EstimatesPostInitialisationHorizontalGyroBias)
{
  const auto params = tiltParams();
  const double bias = 100.0 * M_PI / (180.0 * 3600.0);
  const double rc = params.noise_static * params.noise_static * params.window_s;
  const double k1 = std::sqrt(params.gyr_cov * kDt / rc + 2.0 * std::sqrt(params.b_gyr_cov * kDt / rc));
  const Eigen::Vector3d rate(bias, 0.0, 0.0);
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, params);
  tilt.closeWindow({}, true, 0.0);
  double peak = 0.0;
  for (int i = 1; i <= 50 * kImuHz; ++i) {
    tilt.propagate(i == 1 ? kZero : rate, rate, kAccel, kDt);
    const double t = i * kDt;
    peak = std::max(peak, angle(tilt.downBodyAt(t), kDown));
    if (i % kImuHz == 0) {
      tilt.closeWindow({}, true, t);
      ASSERT_TRUE(tilt.diagnostics().updated);
      peak = std::max(peak, angle(tilt.downBodyAt(t), kDown));
    }
  }
  EXPECT_LE(peak, bias / k1);
  EXPECT_LE(std::abs(tilt.bias().x() - bias), 3.0 * std::sqrt(tilt.covarianceDiagonal()(3)));
}

TEST(ImuTiltReference, NonOverlappingWindowsCountEachSampleOnce)
{
  constexpr double duration = 3.5;
  const auto params = tiltParams();
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, params);
  tilt.closeWindow({}, true, 0.0);
  int windows = 0;
  int samples = 0;
  for (int i = 1; i <= duration * kImuHz; ++i) {
    tilt.propagate(kZero, kZero, kAccel, kDt);
    if (i % (kImuHz / 10) != 0)
      continue;
    tilt.closeWindow({}, true, i * kDt);
    const auto & diag = tilt.diagnostics();
    if (diag.windows > windows) {
      ASSERT_TRUE(diag.updated);
      EXPECT_EQ(diag.samples, kImuHz);
      samples += diag.samples;
      ++windows;
    }
  }
  EXPECT_EQ(windows, static_cast<int>(std::floor(duration / params.window_s)));
  EXPECT_EQ(samples, windows * kImuHz);
}

TEST(ImuTiltReference, HorizontalAccelerometerBiasKeepsItsPhysicalTiltFloor)
{
  const Eigen::Vector3d accel = kAccel + Eigen::Vector3d(kG * 1e-3, 0.0, 0.0);
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, tiltParams());
  tilt.closeWindow({}, true, 0.0);
  for (int i = 1; i <= 60 * kImuHz; ++i) {
    tilt.propagate(kZero, kZero, accel, kDt);
    if (i % kImuHz == 0)
      tilt.closeWindow({}, true, i * kDt);
  }
  ASSERT_TRUE(tilt.diagnostics().updated);
  const double error = angle(tilt.downBodyAt(60.0), kDown);
  EXPECT_GE(error, 0.9e-3);
  EXPECT_LE(error, 1.1e-3);
}

TEST(ImuTiltReference, InsertionMatchesBodyDownAndChangesYawAtSecondOrder)
{
  const auto yaw = [](const Eigen::Quaterniond & q) {
    const Eigen::Vector3d forward = q * Eigen::Vector3d::UnitX();
    return std::atan2(forward.y(), forward.x());
  };
  const Eigen::Quaterniond heading(Eigen::AngleAxisd(0.66, Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d tilt_axis = Eigen::Vector3d(1.0, -2.0, 0.0).normalized();
  const Eigen::Vector3d reference_axis = Eigen::Vector3d(2.0, 1.0, 0.0).normalized();
  double previous_heading_change = 0.0;
  for (double theta : {0.06, 0.03}) {
    const Eigen::Quaterniond rot = heading * Eigen::AngleAxisd(theta, tilt_axis);
    const Eigen::Vector3d down = Eigen::AngleAxisd(-theta, reference_axis) * kDown;
    const auto used = fast_lio::levelledInsertionRotation(rot, down);
    EXPECT_LE(angle(used.conjugate() * kDown, down), 1e-12);
    const double heading_change = std::abs(yaw(used) - yaw(rot));
    ASSERT_GT(heading_change, 0.0);
    if (previous_heading_change > 0.0) {
      EXPECT_GT(heading_change / previous_heading_change, 0.24);
      EXPECT_LT(heading_change / previous_heading_change, 0.26);
    }
    previous_heading_change = heading_change;
  }
}

TEST(ImuTiltReference, PriorMapAndUnavailableReferencePreserveQuaternionBytes)
{
  Eigen::Quaterniond rot(Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitY()));
  rot.x() = -0.0;
  ImuTiltReference tilt;
  expectSameRotation(rot, fast_lio::insertionRotationFromImuTilt(rot, tilt, false, 0.0));
  auto params = tiltParams();
  tilt.initialise(kAccel, kZero, true, params);
  ASSERT_TRUE(tilt.initialised());
  expectSameRotation(rot, fast_lio::insertionRotationFromImuTilt(rot, tilt, true, 0.0));
  params.window_s = 0.0;
  tilt.initialise(kAccel, kZero, true, params);
  ASSERT_FALSE(tilt.initialised());
  expectSameRotation(rot, fast_lio::insertionRotationFromImuTilt(rot, tilt, false, 0.0));
}

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

// ── Sigma policy ─────────────────────────────────────────────────────────────
// The scene policy is the legacy rule, bit for bit; the measurement policy never
// looks at the scene and answers only "is the body static?".
TEST(GravityAlignKinematics, ScenePolicyReproducesTheLegacyRule)
{
  const auto P = LevellingSigmaPolicy::kScene;
  EXPECT_DOUBLE_EQ(levellingSigma(P, true, false, 0.03, 0.008, 0.01, 0.045), 0.008);
  EXPECT_DOUBLE_EQ(levellingSigma(P, true, true, 0.03, 0.008, 0.01, 0.045), 0.008);
  EXPECT_DOUBLE_EQ(levellingSigma(P, false, false, 0.03, 0.008, 0.01, 0.045), 0.03);
  EXPECT_DOUBLE_EQ(levellingSigma(P, true, false, 0.03, 0.0, 0.01, 0.045), 0.03);  // degraded sigma 0 = unused
}

TEST(GravityAlignKinematics, MeasurementPolicyIgnoresTheSceneAndFollowsTheBody)
{
  const auto P = LevellingSigmaPolicy::kMeasurement;
  for (const bool scene : {false, true}) {
    EXPECT_DOUBLE_EQ(levellingSigma(P, scene, true, 0.03, 0.008, 0.01, 0.045), 0.01);
    EXPECT_DOUBLE_EQ(levellingSigma(P, scene, false, 0.03, 0.008, 0.01, 0.045), 0.045);
  }
}

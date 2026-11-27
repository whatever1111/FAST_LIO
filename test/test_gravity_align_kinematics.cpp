#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>

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

TEST(ImuTiltReference, StraightTurnStraightUsesTheWholeForceWindow)
{
  constexpr double turnStart = 0.25;
  constexpr double turnDuration = 0.5;
  constexpr double peakRate = 1.2;
  constexpr double speed = 1.0;
  const double frequency = 2.0 * M_PI / turnDuration;
  const Eigen::Vector3d lever(0.3238, 0.0, 0.2793);
  const Eigen::Vector3d forward = speed * Eigen::Vector3d::UnitX();
  const auto params = tiltParams();

  const auto edge = [&](const Eigen::Vector3d & rate) {
    auto value = legEdge(speed);
    value.velocity = legEdgeVelocity(speed, Eigen::Matrix3d::Identity(), rate, lever);
    return value;
  };
  ImuTiltReference tilt;
  tilt.initialise(kAccel, kZero, true, params);
  tilt.closeWindow(edge(kZero), false, 0.0);
  Eigen::Vector3d ratePrev = kZero;
  for (int i = 1; i <= kImuHz; ++i) {
    const double phase = frequency * std::clamp(i * kDt - turnStart, 0.0, turnDuration);
    const Eigen::Vector3d rate = (0.5 * peakRate * (1.0 - std::cos(phase))) * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d rateDot = (0.5 * peakRate * frequency * std::sin(phase)) * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d imuVelocity = forward + rate.cross(lever);
    const Eigen::Vector3d accel = kAccel + rateDot.cross(lever) + rate.cross(imuVelocity);
    tilt.propagate(ratePrev, rate, accel, kDt);
    ratePrev = rate;
  }
  tilt.closeWindow(edge(ratePrev), false, params.window_s);
  const auto & diag = tilt.diagnostics();
  RecordProperty("residual_deg", std::to_string(diag.residual_angle * 180.0 / M_PI));
  RecordProperty("window_samples", diag.samples);
  ASSERT_TRUE(diag.updated);

  // d(R v_imu)/dt is the world acceleration. Translation is constant on the
  // straight segments. During the turn, differentiate R*(omega x v_imu + omega_dot x lever)
  // to bound the jerk and thus the right-endpoint rule error in the mean force.
  const double radius = lever.head<2>().norm();
  const double rateSlope = 0.5 * peakRate * frequency;
  const double rateCurvature = rateSlope * frequency;
  const double maxSpeed = speed + peakRate * radius;
  const double maxAcceleration = peakRate * maxSpeed + rateSlope * radius;
  const double maxJerk =
    peakRate * maxAcceleration + rateSlope * (maxSpeed + peakRate * radius) + rateCurvature * radius;
  const double quadratureError = 0.5 * kDt * maxJerk * turnDuration / params.window_s;
  // Trapezoidal yaw integration bounds the rotation error of both the
  // acceleration samples and the starting leg edge. Yaw leaves gravity unchanged.
  const double yawError = params.window_s * kDt * kDt * rateCurvature / 12.0;
  const double forceError = quadratureError + yawError * (maxAcceleration + speed / params.window_s);
  const double angleBound = std::atan2(forceError, kG);
  RecordProperty("angle_bound_deg", std::to_string(angleBound * 180.0 / M_PI));
  EXPECT_LE(diag.residual_angle, angleBound);
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

// The measured vertical is the direction of G·up + b_a; the bias column is its derivative at zero bias, and the
// prediction its first-order expansion.
TEST(GravityAlignKinematics, LevellingBiasColumnIsTheDerivativeOfTheBiasedVertical)
{
  constexpr double G = 9.81;
  for (const Eigen::Vector3d & up :
       {Eigen::Vector3d(Eigen::Vector3d::UnitZ()), Eigen::Vector3d(Eigen::Vector3d(0.3, -0.2, 0.93).normalized())}) {
    const auto measured = [&up, G](const Eigen::Vector3d & bias) {
      return Eigen::Vector3d((G * up + bias).normalized());
    };
    const double step = 1e-4;
    Eigen::Matrix3d numeric;
    for (int axis = 0; axis < 3; ++axis) {
      const Eigen::Vector3d delta = step * Eigen::Vector3d::Unit(axis);
      numeric.col(axis) = (measured(delta) - measured(-delta)) / (2.0 * step);
    }
    EXPECT_LT((numeric - fast_lio::levellingBiasJacobian(up, G)).cwiseAbs().maxCoeff(), 1e-8);
    const Eigen::Vector3d bias(0.03, -0.02, 0.05);
    EXPECT_LT((fast_lio::levellingPrediction(up, bias, G) - measured(bias)).norm(), 2.0 * bias.squaredNorm() / (G * G));
  }
}

TEST(GravityAlignKinematics, ZeroBiasMatchesTheLegacyVerticalAndRetainsTheDerivative)
{
  const Eigen::Vector3d up = Eigen::Vector3d(0.1, 0.2, 0.97).normalized();
  EXPECT_EQ(fast_lio::levellingPrediction(up, Eigen::Vector3d::Zero(), 9.81), up);
  const Eigen::Matrix3d jacobian = fast_lio::levellingBiasJacobian(up, 9.81);
  EXPECT_LT((jacobian * up).norm(), 1e-15);
  EXPECT_NEAR(jacobian.trace(), 2.0 / 9.81, 1e-15);
}

TEST(GravityAlignKinematics, BiasedVerticalAtTruthLeavesOnlySecondOrderResidual)
{
  const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d horizontal(0.0316, 0.0, 0.0);
  const Eigen::Vector3d axial(0.0, 0.0, 0.0316);
  for (const auto & bias : {horizontal, axial, Eigen::Vector3d(horizontal + axial)}) {
    const Eigen::Vector3d measured = (kG * up + bias).normalized();
    const double bound = 2.0 * bias.squaredNorm() / (kG * kG);
    const double legacy_residual = (measured - up).norm();
    const double horizontal_ratio = (bias - up * up.dot(bias)).norm() / kG;
    EXPECT_LE((measured - fast_lio::levellingPrediction(up, bias, kG)).norm(), bound);
    EXPECT_LE(std::abs(legacy_residual - horizontal_ratio), bound);
    if (bias == horizontal)
      EXPECT_GT(legacy_residual, 0.5 * horizontal_ratio);
  }
}

TEST(GravityAlignKinematics, LevellingBiasColumnNullSpaceIsTheVertical)
{
  const Eigen::Vector3d up = Eigen::Vector3d(0.25, -0.11, 0.96).normalized();
  const Eigen::Matrix3d jacobian = fast_lio::levellingBiasJacobian(up, kG);
  Eigen::FullPivLU<Eigen::Matrix3d> rank(jacobian);
  rank.setThreshold(1e-12);
  EXPECT_EQ(rank.rank(), 2);
  EXPECT_LE((jacobian * up).norm(), 1e-12);
}

TEST(GravityAlignKinematics, HeadingChangeSeparatesHorizontalBiasFromWorldTilt)
{
  const auto levelling_rows = [](double yaw) {
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    const Eigen::Matrix3d rot = Eigen::AngleAxisd(yaw, up).toRotationMatrix();
    Eigen::Matrix3d attitude;
    for (int i = 0; i < 3; ++i)
      attitude.col(i) = up.cross(Eigen::Vector3d::Unit(i));
    Eigen::Matrix<double, 3, 4> rows;
    rows.leftCols<2>() = attitude * rot.transpose().leftCols<2>();
    rows.rightCols<2>() = fast_lio::levellingBiasJacobian(up, kG).leftCols<2>();
    return rows;
  };
  const auto parked = levelling_rows(0.0);
  Eigen::Matrix<double, 6, 4> turned;
  turned.topRows<3>() = parked;
  turned.bottomRows<3>() = levelling_rows(M_PI / 2.0);
  Eigen::FullPivLU<Eigen::Matrix<double, 3, 4>> parked_rank(parked);
  Eigen::FullPivLU<Eigen::Matrix<double, 6, 4>> turned_rank(turned);
  parked_rank.setThreshold(1e-12);
  turned_rank.setThreshold(1e-12);
  EXPECT_EQ(parked_rank.rank(), 2);
  EXPECT_EQ(turned_rank.rank(), 4);
}

// ── The case this header exists for ──────────────────────────────────────────
// m20 0826 131 s: the dog walks into a doorway at 0.86 m/s and stops inside the
// 1 s window. The window mean of the specific force then leans 5° forward, which
// the prior took as roll/pitch. The term recovers the −0.86 m/s² that explains it.
TEST(GravityAlignKinematics, DoorwayStopIsRecoveredFromTheVelocityEdges)
{
  const auto oldest = scan(100.0, 100.1, {0.86, 0.0, 0.0}, {0.80, 0.0, 0.0});
  const auto newest = scan(100.9, 101.0, {0.05, 0.0, 0.0}, {0.0, 0.0, 0.0});
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, kG, 0.0);
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
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, kG, 1.0);
  ASSERT_TRUE(t.valid);
  EXPECT_NEAR(t.accel.norm(), 0.0, 1e-15);
}

// An edge without a velocity measurement has no derivative. Either edge's flag
// withholds the term.
TEST(GravityAlignKinematics, AnEdgeWithoutAVelocityWithholdsTheTerm)
{
  const auto oldest = scan(0.0, 0.1, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0});
  const auto newest = scan(0.9, 1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  EXPECT_TRUE(linearAccelerationTerm(oldest, newest, kG, 1.0).valid);
  auto o = oldest;
  o.velocity_overwritten = true;
  EXPECT_FALSE(linearAccelerationTerm(o, newest, kG, 1.0).valid);
  auto n = newest;
  n.velocity_overwritten = true;
  EXPECT_FALSE(linearAccelerationTerm(oldest, n, kG, 1.0).valid);
}

TEST(GravityAlignKinematics, SigmaComesFromTheEdgeCovariancesOverTheSpan)
{
  auto oldest = scan(0.0, 0.1, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  auto newest = scan(0.9, 1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  oldest.var_start = 0.01;  // (0.1 m/s)² per edge
  newest.var_end = 0.01;
  const LinearAccelerationTerm t = linearAccelerationTerm(oldest, newest, kG, 1.0);
  ASSERT_TRUE(t.valid);
  EXPECT_NEAR(t.sigma, std::sqrt(0.02) / kG, 1e-12);  // 0.0144 rad ≈ 0.83°
  // the scale multiplies, a longer span divides
  EXPECT_NEAR(linearAccelerationTerm(oldest, newest, kG, 2.0).sigma, 2.0 * std::sqrt(0.02) / kG, 1e-12);
  auto longer = newest;
  longer.t_end = 2.0;
  EXPECT_NEAR(linearAccelerationTerm(oldest, longer, kG, 1.0).sigma, std::sqrt(0.02) / (2.0 * kG), 1e-12);
  // a negative covariance trace (numerical) counts as zero, never as imaginary
  oldest.var_start = -1e-9;
  EXPECT_NEAR(linearAccelerationTerm(oldest, newest, kG, 1.0).sigma, std::sqrt(0.01) / kG, 1e-12);
}

TEST(GravityAlignKinematics, DegenerateInputsAreInvalidNotWild)
{
  const auto ok_old = scan(0.0, 0.1, {0.3, 0.0, 0.0}, {0.3, 0.0, 0.0});
  const auto ok_new = scan(0.9, 1.0, {0.4, 0.0, 0.0}, {0.4, 0.0, 0.0});
  EXPECT_TRUE(linearAccelerationTerm(ok_old, ok_new, kG, 1.0).valid);
  // zero or negative span (clock went backwards, or the same scan passed twice)
  EXPECT_FALSE(linearAccelerationTerm(ok_new, ok_old, kG, 1.0).valid);
  auto same = ok_old;
  same.t_end = same.t_start;
  EXPECT_FALSE(linearAccelerationTerm(same, same, kG, 1.0).valid);
  // non-finite velocity, gravity, covariance or scale
  auto nan_v = ok_new;
  nan_v.v_end = Eigen::Vector3d(kNaN, 0.0, 0.0);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, nan_v, kG, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, kNaN, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, 0.0, 1.0).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, kG, kNaN).valid);
  EXPECT_FALSE(linearAccelerationTerm(ok_old, ok_new, kG, -1.0).valid);
  auto nan_var = ok_old;
  nan_var.var_start = kNaN;
  EXPECT_FALSE(linearAccelerationTerm(nan_var, ok_new, kG, 1.0).valid);
  // an invalid term carries no acceleration at all
  const LinearAccelerationTerm t = linearAccelerationTerm(ok_new, ok_old, kG, 1.0);
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

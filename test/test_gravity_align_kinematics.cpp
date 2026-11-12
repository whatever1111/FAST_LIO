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
using fast_lio::levellingSigma;
using fast_lio::LevellingSigmaPolicy;
using fast_lio::LevellingWindowGates;
using fast_lio::LevellingWindowReference;
using fast_lio::levellingWindowReference;
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

// ── The window read as a vertical reference ──────────────────────────────────
// The rebuild relevel (rebuild_relevel.hpp) reads the levelling window with the block's own gates. Here the window
// is one second of 200 Hz samples on the m20 settings: its oldest entry ends at 0.1 s, the scan in question at 1.0 s.
namespace
{
constexpr double kNoiseStatic = 0.008;
constexpr double kLegSigma = 0.15;

LevellingWindowGates m20Gates()
{
  LevellingWindowGates gates;
  gates.window_s = 1.0;
  gates.accel_tol = 0.05;
  gates.magnitude_ref = kG;
  gates.raw_per_mps2 = 1.0;
  gates.gravity = kG;
  gates.lin_accel_comp = true;
  gates.leg_edges = true;
  gates.leg_stream_alive = true;
  gates.vel_sigma_scale = 1.0;
  gates.leg_vel_sigma = kLegSigma;
  gates.noise_static = kNoiseStatic;
  return gates;
}

// A window of a constant mean specific force whose leg edges read v_start and v_end; a zero edge is a parked one
// (the IMU agrees), any other a moving one.
struct Window
{
  LevellingWindowScan oldest = scan(0.0, 0.1, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  LevellingWindowScan newest = scan(0.9, 1.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  int samples = 200;
  double span = 1.0;
  bool overwritten_inside = false;

  Window(const Eigen::Vector3d & mean_force, const Eigen::Vector3d & v_start, const Eigen::Vector3d & v_end)
  {
    oldest.v_start = v_start;
    oldest.var_start = legEdgeVariance(v_start.x(), v_start.isZero(), kLegSigma);
    newest.v_end = v_end;
    newest.var_end = legEdgeVariance(v_end.x(), v_end.isZero(), kLegSigma);
    sum = mean_force * double(samples);
  }

  LevellingWindowReference read(const LevellingWindowGates & gates = m20Gates()) const
  {
    return levellingWindowReference(sum, samples, span, oldest, newest, overwritten_inside, gates);
  }
};

// The specific force of a body at rest, tilted by `deg` about +y.
Eigen::Vector3d restingForce(double deg)
{
  const double a = deg * M_PI / 180.0;
  return kG * Eigen::Vector3d(std::sin(a), 0.0, std::cos(a));
}
}  // namespace

// Parked at both edges: the window's mean linear acceleration is zero, so its mean is the body's up.
TEST(GravityAlignKinematics, WindowParkedAtBothEdgesIsTheNormalisedMean)
{
  const LevellingWindowReference ref =
    Window(restingForce(5.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()).read();
  ASSERT_TRUE(ref.valid);
  EXPECT_TRUE(ref.parked);
  EXPECT_NEAR((ref.up_body - restingForce(5.0).normalized()).norm(), 0.0, 1e-12);
  // the levelling block's variance for this measurement: the static noise; the compensation adds exactly 0
  EXPECT_DOUBLE_EQ(ref.variance, kNoiseStatic * kNoiseStatic);
}

TEST(GravityAlignKinematics, WindowWalkingAtConstantSpeedIsValidButNotParked)
{
  const Eigen::Vector3d v(0.8, 0.0, 0.0);
  const LevellingWindowReference ref = Window(restingForce(0.0), v, v).read();
  EXPECT_TRUE(ref.valid);
  EXPECT_FALSE(ref.parked);
  EXPECT_DOUBLE_EQ(ref.variance, 0.0);
}

// Setting off inside the window, 0 → 0.5 m/s in its second: the specific-force mean leans 2.9° forward, and the
// reference takes off the mean linear acceleration the two edges give. One parked edge does not make it parked,
// and neither does a stop.
TEST(GravityAlignKinematics, WindowAtAStartTakesOffTheMeanLinearAcceleration)
{
  const Eigen::Vector3d accel(0.5, 0.0, 0.0);
  const Window start(restingForce(0.0) + accel, Eigen::Vector3d::Zero(), accel);
  const LevellingWindowReference ref = start.read();
  ASSERT_TRUE(ref.valid);
  EXPECT_FALSE(ref.parked);
  EXPECT_NEAR((ref.up_body - Eigen::Vector3d::UnitZ()).norm(), 0.0, 1e-12);
  LevellingWindowGates uncompensated = m20Gates();
  uncompensated.lin_accel_comp = false;
  const LevellingWindowReference raw = start.read(uncompensated);
  ASSERT_TRUE(raw.valid);
  EXPECT_NEAR(std::acos(raw.up_body.z()), leanOfAcceleration(accel, kG), 1e-12);
  const Window stop(restingForce(0.0) - accel, accel, Eigen::Vector3d::Zero());
  ASSERT_TRUE(stop.read().valid);
  EXPECT_FALSE(stop.read().parked);
}

TEST(GravityAlignKinematics, WindowNeedsTheBlockCoverage)
{
  Window w(restingForce(0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  w.span = 0.7;
  w.samples = 20;
  w.sum = restingForce(0.0) * 20.0;
  EXPECT_TRUE(w.read().valid);  // the block's limits themselves pass
  Window short_span = w;
  short_span.span = 0.69;
  EXPECT_FALSE(short_span.read().valid);
  Window few = w;
  few.samples = 19;
  few.sum = restingForce(0.0) * 19.0;
  EXPECT_FALSE(few.read().valid);
  Window empty = w;
  empty.samples = 0;
  empty.sum.setZero();
  EXPECT_FALSE(empty.read().valid);
  // a window length of zero keeps no window at all
  LevellingWindowGates no_window = m20Gates();
  no_window.window_s = 0.0;
  EXPECT_FALSE(w.read(no_window).valid);
}

// A lift accelerating vertically changes only the magnitude of the specific force: outside the tolerance the window
// is no reference.
TEST(GravityAlignKinematics, WindowMagnitudeMustMatchTheReference)
{
  const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d parked = Eigen::Vector3d::Zero();
  EXPECT_TRUE(Window(up * kG * 1.04, parked, parked).read().valid);
  EXPECT_FALSE(Window(up * kG * 1.06, parked, parked).read().valid);
  EXPECT_FALSE(Window(up * kG * 0.94, parked, parked).read().valid);
  LevellingWindowGates no_reference = m20Gates();
  no_reference.magnitude_ref = 0.0;
  EXPECT_FALSE(Window(up * kG, parked, parked).read(no_reference).valid);
}

// A leg edge the bridge could not measure leaves no linear-acceleration term. With the leg stream alive the block
// withholds such a window, and so does the reference; with no leg stream at all the block keeps the window
// uncompensated, which is then valid but never parked.
TEST(GravityAlignKinematics, WindowWithoutALegEdgeIsInvalid)
{
  Window w(restingForce(0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  w.newest.velocity_overwritten = true;
  EXPECT_FALSE(w.read().valid);
  LevellingWindowGates no_stream = m20Gates();
  no_stream.leg_stream_alive = false;
  const LevellingWindowReference legacy = w.read(no_stream);
  EXPECT_TRUE(legacy.valid);
  EXPECT_FALSE(legacy.parked);
  // leg edges do not come from the filter's velocity: an overwrite inside the window leaves them usable, while the
  // filter's own velocity edges have no derivative across it
  Window inside(restingForce(0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  inside.overwritten_inside = true;
  EXPECT_TRUE(inside.read().parked);
  LevellingWindowGates filter_edges = m20Gates();
  filter_edges.leg_edges = false;
  EXPECT_FALSE(inside.read(filter_edges).valid);
}

// Parked needs edges that can tell moving from parked: the filter's velocity edges carry a variance of 0 whenever
// the compensation sigma is off, a leg sigma of 0 gives every leg edge 0, and without the compensation there are
// no edges.
TEST(GravityAlignKinematics, WindowIsParkedOnlyOnLegEdgesThatCanTellMoving)
{
  const Window w(restingForce(0.0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  ASSERT_TRUE(w.read().parked);
  LevellingWindowGates filter_edges = m20Gates();
  filter_edges.leg_edges = false;
  LevellingWindowGates zero_sigma = m20Gates();
  zero_sigma.leg_vel_sigma = 0.0;
  LevellingWindowGates uncompensated = m20Gates();
  uncompensated.lin_accel_comp = false;
  for (const LevellingWindowGates & gates : {filter_edges, zero_sigma, uncompensated}) {
    const LevellingWindowReference ref = w.read(gates);
    EXPECT_TRUE(ref.valid);
    EXPECT_FALSE(ref.parked);
    EXPECT_DOUBLE_EQ(ref.variance, 0.0);
  }
}

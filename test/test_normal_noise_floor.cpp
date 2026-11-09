// test_normal_noise_floor.cpp — TLS covariance, complete S0, the joint-row contraction,
// independent information/shift oracles, and the three modes with schema-2 telemetry.

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "degeneracy_policy.hpp"
#include "normal_noise_floor.hpp"
#include "plane_fit.hpp"

using fast_lio::applyFloorProjection;
using fast_lio::FloorAccumulator;
using fast_lio::floorKeepGains;
using fast_lio::FloorMode;
using fast_lio::FloorParams;
using fast_lio::FloorProjectionResult;
using fast_lio::PlaneFitEigen;
using fast_lio::planeNormalCovariance;
using fast_lio::projectionFromGains;
using fast_lio::zFromFalseKeepRate;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
// z(1e-3) of the design (A.3); the margin factor 1 + z * 0.3 is the "c = 1.927 f" of A.4.
constexpr double kZ = 3.090232306167813;
// The Schur ridge the front end runs with (degeneracy_schur_ridge).
const double kRidge = fast_lio::DegeneracyParams{}.schur_ridge;

struct Point
{
  double x, y, z;
};

PlaneFitEigen handBuilt(double lambda0, double lambda1, double lambda2, int count = 5)
{
  PlaneFitEigen fit;
  fit.lambda0 = lambda0;
  fit.lambda1 = lambda1;
  fit.lambda2 = lambda2;
  fit.e1 = Eigen::Vector3d::UnitX();
  fit.e2 = Eigen::Vector3d::UnitY();
  fit.count = count;
  return fit;
}

Eigen::Matrix3d orthonormalBasis()
{
  // The production path takes V from SelfAdjointEigenSolver of M_eff: do the same here.
  Eigen::Matrix3d a;
  a << 4.0, 1.0, 0.5, 1.0, 3.0, -0.7, 0.5, -0.7, 2.0;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(a);
  return solver.eigenvectors();
}

// Uniform in (0, 1) from 53 random bits, and Box-Muller: the same draws on every standard library.
double uniform(std::mt19937_64 & rng)
{
  return (static_cast<double>(rng() >> 11) + 0.5) / 9007199254740992.0;
}

double gaussian(std::mt19937_64 & rng)
{
  const double u1 = uniform(rng);
  const double u2 = uniform(rng);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

// One scan's point-to-plane rows as the front end builds them: h_x N x 12 (position 0-2, rotation 3-5,
// extrinsics 6-11), the residual h, each row's weight and its plane's normal covariance.
struct Rows
{
  Eigen::MatrixXd h_x;
  Eigen::VectorXd h;
  std::vector<double> weights;
  std::vector<std::optional<Eigen::Matrix3d>> covariances;
};

// A corridor along x: a floor 1 m below the sensor on a 40 x 4 grid and walls 2 m to either side, 240 patches,
// each the five-point fit of design A (2 cm rms over 0.4 m), the sensor 5 m off the middle so that position and
// rotation are coupled. Each normal is tilted along the corridor by tilt_scale times a Gaussian draw of its own
// standard deviation there, as fit noise would; that tilt is all the along-corridor information the scan has. The
// residual reads a position error (5, -3, 2) cm and a rotation error (2, 10, -4) mrad through the rows, plus
// Gaussian noise; every draw is seeded. Without floor every patch is coplanar: same rows, no normal covariance.
Rows corridorRows(double tilt_scale, double noise, bool with_floor = true)
{
  struct Patch
  {
    Eigen::Vector3d p, n, e1, e2;
  };
  const Eigen::Vector3d X = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d Y = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d Z = Eigen::Vector3d::UnitZ();
  std::vector<Patch> patches;
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 4; ++j) {
      patches.push_back({Eigen::Vector3d(-14.5 + i, -1.5 + j, -1.0), Z, X, Y});
    }
  }
  for (int i = 0; i < 20; ++i) {
    for (double y : {-2.0, 2.0}) {
      for (double z : {-0.5, 0.5}) {
        patches.push_back({Eigen::Vector3d(-14.0 + 2.0 * i, y, z), Eigen::Vector3d(0.0, -y / 2.0, 0.0), X, Z});
      }
    }
  }
  const Eigen::Vector3d position_error(0.05, -0.03, 0.02);
  const Eigen::Vector3d rotation_error(0.002, 0.01, -0.004);
  const double lambda0 = 0.002;
  const double lambda1 = 0.16;
  const double sigma_tilt = std::sqrt(lambda0 / 2.0 / lambda1);  // Cov(n) along e1 is sigma^2 / lambda1
  std::mt19937_64 rng(20261108);
  const int count = static_cast<int>(patches.size());
  Rows rows;
  rows.h_x = Eigen::MatrixXd::Zero(count, 12);
  rows.h.resize(count);
  for (int i = 0; i < count; ++i) {
    const Patch & patch = patches[i];
    PlaneFitEigen fit;
    fit.lambda0 = with_floor ? lambda0 : 0.0;
    fit.lambda1 = lambda1;
    fit.lambda2 = 0.25;
    fit.e1 = patch.e1;
    fit.e2 = patch.e2;
    fit.count = 5;
    const double tilt = tilt_scale * sigma_tilt * gaussian(rng);
    const Eigen::Vector3d n = (patch.n + tilt * patch.e1).normalized();
    const Eigen::Vector3d rotation = patch.p.cross(n);
    rows.h_x.block<1, 3>(i, 0) = n.transpose();
    rows.h_x.block<1, 3>(i, 3) = rotation.transpose();
    rows.h(i) = n.dot(position_error) + rotation.dot(rotation_error) + noise * gaussian(rng);
    rows.weights.push_back(1.0);
    rows.covariances.push_back(planeNormalCovariance(fit));
  }
  return rows;
}

FloorProjectionResult project(Rows & rows, const FloorParams & params = FloorParams{}, double ridge = kRidge)
{
  return applyFloorProjection(rows.h_x, rows.h, rows.weights, rows.covariances, params, ridge);
}

template<typename Derived>
bool sameBits(const Eigen::PlainObjectBase<Derived> & a, const Eigen::PlainObjectBase<Derived> & b)
{
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), sizeof(double) * static_cast<std::size_t>(a.size())) == 0;
}
}  // namespace

TEST(PlaneNormalCovariance, MatchesTheFirstOrderFormulaOnAHandBuiltFit)
{
  // sigma^2 = lambda0 / (Count - 3) = 0.02 / 2; Cov = sigma^2 (e1 e1^T / 1 + e2 e2^T / 4).
  const auto cov = planeNormalCovariance(handBuilt(0.02, 1.0, 4.0));
  ASSERT_TRUE(cov.has_value());
  const Eigen::Matrix3d expected = Eigen::Vector3d(0.01, 0.0025, 0.0).asDiagonal();
  EXPECT_LT((*cov - expected).norm(), 1e-16);
  // The design's "dof /2" row (calc.out:20): 2 cm rms residual over 5 points (lambda0 = 5 * 0.02^2) on a
  // 0.4 m patch (in-plane scatter 5 * 0.2^2 / 2 = 0.1 per axis) tilts the normal by 5.7 degrees.
  const auto design = planeNormalCovariance(handBuilt(5 * 0.02 * 0.02, 0.1, 0.1));
  ASSERT_TRUE(design.has_value());
  const double tilt_deg = std::sqrt((*design)(0, 0)) * 180.0 / 3.14159265358979323846;
  EXPECT_NEAR(tilt_deg, 5.7, 0.05);
  // ... and the out-of-plane direction carries no first-order variance.
  EXPECT_EQ((*design)(2, 2), 0.0);
}

TEST(PlaneNormalCovariance, CoplanarPointsGiveNoFloor)
{
  const auto cov = planeNormalCovariance(handBuilt(0.0, 0.5, 2.0));
  ASSERT_TRUE(cov.has_value());
  EXPECT_EQ(*cov, Eigen::Matrix3d::Zero());
  // A negative lambda0 within the fit's rank tolerance is rounding of a coplanar patch.
  const double rounding = std::numeric_limits<double>::epsilon() * 5 * 2.0;
  const auto rounded = planeNormalCovariance(handBuilt(-0.5 * rounding, 0.5, 2.0));
  ASSERT_TRUE(rounded.has_value());
  EXPECT_EQ(*rounded, Eigen::Matrix3d::Zero());

  FloorAccumulator accumulator;
  accumulator.add(1.0, cov);
  accumulator.add(3.0, rounded);
  EXPECT_EQ(accumulator.matrix(), Eigen::Matrix3d::Zero());
  EXPECT_EQ(accumulator.pointsWithoutCovariance(), 0);

  // The same through fitPlane: five exactly coplanar points.
  const std::vector<Point> points{{-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0}, {0, 0, 0}};
  Eigen::Vector4d coefficients;
  PlaneFitEigen fit;
  ASSERT_TRUE(fast_lio::fitPlane<5>(points, 0.1, coefficients, &fit));
  const auto fitted = planeNormalCovariance(fit);
  ASSERT_TRUE(fitted.has_value());
  EXPECT_LT(fitted->norm(), 1e-15);  // lambda0 is at most the rounding of an exact zero
}

TEST(PlaneNormalCovariance, RefusesFitsThatCannotCarryOne)
{
  EXPECT_FALSE(planeNormalCovariance(PlaneFitEigen{}).has_value());               // never filled: count 0
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, 1.0, 4.0, 3)).has_value());  // no residual dof
  EXPECT_FALSE(planeNormalCovariance(handBuilt(kNaN, 1.0, 4.0)).has_value());
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, kNaN, 4.0)).has_value());
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, 1.0, kInf)).has_value());
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, 0.0, 4.0)).has_value());  // no in-plane spread
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, -1.0, 4.0)).has_value());
  EXPECT_FALSE(planeNormalCovariance(handBuilt(0.02, 4.0, 1.0)).has_value());  // out of order
  EXPECT_FALSE(planeNormalCovariance(handBuilt(2.0, 1.0, 4.0)).has_value());
  EXPECT_FALSE(planeNormalCovariance(handBuilt(-1e-3, 1.0, 4.0)).has_value());  // not a scatter matrix
  PlaneFitEigen bad_vector = handBuilt(0.02, 1.0, 4.0);
  bad_vector.e2(1) = kNaN;
  EXPECT_FALSE(planeNormalCovariance(bad_vector).has_value());
}

TEST(PlaneNormalCovariance, AgreesWithTheSpreadOfFittedNormalsUnderNoise)
{
  // A 0.6 m x 0.3 m five-point patch (lambda = 0.36 along x, 0.09 along y), 1 cm Gaussian noise along the
  // normal. First order: Var(n_x) = sigma^2 / 0.36, Var(n_y) = sigma^2 / 0.09, and E[lambda0 / (5 - 3)] =
  // sigma^2. Tolerances are five standard errors of K samples: sqrt(2 / K) for a variance, sqrt(1 / K) for
  // the mean of lambda0 / 2 (a chi-square with two degrees of freedom over two). Second-order terms are
  // (sigma^2 / lambda) ~ 1e-3 relative, well inside.
  const std::array<Point, 5> patch{{{-0.3, -0.15, 0}, {-0.3, 0.15, 0}, {0.3, -0.15, 0}, {0.3, 0.15, 0}, {0, 0, 0}}};
  const double sigma = 0.01;
  const int samples = 20000;
  std::mt19937_64 rng(20261108);
  Eigen::Matrix3d spread = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d predicted = Eigen::Matrix3d::Zero();
  double sigma2_sum = 0.0;
  for (int k = 0; k < samples; ++k) {
    std::array<Point, 5> noisy = patch;
    for (auto & point : noisy) {
      point.z += sigma * gaussian(rng);
    }
    Eigen::Vector4d coefficients;
    PlaneFitEigen fit;
    ASSERT_TRUE(fast_lio::fitPlane<5>(noisy, 1.0, coefficients, &fit));
    Eigen::Vector3d n = coefficients.head<3>();
    if (n.z() < 0.0) {
      n = -n;
    }
    const Eigen::Vector3d tilt = n - Eigen::Vector3d::UnitZ();
    spread += tilt * tilt.transpose();
    const auto cov = planeNormalCovariance(fit);
    ASSERT_TRUE(cov.has_value());
    predicted += *cov;
    sigma2_sum += fit.lambda0 / 2.0;
  }
  spread /= samples;
  predicted /= samples;
  const double variance_tolerance = 5.0 * std::sqrt(2.0 / samples);
  const double mean_tolerance = 5.0 * std::sqrt(1.0 / samples);
  const double s2 = sigma * sigma;
  EXPECT_NEAR(spread(0, 0) / (s2 / 0.36), 1.0, variance_tolerance);
  EXPECT_NEAR(spread(1, 1) / (s2 / 0.09), 1.0, variance_tolerance);
  EXPECT_NEAR(sigma2_sum / samples / s2, 1.0, mean_tolerance);
  // The formula as each fit reports it, averaged, against the true covariance.
  EXPECT_NEAR(predicted(0, 0) / (s2 / 0.36), 1.0, mean_tolerance);
  EXPECT_NEAR(predicted(1, 1) / (s2 / 0.09), 1.0, mean_tolerance);
  // The patch's x and y tilts are independent (sum x_i y_i = 0): the formula's zero cross term, within five
  // standard errors sqrt(Var_x Var_y / K) of the sample cross-covariance.
  EXPECT_LT(std::abs(spread(0, 1)), mean_tolerance * std::sqrt(s2 / 0.36 * s2 / 0.09));
}

TEST(FloorAccumulator, AddsTheCompleteWeightedCovariance)
{
  // Along the axes: f_k = kappa sum w^2 C_kk.
  FloorAccumulator axes;
  axes.add(0.5, Eigen::Matrix3d(Eigen::Vector3d(4e-3, 2e-3, 0.0).asDiagonal()));
  axes.add(2.0, Eigen::Matrix3d(Eigen::Vector3d(0.0, 1e-3, 3e-3).asDiagonal()));
  // Independent r2 numerical-d1.txt:L33, including the original 1e-18 tolerance.
  const Eigen::Matrix3d expected = Eigen::Vector3d(0.25 * 4e-3, 0.25 * 2e-3 + 4.0 * 1e-3, 4.0 * 3e-3).asDiagonal();
  EXPECT_LT((axes.floor(1.0) - expected).norm(), 1e-18);
  EXPECT_LT((axes.floor(1.5) - 1.5 * expected).norm(), 1e-18);
  EXPECT_EQ(axes.points(), 2);
  EXPECT_EQ(axes.pointsWithoutCovariance(), 0);

  // For any orthonormal V the floors add up to kappa sum w^2 trace(C): nothing is lost or counted twice.
  FloorAccumulator rotated;
  const auto cov_a = planeNormalCovariance(handBuilt(0.02, 1.0, 4.0));
  PlaneFitEigen tilted = handBuilt(0.004, 0.2, 0.5);
  tilted.e1 = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
  tilted.e2 = Eigen::Vector3d(-1.0, 1.0, 1.0).normalized();
  const auto cov_b = planeNormalCovariance(tilted);
  ASSERT_TRUE(cov_a && cov_b);
  rotated.add(0.7, cov_a);
  rotated.add(1.3, cov_b);
  const double trace = 0.49 * cov_a->trace() + 1.69 * cov_b->trace();
  EXPECT_NEAR(rotated.floor(2.0).trace(), 2.0 * trace, 1e-15);
  EXPECT_NEAR(rotated.matrix()(0, 1), 0.49 * (*cov_a)(0, 1) + 1.69 * (*cov_b)(0, 1), 1e-18);
  EXPECT_NE(rotated.matrix()(0, 1), 0.0);
}

TEST(FloorAccumulator, PointsWithoutAKnownContributionAreCountedNotGuessed)
{
  FloorAccumulator accumulator;
  const auto cov = planeNormalCovariance(handBuilt(0.02, 1.0, 4.0));
  accumulator.add(1.0, cov);
  const Eigen::Matrix3d known = accumulator.matrix();
  accumulator.add(1.0, std::nullopt);
  accumulator.add(kNaN, cov);
  accumulator.add(kInf, cov);
  Eigen::Matrix3d broken = *cov;
  broken(1, 1) = kNaN;
  accumulator.add(1.0, broken);
  EXPECT_EQ(accumulator.points(), 5);
  EXPECT_EQ(accumulator.pointsWithoutCovariance(), 4);
  EXPECT_EQ(accumulator.floor(1.0), known);
}

TEST(FloorKeepGains, ExactZeroUpToTheMarginAndTheDesignsKeptFractions)
{
  const double c = 25.0 * (1.0 + kZ * 0.3);  // what is deducted from a direction whose floor is 25
  for (double lambda : {1.0, 25.0, 48.0, c}) {
    const Eigen::Vector3d g =
      floorKeepGains(Eigen::Vector3d::Constant(lambda), Eigen::Vector3d::Constant(25.0), kZ, 0.3);
    EXPECT_EQ(g, Eigen::Vector3d::Zero()) << "lambda " << lambda;  // exactly 0.0, not a small number
  }
  EXPECT_GT(floorKeepGains(Eigen::Vector3d::Constant(c * (1.0 + 1e-12)), Eigen::Vector3d::Constant(25.0), kZ, 0.3)(0),
            0.0);
  // calc.out:28-30: lambda 1000, 780, 100 with f = 25 keep 95.2 %, 93.8 %, 51.8 % of the information (g^2).
  const Eigen::Vector3d g =
    floorKeepGains(Eigen::Vector3d(1000.0, 780.0, 100.0), Eigen::Vector3d::Constant(25.0), kZ, 0.3);
  EXPECT_NEAR(100.0 * g(0) * g(0), 95.2, 0.05);
  EXPECT_NEAR(100.0 * g(1) * g(1), 93.8, 0.05);
  EXPECT_NEAR(100.0 * g(2) * g(2), 51.8, 0.05);
  // Without a floor every direction with information is kept whole.
  EXPECT_EQ(floorKeepGains(Eigen::Vector3d(1e-9, 1.0, 1e9), Eigen::Vector3d::Zero(), kZ, 0.3), Eigen::Vector3d::Ones());
}

TEST(FloorKeepGains, MonotoneInInformationAndInTheFloor)
{
  const double f = 1.5;
  double previous = 0.0;
  for (int i = 0; i <= 20000; ++i) {
    const double lambda = 1e-3 * i;  // 0 ... 20 = 13.3 f, densest around the onset at 1.927 f
    const double g = floorKeepGains(Eigen::Vector3d::Constant(lambda), Eigen::Vector3d::Constant(f), kZ, 0.3)(0);
    ASSERT_GE(g, previous) << "lambda " << lambda;
    ASSERT_LE(g, 1.0);
    previous = g;
  }
  previous = 1.0;
  for (int i = 0; i <= 20000; ++i) {
    const double noise_floor = 1e-3 * i;
    const double g =
      floorKeepGains(Eigen::Vector3d::Constant(10.0), Eigen::Vector3d::Constant(noise_floor), kZ, 0.3)(0);
    ASSERT_LE(g, previous) << "floor " << noise_floor;
    ASSERT_GE(g, 0.0);
    previous = g;
  }
}

TEST(FloorKeepGains, NoInformationOrUnusableInputsKeepNothing)
{
  const Eigen::Vector3d noise_floor(0.1, 0.1, 0.1);
  EXPECT_EQ(floorKeepGains(Eigen::Vector3d(0.0, -1.0, kNaN), noise_floor, kZ, 0.3), Eigen::Vector3d::Zero());
  EXPECT_EQ(floorKeepGains(Eigen::Vector3d(kInf, -kInf, 0.0), noise_floor, kZ, 0.3), Eigen::Vector3d::Zero());
  const Eigen::Vector3d lambda(10.0, 10.0, 10.0);
  EXPECT_EQ(floorKeepGains(lambda, Eigen::Vector3d(-0.1, kNaN, kInf), kZ, 0.3), Eigen::Vector3d::Zero());
  EXPECT_EQ(floorKeepGains(lambda, noise_floor, kNaN, 0.3), Eigen::Vector3d::Zero());
  EXPECT_EQ(floorKeepGains(lambda, noise_floor, kZ, -0.3), Eigen::Vector3d::Zero());
  EXPECT_EQ(floorKeepGains(lambda, noise_floor, kInf, 0.3), Eigen::Vector3d::Zero());
  // One bad direction does not take the others with it.
  const Eigen::Vector3d mixed = floorKeepGains(Eigen::Vector3d(kNaN, 10.0, 10.0), noise_floor, kZ, 0.3);
  EXPECT_EQ(mixed(0), 0.0);
  EXPECT_GT(mixed(1), 0.9);
  EXPECT_EQ(mixed(1), mixed(2));
}

TEST(ProjectionFromGains, ScalesEachEigenvectorAndRemovesAZeroGainExactly)
{
  const Eigen::Matrix3d V = orthonormalBasis();
  // Gains that are not powers of two, so that the order of the products shows in the rounding.
  const Eigen::Vector3d gains(0.0, 0.3, 0.9);
  const Eigen::Matrix3d G = projectionFromGains(V, gains);
  EXPECT_EQ(G, G.transpose());  // exactly symmetric
  // Tolerances are a few ulps of V's orthonormality (1e-16 per product in practice).
  for (int k = 0; k < 3; ++k) {
    EXPECT_LT((G * V.col(k) - gains(k) * V.col(k)).norm(), 1e-14) << "k " << k;
  }
  // Design D.2: projected rows n^T G carry nothing along the removed direction, down to the
  // orthonormality of V (~1e-16 |n|; velocity is pinned only above 3.2e-9 normals per frame, calc.out:17).
  std::mt19937_64 rng(7);
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector3d n = Eigen::Vector3d(gaussian(rng), gaussian(rng), gaussian(rng)).normalized();
    EXPECT_LT(std::abs(n.transpose() * G * V.col(0)), 1e-14);
  }
  EXPECT_LT((projectionFromGains(V, Eigen::Vector3d::Ones()) - Eigen::Matrix3d::Identity()).norm(), 1e-14);
  EXPECT_EQ(projectionFromGains(V, Eigen::Vector3d::Zero()), Eigen::Matrix3d::Zero());
}

TEST(ZFromFalseKeepRate, IsTheUpperTailQuantileOfTheStandardNormal)
{
  const auto z = zFromFalseKeepRate(1e-3);
  ASSERT_TRUE(z.has_value());
  EXPECT_NEAR(*z, 3.0902, 1e-4);  // design A.3
  EXPECT_NEAR(*z, kZ, 1e-12);
  for (double alpha : {0.4, 0.1, 0.05, 1e-2, 1e-3, 1e-4, 1e-6, 1e-12}) {
    const auto quantile = zFromFalseKeepRate(alpha);
    ASSERT_TRUE(quantile.has_value()) << alpha;
    EXPECT_NEAR(0.5 * std::erfc(*quantile / std::sqrt(2.0)) / alpha, 1.0, 1e-12) << alpha;
  }
  for (double alpha : {0.0, 0.5, 0.7, 1.0, -1e-3, kNaN, kInf}) {
    EXPECT_FALSE(zFromFalseKeepRate(alpha).has_value()) << alpha;
  }
}

TEST(FloorParams, DefaultsAreTheDesignValues)
{
  const FloorParams params;
  EXPECT_EQ(params.alpha, 1e-3);
  EXPECT_EQ(params.kappa, 1.0);
  EXPECT_EQ(params.rel_uncertainty, 0.3);
  const auto z = zFromFalseKeepRate(params.alpha);
  ASSERT_TRUE(z.has_value());
  EXPECT_NEAR(1.0 + *z * params.rel_uncertainty, 1.927, 5e-4);  // design A.4: c = 1.927 f
}

TEST(FloorResidualShift, AScanWithoutFloorKeepsItsResidualBitForBit)
{
  // r2 numerical-d1.txt:L38. Zero S has priority even on a rank-deficient scan.
  for (double tilt : {0.0, std::sqrt(0.5)}) {
    const Rows original = corridorRows(tilt, 0.005, false);
    Rows rows = original;
    const auto result = project(rows);
    ASSERT_TRUE(result.valid);
    EXPECT_STREQ(result.reason, "zero_floor");
    EXPECT_EQ(result.S0, Eigen::Matrix3d::Zero());
    EXPECT_EQ(result.d_t, Eigen::Vector3d::Zero());
    EXPECT_EQ(result.d_r, Eigen::Vector3d::Zero());
    EXPECT_EQ(result.dh_norm, 0.0);
    EXPECT_TRUE(sameBits(rows.h, original.h));
    EXPECT_TRUE(sameBits(rows.h_x, original.h_x));
  }
}

TEST(ApplyFloorProjection, TheShiftedRowsReadNothingAlongTheRemovedJointMode)
{
  // Independent r2 numerical-d1.txt:L39 / numerical_d1.py:440.
  // R36 withdraws the original pure-position/unchanged-extrinsic assertions.
  Rows original = corridorRows(std::sqrt(0.5), 0.005);
  original.covariances[7] = std::nullopt;
  Rows rows = original;
  const auto result = project(rows);
  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.shift_coords_valid);
  EXPECT_EQ(result.points, 240);
  EXPECT_EQ(result.points_without_covariance, 1);
  EXPECT_LT((result.d_t - Eigen::Vector3d(0.05005830761725314, 0.0001378432894548538, -0.00011018954792048842)).norm(),
            1e-10);
  EXPECT_LT(
    (result.d_r - Eigen::Vector3d(-0.00019813140066665571, -2.614213481652736e-06, 1.6477877902155545e-05)).norm(),
    1e-10);
  EXPECT_NEAR(result.dh_norm, 0.041811816569320696, 1e-10);
  EXPECT_LT((rows.h_x.middleCols(3, 3) - original.h_x.middleCols(3, 3)).norm(), 1e-12);
  const Eigen::VectorXd delta = original.h_x.leftCols(3) * result.d_t + original.h_x.middleCols(3, 3) * result.d_r;
  EXPECT_LT((original.h - rows.h - delta).norm(), 1e-12);
  Eigen::Matrix<double, 6, 6> expected = original.h_x.leftCols(6).transpose() * original.h_x.leftCols(6);
  expected.topLeftCorner<3, 3>() -= result.M0 - result.K;
  EXPECT_LT((rows.h_x.leftCols(6).transpose() * rows.h_x.leftCols(6) - expected).norm(), 1e-10 * expected.norm());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> kept(result.K);
  const Eigen::Vector3d u = kept.eigenvectors().col(0);
  Eigen::Matrix<double, 6, 1> q;
  q.head<3>() = u;
  q.tail<3>() = -result.C.ldlt().solve(result.B.transpose() * u);
  EXPECT_LT((rows.h_x.leftCols(6) * q).norm(), 1e-12 * original.h_x.norm());
}

TEST(FloorResidualShift, ADirectionWithoutInformationIsRefusedAtomically)
{
  // R37 / r2 numerical-d1.txt:L40. The old support-extension shift is withdrawn.
  const Rows original = corridorRows(0.0, 0.005);
  Rows rows = original;
  const auto result = project(rows);
  EXPECT_FALSE(result.valid);
  EXPECT_STREQ(result.reason, "rank_m");
  EXPECT_LT(result.rank_m, 3);
  EXPECT_TRUE(sameBits(rows.h_x, original.h_x));
  EXPECT_TRUE(sameBits(rows.h, original.h));
}

TEST(FloorResidualShift, NonFiniteInputsRefuseTheWholeUpdate)
{
  // r2 numerical-d1.txt:L37. Invalid matrices do not masquerade as healthy gains.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  const auto refused = [](Rows rows, const FloorParams & params, double ridge) {
    const Rows given = rows;
    const auto result = project(rows, params, ridge);
    return !result.valid && !result.shift_coords_valid && result.dh_norm == 0.0 && sameBits(rows.h_x, given.h_x) &&
           sameBits(rows.h, given.h);
  };
  for (double bad : {kNaN, kInf, -kInf}) {
    for (int col = 0; col < 12; ++col) {
      Rows broken = original;
      broken.h_x(5, col) = bad;
      EXPECT_TRUE(refused(broken, FloorParams{}, kRidge));
    }
    Rows broken = original;
    broken.h(3) = bad;
    EXPECT_TRUE(refused(broken, FloorParams{}, kRidge));
    FloorParams params;
    params.kappa = bad;
    EXPECT_TRUE(refused(original, params, kRidge));
    params = FloorParams{};
    params.rel_uncertainty = bad;
    EXPECT_TRUE(refused(original, params, kRidge));
    EXPECT_TRUE(refused(original, FloorParams{}, bad));
  }
  FloorParams params;
  params.alpha = 0.0;
  EXPECT_TRUE(refused(original, params, kRidge));
  params = FloorParams{};
  params.rel_uncertainty = -0.1;
  EXPECT_TRUE(refused(original, params, kRidge));
  Rows rows = original;
  Eigen::MatrixXd narrow = rows.h_x.leftCols(11);
  EXPECT_THROW(applyFloorProjection(narrow, rows.h, rows.weights, rows.covariances, FloorParams{}, kRidge),
               std::invalid_argument);
  Eigen::VectorXd short_h = rows.h.head(rows.h.size() - 1);
  EXPECT_THROW(applyFloorProjection(rows.h_x, short_h, rows.weights, rows.covariances, FloorParams{}, kRidge),
               std::invalid_argument);
  rows.weights.push_back(1.0);
  EXPECT_THROW(project(rows), std::invalid_argument);
  rows = original;
  rows.covariances.pop_back();
  EXPECT_THROW(project(rows), std::invalid_argument);
}

// --- the lidar update's modes (degeneracy_floor_mode), the scan's record and its [FLOOR] line ---------------------

TEST(FloorMode, ParsesTheThreeNamesAndRefusesAnyOther)
{
  EXPECT_TRUE(fast_lio::floorModeFromName("off") == FloorMode::kOff);
  EXPECT_TRUE(fast_lio::floorModeFromName("observe") == FloorMode::kObserve);
  EXPECT_TRUE(fast_lio::floorModeFromName("apply") == FloorMode::kApply);
  for (const char * name : {"", "Apply", "OFF", "on", "true", "apply "}) {
    EXPECT_FALSE(fast_lio::floorModeFromName(name).has_value()) << '"' << name << '"';
  }
  for (const FloorMode mode : {FloorMode::kOff, FloorMode::kObserve, FloorMode::kApply}) {
    EXPECT_TRUE(fast_lio::floorModeFromName(fast_lio::floorModeName(mode)) == mode);
  }
}

TEST(FloorStep, OffComputesNothingAndLeavesTheRowsAndTheRecord)
{
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  Rows rows = original;
  fast_lio::FloorScanRecord record;
  fast_lio::floorStep(FloorMode::kOff, rows.h_x, rows.h, rows.weights, rows.covariances, FloorParams{}, kRidge, record);
  EXPECT_TRUE(sameBits(rows.h_x, original.h_x));
  EXPECT_TRUE(sameBits(rows.h, original.h));
  EXPECT_EQ(record.evaluations, 0);
  EXPECT_EQ(record.invalid_evaluations, 0);
  EXPECT_FALSE(record.projected);
  EXPECT_FALSE(record.last.valid);
}

TEST(FloorStep, ObserveReachesTheDecisionApplyActsOnAndLeavesTheRowsBitForBit)
{
  // F0's kappa observation: lambda, f and g of the evaluation, while the update runs on the rows as built. On the
  // noisy corridor the same decision applied removes the axis (g = 0) and does change the rows.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  Rows observed = original;
  Rows applied = original;
  fast_lio::FloorScanRecord observe;
  fast_lio::FloorScanRecord apply;
  fast_lio::floorStep(FloorMode::kObserve,
                      observed.h_x,
                      observed.h,
                      observed.weights,
                      observed.covariances,
                      FloorParams{},
                      kRidge,
                      observe);
  fast_lio::floorStep(
    FloorMode::kApply, applied.h_x, applied.h, applied.weights, applied.covariances, FloorParams{}, kRidge, apply);

  EXPECT_TRUE(sameBits(observed.h_x, original.h_x));
  EXPECT_TRUE(sameBits(observed.h, original.h));
  EXPECT_FALSE(sameBits(applied.h_x, original.h_x));
  ASSERT_TRUE(observe.last.valid);
  ASSERT_TRUE(apply.last.valid);
  EXPECT_FALSE(observe.projected);
  EXPECT_TRUE(apply.projected);
  EXPECT_FALSE(observe.last.shift_coords_valid);
  EXPECT_EQ(observe.last.gains(2), 0.0);  // gains are ordered by ascending tau
  EXPECT_EQ(observe.last.points, apply.last.points);
  EXPECT_EQ(observe.last.points_without_covariance, apply.last.points_without_covariance);
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(observe.last.lambda(k), apply.last.lambda(k), 1e-12 * apply.last.lambda.maxCoeff()) << k;
    EXPECT_NEAR(observe.last.gains(k), apply.last.gains(k), 1e-12) << k;
  }
  EXPECT_LT((observe.last.M0 - apply.last.M0).norm(), 1e-12);
  EXPECT_LT((observe.last.M_ridge - apply.last.M_ridge).norm(), 1e-12);
  EXPECT_LT((observe.last.S0 - apply.last.S0).norm(), 1e-12);
  EXPECT_LT((observe.last.K - apply.last.K).norm(), 1e-12);
  EXPECT_LT((observe.last.tau - apply.last.tau).norm(), 1e-12);
  EXPECT_LT((observe.last.F_sym - apply.last.F_sym).norm(), 1e-12);
  EXPECT_EQ(observe.evaluations, 1);
  EXPECT_EQ(observe.invalid_evaluations, 0);
}

TEST(FloorStep, ARefusedEvaluationIsCountedAndLeavesTheRows)
{
  // valid false (here a non-finite residual) leaves the rows as built in either mode, and the scan's record counts
  // the refusal among its evaluations; the next good evaluation becomes the scan's last decision.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  for (const FloorMode mode : {FloorMode::kObserve, FloorMode::kApply}) {
    Rows bad = original;
    bad.h(3) = kNaN;
    const Rows bad_before = bad;
    fast_lio::FloorScanRecord record;
    fast_lio::floorStep(mode, bad.h_x, bad.h, bad.weights, bad.covariances, FloorParams{}, kRidge, record);
    EXPECT_TRUE(sameBits(bad.h_x, bad_before.h_x));
    EXPECT_TRUE(sameBits(bad.h, bad_before.h));
    EXPECT_FALSE(record.last.valid);
    EXPECT_FALSE(record.projected);
    EXPECT_EQ(record.evaluations, 1);
    EXPECT_EQ(record.invalid_evaluations, 1);

    Rows good = original;
    fast_lio::floorStep(mode, good.h_x, good.h, good.weights, good.covariances, FloorParams{}, kRidge, record);
    EXPECT_TRUE(record.last.valid);
    EXPECT_EQ(record.evaluations, 2);
    EXPECT_EQ(record.invalid_evaluations, 1);
    EXPECT_EQ(record.projected, mode == FloorMode::kApply);
  }
}

TEST(FloorStep, ApplyWithoutFloorIsNotAProjection)
{
  // Every gain exactly one: apply hands the update its rows bit for bit and the evaluation is not a projected one.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005, false);
  Rows rows = original;
  fast_lio::FloorScanRecord record;
  fast_lio::floorStep(
    FloorMode::kApply, rows.h_x, rows.h, rows.weights, rows.covariances, FloorParams{}, kRidge, record);
  ASSERT_TRUE(record.last.valid);
  EXPECT_FALSE(record.projected);
  EXPECT_TRUE(sameBits(rows.h_x, original.h_x));
  EXPECT_TRUE(sameBits(rows.h, original.h));
}

namespace
{
// Independent fixtures/oracles from r2 numerical_d1.py and numerical-d1.txt.
// All spectral roots below belong to the test oracle, never to the row-update API.
using Matrix6 = Eigen::Matrix<double, 6, 6>;

Eigen::MatrixXd psdRoot(const Eigen::MatrixXd & matrix)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  return solver.eigenvectors() * solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
         solver.eigenvectors().transpose();
}

Eigen::MatrixXd supportInverse(const Eigen::MatrixXd & matrix)
{
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  const double tolerance =
    std::numeric_limits<double>::epsilon() * matrix.rows() * solver.eigenvalues().cwiseAbs().maxCoeff();
  Eigen::VectorXd inverse = Eigen::VectorXd::Zero(matrix.rows());
  for (Eigen::Index i = 0; i < inverse.size(); ++i) {
    if (solver.eigenvalues()(i) > tolerance) {
      inverse(i) = 1.0 / solver.eigenvalues()(i);
    }
  }
  return solver.eigenvectors() * inverse.asDiagonal() * solver.eigenvectors().transpose();
}

Rows fixtureRows(const Eigen::MatrixXd & H6, const Eigen::Matrix3d & S)
{
  Rows rows;
  rows.h_x = Eigen::MatrixXd::Zero(H6.rows(), 12);
  rows.h_x.leftCols(H6.cols()) = H6;
  rows.h = Eigen::VectorXd::LinSpaced(H6.rows(), -0.3, 0.4);
  rows.weights.assign(static_cast<std::size_t>(H6.rows()), 1.0);
  // Supply the exact aggregate from the oracle, including its a*tau=1 boundary.
  // Splitting 1 into six rounded 1/6 contributions would change that input.
  rows.covariances.assign(static_cast<std::size_t>(H6.rows()), Eigen::Matrix3d::Zero());
  rows.covariances.front() = S;
  return rows;
}

Matrix6 handRows()
{
  // r2 numerical-d1.txt:L8 / py:197, explicit coupled six-row Jacobian.
  Matrix6 H = Matrix6::Identity();
  H(3, 0) = 1.0;
  H(3, 1) = 1.0;
  return H;
}

Rows orthogonalRows(const Eigen::Matrix3d & M, const Eigen::Matrix3d & S)
{
  Matrix6 H = Matrix6::Identity();
  H.topLeftCorner<3, 3>() = psdRoot(M);
  return fixtureRows(H, S);
}

FloorParams unitMargin()
{
  FloorParams params;
  params.rel_uncertainty = 0.0;
  return params;
}

Eigen::Matrix3d effectiveInformation(const Matrix6 & J, double precision)
{
  const Eigen::Matrix3d C = J.bottomRightCorner<3, 3>() + precision * Eigen::Matrix3d::Identity();
  return J.topLeftCorner<3, 3>() - J.topRightCorner<3, 3>() * C.ldlt().solve(J.bottomLeftCorner<3, 3>());
}

double maxEigenvalue(const Eigen::MatrixXd & matrix)
{
  return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(0.5 * (matrix + matrix.transpose())).eigenvalues().maxCoeff();
}

// Independent symmetric-root oracle for K, with no production decision/apply call.
Eigen::Matrix3d jointKeptOracle(const Eigen::Matrix3d & M, const Eigen::Matrix3d & S, double a)
{
  const Eigen::Matrix3d root = psdRoot(M);
  const Eigen::Matrix3d inverse = root.inverse();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(inverse * S * inverse);
  const Eigen::Vector3d kept = (Eigen::Vector3d::Ones() - a * solver.eigenvalues()).cwiseMax(0.0);
  return root * solver.eigenvectors() * kept.asDiagonal() * solver.eigenvectors().transpose() * root;
}

// Historical wrong coordinate rules, retained only as counterexample oracles.
Eigen::Matrix3d legacyCoordinateRule(const Eigen::Matrix3d & M, const Eigen::Matrix3d & S, double a, int kind)
{
  const Eigen::Matrix3d root = psdRoot(M);
  const Eigen::Matrix3d inverse = root.inverse();
  if (kind == 0) {
    const Eigen::Matrix3d T = inverse * S * inverse;
    return inverse * psdRoot(Eigen::Matrix3d::Identity() - a * T) * root;
  }
  if (kind == 1) {
    return inverse * psdRoot(root * jointKeptOracle(M, S, a) * root) * inverse;
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(M);
  Eigen::Vector3d gains;
  for (int k = 0; k < 3; ++k) {
    const Eigen::Vector3d v = solver.eigenvectors().col(k);
    gains(k) = std::sqrt(std::max(1.0 - a * v.dot(S * v) / solver.eigenvalues()(k), 0.0));
  }
  return solver.eigenvectors() * gains.asDiagonal() * solver.eigenvectors().transpose();
}

void expectJointCertificate(const Rows & original,
                            const Rows & updated,
                            const FloorProjectionResult & result,
                            double tolerance = 1e-10,
                            const FloorParams & params = FloorParams{})
{
  const Eigen::MatrixXd raw = original.h_x.transpose() * original.h_x;
  const Eigen::MatrixXd kept = updated.h_x.transpose() * updated.h_x;
  EXPECT_LE(maxEigenvalue(kept - raw), tolerance * std::max(1.0, raw.norm()));
  const Eigen::Matrix3d M =
    raw.topLeftCorner<3, 3>() - raw.block<3, 3>(0, 3) * supportInverse(raw.block<3, 3>(3, 3)) * raw.block<3, 3>(3, 0);
  Eigen::Matrix3d S0 = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < original.weights.size(); ++i) {
    if (original.covariances[i])
      S0 += original.weights[i] * original.weights[i] * *original.covariances[i];
  }
  const Eigen::Matrix3d K = jointKeptOracle(M, params.kappa * S0, 1.0 + kZ * params.rel_uncertainty);
  EXPECT_LE((result.M0 - M).norm(), tolerance * std::max(1.0, raw.norm()));
  EXPECT_LE((result.K - K).norm(), tolerance * std::max(1.0, raw.norm()));
  Matrix6 target = raw.topLeftCorner<6, 6>();
  target.topLeftCorner<3, 3>() -= M - K;
  EXPECT_LE((kept.topLeftCorner<6, 6>() - target).norm(), tolerance * std::max(1.0, target.norm()));
  for (double variance : {1e-2, 1e-4, 1e-6}) {
    EXPECT_LE(maxEigenvalue(effectiveInformation(kept.topLeftCorner<6, 6>(), 1.0 / variance) -
                            effectiveInformation(raw.topLeftCorner<6, 6>(), 1.0 / variance)),
              tolerance * std::max(1.0, raw.norm()));
  }
}
}  // namespace

TEST(JointFloor, CommutingCouplingCannotUseB1)
{
  // r2 txt:L7-L8 / py:184,197. Old B.1 invents (sqrt(13)-3)/4 information.
  Rows original = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  Matrix6 D = Matrix6::Identity();
  D(0, 0) = 0.0;
  const Matrix6 J = handRows().transpose() * handRows();
  EXPECT_NEAR(
    maxEigenvalue(effectiveInformation(D * J * D, 1.0) - effectiveInformation(J, 1.0)), 0.1513878188659973, 1e-12);
  Rows rows = original;
  const auto result = project(rows, unitMargin());
  ASSERT_TRUE(result.valid);
  Matrix6 expected = handRows();
  expected.row(0).setZero();
  EXPECT_LT((rows.h_x.leftCols(6) - expected).norm(), 1e-12);
  expectJointCertificate(original, rows, result, 1e-12, unitMargin());
}

TEST(JointFloor, ConfoundingDoesNotInventInformation)
{
  // r2 txt:L17-L26 / py:292-312. Fixed a=1.927 and the three prescribed priors.
  FloorParams params;
  params.rel_uncertainty = 0.927 / kZ;
  for (double e : {10.0, 0.001}) {
    const Eigen::Matrix3d A = 1e5 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d B = Eigen::Matrix3d::Zero();
    B(0, 0) = std::sqrt(1e5 * (1e5 - e));
    Matrix6 J = Matrix6::Zero();
    J.topLeftCorner<3, 3>() = A;
    J.topRightCorner<3, 3>() = B;
    J.bottomLeftCorner<3, 3>() = B.transpose();
    J.bottomRightCorner<3, 3>() = A;
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S(0, 0) = 0.95 * e / 1.927;
    S(1, 1) = 1000.0;
    S(0, 1) = S(1, 0) = 0.3 * std::sqrt(S(0, 0) * S(1, 1));
    const Rows original = fixtureRows(psdRoot(J), S);
    Rows rows = original;
    const auto result = project(rows, params);
    ASSERT_TRUE(result.valid) << result.reason;
    expectJointCertificate(original, rows, result, 1e-10, params);
    const Matrix6 kept = rows.h_x.leftCols(6).transpose() * rows.h_x.leftCols(6);
    for (double variance : {1e-2, 1e-4, 1e-6}) {
      EXPECT_LE(maxEigenvalue(effectiveInformation(kept, 1.0 / variance) - effectiveInformation(J, 1.0 / variance)),
                1e-5);
    }
  }
}

TEST(JointFloor, WedgeChecksDirectionalInformation)
{
  // r2 txt:L27-L32 / py:313-338, fixed A/B/C/S independent of the implementation.
  Eigen::Matrix3d A, B, C, S;
  A << 362.7711786561752, 304.7590478799027, -102.37354886317999, 304.7590478799027, 268635.41837121884,
    276.601029831832, -102.37354886317999, 276.601029831832, 131001.81045012521;
  B << -26.245607192768148, 669.2716173167973, 1880.0937720196182, -128050.34115833661, -5893.207528012879,
    4925285.459320568, -17977.19766437482, -2456764.8291369206, 5919.45313520565;
  C << 19572835.253681567, -485075.8472770165, -2330240.9627771494, -485075.8472770165, 49142554.62689758,
    -106212.64232883297, -2330240.9627771494, -106212.64232883297, 96801734.03439139;
  S << 177.9004345230022, -2.3582873759913228, -35.06697876735646, -2.3582873759913228, 0.034937384873912326,
    0.9841251809141385, -35.06697876735646, 0.9841251809141385, 80.27579059746604;
  Matrix6 J;
  J << A, B, B.transpose(), C;
  const Eigen::Matrix3d M = A - B * C.ldlt().solve(B.transpose());
  const double wrong_increases[] = {1.5259708237476637, 1.646154144923794, 1.520403799462299};
  for (int kind = 0; kind < 3; ++kind) {
    Matrix6 D = Matrix6::Identity();
    D.topLeftCorner<3, 3>() = legacyCoordinateRule(M, S, 1.927, kind);
    EXPECT_NEAR(maxEigenvalue(effectiveInformation(D.transpose() * J * D, 1e6) - effectiveInformation(J, 1e6)),
                wrong_increases[kind],
                1e-5);
  }
  const Rows original = fixtureRows(psdRoot(J), S);
  Rows rows = original;
  FloorParams params;
  params.rel_uncertainty = 0.927 / kZ;
  const auto result = project(rows, params);
  ASSERT_TRUE(result.valid) << result.reason;
  const Matrix6 kept = rows.h_x.leftCols(6).transpose() * rows.h_x.leftCols(6);
  Matrix6 target = J;
  target.topLeftCorner<3, 3>() -= M - jointKeptOracle(M, S, 1.927);
  EXPECT_LT((kept - target).norm(), 1e-6);
  for (double variance : {1e-2, 1e-4, 1e-6}) {
    EXPECT_LE(maxEigenvalue(effectiveInformation(kept, 1.0 / variance) - effectiveInformation(J, 1.0 / variance)),
              1e-5);
  }
}

TEST(JointFloor, ZeroFloorRowsStayUnchanged)
{
  // r2 txt:L15 / py:239. The GM coordinate rule moves the zero-floor y row.
  Eigen::Matrix3d M;
  M << 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0;
  const Eigen::Matrix3d S = Eigen::Vector3d(0.2, 0.0, 0.0).asDiagonal();
  EXPECT_NEAR((legacyCoordinateRule(M, S, 1.0, 1) * Eigen::Vector3d::UnitY() - Eigen::Vector3d::UnitY()).norm(),
              0.0198638978616585,
              1e-12);
  const Rows original = orthogonalRows(M, S);
  Rows rows = original;
  const auto result = project(rows, unitMargin());
  ASSERT_TRUE(result.valid);
  EXPECT_LT((rows.h_x.col(1) - original.h_x.col(1)).norm(), 1e-12);
  EXPECT_LT((result.K - (M - S)).norm(), 1e-12);
}

TEST(JointFloor, RepeatedRootTwoPaths)
{
  // r2 txt:L10 / py:221, production a. No eigengap branch is allowed.
  const Eigen::Matrix3d S = Eigen::Vector3d(8.0, 0.5, 0.0).asDiagonal();
  Eigen::Matrix3d Ma = Eigen::Vector3d(10.0 - 1e-8, 10.0, 1000.0 + 1e-8).asDiagonal();
  Eigen::Matrix3d Mb = Eigen::Vector3d(10.0, 10.0, 1000.0).asDiagonal();
  Mb(0, 1) = Mb(1, 0) = 1e-8;
  Rows ra = orthogonalRows(Ma, S);
  Rows rb = orthogonalRows(Mb, S);
  const auto a = project(ra);
  const auto b = project(rb);
  ASSERT_TRUE(a.valid && b.valid);
  EXPECT_NEAR(a.K(1, 1), 9.036465154074829, 1e-10);
  EXPECT_NEAR(b.K(1, 1), 9.036465154074829, 1e-10);
  EXPECT_LT((a.K - b.K).norm(), 2e-8);
  Rows exact = orthogonalRows(Eigen::Vector3d(10.0, 10.0, 1000.0).asDiagonal(), S);
  const auto r = project(exact);
  ASSERT_TRUE(r.valid);
  EXPECT_NEAR(r.K(1, 1), 9.036465154074829, 1e-10);
}

TEST(JointFloor, CrossRepeatedRootContinuity)
{
  // r2 txt:L11-L14 / py:231, fixture-specific bound 1.081, no runtime threshold.
  for (double step : {1e-3, 1e-4, 1e-5, 1e-6}) {
    Eigen::Matrix3d previous = Eigen::Matrix3d::Zero();
    for (int j = -10; j <= 10; ++j) {
      const double t = j * step;
      Eigen::Matrix3d M = Eigen::Vector3d(10.0 + t, 10.0 - t, 1000.0).asDiagonal();
      M(0, 1) = M(1, 0) = 0.3 * t;
      Rows rows = orthogonalRows(M, Eigen::Vector3d(8.0, 0.5, 0.0).asDiagonal());
      const auto result = project(rows);
      ASSERT_TRUE(result.valid);
      if (j > -10) {
        EXPECT_LE((result.K - previous).norm() / step, 1.081);
      }
      previous = result.K;
    }
  }
}

TEST(JointFloor, OrthogonalEquivarianceAndReflection)
{
  // r2 txt:L16 / py:288. Seed/count fixed, independent Gaussian fixtures and algebraic oracle.
  std::mt19937_64 rng(71319);
  for (int trial = 0; trial < 128; ++trial) {
    Matrix6 H;
    Eigen::Matrix3d Z, axes;
    for (int i = 0; i < H.size(); ++i)
      H.data()[i] = gaussian(rng);
    for (int i = 0; i < 9; ++i) {
      Z.data()[i] = gaussian(rng);
      axes.data()[i] = gaussian(rng);
    }
    H += 4.0 * Matrix6::Identity();
    const Eigen::Matrix3d S = 0.1 * Z * Z.transpose();
    const Eigen::HouseholderQR<Eigen::Matrix3d> qr(axes);
    Eigen::Matrix3d Q = qr.householderQ();
    if (trial % 2 == 0)
      Q.col(0) *= -1.0;
    if (trial == 0) {
      Q = Eigen::Matrix3d::Identity();
      Q.col(0).swap(Q.col(1));
    }
    const Rows original = fixtureRows(H, S);
    Rows rows = original;
    Rows rotated = original;
    rotated.h_x.leftCols(3) = original.h_x.leftCols(3) * Q.transpose();
    for (auto & covariance : rotated.covariances)
      covariance = Q * *covariance * Q.transpose();
    const auto a = project(rows);
    const auto b = project(rotated);
    ASSERT_TRUE(a.valid && b.valid) << trial;
    EXPECT_LT((b.K - Q * a.K * Q.transpose()).norm(), 1e-10 * std::max(1.0, a.K.norm()));
    EXPECT_LT((rotated.h_x.leftCols(3) - rows.h_x.leftCols(3) * Q.transpose()).norm(), 1e-10 * rows.h_x.norm());
    EXPECT_LT((rotated.h_x.rightCols(9) - rows.h_x.rightCols(9)).norm(), 1e-10 * rows.h_x.norm());
    EXPECT_LT((rotated.h - rows.h).norm(), 1e-10);
    expectJointCertificate(original, rows, a);
    Rows nuisance = original;
    nuisance.h_x.middleCols(3, 3) = original.h_x.middleCols(3, 3) * Q.transpose();
    ASSERT_TRUE(project(nuisance).valid);
    EXPECT_LT((nuisance.h_x.leftCols(3) - rows.h_x.leftCols(3)).norm(), 1e-10 * rows.h_x.norm());
    EXPECT_LT((nuisance.h_x.middleCols(3, 3) - rows.h_x.middleCols(3, 3) * Q.transpose()).norm(),
              1e-10 * rows.h_x.norm());
    EXPECT_LT((nuisance.h - rows.h).norm(), 1e-10);
    // A coupled rank-one floor also preserves every physical zero-floor row.
    Rows zero = fixtureRows(H, Eigen::Vector3d(0.2, 0.0, 0.0).asDiagonal());
    const Eigen::VectorXd before = zero.h_x.col(1);
    ASSERT_TRUE(project(zero).valid);
    EXPECT_LT((zero.h_x.col(1) - before).norm(), 1e-10 * before.norm());
  }
}

TEST(JointFloor, AppliedOperatorIsBoundedNearRankLoss)
{
  // r2 txt:L3-L6 / py:174. W coordinates diverge, while the actual row map stays bounded.
  for (double epsilon : {1e-2, 1e-4, 1e-8, 1e-12}) {
    const Eigen::Matrix3d M = Eigen::Vector3d(epsilon, 1.0, 1.0).asDiagonal();
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S(0, 0) = epsilon;
    S(0, 1) = S(1, 0) = 0.5 * std::sqrt(epsilon);
    S(1, 1) = 1.0;
    Rows rows = orthogonalRows(M, S);
    const Matrix6 original = rows.h_x.leftCols(6);
    const auto result = project(rows, unitMargin());
    ASSERT_TRUE(result.valid);
    const Matrix6 L = rows.h_x.leftCols(6) * original.inverse();
    EXPECT_LE(L.jacobiSvd().singularValues().maxCoeff(), 1.0 + 1e-12);
    EXPECT_NEAR(result.K(1, 1), 0.25, 1e-12);
  }
  Rows rank = orthogonalRows(Eigen::Vector3d(0.0, 1.0, 1.0).asDiagonal(), Eigen::Vector3d(0.0, 1.0, 0.0).asDiagonal());
  const Rows before = rank;
  const auto result = project(rank, unitMargin());
  EXPECT_FALSE(result.valid);
  EXPECT_STREQ(result.reason, "rank_m");
  EXPECT_TRUE(sameBits(rank.h_x, before.h_x));
  EXPECT_TRUE(sameBits(rank.h, before.h));
}

TEST(JointFloor, FixedShapeScaleMonotonic)
{
  // r2 txt:L16, R31. Only scalar shape scaling is ordered.
  Eigen::Matrix3d M;
  M << 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0;
  Eigen::Matrix3d S;
  S << 0.5, 0.1, 0.0, 0.1, 0.3, 0.05, 0.0, 0.05, 0.2;
  const Rows original = orthogonalRows(M, S);
  Eigen::MatrixXd previous = original.h_x.transpose() * original.h_x;
  Eigen::Matrix3d previous_K = M;
  for (double scale : {0.0, 0.1, 0.5, 1.0, 2.0, 4.0}) {
    FloorParams params = unitMargin();
    params.kappa = scale;
    Rows rows = original;
    const auto result = project(rows, params);
    ASSERT_TRUE(result.valid);
    const Eigen::MatrixXd J = rows.h_x.transpose() * rows.h_x;
    EXPECT_LE(maxEigenvalue(J - previous), 1e-10 * std::max(1.0, previous.norm()));
    EXPECT_LE(maxEigenvalue(result.K - previous_K), 1e-10 * std::max(1.0, previous_K.norm()));
    previous = J;
    previous_K = result.K;
    EXPECT_LT((result.S0 - S).norm(), 1e-12);
    if (scale >= 1.0) {
      Rows equivalent = original;
      params.kappa = 1.0;
      params.rel_uncertainty = (scale - 1.0) / kZ;
      ASSERT_TRUE(project(equivalent, params).valid);
      EXPECT_LT((equivalent.h_x - rows.h_x).norm(), 1e-12);
    }
  }
}

TEST(JointFloor, RemovedJointModeAndPurePositionAreDistinct)
{
  // r2 txt:L8. R36 removes q=(e_x,-e_x), and retains the rotation-confounded pure-position row.
  Rows joint = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  Eigen::Matrix<double, 6, 1> q;
  q << 1, 0, 0, -1, 0, 0;
  joint.h = handRows() * q;
  Rows pure = joint;
  pure.h = handRows().col(0);
  ASSERT_TRUE(project(joint, unitMargin()).valid);
  ASSERT_TRUE(project(pure, unitMargin()).valid);
  EXPECT_LT(joint.h.norm(), 1e-12);
  EXPECT_LT((joint.h_x.leftCols(6) * q).norm(), 1e-12);
  EXPECT_LT((pure.h - Matrix6::Identity().col(3)).norm(), 1e-12);
  EXPECT_LT((joint.h_x.col(1) - handRows().col(1)).norm(), 1e-12);
}

TEST(JointFloor, ResidualAndJacobianUseTheSameMap)
{
  // r2 txt:L8/L35. Independent linear slope/normal-equation oracle.
  const Matrix6 H = handRows();
  Rows original = fixtureRows(H, Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  Rows first = original;
  Rows second = original;
  const Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::LinSpaced(-0.2, 0.3);
  second.h -= H * dx;
  ASSERT_TRUE(project(first, unitMargin()).valid);
  ASSERT_TRUE(project(second, unitMargin()).valid);
  EXPECT_LT((second.h - first.h + first.h_x.leftCols(6) * dx).norm(), 1e-12);
  Matrix6 L = Matrix6::Identity();
  L(0, 0) = 0.0;
  EXPECT_LT((first.h_x.transpose() * first.h - original.h_x.transpose() * L * L * original.h).norm(), 1e-12);
  EXPECT_LT((first.h_x.transpose() * first.h_x - original.h_x.transpose() * L * L * original.h_x).norm(), 1e-12);
}

TEST(JointFloor, NoncommutingHandOracle)
{
  // r2 txt:L35 / py:364. Neither old V/g nor GM reproduces these actual rows.
  Eigen::Matrix3d S;
  S << 4, -6, 0, -6, 9, 0, 0, 0, 1.6;
  Rows rows = orthogonalRows(Eigen::Vector3d(4.0, 9.0, 16.0).asDiagonal(), S);
  rows.h << 0.1, -0.09, 0.08, 0.0, 0.0, 0.0;
  const auto result = project(rows);
  ASSERT_TRUE(result.valid);
  Eigen::Matrix3d expected_H, expected_K;
  expected_H << 1, 1.5, 0, 1, 1.5, 0, 0, 0, 3.593979478661425;
  expected_K << 2, 3, 0, 3, 4.5, 0, 0, 0, 12.91668849303945;
  EXPECT_LT((rows.h_x.topLeftCorner<3, 3>() - expected_H).norm(), 1e-12);
  EXPECT_LT((result.K - expected_K).norm(), 1e-12);
  EXPECT_LT((rows.h.head<3>() - Eigen::Vector3d(0.005, 0.005, 0.0718795895732285)).norm(), 1e-12);
}

TEST(JointFloor, AllColumnsShareTheRowMap)
{
  // r2 txt:L45 / py:489. Keeping extrinsic columns unchanged invents 0.618... information.
  Rows original = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  original.h_x.rightCols(6) = original.h_x.leftCols(6);
  Rows rows = original;
  const auto result = project(rows, unitMargin());
  ASSERT_TRUE(result.valid);
  expectJointCertificate(original, rows, result, 1e-12, unitMargin());
  EXPECT_LT((rows.h_x.rightCols(6) - rows.h_x.leftCols(6)).norm(), 1e-12);
  Eigen::MatrixXd wrong = rows.h_x;
  wrong.rightCols(6) = original.h_x.rightCols(6);
  EXPECT_NEAR(
    maxEigenvalue(wrong.transpose() * wrong - original.h_x.transpose() * original.h_x), 0.6180339887498948, 1e-12);
}

TEST(JointFloor, ArbitraryFullPriorCertificate)
{
  // r2 txt:L16. Full 12-state priors include PSD singular and zero cases.
  std::mt19937_64 rng(71319);
  for (int trial = 0; trial < 128; ++trial) {
    Eigen::MatrixXd H(18, 12), Z(12, 12);
    for (Eigen::Index i = 0; i < H.size(); ++i)
      H.data()[i] = gaussian(rng);
    for (Eigen::Index i = 0; i < Z.size(); ++i)
      Z.data()[i] = gaussian(rng);
    if (trial % 3 == 0)
      Z.setZero();
    if (trial % 3 == 1)
      Z.rightCols(8).setZero();
    const Rows original = fixtureRows(H, 2.0 * Eigen::Matrix3d::Identity());
    Rows rows = original;
    ASSERT_TRUE(project(rows).valid);
    const Eigen::MatrixXd prior = Z * Z.transpose();
    const Eigen::MatrixXd raw = H.transpose() * H + prior;
    const Eigen::MatrixXd kept = rows.h_x.transpose() * rows.h_x + prior;
    const auto schur = [](const Eigen::MatrixXd & J) -> Eigen::Matrix3d {
      return J.topLeftCorner<3, 3>() -
             J.block(0, 3, 3, 9) * supportInverse(J.bottomRightCorner(9, 9)) * J.block(3, 0, 9, 3);
    };
    EXPECT_LE(maxEigenvalue(kept - raw), 1e-10 * raw.norm());
    EXPECT_LE(maxEigenvalue(schur(kept) - schur(raw)), 1e-10 * raw.norm());
  }
}

TEST(JointFloor, ZeroAllPartialAndRankCases)
{
  // r2 txt:L38,L40-L42 / py:504. Zero floor bypasses rank refusal exactly.
  const Eigen::Matrix3d M = Eigen::Vector3d(1.0, 4.0, 9.0).asDiagonal();
  for (bool all : {false, true}) {
    const Eigen::Matrix3d S = Eigen::Vector3d(2.0, all ? 8.0 : 0.0, all ? 18.0 : 0.0).asDiagonal();
    Rows rows = orthogonalRows(M, S);
    rows.h.setOnes();
    const auto result = project(rows, unitMargin());
    ASSERT_TRUE(result.valid);
    const Eigen::Matrix3d K = Eigen::Vector3d(0.0, all ? 0.0 : 4.0, all ? 0.0 : 9.0).asDiagonal();
    EXPECT_LT((result.K - K).norm(), 1e-12);
    EXPECT_LT(rows.h.head(all ? 3 : 1).norm(), 1e-12);
    EXPECT_EQ(rows.h.tail<3>(), Eigen::Vector3d::Ones());
  }
  Rows rank = orthogonalRows(Eigen::Vector3d(0.0, 4.0, 9.0).asDiagonal(), Eigen::Matrix3d::Zero());
  const Rows original = rank;
  const auto result = project(rank);
  ASSERT_TRUE(result.valid);
  EXPECT_STREQ(result.reason, "zero_floor");
  EXPECT_EQ(result.spectral_count, 2);
  EXPECT_TRUE(sameBits(rank.h_x, original.h_x));
  EXPECT_TRUE(sameBits(rank.h, original.h));
  Rows empty;
  empty.h_x.resize(0, 12);
  empty.h.resize(0);
  const auto no_rows = project(empty);
  EXPECT_TRUE(no_rows.valid);
  EXPECT_STREQ(no_rows.reason, "zero_floor");
  EXPECT_EQ(no_rows.points, 0);
  for (int count : {1, 2}) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(count, 6);
    H.leftCols(count).setIdentity();
    Rows short_scan = fixtureRows(H, Eigen::Matrix3d::Identity());
    const Rows before = short_scan;
    const auto refused = project(short_scan);
    EXPECT_FALSE(refused.valid);
    EXPECT_STREQ(refused.reason, "rank_m");
    EXPECT_LE(refused.rank_m, count);
    EXPECT_TRUE(sameBits(short_scan.h_x, before.h_x));
    EXPECT_TRUE(sameBits(short_scan.h, before.h));
  }
  // r2 txt:L34 / py:351. The production row map retains the original commuting fractions too.
  Rows commuting = orthogonalRows(Eigen::Vector3d(1000, 780, 100).asDiagonal(), 25.0 * Eigen::Matrix3d::Identity());
  const auto fractions = project(commuting);
  ASSERT_TRUE(fractions.valid);
  const Eigen::Vector3d expected(951.8232577037414, 731.8232577037413, 51.823257703741405);
  EXPECT_LT((fractions.K.diagonal() - expected).norm(), 1e-10);
}

TEST(JointFloor, RotationSupportRankMayChangeM0AtFullPositionRank)
{
  // R37(4) / r2 E2.4. Exact nuisance support changes can jump while M0 remains full rank.
  for (double scale : {1.0, 1e-10, 0.0}) {
    Matrix6 H = Matrix6::Zero();
    H.topLeftCorner<3, 3>().setIdentity();
    H(0, 3) = H(3, 3) = scale;
    Rows rows = fixtureRows(H, 0.1 * Eigen::Matrix3d::Identity());
    const auto result = project(rows, unitMargin());
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.rank_m, 3);
    EXPECT_EQ(result.rank_r, scale == 0.0 ? 0 : 1);
    EXPECT_NEAR(result.M0(0, 0), scale == 0.0 ? 1.0 : 0.5, 1e-12);
  }
}

TEST(JointFloor, RidgeIsNotAnOrthogonalProjection)
{
  // r2 txt:L9 / py:207. Ridge is diagnostic only for D1.
  Rows rows = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  const auto result = project(rows, unitMargin(), 0.1);
  ASSERT_TRUE(result.valid);
  EXPECT_LT((result.M0 - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_NEAR(result.M_ridge(0, 0), 1.0909090909090908, 1e-12);
  EXPECT_NEAR(result.M_ridge(0, 1), 0.09090909090909094, 1e-12);
  EXPECT_LT((result.K - Eigen::Matrix3d(Eigen::Vector3d(0, 1, 1).asDiagonal())).norm(), 1e-12);
}

TEST(JointFloor, InvalidIsAtomic)
{
  // r2 txt:L37,L40. Also overflow an unused-by-Schur extrinsic column late in staging.
  const Rows original = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  for (int kind = 0; kind < 4; ++kind) {
    Rows rows = original;
    if (kind == 0)
      rows.covariances[0] = -Eigen::Matrix3d::Identity();
    if (kind == 1)
      rows.h_x.col(0).setZero();
    if (kind == 2)
      rows.h_x(0, 11) = kNaN;
    if (kind == 3) {
      const double c = std::sqrt(0.5);
      rows.h_x.row(0) = c * (original.h_x.row(0) - original.h_x.row(1));
      rows.h_x.row(1) = c * (original.h_x.row(0) + original.h_x.row(1));
      rows.h_x.col(11).setConstant(std::numeric_limits<double>::max());
    }
    const Rows before = rows;
    for (const FloorMode mode : {FloorMode::kObserve, FloorMode::kApply}) {
      rows = before;
      fast_lio::FloorScanRecord record;
      fast_lio::floorStep(mode, rows.h_x, rows.h, rows.weights, rows.covariances, unitMargin(), kRidge, record);
      EXPECT_FALSE(record.last.valid) << kind;
      EXPECT_EQ(record.invalid_evaluations, 1);
      EXPECT_FALSE(record.projected);
      EXPECT_TRUE(sameBits(rows.h_x, before.h_x));
      EXPECT_TRUE(sameBits(rows.h, before.h));
    }
  }
}

TEST(JointFloor, OverflowingShiftCoordinatesDoNotRejectFiniteRows)
{
  // E2.3 diagnostic overflow contract. The applied L/delta_h remain finite.
  Matrix6 H = Matrix6::Zero();
  H.topLeftCorner<3, 3>() = 1e-155 * Eigen::Matrix3d::Identity();
  Rows rows = fixtureRows(H, 2e-310 * Eigen::Matrix3d::Identity());
  rows.h.setConstant(1e200);
  const auto result = project(rows, unitMargin());
  ASSERT_TRUE(result.valid) << result.reason;
  EXPECT_FALSE(result.shift_coords_valid);
  EXPECT_TRUE(result.d_t.array().isNaN().all());
  EXPECT_TRUE(result.d_r.array().isNaN().all());
  EXPECT_TRUE(std::isfinite(result.dh_norm));
  EXPECT_TRUE(rows.h_x.allFinite());
  EXPECT_TRUE(rows.h.allFinite());
  EXPECT_EQ(rows.h.head<3>(), Eigen::Vector3d::Zero());
}

TEST(DesignLimits, ArbitraryPsdFloorOrderingIsNotPromised)
{
  // r2 numerical-d1.txt:L1 / py:154. Preserve the explicit impossibility oracle.
  const Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d S1 = Eigen::Vector3d(0.5, 2, 0).asDiagonal();
  const Eigen::Vector3d v(1, 1, 0);
  const Eigen::Matrix3d S2 = S1 + 0.1 * v * v.transpose();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> difference(jointKeptOracle(M, S1, 1.0) -
                                                                  jointKeptOracle(M, S2, 1.0));
  EXPECT_NEAR(difference.eigenvalues().minCoeff(), -0.00873447902177305, 1e-12);
}

TEST(DesignLimits, SupportChangesMayJumpAndCoordinateMapsAreUnbounded)
{
  // r2 numerical-d1.txt:L2-L6 / py:167,174. These are retained limitations, not production fallbacks.
  const Eigen::Matrix2d M = Eigen::Vector2d(1, 0).asDiagonal();
  for (double theta : {0.1, 0.01, 0.001, 0.0}) {
    const Eigen::Vector2d q(-std::sin(theta), std::cos(theta));
    const double weight = q.dot(M * q);
    const double forced = weight > 0.0 ? 1.0 : 0.0;
    EXPECT_EQ(forced, theta == 0.0 ? 0.0 : 1.0);
  }
  double previous_norm = 0.0;
  for (double epsilon : {1e-2, 1e-6, 1e-10, 1e-12}) {
    const Eigen::Matrix3d information = Eigen::Vector3d(epsilon, 1, 1).asDiagonal();
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    S(0, 0) = epsilon;
    S(0, 1) = S(1, 0) = 0.5 * std::sqrt(epsilon);
    S(1, 1) = 1.0;
    const double norm = legacyCoordinateRule(information, S, 1.0, 0).jacobiSvd().singularValues().maxCoeff();
    EXPECT_GT(norm, previous_norm);
    previous_norm = norm;
  }
  EXPECT_NEAR(previous_norm, 353553.3905936273, 1e-5);
}

TEST(FloorSchema2, R35UsesFullMatricesAndFrameEndDirection)
{
  // r2 txt:L43 / py:509, and complete schema field/roundtrip checks.
  fast_lio::FloorScanRecord record;
  auto & r = record.last;
  r.M0 << 4, 1, 0, 1, 3, 0, 0, 0, 2;
  r.S0 = Eigen::Vector3d(2, 1, 0).asDiagonal();
  const Eigen::Vector3d u = Eigen::Vector3d(1, 1, 0).normalized();
  for (double kappa : {0.0, 1.0, 4.0}) {
    r.kappa = kappa;
    const auto calibration = fast_lio::floorCalibration(r, u, true);
    EXPECT_NEAR(calibration.numerator, 4.5, 1e-12);
    EXPECT_NEAR(calibration.denominator, 1.5, 1e-12);
    EXPECT_NEAR(calibration.ratio, 3.0, 1e-12);
    EXPECT_TRUE(calibration.valid);
  }
  const auto zero = fast_lio::floorCalibration(r, Eigen::Vector3d::UnitZ(), true);
  EXPECT_FALSE(zero.valid);
  EXPECT_EQ(zero.zero_denominator, 1);
  const auto mismatch = fast_lio::floorCalibration(r, u, false);
  EXPECT_FALSE(mismatch.valid);
  EXPECT_STREQ(mismatch.reason, "frame_mismatch");
  EXPECT_FALSE(fast_lio::floorCalibration(r, Eigen::Vector3d(kNaN, 0, 0), true).valid);
  const std::string encoded = fast_lio::floorMatrix(r.M0);
  double xx, xy, xz, yy, yz, zz;
  ASSERT_EQ(std::sscanf(encoded.c_str(), "[%lf %lf %lf %lf %lf %lf]", &xx, &xy, &xz, &yy, &yz, &zz), 6);
  EXPECT_DOUBLE_EQ(xx, 4.0);
  EXPECT_DOUBLE_EQ(xy, 1.0);
  for (double value : {0.0, 1.2345678901234567, -1.2345678901234567e-100, 1.2345678901234567e100}) {
    const double decoded = std::stod(fast_lio::floorNumber(value));
    EXPECT_LE(std::abs(decoded - value), 1e-15 * std::max(1.0, std::abs(value)));
  }
  EXPECT_EQ(fast_lio::floorNumber(kNaN), "nan");
  EXPECT_EQ(fast_lio::floorNumber(std::numeric_limits<double>::infinity()), "nan");
  EXPECT_EQ(fast_lio::floorNumber(-std::numeric_limits<double>::infinity()), "nan");
}

TEST(FloorSchema2, InvalidApplyDoesNotDropValidCalibration)
{
  // r2 txt:L40,L43 / E4.2. rank_m is an apply gate, never a calibration filter.
  Rows rows = orthogonalRows(Eigen::Vector3d(0, 4, 9).asDiagonal(), Eigen::Matrix3d::Identity());
  const auto result = project(rows);
  ASSERT_FALSE(result.valid);
  ASSERT_STREQ(result.reason, "rank_m");
  const auto calibration = fast_lio::floorCalibration(result, Eigen::Vector3d::UnitY(), true);
  EXPECT_TRUE(calibration.valid);
  EXPECT_NEAR(calibration.ratio, 4.0, 1e-12);
}

TEST(FloorLine, CarriesTheDecisionAndTheScansCountsInItsFormat)
{
  // r2 txt:L37,L43 and E4.2. Historical schema-1 field meanings are not reused.
  Rows rows = fixtureRows(handRows(), Eigen::Vector3d(1.0, 0.0, 0.0).asDiagonal());
  fast_lio::FloorScanRecord record;
  fast_lio::floorStep(
    FloorMode::kApply, rows.h_x, rows.h, rows.weights, rows.covariances, unitMargin(), kRidge, record);
  const auto stamp = INT64_C(1234567890123456789);
  const std::string line =
    fast_lio::floorLine(1234.5, stamp, FloorMode::kApply, record, Eigen::Vector3d::UnitX(), true);
  for (const char * field : {"schema=2",
                             "rule=joint_row_v1",
                             "mode=apply",
                             "stamp_ns=1234567890123456789",
                             "evaluation_iter=1",
                             "valid=1",
                             "reason=ok",
                             "points=6",
                             "iterations=1",
                             "points_without_covariance=0",
                             "invalid_iterations=0",
                             "projected=1",
                             "rank_m=3",
                             "rank_r=3",
                             "kappa=",
                             "sigma_rel=",
                             "z=",
                             "a=",
                             "m_definition=exact_rotation_schur",
                             "M=[",
                             "M_ridge=[",
                             "S0=[",
                             "lambda=[",
                             "tau=[",
                             "g_gen=[",
                             "K=[",
                             "F_sym=[",
                             "transform_kind=row_contraction",
                             "u=[1 0 0]",
                             "u_source=odom_frame_end",
                             "u_valid=1",
                             "rD_num=",
                             "rD_den=",
                             "rD=",
                             "rD_valid=1",
                             "A=[",
                             "B=[",
                             "C=[",
                             "A_keep=[",
                             "B_keep=[",
                             "C_keep=[",
                             "shift_kind=joint_schur",
                             "shift_coords_valid=1",
                             "d_t=[",
                             "d_r=[",
                             "dh_norm="}) {
    EXPECT_NE(line.find(field), std::string::npos) << field << " in " << line;
  }
  const std::string observed =
    fast_lio::floorLine(1234.5, stamp, FloorMode::kObserve, record, Eigen::Vector3d::UnitX(), true);
  EXPECT_EQ(observed.find(" d_t="), std::string::npos);
  EXPECT_EQ(observed.find(" d_r="), std::string::npos);
  EXPECT_EQ(observed.find(" shift_kind="), std::string::npos);
  EXPECT_TRUE(fast_lio::floorLine(1234.5, stamp, FloorMode::kOff, record, Eigen::Vector3d::UnitX(), true).empty());
}

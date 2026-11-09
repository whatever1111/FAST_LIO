// test_normal_noise_floor.cpp — the per-point normal covariance, the scan's floor, the keep
// gains, the projection built from them and the residual shift that goes with it, and the
// lidar update's three modes with the per-scan record and log line they leave
// (normal_noise_floor.hpp).

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
using fast_lio::DirectionFloorAccumulator;
using fast_lio::floorKeepGains;
using fast_lio::FloorMode;
using fast_lio::FloorParams;
using fast_lio::FloorProjectionResult;
using fast_lio::floorResidualShift;
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

// What rows (H_t, H_r) and a residual h read about the position once the rotation is solved for:
// b_t - M_tr (M_rr + ridge trace(M_rr) / 3 I)^-1 b_r, with the regularisation M_eff is built with.
Eigen::Vector3d profiledInformation(const Eigen::MatrixXd & H_t, const Eigen::MatrixXd & H_r, const Eigen::VectorXd & h)
{
  const Eigen::Matrix3d M_tr = H_t.transpose() * H_r;
  const Eigen::Matrix3d M_rr = H_r.transpose() * H_r;
  const Eigen::Matrix3d rr = M_rr + kRidge * M_rr.trace() / 3.0 * Eigen::Matrix3d::Identity();
  const Eigen::Vector3d b_t = H_t.transpose() * h;
  const Eigen::Vector3d b_r = H_r.transpose() * h;
  return b_t - M_tr * rr.ldlt().solve(b_r);
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

  DirectionFloorAccumulator accumulator(orthonormalBasis());
  accumulator.add(1.0, cov);
  accumulator.add(3.0, rounded);
  EXPECT_EQ(accumulator.floor(1.0), Eigen::Vector3d::Zero());
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

TEST(DirectionFloorAccumulator, AddsTheWeightedVarianceAlongEachDirection)
{
  // Along the axes: f_k = kappa sum w^2 C_kk.
  DirectionFloorAccumulator axes(Eigen::Matrix3d::Identity());
  axes.add(0.5, Eigen::Matrix3d(Eigen::Vector3d(4e-3, 2e-3, 0.0).asDiagonal()));
  axes.add(2.0, Eigen::Matrix3d(Eigen::Vector3d(0.0, 1e-3, 3e-3).asDiagonal()));
  const Eigen::Vector3d expected(0.25 * 4e-3, 0.25 * 2e-3 + 4.0 * 1e-3, 4.0 * 3e-3);
  EXPECT_LT((axes.floor(1.0) - expected).norm(), 1e-18);
  EXPECT_LT((axes.floor(1.5) - 1.5 * expected).norm(), 1e-18);
  EXPECT_EQ(axes.points(), 2);
  EXPECT_EQ(axes.pointsWithoutCovariance(), 0);

  // For any orthonormal V the floors add up to kappa sum w^2 trace(C): nothing is lost or counted twice.
  const Eigen::Matrix3d V = orthonormalBasis();
  DirectionFloorAccumulator rotated(V);
  const auto cov_a = planeNormalCovariance(handBuilt(0.02, 1.0, 4.0));
  PlaneFitEigen tilted = handBuilt(0.004, 0.2, 0.5);
  tilted.e1 = Eigen::Vector3d(1.0, 1.0, 0.0).normalized();
  tilted.e2 = Eigen::Vector3d(-1.0, 1.0, 1.0).normalized();
  const auto cov_b = planeNormalCovariance(tilted);
  ASSERT_TRUE(cov_a && cov_b);
  rotated.add(0.7, cov_a);
  rotated.add(1.3, cov_b);
  const double trace = 0.49 * cov_a->trace() + 1.69 * cov_b->trace();
  EXPECT_NEAR(rotated.floor(2.0).sum(), 2.0 * trace, 1e-15);
  EXPECT_GT(rotated.floor(2.0).minCoeff(), 0.0);
}

TEST(DirectionFloorAccumulator, PointsWithoutAKnownContributionAreCountedNotGuessed)
{
  DirectionFloorAccumulator accumulator(Eigen::Matrix3d::Identity());
  const auto cov = planeNormalCovariance(handBuilt(0.02, 1.0, 4.0));
  accumulator.add(1.0, cov);
  const Eigen::Vector3d known = accumulator.floor(1.0);
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
  // Every gain exactly one: nothing is removed, so nothing is shifted, and the update gets the rows as built,
  // bit for bit. The scan does read an offset; it is the gains, not the residual, that decide.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005, false);
  Rows rows = original;
  const FloorProjectionResult result = project(rows);
  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.noise_floor, Eigen::Vector3d::Zero());
  EXPECT_EQ(result.gains, Eigen::Vector3d::Ones());
  EXPECT_EQ(result.shift, Eigen::Vector3d::Zero());
  EXPECT_TRUE(sameBits(rows.h, original.h));
  EXPECT_TRUE(sameBits(rows.h_x, original.h_x));

  const Eigen::MatrixXd H_t = original.h_x.leftCols(3);
  const Eigen::MatrixXd H_r = original.h_x.middleCols(3, 3);
  const Eigen::Vector3d b_t = H_t.transpose() * original.h;
  const Eigen::Vector3d b_r = H_r.transpose() * original.h;
  const Eigen::Matrix3d M_tr = H_t.transpose() * H_r;
  const Eigen::Matrix3d M_rr = H_r.transpose() * H_r;
  ASSERT_GT(b_t.norm(), 1.0);
  EXPECT_EQ(floorResidualShift(result.eigenvectors,
                               result.lambda,
                               Eigen::Vector3d::Ones(),
                               b_t,
                               b_r,
                               M_tr,
                               M_rr,
                               kRidge,
                               static_cast<int>(original.h.size())),
            Eigen::Vector3d::Zero());
}

TEST(ApplyFloorProjection, TheShiftedRowsReadNothingAlongTheRemovedDirection)
{
  // Design addendum 6.1 (ii). The noisy corridor keeps nothing along its axis (g = 0) and most of the other two
  // directions (0 < g < 1). Before the shift the scan reads its along-corridor offset, mixed with what the
  // rotation explains; after it, the residual against the rows as built reads nothing along the axis once the
  // rotation is solved for (zero up to rounding: 1e-12 of the information vector), and g_k of its reading along
  // a partly kept direction. The rotation and extrinsic columns stay as they are; a row without a normal
  // covariance stays out of the floor and is counted.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  Rows rows = original;
  rows.covariances[7] = std::nullopt;
  const FloorProjectionResult result = project(rows);
  ASSERT_TRUE(result.valid);
  const Eigen::Vector3d u = result.eigenvectors.col(0);
  ASSERT_GT(std::abs(u.x()), 0.999);  // the removed direction is the corridor's axis
  ASSERT_EQ(result.gains(0), 0.0);
  for (int k = 1; k < 3; ++k) {
    ASSERT_GT(result.gains(k), 0.9) << k;
    ASSERT_LT(result.gains(k), 1.0) << k;
  }
  EXPECT_EQ(result.points, static_cast<int>(original.h.size()));
  EXPECT_EQ(result.points_without_covariance, 1);

  const Eigen::MatrixXd H_t = original.h_x.leftCols(3);
  const Eigen::MatrixXd H_r = original.h_x.middleCols(3, 3);
  const Eigen::Vector3d before = result.eigenvectors.transpose() * profiledInformation(H_t, H_r, original.h);
  const Eigen::Vector3d after = result.eigenvectors.transpose() * profiledInformation(H_t, H_r, rows.h);
  const Eigen::Vector3d raw = H_t.transpose() * original.h;
  const double scale = before.norm();
  // There is a reading to remove, and the rotation is part of it: the profiled one is not the raw one.
  EXPECT_GT(std::abs(before(0)), 1e-3 * scale);
  EXPECT_GT(std::abs(u.dot(raw) - before(0)), 0.1 * std::abs(before(0)));
  EXPECT_LE(std::abs(after(0)), 1e-12 * scale);
  for (int k = 1; k < 3; ++k) {
    EXPECT_NEAR(after(k), result.gains(k) * before(k), 1e-12 * scale) << k;
  }

  EXPECT_TRUE((rows.h_x.rightCols(9).array() == original.h_x.rightCols(9).array()).all());
  // The position columns carry nothing along u either, down to V's orthonormality (cf. ProjectionFromGains).
  EXPECT_LT((rows.h_x.leftCols(3) * u).cwiseAbs().maxCoeff(), 1e-14);
}

TEST(FloorResidualShift, ADirectionWithoutInformationIsNotShifted)
{
  // A direction the scan does not see has nothing to divide its reading by. Hand-built, lambda_0 = 0 and then
  // lambda_0 below the rank tolerance eps N lambda_max, each with a reading a division would blow up; the partly
  // kept direction is shifted by (1 - g) beta / lambda as ever, the kept one not at all.
  const Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gains(0.0, 0.5, 1.0);
  const Eigen::Vector3d b_t(1e-3, 0.4, 0.3);
  const Eigen::Matrix3d no_rotation = Eigen::Matrix3d::Zero();  // trace(M_rr) = 0: b_eff = b_t
  for (double lambda0 : {0.0, 1e-20}) {
    const Eigen::Vector3d shift = floorResidualShift(V,
                                                     Eigen::Vector3d(lambda0, 2.0, 3.0),
                                                     gains,
                                                     b_t,
                                                     Eigen::Vector3d::Zero(),
                                                     no_rotation,
                                                     no_rotation,
                                                     kRidge,
                                                     100);
    ASSERT_TRUE(shift.allFinite()) << lambda0;
    EXPECT_EQ(shift(0), 0.0) << lambda0;
    EXPECT_DOUBLE_EQ(shift(1), 0.5 * 0.4 / 2.0) << lambda0;
    EXPECT_EQ(shift(2), 0.0) << lambda0;
  }

  // From rows: a corridor with exact normals has no along-corridor information at all, lambda_0 and the reading
  // both zero up to rounding. Everything stays finite, the directions the scan sees are still shifted, and
  // nothing is shifted along the corridor.
  Rows rows = corridorRows(0.0, 0.005);
  const FloorProjectionResult result = project(rows);
  ASSERT_TRUE(result.valid);
  const Eigen::Vector3d u = result.eigenvectors.col(0);
  ASSERT_GT(std::abs(u.x()), 1.0 - 1e-12);
  ASSERT_LE(result.lambda(0),
            std::numeric_limits<double>::epsilon() * static_cast<double>(rows.h.size()) * result.lambda(2));
  EXPECT_EQ(result.gains(0), 0.0);
  EXPECT_TRUE(result.shift.allFinite());
  EXPECT_TRUE(rows.h.allFinite());
  EXPECT_TRUE(rows.h_x.allFinite());
  EXPECT_GT(result.shift.norm(), 0.0);
  EXPECT_LE(std::abs(u.dot(result.shift)), 1e-14 * result.shift.norm());
}

TEST(FloorResidualShift, NonFiniteInputsGiveNoShift)
{
  // The conservative side for the residual is to leave it: any non-finite input, or a negative row count, gives
  // the zero vector. The same inputs, finite, do shift.
  struct Inputs
  {
    Eigen::Matrix3d V;
    Eigen::Vector3d lambda, gains, b_t, b_r;
    Eigen::Matrix3d M_tr, M_rr;
    double ridge;
    int rows;
  };
  const auto shift = [](const Inputs & in) {
    return floorResidualShift(in.V, in.lambda, in.gains, in.b_t, in.b_r, in.M_tr, in.M_rr, in.ridge, in.rows);
  };
  Eigen::Matrix3d M_tr;
  M_tr << 0.1, -0.2, 0.05, 0.0, 0.3, -0.1, 0.2, 0.1, 0.0;
  const Inputs finite{orthonormalBasis(),
                      Eigen::Vector3d(0.5, 2.0, 3.0),
                      Eigen::Vector3d(0.0, 0.5, 1.0),
                      Eigen::Vector3d(0.3, -0.2, 0.1),
                      Eigen::Vector3d(0.05, 0.02, -0.01),
                      M_tr,
                      Eigen::Matrix3d(Eigen::Vector3d(4.0, 5.0, 6.0).asDiagonal()),
                      kRidge,
                      10};
  ASSERT_GT(shift(finite).norm(), 0.0);
  for (double bad : {kNaN, kInf, -kInf}) {
    std::vector<Inputs> broken(8, finite);
    broken[0].V(1, 2) = bad;
    broken[1].lambda(1) = bad;
    broken[2].gains(1) = bad;
    broken[3].b_t(0) = bad;
    broken[4].b_r(2) = bad;
    broken[5].M_tr(2, 0) = bad;
    broken[6].M_rr(1, 1) = bad;
    broken[7].ridge = bad;
    for (std::size_t i = 0; i < broken.size(); ++i) {
      EXPECT_EQ(shift(broken[i]), Eigen::Vector3d::Zero()) << "argument " << i << " = " << bad;
    }
  }
  Inputs negative_rows = finite;
  negative_rows.rows = -1;
  EXPECT_EQ(shift(negative_rows), Eigen::Vector3d::Zero());

  // applyFloorProjection: rows or parameters it cannot use leave the rows as they are and say so; rows that do not
  // match each other in size are a programming error.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  const auto refused = [](Rows rows, const FloorParams & params, double ridge) {
    const Rows given = rows;
    const FloorProjectionResult result = project(rows, params, ridge);
    return !result.valid && result.gains == Eigen::Vector3d::Ones() && result.shift == Eigen::Vector3d::Zero() &&
           sameBits(rows.h_x, given.h_x) && sameBits(rows.h, given.h);
  };
  Rows bad_row = original;
  bad_row.h_x(5, 1) = kNaN;
  EXPECT_TRUE(refused(bad_row, FloorParams{}, kRidge));
  Rows bad_residual = original;
  bad_residual.h(3) = kInf;
  EXPECT_TRUE(refused(bad_residual, FloorParams{}, kRidge));
  FloorParams no_alpha;
  no_alpha.alpha = 0.0;
  EXPECT_TRUE(refused(original, no_alpha, kRidge));
  FloorParams bad_kappa;
  bad_kappa.kappa = kNaN;
  EXPECT_TRUE(refused(original, bad_kappa, kRidge));
  FloorParams negative_margin;
  negative_margin.rel_uncertainty = -0.1;
  EXPECT_TRUE(refused(original, negative_margin, kRidge));
  EXPECT_TRUE(refused(original, FloorParams{}, kInf));
  Rows usable = original;
  ASSERT_TRUE(project(usable).valid);

  Rows rows = original;
  Eigen::MatrixXd narrow = rows.h_x.leftCols(11);
  EXPECT_THROW(applyFloorProjection(narrow, rows.h, rows.weights, rows.covariances, FloorParams{}, kRidge),
               std::invalid_argument);
  Eigen::VectorXd short_residual = rows.h.head(rows.h.size() - 1);
  EXPECT_THROW(applyFloorProjection(rows.h_x, short_residual, rows.weights, rows.covariances, FloorParams{}, kRidge),
               std::invalid_argument);
  std::vector<double> long_weights = rows.weights;
  long_weights.push_back(1.0);
  EXPECT_THROW(applyFloorProjection(rows.h_x, rows.h, long_weights, rows.covariances, FloorParams{}, kRidge),
               std::invalid_argument);
  std::vector<std::optional<Eigen::Matrix3d>> short_covariances = rows.covariances;
  short_covariances.pop_back();
  EXPECT_THROW(applyFloorProjection(rows.h_x, rows.h, rows.weights, short_covariances, FloorParams{}, kRidge),
               std::invalid_argument);
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
  EXPECT_EQ(observe.last.shift, Eigen::Vector3d::Zero());
  EXPECT_EQ(observe.last.gains(0), 0.0);
  EXPECT_EQ(observe.last.points, apply.last.points);
  EXPECT_EQ(observe.last.points_without_covariance, apply.last.points_without_covariance);
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(observe.last.lambda(k), apply.last.lambda(k), 1e-12 * apply.last.lambda.maxCoeff()) << k;
    EXPECT_NEAR(observe.last.noise_floor(k), apply.last.noise_floor(k), 1e-12 * apply.last.noise_floor.maxCoeff()) << k;
    EXPECT_NEAR(observe.last.gains(k), apply.last.gains(k), 1e-12) << k;
    EXPECT_NEAR(std::abs(observe.last.eigenvectors.col(k).dot(apply.last.eigenvectors.col(k))), 1.0, 1e-12) << k;
  }
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

TEST(FloorLine, CarriesTheDecisionAndTheScansCountsInItsFormat)
{
  // The [FLOOR] line the replay reader parses: the fixed fields in order, every number read back to its 9 digits,
  // and the shift's length and coordinates only in apply mode.
  const Rows original = corridorRows(std::sqrt(0.5), 0.005);
  Rows rows = original;
  rows.covariances[7] = std::nullopt;
  fast_lio::FloorScanRecord record;
  fast_lio::floorStep(
    FloorMode::kApply, rows.h_x, rows.h, rows.weights, rows.covariances, FloorParams{}, kRidge, record);
  ASSERT_TRUE(record.projected);
  const std::string line = fast_lio::floorLine(1234.5, FloorMode::kApply, record);
  const std::string head = "[FLOOR] t=1234.500000 mode=apply valid=1 points=" + std::to_string(original.h.size()) +
                           " points_without_covariance=1 iterations=1 invalid_iterations=0 lambda=[";
  ASSERT_EQ(line.rfind(head, 0), 0u) << line;
  const std::size_t numbers = line.find(" lambda=[");
  ASSERT_NE(numbers, std::string::npos) << line;
  double lambda[3], noise[3], gains[3], shift = 0.0, coef[3];
  ASSERT_EQ(std::sscanf(line.c_str() + numbers,
                        " lambda=[%lf %lf %lf] f=[%lf %lf %lf] g=[%lf %lf %lf] d=%lf coef=[%lf %lf %lf]",
                        &lambda[0],
                        &lambda[1],
                        &lambda[2],
                        &noise[0],
                        &noise[1],
                        &noise[2],
                        &gains[0],
                        &gains[1],
                        &gains[2],
                        &shift,
                        &coef[0],
                        &coef[1],
                        &coef[2]),
            13)
    << line;
  const auto matches = [](double printed, double value) {
    return std::abs(printed - value) <= 1e-8 * std::max(1.0, std::abs(value));
  };
  const Eigen::Vector3d expected_coef = record.last.eigenvectors.transpose() * record.last.shift;
  for (int k = 0; k < 3; ++k) {
    EXPECT_TRUE(matches(lambda[k], record.last.lambda(k))) << k << ' ' << line;
    EXPECT_TRUE(matches(noise[k], record.last.noise_floor(k))) << k << ' ' << line;
    EXPECT_TRUE(matches(gains[k], record.last.gains(k))) << k << ' ' << line;
    EXPECT_TRUE(matches(coef[k], expected_coef(k))) << k << ' ' << line;
  }
  EXPECT_TRUE(matches(shift, record.last.shift.norm())) << line;
  EXPECT_GT(shift, 0.0);

  const std::string observed = fast_lio::floorLine(1234.5, FloorMode::kObserve, record);
  EXPECT_EQ(observed.rfind("[FLOOR] t=1234.500000 mode=observe valid=1 ", 0), 0u) << observed;
  EXPECT_EQ(observed.find(" d="), std::string::npos) << observed;
  EXPECT_EQ(observed.find(" coef="), std::string::npos) << observed;
}

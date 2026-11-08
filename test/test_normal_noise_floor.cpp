// test_normal_noise_floor.cpp — the per-point normal covariance, the scan's floor, the keep
// gains and the projection built from them (normal_noise_floor.hpp).

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "normal_noise_floor.hpp"
#include "plane_fit.hpp"

using fast_lio::DirectionFloorAccumulator;
using fast_lio::floorKeepGains;
using fast_lio::FloorParams;
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

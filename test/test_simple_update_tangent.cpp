#include <Eigen/Eigenvalues>

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <omp.h>

#include "use-ikfom.hpp"

namespace
{
using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;
using Cov = Filter::cov;
using Delta = Filter::vectorized_state;
using Observation = Eigen::Matrix<double, Eigen::Dynamic, state_ikfom::DOF>;

Cov correlatedPrior()
{
  // Deterministic dense SPD: manifold/Euclidean and manifold/manifold cross-blocks are nonzero.
  Cov lower = Cov::Identity();
  for (int row = 0; row < lower.rows(); ++row) {
    lower(row, row) = 0.3 + 0.01 * row;
    for (int col = 0; col < row; ++col) {
      lower(row, col) = 0.035 * std::sin(1.0 + row + 2.0 * col);
    }
  }
  return lower * lower.transpose();
}

state_ikfom initialState()
{
  state_ikfom state;
  state.rot = SO3(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1.0, -2.0, 3.0).normalized()).toRotationMatrix());
  state.offset_R_L_I = SO3(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()).toRotationMatrix());
  state.grav = S2(Eigen::Vector3d(2.0, -1.0, -9.0));
  return state;
}

Cov numericalReset(const state_ikfom & before, const Delta & correction)
{
  // G = d [ (before boxplus (correction + error)) boxminus (before boxplus correction) ] / d error.
  // This oracle uses only the manifold operations, not the production analytic Jacobians.
  constexpr double kStep = 1e-6;
  state_ikfom after = before;
  after.boxplus(correction);
  Cov reset;
  for (int col = 0; col < reset.cols(); ++col) {
    Delta plus = correction;
    Delta minus = correction;
    plus(col) += kStep;
    minus(col) -= kStep;
    state_ikfom plus_state = before;
    state_ikfom minus_state = before;
    plus_state.boxplus(plus);
    minus_state.boxplus(minus);
    Delta plus_error;
    Delta minus_error;
    plus_state.boxminus(plus_error, after);
    minus_state.boxminus(minus_error, after);
    reset.col(col) = (plus_error - minus_error) / (2.0 * kStep);
  }
  return reset;
}

void checkUpdate(const Observation & H, const Eigen::VectorXd & residual, bool expect_reset)
{
  Filter filter;
  state_ikfom before = initialState();
  filter.change_x(before);
  Cov prior = correlatedPrior();
  filter.change_P(prior);
  const Eigen::VectorXd noise = Eigen::VectorXd::Constant(H.rows(), 0.05);
  const Eigen::MatrixXd innovation = H * prior * H.transpose() + noise.asDiagonal().toDenseMatrix();
  const Eigen::MatrixXd gain = innovation.ldlt().solve(H * prior).transpose();
  const Delta correction = gain * residual;
  const Cov conditioned = prior - gain * H * prior;
  const Cov reset = numericalReset(before, correction);
  const Cov expected = reset * conditioned * reset.transpose();
  if (expect_reset) {
    ASSERT_GT((expected - conditioned).norm(), 1e-4);  // fails the former no-reset implementation
  }

  Filter::UpdateDiagnostics diagnostics;
  filter.update_simple(H, residual, noise, &diagnostics);
  state_ikfom expected_state = before;
  expected_state.boxplus(correction);
  Delta state_error;
  filter.get_x().boxminus(state_error, expected_state);
  EXPECT_LT(state_error.norm(), 1e-12);  // the fix must not change the mean update
  EXPECT_LT((filter.get_P() - expected).norm(), 2e-9);
  EXPECT_LT((filter.get_P() - filter.get_P().transpose()).norm(), 1e-12);
  EXPECT_LT((diagnostics.covariance_diag_after - expected.diagonal()).norm(), 2e-9);
  EXPECT_LT((diagnostics.last_dx - correction).norm(), 1e-12);
  Eigen::SelfAdjointEigenSolver<Cov> eigen(filter.get_P());
  ASSERT_EQ(eigen.info(), Eigen::Success);
  EXPECT_GT(eigen.eigenvalues().minCoeff(), 0.0);
}
}  // namespace

TEST(SimpleUpdateTangent, GravityInjectionDerivativeMatchesRetractionAtFiniteCorrections)
{
  const std::array<Eigen::Vector3d, 3> directions = {
    Eigen::Vector3d(2.0, -1.0, -9.0), Eigen::Vector3d(0.0, 0.0, -9.81), Eigen::Vector3d(0.0, 0.0, 9.81)};
  const std::array<Eigen::Vector2d, 4> corrections = {
    Eigen::Vector2d::Zero(), Eigen::Vector2d(1e-9, -2e-9), Eigen::Vector2d(0.2, -0.3), Eigen::Vector2d(-0.15, 0.24)};
  constexpr double kStep = 1e-6;
  for (const auto & direction : directions) {
    for (const auto & correction : corrections) {
      S2 gravity(direction);
      Eigen::Matrix<double, 3, 2> analytic;
      gravity.S2_Mx(analytic, correction);
      Eigen::Matrix<double, 3, 2> numerical;
      for (int col = 0; col < numerical.cols(); ++col) {
        Eigen::Vector2d plus_delta = correction;
        Eigen::Vector2d minus_delta = correction;
        plus_delta(col) += kStep;
        minus_delta(col) -= kStep;
        S2 plus = gravity;
        S2 minus = gravity;
        plus.boxplus(plus_delta);
        minus.boxplus(minus_delta);
        numerical.col(col) = (plus.get_vect() - minus.get_vect()) / (2.0 * kStep);
      }
      EXPECT_LT((analytic - numerical).norm(), 2e-8)
        << "direction=" << direction.transpose() << " correction=" << correction.transpose();
    }
  }
}

TEST(SimpleUpdateTangent, BothRotationsGravityAndCrossCovariancesFollowTheNewTangent)
{
  Observation H = Observation::Zero(8, state_ikfom::DOF);
  H.block<3, 3>(0, 3).setIdentity();
  H.block<3, 3>(3, 6).setIdentity();
  H.block<2, 2>(6, 21).setIdentity();
  Eigen::VectorXd residual(8);
  residual << 0.25, -0.18, 0.12, -0.2, 0.15, 0.3, 0.18, -0.22;
  checkUpdate(H, residual, true);
}

TEST(SimpleUpdateTangent, EuclideanObservationStillTransportsCorrelatedRotationAndGravity)
{
  Observation H = Observation::Zero(3, state_ikfom::DOF);
  H.block<3, 3>(0, 12).setIdentity();
  const Eigen::Vector3d residual(1.2, -0.8, 0.6);
  checkUpdate(H, residual, true);
}

TEST(SimpleUpdateTangent, ZeroCorrectionPreservesTheOrdinaryConditionalCovariance)
{
  Observation H = Observation::Identity(state_ikfom::DOF, state_ikfom::DOF);
  checkUpdate(H, Eigen::VectorXd::Zero(state_ikfom::DOF), false);
}

TEST(SimpleUpdateTangent, RepeatedUpdatesRemainFiniteSymmetricAndPositive)
{
  Filter filter;
  state_ikfom state = initialState();
  filter.change_x(state);
  Cov prior = correlatedPrior();
  filter.change_P(prior);
  const Observation H = Observation::Identity(state_ikfom::DOF, state_ikfom::DOF);
  const Eigen::VectorXd noise = Eigen::VectorXd::Constant(state_ikfom::DOF, 0.05);
  for (int cycle = 0; cycle < 80; ++cycle) {
    Eigen::VectorXd residual = Eigen::VectorXd::Zero(state_ikfom::DOF);
    residual.segment<3>(3) << 0.1, 0.08 * std::sin(cycle), -0.07;
    residual.segment<3>(6) << -0.04, 0.06, 0.05 * std::cos(cycle);
    residual.tail<2>() << 0.07, -0.09;
    filter.update_simple(H, residual, noise);
    const Cov covariance = filter.get_P();
    ASSERT_TRUE(covariance.allFinite()) << cycle;
    EXPECT_LT((covariance - covariance.transpose()).norm(), 1e-12) << cycle;
    Eigen::SelfAdjointEigenSolver<Cov> eigen(covariance);
    ASSERT_EQ(eigen.info(), Eigen::Success);
    ASSERT_GT(eigen.eigenvalues().minCoeff(), 0.0) << cycle;
  }
}

// Tests for rp_consider.hpp: the policy, the projector and the consider covariance form against the explicit
// Joseph form of a Schmidt-Kalman update on a small synthetic system.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include <limits>

#include "rp_consider.hpp"

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

fast_lio::RollPitchConsiderParams allRules()
{
  fast_lio::RollPitchConsiderParams p;
  p.enabled = true;
  p.while_engulfed = true;
  p.post_release_s = 2.5;
  p.min_rp_info = 1500.0;
  p.z_weak_min = 0.9;
  p.max_consider_s = 0.0;
  return p;
}
}  // namespace

TEST(ConsiderPolicy, DisabledNeverConsiders)
{
  fast_lio::RollPitchConsiderParams p;  // disabled
  fast_lio::RollPitchConsiderInputs in;
  in.engulf_latched = true;
  in.rp_info_prev = 10.0;
  in.z_weak_prev = 1.0;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
}

TEST(ConsiderPolicy, EachRuleFiresAloneAndNaNInputsDoNot)
{
  const auto p = allRules();
  fast_lio::RollPitchConsiderInputs in;
  in.rp_info_prev = 5000.0;
  in.z_weak_prev = 0.2;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  in.engulf_latched = true;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
  in.engulf_latched = false;
  in.since_release_s = 2.4;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
  in.since_release_s = 2.6;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  in.since_release_s = kNaN;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  in.since_release_s = -1.0;
  in.rp_info_prev = 800.0;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
  in.rp_info_prev = kNaN;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  in.rp_info_prev = 5000.0;
  in.z_weak_prev = 0.95;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
  in.z_weak_prev = kNaN;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
}

TEST(ConsiderPolicy, RulesCanBeSwitchedOffIndividuallyAndTheCapEndsAStretch)
{
  auto p = allRules();
  fast_lio::RollPitchConsiderInputs in;
  in.rp_info_prev = 100.0;
  in.z_weak_prev = 1.0;
  in.engulf_latched = true;
  p.while_engulfed = false;
  p.min_rp_info = 0.0;
  p.z_weak_min = 1.1;
  p.post_release_s = 0.0;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  p.min_rp_info = 1500.0;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
  p.max_consider_s = 10.0;
  in.consider_age_s = 10.5;
  EXPECT_FALSE(fast_lio::considerRollPitchThisScan(p, in));
  in.consider_age_s = 9.5;
  EXPECT_TRUE(fast_lio::considerRollPitchThisScan(p, in));
}

TEST(ConsiderProjector, KeepsOnlyTheVerticalComponentAndRejectsBadVerticals)
{
  const Eigen::Vector3d u = Eigen::Vector3d(0.2, -0.1, 0.97).normalized();
  const auto proj = fast_lio::yawOnlyProjector(u * 1.2);  // not quite unit: normalised
  ASSERT_TRUE(proj.valid);
  EXPECT_TRUE((proj.keep * u).isApprox(u, 1e-12));
  const Eigen::Vector3d h = u.cross(Eigen::Vector3d::UnitX()).normalized();
  EXPECT_NEAR((proj.keep * h).norm(), 0.0, 1e-12);
  EXPECT_TRUE(proj.keep.isApprox(proj.keep.transpose()));
  EXPECT_FALSE(fast_lio::yawOnlyProjector(Eigen::Vector3d::Zero()).valid);
  EXPECT_FALSE(fast_lio::yawOnlyProjector(Eigen::Vector3d(kNaN, 0.0, 1.0)).valid);
  EXPECT_FALSE(fast_lio::yawOnlyProjector(Eigen::Vector3d(0.0, 0.0, 5.0)).valid);
  EXPECT_TRUE(fast_lio::yawOnlyProjector(Eigen::Vector3d::Zero()).keep.isIdentity());
}

TEST(ConsiderCovariance, MatchesTheExplicitJosephFormOfTheProjectedGain)
{
  // 6 states (pos 0-2, rot 3-5), 8 point-to-plane rows, scalar noise r. K is the optimal gain of the full P; the
  // consider update projects its rotation rows with u u^T and must give the Joseph-form covariance exactly.
  constexpr int n = 6;
  constexpr int m = 8;
  Eigen::Matrix<double, n, n> G;
  G << 1.0, 0.2, 0.1, 0.05, -0.02, 0.03,
       0.0, 0.9, -0.1, 0.02, 0.04, -0.01,
       0.1, 0.0, 1.1, -0.03, 0.01, 0.02,
       0.0, 0.1, 0.0, 0.4, 0.05, -0.02,
       0.05, 0.0, 0.1, 0.0, 0.5, 0.03,
       0.0, 0.05, 0.0, 0.1, 0.0, 0.6;
  const Eigen::Matrix<double, n, n> P = G * G.transpose() * 1e-3 + Eigen::Matrix<double, n, n>::Identity() * 1e-4;
  Eigen::Matrix<double, m, n> H;
  for (int i = 0; i < m; ++i) {
    const double a = 0.7 * i;
    const Eigen::Vector3d normal(std::cos(a), std::sin(a), 0.3 * ((i % 3) - 1));
    const Eigen::Vector3d point(2.0 + i, -1.0 + 0.5 * i, 0.5 * (i % 2));
    H.block<1, 3>(i, 0) = normal.normalized().transpose();
    H.block<1, 3>(i, 3) = point.cross(normal.normalized()).transpose();
  }
  const double r = 1e-3;
  const Eigen::Matrix<double, m, m> S = H * P * H.transpose() + r * Eigen::Matrix<double, m, m>::Identity();
  const Eigen::Matrix<double, n, m> K = P * H.transpose() * S.inverse();
  const Eigen::Matrix<double, n, n> Kx = K * H;
  const Eigen::Matrix<double, n, n> A = Kx * P;
  const Eigen::Vector3d u = Eigen::Vector3d(0.1, 0.05, 1.0).normalized();
  Eigen::Matrix<double, n, n> M = Eigen::Matrix<double, n, n>::Identity();
  M.block<3, 3>(3, 3) = u * u.transpose();
  const Eigen::Matrix<double, n, m> Kt = M * K;
  const Eigen::Matrix<double, n, n> I = Eigen::Matrix<double, n, n>::Identity();
  const Eigen::Matrix<double, n, n> joseph = (I - Kt * H) * P * (I - Kt * H).transpose() + Kt * (r * Kt.transpose());
  const Eigen::Matrix<double, n, n> got = fast_lio::considerCovariance(P, A, M);
  EXPECT_LT((got - joseph).norm(), 1e-12 * joseph.norm());
  // the considered directions keep their prior variance, the identity projector gives the usual (I - K H) P
  const Eigen::Vector3d h = u.cross(Eigen::Vector3d::UnitX()).normalized();
  Eigen::Matrix<double, n, 1> hv = Eigen::Matrix<double, n, 1>::Zero();
  hv.segment<3>(3) = h;
  EXPECT_NEAR((hv.transpose() * got * hv)(0, 0), (hv.transpose() * P * hv)(0, 0), 1e-15);
  const Eigen::Matrix<double, n, n> plain = fast_lio::considerCovariance(P, A, I);
  EXPECT_LT((plain - (I - K * H) * P).norm(), 1e-12 * P.norm());
  EXPECT_TRUE(got.isApprox(got.transpose(), 1e-9));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, n, n>> es(got);
  EXPECT_GT(es.eigenvalues().minCoeff(), -1e-15);
}

TEST(ConsiderIncrement, ProjectedGainLeavesTheHorizontalRotationAtThePropagatedState)
{
  // dx = M K_h + (M K_x - I) dx_new: for the considered directions the increment is -dx_new, i.e. the iterate is
  // pulled back to the propagated state; the yaw direction and the other states keep the full increment.
  constexpr int n = 6;
  const Eigen::Vector3d u = Eigen::Vector3d::UnitZ();
  Eigen::Matrix<double, n, n> M = Eigen::Matrix<double, n, n>::Identity();
  M.block<3, 3>(3, 3) = u * u.transpose();
  Eigen::Matrix<double, n, 1> Kh;
  Kh << 0.01, -0.02, 0.005, 0.004, -0.003, 0.002;
  Eigen::Matrix<double, n, n> Kx = Eigen::Matrix<double, n, n>::Identity() * 0.5;
  Eigen::Matrix<double, n, 1> dx_new;
  dx_new << 0.1, 0.0, 0.0, 0.01, 0.02, 0.03;
  const Eigen::Matrix<double, n, n> I = Eigen::Matrix<double, n, n>::Identity();
  const Eigen::Matrix<double, n, 1> dx = M * Kh + (M * Kx - I) * dx_new;
  EXPECT_NEAR(dx(3), -dx_new(3), 1e-15);  // roll: back to the propagated state
  EXPECT_NEAR(dx(4), -dx_new(4), 1e-15);  // pitch
  EXPECT_NEAR(dx(5), Kh(5) + (0.5 - 1.0) * dx_new(5), 1e-15);  // yaw: the full increment
  EXPECT_NEAR(dx(0), Kh(0) + (0.5 - 1.0) * dx_new(0), 1e-15);  // position: the full increment
}

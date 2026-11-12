#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "covariance_reset.hpp"
#include "rebuild_relevel.hpp"

using fast_lio::assignGravityBlock;
using fast_lio::discardedVelocitySigma;
using fast_lio::kStateGravIndex;
using fast_lio::kStatePosIndex;
using fast_lio::kStateVelIndex;
using fast_lio::resetVelocityBlock;
using fast_lio::rotateWorldCovariance;
using fast_lio::StateCovariance;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A covariance with the structure a blind stretch leaves: velocity correlated with roll/pitch and position.
Eigen::Matrix<double, 23, 23> blindStretchCovariance()
{
  Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Identity() * 1e-4;
  for (int k = 0; k < 3; ++k) {
    P(kStateVelIndex + k, kStateVelIndex + k) = 1e-3;
    P(kStateVelIndex + k, 3 + k) = P(3 + k, kStateVelIndex + k) = -2e-3;  // roll/pitch <-> velocity (gravity leak)
    P(kStateVelIndex + k, k) = P(k, kStateVelIndex + k) = 5e-4;           // position <-> velocity
  }
  return P;
}
}  // namespace

TEST(CovarianceReset, VelocityBlockLosesEveryCrossTermAndTakesTheAssignedVariance)
{
  Eigen::Matrix<double, 23, 23> P = blindStretchCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  ASSERT_TRUE(resetVelocityBlock(P, 0.5));
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      const bool vel_r = r >= kStateVelIndex && r < kStateVelIndex + 3;
      const bool vel_c = c >= kStateVelIndex && c < kStateVelIndex + 3;
      if (vel_r && vel_c) {
        EXPECT_DOUBLE_EQ(P(r, c), r == c ? 0.25 : 0.0);
      } else if (vel_r || vel_c) {
        EXPECT_DOUBLE_EQ(P(r, c), 0.0);  // no relation to any other state error
      } else {
        EXPECT_DOUBLE_EQ(P(r, c), before(r, c));  // the rest of the filter's knowledge is untouched
      }
    }
  }
  // symmetric, as a covariance must stay
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 0.0);
}

TEST(CovarianceReset, RejectsBadSigmaAndOutOfRangeIndex)
{
  Eigen::Matrix<double, 23, 23> P = blindStretchCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  EXPECT_FALSE(resetVelocityBlock(P, kNaN));
  EXPECT_FALSE(resetVelocityBlock(P, -0.1));
  EXPECT_FALSE(resetVelocityBlock(P, 0.5, 21));  // 21 + 3 > 23
  EXPECT_FALSE(resetVelocityBlock(P, 0.5, -1));
  EXPECT_DOUBLE_EQ((P - before).norm(), 0.0);
  // a zero sigma is a legitimate assignment (a measured standstill): the block becomes exactly zero
  EXPECT_TRUE(resetVelocityBlock(P, 0.0));
  EXPECT_DOUBLE_EQ(P.block(kStateVelIndex, kStateVelIndex, 3, 3).norm(), 0.0);
}

TEST(CovarianceReset, WorksOnADynamicMatrixWithAnotherLayout)
{
  Eigen::MatrixXd P = Eigen::MatrixXd::Constant(9, 9, 0.3);
  ASSERT_TRUE(resetVelocityBlock(P, 0.1, 6));
  EXPECT_DOUBLE_EQ(P(6, 6), 0.01);
  EXPECT_DOUBLE_EQ(P(8, 8), 0.01);
  EXPECT_DOUBLE_EQ(P(6, 7), 0.0);
  EXPECT_DOUBLE_EQ(P(0, 6), 0.0);
  EXPECT_DOUBLE_EQ(P(0, 1), 0.3);
}

TEST(CovarianceReset, DiscardedVelocitySigmaIsThePlatformSpeedWithAFloor)
{
  EXPECT_DOUBLE_EQ(discardedVelocitySigma(1.5, 0.05), 1.5);
  EXPECT_DOUBLE_EQ(discardedVelocitySigma(0.0, 0.05), 0.05);
  EXPECT_DOUBLE_EQ(discardedVelocitySigma(kNaN, 0.05), 0.05);
}

// ── The rebuild relevel (rebuild_relevel.hpp) ────────────────────────────────
// The frame turns by L about the current position: P follows it.
namespace
{
// Every block correlated with every other.
StateCovariance denseCovariance()
{
  StateCovariance A;
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      A(r, c) = 0.01 * std::sin(1.3 * r + 0.7 * c + 0.11 * r * c);
    }
  }
  return A * A.transpose() + 1e-4 * StateCovariance::Identity();
}

// An orthonormal basis of the plane tangent to g, the shape of the S2 state's B(g).
Eigen::Matrix<double, 3, 2> tangentBasis(const Eigen::Vector3d & g)
{
  Eigen::Matrix<double, 3, 2> B;
  B.col(0) = g.unitOrthogonal();
  B.col(1) = g.normalized().cross(B.col(0));
  return B;
}

bool inBlock(int i, int first, int size)
{
  return i >= first && i < first + size;
}

// An 8.7° tilt that L takes out; the gravity state turns with the frame.
struct Relevel
{
  Eigen::Matrix3d L =
    Eigen::AngleAxisd(8.7 * M_PI / 180.0, Eigen::Vector3d(1.0, 2.0, 0.0).normalized()).toRotationMatrix();
  Eigen::Vector3d g_old = -9.809 * (L.transpose() * Eigen::Vector3d::UnitZ());
  Eigen::Matrix<double, 3, 2> B_old = tangentBasis(g_old);
  Eigen::Matrix<double, 3, 2> B_new = tangentBasis(L * g_old);
};
}  // namespace

TEST(RelevelCovariance, WorldRowsTurnWithTheFrameAndBodyRowsStay)
{
  const Relevel turn;
  const Eigen::Matrix3d & L = turn.L;
  const StateCovariance P0 = denseCovariance();
  StateCovariance P = P0;
  rotateWorldCovariance(P, L, turn.B_old, turn.B_new);
  constexpr int pos = kStatePosIndex;
  constexpr int vel = kStateVelIndex;
  constexpr int grav = kStateGravIndex;
  // world-frame blocks: L on both sides
  EXPECT_NEAR((P.block<3, 3>(pos, pos) - L * P0.block<3, 3>(pos, pos) * L.transpose()).norm(), 0.0, 1e-15);
  EXPECT_NEAR((P.block<3, 3>(vel, vel) - L * P0.block<3, 3>(vel, vel) * L.transpose()).norm(), 0.0, 1e-15);
  EXPECT_NEAR((P.block<3, 3>(pos, vel) - L * P0.block<3, 3>(pos, vel) * L.transpose()).norm(), 0.0, 1e-15);
  // world frame against body frame (rotation 3, extrinsics 6 and 9, biases 15 and 18): L on the world side only
  for (const int body : {3, 6, 9, 15, 18}) {
    EXPECT_NEAR((P.block<3, 3>(pos, body) - L * P0.block<3, 3>(pos, body)).norm(), 0.0, 1e-15) << body;
    EXPECT_NEAR((P.block<3, 3>(vel, body) - L * P0.block<3, 3>(vel, body)).norm(), 0.0, 1e-15) << body;
  }
  // body-frame blocks keep every element
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      const bool world = inBlock(r, pos, 3) || inBlock(r, vel, 3) || inBlock(r, grav, 2) || inBlock(c, pos, 3) ||
                         inBlock(c, vel, 3) || inBlock(c, grav, 2);
      if (!world) {
        EXPECT_EQ(P(r, c), P0(r, c)) << r << "," << c;
      }
    }
  }
  // the gravity tangent changes basis: the same world-frame covariance of the direction, and of its cross terms
  const auto & B_old = turn.B_old;
  const auto & B_new = turn.B_new;
  EXPECT_NEAR((B_new * P.block<2, 2>(grav, grav) * B_new.transpose() -
               L * B_old * P0.block<2, 2>(grav, grav) * B_old.transpose() * L.transpose())
                .norm(),
              0.0,
              1e-15);
  EXPECT_NEAR(
    (B_new * P.block<2, 3>(grav, pos) - L * B_old * P0.block<2, 3>(grav, pos) * L.transpose()).norm(), 0.0, 1e-15);
  EXPECT_NEAR((B_new * P.block<2, 3>(grav, 3) - L * B_old * P0.block<2, 3>(grav, 3)).norm(), 0.0, 1e-15);
}

// The transform is orthogonal: P stays a symmetric covariance with the same eigenvalues, so the change of
// coordinates neither creates nor loses information.
TEST(RelevelCovariance, StaysASymmetricCovarianceWithTheSameSpectrum)
{
  const Relevel turn;
  const StateCovariance P0 = denseCovariance();
  StateCovariance P = P0;
  rotateWorldCovariance(P, turn.L, turn.B_old, turn.B_new);
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-15);
  const Eigen::SelfAdjointEigenSolver<StateCovariance> before(P0);
  const Eigen::SelfAdjointEigenSolver<StateCovariance> after(P);
  EXPECT_GT(after.eigenvalues().minCoeff(), 0.0);
  EXPECT_NEAR((after.eigenvalues() - before.eigenvalues()).norm(), 0.0, 1e-14);
}

TEST(RelevelCovariance, AnIdentityTurnLeavesEveryElement)
{
  Eigen::Matrix<double, 3, 2> B;  // tangent to a vertical gravity, exactly orthonormal
  B << 1.0, 0.0, 0.0, 1.0, 0.0, 0.0;
  const StateCovariance P0 = denseCovariance();
  StateCovariance P = P0;
  rotateWorldCovariance(P, Eigen::Matrix3d::Identity(), B, B);
  EXPECT_TRUE(P == P0);
}

// A window reference assigns the gravity state: its block takes the measurement's variance and no relation to any
// other state error; the rest of the filter's knowledge is untouched.
TEST(RelevelCovariance, AnAssignedGravityLosesItsCrossTerms)
{
  const StateCovariance P0 = denseCovariance();
  StateCovariance P = P0;
  const double variance = 0.008 * 0.008;
  assignGravityBlock(P, variance);
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      if (inBlock(r, kStateGravIndex, 2) || inBlock(c, kStateGravIndex, 2)) {
        EXPECT_EQ(P(r, c), r == c ? variance : 0.0) << r << "," << c;
      } else {
        EXPECT_EQ(P(r, c), P0(r, c)) << r << "," << c;
      }
    }
  }
}

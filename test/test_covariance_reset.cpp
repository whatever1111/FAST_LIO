#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>

#include "covariance_reset.hpp"

using fast_lio::discardedVelocitySigma;
using fast_lio::kStateVelIndex;
using fast_lio::resetVelocityBlock;

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

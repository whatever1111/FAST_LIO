#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "imu_gap_prior.hpp"

using fast_lio::applyImuGapInflation;
using fast_lio::ImuGapInflation;
using fast_lio::imuGapInflation;
using fast_lio::ImuGapPriorParams;
using fast_lio::ImuGapSummary;
using fast_lio::isImuGap;
using fast_lio::kStatePosIndex;
using fast_lio::kStateRotIndex;
using fast_lio::kStateVelIndex;
using fast_lio::recordImuGap;
using fast_lio::rotationAboutGravityInput;
using fast_lio::zeroAccelerationInput;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kG = 9.81;
const Eigen::Vector3d kUp = Eigen::Vector3d::UnitZ();

ImuGapPriorParams enabledParams()
{
  ImuGapPriorParams p;
  p.enabled = true;
  p.min_gap_s = 0.05;
  p.accel_max = 1.0;
  p.accel_max_z = 0.2;
  p.rate_rp_max = 0.02;
  p.rate_yaw_max = 0.2;
  return p;
}

// A covariance with every cross term populated, the structure a lidar update and a blind stretch leave behind.
Eigen::Matrix<double, 23, 23> denseCovariance()
{
  Eigen::Matrix<double, 23, 23> A;
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      A(r, c) = std::sin(1.0 + r * 23.0 + c) * 0.01;
    }
  }
  return A * A.transpose() + Eigen::Matrix<double, 23, 23>::Identity() * 1e-4;
}

// 0 position, 1 velocity, 2 attitude, 3 the rest of the state
int group(int i)
{
  if (i >= kStatePosIndex && i < kStatePosIndex + 3) {
    return 0;
  }
  if (i >= kStateVelIndex && i < kStateVelIndex + 3) {
    return 1;
  }
  return (i >= kStateRotIndex && i < kStateRotIndex + 3) ? 2 : 3;
}
}  // namespace

TEST(ImuGapPrior, GapThresholdIsExclusiveAndOnlyForFiniteIntervals)
{
  const ImuGapPriorParams p = enabledParams();
  EXPECT_FALSE(isImuGap(0.005, p));  // the nominal 200 Hz period
  EXPECT_FALSE(isImuGap(0.05, p));   // exactly at the threshold
  EXPECT_TRUE(isImuGap(0.0501, p));
  EXPECT_TRUE(isImuGap(1.085, p));  // the 0826 hole
  EXPECT_FALSE(isImuGap(0.0, p));
  EXPECT_FALSE(isImuGap(-0.2, p));
  EXPECT_FALSE(isImuGap(kNaN, p));
  EXPECT_FALSE(isImuGap(kInf, p));
}

TEST(ImuGapPrior, DisabledOrInvalidParametersBridgeNothing)
{
  ImuGapPriorParams p = enabledParams();
  p.enabled = false;
  EXPECT_FALSE(isImuGap(1.0, p));
  p = enabledParams();
  p.min_gap_s = 0.0;
  EXPECT_FALSE(fast_lio::imuGapPriorParamsValid(p));
  EXPECT_FALSE(isImuGap(1.0, p));
  p = enabledParams();
  p.accel_max = kNaN;
  EXPECT_FALSE(isImuGap(1.0, p));
  p = enabledParams();
  p.rate_yaw_max = -0.1;
  EXPECT_FALSE(isImuGap(1.0, p));
  p = enabledParams();
  p.accel_max_z = kInf;
  EXPECT_FALSE(isImuGap(1.0, p));
  EXPECT_TRUE(fast_lio::imuGapPriorParamsValid(enabledParams()));
}

TEST(ImuGapPrior, ZeroAccelerationInputCancelsGravityForAnyAttitudeAndBias)
{
  const Eigen::Vector3d grav(0.0, 0.0, -kG);
  // Level and without bias: the accelerometer of a body at rest reads +g along its z axis.
  const Eigen::Vector3d level = zeroAccelerationInput(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), grav);
  EXPECT_NEAR((level - Eigen::Vector3d(0.0, 0.0, kG)).norm(), 0.0, 1e-12);

  const Eigen::Matrix3d attitudes[] = {
    Eigen::Matrix3d::Identity(),
    Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -0.5, 0.8).normalized()).toRotationMatrix(),
    Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix(),
    Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix(),
  };
  const Eigen::Vector3d ba(0.05, -0.02, 0.11);
  const Eigen::Vector3d tilted_grav = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()).toRotationMatrix() * grav;
  for (const auto & R : attitudes) {
    for (const auto & g : {grav, tilted_grav}) {
      const Eigen::Vector3d a = zeroAccelerationInput(R, ba, g);
      // IKFoM's velocity derivative, use-ikfom.hpp get_f: R (a - ba) + g
      EXPECT_NEAR((R * (a - ba) + g).norm(), 0.0, 1e-12);
    }
  }
}

TEST(ImuGapPrior, InflationFollowsTheGapModel)
{
  const ImuGapPriorParams p = enabledParams();
  const ImuGapInflation infl = imuGapInflation(1.0, kUp, Eigen::Matrix3d::Identity(), p);
  ASSERT_TRUE(infl.valid);
  EXPECT_DOUBLE_EQ(infl.gap_s, 1.0);
  // sigma_v = a T, sigma_p = a T^2 / 2, horizontal / vertical
  EXPECT_DOUBLE_EQ(infl.sigma_vel_h, 1.0);
  EXPECT_DOUBLE_EQ(infl.sigma_vel_z, 0.2);
  EXPECT_DOUBLE_EQ(infl.sigma_pos_h, 0.5);
  EXPECT_DOUBLE_EQ(infl.sigma_pos_z, 0.1);
  EXPECT_DOUBLE_EQ(infl.sigma_rp, 0.02);
  EXPECT_DOUBLE_EQ(infl.sigma_yaw, 0.2);
  const Eigen::Vector3d vel_diag(1.0, 1.0, 0.04);
  const Eigen::Vector3d pos_diag(0.25, 0.25, 0.01);
  const Eigen::Vector3d rot_diag(0.0004, 0.0004, 0.04);
  EXPECT_NEAR((infl.vel - Eigen::Matrix3d(vel_diag.asDiagonal())).norm(), 0.0, 1e-12);
  EXPECT_NEAR((infl.pos - Eigen::Matrix3d(pos_diag.asDiagonal())).norm(), 0.0, 1e-12);
  EXPECT_NEAR((infl.rot - Eigen::Matrix3d(rot_diag.asDiagonal())).norm(), 0.0, 1e-12);
  // Each block is a covariance.
  for (const Eigen::Matrix3d * block : {&infl.pos, &infl.vel, &infl.rot}) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(*block);
    EXPECT_GT(eig.eigenvalues().minCoeff(), 0.0);
    EXPECT_NEAR((*block - block->transpose()).norm(), 0.0, 1e-15);
  }
}

TEST(ImuGapPrior, InflationScalesWithTheGapLength)
{
  const ImuGapPriorParams p = enabledParams();
  const ImuGapInflation half = imuGapInflation(0.5, kUp, Eigen::Matrix3d::Identity(), p);
  const ImuGapInflation full = imuGapInflation(1.0, kUp, Eigen::Matrix3d::Identity(), p);
  ASSERT_TRUE(half.valid && full.valid);
  EXPECT_NEAR((full.vel - 4.0 * half.vel).norm(), 0.0, 1e-12);   // T^2
  EXPECT_NEAR((full.pos - 16.0 * half.pos).norm(), 0.0, 1e-12);  // T^4
  EXPECT_NEAR((full.rot - 4.0 * half.rot).norm(), 0.0, 1e-12);   // T^2
}

TEST(ImuGapPrior, InflationAtTheCoverageLimitStaysFinite)
{
  // 2 s is the longest gap the m20 profile bridges (imu_coverage.max_gap_s); beyond it the scan is skipped.
  const ImuGapInflation infl = imuGapInflation(2.0, kUp, Eigen::Matrix3d::Identity(), enabledParams());
  ASSERT_TRUE(infl.valid);
  EXPECT_DOUBLE_EQ(infl.sigma_pos_h, 2.0);
  EXPECT_DOUBLE_EQ(infl.sigma_vel_h, 2.0);
  EXPECT_DOUBLE_EQ(infl.sigma_yaw, 0.4);
  EXPECT_TRUE(infl.pos.allFinite() && infl.vel.allFinite() && infl.rot.allFinite());
}

TEST(ImuGapPrior, AttitudeInflationIsAboutGravityInTheBodyFrame)
{
  const ImuGapPriorParams p = enabledParams();
  // Body rolled 90 deg: world up is the body's +y or -y axis, so the yaw variance belongs on body y.
  const Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const ImuGapInflation infl = imuGapInflation(1.0, kUp, R, p);
  ASSERT_TRUE(infl.valid);
  const Eigen::Vector3d up_body = R.transpose() * kUp;
  EXPECT_NEAR(up_body.dot(infl.rot * up_body), 0.04, 1e-12);
  EXPECT_NEAR(Eigen::Vector3d::UnitX().dot(infl.rot * Eigen::Vector3d::UnitX()), 0.0004, 1e-12);
  const Eigen::Vector3d other = up_body.cross(Eigen::Vector3d::UnitX());
  EXPECT_NEAR(other.dot(infl.rot * other), 0.0004, 1e-12);
  // The position/velocity blocks are world-frame and do not care about the attitude.
  const ImuGapInflation level = imuGapInflation(1.0, kUp, Eigen::Matrix3d::Identity(), p);
  EXPECT_NEAR((infl.vel - level.vel).norm(), 0.0, 1e-12);
  EXPECT_NEAR((infl.pos - level.pos).norm(), 0.0, 1e-12);
}

TEST(ImuGapPrior, VerticalFollowsTheGivenGravityNotTheWorldZ)
{
  // The FE world frame is tilted against gravity by a fraction of a degree to a few degrees; the vertical bound
  // belongs on the gravity axis (and the up vector need not be unit length).
  const Eigen::Vector3d up = Eigen::Vector3d(0.0, std::sin(0.05), std::cos(0.05)) * kG;
  const ImuGapInflation infl = imuGapInflation(1.0, up, Eigen::Matrix3d::Identity(), enabledParams());
  ASSERT_TRUE(infl.valid);
  const Eigen::Vector3d u = up.normalized();
  EXPECT_NEAR(u.dot(infl.vel * u), 0.04, 1e-12);
  EXPECT_NEAR(Eigen::Vector3d::UnitX().dot(infl.vel * Eigen::Vector3d::UnitX()), 1.0, 1e-12);
}

TEST(ImuGapPrior, ZeroGapAddsNothingAndLeavesTheCovarianceAlone)
{
  const ImuGapInflation infl = imuGapInflation(0.0, kUp, Eigen::Matrix3d::Identity(), enabledParams());
  ASSERT_TRUE(infl.valid);
  EXPECT_DOUBLE_EQ(infl.pos.norm() + infl.vel.norm() + infl.rot.norm(), 0.0);
  Eigen::Matrix<double, 23, 23> P = denseCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  EXPECT_TRUE(applyImuGapInflation(P, infl));
  EXPECT_DOUBLE_EQ((P - before).norm(), 0.0);
}

TEST(ImuGapPrior, InvalidInputsGiveNoInflationAndLeaveTheCovarianceAlone)
{
  const ImuGapPriorParams p = enabledParams();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  EXPECT_FALSE(imuGapInflation(kNaN, kUp, I, p).valid);
  EXPECT_FALSE(imuGapInflation(kInf, kUp, I, p).valid);
  EXPECT_FALSE(imuGapInflation(-0.1, kUp, I, p).valid);
  EXPECT_FALSE(imuGapInflation(1.0, Eigen::Vector3d::Zero(), I, p).valid);
  EXPECT_FALSE(imuGapInflation(1.0, Eigen::Vector3d(kNaN, 0.0, 1.0), I, p).valid);
  Eigen::Matrix3d bad = I;
  bad(1, 2) = kInf;
  EXPECT_FALSE(imuGapInflation(1.0, kUp, bad, p).valid);
  EXPECT_FALSE(imuGapInflation(1.0, kUp, Eigen::Matrix3d::Zero(), p).valid);
  ImuGapPriorParams invalid = p;
  invalid.rate_rp_max = kNaN;
  EXPECT_FALSE(imuGapInflation(1.0, kUp, I, invalid).valid);

  Eigen::Matrix<double, 23, 23> P = denseCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  EXPECT_FALSE(applyImuGapInflation(P, imuGapInflation(kNaN, kUp, I, p)));
  EXPECT_DOUBLE_EQ((P - before).norm(), 0.0);
}

TEST(ImuGapPrior, ApplyDecouplesThePoseAndVelocityAndAddsTheModel)
{
  const ImuGapInflation infl = imuGapInflation(1.085, kUp, Eigen::Matrix3d::Identity(), enabledParams());
  ASSERT_TRUE(infl.valid);
  Eigen::Matrix<double, 23, 23> P = denseCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  ASSERT_TRUE(applyImuGapInflation(P, infl));
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      if (group(r) != group(c)) {
        EXPECT_DOUBLE_EQ(P(r, c), 0.0) << r << "," << c;  // nothing links the gap's motion to the rest
      } else if (group(r) == 3) {
        EXPECT_DOUBLE_EQ(P(r, c), before(r, c)) << r << "," << c;  // biases, gravity, extrinsics untouched
      }
    }
  }
  const auto blk = [](const Eigen::Matrix<double, 23, 23> & M, int r, int c) {
    return M.block<3, 3>(r, c);
  };
  EXPECT_NEAR((blk(P, 0, 0) - blk(before, 0, 0) - infl.pos).norm(), 0.0, 1e-12);
  EXPECT_NEAR((blk(P, 12, 12) - blk(before, 12, 12) - infl.vel).norm(), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(blk(P, 0, 12).norm(), 0.0);  // position and velocity decoupled as well
  EXPECT_NEAR((blk(P, 3, 3) - blk(before, 3, 3) - infl.rot).norm(), 0.0, 1e-12);
  EXPECT_NEAR((P - P.transpose()).norm(), 0.0, 1e-15);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 23, 23>> eig(P);
  EXPECT_GT(eig.eigenvalues().minCoeff(), 0.0);  // still a covariance
}

TEST(ImuGapPrior, ApplyRejectsLayoutsThatDoNotFit)
{
  const ImuGapInflation infl = imuGapInflation(0.5, kUp, Eigen::Matrix3d::Identity(), enabledParams());
  Eigen::Matrix<double, 23, 23> P = denseCovariance();
  const Eigen::Matrix<double, 23, 23> before = P;
  EXPECT_FALSE(applyImuGapInflation(P, infl, 0, 2, 12));   // pos and rot overlap
  EXPECT_FALSE(applyImuGapInflation(P, infl, 0, 3, 21));   // vel runs past the matrix
  EXPECT_FALSE(applyImuGapInflation(P, infl, -1, 3, 12));  // negative index
  EXPECT_DOUBLE_EQ((P - before).norm(), 0.0);
  Eigen::MatrixXd rect = Eigen::MatrixXd::Identity(23, 22);
  EXPECT_FALSE(applyImuGapInflation(rect, infl));
  // Another layout on a dynamic matrix works as well.
  Eigen::MatrixXd small = Eigen::MatrixXd::Constant(9, 9, 0.3) + Eigen::MatrixXd::Identity(9, 9);
  ASSERT_TRUE(applyImuGapInflation(small, infl, 0, 3, 6));
  EXPECT_DOUBLE_EQ(small(0, 3), 0.0);  // pos-rot decoupled
  EXPECT_DOUBLE_EQ(small(0, 6), 0.0);  // pos-vel decoupled
  EXPECT_DOUBLE_EQ(small(0, 1), 0.3);  // inside a block: kept
  EXPECT_NEAR(small(6, 6), 1.3 + infl.vel(0, 0), 1e-12);
}

TEST(ImuGapPrior, SummaryKeepsTheLongestGap)
{
  const ImuGapPriorParams p = enabledParams();
  ImuGapSummary summary;
  recordImuGap(&summary, imuGapInflation(0.115, kUp, Eigen::Matrix3d::Identity(), p));
  recordImuGap(&summary, imuGapInflation(1.085, kUp, Eigen::Matrix3d::Identity(), p));
  recordImuGap(&summary, imuGapInflation(0.25, kUp, Eigen::Matrix3d::Identity(), p));
  recordImuGap(&summary, imuGapInflation(kNaN, kUp, Eigen::Matrix3d::Identity(), p));  // ignored
  recordImuGap(&summary, imuGapInflation(0.0, kUp, Eigen::Matrix3d::Identity(), p));   // no gap, ignored
  recordImuGap(nullptr, imuGapInflation(0.3, kUp, Eigen::Matrix3d::Identity(), p));    // no crash
  EXPECT_EQ(summary.gaps, 3);
  EXPECT_NEAR(summary.total_s, 1.45, 1e-12);
  EXPECT_DOUBLE_EQ(summary.longest_s, 1.085);
  EXPECT_DOUBLE_EQ(summary.longest.sigma_vel_h, 1.085);
}

TEST(ImuGapPrior, ExtrapolationKeepsOnlyTheRotationAboutGravity)
{
  const Eigen::Vector3d grav(0.0, 0.0, -kG);
  const Eigen::Vector3d gyro(0.3, -0.2, 0.5);
  // Level: yaw is the body z rate.
  EXPECT_NEAR((rotationAboutGravityInput(gyro, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(), grav) -
               Eigen::Vector3d(0.0, 0.0, 0.5))
                .norm(),
              0.0,
              1e-12);
  // The bias is removed before the split and handed back, so the model sees (0, 0, 0.47).
  const Eigen::Vector3d bg(0.01, 0.02, 0.03);
  EXPECT_NEAR(
    (rotationAboutGravityInput(gyro, bg, Eigen::Matrix3d::Identity(), grav) - Eigen::Vector3d(0.01, 0.02, 0.5)).norm(),
    0.0,
    1e-12);
  // Rolled 90 deg about x: world up is body +y, so the body y rate is the one kept.
  const Eigen::Matrix3d rolled = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
  EXPECT_NEAR(
    (rotationAboutGravityInput(gyro, Eigen::Vector3d::Zero(), rolled, grav) - Eigen::Vector3d(0.0, -0.2, 0.0)).norm(),
    0.0,
    1e-12);
  // Degenerate gravity or non-finite input: nothing to split, the input comes back.
  EXPECT_EQ(rotationAboutGravityInput(gyro, bg, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()), gyro);
  const Eigen::Vector3d bad(kNaN, 0.0, 0.1);
  EXPECT_TRUE(rotationAboutGravityInput(bad, bg, Eigen::Matrix3d::Identity(), grav).hasNaN());
}

// imu_gap_prior.hpp — carry the state across an IMU dropout: rotate with the gyro, hold the velocity, and say so in P.
//
// A recorder or host stall drops IMU samples while the platform keeps moving (m20 0826, end of the bag: 1.085 s of
// /lio/imu missing while the dog walked 1.33 m, then 85-250 ms holes while it turned 45 deg). FAST-LIO had two
// answers to that and both lose:
//  - upstream integrates the mean of the two samples that bracket the gap in ONE step. The gyro half is right (the
//    turn is smooth on the scale of the gap and the average recovers it); the accelerometer half is not: the mean of
//    two gait samples becomes a constant acceleration for the whole gap (2.37 m/s after a 0.9 s hole walked at
//    1.29 m/s, z -0.9..-1.5 m after the stop);
//  - the coverage skip (imu_coverage_policy.hpp) holds the state and its covariance: the motion of the gap is lost
//    and P still says cm / 0.2 deg, so the next registration converges to a self-consistent wrong minimum.
// Published systems do neither: constant-velocity priors with time-scaled process noise (Cartographer, KISS-ICP,
// RKO-LIO), the last IMU reading stretched over the uncovered period (OpenVINS), a state reset always paired with a
// covariance reset (PX4, OpenVINS).
//
// Across a gap the state is rotated with the bracketing gyro average, the world-frame velocity is held (zero world
// acceleration: zeroAccelerationInput) and the position is extrapolated with it. P gets what that model does not
// know, for a gap of length T:
//  - an unknown acceleration of 1-sigma a_h horizontally and a_z along gravity: sigma_v = a T and sigma_p = a T^2 / 2
//    (the constant-acceleration bounds);
//  - a rate error the bracketing average misses: w_rp T about the horizontal axes and w_yaw T about gravity, on top
//    of the T^2 Q_gyro that IKFoM's single long step already adds.
// P is then re-blocked the way covariance_reset.hpp treats an assigned velocity: position, velocity, attitude and the
// rest of the state keep their own blocks and lose every cross term between them. The motion inside the gap has
// nothing to do with the biases, gravity or the extrinsics, and a position innovation after the gap must not be
// explained as a roll/pitch error through P(rot, pos), the channel of the doorway swings (docs/PGO_LOOP_TUNING.md
// section 7.7). (The gyro-bias share of the attitude error, T sigma_bg, is small next to the rate term and goes with
// it.) Position and velocity are decoupled as well, although a constant unknown acceleration would correlate them.
// With that correlation (sqrt(3)/2, tried on m20 0826) the registration after a short gap reads its position noise
// as a velocity change with a gain of about 2/T: after the 0.095 s gap just before the 1.085 s hole the velocity went
// from 1.17 to 1.42 m/s (FP 1.16), the hole carried that for a second, and the pose came out of it 0.88 m off
// (0.08 m decoupled). Decoupled, the velocity is re-estimated through the propagation of the scans that follow.
//
// Pure C++ + Eigen. No ROS, no estimator types: the caller passes attitude, bias and gravity as Eigen types and the
// 23x23 IKFoM covariance with the indices of its pos, rot and vel blocks.

#pragma once

#include <Eigen/Core>

#include <cmath>

#include "covariance_reset.hpp"

namespace fast_lio
{

/// FAST-LIO state layout (velocity: kStateVelIndex in covariance_reset.hpp): pos 0-2, rot 3-5.
constexpr int kStatePosIndex = 0;
constexpr int kStateRotIndex = 3;
constexpr double kImuGapDegToRad = 0.017453292519943295;

struct ImuGapPriorParams
{
  bool enabled = false;      ///< imu_gap_prior_en
  double min_gap_s = 0.05;   ///< imu_gap_min_s: an IMU interval longer than this is a gap (the nominal period is 5 ms)
  double accel_max = 1.0;    ///< imu_gap_accel_max (m/s^2): 1-sigma horizontal acceleration inside a gap
  double accel_max_z = 0.2;  ///< imu_gap_accel_max_z (m/s^2): the same along gravity (a walking platform stays level)
  double rate_rp_max = 0.0;  ///< rad/s (imu_gap_rate_rp_max is in deg/s): roll/pitch rate the gyro average misses
  double rate_yaw_max = 10.0 * kImuGapDegToRad;  ///< rad/s (imu_gap_rate_yaw_max in deg/s): the same about gravity
  /// imu_gap_empty_scans: also carry a scan that has no IMU sample at all (the lidar kept recording through the
  /// IMU dropout), at the held velocity and the last yaw rate, instead of dropping it unregistered.
  bool empty_scans = false;
};

/// Every bound finite and non-negative, the gap threshold positive.
inline bool imuGapPriorParamsValid(const ImuGapPriorParams & p)
{
  const auto bound = [](double v) {
    return std::isfinite(v) && v >= 0.0;
  };
  return std::isfinite(p.min_gap_s) && p.min_gap_s > 0.0 && bound(p.accel_max) && bound(p.accel_max_z) &&
         bound(p.rate_rp_max) && bound(p.rate_yaw_max);
}

/// Whether an interval between two IMU samples (or the stretch past the last one) is a gap the prior bridges. The
/// threshold is exclusive; a non-finite interval never is (the coverage check owns those).
inline bool isImuGap(double interval_s, const ImuGapPriorParams & p)
{
  return p.enabled && imuGapPriorParamsValid(p) && std::isfinite(interval_s) && interval_s > p.min_gap_s;
}

/// The accelerometer input under which IKFoM's velocity does not change. Its model is dv/dt = R (a - ba) + g, zero for
/// a = ba - R^T g: the specific force of a body at rest plus the bias the model subtracts again.
inline Eigen::Vector3d
zeroAccelerationInput(const Eigen::Matrix3d & R_wb, const Eigen::Vector3d & ba, const Eigen::Vector3d & grav_w)
{
  return ba - R_wb.transpose() * grav_w;
}

/// The gyro input for a stretch past the last IMU sample (no sample closes it): only the rotation about gravity is
/// carried on. A walking or driving platform turns steadily but rocks about level, and a held roll/pitch rate of the
/// gait (45 deg/s at 354.2 s on m20 0826) would tilt the state by degrees within one scan. The rate is split in the
/// body frame after removing the bias the model subtracts again; a degenerate gravity or attitude returns the input.
inline Eigen::Vector3d rotationAboutGravityInput(const Eigen::Vector3d & gyro,
                                                 const Eigen::Vector3d & bg,
                                                 const Eigen::Matrix3d & R_wb,
                                                 const Eigen::Vector3d & grav_w)
{
  const Eigen::Vector3d up_body = -(R_wb.transpose() * grav_w);
  const double norm = up_body.norm();
  if (!gyro.allFinite() || !bg.allFinite() || !up_body.allFinite() || !(norm > 0.0)) {
    return gyro;
  }
  const Eigen::Vector3d u = up_body / norm;
  const Eigen::Vector3d rate = gyro - bg;
  return rate.dot(u) * u + bg;
}

/// The covariance a gap adds. pos and vel are world-frame blocks, rot is IKFoM's rotation error, which is body-frame
/// (R = R_est Exp(dtheta)). The sigmas are for the log.
struct ImuGapInflation
{
  bool valid = false;
  double gap_s = 0.0;
  Eigen::Matrix3d pos = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d vel = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d rot = Eigen::Matrix3d::Zero();
  double sigma_pos_h = 0.0;
  double sigma_pos_z = 0.0;
  double sigma_vel_h = 0.0;
  double sigma_vel_z = 0.0;
  double sigma_rp = 0.0;
  double sigma_yaw = 0.0;
};

/// The inflation for a gap of gap_s seconds. up_w is the world-frame "up" (the caller passes -gravity), R_wb the
/// attitude at the end of the gap. A zero gap is valid and adds nothing; a negative or non-finite gap, a degenerate
/// up vector, a non-finite attitude or invalid parameters give an invalid inflation.
inline ImuGapInflation
imuGapInflation(double gap_s, const Eigen::Vector3d & up_w, const Eigen::Matrix3d & R_wb, const ImuGapPriorParams & p)
{
  ImuGapInflation out;
  if (!std::isfinite(gap_s) || gap_s < 0.0 || !imuGapPriorParamsValid(p) || !up_w.allFinite() || !R_wb.allFinite()) {
    return out;
  }
  const double up_norm = up_w.norm();
  const Eigen::Vector3d up_body = R_wb.transpose() * up_w;
  const double up_body_norm = up_body.norm();
  if (!(up_norm > 0.0) || !(up_body_norm > 0.0)) {
    return out;
  }
  const Eigen::Vector3d u = up_w / up_norm;
  const Eigen::Vector3d ub = up_body / up_body_norm;
  const Eigen::Matrix3d vertical = u * u.transpose();
  const Eigen::Matrix3d horizontal = Eigen::Matrix3d::Identity() - vertical;
  const Eigen::Matrix3d vertical_body = ub * ub.transpose();
  const Eigen::Matrix3d horizontal_body = Eigen::Matrix3d::Identity() - vertical_body;

  const double t = gap_s;
  out.sigma_vel_h = p.accel_max * t;
  out.sigma_vel_z = p.accel_max_z * t;
  out.sigma_pos_h = 0.5 * p.accel_max * t * t;
  out.sigma_pos_z = 0.5 * p.accel_max_z * t * t;
  out.sigma_rp = p.rate_rp_max * t;
  out.sigma_yaw = p.rate_yaw_max * t;
  out.vel = out.sigma_vel_h * out.sigma_vel_h * horizontal + out.sigma_vel_z * out.sigma_vel_z * vertical;
  out.pos = out.sigma_pos_h * out.sigma_pos_h * horizontal + out.sigma_pos_z * out.sigma_pos_z * vertical;
  out.rot = out.sigma_rp * out.sigma_rp * horizontal_body + out.sigma_yaw * out.sigma_yaw * vertical_body;
  out.gap_s = t;
  out.valid = true;
  return out;
}

/// Decouple and inflate P after a gap: every cross term between the pos, vel and rot blocks and the rest of the state
/// is zeroed (each block and the rest keep their own), then the inflation is added. Returns false and leaves P alone
/// for an invalid inflation or a layout that does not fit P; a zero gap leaves P alone and returns true.
template<typename Derived>
inline bool applyImuGapInflation(Eigen::MatrixBase<Derived> & P,
                                 const ImuGapInflation & inflation,
                                 int pos_index = kStatePosIndex,
                                 int rot_index = kStateRotIndex,
                                 int vel_index = kStateVelIndex)
{
  const int n = static_cast<int>(P.rows());
  const auto fits = [n](int index) {
    return index >= 0 && index + 3 <= n;
  };
  const auto disjoint = [](int a, int b) {
    return a + 3 <= b || b + 3 <= a;
  };
  if (!inflation.valid || P.rows() != P.cols() || !fits(pos_index) || !fits(rot_index) || !fits(vel_index) ||
      !disjoint(pos_index, rot_index) || !disjoint(pos_index, vel_index) || !disjoint(rot_index, vel_index)) {
    return false;
  }
  if (!(inflation.gap_s > 0.0)) {
    return true;
  }
  const auto group = [&](int i) {
    if (i >= pos_index && i < pos_index + 3) {
      return 0;
    }
    if (i >= vel_index && i < vel_index + 3) {
      return 1;
    }
    return (i >= rot_index && i < rot_index + 3) ? 2 : 3;
  };
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      if (group(r) != group(c)) {
        P(r, c) = 0.0;
      }
    }
  }
  P.template block<3, 3>(pos_index, pos_index) += inflation.pos;
  P.template block<3, 3>(vel_index, vel_index) += inflation.vel;
  P.template block<3, 3>(rot_index, rot_index) += inflation.rot;
  return true;
}

/// What one scan's propagation bridged: for the log line and the re-anchor decision.
struct ImuGapSummary
{
  int gaps = 0;
  double total_s = 0.0;
  double longest_s = 0.0;           ///< the longest stretch this scan bridged
  double longest_interval_s = 0.0;  ///< the longest IMU interval behind them (it may extend into other scans)
  ImuGapInflation longest;          ///< the inflation of the longest bridged stretch
};

/// Record one bridged stretch; interval_s is the IMU interval it belongs to (sample to sample, or from the last
/// sample to the end of the stretch), which is longer when the gap started in an earlier scan.
inline void recordImuGap(ImuGapSummary * summary, const ImuGapInflation & inflation, double interval_s = 0.0)
{
  if (summary == nullptr || !inflation.valid || !(inflation.gap_s > 0.0)) {
    return;
  }
  ++summary->gaps;
  summary->total_s += inflation.gap_s;
  if (inflation.gap_s > summary->longest_s) {
    summary->longest_s = inflation.gap_s;
    summary->longest = inflation;
  }
  if (std::isfinite(interval_s)) {
    summary->longest_interval_s = std::fmax(summary->longest_interval_s, std::fmax(interval_s, inflation.gap_s));
  }
}

}  // namespace fast_lio

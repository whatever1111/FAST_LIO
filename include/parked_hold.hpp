// parked_hold.hpp — the measurement a provably parked platform is entitled to.
//
// The blind-stretch static hold answers a scan that stopped constraining the
// state. This answers the opposite case: a scan that constrains it perfectly
// well and is still wrong. A person walking through a confined field of view
// registers as cleanly as a wall, so the map can pull a parked platform along
// with them at full correspondence count and a tight residual — nothing inside
// the estimator can object, because the map and the mover agree.
//
// The answer is a measurement, not an override. Pose is pulled toward the
// anchor and velocity toward zero with a covariance that says what the prior is
// worth, so a platform that is pushed or slides can still out-vote it. Pinning
// the state instead would make that impossible.
//
// Pure C++ + Eigen. No ROS, no ikd-tree, no estimator types.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

namespace fast_lio
{

struct ParkedHoldParams
{
  bool enabled = false;         ///< off by default; profiles opt in
  int engage_scans = 3;         ///< consecutive parked scans before the prior applies
  double pos_noise = 0.05;      ///< m, 1-sigma
  double rot_noise = 0.02;      ///< rad, 1-sigma
  double vel_noise = 0.05;      ///< m/s, 1-sigma
  double deadband_sigma = 3.0;  ///< apply only past this many sigma of deviation; 0 = every scan
};

struct ParkedHoldState
{
  int parked_scans = 0;
  bool active = false;
};

struct ParkedHoldVerdict
{
  bool engaged = false;
  bool just_engaged = false;
  bool just_released = false;
};

/// Slow to engage, instant to release: a single scan of evidence that the
/// platform moved drops the prior, while engaging waits for `engage_scans` so a
/// lone spurious verdict mid-motion cannot pin a moving robot.
inline ParkedHoldVerdict updateParkedHold(ParkedHoldState * state, bool body_static, const ParkedHoldParams & params)
{
  ParkedHoldVerdict out;
  state->parked_scans = body_static ? state->parked_scans + 1 : 0;
  const bool engaged = params.enabled && body_static && state->parked_scans >= params.engage_scans;
  out.engaged = engaged;
  out.just_engaged = engaged && !state->active;
  out.just_released = !engaged && state->active;
  state->active = engaged;
  return out;
}

/// The prior carries the same information every scan, so applying it on all of
/// them is not a repeated measurement — it is one fact counted thousands of
/// times, and the position covariance collapses under it (IMU process noise at
/// rest re-inflates almost nothing between scans). A filter that has convinced
/// itself the pose is certain to a micron is slow to accept the LiDAR again
/// when the platform moves off. So the hold is a fault response, not a standing
/// prior: inside the deadband the estimate is where it should be and nothing is
/// applied. Drift is then bounded by the deadband rather than removed, which is
/// the point — the failure being answered is metres wide.
inline bool parkedHoldShouldApply(const Eigen::Vector3d & anchor_pos,
                                  const Eigen::Matrix3d & anchor_rot,
                                  const Eigen::Vector3d & est_pos,
                                  const Eigen::Matrix3d & est_rot,
                                  const ParkedHoldParams & params)
{
  if (!(params.deadband_sigma > 0.0)) {
    return true;
  }
  if ((anchor_pos - est_pos).norm() > params.deadband_sigma * params.pos_noise) {
    return true;
  }
  const Eigen::AngleAxisd err(est_rot.transpose() * anchor_rot);
  return std::abs(err.angle()) > params.deadband_sigma * params.rot_noise;
}

/// Rows of the pose-and-zero-velocity prior, in the ESIKF's RIGHT-perturbation
/// error state (R <- R * Exp(dx)); column layout pos 0-2, rot 3-5, vel 12-14.
struct ParkedHoldMeasurement
{
  Eigen::Matrix<double, 9, 23> H = Eigen::Matrix<double, 9, 23>::Zero();
  Eigen::Matrix<double, 9, 1> residual = Eigen::Matrix<double, 9, 1>::Zero();
  Eigen::Matrix<double, 9, 1> R_diag = Eigen::Matrix<double, 9, 1>::Ones();
};

inline ParkedHoldMeasurement parkedHoldMeasurement(const Eigen::Vector3d & anchor_pos,
                                                   const Eigen::Matrix3d & anchor_rot,
                                                   const Eigen::Vector3d & est_pos,
                                                   const Eigen::Matrix3d & est_rot,
                                                   const Eigen::Vector3d & est_vel,
                                                   const ParkedHoldParams & params)
{
  ParkedHoldMeasurement m;
  m.H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  m.H.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  m.H.block<3, 3>(6, 12) = Eigen::Matrix3d::Identity();

  // Attitude residual is the right perturbation that carries the estimate onto
  // the anchor: est_rot * Exp(r) == anchor_rot.
  const Eigen::AngleAxisd err(est_rot.transpose() * anchor_rot);
  m.residual.segment<3>(0) = anchor_pos - est_pos;
  m.residual.segment<3>(3) = err.angle() * err.axis();
  m.residual.segment<3>(6) = -est_vel;

  m.R_diag.segment<3>(0).setConstant(params.pos_noise * params.pos_noise);
  m.R_diag.segment<3>(3).setConstant(params.rot_noise * params.rot_noise);
  m.R_diag.segment<3>(6).setConstant(params.vel_noise * params.vel_noise);
  return m;
}

}  // namespace fast_lio

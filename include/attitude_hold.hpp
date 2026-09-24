// attitude_hold.hpp — whether a scan's point-to-plane update may rotate roll/pitch.
//
// With no horizontal surface in view (pos_obs_z_weak ≈ 1) the residuals constrain
// roll/pitch only through the short levers of near-field vertical structure, and
// against a frozen map that lacks the space the scan is entering, the unmatched
// half of the scan is pulled onto the nearest door frame or wall. The iterated
// update then tilts the state to absorb residuals that carry no vertical
// information: 0.4–1.7° per scan on the m20 doorway windows against 0.07° per
// scan in healthy walking (docs/PGO_LOOP_TUNING.md §7.5). Over those seconds the
// gyro alone drifts 0.1–1°; it is the better estimate.
//
// The remedy keeps the update but removes roll/pitch from what it may decide:
// each point's rotation Jacobian row is projected onto the current vertical, so
// the residuals are modelled as depending on the rotation about the vertical
// (yaw) only. Translation, yaw, velocity and the rest are updated as before;
// roll/pitch ride on the propagation until healthy scans return, and their
// covariance is not shrunk by information the update never had.
//
// Pure C++ + Eigen. No ROS, no estimator types.

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace fast_lio
{

struct AttitudeHoldParams
{
  bool enabled = false;       ///< lidar_attitude_hold_en
  double far_frac_max = 0.0;  ///< also hold on unfrozen scans whose far-field share is below this (0 = never)
  double z_weak_min = 0.9;    ///< ...and whose previous scan saw no horizontal surface (pos_obs_z_weak above this)
};

/// Whether this scan's lidar update is restricted to translation and yaw.
/// `map_frozen` and `z_weak_prev` are the previous scan's verdicts (the guard and
/// the observability run after the update); `engulf_latched` is the sticky
/// engulfment state; `far_frac` is this scan's own far-field share.
inline bool holdAttitudeThisScan(const AttitudeHoldParams & p,
                                 bool map_frozen,
                                 bool engulf_latched,
                                 double far_frac,
                                 double z_weak_prev)
{
  if (!p.enabled) {
    return false;
  }
  if (map_frozen || engulf_latched) {
    return true;
  }
  if (p.far_frac_max > 0.0 && std::isfinite(far_frac) && std::isfinite(z_weak_prev) && far_frac < p.far_frac_max &&
      z_weak_prev > p.z_weak_min) {
    return true;
  }
  return false;
}

/// A point's rotation Jacobian row projected onto the vertical axis `up` (unit,
/// body frame): only the component that a rotation about the vertical would
/// produce survives, so the residual cannot ask for roll or pitch.
inline Eigen::Vector3d projectRotationRowToYaw(const Eigen::Vector3d & row, const Eigen::Vector3d & up)
{
  return row.dot(up) * up;
}

}  // namespace fast_lio

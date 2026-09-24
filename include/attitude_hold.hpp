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
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace fast_lio
{

struct AttitudeHoldParams
{
  bool enabled = false;       ///< lidar_attitude_hold_en
  int min_effct = 100;        ///< release when the previous scan matched fewer points: a starved scan needs all six
                              ///< degrees of freedom to re-lock, and the hold has no trustworthy attitude to hold
  double max_hold_s = 3.0;    ///< release after this long on the gyro alone
  double max_scan_gap_s = 0.3;  ///< a scan hole longer than this means the propagation itself is suspect
  double far_frac_max = 0.0;  ///< also hold on unlatched scans whose far-field share is below this (0 = never)
  double z_weak_min = 0.9;    ///< ...and whose previous scan saw no horizontal surface (pos_obs_z_weak above this)
};

/// What the caller knows about this scan when it decides.
struct AttitudeHoldInputs
{
  bool engulf_latched = false;  ///< the guard's sticky far-field-collapse state (this scan's, computed last scan)
  int effct_prev = 0;           ///< effective correspondences of the previous scan
  double hold_age_s = 0.0;      ///< how long the hold has already been on (0 if it is off)
  double scan_gap_s = 0.1;      ///< time since the previous processed scan
  double far_frac = 1.0;        ///< this scan's far-field share
  double z_weak_prev = 0.0;     ///< previous scan's pos_obs_z_weak
};

/// Whether this scan's lidar update is restricted to translation and yaw.
///
/// The hold is for one situation: a still-locked front end whose scan has just
/// collapsed into the near field (engulfment). It must not engage where the
/// attitude it would hold is not trustworthy or cannot be propagated — the
/// re-anchor verification at start-up (m20 0825: held from scan 1 with a
/// walking-start init, the tilt grew to 143° and the guard never re-locked),
/// scan starvation (a lost front end needs six degrees of freedom to find the
/// map again), scan holes, or a hold that has already run for seconds.
inline bool holdAttitudeThisScan(const AttitudeHoldParams & p, const AttitudeHoldInputs & in)
{
  if (!p.enabled) {
    return false;
  }
  if (!std::isfinite(in.hold_age_s) || !std::isfinite(in.scan_gap_s)) {
    return false;
  }
  if (in.effct_prev < p.min_effct || in.hold_age_s > p.max_hold_s || in.scan_gap_s > p.max_scan_gap_s) {
    return false;
  }
  if (in.engulf_latched) {
    return true;
  }
  if (p.far_frac_max > 0.0 && std::isfinite(in.far_frac) && std::isfinite(in.z_weak_prev) &&
      in.far_frac < p.far_frac_max && in.z_weak_prev > p.z_weak_min) {
    return true;
  }
  return false;
}

/// How much this scan's correspondences say about roll and pitch: the smaller of
/// the two eigenvalues of the rotation information Σ AAᵀ restricted to the plane
/// orthogonal to the vertical `up` (unit, body frame). Rows are the point-to-plane
/// rotation Jacobians A_i = p_i × Rᵀn_i; a floor at range r contributes ~r² per
/// point to this subspace, a near-field wall only its points' vertical offsets
/// squared, so a doorway transition sits two orders of magnitude below a healthy
/// scan. Units: m² per point (before the point covariance). Returns 0 for a
/// degenerate input.
inline double rollPitchInformation(const Eigen::Matrix3d & rotation_information, const Eigen::Vector3d & up)
{
  if (!rotation_information.allFinite() || !up.allFinite() || up.norm() < 1e-9) {
    return 0.0;
  }
  const Eigen::Vector3d u = up.normalized();
  const Eigen::Matrix3d proj = Eigen::Matrix3d::Identity() - u * u.transpose();
  const Eigen::Matrix3d m = proj * rotation_information * proj;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(m);
  if (es.info() != Eigen::Success) {
    return 0.0;
  }
  // ascending: the first is the (numerically zero) direction along up, the second the weakest roll/pitch axis
  return std::max(es.eigenvalues()(1), 0.0);
}

/// A point's rotation Jacobian row projected onto the vertical axis `up` (unit,
/// body frame): only the component that a rotation about the vertical would
/// produce survives, so the residual cannot ask for roll or pitch.
inline Eigen::Vector3d projectRotationRowToYaw(const Eigen::Vector3d & row, const Eigen::Vector3d & up)
{
  return row.dot(up) * up;
}

}  // namespace fast_lio

// correspondence_gate.hpp — whether one point-to-plane residual is consistent with a correct
// correspondence, judged against what the filter itself expects of it.
//
// FAST-LIO admits a correspondence when |d| < sqrt(r)/9 (the s > 0.9 test): 0.11 m at 1 m, 0.26 m at
// 5.4 m. On the m20 doorways the roll/pitch swing is owned by correspondences with residuals of
// 10–25 cm at 4–7 m, 2–3 m above the sensor, whose nearest map point is 0.2–0.4 m away: points on a
// surface the frozen map does not hold, matched to the plane through the boundary of what it does
// hold. Residuals above 10 cm are 14 % of the effective points and own 74 % of the harmful roll/pitch
// increment (docs/PGO_LOOP_TUNING.md §7.7); the healthy per-point noise is 3 cm.
//
// The principled gate is the innovation test used by MSCKF-style filters (OpenVINS) and, with the
// plane covariance, by VoxelMap: a residual is admitted when it lies within k sigma of its own
// predicted variance S = H P Hᵀ + R, H = [n_world, A_body] over the [pos, rot] block of the prior.
// It is a 3-sigma test of the measurement noise when the pose is certain and opens by r·sigma_theta +
// sigma_pos when it is not (after a scan hole, at start-up), so the convergence basin is kept without
// a range heuristic. The first-neighbour bound is the separate, weaker statement that a point on a
// mapped surface has a map point within the grid spacing.
//
// Pure C++ + Eigen. No ROS, no estimator types.

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace fast_lio
{

struct CorrespondenceGateParams
{
  double innovation_k = 0.0;  ///< corr_innov_gate_k: admit |d| <= k*sqrt(H P H^T + R); <= 0 keeps the legacy test only
  double nn0_max = 0.0;       ///< corr_nn0_max (m): admit only when the nearest map point is closer than this; <= 0 off
};

/// Predicted variance of one point-to-plane residual (m²): H P Hᵀ + R with H = [n_world; A_body],
/// P6 the prior covariance over [pos(3), rot(3)] and point_cov the per-point measurement variance.
/// Returns a non-finite value for non-finite inputs so the caller's test rejects.
inline double residualVariance(const Eigen::Vector3d & n_world,
                               const Eigen::Vector3d & A_body,
                               const Eigen::Matrix<double, 6, 6> & P6,
                               double point_cov)
{
  Eigen::Matrix<double, 6, 1> H;
  H << n_world, A_body;
  return H.dot(P6 * H) + point_cov;
}

/// Whether a correspondence with this residual, predicted variance and nearest-map distance is admitted.
/// Non-finite inputs reject; a disabled test (parameter <= 0) never rejects on its own.
inline bool correspondenceAdmitted(const CorrespondenceGateParams & p, double residual, double variance, double nn0)
{
  if (!std::isfinite(residual)) {
    return false;
  }
  if (p.innovation_k > 0.0) {
    if (!std::isfinite(variance) || variance < 0.0) {
      return false;
    }
    if (std::abs(residual) > p.innovation_k * std::sqrt(variance)) {
      return false;
    }
  }
  if (p.nn0_max > 0.0) {
    if (!std::isfinite(nn0) || nn0 > p.nn0_max) {
      return false;
    }
  }
  return true;
}

}  // namespace fast_lio

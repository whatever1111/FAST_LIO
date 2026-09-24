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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace fast_lio
{

struct CorrespondenceGateParams
{
  double innovation_k = 0.0;  ///< corr_innov_gate_k: admit |d| <= k*sigma_i; <= 0 keeps the legacy test only
  double nn0_max = 0.0;       ///< corr_nn0_max (m): admit only when the nearest map point is closer than this; <= 0 off
  /// corr_gate_mad_scale: sigma_i is never below this multiple of the scan's own robust residual scale
  /// (1.4826 MAD of the signed residuals at this iteration). A pose the filter is overconfident about
  /// raises every residual; the scan's scale follows, the model's sqrt(H P H^T + R) does not, and a gate
  /// on the model alone then rejects the very matches that carry the correction (m20 0825 165 s: 365 ->
  /// 139 matches, the tilt grew to 4.4 deg). 0 = no floor.
  double mad_scale = 0.0;
  /// corr_gate_rot_sigma_floor_deg: the prior attitude covariance used in H P H^T is at least this on
  /// each axis, an allowance of r*sigma for far points where the filter's 0.1 deg is optimistic. 0 = none.
  double rot_sigma_floor_deg = 0.0;
};

/// Robust residual scale of one iteration: 1.4826 x the median absolute deviation of the signed
/// residuals from their median (the standard deviation of a Gaussian core with up to half the
/// samples arbitrary). Returns 0 for fewer than three residuals or non-finite input; the caller's
/// gate then relies on the model alone. `values` is consumed (reordered).
inline double robustResidualScale(std::vector<double> & values)
{
  const auto finite_end = std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); });
  values.erase(finite_end, values.end());
  if (values.size() < 3) {
    return 0.0;
  }
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  const double median = *mid;
  for (double & v : values) {
    v = std::abs(v - median);
  }
  std::nth_element(values.begin(), mid, values.end());
  return 1.4826 * *mid;
}

/// The prior [pos, rot] block with its attitude variances floored at sigma_floor_rad² (off-diagonals kept).
inline Eigen::Matrix<double, 6, 6> flooredPriorBlock(const Eigen::Matrix<double, 6, 6> & P6, double sigma_floor_rad)
{
  Eigen::Matrix<double, 6, 6> out = P6;
  if (std::isfinite(sigma_floor_rad) && sigma_floor_rad > 0.0) {
    for (int k = 3; k < 6; ++k) {
      out(k, k) = std::max(out(k, k), sigma_floor_rad * sigma_floor_rad);
    }
  }
  return out;
}

/// The sigma one residual is judged against: the model's sqrt(variance), never below the scan floor.
inline double gateSigma(double variance, double scan_sigma_floor)
{
  if (!std::isfinite(variance) || variance < 0.0) {
    return variance;  // propagate the bad value; correspondenceAdmitted rejects it
  }
  return std::max(std::sqrt(variance), std::isfinite(scan_sigma_floor) ? std::max(scan_sigma_floor, 0.0) : 0.0);
}

/// Geman-McClure and Cauchy weights on a residual with kernel scale c (IRLS weights on the squared
/// residual: GM (c²/(c²+r²))², Cauchy 1/(1+(r/c)²); the caller scales the measurement row and residual
/// by sqrt(w)). 1 for c <= 0 or bad input.
inline double gemanMcClureWeight(double residual, double c)
{
  if (!(c > 0.0) || !std::isfinite(residual) || !std::isfinite(c)) {
    return 1.0;
  }
  const double c2 = c * c;
  const double w = c2 / (c2 + residual * residual);
  return w * w;
}

inline double cauchyWeight(double residual, double c)
{
  if (!(c > 0.0) || !std::isfinite(residual) || !std::isfinite(c)) {
    return 1.0;
  }
  const double r = residual / c;
  return 1.0 / (1.0 + r * r);
}

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

/// Whether a correspondence with this residual, predicted variance and nearest-map distance is admitted;
/// `scan_sigma_floor` is the scan's robust scale times corr_gate_mad_scale (0 when the floor is off).
/// Non-finite inputs reject; a disabled test (parameter <= 0) never rejects on its own.
inline bool correspondenceAdmitted(const CorrespondenceGateParams & p,
                                   double residual,
                                   double variance,
                                   double nn0,
                                   double scan_sigma_floor = 0.0)
{
  if (!std::isfinite(residual)) {
    return false;
  }
  if (p.innovation_k > 0.0) {
    const double sigma = gateSigma(variance, scan_sigma_floor);
    if (!std::isfinite(sigma) || sigma < 0.0) {
      return false;
    }
    if (std::abs(residual) > p.innovation_k * sigma) {
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

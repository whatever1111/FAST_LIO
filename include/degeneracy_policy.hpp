// degeneracy_policy.hpp — directional down-weighting of the LiDAR position update.
//
// A degenerate scan is rarely degenerate in every direction. A doorway, a corridor
// or a wall-hugging pass leaves the lateral and yaw directions well constrained and
// blinds only the along-corridor and vertical ones, yet the existing defences are
// scalar: effct_feat_num / far_frac collapse into one number, and the response
// (freeze the map, pin the pose) throws away the directions the scan still sees.
//
// The information the scan carries about position is M = sum(n n^T) over the matched
// plane normals — exactly the top-left 3x3 block of the iEKF's H^T H, since the
// position columns of the measurement Jacobian ARE the normals. Its eigenvectors are
// the directions the scan constrains, its eigenvalues how strongly.
//
// ── How the weight is applied ────────────────────────────────────────────────────
// This header hands back W = V diag(w) V^T and a PER-POINT scalar
//
//     s_i = sqrt(n_i^T W n_i)  in [0, 1],
//
// which the caller multiplies into the WHOLE Jacobian row AND the residual of that
// point. That is exactly R_i <- R / s_i^2: an anisotropic measurement covariance, mean
// and covariance consistent.
//
// Scaling only the position COLUMNS of H (H_pos <- H_pos * sqrt(W)) is not the same
// thing and must not be done. Leaving the residual alone turns n_i^T into (S n_i)^T,
// which does not down-weight the measurement — it claims the plane faces a different
// way and is shallower than it is. Explaining the same point-to-plane distance against
// a shallower plane needs a LARGER position step, so in a direction the scan dominates
// the prior the correction grows by up to 1/s instead of shrinking. Only s in {0, 1}
// survives that mistake, because a projector is also an exact constrained solve; every
// fractional weight is wrong in sign.
//
// ── Which directions are held back ──────────────────────────────────────────────
// Two tests, and a direction is only held back when it fails BOTH:
//
//   relative — the eigenvalue normalised by the effective point count, i.e. the mean
//     cos^2 between the matched normals and that direction. These SUM TO 1
//     (trace(M) = N for unit normals), so each is a share of the scan's information
//     and the weakest of three is capped at 1/3. A restricted field of view keeps that
//     share structurally low — a forward wedge sits near 0.1 even in a rich open scene
//     — so the share alone cannot tell a blind direction from a narrow sensor.
//
//   absolute — the raw eigenvalue: how many normals' worth of information the
//     direction actually carries. Compare it against R / sigma_pos^2, the point at
//     which the propagated prior stops being outweighed by the scan. This is the only
//     one of the two that can see a starved scan, because normalising by the point
//     count is precisely what erases "there were only 130 points".
//
// Combining them with max() means: strong in absolute terms => trust it however small
// its share; a healthy share => trust it however starved the scan (a uniformly starved
// scan is a job for the scalar guards, not for a subspace).
//
// Shrinking the information is not the same as adding any: the state then follows IMU
// propagation in the released direction, so this belongs WITH a prior that supplies it
// (gravity/planar for Z, ZUPT when parked, wheel speed along-track), not instead of one.
//
// ── Which information matrix ────────────────────────────────────────────────────
// A joint 6x6 spectrum is not an option: the rotation columns of H carry a lever arm
// and are in metres while the position columns are dimensionless, so its eigenvalues
// would be ordered by scene scale rather than by degeneracy. But H_tt = sum(n n^T)
// alone is not the position information either — it is the information the scan would
// carry IF the rotation were known. What is left once rotation has been solved for is
// the Schur complement
//
//     M_eff = H_tt - H_tr (H_rr + ridge)^-1 H_rt,
//
// which is dimensionless like H_tt (metre * metre^-2 * metre) and therefore directly
// comparable with it, and never larger. The difference is the information rotation
// steals: points on one plane seen at a constant lever arm make translation along that
// normal indistinguishable from a rotation about the perpendicular axis, and H_tt is
// blind to it — it counts those normals at full strength. The ridge is relative to
// trace(H_rr) because H_rr scales with the scene's lever arms.
//
// The per-point weight still scales the rotation columns of the row it belongs to —
// that is what makes it a weight on the MEASUREMENT rather than a change to the model.
//
// Pure C++ and Eigen. No ROS, no filter state.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace fast_lio
{

struct DegeneracyParams
{
  bool enable = false;  ///< off = unit weights, the policy is inert

  /// Relative test, on the eigenvalue share (sums to 1 across the three directions).
  double lambda_bad = 0.02;   ///< share at/below which a direction is unobservable (w=0)
  double lambda_good = 0.10;  ///< share at/above which it is fully trusted (w=1)

  /// Absolute test, on the raw eigenvalue of M (units: effective normals). Scales with
  /// the downsample density, so it is a platform value, not a universal one. Set both
  /// to 0 (good <= bad) to disable it and fall back to the share alone — which is
  /// field-of-view biased and will hold back healthy directions on a narrow sensor.
  double lambda_abs_bad = 2.0;    ///< raw eigenvalue at/below which a direction is unobservable
  double lambda_abs_good = 20.0;  ///< raw eigenvalue at/above which it is fully trusted

  /// Collapse the ramp to a step at its midpoint, so every weight is 0 or 1. Costs the
  /// graded response and buys an exact constrained solve.
  bool hard_projection = false;

  /// Judge the Schur complement rather than the raw translation block, so a direction
  /// rotation can explain away is not counted as observed. Off = the H_tt spectrum,
  /// whose normalised eigenvalues sum to exactly 1; on, they sum to at most 1 and the
  /// shortfall IS the coupling, so both thresholds want recalibrating when it changes.
  bool schur_en = false;
  double schur_ridge = 1e-6;  ///< relative to trace(H_rr)/3; keeps a singular H_rr invertible

  double min_weight = 0.0;  ///< floor on the weight; >0 keeps a trickle of LiDAR in blind directions
};

struct DegeneracyResult
{
  /// W = V diag(w) V^T, the source of the per-point weight. Identity when inert.
  Eigen::Matrix3d weight_matrix = Eigen::Matrix3d::Identity();
  /// Eigen-directions of M (columns, world frame), ascending in eigenvalue.
  Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
  /// Eigenvalue shares (mean cos^2), ascending. Clamped to [0, 1].
  Eigen::Vector3d lambda_norm = Eigen::Vector3d::Ones();
  /// Raw eigenvalues of M, ascending. Zero when inert.
  Eigen::Vector3d lambda_abs = Eigen::Vector3d::Zero();
  /// Weight per eigen-direction, in the same order.
  Eigen::Vector3d weights = Eigen::Vector3d::Ones();
  /// True when at least one direction is being held back.
  bool engaged = false;
};

/// Linear ramp: 0 at/below bad, 1 at/above good. Degenerate bounds (good <= bad) mean
/// the test is not configured and return when_disabled; a non-finite input returns full
/// trust, matching the "misconfigured means full trust" convention of the other quality
/// factors.
inline double rampWeight(double x, double bad, double good, double when_disabled)
{
  if (!std::isfinite(x)) {
    return 1.0;
  }
  if (good <= bad) {
    return when_disabled;
  }
  return std::clamp((x - bad) / (good - bad), 0.0, 1.0);
}

/// Weight for one eigen-direction from its share and its raw eigenvalue. Held back only
/// where both tests say so; see the header comment for why max() and not min().
inline double degeneracyWeight(double lambda_norm, double lambda_abs, const DegeneracyParams & params)
{
  double w = std::max(rampWeight(lambda_norm, params.lambda_bad, params.lambda_good, 1.0),
                      rampWeight(lambda_abs, params.lambda_abs_bad, params.lambda_abs_good, 0.0));
  if (params.hard_projection) {
    w = (w >= 0.5) ? 1.0 : 0.0;
  }
  return std::clamp(std::max(w, std::clamp(params.min_weight, 0.0, 1.0)), 0.0, 1.0);
}

/// Spectrum of the scan's position information and the weights it implies.
///   M     : sum(n n^T) over the effective matched normals, world frame
///   effct : how many normals went into M (the normaliser for the relative test)
/// A non-finite or empty M, or a disabled policy, yields unit weights — the update then
/// runs exactly as it does today.
inline DegeneracyResult computePositionDegeneracy(const Eigen::Matrix3d & M, int effct, const DegeneracyParams & params)
{
  DegeneracyResult out;
  if (!params.enable || effct <= 0 || !M.allFinite()) {
    return out;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
  if (es.info() != Eigen::Success) {
    return out;
  }

  const double inv_n = 1.0 / static_cast<double>(effct);
  for (int i = 0; i < 3; ++i) {
    // M is PSD by construction; clamp the numerical noise that can push the
    // smallest eigenvalue a hair below zero.
    out.lambda_abs(i) = std::max(es.eigenvalues()(i), 0.0);
    out.lambda_norm(i) = std::clamp(out.lambda_abs(i) * inv_n, 0.0, 1.0);
    out.weights(i) = degeneracyWeight(out.lambda_norm(i), out.lambda_abs(i), params);
    if (out.weights(i) < 1.0) {
      out.engaged = true;
    }
  }
  out.basis = es.eigenvectors();
  if (!out.engaged) {
    return out;  // keep the matrix exactly identity rather than a rounded reconstruction
  }
  out.weight_matrix = out.basis * out.weights.asDiagonal() * out.basis.transpose();
  return out;
}

/// Position information with the rotation coupling removed, from the three blocks of
/// H^T H over the position (t) and rotation (r) columns of the measurement Jacobian:
///
///     M_eff = H_tt - H_tr (H_rr + (ridge * trace(H_rr) / 3) I)^-1 H_rt
///
/// A rotation block with no information at all (trace <= 0) leaves H_tt untouched: the
/// scan then says nothing about rotation, so rotation can explain away nothing.
inline Eigen::Matrix3d schurPositionInformation(const Eigen::Matrix3d & H_tt,
                                                const Eigen::Matrix3d & H_tr,
                                                const Eigen::Matrix3d & H_rr,
                                                double ridge = 1e-6)
{
  const double trace_rr = H_rr.trace();
  if (!H_tt.allFinite() || !H_tr.allFinite() || !H_rr.allFinite() || !(trace_rr > 0.0)) {
    return H_tt;
  }
  const double lambda = std::max(ridge, 0.0) * trace_rr / 3.0;
  const Eigen::Matrix3d rr = H_rr + lambda * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d coupled = H_tr * rr.ldlt().solve(H_tr.transpose());
  if (!coupled.allFinite()) {
    return H_tt;
  }
  // Symmetrise: the product is symmetric in exact arithmetic, not in floating point.
  const Eigen::Matrix3d m = H_tt - coupled;
  return 0.5 * (m + m.transpose());
}

/// As above, from the Jacobian blocks. Falls back to the translation block alone when
/// the policy is not asked for the Schur complement.
inline DegeneracyResult computePositionDegeneracy(const Eigen::Matrix3d & H_tt,
                                                  const Eigen::Matrix3d & H_tr,
                                                  const Eigen::Matrix3d & H_rr,
                                                  int effct,
                                                  const DegeneracyParams & params)
{
  const Eigen::Matrix3d m = params.schur_en ? schurPositionInformation(H_tt, H_tr, H_rr, params.schur_ridge) : H_tt;
  return computePositionDegeneracy(m, effct, params);
}

/// Per-point weight for a plane whose (unit) normal is `normal`: s = sqrt(n^T W n).
/// Multiply the point's WHOLE Jacobian row and its residual by this — see the header
/// comment for why both, and why scaling the position columns alone is not equivalent.
inline double pointWeight(const DegeneracyResult & result, const Eigen::Vector3d & normal)
{
  if (!result.engaged) {
    return 1.0;
  }
  const double q = normal.dot(result.weight_matrix * normal);
  if (!std::isfinite(q)) {
    return 1.0;
  }
  return std::sqrt(std::clamp(q, 0.0, 1.0));
}

}  // namespace fast_lio

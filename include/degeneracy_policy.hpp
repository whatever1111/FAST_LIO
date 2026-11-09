// degeneracy_policy.hpp — the position information a scan carries once rotation has been
// solved for.
//
// The information the scan carries about position is the top-left 3x3 block of the
// iEKF's H^T H: the position columns of the measurement Jacobian ARE the matched plane
// normals, so H_tt = sum(n n^T). Its eigenvectors are the directions the scan
// constrains, its eigenvalues how strongly.
//
// H_tt alone is not the position information, though: it is the information the scan
// would carry IF the rotation were known. What is left once rotation has been solved for
// is the Schur complement
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
// A joint 6x6 spectrum is not an option: the rotation columns of H carry a lever arm and
// are in metres while the position columns are dimensionless, so its eigenvalues would
// be ordered by scene scale rather than by degeneracy.
//
// normal_noise_floor.hpp judges M_eff's directions against the information the plane
// normals' own noise can fake; the [DEGEN-SPECTRUM] trace prints its spectrum.
//
// Pure C++ and Eigen. No ROS, no filter state.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace fast_lio
{

/// The Schur complement's regularisation (degeneracy_schur_ridge).
struct DegeneracyParams
{
  double schur_ridge = 1e-6;  ///< relative to trace(H_rr)/3; keeps a singular H_rr invertible
};

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

}  // namespace fast_lio

// rp_consider.hpp — the lidar update as a consider (Schmidt-Kalman) update for roll and pitch.
//
// On the m20 doorways the lidar update keeps rotating roll/pitch on scans that cannot observe them: the frozen-map
// boundary matches inside an engulfment stretch and the 2-5 m wall points above the sensor after the release, in
// scenes without a horizontal surface (docs/PGO_LOOP_TUNING.md §7.7-7.8). The gyro-propagated attitude agrees with
// the vendor INS within 0.6° through every window, so the honest estimator there is "roll/pitch follow the gyro,
// everything else updates". Projecting the rotation rows of H to yaw (attitude_hold.hpp, §7.6) did not do that:
// the position innovation still rotated the state through P(rot, pos). The consider update does: the rows of the
// Kalman gain that would move the horizontal rotation directions are zeroed, the other rows keep the gain computed
// with the full covariance (so the consider uncertainty still widens the innovation covariance), and P is updated
// in Joseph form with the projected gain — the considered block keeps its variance, the cross terms shrink as the
// Schmidt filter prescribes. Textbook: Schmidt 1966, Bar-Shalom, Li & Kirubarajan 2001 §10.4 (consider states).
//
// Pure C++ + Eigen. No ROS, no estimator types: the estimator applies `keep` to its gain rows and the covariance
// helper to its own matrices; the policy takes numbers the front end already has.

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace fast_lio
{

struct RollPitchConsiderParams
{
  bool enabled = false;
  bool while_engulfed = true;    ///< consider roll/pitch on every scan of an engulfment stretch
  double post_release_s = 0.0;   ///< s; keep considering this long after an engulfment release (0 = off)
  double min_rp_info = 0.0;      ///< consider when the previous scan's roll/pitch information is below this (0 = off)
  double z_weak_min = 1.1;       ///< consider when the previous scan's z_weak is at least this (> 1 = off)
  double max_consider_s = 0.0;   ///< s; a cap on one continuous stretch of considering (0 = no cap)
};

struct RollPitchConsiderInputs
{
  bool engulf_latched = false;
  double since_release_s = -1.0;  ///< s since the last engulfment release; < 0 = none yet
  double rp_info_prev = 0.0;      ///< the previous scan's roll/pitch information (attitude_hold.hpp)
  double z_weak_prev = 0.0;       ///< the previous scan's pos_obs_z_weak
  double consider_age_s = 0.0;    ///< s this stretch has been considering already
};

/// Whether this scan's lidar update considers (does not estimate) roll and pitch.
inline bool considerRollPitchThisScan(const RollPitchConsiderParams & p, const RollPitchConsiderInputs & in)
{
  if (!p.enabled) {
    return false;
  }
  if (p.max_consider_s > 0.0 && std::isfinite(in.consider_age_s) && in.consider_age_s > p.max_consider_s) {
    return false;
  }
  if (p.while_engulfed && in.engulf_latched) {
    return true;
  }
  if (p.post_release_s > 0.0 && std::isfinite(in.since_release_s) && in.since_release_s >= 0.0 &&
      in.since_release_s <= p.post_release_s) {
    return true;
  }
  if (p.min_rp_info > 0.0 && std::isfinite(in.rp_info_prev) && in.rp_info_prev < p.min_rp_info) {
    return true;
  }
  if (p.z_weak_min <= 1.0 && std::isfinite(in.z_weak_prev) && in.z_weak_prev >= p.z_weak_min) {
    return true;
  }
  return false;
}

struct RotationConsiderProjector
{
  bool valid = false;
  Eigen::Matrix3d keep = Eigen::Matrix3d::Identity();  ///< applied to the rotation rows of the gain
};

/// The projector that keeps only the yaw part of a body-frame rotation increment: u uᵀ with u the body-frame
/// vertical. Invalid (identity, i.e. no consider) for a non-finite or degenerate vertical.
inline RotationConsiderProjector yawOnlyProjector(const Eigen::Vector3d & up_body)
{
  RotationConsiderProjector out;
  const double n = up_body.norm();
  if (!up_body.allFinite() || !(n > 0.5) || !(n < 2.0)) {
    return out;
  }
  const Eigen::Vector3d u = up_body / n;
  out.keep = u * u.transpose();
  out.valid = true;
  return out;
}

/// The Joseph-form covariance of a consider update written with what the estimator has: L the (Jacobian-adjusted)
/// prior covariance, A = K H P the correction of the full gain, M the row projector of the gain (identity on the
/// estimated states, `keep` on the considered block). (I − M K H) P (I − M K H)ᵀ + M K R Kᵀ M = P − M A − Aᵀ M + M A M
/// because K R Kᵀ = A − K H P Hᵀ Kᵀ for the optimal gain. M = I gives the usual P − A.
template <typename DerivedL, typename DerivedA, typename DerivedM>
inline Eigen::Matrix<typename DerivedL::Scalar, DerivedL::RowsAtCompileTime, DerivedL::ColsAtCompileTime>
considerCovariance(const Eigen::MatrixBase<DerivedL> & L, const Eigen::MatrixBase<DerivedA> & A,
                   const Eigen::MatrixBase<DerivedM> & M)
{
  return L - M * A - A.transpose() * M + M * A * M;
}

}  // namespace fast_lio

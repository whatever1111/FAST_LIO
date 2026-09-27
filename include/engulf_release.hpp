// engulf_release.hpp — what the state and its covariance become when an engulfment stretch is released.
//
// The guard freezes the map while the scan is engulfed (far field collapsed) and releases it once the far
// field has been back for a few scans. Two things about that release moment were measured on the m20
// doorways (docs/PGO_LOOP_TUNING.md §7.8): the velocity was discarded to zero while the platform was
// walking out of the door at 0.4-0.9 m/s, and the covariance still claimed centimetre / 0.06° certainty
// although the pose had drifted 0.3-0.5 m and up to 2° against the frozen map. The next scans then
// explained the resulting along-track innovation partly through roll/pitch (2-5 m wall points above the
// sensor) and the map was rebuilt around the wrong pose.
//
// This header decides the released velocity (legs, kept estimate, or the legacy zero) with the 1-σ that
// assignment deserves, and the additive inflation of the position and roll/pitch blocks by the length of
// the blind stretch. Pure C++ + Eigen, no ROS, no estimator types: the caller passes the 23x23 IKFoM
// covariance (pos 0-2, rot 3-5, vel 12-14) and applies the velocity through change_x / resetVelocityBlock.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace fast_lio
{

enum class ReleaseVelocitySource
{
  kZero,  ///< legacy: discard the blind stretch's velocity
  kLeg,   ///< the newest leg-odometry sample, rotated into the world; falls back to kKeep when stale
  kKeep   ///< the estimate carried through the blind stretch; falls back to kZero when it is a runaway
};

struct ReleaseVelocityParams
{
  ReleaseVelocitySource source = ReleaseVelocitySource::kZero;
  double leg_max_age_s = 0.5;  ///< s; an older leg sample is not evidence of the current speed
  double leg_sigma = 0.3;      ///< m/s; 1-σ of a leg-derived velocity (scale and slip on this platform)
  double keep_sigma = 0.5;     ///< m/s; 1-σ of the estimate carried through the blind stretch
  double zero_sigma = 0.0;     ///< m/s; 1-σ written for a discarded velocity (0 = leave P alone, legacy)
  double max_speed = 3.0;      ///< m/s; an estimate above this is a runaway and is discarded
};

struct ReleaseVelocityDecision
{
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  ///< world-frame velocity to write into the state
  double sigma = 0.0;                                   ///< 1-σ for the velocity block (0 = leave P alone)
  ReleaseVelocitySource used = ReleaseVelocitySource::kZero;
};

/// The velocity to carry out of an engulfment stretch. `leg_age_s` < 0 means there is no leg sample at all.
inline ReleaseVelocityDecision releaseVelocity(const Eigen::Vector3d & vel_estimate,
                                               const Eigen::Matrix3d & R_world_body,
                                               const Eigen::Vector3d & leg_velocity_body,
                                               double leg_age_s,
                                               const ReleaseVelocityParams & p)
{
  ReleaseVelocityDecision d;
  const bool estimate_ok = vel_estimate.allFinite() && std::isfinite(p.max_speed) && vel_estimate.norm() <= p.max_speed;
  const bool leg_ok = leg_velocity_body.allFinite() && R_world_body.allFinite() && std::isfinite(leg_age_s) &&
                      leg_age_s >= 0.0 && leg_age_s <= p.leg_max_age_s;
  if (p.source == ReleaseVelocitySource::kLeg && leg_ok) {
    d.velocity = R_world_body * leg_velocity_body;
    d.sigma = p.leg_sigma;
    d.used = ReleaseVelocitySource::kLeg;
    return d;
  }
  if (p.source != ReleaseVelocitySource::kZero && estimate_ok) {
    d.velocity = vel_estimate;
    d.sigma = p.keep_sigma;
    d.used = ReleaseVelocitySource::kKeep;
    return d;
  }
  d.velocity.setZero();
  d.sigma = p.zero_sigma;
  d.used = ReleaseVelocitySource::kZero;
  return d;
}

struct ReleaseInflationParams
{
  bool enabled = false;
  double pos_sigma_min = 0.1;      ///< m; position 1-σ added at release, floor
  double pos_sigma_rate = 0.2;     ///< m per second of blind stretch
  double pos_sigma_max = 1.0;      ///< m; ceiling
  double rp_sigma_min_deg = 0.5;   ///< deg; roll/pitch 1-σ added at release, floor
  double rp_sigma_rate_deg = 0.3;  ///< deg per second of blind stretch
  double rp_sigma_max_deg = 3.0;   ///< deg; ceiling
  double yaw_sigma_deg = 0.0;      ///< deg; 0 leaves the yaw variance alone
};

/// Clamp helper: a non-finite or negative blind duration counts as zero (the floor still applies).
inline double releaseSigma(double blind_s, double floor_value, double rate, double ceiling)
{
  const double t = (std::isfinite(blind_s) && blind_s > 0.0) ? blind_s : 0.0;
  const double lo = std::fmax(0.0, floor_value);
  const double hi = std::fmax(lo, ceiling);
  return std::fmin(hi, std::fmax(lo, lo + rate * t));
}

inline double releasePositionSigma(double blind_s, const ReleaseInflationParams & p)
{
  return releaseSigma(blind_s, p.pos_sigma_min, p.pos_sigma_rate, p.pos_sigma_max);
}

inline double releaseRollPitchSigmaRad(double blind_s, const ReleaseInflationParams & p)
{
  constexpr double kDegToRad = M_PI / 180.0;
  return releaseSigma(blind_s, p.rp_sigma_min_deg, p.rp_sigma_rate_deg, p.rp_sigma_max_deg) * kDegToRad;
}

/// Add σ_pos² I to the position block and σ_rp² (I − u uᵀ) + σ_yaw² u uᵀ to the rotation block, u being the
/// body-frame vertical (the error state of the rotation is a body-frame small angle). Cross terms are left
/// as they are: an added independent uncertainty keeps P symmetric positive semi-definite and only lowers the
/// weight of the correlations the blind stretch accumulated. Returns false (P untouched) when disabled, for a
/// non-finite or non-unit vertical, or for blocks outside P.
template <typename Derived>
inline bool inflateReleaseCovariance(Eigen::MatrixBase<Derived> & P,
                                     double blind_s,
                                     const Eigen::Vector3d & up_body,
                                     const ReleaseInflationParams & p,
                                     int pos_index = 0,
                                     int rot_index = 3)
{
  if (!p.enabled || !up_body.allFinite() || pos_index < 0 || rot_index < 0 || pos_index + 3 > P.rows() ||
      rot_index + 3 > P.rows() || pos_index + 3 > P.cols() || rot_index + 3 > P.cols()) {
    return false;
  }
  const double un = up_body.norm();
  if (!(un > 0.5) || !(un < 2.0)) {
    return false;
  }
  const Eigen::Vector3d u = up_body / un;
  const double sp = releasePositionSigma(blind_s, p);
  const double sr = releaseRollPitchSigmaRad(blind_s, p);
  const double sy = (std::isfinite(p.yaw_sigma_deg) && p.yaw_sigma_deg > 0.0) ? p.yaw_sigma_deg * M_PI / 180.0 : 0.0;
  const Eigen::Matrix3d uu = u * u.transpose();
  const Eigen::Matrix3d rot_add = sr * sr * (Eigen::Matrix3d::Identity() - uu) + sy * sy * uu;
  for (int k = 0; k < 3; ++k) {
    P(pos_index + k, pos_index + k) += sp * sp;
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      P(rot_index + r, rot_index + c) += rot_add(r, c);
    }
  }
  return true;
}

}  // namespace fast_lio

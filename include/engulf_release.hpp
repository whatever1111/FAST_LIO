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
// assignment deserves. Pure C++ + Eigen, no ROS, no estimator types: the caller applies the velocity through
// change_x / resetVelocityBlock.

#pragma once

#include <Eigen/Core>

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

}  // namespace fast_lio

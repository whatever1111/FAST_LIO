// gravity_align_kinematics.hpp — the kinematic term the levelling prior has to
// subtract before a window-mean specific force can be read as a gravity direction.
//
// A rigid body's accelerometer measures f = dv_b/dt + ω×v_b − Rᵀg + b_a. The
// levelling prior in laserMapping.cpp subtracted only the centripetal ω×v_b and
// took the 1 s window mean of the rest as the body-frame vertical. The window
// mean of dv_b/dt is not zero: it is (v_b(end) − v_b(start))/span, and on the m20
// a doorway stop or start moves that by 4–7° of lean inside a window the prior
// then trusts at 0.46° (docs/PGO_LOOP_TUNING.md §7.3). Only the body velocity at
// the two window edges enters, so the filter's own per-scan velocity history is
// enough to compute it — as long as nothing overwrote that velocity inside the
// window. A reset, pin or clamp has no derivative; the term is undefined there,
// and the honest answer is to withhold the prior for that window rather than
// level onto a number that is neither gravity nor acceleration.
//
// The same edges carry the filter's velocity covariance, which bounds the error
// of the compensation: σ = scale · sqrt(var_start + var_end) / (g · span), in the
// unit-vector units the prior's residual uses. Where the velocity is
// unobserved the sigma grows and the prior fades on its own — no threshold.
//
// Pure C++ + Eigen. No ROS, no estimator types.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace fast_lio
{

/// One scan's entry in the levelling window: the body-frame velocity at both ends
/// of the span its IMU samples cover.
struct LevellingWindowScan
{
  double t_start = 0.0;  ///< s; lidar_end_time of the previous scan (this scan's IMU samples begin here)
  double t_end = 0.0;    ///< s; lidar_end_time of this scan
  Eigen::Vector3d v_start = Eigen::Vector3d::Zero();  ///< body-frame velocity at t_start (m/s)
  Eigen::Vector3d v_end = Eigen::Vector3d::Zero();    ///< body-frame velocity at t_end (m/s)
  double var_start = 0.0;  ///< trace of the velocity covariance block at t_start ((m/s)²)
  double var_end = 0.0;    ///< trace of the velocity covariance block at t_end ((m/s)²)
  bool velocity_overwritten = false;  ///< the state velocity was assigned (reset/pin/clamp) inside (t_start, t_end]
};

struct LinearAccelerationTerm
{
  bool valid = false;
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();  ///< window-mean body-frame linear acceleration (m/s²)
  double span = 0.0;                                ///< s, from the oldest scan's start to the newest scan's end
  double sigma = 0.0;                               ///< 1-σ lean of the compensation on the unit vector (rad)
};

/// The window-mean linear acceleration between the oldest scan's start edge and
/// the newest scan's end edge. Invalid when the span is not positive, any input is
/// non-finite, or a velocity overwrite lies anywhere inside the window
/// (`overwritten_inside` is the OR of every scan's flag, the caller's job because
/// the middle scans are not passed in).
inline LinearAccelerationTerm linearAccelerationTerm(const LevellingWindowScan & oldest,
                                                     const LevellingWindowScan & newest,
                                                     bool overwritten_inside,
                                                     double gravity,
                                                     double sigma_scale)
{
  LinearAccelerationTerm term;
  const double span = newest.t_end - oldest.t_start;
  const bool finite = std::isfinite(span) && std::isfinite(gravity) && std::isfinite(sigma_scale) &&
                      oldest.v_start.allFinite() && newest.v_end.allFinite() &&
                      std::isfinite(oldest.var_start) && std::isfinite(newest.var_end);
  if (!finite || overwritten_inside || oldest.velocity_overwritten || newest.velocity_overwritten || span <= 0.0 ||
      gravity <= 0.0 || sigma_scale < 0.0) {
    return term;
  }
  term.valid = true;
  term.span = span;
  term.accel = (newest.v_end - oldest.v_start) / span;
  const double var = std::max(oldest.var_start, 0.0) + std::max(newest.var_end, 0.0);
  term.sigma = sigma_scale * std::sqrt(var) / (gravity * span);
  return term;
}

/// The lean, in radians, that a horizontal acceleration error of this size puts on
/// the measured vertical — for logs and tests, not for the update.
inline double leanOfAcceleration(const Eigen::Vector3d & accel, double gravity)
{
  if (!(gravity > 0.0) || !accel.allFinite()) {
    return 0.0;
  }
  return std::atan2(std::hypot(accel.x(), accel.y()), gravity);
}

}  // namespace fast_lio

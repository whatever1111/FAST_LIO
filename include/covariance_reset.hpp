// covariance_reset.hpp — keep the error-state covariance consistent with a state that was assigned
// outside an update.
//
// The guard assigns the velocity in several places (discarded at an engulfment release, zeroed by a
// ZUPT, bounded across a scan hole) through change_x(), which leaves P untouched. P then still says
// that the velocity is known to a few cm/s and that its error is correlated with the roll/pitch error
// through the gravity leak accumulated over the blind stretch. The next position innovations are
// explained through exactly those cross terms: on the m20 doorways the first scans after a release
// rotate roll by ~1° per 5 cm of innovation while the rotation rows carry no roll/pitch information
// at all (docs/PGO_LOOP_TUNING.md §7.7). An assigned value has the uncertainty of the assignment and
// no relation to any other state error, and that is what the block below writes.
//
// Pure C++ + Eigen. No ROS, no estimator types: the caller passes the 23x23 IKFoM covariance and the
// index of the first velocity row (12 in the FAST-LIO state layout).

#pragma once

#include <Eigen/Core>

#include <cmath>

namespace fast_lio
{

/// FAST-LIO state layout: pos 0-2, rot 3-5, ext_R 6-8, ext_T 9-11, vel 12-14, bg 15-17, ba 18-20, grav 21-22.
constexpr int kStateVelIndex = 12;

/// Zero every cross term of the velocity block and set its variance to sigma_v² on each axis.
/// Returns false (and leaves P alone) for a non-finite or negative sigma or an index outside P.
template <typename Derived>
inline bool resetVelocityBlock(Eigen::MatrixBase<Derived> & P, double sigma_v, int vel_index = kStateVelIndex)
{
  if (!std::isfinite(sigma_v) || sigma_v < 0.0 || vel_index < 0 || vel_index + 3 > P.rows() ||
      vel_index + 3 > P.cols()) {
    return false;
  }
  P.middleRows(vel_index, 3).setZero();
  P.middleCols(vel_index, 3).setZero();
  for (int k = 0; k < 3; ++k) {
    P(vel_index + k, vel_index + k) = sigma_v * sigma_v;
  }
  return true;
}

/// The 1-σ speed a discarded velocity leaves behind: the platform can be anywhere between parked
/// and its walking speed, so the assignment "velocity = 0" is uncertain by that speed, never less
/// than the floor (a body the IMU calls static still creeps).
inline double discardedVelocitySigma(double platform_max_speed, double floor_sigma)
{
  if (!std::isfinite(platform_max_speed) || !std::isfinite(floor_sigma)) {
    return floor_sigma;
  }
  return std::fmax(platform_max_speed, floor_sigma);
}

}  // namespace fast_lio

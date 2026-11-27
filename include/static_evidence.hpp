// static_evidence.hpp — "is the platform parked?" from sensors that do NOT
// depend on the pose estimate.
//
// The divergence guard freezes the map when the scan stops constraining the
// state, but a frozen map does not stop the iEKF: it keeps integrating IMU.
// Measured on dog 2 (2026-09-04), 9 s of effct=0 while parked cost 3.9 m of Z
// and 1.4 m/s of phantom velocity, and the front end never recovered — once
// the map unfroze it was rebuilt around the wrong pose. A static hold needs an
// answer to "is the robot actually moving?" that the pose estimate cannot
// contaminate, which leaves the IMU (no rotation, |a| flat at gravity) and the
// wheel stream.
//
// The verdict is deliberately conservative: it may only ever be true when the
// evidence positively says parked, which takes both: an IMU window at rest and a
// fresh, finite wheel sample at rest. A quiet IMU alone cannot tell parked from a
// slow, smooth motion, so without a usable wheel sample nothing is static. A
// non-finite IMU window is never static.
//
// A wheel sample that is ABSENT, STALE, from the scan's FUTURE or NON-FINITE is
// the same thing: no wheel evidence. The wheelspeed bridge says "cannot measure"
// in band, with NaN, and it does so whenever it is gated — while the legs work,
// and permanently if its wheel signs never lock or a sign mismatch latches. A
// configured wheel topic that never publishes is no evidence either. Only a
// sample the bridge measured testifies; runaway_watchdog.hpp folds the same
// finiteness into its freshness test.
//
// Pure C++. No ROS, no Eigen.

#pragma once

#include <cmath>

namespace fast_lio
{

struct StaticEvidenceParams
{
  double gyro_thresh = 0.05;         ///< rad/s; max |omega| in the window
  double acc_span_ratio = 0.02;      ///< (max-min)|a| / mean|a| — free of the IMU's unit scale
  double wheel_speed_thresh = 0.05;  ///< m/s
  double wheel_max_age = 0.5;        ///< s; older wheel samples do not testify
};

/// Per-scan IMU window summary. `samples == 0` means no IMU covered the scan.
struct ImuWindowStats
{
  int samples = 0;
  double gyr_max = 0.0;   ///< max |omega| (rad/s)
  double acc_min = 0.0;   ///< min |a| (whatever unit the driver publishes)
  double acc_max = 0.0;   ///< max |a|
  double acc_mean = 0.0;  ///< mean |a| — the scale the span is judged against
};

/// Is a wheel-odometry sample a measurement at all? The bridge publishes NaN to
/// mean "cannot measure" — the only in-band way for a Twist to say it, since 0
/// there is a measurement of standstill. Callers must not feed a sample that
/// fails this into anything that treats it as a number.
inline bool wheelSampleIsEvidence(double vx, double vy, double vz)
{
  return std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz);
}

/// `wheel_age` < 0 means "no wheel sample at all"; without a fresh, finite wheel sample nothing is static.
inline bool bodyIsStatic(const ImuWindowStats & imu,
                         double wheel_age,
                         double wheel_speed,
                         const StaticEvidenceParams & params)
{
  if (imu.samples <= 0) {
    return false;
  }
  if (!std::isfinite(imu.gyr_max) || !std::isfinite(imu.acc_min) || !std::isfinite(imu.acc_max) ||
      !std::isfinite(imu.acc_mean)) {
    return false;
  }
  if (imu.gyr_max > params.gyro_thresh) {
    return false;
  }
  // A body-fixed accelerometer at rest reads a constant |g|; motion of a legged
  // robot shows up as a spread even when the mean is unchanged. The test is a
  // ratio so it holds for drivers publishing m/s^2 or g.
  if (!(imu.acc_mean > 0.0) || (imu.acc_max - imu.acc_min) > params.acc_span_ratio * imu.acc_mean) {
    return false;
  }
  const bool wheel_usable = wheel_age >= 0.0 && std::isfinite(wheel_age) &&
                            wheel_age <= params.wheel_max_age && std::isfinite(wheel_speed);
  return wheel_usable && wheel_speed <= params.wheel_speed_thresh;
}

}  // namespace fast_lio

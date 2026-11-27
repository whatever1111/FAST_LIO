#pragma once

namespace fast_lio
{
// Gravity magnitude in the normalized units used by propagation and the S2 state, in m/s². It is the one gravity
// value of LIO-SLAM: the parent repository's lio_slam_core::kGravityMps2 (core/include/lio_slam_core/gravity.hpp)
// holds the same number for the fusion layer, and the two stay equal. A model constant, not a measured local
// gravity; the standard gravity 9.80665 m/s² that defines the unit g is a unit conversion and stays separate.
inline constexpr double G_m_s2 = 9.81;
inline constexpr int kGravityDenominator = 10000;
inline constexpr int kGravityNumerator = static_cast<int>(G_m_s2 * kGravityDenominator + 0.5);
static_assert(static_cast<double>(kGravityNumerator) / kGravityDenominator == G_m_s2);

// Default static-init reference in raw IMU units: the rest magnitude of an IMU that reports m/s², which is the
// gravity value above. An IMU that reports in g sets imu_init_still_gravity_ref to 1.
inline constexpr double kDefaultRawImuStillGravity = G_m_s2;
}  // namespace fast_lio

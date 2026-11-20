#pragma once

namespace fast_lio
{
// Gravity magnitude in the normalized units used by propagation and the S2 state, in m/s².
inline constexpr double G_m_s2 = 9.81;
inline constexpr int kGravityDenominator = 10000;
inline constexpr int kGravityNumerator = static_cast<int>(G_m_s2 * kGravityDenominator + 0.5);
static_assert(static_cast<double>(kGravityNumerator) / kGravityDenominator == G_m_s2);

// Default static-init reference in raw IMU units.
inline constexpr double kDefaultRawImuStillGravity = 9.81;
}  // namespace fast_lio

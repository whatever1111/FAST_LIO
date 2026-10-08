// Verify the frozen map after a long LiDAR hole without discarding its IMU propagation.
#pragma once

#include <cmath>

namespace fast_lio
{

inline bool validScanHoleVerifySec(double verify_s) noexcept
{
  return std::isfinite(verify_s) && verify_s >= 0.0;
}

// Zero disables verification. A gate already blind must keep its current verification progress.
inline bool scanHoleRequiresVerification(double gap_s, double verify_s, bool gate_already_blind) noexcept
{
  return verify_s > 0.0 && std::isfinite(gap_s) && gap_s > verify_s && !gate_already_blind;
}

}  // namespace fast_lio

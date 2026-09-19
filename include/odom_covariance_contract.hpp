#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>

namespace fast_lio
{

// Convert the iEKF's [position, rotation] covariance ordering to the existing
// Odometry wire ordering [rotation, position].  The permutation is applied to
// both axes, retaining all cross-covariances.  The degraded guard is expressed
// in wire coordinates and therefore affects only position diagonals.
inline std::array<double, 36> makeOdomCovariance(const Eigen::Matrix<double, 6, 6> & ekfCovariance, bool degraded)
{
  std::array<double, 36> covariance{};
  for (int row = 0; row < 6; ++row) {
    const int sourceRow = row < 3 ? row + 3 : row - 3;
    for (int col = 0; col < 6; ++col) {
      const int sourceCol = col < 3 ? col + 3 : col - 3;
      covariance[static_cast<std::size_t>(row * 6 + col)] = ekfCovariance(sourceRow, sourceCol);
    }
  }
  if (degraded) {
    constexpr double kDegradedPositionVariance = 100.0;
    covariance[21] += kDegradedPositionVariance;  // wire translation x
    covariance[28] += kDegradedPositionVariance;  // wire translation y
    covariance[35] += kDegradedPositionVariance;  // wire translation z
  }
  return covariance;
}

}  // namespace fast_lio

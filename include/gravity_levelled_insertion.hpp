#pragma once

#include <Eigen/Geometry>

#include "imu_tilt_reference.hpp"

namespace fast_lio
{
inline Eigen::Quaterniond levelledInsertionRotation(const Eigen::Quaterniond & rot, const Eigen::Vector3d & down_body)
{
  const Eigen::Vector3d reference = rot * down_body;
  if (!reference.allFinite() || reference.z() >= 0.0)
    return rot;

  const Eigen::Vector3d direction = reference.stableNormalized();
  if (direction.z() >= 0.0)
    return rot;

  const Eigen::Quaterniond align = Eigen::Quaterniond::FromTwoVectors(direction, -Eigen::Vector3d::UnitZ());
  if (align.w() == 1.0 && align.vec().isZero(0.0))
    return rot;
  return align * rot;
}

inline Eigen::Quaterniond insertionRotationFromImuTilt(const Eigen::Quaterniond & rot,
                                                       const ImuTiltReference & tilt,
                                                       bool prior_map_installed,
                                                       double t)
{
  if (prior_map_installed || !tilt.initialised())
    return rot;
  return levelledInsertionRotation(rot, tilt.downBodyAt(t));
}
}  // namespace fast_lio

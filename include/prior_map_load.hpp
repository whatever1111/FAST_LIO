// prior_map_load.hpp — the decisions taken when a prior map is loaded, before any of it
// reaches the map backend.
//
// A prior map arrives as a PCD plus an operator-supplied initial pose, and both can be
// wrong in ways that are silent until the estimate has already walked off: a PCD saved in
// a global frame (UTM) where a local ENU one was expected, an initial_pose whose roll and
// pitch disagree with gravity, a pose vector that is simply too short. None of those raise
// an error on their own — the map loads, the filter starts, and the first symptom is a
// diverged trajectory minutes later.
//
// So the checks live here, as values a caller can log and act on, rather than as branches
// buried in the loading code. Pure C++ + Eigen: no ROS, no PCL — the bounds pass takes a
// pair of iterators over anything with .x/.y/.z members.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <vector>

namespace fast_lio
{

/// A prior map centred this far from the origin is a global/UTM frame, not a local one.
/// Field maps run to a few hundred metres; a UTM easting is ~10^5-10^6 m.
inline constexpr double kPriorMapGlobalFrameDistM = 10000.0;

/// Number of elements an initial_pose vector must have: x, y, z, roll°, pitch°, yaw°.
inline constexpr std::size_t kInitialPoseSize = 6;

/// Axis-aligned bounds of a prior map, plus what they say about its coordinate frame.
struct PriorMapBounds
{
  bool valid = false;  ///< false when the cloud held no finite point
  double xmin = 0.0, xmax = 0.0;
  double ymin = 0.0, ymax = 0.0;
  double zmin = 0.0, zmax = 0.0;
  std::size_t finite_points = 0;
  double center_to_origin_m = 0.0;  ///< distance from the AABB centre to (0,0,0)
  /// centre further than kPriorMapGlobalFrameDistM: almost certainly UTM/global, and the
  /// initial_pose it will be used with is almost certainly local.
  bool likely_global_frame = false;
};

/// AABB over [first, last), skipping non-finite points (a PCD may carry NaNs).
template <typename It>
inline PriorMapBounds computePriorMapBounds(It first, It last)
{
  PriorMapBounds b;
  for (It it = first; it != last; ++it) {
    const double x = static_cast<double>(it->x);
    const double y = static_cast<double>(it->y);
    const double z = static_cast<double>(it->z);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    if (b.finite_points == 0) {
      b.xmin = b.xmax = x;
      b.ymin = b.ymax = y;
      b.zmin = b.zmax = z;
    } else {
      b.xmin = std::min(b.xmin, x);
      b.xmax = std::max(b.xmax, x);
      b.ymin = std::min(b.ymin, y);
      b.ymax = std::max(b.ymax, y);
      b.zmin = std::min(b.zmin, z);
      b.zmax = std::max(b.zmax, z);
    }
    ++b.finite_points;
  }
  if (b.finite_points == 0) {
    return b;
  }
  b.valid = true;
  const double cx = 0.5 * (b.xmin + b.xmax);
  const double cy = 0.5 * (b.ymin + b.ymax);
  const double cz = 0.5 * (b.zmin + b.zmax);
  b.center_to_origin_m = std::sqrt(cx * cx + cy * cy + cz * cz);
  b.likely_global_frame = b.center_to_origin_m > kPriorMapGlobalFrameDistM;
  return b;
}

/// The pose an initial_pose vector asks the filter to start from.
struct InitialPoseInjection
{
  bool valid = false;              ///< false when the vector was too short to be used
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  double applied_roll_rad = 0.0;   ///< what was actually used, after the override decision
  double applied_pitch_rad = 0.0;
  double applied_yaw_rad = 0.0;
  bool kept_gravity_rp = false;    ///< true when roll/pitch came from IMU gravity, not the vector
};

/// Resolve [x, y, z, roll°, pitch°, yaw°] against the filter's gravity-aligned attitude.
///
/// Yaw always comes from the vector — it is the one attitude component an operator can
/// supply and gravity cannot. Roll and pitch stay with the IMU unless the caller overrides:
/// a supplied roll/pitch that disagrees with gravity makes every body-to-world transform
/// wrong, so the caller has to say out loud that its values are gravity-consistent.
inline InitialPoseInjection resolveInitialPose(const std::vector<double> & pose_vec,
                                               bool full_rpy_override,
                                               double gravity_roll_rad,
                                               double gravity_pitch_rad)
{
  InitialPoseInjection out;
  if (pose_vec.size() < kInitialPoseSize) {
    return out;
  }
  for (std::size_t i = 0; i < kInitialPoseSize; ++i) {
    if (!std::isfinite(pose_vec[i])) {
      return out;
    }
  }
  constexpr double kDegToRad = M_PI / 180.0;
  out.valid = true;
  out.position = Eigen::Vector3d(pose_vec[0], pose_vec[1], pose_vec[2]);
  out.applied_roll_rad = full_rpy_override ? pose_vec[3] * kDegToRad : gravity_roll_rad;
  out.applied_pitch_rad = full_rpy_override ? pose_vec[4] * kDegToRad : gravity_pitch_rad;
  out.applied_yaw_rad = pose_vec[5] * kDegToRad;
  out.kept_gravity_rp = !full_rpy_override;
  const Eigen::AngleAxisd roll(out.applied_roll_rad, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(out.applied_pitch_rad, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(out.applied_yaw_rad, Eigen::Vector3d::UnitZ());
  out.orientation = Eigen::Quaterniond(yaw * pitch * roll);
  return out;
}

/// Whether a prior map of `voxels` voxels fits a backend bounded at `capacity`.
/// Capacity is enforced by eviction, so "does not fit" means "was silently truncated".
inline bool priorMapFitsCapacity(std::size_t voxels, std::size_t capacity)
{
  return capacity == 0u || voxels <= capacity;
}

}  // namespace fast_lio

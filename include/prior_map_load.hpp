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
#include <string>
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

/// Where the transform that puts a prior map into the front end's own frame comes from.
///
/// The map is delivered in the frame of the run that built it, and that frame is not a
/// property of the site: it is wherever the front end that built the map happened to
/// initialise, with whatever gravity estimate it had. A different binary, or the same
/// binary started two seconds later while the platform was moving, defines a different
/// one. Assuming the two coincide is the mistake this enum exists to make explicit.
enum class PriorMapAlign
{
  /// The map is already in this run's frame; load it as delivered (optionally moving the
  /// filter's start pose with initial_pose). The caller has to know this to be true.
  kAsDelivered,
  /// Wait for a relocalization result (T_map_camera_init) and install the map transformed
  /// into the front end's frame. Nothing is loaded until an alignment exists, and a later,
  /// better alignment replaces the whole tier rather than adding a second copy.
  kRelocalization,
};

/// Parse the prior_map_align parameter. Returns false (and kAsDelivered) on an unknown
/// value so the caller can report the name it did not recognise.
inline bool parsePriorMapAlign(const std::string & name, PriorMapAlign * out)
{
  if (name.empty() || name == "as_delivered" || name == "none") {
    *out = PriorMapAlign::kAsDelivered;
    return true;
  }
  if (name == "relocalization" || name == "reloc") {
    *out = PriorMapAlign::kRelocalization;
    return true;
  }
  *out = PriorMapAlign::kAsDelivered;
  return false;
}

/// Leaf size the prior map is downsampled to before it enters the map backend.
/// `requested` <= 0 means "same as the live map's dedup grid", which is the historical
/// behaviour; a smaller leaf keeps more of the map so a plane fit has neighbours to
/// choose from, at the cost of voxels.
inline double priorMapLeaf(double requested, double filter_size_map)
{
  return requested > 0.0 ? requested : filter_size_map;
}

/// Minimum movement of the alignment before an installed prior map is worth reinstalling.
/// Reinstalling costs a full clear + add of the tier, and the relocalization producer
/// republishes the same answer as it re-confirms it.
inline constexpr double kPriorMapReinstallTransM = 0.10;
inline constexpr double kPriorMapReinstallYawRad = 0.5 * M_PI / 180.0;

/// Whether a new alignment differs from the installed one by enough to reinstall.
/// `installed_valid` false means nothing is installed yet, so any alignment counts.
inline bool priorMapReinstallNeeded(bool installed_valid,
                                    const Eigen::Isometry3d & installed,
                                    const Eigen::Isometry3d & candidate,
                                    double trans_eps_m = kPriorMapReinstallTransM,
                                    double yaw_eps_rad = kPriorMapReinstallYawRad)
{
  if (!installed_valid) {
    return true;
  }
  const Eigen::Isometry3d delta = installed.inverse() * candidate;
  if (delta.translation().norm() > trans_eps_m) {
    return true;
  }
  const Eigen::Matrix3d & r = delta.linear();
  const double yaw = std::atan2(r(1, 0), r(0, 0));
  return std::abs(yaw) > yaw_eps_rad;
}

}  // namespace fast_lio

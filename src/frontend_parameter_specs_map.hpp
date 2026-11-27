#pragma once

#include <string>
#include <vector>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// The local map: filters, extent, insertion and saving, the prior map, the initial pose and the iVox backend.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct MapParameterSpecs
{
  const ParameterSpec<std::string> map_file_path{"map_file_path", "", ""};
  const ParameterSpec<double> filter_size_surf{"filter_size_surf", 0.5, 0.5};
  const ParameterSpec<double> filter_size_map{"filter_size_map", 0.5, 0.5};
  const ParameterSpec<double> cube_side_length{"cube_side_length", 200., 200.f};
  const ParameterSpec<float> mapping_det_range{"mapping.det_range", 300., 300.f};
  const ParameterSpec<int> mapping_map_insert_delay_scans{"mapping.map_insert_delay_scans", 0, 0};
  const ParameterSpec<bool> pcd_save_pcd_save_en{"pcd_save.pcd_save_en", false, false};
  const ParameterSpec<int> pcd_save_interval{"pcd_save.interval", -1, -1};
  // A delivered prior map goes into a map-backend tier of its own: it is not something this run
  // learned, so nothing this run does — LRU eviction as the map fills, or the guard dropping a
  // local map it can no longer register against — may take it away. On the 0831 lobby bag,
  // forcing the guard to lose its re-registration ends the run 11.1 m from truth with the prior
  // merged into the live map (the map goes 70023 -> 532 points, so there is nothing left to
  // re-register against) versus 1.5 m with its own tier. Keeping a map to re-register against is
  // not the same as knowing where in it you are, which is what relocalization decides, and the
  // tier cannot heal: the delivered map owns its cells for the whole run. iVox backend only; an
  // ikd-Tree build, which has no tier to keep it in, refuses a prior map.
  const ParameterSpec<std::string> prior_map_pcd{"prior_map_pcd", "", std::string("")};
  // prior_map_align — where the transform that puts the map into THIS run's frame
  // comes from. The map is delivered in the frame of the run that built it, and that
  // frame is not a property of the site: a different front-end binary, or the same one
  // initialising two seconds later while the platform walks, defines a different one.
  //   "as_delivered" (default) — the caller asserts the two frames coincide; the map is
  //       added as it is, and initial_pose moves the filter's start pose within it.
  //   "relocalization"         — nothing is loaded until /relocalization/result says
  //       where the map is (T_map_lidar_odom); the map is then installed transformed
  //       into the front end's own frame, and a later alignment replaces the whole tier.
  // Measured on the 0831 lobby bag: the delivered map's frame sat 1.22 m / 3.4 deg from
  // the replaying front end's, and loading it as delivered cost 0.253 -> 2.366 m of 2D
  // error against RTK. Supplying that offset alone brought it back to 0.717 m.
  const ParameterSpec<std::string> prior_map_align{"prior_map_align", "as_delivered", std::string("as_delivered")};
  // Leaf the prior map is downsampled to. 0 = filter_size_map, the live map's dedup grid,
  // which is what the map backend enforces anyway. A finer leaf survives that grid only
  // where the live map is sparse, but it costs nothing to ask for: at 0.3 the prior map
  // enters as one centroid per cell and the effective correspondence count per scan fell
  // 25-38% on the 0831 bag (accuracy held there; a starved scene may not).
  const ParameterSpec<double> prior_map_voxel{"prior_map_voxel", 0.0, 0.0};
  // Stop growing the map once a prior map is installed (pure localization). Without this
  // the map is neither: the prior owns its cells, live scans fill the gaps between them.
  const ParameterSpec<bool> prior_map_freeze{"prior_map_freeze", false, false};
  // [x, y, z, roll°, pitch°, yaw°]: the position and yaw are injected, roll and pitch stay with the IMU's gravity
  // alignment (prior_map_load.hpp resolveInitialPose).
  const ParameterSpec<std::vector<double>> initial_pose{"initial_pose", std::vector<double>(), std::vector<double>()};
  // iVox map backend (option B; declared only by USE_IVOX builds). Its neighbour search covers all 26 voxels around
  // a point's own (laserMapping.cpp). res=0.5 NN voxels give the best probe NN-agreement (99.9%, tightest normals);
  // a GLOBAL fine-grid dedup (= filter_size_map) keeps the map at ikd-Tree's 542k.
  const DeclaredValueSpec<double> ivox_grid_resolution{"ivox_grid_resolution", 0.5, false};
  const DeclaredValueSpec<int> ivox_voxel_capacity{"ivox_voxel_capacity", 50, false};
  const DeclaredValueSpec<int> ivox_max_voxels{"ivox_max_voxels", 5000000, false};
};
}  // namespace fast_lio

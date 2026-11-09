#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// The lidar measurement update: iterations, the normal-noise floor, attitude hold, correspondence gating and kernels,
/// canopy and ground recovery, nearest-neighbour refresh. One spec per parameter, in declaration order, named after the
/// parameter with '.' written '_'. FrontendParameters declares and reads them in the constructor's original order, not
/// domain by domain.
struct MeasurementParameterSpecs
{
  const ParameterSpec<int> max_iteration{"max_iteration", 4, 4};
  const ParameterSpec<bool> degeneracy_debug{"degeneracy_debug", false, false};
  const ParameterSpec<bool> degeneracy_spectrum_trace{"degeneracy_spectrum_trace", false, false};
  // The normal-noise floor of the LiDAR position update (normal_noise_floor.hpp): off | observe | apply.
  const ParameterSpec<std::string> degeneracy_floor_mode{"degeneracy_floor_mode", "off", std::string("off")};
  // alpha: the rate at which a direction made only of floor is kept anyway, per direction and frame; in (0, 0.5).
  const ParameterSpec<double> degeneracy_false_keep_rate{"degeneracy_false_keep_rate", 1e-3, 1e-3};
  // kappa, the scale of the floor model (> 0), and sigma_rel, its relative uncertainty (>= 0).
  const ParameterSpec<double> degeneracy_floor_model_scale{"degeneracy_floor_model_scale", 1.0, 1.0};
  const ParameterSpec<double> degeneracy_floor_rel_uncertainty{"degeneracy_floor_rel_uncertainty", 0.3, 0.3};
  // Regularisation of the Schur complement the floor judges, relative to trace(H_rr)/3.
  const ParameterSpec<double> degeneracy_schur_ridge{"degeneracy_schur_ridge", 1e-6, 1e-6};
  const ParameterSpec<bool> lidar_attitude_hold_en{"lidar_attitude_hold_en", false, false};
  const ParameterSpec<double> corr_innov_gate_k{"corr_innov_gate_k", 0.0, 0.0};
  const ParameterSpec<double> corr_nn0_max{"corr_nn0_max", 0.0, 0.0};
  const ParameterSpec<double> corr_gate_mad_scale{"corr_gate_mad_scale", 0.0, 0.0};
  const ParameterSpec<double> corr_gate_rot_sigma_floor_deg{"corr_gate_rot_sigma_floor_deg", 0.0, 0.0};
  const ParameterSpec<std::string> corr_kernel{"corr_kernel", "none", std::string("none")};
  const ParameterSpec<double> corr_kernel_mad_scale{"corr_kernel_mad_scale", 3.0, 3.0};
  const ParameterSpec<double> corr_kernel_min_scale{"corr_kernel_min_scale", 0.03, 0.03};
  const ParameterSpec<double> lidar_attitude_hold_far_frac{"lidar_attitude_hold_far_frac", 0.0, 0.0};
  const ParameterSpec<int> lidar_attitude_hold_min_effct{"lidar_attitude_hold_min_effct", 100, 100};
  const ParameterSpec<double> lidar_attitude_hold_max_s{"lidar_attitude_hold_max_s", 3.0, 3.0};
  const ParameterSpec<double> lidar_attitude_hold_max_scan_gap_s{"lidar_attitude_hold_max_scan_gap_s", 0.3, 0.3};
  const ParameterSpec<double> lidar_attitude_hold_min_rp_info{"lidar_attitude_hold_min_rp_info", 0.0, 0.0};
  const ParameterSpec<double> lidar_attitude_hold_z_weak{"lidar_attitude_hold_z_weak", 0.9, 0.9};
  const ParameterSpec<bool> lidar_consider_rp_en{"lidar_consider_rp_en", false, false};
  const ParameterSpec<bool> lidar_consider_rp_while_engulfed{"lidar_consider_rp_while_engulfed", true, true};
  const ParameterSpec<double> lidar_consider_rp_post_release_s{"lidar_consider_rp_post_release_s", 0.0, 0.0};
  const ParameterSpec<double> lidar_consider_rp_min_info{"lidar_consider_rp_min_info", 0.0, 0.0};
  const ParameterSpec<double> lidar_consider_rp_z_weak_min{"lidar_consider_rp_z_weak_min", 1.1, 1.1};
  const ParameterSpec<double> lidar_consider_rp_max_s{"lidar_consider_rp_max_s", 0.0, 0.0};
  const ParameterSpec<double> canopy_near_range{"canopy_near_range", 20.0, 20.0};
  const ParameterSpec<double> canopy_ground_z{"canopy_ground_z", -0.47, -0.47};
  const ParameterSpec<double> canopy_ground_halfband{"canopy_ground_halfband", 0.8, 0.8};
  const ParameterSpec<bool> ground_recover_en{"ground_recover_en", false, false};
  const ParameterSpec<double> ground_recover_scan_band{"ground_recover_scan_band", 0.15, 0.15};
  const ParameterSpec<int> ground_recover_min_inliers{"ground_recover_min_inliers", 100, 100};
  const ParameterSpec<double> ground_recover_max_residual{"ground_recover_max_residual", 1.0, 1.0};
  const ParameterSpec<double> ground_recover_sector_tol{"ground_recover_sector_tol", 0.25, 0.25};
  const ParameterSpec<int> ground_recover_max_rows{"ground_recover_max_rows", 4000, 4000};
  const ParameterSpec<bool> nn_refresh_gate_en{"nn_refresh_gate_en", false, false};
  const ParameterSpec<double> nn_refresh_min_shift{"nn_refresh_min_shift", 0.0, 0.0};
  const ParameterSpec<double> nn_refresh_ref_range{"nn_refresh_ref_range", 20.0, 20.0};
};
}  // namespace fast_lio

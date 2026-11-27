#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// The lidar measurement update: iterations, the degeneracy traces, correspondence gating, the roll/pitch consider
/// rule, nearest-neighbour refresh. One spec per parameter, in declaration order, named
/// after the parameter with '.' written '_'. FrontendParameters declares and reads them in the constructor's original
/// order, not domain by domain.
struct MeasurementParameterSpecs
{
  const ParameterSpec<int> max_iteration{"max_iteration", 4, 4};
  const ParameterSpec<bool> degeneracy_debug{"degeneracy_debug", false, false};
  const ParameterSpec<bool> degeneracy_spectrum_trace{"degeneracy_spectrum_trace", false, false};
  // Regularisation of the Schur complement the spectrum trace reports, relative to trace(H_rr)/3.
  const ParameterSpec<double> degeneracy_schur_ridge{"degeneracy_schur_ridge", 1e-6, 1e-6};
  const ParameterSpec<double> corr_innov_gate_k{"corr_innov_gate_k", 0.0, 0.0};
  const ParameterSpec<double> corr_nn0_max{"corr_nn0_max", 0.0, 0.0};
  const ParameterSpec<double> corr_gate_mad_scale{"corr_gate_mad_scale", 0.0, 0.0};
  const ParameterSpec<double> corr_gate_rot_sigma_floor_deg{"corr_gate_rot_sigma_floor_deg", 0.0, 0.0};
  const ParameterSpec<bool> lidar_consider_rp_en{"lidar_consider_rp_en", false, false};
  const ParameterSpec<bool> lidar_consider_rp_while_engulfed{"lidar_consider_rp_while_engulfed", true, true};
  const ParameterSpec<double> lidar_consider_rp_post_release_s{"lidar_consider_rp_post_release_s", 0.0, 0.0};
  const ParameterSpec<double> lidar_consider_rp_min_info{"lidar_consider_rp_min_info", 0.0, 0.0};
  const ParameterSpec<double> lidar_consider_rp_z_weak_min{"lidar_consider_rp_z_weak_min", 1.1, 1.1};
  const ParameterSpec<double> lidar_consider_rp_max_s{"lidar_consider_rp_max_s", 0.0, 0.0};
  const ParameterSpec<bool> nn_refresh_gate_en{"nn_refresh_gate_en", false, false};
  const ParameterSpec<double> nn_refresh_min_shift{"nn_refresh_min_shift", 0.0, 0.0};
  const ParameterSpec<double> nn_refresh_ref_range{"nn_refresh_ref_range", 20.0, 20.0};
};
}  // namespace fast_lio

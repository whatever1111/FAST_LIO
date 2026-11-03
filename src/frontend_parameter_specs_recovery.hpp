#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// Divergence guards, engulfment release, holes, ZUPT and parked hold, re-anchoring, the runaway watchdog and LOST
/// recovery. One spec per parameter, in declaration order, named after the parameter with '.' written '_'.
/// FrontendParameters declares and reads them in the constructor's original order, not domain by domain.
struct RecoveryParameterSpecs
{
  const ParameterSpec<double> guard_vel_reset_sigma{"guard_vel_reset_sigma", 0.0, 0.0};
  const ParameterSpec<double> zupt_vel_reset_sigma{"zupt_vel_reset_sigma", 0.0, 0.0};
  const ParameterSpec<bool> divergence_guard_en{"divergence_guard_en", true, true};
  const ParameterSpec<int> divergence_guard_min_eff{"divergence_guard_min_eff", 50, 50};
  const ParameterSpec<bool> guard_engulf_en{"guard_engulf_en", true, true};
  const ParameterSpec<double> guard_engulf_far_range{"guard_engulf_far_range", 2.0, 2.0};
  const ParameterSpec<double> guard_engulf_far_frac_min{"guard_engulf_far_frac_min", 0.15, 0.15};
  const ParameterSpec<int> guard_engulf_streak{"guard_engulf_streak", 2, 2};
  const ParameterSpec<double> guard_engulf_far_frac_release{"guard_engulf_far_frac_release", 0.35, 0.35};
  const ParameterSpec<int> guard_engulf_release_scans{"guard_engulf_release_scans", 5, 5};
  const ParameterSpec<std::string> guard_release_vel_source{"guard_release_vel_source", "zero", "zero"};
  const ParameterSpec<double> guard_release_leg_max_age_s{"guard_release_leg_max_age_s", 0.5, 0.5};
  const ParameterSpec<double> guard_release_leg_sigma{"guard_release_leg_sigma", 0.3, 0.3};
  const ParameterSpec<double> guard_release_keep_sigma{"guard_release_keep_sigma", 0.5, 0.5};
  const ParameterSpec<bool> guard_release_inflate_en{"guard_release_inflate_en", false, false};
  const ParameterSpec<double> guard_release_pos_sigma_min{"guard_release_pos_sigma_min", 0.1, 0.1};
  const ParameterSpec<double> guard_release_pos_sigma_rate{"guard_release_pos_sigma_rate", 0.2, 0.2};
  const ParameterSpec<double> guard_release_pos_sigma_max{"guard_release_pos_sigma_max", 1.0, 1.0};
  const ParameterSpec<double> guard_release_rp_sigma_deg{"guard_release_rp_sigma_deg", 0.5, 0.5};
  const ParameterSpec<double> guard_release_rp_sigma_rate_deg{"guard_release_rp_sigma_rate_deg", 0.3, 0.3};
  const ParameterSpec<double> guard_release_rp_sigma_max_deg{"guard_release_rp_sigma_max_deg", 3.0, 3.0};
  const ParameterSpec<double> guard_release_yaw_sigma_deg{"guard_release_yaw_sigma_deg", 0.0, 0.0};
  const ParameterSpec<bool> guard_hole_en{"guard_hole_en", false, false};
  const ParameterSpec<double> guard_hole_min_sec{"guard_hole_min_sec", 0.3, 0.3};
  const ParameterSpec<double> guard_hole_max_accel{"guard_hole_max_accel", 1.0, 1.0};
  const ParameterSpec<bool> guard_hole_reanchor_pos{"guard_hole_reanchor_pos", true, true};
  const ParameterSpec<int> divergence_guard_streak{"divergence_guard_streak", 5, 5};
  const ParameterSpec<double> divergence_guard_max_speed{"divergence_guard_max_speed", 30.0, 30.0};
  const ParameterSpec<bool> zupt_en{"zupt_en", true, true};
  const ParameterSpec<double> zupt_gyro_thresh{"zupt_gyro_thresh", 0.05, 0.05};
  const ParameterSpec<double> zupt_acc_span_ratio{"zupt_acc_span_ratio", 0.02, 0.02};
  const ParameterSpec<double> zupt_wheel_speed_thresh{"zupt_wheel_speed_thresh", 0.05, 0.05};
  const ParameterSpec<double> zupt_wheel_max_age{"zupt_wheel_max_age", 0.5, 0.5};
  const ParameterSpec<bool> zupt_require_wheel{"zupt_require_wheel", false, false};
  const ParameterSpec<bool> zupt_anchor_trace_en{"zupt_anchor_trace_en", false, false};
  const ParameterSpec<bool> zupt_parked_hold_en{"zupt_parked_hold_en", false, false};
  const ParameterSpec<int> zupt_parked_hold_scans{"zupt_parked_hold_scans", 3, 3};
  const ParameterSpec<double> zupt_hold_pos_noise{"zupt_hold_pos_noise", 0.05, 0.05};
  const ParameterSpec<double> zupt_hold_rot_noise{"zupt_hold_rot_noise", 0.02, 0.02};
  const ParameterSpec<double> zupt_hold_vel_noise{"zupt_hold_vel_noise", 0.05, 0.05};
  const ParameterSpec<double> zupt_hold_deadband_sigma{"zupt_hold_deadband_sigma", 3.0, 3.0};
  const ParameterSpec<int> zupt_anchor_confirm_scans{"zupt_anchor_confirm_scans", 3, 3};
  const ParameterSpec<double> zupt_anchor_max_scan_gap_s{"zupt_anchor_max_scan_gap_s", 0.0, 0.0};
  const ParameterSpec<bool> reanchor_en{"reanchor_en", true, true};
  const ParameterSpec<int> reanchor_min_eff{"reanchor_min_eff", 200, 200};
  const ParameterSpec<double> reanchor_max_res{"reanchor_max_res", 0.10, 0.10};
  const ParameterSpec<int> reanchor_confirm_scans{"reanchor_confirm_scans", 3, 3};
  const ParameterSpec<double> reanchor_timeout_sec{"reanchor_timeout_sec", 5.0, 5.0};
  const ParameterSpec<std::string> reanchor_on_fail{"reanchor_on_fail", "hold", std::string("hold")};
  const ParameterSpec<double> reanchor_max_res_ratio{"reanchor_max_res_ratio", 0.0, 0.0};
  const ParameterSpec<double> reanchor_res_floor{"reanchor_res_floor", 0.04, 0.04};
  const ParameterSpec<double> reanchor_min_eff_ratio{"reanchor_min_eff_ratio", 0.0, 0.0};
  const ParameterSpec<int> reanchor_healthy_window{"reanchor_healthy_window", 100, 100};
  const ParameterSpec<double> reanchor_total_timeout_sec{"reanchor_total_timeout_sec", 0.0, 0.0};
  const ParameterSpec<double> reanchor_recovery_stable_sec{"reanchor_recovery_stable_sec", 1.0, 1.0};
  const ParameterSpec<bool> runaway_watchdog_en{"runaway_watchdog_en", true, true};
  const ParameterSpec<double> runaway_parked_speed_thresh{"runaway_parked_speed_thresh", 0.3, 0.3};
  const ParameterSpec<double> runaway_wheel_speed_margin{"runaway_wheel_speed_margin", 1.0, 1.0};
  const ParameterSpec<double> runaway_hold_sec{"runaway_hold_sec", 3.0, 3.0};
  const ParameterSpec<std::string> runaway_on_trip{"runaway_on_trip", "hold", std::string("hold")};
  const DeclaredValueSpec<bool> lost_recovery_en{"lost_recovery_en", false, true};
  const DeclaredValueSpec<double> lost_recovery_stable_sec{"lost_recovery_stable_sec", 2.0, true};
};
}  // namespace fast_lio

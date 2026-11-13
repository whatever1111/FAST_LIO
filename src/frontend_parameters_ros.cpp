#include "frontend_parameters_ros.hpp"

#include <rclcpp/logging.hpp>

#include <Eigen/Core>

#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "frontend_observation.hpp"
#include "frontend_parameter_spec.hpp"
#include "frontend_parameter_specs_constraints.hpp"
#include "frontend_parameter_specs_diagnostics.hpp"
#include "frontend_parameter_specs_downsample.hpp"
#include "frontend_parameter_specs_imu.hpp"
#include "frontend_parameter_specs_input.hpp"
#include "frontend_parameter_specs_map.hpp"
#include "frontend_parameter_specs_measurement.hpp"
#include "frontend_parameter_specs_output.hpp"
#include "frontend_parameter_specs_recovery.hpp"
#include "prior_map_load.hpp"

namespace fast_lio
{
// Each function builds the spec objects of the domains it declares or reads: constants made per call, so that nothing
// but the values read outlives the call.

void FrontendParameters::declareFront(rclcpp::Node & node)
{
  declareOutputsAndMeasurement(node);
  declareGravityToConsider(node);
  declareHoleToLostRecovery(node);
}

void FrontendParameters::declareOutputsAndMeasurement(rclcpp::Node & node)
{
  const OutputParameterSpecs output{};
  const MeasurementParameterSpecs measurement{};
  const InputParameterSpecs input{};
  const DownsampleParameterSpecs downsample{};
  const ConstraintParameterSpecs constraints{};
  const DiagnosticsParameterSpecs diagnostics{};
  declareOnly(node, output.publish_path_en);
  declareOnly(node, output.publish_effect_map_en);
  declareOnly(node, output.publish_map_en);
  declareOnly(node, output.publish_map_period_sec);
  declareOnly(node, output.publish_scan_publish_en);
  declareOnly(node, output.publish_dense_publish_en);
  declareOnly(node, output.publish_scan_bodyframe_pub_en);
  declareOnly(node, output.publish_odom_frame_id);
  declareOnly(node, output.publish_body_frame_id);
  declareOnly(node, measurement.max_iteration);
  declareOnly(node, input.max_buffered_scans);
  declareOnly(node, input.max_scan_backlog_sec);
  declareOnly(node, input.max_imu_backlog_sec);
  declareOnly(node, downsample.adaptive_downsample_en);
  declareOnly(node, downsample.adaptive_downsample_budget_sec);
  declareOnly(node, downsample.adaptive_downsample_release_sec);
  declareOnly(node, downsample.adaptive_downsample_step);
  declareOnly(node, downsample.adaptive_downsample_max_voxel);
  declareOnly(node, downsample.adaptive_downsample_raise_after);
  declareOnly(node, downsample.adaptive_downsample_lower_after);
  declareOnly(node, measurement.degeneracy_debug);
  declareOnly(node, measurement.degeneracy_spectrum_trace);
  declareOnly(node, measurement.degeneracy_floor_mode);
  declareOnly(node, measurement.degeneracy_false_keep_rate);
  declareOnly(node, measurement.degeneracy_floor_model_scale);
  declareOnly(node, measurement.degeneracy_floor_rel_uncertainty);
  declareOnly(node, measurement.degeneracy_schur_ridge);
  declareOnly(node, output.health_pub_en);
  declareOnly(node, constraints.planar_constraint_en);
  declareOnly(node, constraints.planar_constraint_noise);
  declareOnly(node, constraints.planar_constraint_z_weak_thresh);
  declareOnly(node, constraints.gravity_align_en);
  declareOnly(node, constraints.gravity_align_noise);
  declareOnly(node, constraints.gravity_align_z_weak_thresh);
  declareOnly(node, constraints.gravity_align_accel_tol);
  declareOnly(node, constraints.gravity_align_gyro_tol);
  declareOnly(node, constraints.gravity_align_window_s);
  declareOnly(node, constraints.gravity_align_continuous);
  declareOnly(node, constraints.gravity_align_hold_s);
  declareOnly(node, constraints.gravity_align_noise_degraded);
  declareOnly(node, constraints.gravity_align_mag_ref);
  declareOnly(node, constraints.gravity_align_grav_leak);
  declareOnly(node, constraints.gravity_align_grav_cap_deg);
  declareOnly(node, constraints.gravity_align_grav_freeze_degraded);
  declareOnly(node, measurement.lidar_attitude_hold_en);
  declareOnly(node, measurement.corr_innov_gate_k);
  declareOnly(node, measurement.corr_nn0_max);
  declareOnly(node, measurement.corr_gate_mad_scale);
  declareOnly(node, measurement.corr_gate_rot_sigma_floor_deg);
  declareOnly(node, measurement.corr_kernel);
  declareOnly(node, measurement.corr_kernel_mad_scale);
  declareOnly(node, measurement.corr_kernel_min_scale);
  declareOnly(node, diagnostics.corr_dump_en);
  declareOnly(node, diagnostics.corr_dump_windows);
}

void FrontendParameters::declareGravityToConsider(rclcpp::Node & node)
{
  const MeasurementParameterSpecs measurement{};
  const RecoveryParameterSpecs recovery{};
  const ConstraintParameterSpecs constraints{};
  const ImuParameterSpecs imu{};
  declareOnly(node, measurement.lidar_attitude_hold_far_frac);
  declareOnly(node, measurement.lidar_attitude_hold_min_effct);
  declareOnly(node, measurement.lidar_attitude_hold_max_s);
  declareOnly(node, measurement.lidar_attitude_hold_max_scan_gap_s);
  declareOnly(node, measurement.lidar_attitude_hold_min_rp_info);
  declareOnly(node, recovery.guard_vel_reset_sigma);
  declareOnly(node, recovery.zupt_vel_reset_sigma);
  declareOnly(node, measurement.lidar_attitude_hold_z_weak);
  declareOnly(node, constraints.gravity_align_lin_accel_comp);
  declareOnly(node, constraints.gravity_align_vel_sigma_scale);
  declareOnly(node, constraints.gravity_align_lin_accel_source);
  declareOnly(node, constraints.gravity_align_withhold_blind_moving);
  declareOnly(node, constraints.gravity_align_sigma_policy);
  declareOnly(node, constraints.gravity_align_noise_static);
  declareOnly(node, constraints.gravity_align_noise_moving);
  declareOnly(node, constraints.gravity_align_leg_vel_sigma);
  declareOnly(node, constraints.gravity_align_leg_edge_window_s);
  declareOnly(node, constraints.gravity_align_leg_lever_arm);
  declareOnly(node, imu.imu_init_require_still);
  declareOnly(node, imu.imu_init_still_tol);
  declareOnly(node, imu.imu_init_still_timeout_s);
  declareOnly(node, imu.imu_init_still_sustained_en);
  declareOnly(node, imu.imu_init_still_min_duration_s);
  declareOnly(node, imu.imu_init_still_min_samples);
  declareOnly(node, imu.imu_init_still_gyro_rms_max);
  declareOnly(node, imu.imu_init_still_gravity_ref);
  declareOnly(node, imu.imu_init_still_gravity_tol);
  declareOnly(node, imu.imu_init_still_max_gap_s);
  declareOnly(node, recovery.divergence_guard_en);
  declareOnly(node, recovery.divergence_guard_min_eff);
  declareOnly(node, recovery.guard_engulf_en);
  declareOnly(node, recovery.guard_engulf_far_range);
  declareOnly(node, recovery.guard_engulf_far_frac_min);
  declareOnly(node, recovery.guard_engulf_streak);
  declareOnly(node, recovery.guard_engulf_far_frac_release);
  declareOnly(node, recovery.guard_engulf_release_scans);
  declareOnly(node, recovery.guard_release_vel_source);
  declareOnly(node, recovery.guard_release_leg_max_age_s);
  declareOnly(node, recovery.guard_release_leg_sigma);
  declareOnly(node, recovery.guard_release_keep_sigma);
  declareOnly(node, recovery.guard_release_inflate_en);
  declareOnly(node, recovery.guard_release_pos_sigma_min);
  declareOnly(node, recovery.guard_release_pos_sigma_rate);
  declareOnly(node, recovery.guard_release_pos_sigma_max);
  declareOnly(node, recovery.guard_release_rp_sigma_deg);
  declareOnly(node, recovery.guard_release_rp_sigma_rate_deg);
  declareOnly(node, recovery.guard_release_rp_sigma_max_deg);
  declareOnly(node, recovery.guard_release_yaw_sigma_deg);
  declareOnly(node, measurement.lidar_consider_rp_en);
  declareOnly(node, measurement.lidar_consider_rp_while_engulfed);
  declareOnly(node, measurement.lidar_consider_rp_post_release_s);
  declareOnly(node, measurement.lidar_consider_rp_min_info);
  declareOnly(node, measurement.lidar_consider_rp_z_weak_min);
  declareOnly(node, measurement.lidar_consider_rp_max_s);
}

void FrontendParameters::declareHoleToLostRecovery(rclcpp::Node & node)
{
  const RecoveryParameterSpecs recovery{};
  declareOnly(node, recovery.guard_hole_en);
  declareOnly(node, recovery.guard_hole_min_sec);
  declareOnly(node, recovery.guard_hole_max_accel);
  declareOnly(node, recovery.guard_hole_reanchor_pos);
  declareOnly(node, recovery.divergence_guard_streak);
  declareOnly(node, recovery.divergence_guard_max_speed);
  declareOnly(node, recovery.zupt_en);
  declareOnly(node, recovery.zupt_gyro_thresh);
  declareOnly(node, recovery.zupt_acc_span_ratio);
  declareOnly(node, recovery.zupt_wheel_speed_thresh);
  declareOnly(node, recovery.zupt_wheel_max_age);
  declareOnly(node, recovery.zupt_require_wheel);
  declareOnly(node, recovery.zupt_anchor_trace_en);
  declareOnly(node, recovery.zupt_parked_hold_en);
  declareOnly(node, recovery.zupt_parked_hold_scans);
  declareOnly(node, recovery.zupt_hold_pos_noise);
  declareOnly(node, recovery.zupt_hold_rot_noise);
  declareOnly(node, recovery.zupt_hold_vel_noise);
  declareOnly(node, recovery.zupt_hold_deadband_sigma);
  declareOnly(node, recovery.zupt_anchor_confirm_scans);
  declareOnly(node, recovery.zupt_anchor_max_scan_gap_s);
  declareOnly(node, recovery.reanchor_en);
  declareOnly(node, recovery.reanchor_min_eff);
  declareOnly(node, recovery.reanchor_max_res);
  declareOnly(node, recovery.reanchor_confirm_scans);
  declareOnly(node, recovery.reanchor_timeout_sec);
  declareOnly(node, recovery.reanchor_on_fail);
  declareOnly(node, recovery.reanchor_max_res_ratio);
  declareOnly(node, recovery.reanchor_res_floor);
  declareOnly(node, recovery.reanchor_min_eff_ratio);
  declareOnly(node, recovery.reanchor_require_guard_clear);
  declareOnly(node, recovery.reanchor_healthy_window);
  declareOnly(node, recovery.reanchor_total_timeout_sec);
  declareOnly(node, recovery.reanchor_recovery_stable_sec);
  declareOnly(node, recovery.reanchor_recovery_entry_view);
  declareValue(node, recovery.lost_recovery_en);
  declareValue(node, recovery.lost_recovery_stable_sec);
}

void FrontendParameters::declareActivation(rclcpp::Node & node,
                                           bool & observation_en,
                                           bool & reference_health_en,
                                           std::string & clock_domain,
                                           std::int64_t & idle_period_ns)
{
  const OutputParameterSpecs output{};
  observation_en = declareValue(node, output.publish_frontend_observation);
  reference_health_en = declareValue(node, output.publish_reference_health);
  clock_domain = declareValue(node, output.publish_frontend_clock_domain);
  const double frontendIdlePeriod = declareValue(node, output.publish_frontend_idle_period_sec);
  if (reference_health_en && !observation_en)
    throw std::invalid_argument("reference health requires publish.frontend_observation=true and its clock domain");
  if (observation_en) {
    if (!fast_lio::frontendIdentifier(clock_domain, 48) || !std::isfinite(frontendIdlePeriod) ||
        frontendIdlePeriod < 1e-9 || frontendIdlePeriod > 3600.0) {
      throw std::invalid_argument(
        "frontend observation requires an explicit clock domain (1..48 bytes) and idle period 1ns..3600s");
    }
    idle_period_ns = static_cast<std::int64_t>(frontendIdlePeriod * 1e9);
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("publish.frontend_idle_period_sec", frontendIdlePeriod);
#endif
}

void FrontendParameters::declareBack(rclcpp::Node & node)
{
  declareRunawayToPreprocess(node);
  declareImuGapToWheel(node);
}

void FrontendParameters::declareRunawayToPreprocess(rclcpp::Node & node)
{
  const RecoveryParameterSpecs recovery{};
  const MeasurementParameterSpecs measurement{};
  const MapParameterSpecs map{};
  const DiagnosticsParameterSpecs diagnostics{};
  const InputParameterSpecs input{};
  const ImuParameterSpecs imu{};
  declareOnly(node, recovery.runaway_watchdog_en);
  declareOnly(node, recovery.runaway_parked_speed_thresh);
  declareOnly(node, recovery.runaway_wheel_speed_margin);
  declareOnly(node, recovery.runaway_hold_sec);
  declareOnly(node, recovery.runaway_on_trip);
  declareOnly(node, recovery.input_epoch_on_fault);
  declareOnly(node, measurement.canopy_near_range);
  declareOnly(node, measurement.canopy_ground_z);
  declareOnly(node, measurement.canopy_ground_halfband);
  declareOnly(node, measurement.ground_recover_en);
  declareOnly(node, measurement.ground_recover_scan_band);
  declareOnly(node, measurement.ground_recover_min_inliers);
  declareOnly(node, measurement.ground_recover_max_residual);
  declareOnly(node, measurement.ground_recover_sector_tol);
  declareOnly(node, measurement.ground_recover_max_rows);
  declareOnly(node, map.map_file_path);
  declareOnly(node, diagnostics.data_dir);
  declareOnly(node, input.common_lid_topic);
  declareOnly(node, input.common_imu_topic);
  declareOnly(node, input.common_imu_input_type);
  declareOnly(node, input.common_imu_bias_topic);
  declareOnly(node, input.common_time_sync_en);
  declareOnly(node, input.common_time_offset_lidar_to_imu);
  declareOnly(node, input.common_imu_time_offset);
  declareOnly(node, map.filter_size_corner);
  declareOnly(node, map.filter_size_surf);
  declareOnly(node, map.filter_size_map);
  declareOnly(node, map.cube_side_length);
  declareOnly(node, map.mapping_det_range);
  declareOnly(node, map.mapping_fov_degree);
  declareOnly(node, imu.mapping_gyr_cov);
  declareOnly(node, imu.mapping_acc_cov);
  declareOnly(node, imu.mapping_b_gyr_cov);
  declareOnly(node, imu.mapping_b_acc_cov);
  declareOnly(node, input.preprocess_blind);
  declareOnly(node, input.preprocess_lidar_type);
  declareOnly(node, input.preprocess_scan_line);
  declareOnly(node, input.preprocess_timestamp_unit);
  declareOnly(node, input.preprocess_scan_rate);
  declareOnly(node, input.preprocess_max_scan_duration_s);
}

void FrontendParameters::declareImuGapToWheel(rclcpp::Node & node)
{
  const ImuParameterSpecs imu{};
  const InputParameterSpecs input{};
  const MeasurementParameterSpecs measurement{};
  const DiagnosticsParameterSpecs diagnostics{};
  const MapParameterSpecs map{};
  const ConstraintParameterSpecs constraints{};
  declareOnly(node, imu.imu_coverage_max_gap_s);
  declareOnly(node, imu.imu_coverage_max_extrapolation_s);
  declareOnly(node, imu.imu_gap_prior_en);
  declareOnly(node, imu.imu_gap_min_s);
  declareOnly(node, imu.imu_gap_accel_max);
  declareOnly(node, imu.imu_gap_accel_max_z);
  declareOnly(node, imu.imu_gap_rate_rp_max);
  declareOnly(node, imu.imu_gap_rate_yaw_max);
  declareOnly(node, imu.imu_gap_verify_s);
  declareOnly(node, imu.imu_gap_empty_scans);
  declareOnly(node, input.point_filter_num);
  declareOnly(node, measurement.nn_refresh_gate_en);
  declareOnly(node, measurement.nn_refresh_min_shift);
  declareOnly(node, measurement.nn_refresh_ref_range);
  declareOnly(node, input.feature_extract_enable);
  declareOnly(node, diagnostics.runtime_pos_log_enable);
  declareOnly(node, diagnostics.diagnostics_latency_enable);
  declareOnly(node, diagnostics.diagnostics_latency_log_period_sec);
  declareOnly(node, diagnostics.diagnostics_kalman_channel_enable);
  values_.time_admission_counts_enable = declareValue(node, diagnostics.diagnostics_time_admission_counts_enable);
  declareOnly(node, imu.mapping_extrinsic_est_en);
  declareOnly(node, map.mapping_map_insert_delay_scans);
  declareOnly(node, map.pcd_save_pcd_save_en);
  declareOnly(node, map.pcd_save_interval);
  declareOnly(node, imu.mapping_extrinsic_T);
  declareOnly(node, imu.mapping_extrinsic_R);
  declareOnly(node, map.prior_map_pcd);
  declareOnly(node, map.prior_map_pinned);
  declareOnly(node, map.prior_map_align);
  declareOnly(node, map.prior_map_voxel);
  declareOnly(node, map.prior_map_freeze);
  declareOnly(node, map.initial_pose);
  declareOnly(node, map.initial_pose_full_rpy_override);
  declareOnly(node, map.initial_pose_apply_roll_pitch);
  declareOnly(node, constraints.wheel_odom_en);
  declareOnly(node, constraints.wheel_topic);
  declareOnly(node, constraints.wheel_speed_scale);
  declareOnly(node, constraints.wheel_vel_noise_vx);
  declareOnly(node, constraints.wheel_vel_noise_vy);
  declareOnly(node, constraints.wheel_vel_noise_vz);
  declareOnly(node, constraints.wheel_extrinsic_R);
  declareOnly(node, constraints.wheel_cov_topic);
  declareOnly(node, constraints.wheel_slip_k);
  declareOnly(node, constraints.wheel_slip_phi0);
  declareOnly(node, constraints.wheel_slip_fx_window_s);
  declareOnly(node, constraints.wheel_lever_arm);
  declareOnly(node, constraints.wheel_lever_sigma);
}

void FrontendParameters::readOutputs(const rclcpp::Node & node,
                                     FrontendStartup & startup,
                                     bool & scan_pub_en,
                                     bool & dense_pub_en,
                                     std::string & odom_frame_id,
                                     std::string & body_frame_id)
{
  const OutputParameterSpecs output{};
  const MeasurementParameterSpecs measurement{};
  readInto(node, output.publish_path_en, values_.path_en);
  readInto(node, output.publish_effect_map_en, values_.effect_pub_en);
  readInto(node, output.publish_map_en, values_.map_pub_en);
  readInto(node, output.publish_scan_publish_en, scan_pub_en);
  readInto(node, output.publish_dense_publish_en, dense_pub_en);
  readInto(node, output.publish_scan_bodyframe_pub_en, values_.scan_body_pub_en);
  readInto(node, output.publish_odom_frame_id, odom_frame_id);
  readInto(node, output.publish_body_frame_id, body_frame_id);
  readInto(node, measurement.max_iteration, startup.num_max_iterations);
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("max_iteration", startup.num_max_iterations);
#endif
}

void FrontendParameters::readBacklog(const rclcpp::Node & node,
                                     int & max_buffered_scans,
                                     double & max_scan_backlog_sec,
                                     double & max_imu_backlog_sec)
{
  const InputParameterSpecs input{};
  readInto(node, input.max_buffered_scans, max_buffered_scans);
  readInto(node, input.max_scan_backlog_sec, max_scan_backlog_sec);
  readInto(node, input.max_imu_backlog_sec, max_imu_backlog_sec);
  if (!std::isfinite(max_imu_backlog_sec)) {
    RCLCPP_FATAL(
      node.get_logger(), "max_imu_backlog_sec must be finite (<= 0 disables the bound), got %f", max_imu_backlog_sec);
    throw std::invalid_argument("max_imu_backlog_sec must be finite");
  }
  if (!(max_imu_backlog_sec > 0.0)) {
    RCLCPP_INFO(node.get_logger(),
                "[IMU-BACKLOG] max_imu_backlog_sec is %.3f: the IMU queue is not bounded",
                max_imu_backlog_sec);
  }
}

void FrontendParameters::readDownsampleAndDegeneracy(const rclcpp::Node & node, bool & degeneracy_debug)
{
  const DownsampleParameterSpecs downsample{};
  const MeasurementParameterSpecs measurement{};
  readInto(node, downsample.adaptive_downsample_en, values_.adaptive_ds.enabled);
  readInto(node, downsample.adaptive_downsample_budget_sec, values_.adaptive_ds.budget_sec);
  readInto(node, downsample.adaptive_downsample_release_sec, values_.adaptive_ds.release_sec);
  readInto(node, downsample.adaptive_downsample_step, values_.adaptive_ds.step);
  readInto(node, downsample.adaptive_downsample_max_voxel, values_.adaptive_ds.max_voxel);
  readInto(node, downsample.adaptive_downsample_raise_after, values_.adaptive_ds.raise_after);
  readInto(node, downsample.adaptive_downsample_lower_after, values_.adaptive_ds.lower_after);
  readInto(node, measurement.degeneracy_debug, degeneracy_debug);
  readInto(node, measurement.degeneracy_spectrum_trace, values_.degeneracy_spectrum_trace);
  std::string floor_mode_name = "off";
  readInto(node, measurement.degeneracy_floor_mode, floor_mode_name);
  const std::optional<fast_lio::FloorMode> floor_mode = fast_lio::floorModeFromName(floor_mode_name);
  if (!floor_mode) {
    RCLCPP_FATAL(node.get_logger(),
                 "degeneracy_floor_mode must be 'off', 'observe' or 'apply', got '%s'",
                 floor_mode_name.c_str());
    throw std::invalid_argument("degeneracy_floor_mode");
  }
  values_.degeneracy_floor_mode = *floor_mode;
  fast_lio::FloorParams & floor_params = values_.degeneracy_floor;
  readInto(node, measurement.degeneracy_false_keep_rate, floor_params.alpha);
  readInto(node, measurement.degeneracy_floor_model_scale, floor_params.kappa);
  readInto(node, measurement.degeneracy_floor_rel_uncertainty, floor_params.rel_uncertainty);
  readInto(node, measurement.degeneracy_schur_ridge, values_.degeneracy_params.schur_ridge);
  if (!(floor_params.alpha > 0.0 && floor_params.alpha < 0.5)) {
    RCLCPP_FATAL(node.get_logger(), "degeneracy_false_keep_rate must lie in (0, 0.5), got %g", floor_params.alpha);
    throw std::invalid_argument("degeneracy_false_keep_rate");
  }
  if (!std::isfinite(floor_params.kappa) || !(floor_params.kappa > 0.0)) {
    RCLCPP_FATAL(node.get_logger(), "degeneracy_floor_model_scale must be finite and > 0, got %g", floor_params.kappa);
    throw std::invalid_argument("degeneracy_floor_model_scale");
  }
  if (!std::isfinite(floor_params.rel_uncertainty) || !(floor_params.rel_uncertainty >= 0.0)) {
    RCLCPP_FATAL(node.get_logger(),
                 "degeneracy_floor_rel_uncertainty must be finite and >= 0, got %g",
                 floor_params.rel_uncertainty);
    throw std::invalid_argument("degeneracy_floor_rel_uncertainty");
  }
  if (!std::isfinite(values_.degeneracy_params.schur_ridge)) {
    RCLCPP_FATAL(
      node.get_logger(), "degeneracy_schur_ridge must be finite, got %g", values_.degeneracy_params.schur_ridge);
    throw std::invalid_argument("degeneracy_schur_ridge");
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("degeneracy_floor_mode", floor_mode_name);
#endif
}

void FrontendParameters::readConstraints(const rclcpp::Node & node, bool & health_pub_en)
{
  const OutputParameterSpecs output{};
  const ConstraintParameterSpecs constraints{};
  const MeasurementParameterSpecs measurement{};
  readInto(node, output.health_pub_en, health_pub_en);
  readInto(node, constraints.planar_constraint_en, values_.planar_constraint_en);
  readInto(node, constraints.planar_constraint_noise, values_.planar_constraint_noise);
  readInto(node, constraints.planar_constraint_z_weak_thresh, values_.planar_constraint_z_weak_thresh);
  readInto(node, constraints.gravity_align_en, values_.gravity_align_en);
  readInto(node, constraints.gravity_align_noise, values_.gravity_align_noise);
  readInto(node, constraints.gravity_align_z_weak_thresh, values_.gravity_align_z_weak_thresh);
  readInto(node, constraints.gravity_align_accel_tol, values_.gravity_align_accel_tol);
  readInto(node, constraints.gravity_align_gyro_tol, values_.gravity_align_gyro_tol);
  readInto(node, constraints.gravity_align_window_s, values_.gravity_align_window_s);
  readInto(node, constraints.gravity_align_continuous, values_.gravity_align_continuous);
  readInto(node, constraints.gravity_align_hold_s, values_.gravity_align_hold_s);
  readInto(node, constraints.gravity_align_noise_degraded, values_.gravity_align_noise_degraded);
  readInto(node, constraints.gravity_align_mag_ref, values_.gravity_align_mag_ref);
  readInto(node, constraints.gravity_align_grav_leak, values_.gravity_align_grav_leak);
  readInto(node, constraints.gravity_align_grav_cap_deg, values_.gravity_align_grav_cap_deg);
  readInto(node, constraints.gravity_align_grav_freeze_degraded, values_.gravity_align_grav_freeze_degraded);
  readInto(node, measurement.lidar_attitude_hold_en, values_.attitude_hold.enabled);
}

void FrontendParameters::readCorrespondence(const rclcpp::Node & node)
{
  const MeasurementParameterSpecs measurement{};
  readInto(node, measurement.corr_innov_gate_k, values_.corr_gate.innovation_k);
  readInto(node, measurement.corr_nn0_max, values_.corr_gate.nn0_max);
  readInto(node, measurement.corr_gate_mad_scale, values_.corr_gate.mad_scale);
  readInto(node, measurement.corr_gate_rot_sigma_floor_deg, values_.corr_gate.rot_sigma_floor_deg);
  std::string corr_kernel_str = "none";
  readInto(node, measurement.corr_kernel, corr_kernel_str);
  readInto(node, measurement.corr_kernel_mad_scale, values_.corr_kernel_mad_scale);
  readInto(node, measurement.corr_kernel_min_scale, values_.corr_kernel_min_scale);
  values_.corr_kernel = corr_kernel_str == "gm" ? 1 : corr_kernel_str == "cauchy" ? 2 : 0;
  if (corr_kernel_str != "none" && values_.corr_kernel == 0) {
    RCLCPP_WARN(node.get_logger(), "corr_kernel '%s' unknown (none|gm|cauchy): kernel off", corr_kernel_str.c_str());
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("corr_kernel", corr_kernel_str);
#endif
}

void FrontendParameters::readCorrespondenceDump(const rclcpp::Node & node)
{
  const DiagnosticsParameterSpecs diagnostics{};
  readInto(node, diagnostics.corr_dump_en, values_.corr_dump_en);
  std::string corr_dump_windows_str;
  readInto(node, diagnostics.corr_dump_windows, corr_dump_windows_str);
  if (values_.corr_dump_en) {
    std::stringstream ws(corr_dump_windows_str);
    std::string item;
    while (std::getline(ws, item, ',')) {
      const auto colon = item.find(':');
      if (colon == std::string::npos) {
        continue;
      }
      values_.corr_dump_windows.emplace_back(std::stod(item.substr(0, colon)), std::stod(item.substr(colon + 1)));
    }
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("corr_dump_windows", corr_dump_windows_str);
#endif
}

void FrontendParameters::readAttitudeAndGravity(const rclcpp::Node & node, double & zupt_vel_reset_sigma)
{
  const MeasurementParameterSpecs measurement{};
  const RecoveryParameterSpecs recovery{};
  const ConstraintParameterSpecs constraints{};
  readInto(node, measurement.lidar_attitude_hold_far_frac, values_.attitude_hold.far_frac_max);
  readInto(node, measurement.lidar_attitude_hold_min_effct, values_.attitude_hold.min_effct);
  readInto(node, measurement.lidar_attitude_hold_max_s, values_.attitude_hold.max_hold_s);
  readInto(node, measurement.lidar_attitude_hold_max_scan_gap_s, values_.attitude_hold.max_scan_gap_s);
  readInto(node, measurement.lidar_attitude_hold_min_rp_info, values_.lidar_attitude_hold_min_rp_info);
  readInto(node, recovery.guard_vel_reset_sigma, values_.guard_vel_reset_sigma);
  readInto(node, recovery.zupt_vel_reset_sigma, zupt_vel_reset_sigma);
  readInto(node, measurement.lidar_attitude_hold_z_weak, values_.attitude_hold.z_weak_min);
  readInto(node, constraints.gravity_align_lin_accel_comp, values_.gravity_align_lin_accel_comp);
  readInto(node, constraints.gravity_align_vel_sigma_scale, values_.gravity_align_vel_sigma_scale);
  readInto(node, constraints.gravity_align_lin_accel_source, values_.gravity_align_lin_accel_source);
  readInto(node, constraints.gravity_align_withhold_blind_moving, values_.gravity_align_withhold_blind_moving);
  readInto(node, constraints.gravity_align_sigma_policy, values_.gravity_align_sigma_policy);
  readInto(node, constraints.gravity_align_noise_static, values_.gravity_align_noise_static);
  readInto(node, constraints.gravity_align_noise_moving, values_.gravity_align_noise_moving);
  if (values_.gravity_align_sigma_policy != "scene" && values_.gravity_align_sigma_policy != "measurement") {
    RCLCPP_FATAL(node.get_logger(),
                 "gravity_align_sigma_policy must be 'scene' or 'measurement', got '%s'",
                 values_.gravity_align_sigma_policy.c_str());
    throw std::invalid_argument("gravity_align_sigma_policy");
  }
  readInto(node, constraints.gravity_align_leg_vel_sigma, values_.gravity_align_leg_vel_sigma);
  readInto(node, constraints.gravity_align_leg_edge_window_s, values_.gravity_align_leg_edge_window_s);
  std::vector<double> gravity_align_leg_lever_arm_vec;
  readInto(node, constraints.gravity_align_leg_lever_arm, gravity_align_leg_lever_arm_vec);
  if (gravity_align_leg_lever_arm_vec.size() != 3u) {
    RCLCPP_FATAL(node.get_logger(),
                 "gravity_align_leg_lever_arm must have exactly 3 elements, got %zu",
                 gravity_align_leg_lever_arm_vec.size());
    throw std::invalid_argument("gravity_align_leg_lever_arm size mismatch");
  }
  values_.gravity_align_leg_lever_arm = Eigen::Vector3d(
    gravity_align_leg_lever_arm_vec[0], gravity_align_leg_lever_arm_vec[1], gravity_align_leg_lever_arm_vec[2]);
  if (values_.gravity_align_lin_accel_source != "leg" && values_.gravity_align_lin_accel_source != "filter") {
    RCLCPP_FATAL(node.get_logger(),
                 "gravity_align_lin_accel_source must be 'leg' or 'filter', got '%s'",
                 values_.gravity_align_lin_accel_source.c_str());
    throw std::invalid_argument("gravity_align_lin_accel_source");
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("gravity_align_leg_lever_arm", gravity_align_leg_lever_arm_vec);
#endif
}

void FrontendParameters::readImuInit(const rclcpp::Node & node, FrontendStartup & startup)
{
  const ImuParameterSpecs imu{};
  readInto(node, imu.imu_init_require_still, startup.imu_init_require_still);
  readInto(node, imu.imu_init_still_tol, startup.imu_init_still_tol);
  readInto(node, imu.imu_init_still_timeout_s, startup.imu_init_still_timeout_s);
  readInto(node, imu.imu_init_still_sustained_en, startup.imu_init_static_params.enabled);
  readInto(node, imu.imu_init_still_min_duration_s, startup.imu_init_static_params.minDurationS);
  readInto(node, imu.imu_init_still_min_samples, startup.imu_init_static_params.minSamples);
  readInto(node, imu.imu_init_still_gyro_rms_max, startup.imu_init_static_params.gyroRmsMax);
  readInto(node, imu.imu_init_still_gravity_ref, startup.imu_init_static_params.gravityRef);
  readInto(node, imu.imu_init_still_gravity_tol, startup.imu_init_static_params.gravityTol);
  readInto(node, imu.imu_init_still_max_gap_s, startup.imu_init_static_params.maxGapS);
  if (!fast_lio::validImuStaticInitParams(startup.imu_init_static_params))
    throw std::invalid_argument("imu_init_still sustained-window limits must be finite and positive");
  if (fast_lio::imuStaticInitActive(startup.imu_init_require_still, startup.imu_init_static_params) &&
      (!std::isfinite(startup.imu_init_still_tol) || startup.imu_init_still_tol <= 0.0 ||
       !std::isfinite(startup.imu_init_still_timeout_s) || startup.imu_init_still_timeout_s <= 0.0))
    throw std::invalid_argument("imu_init_still_tol and imu_init_still_timeout_s must be finite and positive");
  if (startup.imu_init_static_params.enabled && !startup.imu_init_require_still)
    RCLCPP_WARN(node.get_logger(), "imu_init_still_sustained_en requires imu_init_require_still; gate is inactive");
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("imu_init_require_still", startup.imu_init_require_still);
  locals_for_test_.value("imu_init_still_tol", startup.imu_init_still_tol);
  locals_for_test_.value("imu_init_still_timeout_s", startup.imu_init_still_timeout_s);
  locals_for_test_.value("imu_init_still_sustained_en", startup.imu_init_static_params.enabled);
  locals_for_test_.value("imu_init_still_min_duration_s", startup.imu_init_static_params.minDurationS);
  locals_for_test_.value("imu_init_still_min_samples", startup.imu_init_static_params.minSamples);
  locals_for_test_.value("imu_init_still_gyro_rms_max", startup.imu_init_static_params.gyroRmsMax);
  locals_for_test_.value("imu_init_still_gravity_ref", startup.imu_init_static_params.gravityRef);
  locals_for_test_.value("imu_init_still_gravity_tol", startup.imu_init_static_params.gravityTol);
  locals_for_test_.value("imu_init_still_max_gap_s", startup.imu_init_static_params.maxGapS);
#endif
}

void FrontendParameters::readGuards(const rclcpp::Node & node, ReleaseVelocityParams & release_vel_params)
{
  const RecoveryParameterSpecs recovery{};
  const MeasurementParameterSpecs measurement{};
  readInto(node, recovery.divergence_guard_en, values_.divergence_guard_en);
  readInto(node, recovery.divergence_guard_min_eff, values_.divergence_guard_min_eff);
  readInto(node, recovery.guard_engulf_en, values_.guard_engulf_en);
  readInto(node, recovery.guard_engulf_far_range, values_.guard_engulf_far_range);
  readInto(node, recovery.guard_engulf_far_frac_min, values_.guard_engulf_far_frac_min);
  readInto(node, recovery.guard_engulf_streak, values_.guard_engulf_streak);
  readInto(node, recovery.guard_engulf_far_frac_release, values_.guard_engulf_far_frac_release);
  readInto(node, recovery.guard_engulf_release_scans, values_.guard_engulf_release_scans);
  {
    std::string release_source;
    readInto(node, recovery.guard_release_vel_source, release_source);
    release_vel_params.source = release_source == "leg"    ? fast_lio::ReleaseVelocitySource::kLeg
                                : release_source == "keep" ? fast_lio::ReleaseVelocitySource::kKeep
                                                           : fast_lio::ReleaseVelocitySource::kZero;
    if (release_source != "leg" && release_source != "keep" && release_source != "zero") {
      RCLCPP_WARN(node.get_logger(), "guard_release_vel_source '%s' unknown: using 'zero'", release_source.c_str());
    }
#ifdef FASTLIO_TEST_HOOKS
    locals_for_test_.value("guard_release_vel_source", release_source);
#endif
  }
  readInto(node, recovery.guard_release_leg_max_age_s, release_vel_params.leg_max_age_s);
  readInto(node, recovery.guard_release_leg_sigma, release_vel_params.leg_sigma);
  readInto(node, recovery.guard_release_keep_sigma, release_vel_params.keep_sigma);
  readInto(node, recovery.guard_release_inflate_en, values_.release_inflate.enabled);
  readInto(node, recovery.guard_release_pos_sigma_min, values_.release_inflate.pos_sigma_min);
  readInto(node, recovery.guard_release_pos_sigma_rate, values_.release_inflate.pos_sigma_rate);
  readInto(node, recovery.guard_release_pos_sigma_max, values_.release_inflate.pos_sigma_max);
  readInto(node, recovery.guard_release_rp_sigma_deg, values_.release_inflate.rp_sigma_min_deg);
  readInto(node, recovery.guard_release_rp_sigma_rate_deg, values_.release_inflate.rp_sigma_rate_deg);
  readInto(node, recovery.guard_release_rp_sigma_max_deg, values_.release_inflate.rp_sigma_max_deg);
  readInto(node, recovery.guard_release_yaw_sigma_deg, values_.release_inflate.yaw_sigma_deg);
  readInto(node, measurement.lidar_consider_rp_en, values_.rp_consider.enabled);
  readInto(node, measurement.lidar_consider_rp_while_engulfed, values_.rp_consider.while_engulfed);
  readInto(node, measurement.lidar_consider_rp_post_release_s, values_.rp_consider.post_release_s);
  readInto(node, measurement.lidar_consider_rp_min_info, values_.rp_consider.min_rp_info);
  readInto(node, measurement.lidar_consider_rp_z_weak_min, values_.rp_consider.z_weak_min);
  readInto(node, measurement.lidar_consider_rp_max_s, values_.rp_consider.max_consider_s);
  if (values_.rp_consider.enabled) {
    RCLCPP_INFO(node.get_logger(),
                "[RP-CONSIDER] lidar update considers roll/pitch while engulfed=%d, %.1fs after a release, when the "
                "previous scan's roll/pitch information < %.0f or z_weak >= %.2f (cap %.1fs)",
                values_.rp_consider.while_engulfed,
                values_.rp_consider.post_release_s,
                values_.rp_consider.min_rp_info,
                values_.rp_consider.z_weak_min,
                values_.rp_consider.max_consider_s);
  }
}

void FrontendParameters::readHoleAndZupt(const rclcpp::Node & node,
                                         FrontendStartup & startup,
                                         double & zupt_gyro_thresh,
                                         double & zupt_acc_span_ratio,
                                         double & zupt_wheel_speed_thresh,
                                         double & zupt_wheel_max_age,
                                         bool & zupt_require_wheel,
                                         bool & zupt_anchor_trace_en,
                                         StationaryAnchorParams & stationary_anchor_params)
{
  const RecoveryParameterSpecs recovery{};
  readInto(node, recovery.guard_hole_en, values_.guard_hole_en);
  readInto(node, recovery.guard_hole_min_sec, values_.guard_hole_min_sec);
  readInto(node, recovery.guard_hole_max_accel, values_.guard_hole_max_accel);
  readInto(node, recovery.guard_hole_reanchor_pos, values_.guard_hole_reanchor_pos);
  readInto(node, recovery.divergence_guard_streak, values_.divergence_guard_streak);
  readInto(node, recovery.zupt_en, values_.zupt_en);
  readInto(node, recovery.zupt_gyro_thresh, zupt_gyro_thresh);
  readInto(node, recovery.zupt_acc_span_ratio, zupt_acc_span_ratio);
  readInto(node, recovery.zupt_wheel_speed_thresh, zupt_wheel_speed_thresh);
  readInto(node, recovery.zupt_wheel_max_age, zupt_wheel_max_age);
  readInto(node, recovery.zupt_require_wheel, zupt_require_wheel);
  readInto(node, recovery.zupt_anchor_trace_en, zupt_anchor_trace_en);
  readInto(node, recovery.zupt_parked_hold_en, values_.parked_hold.enabled);
  readInto(node, recovery.zupt_parked_hold_scans, values_.parked_hold.engage_scans);
  readInto(node, recovery.zupt_hold_pos_noise, values_.parked_hold.pos_noise);
  readInto(node, recovery.zupt_hold_rot_noise, values_.parked_hold.rot_noise);
  readInto(node, recovery.zupt_hold_vel_noise, values_.parked_hold.vel_noise);
  readInto(node, recovery.zupt_hold_deadband_sigma, values_.parked_hold.deadband_sigma);
  readInto(node, recovery.zupt_anchor_confirm_scans, stationary_anchor_params.confirm_scans);
  readInto(node, recovery.zupt_anchor_max_scan_gap_s, startup.zupt_anchor_max_scan_gap_s);
}

void FrontendParameters::readReanchor(const rclcpp::Node & node,
                                      FrontendStartup & startup,
                                      ReanchorParams & reanchor_params,
                                      RecoveryParams & recovery_params)
{
  const RecoveryParameterSpecs recovery{};
  readInto(node, recovery.reanchor_en, reanchor_params.enabled);
  readInto(node, recovery.reanchor_min_eff, reanchor_params.min_eff);
  readInto(node, recovery.reanchor_max_res, reanchor_params.max_res);
  readInto(node, recovery.reanchor_confirm_scans, reanchor_params.confirm_scans);
  readInto(node, recovery.reanchor_timeout_sec, reanchor_params.timeout_sec);
  readInto(node, recovery.reanchor_on_fail, values_.reanchor_on_fail);
  readInto(node, recovery.reanchor_max_res_ratio, reanchor_params.max_res_ratio);
  readInto(node, recovery.reanchor_res_floor, reanchor_params.res_floor);
  readInto(node, recovery.reanchor_min_eff_ratio, reanchor_params.min_eff_ratio);
  readInto(node, recovery.reanchor_require_guard_clear, values_.reanchor_require_guard_clear);
  readInto(node, recovery.reanchor_healthy_window, startup.reanchor_healthy_window);
  if (!std::isfinite(reanchor_params.max_res_ratio) || reanchor_params.max_res_ratio < 0.0 ||
      !std::isfinite(reanchor_params.res_floor) || reanchor_params.res_floor < 0.0 ||
      !std::isfinite(reanchor_params.min_eff_ratio) || reanchor_params.min_eff_ratio < 0.0 ||
      startup.reanchor_healthy_window < 1) {
    RCLCPP_FATAL(node.get_logger(),
                 "reanchor_max_res_ratio / reanchor_res_floor / reanchor_min_eff_ratio must be finite and "
                 "nonnegative and reanchor_healthy_window at least 1");
    throw std::invalid_argument("reanchor relative verification parameters");
  }
  double reanchor_recovery_stable_sec = 1.0;
  readInto(node, recovery.reanchor_total_timeout_sec, startup.reanchor_total_timeout_sec);
  readInto(node, recovery.reanchor_recovery_stable_sec, reanchor_recovery_stable_sec);
  if (!fast_lio::resolveRecoveryParams(
        startup.reanchor_total_timeout_sec, reanchor_recovery_stable_sec, &recovery_params)) {
    RCLCPP_FATAL(node.get_logger(),
                 "reanchor_total_timeout_sec must be finite and nonnegative (0 = off) and "
                 "reanchor_recovery_stable_sec finite, positive and, with the total on, shorter than it");
    throw std::invalid_argument("reanchor recovery budget parameters");
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("reanchor_healthy_window", startup.reanchor_healthy_window);
  locals_for_test_.value("reanchor_total_timeout_sec", startup.reanchor_total_timeout_sec);
  locals_for_test_.value("reanchor_recovery_stable_sec", reanchor_recovery_stable_sec);
#endif
}

void FrontendParameters::loadLostRecovery(const rclcpp::Node & node,
                                          bool health_pub_en,
                                          double total_s,
                                          LostRecoveryParams & lost_recovery_params)
{
  const RecoveryParameterSpecs recovery{};
  const bool enabled = node.get_parameter(recovery.lost_recovery_en.name).as_bool();
  const double stable_s = node.get_parameter(recovery.lost_recovery_stable_sec.name).as_double();
  if (enabled && !health_pub_en) {
    throw std::invalid_argument(
      "lost_recovery_en=true requires health_pub_en=true "
      "(the coordinate epoch is published in the /Odometry health block)");
  }
  if (!fast_lio::resolveLostRecoveryParams(enabled, stable_s, total_s, &lost_recovery_params)) {
    throw std::invalid_argument(
      "lost_recovery_stable_sec must be finite and positive (representable in ns); "
      "lost_recovery_en=true requires reanchor_total_timeout_sec > 0");
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("lost_recovery_en", enabled);
  locals_for_test_.value("lost_recovery_stable_sec", stable_s);
#endif
}

void FrontendParameters::loadRecoveryEntryView(const rclcpp::Node & node)
{
  const RecoveryParameterSpecs recovery{};
  readInto(node, recovery.reanchor_recovery_entry_view, values_.reanchor_recovery_entry_view);
#ifndef USE_IVOX
  if (values_.reanchor_recovery_entry_view) {
    RCLCPP_FATAL(node.get_logger(), "reanchor_recovery_entry_view needs the iVox map backend (USE_IVOX)");
    throw std::invalid_argument("reanchor_recovery_entry_view needs the iVox map backend");
  }
#endif
}

void FrontendParameters::readRunaway(const rclcpp::Node & node,
                                     double zupt_wheel_max_age,
                                     double zupt_wheel_speed_thresh)
{
  const RecoveryParameterSpecs recovery{};
  readInto(node, recovery.runaway_watchdog_en, values_.runaway.enabled);
  readInto(node, recovery.runaway_parked_speed_thresh, values_.runaway.parked_speed_thresh);
  readInto(node, recovery.runaway_wheel_speed_margin, values_.runaway.wheel_speed_margin);
  readInto(node, recovery.runaway_hold_sec, values_.runaway.hold_sec);
  readInto(node, recovery.runaway_on_trip, values_.runaway_on_trip);
  readInto(node, recovery.input_epoch_on_fault, values_.input_epoch_on_fault);
  if (values_.input_epoch_on_fault != "hold" && values_.input_epoch_on_fault != "fault_exit") {
    RCLCPP_FATAL(node.get_logger(),
                 "input_epoch_on_fault must be 'hold' or 'fault_exit', got '%s'",
                 values_.input_epoch_on_fault.c_str());
    throw std::invalid_argument("input_epoch_on_fault");
  }
  values_.runaway.wheel_max_age = zupt_wheel_max_age;
  // 轮速要先自己报出运动,才有资格指认跑飞:恒 0 的速度源(m20 的 /lio/twist 是 GNSS
  // 代理,拒止段整段读 0)否则会把正常行走判成跑飞。
  values_.runaway.wheel_moving_thresh = zupt_wheel_speed_thresh;
  readInto(node, recovery.divergence_guard_max_speed, values_.divergence_guard_max_speed);
}

void FrontendParameters::readCanopyAndGround(const rclcpp::Node & node)
{
  const MeasurementParameterSpecs measurement{};
  readInto(node, measurement.canopy_near_range, values_.canopy_near_range);
  readInto(node, measurement.canopy_ground_z, values_.canopy_ground_z);
  readInto(node, measurement.canopy_ground_halfband, values_.canopy_ground_halfband);
  readInto(node, measurement.ground_recover_en, values_.ground_recover_en);
  readInto(node, measurement.ground_recover_scan_band, values_.ground_recover_scan_band);
  readInto(node, measurement.ground_recover_min_inliers, values_.ground_recover_min_inliers);
  readInto(node, measurement.ground_recover_max_residual, values_.ground_recover_max_residual);
  readInto(node, measurement.ground_recover_sector_tol, values_.ground_recover_sector_tol);
  readInto(node, measurement.ground_recover_max_rows, values_.ground_recover_max_rows);
}

void FrontendParameters::readDataPaths(const rclcpp::Node & node, std::string & map_file_path, std::string & root_dir)
{
  const MapParameterSpecs map{};
  const DiagnosticsParameterSpecs diagnostics{};
  readInto(node, map.map_file_path, map_file_path);
  std::string data_dir_param;
  readInto(node, diagnostics.data_dir, data_dir_param);
  if (!data_dir_param.empty()) {
    root_dir = data_dir_param;
    if (root_dir.back() != '/')
      root_dir += '/';
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("data_dir", data_dir_param);
#endif
}

void FrontendParameters::readInputs(const rclcpp::Node & node,
                                    FrontendStartup & startup,
                                    bool & time_sync_en,
                                    double & time_diff_lidar_to_imu,
                                    double & imu_time_offset)
{
  const InputParameterSpecs input{};
  readInto(node, input.common_lid_topic, startup.lid_topic);
  readInto(node, input.common_imu_topic, startup.imu_topic);
  readInto(node, input.common_imu_input_type, startup.imu_input_type);
  readInto(node, input.common_imu_bias_topic, startup.imu_bias_topic);
  readInto(node, input.common_time_sync_en, time_sync_en);
  readInto(node, input.common_time_offset_lidar_to_imu, time_diff_lidar_to_imu);
  readInto(node, input.common_imu_time_offset, imu_time_offset);
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("common.lid_topic", startup.lid_topic);
  locals_for_test_.value("common.imu_topic", startup.imu_topic);
  locals_for_test_.value("common.imu_input_type", startup.imu_input_type);
  locals_for_test_.value("common.imu_bias_topic", startup.imu_bias_topic);
#endif
}

void FrontendParameters::readMapAndNoise(const rclcpp::Node & node,
                                         FrontendStartup & startup,
                                         double & filter_size_map_min,
                                         double & cube_len,
                                         float & det_range)
{
  const MapParameterSpecs map{};
  const ImuParameterSpecs imu{};
  double filter_size_corner_min = 0;
  readInto(node, map.filter_size_corner, filter_size_corner_min);
  readInto(node, map.filter_size_surf, values_.filter_size_surf_min);
  readInto(node, map.filter_size_map, filter_size_map_min);
  readInto(node, map.cube_side_length, cube_len);
  readInto(node, map.mapping_det_range, det_range);
  readInto(node, map.mapping_fov_degree, startup.fov_deg);
  readInto(node, imu.mapping_gyr_cov, startup.gyr_cov);
  readInto(node, imu.mapping_acc_cov, startup.acc_cov);
  readInto(node, imu.mapping_b_gyr_cov, startup.b_gyr_cov);
  readInto(node, imu.mapping_b_acc_cov, startup.b_acc_cov);
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("filter_size_corner", filter_size_corner_min);
  locals_for_test_.value("mapping.fov_degree", startup.fov_deg);
  locals_for_test_.value("mapping.gyr_cov", startup.gyr_cov);
  locals_for_test_.value("mapping.acc_cov", startup.acc_cov);
  locals_for_test_.value("mapping.b_gyr_cov", startup.b_gyr_cov);
  locals_for_test_.value("mapping.b_acc_cov", startup.b_acc_cov);
#endif
}

void FrontendParameters::readPreprocess(
  const rclcpp::Node & node, double & blind, int & lidar_type, int & scan_lines, int & time_unit, int & scan_rate)
{
  const InputParameterSpecs input{};
  readInto(node, input.preprocess_blind, blind);
  readInto(node, input.preprocess_lidar_type, lidar_type);
  readInto(node, input.preprocess_scan_line, scan_lines);
  readInto(node, input.preprocess_timestamp_unit, time_unit);
  readInto(node, input.preprocess_scan_rate, scan_rate);
}

void FrontendParameters::resolveAnchorGap(const rclcpp::Node & node,
                                          const FrontendStartup & startup,
                                          int scan_rate,
                                          StationaryAnchorParams & stationary_anchor_params,
                                          const RecoveryParams & recovery_params)
{
  // The stationary anchor ends at any gap in the scan evidence longer than this bound (0 = three scan periods).
  if (!fast_lio::anchorConfirmScansValid(stationary_anchor_params.confirm_scans) ||
      !fast_lio::resolveAnchorScanGapNs(startup.zupt_anchor_max_scan_gap_s,
                                        static_cast<double>(scan_rate),
                                        &stationary_anchor_params.max_scan_gap_ns)) {
    RCLCPP_FATAL(node.get_logger(),
                 "zupt_anchor_confirm_scans must be at least 1 and zupt_anchor_max_scan_gap_s finite and positive, "
                 "or 0 with a finite positive preprocess.scan_rate");
    throw std::invalid_argument("stationary anchor parameters");
  }
  RCLCPP_INFO(node.get_logger(),
              "stationary anchor: confirm_scans=%d max_scan_gap_ns=%lld; recovery budget: total_ns=%lld "
              "stable_ns=%lld (%s)",
              stationary_anchor_params.confirm_scans,
              static_cast<long long>(stationary_anchor_params.max_scan_gap_ns),
              static_cast<long long>(recovery_params.total_ns),
              static_cast<long long>(recovery_params.stable_ns),
              recovery_params.enabled() ? "platform policy on" : "platform policy off");
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("zupt_anchor_max_scan_gap_s", startup.zupt_anchor_max_scan_gap_s);
#endif
}

void FrontendParameters::readImuCoverage(const rclcpp::Node & node,
                                         double & max_scan_duration_s,
                                         ImuCoverageParams & coverage,
                                         ImuGapPriorParams & gap,
                                         bool reanchor_en)
{
  const InputParameterSpecs input{};
  const ImuParameterSpecs imu{};
  readInto(node, input.preprocess_max_scan_duration_s, max_scan_duration_s);
  readInto(node, imu.imu_coverage_max_gap_s, coverage.max_gap_s);
  readInto(node, imu.imu_coverage_max_extrapolation_s, coverage.max_extrapolation_s);
  if (!std::isfinite(coverage.max_gap_s) || coverage.max_gap_s < 0.0 || !std::isfinite(coverage.max_extrapolation_s) ||
      coverage.max_extrapolation_s < 0.0)
    throw std::invalid_argument("imu_coverage limits must be finite and nonnegative");
  if ((coverage.max_gap_s > 0.0 || coverage.max_extrapolation_s > 0.0) && !reanchor_en)
    throw std::invalid_argument("imu_coverage requires reanchor_en for recovery against the frozen map");
  {
    readInto(node, imu.imu_gap_prior_en, gap.enabled);
    readInto(node, imu.imu_gap_min_s, gap.min_gap_s);
    readInto(node, imu.imu_gap_accel_max, gap.accel_max);
    readInto(node, imu.imu_gap_accel_max_z, gap.accel_max_z);
    double rate_rp_deg = 0.0;
    double rate_yaw_deg = 10.0;
    readInto(node, imu.imu_gap_rate_rp_max, rate_rp_deg);
    readInto(node, imu.imu_gap_rate_yaw_max, rate_yaw_deg);
    gap.rate_rp_max = rate_rp_deg * fast_lio::kImuGapDegToRad;
    gap.rate_yaw_max = rate_yaw_deg * fast_lio::kImuGapDegToRad;
    readInto(node, imu.imu_gap_verify_s, values_.imu_gap_verify_s);
    readInto(node, imu.imu_gap_empty_scans, gap.empty_scans);
    if (gap.enabled && (!fast_lio::imuGapPriorParamsValid(gap) || !std::isfinite(values_.imu_gap_verify_s) ||
                        values_.imu_gap_verify_s < 0.0)) {
      RCLCPP_FATAL(node.get_logger(),
                   "imu_gap_min_s must be positive and imu_gap_accel_max(_z), imu_gap_rate_*_max and "
                   "imu_gap_verify_s finite and nonnegative");
      throw std::invalid_argument("imu_gap_* parameters");
    }
    if (gap.enabled) {
      RCLCPP_INFO(node.get_logger(),
                  "[IMU-GAP] prior on: IMU intervals > %.3fs up to imu_coverage.max_gap_s %.2fs are bridged "
                  "(gyro average, velocity held; 1-sigma accel %.2f / %.2f m/s^2 horizontal / vertical, rate "
                  "roll/pitch %.1f yaw %.1f deg/s), scans without IMU samples %s; gaps > %.2fs are re-verified "
                  "against the frozen map",
                  gap.min_gap_s,
                  coverage.max_gap_s,
                  gap.accel_max,
                  gap.accel_max_z,
                  rate_rp_deg,
                  rate_yaw_deg,
                  gap.empty_scans ? "carried" : "dropped",
                  values_.imu_gap_verify_s);
      if (!(coverage.max_gap_s > 0.0)) {
        RCLCPP_WARN(node.get_logger(),
                    "[IMU-GAP] imu_coverage.max_gap_s is 0: gaps of any length are bridged by dead reckoning");
      }
    }
#ifdef FASTLIO_TEST_HOOKS
    locals_for_test_.value("imu_gap_rate_rp_max", rate_rp_deg);
    locals_for_test_.value("imu_gap_rate_yaw_max", rate_yaw_deg);
#endif
  }
  if (!std::isfinite(max_scan_duration_s) || max_scan_duration_s < 0.0)
    throw std::invalid_argument("preprocess.max_scan_duration_s must be finite and nonnegative");
}

void FrontendParameters::readFeatureInputs(const rclcpp::Node & node,
                                           int & point_filter_num,
                                           bool & feature_enabled,
                                           bool & runtime_pos_log)
{
  const InputParameterSpecs input{};
  const MeasurementParameterSpecs measurement{};
  const DiagnosticsParameterSpecs diagnostics{};
  readInto(node, input.point_filter_num, point_filter_num);
  readInto(node, measurement.nn_refresh_gate_en, values_.nn_refresh_gate_en);
  readInto(node, measurement.nn_refresh_min_shift, values_.nn_refresh_min_shift);
  readInto(node, measurement.nn_refresh_ref_range, values_.nn_refresh_ref_range);
  readInto(node, input.feature_extract_enable, feature_enabled);
  readInto(node, diagnostics.runtime_pos_log_enable, runtime_pos_log);
}

#ifdef USE_IVOX
IvoxDeclaredValues FrontendParameters::declareIvox(rclcpp::Node & node)
{
  const MapParameterSpecs map{};
  const double ivox_res = declareValue(node, map.ivox_grid_resolution);
  const int ivox_nearby = declareValue(node, map.ivox_nearby_type);
  const int ivox_vcap = declareValue(node, map.ivox_voxel_capacity);
  const int ivox_maxvox = declareValue(node, map.ivox_max_voxels);
  if (ivox_maxvox < 0) {
    RCLCPP_FATAL(node.get_logger(), "ivox_max_voxels must be nonnegative (0 uses 5000000); got %d", ivox_maxvox);
    throw std::invalid_argument("ivox_max_voxels must be nonnegative");
  }
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("ivox_grid_resolution", ivox_res);
  locals_for_test_.value("ivox_nearby_type", ivox_nearby);
  locals_for_test_.value("ivox_voxel_capacity", ivox_vcap);
  locals_for_test_.value("ivox_max_voxels", ivox_maxvox);
#endif
  return IvoxDeclaredValues{ivox_res, ivox_nearby, ivox_vcap, ivox_maxvox};
}
#endif

void FrontendParameters::readDiagnosticsAndMapSave(const rclcpp::Node & node,
                                                   FrontendStartup & startup,
                                                   bool & latency_diag_en,
                                                   double & latency_log_period_sec,
                                                   int & map_insert_delay_scans,
                                                   bool & pcd_save_en,
                                                   int & pcd_save_interval)
{
  const DiagnosticsParameterSpecs diagnostics{};
  const ImuParameterSpecs imu{};
  const MapParameterSpecs map{};
  readInto(node, diagnostics.diagnostics_latency_enable, latency_diag_en);
  readInto(node, diagnostics.diagnostics_latency_log_period_sec, latency_log_period_sec);
  readInto(node, diagnostics.diagnostics_kalman_channel_enable, values_.kalman_channel_diag_en);
  values_.time_admission_counts_enable =
    node.get_parameter(diagnostics.diagnostics_time_admission_counts_enable.name).as_bool();
  readInto(node, imu.mapping_extrinsic_est_en, values_.extrinsic_est_en);
  if (values_.attitude_hold.enabled && values_.extrinsic_est_en) {
    RCLCPP_FATAL(node.get_logger(),
                 "lidar_attitude_hold_en and mapping.extrinsic_est_en cannot both be true: the hold projects only the "
                 "attitude columns of H onto yaw, so the extrinsic rotation columns would let roll and pitch "
                 "information into the state");
    throw std::invalid_argument(
      "lidar_attitude_hold_en=true requires mapping.extrinsic_est_en=false "
      "(the hold does not project the extrinsic rotation columns of H)");
  }
  readInto(node, map.mapping_map_insert_delay_scans, map_insert_delay_scans);
  if (map_insert_delay_scans > 0) {
    RCLCPP_WARN(node.get_logger(), "[MAP] deterministic insertion lag: %d scan(s)", map_insert_delay_scans);
  }
  readInto(node, map.pcd_save_pcd_save_en, pcd_save_en);
  readInto(node, map.pcd_save_interval, pcd_save_interval);
  readInto(node, imu.mapping_extrinsic_T, startup.extrinT);
  readInto(node, imu.mapping_extrinsic_R, startup.extrinR);
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("mapping.extrinsic_T", startup.extrinT);
  locals_for_test_.value("mapping.extrinsic_R", startup.extrinR);
#endif
}

void FrontendParameters::readPriorMapPath(const rclcpp::Node & node, std::string & prior_map_pcd)
{
  const MapParameterSpecs map{};
  readInto(node, map.prior_map_pcd, prior_map_pcd);
}

void FrontendParameters::readPriorMapAndPose(const rclcpp::Node & node)
{
  const MapParameterSpecs map{};
  readInto(node, map.prior_map_pinned, values_.prior_map_pinned);
  readInto(node, map.prior_map_voxel, values_.prior_map_voxel);
  readInto(node, map.prior_map_freeze, values_.prior_map_freeze);
  {
    std::string align_name;
    readInto(node, map.prior_map_align, align_name);
    if (!fast_lio::parsePriorMapAlign(align_name, &values_.prior_map_align)) {
      RCLCPP_ERROR(node.get_logger(),
                   "prior_map_align='%s' is not a known value (as_delivered | relocalization) - "
                   "loading the map as delivered",
                   align_name.c_str());
    }
    if (values_.prior_map_align == fast_lio::PriorMapAlign::kRelocalization) {
#ifdef USE_IVOX
      // The map has to be replaceable as a unit when a better alignment arrives, which is
      // what the pinned tier is: forcing it here rather than asking the operator to set two
      // parameters that only work together.
      values_.prior_map_pinned = true;
      RCLCPP_INFO(node.get_logger(),
                  "prior_map_align=relocalization: the map waits for /relocalization/result and is "
                  "installed transformed into this run's frame (pinned tier, replaced on realignment)");
#else
      values_.prior_map_align = fast_lio::PriorMapAlign::kAsDelivered;
      RCLCPP_ERROR(node.get_logger(),
                   "prior_map_align=relocalization needs the iVox backend (it replaces a pinned tier); "
                   "this build uses ikd-Tree - loading the map as delivered instead");
#endif
    }
#ifdef FASTLIO_TEST_HOOKS
    locals_for_test_.value("prior_map_align", align_name);
#endif
  }
  readInto(node, map.initial_pose, values_.initial_pose_vec);
  {
    bool full_rpy_override = false;
    bool legacy_apply_rp = false;
    readInto(node, map.initial_pose_full_rpy_override, full_rpy_override);
    readInto(node, map.initial_pose_apply_roll_pitch, legacy_apply_rp);
    if (legacy_apply_rp) {
      RCLCPP_WARN(node.get_logger(),
                  "Parameter 'initial_pose_apply_roll_pitch' is deprecated; "
                  "use 'initial_pose_full_rpy_override' instead.");
    }
    values_.initial_pose_full_rpy_override = full_rpy_override || legacy_apply_rp;
    if (values_.initial_pose_full_rpy_override) {
      RCLCPP_WARN(node.get_logger(),
                  "initial_pose_full_rpy_override=true: overriding IMU-aligned "
                  "roll/pitch with values from initial_pose. Caller must ensure "
                  "these match gravity (accel along -Z), else point cloud "
                  "transforms will be incorrect.");
    }
#ifdef FASTLIO_TEST_HOOKS
    locals_for_test_.value("initial_pose_full_rpy_override", full_rpy_override);
    locals_for_test_.value("initial_pose_apply_roll_pitch", legacy_apply_rp);
#endif
  }
}

namespace
{
// The wheel measurement model's entry checks (wheel_measurement.hpp): every value finite, the slip window positive,
// the sigmas nonnegative. Without a variance topic the two parameter sigmas are every sample's variances, which the
// sample ring takes only when positive.
void checkWheelModel(const rclcpp::Node & node, const FrontendParams & v)
{
  const auto require = [&node](bool ok, const char * key, const char * rule, double value) {
    if (ok)
      return;
    RCLCPP_FATAL(node.get_logger(), "%s must be %s, got %g", key, rule, value);
    throw std::invalid_argument(key);
  };
  require(std::isfinite(v.wheel_slip_k), "wheel_slip_k", "finite", v.wheel_slip_k);
  require(std::isfinite(v.wheel_slip_phi0), "wheel_slip_phi0", "finite", v.wheel_slip_phi0);
  require(std::isfinite(v.wheel_slip_fx_window_s) && v.wheel_slip_fx_window_s > 0.0,
          "wheel_slip_fx_window_s",
          "finite and > 0",
          v.wheel_slip_fx_window_s);
  for (int i = 0; i < 3; ++i)
    require(std::isfinite(v.wheel_lever_arm(i)), "wheel_lever_arm", "finite in every element", v.wheel_lever_arm(i));
  require(std::isfinite(v.wheel_lever_sigma) && v.wheel_lever_sigma >= 0.0,
          "wheel_lever_sigma",
          "finite and >= 0",
          v.wheel_lever_sigma);
  require(std::isfinite(v.wheel_vel_noise_vx) && v.wheel_vel_noise_vx >= 0.0,
          "wheel_vel_noise_vx",
          "finite and >= 0",
          v.wheel_vel_noise_vx);
  require(std::isfinite(v.wheel_vel_noise_vy) && v.wheel_vel_noise_vy >= 0.0,
          "wheel_vel_noise_vy",
          "finite and >= 0",
          v.wheel_vel_noise_vy);
  require(std::isfinite(v.wheel_vel_noise_vz) && v.wheel_vel_noise_vz >= 0.0,
          "wheel_vel_noise_vz",
          "finite and >= 0",
          v.wheel_vel_noise_vz);
  if (v.wheel_odom_en && v.wheel_cov_topic.empty()) {
    require(v.wheel_vel_noise_vx > 0.0, "wheel_vel_noise_vx", "> 0 without wheel_cov_topic", v.wheel_vel_noise_vx);
    require(v.wheel_vel_noise_vy > 0.0, "wheel_vel_noise_vy", "> 0 without wheel_cov_topic", v.wheel_vel_noise_vy);
  }
}
}  // namespace

void FrontendParameters::readWheel(const rclcpp::Node & node, FrontendStartup & startup)
{
  const ConstraintParameterSpecs constraints{};
  readInto(node, constraints.wheel_odom_en, values_.wheel_odom_en);
  readInto(node, constraints.wheel_topic, startup.wheel_topic);
  readInto(node, constraints.wheel_speed_scale, values_.wheel_speed_scale);
  readInto(node, constraints.wheel_vel_noise_vx, values_.wheel_vel_noise_vx);
  readInto(node, constraints.wheel_vel_noise_vy, values_.wheel_vel_noise_vy);
  readInto(node, constraints.wheel_vel_noise_vz, values_.wheel_vel_noise_vz);
  std::vector<double> wext;
  readInto(node, constraints.wheel_extrinsic_R, wext);
  if (wext.size() == 9)
    values_.R_robot_to_imu = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(wext.data());
  readInto(node, constraints.wheel_cov_topic, values_.wheel_cov_topic);
  readInto(node, constraints.wheel_slip_k, values_.wheel_slip_k);
  readInto(node, constraints.wheel_slip_phi0, values_.wheel_slip_phi0);
  readInto(node, constraints.wheel_slip_fx_window_s, values_.wheel_slip_fx_window_s);
  std::vector<double> lever;
  readInto(node, constraints.wheel_lever_arm, lever);
  readInto(node, constraints.wheel_lever_sigma, values_.wheel_lever_sigma);
  if (lever.size() != 3u) {
    RCLCPP_FATAL(node.get_logger(), "wheel_lever_arm must have exactly 3 elements, got %zu", lever.size());
    throw std::invalid_argument("wheel_lever_arm size mismatch");
  }
  values_.wheel_lever_arm = Eigen::Vector3d(lever[0], lever[1], lever[2]);
  checkWheelModel(node, values_);
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("wheel_extrinsic_R", wext);
  locals_for_test_.value("wheel_topic", startup.wheel_topic);
  locals_for_test_.value("wheel_lever_arm", lever);
#endif
}

double FrontendParameters::readMapPeriod(const rclcpp::Node & node)
{
  const OutputParameterSpecs output{};
  double map_pub_period_sec = 5.0;
  readInto(node, output.publish_map_period_sec, map_pub_period_sec);
  if (map_pub_period_sec <= 0.0)
    map_pub_period_sec = 5.0;
#ifdef FASTLIO_TEST_HOOKS
  locals_for_test_.value("publish.map_period_sec", map_pub_period_sec);
#endif
  return map_pub_period_sec;
}
}  // namespace fast_lio

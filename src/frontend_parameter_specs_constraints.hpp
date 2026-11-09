#pragma once

#include <string>
#include <vector>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// Soft constraints the filter takes besides the lidar: planar, gravity alignment and wheel odometry.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct ConstraintParameterSpecs
{
  const ParameterSpec<bool> planar_constraint_en{"planar_constraint_en", false, false};
  const ParameterSpec<double> planar_constraint_noise{"planar_constraint_noise", 0.1, 0.1};
  const ParameterSpec<double> planar_constraint_z_weak_thresh{"planar_constraint_z_weak_thresh", 0.5, 0.5};
  const ParameterSpec<bool> gravity_align_en{"gravity_align_en", false, false};
  const ParameterSpec<double> gravity_align_noise{"gravity_align_noise", 0.05, 0.05};
  const ParameterSpec<double> gravity_align_z_weak_thresh{"gravity_align_z_weak_thresh", 0.3, 0.3};
  const ParameterSpec<double> gravity_align_accel_tol{"gravity_align_accel_tol", 0.05, 0.05};
  const ParameterSpec<double> gravity_align_gyro_tol{"gravity_align_gyro_tol", 0.35, 0.35};
  const ParameterSpec<double> gravity_align_window_s{"gravity_align_window_s", 0.0, 0.0};
  const ParameterSpec<bool> gravity_align_continuous{"gravity_align_continuous", false, false};
  const ParameterSpec<double> gravity_align_hold_s{"gravity_align_hold_s", 0.0, 0.0};
  const ParameterSpec<double> gravity_align_noise_degraded{"gravity_align_noise_degraded", 0.0, 0.0};
  const ParameterSpec<double> gravity_align_mag_ref{"gravity_align_mag_ref", 0.0, 0.0};
  const ParameterSpec<double> gravity_align_grav_leak{"gravity_align_grav_leak", 1.0, 1.0};
  const ParameterSpec<double> gravity_align_grav_cap_deg{"gravity_align_grav_cap_deg", 3.0, 3.0};
  const ParameterSpec<bool> gravity_align_grav_freeze_degraded{"gravity_align_grav_freeze_degraded", false, false};
  const ParameterSpec<bool> gravity_align_lin_accel_comp{"gravity_align_lin_accel_comp", false, false};
  const ParameterSpec<double> gravity_align_vel_sigma_scale{"gravity_align_vel_sigma_scale", 0.0, 0.0};
  const ParameterSpec<std::string> gravity_align_lin_accel_source{
    "gravity_align_lin_accel_source", "leg", std::string("leg")};
  const ParameterSpec<bool> gravity_align_withhold_blind_moving{"gravity_align_withhold_blind_moving", false, false};
  const ParameterSpec<std::string> gravity_align_sigma_policy{
    "gravity_align_sigma_policy", "scene", std::string("scene")};
  const ParameterSpec<double> gravity_align_noise_static{"gravity_align_noise_static", 0.008, 0.008};
  const ParameterSpec<double> gravity_align_noise_moving{"gravity_align_noise_moving", 0.045, 0.045};
  const ParameterSpec<double> gravity_align_leg_vel_sigma{"gravity_align_leg_vel_sigma", 0.15, 0.15};
  const ParameterSpec<double> gravity_align_leg_edge_window_s{"gravity_align_leg_edge_window_s", 0.2, 0.2};
  const ParameterSpec<std::vector<double>> gravity_align_leg_lever_arm{
    "gravity_align_leg_lever_arm", std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{0.0, 0.0, 0.0}};
  const ParameterSpec<bool> wheel_odom_en{"wheel_odom_en", false, false};
  const ParameterSpec<std::string> wheel_topic{"wheel_topic", "/lio/twist", std::string("/lio/twist")};
  const ParameterSpec<double> wheel_speed_scale{"wheel_speed_scale", 1.0, 1.0};
  const ParameterSpec<double> wheel_vel_noise_vx{"wheel_vel_noise_vx", 0.10, 0.10};
  const ParameterSpec<double> wheel_vel_noise_vy{"wheel_vel_noise_vy", 0.05, 0.05};
  const ParameterSpec<double> wheel_vel_noise_vz{"wheel_vel_noise_vz", 0.10, 0.10};
  const ParameterSpec<std::vector<double>> wheel_extrinsic_R{
    "wheel_extrinsic_R", std::vector<double>(), std::vector<double>()};
  const ParameterSpec<std::string> wheel_cov_topic{"wheel_cov_topic", "", std::string("")};
  const ParameterSpec<double> wheel_slip_k{"wheel_slip_k", 0.0, 0.0};
  const ParameterSpec<double> wheel_slip_phi0{"wheel_slip_phi0", 0.0, 0.0};
  const ParameterSpec<double> wheel_slip_fx_window_s{"wheel_slip_fx_window_s", 1.0, 1.0};
  const ParameterSpec<std::vector<double>> wheel_lever_arm{
    "wheel_lever_arm", std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{0.0, 0.0, 0.0}};
  const ParameterSpec<double> wheel_lever_sigma{"wheel_lever_sigma", 0.0, 0.0};
};
}  // namespace fast_lio

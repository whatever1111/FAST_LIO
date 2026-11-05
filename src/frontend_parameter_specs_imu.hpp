#pragma once

#include <string>
#include <vector>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// IMU initialisation, noise, coverage and gap priors, and the lidar-IMU extrinsics.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct ImuParameterSpecs
{
  const ParameterSpec<bool> imu_init_require_still{"imu_init_require_still", false, false};
  const ParameterSpec<double> imu_init_still_tol{"imu_init_still_tol", 0.03, 0.03};
  const ParameterSpec<double> imu_init_still_timeout_s{"imu_init_still_timeout_s", 20.0, 20.0};
  const ParameterSpec<bool> imu_init_still_sustained_en{"imu_init_still_sustained_en", false, false};
  const ParameterSpec<double> imu_init_still_min_duration_s{"imu_init_still_min_duration_s", 1.0, 1.0};
  const ParameterSpec<int> imu_init_still_min_samples{"imu_init_still_min_samples", 100, 100};
  const ParameterSpec<double> imu_init_still_gyro_rms_max{"imu_init_still_gyro_rms_max", 0.1, 0.1};
  const ParameterSpec<double> imu_init_still_gravity_ref{"imu_init_still_gravity_ref", 9.81, 9.81};
  const ParameterSpec<double> imu_init_still_gravity_tol{"imu_init_still_gravity_tol", 0.05, 0.05};
  const ParameterSpec<double> imu_init_still_max_gap_s{"imu_init_still_max_gap_s", 0.1, 0.1};
  const ParameterSpec<double> mapping_gyr_cov{"mapping.gyr_cov", 0.1, 0.1};
  const ParameterSpec<double> mapping_acc_cov{"mapping.acc_cov", 0.1, 0.1};
  const ParameterSpec<double> mapping_b_gyr_cov{"mapping.b_gyr_cov", 0.0001, 0.0001};
  const ParameterSpec<double> mapping_b_acc_cov{"mapping.b_acc_cov", 0.0001, 0.0001};
  const ParameterSpec<double> imu_coverage_max_gap_s{"imu_coverage.max_gap_s", 0.0, 0.0};
  const ParameterSpec<double> imu_coverage_max_extrapolation_s{"imu_coverage.max_extrapolation_s", 0.0, 0.0};
  const ParameterSpec<bool> imu_gap_prior_en{"imu_gap_prior_en", false, false};
  const ParameterSpec<double> imu_gap_min_s{"imu_gap_min_s", 0.1, 0.1};
  const ParameterSpec<double> imu_gap_accel_max{"imu_gap_accel_max", 1.0, 1.0};
  const ParameterSpec<double> imu_gap_accel_max_z{"imu_gap_accel_max_z", 0.2, 0.2};
  const ParameterSpec<double> imu_gap_rate_rp_max{"imu_gap_rate_rp_max", 0.0, 0.0};
  const ParameterSpec<double> imu_gap_rate_yaw_max{"imu_gap_rate_yaw_max", 10.0, 10.0};
  const ParameterSpec<double> imu_gap_verify_s{"imu_gap_verify_s", 0.3, 0.3};
  const ParameterSpec<bool> imu_gap_empty_scans{"imu_gap_empty_scans", false, false};
  const ParameterSpec<bool> mapping_extrinsic_est_en{"mapping.extrinsic_est_en", true, true};
  const ParameterSpec<std::vector<double>> mapping_extrinsic_T{
    "mapping.extrinsic_T", std::vector<double>(), std::vector<double>()};
  const ParameterSpec<std::vector<double>> mapping_extrinsic_R{
    "mapping.extrinsic_R", std::vector<double>(), std::vector<double>()};
};
}  // namespace fast_lio

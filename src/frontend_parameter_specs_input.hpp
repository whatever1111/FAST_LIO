#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"
#include "preprocess.h"  // LID_TYPE and TIME_UNIT: the lidar type and timestamp unit defaults

namespace fast_lio
{
/// Sensor input: topics, buffering and time alignment, and the lidar preprocessing.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct InputParameterSpecs
{
  const ParameterSpec<int> max_buffered_scans{"max_buffered_scans", 100, 100};
  const ParameterSpec<double> max_scan_backlog_sec{"max_scan_backlog_sec", 0.5, 0.5};
  // Bound on the IMU sensor time imu_buffer holds, in seconds behind the newest sample; must be finite, <= 0
  // disables it. 10 s never binds on normal data: the four M20 regression bags hold at most 313 samples (~1.5 s)
  // before the first scan and well under 1 s in steady state.
  const ParameterSpec<double> max_imu_backlog_sec{"max_imu_backlog_sec", 10.0, 10.0};
  const ParameterSpec<std::string> common_lid_topic{"common.lid_topic", "/livox/lidar", "/livox/lidar"};
  const ParameterSpec<std::string> common_imu_topic{"common.imu_topic", "/livox/imu", "/livox/imu"};
  const ParameterSpec<std::string> common_imu_input_type{
    "common.imu_input_type", "sensor_msgs", std::string("sensor_msgs")};
  const ParameterSpec<std::string> common_imu_bias_topic{
    "common.imu_bias_topic", "/fixposition/fpa/imubias", std::string("/fixposition/fpa/imubias")};
  const ParameterSpec<bool> common_time_sync_en{"common.time_sync_en", false, false};
  const ParameterSpec<double> common_time_offset_lidar_to_imu{"common.time_offset_lidar_to_imu", 0.0, 0.0};
  const ParameterSpec<double> common_imu_time_offset{"common.imu_time_offset", 0.0, 0.0};
  const ParameterSpec<double> preprocess_blind{"preprocess.blind", 0.01, 0.01};
  const ParameterSpec<int> preprocess_lidar_type{"preprocess.lidar_type", AVIA, AVIA};
  const ParameterSpec<int> preprocess_scan_line{"preprocess.scan_line", 16, 16};
  const ParameterSpec<int> preprocess_timestamp_unit{"preprocess.timestamp_unit", US, US};
  const ParameterSpec<int> preprocess_scan_rate{"preprocess.scan_rate", 10, 10};
  const ParameterSpec<double> preprocess_max_scan_duration_s{"preprocess.max_scan_duration_s", 0.0, 0.0};
  const ParameterSpec<int> point_filter_num{"point_filter_num", 2, 2};
  const ParameterSpec<bool> feature_extract_enable{"feature_extract_enable", false, false};
};
}  // namespace fast_lio

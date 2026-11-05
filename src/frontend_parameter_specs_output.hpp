#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// What the front end publishes, in which frames, and its typed observation outputs.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct OutputParameterSpecs
{
  const ParameterSpec<bool> publish_path_en{"publish.path_en", true, true};
  const ParameterSpec<bool> publish_effect_map_en{"publish.effect_map_en", false, false};
  const ParameterSpec<bool> publish_map_en{"publish.map_en", false, false};
  // /Laser_map publish period (s); 5s = 0.2Hz
  const ParameterSpec<double> publish_map_period_sec{"publish.map_period_sec", 5.0, 5.0};
  const ParameterSpec<bool> publish_scan_publish_en{"publish.scan_publish_en", true, true};
  const ParameterSpec<bool> publish_dense_publish_en{"publish.dense_publish_en", true, true};
  const ParameterSpec<bool> publish_scan_bodyframe_pub_en{"publish.scan_bodyframe_pub_en", true, true};
  const ParameterSpec<std::string> publish_odom_frame_id{
    "publish.odom_frame_id", "lidar_odom", std::string("lidar_odom")};
  const ParameterSpec<std::string> publish_body_frame_id{"publish.body_frame_id", "body", std::string("body")};
  const ParameterSpec<bool> health_pub_en{"health_pub_en", true, true};
  const DeclaredValueSpec<bool> publish_frontend_observation{"publish.frontend_observation", false, false};
  const DeclaredValueSpec<bool> publish_reference_health{"publish.reference_health", false, true};
  const DeclaredValueSpec<std::string> publish_frontend_clock_domain{"publish.frontend_clock_domain", "", false};
  const DeclaredValueSpec<double> publish_frontend_idle_period_sec{"publish.frontend_idle_period_sec", 0.5, false};
};
}  // namespace fast_lio

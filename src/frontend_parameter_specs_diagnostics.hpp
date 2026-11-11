#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// Diagnostic outputs: correspondence dumps, the position log, latency and Kalman-channel records, the data directory.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct DiagnosticsParameterSpecs
{
  const ParameterSpec<bool> corr_dump_en{"corr_dump_en", false, false};
  const ParameterSpec<std::string> corr_dump_windows{"corr_dump_windows", "", std::string("")};
  const ParameterSpec<std::string> data_dir{"data_dir", "", ""};
  const ParameterSpec<bool> runtime_pos_log_enable{"runtime_pos_log_enable", false, 0};
  const ParameterSpec<bool> diagnostics_latency_enable{"diagnostics.latency_enable", true, true};
  const ParameterSpec<double> diagnostics_latency_log_period_sec{"diagnostics.latency_log_period_sec", 1.0, 1.0};
  const ParameterSpec<bool> diagnostics_kalman_channel_enable{"diagnostics.kalman_channel_enable", false, false};
  const DeclaredValueSpec<bool> diagnostics_time_admission_counts_enable{
    "diagnostics.time_admission_counts_enable", false, true};
};
}  // namespace fast_lio

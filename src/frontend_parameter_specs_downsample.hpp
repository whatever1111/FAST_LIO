#pragma once

#include <string>

#include "frontend_parameter_spec.hpp"

namespace fast_lio
{
/// Adaptive downsampling of the scan under a time budget.
/// One spec per parameter, in declaration order, named after the parameter with '.' written '_'. FrontendParameters
/// declares and reads them in the constructor's original order, not domain by domain.
struct DownsampleParameterSpecs
{
  const ParameterSpec<bool> adaptive_downsample_en{"adaptive_downsample_en", false, false};
  const ParameterSpec<double> adaptive_downsample_budget_sec{"adaptive_downsample_budget_sec", 0.06, 0.06};
  const ParameterSpec<double> adaptive_downsample_release_sec{"adaptive_downsample_release_sec", 0.035, 0.035};
  const ParameterSpec<double> adaptive_downsample_step{"adaptive_downsample_step", 0.05, 0.05};
  const ParameterSpec<double> adaptive_downsample_max_voxel{"adaptive_downsample_max_voxel", 0.5, 0.5};
  const ParameterSpec<int> adaptive_downsample_raise_after{"adaptive_downsample_raise_after", 5, 5};
  const ParameterSpec<int> adaptive_downsample_lower_after{"adaptive_downsample_lower_after", 100, 100};
};
}  // namespace fast_lio

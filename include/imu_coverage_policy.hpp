#pragma once

#include <algorithm>
#include <cmath>

namespace fast_lio
{

enum class ImuCoverageStatus
{
  kCovered,
  kEmpty,
  kInvalid,
  kUnordered,
  kGap,
  kStart,
  kEnd
};
struct ImuCoverageParams
{
  double max_gap_s = 0.0;  // zero leaves the device-specific gap bound disabled
  double max_extrapolation_s = 0.0;
};
struct ImuCoverageResult
{
  ImuCoverageStatus status = ImuCoverageStatus::kEmpty;
  double max_gap_s = 0.0;
  double start_extrapolation_s = 0.0;
  double end_extrapolation_s = 0.0;
  bool covered() const { return status == ImuCoverageStatus::kCovered; }
};

// The window begins at the estimator's last propagated time, which may be
// earlier than this scan's first point. Include the retained previous IMU.
template<class Stamps>
ImuCoverageResult imuCoverage(double begin, double end, const Stamps & stamps, const ImuCoverageParams & params)
{
  ImuCoverageResult result;
  if (!std::isfinite(begin) || !std::isfinite(end) || begin > end || !std::isfinite(params.max_gap_s) ||
      params.max_gap_s < 0.0 || !std::isfinite(params.max_extrapolation_s) || params.max_extrapolation_s < 0.0) {
    result.status = ImuCoverageStatus::kInvalid;
    return result;
  }
  if (stamps.empty())
    return result;
  double previous = stamps.front();
  for (const double stamp : stamps) {
    if (!std::isfinite(stamp) || stamp < 0.0 || stamp > end) {
      result.status = ImuCoverageStatus::kInvalid;
      return result;
    }
    if (stamp < previous) {
      result.status = ImuCoverageStatus::kUnordered;
      return result;
    }
    result.max_gap_s = std::max(result.max_gap_s, stamp - previous);
    previous = stamp;
  }
  result.start_extrapolation_s = std::max(0.0, stamps.front() - begin);
  result.end_extrapolation_s = std::max(0.0, end - stamps.back());
  if (params.max_gap_s > 0.0 && result.max_gap_s > params.max_gap_s)
    result.status = ImuCoverageStatus::kGap;
  else if (params.max_extrapolation_s > 0.0 && result.start_extrapolation_s > params.max_extrapolation_s)
    result.status = ImuCoverageStatus::kStart;
  else if (params.max_extrapolation_s > 0.0 && result.end_extrapolation_s > params.max_extrapolation_s)
    result.status = ImuCoverageStatus::kEnd;
  else
    result.status = ImuCoverageStatus::kCovered;
  return result;
}

}  // namespace fast_lio

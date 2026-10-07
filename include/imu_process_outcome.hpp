#pragma once

#include <cstddef>

#include "imu_coverage_policy.hpp"

namespace fast_lio
{
enum class ImuDisposition
{
  kUncommitted,
  kCommitted,
  kInvalid,
  kFatal
};

enum class ImuProcessReason
{
  kNone,
  kNoOutput,
  kNoLidar,
  kNoImu,
  kNonadvancing,
  kInvalidScanTime,
  kInvalidImu,
  kInitializationInvalidMean,
  kInitializationInvalidState,
  kInitializationAccumulated,
  kInitializationMotion,
  kInitializationComplete,
  kHistory,
  kCoverageEmpty,
  kCoverageInvalid,
  kCoverageUnordered,
  kCoverageGap,
  kCoverageStart,
  kCoverageEnd,
  kProcessed,
  kProcessedNoCloud,
  kBootstrap,
  kPartialPropagation,
  kCount
};

struct ImuProcessOutcome
{
  ImuDisposition disposition = ImuDisposition::kUncommitted;
  ImuProcessReason reason = ImuProcessReason::kNone;
  bool producedOutput = false;
};

inline ImuProcessOutcome imuCoverageOutcome(ImuCoverageStatus status)
{
  switch (status) {
    case ImuCoverageStatus::kCovered:
      return {ImuDisposition::kCommitted, ImuProcessReason::kProcessed};
    case ImuCoverageStatus::kEmpty:
      return {ImuDisposition::kUncommitted, ImuProcessReason::kCoverageEmpty};
    case ImuCoverageStatus::kInvalid:
      return {ImuDisposition::kInvalid, ImuProcessReason::kCoverageInvalid};
    case ImuCoverageStatus::kUnordered:
      return {ImuDisposition::kInvalid, ImuProcessReason::kCoverageUnordered};
    case ImuCoverageStatus::kGap:
      return {ImuDisposition::kCommitted, ImuProcessReason::kCoverageGap};
    case ImuCoverageStatus::kStart:
      return {ImuDisposition::kCommitted, ImuProcessReason::kCoverageStart};
    case ImuCoverageStatus::kEnd:
      return {ImuDisposition::kCommitted, ImuProcessReason::kCoverageEnd};
  }
  return {ImuDisposition::kFatal, ImuProcessReason::kNone};
}

inline const char * imuProcessReasonName(ImuProcessReason reason)
{
  constexpr const char * names[] = {"none",
                                    "no_output",
                                    "no_lidar",
                                    "no_imu",
                                    "nonadvancing",
                                    "invalid_scan_time",
                                    "invalid_imu",
                                    "initialization_invalid_mean",
                                    "initialization_invalid_state",
                                    "initialization_accumulated",
                                    "initialization_motion",
                                    "initialization_complete",
                                    "history",
                                    "coverage_empty",
                                    "coverage_invalid",
                                    "coverage_unordered",
                                    "coverage_gap",
                                    "coverage_start",
                                    "coverage_end",
                                    "processed",
                                    "processed_no_cloud",
                                    "bootstrap",
                                    "partial_propagation"};
  const auto index = static_cast<std::size_t>(reason);
  return index < static_cast<std::size_t>(ImuProcessReason::kCount) ? names[index] : "unknown";
}
}  // namespace fast_lio

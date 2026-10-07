#pragma once

#include <algorithm>
#include <cstddef>

namespace fast_lio
{

enum class ScanHistoryAction
{
  kKeepAll,
  kDiscardEarly,
  kReject
};

// Use the same relative-time comparison as deskew's earliest segment. The
// complete original scan must pass scanTime before this policy is consulted;
// discarding points must never hide invalid offsets. Equality is retained.
inline bool pointInScanHistory(double curvatureMilliseconds, double historyOffsetSeconds)
{
  constexpr double kMillisecondsPerSecond = 1000.0;
  return curvatureMilliseconds / kMillisecondsPerSecond >= historyOffsetSeconds;
}

template<class Points, class Offset>
ScanHistoryAction scanHistoryAction(const Points & points, double historyOffsetSeconds, Offset offset)
{
  std::size_t retained = 0;
  for (const auto & point : points) {
    if (pointInScanHistory(static_cast<double>(offset(point)), historyOffsetSeconds))
      ++retained;
  }
  if (retained == 0)
    return ScanHistoryAction::kReject;
  return retained == points.size() ? ScanHistoryAction::kKeepAll : ScanHistoryAction::kDiscardEarly;
}

// Compact only the already-selected working points. Stable removal preserves
// every retained field and its original order; it does not repeat sensor stride.
template<class Points, class Offset>
std::size_t discardPointsBeforeHistory(Points & points, double historyOffsetSeconds, Offset offset)
{
  const auto originalSize = points.size();
  points.erase(std::remove_if(points.begin(),
                              points.end(),
                              [&](const auto & point) {
                                return !pointInScanHistory(static_cast<double>(offset(point)), historyOffsetSeconds);
                              }),
               points.end());
  return originalSize - points.size();
}

}  // namespace fast_lio

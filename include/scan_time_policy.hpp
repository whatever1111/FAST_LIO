#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace fast_lio
{

enum class ScanTimeStatus
{
  kValid,
  kEmpty,
  kInvalidBegin,
  kInvalidOffset,
  kExcessiveSpan
};

struct ScanTimeResult
{
  ScanTimeStatus status = ScanTimeStatus::kEmpty;
  double end = 0.0;
  double duration = 0.0;
  bool valid() const { return status == ScanTimeStatus::kValid; }
};

// Offsets are milliseconds at the preprocessor boundary. Inspect the effective
// point set without changing its order or the preprocessor's index sampling.
// A present zero offset is valid, including a short or instantaneous scan.
template<class Points, class Offset>
ScanTimeResult scanTime(double begin, const Points & points, Offset offset, double max_duration_s = 0.0)
{
  ScanTimeResult result;
  if (!std::isfinite(begin) || begin < 0.0 || !std::isfinite(max_duration_s) || max_duration_s < 0.0) {
    result.status = ScanTimeStatus::kInvalidBegin;
    return result;
  }
  if (points.empty())
    return result;
  double maximum = 0.0;
  for (const auto & point : points) {
    const double value = offset(point);
    if (!std::isfinite(value) || value < 0.0) {
      result.status = ScanTimeStatus::kInvalidOffset;
      return result;
    }
    maximum = std::max(maximum, value);
  }
  constexpr double kMillisecondsPerSecond = 1000.0;
  result.duration = maximum / kMillisecondsPerSecond;
  result.end = begin + result.duration;
  if (!std::isfinite(result.end) || (max_duration_s > 0.0 && result.duration > max_duration_s)) {
    result.status = ScanTimeStatus::kExcessiveSpan;
    return result;
  }
  result.status = ScanTimeStatus::kValid;
  return result;
}

// Commit only after IMU processing (including initialization) consumed a scan.
// Rejected/empty input must never advance this watermark or reuse old output.
class ScanConsumption
{
public:
  bool canProcess(double end) const { return std::isfinite(end) && end >= 0.0 && end > last_end_; }
  bool commit(double end)
  {
    if (!canProcess(end))
      return false;
    last_end_ = end;
    return true;
  }
  double lastEnd() const { return last_end_; }
  void reset() { last_end_ = -std::numeric_limits<double>::infinity(); }

private:
  double last_end_ = -std::numeric_limits<double>::infinity();
};

// A changed clock epoch invalidates all estimator/map temporal state. Latch a
// fault until the node is restarted; clearing just one queue is not a reset.
class InputEpochGuard
{
public:
  bool observe(double previous, double current, double max_rollback_s)
  {
    if (!std::isfinite(current) || current < 0.0 || !std::isfinite(previous) || !std::isfinite(max_rollback_s) ||
        max_rollback_s < 0.0 || (previous >= 0.0 && previous - current > max_rollback_s))
      faulted_ = true;
    return !faulted_;
  }
  bool faulted() const { return faulted_; }

private:
  bool faulted_ = false;
};

// One per input stream, consulted before InputEpochGuard with the same baseline: the stream's last accepted stamp
// (< 0 before the first one). A stamp more than max_lead_s past it is held back instead of becoming the baseline.
// Taken as it is, one corrupt stamp ahead would make every later sample look like a rollback past the guard's bound
// and latch it. The next sample decides. Back within max_lead_s of the baseline, the held stamp was a lone bad one
// and only that sample is lost. Within max_lead_s after the held stamp, the stream really moved on (an outage ended,
// or its clock stepped forward): it continues from there, one sample short. Any other sample is held in its place.
// A rollback is not this filter's business; it passes, and the guard judges it.
class InputLeadFilter
{
public:
  bool admit(double previous, double current, double max_lead_s)
  {
    const bool lead = previous >= 0.0 && std::isfinite(current) && current - previous > max_lead_s;
    if (!lead) {
      held_ = kNone;
      return true;
    }
    const bool confirms = held_ >= 0.0 && current >= held_ && current - held_ <= max_lead_s;
    held_ = confirms ? kNone : current;
    return confirms;
  }

private:
  static constexpr double kNone = -1.0;
  double held_ = kNone;
};

}  // namespace fast_lio

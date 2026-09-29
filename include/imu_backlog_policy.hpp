// The front end's IMU queue bound: how many of the oldest samples imu_buffer drops after a push.
//
// Only a LiDAR scan drains imu_buffer (sync_packages hands a scan every sample up to its end time), so while the
// LiDAR is absent and the IMU keeps publishing nothing else bounds the queue, and the first scan afterwards would
// integrate the whole backlog. The queue is bounded in sensor time instead: a sample more than max_backlog_sec older
// than the newest one (newest - stamp > max_backlog_sec) is dropped, except that
//  - the newest sample is never dropped, and
//  - no sample stamped at or after protect_from is dropped: that is the begin time of the oldest queued scan, which
//    still integrates those samples whatever their age (+infinity when no scan is queued).
// A sample exactly max_backlog_sec older than the newest stays.
//
// Samples only ever leave from the front, so the dropped samples are a prefix of the queue: the scan starts at the
// oldest sample and stops at the first one it must keep, reading at most one stamp more than it drops. A sample
// stamped out of order behind a kept one therefore stays until it reaches the front; no ordering of the stamps can
// make a sample inside the window, or one from protect_from on, go.
//
// Garbage never trims: a non-finite newest stamp, max_backlog_sec or examined stamp, or a protect_from that is NaN or
// -infinity, drops nothing. max_backlog_sec <= 0 disables the bound.
#pragma once

#include <cstddef>
#include <limits>

namespace fast_lio
{

// True for a finite value: false for NaN and both infinities. std::isfinite is not constexpr before C++23, and an
// ordered comparison with a NaN is not a constant expression, so this uses equality comparisons only.
constexpr bool imuBacklogFinite(double value) noexcept
{
  return value == value && value != std::numeric_limits<double>::infinity() &&
         value != -std::numeric_limits<double>::infinity();
}

// How many of the oldest samples to drop from a queue of `size` samples. stamp_at(i) returns the stamp of sample i,
// oldest first; it is only asked for i < size - 1. newest is the stamp of the newest sample (i = size - 1).
// Every value is checked for garbage before an ordered comparison reads it.
template<class StampAt>
constexpr std::size_t
imuBacklogExcess(std::size_t size, const StampAt & stamp_at, double newest, double max_backlog_sec, double protect_from)
{
  const bool guard_valid = protect_from == protect_from && protect_from != -std::numeric_limits<double>::infinity();
  if (!imuBacklogFinite(max_backlog_sec) || !(max_backlog_sec > 0.0) || !imuBacklogFinite(newest) || !guard_valid) {
    return 0;
  }
  std::size_t excess = 0;
  while (excess + 1 < size) {
    const double stamp = stamp_at(excess);
    if (!imuBacklogFinite(stamp)) {
      return 0;
    }
    if (!(stamp < protect_from) || !(newest - stamp > max_backlog_sec)) {
      break;
    }
    ++excess;
  }
  return excess;
}

}  // namespace fast_lio

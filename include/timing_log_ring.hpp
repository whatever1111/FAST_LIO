// The front end's timing log as a ring: the T1 / s_plot* arrays of laserMapping.cpp.
//
// The log keeps one row per processed scan and one preprocessing duration per LiDAR callback in arrays
// of kTimingLogCapacity entries, indexed by counters that only ever grow for the life of the node.
// The arrays are a ring over those counters: a counter's slot is its value modulo the capacity, so a
// node that outlives the capacity keeps the newest kTimingLogCapacity rows instead of writing past
// the arrays into whatever the linker placed after them. The shutdown dump reads timingLogRows()
// rows, oldest first, through timingLogRowSlot(). A negative counter is an int that wrapped: it has
// no slot and its sample is dropped.
#pragma once

#include <cstddef>

namespace fast_lio
{

constexpr std::size_t kTimingLogCapacity = 720000;

// True when a counter value may index the ring at all.
constexpr bool timingLogAccepts(long long counter) noexcept
{
  return counter >= 0;
}

// The slot a counter value writes. Requires timingLogAccepts(counter).
constexpr std::size_t timingLogSlot(long long counter) noexcept
{
  return static_cast<std::size_t>(static_cast<unsigned long long>(counter) % kTimingLogCapacity);
}

// The number of rows the ring holds for a counter: the counter clamped to the capacity.
constexpr std::size_t timingLogRows(long long counter) noexcept
{
  if (counter <= 0) {
    return 0;
  }
  return counter < static_cast<long long>(kTimingLogCapacity) ? static_cast<std::size_t>(counter) : kTimingLogCapacity;
}

// The slot of the row-th oldest row the ring still holds (row < timingLogRows(counter)).
constexpr std::size_t timingLogRowSlot(long long counter, std::size_t row) noexcept
{
  const unsigned long long oldest = static_cast<unsigned long long>(counter) - timingLogRows(counter);
  return static_cast<std::size_t>((oldest + row) % kTimingLogCapacity);
}

}  // namespace fast_lio

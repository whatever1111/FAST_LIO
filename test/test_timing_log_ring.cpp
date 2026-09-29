// Tests for timing_log_ring.hpp: where the timing log's counters write and how a dump reads them back.
#include <gtest/gtest.h>

#include <cstddef>

#include "timing_log_ring.hpp"

namespace
{

constexpr long long kCapacity = static_cast<long long>(fast_lio::kTimingLogCapacity);

TEST(TimingLogRing, CapacityMatchesTheArrays)
{
  EXPECT_EQ(fast_lio::kTimingLogCapacity, 720000u);
}

TEST(TimingLogRing, CountersInsideTheFirstCycleWriteTheirOwnSlot)
{
  for (long long counter : {0LL, 1LL, 4711LL, kCapacity - 1}) {
    EXPECT_TRUE(fast_lio::timingLogAccepts(counter));
    EXPECT_EQ(fast_lio::timingLogSlot(counter), static_cast<std::size_t>(counter));
  }
}

TEST(TimingLogRing, CountersPastTheCapacityWrapInsideTheArrays)
{
  EXPECT_EQ(fast_lio::timingLogSlot(kCapacity), 0u);
  EXPECT_EQ(fast_lio::timingLogSlot(kCapacity + 1), 1u);
  EXPECT_EQ(fast_lio::timingLogSlot(2 * kCapacity - 1), fast_lio::kTimingLogCapacity - 1);
  // Ten days of scans at 10 Hz: the indices a long-running node reaches.
  EXPECT_EQ(fast_lio::timingLogSlot(8640048), 48u);
  EXPECT_EQ(fast_lio::timingLogSlot(8640052), 52u);
  EXPECT_LT(fast_lio::timingLogSlot(2147483647LL), fast_lio::kTimingLogCapacity);
}

TEST(TimingLogRing, WrappedCountersAreNotAccepted)
{
  EXPECT_FALSE(fast_lio::timingLogAccepts(-1));
  EXPECT_FALSE(fast_lio::timingLogAccepts(-2147483648LL));
}

TEST(TimingLogRing, RowsClampTheCounterToTheCapacity)
{
  EXPECT_EQ(fast_lio::timingLogRows(0), 0u);
  EXPECT_EQ(fast_lio::timingLogRows(-3), 0u);
  EXPECT_EQ(fast_lio::timingLogRows(5), 5u);
  EXPECT_EQ(fast_lio::timingLogRows(kCapacity - 1), static_cast<std::size_t>(kCapacity - 1));
  EXPECT_EQ(fast_lio::timingLogRows(kCapacity), fast_lio::kTimingLogCapacity);
  EXPECT_EQ(fast_lio::timingLogRows(8640052), fast_lio::kTimingLogCapacity);
}

TEST(TimingLogRing, DumpReadsAPartialRingInWriteOrder)
{
  for (std::size_t row = 0; row < 5; ++row) {
    EXPECT_EQ(fast_lio::timingLogRowSlot(5, row), row);
  }
}

TEST(TimingLogRing, DumpReadsAFullRingOldestFirst)
{
  const long long counter = kCapacity + 5;  // rows 5 .. capacity+4 survive; 0 .. 4 were overwritten
  ASSERT_EQ(fast_lio::timingLogRows(counter), fast_lio::kTimingLogCapacity);
  EXPECT_EQ(fast_lio::timingLogRowSlot(counter, 0), 5u);
  EXPECT_EQ(fast_lio::timingLogRowSlot(counter, fast_lio::kTimingLogCapacity - 6), fast_lio::kTimingLogCapacity - 1);
  EXPECT_EQ(fast_lio::timingLogRowSlot(counter, fast_lio::kTimingLogCapacity - 5), 0u);
  EXPECT_EQ(fast_lio::timingLogRowSlot(counter, fast_lio::kTimingLogCapacity - 1), 4u);
  // The newest row read is the slot the counter wrote last.
  EXPECT_EQ(fast_lio::timingLogRowSlot(counter, fast_lio::kTimingLogCapacity - 1), fast_lio::timingLogSlot(counter - 1));
}

TEST(TimingLogRing, DumpOfExactlyOneCycleStartsAtSlotZero)
{
  EXPECT_EQ(fast_lio::timingLogRowSlot(kCapacity, 0), 0u);
  EXPECT_EQ(fast_lio::timingLogRowSlot(kCapacity, fast_lio::kTimingLogCapacity - 1), fast_lio::kTimingLogCapacity - 1);
}

TEST(TimingLogRing, TheDecisionsAreConstexpr)
{
  static_assert(fast_lio::timingLogAccepts(0), "row 0 is a row");
  static_assert(!fast_lio::timingLogAccepts(-1), "a wrapped counter is not a row");
  static_assert(fast_lio::timingLogSlot(8640052) == 52u, "the ring wraps");
  static_assert(fast_lio::timingLogRows(8640052) == fast_lio::kTimingLogCapacity, "rows clamp");
  static_assert(fast_lio::timingLogRowSlot(8640052, 0) == 52u, "the oldest surviving row follows the newest");
  SUCCEED();
}

}  // namespace

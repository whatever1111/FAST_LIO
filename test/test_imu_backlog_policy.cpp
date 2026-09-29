// Tests for imu_backlog_policy.hpp: how much of the IMU queue a LiDAR outage may leave behind.
#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "imu_backlog_policy.hpp"

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kWindow = 10.0;                           // the node's default max_imu_backlog_sec
constexpr double kImuRate = 200.0;                         // Hz
constexpr double kEpoch = 1758700000.0;                    // a sensor-epoch stamp; whole seconds on top of it are exact
constexpr std::size_t kOutageSamples = 30 * 60 * 200 + 1;  // 30 min at 200 Hz, 0 .. 1800 s

// Sample i of a 200 Hz stream; exact whenever i is a multiple of the rate.
double imuStamp(std::size_t i)
{
  return kEpoch + static_cast<double>(i) / kImuRate;
}

std::vector<double> imuStream(std::size_t count)
{
  std::vector<double> stamps(count);
  for (std::size_t i = 0; i < count; ++i) {
    stamps[i] = imuStamp(i);
  }
  return stamps;
}

// The node's call: the newest stamp is the queue's last one.
std::size_t excess(const std::vector<double> & stamps, double max_backlog_sec, double protect_from = kInf)
{
  const double newest = stamps.empty() ? 0.0 : stamps.back();
  return fast_lio::imuBacklogExcess(
    stamps.size(), [&stamps](std::size_t i) { return stamps.at(i); }, newest, max_backlog_sec, protect_from);
}

TEST(ImuBacklogPolicy, EmptyBufferDropsNothing)
{
  std::size_t reads = 0;
  const auto stamp_at = [&reads](std::size_t) {
    ++reads;
    return 0.0;
  };
  EXPECT_EQ(fast_lio::imuBacklogExcess(0, stamp_at, kEpoch, kWindow, kInf), 0u);
  EXPECT_EQ(reads, 0u);
  EXPECT_EQ(excess({}, kWindow), 0u);
}

TEST(ImuBacklogPolicy, ASingleSampleIsTheNewestAndStays)
{
  EXPECT_EQ(excess({kEpoch}, kWindow), 0u);
  // Even against a newest stamp an hour ahead of it.
  EXPECT_EQ(fast_lio::imuBacklogExcess(1, [](std::size_t) { return kEpoch; }, kEpoch + 3600.0, kWindow, kInf), 0u);
}

TEST(ImuBacklogPolicy, SamplesInsideTheWindowStay)
{
  // The largest backlog the M20 regression bags queue before their first scan, and one second of steady state.
  EXPECT_EQ(excess(imuStream(313), kWindow), 0u);
  EXPECT_EQ(excess(imuStream(201), kWindow), 0u);
  // The whole window: its oldest sample is exactly 10 s old.
  EXPECT_EQ(excess(imuStream(2001), kWindow), 0u);
}

TEST(ImuBacklogPolicy, ThirtyMinuteOutageKeepsExactlyTheLastTenSeconds)
{
  const std::vector<double> stamps = imuStream(kOutageSamples);
  const std::size_t dropped = excess(stamps, kWindow);
  // 10 s at 200 Hz is 2000 intervals: the 2001 samples from newest - 10 s to the newest stay.
  EXPECT_EQ(dropped, kOutageSamples - 2001);
  ASSERT_GT(dropped, 0u);
  EXPECT_EQ(stamps[dropped], stamps.back() - kWindow);
  EXPECT_GT(stamps.back() - stamps[dropped - 1], kWindow);
}

TEST(ImuBacklogPolicy, TrimAfterEveryPushHoldsTheQueueAtTheWindow)
{
  // The node's use over a 30 min outage with no scan queued: push one sample, drop the excess from the front.
  std::deque<double> queue;
  std::size_t reads = 0;
  std::size_t largest = 0;
  std::size_t total = 0;
  const auto stamp_at = [&queue, &reads](std::size_t i) {
    ++reads;
    return queue[i];
  };
  for (std::size_t i = 0; i < kOutageSamples; ++i) {
    queue.push_back(imuStamp(i));
    const std::size_t dropped = fast_lio::imuBacklogExcess(queue.size(), stamp_at, queue.back(), kWindow, kInf);
    for (std::size_t k = 0; k < dropped; ++k) {
      queue.pop_front();
    }
    total += dropped;
    largest = std::max(largest, queue.size());
  }
  EXPECT_EQ(queue.size(), 2001u);
  EXPECT_EQ(largest, 2001u);
  EXPECT_EQ(total, kOutageSamples - 2001);
  EXPECT_EQ(queue.front(), queue.back() - kWindow);
  // Each push reads one stamp per dropped sample plus the one that stops the scan.
  EXPECT_EQ(reads, total + kOutageSamples - 1);
}

TEST(ImuBacklogPolicy, ExactlyAtCapIsKept)
{
  // 1 Hz, 0 .. 12 s: the 10 s cap falls exactly on the sample at 2 s.
  std::vector<double> stamps;
  for (int second = 0; second <= 12; ++second) {
    stamps.push_back(kEpoch + second);
  }
  EXPECT_EQ(excess(stamps, kWindow), 2u);  // 12 s and 11 s old go, exactly 10 s old stays
  // A window that is not a whole number of seconds (quarter seconds are exact).
  const std::vector<double> quarters{kEpoch, kEpoch + 0.25, kEpoch + 0.5, kEpoch + 0.75};
  EXPECT_EQ(excess(quarters, 0.5), 1u);  // 0.75 s old goes, exactly 0.5 s old stays
  EXPECT_EQ(excess(quarters, 0.75), 0u);
  EXPECT_EQ(excess(quarters, 0.7499), 1u);
}

TEST(ImuBacklogPolicy, OldestQueuedFrameGuardKeepsWhatAQueuedScanNeeds)
{
  const std::vector<double> stamps = imuStream(kOutageSamples);
  const double newest = stamps.back();
  // A scan that began 20 s before the newest sample is still queued (waiting for its IMU): everything from its
  // begin on stays, although the window alone keeps only the last 10 s.
  const double scan_begin = newest - 20.0;
  const std::size_t dropped = excess(stamps, kWindow, scan_begin);
  EXPECT_EQ(dropped, kOutageSamples - 4001);
  ASSERT_GT(dropped, 0u);
  EXPECT_EQ(stamps[dropped], scan_begin);  // stamped exactly at the scan's begin: kept
  EXPECT_LT(stamps[dropped - 1], scan_begin);
  // A scan that began inside the window changes nothing: the window binds first.
  EXPECT_EQ(excess(stamps, kWindow, newest - 5.0), kOutageSamples - 2001);
  // A scan that began at or before the oldest sample keeps the whole queue.
  EXPECT_EQ(excess(stamps, kWindow, stamps.front()), 0u);
  EXPECT_EQ(excess(stamps, kWindow, stamps.front() - 1.0), 0u);
  // No scan queued: +infinity protects nothing.
  EXPECT_EQ(excess(stamps, kWindow, kInf), kOutageSamples - 2001);
}

TEST(ImuBacklogPolicy, ZeroOrNegativeWindowDisablesTheBound)
{
  const std::vector<double> stamps = imuStream(30 * 200 + 1);  // 30 s
  ASSERT_EQ(excess(stamps, kWindow), 20u * 200u);
  EXPECT_EQ(excess(stamps, 0.0), 0u);
  EXPECT_EQ(excess(stamps, -0.0), 0u);
  EXPECT_EQ(excess(stamps, -1.0), 0u);
  EXPECT_EQ(excess(stamps, -kInf), 0u);
}

TEST(ImuBacklogPolicy, NonFiniteStampsDropNothing)
{
  // Every sample but the newest is 30 s old, so finite stamps would all go.
  for (const double garbage : {kNaN, kInf, -kInf}) {
    EXPECT_EQ(excess({garbage, kEpoch, kEpoch + 30.0}, kWindow), 0u) << garbage;  // the oldest
    EXPECT_EQ(excess({kEpoch, garbage, kEpoch + 30.0}, kWindow), 0u) << garbage;  // inside the old prefix
    EXPECT_EQ(excess({kEpoch, kEpoch + 1.0, garbage}, kWindow), 0u) << garbage;   // the newest
  }
}

TEST(ImuBacklogPolicy, NonFiniteNewestWindowOrGuardDropsNothing)
{
  const std::vector<double> stamps{kEpoch, kEpoch + 1.0, kEpoch + 30.0};
  const auto stamp_at = [&stamps](std::size_t i) {
    return stamps.at(i);
  };
  ASSERT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kWindow, kInf), 2u);
  for (const double garbage : {kNaN, kInf, -kInf}) {
    EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, garbage, kWindow, kInf), 0u) << garbage;
  }
  EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kNaN, kInf), 0u);
  EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kInf, kInf), 0u);
  // +infinity is the no-scan-queued guard; NaN and -infinity are garbage.
  EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kWindow, kNaN), 0u);
  EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kWindow, -kInf), 0u);
}

TEST(ImuBacklogPolicy, NeverDropsTheNewestSample)
{
  // Measured against a newest stamp ten hours ahead, every sample is old, the newest one included.
  std::size_t highest_read = 0;
  const auto stamp_at = [&highest_read](std::size_t i) {
    highest_read = std::max(highest_read, i);
    return kEpoch + static_cast<double>(i);
  };
  EXPECT_EQ(fast_lio::imuBacklogExcess(3, stamp_at, kEpoch + 36000.0, kWindow, kInf), 2u);
  EXPECT_EQ(highest_read, 1u);  // the newest (index 2) is not even read
  EXPECT_EQ(fast_lio::imuBacklogExcess(2, stamp_at, kEpoch + 36000.0, 1e-9, kInf), 1u);
  // A tiny window drops everything older than the newest, and nothing stamped with it.
  EXPECT_EQ(excess({kEpoch, kEpoch + 1.0, kEpoch + 2.0}, 1e-9), 2u);
  EXPECT_EQ(excess({kEpoch + 2.0, kEpoch + 2.0, kEpoch + 2.0}, 1e-9), 0u);
}

TEST(ImuBacklogPolicy, OutOfOrderStampsNeverDropInsideTheWindow)
{
  // The node's queue is in stamp order (enqueue_imu_msg rejects late samples); the rule for any other order: the
  // trim is a prefix that ends at the first sample it must keep, and an older sample behind that one waits until
  // it reaches the front.
  EXPECT_EQ(excess({kEpoch, kEpoch + 1.0, kEpoch + 100.0, kEpoch + 2.0, kEpoch + 3.0, kEpoch + 105.0}, kWindow), 2u);
  // A newest stamp that rolled back behind the queue drops nothing ahead of it.
  EXPECT_EQ(excess({kEpoch + 100.0, kEpoch + 101.0, kEpoch + 102.0, kEpoch + 50.0}, kWindow), 0u);
  // A sample stamped ahead of the newest, at the front, holds the queue until the newest passes it.
  EXPECT_EQ(excess({kEpoch + 200.0, kEpoch, kEpoch + 1.0, kEpoch + 105.0}, kWindow), 0u);
}

TEST(ImuBacklogPolicy, ReadsOnlyTheSamplesItDropsPlusOne)
{
  // A stamp behind the first kept sample is never read, garbage or not.
  const std::vector<double> stamps{kEpoch, kEpoch + 1.0, kEpoch + 25.0, kNaN, kEpoch + 30.0};
  std::size_t reads = 0;
  const auto stamp_at = [&stamps, &reads](std::size_t i) {
    ++reads;
    return stamps.at(i);
  };
  EXPECT_EQ(fast_lio::imuBacklogExcess(stamps.size(), stamp_at, stamps.back(), kWindow, kInf), 2u);
  EXPECT_EQ(reads, 3u);
}

// A fixed queue for constant evaluation: ages 20, 10, 5 and 0 s behind the newest.
struct FixedQueue
{
  std::array<double, 4> stamps;
  constexpr double operator()(std::size_t i) const { return stamps[i]; }
};
constexpr FixedQueue kFixedQueue{{0.0, 10.0, 15.0, 20.0}};

TEST(ImuBacklogPolicy, TheDecisionIsConstexpr)
{
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 10.0, kInf) == 1u, "exactly 10 s old stays");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 9.5, kInf) == 2u, "10 s old goes past 9.5 s");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 10.0, 0.0) == 0u, "a scan from 0 s keeps all");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 0.0, kInf) == 0u, "0 disables the bound");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 1e-9, kInf) == 3u, "the newest stays");
  static_assert(fast_lio::imuBacklogFinite(0.0) && !fast_lio::imuBacklogFinite(kNaN), "NaN is not finite");
  static_assert(!fast_lio::imuBacklogFinite(kInf) && !fast_lio::imuBacklogFinite(-kInf), "nor are infinities");
  // Garbage drops nothing in a constant expression too: no NaN reaches an ordered comparison.
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, kNaN, 10.0, kInf) == 0u, "NaN newest");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, kNaN, kInf) == 0u, "NaN window");
  static_assert(fast_lio::imuBacklogExcess(4, kFixedQueue, 20.0, 10.0, kNaN) == 0u, "NaN guard");
  static_assert(fast_lio::imuBacklogExcess(4, FixedQueue{{0.0, kNaN, 15.0, 20.0}}, 20.0, 10.0, kInf) == 0u,
                "NaN stamp");
  SUCCEED();
}

}  // namespace

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <utility>
#include <vector>

#include "delayed_map_insertion.hpp"
#include "ivox/ivox.hpp"
#include "laser_mapping_test_hooks.hpp"
#include "laser_mapping_test_types.hpp"

extern lio_ivox::IVox<PointType> ikdtree;
extern PointCloudXYZI::Ptr feats_down_body;
extern fast_lio::DelayedMapInsertion<PointVector> map_insert_pending;
extern bool async_map_en;
extern bool prepareScanCorrespondenceStorage(std::size_t);
extern void startMapWorker();
extern void stopMapWorker();
extern void dispatchMapAdd(PointVector &, PointVector &);
extern bool rebuildLocalMapStorage(int &);

namespace
{
PointVector pointAt(float x)
{
  PointType point{};
  point.x = x;
  return PointVector{point};
}

// The map as every case starts it: one original point at x = 1, a one-point scan at
// x = 50 to rebuild from, and no delayed batch left over from an earlier case.
void prepareMap(bool asyncMap)
{
  async_map_en = asyncMap;
  map_insert_pending.reset();
  ikdtree.Init(0.5f, 26, 50, 100);
  auto original = pointAt(1.0f);
  ikdtree.Build(original);
  feats_down_body->points = pointAt(50.0f);
}

// Stages `delayScans` scans' batches at x = firstX, firstX + 1, ...; each waits for
// its turn, so none comes out.
void stageHeldBatches(float firstX, int delayScans)
{
  for (int scan = 0; scan < delayScans; ++scan) {
    EXPECT_FALSE(map_insert_pending.stage(pointAt(firstX + static_cast<float>(scan)), {}, delayScans).has_value())
      << "scan " << scan;
  }
}

// After a reset: only the scan at x = 50 is in the map, nothing is pending, and the
// first batch staged afterwards is the first to come out (at once when there is no
// delay), so neither the discarded delayed batches nor an in-flight one resurfaces.
void expectRebuiltWithoutOldHistory(int delayScans)
{
  EXPECT_EQ(ikdtree.size(), 1);
  EXPECT_EQ(map_insert_pending.pendingSize(), 0u);
  PointVector nearest;
  std::vector<float> distances;
  const auto rebuilt = pointAt(50.0f);
  ikdtree.Nearest_Search(rebuilt.front(), 1, nearest, distances);
  ASSERT_EQ(nearest.size(), 1u);
  EXPECT_EQ(nearest.front().x, 50.0f);
  stageHeldBatches(60.0f, delayScans);
  const auto next = map_insert_pending.stage(pointAt(70.0f), {}, delayScans);
  ASSERT_TRUE(next.has_value());
  ASSERT_EQ(next->first.size(), 1u);
  EXPECT_EQ(next->first.front().x, delayScans > 0 ? 60.0f : 70.0f);
}

// A reset while the real worker is inserting a batch blocks until that batch is in the
// map, and only then discards the delayed batches and rebuilds.
void expectResetWaitsForTheWorker(int delayScans)
{
  using namespace std::chrono_literals;
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  std::promise<void> resetEntered;
  std::atomic<bool> discarded{false};
  prepareMap(true);
  ASSERT_TRUE(prepareScanCorrespondenceStorage(1));
  stageHeldBatches(20.0f, delayScans);
  ASSERT_EQ(map_insert_pending.pendingSize(), static_cast<std::size_t>(delayScans));

  fast_lio_test::beforeMapAdd = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  };
  fast_lio_test::beforeResetJoin = [&] {
    resetEntered.set_value();
  };
  fast_lio_test::afterPendingReset = [&] {
    discarded.store(true);
  };
  startMapWorker();
  auto inFlight = pointAt(30.0f);
  PointVector direct;
  dispatchMapAdd(inFlight, direct);
  bool workerEntered;
  {
    std::unique_lock<std::mutex> lock(mutex);
    workerEntered = cv.wait_for(lock, 5s, [&] { return entered; });
  }
  auto reset = std::async(std::launch::async, [&] {
    int before = 0;
    const bool kept = rebuildLocalMapStorage(before);
    return std::make_pair(before, kept);
  });
  const auto resetArrived = resetEntered.get_future().wait_for(5s);
  const auto resetWhileBlocked = reset.wait_for(50ms);
  const bool discardedWhileBlocked = discarded.load();
  // Always release and join before assertions can abort the fixture.
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();
  const auto result = reset.get();
  stopMapWorker();
  fast_lio_test::beforeMapAdd = {};
  fast_lio_test::beforeResetJoin = {};
  fast_lio_test::afterPendingReset = {};

  ASSERT_TRUE(workerEntered);
  EXPECT_EQ(resetArrived, std::future_status::ready);
  EXPECT_EQ(resetWhileBlocked, std::future_status::timeout);
  EXPECT_FALSE(discardedWhileBlocked);
  EXPECT_TRUE(discarded.load());
  EXPECT_EQ(result.first, 2);  // original + completed real worker batch, before reset
  EXPECT_FALSE(result.second);
  expectRebuiltWithoutOldHistory(delayScans);
}

TEST(AsyncMapReset, RealWorkerMustFinishBeforeDiscardingDelayedHistory)
{
  expectResetWaitsForTheWorker(1);
}

TEST(AsyncMapReset, WithNoDelayTheResetStillWaitsForTheInFlightBatch)
{
  expectResetWaitsForTheWorker(0);
}

TEST(AsyncMapReset, EveryBatchOfAMultiScanDelayIsDiscarded)
{
  expectResetWaitsForTheWorker(3);
}

// FLIO_ASYNC_MAP=0: map_incremental inserts on the scan thread, so there is no worker to
// wait for; the reset still discards the delayed batches before it rebuilds.
void expectResetWithoutTheWorker(int delayScans)
{
  prepareMap(false);
  ASSERT_TRUE(prepareScanCorrespondenceStorage(1));
  stageHeldBatches(20.0f, delayScans);
  ASSERT_EQ(map_insert_pending.pendingSize(), static_cast<std::size_t>(delayScans));

  bool resetEntered = false;
  std::size_t pendingAtDiscard = 1;
  fast_lio_test::beforeResetJoin = [&] {
    resetEntered = true;
  };
  fast_lio_test::afterPendingReset = [&] {
    pendingAtDiscard = map_insert_pending.pendingSize();
  };
  int before = 0;
  const bool kept = rebuildLocalMapStorage(before);
  fast_lio_test::beforeResetJoin = {};
  fast_lio_test::afterPendingReset = {};

  EXPECT_TRUE(resetEntered);
  EXPECT_EQ(pendingAtDiscard, 0u);
  EXPECT_EQ(before, 1);  // the original point only: nothing was in flight
  EXPECT_FALSE(kept);
  expectRebuiltWithoutOldHistory(delayScans);
}

TEST(AsyncMapReset, WithoutTheWorkerTheResetDiscardsDelayedHistoryAndRebuilds)
{
  expectResetWithoutTheWorker(1);
}

TEST(AsyncMapReset, WithoutTheWorkerOrADelayTheResetStillRebuilds)
{
  expectResetWithoutTheWorker(0);
}

TEST(AsyncMapReset, WithoutTheWorkerEveryBatchOfAMultiScanDelayIsDiscarded)
{
  expectResetWithoutTheWorker(3);
}
}  // namespace

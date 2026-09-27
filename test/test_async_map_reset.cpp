#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <utility>

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

TEST(AsyncMapReset, RealWorkerMustFinishBeforeDiscardingDelayedHistory)
{
  using namespace std::chrono_literals;
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool release = false;
  std::promise<void> resetEntered;
  std::atomic<bool> discarded{false};
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
  async_map_en = true;
  ikdtree.Init(0.5f, 26, 50, 100);
  auto original = pointAt(1.0f);
  ikdtree.Build(original);
  ASSERT_TRUE(prepareScanCorrespondenceStorage(1));
  feats_down_body->points = pointAt(50.0f);
  ASSERT_FALSE(map_insert_pending.stage(pointAt(20.0f), {}, 1).has_value());

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
  EXPECT_EQ(ikdtree.size(), 1);
  EXPECT_EQ(map_insert_pending.pendingSize(), 0u);
  PointVector nearest;
  std::vector<float> distances;
  const auto rebuilt = pointAt(50.0f);
  ikdtree.Nearest_Search(rebuilt.front(), 1, nearest, distances);
  ASSERT_EQ(nearest.size(), 1u);
  EXPECT_EQ(nearest.front().x, 50.0f);
  EXPECT_FALSE(map_insert_pending.stage(pointAt(60.0f), {}, 1).has_value());
  const auto next = map_insert_pending.stage(pointAt(70.0f), {}, 1);
  ASSERT_TRUE(next.has_value());
  ASSERT_EQ(next->first.size(), 1u);
  EXPECT_EQ(next->first.front().x, 60.0f);  // neither the old delayed nor in-flight batch resurfaces
}
}  // namespace

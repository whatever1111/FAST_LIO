#include <pcl/point_types.h>

#include <gtest/gtest.h>
#include <vector>

#include "delayed_map_insertion.hpp"
#include "ivox/ivox.hpp"

namespace
{
using Map = lio_ivox::IVox<pcl::PointXYZ>;
using Cloud = Map::PointVector;
Cloud point(float x)
{
  pcl::PointXYZ p;
  p.x = x;
  p.y = p.z = 0.0f;
  return {p};
}
void insert(Map & map, fast_lio::DelayedMapInsertion<Cloud>::Batch & batch)
{
  map.Add_Points(batch.first, true);
  map.Add_Points(batch.second, false);
}
bool contains(Map & map, float x)
{
  Cloud nearest;
  std::vector<float> distances;
  map.Nearest_Search(point(x).front(), 1, nearest, distances);
  return !nearest.empty() && distances.front() < 1e-6f;
}
}  // namespace

TEST(DelayedMapInsertion, ZeroOneAndManyScansPreservePairedBatchOrder)
{
  for (int delay : {0, 1, 4}) {
    fast_lio::DelayedMapInsertion<std::vector<int>> queue;
    for (int frame = 0; frame < 12; ++frame) {
      const auto ready = queue.stage({frame, frame + 100}, {-frame, frame + 200}, delay);
      if (frame < delay) {
        EXPECT_FALSE(ready);
      } else {
        ASSERT_TRUE(ready);
        const int expected = frame - delay;
        EXPECT_EQ(ready->first, (std::vector<int>{expected, expected + 100}));
        EXPECT_EQ(ready->second, (std::vector<int>{-expected, expected + 200}));
      }
    }
    EXPECT_EQ(queue.pendingSize(), static_cast<std::size_t>(delay));
    queue.reset();
    EXPECT_EQ(queue.pendingSize(), 0u);
    queue.reset();
  }
}

TEST(DelayedMapInsertion, EmptyFramesStillCountAsScans)
{
  fast_lio::DelayedMapInsertion<std::vector<int>> queue;
  EXPECT_FALSE(queue.stage({}, {}, 1));
  auto ready = queue.stage({1}, {2}, 1);
  ASSERT_TRUE(ready);
  EXPECT_TRUE(ready->first.empty());
  EXPECT_TRUE(ready->second.empty());
  ready = queue.stage({}, {}, 1);
  ASSERT_TRUE(ready);
  EXPECT_EQ(ready->first, std::vector<int>{1});
  EXPECT_EQ(ready->second, std::vector<int>{2});
}

TEST(DelayedMapInsertion, RebuildOrClearLiveCannotFlushOldCoordinatesAfterReset)
{
  for (bool pinned : {false, true}) {
    Map map;
    map.Init(0.3f, 26, 50, 1000);
    map.set_downsample_param(0.3f);
    auto prior = point(-20);
    auto oldLive = point(10);
    if (pinned)
      map.AddPinnedPoints(prior);
    map.Add_Points(oldLive, true);
    fast_lio::DelayedMapInsertion<Cloud> queue;
    EXPECT_FALSE(queue.stage(point(11), point(12), 2));
    EXPECT_FALSE(queue.stage(point(13), point(14), 2));
    // Production joins any active add before this single-threaded boundary.
    queue.reset();
    auto seed = point(100);
    if (pinned) {
      map.clearLive();
      map.Add_Points(seed, true);
    } else {
      map.Build(seed);
    }
    EXPECT_FALSE(queue.stage(point(101), point(102), 2));
    EXPECT_FALSE(queue.stage(point(103), point(104), 2));
    auto ready = queue.stage(point(105), point(106), 2);
    ASSERT_TRUE(ready);
    insert(map, *ready);
    EXPECT_TRUE(contains(map, 100));
    EXPECT_TRUE(contains(map, 101));
    EXPECT_TRUE(contains(map, 102));
    for (float old : {10, 11, 12, 13, 14})
      EXPECT_FALSE(contains(map, old));
    EXPECT_EQ(map.pinnedVoxels(), pinned ? 1u : 0u);
    if (pinned)
      EXPECT_TRUE(contains(map, -20));
  }
}

TEST(DelayedMapInsertion, ControlRebuildWithoutQueueResetWouldReinsertOldWorldBatch)
{
  Map map;
  map.Init(0.3f, 26, 50, 1000);
  map.set_downsample_param(0.3f);
  fast_lio::DelayedMapInsertion<Cloud> queue;
  EXPECT_FALSE(queue.stage(point(11), point(12), 1));
  auto seed = point(100);
  map.Build(seed);
  // Control only: deliberately omit the production reset to expose why a
  // backend rebuild alone cannot invalidate world-frame points held elsewhere.
  auto ready = queue.stage(point(101), point(102), 1);
  ASSERT_TRUE(ready);
  insert(map, *ready);
  EXPECT_TRUE(contains(map, 11));
  EXPECT_TRUE(contains(map, 12));
}

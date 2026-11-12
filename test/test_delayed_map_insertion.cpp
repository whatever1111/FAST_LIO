#include <pcl/point_types.h>

#include <cstring>
#include <gtest/gtest.h>
#include <thread>
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

TEST(DelayedMapInsertion, ProvenanceLeavesWithItsOwnBatch)
{
  fast_lio::DelayedMapInsertion<std::vector<int>, int> queue;
  auto immediate = queue.stage({1}, {-1}, 0, 1);
  ASSERT_TRUE(immediate);
  EXPECT_EQ(immediate->provenance, 1);
  for (int frame = 2; frame < 10; ++frame) {
    const auto ready = queue.stage({frame}, {-frame}, 4, frame);
    if (frame < 6) {
      EXPECT_FALSE(ready);
    } else {
      ASSERT_TRUE(ready);
      EXPECT_EQ(ready->provenance, frame - 4);
      EXPECT_EQ(ready->first, std::vector<int>{ready->provenance});
      EXPECT_EQ(ready->second, std::vector<int>{-ready->provenance});
    }
  }
  queue.reset();
  EXPECT_EQ(queue.pendingSize(), 0u);
  auto fresh = queue.stage({20}, {-20}, 0, 20);
  ASSERT_TRUE(fresh);
  EXPECT_EQ(fresh->provenance, 20);
}

TEST(DelayedMapInsertion, ADelayedBatchIsLabelledWithItsStagingScan)
{
  Map map;
  map.Init(0.3f, 26, 50, 1000);
  ASSERT_TRUE(map.enablePointIdentity());
  fast_lio::DelayedMapInsertion<Cloud, fast_lio::MapInsertProvenance> queue;
  for (int frame = 1; frame <= 6; ++frame) {
    auto provenance = fast_lio::beginScanProvenance(true, frame, 1);
    provenance.noteDownsampled(frame + 10);
    auto ready = queue.stage(point(frame * 10), {}, 2, std::move(provenance));
    if (frame <= 2) {
      EXPECT_FALSE(ready);
      continue;
    }
    ASSERT_TRUE(ready);
    map.stagePointProvenance(std::move(ready->provenance));
    map.addStagedPoints(ready->first, ready->second);
    Cloud near;
    std::vector<float> distances;
    std::vector<fast_lio::MapPointIdentity> ids;
    EXPECT_EQ(map.Nearest_Search_Identified(point((frame - 2) * 10).front(), 1, near, distances, ids),
              fast_lio::MapIdentityStatus::kOk);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0].origin, fast_lio::MapPointOrigin::kScanIncremental);
    EXPECT_EQ(ids[0].frame_seq, static_cast<std::uint32_t>(frame - 2));
    EXPECT_EQ(ids[0].point_index, static_cast<std::uint32_t>(frame + 8));
  }
  EXPECT_FALSE(contains(map, 50));
  EXPECT_FALSE(contains(map, 60));
}

TEST(DelayedMapInsertion, ResetDropsProvenanceWithItsPoints)
{
  Map map;
  map.Init(0.3f, 26, 50, 1000);
  ASSERT_TRUE(map.enablePointIdentity());
  fast_lio::DelayedMapInsertion<Cloud, fast_lio::MapInsertProvenance> queue;
  auto provenance = fast_lio::beginScanProvenance(true, 8, 1);
  provenance.noteDownsampled(0);
  EXPECT_FALSE(queue.stage(point(80), {}, 2, std::move(provenance)));
  queue.reset();
  // The rebuild replaces what the map held at the seed's own position, identity included.
  auto replaced = point(90);
  map.Add_Points(replaced, true, fast_lio::seedLabel(7));
  auto seed = point(90);
  map.Build(seed, fast_lio::seedLabel(9));
  for (int frame = 10; frame <= 12; ++frame) {
    auto p = fast_lio::beginScanProvenance(true, frame, 1);
    p.noteDownsampled(4);
    auto ready = queue.stage(point(frame * 10), {}, 2, std::move(p));
    if (ready) {
      map.stagePointProvenance(std::move(ready->provenance));
      map.addStagedPoints(ready->first, ready->second);
    }
  }
  EXPECT_FALSE(contains(map, 80));
  for (int frame : {9, 10}) {
    Cloud near;
    std::vector<float> distances;
    std::vector<fast_lio::MapPointIdentity> ids;
    map.Nearest_Search_Identified(point(frame * 10).front(), 1, near, distances, ids);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0].frame_seq, static_cast<std::uint32_t>(frame));
    EXPECT_EQ(ids[0].origin,
              frame == 9 ? fast_lio::MapPointOrigin::kScanSeed : fast_lio::MapPointOrigin::kScanIncremental);
  }
}

TEST(DelayedMapInsertion, TheWorkerThreadAddsTheSameIdentities)
{
  Map sync, worker;
  sync.Init(0.3f, 26, 50, 1000);
  worker.Init(0.3f, 26, 50, 1000);
  ASSERT_TRUE(sync.enablePointIdentity());
  ASSERT_TRUE(worker.enablePointIdentity());
  for (int frame = 1; frame <= 8; ++frame) {
    auto p = fast_lio::beginScanProvenance(true, frame, 1);
    p.noteDownsampled(frame + 10);
    p.noteDirect(frame + 20);
    auto other = p;
    sync.stagePointProvenance(std::move(p));
    worker.stagePointProvenance(std::move(other));
    auto ds = point(frame * 10), direct = point(frame * 10 + 1);
    sync.addStagedPoints(ds, direct);
    std::thread thread([&] { worker.addStagedPoints(ds, direct); });
    thread.join();
  }
  for (int frame = 1; frame <= 8; ++frame) {
    Cloud a, b;
    std::vector<float> da, db;
    std::vector<fast_lio::MapPointIdentity> ia, ib;
    sync.Nearest_Search_Identified(point(frame * 10).front(), 5, a, da, ia);
    worker.Nearest_Search_Identified(point(frame * 10).front(), 5, b, db, ib);
    ASSERT_EQ(a.size(), b.size());
    ASSERT_FALSE(a.empty());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])), 0);
    EXPECT_EQ(da, db);
    ASSERT_EQ(ia.size(), ib.size());
    EXPECT_EQ(std::memcmp(ia.data(), ib.data(), ia.size() * sizeof(ia[0])), 0);
    // Each list is labelled from its own index array: the direct point is not given the downsampled one's.
    ASSERT_EQ(ib.size(), 1u);
    EXPECT_EQ(ib[0].point_index, static_cast<std::uint32_t>(frame + 10));
    for (const Map * map : {&sync, &worker}) {
      map->Nearest_Search_Identified(point(frame * 10 + 1).front(), 1, a, da, ia);
      ASSERT_EQ(ia.size(), 1u);
      EXPECT_EQ(ia[0].origin, fast_lio::MapPointOrigin::kScanIncremental);
      EXPECT_EQ(ia[0].frame_seq, static_cast<std::uint32_t>(frame));
      EXPECT_EQ(ia[0].point_index, static_cast<std::uint32_t>(frame + 20));
    }
  }
}

TEST(MapProvenance, AFrameSequenceOutsideTheRecordWidthIsUnlabelled)
{
  const auto max = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(fast_lio::mapFrameSeq(0));
  EXPECT_EQ(fast_lio::mapFrameSeq(1), 1u);
  EXPECT_EQ(fast_lio::mapFrameSeq(max), max);
  const auto overflow = static_cast<std::uint64_t>(max) + 1;
  EXPECT_FALSE(fast_lio::mapFrameSeq(overflow));
  EXPECT_EQ(fast_lio::seedLabel(overflow).origin, fast_lio::MapPointOrigin::kUnlabelled);
  EXPECT_EQ(fast_lio::beginScanProvenance(true, overflow, 100).frame_seq, 0u);
  auto disabled = fast_lio::beginScanProvenance(false, 12, 100);
  disabled.noteDownsampled(4);
  disabled.noteDirect(9);
  EXPECT_TRUE(disabled.downsampled_index.empty());
  EXPECT_TRUE(disabled.direct_index.empty());
  EXPECT_EQ(disabled.downsampled_index.capacity(), 0u);
  EXPECT_EQ(disabled.direct_index.capacity(), 0u);
}

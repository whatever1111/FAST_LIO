// The prior-map tier: what the live run may and may not take away from it.

#include <pcl/point_types.h>

#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "ivox/ivox.hpp"

namespace
{
using Ivox = lio_ivox::IVox<pcl::PointXYZ>;
using Cloud = Ivox::PointVector;

// Points on a 1 m lattice starting at `x0`, one per voxel at res 0.5 / fine grid 0.25.
Cloud lattice(float x0, int n)
{
  Cloud pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    pcl::PointXYZ p;
    p.x = x0 + static_cast<float>(i);
    p.y = 0.0f;
    p.z = 0.0f;
    pts.push_back(p);
  }
  return pts;
}

Ivox makeIvox(std::size_t max_voxels)
{
  Ivox v;
  v.Init(0.5f, 26, 50, max_voxels);
  v.set_downsample_param(0.25f);
  return v;
}
}  // namespace

TEST(IvoxLayering, PinnedVoxelsAreNeverEvicted)
{
  Ivox v = makeIvox(10);
  Cloud prior = lattice(0.0f, 6);
  v.AddPinnedPoints(prior);
  EXPECT_EQ(v.pinnedVoxels(), 6u);

  // Far more live points than the capacity: eviction has to come from the live tier only.
  Cloud live = lattice(100.0f, 40);
  v.Add_Points(live, true);

  EXPECT_GT(v.evictedVoxels(), 0u) << "capacity was never exceeded — test does not test anything";
  EXPECT_EQ(v.pinnedVoxels(), 6u);
  for (const auto & p : prior) {
    Cloud near;
    std::vector<float> sqdist;
    v.Nearest_Search(p, 1, near, sqdist);
    ASSERT_EQ(near.size(), 1u);
    EXPECT_NEAR(sqdist[0], 0.0f, 1e-6f) << "pinned point at x=" << p.x << " was evicted";
  }
}

TEST(IvoxLayering, ClearLiveKeepsThePinnedTier)
{
  Ivox v = makeIvox(1000);
  Cloud prior = lattice(0.0f, 5);
  Cloud live = lattice(50.0f, 7);
  v.AddPinnedPoints(prior);
  v.Add_Points(live, true);
  ASSERT_EQ(v.size(), 12);

  v.clearLive();

  EXPECT_EQ(v.size(), 5) << "clearLive must drop exactly the live points";
  EXPECT_EQ(v.pinnedVoxels(), 5u);
  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(prior.front(), 1, near, sqdist);
  ASSERT_EQ(near.size(), 1u);
  EXPECT_NEAR(sqdist[0], 0.0f, 1e-6f);
  v.Nearest_Search(live.front(), 1, near, sqdist);
  EXPECT_TRUE(near.empty() || sqdist[0] > 1.0f) << "a live point survived clearLive";
}

TEST(IvoxLayering, ClearLiveFreesTheFineGridSoTheSameCellCanBeRelearned)
{
  // The global fine-grid dedup drops a point whose cell is taken. If clearLive forgot to
  // release the cells it dropped, the map could never re-learn the area it just cleared.
  Ivox v = makeIvox(1000);
  Cloud live = lattice(50.0f, 4);
  v.Add_Points(live, true);
  ASSERT_EQ(v.size(), 4);
  v.clearLive();
  ASSERT_EQ(v.size(), 0);
  v.Add_Points(live, true);
  EXPECT_EQ(v.size(), 4) << "the cleared cells stayed occupied";
}

TEST(IvoxLayering, BuildClearsEverythingIncludingPinned)
{
  // ikd-Tree's Build deletes the tree first; the guard's recovery depends on that meaning
  // the same thing on both backends.
  Ivox v = makeIvox(1000);
  Cloud prior = lattice(0.0f, 5);
  v.AddPinnedPoints(prior);
  Cloud fresh = lattice(200.0f, 3);
  v.Build(fresh);

  EXPECT_EQ(v.size(), 3);
  EXPECT_EQ(v.pinnedVoxels(), 0u);
  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(prior.front(), 1, near, sqdist);
  EXPECT_TRUE(near.empty() || sqdist[0] > 1.0f) << "Build kept a pinned point";
}

TEST(IvoxLayering, LivePointInAPinnedVoxelDoesNotMakeItEvictable)
{
  Ivox v = makeIvox(4);
  Cloud prior = lattice(0.0f, 2);
  v.AddPinnedPoints(prior);
  // A live point 0.25 m away shares the 0.5 m voxel but not the 0.25 m fine cell.
  Cloud overlapping;
  pcl::PointXYZ q;
  q.x = prior.front().x + 0.25f;
  q.y = 0.0f;
  q.z = 0.0f;
  overlapping.push_back(q);
  v.Add_Points(overlapping, true);
  EXPECT_EQ(v.pinnedVoxels(), 2u);

  Cloud flood = lattice(100.0f, 30);  // flood well past capacity
  v.Add_Points(flood, true);
  EXPECT_EQ(v.pinnedVoxels(), 2u);
  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(prior.front(), 1, near, sqdist);
  ASSERT_EQ(near.size(), 1u);
  EXPECT_NEAR(sqdist[0], 0.0f, 1e-6f);
}

TEST(IvoxLayering, WithoutPinningTheBackendBehavesExactlyAsBefore)
{
  // The no-prior-map case: nothing is pinned, so capacity, eviction and Build are unchanged.
  Ivox v = makeIvox(8);
  Cloud live = lattice(0.0f, 20);
  v.Add_Points(live, true);
  EXPECT_EQ(v.pinnedVoxels(), 0u);
  EXPECT_LE(v.numVoxels(), 8u);  // eviction precedes insertion, never exceeding capacity
  EXPECT_GT(v.evictedVoxels(), 0u);
  v.clearLive();
  EXPECT_EQ(v.size(), 0) << "with nothing pinned, clearLive empties the map";
}

TEST(IvoxLayering, ClearPinnedDropsThePriorTierAndKeepsTheLiveOne)
{
  // Reinstalling a prior map at a better alignment: the old tier has to leave whole,
  // and what this run learned has to stay.
  Ivox v = makeIvox(1000);
  Cloud prior = lattice(0.0f, 5);
  Cloud live = lattice(100.0f, 7);
  v.AddPinnedPoints(prior);
  v.Add_Points(live, true);
  const int total = v.size();
  ASSERT_EQ(total, 12);
  EXPECT_EQ(v.pinnedVoxels(), 5u);

  v.clearPinned();
  EXPECT_EQ(v.pinnedVoxels(), 0u);
  EXPECT_EQ(v.size(), 7) << "the live points survive the prior tier leaving";
  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(live.front(), 1, near, sqdist);
  ASSERT_EQ(near.size(), 1u);
  EXPECT_NEAR(sqdist[0], 0.0f, 1e-6f);
}

TEST(IvoxLayering, ClearPinnedFreesTheFineGridSoTheSameCellsCanBeRelearned)
{
  // The dedup grid is keep-first: if clearPinned left the prior's fine boxes marked,
  // the reinstalled map could not occupy the cells it just left.
  Ivox v = makeIvox(1000);
  Cloud prior = lattice(0.0f, 5);
  v.AddPinnedPoints(prior);
  v.clearPinned();
  EXPECT_EQ(v.size(), 0);
  Cloud again = lattice(0.0f, 5);
  EXPECT_EQ(v.AddPinnedPoints(again), 5) << "every cell the cleared tier held is free again";
  EXPECT_EQ(v.pinnedVoxels(), 5u);
}

TEST(IvoxLayering, ClearPinnedOnAMapWithNoPriorTierIsANoOp)
{
  Ivox v = makeIvox(1000);
  Cloud live = lattice(0.0f, 6);
  v.Add_Points(live, true);
  const int before = v.size();
  v.clearPinned();
  EXPECT_EQ(v.size(), before);
  EXPECT_EQ(v.pinnedVoxels(), 0u);
}

TEST(IvoxLayering, APriorMapFillsWhatTheLiveMapHasNotAlreadyClaimed)
{
  // The dedup grid is first-point-wins, and that applies to a pinned add too: where this
  // run has already put a point, the delivered map does not displace it. That is the right
  // way round — those live points came from the scans being matched right now — but it
  // means the pinned tier is not a complete copy of the delivered map, and the count that
  // came back is the only honest report of how much of it went in.
  Ivox v = makeIvox(1000);
  Cloud live = lattice(0.0f, 3);
  v.Add_Points(live, true);
  Cloud prior = lattice(0.0f, 5);  // the same 3 cells, plus 2 the live map never saw
  EXPECT_EQ(v.AddPinnedPoints(prior), 2) << "only the cells the live map left free are taken";
  EXPECT_EQ(v.pinnedVoxels(), 2u);
  EXPECT_EQ(v.size(), 5);

  v.clearPinned();
  EXPECT_EQ(v.pinnedVoxels(), 0u);
  EXPECT_EQ(v.size(), 3) << "clearing the prior tier leaves this run's own map standing";
}

TEST(IvoxLayering, ReinstallingAPriorMapAtANewAlignmentReplacesItWhole)
{
  // What the relocalization path does: install, then install again somewhere else. The
  // second install must not leave the first one in the map beside it.
  Ivox v = makeIvox(1000);
  Cloud first = lattice(0.0f, 4);
  v.AddPinnedPoints(first);
  EXPECT_EQ(v.pinnedVoxels(), 4u);

  v.clearPinned();
  Cloud second = lattice(50.0f, 4);  // the same map, 50 m away
  EXPECT_EQ(v.AddPinnedPoints(second), 4);
  EXPECT_EQ(v.pinnedVoxels(), 4u);
  EXPECT_EQ(v.size(), 4) << "one copy of the prior map, at the new alignment";

  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(first.front(), 1, near, sqdist);
  EXPECT_TRUE(near.empty() || sqdist[0] > 1.0f) << "nothing left where the old alignment put it";
}

TEST(IvoxLayering, AllPinnedCapacityRejectsLiveWithoutCountingOrClaimingCells)
{
  Ivox v = makeIvox(2);
  auto prior = lattice(0, 2);
  ASSERT_EQ(v.AddPinnedPoints(prior), 2);
  auto live = lattice(10, 1);
  EXPECT_EQ(v.Add_Points(live, true), 0);
  EXPECT_EQ(v.Add_Points(live, false), 0);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.numVoxels(), 2u);
  EXPECT_EQ(v.evictedVoxels(), 0u);
  v.clearPinned();
  EXPECT_EQ(v.Add_Points(live, true), 1);  // rejected point never claimed its fine cell
  EXPECT_EQ(v.size(), 1);
}

TEST(IvoxLayering, OversizedPinnedAdditionOrReplacementPreservesTheOldMap)
{
  Ivox v = makeIvox(2);
  auto prior = lattice(0, 1);
  auto live = lattice(10, 1);
  v.AddPinnedPoints(prior);
  v.Add_Points(live, true);
  auto tooLarge = lattice(20, 3);
  EXPECT_FALSE(v.canAddPinnedPoints(tooLarge));
  EXPECT_FALSE(v.canReplacePinnedPoints(tooLarge));
  EXPECT_THROW(v.AddPinnedPoints(tooLarge), std::length_error);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.pinnedVoxels(), 1u);
  EXPECT_EQ(v.evictedVoxels(), 0u);
  for (const auto & p : {prior.front(), live.front()}) {
    Cloud nearest;
    std::vector<float> distances;
    v.Nearest_Search(p, 1, nearest, distances);
    ASSERT_EQ(nearest.size(), 1u);
    EXPECT_FLOAT_EQ(distances[0], 0.0f);
  }
  auto replacement = lattice(20, 2);
  EXPECT_FALSE(v.canAddPinnedPoints(replacement));  // union with the existing pinned voxel
  EXPECT_TRUE(v.canReplacePinnedPoints(replacement));
}

TEST(IvoxLayering, MixedCapacityEvictsOnlyTheOldestLiveAndCountsRetainedPoints)
{
  Ivox v = makeIvox(2);
  auto pinned = lattice(0, 1);
  auto first = lattice(10, 1);
  auto second = lattice(20, 1);
  v.AddPinnedPoints(pinned);
  EXPECT_EQ(v.Add_Points(first, true), 1);
  EXPECT_EQ(v.Add_Points(second, true), 1);
  EXPECT_EQ(v.numVoxels(), 2u);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.evictedVoxels(), 1u);
  for (const auto & p : {pinned.front(), second.front()}) {
    Cloud nearest;
    std::vector<float> distances;
    v.Nearest_Search(p, 1, nearest, distances);
    ASSERT_FALSE(nearest.empty());
    EXPECT_FLOAT_EQ(distances[0], 0.0f);
  }
  Cloud nearest;
  std::vector<float> distances;
  v.Nearest_Search(first.front(), 1, nearest, distances);
  EXPECT_TRUE(nearest.empty());
}

TEST(IvoxLayering, ZeroCapacityKeepsExistingDefaultAndExtremePriorPreflightIsReadOnly)
{
  Ivox v = makeIvox(0);
  EXPECT_EQ(v.maxVoxels(), 5000000u);
  auto prior = lattice(0, 1);
  v.AddPinnedPoints(prior);
  auto huge = lattice(std::numeric_limits<float>::max(), 1);
  EXPECT_FALSE(v.canAddPinnedPoints(huge));
  EXPECT_FALSE(v.canReplacePinnedPoints(huge));
  EXPECT_THROW(v.AddPinnedPoints(huge), std::length_error);
  EXPECT_EQ(v.Add_Points(huge, false), 0);
  EXPECT_EQ(v.Add_Points(huge, true), 0);
  EXPECT_EQ(v.size(), 1);
}

TEST(IvoxLayering, PinnedPreflightCountsUniqueCoarseBucketsNotPoints)
{
  Ivox v = makeIvox(2);
  auto first = lattice(0, 1);
  v.AddPinnedPoints(first);
  Cloud duplicates(100, first.front());
  auto second = lattice(1, 1);
  duplicates.push_back(second.front());
  EXPECT_TRUE(v.canAddPinnedPoints(duplicates));
  EXPECT_EQ(v.AddPinnedPoints(duplicates), 1);
  EXPECT_EQ(v.pinnedVoxels(), 2u);
  EXPECT_EQ(v.size(), 2);
}

namespace
{
using Origin = fast_lio::MapPointOrigin;
using Status = fast_lio::MapIdentityStatus;
using Ids = std::vector<fast_lio::MapPointIdentity>;

Ids identitiesAt(const Ivox & map, const pcl::PointXYZ & point, int k = 1)
{
  Cloud near;
  std::vector<float> distances;
  Ids ids;
  EXPECT_EQ(map.Nearest_Search_Identified(point, k, near, distances, ids), Status::kOk);
  EXPECT_EQ(ids.size(), near.size());
  return ids;
}

void expectIdentity(const fast_lio::MapPointIdentity & id,
                    Origin origin,
                    std::uint32_t frame,
                    std::uint32_t index,
                    std::uint32_t generation)
{
  EXPECT_EQ(id.origin, origin);
  EXPECT_EQ(id.frame_seq, frame);
  EXPECT_EQ(id.point_index, index);
  EXPECT_EQ(id.generation, generation);
  for (auto byte : id.reserved)
    EXPECT_EQ(byte, 0);
}
}  // namespace

TEST(IvoxIdentity, AdmittedPointsCarryTheirLabel)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto points = lattice(0, 3);
  const std::vector<std::uint32_t> index{4, 9, 12};
  ASSERT_EQ(map.Add_Points(points, true, {Origin::kScanIncremental, 7, &index}), 3);
  for (std::size_t i = 0; i < points.size(); ++i) {
    Cloud near;
    std::vector<float> distances;
    Ids ids;
    ASSERT_EQ(map.Nearest_Search_Identified(points[i], 1, near, distances, ids), Status::kOk);
    ASSERT_EQ(near.size(), 1u);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_FLOAT_EQ(near[0].x, points[i].x);
    expectIdentity(ids[0], Origin::kScanIncremental, 7, index[i], map.mapGeneration());
  }
}

TEST(IvoxIdentity, RejectedOffersLeaveNoIdentity)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto points = lattice(0, 2);
  points.insert(points.begin() + 1, points.front());
  points.push_back(lattice(std::numeric_limits<float>::quiet_NaN(), 1).front());
  points.push_back(lattice(std::numeric_limits<float>::max(), 1).front());
  const std::vector<std::uint32_t> index{4, 9, 12, 15, 18};
  EXPECT_EQ(map.Add_Points(points, true, {Origin::kScanIncremental, 7, &index}), 2);
  EXPECT_EQ(map.size(), 2);
  for (const auto & point : {points[0], points[2]}) {
    const auto ids = identitiesAt(map, point, 20);
    ASSERT_FALSE(ids.empty());
    for (const auto & id : ids) {
      EXPECT_NE(id.point_index, 9u);
      EXPECT_NE(id.point_index, 15u);
      EXPECT_NE(id.point_index, 18u);
    }
  }
  EXPECT_EQ(map.unlabelledPoints(), 0u);
  // With no evictable voxel, a rejected live offer cannot acquire an identity either.
  auto pinned = makeIvox(1);
  ASSERT_TRUE(pinned.enablePointIdentity());
  auto prior = lattice(0, 1);
  pinned.AddPinnedPoints(prior);
  auto live = lattice(10, 1);
  EXPECT_EQ(pinned.Add_Points(live, false, fast_lio::seedLabel(8)), 0);
  EXPECT_TRUE(identitiesAt(pinned, live.front()).empty());
}

TEST(IvoxIdentity, EvictedVoxelTakesItsIdentitiesAway)
{
  auto map = makeIvox(2);
  ASSERT_TRUE(map.enablePointIdentity());
  auto a = lattice(0, 1), b = lattice(10, 1), c = lattice(20, 1);
  map.Add_Points(a, true, fast_lio::seedLabel(1));
  map.Add_Points(b, true, fast_lio::seedLabel(2));
  map.Add_Points(c, true, fast_lio::seedLabel(3));
  EXPECT_TRUE(identitiesAt(map, a.front()).empty());
  ASSERT_EQ(identitiesAt(map, b.front()).size(), 1u);
  EXPECT_EQ(identitiesAt(map, b.front())[0].frame_seq, 2u);
  ASSERT_EQ(identitiesAt(map, c.front()).size(), 1u);
  EXPECT_EQ(identitiesAt(map, c.front())[0].frame_seq, 3u);
  map.Add_Points(a, true, fast_lio::seedLabel(4));
  const auto ids = identitiesAt(map, a.front());
  ASSERT_EQ(ids.size(), 1u);
  EXPECT_EQ(ids[0].frame_seq, 4u);
}

TEST(IvoxIdentity, ClearLiveKeepsThePinnedTierWithItsIdentities)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto prior = lattice(0, 2), live = lattice(50, 1), mixed = lattice(0.25f, 1);
  map.AddPinnedPoints(prior);
  map.Add_Points(live, true, {Origin::kScanIncremental, 5});
  map.Add_Points(mixed, false, {Origin::kScanIncremental, 6});
  const auto g = map.mapGeneration();
  map.clearLive();
  EXPECT_EQ(map.mapGeneration(), g + 1);
  for (std::size_t i = 0; i < prior.size(); ++i) {
    const auto ids = identitiesAt(map, prior[i]);
    ASSERT_EQ(ids.size(), 1u);
    expectIdentity(ids[0], Origin::kPrior, 1, i, g);
  }
  const auto ids = identitiesAt(map, mixed.front());
  ASSERT_EQ(ids.size(), 1u);
  expectIdentity(ids[0], Origin::kScanIncremental, 6, 0, g);
  EXPECT_TRUE(identitiesAt(map, live.front()).empty());
  // A voxel the reset removed starts its identities afresh when a later scan refills it.
  map.Add_Points(live, true, {Origin::kScanIncremental, 7});
  const auto refilled = identitiesAt(map, live.front());
  ASSERT_EQ(refilled.size(), 1u);
  expectIdentity(refilled[0], Origin::kScanIncremental, 7, 0, g + 1);
}

TEST(IvoxIdentity, EveryResetAdvancesTheGeneration)
{
  const auto sequence = [](bool enabled) {
    auto map = makeIvox(1);
    if (enabled)
      EXPECT_TRUE(map.enablePointIdentity());
    std::vector<std::uint32_t> values{map.mapGeneration()};
    map.clear();
    values.push_back(map.mapGeneration());
    auto points = lattice(0, 1);
    map.Build(points, fast_lio::seedLabel(1));
    values.push_back(map.mapGeneration());
    map.clearLive();
    values.push_back(map.mapGeneration());
    map.clearPinned();
    values.push_back(map.mapGeneration());
    map.Add_Points(points, true);
    auto other = lattice(10, 1);
    map.Add_Points(other, true);
    EXPECT_EQ(map.evictedVoxels(), 1u);
    values.push_back(map.mapGeneration());
    map.Add_Points(other, false);
    values.push_back(map.mapGeneration());
    return values;
  };
  const std::vector<std::uint32_t> expected{0, 1, 2, 3, 3, 3, 3};
  EXPECT_EQ(sequence(false), expected);
  EXPECT_EQ(sequence(true), expected);
}

TEST(IvoxIdentity, SearchIsBitIdenticalWithAndWithoutIdentity)
{
  auto off = makeIvox(128), on = makeIvox(128);
  ASSERT_TRUE(on.enablePointIdentity());
  std::uint32_t state = 17;
  const auto coordinate = [&state]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state % 1000) * 0.003f;
  };
  Cloud queries;
  for (int batch = 1; batch <= 8; ++batch) {
    Cloud points;
    for (int i = 0; i < 400; ++i) {
      pcl::PointXYZ p;
      p.x = coordinate();
      p.y = coordinate();
      p.z = coordinate();
      points.push_back(p);
      if (i < 40)
        queries.push_back(p);
    }
    EXPECT_EQ(off.Add_Points(points, batch % 2 == 0),
              on.Add_Points(points, batch % 2 == 0, fast_lio::seedLabel(batch)));
  }
  // Equal distances and exact duplicates exercise ordering and storage-position resolution.
  auto ties = lattice(0, 1);
  ties.front().y = ties.front().z = 0.1f;
  auto left = ties.front(), right = ties.front();
  left.x = 0.25f;
  right.x = -0.25f;
  ties = {left, right, left};
  off.Add_Points(ties, false);
  on.Add_Points(ties, false, fast_lio::seedLabel(9));
  queries.push_back(ties.front());
  auto center = left;
  center.x = 0.0f;
  queries.push_back(center);
  ASSERT_TRUE(off.BeginEntryView());
  ASSERT_TRUE(on.BeginEntryView());
  auto later = lattice(0.1f, 2);
  off.Add_Points(later, false);
  on.Add_Points(later, false, fast_lio::seedLabel(10));
  const auto sameBytes = [](const auto & a, const auto & b) {
    ASSERT_EQ(a.size(), b.size());
    if (!a.empty())
      EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])), 0);
  };
  for (const auto & query : queries) {
    Cloud a, b, identified;
    std::vector<float> da, db, di;
    Ids ids;
    off.Nearest_Search(query, 5, a, da);
    on.Nearest_Search(query, 5, b, db);
    ASSERT_EQ(on.Nearest_Search_Identified(query, 5, identified, di, ids), Status::kOk);
    sameBytes(a, b);
    sameBytes(a, identified);
    sameBytes(da, db);
    sameBytes(da, di);
    ASSERT_EQ(ids.size(), identified.size());
    off.Nearest_Search_View(query, 5, a, da);
    on.Nearest_Search_View(query, 5, b, db);
    sameBytes(a, b);
    sameBytes(da, db);
  }
  const auto duplicateIds = identitiesAt(on, left, 2);
  ASSERT_EQ(duplicateIds.size(), 2u);
  EXPECT_EQ(duplicateIds[0].frame_seq, 9u);
  EXPECT_EQ(duplicateIds[1].frame_seq, 9u);
  EXPECT_NE(duplicateIds[0].point_index, duplicateIds[1].point_index);
  EXPECT_EQ(off.size(), on.size());
  EXPECT_EQ(off.numVoxels(), on.numVoxels());
  EXPECT_EQ(off.evictedVoxels(), on.evictedVoxels());
  EXPECT_GT(on.evictedVoxels(), 0u);
}

TEST(IvoxIdentity, DisabledSearchSaysSoAndReturnsNone)
{
  auto map = makeIvox(100);
  auto points = lattice(0, 2);
  map.Add_Points(points, true);
  Cloud near, expected;
  std::vector<float> distances, expectedDistances;
  Ids ids(1);
  EXPECT_EQ(map.Nearest_Search_Identified(points[0], 2, near, distances, ids), Status::kDisabled);
  EXPECT_TRUE(ids.empty());
  map.Nearest_Search(points[0], 2, expected, expectedDistances);
  ASSERT_EQ(near.size(), expected.size());
  EXPECT_EQ(std::memcmp(near.data(), expected.data(), near.size() * sizeof(near[0])), 0);
  EXPECT_EQ(distances, expectedDistances);
}

TEST(IvoxIdentity, TurnsOnOnlyOnAnEmptyMap)
{
  auto map = makeIvox(100);
  auto points = lattice(0, 1);
  map.Add_Points(points, true);
  EXPECT_FALSE(map.enablePointIdentity());
  EXPECT_FALSE(map.pointIdentityEnabled());
  map.clear();
  EXPECT_TRUE(map.enablePointIdentity());
  Cloud near;
  std::vector<float> distances;
  Ids ids(1);
  const auto invalid = lattice(std::numeric_limits<float>::quiet_NaN(), 1).front();
  EXPECT_EQ(map.Nearest_Search_Identified(invalid, 5, near, distances, ids), Status::kOk);
  EXPECT_TRUE(near.empty());
  EXPECT_TRUE(distances.empty());
  EXPECT_TRUE(ids.empty());
}

TEST(IvoxIdentity, UnlabelledAdditionsAreMarkedNotGuessed)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto points = lattice(0, 2), malformed = lattice(10, 2);
  map.Add_Points(points, true);
  const std::vector<std::uint32_t> wrongSize{3};
  map.Add_Points(malformed, true, {Origin::kScanIncremental, 8, &wrongSize});
  for (const auto & p : {points[0], points[1], malformed[0], malformed[1]}) {
    const auto ids = identitiesAt(map, p);
    ASSERT_EQ(ids.size(), 1u);
    expectIdentity(ids[0], Origin::kUnlabelled, 0, 0, map.mapGeneration());
  }
  EXPECT_EQ(map.unlabelledPoints(), 4u);
}

TEST(IvoxIdentity, MergedPriorBuildSplitsTheOrigins)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto points = lattice(0, 5);
  map.Build(points, fast_lio::seedLabel(3, 2));
  EXPECT_EQ(map.priorBatchSeq(), 1u);
  for (std::uint32_t i = 0; i < points.size(); ++i) {
    const auto ids = identitiesAt(map, points[i]);
    ASSERT_EQ(ids.size(), 1u);
    expectIdentity(
      ids[0], i < 2 ? Origin::kPrior : Origin::kScanSeed, i < 2 ? 1 : 3, i < 2 ? i : i - 2, map.mapGeneration());
  }
}

TEST(IvoxIdentity, AReinstalledPriorMapIsANewBatch)
{
  auto map = makeIvox(100);
  ASSERT_TRUE(map.enablePointIdentity());
  auto points = lattice(0, 2);
  map.AddPinnedPoints(points);
  const auto g = map.mapGeneration();
  EXPECT_EQ(map.priorBatchSeq(), 1u);
  map.clearPinned();
  map.AddPinnedPoints(points);
  EXPECT_EQ(map.priorBatchSeq(), 2u);
  EXPECT_EQ(map.mapGeneration(), g);
  for (std::uint32_t i = 0; i < points.size(); ++i) {
    const auto ids = identitiesAt(map, points[i]);
    ASSERT_EQ(ids.size(), 1u);
    expectIdentity(ids[0], Origin::kPrior, 2, i, g);
  }
}

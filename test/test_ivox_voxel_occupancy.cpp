// How many points a voxel ends up holding — the relation the per-voxel reserve is sized from —
// and which of them the entry view shows.

#include <pcl/point_types.h>

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <vector>

#include "ivox/ivox.hpp"

namespace
{
using Ivox = lio_ivox::IVox<pcl::PointXYZ>;
using Cloud = Ivox::PointVector;

Cloud cloudOf(const std::vector<std::array<float, 3>> & xyz)
{
  Cloud pts;
  pts.reserve(xyz.size());
  for (const auto & c : xyz) {
    pcl::PointXYZ p;
    p.x = c[0];
    p.y = c[1];
    p.z = c[2];
    pts.push_back(p);
  }
  return pts;
}

// The deployed m20 geometry: the NN voxel is the dedup box.
Ivox alignedGrid(int per_voxel_cap = 50)
{
  Ivox v;
  v.Init(0.3f, 26, per_voxel_cap, 1000);
  v.set_downsample_param(0.3f);
  return v;
}

// 1 m voxels over a 0.1 m dedup grid: a voxel holds several points, so a view can cut one short.
Ivox fineGrid(std::size_t max_voxels = 1000)
{
  Ivox v;
  v.Init(1.0f, 26, 50, max_voxels);
  v.set_downsample_param(0.1f);
  return v;
}

pcl::PointXYZ pointAt(float x, float y, float z)
{
  return cloudOf({{x, y, z}}).front();
}

// The x of every point a search finds near `query` (up to 20 of them), sorted.
std::vector<float> nearestXs(const Ivox & v, const pcl::PointXYZ & query, bool in_view)
{
  Cloud near;
  std::vector<float> sqdist;
  if (in_view)
    v.Nearest_Search_View(query, 20, near, sqdist);
  else
    v.Nearest_Search(query, 20, near, sqdist);
  std::vector<float> xs;
  for (const auto & p : near)
    xs.push_back(p.x);
  std::sort(xs.begin(), xs.end());
  return xs;
}
}  // namespace

TEST(IvoxVoxelOccupancy, PointsPastTheCapLeaveNoIdentity)
{
  Ivox v = alignedGrid(2);
  ASSERT_TRUE(v.enablePointIdentity());
  auto points = cloudOf({{0.01f, 0, 0}, {0.05f, 0, 0}, {0.1f, 0, 0}, {0.15f, 0, 0}});
  EXPECT_EQ(v.Add_Points(points, false, fast_lio::seedLabel(7)), 2);
  Cloud near;
  std::vector<float> distances;
  std::vector<fast_lio::MapPointIdentity> ids;
  EXPECT_EQ(v.Nearest_Search_Identified(points[0], 10, near, distances, ids), fast_lio::MapIdentityStatus::kOk);
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_EQ(ids[0].point_index, 0u);
  EXPECT_EQ(ids[1].point_index, 1u);
  EXPECT_EQ(v.unlabelledPoints(), 0u);
}

TEST(IvoxEntryView, TheViewSearchIsUnchangedByIdentity)
{
  auto off = fineGrid(), on = fineGrid();
  ASSERT_TRUE(on.enablePointIdentity());
  auto entry = cloudOf({{0.05f, 0.05f, 0.05f}, {0.55f, 0.05f, 0.05f}});
  off.Add_Points(entry, true);
  on.Add_Points(entry, true, fast_lio::seedLabel(1));
  ASSERT_TRUE(off.BeginEntryView());
  ASSERT_TRUE(on.BeginEntryView());
  auto later = cloudOf({{0.25f, 0.05f, 0.05f}, {1.25f, 0.05f, 0.05f}});
  auto direct = cloudOf({{0.35f, 0.05f, 0.05f}});
  off.Add_Points(later, true);
  on.Add_Points(later, true, fast_lio::seedLabel(2));
  off.Add_Points(direct, false);
  on.Add_Points(direct, false, fast_lio::seedLabel(3));
  const auto query = pointAt(0.6f, 0.05f, 0.05f);
  EXPECT_EQ(nearestXs(on, query, true), nearestXs(off, query, true));
  EXPECT_EQ(nearestXs(on, query, true), (std::vector<float>{0.05f, 0.55f}));
  EXPECT_EQ(nearestXs(on, query, false), nearestXs(off, query, false));
}

TEST(IvoxVoxelOccupancy, AlignedGridKeepsOnePointPerVoxel)
{
  Ivox v = alignedGrid();
  // Four points inside one 0.3 m voxel, which is also one dedup box.
  Cloud pts = cloudOf({{0.01f, 0.01f, 0.01f}, {0.11f, 0.05f, 0.02f}, {0.29f, 0.29f, 0.29f}, {0.15f, 0.15f, 0.15f}});
  v.Add_Points(pts, true);

  EXPECT_EQ(v.numVoxels(), 1u);
  EXPECT_EQ(v.size(), 1) << "the dedup grid is the voxel grid — a voxel cannot hold two points here";
}

TEST(IvoxVoxelOccupancy, ACoarserVoxelHoldsOnePointPerFineBox)
{
  Ivox v;
  v.Init(0.5f, 26, 50, 1000);
  v.set_downsample_param(0.25f);  // 2x2x2 fine boxes per voxel

  std::vector<std::array<float, 3>> xyz;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        xyz.push_back({0.05f + 0.25f * static_cast<float>(i),
                       0.05f + 0.25f * static_cast<float>(j),
                       0.05f + 0.25f * static_cast<float>(k)});
  Cloud pts = cloudOf(xyz);
  v.Add_Points(pts, true);

  EXPECT_EQ(v.numVoxels(), 1u);
  EXPECT_EQ(v.size(), 8) << "all eight fine boxes of the voxel are distinct and all should be kept";
}

TEST(IvoxVoxelOccupancy, TheReserveIsNotACapOnPointsThatSkipTheDedup)
{
  Ivox v = alignedGrid();
  // downsample_on = false is the map_incremental path for points the dedup must not thin.
  Cloud pts = cloudOf({{0.01f, 0.01f, 0.01f}, {0.11f, 0.05f, 0.02f}, {0.29f, 0.29f, 0.29f}});
  v.Add_Points(pts, false);

  EXPECT_EQ(v.numVoxels(), 1u);
  ASSERT_EQ(v.size(), 3) << "a voxel must still grow past its initial buffer";

  // The last one is retrievable, so growing the buffer did not lose or alias anything.
  Cloud near;
  std::vector<float> sqdist;
  v.Nearest_Search(pts.back(), 1, near, sqdist);
  ASSERT_EQ(near.size(), 1u);
  EXPECT_NEAR(sqdist[0], 0.0f, 1e-6f);
}

TEST(IvoxVoxelOccupancy, PerVoxelCapStillBoundsAVoxel)
{
  Ivox v = alignedGrid(3);
  std::vector<std::array<float, 3>> xyz;
  for (int i = 0; i < 10; ++i)
    xyz.push_back({0.01f + 0.02f * static_cast<float>(i), 0.01f, 0.01f});
  Cloud pts = cloudOf(xyz);
  v.Add_Points(pts, false);

  EXPECT_EQ(v.numVoxels(), 1u);
  EXPECT_EQ(v.size(), 3) << "per_voxel_cap is what bounds a voxel, not the reserve";
}

TEST(IvoxVoxelOccupancy, SignedCoordinatesRemainRetrievableBeyondHashOverflowBoundary)
{
  Ivox v = alignedGrid();
  // x keys 30/-31 and z keys 26/-27 already overflow the old signed products.
  // Exercise both coarse-grid lookup and fine-grid dedup through the public API.
  Cloud pts = cloudOf({{0.01f, 0.01f, 0.01f},
                       {9.01f, 0.01f, 0.01f},
                       {-9.01f, 0.01f, 0.01f},
                       {0.01f, 1400.01f, 0.01f},
                       {0.01f, -1400.01f, 0.01f},
                       {0.01f, 0.01f, 7.81f},
                       {0.01f, 0.01f, -7.81f},
                       {300000.0f, -300000.0f, 300000.0f}});
  EXPECT_EQ(v.Add_Points(pts, true), static_cast<int>(pts.size()));
  EXPECT_EQ(v.Add_Points(pts, true), 0);
  ASSERT_EQ(v.size(), static_cast<int>(pts.size()));
  for (const auto & p : pts) {
    Cloud near;
    std::vector<float> sqdist;
    v.Nearest_Search(p, 1, near, sqdist);
    ASSERT_EQ(near.size(), 1u);
    ASSERT_EQ(sqdist.size(), 1u);
    EXPECT_FLOAT_EQ(sqdist[0], 0.0f);
    EXPECT_FLOAT_EQ(near[0].x, p.x);
    EXPECT_FLOAT_EQ(near[0].y, p.y);
    EXPECT_FLOAT_EQ(near[0].z, p.z);
  }
}

TEST(IvoxVoxelOccupancy, HashSupportsKeysNearBothEndsOfTheIntegerRange)
{
  Ivox v;
  v.Init(1.0f, 26, 50, 1000);
  v.set_downsample_param(1.0f);
  // Largest float below INT32_MAX; its negative and neighbor +/-1 remain
  // representable as int. This test does not exercise out-of-range float casts.
  constexpr float kEdge = 2147483520.0f;
  Cloud pts = cloudOf({{kEdge, 0.0f, 0.0f},
                       {-kEdge, 0.0f, 0.0f},
                       {0.0f, kEdge, 0.0f},
                       {0.0f, -kEdge, 0.0f},
                       {0.0f, 0.0f, kEdge},
                       {0.0f, 0.0f, -kEdge}});
  ASSERT_EQ(v.Add_Points(pts, true), static_cast<int>(pts.size()));
  for (const auto & p : pts) {
    Cloud near;
    std::vector<float> sqdist;
    v.Nearest_Search(p, 1, near, sqdist);
    ASSERT_EQ(near.size(), 1u);
    ASSERT_EQ(sqdist.size(), 1u);
    EXPECT_FLOAT_EQ(sqdist[0], 0.0f);
  }
  EXPECT_EQ(v.Add_Points(pts, true), 0);
  v.clearLive();
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.Add_Points(pts, true), static_cast<int>(pts.size()));
}

TEST(IvoxVoxelOccupancy, FalsePointsCannotEraseAnotherCoarseVoxelsFineOwnership)
{
  for (int removal = 0; removal < 3; ++removal) {
    Ivox v;
    v.Init(0.5f, 26, 50, removal == 2 ? 2 : 100);
    v.set_downsample_param(0.3f);
    auto owner = cloudOf({{0.49f, 0, 0}});   // coarse 0, fine 1
    auto direct = cloudOf({{0.51f, 0, 0}});  // coarse 1, same fine cell, no ownership
    if (removal == 0) {
      v.AddPinnedPoints(owner);
      v.Add_Points(direct, false);
      v.clearLive();
    } else if (removal == 1) {
      v.Add_Points(owner, true);
      v.Add_Points(direct, false);
      auto promote = cloudOf({{0.8f, 0, 0}});  // different fine cell promotes coarse 1
      v.AddPinnedPoints(promote);
      v.clearPinned();
    } else {
      v.Add_Points(direct, false);  // oldest live coarse voxel
      v.Add_Points(owner, true);
      auto newVoxel = cloudOf({{1.2f, 0, 0}});
      v.Add_Points(newVoxel, true);
    }
    auto duplicate = cloudOf({{0.52f, 0, 0}});
    EXPECT_EQ(v.Add_Points(duplicate, true), 0) << removal;
    Cloud near;
    std::vector<float> distance;
    v.Nearest_Search(owner.front(), 1, near, distance);
    ASSERT_FALSE(near.empty());
    EXPECT_FLOAT_EQ(distance[0], 0.0f);
  }
}

TEST(IvoxVoxelOccupancy, InvalidQueriesClearOutputsAndIntegerEdgeNeighborsDoNotOverflow)
{
  Ivox v;
  v.Init(1.0f, 26, 50, 10);
  v.set_downsample_param(1.0f);
  auto edges = cloudOf({{-2147483648.0f, 0, 0}, {2147483520.0f, 0, 0}});
  ASSERT_EQ(v.Add_Points(edges, true), 2);
  for (const auto & p : edges) {
    Cloud near;
    std::vector<float> distances;
    v.Nearest_Search(p, 1, near, distances);
    ASSERT_EQ(near.size(), 1u);
    EXPECT_FLOAT_EQ(distances[0], 0.0f);
  }
  for (float invalid : {2147483648.0f,
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    auto query = cloudOf({{invalid, 0, 0}});
    Cloud near = edges;
    std::vector<float> distances{1.0f};
    v.Nearest_Search(query.front(), 1, near, distances);
    EXPECT_TRUE(near.empty());
    EXPECT_TRUE(distances.empty());
    EXPECT_EQ(v.Add_Points(query, true), 0);
  }
}

TEST(IvoxEntryView, ShowsOnlyThePointsTheMapHeldWhenItOpened)
{
  Ivox v = fineGrid();
  Cloud entry = cloudOf({{0.05f, 0.05f, 0.05f}, {0.55f, 0.05f, 0.05f}});  // one voxel
  ASSERT_EQ(v.Add_Points(entry, true), 2);
  ASSERT_TRUE(v.BeginEntryView());
  EXPECT_TRUE(v.entryViewActive());
  // After the opening, both of map_incremental's paths: into the voxel the view holds, and into a new one.
  Cloud later = cloudOf({{0.25f, 0.05f, 0.05f}, {1.25f, 0.05f, 0.05f}});
  ASSERT_EQ(v.Add_Points(later, true), 2);
  Cloud later_raw = cloudOf({{0.35f, 0.05f, 0.05f}});
  ASSERT_EQ(v.Add_Points(later_raw, false), 1);

  const pcl::PointXYZ query = pointAt(0.6f, 0.05f, 0.05f);
  EXPECT_EQ(nearestXs(v, query, true), (std::vector<float>{0.05f, 0.55f}));
  EXPECT_EQ(nearestXs(v, query, false), (std::vector<float>{0.05f, 0.25f, 0.35f, 0.55f, 1.25f}));

  v.EndEntryView();
  EXPECT_FALSE(v.entryViewActive());
  EXPECT_EQ(nearestXs(v, query, true), nearestXs(v, query, false));
}

TEST(IvoxEntryView, WithNothingAddedSinceTheOpeningTheViewSearchIsTheMapSearch)
{
  Ivox v = fineGrid();
  std::vector<std::array<float, 3>> xyz;
  for (int i = 0; i < 30; ++i)
    for (int j = 0; j < 30; ++j)
      xyz.push_back({0.1f * static_cast<float>(i) + 0.03f,
                     0.1f * static_cast<float>(j) + 0.04f,
                     0.02f * static_cast<float>((i * j) % 5)});
  Cloud map = cloudOf(xyz);
  ASSERT_GT(v.Add_Points(map, true), 0);
  ASSERT_TRUE(v.BeginEntryView());
  // A point the dedup refuses changes no voxel, so it opens no entry either.
  Cloud refused = cloudOf({xyz.front()});
  ASSERT_EQ(v.Add_Points(refused, true), 0);

  for (int q = 0; q < 40; ++q) {
    const pcl::PointXYZ query = pointAt(0.07f * static_cast<float>(q), 0.05f * static_cast<float>(q), 0.03f);
    Cloud near_map, near_view;
    std::vector<float> d_map, d_view;
    v.Nearest_Search(query, 5, near_map, d_map);
    v.Nearest_Search_View(query, 5, near_view, d_view);
    ASSERT_EQ(near_view.size(), near_map.size()) << q;
    for (std::size_t i = 0; i < near_map.size(); ++i) {
      EXPECT_EQ(d_view[i], d_map[i]) << q;
      EXPECT_EQ(near_view[i].x, near_map[i].x) << q;
      EXPECT_EQ(near_view[i].y, near_map[i].y) << q;
      EXPECT_EQ(near_view[i].z, near_map[i].z) << q;
    }
  }
}

TEST(IvoxEntryView, AnEvictedVoxelLeavesTheViewAndComesBackOutsideIt)
{
  Ivox v = fineGrid(2);
  Cloud first = cloudOf({{0.5f, 0.5f, 0.5f}});  // the oldest voxel
  Cloud second = cloudOf({{5.5f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(first, true), 1);
  ASSERT_EQ(v.Add_Points(second, true), 1);
  ASSERT_TRUE(v.BeginEntryView());

  Cloud third = cloudOf({{10.5f, 0.5f, 0.5f}});  // a new voxel at capacity: the oldest goes
  ASSERT_EQ(v.Add_Points(third, true), 1);
  ASSERT_EQ(v.evictedVoxels(), 1u);
  EXPECT_TRUE(nearestXs(v, first.front(), true).empty()) << "evicted: gone from the view with the map";
  EXPECT_EQ(nearestXs(v, second.front(), true), (std::vector<float>{5.5f}));
  EXPECT_TRUE(nearestXs(v, third.front(), true).empty()) << "added after the opening";
  EXPECT_EQ(nearestXs(v, third.front(), false), (std::vector<float>{10.5f}));

  // The evicted voxel comes back with the very point it held at the opening: a voxel created after the opening.
  ASSERT_EQ(v.Add_Points(first, true), 1);
  EXPECT_TRUE(nearestXs(v, first.front(), true).empty());
  EXPECT_EQ(nearestXs(v, first.front(), false), (std::vector<float>{0.5f}));
}

TEST(IvoxEntryView, APerVoxelCapKeepsWhatEachVoxelHeldAtTheOpening)
{
  Ivox v;
  v.Init(1.0f, 26, 3, 1000);
  v.set_downsample_param(0.1f);
  // One voxel with room for one more point, and one already full.
  Cloud entry = cloudOf({{0.05f, 0.05f, 0.05f},
                         {0.25f, 0.05f, 0.05f},
                         {5.05f, 0.05f, 0.05f},
                         {5.25f, 0.05f, 0.05f},
                         {5.45f, 0.05f, 0.05f}});
  ASSERT_EQ(v.Add_Points(entry, true), 5);
  ASSERT_TRUE(v.BeginEntryView());

  // The last point of the first voxel, one over its cap, and one over the cap of the full voxel.
  Cloud later = cloudOf({{0.45f, 0.05f, 0.05f}, {0.65f, 0.05f, 0.05f}, {5.65f, 0.05f, 0.05f}});
  ASSERT_EQ(v.Add_Points(later, true), 1);

  const pcl::PointXYZ query = pointAt(0.3f, 0.05f, 0.05f);
  EXPECT_EQ(nearestXs(v, query, true), (std::vector<float>{0.05f, 0.25f}));
  EXPECT_EQ(nearestXs(v, query, false), (std::vector<float>{0.05f, 0.25f, 0.45f}));
  const pcl::PointXYZ query_full = pointAt(5.3f, 0.05f, 0.05f);
  EXPECT_EQ(nearestXs(v, query_full, true), (std::vector<float>{5.05f, 5.25f, 5.45f}));
  EXPECT_EQ(nearestXs(v, query_full, false), (std::vector<float>{5.05f, 5.25f, 5.45f}));
}

TEST(IvoxEntryView, ClearAndBuildReopenAnOpenViewOnTheNewMap)
{
  Ivox v = fineGrid();
  Cloud entry = cloudOf({{0.5f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(entry, true), 1);
  ASSERT_TRUE(v.BeginEntryView());

  v.clear();
  EXPECT_TRUE(v.entryViewActive());
  Cloud after_clear = cloudOf({{0.55f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(after_clear, true), 1);
  EXPECT_TRUE(nearestXs(v, entry.front(), true).empty()) << "reopened on the empty map";
  EXPECT_EQ(nearestXs(v, entry.front(), false), (std::vector<float>{0.55f}));

  Cloud built = cloudOf({{0.15f, 0.5f, 0.5f}, {1.15f, 0.5f, 0.5f}});
  v.Build(built);
  EXPECT_TRUE(v.entryViewActive());
  // After the build: into a built voxel, and into a new one.
  Cloud after_build = cloudOf({{0.35f, 0.5f, 0.5f}, {0.45f, 1.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(after_build, true), 2);
  const pcl::PointXYZ query = pointAt(0.3f, 0.5f, 0.5f);
  EXPECT_EQ(nearestXs(v, query, true), (std::vector<float>{0.15f, 1.15f})) << "the built points are the view";
  EXPECT_EQ(nearestXs(v, query, false), (std::vector<float>{0.15f, 0.35f, 0.45f, 1.15f}));
}

TEST(IvoxEntryView, DoesNotOpenOverAPinnedTier)
{
  Ivox v = fineGrid();
  Cloud prior = cloudOf({{0.5f, 0.5f, 0.5f}});
  ASSERT_EQ(v.AddPinnedPoints(prior), 1);
  Cloud live = cloudOf({{0.15f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(live, true), 1);

  EXPECT_FALSE(v.BeginEntryView());
  EXPECT_FALSE(v.entryViewActive());
  Cloud later = cloudOf({{0.25f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(later, true), 1);
  const pcl::PointXYZ query = pointAt(0.3f, 0.5f, 0.5f);
  EXPECT_EQ(nearestXs(v, query, false), (std::vector<float>{0.15f, 0.25f, 0.5f}));
  EXPECT_EQ(nearestXs(v, query, true), nearestXs(v, query, false)) << "no view: the view search is the map search";
}

TEST(IvoxEntryView, APinnedAdditionWhileOpenStaysOutOfTheViewAndIsReported)
{
  Ivox v = fineGrid();
  Cloud live = cloudOf({{0.15f, 0.5f, 0.5f}});
  ASSERT_EQ(v.Add_Points(live, true), 1);
  ASSERT_TRUE(v.BeginEntryView());
  EXPECT_FALSE(v.entryViewPinned());

  // Into the live voxel, which it pins, and into a new voxel.
  Cloud prior = cloudOf({{0.55f, 0.5f, 0.5f}, {0.5f, 1.5f, 0.5f}});
  ASSERT_EQ(v.AddPinnedPoints(prior), 2);
  EXPECT_TRUE(v.entryViewPinned());
  EXPECT_TRUE(v.entryViewActive());
  const pcl::PointXYZ query = pointAt(0.3f, 0.5f, 0.5f);
  EXPECT_EQ(nearestXs(v, query, true), (std::vector<float>{0.15f}));
  EXPECT_EQ(nearestXs(v, query, false), (std::vector<float>{0.15f, 0.5f, 0.55f}));

  v.EndEntryView();
  EXPECT_FALSE(v.entryViewPinned());
}

TEST(IvoxEntryView, CountsTheCallersFramesOnlyWhileOpen)
{
  Ivox v = fineGrid();
  v.countEntryViewFrame();
  EXPECT_EQ(v.entryViewFrames(), 0u);
  ASSERT_TRUE(v.BeginEntryView());
  v.countEntryViewFrame();
  v.countEntryViewFrame();
  EXPECT_EQ(v.entryViewFrames(), 2u);
  ASSERT_TRUE(v.BeginEntryView());  // reopening starts over
  EXPECT_EQ(v.entryViewFrames(), 0u);
  v.countEntryViewFrame();
  v.EndEntryView();
  EXPECT_EQ(v.entryViewFrames(), 0u);
}

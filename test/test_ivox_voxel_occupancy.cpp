// How many points a voxel ends up holding — the relation the per-voxel reserve is sized from.

#include <pcl/point_types.h>

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
}  // namespace

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

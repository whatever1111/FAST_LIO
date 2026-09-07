// How many points a voxel ends up holding — the relation the per-voxel reserve is sized from.

#include "ivox/ivox.hpp"

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include <array>
#include <vector>

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
  for (int i = 0; i < 10; ++i) xyz.push_back({0.01f + 0.02f * static_cast<float>(i), 0.01f, 0.01f});
  Cloud pts = cloudOf(xyz);
  v.Add_Points(pts, false);

  EXPECT_EQ(v.numVoxels(), 1u);
  EXPECT_EQ(v.size(), 3) << "per_voxel_cap is what bounds a voxel, not the reserve";
}

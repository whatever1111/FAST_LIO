// The front end's first local reference: the normal initial map build has to leave the map backend holding points.
// Both backends are checked in every build, whichever one the node is compiled with.

#include <pcl/point_types.h>

#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "map_backend_reference.hpp"

namespace
{
using PointType = pcl::PointXYZINormal;
using Ivox = lio_ivox::IVox<PointType>;
using KdTree = KD_TREE<PointType>;
using Points = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

PointType at(float x, float y, float z)
{
  PointType p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

// A scan's downsampled points on a 2 m x 2 m patch of ground, 0.1 m apart.
Points ground()
{
  Points points;
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 20; ++j)
      points.push_back(at(0.1f * static_cast<float>(i), 0.1f * static_cast<float>(j), 0.0f));
  }
  return points;
}

// Points iVox skips: non-finite coordinates, and positions whose voxel index leaves the int range at the node's
// map grid (0.5 m voxels, 0.05 m dedup grid).
Points unplaceable()
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  return Points{at(nan, 0.0f, 0.0f),
                at(0.0f, nan, 0.0f),
                at(0.0f, 0.0f, nan),
                at(inf, 0.0f, 0.0f),
                at(0.0f, -inf, 1.0f),
                at(2.0e8f, 0.0f, 0.0f),
                at(0.0f, -3.0e9f, 0.0f)};
}

void configure(Ivox * map)
{
  map->Init(0.5f, 26, 50, 5000000u);
  map->set_downsample_param(0.05f);
}
}  // namespace

TEST(MapBackendReference, IvoxHoldsPointsOnlyWhenTheBuildPlacedOne)
{
  Ivox map;
  configure(&map);
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map)) << "never built";
  Points none;
  map.Build(none);
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map)) << "built from nothing";
  Points rejected = unplaceable();
  map.Build(rejected);
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map)) << "built from points it skipped, every one";
  EXPECT_EQ(map.size(), 0);

  Points scan = ground();
  map.Build(scan);
  EXPECT_TRUE(fast_lio::mapBackendHoldsPoints(map));
  EXPECT_GT(map.size(), 0);

  // One point it can place among those it cannot is a map.
  Points mixed = unplaceable();
  mixed.push_back(at(1.0f, 2.0f, 0.5f));
  map.Build(mixed);
  EXPECT_TRUE(fast_lio::mapBackendHoldsPoints(map));
  EXPECT_EQ(map.size(), 1);

  // A build clears first: rebuilt from rejected points, the map that was there is gone.
  rejected = unplaceable();
  map.Build(rejected);
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map));
}

TEST(MapBackendReference, IkdTreeHoldsPointsOnlyWhenTheBuildHadOne)
{
  KdTree map;
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map)) << "never built";
  map.Build(Points());
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map)) << "built from nothing";
  map.Build(ground());
  EXPECT_TRUE(fast_lio::mapBackendHoldsPoints(map));
  EXPECT_GT(map.size(), 0);
  // A build deletes the tree first: rebuilt from nothing, nothing is left.
  map.Build(Points());
  EXPECT_FALSE(fast_lio::mapBackendHoldsPoints(map));
}

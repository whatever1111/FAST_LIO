// The prior-map tier: what the live run may and may not take away from it.

#include "ivox/ivox.hpp"

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include <vector>

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
  EXPECT_LE(v.numVoxels(), 9u);  // bounded by capacity (the check runs after each insert)
  EXPECT_GT(v.evictedVoxels(), 0u);
  v.clearLive();
  EXPECT_EQ(v.size(), 0) << "with nothing pinned, clearLive empties the map";
}

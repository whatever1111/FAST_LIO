// test_voxel_downsample.cpp — 一趟哈希占格降采样的行为约定。
// Behaviour contract for the one-pass hash-bucketed scan downsampler.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "voxel_downsample.hpp"

namespace
{

struct P
{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  int id = 0;
};

using Cloud = std::vector<P>;

Cloud run(fast_lio::VoxelGridHash & f, const Cloud & in)
{
  Cloud out;
  f.filter(in, out);
  return out;
}

fast_lio::VoxelGridHash filterWithLeaf(double leaf)
{
  fast_lio::VoxelGridHash f;
  f.setLeaf(leaf);
  return f;
}

}  // namespace

TEST(VoxelDownsample, EmptyInputGivesEmptyOutput)
{
  auto f = filterWithLeaf(0.15);
  EXPECT_TRUE(run(f, {}).empty());
}

TEST(VoxelDownsample, NonPositiveLeafPassesTheScanThrough)
{
  auto f = filterWithLeaf(0.0);
  const Cloud in{{0.01f, 0.0f, 0.0f, 1}, {0.02f, 0.0f, 0.0f, 2}};
  const Cloud out = run(f, in);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].id, 1);
  EXPECT_EQ(out[1].id, 2);
}

TEST(VoxelDownsample, SinglePointSurvivesUnchanged)
{
  auto f = filterWithLeaf(0.2);
  const Cloud out = run(f, {{1.234f, -5.0f, 0.5f, 7}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 7);
  EXPECT_FLOAT_EQ(out[0].x, 1.234f);
}

TEST(VoxelDownsample, KeepsTheMeasuredPointNearestTheCellCentre)
{
  auto f = filterWithLeaf(1.0);
  // Cell [0,1)^3, centre (0.5, 0.5, 0.5). The second point is nearer, and it wins even
  // though it arrives later; the survivor is a real input point, not an average.
  const Cloud in{{0.10f, 0.5f, 0.5f, 1}, {0.45f, 0.5f, 0.5f, 2}, {0.90f, 0.5f, 0.5f, 3}};
  const Cloud out = run(f, in);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 2);
  EXPECT_FLOAT_EQ(out[0].x, 0.45f);
}

TEST(VoxelDownsample, OneRepresentativePerOccupiedCellInFirstSeenOrder)
{
  auto f = filterWithLeaf(1.0);
  const Cloud in{
    {0.5f, 0.5f, 0.5f, 1},    // cell (0,0,0)
    {5.5f, 0.5f, 0.5f, 2},    // cell (5,0,0)
    {0.6f, 0.5f, 0.5f, 3},    // cell (0,0,0) again, farther from the centre
    {-1.5f, 0.5f, 0.5f, 4}};  // cell (-2,0,0)
  const Cloud out = run(f, in);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].id, 1);
  EXPECT_EQ(out[1].id, 2);
  EXPECT_EQ(out[2].id, 4);
}

TEST(VoxelDownsample, NegativeCoordinatesGetTheirOwnCells)
{
  auto f = filterWithLeaf(1.0);
  // floor() must bucket -0.5 and +0.5 apart; a truncating cast would merge them.
  const Cloud out = run(f, {{-0.5f, 0.0f, 0.0f, 1}, {0.5f, 0.0f, 0.0f, 2}});
  EXPECT_EQ(out.size(), 2u);
}

TEST(VoxelDownsample, DropsNonFinitePoints)
{
  auto f = filterWithLeaf(0.5);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const Cloud in{{nan, 0.0f, 0.0f, 1}, {0.0f, inf, 0.0f, 2}, {0.0f, 0.0f, -inf, 3}, {1.0f, 1.0f, 1.0f, 4}};
  const Cloud out = run(f, in);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 4);
}

TEST(VoxelDownsample, DropsPointsTooFarToIndex)
{
  auto f = filterWithLeaf(0.1);
  // 2^20 cells at 0.1 m is ~105 km out; nothing real reaches it, and it must not alias.
  const Cloud out = run(f, {{2.0e6f, 0.0f, 0.0f, 1}, {1.0f, 2.0f, 3.0f, 2}});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 2);
}

TEST(VoxelDownsample, RepeatedCallsAreIndependentAndRepeatable)
{
  auto f = filterWithLeaf(0.5);
  Cloud big;
  for (int i = 0; i < 500; ++i) {
    big.push_back({static_cast<float>(i) * 0.5f, 0.0f, 0.0f, i});
  }
  const Cloud first = run(f, big);
  EXPECT_EQ(first.size(), 500u);

  // A smaller scan right after a big one must not see stale cells from the big one.
  const Cloud small = run(f, {{0.25f, 0.25f, 0.25f, 42}});
  ASSERT_EQ(small.size(), 1u);
  EXPECT_EQ(small[0].id, 42);

  const Cloud again = run(f, big);
  ASSERT_EQ(again.size(), first.size());
  for (size_t i = 0; i < again.size(); ++i) {
    EXPECT_EQ(again[i].id, first[i].id) << "at " << i;
  }
}

TEST(VoxelDownsample, GrowsWhenAScanFillsFarMoreCellsThanTheLast)
{
  auto f = filterWithLeaf(1.0);
  // The table is sized from the previous scan's occupancy, so a sudden 200x scene has to
  // grow and redo the pass without losing or duplicating a cell.
  EXPECT_EQ(run(f, {{0.5f, 0.5f, 0.5f, 0}}).size(), 1u);
  Cloud wide;
  for (int i = 0; i < 4000; ++i) {
    wide.push_back({static_cast<float>(i) * 2.0f, 0.0f, 0.0f, i});
  }
  const Cloud out = run(f, wide);
  ASSERT_EQ(out.size(), 4000u);
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i].id, static_cast<int>(i)) << "at " << i;
  }
  // And it still works once the scene shrinks back.
  EXPECT_EQ(run(f, {{0.5f, 0.5f, 0.5f, 0}}).size(), 1u);
}

TEST(VoxelDownsample, LeafChangesTakeEffectOnTheNextScan)
{
  auto f = filterWithLeaf(0.1);
  const Cloud in{{0.00f, 0.0f, 0.0f, 1}, {0.15f, 0.0f, 0.0f, 2}, {0.30f, 0.0f, 0.0f, 3}};
  EXPECT_EQ(run(f, in).size(), 3u);
  f.setLeaf(1.0);
  EXPECT_EQ(run(f, in).size(), 1u);
  f.setLeaf(0.1);
  EXPECT_EQ(run(f, in).size(), 3u);
}

TEST(VoxelDownsample, DenseCloudCollapsesToOccupiedCellCount)
{
  auto f = filterWithLeaf(1.0);
  Cloud in;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int r = 0; r < 7; ++r) {  // 7 samples inside every cell
        in.push_back({static_cast<float>(i) + 0.1f * static_cast<float>(r),
                      static_cast<float>(j) + 0.1f * static_cast<float>(r),
                      0.5f,
                      r});
      }
    }
  }
  EXPECT_EQ(run(f, in).size(), 100u);
}

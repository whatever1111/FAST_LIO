#include <cmath>
#include <gtest/gtest.h>

#include "adaptive_downsample.hpp"

using fast_lio::AdaptiveDownsampleParams;
using fast_lio::AdaptiveDownsampleState;
using fast_lio::updateAdaptiveDownsample;

namespace
{
constexpr double kBaseVoxel = 0.15;

AdaptiveDownsampleParams enabled()
{
  AdaptiveDownsampleParams p;
  p.enabled = true;
  return p;
}

// Feed `scans` scans of the same cost; return how many times the voxel changed.
int runScans(AdaptiveDownsampleState & s, const AdaptiveDownsampleParams & p, int scans, double cost_sec)
{
  int changes = 0;
  for (int i = 0; i < scans; ++i) {
    if (updateAdaptiveDownsample(&s, cost_sec, kBaseVoxel, p))
      ++changes;
  }
  return changes;
}
}  // namespace

TEST(AdaptiveDownsample, DisabledNeverMovesButStillInitialises)
{
  AdaptiveDownsampleState s;
  AdaptiveDownsampleParams p;  // enabled = false
  EXPECT_EQ(runScans(s, p, 500, 0.5), 0);
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
  EXPECT_EQ(s.raises, 0u);
}

TEST(AdaptiveDownsample, AffordableScansLeaveTheVoxelAtBase)
{
  AdaptiveDownsampleState s;
  EXPECT_EQ(runScans(s, enabled(), 1000, 0.020), 0);
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
}

TEST(AdaptiveDownsample, OverBudgetRaisesOneStepPerStreak)
{
  AdaptiveDownsampleState s;
  const auto p = enabled();                // raise_after = 5, step = 0.05
  EXPECT_EQ(runScans(s, p, 4, 0.090), 0);  // a burst shorter than the streak does nothing
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
  EXPECT_EQ(runScans(s, p, 1, 0.090), 1);
  EXPECT_NEAR(s.voxel, 0.20, 1e-12);
  EXPECT_EQ(runScans(s, p, 5, 0.090), 1);
  EXPECT_NEAR(s.voxel, 0.25, 1e-12);
}

TEST(AdaptiveDownsample, TheVoxelIsCappedAndTheStateStopsGrowing)
{
  AdaptiveDownsampleState s;
  auto p = enabled();
  p.max_voxel = 0.25;
  runScans(s, p, 1000, 0.5);
  EXPECT_NEAR(s.voxel, 0.25, 1e-12);
  const auto raises_at_cap = s.raises;
  EXPECT_EQ(runScans(s, p, 200, 0.5), 0);
  EXPECT_EQ(s.raises, raises_at_cap);
}

TEST(AdaptiveDownsample, ItStepsBackDownOnlyAfterASustainedCalm)
{
  AdaptiveDownsampleState s;
  const auto p = enabled();  // lower_after = 100
  runScans(s, p, 5, 0.090);
  ASSERT_NEAR(s.voxel, 0.20, 1e-12);
  EXPECT_EQ(runScans(s, p, 99, 0.010), 0);  // 99 calm scans are not enough
  EXPECT_NEAR(s.voxel, 0.20, 1e-12);
  EXPECT_EQ(runScans(s, p, 1, 0.010), 1);
  EXPECT_NEAR(s.voxel, kBaseVoxel, 1e-12);
  EXPECT_EQ(runScans(s, p, 500, 0.010), 0);  // and it never goes below base
  EXPECT_NEAR(s.voxel, kBaseVoxel, 1e-12);
}

TEST(AdaptiveDownsample, TheDeadZoneClearsBothStreaks)
{
  AdaptiveDownsampleState s;
  const auto p = enabled();                // budget 0.060, release 0.035
  runScans(s, p, 4, 0.090);                // four over-budget scans, one short of a raise
  EXPECT_EQ(runScans(s, p, 1, 0.045), 0);  // a dead-zone scan resets the streak
  EXPECT_EQ(runScans(s, p, 4, 0.090), 0);  // so four more still do not raise
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
  EXPECT_EQ(runScans(s, p, 1, 0.090), 1);
}

TEST(AdaptiveDownsample, AlternatingLoadDoesNotOscillate)
{
  AdaptiveDownsampleState s;
  const auto p = enabled();
  int changes = 0;
  for (int i = 0; i < 1000; ++i) {
    changes += updateAdaptiveDownsample(&s, (i % 2 == 0) ? 0.090 : 0.010, kBaseVoxel, p) ? 1 : 0;
  }
  EXPECT_EQ(changes, 0);
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
}

TEST(AdaptiveDownsample, NonFiniteAndNegativeCostsAreIgnored)
{
  AdaptiveDownsampleState s;
  const auto p = enabled();
  EXPECT_EQ(runScans(s, p, 50, std::nan("")), 0);
  EXPECT_EQ(runScans(s, p, 50, -1.0), 0);
  EXPECT_DOUBLE_EQ(s.voxel, kBaseVoxel);
  EXPECT_FALSE(updateAdaptiveDownsample(nullptr, 0.5, kBaseVoxel, p));
}

#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "scan_time_policy.hpp"

namespace
{
auto timing(double begin, const std::vector<double> & offsets, double limit = 0.0)
{
  return fast_lio::scanTime(begin, offsets, [](double offset) { return offset; }, limit);
}
}  // namespace

TEST(ScanTime, UsesMaximumWithoutReorderingTheEffectivePointSet)
{
  const std::vector<double> offsets{0.0, 99.808216, 19.599199, 73.067188};
  const auto before = offsets;
  const auto result = timing(1788617881.300125, offsets, 0.2);
  ASSERT_TRUE(result.valid());
  EXPECT_NEAR(result.end, 1788617881.399933, 0.0000003);
  EXPECT_EQ(offsets, before);
}

TEST(ScanTime, PreservesShortAndZeroDurationInsteadOfMeanFallback)
{
  EXPECT_DOUBLE_EQ(timing(10.0, {0.0, 1.0, 0.0}).end, 10.001);
  EXPECT_DOUBLE_EQ(timing(10.0, {0.0}).end, 10.0);
  EXPECT_DOUBLE_EQ(timing(10.0, {5.0}).end, 10.005);
}

TEST(ScanTime, RejectsInvalidAndExcessiveTimes)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(timing(10.0, {}).valid());
  for (double bad : {nan, inf, -1.0}) {
    EXPECT_FALSE(timing(bad, {0.0}).valid());
    EXPECT_FALSE(timing(10.0, {0.0, bad, 1.0}).valid());
  }
  EXPECT_TRUE(timing(10.0, {200.0}, 0.2).valid());
  EXPECT_FALSE(timing(10.0, {200.1}, 0.2).valid());
  EXPECT_FALSE(timing(10.0, {1.0}, nan).valid());
}

TEST(ScanConsumption, CommitsOnlyAdvancingConsumedScans)
{
  fast_lio::ScanConsumption state;
  EXPECT_TRUE(state.canProcess(10.1));
  EXPECT_TRUE(state.commit(10.1));
  EXPECT_FALSE(state.canProcess(10.1));
  EXPECT_FALSE(state.commit(10.0));
  EXPECT_DOUBLE_EQ(state.lastEnd(), 10.1);
  EXPECT_FALSE(state.commit(std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(state.commit(10.2));
  state.reset();
  EXPECT_TRUE(state.canProcess(1.0));
}

TEST(InputEpochGuard, SmallLateScansDoNotResetButClockEpochChangeLatchesAcrossSensors)
{
  fast_lio::InputEpochGuard guard;
  EXPECT_TRUE(guard.observe(-1.0, 100.0, 1.0));
  EXPECT_TRUE(guard.observe(100.0, 99.8, 1.0));
  EXPECT_TRUE(guard.observe(100.0, 99.0, 1.0));
  EXPECT_FALSE(guard.observe(100.0, 98.999, 1.0));
  // Even another stream in the old epoch must stop: no mixed-epoch propagation.
  EXPECT_FALSE(guard.observe(100.0, 100.1, 1.0));
  EXPECT_TRUE(guard.faulted());
  fast_lio::InputEpochGuard restarted;
  EXPECT_TRUE(restarted.observe(-1.0, 1.0, 1.0));
}

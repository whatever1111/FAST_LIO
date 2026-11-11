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

namespace
{
constexpr double kRollback = 1.0;
constexpr double kLead = 10.0;
constexpr double kDay = 86400.0;

// One input stream as the node takes it: the lead filter, then the epoch guard, then the late-sample drop; only a
// sample that passes all three becomes the baseline.
struct Stream
{
  fast_lio::InputEpochGuard guard;
  fast_lio::InputLeadFilter lead;
  double baseline = -1.0;
  std::vector<double> accepted;

  void feed(double stamp)
  {
    if (!lead.admit(baseline, stamp, kLead) || !guard.observe(baseline, stamp, kRollback))
      return;
    if (baseline >= 0.0 && stamp < baseline)
      return;
    baseline = stamp;
    accepted.push_back(stamp);
  }
};
}  // namespace

TEST(InputLeadFilter, OneStampADayAheadCostsThatSampleAndNeverLatches)
{
  Stream stream;
  for (double stamp : {100.0, 100.1, 100.0 + kDay, 100.2, 100.3})
    stream.feed(stamp);
  EXPECT_EQ(stream.accepted, (std::vector<double>{100.0, 100.1, 100.2, 100.3}));
  EXPECT_FALSE(stream.guard.faulted());

  // The guard alone takes the bad stamp as the baseline and latches on the next good sample.
  fast_lio::InputEpochGuard alone;
  EXPECT_TRUE(alone.observe(100.1, 100.1 + kDay, kRollback));
  EXPECT_FALSE(alone.observe(100.1 + kDay, 100.2, kRollback));
}

TEST(InputLeadFilter, AStreamThatMovedOnContinuesOneSampleShort)
{
  Stream stream;
  for (double stamp : {100.0, 100.1, 160.0, 160.1, 160.2})
    stream.feed(stamp);
  EXPECT_EQ(stream.accepted, (std::vector<double>{100.0, 100.1, 160.1, 160.2}));
  EXPECT_FALSE(stream.guard.faulted());
}

TEST(InputLeadFilter, GapsUpToTheBoundAndTheFirstSamplePassUntouched)
{
  Stream stream;
  for (double stamp : {5000.0, 5010.0, 5010.1})
    stream.feed(stamp);
  EXPECT_EQ(stream.accepted, (std::vector<double>{5000.0, 5010.0, 5010.1}));
}

TEST(InputLeadFilter, UnrelatedStampsAheadAreEachHeldInTurn)
{
  Stream stream;
  for (double stamp : {100.0, 100.0 + kDay, 100.0 + 2.0 * kDay, 100.1})
    stream.feed(stamp);
  EXPECT_EQ(stream.accepted, (std::vector<double>{100.0, 100.1}));
  EXPECT_FALSE(stream.guard.faulted());
}

TEST(InputLeadFilter, ARollbackPassesToTheGuardWhichLatches)
{
  Stream stream;
  for (double stamp : {100.0, 100.1, 99.0, 100.2})
    stream.feed(stamp);
  EXPECT_EQ(stream.accepted, (std::vector<double>{100.0, 100.1}));
  EXPECT_TRUE(stream.guard.faulted());
}

#include <gtest/gtest.h>

#include "lost_recovery_policy.hpp"

namespace
{
using namespace fast_lio;

class LostRecovery : public ::testing::Test
{
protected:
  LostRecoveryParams params{true, 200};
  LostRecoveryState lost;
  RecoveryState recovery;

  void SetUp() override
  {
    recovery.trust = RecoveryTrust::kUnanchored;
    recovery.cause = RecoveryCause::kDeadline;
    recovery.episode = 7;
    recovery.map_resets = 3;
    recovery.reference_initialized_once = true;
  }

  bool scan(std::uint64_t seq, std::int64_t begin, std::int64_t end, bool match = true, std::uint64_t gen = 4)
  {
    return observeLostRecoveryCandidate(&lost, &recovery, seq, begin, end, match, gen, params);
  }

  void begin() { beginLostRecoveryCandidate(&lost, 4, 1, 0, params); }
};

TEST_F(LostRecovery, DisabledOperationsPreserveAnExistingCandidateAndLatch)
{
  begin();
  params.enabled = false;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
  beginLostRecoveryCandidate(&lost, 9, 9, 999, params);
  abandonLostRecovery(&lost, params);
  EXPECT_FALSE(scan(2, 100, 300));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 300, params));
  EXPECT_TRUE(lost.candidate);
  EXPECT_EQ(lost.candidate_begin_ns, 0);
  EXPECT_EQ(lost.candidate_generation, 4u);
  EXPECT_EQ(lost.candidate_last_scan, 1u);
  EXPECT_EQ(lost.attempts, 1u);
  EXPECT_EQ(lost.coordinate_epoch, 1u);
  EXPECT_EQ(lost.epochs_committed, 0u);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
  EXPECT_EQ(recovery.cause, RecoveryCause::kDeadline);
}

TEST_F(LostRecovery, OnlyAnEnabledLatchArmsRecovery)
{
  EXPECT_TRUE(lostRecoveryArmed(recovery, params));
  recovery.trust = RecoveryTrust::kPending;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
  recovery.trust = RecoveryTrust::kLocalQualified;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
}

TEST_F(LostRecovery, StableBoundaryCommitsOnceAndPreservesHistory)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 99));
  EXPECT_FALSE(scan(2, 100, 199));
  EXPECT_TRUE(scan(3, 199, 200));
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(recovery.cause, RecoveryCause::kStable);
  EXPECT_EQ(recovery.last_supported_end_ns, 200);
  EXPECT_EQ(recovery.stable_begin_ns, -1);
  EXPECT_EQ(recovery.episode, 7u);
  EXPECT_EQ(recovery.map_resets, 3u);
  EXPECT_TRUE(recovery.reference_initialized_once);
  EXPECT_EQ(lost.coordinate_epoch, 2u);
  EXPECT_EQ(lost.epochs_committed, 1u);
  EXPECT_FALSE(lost.candidate);
  EXPECT_FALSE(scan(4, 200, 400));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 400, params));
  EXPECT_EQ(lost.coordinate_epoch, 2u);
}

TEST_F(LostRecovery, MissingScanRestartsTheStableWindow)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(3, 100, 200));
  EXPECT_TRUE(scan(4, 200, 300));
}

TEST_F(LostRecovery, ChangedGenerationRestartsTheStableWindow)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(2, 100, 200, true, 5));
  EXPECT_TRUE(scan(3, 200, 300, true, 5));
}

TEST_F(LostRecovery, FailedMatchCannotSupplyStableTime)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(2, 100, 300, false));
  EXPECT_FALSE(scan(3, 300, 400));
  EXPECT_TRUE(scan(4, 400, 500));
}

TEST_F(LostRecovery, ExcludedAndInvalidScansCannotQualify)
{
  begin();
  recovery.excluded_scan = 1;
  EXPECT_FALSE(scan(1, 0, 300));
  EXPECT_FALSE(scan(0, 0, 300));
  EXPECT_FALSE(scan(2, -1, 300));
  EXPECT_FALSE(scan(3, 300, 200));
  EXPECT_FALSE(scan(4, 300, 400));
  EXPECT_TRUE(scan(5, 400, 500));
}

TEST_F(LostRecovery, AbandonRequiresANewCandidate)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  abandonLostRecovery(&lost, params);
  EXPECT_FALSE(scan(2, 100, 400));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 400, params));
  EXPECT_EQ(lost.coordinate_epoch, 1u);
  beginLostRecoveryCandidate(&lost, 5, 3, 400, params);
  EXPECT_FALSE(scan(3, 400, 500, true, 5));
  EXPECT_TRUE(scan(4, 500, 600, true, 5));
  EXPECT_EQ(lost.attempts, 2u);
}

TEST_F(LostRecovery, RepeatedLossCommitsEpochsTwoThenThree)
{
  begin();
  EXPECT_TRUE(scan(1, 0, 200));
  EXPECT_EQ(lost.coordinate_epoch, 2u);
  recovery.trust = RecoveryTrust::kUnanchored;
  beginLostRecoveryCandidate(&lost, 6, 2, 200, params);
  EXPECT_TRUE(scan(2, 200, 400, true, 6));
  EXPECT_EQ(lost.coordinate_epoch, 3u);
  EXPECT_EQ(lost.epochs_committed, 2u);
  EXPECT_EQ(recovery.episode, 7u);
  EXPECT_EQ(recovery.map_resets, 3u);
}

TEST_F(LostRecovery, EpochExhaustionNeverWrapsOrQualifies)
{
  lost.coordinate_epoch = std::numeric_limits<std::uint64_t>::max();
  begin();
  EXPECT_FALSE(scan(1, 0, 200));
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
  EXPECT_EQ(lost.coordinate_epoch, std::numeric_limits<std::uint64_t>::max());
}

TEST(LostRecoveryParameters, ValidatesDurationAndRequiresTheLatchPolicy)
{
  LostRecoveryParams params;
  EXPECT_TRUE(resolveLostRecoveryParams(false, 2.0, 0.0, &params));
  EXPECT_FALSE(params.enabled);
  EXPECT_EQ(params.stable_ns, 2000000000);
  EXPECT_TRUE(resolveLostRecoveryParams(true, 2.0, 10.0, &params));
  EXPECT_TRUE(params.enabled);
  EXPECT_FALSE(resolveLostRecoveryParams(true, 2.0, 0.0, &params));
  EXPECT_FALSE(resolveLostRecoveryParams(true, 2.0, -1.0, &params));
  for (double invalid :
       {0.0, -1.0, 1e-12, 1e20, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(resolveLostRecoveryParams(false, invalid, 0.0, &params));
    EXPECT_FALSE(resolveLostRecoveryParams(true, invalid, 10.0, &params));
  }
}
}  // namespace

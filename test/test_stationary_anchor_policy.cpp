#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <vector>

#include "stationary_anchor_policy.hpp"
#include "zupt_policy.hpp"

using fast_lio::AnchorDrop;
using fast_lio::AnchorHealth;
using fast_lio::AnchorHealthInputs;
using fast_lio::AnchorScanStart;
using fast_lio::AnchorState;
using fast_lio::AnchorStep;
using fast_lio::AnchorStepInput;
using fast_lio::RecoveryCause;
using fast_lio::RecoveryEvent;
using fast_lio::RecoveryParams;
using fast_lio::RecoveryState;
using fast_lio::RecoveryTrust;
using fast_lio::StaticEvidence;
using fast_lio::StaticEvidenceReason;
using fast_lio::StationaryAnchor;
using fast_lio::StationaryAnchorParams;

namespace
{
constexpr std::int64_t kSecond = 1000000000;
constexpr std::int64_t kPeriod = 100000000;  // 10 Hz scans that span their period
constexpr std::int64_t kFirstBegin = 5 * kSecond;

std::int64_t beginOf(int scan)
{
  return kFirstBegin + scan * kPeriod;
}

std::int64_t endOf(int scan)
{
  return beginOf(scan) + kPeriod;
}

// Runs the lifecycle the way the node does: scan start (hole check), then the anchor step.
class Lifecycle
{
public:
  Lifecycle() = default;
  explicit Lifecycle(const StationaryAnchorParams & params) : params(params) {}

  AnchorStep scan(StaticEvidence evidence, bool degraded = false, bool trusted = true, bool lost = false)
  {
    last_start = fast_lio::beginAnchorScan(&anchor, beginOf(next), params);
    AnchorStepInput in;
    in.end_ns = endOf(next);
    in.evidence = evidence;
    in.degraded = degraded;
    in.reference_trusted = trusted;
    in.lost = lost;
    ++next;
    return fast_lio::stepAnchor(&anchor, in, params);
  }
  AnchorStep stationary(bool degraded = false) { return scan(StaticEvidence::kStationary, degraded); }
  AnchorStep moving(bool degraded = false) { return scan(StaticEvidence::kMoving, degraded); }

  // Healthy stationary scans until the candidate is committed; returns the anchor's ID.
  std::uint64_t commit()
  {
    for (int i = 0; i < params.confirm_scans; ++i)
      stationary();
    EXPECT_EQ(anchor.state, AnchorState::kValid);
    return anchor.id;
  }

  // Scans that return before their anchor step (no evidence evaluated, nothing dropped explicitly).
  void skip(int scans) { next += scans; }

  // A scan that returns before its anchor step, closed the way the node closes it: the static evidence of the IMU
  // window the filter committed for it (`committed`) may end the anchor, then a candidate ends as unevaluated.
  // Returns what the window check dropped.
  AnchorDrop exit(StaticEvidence window, bool committed = true)
  {
    last_start = fast_lio::beginAnchorScan(&anchor, beginOf(next), params);
    const AnchorDrop drop = fast_lio::unevaluatedConsumedImuDrop(anchor, committed, window);
    if (drop != AnchorDrop::kNone)
      fast_lio::dropAnchor(&anchor, drop);
    fast_lio::dropUnevaluatedCandidate(&anchor, endOf(next));
    ++next;
    return drop;
  }

  StationaryAnchor anchor;
  StationaryAnchorParams params;
  AnchorScanStart last_start;
  int next = 0;
};

fast_lio::ImuWindowStats imuAtRest()
{
  fast_lio::ImuWindowStats imu;
  imu.samples = 20;
  imu.gyr_max = 0.01;
  imu.acc_min = 9.80;
  imu.acc_max = 9.82;
  imu.acc_mean = 9.81;
  return imu;
}

fast_lio::WheelEvidence wheel(bool valid, double age, double speed)
{
  fast_lio::WheelEvidence w;
  w.valid = valid;
  w.age = age;
  w.speed = speed;
  return w;
}

RecoveryParams platformPolicy()
{
  RecoveryParams p;
  p.total_ns = 10 * kSecond;
  p.stable_ns = kSecond;
  return p;
}
}  // namespace

// ── P2-1: creation ───────────────────────────────────────────────────────────

TEST(StationaryAnchorPolicy, StartsEmptyWithNothingToUse)
{
  const StationaryAnchor anchor;
  EXPECT_EQ(anchor.state, AnchorState::kEmpty);
  EXPECT_EQ(fast_lio::anchorCurrentId(anchor), 0u);
  EXPECT_EQ(anchor.last_evidence_end_ns, -1);
  const RecoveryState recovery;
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
}

TEST(StationaryAnchorPolicy, HealthyMovingScansNeverCreateAnAnchor)
{
  Lifecycle life;
  for (int i = 0; i < 50; ++i) {
    const AnchorStep step = life.moving();
    EXPECT_FALSE(step.usable);
    EXPECT_FALSE(step.started);
    EXPECT_FALSE(step.committed);
  }
  EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
  EXPECT_EQ(life.anchor.last_id, 0u);
}

TEST(StationaryAnchorPolicy, ThirdStationaryScanCommitsTheFirstScansCandidate)
{
  Lifecycle life;
  const AnchorStep first = life.stationary();
  EXPECT_TRUE(first.started);
  EXPECT_FALSE(first.committed);
  EXPECT_FALSE(first.usable);
  EXPECT_EQ(life.anchor.state, AnchorState::kQualifying);
  const std::uint64_t id = life.anchor.id;
  EXPECT_NE(id, 0u);

  const AnchorStep second = life.stationary();
  EXPECT_FALSE(second.started);
  EXPECT_FALSE(second.committed);
  EXPECT_FALSE(second.usable);

  const AnchorStep third = life.stationary();
  EXPECT_TRUE(third.committed);
  EXPECT_FALSE(third.usable) << "the commit scan does not use the anchor it commits";
  EXPECT_EQ(life.anchor.state, AnchorState::kValid);
  EXPECT_EQ(life.anchor.id, id);
  EXPECT_EQ(life.anchor.created_ns, endOf(0)) << "the candidate carries the first scan's pose";
  EXPECT_EQ(life.anchor.committed_ns, endOf(2));

  const AnchorStep fourth = life.stationary();
  EXPECT_TRUE(fourth.usable);
  EXPECT_FALSE(fourth.started);
}

TEST(StationaryAnchorPolicy, AnchorAndPriorStreakDoNotWaitTwice)
{
  // The parked streak counts the candidate's scans: with engage_scans 3 the prior may engage on the commit scan
  // and apply from the next one, not 3 scans after the commit.
  Lifecycle life;
  int counted = 0;
  for (int i = 0; i < 3; ++i)
    counted += life.stationary().parked_evidence ? 1 : 0;
  EXPECT_EQ(counted, 3);
}

TEST(StationaryAnchorPolicy, OneConfirmScanCommitsAtOnceButIsNotUsedOnThatScan)
{
  StationaryAnchorParams params;
  params.confirm_scans = 1;
  Lifecycle life(params);
  const AnchorStep step = life.stationary();
  EXPECT_TRUE(step.started);
  EXPECT_TRUE(step.committed);
  EXPECT_FALSE(step.usable);
  EXPECT_TRUE(life.stationary().usable);
}

TEST(StationaryAnchorPolicy, ALongParkingKeepsOneAnchor)
{
  Lifecycle life;
  const std::uint64_t id = life.commit();
  for (int i = 0; i < 6000; ++i) {  // ten minutes at 10 Hz: age alone is not evidence of motion
    const AnchorStep step = life.stationary(i % 2 == 0);
    ASSERT_TRUE(step.usable) << i;
  }
  EXPECT_EQ(life.anchor.id, id);
  EXPECT_EQ(life.anchor.last_id, id);
}

// ── P2-2: stationary → moving → stationary ───────────────────────────────────

TEST(StationaryAnchorPolicy, OneMovingScanEndsTheEpisodeInAllFourAlternations)
{
  for (const bool moving_degraded : {false, true}) {
    for (const bool parked_degraded : {false, true}) {
      Lifecycle life;
      const std::uint64_t old_id = life.commit();
      const AnchorStep moved = life.moving(moving_degraded);
      EXPECT_TRUE(moved.dropped);
      EXPECT_EQ(moved.drop, AnchorDrop::kMotion);
      EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
      EXPECT_EQ(fast_lio::anchorCurrentId(life.anchor), 0u);

      const AnchorStep parked = life.stationary(parked_degraded);
      EXPECT_FALSE(parked.usable) << "moving degraded=" << moving_degraded << " parked degraded=" << parked_degraded;
      if (parked_degraded) {
        EXPECT_FALSE(parked.started) << "a degraded scan does not start a candidate";
        fast_lio::ZuptPolicyInputs zupt;
        zupt.scan_degraded = true;
        zupt.zupt_enabled = true;
        zupt.anchor_valid = parked.usable;
        zupt.body_static = true;
        zupt.speed = 0.4;
        zupt.max_speed = 30.0;
        const fast_lio::ZuptDecision decision = fast_lio::decideZupt(zupt);
        EXPECT_FALSE(decision.hold);
        EXPECT_EQ(decision.action, fast_lio::ZuptAction::kZeroVelocity) << "only the velocity is zeroed";
      } else {
        EXPECT_TRUE(parked.started);
        EXPECT_NE(life.anchor.id, old_id) << "a new stationary stretch is a new anchor";
      }
    }
  }
}

TEST(StationaryAnchorPolicy, HealthyMovingScanDoesNotRebuildAnAnchorOnTheSameScan)
{
  Lifecycle life;
  life.commit();
  const AnchorStep moved = life.moving(false);
  EXPECT_FALSE(moved.started);
  EXPECT_FALSE(moved.committed);
  EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
}

TEST(StationaryAnchorPolicy, UnknownEvidenceEndsTheEpisodeLikeMotion)
{
  Lifecycle life;
  life.commit();
  const AnchorStep step = life.scan(StaticEvidence::kUnknown, true);
  EXPECT_TRUE(step.dropped);
  EXPECT_EQ(step.drop, AnchorDrop::kEvidenceUnknown);
  EXPECT_FALSE(step.usable);
}

TEST(StationaryAnchorPolicy, DegradedScanBreaksACandidate)
{
  Lifecycle life;
  life.stationary();
  life.stationary();
  const AnchorStep degraded = life.stationary(true);
  EXPECT_TRUE(degraded.dropped);
  EXPECT_EQ(degraded.drop, AnchorDrop::kNotQualified);
  EXPECT_FALSE(degraded.parked_evidence);
  const std::uint64_t broken = life.anchor.id;
  // Three more healthy scans are needed, under a new ID.
  EXPECT_TRUE(life.stationary().started);
  EXPECT_NE(life.anchor.id, broken);
  EXPECT_FALSE(life.stationary().committed);
  EXPECT_TRUE(life.stationary().committed);
}

// ── P2-3 / P2-4: holes ───────────────────────────────────────────────────────

TEST(StationaryAnchorPolicy, AScanHoleEndsTheEpisodeWhateverHappenedInside)
{
  // Nothing evaluated the evidence inside the hole, so a platform that moved there, one that stayed put, and one
  // parked on both sides look the same: all three end the episode at the first returning scan, before its evidence
  // is looked at, and the old anchor is never used again.
  struct Return
  {
    StaticEvidence evidence;
    bool degraded;
  };
  const Return returns[] = {{StaticEvidence::kMoving, true},        // moved inside and still moving
                            {StaticEvidence::kStationary, true},    // parked before and after, scan still blind
                            {StaticEvidence::kStationary, false}};  // parked before and after, scan healthy
  for (const Return & r : returns) {
    Lifecycle life;
    const std::uint64_t old_id = life.commit();
    life.skip(4);  // 0.4 s without evaluated evidence; the IMU may well have kept running
    const AnchorStep first = life.scan(r.evidence, r.degraded);
    EXPECT_TRUE(life.last_start.hole);
    EXPECT_TRUE(life.last_start.dropped);
    EXPECT_EQ(life.anchor.last_drop, AnchorDrop::kScanHole);
    EXPECT_FALSE(first.usable);
    for (int i = 0; i < 10; ++i) {
      const AnchorStep step = life.scan(StaticEvidence::kStationary, r.degraded);
      if (step.usable) {
        EXPECT_NE(life.anchor.id, old_id);
      }
      EXPECT_NE(fast_lio::anchorCurrentId(life.anchor), old_id);
    }
  }
}

TEST(StationaryAnchorPolicy, AClockThatStepsBackBeyondTheBoundIsAHole)
{
  for (const std::int64_t delta : {-1, 0, 1}) {
    Lifecycle life;
    life.commit();
    const std::int64_t begin = life.anchor.last_evidence_end_ns - life.params.max_scan_gap_ns - delta;
    const AnchorScanStart start = fast_lio::beginAnchorScan(&life.anchor, begin, life.params);
    EXPECT_EQ(start.hole, delta > 0) << "back by the bound " << (delta >= 0 ? "+" : "") << delta << " ns";
    EXPECT_EQ(life.anchor.state == AnchorState::kValid, delta <= 0);
  }
}

TEST(StationaryAnchorPolicy, ARepeatedOrOutOfOrderScanAddsNothing)
{
  Lifecycle life;
  life.stationary();
  life.stationary();
  const std::int64_t watermark = life.anchor.last_evidence_end_ns;
  for (const std::int64_t end : {watermark, watermark - kPeriod}) {
    fast_lio::beginAnchorScan(&life.anchor, end - kPeriod, life.params);
    AnchorStepInput repeat;
    repeat.end_ns = end;
    repeat.evidence = StaticEvidence::kStationary;
    const AnchorStep step = fast_lio::stepAnchor(&life.anchor, repeat, life.params);
    EXPECT_FALSE(step.committed) << "a copy of the second scan is not the third";
    EXPECT_FALSE(step.parked_evidence);
    EXPECT_EQ(life.anchor.confirmed_scans, 2);
    EXPECT_EQ(life.anchor.last_evidence_end_ns, watermark);
  }
  life.next = 2;
  EXPECT_TRUE(life.stationary().committed);
  const std::int64_t committed_end = life.anchor.last_evidence_end_ns;
  AnchorStepInput repeat;
  repeat.end_ns = committed_end;
  repeat.evidence = StaticEvidence::kStationary;
  repeat.degraded = true;
  fast_lio::beginAnchorScan(&life.anchor, committed_end - kPeriod, life.params);
  EXPECT_FALSE(fast_lio::stepAnchor(&life.anchor, repeat, life.params).usable);
  EXPECT_EQ(life.anchor.state, AnchorState::kValid) << "nor does it end anything";
}

TEST(StationaryAnchorPolicy, GapBoundIsStrictToTheNanosecond)
{
  const StationaryAnchorParams params;  // 0.3 s
  for (const std::int64_t delta : {-1, 0, 1}) {
    StationaryAnchor anchor;
    AnchorStepInput in;
    in.evidence = StaticEvidence::kStationary;
    for (int i = 0; i < 3; ++i) {
      fast_lio::beginAnchorScan(&anchor, beginOf(i), params);
      in.end_ns = endOf(i);
      fast_lio::stepAnchor(&anchor, in, params);
    }
    ASSERT_EQ(anchor.state, AnchorState::kValid);
    const std::int64_t begin = endOf(2) + params.max_scan_gap_ns + delta;
    const AnchorScanStart start = fast_lio::beginAnchorScan(&anchor, begin, params);
    EXPECT_EQ(start.gap_ns, params.max_scan_gap_ns + delta);
    EXPECT_EQ(start.hole, delta > 0) << "gap = bound " << (delta >= 0 ? "+" : "") << delta << " ns";
    EXPECT_EQ(anchor.state == AnchorState::kValid, delta <= 0);
  }
}

TEST(StationaryAnchorPolicy, OnlyEvaluatedEvidenceAdvancesTheWatermark)
{
  // Scans that return before the anchor step (coverage skip, no cloud) leave the watermark where it was, so a
  // run of them becomes a hole even though scans kept arriving.
  Lifecycle life;
  life.commit();
  const std::int64_t watermark = life.anchor.last_evidence_end_ns;
  life.skip(3);
  life.stationary();
  EXPECT_FALSE(life.last_start.hole) << "three missing scans span exactly the bound";
  EXPECT_GT(life.anchor.last_evidence_end_ns, watermark);
  life.skip(4);
  life.stationary();
  EXPECT_TRUE(life.last_start.hole);
}

TEST(StationaryAnchorPolicy, AScanThatWasNotEvaluatedEndsACandidateAndLeavesAnAnchor)
{
  // Two confirming scans, then a scan that ends before the anchor step: the three scans the anchor needs are no
  // longer consecutive, so the candidate ends and the count starts again on a new one.
  Lifecycle life;
  life.stationary();
  life.stationary();
  ASSERT_EQ(life.anchor.state, AnchorState::kQualifying);
  const std::uint64_t first_id = life.anchor.id;
  fast_lio::beginAnchorScan(&life.anchor, beginOf(life.next), life.params);
  EXPECT_TRUE(fast_lio::dropUnevaluatedCandidate(&life.anchor, endOf(life.next)));
  life.skip(1);
  EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
  EXPECT_EQ(life.anchor.last_drop, AnchorDrop::kScanNotEvaluated);
  EXPECT_EQ(life.anchor.confirmed_scans, 0);
  const AnchorStep restarted = life.stationary();
  EXPECT_TRUE(restarted.started);
  EXPECT_NE(life.anchor.id, first_id);
  EXPECT_FALSE(life.stationary().committed) << "the two scans before the unevaluated one still counted";
  EXPECT_TRUE(life.stationary().committed);

  // An anchor is not a count of consecutive scans: the scan-gap bound says how long it may go unevaluated.
  Lifecycle parked;
  const std::uint64_t anchor_id = parked.commit();
  fast_lio::beginAnchorScan(&parked.anchor, beginOf(parked.next), parked.params);
  EXPECT_FALSE(fast_lio::dropUnevaluatedCandidate(&parked.anchor, endOf(parked.next)));
  parked.skip(1);
  EXPECT_EQ(parked.anchor.state, AnchorState::kValid);
  EXPECT_TRUE(parked.stationary().usable);
  EXPECT_EQ(parked.anchor.id, anchor_id);
}

TEST(StationaryAnchorPolicy, ARepeatedOrOldScanThatWasNotEvaluatedIsNoMissingScan)
{
  Lifecycle life;
  life.stationary();
  life.stationary();
  const std::int64_t watermark = life.anchor.last_evidence_end_ns;
  EXPECT_FALSE(fast_lio::dropUnevaluatedCandidate(&life.anchor, watermark)) << "the evaluated scan again";
  EXPECT_FALSE(fast_lio::dropUnevaluatedCandidate(&life.anchor, watermark - kPeriod)) << "an older scan";
  EXPECT_EQ(life.anchor.state, AnchorState::kQualifying);
  EXPECT_EQ(life.anchor.confirmed_scans, 2);
  // Without an end time the scan cannot be shown to be a repeat, and counts as a new one.
  EXPECT_TRUE(fast_lio::dropUnevaluatedCandidate(&life.anchor, -1));
  EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
  EXPECT_FALSE(fast_lio::dropUnevaluatedCandidate(&life.anchor, endOf(life.next))) << "nothing left to end";
}

// ── Scans that took their IMU window and returned before the anchor step ─────

TEST(StationaryAnchorPolicy, ACommittedWindowThatIsNotParkedEndsTheAnchor)
{
  // No later scan sees a window the filter has taken, so motion in it, or evidence that cannot tell, ends the
  // anchor on the scan that took it, for good. The degraded parked scan after it has nothing to hold to and only
  // zeroes its velocity; a new anchor needs its own run of healthy parked scans and a new ID.
  for (const StaticEvidence window : {StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
    const AnchorDrop expected = window == StaticEvidence::kMoving ? AnchorDrop::kMotion : AnchorDrop::kEvidenceUnknown;
    Lifecycle life;
    const std::uint64_t old_id = life.commit();
    const std::int64_t watermark = life.anchor.last_evidence_end_ns;
    EXPECT_EQ(life.exit(window), expected);
    EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
    EXPECT_EQ(life.anchor.last_drop, expected);
    EXPECT_EQ(life.anchor.id, old_id) << "the anchor that ended keeps its ID";
    EXPECT_EQ(life.anchor.last_evidence_end_ns, watermark) << "the window is no evaluated evidence";

    const AnchorStep parked = life.stationary(true);
    EXPECT_FALSE(parked.usable || parked.started || parked.committed);
    fast_lio::ZuptPolicyInputs in;
    in.scan_degraded = true;
    in.zupt_enabled = true;
    in.anchor_valid = parked.usable;
    in.body_static = true;
    in.speed = 0.4;
    in.max_speed = 30.0;
    const fast_lio::ZuptDecision decision = fast_lio::decideZupt(in);
    EXPECT_EQ(decision.action, fast_lio::ZuptAction::kZeroVelocity);
    EXPECT_FALSE(decision.hold);

    const std::uint64_t new_id = life.commit();
    EXPECT_GT(new_id, old_id);
    EXPECT_TRUE(life.stationary(true).usable);
    EXPECT_EQ(life.anchor.id, new_id);
  }
}

TEST(StationaryAnchorPolicy, ParkedCommittedWindowsKeepTheAnchorAndAreNoEvidence)
{
  // Up to three such scans with the platform parked in their windows end nothing and count for nothing: the anchor
  // keeps its ID, episode, count and times, and the degraded parked scan after them, as far from the last evaluated
  // scan as the bound allows, holds to it.
  for (int exits = 1; exits <= 3; ++exits) {
    Lifecycle life;
    const std::uint64_t id = life.commit();
    const StationaryAnchor before = life.anchor;
    for (int i = 0; i < exits; ++i) {
      EXPECT_EQ(life.exit(StaticEvidence::kStationary), AnchorDrop::kNone) << exits << " exits";
      EXPECT_FALSE(life.last_start.hole) << exits << " exits";
    }
    EXPECT_EQ(life.anchor.state, AnchorState::kValid) << exits << " exits";
    EXPECT_EQ(life.anchor.id, id);
    EXPECT_EQ(life.anchor.episode, before.episode);
    EXPECT_EQ(life.anchor.confirmed_scans, before.confirmed_scans);
    EXPECT_EQ(life.anchor.created_ns, before.created_ns);
    EXPECT_EQ(life.anchor.committed_ns, before.committed_ns);
    EXPECT_EQ(life.anchor.last_evidence_end_ns, before.last_evidence_end_ns);
    const AnchorStep parked = life.stationary(true);
    EXPECT_FALSE(life.last_start.hole) << exits << " exits";
    EXPECT_EQ(life.last_start.gap_ns, exits * kPeriod);
    EXPECT_TRUE(parked.usable) << exits << " exits";
    EXPECT_EQ(life.anchor.id, id);
  }
}

TEST(StationaryAnchorPolicy, AWindowTheFilterDidNotCommitIsLeftToTheScanThatTakesIt)
{
  // A window the filter kept, or never took, is not spent: whatever it would show, the scan that takes it judges
  // it at its own anchor step. The "unknown" of a window nobody evaluated ends nothing.
  for (const StaticEvidence window : {StaticEvidence::kStationary, StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
    Lifecycle life;
    const std::uint64_t id = life.commit();
    EXPECT_EQ(life.exit(window, false), AnchorDrop::kNone);
    EXPECT_EQ(life.anchor.state, AnchorState::kValid);
    EXPECT_EQ(life.anchor.id, id);
    const AnchorStep taken = life.scan(window, true);
    EXPECT_EQ(taken.usable, window == StaticEvidence::kStationary);
    EXPECT_EQ(taken.dropped, window != StaticEvidence::kStationary);
  }
}

TEST(StationaryAnchorPolicy, ACandidateKeepsTheRuleForScansThatWereNotEvaluated)
{
  // A candidate's confirming scans must be consecutive, whatever the window of the scan that broke the run: it
  // ends as scan_not_evaluated, and the count starts again on a new candidate.
  for (const bool committed : {true, false}) {
    for (const StaticEvidence window :
         {StaticEvidence::kStationary, StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
      Lifecycle life;
      life.stationary();
      life.stationary();
      ASSERT_EQ(life.anchor.state, AnchorState::kQualifying);
      const std::uint64_t candidate = life.anchor.id;
      EXPECT_EQ(life.exit(window, committed), AnchorDrop::kNone);
      EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
      EXPECT_EQ(life.anchor.last_drop, AnchorDrop::kScanNotEvaluated);
      EXPECT_TRUE(life.stationary().started);
      EXPECT_GT(life.anchor.id, candidate);
      EXPECT_FALSE(life.stationary().committed);
      EXPECT_TRUE(life.stationary().committed);
    }
  }
}

TEST(StationaryAnchorPolicy, AWindowFindsNothingToEndOnceTheAnchorIsGone)
{
  // No anchor yet, or one an earlier event of the same scan already ended: the window ends nothing, the first
  // reason stands, and nothing is started or revived.
  for (const StaticEvidence window : {StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
    Lifecycle empty;
    EXPECT_EQ(empty.exit(window), AnchorDrop::kNone);
    EXPECT_EQ(empty.anchor.state, AnchorState::kEmpty);
    EXPECT_EQ(empty.anchor.id, 0u);

    Lifecycle life;
    const std::uint64_t id = life.commit();
    fast_lio::beginAnchorScan(&life.anchor, beginOf(life.next), life.params);
    ASSERT_TRUE(fast_lio::dropAnchor(&life.anchor, AnchorDrop::kImuCoverageGap));
    EXPECT_EQ(fast_lio::unevaluatedConsumedImuDrop(life.anchor, true, window), AnchorDrop::kNone);
    EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
    EXPECT_EQ(life.anchor.last_drop, AnchorDrop::kImuCoverageGap);
    EXPECT_EQ(life.anchor.id, id);
  }
}

TEST(StationaryAnchorPolicy, OneWindowThatIsNotParkedInARunOfExitsEndsTheAnchorOnce)
{
  // One to three such scans, the first, a middle or the last of them with motion (or evidence that cannot tell)
  // in its window and the platform parked in the others: each window is judged on its own, so parked windows
  // neither average the motion away nor bring the anchor back. It ends once, on the window that showed it.
  for (const StaticEvidence shown : {StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
    const AnchorDrop expected = shown == StaticEvidence::kMoving ? AnchorDrop::kMotion : AnchorDrop::kEvidenceUnknown;
    for (int exits = 1; exits <= 3; ++exits) {
      for (int at = 0; at < exits; ++at) {
        Lifecycle life;
        const std::uint64_t id = life.commit();
        int drops = 0;
        for (int i = 0; i < exits; ++i) {
          const AnchorDrop drop = life.exit(i == at ? shown : StaticEvidence::kStationary);
          EXPECT_EQ(drop, i == at ? expected : AnchorDrop::kNone) << exits << " exits, shown at " << at << ", " << i;
          drops += drop != AnchorDrop::kNone ? 1 : 0;
          EXPECT_EQ(life.anchor.state, i < at ? AnchorState::kValid : AnchorState::kEmpty);
          EXPECT_EQ(life.anchor.id, id);
        }
        EXPECT_EQ(drops, 1);
        EXPECT_EQ(life.anchor.last_drop, expected);
        EXPECT_FALSE(life.stationary(true).usable) << exits << " exits, shown at " << at;
      }
    }
  }
}

TEST(StationaryAnchorPolicy, ParkedWindowsDoNotMoveTheGapBound)
{
  // Parked windows are no evaluated evidence: the bound still runs from the last scan the anchor step evaluated,
  // to the nanosecond and both ways.
  for (const std::int64_t delta : {-1, 0, 1}) {
    for (const std::int64_t sign : {1, -1}) {
      Lifecycle life;
      life.commit();
      const std::int64_t watermark = life.anchor.last_evidence_end_ns;
      life.exit(StaticEvidence::kStationary);
      life.exit(StaticEvidence::kStationary);
      ASSERT_EQ(life.anchor.state, AnchorState::kValid);
      const std::int64_t gap = sign * (life.params.max_scan_gap_ns + delta);
      const AnchorScanStart start = fast_lio::beginAnchorScan(&life.anchor, watermark + gap, life.params);
      EXPECT_EQ(start.gap_ns, gap);
      EXPECT_EQ(start.hole, delta > 0) << "gap " << gap << " ns";
      EXPECT_EQ(life.anchor.state == AnchorState::kValid, delta <= 0) << "gap " << gap << " ns";
    }
  }
  // Four parked windows in a row: the fourth still begins within the bound, the scan after it does not.
  Lifecycle life;
  life.commit();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(life.exit(StaticEvidence::kStationary), AnchorDrop::kNone);
    EXPECT_FALSE(life.last_start.hole) << "exit " << i + 1;
  }
  EXPECT_FALSE(life.stationary(true).usable);
  EXPECT_TRUE(life.last_start.hole);
  EXPECT_EQ(life.last_start.gap_ns, 4 * kPeriod);
  EXPECT_EQ(life.anchor.last_drop, AnchorDrop::kScanHole);
}

TEST(StationaryAnchorPolicy, AfterAWindowEndedTheAnchorZuptKeepsItsTable)
{
  // The window check only ends the anchor; what a later scan writes is decideZupt's, from that scan's own
  // evidence and without the anchor. Evidence that still cannot tell is not "parked": nothing is pinned, and the
  // velocity is zeroed only above the speed ceiling, as with ZUPT off.
  Lifecycle life;
  life.commit();
  ASSERT_EQ(life.exit(StaticEvidence::kUnknown), AnchorDrop::kEvidenceUnknown);
  const AnchorStep unknown = life.scan(StaticEvidence::kUnknown, true);
  EXPECT_FALSE(unknown.usable || unknown.started || unknown.dropped);
  for (const bool enabled : {true, false}) {
    for (const bool body_static : {false, true}) {
      for (const double speed : {0.4, 31.0}) {
        fast_lio::ZuptPolicyInputs in;
        in.scan_degraded = true;
        in.zupt_enabled = enabled;
        in.anchor_valid = unknown.usable;
        in.body_static = body_static;
        in.speed = speed;
        in.max_speed = 30.0;
        const fast_lio::ZuptDecision decision = fast_lio::decideZupt(in);
        const bool zero = (enabled && body_static) || speed > in.max_speed;
        EXPECT_EQ(decision.action, zero ? fast_lio::ZuptAction::kZeroVelocity : fast_lio::ZuptAction::kNone)
          << "zupt " << enabled << " static " << body_static << " speed " << speed;
        EXPECT_FALSE(decision.hold);
      }
    }
  }
}

TEST(StationaryAnchorPolicy, AScanWithoutABeginTimeIsAHole)
{
  Lifecycle life;
  life.commit();
  const AnchorScanStart start = fast_lio::beginAnchorScan(&life.anchor, -1, life.params);
  EXPECT_TRUE(start.hole);
  EXPECT_TRUE(start.dropped);
}

TEST(StationaryAnchorPolicy, OverlappingScansAreNotHoles)
{
  Lifecycle life;
  life.commit();
  const AnchorScanStart start = fast_lio::beginAnchorScan(&life.anchor, endOf(2) - kPeriod / 2, life.params);
  EXPECT_FALSE(start.hole);
  EXPECT_EQ(life.anchor.state, AnchorState::kValid);
}

TEST(StationaryAnchorPolicy, CoverageGapsAndBridgesDropTheAnchorAtOnce)
{
  for (const AnchorDrop reason : {AnchorDrop::kImuCoverageGap,
                                  AnchorDrop::kImuBridge,
                                  AnchorDrop::kTimeEpoch,
                                  AnchorDrop::kLost,
                                  AnchorDrop::kEvidenceUnknown}) {
    Lifecycle life;
    life.commit();
    fast_lio::beginAnchorScan(&life.anchor, beginOf(life.next), life.params);
    EXPECT_TRUE(fast_lio::dropAnchor(&life.anchor, reason));
    EXPECT_EQ(life.anchor.last_drop, reason);
    EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
  }
}

// ── Same-scan order: invalidate before use or create ─────────────────────────

TEST(StationaryAnchorPolicy, ADropEarlierInTheScanForbidsUseAndCreation)
{
  for (const bool degraded : {false, true}) {
    Lifecycle life;
    life.commit();
    fast_lio::beginAnchorScan(&life.anchor, beginOf(life.next), life.params);
    fast_lio::dropAnchor(&life.anchor, AnchorDrop::kImuBridge);
    AnchorStepInput in;
    in.end_ns = endOf(life.next);
    in.evidence = StaticEvidence::kStationary;
    in.degraded = degraded;
    const AnchorStep step = fast_lio::stepAnchor(&life.anchor, in, life.params);
    EXPECT_FALSE(step.usable);
    EXPECT_FALSE(step.started);
    EXPECT_FALSE(step.parked_evidence);
  }
}

TEST(StationaryAnchorPolicy, AnEventOnAnEmptyAnchorStillBlocksThatScansCandidate)
{
  // The bridged or skipped stretch belongs to this scan's window, so the scan is not a complete stationary scan.
  Lifecycle life;
  fast_lio::beginAnchorScan(&life.anchor, beginOf(0), life.params);
  EXPECT_FALSE(fast_lio::dropAnchor(&life.anchor, AnchorDrop::kImuBridge));
  AnchorStepInput in;
  in.end_ns = endOf(0);
  in.evidence = StaticEvidence::kStationary;
  EXPECT_FALSE(fast_lio::stepAnchor(&life.anchor, in, life.params).started);
}

// ── P2-6: map reset and reference replacement ────────────────────────────────

TEST(StationaryAnchorPolicy, AMapResetEndsEverythingAndTheResetScanSeedsNothing)
{
  for (const AnchorDrop reason : {AnchorDrop::kMapReset, AnchorDrop::kReferenceChanged}) {
    Lifecycle life;
    const std::uint64_t old_id = life.commit();
    const std::uint64_t generation = life.anchor.reference_generation;
    // The late reset: the anchor step of the scan has already run.
    EXPECT_TRUE(life.stationary().usable);
    EXPECT_TRUE(fast_lio::changeAnchorReference(&life.anchor, reason));
    EXPECT_EQ(life.anchor.reference_generation, generation + 1);
    EXPECT_EQ(life.anchor.state, AnchorState::kEmpty);
    // A reset before the anchor step of a scan: that scan neither uses nor seeds.
    fast_lio::beginAnchorScan(&life.anchor, beginOf(life.next), life.params);
    fast_lio::changeAnchorReference(&life.anchor, reason);
    AnchorStepInput in;
    in.end_ns = endOf(life.next++);
    in.evidence = StaticEvidence::kStationary;
    const AnchorStep step = fast_lio::stepAnchor(&life.anchor, in, life.params);
    EXPECT_FALSE(step.started);
    EXPECT_FALSE(step.usable);
    // The next stationary stretch builds a new anchor in the new generation.
    EXPECT_NE(life.commit(), old_id);
  }
}

TEST(StationaryAnchorPolicy, OrdinaryScansDoNotChangeTheReference)
{
  // Map growth, trimming and the first map build make no call here; the anchor and its generation stay.
  Lifecycle life;
  const std::uint64_t id = life.commit();
  for (int i = 0; i < 100; ++i)
    life.stationary();
  EXPECT_EQ(life.anchor.id, id);
  EXPECT_EQ(life.anchor.reference_generation, 0u);
}

// ── P2-7: degraded flips ─────────────────────────────────────────────────────

TEST(StationaryAnchorPolicy, ThirtyTwoDegradedFlipsNeverReachTheOldAnchor)
{
  Lifecycle life;
  const std::uint64_t old_id = life.commit();
  int pins_after_release = 0;
  int flips = 0;
  bool released = false;
  for (int i = 0; i < 80; ++i) {
    const bool moving = i % 2 == 1;
    const AnchorStep step = moving ? life.moving(true) : life.stationary(true);
    if (moving) {
      ++flips;
      released = true;
    }
    if (released && step.usable)
      ++pins_after_release;
    EXPECT_EQ(life.anchor.last_id, old_id) << "no anchor is created while degraded";
  }
  EXPECT_GE(flips, 32);
  EXPECT_EQ(pins_after_release, 0);
}

TEST(StationaryAnchorPolicy, PendingDeadlineIgnoresFlipsReanchorsAndMatchRuns)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  const RecoveryEvent started = fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), params);
  ASSERT_TRUE(started.pending);
  EXPECT_EQ(recovery.start_ns, endOf(0)) << "timed from the last supported scan";
  const std::int64_t deadline = recovery.deadline_ns;
  EXPECT_EQ(deadline, endOf(0) + params.total_ns);
  // Blind / verifying flips and short successful runs: the gate may re-anchor every few scans.
  std::uint64_t seq = 10;
  for (int i = 2; i < 80; ++i) {
    const bool match = i % 4 != 0;  // never 10 scans in a row
    fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(i), params);
    const RecoveryEvent event = fast_lio::observeRecoveryScan(&recovery, seq++, beginOf(i), endOf(i), match, 0, params);
    EXPECT_FALSE(event.qualified);
    EXPECT_EQ(recovery.deadline_ns, deadline);
    EXPECT_EQ(recovery.start_ns, endOf(0));
  }
  EXPECT_EQ(recovery.trust, RecoveryTrust::kPending);
}

TEST(StationaryAnchorPolicy, ExpiryWinsOverARecoveryAtTheSameInstant)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, 0);
  fast_lio::startRecovery(&recovery, RecoveryCause::kScanHole, kPeriod, params);
  // A run of matching scans that reaches the stable time exactly at the deadline.
  const std::int64_t deadline = recovery.deadline_ns;
  std::uint64_t seq = 1;
  for (std::int64_t begin = deadline - params.stable_ns; begin < deadline - kPeriod; begin += kPeriod) {
    fast_lio::expireRecovery(&recovery, begin + kPeriod);
    fast_lio::observeRecoveryScan(&recovery, seq++, begin, begin + kPeriod, true, 0, params);
    ASSERT_EQ(recovery.trust, RecoveryTrust::kPending);
  }
  const RecoveryEvent expired = fast_lio::expireRecovery(&recovery, deadline);
  EXPECT_TRUE(expired.unanchored);
  const RecoveryEvent late =
    fast_lio::observeRecoveryScan(&recovery, seq++, deadline - kPeriod, deadline, true, 0, params);
  EXPECT_FALSE(late.qualified);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
}

TEST(StationaryAnchorPolicy, OneSecondOfUnbrokenMatchingEndsPendingBeforeTheDeadline)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), params);
  std::uint64_t seq = 100;
  int scans = 0;
  for (int i = 30; i < 60; ++i) {
    ++scans;
    fast_lio::expireRecovery(&recovery, endOf(i));
    const RecoveryEvent event = fast_lio::observeRecoveryScan(&recovery, seq++, beginOf(i), endOf(i), true, 0, params);
    if (event.qualified)
      break;
  }
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(scans, 10) << "ten 0.1 s scans make one second";
}

TEST(StationaryAnchorPolicy, AGapARescanOrANewGenerationRestartsTheRun)
{
  const RecoveryParams params = platformPolicy();
  for (int breaker = 0; breaker < 3; ++breaker) {
    RecoveryState recovery;
    fast_lio::noteSupportedScan(&recovery, endOf(0));
    fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), params);
    std::uint64_t seq = 1;
    std::uint64_t generation = 0;
    for (int i = 10; i < 15; ++i)
      fast_lio::observeRecoveryScan(&recovery, seq++, beginOf(i), endOf(i), true, generation, params);
    if (breaker == 0)
      ++seq;  // a scan that never reached the test (no cloud, coverage skip)
    if (breaker == 1)
      fast_lio::breakRecoveryRun(&recovery);  // a gap or a loss
    if (breaker == 2)
      ++generation;
    for (int i = 15; i < 20; ++i) {
      const RecoveryEvent event =
        fast_lio::observeRecoveryScan(&recovery, seq++, beginOf(i), endOf(i), true, generation, params);
      EXPECT_FALSE(event.qualified) << "breaker " << breaker << " scan " << i;
    }
    EXPECT_EQ(recovery.trust, RecoveryTrust::kPending);
  }
}

TEST(StationaryAnchorPolicy, TheScanThatCrossedAnUnobservedImuStretchStartsTheEpisodeAndCountsForNothing)
{
  // An IMU gap the motion model bridged on scan 1: Pending starts at the last supported scan, and the bridged scan
  // itself, however well it registers, is not one of the scans that can end it.
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  fast_lio::excludeRecoveryScan(&recovery, 1);
  const RecoveryEvent started = fast_lio::startRecovery(&recovery, RecoveryCause::kImuBridge, endOf(1), params);
  ASSERT_TRUE(started.pending);
  EXPECT_EQ(started.cause, RecoveryCause::kImuBridge);
  EXPECT_EQ(recovery.start_ns, endOf(0));
  EXPECT_EQ(recovery.deadline_ns, endOf(0) + params.total_ns);
  EXPECT_TRUE(fast_lio::recoveryScanExcluded(recovery, 1));
  EXPECT_FALSE(fast_lio::observeRecoveryScan(&recovery, 1, beginOf(1), endOf(1), true, 0, params).qualified);
  EXPECT_EQ(recovery.stable_begin_ns, -1) << "the bridged scan started a run";
  int scans = 0;
  for (int i = 2; i < 30; ++i) {
    ++scans;
    fast_lio::expireRecovery(&recovery, endOf(i));
    if (fast_lio::observeRecoveryScan(&recovery, static_cast<std::uint64_t>(i), beginOf(i), endOf(i), true, 0, params)
          .qualified)
      break;
  }
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(scans, 10) << "the run starts on the scan after the bridged one";
}

TEST(StationaryAnchorPolicy, AnExcludedScanBreaksTheRunItFallsInto)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), params);
  for (int i = 1; i <= 5; ++i)
    fast_lio::observeRecoveryScan(&recovery, static_cast<std::uint64_t>(i), beginOf(i), endOf(i), true, 0, params);
  ASSERT_EQ(recovery.stable_begin_ns, beginOf(1));
  // Scan 6 carries a lost IMU batch. Pending is already running: no second episode, no new clock.
  fast_lio::excludeRecoveryScan(&recovery, 6);
  EXPECT_EQ(recovery.stable_begin_ns, -1);
  const RecoveryEvent again = fast_lio::startRecovery(&recovery, RecoveryCause::kImuCoverageGap, endOf(6), params);
  EXPECT_FALSE(again.pending);
  EXPECT_EQ(recovery.episode, 1u);
  EXPECT_EQ(recovery.start_ns, endOf(0));
  EXPECT_EQ(recovery.cause, RecoveryCause::kDegraded);
  fast_lio::observeRecoveryScan(&recovery, 6, beginOf(6), endOf(6), true, 0, params);
  EXPECT_EQ(recovery.stable_begin_ns, -1);
  int last = 0;
  for (int i = 7; i < 30 && recovery.trust == RecoveryTrust::kPending; ++i) {
    last = i;
    fast_lio::observeRecoveryScan(&recovery, static_cast<std::uint64_t>(i), beginOf(i), endOf(i), true, 0, params);
  }
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(last, 16) << "one second of scans from the first one after the excluded scan";
}

TEST(StationaryAnchorPolicy, OnlyTheNamedScanIsExcluded)
{
  RecoveryState recovery;
  EXPECT_FALSE(fast_lio::recoveryScanExcluded(recovery, 0)) << "no scan has number 0";
  fast_lio::excludeRecoveryScan(&recovery, 0);
  EXPECT_FALSE(fast_lio::recoveryScanExcluded(recovery, 0));
  fast_lio::excludeRecoveryScan(&recovery, 41);
  EXPECT_TRUE(fast_lio::recoveryScanExcluded(recovery, 41));
  EXPECT_FALSE(fast_lio::recoveryScanExcluded(recovery, 40));
  EXPECT_FALSE(fast_lio::recoveryScanExcluded(recovery, 42));
}

// ── P2-8: re-anchoring and the latch ─────────────────────────────────────────

TEST(StationaryAnchorPolicy, ReanchoringKeepsALiveAnchorButRevivesNoDroppedOne)
{
  // Same continuous parking through a blind stretch and its re-anchor: the anchor was never dropped.
  Lifecycle live;
  const std::uint64_t id = live.commit();
  for (int i = 0; i < 20; ++i)
    EXPECT_TRUE(live.stationary(true).usable);
  EXPECT_TRUE(live.stationary(false).usable);
  EXPECT_EQ(live.anchor.id, id);

  // Dropped, then the gate re-anchors: nothing comes back; the next stretch is new.
  Lifecycle dropped;
  const std::uint64_t old_id = dropped.commit();
  dropped.moving(true);
  for (int i = 0; i < 5; ++i)
    EXPECT_FALSE(dropped.stationary(true).usable);
  EXPECT_TRUE(dropped.stationary(false).started);
  EXPECT_NE(dropped.anchor.id, old_id);
}

TEST(StationaryAnchorPolicy, UnanchoredIsLatchedAgainstLowResidualsAndHealthyScans)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  const RecoveryEvent reset = fast_lio::resetRecoveryReference(&recovery, endOf(1), params);
  EXPECT_TRUE(reset.unanchored);
  EXPECT_EQ(reset.cause, RecoveryCause::kMapReset);
  std::uint64_t seq = 1;
  for (int i = 2; i < 400; ++i) {
    fast_lio::noteSupportedScan(&recovery, endOf(i));
    fast_lio::observeRecoveryScan(&recovery, seq++, beginOf(i), endOf(i), true, 1, params);
    fast_lio::expireRecovery(&recovery, endOf(i));
    EXPECT_FALSE(fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(i), params).pending);
  }
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
  EXPECT_EQ(recovery.cause, RecoveryCause::kMapReset);

  // And no anchor is built on an untrusted reference.
  Lifecycle life;
  for (int i = 0; i < 10; ++i)
    EXPECT_FALSE(life.scan(StaticEvidence::kStationary, false, false).started);
}

TEST(StationaryAnchorPolicy, WithThePolicyOffNothingIsPendingOrLatched)
{
  const RecoveryParams off;  // generic: total 0
  RecoveryState recovery;
  EXPECT_FALSE(fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), off).pending);
  EXPECT_FALSE(fast_lio::resetRecoveryReference(&recovery, endOf(2), off).unanchored);
  EXPECT_FALSE(fast_lio::expireRecovery(&recovery, std::numeric_limits<std::int64_t>::max()).unanchored);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(recovery.map_resets, 1u);
  // The reset still ends the anchor: that part is the lifecycle, on for every profile.
  Lifecycle life;
  life.commit();
  EXPECT_TRUE(fast_lio::changeAnchorReference(&life.anchor, AnchorDrop::kMapReset));
}

TEST(StationaryAnchorPolicy, ALongHoleIsTimedFromItsStartAndExpiresOnTheFirstReturningScan)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  const std::int64_t returning_end = endOf(0) + 30 * kSecond;
  const RecoveryEvent started = fast_lio::startRecovery(&recovery, RecoveryCause::kScanHole, returning_end, params);
  EXPECT_TRUE(started.pending);
  EXPECT_EQ(recovery.start_ns, endOf(0));
  EXPECT_TRUE(fast_lio::expireRecovery(&recovery, returning_end).unanchored);
}

TEST(StationaryAnchorPolicy, NoTimeProgressKeepsPendingAndBackwardTimeMovesNothing)
{
  const RecoveryParams params = platformPolicy();
  RecoveryState recovery;
  fast_lio::noteSupportedScan(&recovery, endOf(0));
  fast_lio::startRecovery(&recovery, RecoveryCause::kDegraded, endOf(1), params);
  const std::int64_t deadline = recovery.deadline_ns;
  for (int i = 0; i < 100; ++i)
    EXPECT_FALSE(fast_lio::expireRecovery(&recovery, endOf(1)).unanchored);
  EXPECT_FALSE(fast_lio::expireRecovery(&recovery, endOf(0) - 50 * kSecond).unanchored);
  EXPECT_FALSE(fast_lio::expireRecovery(&recovery, -1).unanchored);
  EXPECT_EQ(recovery.deadline_ns, deadline);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kPending);
}

TEST(StationaryAnchorPolicy, WithoutAnyTimeAnEpisodeCannotBeMeasuredAndIsNotTrusted)
{
  RecoveryState recovery;
  const RecoveryEvent event = fast_lio::startRecovery(&recovery, RecoveryCause::kScanHole, -1, platformPolicy());
  EXPECT_TRUE(event.unanchored);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
}

// ── P2-9: what counts as static evidence ─────────────────────────────────────

TEST(StationaryAnchorPolicy, StationaryAgreesWithBodyIsStaticEverywhere)
{
  fast_lio::StaticEvidenceParams params;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  int cases = 0;
  for (const bool require : {false, true}) {
    params.require_wheel = require;
    for (const int samples : {0, 1, 20}) {
      for (const double gyr : {0.0, 0.05, 0.0500001, 0.3, nan}) {
        for (const double span : {0.0, 0.19, 0.2, 0.25, inf}) {
          for (const double mean : {9.81, 0.0, -1.0}) {
            for (const double age : {-1.0, 0.0, 0.5, 0.51, 3.0, nan}) {
              for (const double speed : {0.0, 0.05, 0.06, nan, inf}) {
                fast_lio::ImuWindowStats imu;
                imu.samples = samples;
                imu.gyr_max = gyr;
                imu.acc_min = mean;
                imu.acc_max = mean + span;
                imu.acc_mean = mean;
                const bool legacy = fast_lio::bodyIsStatic(imu, age, speed, params);
                const auto verdict = fast_lio::classifyStaticEvidence(imu, wheel(age >= 0.0, age, speed), params);
                ASSERT_EQ(verdict.stationary(), legacy)
                  << "require=" << require << " samples=" << samples << " gyr=" << gyr << " span=" << span
                  << " mean=" << mean << " age=" << age << " speed=" << speed;
                ++cases;
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(cases, 1000);
}

TEST(StationaryAnchorPolicy, OnlyAValidFreshWheelZeroIsAMeasuredZero)
{
  fast_lio::StaticEvidenceParams params;
  const auto measured = fast_lio::classifyStaticEvidence(imuAtRest(), wheel(true, 0.1, 0.0), params);
  EXPECT_EQ(measured.evidence, StaticEvidence::kStationary);
  EXPECT_EQ(measured.reason, StaticEvidenceReason::kImuAndWheel);
  EXPECT_FALSE(measured.imu_only);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // Not measurements: an invalid 0.0, NaN, Inf, a missing sample, a stale one (including one from a clock far
  // away), and ages that are negative or not finite.
  const std::vector<fast_lio::WheelEvidence> not_measured = {wheel(false, 0.1, 0.0),
                                                             wheel(true, 0.1, nan),
                                                             wheel(true, 0.1, inf),
                                                             wheel(false, -1.0, nan),
                                                             wheel(true, 0.6, 0.0),
                                                             wheel(true, 1.0e9, 0.0),
                                                             wheel(true, -0.2, 0.0),
                                                             wheel(true, nan, 0.0),
                                                             wheel(true, inf, 0.0)};
  for (const bool require : {false, true}) {
    params.require_wheel = require;
    for (const auto & w : not_measured) {
      const auto verdict = fast_lio::classifyStaticEvidence(imuAtRest(), w, params);
      if (require) {
        EXPECT_EQ(verdict.evidence, StaticEvidence::kUnknown);
        EXPECT_EQ(verdict.reason, StaticEvidenceReason::kWheelMissing);
      } else {
        EXPECT_EQ(verdict.evidence, StaticEvidence::kStationary);
        EXPECT_EQ(verdict.reason, StaticEvidenceReason::kImuOnly);
        EXPECT_TRUE(verdict.imu_only) << "the IMU alone testified, and the record says so";
      }
    }
  }
}

TEST(StationaryAnchorPolicy, MotionAndMissingProofAreTold)
{
  fast_lio::StaticEvidenceParams params;
  fast_lio::ImuWindowStats turning = imuAtRest();
  turning.gyr_max = 0.3;
  EXPECT_EQ(fast_lio::classifyStaticEvidence(turning, wheel(false, -1, 0), params).reason, StaticEvidenceReason::kGyro);
  fast_lio::ImuWindowStats shaking = imuAtRest();
  shaking.acc_max = 10.5;
  EXPECT_EQ(fast_lio::classifyStaticEvidence(shaking, wheel(false, -1, 0), params).reason,
            StaticEvidenceReason::kAccSpan);
  EXPECT_EQ(fast_lio::classifyStaticEvidence(imuAtRest(), wheel(true, 0.1, 0.4), params).reason,
            StaticEvidenceReason::kWheel);
  const fast_lio::ImuWindowStats empty;
  EXPECT_EQ(fast_lio::classifyStaticEvidence(empty, wheel(true, 0.1, 0.0), params).evidence, StaticEvidence::kUnknown);
  fast_lio::ImuWindowStats broken = imuAtRest();
  broken.acc_mean = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(fast_lio::classifyStaticEvidence(broken, wheel(true, 0.1, 0.0), params).reason,
            StaticEvidenceReason::kImuInvalid);
}

TEST(StationaryAnchorPolicy, ImuOnlyEvidenceCannotBringBackADroppedAnchor)
{
  Lifecycle life;
  const std::uint64_t old_id = life.commit();
  life.scan(StaticEvidence::kUnknown, true);  // a required wheel went missing
  for (int i = 0; i < 3; ++i)
    life.stationary();  // IMU-only stationary, healthy: a NEW anchor
  EXPECT_EQ(life.anchor.state, AnchorState::kValid);
  EXPECT_NE(life.anchor.id, old_id);
}

// ── P2-10: sequences, parameters, restarts ───────────────────────────────────

TEST(StationaryAnchorPolicy, AnExhaustedIdSequenceFailsClosedInsteadOfWrapping)
{
  Lifecycle life;
  life.anchor.last_id = std::numeric_limits<std::uint64_t>::max() - 1;
  const std::uint64_t last = life.commit();
  EXPECT_EQ(last, std::numeric_limits<std::uint64_t>::max());
  life.moving();
  for (int i = 0; i < 10; ++i) {
    const AnchorStep step = life.stationary();
    EXPECT_FALSE(step.started);
    EXPECT_FALSE(step.usable);
    EXPECT_EQ(step.drop, AnchorDrop::kIdExhausted);
  }
  EXPECT_TRUE(life.anchor.ids_exhausted);
  EXPECT_EQ(fast_lio::anchorCurrentId(life.anchor), 0u);
}

TEST(StationaryAnchorPolicy, AnExhaustedGenerationFailsClosed)
{
  Lifecycle life;
  life.anchor.reference_generation = std::numeric_limits<std::uint64_t>::max();
  fast_lio::changeAnchorReference(&life.anchor, AnchorDrop::kMapReset);
  EXPECT_EQ(life.anchor.reference_generation, std::numeric_limits<std::uint64_t>::max());
  for (int i = 0; i < 5; ++i)
    EXPECT_FALSE(life.stationary().started);
}

TEST(StationaryAnchorPolicy, ARestartedProcessStartsFromNothing)
{
  Lifecycle before;
  before.commit();
  const Lifecycle after;  // a new process: nothing carries over
  EXPECT_EQ(after.anchor.state, AnchorState::kEmpty);
  EXPECT_EQ(after.anchor.last_id, 0u);
  EXPECT_EQ(after.anchor.last_evidence_end_ns, -1);
}

TEST(StationaryAnchorPolicy, ParametersOutsideTheirDomainAreRefused)
{
  EXPECT_TRUE(fast_lio::anchorConfirmScansValid(1));
  EXPECT_TRUE(fast_lio::anchorConfirmScansValid(3));
  EXPECT_FALSE(fast_lio::anchorConfirmScansValid(0));
  EXPECT_FALSE(fast_lio::anchorConfirmScansValid(-3));

  std::int64_t gap = 0;
  EXPECT_TRUE(fast_lio::resolveAnchorScanGapNs(0.0, 10.0, &gap));
  EXPECT_EQ(gap, 300000000) << "0 = three scan periods";
  EXPECT_TRUE(fast_lio::resolveAnchorScanGapNs(0.3, 0.0, &gap));
  EXPECT_EQ(gap, 300000000) << "an explicit bound needs no scan rate";
  EXPECT_TRUE(fast_lio::resolveAnchorScanGapNs(0.0, 20.0, &gap));
  EXPECT_EQ(gap, 150000000);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double rate : {0.0, -10.0, nan, inf})
    EXPECT_FALSE(fast_lio::resolveAnchorScanGapNs(0.0, rate, &gap)) << rate;
  for (const double seconds : {-0.3, nan, inf, 1e-12, 1e11})
    EXPECT_FALSE(fast_lio::resolveAnchorScanGapNs(seconds, 10.0, &gap)) << seconds;

  RecoveryParams recovery;
  EXPECT_TRUE(fast_lio::resolveRecoveryParams(0.0, 1.0, &recovery));
  EXPECT_FALSE(recovery.enabled());
  EXPECT_TRUE(fast_lio::resolveRecoveryParams(10.0, 1.0, &recovery));
  EXPECT_TRUE(recovery.enabled());
  EXPECT_EQ(recovery.total_ns, 10 * kSecond);
  EXPECT_EQ(recovery.stable_ns, kSecond);
  EXPECT_FALSE(fast_lio::resolveRecoveryParams(1.0, 1.0, &recovery)) << "stable must be shorter than the total";
  EXPECT_FALSE(fast_lio::resolveRecoveryParams(0.5, 1.0, &recovery));
  EXPECT_TRUE(fast_lio::resolveRecoveryParams(0.0, 30.0, &recovery)) << "with the total off the stable time is unused";
  for (const double total : {-1.0, nan, inf, 1e-12})
    EXPECT_FALSE(fast_lio::resolveRecoveryParams(total, 1.0, &recovery)) << total;
  for (const double stable : {0.0, -1.0, nan, inf})
    EXPECT_FALSE(fast_lio::resolveRecoveryParams(10.0, stable, &recovery)) << stable;
}

// ── P2-11 and K6: what the outputs may claim ─────────────────────────────────

TEST(StationaryAnchorPolicy, StaticHoldIsClaimedOnlyForAHoldOnAUsableAnchor)
{
  for (const bool degraded : {false, true}) {
    for (const bool hold : {false, true}) {
      for (const bool usable : {false, true}) {
        for (const bool lost : {false, true}) {
          for (const RecoveryTrust trust :
               {RecoveryTrust::kLocalQualified, RecoveryTrust::kPending, RecoveryTrust::kUnanchored}) {
            AnchorHealthInputs in;
            in.degraded = degraded;
            in.hold_active = hold;
            in.anchor_usable = usable;
            in.lost = lost;
            in.trust = trust;
            const AnchorHealth health = fast_lio::anchorHealth(in);
            const bool trusted = trust == RecoveryTrust::kLocalQualified;
            EXPECT_EQ(health.effective_degraded, degraded || trust == RecoveryTrust::kUnanchored)
              << "Pending adds nothing to the scan's own verdict";
            EXPECT_EQ(health.reported_static_hold, hold && usable && !lost && trusted);
            EXPECT_EQ(health.lost, lost) << "a loss is reported whatever else the scan is";
            if (!usable) {
              EXPECT_FALSE(health.reported_static_hold) << "a zeroed velocity or a static verdict is not a hold";
            }
          }
        }
      }
    }
  }
}

// Every cell of trust x degraded x hold_active x anchor_usable x lost, written out rather than derived. Degraded is
// the scan's own verdict, set on every Unanchored cell and on no Pending cell that is not degraded itself; StaticHold
// is the one cell per trust with a hold on a usable anchor and no loss, and only on LocalQualified; Lost is the input.
TEST(StationaryAnchorPolicy, HealthTruthTableCellByCell)
{
  struct Cell
  {
    RecoveryTrust trust;
    int degraded, hold, usable, lost;  // inputs
    int want_degraded, want_hold, want_lost;
  };
  constexpr RecoveryTrust kL = RecoveryTrust::kLocalQualified;
  constexpr RecoveryTrust kP = RecoveryTrust::kPending;
  constexpr RecoveryTrust kU = RecoveryTrust::kUnanchored;
  const Cell cells[] = {
    // {trust, degraded, hold, usable, lost, -> degraded, static hold, lost}: one line per trust, degraded and hold;
    // usable and lost run through 00, 01, 10, 11 along it.
    {kL, 0, 0, 0, 0, 0, 0, 0}, {kL, 0, 0, 0, 1, 0, 0, 1}, {kL, 0, 0, 1, 0, 0, 0, 0}, {kL, 0, 0, 1, 1, 0, 0, 1},
    {kL, 0, 1, 0, 0, 0, 0, 0}, {kL, 0, 1, 0, 1, 0, 0, 1}, {kL, 0, 1, 1, 0, 0, 1, 0}, {kL, 0, 1, 1, 1, 0, 0, 1},
    {kL, 1, 0, 0, 0, 1, 0, 0}, {kL, 1, 0, 0, 1, 1, 0, 1}, {kL, 1, 0, 1, 0, 1, 0, 0}, {kL, 1, 0, 1, 1, 1, 0, 1},
    {kL, 1, 1, 0, 0, 1, 0, 0}, {kL, 1, 1, 0, 1, 1, 0, 1}, {kL, 1, 1, 1, 0, 1, 1, 0}, {kL, 1, 1, 1, 1, 1, 0, 1},
    {kP, 0, 0, 0, 0, 0, 0, 0}, {kP, 0, 0, 0, 1, 0, 0, 1}, {kP, 0, 0, 1, 0, 0, 0, 0}, {kP, 0, 0, 1, 1, 0, 0, 1},
    {kP, 0, 1, 0, 0, 0, 0, 0}, {kP, 0, 1, 0, 1, 0, 0, 1}, {kP, 0, 1, 1, 0, 0, 0, 0}, {kP, 0, 1, 1, 1, 0, 0, 1},
    {kP, 1, 0, 0, 0, 1, 0, 0}, {kP, 1, 0, 0, 1, 1, 0, 1}, {kP, 1, 0, 1, 0, 1, 0, 0}, {kP, 1, 0, 1, 1, 1, 0, 1},
    {kP, 1, 1, 0, 0, 1, 0, 0}, {kP, 1, 1, 0, 1, 1, 0, 1}, {kP, 1, 1, 1, 0, 1, 0, 0}, {kP, 1, 1, 1, 1, 1, 0, 1},
    {kU, 0, 0, 0, 0, 1, 0, 0}, {kU, 0, 0, 0, 1, 1, 0, 1}, {kU, 0, 0, 1, 0, 1, 0, 0}, {kU, 0, 0, 1, 1, 1, 0, 1},
    {kU, 0, 1, 0, 0, 1, 0, 0}, {kU, 0, 1, 0, 1, 1, 0, 1}, {kU, 0, 1, 1, 0, 1, 0, 0}, {kU, 0, 1, 1, 1, 1, 0, 1},
    {kU, 1, 0, 0, 0, 1, 0, 0}, {kU, 1, 0, 0, 1, 1, 0, 1}, {kU, 1, 0, 1, 0, 1, 0, 0}, {kU, 1, 0, 1, 1, 1, 0, 1},
    {kU, 1, 1, 0, 0, 1, 0, 0}, {kU, 1, 1, 0, 1, 1, 0, 1}, {kU, 1, 1, 1, 0, 1, 0, 0}, {kU, 1, 1, 1, 1, 1, 0, 1},
  };
  std::set<int> seen;
  for (const Cell & cell : cells) {
    const int key = static_cast<int>(cell.trust) * 16 + cell.degraded * 8 + cell.hold * 4 + cell.usable * 2 + cell.lost;
    EXPECT_TRUE(seen.insert(key).second) << "cell " << key << " is listed twice";
    AnchorHealthInputs in;
    in.degraded = cell.degraded != 0;
    in.hold_active = cell.hold != 0;
    in.anchor_usable = cell.usable != 0;
    in.lost = cell.lost != 0;
    in.trust = cell.trust;
    const AnchorHealth health = fast_lio::anchorHealth(in);
    EXPECT_EQ(health.effective_degraded, cell.want_degraded != 0) << "cell " << key;
    EXPECT_EQ(health.reported_static_hold, cell.want_hold != 0) << "cell " << key;
    EXPECT_EQ(health.lost, cell.want_lost != 0) << "cell " << key;
  }
  EXPECT_EQ(seen.size(), 48u) << "3 trust levels x 2^4 inputs";
}

TEST(StationaryAnchorPolicy, ZeroOnlyNeverPinsWhateverTheSpeed)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double speed : {0.0, 0.4, 2.9, 3.0, 45.0, nan}) {
    fast_lio::ZuptPolicyInputs in;
    in.scan_degraded = true;
    in.zupt_enabled = true;
    in.anchor_valid = false;  // no usable anchor
    in.body_static = true;
    in.speed = speed;
    in.max_speed = 30.0;
    const fast_lio::ZuptDecision decision = fast_lio::decideZupt(in);
    EXPECT_FALSE(decision.hold) << speed;
    EXPECT_EQ(decision.action, fast_lio::ZuptAction::kZeroVelocity) << speed;
  }
}

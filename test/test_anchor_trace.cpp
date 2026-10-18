// The stationary-anchor trace is the only evidence a checker gets of which
// anchor a scan used, so its format is pinned here exactly as anchor_trace.hpp
// specifies it: key order, units, exact round trip of every real, hashes that
// see single bits, and sequences that never wrap onto old values.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "anchor_trace.hpp"

namespace
{

std::vector<std::string> tokens(const std::string & line)
{
  std::istringstream in(line);
  std::vector<std::string> out;
  std::string token;
  while (in >> token) {
    out.push_back(token);
  }
  return out;
}

std::string valueOf(const std::string & line, const std::string & key)
{
  for (const auto & token : tokens(line)) {
    if (token.compare(0, key.size() + 1, key + "=") == 0) {
      return token.substr(key.size() + 1);
    }
  }
  return "<missing>";
}

std::vector<std::string> keysOf(const std::string & line)
{
  std::vector<std::string> keys;
  for (const auto & token : tokens(line)) {
    const auto eq = token.find('=');
    if (eq != std::string::npos) {
      keys.push_back(token.substr(0, eq));
    }
  }
  return keys;
}

bool sameBits(double a, double b)
{
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

}  // namespace

// `v` names the specification a record follows: the header comment of anchor_trace.hpp. A reader rejects a version
// it does not know, so the number moves with any change to what a record says, here as well as there.
TEST(AnchorTrace, TheFormatVersionIsSix)
{
  EXPECT_EQ(fast_lio::kAnchorTraceVersion, 6);
}

TEST(AnchorTrace, TransitionRecordHasTheSpecifiedKeysInOrder)
{
  fast_lio::AnchorTransition t;
  t.event = fast_lio::AnchorTraceEvent::kPinEngaged;
  t.event_seq = 7;
  t.scan_seq = 42;
  t.stamp_ns = 1700000000123456789LL;
  t.anchor_id = 3;
  t.anchor_valid = true;
  t.created_ns = 1700000000000000000LL;
  t.last_evidence_ns = 1700000000100000000LL;
  t.anchor_state = fast_lio::AnchorState::kValid;
  t.episode = 4;
  t.ref_gen = 2;
  t.streak = 5;
  EXPECT_EQ(fast_lio::formatAnchorTransition(t),
            "[ZUPT-ANCHOR] v=6 event=pin_engaged reason=- event_seq=7 scan_seq=42 stamp_ns=1700000000123456789 "
            "anchor_id=3 anchor_valid=1 created_ns=1700000000000000000 last_evidence_ns=1700000000100000000 "
            "anchor_state=valid stationary_episode=4 reference_generation=2 streak=5");
}

TEST(AnchorTrace, InvalidationNamesTheAnchorThatEndedAndWhy)
{
  fast_lio::AnchorTransition t;
  t.event = fast_lio::AnchorTraceEvent::kInvalidated;
  t.reason = fast_lio::anchorTraceReasonOf(fast_lio::AnchorDrop::kScanHole);
  t.anchor_id = 9;
  t.anchor_state = fast_lio::AnchorState::kEmpty;
  const std::string line = fast_lio::formatAnchorTransition(t);
  EXPECT_EQ(valueOf(line, "event"), "invalidated");
  EXPECT_EQ(valueOf(line, "reason"), "scan_hole");
  EXPECT_EQ(valueOf(line, "anchor_id"), "9");
  EXPECT_EQ(valueOf(line, "anchor_valid"), "0");
  EXPECT_EQ(valueOf(line, "anchor_state"), "empty");
}

TEST(AnchorTrace, EveryDropReasonHasItsOwnToken)
{
  std::vector<std::string> seen;
  for (const auto drop : fast_lio::kAnchorInvalidationReasons) {
    const std::string name = fast_lio::anchorTraceReasonName(fast_lio::anchorTraceReasonOf(drop));
    EXPECT_NE(name, "-");
    EXPECT_NE(name, "unknown");
    for (const auto & other : seen) {
      EXPECT_NE(name, other);
    }
    seen.push_back(name);
  }
  EXPECT_EQ(seen.size(), 10u);
  EXPECT_EQ(fast_lio::anchorTraceReasonOf(fast_lio::AnchorDrop::kNone), fast_lio::AnchorTraceReason::kNone);
  EXPECT_EQ(fast_lio::anchorTraceReasonOf(fast_lio::AnchorDrop::kIdExhausted), fast_lio::AnchorTraceReason::kNone);
  // Only a candidate ends for a scan that was not evaluated: the reason has a token and no invalidated_* counter.
  const std::string unevaluated =
    fast_lio::anchorTraceReasonName(fast_lio::anchorTraceReasonOf(fast_lio::AnchorDrop::kScanNotEvaluated));
  EXPECT_EQ(unevaluated, "scan_not_evaluated");
  for (const auto & other : seen) {
    EXPECT_NE(unevaluated, other);
  }
  for (const auto drop : fast_lio::kAnchorInvalidationReasons) {
    EXPECT_NE(drop, fast_lio::AnchorDrop::kScanNotEvaluated);
  }
  EXPECT_LT(static_cast<std::size_t>(fast_lio::AnchorDrop::kScanNotEvaluated), fast_lio::kAnchorDropCount);
}

TEST(AnchorTrace, ACandidateDroppedForAnUnevaluatedScanSaysSoAndCarriesTheClearedStreak)
{
  fast_lio::AnchorTransition t;
  t.event = fast_lio::AnchorTraceEvent::kCandidateDropped;
  t.reason = fast_lio::anchorTraceReasonOf(fast_lio::AnchorDrop::kScanNotEvaluated);
  t.anchor_id = 6;
  t.anchor_state = fast_lio::AnchorState::kEmpty;
  const std::string line = fast_lio::formatAnchorTransition(t);
  EXPECT_EQ(valueOf(line, "event"), "candidate_dropped");
  EXPECT_EQ(valueOf(line, "reason"), "scan_not_evaluated");
  EXPECT_EQ(valueOf(line, "anchor_state"), "empty");
  EXPECT_EQ(valueOf(line, "streak"), "0");
  EXPECT_EQ(keysOf(line).back(), "streak");
}

// An anchor that ends on the IMU window of a scan that stopped before the anchor step ends with one of the two
// reasons the anchor step's own evidence uses, never with scan_not_evaluated, which is a candidate's. Its records,
// in the order the node writes them: the invalidation and the release of the parked prior, numbered on from the
// events before them and belonging to that scan; the scan's record, every anchor-step field na and no event of its
// own; and a counter summary that counts an invalidation by that reason and no evidence the anchor step saw.
TEST(AnchorTrace, AnAnchorEndedOnTheWindowOfAScanThatStoppedEarlyIsAnInvalidationOfThatScan)
{
  using fast_lio::AnchorDrop;
  using fast_lio::AnchorState;
  using fast_lio::StaticEvidence;
  for (const AnchorState state : {AnchorState::kEmpty, AnchorState::kQualifying, AnchorState::kValid}) {
    for (const bool committed : {false, true}) {
      for (const StaticEvidence evidence :
           {StaticEvidence::kStationary, StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
        fast_lio::StationaryAnchor anchor;
        anchor.state = state;
        const AnchorDrop drop = fast_lio::unevaluatedConsumedImuDrop(anchor, committed, evidence);
        const bool ends = committed && state == AnchorState::kValid && evidence != StaticEvidence::kStationary;
        const AnchorDrop expected = !ends                                 ? AnchorDrop::kNone
                                    : evidence == StaticEvidence::kMoving ? AnchorDrop::kMotion
                                                                          : AnchorDrop::kEvidenceUnknown;
        EXPECT_EQ(drop, expected) << static_cast<int>(state) << committed << static_cast<int>(evidence);
      }
    }
  }

  for (const StaticEvidence window : {StaticEvidence::kMoving, StaticEvidence::kUnknown}) {
    const std::string reason = window == StaticEvidence::kMoving ? "motion" : "evidence_unknown";
    const std::string other = window == StaticEvidence::kMoving ? "evidence_unknown" : "motion";
    fast_lio::StationaryAnchor anchor;
    anchor.state = AnchorState::kValid;
    anchor.id = 3;
    anchor.last_id = 3;
    anchor.episode = 2;
    anchor.confirmed_scans = 3;
    anchor.created_ns = 1700000000000000000LL;
    anchor.committed_ns = 1700000000200000000LL;
    anchor.last_evidence_end_ns = 1700000000200000000LL;
    const AnchorDrop drop = fast_lio::unevaluatedConsumedImuDrop(anchor, true, window);
    ASSERT_TRUE(fast_lio::dropAnchor(&anchor, drop));
    bool counted = false;
    for (const auto invalidation : fast_lio::kAnchorInvalidationReasons) {
      counted = counted || invalidation == drop;
    }
    EXPECT_TRUE(counted) << reason << " has no invalidated_* counter";

    // The scan after the anchor's, taken as the eighth; the records of the events come while it is open.
    fast_lio::MonotonicSequence scan_seq;
    fast_lio::MonotonicSequence event_seq;
    scan_seq.skipTo(7);
    event_seq.skipTo(5);
    fast_lio::AnchorScanRecord scan;
    scan.scan_seq = scan_seq.next();
    scan.outcome = fast_lio::AnchorScanOutcome::kBeforeAnchorStep;
    scan.beg_ns = 1700000000200000000LL;
    scan.end_ns = 1700000000300000000LL;
    scan.gap_ns = scan.beg_ns - anchor.last_evidence_end_ns;
    std::vector<std::string> lines;
    fast_lio::AnchorTraceCounters counters;
    for (const auto event :
         {fast_lio::AnchorTraceEvent::kInvalidated, fast_lio::AnchorTraceEvent::kParkedPriorReleased}) {
      fast_lio::AnchorTransition t;
      t.event = event;
      t.reason = fast_lio::anchorTraceReasonOf(drop);
      t.event_seq = event_seq.next();
      t.scan_seq = scan_seq.last();
      t.stamp_ns = scan.end_ns;
      t.anchor_id = anchor.id;
      t.anchor_valid = anchor.state == AnchorState::kValid;
      t.created_ns = anchor.created_ns;
      t.last_evidence_ns = anchor.last_evidence_end_ns;
      t.anchor_state = anchor.state;
      t.episode = anchor.episode;
      fast_lio::saturatingIncrement(&counters.events);
      lines.push_back(fast_lio::formatAnchorTransition(t));
    }
    fast_lio::saturatingIncrement(&counters.invalidated);
    fast_lio::saturatingIncrement(&counters.invalidated_by[static_cast<std::size_t>(drop)]);
    scan.anchor_id = anchor.id;
    scan.anchor_valid = anchor.state == AnchorState::kValid;
    scan.anchor_created_ns = anchor.created_ns;
    scan.anchor_state = anchor.state;
    scan.episode = anchor.episode;
    lines.push_back(fast_lio::formatAnchorScanRecord(scan));
    lines.push_back(fast_lio::formatAnchorCounters(counters, event_seq.next(), false));

    const std::string & ended = lines[0];
    const std::string & released = lines[1];
    const std::string & record = lines[2];
    const std::string & summary = lines[3];
    EXPECT_EQ(valueOf(ended, "event"), "invalidated");
    EXPECT_EQ(valueOf(released, "event"), "parked_prior_released");
    for (const std::string & line : {ended, released}) {
      EXPECT_EQ(valueOf(line, "reason"), reason) << line;
      EXPECT_EQ(valueOf(line, "scan_seq"), "8") << line;
      EXPECT_EQ(valueOf(line, "stamp_ns"), "1700000000300000000") << line;
      EXPECT_EQ(valueOf(line, "anchor_id"), "3") << line;
      EXPECT_EQ(valueOf(line, "anchor_valid"), "0") << line;
      EXPECT_EQ(valueOf(line, "anchor_state"), "empty") << line;
      EXPECT_EQ(valueOf(line, "last_evidence_ns"), "1700000000200000000") << line;
      EXPECT_EQ(valueOf(line, "streak"), "0") << line;
    }
    EXPECT_EQ(valueOf(ended, "event_seq"), "6");
    EXPECT_EQ(valueOf(released, "event_seq"), "7");

    EXPECT_EQ(valueOf(record, "scan_seq"), "8");
    EXPECT_EQ(valueOf(record, "outcome"), "before_anchor_step");
    EXPECT_EQ(valueOf(record, "gap_ns"), "0");
    EXPECT_EQ(valueOf(record, "anchor_id"), "3");
    EXPECT_EQ(valueOf(record, "anchor_valid"), "0");
    EXPECT_EQ(valueOf(record, "anchor_state"), "empty");
    EXPECT_EQ(valueOf(record, "events"), "-") << "the window's verdict is no event of the scan";
    EXPECT_EQ(valueOf(record, "action"), "na");
    EXPECT_EQ(valueOf(record, "anchor_used"), "0");
    for (const char * key :
         {"body",        "imu_n",    "gyr_max",        "acc_span",    "wheel_age", "wheel_speed",     "guard_degraded",
          "degraded",    "parked",   "parked_applied", "zupt_active", "evidence",  "evidence_reason", "imu_only",
          "wheel_valid", "usable",   "zero_only",      "supported",   "act_p",     "act_q",           "act_v",
          "p_pose_act",  "target_p", "target_q",       "anchor_p",    "anchor_q"}) {
      EXPECT_EQ(valueOf(record, key), "na") << key;
    }

    EXPECT_EQ(valueOf(summary, "event_seq"), "8");
    EXPECT_EQ(valueOf(summary, "events"), "2");
    EXPECT_EQ(valueOf(summary, "invalidated"), "1");
    EXPECT_EQ(valueOf(summary, "invalidated_" + reason), "1");
    EXPECT_EQ(valueOf(summary, "invalidated_" + other), "0");
    EXPECT_EQ(valueOf(summary, "candidates_dropped"), "0");
    EXPECT_EQ(valueOf(summary, "motion"), "0") << "the anchor step saw no motion";
    EXPECT_EQ(valueOf(summary, "evidence_unknown"), "0") << "the anchor step saw no unknown evidence";
    for (const auto & line : lines) {
      EXPECT_FALSE(valueOf(line, "event") == "invalidated" && valueOf(line, "reason") == "scan_not_evaluated") << line;
    }
  }
}

TEST(AnchorTrace, RecoveryRecordHasTheSpecifiedKeysInOrder)
{
  fast_lio::RecoveryTransition t;
  t.event = fast_lio::RecoveryTraceEvent::kUnanchored;
  t.reason = fast_lio::RecoveryCause::kDeadline;
  t.event_seq = 12;
  t.scan_seq = 300;
  t.stamp_ns = 1700000010100000000LL;
  t.episode = 2;
  t.start_ns = 1700000000100000000LL;
  t.deadline_ns = 1700000010100000000LL;
  t.elapsed_ns = 10000000000LL;
  t.reset_count = 1;
  EXPECT_EQ(fast_lio::formatRecoveryTransition(t),
            "[FE-RECOVERY] v=6 event=unanchored reason=deadline event_seq=12 scan_seq=300 "
            "stamp_ns=1700000010100000000 episode=2 start_ns=1700000000100000000 "
            "deadline_ns=1700000010100000000 elapsed_ns=10000000000 reset_count=1 path=scan");
  t.path = fast_lio::RecoveryTracePath::kIdle;
  const std::string idle = fast_lio::formatRecoveryTransition(t);
  EXPECT_EQ(valueOf(idle, "path"), "idle");
  EXPECT_EQ(keysOf(idle).back(), "path");
}

TEST(AnchorTrace, ARecoveryStartedByAnImuBridgeHasItsOwnReason)
{
  fast_lio::RecoveryTransition t;
  t.event = fast_lio::RecoveryTraceEvent::kPending;
  t.reason = fast_lio::RecoveryCause::kImuBridge;
  EXPECT_EQ(valueOf(fast_lio::formatRecoveryTransition(t), "reason"), "imu_bridge");
  t.reason = fast_lio::RecoveryCause::kImuCoverageGap;
  EXPECT_EQ(valueOf(fast_lio::formatRecoveryTransition(t), "reason"), "imu_coverage_gap");
}

TEST(AnchorTrace, ScanRecordPrintsEveryKeyEvenWhenTheAnchorStepDidNotRun)
{
  fast_lio::AnchorScanRecord r;
  r.scan_seq = 5;
  r.outcome = fast_lio::AnchorScanOutcome::kImuCoverageGap;
  r.beg_ns = 100;
  r.end_ns = 200;
  r.events = fast_lio::kAnchorEventImuCoverageGap;
  const std::string line = fast_lio::formatAnchorScanRecord(r);
  const std::vector<std::string> expected = {"v",
                                             "scan_seq",
                                             "outcome",
                                             "beg_ns",
                                             "end_ns",
                                             "gap_ns",
                                             "anchor_id",
                                             "anchor_valid",
                                             "anchor_created_ns",
                                             "body",
                                             "imu_n",
                                             "gyr_max",
                                             "acc_span",
                                             "wheel_age",
                                             "wheel_speed",
                                             "guard_degraded",
                                             "degraded",
                                             "gate_pre",
                                             "gate_post",
                                             "action",
                                             "anchor_used",
                                             "parked",
                                             "parked_applied",
                                             "zupt_active",
                                             "events",
                                             "pre_p",
                                             "pre_q",
                                             "pre_v",
                                             "post_p",
                                             "post_q",
                                             "post_v",
                                             "p_pose_pre",
                                             "p_pose_post",
                                             "p_finite",
                                             "p_sym",
                                             "p_min_eig",
                                             "published",
                                             "flags",
                                             "anchor_state",
                                             "stationary_episode",
                                             "reference_generation",
                                             "evidence",
                                             "evidence_reason",
                                             "imu_only",
                                             "wheel_valid",
                                             "usable",
                                             "zero_only",
                                             "supported",
                                             "trust_pre",
                                             "trust_post",
                                             "recovery_episode",
                                             "recovery_start_ns",
                                             "recovery_deadline_ns",
                                             "eff_degraded",
                                             "static_hold",
                                             "act_p",
                                             "act_q",
                                             "act_v",
                                             "p_pose_act",
                                             "target_p",
                                             "target_q",
                                             "anchor_p",
                                             "anchor_q",
                                             "streak",
                                             "map_updated",
                                             "upd_dp",
                                             "upd_dq",
                                             "startup_reference",
                                             "r1_present",
                                             "r1_schema",
                                             "r1_snapshot_sequence",
                                             "r1_state_revision",
                                             "r1_scan_sequence",
                                             "r1_kind",
                                             "r1_lifecycle",
                                             "r1_startup_reference",
                                             "r1_trust",
                                             "r1_local",
                                             "r1_legacy_flags",
                                             "r1_qualified_sequence",
                                             "r1_present_mask",
                                             "r1_true_mask",
                                             "r1_cause_mask",
                                             "r1_taint_mask",
                                             "r1_hard_mask",
                                             "r1_has_pose",
                                             "r1_evaluated_input_ns",
                                             "r1_effective_ns",
                                             "r1_state_valid_until_ns",
                                             "r1_hr_kind",
                                             "r1_hr_valid_until_ns",
                                             "rw_present",
                                             "gprime_present"};
  EXPECT_EQ(keysOf(line), expected);
  EXPECT_EQ(line.rfind("[ZUPT-ANCHOR-SCAN] v=6 ", 0), 0u);
  for (const char * key :
       {"act_p", "act_q", "act_v", "p_pose_act", "target_p", "target_q", "anchor_p", "anchor_q", "upd_dp", "upd_dq"}) {
    EXPECT_EQ(valueOf(line, key), "na") << key;
  }
  EXPECT_EQ(valueOf(line, "streak"), "0");
  EXPECT_EQ(valueOf(line, "map_updated"), "0");
  EXPECT_EQ(valueOf(line, "outcome"), "imu_coverage_gap");
  EXPECT_EQ(valueOf(line, "body"), "na");
  EXPECT_EQ(valueOf(line, "action"), "na");
  EXPECT_EQ(valueOf(line, "pre_p"), "na");
  EXPECT_EQ(valueOf(line, "p_pose_pre"), "na");
  EXPECT_EQ(valueOf(line, "events"), "imu_coverage_gap");
  EXPECT_EQ(valueOf(line, "published"), "0");
  EXPECT_EQ(valueOf(line, "flags"), "-1");
  for (const char * key :
       {"evidence", "evidence_reason", "imu_only", "wheel_valid", "usable", "zero_only", "supported"}) {
    EXPECT_EQ(valueOf(line, key), "na") << key;
  }
  EXPECT_EQ(valueOf(line, "anchor_state"), "empty");
  EXPECT_EQ(valueOf(line, "trust_pre"), "local");
  EXPECT_EQ(valueOf(line, "recovery_start_ns"), "-1");
  EXPECT_EQ(valueOf(line, "eff_degraded"), "0");
}

TEST(AnchorTrace, ScanRecordCarriesTheLifecycleAndTheTrust)
{
  fast_lio::AnchorScanRecord r;
  r.anchor_step = true;
  r.anchor_id = 4;
  r.anchor_state = fast_lio::AnchorState::kQualifying;
  r.episode = 4;
  r.ref_gen = 1;
  r.evidence = fast_lio::StaticEvidence::kStationary;
  r.evidence_reason = fast_lio::StaticEvidenceReason::kImuOnly;
  r.imu_only = true;
  r.zero_only = true;
  r.trust_pre = fast_lio::RecoveryTrust::kPending;
  r.trust_post = fast_lio::RecoveryTrust::kUnanchored;
  r.recovery_episode = 3;
  r.recovery_start_ns = 100;
  r.recovery_deadline_ns = 10000000100LL;
  r.eff_degraded = true;
  const std::string line = fast_lio::formatAnchorScanRecord(r);
  EXPECT_EQ(valueOf(line, "anchor_state"), "qualifying");
  EXPECT_EQ(valueOf(line, "stationary_episode"), "4");
  EXPECT_EQ(valueOf(line, "reference_generation"), "1");
  EXPECT_EQ(valueOf(line, "evidence"), "stationary");
  EXPECT_EQ(valueOf(line, "evidence_reason"), "imu_only");
  EXPECT_EQ(valueOf(line, "imu_only"), "1");
  EXPECT_EQ(valueOf(line, "wheel_valid"), "0");
  EXPECT_EQ(valueOf(line, "usable"), "0");
  EXPECT_EQ(valueOf(line, "zero_only"), "1");
  EXPECT_EQ(valueOf(line, "trust_pre"), "pending");
  EXPECT_EQ(valueOf(line, "trust_post"), "unanchored");
  EXPECT_EQ(valueOf(line, "recovery_episode"), "3");
  EXPECT_EQ(valueOf(line, "recovery_deadline_ns"), "10000000100");
  EXPECT_EQ(valueOf(line, "eff_degraded"), "1");
  EXPECT_EQ(valueOf(line, "static_hold"), "0");
}

// The keys version 3 appends, each group present or na on its own condition: the state and covariance after the
// anchor step's last writer, the target the pin or prior took, the pose the candidate or anchor holds, the parked
// streak, whether the map took the scan's points, and what the lidar update moved.
TEST(AnchorTrace, ScanRecordAppendsTheAnchorWriteItsTargetAndTheUpdate)
{
  fast_lio::AnchorScanRecord r;
  r.have_act = true;
  r.act_p[0] = 1.5;
  r.act_p[1] = -2.0;
  r.act_p[2] = 0.25;
  r.act_q[0] = 0.5;
  r.act_q[1] = 0.5;
  r.act_q[2] = -0.5;
  r.act_q[3] = 0.5;
  r.act_v[2] = 0.125;
  for (int i = 0; i < 36; ++i) {
    r.p_pose_act[i] = (i % 7 == 0) ? 0.5 : 0.0;
  }
  r.have_target = true;
  r.target_p[0] = 3.0;
  r.target_q[3] = 0.25;
  r.have_anchor_pose = true;
  r.anchor_p[1] = 4.0;
  r.anchor_q[1] = -0.75;
  r.streak = 7;
  r.map_updated = true;
  r.have_update = true;
  r.upd_dp = 0.0625;
  r.upd_dq = 0.001953125;
  const std::string line = fast_lio::formatAnchorScanRecord(r);
  char hash[24];
  std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(fast_lio::fnv1a64(r.p_pose_act, 36)));
  const std::string tail = std::string(
                             " static_hold=0 act_p=1.5,-2,0.25 act_q=0.5,0.5,-0.5,0.5 act_v=0,0,0.125 "
                             "p_pose_act=") +
                           hash +
                           " target_p=3,0,0 target_q=1,0,0,0.25 anchor_p=0,4,0 anchor_q=1,-0.75,0,0 streak=7 "
                           "map_updated=1 upd_dp=0.0625 upd_dq=0.001953125 startup_reference=excluded";
  ASSERT_GE(line.size(), tail.size());
  EXPECT_EQ(line.substr(line.size() - tail.size()), tail);

  // Each group is na by itself.
  fast_lio::AnchorScanRecord only_anchor;
  only_anchor.have_anchor_pose = true;
  only_anchor.streak = 2;
  const std::string candidate = fast_lio::formatAnchorScanRecord(only_anchor);
  EXPECT_EQ(valueOf(candidate, "act_p"), "na");
  EXPECT_EQ(valueOf(candidate, "p_pose_act"), "na");
  EXPECT_EQ(valueOf(candidate, "target_q"), "na");
  EXPECT_EQ(valueOf(candidate, "anchor_p"), "0,0,0");
  EXPECT_EQ(valueOf(candidate, "anchor_q"), "1,0,0,0");
  EXPECT_EQ(valueOf(candidate, "streak"), "2");
  EXPECT_EQ(valueOf(candidate, "map_updated"), "0");
  EXPECT_EQ(valueOf(candidate, "upd_dq"), "na");

  fast_lio::AnchorScanRecord zero_only;
  zero_only.have_act = true;
  zero_only.have_update = true;
  zero_only.upd_dp = 0.0;
  zero_only.upd_dq = 0.0;
  const std::string zeroed = fast_lio::formatAnchorScanRecord(zero_only);
  EXPECT_EQ(valueOf(zeroed, "act_v"), "0,0,0");
  EXPECT_EQ(valueOf(zeroed, "p_pose_act").size(), 16u);
  EXPECT_EQ(valueOf(zeroed, "target_p"), "na");
  EXPECT_EQ(valueOf(zeroed, "anchor_q"), "na");
  EXPECT_EQ(valueOf(zeroed, "upd_dp"), "0");
}

// Version 6 preserves the complete v5 prefix, then appends R1 fields.
TEST(AnchorTrace, ScanRecordPreservesStartupBeforeTheR1Extension)
{
  const struct
  {
    fast_lio::StartupReference phase;
    const char * token;
  } phases[] = {
    {fast_lio::StartupReference::kCold, "cold"},
    {fast_lio::StartupReference::kEstablished, "established"},
    {fast_lio::StartupReference::kExcluded, "excluded"},
  };
  for (const auto & phase : phases) {
    fast_lio::AnchorScanRecord r;
    r.startup_reference = phase.phase;
    const std::string line = fast_lio::formatAnchorScanRecord(r);
    EXPECT_LT(line.find(" startup_reference="), line.find(" r1_present="));
    EXPECT_EQ(valueOf(line, "startup_reference"), phase.token);
  }
  EXPECT_EQ(valueOf(fast_lio::formatAnchorScanRecord(fast_lio::AnchorScanRecord()), "startup_reference"), "excluded")
    << "a record nobody filled claims no cold start";

  fast_lio::RecoveryState recovery;
  EXPECT_STREQ(fast_lio::startupReferenceName(fast_lio::recoveryStartupReference(recovery)), "excluded");
  fast_lio::configureRecoveryStartup(&recovery, true);
  EXPECT_STREQ(fast_lio::startupReferenceName(fast_lio::recoveryStartupReference(recovery)), "cold");
  fast_lio::commitInitialReference(&recovery);
  EXPECT_STREQ(fast_lio::startupReferenceName(fast_lio::recoveryStartupReference(recovery)), "established");

  EXPECT_EQ(valueOf(fast_lio::formatAnchorTransition(fast_lio::AnchorTransition()), "startup_reference"), "<missing>");
  EXPECT_EQ(valueOf(fast_lio::formatRecoveryTransition(fast_lio::RecoveryTransition()), "startup_reference"),
            "<missing>");
  EXPECT_EQ(valueOf(fast_lio::formatAnchorCounters(fast_lio::AnchorTraceCounters(), 1, false), "startup_reference"),
            "<missing>");
}

// p_pose_act is the same fingerprint as p_pose_pre: an anchor write that leaves the pose covariance block alone
// shows as two equal hashes.
TEST(AnchorTrace, ThePoseCovarianceAfterTheAnchorWriteHashesLikeTheOneBeforeIt)
{
  fast_lio::AnchorScanRecord r;
  r.have_cov = true;
  r.have_act = true;
  for (int i = 0; i < 36; ++i) {
    r.p_pose_pre[i] = 0.001 * (i + 1);
    r.p_pose_act[i] = r.p_pose_pre[i];
  }
  const std::string same = fast_lio::formatAnchorScanRecord(r);
  EXPECT_EQ(valueOf(same, "p_pose_act"), valueOf(same, "p_pose_pre"));
  r.p_pose_act[17] = std::nextafter(r.p_pose_act[17], 1.0);
  const std::string moved = fast_lio::formatAnchorScanRecord(r);
  EXPECT_NE(valueOf(moved, "p_pose_act"), valueOf(moved, "p_pose_pre"));
}

TEST(AnchorTrace, RealsRoundTripBitExactly)
{
  const double values[] = {0.1,
                           -0.0,
                           0.0,
                           1e-300,
                           -36.33,
                           28.41,
                           std::nextafter(1.0, 2.0),
                           std::numeric_limits<double>::denorm_min(),
                           std::numeric_limits<double>::max()};
  for (const double value : values) {
    fast_lio::AnchorScanRecord r;
    r.have_post = true;
    r.post_p[0] = value;
    const std::string printed = valueOf(fast_lio::formatAnchorScanRecord(r), "post_p");
    const std::string first = printed.substr(0, printed.find(','));
    const double parsed = std::strtod(first.c_str(), nullptr);
    EXPECT_TRUE(sameBits(parsed, value)) << printed;
  }
  fast_lio::AnchorScanRecord r;
  r.have_post = true;
  r.post_v[0] = std::numeric_limits<double>::quiet_NaN();
  r.post_v[1] = std::numeric_limits<double>::infinity();
  r.post_v[2] = -std::numeric_limits<double>::infinity();
  EXPECT_EQ(valueOf(fast_lio::formatAnchorScanRecord(r), "post_v"), "nan,inf,-inf");
}

TEST(AnchorTrace, QuaternionIsPrintedWxyz)
{
  fast_lio::AnchorScanRecord r;
  r.have_pre = true;
  r.pre_q[0] = 1.0;
  r.pre_q[1] = 0.25;
  r.pre_q[2] = 0.5;
  r.pre_q[3] = 0.75;
  EXPECT_EQ(valueOf(fast_lio::formatAnchorScanRecord(r), "pre_q"), "1,0.25,0.5,0.75");
}

TEST(AnchorTrace, CovarianceHashSeesASingleBit)
{
  double block[36];
  for (int i = 0; i < 36; ++i) {
    block[i] = (i % 7 == 0) ? 0.01 * (i + 1) : 0.0;
  }
  double same[36];
  std::memcpy(same, block, sizeof(block));
  EXPECT_EQ(fast_lio::fnv1a64(block, 36), fast_lio::fnv1a64(same, 36));
  same[1] = -0.0;  // equal as a number, different as bits
  EXPECT_NE(fast_lio::fnv1a64(block, 36), fast_lio::fnv1a64(same, 36));
  std::memcpy(same, block, sizeof(block));
  same[35] = std::nextafter(same[35], 1.0);
  EXPECT_NE(fast_lio::fnv1a64(block, 36), fast_lio::fnv1a64(same, 36));

  fast_lio::AnchorScanRecord r;
  r.have_cov = true;
  std::memcpy(r.p_pose_pre, block, sizeof(block));
  std::memcpy(r.p_pose_post, block, sizeof(block));
  const std::string line = fast_lio::formatAnchorScanRecord(r);
  EXPECT_EQ(valueOf(line, "p_pose_pre"), valueOf(line, "p_pose_post"));
  EXPECT_EQ(valueOf(line, "p_pose_pre").size(), 16u);
}

TEST(AnchorTrace, EventsArePrintedInBitOrder)
{
  EXPECT_EQ(fast_lio::formatAnchorEvents(0u), "-");
  EXPECT_EQ(fast_lio::formatAnchorEvents(fast_lio::kAnchorEventTimeEpoch | fast_lio::kAnchorEventMotion |
                                         fast_lio::kAnchorEventLost),
            "motion,lost,time_epoch");
  std::uint32_t all = 0u;
  for (int bit = 0; bit < fast_lio::kAnchorEventBitCount; ++bit) {
    all |= 1u << bit;
  }
  EXPECT_EQ(fast_lio::formatAnchorEvents(all),
            "motion,scan_hole,imu_coverage_gap,imu_bridge,map_reset,reference_changed,lost,reanchored,runaway_"
            "trip,time_epoch,evidence_unknown");
}

TEST(AnchorTrace, EveryEnumValueHasASpaceFreeToken)
{
  const auto check = [](const char * token) {
    const std::string text(token);
    EXPECT_NE(text, "unknown");
    EXPECT_EQ(text.find(' '), std::string::npos) << text;
    EXPECT_EQ(text.find('='), std::string::npos) << text;
  };
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceEvent::kInvalidated); ++i) {
    check(fast_lio::anchorTraceEventName(static_cast<fast_lio::AnchorTraceEvent>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceReason::kScanNotEvaluated); ++i) {
    check(fast_lio::anchorTraceReasonName(static_cast<fast_lio::AnchorTraceReason>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorState::kValid); ++i) {
    check(fast_lio::anchorStateName(static_cast<fast_lio::AnchorState>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::StaticEvidenceReason::kWheelMissing); ++i) {
    check(fast_lio::staticEvidenceReasonName(static_cast<fast_lio::StaticEvidenceReason>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::RecoveryTrust::kUnanchored); ++i) {
    check(fast_lio::recoveryTrustName(static_cast<fast_lio::RecoveryTrust>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::StartupReference::kExcluded); ++i) {
    check(fast_lio::startupReferenceName(static_cast<fast_lio::StartupReference>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::RecoveryCause::kStable); ++i) {
    check(fast_lio::recoveryCauseName(static_cast<fast_lio::RecoveryCause>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::RecoveryTraceEvent::kUnanchored); ++i) {
    check(fast_lio::recoveryTraceEventName(static_cast<fast_lio::RecoveryTraceEvent>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::RecoveryTracePath::kIdle); ++i) {
    check(fast_lio::recoveryTracePathName(static_cast<fast_lio::RecoveryTracePath>(i)));
  }
  // "unknown" is one of the evidence values, so these are pinned one by one.
  EXPECT_STREQ(fast_lio::staticEvidenceName(fast_lio::StaticEvidence::kStationary), "stationary");
  EXPECT_STREQ(fast_lio::staticEvidenceName(fast_lio::StaticEvidence::kMoving), "moving");
  EXPECT_STREQ(fast_lio::staticEvidenceName(fast_lio::StaticEvidence::kUnknown), "unknown");
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorScanOutcome::kDroppedEmptyImu); ++i) {
    check(fast_lio::anchorScanOutcomeName(static_cast<fast_lio::AnchorScanOutcome>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceGate::kLost); ++i) {
    check(fast_lio::anchorTraceGateName(static_cast<fast_lio::AnchorTraceGate>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceBody::kMoving); ++i) {
    check(fast_lio::anchorTraceBodyName(static_cast<fast_lio::AnchorTraceBody>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceAction::kZeroVelocity); ++i) {
    check(fast_lio::anchorTraceActionName(static_cast<fast_lio::AnchorTraceAction>(i)));
  }
  for (int bit = 0; bit < fast_lio::kAnchorEventBitCount; ++bit) {
    check(fast_lio::anchorEventBitName(bit));
  }
  EXPECT_STREQ(fast_lio::anchorEventBitName(fast_lio::kAnchorEventBitCount), "unknown");
}

TEST(AnchorTrace, CountersSaturateInsteadOfWrapping)
{
  std::uint64_t counter = std::numeric_limits<std::uint64_t>::max() - 1;
  fast_lio::saturatingIncrement(&counter);
  EXPECT_EQ(counter, std::numeric_limits<std::uint64_t>::max());
  fast_lio::saturatingIncrement(&counter);
  EXPECT_EQ(counter, std::numeric_limits<std::uint64_t>::max());
}

TEST(AnchorTrace, SequenceStartsAtOneAndFailsClosedAtTheTop)
{
  fast_lio::MonotonicSequence sequence;
  EXPECT_EQ(sequence.last(), 0u);
  EXPECT_EQ(sequence.next(), 1u);
  EXPECT_EQ(sequence.next(), 2u);
  sequence.skipTo(std::numeric_limits<std::uint64_t>::max() - 1);
  EXPECT_FALSE(sequence.exhausted());
  EXPECT_EQ(sequence.next(), std::numeric_limits<std::uint64_t>::max());
  EXPECT_TRUE(sequence.exhausted());
  // Exhausted: 0, never a wrapped small number that could alias an old ID.
  EXPECT_EQ(sequence.next(), 0u);
  EXPECT_EQ(sequence.next(), 0u);
  EXPECT_TRUE(sequence.exhausted());
}

TEST(AnchorTrace, SensorNanosecondsRoundAndRejectWhatTheyCannotHold)
{
  EXPECT_EQ(fast_lio::traceNs(1.5), 1500000000LL);
  EXPECT_EQ(fast_lio::traceNs(0.0), 0LL);
  EXPECT_EQ(fast_lio::traceNs(1e-9), 1LL);
  EXPECT_EQ(fast_lio::traceNs(std::numeric_limits<double>::quiet_NaN()), -1LL);
  EXPECT_EQ(fast_lio::traceNs(std::numeric_limits<double>::infinity()), -1LL);
  EXPECT_EQ(fast_lio::traceNs(1e10), -1LL);  // 1e19 ns does not fit
  // A 2026 epoch stamp is carried to well under a microsecond.
  const double stamp = 1791590400.25;
  EXPECT_NEAR(static_cast<double>(fast_lio::traceNs(stamp) - 1791590400250000000LL), 0.0, 1000.0);
}

TEST(AnchorTrace, CounterSummaryCarriesEveryCounter)
{
  fast_lio::AnchorTraceCounters c;
  c.scans = 10;
  c.events = 4;
  c.candidates = 3;
  c.created = 2;
  c.candidates_dropped = 1;
  c.invalidated = 2;
  c.invalidated_by[static_cast<std::size_t>(fast_lio::AnchorDrop::kMotion)] = 21;
  c.invalidated_by[static_cast<std::size_t>(fast_lio::AnchorDrop::kNotQualified)] = 22;
  c.pin_applied = 2;
  c.parked_prior_applied = 1;
  c.zero_only_unanchored = 14;
  c.blocked_stale_use = 15;
  c.evidence_unknown = 16;
  c.motion = 5;
  c.scan_hole = 6;
  c.imu_coverage_gap = 7;
  c.imu_bridge = 8;
  c.map_reset = 9;
  c.reference_changed = 11;
  c.lost = 12;
  c.time_epoch = 13;
  c.recovery_started = 17;
  c.recovery_qualified = 18;
  c.recovery_expired = 19;
  c.map_reset_unanchored = 20;
  const std::string line = fast_lio::formatAnchorCounters(c, 99, true);
  const std::vector<std::string> expected = {"v",
                                             "event",
                                             "event_seq",
                                             "scans",
                                             "events",
                                             "candidates",
                                             "created",
                                             "candidates_dropped",
                                             "invalidated",
                                             "invalidated_motion",
                                             "invalidated_evidence_unknown",
                                             "invalidated_scan_hole",
                                             "invalidated_imu_coverage_gap",
                                             "invalidated_imu_bridge",
                                             "invalidated_map_reset",
                                             "invalidated_reference_changed",
                                             "invalidated_time_epoch",
                                             "invalidated_lost",
                                             "invalidated_not_qualified",
                                             "pin_applied",
                                             "parked_prior_applied",
                                             "zero_only_unanchored",
                                             "blocked_stale_use",
                                             "evidence_unknown",
                                             "motion",
                                             "scan_hole",
                                             "imu_coverage_gap",
                                             "imu_bridge",
                                             "map_reset",
                                             "reference_changed",
                                             "lost",
                                             "time_epoch",
                                             "recovery_started",
                                             "recovery_qualified",
                                             "recovery_expired",
                                             "map_reset_unanchored",
                                             "sequence_exhausted"};
  EXPECT_EQ(keysOf(line), expected);
  EXPECT_EQ(line.rfind("[ZUPT-ANCHOR] v=6 ", 0), 0u);
  EXPECT_EQ(valueOf(line, "event"), "counters");
  EXPECT_EQ(valueOf(line, "event_seq"), "99");
  EXPECT_EQ(valueOf(line, "invalidated_motion"), "21");
  EXPECT_EQ(valueOf(line, "invalidated_not_qualified"), "22");
  EXPECT_EQ(valueOf(line, "invalidated_scan_hole"), "0");
  EXPECT_EQ(valueOf(line, "reference_changed"), "11");
  EXPECT_EQ(valueOf(line, "zero_only_unanchored"), "14");
  EXPECT_EQ(valueOf(line, "blocked_stale_use"), "15");
  EXPECT_EQ(valueOf(line, "map_reset_unanchored"), "20");
  EXPECT_EQ(valueOf(line, "sequence_exhausted"), "1");
}

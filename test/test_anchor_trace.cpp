// The stationary-anchor trace is the only evidence a checker gets of which
// anchor a scan used, so its format is pinned here exactly as anchor_trace.hpp
// specifies it: key order, units, exact round trip of every real, hashes that
// see single bits, and sequences that never wrap onto old values.

#include <cmath>
#include <cstdint>
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
  EXPECT_EQ(fast_lio::formatAnchorTransition(t),
            "[ZUPT-ANCHOR] v=1 event=pin_engaged reason=- event_seq=7 scan_seq=42 stamp_ns=1700000000123456789 "
            "anchor_id=3 anchor_valid=1 created_ns=1700000000000000000 last_evidence_ns=1700000000100000000");
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
                                             "flags"};
  EXPECT_EQ(keysOf(line), expected);
  EXPECT_EQ(line.rfind("[ZUPT-ANCHOR-SCAN] ", 0), 0u);
  EXPECT_EQ(valueOf(line, "outcome"), "imu_coverage_gap");
  EXPECT_EQ(valueOf(line, "body"), "na");
  EXPECT_EQ(valueOf(line, "action"), "na");
  EXPECT_EQ(valueOf(line, "pre_p"), "na");
  EXPECT_EQ(valueOf(line, "p_pose_pre"), "na");
  EXPECT_EQ(valueOf(line, "events"), "imu_coverage_gap");
  EXPECT_EQ(valueOf(line, "published"), "0");
  EXPECT_EQ(valueOf(line, "flags"), "-1");
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
            "trip,time_epoch");
}

TEST(AnchorTrace, EveryEnumValueHasASpaceFreeToken)
{
  const auto check = [](const char * token) {
    const std::string text(token);
    EXPECT_NE(text, "unknown");
    EXPECT_EQ(text.find(' '), std::string::npos) << text;
    EXPECT_EQ(text.find('='), std::string::npos) << text;
  };
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceEvent::kEvidence); ++i) {
    check(fast_lio::anchorTraceEventName(static_cast<fast_lio::AnchorTraceEvent>(i)));
  }
  for (int i = 0; i <= static_cast<int>(fast_lio::AnchorTraceReason::kScanHealthy); ++i) {
    check(fast_lio::anchorTraceReasonName(static_cast<fast_lio::AnchorTraceReason>(i)));
  }
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
  c.anchor_refreshed = 3;
  c.pin_applied = 2;
  c.parked_prior_applied = 1;
  c.motion = 5;
  c.scan_hole = 6;
  c.imu_coverage_gap = 7;
  c.imu_bridge = 8;
  c.map_reset = 9;
  c.reference_changed = 11;
  c.lost = 12;
  c.time_epoch = 13;
  const std::string line = fast_lio::formatAnchorCounters(c, 99, true);
  const std::vector<std::string> expected = {"v",
                                             "event",
                                             "event_seq",
                                             "scans",
                                             "events",
                                             "anchor_refreshed",
                                             "pin_applied",
                                             "parked_prior_applied",
                                             "motion",
                                             "scan_hole",
                                             "imu_coverage_gap",
                                             "imu_bridge",
                                             "map_reset",
                                             "reference_changed",
                                             "lost",
                                             "time_epoch",
                                             "sequence_exhausted"};
  EXPECT_EQ(keysOf(line), expected);
  EXPECT_EQ(valueOf(line, "event"), "counters");
  EXPECT_EQ(valueOf(line, "event_seq"), "99");
  EXPECT_EQ(valueOf(line, "reference_changed"), "11");
  EXPECT_EQ(valueOf(line, "sequence_exhausted"), "1");
}

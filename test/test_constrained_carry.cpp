// Tests for constrained_carry.hpp: origin and raw-episode lifecycle, the fixed certificate gates and bounds, leg
// witnesses, taint/hard latching and the legal recoveries that alone clear them.
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <deque>
#include <gtest/gtest.h>
#include <limits>

#include "constrained_carry.hpp"

namespace
{
constexpr std::int64_t kMs = 1000000;
constexpr std::int64_t kScanNs = 100 * kMs;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kLateDeadline = std::numeric_limits<std::int64_t>::max();

struct Facts
{
  bool degraded = false, pureEngulf = false, staticHold = false;
  fast_lio::RecoveryTrust trust = fast_lio::RecoveryTrust::kLocalQualified;
  std::int64_t deadlineNs = -1;
  double residual = 0.03;
};

Facts engulfed(std::int64_t deadlineNs = kLateDeadline)
{
  return {true, true, false, fast_lio::RecoveryTrust::kPending, deadlineNs};
}

// 10 Hz scans of a platform moving along world x; 200 Hz gyro and 50 Hz legs from the same motion unless changed.
class Drive
{
public:
  double speed = 1.0;      // FE base speed, m/s
  double yawRate = 0.0;    // FE yaw rate, rad/s
  double gyroExtra = 0.0;  // gyro minus FE about gravity, rad/s
  double climb = 0.0;      // FE vertical speed, m/s
  double legs = 1.0;       // reported leg speed; NaN is "cannot measure"
  bool imu = true;
  std::uint64_t prepareShift = 0;  // nonzero prepares another scan sequence than the final one
  std::int64_t stableNs = 1000 * kMs;
  int gapNegatives = 0;
  fast_lio::ConstrainedCarry carry;
  std::deque<fast_lio::CarryWheelSample> wheel;

  // As the node: the IMU and the window before the #273 step, a new structural gap reported once as a negative.
  fast_lio::CarryVerdict scan(const Facts & facts = {}, std::uint64_t skip = 0)
  {
    const std::int64_t begin = end_;
    for (std::int64_t t = begin; imu && t < begin + kScanNs; t += 5 * kMs)
      carry.noteImu(t, Eigen::Vector3d(0.0, 0.0, yawRate + gyroExtra));
    for (std::int64_t t = begin + 20 * kMs; t <= begin + kScanNs; t += 20 * kMs)
      wheel.push_back({t, legs});
    end_ = begin + kScanNs;
    x_ += speed * 0.1;
    z_ += climb * 0.1;
    yaw_ += yawRate * 0.1;
    sequence_ += 1 + skip;
    if (carry.prepareScanWindow(sequence_ + prepareShift, end_, 1, facts.trust).newGap) {
      carry.negative(fast_lio::kReferenceImuCoverage, 0);
      ++gapNegatives;
    }
    fast_lio::CarryScan in;
    in.sequence = sequence_;
    in.beginNs = begin;
    in.endNs = end_;
    in.generation = 1;
    in.established = true;
    in.trust = facts.trust;
    in.deadlineNs = facts.deadlineNs;
    in.degraded = facts.degraded;
    in.pureEngulf = facts.pureEngulf;
    in.staticHold = facts.staticHold;
    in.iterations = 3;
    in.effective = 500;
    in.residual = facts.residual;
    in.base = Eigen::Vector3d(x_, 0.0, z_);
    in.rotation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()));
    return carry.scan(in, wheel, stableNs);
  }

  void origin(int scans = 20)
  {
    for (int i = 0; i < scans; ++i)
      scan();
  }

  std::int64_t end() const { return end_; }

private:
  std::int64_t end_ = 1000000 * kMs;
  std::uint64_t sequence_ = 0;
  double x_ = 0.0, z_ = 0.0, yaw_ = 0.0;
};
}  // namespace

TEST(ConstrainedCarry, PureEngulfOnTwentySampleOriginCertifiesPendingAndItsRawClearTail)
{
  Drive drive;
  drive.origin();
  ASSERT_TRUE(drive.carry.originQualified());
  const auto first = drive.scan(engulfed());
  EXPECT_TRUE(first.permitted);
  EXPECT_TRUE(first.constrained);
  EXPECT_EQ(first.witnessKind, fast_lio::kCarryWitnessCurrent);
  EXPECT_EQ(first.validUntilNs, drive.end() + 200 * kMs);
  EXPECT_EQ(drive.carry.rawStartNs(), drive.end());
  const auto rawStart = drive.end();

  // Released but #273 still Pending: the same origin may vouch, without a raw-degraded odometry claim.
  const auto tail = drive.scan({false, false, false, fast_lio::RecoveryTrust::kPending, kLateDeadline});
  EXPECT_TRUE(tail.permitted);
  EXPECT_FALSE(tail.constrained);
  EXPECT_EQ(drive.carry.rawStartNs(), rawStart);

  // Local-qualified enumeration without a qualified event is no recovery; the event plus a trusted scan is.
  EXPECT_FALSE(drive.scan().rearmed);
  EXPECT_TRUE(drive.carry.rawActive());
  drive.carry.qualified();
  const auto back = drive.scan();
  EXPECT_TRUE(back.rearmed);
  EXPECT_FALSE(back.permitted);
  EXPECT_FALSE(drive.carry.rawActive());
  EXPECT_EQ(drive.carry.rearmEvents(), 1u);
}

TEST(ConstrainedCarry, ProjectionIsAValidRwBlockThroughTheEpisode)
{
  Drive drive;
  fast_lio::RecoveryState recovery;
  fast_lio::configureRecoveryStartup(&recovery, true);
  fast_lio::commitInitialReference(&recovery);
  fast_lio::ReferenceHealthProgress progress;
  auto project = [&](const Facts & facts) {
    const auto verdict = drive.scan(facts);
    fast_lio::ReferenceHealthSnapshot value;
    value.kind = 1;
    value.enabledPolicyMask = fast_lio::kReferenceR1Enabled | 2u;
    value.hasSourceInstance = value.hasCoordinateEpoch = value.hasObservationId = true;
    value.hasScanSequence = value.hasScanBegin = value.hasScanEnd = value.hasPoseBinding = true;
    value.pose = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    value.scanBeginNs = drive.end() - kScanNs;
    value.scanEndNs = drive.end();
    value.legacyFlags = facts.degraded ? 1u : 0u;
    recovery.trust = facts.trust;
    recovery.deadline_ns = facts.deadlineNs;
    fast_lio::ReferenceScanFacts scan;
    scan.completed = scan.initialized = scan.assessed = true;
    scan.rawDegraded = facts.degraded;
    value = fast_lio::projectReferenceHealth(value, recovery, scan);
    value = progress.settle(fast_lio::projectCarry(value, drive.carry, verdict));
    EXPECT_TRUE(fast_lio::referenceHealthEncodingValid(value));
    EXPECT_EQ(value.hasRwValidUntil, verdict.permitted);
    return value;
  };
  for (int i = 0; i < 19; ++i)
    EXPECT_EQ(project({}).trueMask & 48u, 0u);
  EXPECT_EQ(project({}).trueMask & 48u, 16u);
  const auto deadline = drive.end() + 350 * kMs;  // the next scan's end + 250 ms
  const auto carried = project(engulfed(deadline));
  EXPECT_EQ(carried.trueMask & 48u, 48u);
  EXPECT_EQ(carried.rwValidUntilNs, drive.end() + 200 * kMs);
  EXPECT_EQ(carried.rawStartNs, drive.end());
  EXPECT_DOUBLE_EQ(carried.frozenResidualMedian, 0.03);
  const auto capped = project(engulfed(deadline));
  EXPECT_EQ(capped.rwValidUntilNs, deadline - 1);
  drive.carry.qualified();
  const auto back = project({});
  EXPECT_EQ(back.trueMask & 48u, 16u);
  EXPECT_FALSE(back.hasRawStart);
  EXPECT_EQ(back.withdrawalRearmEvent, 1u);
}

TEST(ConstrainedCarry, OriginNeedsTwentyTrustedSamplesAndAPureEngulfStart)
{
  Drive shortHistory;
  shortHistory.origin(19);
  EXPECT_FALSE(shortHistory.carry.originQualified());
  EXPECT_FALSE(shortHistory.scan(engulfed()).permitted);
  EXPECT_FALSE(shortHistory.scan(engulfed()).permitted);  // a later sample cannot qualify a started episode

  Drive otherCause;
  otherCause.origin();
  EXPECT_FALSE(otherCause.scan({true, false, false, fast_lio::RecoveryTrust::kPending}).permitted);
  EXPECT_FALSE(otherCause.scan(engulfed()).permitted);  // engulf joining a non-engulf episode gets no provenance
  EXPECT_FALSE(otherCause.carry.originQualified());
}

TEST(ConstrainedCarry, CertificateStopsAtRawBudgetAndRecoveryDeadline)
{
  Drive drive;
  drive.origin();
  drive.scan(engulfed());
  const auto rawStart = drive.carry.rawStartNs();
  while (drive.end() < rawStart + 7900 * kMs)
    ASSERT_TRUE(drive.scan(engulfed()).permitted);
  EXPECT_EQ(drive.scan(engulfed()).validUntilNs, rawStart + 8000 * kMs);  // end = rawStart + 8.0 s, inclusive
  EXPECT_FALSE(drive.scan(engulfed()).permitted);                         // end past the raw budget

  Drive pending;
  pending.origin();
  const auto deadline = pending.end() + kScanNs + 50 * kMs;
  EXPECT_EQ(pending.scan(engulfed(deadline)).validUntilNs, deadline - 1);
  EXPECT_FALSE(pending.scan(engulfed(pending.end() + kScanNs)).permitted);  // deadline at the scan end: expired
  EXPECT_FALSE(pending.scan(engulfed(-1)).permitted);                       // a Pending without its deadline
  EXPECT_TRUE(pending.scan(engulfed()).permitted);
}

TEST(ConstrainedCarry, StaticHoldNeitherCertifiesNorRestartsTheRawBudget)
{
  Drive drive;
  drive.origin();
  const Facts held{true, false, true, fast_lio::RecoveryTrust::kLocalQualified};
  EXPECT_FALSE(drive.scan(held).permitted);
  const auto rawStart = drive.carry.rawStartNs();
  for (int i = 0; i < 30; ++i)
    EXPECT_FALSE(drive.scan(held).permitted);
  EXPECT_TRUE(drive.scan(engulfed()).permitted);  // the held episode attaches the engulf that follows it
  EXPECT_FALSE(drive.scan(held).permitted);
  EXPECT_TRUE(drive.scan(engulfed()).permitted);
  EXPECT_EQ(drive.carry.rawStartNs(), rawStart);
  EXPECT_EQ(drive.carry.rawEpisode(), 1u);
}

TEST(ConstrainedCarry, CurrentLegContradictionIsFinal)
{
  Drive fast;
  fast.speed = 2.14;
  fast.legs = 0.9;
  fast.origin();
  for (int i = 0; i < 5; ++i)
    EXPECT_FALSE(fast.scan(engulfed()).permitted);

  Drive slowed;
  slowed.legs = 1.8;
  slowed.speed = 1.8;
  slowed.origin();
  for (int i = 0; i < 6; ++i)
    EXPECT_TRUE(slowed.scan(engulfed()).permitted);
  slowed.legs = 0.1;
  slowed.speed = 0.5;
  for (int i = 0; i < 4; ++i)
    slowed.scan(engulfed());
  EXPECT_FALSE(slowed.scan(engulfed()).permitted);  // the earlier 1.8 cannot widen a sufficient current 0.1
}

TEST(ConstrainedCarry, InsufficientCurrentLegsFallBackToRealWitnessesOnly)
{
  const std::int64_t end = 1000000 * kMs;
  const std::int64_t start = end - 300 * kMs;
  std::deque<fast_lio::CarryWheelSample> wheel{{end - 500 * kMs, 1.0}};
  auto legs = fast_lio::carryWheelEvidence(wheel, start, end, 1.0);
  EXPECT_TRUE(legs.supported);
  EXPECT_EQ(legs.kind, fast_lio::kCarryWitnessFallback);
  EXPECT_EQ(legs.untilNs, end);
  wheel.front().ns -= 1;  // newest finite sample 1 ns past the 0.5 s freshness
  EXPECT_FALSE(fast_lio::carryWheelEvidence(wheel, start, end, 1.0).supported);

  // 2.14 m/s is above anything a recent 0.1 supports; the 1.8 peak supports it only until its own epoch + 1 s.
  wheel = {{end - 950 * kMs, 1.8}, {end - 400 * kMs, 0.1}};
  legs = fast_lio::carryWheelEvidence(wheel, start, end, 2.14);
  EXPECT_TRUE(legs.supported);
  EXPECT_EQ(legs.upperNs, end - 950 * kMs);
  EXPECT_EQ(legs.lowerNs, end - 400 * kMs);
  EXPECT_EQ(legs.untilNs, end + 50 * kMs);
  wheel.front().ns = end - 1000 * kMs - 1;
  EXPECT_FALSE(fast_lio::carryWheelEvidence(wheel, start, end, 2.14).supported);

  // NaN keeps its place in the raw count; a dense finite current window is still sufficient and decides alone.
  wheel.clear();
  for (std::int64_t t = start; t <= end; t += 20 * kMs)
    wheel.push_back({t, (t / (20 * kMs)) % 4 == 0 ? kNaN : 0.1});
  EXPECT_FALSE(fast_lio::carryWheelEvidence(wheel, start, end, 2.14).supported);
  EXPECT_TRUE(std::isnan(fast_lio::carryWheelSpeed(0.3, 0.4, kNaN)));
  EXPECT_TRUE(std::isnan(fast_lio::carryWheelSpeed(kNaN, 0.0, 0.0)));
  EXPECT_DOUBLE_EQ(fast_lio::carryWheelSpeed(0.3, 0.4, 9.0), 0.5);
}

TEST(ConstrainedCarry, KinematicAndMeasurementGatesReject)
{
  auto certified = [](void (*change)(Drive &), const Facts & facts) {
    Drive drive;
    drive.origin();
    change(drive);
    for (int i = 0; i < 3; ++i)
      drive.scan(facts);
    return drive.scan(facts).permitted;
  };
  EXPECT_TRUE(certified([](Drive & d) { d.gyroExtra = 0.2; }, engulfed()));   // 11.5 deg/s
  EXPECT_FALSE(certified([](Drive & d) { d.gyroExtra = 0.3; }, engulfed()));  // 17.2 deg/s
  EXPECT_TRUE(certified([](Drive & d) { d.yawRate = 0.8; }, engulfed()));     // FE and gyro agree
  EXPECT_FALSE(certified([](Drive & d) { d.climb = 0.6; }, engulfed()));
  Facts loose = engulfed();
  loose.residual = 0.061;  // above min(.10, 2 * max(frozen .03, .03))
  EXPECT_FALSE(certified([](Drive &) {}, loose));

  // A scan without gyro is one structural gap: one negative, a taint that outlives the window, and only a qualified
  // event after it, or a 20-scan stable run after it, rearms.
  Drive gap;
  gap.origin();
  EXPECT_TRUE(gap.scan(engulfed()).permitted);
  gap.carry.qualified();  // a receipt before the gap does not count
  gap.imu = false;
  EXPECT_FALSE(gap.scan(engulfed()).permitted);
  gap.imu = true;
  for (int i = 0; i < 6; ++i)
    EXPECT_FALSE(gap.scan(engulfed()).permitted);  // the window rolled past the gap
  EXPECT_EQ(gap.gapNegatives, 1);
  EXPECT_EQ(gap.carry.taint(), fast_lio::kReferenceImuCoverage);
  EXPECT_FALSE(gap.scan().rearmed);
  gap.carry.qualified();
  EXPECT_TRUE(gap.scan().rearmed);

  Drive local;
  local.origin();
  local.imu = false;
  local.scan();
  local.imu = true;
  for (int i = 0; i < 21; ++i)
    EXPECT_FALSE(local.scan().rearmed);  // two more windows hold the gap; 19 covered scans are not 20
  EXPECT_TRUE(local.scan().rearmed);
  EXPECT_EQ(local.gapNegatives, 1);

  // The window a scan was prepared with certifies that scan only.
  Drive other;
  other.origin();
  other.prepareShift = 1;
  EXPECT_FALSE(other.scan(engulfed()).permitted);
  other.prepareShift = 0;
  EXPECT_TRUE(other.scan(engulfed()).permitted);
}

TEST(ConstrainedCarry, ImuCoverageHoldsEachRealSampleAtMostFiftyMilliseconds)
{
  const std::int64_t start = 1000 * kMs;
  const std::int64_t end = start + 100 * kMs;
  std::int64_t left = -1;
  std::deque<fast_lio::CarryImu> imu;
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kInsufficientHistory);
  imu = {{start + 1}, {start + 50 * kMs}};  // no sample at or before the start: start-up, not a gap
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kInsufficientHistory);
  imu = {{start}, {start + 50 * kMs}};
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kCovered);
  imu.back().ns += 1;
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kStructuralGap);
  EXPECT_EQ(left, start);
  imu = {{start}, {start + 50 * kMs - 1}};  // the end is 50 ms + 1 ns past the last sample
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kStructuralGap);
  EXPECT_EQ(left, start + 50 * kMs - 1);

  // A fresh end does not hide a 60 ms hole inside the window.
  imu.clear();
  for (std::int64_t t = start; t < end; t += 5 * kMs)
    if (t <= start + 20 * kMs || t >= start + 80 * kMs)
      imu.push_back({t});
  EXPECT_EQ(fast_lio::carryImuCoverage(imu, start, end, left), fast_lio::CarryImuCoverage::kStructuralGap);
  EXPECT_EQ(left, start + 20 * kMs);
}

TEST(ConstrainedCarry, TaintInsideAPendingEpisodeIsNeverWashedOut)
{
  Drive drive;
  drive.origin();
  EXPECT_TRUE(drive.scan(engulfed()).permitted);
  drive.carry.qualified();  // a receipt before the gap does not count
  drive.carry.negative(fast_lio::kReferenceImuCoverage, 0);
  for (int i = 0; i < 10; ++i)
    EXPECT_FALSE(drive.scan(engulfed()).permitted);
  EXPECT_EQ(drive.carry.taint(), fast_lio::kReferenceImuCoverage);
  EXPECT_FALSE(drive.scan({false, false, false, fast_lio::RecoveryTrust::kPending}).permitted);
  EXPECT_FALSE(drive.scan().rearmed);
  drive.carry.qualified();
  EXPECT_TRUE(drive.scan().rearmed);
  EXPECT_EQ(drive.carry.cause(), 0u);
  EXPECT_TRUE(drive.scan(engulfed()).permitted);  // a new episode on the same, still qualified origin
  EXPECT_EQ(drive.carry.rawEpisode(), 2u);
}

TEST(ConstrainedCarry, LocalQualifiedTaintRearmsOnlyAfterTwentyContiguousStableScans)
{
  Drive drive;
  drive.origin();
  drive.stableNs = 2500 * kMs;
  drive.carry.negative(fast_lio::kReferenceScanHole, 0);
  for (int i = 0; i < 10; ++i)
    EXPECT_FALSE(drive.scan().rearmed);
  EXPECT_FALSE(drive.scan({}, 1).rearmed);  // a scan dropped in between restarts the run
  for (int i = 0; i < 23; ++i)
    EXPECT_FALSE(drive.scan().rearmed);  // 20 scans are not enough until they also span the 2.5 s
  EXPECT_TRUE(drive.scan().rearmed);
  EXPECT_FALSE(drive.carry.latched());
}

TEST(ConstrainedCarry, PermanentHardFailuresNeverRearmInProcess)
{
  for (const std::uint32_t hard : {4u, 64u}) {
    Drive drive;
    drive.origin();
    drive.carry.negative(0, hard);
    drive.carry.qualified();
    for (int i = 0; i < 40; ++i)
      EXPECT_FALSE(drive.scan().rearmed);
    EXPECT_TRUE(drive.carry.latched());
    EXPECT_FALSE(drive.carry.originQualified());
  }
  Drive unanchored;
  unanchored.origin();
  unanchored.scan({false, false, false, fast_lio::RecoveryTrust::kUnanchored});
  EXPECT_EQ(unanchored.carry.hard(), 2u);
  for (int i = 0; i < 40; ++i)
    EXPECT_FALSE(unanchored.scan().rearmed);
}

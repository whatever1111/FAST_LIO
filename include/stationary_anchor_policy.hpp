// stationary_anchor_policy.hpp — which anchor, if any, a parked platform may be
// held to, and when the front end may call its reference trusted again.
//
// The static hold pins a blind scan to an anchor pose and the parked prior pulls
// a healthy scan toward it. Both are right only while the platform has not moved
// since the anchor was taken, so an anchor is the reference of ONE continuous
// stationary episode, not "the last pose a healthy scan saw":
//   - A scan whose static evidence is not "stationary" (measured motion, or no
//     proof either way), a stretch the evidence did not observe (scan hole, IMU
//     coverage gap, IMU gap bridged by the motion model), a map reset, a
//     replaced reference, an input clock fault and a lost front end all end the
//     episode and drop the anchor. A dropped anchor ID is never used again; a
//     later anchor at the same pose is a new anchor with a new ID.
//   - A new anchor needs `confirm_scans` consecutive healthy, stationary scans
//     on a trusted reference with nothing invalidated in between. Its pose is
//     the first of them: the stretch is stationary, so a later scan adds no
//     information and may already carry a slow registration bias. Consecutive
//     is meant literally: a scan that was taken and ended without its static
//     evidence being evaluated ends the candidate too.
//   - Within a scan, invalidation comes first: a scan that dropped the anchor
//     neither uses one nor starts a new one.
//   - Without a usable anchor a stationary platform gets its velocity zeroed and
//     nothing else; no pose is made up for it.
//
// Recovery trust, the platform policy (on when total_ns > 0). A degraded scan
// without a usable hold, a scan hole, and a stretch the IMU did not observe (a
// coverage gap, a lost batch, a gap the motion model bridged, however short)
// start an unsupported-recovery episode, Pending. Its clock starts at the last
// supported evidence, not at the return of the data, so a long hole is timed
// from its start. Pending ends when scans on the unchanged reference keep
// passing the re-anchor match test for `stable_ns` (the scan that carried the
// unobserved IMU stretch is not one of them), or at start + total_ns in
// Unanchored. A
// map reset leads to Unanchored as well: the rebuilt map is seeded from the very
// pose in question. Unanchored is latched for the life of the process, because a
// rebuilt or re-matched local map shows that the scans agree with each other,
// not where the platform is. While Pending the caller keeps the local map
// frozen, so the scans cannot vouch for themselves with points they added.
// Otherwise, Unanchored included, freezing and growing the local map stays with
// the re-anchor gate: a latched Unanchored must not freeze a rebuilt map for
// good, and this decides only what the outputs may claim.
//
// Pure C++. No ROS, no estimator types.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "static_evidence.hpp"

namespace fast_lio
{

// ── Static evidence, three-valued ────────────────────────────────────────────

enum class StaticEvidence : std::uint8_t
{
  kStationary,
  kMoving,   ///< the evidence measured motion
  kUnknown,  ///< the evidence cannot tell; never a zero-velocity measurement
};

enum class StaticEvidenceReason : std::uint8_t
{
  kImuAndWheel,   ///< stationary: IMU at rest and a fresh wheel sample at rest
  kImuOnly,       ///< stationary: IMU at rest, no usable wheel sample, and none is required
  kGyro,          ///< moving: angular rate above the threshold
  kAccSpan,       ///< moving: spread of |a| above the threshold
  kWheel,         ///< moving: a fresh wheel sample above the threshold
  kNoImu,         ///< unknown: no IMU sample in the scan's window
  kImuInvalid,    ///< unknown: non-finite IMU statistics, or no positive |a| to scale the spread by
  kWheelMissing,  ///< unknown: a wheel sample is required and none is usable
};

/// The newest wheel sample as the static test sees it. `valid` says a sample exists; without one, `speed` is not
/// a measured zero whatever it holds.
struct WheelEvidence
{
  bool valid = false;
  double age = -1.0;  ///< s between the sample and the scan, in the scan's clock
  double speed = std::numeric_limits<double>::quiet_NaN();
};

struct StaticEvidenceVerdict
{
  StaticEvidence evidence = StaticEvidence::kUnknown;
  StaticEvidenceReason reason = StaticEvidenceReason::kNoImu;
  bool imu_only = false;  ///< stationary on the IMU alone: no wheel sample testified
  bool stationary() const { return evidence == StaticEvidence::kStationary; }
};

/// A wheel sample testifies only when it exists, is fresh by a finite, non-negative age and carries a finite speed.
inline bool wheelEvidenceUsable(const WheelEvidence & wheel, const StaticEvidenceParams & params)
{
  return wheel.valid && wheel.age >= 0.0 && std::isfinite(wheel.age) && wheel.age <= params.wheel_max_age &&
         std::isfinite(wheel.speed);
}

/// bodyIsStatic() with its "no" split into measured motion and missing proof. stationary() equals bodyIsStatic()
/// for the same inputs (a wheel age of -1 standing for "no sample"), so the hold, the parked prior and the runaway
/// watchdog keep answering "is it parked?" the same way.
inline StaticEvidenceVerdict
classifyStaticEvidence(const ImuWindowStats & imu, const WheelEvidence & wheel, const StaticEvidenceParams & params)
{
  StaticEvidenceVerdict verdict;
  if (imu.samples <= 0) {
    verdict.reason = StaticEvidenceReason::kNoImu;
    return verdict;
  }
  if (!std::isfinite(imu.gyr_max) || !std::isfinite(imu.acc_min) || !std::isfinite(imu.acc_max) ||
      !std::isfinite(imu.acc_mean)) {
    verdict.reason = StaticEvidenceReason::kImuInvalid;
    return verdict;
  }
  if (imu.gyr_max > params.gyro_thresh) {
    verdict.evidence = StaticEvidence::kMoving;
    verdict.reason = StaticEvidenceReason::kGyro;
    return verdict;
  }
  if (!(imu.acc_mean > 0.0)) {
    verdict.reason = StaticEvidenceReason::kImuInvalid;
    return verdict;
  }
  if ((imu.acc_max - imu.acc_min) > params.acc_span_ratio * imu.acc_mean) {
    verdict.evidence = StaticEvidence::kMoving;
    verdict.reason = StaticEvidenceReason::kAccSpan;
    return verdict;
  }
  if (!wheelEvidenceUsable(wheel, params)) {
    if (params.require_wheel) {
      verdict.reason = StaticEvidenceReason::kWheelMissing;
      return verdict;
    }
    verdict.evidence = StaticEvidence::kStationary;
    verdict.reason = StaticEvidenceReason::kImuOnly;
    verdict.imu_only = true;
    return verdict;
  }
  if (wheel.speed <= params.wheel_speed_thresh) {
    verdict.evidence = StaticEvidence::kStationary;
    verdict.reason = StaticEvidenceReason::kImuAndWheel;
    return verdict;
  }
  verdict.evidence = StaticEvidence::kMoving;
  verdict.reason = StaticEvidenceReason::kWheel;
  return verdict;
}

// ── Parameters ───────────────────────────────────────────────────────────────

constexpr double kAnchorMaxNs = 9.2e18;  ///< times beyond this do not fit the signed 64-bit nanosecond fields

struct StationaryAnchorParams
{
  int confirm_scans = 3;                     ///< zupt_anchor_confirm_scans
  std::int64_t max_scan_gap_ns = 300000000;  ///< zupt_anchor_max_scan_gap_s, resolved
};

/// zupt_anchor_confirm_scans: at least one scan.
inline bool anchorConfirmScansValid(int scans)
{
  return scans >= 1;
}

/// zupt_anchor_max_scan_gap_s to whole nanoseconds. 0 means three scan periods, and then the scan rate must be
/// finite and positive; any other value must itself be finite and positive. False: refuse the configuration.
inline bool resolveAnchorScanGapNs(double gap_s, double scan_rate_hz, std::int64_t * gap_ns)
{
  double seconds = gap_s;
  if (gap_s == 0.0) {
    if (!std::isfinite(scan_rate_hz) || !(scan_rate_hz > 0.0)) {
      return false;
    }
    seconds = 3.0 / scan_rate_hz;
  }
  if (!std::isfinite(seconds) || !(seconds > 0.0)) {
    return false;
  }
  const double ns = std::round(seconds * 1e9);
  if (!(ns >= 1.0) || ns > kAnchorMaxNs) {
    return false;
  }
  *gap_ns = static_cast<std::int64_t>(ns);
  return true;
}

struct RecoveryParams
{
  std::int64_t total_ns = 0;            ///< reanchor_total_timeout_sec; 0 = the platform policy is off
  std::int64_t stable_ns = 1000000000;  ///< reanchor_recovery_stable_sec
  bool enabled() const { return total_ns > 0; }
};

/// reanchor_total_timeout_sec (finite, >= 0; 0 = off) and reanchor_recovery_stable_sec (finite, > 0, and shorter
/// than the total when the total is on) to whole nanoseconds. False: refuse the configuration.
inline bool resolveRecoveryParams(double total_s, double stable_s, RecoveryParams * params)
{
  if (!std::isfinite(total_s) || total_s < 0.0 || !std::isfinite(stable_s) || !(stable_s > 0.0)) {
    return false;
  }
  const double total_ns = std::round(total_s * 1e9);
  const double stable_ns = std::round(stable_s * 1e9);
  if (total_ns > kAnchorMaxNs || stable_ns > kAnchorMaxNs || !(stable_ns >= 1.0)) {
    return false;
  }
  if (total_s > 0.0 && (!(total_ns >= 1.0) || !(stable_ns < total_ns))) {
    return false;
  }
  params->total_ns = static_cast<std::int64_t>(total_ns);
  params->stable_ns = static_cast<std::int64_t>(stable_ns);
  return true;
}

// ── Anchor lifecycle ─────────────────────────────────────────────────────────

enum class AnchorState : std::uint8_t
{
  kEmpty,       ///< no anchor and no candidate
  kQualifying,  ///< a candidate collecting its confirming scans; never used
  kValid,       ///< the anchor of the current stationary episode
};

/// Why a candidate or an anchor ended.
enum class AnchorDrop : std::uint8_t
{
  kNone,
  kMotion,            ///< the static evidence measured motion
  kEvidenceUnknown,   ///< the static evidence could not prove "stationary"
  kScanHole,          ///< the time since the last evaluated evidence exceeded the gap bound
  kImuCoverageGap,    ///< an IMU coverage gap: the stretch was not observed
  kImuBridge,         ///< an IMU gap the motion model bridged: not observed either
  kMapReset,          ///< the local map was rebuilt
  kReferenceChanged,  ///< the reference map was replaced
  kTimeEpoch,         ///< the input clock jumped
  kLost,              ///< the front end declared itself lost
  kNotQualified,      ///< a candidate's scan was degraded, or the reference was not trusted
  kScanNotEvaluated,  ///< a candidate: a scan ended without its static evidence being evaluated
  kIdExhausted,       ///< no anchor ID left; nothing is created rather than an ID reused
};

struct StationaryAnchor
{
  AnchorState state = AnchorState::kEmpty;
  std::uint64_t id = 0;       ///< the current or the last dropped candidate/anchor; 0 = none yet
  std::uint64_t last_id = 0;  ///< the highest ID handed out; IDs are never reused
  std::uint64_t episode = 0;  ///< stationary episodes started, one per candidate
  std::uint64_t reference_generation = 0;
  int confirmed_scans = 0;                 ///< healthy stationary scans of the current candidate
  std::int64_t created_ns = -1;            ///< end of the scan whose pose the candidate/anchor carries
  std::int64_t committed_ns = -1;          ///< end of the scan that made it the anchor
  std::int64_t last_evidence_end_ns = -1;  ///< end of the last scan whose static evidence was evaluated
  bool invalidated_this_scan = false;      ///< an event of the current scan broke the evidence
  bool ids_exhausted = false;
  AnchorDrop last_drop = AnchorDrop::kNone;
};

/// The ID a record should carry: the current candidate's or anchor's, 0 when there is none.
inline std::uint64_t anchorCurrentId(const StationaryAnchor & anchor)
{
  return anchor.state == AnchorState::kEmpty ? 0 : anchor.id;
}

/// A candidate or an anchor ends. Marks the scan as broken even when there is nothing to drop: the same scan may
/// not start a candidate or use an anchor afterwards. True when something was dropped; `id` keeps its ID.
inline bool dropAnchor(StationaryAnchor * anchor, AnchorDrop reason)
{
  anchor->invalidated_this_scan = true;
  if (anchor->state == AnchorState::kEmpty) {
    return false;
  }
  anchor->state = AnchorState::kEmpty;
  anchor->confirmed_scans = 0;
  anchor->last_drop = reason;
  return true;
}

/// The reference the anchors live in was replaced (map reset, new prior map). Everything built so far ends, and
/// what is built next belongs to the new generation. A generation counter at its top fails closed: no further
/// anchors, rather than two references sharing a generation.
inline bool changeAnchorReference(StationaryAnchor * anchor, AnchorDrop reason)
{
  if (anchor->reference_generation == std::numeric_limits<std::uint64_t>::max()) {
    anchor->ids_exhausted = true;
  } else {
    ++anchor->reference_generation;
  }
  return dropAnchor(anchor, reason);
}

struct AnchorScanStart
{
  std::int64_t gap_ns = -1;  ///< begin minus the last evidence end; -1 when there is no earlier evidence
  bool hole = false;
  bool dropped = false;
};

/// The start of a scan, before any of its processing. A gap since the last evaluated evidence that exceeds the
/// bound (strictly: a gap equal to the bound is continuous) ends the episode whether or not the platform moved in
/// it — nothing observed the stretch. A scan without a usable begin time, or one that starts further before the
/// evidence than the bound allows (a clock that stepped back), cannot be placed after the evidence either.
inline AnchorScanStart
beginAnchorScan(StationaryAnchor * anchor, std::int64_t begin_ns, const StationaryAnchorParams & params)
{
  anchor->invalidated_this_scan = false;
  AnchorScanStart start;
  if (anchor->last_evidence_end_ns < 0) {
    return start;
  }
  if (begin_ns < 0) {
    start.hole = true;
  } else {
    start.gap_ns = begin_ns - anchor->last_evidence_end_ns;
    start.hole = start.gap_ns > params.max_scan_gap_ns || start.gap_ns < -params.max_scan_gap_ns;
  }
  if (start.hole) {
    start.dropped = dropAnchor(anchor, AnchorDrop::kScanHole);
  }
  return start;
}

/// A scan was taken and ended without reaching the anchor step (no usable timing or cloud, its IMU batch not
/// processed, map initialisation, too few points). A candidate's confirming scans must be consecutive, so the
/// candidate ends. An anchor does not: how long it may go without evaluated evidence is the scan-gap bound's
/// decision. A scan that does not end after the last evaluated evidence is a repeated or old sample, not a new
/// scan, and changes nothing; a scan without a usable end time (`end_ns` < 0) cannot be shown to be one, and counts
/// as new. True when the candidate was dropped.
inline bool dropUnevaluatedCandidate(StationaryAnchor * anchor, std::int64_t end_ns)
{
  if (anchor->state != AnchorState::kQualifying) {
    return false;
  }
  if (end_ns >= 0 && anchor->last_evidence_end_ns >= 0 && end_ns <= anchor->last_evidence_end_ns) {
    return false;
  }
  return dropAnchor(anchor, AnchorDrop::kScanNotEvaluated);
}

struct AnchorStepInput
{
  std::int64_t end_ns = -1;
  StaticEvidence evidence = StaticEvidence::kUnknown;
  bool degraded = false;          ///< the re-anchor gate's verdict for this scan
  bool reference_trusted = true;  ///< recovery trust is LocalQualified
  bool lost = false;              ///< the re-anchor gate is Lost
};

struct AnchorStep
{
  bool usable = false;     ///< a hold or prior may take its pose from the anchor on this scan
  bool started = false;    ///< a candidate started: the caller keeps the current pose as its pose
  bool committed = false;  ///< the candidate became the anchor; nothing uses it before the next scan
  bool dropped = false;    ///< this step ended a candidate or an anchor
  AnchorDrop drop = AnchorDrop::kNone;
  bool parked_evidence = false;  ///< the scan counts toward the parked hold's streak
};

/// The anchor step of one scan, after every earlier event of that scan has been applied. The parked streak counts
/// the candidate's scans as well, so the prior can engage once the anchor is committed instead of waiting for
/// confirm_scans and then engage_scans more. A scan that does not end after the last evaluated evidence (repeated
/// or out of order) adds nothing: it neither confirms a candidate nor uses the anchor, and moves no watermark.
inline AnchorStep
stepAnchor(StationaryAnchor * anchor, const AnchorStepInput & in, const StationaryAnchorParams & params)
{
  AnchorStep step;
  if (in.end_ns < 0) {
    step.drop = AnchorDrop::kEvidenceUnknown;
    step.dropped = dropAnchor(anchor, step.drop);
    return step;
  }
  if (anchor->last_evidence_end_ns >= 0 && in.end_ns <= anchor->last_evidence_end_ns) {
    return step;
  }
  anchor->last_evidence_end_ns = in.end_ns;
  if (in.evidence != StaticEvidence::kStationary) {
    step.drop = in.evidence == StaticEvidence::kMoving ? AnchorDrop::kMotion : AnchorDrop::kEvidenceUnknown;
    step.dropped = dropAnchor(anchor, step.drop);
    return step;
  }
  const bool trusted = in.reference_trusted && !in.lost;
  if (anchor->state == AnchorState::kValid && !anchor->invalidated_this_scan) {
    if (trusted) {
      step.usable = true;
      step.parked_evidence = true;
      return step;
    }
    step.drop = AnchorDrop::kNotQualified;
    step.dropped = dropAnchor(anchor, step.drop);
    return step;
  }
  if (in.degraded || !trusted || anchor->invalidated_this_scan) {
    if (anchor->state == AnchorState::kQualifying) {
      step.drop = AnchorDrop::kNotQualified;
      step.dropped = dropAnchor(anchor, step.drop);
    }
    return step;
  }
  if (anchor->state == AnchorState::kEmpty) {
    if (anchor->ids_exhausted || anchor->last_id == std::numeric_limits<std::uint64_t>::max()) {
      anchor->ids_exhausted = true;
      step.drop = AnchorDrop::kIdExhausted;
      return step;
    }
    anchor->id = ++anchor->last_id;
    if (anchor->episode != std::numeric_limits<std::uint64_t>::max()) {
      ++anchor->episode;
    }
    anchor->state = AnchorState::kQualifying;
    anchor->confirmed_scans = 0;
    anchor->created_ns = in.end_ns;
    anchor->committed_ns = -1;
    step.started = true;
  }
  ++anchor->confirmed_scans;
  step.parked_evidence = true;
  if (anchor->confirmed_scans >= params.confirm_scans) {
    anchor->state = AnchorState::kValid;
    anchor->committed_ns = in.end_ns;
    step.committed = true;
  }
  return step;
}

// ── Recovery trust: the platform policy ──────────────────────────────────────

enum class RecoveryTrust : std::uint8_t
{
  kLocalQualified,  ///< the reference is trusted as far as the local checks go
  kPending,         ///< an unsupported-recovery episode is running against its deadline
  kUnanchored,      ///< latched: the position is not trusted again in this process
};

enum class RecoveryCause : std::uint8_t
{
  kNone,
  kDegraded,        ///< a degraded scan without a usable hold
  kScanHole,        ///< a scan hole
  kImuCoverageGap,  ///< an IMU coverage gap, or an IMU batch that was lost
  kImuBridge,       ///< an IMU gap the motion model bridged
  kMapReset,        ///< the local map was rebuilt
  kDeadline,        ///< the episode outlasted the total budget
  kStable,          ///< the scans re-registered for the stable time
};

struct RecoveryState
{
  RecoveryTrust trust = RecoveryTrust::kLocalQualified;
  RecoveryCause cause = RecoveryCause::kNone;  ///< what started the current Pending, or caused Unanchored
  std::uint64_t episode = 0;                   ///< unsupported-recovery episodes started
  std::int64_t start_ns = -1;
  std::int64_t deadline_ns = -1;
  std::int64_t last_supported_end_ns = -1;  ///< end of the last scan the reference was trusted on
  std::int64_t stable_begin_ns = -1;        ///< begin of the current run of matching scans; -1 = none
  std::uint64_t stable_last_scan = 0;       ///< scan sequence number of the run's last scan
  std::uint64_t stable_generation = 0;      ///< reference generation the run is on
  std::uint64_t excluded_scan = 0;          ///< scan sequence number that may not count toward a run; 0 = none
  std::uint64_t map_resets = 0;
};

struct RecoveryEvent
{
  bool pending = false;     ///< Pending started
  bool qualified = false;   ///< Pending ended with the reference trusted again
  bool unanchored = false;  ///< Unanchored latched
  RecoveryCause cause = RecoveryCause::kNone;
};

inline bool recoveryTrusted(const RecoveryState & recovery)
{
  return recovery.trust == RecoveryTrust::kLocalQualified;
}

/// A scan the reference was trusted on: a healthy scan, or a degraded one held by a usable anchor. Pending clocks
/// start from the last of these.
inline void noteSupportedScan(RecoveryState * recovery, std::int64_t end_ns)
{
  if (recovery->trust == RecoveryTrust::kLocalQualified && end_ns >= 0) {
    recovery->last_supported_end_ns = end_ns;
  }
}

/// A gap, a loss or a reset breaks a run of matching scans: the stable time has to be shown again in one piece.
inline void breakRecoveryRun(RecoveryState * recovery)
{
  recovery->stable_begin_ns = -1;
}

/// The scan `scan_seq` carried a stretch the IMU did not observe: a bridged gap, a coverage gap, a lost batch. Its
/// pose crossed that stretch on the motion model, so whatever its registration says it confirms nothing: the run so
/// far is broken, and the scan itself does not start or continue one. The next scan can.
inline void excludeRecoveryScan(RecoveryState * recovery, std::uint64_t scan_seq)
{
  recovery->excluded_scan = scan_seq;
  recovery->stable_begin_ns = -1;
}

inline bool recoveryScanExcluded(const RecoveryState & recovery, std::uint64_t scan_seq)
{
  return scan_seq != 0 && recovery.excluded_scan == scan_seq;
}

/// The evidence for the reference broke. With the policy on and the reference trusted, Pending starts with its
/// clock at the last supported scan (or at `now_ns` when there was none). Without any usable time there is nothing
/// to measure the episode by, and the reference is not trusted again.
inline RecoveryEvent
startRecovery(RecoveryState * recovery, RecoveryCause cause, std::int64_t now_ns, const RecoveryParams & params)
{
  RecoveryEvent event;
  if (!params.enabled() || recovery->trust != RecoveryTrust::kLocalQualified) {
    return event;
  }
  if (recovery->episode != std::numeric_limits<std::uint64_t>::max()) {
    ++recovery->episode;
  }
  recovery->cause = cause;
  recovery->stable_begin_ns = -1;
  const std::int64_t start = recovery->last_supported_end_ns >= 0 ? recovery->last_supported_end_ns : now_ns;
  if (start < 0) {
    recovery->trust = RecoveryTrust::kUnanchored;
    recovery->start_ns = -1;
    recovery->deadline_ns = -1;
    event.unanchored = true;
    event.cause = cause;
    return event;
  }
  recovery->trust = RecoveryTrust::kPending;
  recovery->start_ns = start;
  recovery->deadline_ns = start > std::numeric_limits<std::int64_t>::max() - params.total_ns
                            ? std::numeric_limits<std::int64_t>::max()
                            : start + params.total_ns;
  event.pending = true;
  event.cause = cause;
  return event;
}

/// Checked first at every processing event: a recovery confirmed at the same instant as the deadline is too late.
inline RecoveryEvent expireRecovery(RecoveryState * recovery, std::int64_t now_ns)
{
  RecoveryEvent event;
  if (recovery->trust != RecoveryTrust::kPending || now_ns < 0 || now_ns < recovery->deadline_ns) {
    return event;
  }
  recovery->trust = RecoveryTrust::kUnanchored;
  recovery->cause = RecoveryCause::kDeadline;
  recovery->stable_begin_ns = -1;
  event.unanchored = true;
  event.cause = RecoveryCause::kDeadline;
  return event;
}

/// One scan's re-registration while Pending: `match_ok` is the re-anchor gate's own match test on a scan the
/// divergence guard did not call degraded. The run must be unbroken (consecutive scan sequence numbers on one
/// reference generation, none of them excluded) and span `stable_ns` from the first scan's begin to this scan's
/// end. The re-anchor gate's own state is not consulted: blind/verifying flips and a 3-scan re-anchor neither end
/// Pending nor move its clock.
inline RecoveryEvent observeRecoveryScan(RecoveryState * recovery,
                                         std::uint64_t scan_seq,
                                         std::int64_t begin_ns,
                                         std::int64_t end_ns,
                                         bool match_ok,
                                         std::uint64_t generation,
                                         const RecoveryParams & params)
{
  RecoveryEvent event;
  if (recovery->trust != RecoveryTrust::kPending) {
    return event;
  }
  if (!match_ok || scan_seq == 0 || begin_ns < 0 || end_ns < 0 || recoveryScanExcluded(*recovery, scan_seq)) {
    recovery->stable_begin_ns = -1;
    return event;
  }
  const bool continues = recovery->stable_begin_ns >= 0 && recovery->stable_last_scan + 1 == scan_seq &&
                         recovery->stable_generation == generation;
  if (!continues) {
    recovery->stable_begin_ns = begin_ns;
    recovery->stable_generation = generation;
  }
  recovery->stable_last_scan = scan_seq;
  if (end_ns - recovery->stable_begin_ns >= params.stable_ns) {
    recovery->trust = RecoveryTrust::kLocalQualified;
    recovery->cause = RecoveryCause::kStable;
    recovery->stable_begin_ns = -1;
    recovery->last_supported_end_ns = end_ns;
    event.qualified = true;
    event.cause = RecoveryCause::kStable;
  }
  return event;
}

/// A map reset. With the policy on, the reference is not trusted again in this process.
inline RecoveryEvent
resetRecoveryReference(RecoveryState * recovery, std::int64_t now_ns, const RecoveryParams & params)
{
  RecoveryEvent event;
  if (recovery->map_resets != std::numeric_limits<std::uint64_t>::max()) {
    ++recovery->map_resets;
  }
  if (!params.enabled() || recovery->trust == RecoveryTrust::kUnanchored) {
    return event;
  }
  if (recovery->trust == RecoveryTrust::kLocalQualified) {
    if (recovery->episode != std::numeric_limits<std::uint64_t>::max()) {
      ++recovery->episode;
    }
    recovery->start_ns = recovery->last_supported_end_ns >= 0 ? recovery->last_supported_end_ns : now_ns;
    recovery->deadline_ns = -1;
  }
  recovery->trust = RecoveryTrust::kUnanchored;
  recovery->cause = RecoveryCause::kMapReset;
  recovery->stable_begin_ns = -1;
  event.unanchored = true;
  event.cause = RecoveryCause::kMapReset;
  return event;
}

// ── What the outputs may claim ───────────────────────────────────────────────

struct AnchorHealthInputs
{
  bool degraded = false;       ///< the front end's own degraded verdict (guard and re-anchor gate)
  bool hold_active = false;    ///< the static hold or the parked hold is engaged
  bool anchor_usable = false;  ///< the hold's anchor was usable on this scan and is still the anchor
  /// The re-anchor gate is Lost, or it declared this scan lost and a map reset of the same scan re-armed it.
  bool lost = false;
  RecoveryTrust trust = RecoveryTrust::kLocalQualified;
};

struct AnchorHealth
{
  bool effective_degraded = false;    ///< what the Degraded flag and the typed state report
  bool reported_static_hold = false;  ///< what the StaticHold flag reports
  bool lost = false;                  ///< what the Lost flag and the typed state report
};

/// One verdict for every encoding of the scan. Pending and Unanchored are degraded whatever the local registration
/// says; static hold is claimed only for a hold on a usable anchor, never for a zeroed velocity or a static verdict
/// alone, and never while lost or unanchored. A loss is reported by the scan that declared it even when that scan
/// went on to rebuild the map: the reset re-arms the gate for the scans after it, it does not undo the loss.
inline AnchorHealth anchorHealth(const AnchorHealthInputs & in)
{
  AnchorHealth health;
  const bool untrusted = in.trust != RecoveryTrust::kLocalQualified;
  health.effective_degraded = in.degraded || untrusted;
  health.reported_static_hold = in.hold_active && in.anchor_usable && !in.lost && !untrusted;
  health.lost = in.lost;
  return health;
}

}  // namespace fast_lio

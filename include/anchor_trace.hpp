// anchor_trace.hpp — a machine-readable trace of the ZUPT stationary anchor.
//
// The degraded-scan static hold and the healthy-scan parked prior both pull the
// estimate onto an anchor pose. Whether the anchor a scan used still belonged to
// the stationary stretch it was taken in cannot be read off the published pose:
// a pin to a stale anchor and a pin to a fresh one look the same downstream.
// This trace names the anchor every scan used, every event that breaks the
// evidence behind an anchor, the anchor's lifecycle and the front end's recovery
// trust, so an offline checker can decide it scan by scan.
//
// Observation only: nothing in this header, and nothing the node does to fill
// it, feeds back into the estimate.
//
// Pure C++. No ROS, no Eigen.
//
// ===========================================================================
// Trace format, version 3. This comment is the specification; a checker must
// depend on nothing else.
// ===========================================================================
//
// Every record is one log line on the logger `laser_mapping.zupt_anchor`: a
// fixed tag, then space-separated `key=value` tokens in the order listed below.
// Keys are lowercase, values never contain spaces. A parser must reject a
// record whose `v` it does not know, and a record that lacks a listed key.
//
// Units and encodings
//   *_ns        signed 64-bit integer nanoseconds of the sensor clock (the
//               lidar/IMU header time base, rounded from the double seconds the
//               front end carries, so the resolution is sub-microsecond, not one
//               nanosecond); -1 = unknown or never.
//   *_p, *_v    metres, metres per second: x,y,z in the front end's world frame
//               (the frame /Odometry is published in).
//   *_q         unit quaternion w,x,y,z, body to world.
//   upd_dp, upd_dq  one real each: metres, radians.
//   reals       printf %.17g, so every double round-trips bit-exactly; non-finite
//               values print as nan, inf or -inf.
//   booleans    0 or 1.
//   na          the field was not evaluated for this scan (see `outcome`), or
//               does not apply to it; every key that can be na says when.
//   p_pose_*    FNV-1a 64-bit hash, 16 lowercase hex digits, of the raw bytes of
//               the 6x6 position/attitude block of the state covariance (rows and
//               columns 0..5, row-major doubles). Equal hashes mean the block is
//               bit-identical.
//
// Since version 2: the reason scan_not_evaluated (section 1), the recovery
// reason imu_bridge and the lost IMU batch under imu_coverage_gap (section 2),
// the anchor writes a late event of the same scan takes back (section 3), and
// the keys appended to each record: `streak` (section 1), `path` (section 2),
// act_p ... upd_dq (section 3). The version-2 keys keep their order and meaning.
//
// Sequence rules
//   event_seq   1, 2, 3, ... over the process lifetime, one per transition record
//               of either tag (the counter summary included). Never repeats,
//               never decreases.
//   scan_seq    1, 2, 3, ... one per scan the front end takes off its input
//               queue, whatever happens to it next, in processing order. Scan
//               records carry it only with the per-scan trace on; the counter
//               summary's `scans` is its final value either way.
//   anchor_id   1, 2, 3, ... one per candidate: handed out when a candidate
//               starts, kept when it becomes the anchor. 0 = none yet. An ID is
//               never reused, so a pin or prior that names an ID names the
//               stationary episode it was taken in. With anchor_state=empty it
//               names the last candidate or anchor, which has ended.
//   stationary_episode    stationary episodes started so far: one per
//               candidate.
//   reference_generation  +1 on every local map reset and every replacement
//               of the prior map. Anchors live in one generation.
//   A sequence that reaches 2^64-1 is exhausted: from then on it yields 0 (IDs:
//   no further candidates), which no record carries legitimately, and the
//   counter summary says so. It never wraps back onto old values.
//
// 1. Anchor transition records: always on, never throttled, one line per event.
//
//   [ZUPT-ANCHOR] v=3 event=<event> reason=<reason> event_seq=<n> scan_seq=<n>
//     stamp_ns=<ns> anchor_id=<id> anchor_valid=<0|1> created_ns=<ns>
//     last_evidence_ns=<ns> anchor_state=<state> stationary_episode=<n>
//     reference_generation=<n> streak=<n>
//
//   event              candidate_started: a candidate took this scan's pose.
//                      created: the candidate became the anchor, after
//                      zupt_anchor_confirm_scans consecutive healthy stationary
//                      scans on a trusted reference.
//                      candidate_dropped, invalidated: a candidate, an anchor,
//                      ended; `reason` says why.
//                      pin_engaged, pin_released, parked_prior_engaged,
//                      parked_prior_released: the holds.
//                      evidence: an input event that breaks static evidence,
//                      whether or not there was an anchor for it to end.
//   reason             for candidate_dropped and invalidated: motion,
//                      evidence_unknown (the static evidence could not prove
//                      "stationary"), scan_hole, imu_coverage_gap, imu_bridge,
//                      map_reset, reference_changed, time_epoch, lost,
//                      not_qualified (a candidate's scan was degraded, or the
//                      reference was not trusted). For candidate_dropped also
//                      scan_not_evaluated: a scan was taken and ended before the
//                      anchor step evaluated its static evidence (a scan without
//                      usable timing or cloud, an IMU batch not processed, map
//                      initialisation, too few points), so the candidate's scans
//                      are no longer consecutive. An anchor does not end with
//                      such a scan: the scan-gap bound decides how long it may
//                      go without evaluated evidence. For evidence: motion (the
//                      static verdict went from stationary to moving),
//                      evidence_unknown (from stationary to unknown), scan_hole,
//                      imu_coverage_gap, imu_bridge, map_reset,
//                      reference_changed, lost, time_epoch. For pin_released and
//                      parked_prior_released: scan_healthy, motion, or the
//                      reason the candidate or anchor ended. Otherwise `-`.
//                      imu_coverage_gap and imu_bridge evidence is reported on
//                      the first scan of a consecutive run; the counters count
//                      every one.
//   scan_seq           the scan the event belongs to: the last scan taken when
//                      the record was written.
//   stamp_ns           the sensor time of the input being processed when the
//                      record was written. On the scan path that is the scan's
//                      end time. A scan hole found while no scan arrives carries
//                      the time of the IMU sample that found it: that sample's
//                      own time, not reduced by the allowance the check gives a
//                      scan in transit. time_epoch carries the stamp of the
//                      input that faulted. -1 when that input has no usable
//                      time (a scan dropped for its timing).
//   anchor_id ... created_ns   the candidate or anchor after the event (for
//                      candidate_dropped and invalidated: the one that ended),
//                      whether it is the anchor, and the end time of the scan
//                      whose pose it carries.
//   last_evidence_ns   end time of the last scan whose static evidence the anchor
//                      logic evaluated.
//   anchor_state       empty, qualifying or valid, after the event.
//   streak             the parked streak after the event: the consecutive
//                      stationary scans the anchor step has counted on one
//                      candidate or usable anchor (the parked prior engages on
//                      this count). It is 0 on evidence, candidate_dropped and
//                      invalidated and on the releases that follow them, since
//                      each of these ends the streak; a record written by a
//                      scan's anchor step (candidate_started, created, the
//                      holds) counts that scan.
//
// 2. Recovery transition records: always on, never throttled, one line per
//    event; event_seq is drawn from the same sequence as the anchor records.
//    Only with the platform policy on (reanchor_total_timeout_sec > 0).
//
//   [FE-RECOVERY] v=3 event=<event> reason=<reason> event_seq=<n> scan_seq=<n>
//     stamp_ns=<ns> episode=<n> start_ns=<ns> deadline_ns=<ns> elapsed_ns=<ns>
//     reset_count=<n> path=<path>
//
//   event        pending: an unsupported-recovery episode started (reason
//                degraded: a degraded scan without a usable hold; scan_hole;
//                imu_coverage_gap: an IMU coverage gap, or an IMU batch that was
//                lost, dropped as invalid or with the input failing closed;
//                imu_bridge: an IMU gap the motion model bridged, whatever its
//                length). qualified: it ended with the reference trusted again
//                (reason stable: the scans kept passing the re-anchor match test
//                for reanchor_recovery_stable_sec on one reference; the scan
//                that carried an IMU bridge, coverage gap or lost batch is not
//                one of them). unanchored: latched until the process restarts
//                (reason deadline, or map_reset).
//   scan_seq     the last scan taken when the event happened.
//   stamp_ns     the sensor time of the input being processed when the record
//                was written: a scan's end time on the scan path, the IMU
//                sample's own time on the idle path. Never that time reduced by
//                the allowance the idle check gives a scan in transit.
//   episode      unsupported-recovery episodes started so far.
//   start_ns     where the episode's clock starts: the end of the last supported
//                scan before it (when there was none, the first unsupported
//                event). -1 unknown.
//   deadline_ns  start_ns + reanchor_total_timeout_sec; -1 when there is none.
//   elapsed_ns   stamp_ns - start_ns; -1 when either is unknown.
//   reset_count  local map resets so far.
//   path         scan: written while a scan was being processed. idle: written
//                by the check each IMU sample runs while no scan is queued; that
//                check finds a deadline that has passed, and a scan hole once
//                the allowance for a scan in transit has run out. A scan that
//                is queued is judged on the scan path, at its own end time.
//
// 3. Scan records: one per scan, only with the parameter zupt_anchor_trace_en.
//
//   [ZUPT-ANCHOR-SCAN] v=3 scan_seq=<n> outcome=<outcome> beg_ns=<ns>
//     end_ns=<ns> gap_ns=<ns> anchor_id=<id> anchor_valid=<0|1>
//     anchor_created_ns=<ns> body=<body> imu_n=<n> gyr_max=<rad/s>
//     acc_span=<ratio> wheel_age=<s> wheel_speed=<m/s> guard_degraded=<0|1>
//     degraded=<0|1> gate_pre=<gate> gate_post=<gate> action=<action>
//     anchor_used=<id> parked=<0|1> parked_applied=<0|1> zupt_active=<0|1>
//     events=<list> pre_p=<x,y,z> pre_q=<w,x,y,z> pre_v=<x,y,z> post_p=<x,y,z>
//     post_q=<w,x,y,z> post_v=<x,y,z> p_pose_pre=<hash> p_pose_post=<hash>
//     p_finite=<0|1> p_sym=<real> p_min_eig=<real> published=<0|1>
//     flags=<bits> anchor_state=<state> stationary_episode=<n>
//     reference_generation=<n> evidence=<evidence> evidence_reason=<reason> imu_only=<0|1>
//     wheel_valid=<0|1> usable=<0|1> zero_only=<0|1> supported=<0|1>
//     trust_pre=<trust> trust_post=<trust> recovery_episode=<n>
//     recovery_start_ns=<ns> recovery_deadline_ns=<ns> eff_degraded=<0|1>
//     static_hold=<0|1> act_p=<x,y,z> act_q=<w,x,y,z> act_v=<x,y,z>
//     p_pose_act=<hash> target_p=<x,y,z> target_q=<w,x,y,z> anchor_p=<x,y,z>
//     anchor_q=<w,x,y,z> streak=<n> map_updated=<0|1> upd_dp=<m> upd_dq=<rad>
//
//   outcome     published: went through the whole scan path and published
//               /Odometry. Otherwise where it stopped: bootstrap (first scan,
//               IMU bootstrap only), imu_fault, imu_not_processed,
//               imu_coverage_gap, no_cloud, before_anchor_step (map
//               initialisation, too few points), after_anchor_step; or dropped
//               before processing: dropped_invalid_time, dropped_nonadvancing,
//               dropped_empty_imu.
//   gap_ns      beg_ns minus last_evidence_ns as it stood before this scan; -1
//               when there was none.
//   anchor_*    the candidate or anchor after this scan (or as it stands, for
//               scans that stopped earlier), as in the transition records.
//   body        static or moving: whether the static evidence (IMU window plus
//               newest wheel sample) proved "stationary". imu_n, gyr_max,
//               acc_span ((max-min)/mean of |a|), wheel_age (s, -1 = no sample)
//               and wheel_speed are its inputs.
//   guard_degraded  the divergence guard's own verdict; degraded is the
//               re-anchor gate's (what the anchor logic acted on).
//   gate        healthy, blind, verifying, lost.
//   action      what the anchor step wrote: none, pin (position and attitude set
//               to the anchor, velocity zeroed) or zero_velocity.
//   anchor_used the anchor a pin or an applied parked prior took its pose from;
//               0 = none.
//   parked      the parked hold was engaged; parked_applied = its prior was
//               applied this scan.
//               A late event of the same scan that ends the anchor (the runaway
//               watchdog resetting the map or declaring the front end lost)
//               takes the scan's pin or parked prior back, state and
//               covariance: the scan is published as if the anchor had not been
//               usable at the anchor step. action, anchor_used, parked_applied,
//               zero_only, act_* and target_* then describe the published
//               state: a pin taken back reads action=zero_velocity
//               anchor_used=0 zero_only=1, the velocity alone having been
//               zeroed in its place; a parked prior taken back reads
//               parked_applied=0 anchor_used=0 and leaves action as the healthy
//               scan had it. usable still says what the anchor step found.
//   events      comma-separated, `-` when empty: motion, scan_hole,
//               imu_coverage_gap, imu_bridge, map_reset, reference_changed,
//               lost, reanchored, runaway_trip, time_epoch, evidence_unknown.
//               motion marks the scan whose verdict went from stationary to
//               moving, evidence_unknown every scan whose evidence was unknown.
//   pre_*       the state right before the anchor step; post_* the state after
//               the last writer of the scan (what /Odometry carries).
//   p_*         only on scans whose anchor step wrote the state (a write taken
//               back later in the scan included), na otherwise.
//               p_pose_pre is taken before the anchor step, p_pose_post after the
//               scan's last writer. p_finite = every covariance entry finite;
//               p_sym = max|P-P^T| / max(1, rho) with rho the largest |eigenvalue|
//               of (P+P^T)/2; p_min_eig = smallest eigenvalue of (P+P^T)/2. The
//               last three need zupt_anchor_trace_en (nan without it).
//   flags       the legacy health bits published in /Odometry twist.covariance[1]
//               (bit0 degraded, bit1 static hold, bit2 lost, bit3 levelling);
//               -1 = nothing published.
//   anchor_state  empty, qualifying or valid, after this scan.
//   evidence    stationary, moving (the evidence measured motion) or unknown (it
//               could prove neither). evidence_reason: imu_and_wheel, imu_only,
//               gyro, acc_span, wheel, no_imu, imu_invalid, wheel_missing.
//               imu_only = stationary on the IMU alone, no wheel sample
//               testified; wheel_valid = a wheel sample was usable as evidence
//               (present, fresh and finite).
//   usable      the anchor step found a usable anchor of the current episode.
//   zero_only   degraded, stationary, no usable anchor: the velocity alone was
//               zeroed.
//   supported   the scan was healthy, or degraded and held on a usable anchor,
//               and its IMU stretch was observed (no IMU bridge, coverage gap or
//               lost batch on it): it did not start an unsupported-recovery
//               episode. A hold taken back later in the scan supports nothing.
//               evidence ... supported are na when the anchor step did not run.
//   trust_*     local, pending or unanchored, at the start and at the end of the
//               scan.
//   recovery_*  the current or last unsupported-recovery episode: its number,
//               start_ns and deadline_ns as in the recovery records.
//   eff_degraded, static_hold  what every encoding of this scan reports: flags
//               bit 0 and bit 1 on a published scan, and the typed observation's
//               degraded state and static-hold substate.
//   act_p, act_q, act_v  the state right after the anchor step's last writer
//               returned (the pin, the velocity zeroing or the parked prior,
//               whichever ran last and was not taken back) and before any later
//               writer of the scan (the runaway watchdog's velocity reset, the
//               wheel update). na when the anchor step left nothing in the
//               state: action=none and parked_applied=0. A parked prior applied
//               with action=none is a writer, so its scan has them.
//   p_pose_act  the pose covariance block at that same moment; na with act_*.
//   target_p, target_q  the pose the pin set, or the parked prior pulled toward,
//               as that writer read it from anchor_used. na when anchor_used=0.
//               The pin runs on degraded scans and the prior on healthy ones, so
//               a scan has one target at most.
//   anchor_p, anchor_q  the pose the candidate or anchor holds after this scan:
//               the pose of the scan that started the candidate, never changed
//               afterwards. na when anchor_state=empty.
//   streak      the parked streak after this scan (see section 1); as it stands
//               for a scan that stopped before the anchor step.
//   map_updated this scan's points were handed to the local map: by the first
//               build, by the rebuild of a map reset, or by the incremental
//               update (which may insert them a configured number of scans
//               later). 0 when the map was frozen or paused, and when the scan
//               stopped before that step.
//   upd_dp, upd_dq  the distance and the rotation angle between the state before
//               and after this scan's lidar measurement update (the iterated
//               update alone, before the priors and the anchor step). na when
//               the scan stopped before it.
//
// 4. Counter summary: once, when the node shuts down. An anchor transition
//    record with its own keys, all saturating unsigned 64-bit counts:
//
//   [ZUPT-ANCHOR] v=3 event=counters event_seq=<n> scans=<n> events=<n>
//     candidates=<n> created=<n> candidates_dropped=<n> invalidated=<n>
//     invalidated_motion=<n> invalidated_evidence_unknown=<n>
//     invalidated_scan_hole=<n> invalidated_imu_coverage_gap=<n>
//     invalidated_imu_bridge=<n> invalidated_map_reset=<n>
//     invalidated_reference_changed=<n> invalidated_time_epoch=<n>
//     invalidated_lost=<n> invalidated_not_qualified=<n> pin_applied=<n>
//     parked_prior_applied=<n> zero_only_unanchored=<n> blocked_stale_use=<n>
//     evidence_unknown=<n> motion=<n> scan_hole=<n> imu_coverage_gap=<n>
//     imu_bridge=<n> map_reset=<n> reference_changed=<n> lost=<n>
//     time_epoch=<n> recovery_started=<n> recovery_qualified=<n>
//     recovery_expired=<n> map_reset_unanchored=<n> sequence_exhausted=<0|1>
//
//   events counts the transition records of both tags before this one; scans
//   the scans taken; candidates and created the candidates started and the
//   anchors made of them; invalidated the anchors that ended (by reason in the
//   invalidated_* keys), candidates_dropped the candidates that never became
//   one; pin_applied and parked_prior_applied the scans whose published state
//   carries a pin, a parked prior (one taken back later in its scan is not
//   counted); zero_only_unanchored the zero-only scans
//   (degraded, stationary, no usable anchor), blocked_stale_use those of them
//   that came after an anchor had been created in this process (scans on which
//   only an anchor that outlived its episode could have been held to). A scan
//   whose pin was taken back is a zero-only scan and counts in both: this is
//   the one source of blocked_stale_use on which an anchor of the current
//   episode existed at the anchor step. evidence_unknown the scans whose static
//   evidence was unknown; motion the changes from stationary to moving;
//   scan_hole ... time_epoch the input events, every occurrence;
//   recovery_started the unsupported-recovery episodes started by a degraded
//   scan, a scan hole, an IMU coverage gap or lost batch, or an IMU bridge,
//   recovery_qualified those that ended trusted again, recovery_expired those
//   that ran out; map_reset_unanchored the map resets that latched Unanchored.
//   The summary has no `streak` key.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

#include "stationary_anchor_policy.hpp"

namespace fast_lio
{

constexpr int kAnchorTraceVersion = 3;

/// Counts that reach the top stay there: a wrapped counter would read as a
/// small, plausible number.
inline void saturatingIncrement(std::uint64_t * counter)
{
  if (*counter != std::numeric_limits<std::uint64_t>::max()) {
    ++*counter;
  }
}

/// 1, 2, 3, ...; once the top is reached, 0 for ever. 0 is never a valid ID, so
/// an exhausted sequence cannot alias an earlier value.
class MonotonicSequence
{
public:
  std::uint64_t next()
  {
    if (last_ == std::numeric_limits<std::uint64_t>::max()) {
      return 0;
    }
    return ++last_;
  }
  std::uint64_t last() const { return last_; }
  bool exhausted() const { return last_ == std::numeric_limits<std::uint64_t>::max(); }
  /// Tests only: start just below the top to exercise exhaustion.
  void skipTo(std::uint64_t last) { last_ = last; }

private:
  std::uint64_t last_ = 0;
};

/// Seconds as carried by the front end to integer sensor nanoseconds; -1 for a
/// time that cannot be represented.
inline std::int64_t traceNs(double seconds)
{
  constexpr double kLimit = 9.2e18;
  const double ns = seconds * 1e9;
  if (!std::isfinite(ns) || ns > kLimit || ns < -kLimit) {
    return -1;
  }
  return static_cast<std::int64_t>(std::llround(ns));
}

/// FNV-1a over the raw bytes of `count` doubles.
inline std::uint64_t fnv1a64(const double * values, std::size_t count)
{
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t i = 0; i < count; ++i) {
    unsigned char bytes[sizeof(double)];
    std::memcpy(bytes, &values[i], sizeof(double));
    for (unsigned char byte : bytes) {
      hash ^= byte;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

enum class AnchorTraceEvent : std::uint8_t
{
  kPinEngaged,
  kPinReleased,
  kParkedPriorEngaged,
  kParkedPriorReleased,
  kEvidence,
  kCandidateStarted,
  kCreated,
  kCandidateDropped,
  kInvalidated,
};

enum class AnchorTraceReason : std::uint8_t
{
  kNone,
  kMotion,
  kScanHole,
  kImuCoverageGap,
  kImuBridge,
  kMapReset,
  kReferenceChanged,
  kLost,
  kTimeEpoch,
  kScanHealthy,
  kEvidenceUnknown,
  kNotQualified,
  kScanNotEvaluated,
};

enum class AnchorScanOutcome : std::uint8_t
{
  kPublished,
  kBootstrap,
  kImuFault,
  kImuNotProcessed,
  kImuCoverageGap,
  kNoCloud,
  kBeforeAnchorStep,
  kAfterAnchorStep,
  kDroppedInvalidTime,
  kDroppedNonadvancing,
  kDroppedEmptyImu,
};

enum class AnchorTraceGate : std::uint8_t
{
  kNa,
  kHealthy,
  kBlind,
  kVerifying,
  kLost,
};

enum class AnchorTraceBody : std::uint8_t
{
  kNa,
  kStatic,
  kMoving,
};

enum class AnchorTraceAction : std::uint8_t
{
  kNa,
  kNone,
  kPin,
  kZeroVelocity,
};

enum class RecoveryTraceEvent : std::uint8_t
{
  kPending,
  kQualified,
  kUnanchored,
};

/// Where a recovery record was written: on the scan path, or by the check the IMU samples run while no scan is queued.
enum class RecoveryTracePath : std::uint8_t
{
  kScan,
  kIdle,
};

/// Bits of AnchorScanRecord::events, in the order they are printed.
constexpr std::uint32_t kAnchorEventMotion = 1u << 0;
constexpr std::uint32_t kAnchorEventScanHole = 1u << 1;
constexpr std::uint32_t kAnchorEventImuCoverageGap = 1u << 2;
constexpr std::uint32_t kAnchorEventImuBridge = 1u << 3;
constexpr std::uint32_t kAnchorEventMapReset = 1u << 4;
constexpr std::uint32_t kAnchorEventReferenceChanged = 1u << 5;
constexpr std::uint32_t kAnchorEventLost = 1u << 6;
constexpr std::uint32_t kAnchorEventReanchored = 1u << 7;
constexpr std::uint32_t kAnchorEventRunawayTrip = 1u << 8;
constexpr std::uint32_t kAnchorEventTimeEpoch = 1u << 9;
constexpr std::uint32_t kAnchorEventEvidenceUnknown = 1u << 10;
constexpr int kAnchorEventBitCount = 11;

inline const char * anchorTraceEventName(AnchorTraceEvent event)
{
  switch (event) {
    case AnchorTraceEvent::kPinEngaged:
      return "pin_engaged";
    case AnchorTraceEvent::kPinReleased:
      return "pin_released";
    case AnchorTraceEvent::kParkedPriorEngaged:
      return "parked_prior_engaged";
    case AnchorTraceEvent::kParkedPriorReleased:
      return "parked_prior_released";
    case AnchorTraceEvent::kEvidence:
      return "evidence";
    case AnchorTraceEvent::kCandidateStarted:
      return "candidate_started";
    case AnchorTraceEvent::kCreated:
      return "created";
    case AnchorTraceEvent::kCandidateDropped:
      return "candidate_dropped";
    case AnchorTraceEvent::kInvalidated:
      return "invalidated";
  }
  return "unknown";
}

inline const char * anchorTraceReasonName(AnchorTraceReason reason)
{
  switch (reason) {
    case AnchorTraceReason::kNone:
      return "-";
    case AnchorTraceReason::kMotion:
      return "motion";
    case AnchorTraceReason::kScanHole:
      return "scan_hole";
    case AnchorTraceReason::kImuCoverageGap:
      return "imu_coverage_gap";
    case AnchorTraceReason::kImuBridge:
      return "imu_bridge";
    case AnchorTraceReason::kMapReset:
      return "map_reset";
    case AnchorTraceReason::kReferenceChanged:
      return "reference_changed";
    case AnchorTraceReason::kLost:
      return "lost";
    case AnchorTraceReason::kTimeEpoch:
      return "time_epoch";
    case AnchorTraceReason::kScanHealthy:
      return "scan_healthy";
    case AnchorTraceReason::kEvidenceUnknown:
      return "evidence_unknown";
    case AnchorTraceReason::kNotQualified:
      return "not_qualified";
    case AnchorTraceReason::kScanNotEvaluated:
      return "scan_not_evaluated";
  }
  return "unknown";
}

/// The reason a lifecycle drop is printed with. kNone and kIdExhausted drop nothing.
inline AnchorTraceReason anchorTraceReasonOf(AnchorDrop drop)
{
  switch (drop) {
    case AnchorDrop::kMotion:
      return AnchorTraceReason::kMotion;
    case AnchorDrop::kEvidenceUnknown:
      return AnchorTraceReason::kEvidenceUnknown;
    case AnchorDrop::kScanHole:
      return AnchorTraceReason::kScanHole;
    case AnchorDrop::kImuCoverageGap:
      return AnchorTraceReason::kImuCoverageGap;
    case AnchorDrop::kImuBridge:
      return AnchorTraceReason::kImuBridge;
    case AnchorDrop::kMapReset:
      return AnchorTraceReason::kMapReset;
    case AnchorDrop::kReferenceChanged:
      return AnchorTraceReason::kReferenceChanged;
    case AnchorDrop::kTimeEpoch:
      return AnchorTraceReason::kTimeEpoch;
    case AnchorDrop::kLost:
      return AnchorTraceReason::kLost;
    case AnchorDrop::kNotQualified:
      return AnchorTraceReason::kNotQualified;
    case AnchorDrop::kScanNotEvaluated:
      return AnchorTraceReason::kScanNotEvaluated;
    case AnchorDrop::kNone:
    case AnchorDrop::kIdExhausted:
      break;
  }
  return AnchorTraceReason::kNone;
}

inline const char * anchorScanOutcomeName(AnchorScanOutcome outcome)
{
  switch (outcome) {
    case AnchorScanOutcome::kPublished:
      return "published";
    case AnchorScanOutcome::kBootstrap:
      return "bootstrap";
    case AnchorScanOutcome::kImuFault:
      return "imu_fault";
    case AnchorScanOutcome::kImuNotProcessed:
      return "imu_not_processed";
    case AnchorScanOutcome::kImuCoverageGap:
      return "imu_coverage_gap";
    case AnchorScanOutcome::kNoCloud:
      return "no_cloud";
    case AnchorScanOutcome::kBeforeAnchorStep:
      return "before_anchor_step";
    case AnchorScanOutcome::kAfterAnchorStep:
      return "after_anchor_step";
    case AnchorScanOutcome::kDroppedInvalidTime:
      return "dropped_invalid_time";
    case AnchorScanOutcome::kDroppedNonadvancing:
      return "dropped_nonadvancing";
    case AnchorScanOutcome::kDroppedEmptyImu:
      return "dropped_empty_imu";
  }
  return "unknown";
}

inline const char * anchorTraceGateName(AnchorTraceGate gate)
{
  switch (gate) {
    case AnchorTraceGate::kNa:
      return "na";
    case AnchorTraceGate::kHealthy:
      return "healthy";
    case AnchorTraceGate::kBlind:
      return "blind";
    case AnchorTraceGate::kVerifying:
      return "verifying";
    case AnchorTraceGate::kLost:
      return "lost";
  }
  return "unknown";
}

inline const char * anchorTraceBodyName(AnchorTraceBody body)
{
  switch (body) {
    case AnchorTraceBody::kNa:
      return "na";
    case AnchorTraceBody::kStatic:
      return "static";
    case AnchorTraceBody::kMoving:
      return "moving";
  }
  return "unknown";
}

inline const char * anchorTraceActionName(AnchorTraceAction action)
{
  switch (action) {
    case AnchorTraceAction::kNa:
      return "na";
    case AnchorTraceAction::kNone:
      return "none";
    case AnchorTraceAction::kPin:
      return "pin";
    case AnchorTraceAction::kZeroVelocity:
      return "zero_velocity";
  }
  return "unknown";
}

inline const char * anchorStateName(AnchorState state)
{
  switch (state) {
    case AnchorState::kEmpty:
      return "empty";
    case AnchorState::kQualifying:
      return "qualifying";
    case AnchorState::kValid:
      return "valid";
  }
  return "unknown";
}

inline const char * staticEvidenceName(StaticEvidence evidence)
{
  switch (evidence) {
    case StaticEvidence::kStationary:
      return "stationary";
    case StaticEvidence::kMoving:
      return "moving";
    case StaticEvidence::kUnknown:
      return "unknown";
  }
  return "invalid";
}

inline const char * staticEvidenceReasonName(StaticEvidenceReason reason)
{
  switch (reason) {
    case StaticEvidenceReason::kImuAndWheel:
      return "imu_and_wheel";
    case StaticEvidenceReason::kImuOnly:
      return "imu_only";
    case StaticEvidenceReason::kGyro:
      return "gyro";
    case StaticEvidenceReason::kAccSpan:
      return "acc_span";
    case StaticEvidenceReason::kWheel:
      return "wheel";
    case StaticEvidenceReason::kNoImu:
      return "no_imu";
    case StaticEvidenceReason::kImuInvalid:
      return "imu_invalid";
    case StaticEvidenceReason::kWheelMissing:
      return "wheel_missing";
  }
  return "unknown";
}

inline const char * recoveryTrustName(RecoveryTrust trust)
{
  switch (trust) {
    case RecoveryTrust::kLocalQualified:
      return "local";
    case RecoveryTrust::kPending:
      return "pending";
    case RecoveryTrust::kUnanchored:
      return "unanchored";
  }
  return "unknown";
}

inline const char * recoveryCauseName(RecoveryCause cause)
{
  switch (cause) {
    case RecoveryCause::kNone:
      return "-";
    case RecoveryCause::kDegraded:
      return "degraded";
    case RecoveryCause::kScanHole:
      return "scan_hole";
    case RecoveryCause::kImuCoverageGap:
      return "imu_coverage_gap";
    case RecoveryCause::kImuBridge:
      return "imu_bridge";
    case RecoveryCause::kMapReset:
      return "map_reset";
    case RecoveryCause::kDeadline:
      return "deadline";
    case RecoveryCause::kStable:
      return "stable";
  }
  return "unknown";
}

inline const char * recoveryTraceEventName(RecoveryTraceEvent event)
{
  switch (event) {
    case RecoveryTraceEvent::kPending:
      return "pending";
    case RecoveryTraceEvent::kQualified:
      return "qualified";
    case RecoveryTraceEvent::kUnanchored:
      return "unanchored";
  }
  return "unknown";
}

inline const char * recoveryTracePathName(RecoveryTracePath path)
{
  switch (path) {
    case RecoveryTracePath::kScan:
      return "scan";
    case RecoveryTracePath::kIdle:
      return "idle";
  }
  return "unknown";
}

inline const char * anchorEventBitName(int bit)
{
  static const char * const kNames[kAnchorEventBitCount] = {"motion",
                                                            "scan_hole",
                                                            "imu_coverage_gap",
                                                            "imu_bridge",
                                                            "map_reset",
                                                            "reference_changed",
                                                            "lost",
                                                            "reanchored",
                                                            "runaway_trip",
                                                            "time_epoch",
                                                            "evidence_unknown"};
  return (bit >= 0 && bit < kAnchorEventBitCount) ? kNames[bit] : "unknown";
}

struct AnchorTransition
{
  AnchorTraceEvent event = AnchorTraceEvent::kEvidence;
  AnchorTraceReason reason = AnchorTraceReason::kNone;
  std::uint64_t event_seq = 0;
  std::uint64_t scan_seq = 0;
  std::int64_t stamp_ns = -1;
  std::uint64_t anchor_id = 0;
  bool anchor_valid = false;
  std::int64_t created_ns = -1;
  std::int64_t last_evidence_ns = -1;
  AnchorState anchor_state = AnchorState::kEmpty;
  std::uint64_t episode = 0;
  std::uint64_t ref_gen = 0;
  std::uint64_t streak = 0;
};

struct RecoveryTransition
{
  RecoveryTraceEvent event = RecoveryTraceEvent::kPending;
  RecoveryCause reason = RecoveryCause::kNone;
  std::uint64_t event_seq = 0;
  std::uint64_t scan_seq = 0;
  std::int64_t stamp_ns = -1;
  std::uint64_t episode = 0;
  std::int64_t start_ns = -1;
  std::int64_t deadline_ns = -1;
  std::int64_t elapsed_ns = -1;
  std::uint64_t reset_count = 0;
  RecoveryTracePath path = RecoveryTracePath::kScan;
};

/// One scan, as the scan record prints it. `anchor_step` says whether the
/// fields the anchor step fills were evaluated; `have_cov` whether the anchor
/// step wrote the state (the covariance fields are printed only then). The
/// fields from `have_act` on are filled only with the per-scan trace on.
struct AnchorScanRecord
{
  std::uint64_t scan_seq = 0;
  AnchorScanOutcome outcome = AnchorScanOutcome::kBeforeAnchorStep;
  std::int64_t beg_ns = -1;
  std::int64_t end_ns = -1;
  std::int64_t gap_ns = -1;
  std::uint64_t anchor_id = 0;
  bool anchor_valid = false;
  std::int64_t anchor_created_ns = -1;

  bool anchor_step = false;
  AnchorTraceBody body = AnchorTraceBody::kNa;
  int imu_n = 0;
  double gyr_max = std::numeric_limits<double>::quiet_NaN();
  double acc_span = std::numeric_limits<double>::quiet_NaN();
  double wheel_age = -1.0;
  double wheel_speed = std::numeric_limits<double>::quiet_NaN();
  bool guard_degraded = false;
  bool degraded = false;
  AnchorTraceGate gate_pre = AnchorTraceGate::kNa;
  AnchorTraceGate gate_post = AnchorTraceGate::kNa;
  AnchorTraceAction action = AnchorTraceAction::kNa;
  std::uint64_t anchor_used = 0;
  bool parked = false;
  bool parked_applied = false;
  bool zupt_active = false;
  std::uint32_t events = 0;

  bool have_pre = false;
  double pre_p[3] = {0.0, 0.0, 0.0};
  double pre_q[4] = {1.0, 0.0, 0.0, 0.0};
  double pre_v[3] = {0.0, 0.0, 0.0};
  bool have_post = false;
  double post_p[3] = {0.0, 0.0, 0.0};
  double post_q[4] = {1.0, 0.0, 0.0, 0.0};
  double post_v[3] = {0.0, 0.0, 0.0};

  bool have_cov = false;
  double p_pose_pre[36] = {};
  double p_pose_post[36] = {};
  bool p_finite = false;
  double p_sym = std::numeric_limits<double>::quiet_NaN();
  double p_min_eig = std::numeric_limits<double>::quiet_NaN();

  bool published = false;
  int flags = -1;

  AnchorState anchor_state = AnchorState::kEmpty;
  std::uint64_t episode = 0;
  std::uint64_t ref_gen = 0;
  StaticEvidence evidence = StaticEvidence::kUnknown;
  StaticEvidenceReason evidence_reason = StaticEvidenceReason::kNoImu;
  bool imu_only = false;
  bool wheel_valid = false;
  bool usable = false;
  bool zero_only = false;
  bool supported = false;
  RecoveryTrust trust_pre = RecoveryTrust::kLocalQualified;
  RecoveryTrust trust_post = RecoveryTrust::kLocalQualified;
  std::uint64_t recovery_episode = 0;
  std::int64_t recovery_start_ns = -1;
  std::int64_t recovery_deadline_ns = -1;
  bool eff_degraded = false;
  bool static_hold = false;

  bool have_act = false;  ///< an anchor-step write stands in the state
  double act_p[3] = {0.0, 0.0, 0.0};
  double act_q[4] = {1.0, 0.0, 0.0, 0.0};
  double act_v[3] = {0.0, 0.0, 0.0};
  double p_pose_act[36] = {};
  bool have_target = false;  ///< a pin or an applied parked prior stands
  double target_p[3] = {0.0, 0.0, 0.0};
  double target_q[4] = {1.0, 0.0, 0.0, 0.0};
  bool have_anchor_pose = false;  ///< a candidate or an anchor exists after the scan
  double anchor_p[3] = {0.0, 0.0, 0.0};
  double anchor_q[4] = {1.0, 0.0, 0.0, 0.0};
  std::uint64_t streak = 0;
  bool map_updated = false;
  bool have_update = false;  ///< the lidar measurement update ran
  double upd_dp = std::numeric_limits<double>::quiet_NaN();
  double upd_dq = std::numeric_limits<double>::quiet_NaN();
};

/// Indexed by AnchorDrop; only the reasons an anchor can end with are printed.
constexpr std::size_t kAnchorDropCount = static_cast<std::size_t>(AnchorDrop::kIdExhausted) + 1;

struct AnchorTraceCounters
{
  std::uint64_t scans = 0;
  std::uint64_t events = 0;
  std::uint64_t candidates = 0;
  std::uint64_t created = 0;
  std::uint64_t candidates_dropped = 0;
  std::uint64_t invalidated = 0;
  std::uint64_t invalidated_by[kAnchorDropCount] = {};
  std::uint64_t pin_applied = 0;
  std::uint64_t parked_prior_applied = 0;
  std::uint64_t zero_only_unanchored = 0;
  std::uint64_t blocked_stale_use = 0;
  std::uint64_t evidence_unknown = 0;
  std::uint64_t motion = 0;
  std::uint64_t scan_hole = 0;
  std::uint64_t imu_coverage_gap = 0;
  std::uint64_t imu_bridge = 0;
  std::uint64_t map_reset = 0;
  std::uint64_t reference_changed = 0;
  std::uint64_t lost = 0;
  std::uint64_t time_epoch = 0;
  std::uint64_t recovery_started = 0;
  std::uint64_t recovery_qualified = 0;
  std::uint64_t recovery_expired = 0;
  std::uint64_t map_reset_unanchored = 0;
};

/// The drops printed as invalidated_<reason>, in order.
constexpr AnchorDrop kAnchorInvalidationReasons[] = {AnchorDrop::kMotion,
                                                     AnchorDrop::kEvidenceUnknown,
                                                     AnchorDrop::kScanHole,
                                                     AnchorDrop::kImuCoverageGap,
                                                     AnchorDrop::kImuBridge,
                                                     AnchorDrop::kMapReset,
                                                     AnchorDrop::kReferenceChanged,
                                                     AnchorDrop::kTimeEpoch,
                                                     AnchorDrop::kLost,
                                                     AnchorDrop::kNotQualified};

namespace anchor_trace_detail
{

class LineBuilder
{
public:
  void raw(const char * text) { line_ += text; }
  void key(const char * name)
  {
    if (!line_.empty()) {
      line_ += ' ';
    }
    line_ += name;
    line_ += '=';
  }
  void text(const char * name, const char * value)
  {
    key(name);
    line_ += value;
  }
  void u64(const char * name, std::uint64_t value)
  {
    key(name);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(value));
    line_ += buffer;
  }
  void i64(const char * name, std::int64_t value)
  {
    key(name);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(value));
    line_ += buffer;
  }
  void boolean(const char * name, bool value) { text(name, value ? "1" : "0"); }
  void real(const char * name, double value)
  {
    key(name);
    appendReal(value);
  }
  void reals(const char * name, const double * values, int count)
  {
    key(name);
    for (int i = 0; i < count; ++i) {
      if (i > 0) {
        line_ += ',';
      }
      appendReal(values[i]);
    }
  }
  void hash(const char * name, const double * values, std::size_t count)
  {
    key(name);
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(fnv1a64(values, count)));
    line_ += buffer;
  }
  const std::string & str() const { return line_; }

private:
  void appendReal(double value)
  {
    if (std::isnan(value)) {
      line_ += "nan";
      return;
    }
    if (std::isinf(value)) {
      line_ += value > 0.0 ? "inf" : "-inf";
      return;
    }
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    line_ += buffer;
  }

  std::string line_;
};

}  // namespace anchor_trace_detail

inline std::string formatAnchorTransition(const AnchorTransition & t)
{
  anchor_trace_detail::LineBuilder line;
  line.raw("[ZUPT-ANCHOR]");
  line.u64("v", kAnchorTraceVersion);
  line.text("event", anchorTraceEventName(t.event));
  line.text("reason", anchorTraceReasonName(t.reason));
  line.u64("event_seq", t.event_seq);
  line.u64("scan_seq", t.scan_seq);
  line.i64("stamp_ns", t.stamp_ns);
  line.u64("anchor_id", t.anchor_id);
  line.boolean("anchor_valid", t.anchor_valid);
  line.i64("created_ns", t.created_ns);
  line.i64("last_evidence_ns", t.last_evidence_ns);
  line.text("anchor_state", anchorStateName(t.anchor_state));
  line.u64("stationary_episode", t.episode);
  line.u64("reference_generation", t.ref_gen);
  line.u64("streak", t.streak);
  return line.str();
}

inline std::string formatRecoveryTransition(const RecoveryTransition & t)
{
  anchor_trace_detail::LineBuilder line;
  line.raw("[FE-RECOVERY]");
  line.u64("v", kAnchorTraceVersion);
  line.text("event", recoveryTraceEventName(t.event));
  line.text("reason", recoveryCauseName(t.reason));
  line.u64("event_seq", t.event_seq);
  line.u64("scan_seq", t.scan_seq);
  line.i64("stamp_ns", t.stamp_ns);
  line.u64("episode", t.episode);
  line.i64("start_ns", t.start_ns);
  line.i64("deadline_ns", t.deadline_ns);
  line.i64("elapsed_ns", t.elapsed_ns);
  line.u64("reset_count", t.reset_count);
  line.text("path", recoveryTracePathName(t.path));
  return line.str();
}

inline std::string formatAnchorEvents(std::uint32_t events)
{
  if (events == 0u) {
    return "-";
  }
  std::string out;
  for (int bit = 0; bit < kAnchorEventBitCount; ++bit) {
    if ((events & (1u << bit)) != 0u) {
      if (!out.empty()) {
        out += ',';
      }
      out += anchorEventBitName(bit);
    }
  }
  return out;
}

inline std::string formatAnchorScanRecord(const AnchorScanRecord & r)
{
  anchor_trace_detail::LineBuilder line;
  line.raw("[ZUPT-ANCHOR-SCAN]");
  line.u64("v", kAnchorTraceVersion);
  line.u64("scan_seq", r.scan_seq);
  line.text("outcome", anchorScanOutcomeName(r.outcome));
  line.i64("beg_ns", r.beg_ns);
  line.i64("end_ns", r.end_ns);
  line.i64("gap_ns", r.gap_ns);
  line.u64("anchor_id", r.anchor_id);
  line.boolean("anchor_valid", r.anchor_valid);
  line.i64("anchor_created_ns", r.anchor_created_ns);
  if (r.anchor_step) {
    line.text("body", anchorTraceBodyName(r.body));
    line.i64("imu_n", r.imu_n);
    line.real("gyr_max", r.gyr_max);
    line.real("acc_span", r.acc_span);
    line.real("wheel_age", r.wheel_age);
    line.real("wheel_speed", r.wheel_speed);
    line.boolean("guard_degraded", r.guard_degraded);
    line.boolean("degraded", r.degraded);
  } else {
    for (const char * name :
         {"body", "imu_n", "gyr_max", "acc_span", "wheel_age", "wheel_speed", "guard_degraded", "degraded"}) {
      line.text(name, "na");
    }
  }
  line.text("gate_pre", anchorTraceGateName(r.gate_pre));
  line.text("gate_post", anchorTraceGateName(r.gate_post));
  line.text("action", anchorTraceActionName(r.action));
  line.u64("anchor_used", r.anchor_used);
  if (r.anchor_step) {
    line.boolean("parked", r.parked);
    line.boolean("parked_applied", r.parked_applied);
    line.boolean("zupt_active", r.zupt_active);
  } else {
    for (const char * name : {"parked", "parked_applied", "zupt_active"}) {
      line.text(name, "na");
    }
  }
  line.text("events", formatAnchorEvents(r.events).c_str());
  if (r.have_pre) {
    line.reals("pre_p", r.pre_p, 3);
    line.reals("pre_q", r.pre_q, 4);
    line.reals("pre_v", r.pre_v, 3);
  } else {
    for (const char * name : {"pre_p", "pre_q", "pre_v"}) {
      line.text(name, "na");
    }
  }
  if (r.have_post) {
    line.reals("post_p", r.post_p, 3);
    line.reals("post_q", r.post_q, 4);
    line.reals("post_v", r.post_v, 3);
  } else {
    for (const char * name : {"post_p", "post_q", "post_v"}) {
      line.text(name, "na");
    }
  }
  if (r.have_cov) {
    line.hash("p_pose_pre", r.p_pose_pre, 36);
    line.hash("p_pose_post", r.p_pose_post, 36);
    line.boolean("p_finite", r.p_finite);
    line.real("p_sym", r.p_sym);
    line.real("p_min_eig", r.p_min_eig);
  } else {
    for (const char * name : {"p_pose_pre", "p_pose_post", "p_finite", "p_sym", "p_min_eig"}) {
      line.text(name, "na");
    }
  }
  line.boolean("published", r.published);
  line.i64("flags", r.flags);
  line.text("anchor_state", anchorStateName(r.anchor_state));
  line.u64("stationary_episode", r.episode);
  line.u64("reference_generation", r.ref_gen);
  if (r.anchor_step) {
    line.text("evidence", staticEvidenceName(r.evidence));
    line.text("evidence_reason", staticEvidenceReasonName(r.evidence_reason));
    line.boolean("imu_only", r.imu_only);
    line.boolean("wheel_valid", r.wheel_valid);
    line.boolean("usable", r.usable);
    line.boolean("zero_only", r.zero_only);
    line.boolean("supported", r.supported);
  } else {
    for (const char * name :
         {"evidence", "evidence_reason", "imu_only", "wheel_valid", "usable", "zero_only", "supported"}) {
      line.text(name, "na");
    }
  }
  line.text("trust_pre", recoveryTrustName(r.trust_pre));
  line.text("trust_post", recoveryTrustName(r.trust_post));
  line.u64("recovery_episode", r.recovery_episode);
  line.i64("recovery_start_ns", r.recovery_start_ns);
  line.i64("recovery_deadline_ns", r.recovery_deadline_ns);
  line.boolean("eff_degraded", r.eff_degraded);
  line.boolean("static_hold", r.static_hold);
  if (r.have_act) {
    line.reals("act_p", r.act_p, 3);
    line.reals("act_q", r.act_q, 4);
    line.reals("act_v", r.act_v, 3);
    line.hash("p_pose_act", r.p_pose_act, 36);
  } else {
    for (const char * name : {"act_p", "act_q", "act_v", "p_pose_act"}) {
      line.text(name, "na");
    }
  }
  if (r.have_target) {
    line.reals("target_p", r.target_p, 3);
    line.reals("target_q", r.target_q, 4);
  } else {
    for (const char * name : {"target_p", "target_q"}) {
      line.text(name, "na");
    }
  }
  if (r.have_anchor_pose) {
    line.reals("anchor_p", r.anchor_p, 3);
    line.reals("anchor_q", r.anchor_q, 4);
  } else {
    for (const char * name : {"anchor_p", "anchor_q"}) {
      line.text(name, "na");
    }
  }
  line.u64("streak", r.streak);
  line.boolean("map_updated", r.map_updated);
  if (r.have_update) {
    line.real("upd_dp", r.upd_dp);
    line.real("upd_dq", r.upd_dq);
  } else {
    for (const char * name : {"upd_dp", "upd_dq"}) {
      line.text(name, "na");
    }
  }
  return line.str();
}

inline std::string formatAnchorCounters(const AnchorTraceCounters & c, std::uint64_t event_seq, bool exhausted)
{
  anchor_trace_detail::LineBuilder line;
  line.raw("[ZUPT-ANCHOR]");
  line.u64("v", kAnchorTraceVersion);
  line.text("event", "counters");
  line.u64("event_seq", event_seq);
  line.u64("scans", c.scans);
  line.u64("events", c.events);
  line.u64("candidates", c.candidates);
  line.u64("created", c.created);
  line.u64("candidates_dropped", c.candidates_dropped);
  line.u64("invalidated", c.invalidated);
  for (const AnchorDrop drop : kAnchorInvalidationReasons) {
    const std::string name = std::string("invalidated_") + anchorTraceReasonName(anchorTraceReasonOf(drop));
    line.u64(name.c_str(), c.invalidated_by[static_cast<std::size_t>(drop)]);
  }
  line.u64("pin_applied", c.pin_applied);
  line.u64("parked_prior_applied", c.parked_prior_applied);
  line.u64("zero_only_unanchored", c.zero_only_unanchored);
  line.u64("blocked_stale_use", c.blocked_stale_use);
  line.u64("evidence_unknown", c.evidence_unknown);
  line.u64("motion", c.motion);
  line.u64("scan_hole", c.scan_hole);
  line.u64("imu_coverage_gap", c.imu_coverage_gap);
  line.u64("imu_bridge", c.imu_bridge);
  line.u64("map_reset", c.map_reset);
  line.u64("reference_changed", c.reference_changed);
  line.u64("lost", c.lost);
  line.u64("time_epoch", c.time_epoch);
  line.u64("recovery_started", c.recovery_started);
  line.u64("recovery_qualified", c.recovery_qualified);
  line.u64("recovery_expired", c.recovery_expired);
  line.u64("map_reset_unanchored", c.map_reset_unanchored);
  line.boolean("sequence_exhausted", exhausted);
  return line.str();
}

}  // namespace fast_lio

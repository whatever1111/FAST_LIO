// anchor_trace.hpp — a machine-readable trace of the ZUPT stationary anchor.
//
// The degraded-scan static hold and the healthy-scan parked prior both pull the
// estimate onto an anchor pose. Whether the anchor a scan used still belonged to
// the stationary stretch it was taken in cannot be read off the published pose:
// a pin to a stale anchor and a pin to a fresh one look the same downstream.
// This trace names the anchor every scan used and every event that breaks the
// evidence behind an anchor, so an offline checker can decide it scan by scan.
//
// Observation only: nothing in this header, and nothing the node does to fill
// it, feeds back into the estimate.
//
// Pure C++. No ROS, no Eigen.
//
// ===========================================================================
// Trace format, version 1. This comment is the specification; a checker must
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
//   reals       printf %.17g, so every double round-trips bit-exactly; non-finite
//               values print as nan, inf or -inf.
//   booleans    0 or 1.
//   na          the field was not evaluated for this scan (see `outcome`).
//   p_pose_*    FNV-1a 64-bit hash, 16 lowercase hex digits, of the raw bytes of
//               the 6x6 position/attitude block of the state covariance (rows and
//               columns 0..5, row-major doubles). Equal hashes mean the block is
//               bit-identical.
//
// Sequence rules
//   event_seq   1, 2, 3, ... over the process lifetime, one per transition record
//               (the counter summary included). Never repeats, never decreases.
//   scan_seq    1, 2, 3, ... one per scan the front end takes off its input
//               queue, whatever happens to it next, in processing order. Scan
//               records carry it only with the per-scan trace on; the counter
//               summary's `scans` is its final value either way.
//   anchor_id   1, 2, 3, ... incremented every time the anchor pose is written;
//               0 = no anchor. An ID is never reused.
//   A sequence that reaches 2^64-1 is exhausted: from then on it yields 0, which
//   no record carries legitimately, and the counter summary says so. It never
//   wraps back onto old values.
//
// 1. Transition records: always on, never throttled, one line per event.
//
//   [ZUPT-ANCHOR] v=1 event=<event> reason=<reason> event_seq=<n> scan_seq=<n>
//     stamp_ns=<ns> anchor_id=<id> anchor_valid=<0|1> created_ns=<ns>
//     last_evidence_ns=<ns>
//
//   event              pin_engaged, pin_released, parked_prior_engaged,
//                      parked_prior_released, evidence.
//   reason             for `evidence`: motion (the static verdict went from
//                      static to not static), scan_hole, imu_coverage_gap,
//                      imu_bridge, map_reset, reference_changed, lost,
//                      time_epoch. For the release events: scan_healthy or
//                      motion. Otherwise `-`.
//                      imu_coverage_gap and imu_bridge are reported on the first
//                      scan of a consecutive run; the counters count every one.
//   scan_seq           the scan the event belongs to.
//   stamp_ns           that scan's sensor end time.
//   anchor_id ... created_ns   the anchor at the time of the event, and the end
//                      time of the scan whose pose became that anchor.
//   last_evidence_ns   end time of the last scan whose static evidence the anchor
//                      logic evaluated.
//
// 2. Scan records: one per scan, only with the parameter zupt_anchor_trace_en.
//
//   [ZUPT-ANCHOR-SCAN] v=1 scan_seq=<n> outcome=<outcome> beg_ns=<ns>
//     end_ns=<ns> gap_ns=<ns> anchor_id=<id> anchor_valid=<0|1>
//     anchor_created_ns=<ns> body=<body> imu_n=<n> gyr_max=<rad/s>
//     acc_span=<ratio> wheel_age=<s> wheel_speed=<m/s> guard_degraded=<0|1>
//     degraded=<0|1> gate_pre=<gate> gate_post=<gate> action=<action>
//     anchor_used=<id> parked=<0|1> parked_applied=<0|1> zupt_active=<0|1>
//     events=<list> pre_p=<x,y,z> pre_q=<w,x,y,z> pre_v=<x,y,z> post_p=<x,y,z>
//     post_q=<w,x,y,z> post_v=<x,y,z> p_pose_pre=<hash> p_pose_post=<hash>
//     p_finite=<0|1> p_sym=<real> p_min_eig=<real> published=<0|1>
//     flags=<bits>
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
//   anchor_*    the anchor after this scan's anchor step (or as it stands, for
//               scans that stopped earlier).
//   body        static or moving: the static-evidence verdict (IMU window plus
//               newest wheel sample) the anchor logic used. imu_n, gyr_max,
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
//   events      comma-separated, `-` when empty: motion, scan_hole,
//               imu_coverage_gap, imu_bridge, map_reset, reference_changed,
//               lost, reanchored, runaway_trip, time_epoch.
//   pre_*       the state right before the anchor step; post_* the state after
//               the last writer of the scan (what /Odometry carries).
//   p_*         only on scans whose anchor step wrote the state, na otherwise.
//               p_pose_pre is taken before the anchor step, p_pose_post after the
//               scan's last writer. p_finite = every covariance entry finite;
//               p_sym = max|P-P^T| / max(1, rho) with rho the largest |eigenvalue|
//               of (P+P^T)/2; p_min_eig = smallest eigenvalue of (P+P^T)/2. The
//               last three need zupt_anchor_trace_en (nan without it).
//   flags       the legacy health bits published in /Odometry twist.covariance[1]
//               (bit0 degraded, bit1 static hold, bit2 lost, bit3 levelling);
//               -1 = nothing published.
//
// 3. Counter summary: once, when the node shuts down. A transition record with
//    its own keys, all saturating unsigned 64-bit counts:
//
//   [ZUPT-ANCHOR] v=1 event=counters event_seq=<n> scans=<n> events=<n>
//     anchor_refreshed=<n> pin_applied=<n> parked_prior_applied=<n>
//     motion=<n> scan_hole=<n> imu_coverage_gap=<n> imu_bridge=<n>
//     map_reset=<n> reference_changed=<n> lost=<n> time_epoch=<n>
//     sequence_exhausted=<0|1>
//
//   events counts the transition records before this one; scans the scans
//   taken; the rest count occurrences (every scan, not only run starts).

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

namespace fast_lio
{

constexpr int kAnchorTraceVersion = 1;

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
constexpr int kAnchorEventBitCount = 10;

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
  }
  return "unknown";
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
                                                            "time_epoch"};
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
};

/// One scan, as the scan record prints it. `anchor_step` says whether the
/// fields the anchor step fills were evaluated; `have_cov` whether the anchor
/// step wrote the state (the covariance fields are printed only then).
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
};

struct AnchorTraceCounters
{
  std::uint64_t scans = 0;
  std::uint64_t events = 0;
  std::uint64_t anchor_refreshed = 0;
  std::uint64_t pin_applied = 0;
  std::uint64_t parked_prior_applied = 0;
  std::uint64_t motion = 0;
  std::uint64_t scan_hole = 0;
  std::uint64_t imu_coverage_gap = 0;
  std::uint64_t imu_bridge = 0;
  std::uint64_t map_reset = 0;
  std::uint64_t reference_changed = 0;
  std::uint64_t lost = 0;
  std::uint64_t time_epoch = 0;
};

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
  line.u64("anchor_refreshed", c.anchor_refreshed);
  line.u64("pin_applied", c.pin_applied);
  line.u64("parked_prior_applied", c.parked_prior_applied);
  line.u64("motion", c.motion);
  line.u64("scan_hole", c.scan_hole);
  line.u64("imu_coverage_gap", c.imu_coverage_gap);
  line.u64("imu_bridge", c.imu_bridge);
  line.u64("map_reset", c.map_reset);
  line.u64("reference_changed", c.reference_changed);
  line.u64("lost", c.lost);
  line.u64("time_epoch", c.time_epoch);
  line.boolean("sequence_exhausted", exhausted);
  return line.str();
}

}  // namespace fast_lio

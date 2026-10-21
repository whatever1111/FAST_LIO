// RW near-field carry, carry contract 3: the producer's pure state and per-scan certificate.
// It reads the scan's final verdicts and the reference lifecycle and changes neither; RecoveryState stays the only
// reference FSM and the map, the estimate and the legacy outputs are untouched.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>

#include "reanchor_gate.hpp"
#include "reference_health_contract.hpp"

namespace fast_lio
{
// Fixed design gates; none of them is a tuning surface.
inline constexpr int kCarryMinEffective = 200;     // reanchor_min_eff
inline constexpr double kCarryMaxResidual = 0.10;  // reanchor_max_res
inline constexpr double kCarryResidualFloor = 0.03;
inline constexpr std::size_t kCarryOriginCapacity = 100;
inline constexpr std::size_t kCarryOriginMinimum = 20;
inline constexpr std::int64_t kCarryWindowNs = 300000000;
inline constexpr std::int64_t kCarryWindowMaxNs = 450000000;
inline constexpr std::int64_t kCarryPoseGapNs = 200000000;
inline constexpr double kCarryMaxVerticalSpeed = 0.5;  // m/s
inline constexpr double kCarryMaxTwistRateDeg = 15.0;  // deg/s, FE against gyro about the gravity axis
inline constexpr double kCarryTwistMinNorm = 1e-6;     // below it the swing is ~180 deg and the twist undefined
inline constexpr std::int64_t kCarryImuGapNs = 50000000;
inline constexpr std::int64_t kCarryWheelFreshNs = 500000000;
inline constexpr std::int64_t kCarryCurrentAgeNs = 100000000;
inline constexpr std::size_t kCarryCurrentMinCount = 3;
inline constexpr double kCarryCurrentMinFraction = 0.6;
inline constexpr std::int64_t kCarryCurrentMinSpanNs = 180000000;
inline constexpr std::int64_t kCarryCurrentMaxGapNs = 100000000;
inline constexpr double kCarryLowerScale = 0.5;
inline constexpr double kCarryLowerSlack = 0.2;  // m/s
inline constexpr double kCarryUpperScale = 1.5;
inline constexpr double kCarryUpperSlack = 0.3;  // m/s
inline constexpr std::size_t kCarryRearmScans = 20;
inline constexpr std::size_t kCarryPoseCapacity = 32;
inline constexpr std::size_t kCarryImuCapacity = 1024;
inline constexpr std::size_t kCarryWheelCapacity = 256;
inline constexpr std::uint32_t kCarryPermanentHard = 2u | 4u | 64u;  // Unanchored, map reset, fatal epoch
inline constexpr std::uint8_t kCarryWitnessCurrent = 1;
inline constexpr std::uint8_t kCarryWitnessFallback = 2;

// One leg-odometry sample at its own sensor epoch. NaN marks "cannot measure" and stays NaN.
struct CarryWheelSample
{
  std::int64_t ns = 0;
  double speed = std::numeric_limits<double>::quiet_NaN();
};

// Horizontal leg speed of the base point; any non-finite axis leaves the sample unknown, never zero.
inline double carryWheelSpeed(double vx, double vy, double vz)
{
  return std::isfinite(vx) && std::isfinite(vy) && std::isfinite(vz) ? std::hypot(vx, vy)
                                                                     : std::numeric_limits<double>::quiet_NaN();
}

inline bool carryLowerHolds(double wheel, double fe)
{
  return std::max(0.0, kCarryLowerScale * wheel - kCarryLowerSlack) <= fe;
}

inline bool carryUpperHolds(double wheel, double fe)
{
  return fe <= kCarryUpperScale * wheel + kCarryUpperSlack;
}

struct CarryWheelVerdict
{
  bool supported = false;
  std::uint8_t kind = 0;
  std::int64_t untilNs = 0;
  std::int64_t lowerNs = 0, upperNs = 0;  // fallback witnesses only
};

// Leg evidence for the FE horizontal speed over the FE window [startNs, endNs]. `wheel` is ascending in ns. A window
// with sufficient current samples decides alone, and a contradiction there is final. Only an insufficient window falls
// back to the last second, where the newest sample supporting each bound is its witness and expires 1 s after itself.
inline CarryWheelVerdict
carryWheelEvidence(const std::deque<CarryWheelSample> & wheel, std::int64_t startNs, std::int64_t endNs, double feSpeed)
{
  CarryWheelVerdict verdict;
  std::size_t raw = 0, finite = 0;
  double sum = 0.0;
  std::int64_t firstFinite = -1, lastFinite = -1, previousRaw = -1, rawGap = 0, finiteGap = 0;
  for (const auto & sample : wheel) {
    if (sample.ns < startNs || sample.ns > endNs)
      continue;
    if (previousRaw >= 0)
      rawGap = std::max(rawGap, sample.ns - previousRaw);
    previousRaw = sample.ns;
    ++raw;
    if (!std::isfinite(sample.speed))
      continue;
    if (lastFinite >= 0)
      finiteGap = std::max(finiteGap, sample.ns - lastFinite);
    else
      firstFinite = sample.ns;
    lastFinite = sample.ns;
    ++finite;
    sum += sample.speed;
  }
  const bool sufficient = finite >= kCarryCurrentMinCount && endNs - lastFinite <= kCarryCurrentAgeNs &&
                          static_cast<double>(finite) >= kCarryCurrentMinFraction * static_cast<double>(raw) &&
                          lastFinite - firstFinite >= kCarryCurrentMinSpanNs && rawGap <= kCarryCurrentMaxGapNs &&
                          finiteGap <= kCarryCurrentMaxGapNs;
  if (sufficient) {
    const double current = sum / static_cast<double>(finite);
    if (carryLowerHolds(current, feSpeed) && carryUpperHolds(current, feSpeed)) {
      verdict.supported = true;
      verdict.kind = kCarryWitnessCurrent;
      verdict.untilNs = referenceAdd(lastFinite, kCarryWheelFreshNs);
    }
    return verdict;
  }
  std::int64_t newest = -1, lower = -1, upper = -1;
  for (const auto & sample : wheel) {
    if (sample.ns < endNs - kCarryWitnessNs || sample.ns > endNs || !std::isfinite(sample.speed))
      continue;
    newest = sample.ns;
    if (carryLowerHolds(sample.speed, feSpeed))
      lower = sample.ns;
    if (carryUpperHolds(sample.speed, feSpeed))
      upper = sample.ns;
  }
  if (newest < 0 || endNs - newest > kCarryWheelFreshNs || lower < 0 || upper < 0)
    return verdict;
  verdict.supported = true;
  verdict.kind = kCarryWitnessFallback;
  verdict.lowerNs = lower;
  verdict.upperNs = upper;
  verdict.untilNs = std::min({referenceAdd(newest, kCarryWheelFreshNs),
                              referenceAdd(lower, kCarryWitnessNs),
                              referenceAdd(upper, kCarryWitnessNs)});
  return verdict;
}

struct CarryImu
{
  std::int64_t ns = 0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // rad/s, IMU (= FE body) frame
};

// Body rotation over [startNs, endNs] from bias-corrected gyro samples, each held until the next one. The sample in
// force at every instant, the window's start and end included, must be no older than kCarryImuGapNs.
inline bool carryGyroRotation(const std::deque<CarryImu> & imu,
                              std::int64_t startNs,
                              std::int64_t endNs,
                              const Eigen::Vector3d & bias,
                              Eigen::Quaterniond & rotation)
{
  auto it = std::upper_bound(
    imu.begin(), imu.end(), startNs, [](std::int64_t stamp, const CarryImu & sample) { return stamp < sample.ns; });
  if (it == imu.begin())
    return false;
  --it;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  std::int64_t t = startNs;
  while (true) {
    const auto next = std::next(it);
    const std::int64_t until = next != imu.end() && next->ns < endNs ? next->ns : endNs;
    const Eigen::Vector3d omega = it->gyro - bias;
    if (until - it->ns > kCarryImuGapNs || !omega.allFinite())
      return false;
    const double angle = omega.norm() * static_cast<double>(until - t) * 1e-9;
    if (angle > 0.0)
      q = q * Eigen::Quaterniond(Eigen::AngleAxisd(angle, omega.normalized()));
    t = until;
    if (until == endNs)
      break;
    it = next;
  }
  rotation = q.normalized();
  return true;
}

// Rotation angle of `q` about the unit axis `axis` (swing-twist), in [-pi, pi]; NaN where it is undefined.
inline double carryTwistAngle(const Eigen::Quaterniond & q, const Eigen::Vector3d & axis)
{
  const double along = q.vec().dot(axis);
  if (std::hypot(along, q.w()) < kCarryTwistMinNorm)
    return std::numeric_limits<double>::quiet_NaN();
  return 2.0 * std::atan2(q.w() < 0.0 ? -along : along, std::abs(q.w()));
}

enum class CarryProvenance : std::uint8_t
{
  kNone,        // the raw episode started from anything but a pure engulf on a qualified origin
  kHoldOnly,    // it started under a genuine StaticHold; a pure engulf may still attach
  kPureEngulf,  // the frozen origin may vouch for this episode
};

// The completed scan's final facts, read after every late guard.
struct CarryScan
{
  std::uint64_t sequence = 0;
  std::int64_t beginNs = 0, endNs = 0;
  std::uint64_t generation = 0;  // stationary-anchor reference generation
  bool established = false;      // lifecycle Established
  RecoveryTrust trust = RecoveryTrust::kLocalQualified;
  std::int64_t deadlineNs = -1;  // #273 recovery deadline, -1 = none
  bool degraded = false;         // D_scan, the published raw registration verdict
  bool pureEngulf = false;       // the engulf latch alone made this scan degraded
  bool staticHold = false;
  bool lost = false;
  int iterations = 0, effective = 0;
  double residual = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector3d base = Eigen::Vector3d::Zero();  // world position of the legs' reference point
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d up = Eigen::Vector3d::UnitZ();  // world unit vector against gravity
  Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
};

struct CarryVerdict
{
  bool permitted = false;    // R1 positive bit 5, current pure-engulf carry
  bool constrained = false;  // odom bit 4: permitted on a raw degraded scan
  std::int64_t validUntilNs = 0;
  std::uint8_t witnessKind = 0;
  std::int64_t lowerWitnessNs = 0, upperWitnessNs = 0;
  bool rearmed = false;  // this scan legally recovered and cleared the latch
};

class ConstrainedCarry
{
public:
  // Every negative entry point, scan or not; cause bits 0/1 alone are no taint.
  void negative(std::uint32_t cause, std::uint32_t hard)
  {
    cause_ |= cause;
    taint_ |= cause & kReferenceTaintBits;
    hard_ |= hard;
    if ((cause & kReferenceTaintBits) != 0 || hard != 0) {
      scanNegative_ = true;
      qualifiedSince_ = pendingSeen_ = false;
      trustedRun_ = 0;
    }
  }

  // A real #273 qualified event.
  void qualified() { qualifiedSince_ = true; }

  void noteImu(std::int64_t ns, const Eigen::Vector3d & gyro)
  {
    if (!imu_.empty() && ns <= imu_.back().ns)
      return;
    imu_.push_back({ns, gyro});
    if (imu_.size() > kCarryImuCapacity)
      imu_.pop_front();
  }

  // Once per completed scan, after its negatives and the #273 step. `stableNs` is reanchor_recovery_stable_sec.
  CarryVerdict scan(const CarryScan & in, const std::deque<CarryWheelSample> & wheel, std::int64_t stableNs)
  {
    if (in.generation != generation_) {
      generation_ = in.generation;
      origin_.clear();
      poses_.clear();
      originScan_ = 0;
      trustedRun_ = 0;
    }
    if (in.trust == RecoveryTrust::kUnanchored)
      hard_ |= 2u;
    poses_.push_back({in.endNs, in.base, in.rotation.toRotationMatrix()});
    if (poses_.size() > kCarryPoseCapacity)
      poses_.pop_front();
    const bool measured = in.iterations > 0 && in.effective > 0 && std::isfinite(in.residual) && in.residual >= 0.0;
    const bool trusted = in.established && in.trust == RecoveryTrust::kLocalQualified && !in.degraded && !in.lost &&
                         !scanNegative_ && measured;
    scanNegative_ = false;
    if (trusted && trustedRun_ > 0 && in.sequence == lastSequence_ + 1) {
      ++trustedRun_;
    } else if (trusted) {
      trustedRun_ = 1;
      runBeginNs_ = in.beginNs;
    } else {
      trustedRun_ = 0;
    }
    lastSequence_ = in.sequence;

    if (in.degraded && !rawActive_) {
      rawActive_ = true;
      rawStartNs_ = in.endNs;
      ++rawEpisode_;
      frozen_ = origin_.median();
      qualifiedSince_ = pendingSeen_ = false;
      provenance_ = !in.established || latched() || !std::isfinite(frozen_) ? CarryProvenance::kNone
                    : in.staticHold                                         ? CarryProvenance::kHoldOnly
                    : in.pureEngulf                                         ? CarryProvenance::kPureEngulf
                                                                            : CarryProvenance::kNone;
    } else if (rawActive_ && provenance_ == CarryProvenance::kHoldOnly && in.degraded && !in.staticHold) {
      provenance_ = in.pureEngulf && !latched() ? CarryProvenance::kPureEngulf : CarryProvenance::kNone;
    }

    CarryVerdict verdict;
    if (unresolved() && in.trust == RecoveryTrust::kPending)
      pendingSeen_ = true;
    if (unresolved() && (hard_ & kCarryPermanentHard) == 0 &&
        (pendingSeen_ ? qualifiedSince_ && trusted
                      : trustedRun_ >= kCarryRearmScans && in.endNs - runBeginNs_ >= stableNs)) {
      rawActive_ = false;
      provenance_ = CarryProvenance::kNone;
      frozen_ = std::numeric_limits<double>::quiet_NaN();
      cause_ = taint_ = hard_ = 0;
      qualifiedSince_ = pendingSeen_ = false;
      ++rearm_;
      verdict.rearmed = true;
    }
    if (!unresolved() && trusted) {
      origin_.push(in.residual);
      originScan_ = in.sequence;
    }

    if (!rawActive_ || provenance_ != CarryProvenance::kPureEngulf || latched() || !in.established || in.staticHold ||
        in.lost || !(in.degraded || in.trust == RecoveryTrust::kPending))
      return verdict;
    const double bound = std::min(kCarryMaxResidual, 2.0 * std::max(frozen_, kCarryResidualFloor));
    if (!measured || in.effective < kCarryMinEffective || in.residual > bound)
      return verdict;

    // FE window: the newest pose no later than end - 0.3 s, through this scan, without interpolation.
    auto start = poses_.rend();
    for (auto it = poses_.rbegin(); it != poses_.rend(); ++it) {
      if (it != poses_.rbegin() && std::prev(it)->ns - it->ns > kCarryPoseGapNs)
        return verdict;
      if (it->ns <= in.endNs - kCarryWindowNs) {
        start = it;
        break;
      }
    }
    if (start == poses_.rend() || in.endNs - start->ns > kCarryWindowMaxNs)
      return verdict;
    const double span = static_cast<double>(in.endNs - start->ns) * 1e-9;
    const Eigen::Vector3d up = in.up.normalized();
    const Eigen::Vector3d moved = in.base - start->base;
    const double vertical = moved.dot(up) / span;
    const double horizontal = (moved - moved.dot(up) * up).norm() / span;
    if (!std::isfinite(horizontal) || !(std::abs(vertical) <= kCarryMaxVerticalSpeed))
      return verdict;
    Eigen::Quaterniond gyro;
    if (!carryGyroRotation(imu_, start->ns, in.endNs, in.gyroBias, gyro))
      return verdict;
    const Eigen::Quaterniond startRotation(start->rotation);
    const Eigen::Vector3d axis = startRotation.conjugate() * up;
    const double feTwist = carryTwistAngle(startRotation.conjugate() * in.rotation, axis);
    const double gyroTwist = carryTwistAngle(gyro, axis);
    const double gap = std::remainder(feTwist - gyroTwist, 2.0 * M_PI);
    if (!(std::abs(gap) / span <= kCarryMaxTwistRateDeg * M_PI / 180.0))
      return verdict;

    const auto legs = carryWheelEvidence(wheel, start->ns, in.endNs, horizontal);
    if (!legs.supported)
      return verdict;
    std::int64_t until =
      std::min({referenceUntil(in.endNs), legs.untilNs, referenceAdd(rawStartNs_, kCarryRawBudgetNs)});
    if (in.trust == RecoveryTrust::kPending && in.deadlineNs >= 0)
      until = std::min(until, in.deadlineNs - 1);
    if (until < in.endNs)
      return verdict;
    verdict.permitted = true;
    verdict.constrained = in.degraded;
    verdict.validUntilNs = until;
    verdict.witnessKind = legs.kind;
    verdict.lowerWitnessNs = legs.lowerNs;
    verdict.upperWitnessNs = legs.upperNs;
    return verdict;
  }

  bool latched() const { return taint_ != 0 || hard_ != 0; }
  // Raw episode or latch still waiting for a legal recovery.
  bool unresolved() const { return rawActive_ || latched(); }
  bool rawActive() const { return rawActive_; }
  std::int64_t rawStartNs() const { return rawStartNs_; }
  std::uint64_t rawEpisode() const { return rawEpisode_; }
  // A qualified origin on an untainted reference: the 20-sample history, frozen for a raw episode it may vouch for.
  bool originQualified() const
  {
    return !latched() && (rawActive_ ? provenance_ != CarryProvenance::kNone && std::isfinite(frozen_)
                                     : std::isfinite(origin_.median()));
  }
  std::uint64_t originScan() const { return originScan_; }
  std::uint64_t originGeneration() const { return generation_; }
  double frozenResidual() const { return frozen_; }
  std::uint64_t rearmEvents() const { return rearm_; }
  std::uint32_t cause() const { return cause_; }
  std::uint32_t taint() const { return taint_; }
  std::uint32_t hard() const { return hard_; }

private:
  // No fixed-size vectorizable Eigen member, so the deque needs no aligned allocator.
  struct Pose
  {
    std::int64_t ns = 0;
    Eigen::Vector3d base = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  };

  HealthyScanWindow origin_{kCarryOriginCapacity, kCarryOriginMinimum};
  std::deque<Pose> poses_;
  std::deque<CarryImu> imu_;
  std::uint64_t generation_ = 0, originScan_ = 0, rawEpisode_ = 0, rearm_ = 0, lastSequence_ = 0;
  std::uint32_t cause_ = 0, taint_ = 0, hard_ = 0;
  bool rawActive_ = false, scanNegative_ = false, qualifiedSince_ = false, pendingSeen_ = false;
  std::int64_t rawStartNs_ = 0, runBeginNs_ = 0;
  std::size_t trustedRun_ = 0;
  double frozen_ = std::numeric_limits<double>::quiet_NaN();
  CarryProvenance provenance_ = CarryProvenance::kNone;
};

// The RW block of an RW-enabled snapshot, before settle(); the negative masks it relies on are the progress masks,
// which equal the carry latch while RW is enabled.
inline ReferenceHealthSnapshot
projectCarry(ReferenceHealthSnapshot value, const ConstrainedCarry & carry, const CarryVerdict & verdict)
{
  const bool origin = carry.originQualified() && value.lifecycle == 2 && value.trust != 3;
  const bool permitted = origin && value.kind == 1 && verdict.permitted;
  value.hasRw = true;
  value.rwContract = kCarryContract;
  value.presentMask |= kReferenceRwPresent;
  value.trueMask = (value.trueMask & ~kReferenceRwPresent) | (origin ? 16u : 0u) | (permitted ? 32u : 0u);
  value.hasRawEpisode = value.hasRawStart = carry.rawActive();
  value.rawEpisodeId = carry.rawActive() ? carry.rawEpisode() : 0;
  value.rawStartNs = carry.rawActive() ? carry.rawStartNs() : 0;
  value.hasRwOrigin = origin;
  value.originScan = origin ? carry.originScan() : 0;
  value.originReferenceGeneration = origin ? carry.originGeneration() : 0;
  value.hasFrozenResidual = carry.rawActive() && std::isfinite(carry.frozenResidual());
  value.frozenResidualMedian = value.hasFrozenResidual ? carry.frozenResidual() : 0.0;
  value.hasWithdrawalRearm = true;
  value.withdrawalRearmEvent = carry.rearmEvents();
  value.hasRwValidUntil = permitted;
  value.rwValidUntilNs = permitted ? verdict.validUntilNs : 0;
  value.wheelWitnessKind = permitted ? verdict.witnessKind : 0;
  value.hasWheelWitnessEpochs = permitted && verdict.witnessKind == kCarryWitnessFallback;
  value.lowerWheelWitnessEpoch = value.hasWheelWitnessEpochs ? static_cast<std::uint64_t>(verdict.lowerWitnessNs) : 0;
  value.upperWheelWitnessEpoch = value.hasWheelWitnessEpochs ? static_cast<std::uint64_t>(verdict.upperWitnessNs) : 0;
  return value;
}
}  // namespace fast_lio

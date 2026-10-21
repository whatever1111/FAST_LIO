// R1-OBS-1: observation only. RecoveryState remains the sole reference FSM.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "stationary_anchor_policy.hpp"

namespace fast_lio
{
inline constexpr std::uint32_t kReferenceImplemented = 7u;  // R1, RW and the read-only HR witness
inline constexpr std::uint32_t kReferenceR1Enabled = 5u;    // R1 and the HR witness projection, without RW
inline constexpr std::uint32_t kReferencePositiveKnown = 255u;
inline constexpr std::uint32_t kReferenceR1Present = 143u;  // bits 0, 1, 2, 3, 7
inline constexpr std::uint32_t kReferenceRwPresent = 48u;   // bit 4 qualified RW origin, bit 5 current carry
inline constexpr std::uint32_t kReferenceCauseKnown = 8191u;
inline constexpr std::uint32_t kReferenceTaintBits = 8188u;  // raw degraded and pure engulf alone are no taint
inline constexpr std::uint32_t kReferenceHardKnown = 255u;
inline constexpr std::int64_t kReferenceLeaseNs = 200000000;
inline constexpr std::int64_t kReferenceInputPeriodNs = 100000000;
inline constexpr std::uint32_t kReferenceRawDegraded = 1u;
inline constexpr std::uint32_t kReferencePureEngulf = 2u;
inline constexpr std::uint32_t kReferenceScanHole = 4u;
inline constexpr std::uint32_t kReferenceImuCoverage = 8u;
inline constexpr std::uint32_t kReferenceImuBridge = 16u;
inline constexpr std::uint32_t kReferenceLostBatch = 32u;
inline constexpr std::uint32_t kReferenceUnassessed = 64u;
inline constexpr std::uint32_t kReferenceMapReset = 128u;
inline constexpr std::uint32_t kReferenceSourceFault = 256u;
inline constexpr std::uint32_t kReferenceRunaway = 512u;
inline constexpr std::uint32_t kReferenceUnknownRewrite = 4096u;

// RW carry contract 3: odom flag bits 4/5 and twist-covariance cells 6/8/9, and the RW block of R1.
inline constexpr std::uint16_t kCarryContract = 3;
inline constexpr unsigned kCarryFlagConstrained = 16u;  // bit 4 NearFieldConstrained
inline constexpr unsigned kCarryFlagHardFailure = 32u;  // bit 5 HardFailure
inline constexpr std::int64_t kCarryRawBudgetNs = 8000000000;
inline constexpr std::int64_t kCarryWitnessNs = 1000000000;

inline std::int64_t referenceAdd(std::int64_t stamp, std::int64_t span)
{
  return stamp > std::numeric_limits<std::int64_t>::max() - span ? std::numeric_limits<std::int64_t>::max()
                                                                 : stamp + span;
}

inline std::int64_t referenceUntil(std::int64_t stamp)
{
  return referenceAdd(stamp, kReferenceLeaseNs);
}

// Cells 6 and 8 carry sensor-epoch seconds in a double, which cannot hold every nanosecond. The instant is floored
// to the microsecond and one more microsecond is taken off, so the decoded value is never later than the exact one
// R1 carries. Zero means none.
inline double carryCellSeconds(std::int64_t ns)
{
  return ns <= 1000 ? 0.0 : static_cast<double>(ns / 1000 - 1) * 1e-6;
}

inline std::int64_t carryCellNs(double seconds)
{
  return std::isfinite(seconds) && seconds > 0.0 && seconds < 9.2e9 ? std::llround(seconds * 1e6) * 1000 : 0;
}

// These are explicit wire encodings, not casts of the zero-based policy enums.
inline std::uint8_t referenceStartupWire(StartupReference phase)
{
  switch (phase) {
    case StartupReference::kCold:
      return 1;
    case StartupReference::kEstablished:
      return 2;
    case StartupReference::kExcluded:
      return 3;
  }
  return 0;
}

inline std::uint8_t referenceTrustWire(RecoveryTrust trust)
{
  switch (trust) {
    case RecoveryTrust::kLocalQualified:
      return 1;
    case RecoveryTrust::kPending:
      return 2;
    case RecoveryTrust::kUnanchored:
      return 3;
  }
  return 0;
}

struct ReferenceHealthSnapshot
{
  std::uint16_t schemaVersion = 1;
  std::uint8_t identityVersion = 1;
  bool hasSourceInstance = false;
  std::array<std::uint8_t, 16> sourceInstance{};
  std::string clockDomain;
  std::string frameId;
  std::string childFrameId;
  bool hasCoordinateEpoch = false;
  std::uint64_t coordinateEpoch = 0;
  std::array<std::uint8_t, 32> producerContractId{};
  std::array<std::uint8_t, 32> configId{};
  std::uint32_t implementedMask = kReferenceImplemented;
  std::uint32_t enabledPolicyMask = 0;
  bool hasSnapshotSequence = false;
  std::uint64_t snapshotSequence = 0;
  bool hasStateRevision = false;
  std::uint64_t stateRevision = 0;
  bool hasScanSequence = false;
  std::uint64_t scanSequence = 0;
  bool hasObservationId = false;
  std::array<std::uint8_t, 32> observationId{};
  std::uint8_t kind = 0;  // Unknown, Scan, StateChange, InputProgress, StateRepeat
  bool hasEvaluatedInput = false;
  std::int64_t evaluatedInputNs = 0;
  bool hasEffective = false;
  std::int64_t effectiveNs = 0;
  bool hasStateValidUntil = false;
  std::int64_t stateValidUntilNs = 0;
  bool hasScanBegin = false;
  std::int64_t scanBeginNs = 0;
  bool hasScanEnd = false;
  std::int64_t scanEndNs = 0;
  bool hasPoseBinding = false;
  std::array<double, 7> pose{};       // xyz, quaternion xyzw, double throughout
  std::uint8_t lifecycle = 0;         // Unknown, PreReference, Established
  std::uint8_t startupReference = 0;  // Unknown, Cold, Established, Excluded
  std::uint8_t trust = 0;             // Unknown, LocalQualified, Pending, Unanchored
  std::uint8_t local = 0;             // Unknown, Healthy, Degraded, Invalid
  std::uint32_t legacyFlags = 0;
  bool hasRawRegistrationDegraded = false;
  bool rawRegistrationDegraded = false;
  std::uint64_t referenceGeneration = 0;
  std::uint64_t recoveryEpisode = 0;
  bool hasQualifiedEventSequence = false;
  std::uint64_t qualifiedEventSequence = 0;
  bool hasOriginalStart = false;
  std::int64_t originalStartNs = 0;
  bool hasOriginalDeadline = false;
  std::int64_t originalDeadlineNs = 0;
  bool hasLastSupportedEnd = false;
  std::int64_t lastSupportedEndNs = 0;
  std::uint32_t presentMask = 0;
  std::uint32_t trueMask = 0;
  bool hasNegativeMasks = false;
  std::uint32_t causeMask = 0;
  std::uint32_t taintMask = 0;
  std::uint32_t hardMask = 0;
  std::uint8_t hrWitnessKind = 0;  // None, Normal, TrustedStaticHold, Constrained
  bool hasHrValidUntil = false;
  std::int64_t hrValidUntilNs = 0;
  // RW contract 3, present exactly when RW is enabled.
  bool hasRw = false;
  std::uint16_t rwContract = 0;
  bool hasRawEpisode = false;
  std::uint64_t rawEpisodeId = 0;
  bool hasRawStart = false;
  std::int64_t rawStartNs = 0;
  bool hasRwOrigin = false;
  std::uint64_t originScan = 0;
  std::uint64_t originReferenceGeneration = 0;
  bool hasFrozenResidual = false;
  double frozenResidualMedian = 0.0;
  bool hasWithdrawalRearm = false;
  std::uint64_t withdrawalRearmEvent = 0;
  bool hasRwValidUntil = false;
  std::int64_t rwValidUntilNs = 0;
  bool hasWheelWitnessEpochs = false;
  std::uint64_t lowerWheelWitnessEpoch = 0;  // sensor ns of the supporting leg sample
  std::uint64_t upperWheelWitnessEpoch = 0;
  std::uint8_t wheelWitnessKind = 0;  // None, CurrentWindow, Fallback
  bool hasSupport = false;            // support contract 1 is reserved, never asserted
};

struct ReferenceScanFacts
{
  bool completed = false;
  bool initialized = false;
  bool rawDegraded = false;
  bool lost = false;
  bool runaway = false;
  bool blocked = false;
  bool blindOrVerifying = false;
  bool levelling = false;
  bool staticHold = false;  // the existing anchorHealth verdict, never an inferred pin
  bool assessed = false;
};

inline bool referencePoseFinite(const std::array<double, 7> & pose)
{
  for (double value : pose)
    if (!std::isfinite(value))
      return false;
  double norm = 0.0;
  for (std::size_t i = 3; i < pose.size(); ++i)
    norm += pose[i] * pose[i];
  return std::abs(norm - 1.0) <= 1e-6;
}

inline ReferenceHealthSnapshot
projectReferenceHealth(ReferenceHealthSnapshot value, const RecoveryState & recovery, const ReferenceScanFacts & scan)
{
  value.startupReference = referenceStartupWire(recoveryStartupReference(recovery));
  value.lifecycle = value.startupReference == 1 ? 1 : value.startupReference == 2 ? 2 : 0;
  value.trust = referenceTrustWire(recovery.trust);
  value.recoveryEpisode = recovery.episode;
  value.hasOriginalStart = recovery.start_ns >= 0;
  value.originalStartNs = value.hasOriginalStart ? recovery.start_ns : 0;
  value.hasOriginalDeadline = recovery.deadline_ns >= 0;
  value.originalDeadlineNs = value.hasOriginalDeadline ? recovery.deadline_ns : 0;
  value.hasLastSupportedEnd = recovery.last_supported_end_ns >= 0;
  value.lastSupportedEndNs = value.hasLastSupportedEnd ? recovery.last_supported_end_ns : 0;
  value.hasRawRegistrationDegraded = scan.completed;
  value.rawRegistrationDegraded = scan.completed && scan.rawDegraded;
  const bool validPose = value.hasPoseBinding && referencePoseFinite(value.pose);
  const bool invalid = scan.lost || scan.runaway || scan.blocked || (scan.completed && !validPose);
  value.local = !scan.completed                                               ? 0
                : invalid                                                     ? 3
                : !scan.initialized || !scan.assessed                         ? 0
                : scan.rawDegraded || scan.blindOrVerifying || scan.levelling ? 2
                                                                              : 1;
  value.presentMask = kReferenceR1Present;
  value.trueMask = (value.lifecycle == 2 ? 1u : 0u) | (value.trust == 1 ? 2u : 0u) | (value.local == 1 ? 128u : 0u);
  const bool certificate = value.kind == 1 && scan.completed && validPose && value.hasObservationId &&
                           value.hasSourceInstance && value.hasCoordinateEpoch && value.hasScanEnd &&
                           value.lifecycle == 2 && value.trust == 1 && !invalid;
  const bool normal = certificate && value.local == 1;
  const bool held = certificate && scan.staticHold;
  value.trueMask |= (normal ? 8u : 0u) | (held ? 4u : 0u);
  value.hrWitnessKind = held ? 2 : normal ? 1 : 0;
  value.hasHrValidUntil = value.hrWitnessKind != 0;
  value.hrValidUntilNs = value.hasHrValidUntil ? referenceUntil(value.scanEndNs) : 0;
  return value;
}

// The RW block of an RW-enabled snapshot. Bit 4 is a qualified origin on an untainted reference; bit 5 is the
// strict carry permission of one scan, bounded by the scan lease, the raw budget, the recovery deadline and, for
// the fallback, its two leg witnesses. Without bit 5 no certificate field is set.
inline bool referenceCarryValid(const ReferenceHealthSnapshot & value)
{
  const bool origin = (value.trueMask & 16u) != 0;
  const bool carry = (value.trueMask & 32u) != 0;
  if (value.rwContract != kCarryContract || (value.presentMask & kReferenceRwPresent) != kReferenceRwPresent ||
      !value.hasWithdrawalRearm || value.hasRawEpisode != value.hasRawStart ||
      (value.hasFrozenResidual && !(std::isfinite(value.frozenResidualMedian) && value.frozenResidualMedian >= 0.0)) ||
      (origin && (!value.hasRwOrigin || value.lifecycle != 2 || value.taintMask != 0 || value.hardMask != 0)))
    return false;
  if (!carry)
    return !value.hasRwValidUntil && value.wheelWitnessKind == 0 && !value.hasWheelWitnessEpochs;
  const auto until = value.rwValidUntilNs;
  const bool witnesses =
    value.wheelWitnessKind == 1
      ? !value.hasWheelWitnessEpochs
      : value.wheelWitnessKind == 2 && value.hasWheelWitnessEpochs &&
          value.lowerWheelWitnessEpoch <= static_cast<std::uint64_t>(value.scanEndNs) &&
          value.upperWheelWitnessEpoch <= static_cast<std::uint64_t>(value.scanEndNs) &&
          until <= referenceAdd(static_cast<std::int64_t>(value.lowerWheelWitnessEpoch), kCarryWitnessNs) &&
          until <= referenceAdd(static_cast<std::int64_t>(value.upperWheelWitnessEpoch), kCarryWitnessNs);
  return witnesses && value.kind == 1 && origin && value.hasRawStart && value.hasFrozenResidual &&
         value.hasRwValidUntil && value.hasScanEnd && (value.trust == 1 || value.trust == 2) &&
         (value.rawRegistrationDegraded || value.trust == 2) && (value.legacyFlags & 6u) == 0 &&
         value.rawStartNs <= value.scanEndNs && until >= value.scanEndNs && until <= referenceUntil(value.scanEndNs) &&
         until <= referenceAdd(value.rawStartNs, kCarryRawBudgetNs) &&
         (value.trust != 2 || !value.hasOriginalDeadline || until < value.originalDeadlineNs);
}

inline bool referenceHealthEncodingValid(const ReferenceHealthSnapshot & value)
{
  const bool rw = (value.enabledPolicyMask & 2u) != 0;
  if (value.schemaVersion != 1 || value.identityVersion != 1 || value.kind < 1 || value.kind > 4 ||
      value.lifecycle > 2 || value.startupReference > 3 || value.trust > 3 || value.local > 3 ||
      value.hrWitnessKind > 3 || (value.legacyFlags & ~15u) != 0 || (value.implementedMask & ~15u) != 0 ||
      (value.enabledPolicyMask & ~value.implementedMask) != 0 ||
      (value.presentMask & ~(kReferenceR1Present | (rw ? kReferenceRwPresent : 0u))) != 0 ||
      (value.trueMask & ~value.presentMask) != 0 || (value.causeMask & ~kReferenceCauseKnown) != 0 ||
      (value.taintMask & ~value.causeMask) != 0 || (value.hardMask & ~kReferenceHardKnown) != 0 || value.hasRw != rw ||
      value.hasSupport || (rw && !referenceCarryValid(value)))
    return false;
  const auto lifecycle = value.startupReference == 1 ? 1 : value.startupReference == 2 ? 2 : 0;
  if (value.lifecycle != lifecycle || ((value.trueMask & 1u) != 0) != (value.lifecycle == 2) ||
      ((value.trueMask & 2u) != 0) != (value.trust == 1) || ((value.trueMask & 128u) != 0) != (value.local == 1))
    return false;
  if (value.kind != 1 &&
      (value.hasPoseBinding || value.hasRawRegistrationDegraded || value.local != 0 || value.hrWitnessKind != 0))
    return false;
  if (value.kind == 1 && (!value.hasPoseBinding || !referencePoseFinite(value.pose)))
    return false;
  if ((value.legacyFlags & 2u) != 0 && value.trust != 1)
    return false;
  if (value.kind == 1 && (((value.legacyFlags & 1u) != 0) != (value.rawRegistrationDegraded || value.trust == 3) ||
                          (value.rawRegistrationDegraded && value.local == 1)))
    return false;
  if ((value.trueMask & 12u) != 0 && (value.kind != 1 || value.lifecycle != 2 || value.trust != 1 || value.local == 3 ||
                                      value.hardMask != 0 || (value.legacyFlags & 4u) != 0))
    return false;
  // Check both witness bits, including when held wins the wire-kind priority.
  if (((value.trueMask & 8u) != 0 && (value.local != 1 || (value.legacyFlags & 8u) != 0)) ||
      ((value.trueMask & 4u) != 0 && (value.legacyFlags & 2u) == 0))
    return false;
  return value.hrWitnessKind == ((value.trueMask & 4u) != 0 ? 2 : (value.trueMask & 8u) != 0 ? 1 : 0);
}

// Only observation order, negative-event accounting and real qualified receipts live here.
// No method changes RecoveryState or any estimator/map/legacy-output decision.
class ReferenceHealthProgress
{
public:
  // `keep` holds the masks while RW has not seen a legal recovery: its latch outlives the unsupported episode,
  // and R1 reports the same OR until rearm().
  void beginScan(const RecoveryState & recovery, bool keep = false)
  {
    // OR is scoped to an actual unsupported episode. Outside one, the next scan
    // supplies its own current evidence; initialization has no pose certificate to carry.
    if (recovery.trust == RecoveryTrust::kLocalQualified && !keep)
      causeMask_ = taintMask_ = hardMask_ = 0;
  }
  void negative(std::uint32_t cause, std::uint32_t hard = 0)
  {
    causeMask_ |= cause;
    taintMask_ |= cause & kReferenceTaintBits;
    hardMask_ |= hard;
    advance(negativeSequence_);
  }

  void qualified(bool keep = false)
  {
    advance(qualifiedSequence_);
    if (!keep)
      causeMask_ = taintMask_ = hardMask_ = 0;
  }

  void rearm() { causeMask_ = taintMask_ = hardMask_ = 0; }

  bool stateChanged(const ReferenceHealthSnapshot & value) const
  {
    return !hasPrevious_ || negativeSequence_ != settledNegative_ || value.trust != previous_.trust ||
           value.lifecycle != previous_.lifecycle || value.startupReference != previous_.startupReference ||
           value.referenceGeneration != previous_.referenceGeneration ||
           value.recoveryEpisode != previous_.recoveryEpisode ||
           value.hasCoordinateEpoch != previous_.hasCoordinateEpoch ||
           value.coordinateEpoch != previous_.coordinateEpoch ||
           value.hasSourceInstance != previous_.hasSourceInstance || value.sourceInstance != previous_.sourceInstance ||
           value.enabledPolicyMask != previous_.enabledPolicyMask || previous_.taintMask != taintMask_ ||
           previous_.hardMask != (hardMask_ | (value.trust == 3 ? 2u : 0u));
  }

  ReferenceHealthSnapshot settle(ReferenceHealthSnapshot value)
  {
    value.hasNegativeMasks = true;
    value.causeMask = causeMask_;
    value.taintMask = taintMask_;
    value.hardMask = hardMask_ | (value.trust == 3 ? 2u : 0u);
    const bool changed = stateChanged(value);
    advance(sequence_);
    if (changed)
      advance(revision_);
    value.hasSnapshotSequence = value.hasStateRevision = value.hasQualifiedEventSequence = valid_;
    value.snapshotSequence = valid_ ? sequence_ : 0;
    value.stateRevision = valid_ ? revision_ : 0;
    value.qualifiedEventSequence = valid_ ? qualifiedSequence_ : 0;
    if (value.kind != 1 && changed)
      value.kind = 2;
    if (!valid_ || value.hardMask != 0 || value.taintMask != 0) {
      value.trueMask &= ~12u;
      value.hrWitnessKind = 0;
      value.hasHrValidUntil = false;
    }
    if (!valid_) {
      value.trueMask = 0;
      value.hrWitnessKind = 0;
      value.hasHrValidUntil = false;
    }
    previous_ = value;
    hasPrevious_ = true;
    settledNegative_ = negativeSequence_;
    if (!value.hasPoseBinding) {
      lastState_ = value;
      hasState_ = true;
    }
    return value;
  }

  bool inputDue(const ReferenceHealthSnapshot & value) const
  {
    if (stateChanged(value) || !hasInput_)
      return true;
    return value.hasEvaluatedInput && value.evaluatedInputNs > lastInputNs_ &&
           static_cast<std::uint64_t>(value.evaluatedInputNs) - static_cast<std::uint64_t>(lastInputNs_) >=
             static_cast<std::uint64_t>(kReferenceInputPeriodNs);
  }
  void noteInput(std::int64_t stamp)
  {
    hasInput_ = true;
    lastInputNs_ = stamp;
  }
  ReferenceHealthSnapshot repeat() const
  {
    if (!hasState_ || lastState_.stateRevision != previous_.stateRevision)
      return {};
    auto value = lastState_;
    value.kind = 4;
    return value;  // same sequence, effective time and lease; receipt must not rejuvenate it
  }

private:
  void advance(std::uint64_t & counter)
  {
    if (counter == std::numeric_limits<std::uint64_t>::max())
      valid_ = false;
    else
      ++counter;
  }
  bool valid_ = true;
  bool hasPrevious_ = false;
  bool hasInput_ = false;
  bool hasState_ = false;
  std::uint64_t sequence_ = 0, revision_ = 0, qualifiedSequence_ = 0;
  std::uint64_t negativeSequence_ = 0, settledNegative_ = 0;
  std::int64_t lastInputNs_ = 0;
  std::uint32_t causeMask_ = 0, taintMask_ = 0, hardMask_ = 0;
  ReferenceHealthSnapshot previous_;
  ReferenceHealthSnapshot lastState_;
};

// A numeric projection keeps AnchorScanRecord trivially copyable for existing real-node fixtures.
struct ReferenceHealthTrace
{
  bool present = false;
  std::uint64_t snapshotSequence = 0, stateRevision = 0, scanSequence = 0, qualifiedSequence = 0;
  std::uint8_t kind = 0, lifecycle = 0, startup = 0, trust = 0, local = 0, hrKind = 0;
  std::uint32_t legacyFlags = 0, presentMask = 0, trueMask = 0, cause = 0, taint = 0, hard = 0;
  bool hasPose = false;
  std::int64_t evaluatedNs = 0, effectiveNs = 0, stateUntilNs = 0, hrUntilNs = 0;
  bool rwPresent = false;
};

inline ReferenceHealthTrace referenceHealthTrace(const ReferenceHealthSnapshot & value)
{
  return {value.enabledPolicyMask != 0 && value.hasSnapshotSequence,
          value.snapshotSequence,
          value.stateRevision,
          value.scanSequence,
          value.qualifiedEventSequence,
          value.kind,
          value.lifecycle,
          value.startupReference,
          value.trust,
          value.local,
          value.hrWitnessKind,
          value.legacyFlags,
          value.presentMask,
          value.trueMask,
          value.causeMask,
          value.taintMask,
          value.hardMask,
          value.hasPoseBinding,
          value.evaluatedInputNs,
          value.effectiveNs,
          value.stateValidUntilNs,
          value.hrValidUntilNs,
          value.enabledPolicyMask != 0 && value.hasSnapshotSequence && value.hasRw};
}
}  // namespace fast_lio

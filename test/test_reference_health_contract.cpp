#include <gtest/gtest.h>
#include <limits>

#include "reference_health_contract.hpp"
#include "reference_health_literals.hpp"

namespace
{
fast_lio::ReferenceHealthSnapshot scanValue()
{
  fast_lio::ReferenceHealthSnapshot value;
  value.kind = 1;
  value.enabledPolicyMask = 5;
  value.hasSourceInstance = value.hasCoordinateEpoch = value.hasObservationId = true;
  value.hasScanSequence = value.hasScanBegin = value.hasScanEnd = value.hasPoseBinding = true;
  value.pose = {{1.25, -2.5, 0.0, 0.0, 0.0, 0.0, 1.0}};
  value.hasEvaluatedInput = value.hasEffective = value.hasStateValidUntil = true;
  value.scanBeginNs = 1000000000;
  value.scanEndNs = value.evaluatedInputNs = value.effectiveNs = 1100000000;
  value.stateValidUntilNs = 1300000000;
  value.scanSequence = 1;
  return value;
}
fast_lio::RecoveryState established()
{
  fast_lio::RecoveryState recovery;
  fast_lio::configureRecoveryStartup(&recovery, true);
  fast_lio::commitInitialReference(&recovery);
  return recovery;
}
fast_lio::ReferenceScanFacts healthy()
{
  fast_lio::ReferenceScanFacts facts;
  facts.completed = facts.initialized = facts.assessed = true;
  return facts;
}
}  // namespace

TEST(ReferenceHealthContract, IndependentWireLiterals)
{
  for (const auto & literal : reference_health_literals::kVectors) {
    SCOPED_TRACE(literal.name);
    auto value = scanValue();
    value.kind = literal.kind;
    value.startupReference = literal.startup;
    value.lifecycle = literal.lifecycle;
    value.trust = literal.trust;
    value.local = literal.local;
    value.hrWitnessKind = literal.hr;
    value.legacyFlags = literal.flags;
    value.presentMask = literal.present;
    value.trueMask = literal.truth;
    value.hasPoseBinding = literal.pose;
    value.hasRawRegistrationDegraded = literal.pose;
    value.rawRegistrationDegraded = (literal.flags & 1u) != 0 && literal.trust != 3;
    EXPECT_EQ(fast_lio::referenceHealthEncodingValid(value), literal.valid);
  }
}

TEST(ReferenceHealthContract, StartupIsReadOnlyAndNeverInferredFromTrustOrGeneration)
{
  fast_lio::RecoveryState recovery;
  auto value = scanValue();
  value.kind = 2;
  value.hasPoseBinding = false;
  auto facts = fast_lio::ReferenceScanFacts{};
  EXPECT_EQ(fast_lio::projectReferenceHealth(value, recovery, facts).lifecycle, 0);
  fast_lio::configureRecoveryStartup(&recovery, true);
  auto result = fast_lio::projectReferenceHealth(value, recovery, facts);
  EXPECT_EQ(result.startupReference, 1);
  EXPECT_EQ(result.lifecycle, 1);
  EXPECT_EQ(result.trueMask, 2u);
  EXPECT_FALSE(recovery.reference_initialized_once);
  fast_lio::commitInitialReference(&recovery);
  result = fast_lio::projectReferenceHealth(value, recovery, facts);
  EXPECT_EQ(result.lifecycle, 2);
  EXPECT_EQ(result.local, 0);
  EXPECT_EQ(result.trueMask, 3u);
  EXPECT_EQ(result.hrWitnessKind, 0);
  EXPECT_EQ(result.referenceGeneration, 0u);
  fast_lio::RecoveryParams params;
  params.total_ns = 10000000000LL;
  fast_lio::resetRecoveryReference(&recovery, 2000000000, params);
  result = fast_lio::projectReferenceHealth(value, recovery, facts);
  EXPECT_EQ(result.startupReference, 2);
  EXPECT_EQ(result.trust, 3);
  EXPECT_TRUE(recovery.reference_initialized_once);
}

TEST(ReferenceHealthContract, PendingRawClearAndUnanchoredHealthyNeverCreateNormal)
{
  auto recovery = established();
  auto facts = healthy();
  auto value = scanValue();
  for (auto trust : {fast_lio::RecoveryTrust::kPending, fast_lio::RecoveryTrust::kUnanchored}) {
    recovery.trust = trust;
    auto result = fast_lio::projectReferenceHealth(value, recovery, facts);
    EXPECT_EQ(result.local, 1);
    EXPECT_EQ(result.trueMask, 129u);
    EXPECT_EQ(result.hrWitnessKind, 0);
  }
}

TEST(ReferenceHealthContract, NormalAndGenuineHoldHaveOnlyTheR1WitnessBits)
{
  auto recovery = established();
  auto facts = healthy();
  const auto normal = fast_lio::projectReferenceHealth(scanValue(), recovery, facts);
  EXPECT_EQ(normal.trueMask, 139u);
  EXPECT_EQ(normal.hrWitnessKind, 1);
  EXPECT_EQ(normal.hrValidUntilNs, 1300000000);
  facts.rawDegraded = facts.staticHold = true;
  const auto held = fast_lio::projectReferenceHealth(scanValue(), recovery, facts);
  EXPECT_EQ(held.trueMask, 7u);
  EXPECT_EQ(held.hrWitnessKind, 2);
  EXPECT_EQ(held.presentMask, 143u);
  EXPECT_FALSE(held.hasRw);
  EXPECT_FALSE(held.hasSupport);
}

TEST(ReferenceHealthContract, LateFaultsOverrideEarlierHealthyMeasurement)
{
  for (int fault = 0; fault < 4; ++fault) {
    auto value = scanValue();
    auto facts = healthy();
    if (fault == 0)
      facts.lost = true;
    if (fault == 1)
      facts.runaway = true;
    if (fault == 2)
      facts.blocked = true;
    if (fault == 3)
      value.pose[0] = std::numeric_limits<double>::quiet_NaN();
    const auto result = fast_lio::projectReferenceHealth(value, established(), facts);
    EXPECT_EQ(result.local, 3);
    EXPECT_EQ(result.trueMask & 12u, 0u);
    EXPECT_EQ(result.hrWitnessKind, 0);
  }
}

TEST(ReferenceHealthContract, OnlyRealQualifiedReceiptIncrementsSequence)
{
  fast_lio::ReferenceHealthProgress progress;
  auto value = fast_lio::projectReferenceHealth(scanValue(), established(), healthy());
  auto first = progress.settle(value);
  EXPECT_EQ(first.qualifiedEventSequence, 0u);
  EXPECT_EQ(progress.settle(value).qualifiedEventSequence, 0u);
  progress.negative(fast_lio::kReferenceImuCoverage);
  auto negative = progress.settle(value);
  EXPECT_GT(negative.stateRevision, first.stateRevision);
  EXPECT_EQ(negative.taintMask, 8u);
  progress.qualified();
  auto qualified = progress.settle(value);
  EXPECT_EQ(qualified.qualifiedEventSequence, 1u);
  EXPECT_EQ(qualified.taintMask, 0u);
  EXPECT_EQ(progress.settle(value).qualifiedEventSequence, 1u);
}

TEST(ReferenceHealthContract, InputProgressLeasesDoNotRefreshScanWitnessAndRepeatsDoNotRenew)
{
  fast_lio::ReferenceHealthProgress progress;
  auto value = scanValue();
  value.kind = 3;
  value.hasPoseBinding = false;
  value = fast_lio::projectReferenceHealth(value, established(), {});
  const auto first = progress.settle(value);
  progress.noteInput(first.evaluatedInputNs);
  const auto repeat = progress.repeat();
  EXPECT_EQ(repeat.kind, 4);
  EXPECT_EQ(repeat.snapshotSequence, first.snapshotSequence);
  EXPECT_EQ(repeat.stateValidUntilNs, first.stateValidUntilNs);
  EXPECT_EQ(repeat.local, 0);
  EXPECT_FALSE(repeat.hasHrValidUntil);
  value.evaluatedInputNs += 99999999;
  EXPECT_FALSE(progress.inputDue(value));
  ++value.evaluatedInputNs;
  EXPECT_TRUE(progress.inputDue(value));
  const auto next = progress.settle(value);
  EXPECT_GT(next.snapshotSequence, first.snapshotSequence);
  EXPECT_EQ(next.stateRevision, first.stateRevision);
}

TEST(ReferenceHealthContract, IntegerAndMaskBoundsFailClosed)
{
  EXPECT_EQ(fast_lio::referenceUntil(std::numeric_limits<std::int64_t>::max()),
            std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(fast_lio::referenceStartupWire(static_cast<fast_lio::StartupReference>(255)), 0);
  auto value = fast_lio::projectReferenceHealth(scanValue(), established(), healthy());
  value.scanSequence = value.referenceGeneration = std::numeric_limits<std::uint64_t>::max();
  EXPECT_TRUE(fast_lio::referenceHealthEncodingValid(value));
  value.enabledPolicyMask = 8;
  EXPECT_FALSE(fast_lio::referenceHealthEncodingValid(value));
  value.enabledPolicyMask = 5;
  value.causeMask = 8192;
  EXPECT_FALSE(fast_lio::referenceHealthEncodingValid(value));
}

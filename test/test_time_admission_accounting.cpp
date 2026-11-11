#include <gtest/gtest.h>
#include <stdexcept>

#include "time_admission_accounting.hpp"

namespace
{
using fast_lio::AdmissionBucket;
using fast_lio::AdmissionCount;
using fast_lio::AdmissionMapKind;
using fast_lio::TimeAdmissionAccounting;
using fast_lio::TimeAdmissionSnapshot;

TEST(TimeAdmissionAccounting, CheckedSaturationKeepsDiagnosticFailureOutOfEstimatorControl)
{
  TimeAdmissionAccounting a{};
  a.add(a.local_event_seq, 0);
  EXPECT_EQ(a.local_event_seq, 0u);
  a.add(a.local_event_seq, UINT64_MAX - 1);
  a.add(a.local_event_seq);
  EXPECT_TRUE(a.exact);
  a.add(a.local_event_seq);
  EXPECT_EQ(a.local_event_seq, UINT64_MAX);
  EXPECT_TRUE(a.overflow);
  EXPECT_FALSE(a.exact);
  EXPECT_FALSE(a.accounting_complete);
  a.record(AdmissionBucket::kEnqueued, 7);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEnqueued)].points, 7u);
  TimeAdmissionAccounting b{};
  AdmissionCount count{};
  b.count(count, UINT64_MAX, UINT64_MAX);
  EXPECT_TRUE(b.exact);
  b.count(count, 1);
  EXPECT_EQ(count.messages, UINT64_MAX);
  EXPECT_EQ(count.points, UINT64_MAX);
  EXPECT_TRUE(b.overflow);
  EXPECT_FALSE(b.exact || b.accounting_complete);
}

TEST(TimeAdmissionAccounting, SeparateUnitsCloseOnlyLocalPartitions)
{
  TimeAdmissionSnapshot s{};
  auto & a = s.accounting;
  a.enabled = true;
  a.record(AdmissionBucket::kReceived, 100);
  a.record(AdmissionBucket::kReceived, 60);
  a.record(AdmissionBucket::kLateRejected, 60);
  a.record(AdmissionBucket::kEnqueuedDeclared, 100);
  a.record(AdmissionBucket::kEnqueued, 20);  // preprocessing selected a different population
  s.queue_pending = {1, 20};
  s.pending_known = true;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_TRUE(a.accounting_complete);
  EXPECT_TRUE(s.callback_partition);
  EXPECT_TRUE(s.queue_partition);
  EXPECT_EQ(s.callback_unresolved_declared.messages, 0u);
  EXPECT_EQ(fast_lio::admissionPointUnit(AdmissionBucket::kReceived), fast_lio::AdmissionUnit::kDeclaredInput);
  EXPECT_EQ(fast_lio::admissionPointUnit(AdmissionBucket::kEnqueued), fast_lio::AdmissionUnit::kEffective);
  EXPECT_FALSE(s.source_identity_known);  // equal totals cannot establish unique physical identities
  EXPECT_FALSE(s.config_identity_known);
  EXPECT_FALSE(s.reader_eof_verified);
  s.queue_pending.points = 19;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_FALSE(s.queue_partition);
  EXPECT_FALSE(a.accounting_complete);
}

TEST(TimeAdmissionAccounting, UnknownAndInterruptedStagesRemainVisibleWithoutInventedZeroEvidence)
{
  TimeAdmissionSnapshot s{};
  auto & a = s.accounting;
  a.enabled = true;
  fast_lio::ProcessAudit audit{};
  audit.entered = true;
  audit.input_points = 9;
  audit.fc_before_known = true;
  audit.fc_before_bits = fast_lio::admissionDoubleBits(-1.0);
  a.process(audit);
  EXPECT_EQ(a.process_exceptions.messages, 1u);
  EXPECT_EQ(a.history_unknown.points, 9u);
  EXPECT_FALSE(a.last_process.history_evaluated);
  EXPECT_FALSE(a.last_process.fc_after_known);
  EXPECT_FALSE(a.accounting_complete);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kDeskew)].messages, 0u);
  // A wait is an observation, never a second terminal record for the queued cloud.
  a.add(a.wait_no_imu);
  a.add(a.wait_no_imu);
  EXPECT_EQ(a.wait_no_imu, 2u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEmptyImu)].messages, 0u);
}

TEST(TimeAdmissionAccounting, IsolatedFutureCloudHasLeadBucketWithoutAnEpochFault)
{
  TimeAdmissionSnapshot s{};
  auto & a = s.accounting;
  a.enabled = true;
  fast_lio::InputEpochGuard epoch;
  fast_lio::InputLeadFilter lead;
  double previous = -1.0;
  for (const double stamp : {10.0, 25.0, 10.1}) {
    a.record(AdmissionBucket::kReceived, 4);
    const auto input = fast_lio::classifyAdmissionInput(epoch, lead, previous, stamp, !epoch.faulted(), 1.0, 10.0);
    if (input == fast_lio::AdmissionInputResult::kAccepted) {
      previous = stamp;
      a.record(AdmissionBucket::kEnqueuedDeclared, 4);
      a.record(AdmissionBucket::kEnqueued, 4);
      a.count(s.queue_pending, 4);
    } else {
      a.recordInputRejection(input, 4);
    }
  }
  s.pending_known = true;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_FALSE(epoch.faulted());
  EXPECT_EQ(previous, 10.1);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kLeadRejected)].messages, 1u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kLeadRejected)].points, 4u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEpochRejected)].messages, 0u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEnqueued)].messages, 2u);
  EXPECT_TRUE(s.callback_partition && a.accounting_complete);
  EXPECT_EQ(fast_lio::admissionPointUnit(AdmissionBucket::kLeadRejected), fast_lio::AdmissionUnit::kDeclaredInput);
}

TEST(TimeAdmissionAccounting, TransactionFaultAndLatchedEpochKeepDistinctRejectionReasons)
{
  TimeAdmissionAccounting a{};
  a.enabled = true;
  fast_lio::InputEpochGuard epoch;
  fast_lio::InputLeadFilter lead;
  const auto transaction = fast_lio::classifyAdmissionInput(epoch, lead, 10.0, 10.1, false, 1.0, 10.0);
  EXPECT_EQ(transaction, fast_lio::AdmissionInputResult::kTransactionRejected);
  a.recordInputRejection(transaction, 3);
  EXPECT_FALSE(epoch.faulted());
  const auto rollback = fast_lio::classifyAdmissionInput(epoch, lead, 10.0, 8.0, true, 1.0, 10.0);
  EXPECT_EQ(rollback, fast_lio::AdmissionInputResult::kEpochRejected);
  a.recordInputRejection(rollback, 5);
  const auto latched = fast_lio::classifyAdmissionInput(epoch, lead, 10.0, 10.1, false, 1.0, 10.0);
  EXPECT_EQ(latched, fast_lio::AdmissionInputResult::kEpochRejected);
  a.recordInputRejection(latched, 7);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kTransactionRejected)].points, 3u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEpochRejected)].messages, 2u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kEpochRejected)].points, 12u);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kLeadRejected)].messages, 0u);
}

TEST(TimeAdmissionAccounting, NormalInitializationEarlyReturnHasKnownUnreachedHistory)
{
  TimeAdmissionSnapshot s{};
  auto & a = s.accounting;
  a.enabled = true;
  for (const auto bucket : {AdmissionBucket::kReceived,
                            AdmissionBucket::kEnqueuedDeclared,
                            AdmissionBucket::kEnqueued,
                            AdmissionBucket::kBorrowed})
    a.record(bucket, 9);
  fast_lio::ProcessAudit audit{};
  fast_lio::beginProcessAudit(audit, 9, -1.0);
  fast_lio::finishProcessAudit(
    audit, {fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kInitializationAccumulated, false}, 0.05);
  a.process(audit);
  s.pending_known = true;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_EQ(a.history_not_reached.messages, 1u);
  EXPECT_EQ(a.history_not_reached.points, 9u);
  EXPECT_EQ(a.history_unknown.messages, 0u);
  EXPECT_EQ(a.process_exceptions.messages, 0u);
  EXPECT_TRUE(s.history_partition && a.accounting_complete);
}

TEST(TimeAdmissionAccounting, CallbackExceptionsAfterEnqueueStillMakeAccountingIncomplete)
{
  TimeAdmissionAccounting a{};
  try {
    fast_lio::AdmissionCallbackScope scope{&a};
    a.record(AdmissionBucket::kReceived, 17);
    a.record(AdmissionBucket::kEnqueuedDeclared, 17);
    a.record(AdmissionBucket::kEnqueued, 4);
    scope.resolved = true;
    throw std::runtime_error("existing callback failure after enqueue");
  } catch (const std::runtime_error &) {}
  EXPECT_EQ(a.callback_exceptions, 1u);
  EXPECT_FALSE(a.accounting_complete);
}

TEST(TimeAdmissionAccounting, InterruptedHistoryCopyDoesNotInventEraseOrRetention)
{
  TimeAdmissionAccounting a{};
  fast_lio::ProcessAudit audit{};
  audit.entered = audit.history_evaluated = true;
  audit.input_points = 3;
  audit.history_action = fast_lio::ScanHistoryAction::kDiscardEarly;
  a.process(audit);
  EXPECT_EQ(a.process_exceptions.messages, 1u);
  EXPECT_EQ(a.history_unknown.points, 3u);
  EXPECT_EQ(a.history_mixed.messages, 0u);
  EXPECT_EQ(a.history_erased, 0u);
  EXPECT_EQ(a.history_retained, 0u);
  EXPECT_FALSE(a.last_process.history_erase_completed);
  EXPECT_FALSE(a.accounting_complete);
}

TEST(TimeAdmissionAccounting, EqualTotalsWithRepeatedIdentityCannotEstablishLineage)
{
  const std::array<unsigned, 2> complete_fixture{7, 9}, duplicate_fixture{7, 7};
  EXPECT_NE(complete_fixture, duplicate_fixture);
  const auto observe = [](const auto & fixture) {
    TimeAdmissionSnapshot s{};
    s.pending_known = true;
    for (std::size_t i = 0; i < fixture.size(); ++i) {
      s.accounting.record(AdmissionBucket::kReceived, 4);
      s.accounting.record(AdmissionBucket::kEnqueuedDeclared, 4);
      s.accounting.record(AdmissionBucket::kEnqueued, 2);
      s.accounting.count(s.queue_pending, 2);
    }
    fast_lio::checkAdmissionSnapshot(s);
    return s;
  };
  const auto complete = observe(complete_fixture), duplicate = observe(duplicate_fixture);
  EXPECT_TRUE(complete.accounting.accounting_complete && duplicate.accounting.accounting_complete);
  EXPECT_EQ(complete.queue_pending.points, duplicate.queue_pending.points);
  EXPECT_FALSE(complete.source_identity_known || duplicate.source_identity_known);
  EXPECT_FALSE(complete.reader_eof_verified || duplicate.reader_eof_verified);
}

TEST(TimeAdmissionAccounting, HistoryPartitionsDoNotCountCoverageOrDeskewAsAnotherLoss)
{
  TimeAdmissionAccounting a{};
  fast_lio::ProcessAudit audit{};
  audit.entered = audit.completed = audit.history_evaluated = audit.history_erase_completed = true;
  audit.input_points = 5;
  audit.history_action = fast_lio::ScanHistoryAction::kDiscardEarly;
  audit.history_erased_points = 2;
  audit.history_retained_points = 3;
  audit.reason = fast_lio::ImuProcessReason::kCoverageGap;
  audit.disposition = fast_lio::ImuDisposition::kCommitted;
  a.process(audit);
  EXPECT_EQ(a.history_mixed.points, 5u);
  EXPECT_EQ(a.history_erased, 2u);
  EXPECT_EQ(a.history_retained, 3u);
  EXPECT_FALSE(a.last_process.deskew_completed);
  EXPECT_EQ(a.process_outcomes[static_cast<std::size_t>(audit.reason)].points, 5u);
}

TEST(TimeAdmissionAccounting, AsyncSubmissionCompletesOnlyAtOwnerJoinAndDelayedResetClosesPoints)
{
  TimeAdmissionSnapshot s{};
  auto & a = s.accounting;
  a.enabled = true;
  s.pending_known = true;
  a.count(a.staged, 12, 3);
  a.count(a.released, 5);
  a.count(a.reset_discarded, 3);
  s.delayed_pending = {1, 4};
  a.submit(AdmissionMapKind::kIncrementalDownsampled, 3);
  a.submit(AdmissionMapKind::kIncrementalDirect, 2);
  a.async_inflight = true;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_TRUE(s.map_partition);
  EXPECT_EQ(a.map[static_cast<std::size_t>(AdmissionMapKind::kIncrementalDownsampled)].completed, 0u);
  a.joined();
  a.joined();  // an idempotent stop does not complete the same operation twice
  EXPECT_EQ(a.async_joins, 1u);
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_TRUE(s.map_partition);
  EXPECT_EQ(a.map[static_cast<std::size_t>(AdmissionMapKind::kIncrementalDirect)].completed, 1u);
  s.delayed_pending.points = 5;
  fast_lio::checkAdmissionSnapshot(s);
  EXPECT_FALSE(s.map_partition);
  EXPECT_FALSE(a.accounting_complete);
}

TEST(TimeAdmissionAccounting, BootstrapSettlementIsIndependentOfProcessAndInputCursor)
{
  TimeAdmissionAccounting a{};
  a.record(AdmissionBucket::kBootstrap, 7);
  EXPECT_EQ(a.counts[static_cast<std::size_t>(AdmissionBucket::kProcessEntered)].messages, 0u);
  EXPECT_EQ(a.process_outcomes[static_cast<std::size_t>(fast_lio::ImuProcessReason::kInitializationComplete)].messages,
            0u);
  EXPECT_FALSE(a.bootstrap_fc_known);
  EXPECT_FALSE(a.last_process.fc_after_known);
}
}  // namespace

#include <deque>
#include <gtest/gtest.h>
#include <limits>

#include "imu_backlog_policy.hpp"
#include "imu_consumption_policy.hpp"
#include "scan_time_policy.hpp"

namespace
{
struct Sample
{
  int identity;
  double stamp;
  bool valid = true;
  bool operator==(const Sample & other) const { return identity == other.identity; }
};
using Queue = std::deque<Sample>;
using Transaction = fast_lio::ImuFifoTransaction<Sample>;
using fast_lio::ImuConfirmation;
using fast_lio::ImuDisposition;
const auto stamp = [](const Sample & sample) {
  return sample.stamp;
};
const auto valid = [](const Sample & sample) {
  return sample.valid;
};
}  // namespace

TEST(ImuConsumptionPolicy, InclusivePrefixPreservesDistinctSamplesWithEqualStamps)
{
  Queue queue{{1, 1.0}, {2, 1.0}, {3, 1.1}};
  Transaction transaction;
  const auto count = fast_lio::imuPrefixThrough(queue, 1.0, stamp);
  ASSERT_EQ(count, 2u);
  const auto token = transaction.borrow(queue, count);
  ASSERT_EQ(transaction.settle(queue, token, ImuDisposition::kUncommitted), ImuConfirmation::kRetained);
  ASSERT_EQ(queue.size(), 3u);
  EXPECT_EQ(queue[0].identity, 1);
  EXPECT_EQ(queue[1].identity, 2);
}

TEST(ImuConsumptionPolicy, PendingPrefixIsVisibleBeforeTheNextEmptyWindowDecision)
{
  Queue queue{{1, 1.0}, {2, 1.2}};
  Transaction transaction;
  const auto rejected = transaction.borrow(queue, 1);
  transaction.settle(queue, rejected, ImuDisposition::kUncommitted);
  EXPECT_EQ(fast_lio::imuPrefixThrough(queue, 1.1, stamp), 1u);
  const auto successor = transaction.borrow(queue, 1);
  EXPECT_GT(successor.sequence, rejected.sequence);
  EXPECT_EQ(successor.prefix.front().identity, rejected.prefix.front().identity);
  EXPECT_EQ(transaction.settle(queue, successor, ImuDisposition::kCommitted), ImuConfirmation::kReleased);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_EQ(queue.front().identity, 2);
}

TEST(ImuConsumptionPolicy, ConsecutiveRejectionsThenAppendCommitOnlyTheSamePrefixOnce)
{
  Queue queue{{1, 1.0}, {2, 1.01}};
  Transaction transaction;
  for (int i = 0; i < 3; ++i) {
    auto token = transaction.borrow(queue, queue.size());
    EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kUncommitted), ImuConfirmation::kRetained);
    queue.push_back({3 + i, 1.02 + i * 0.01});
  }
  const auto token = transaction.borrow(queue, 4);
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kReleased);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_EQ(queue.front().identity, 5);
  EXPECT_EQ(transaction.duplicateConfirmations(), 0u);
}

TEST(ImuConsumptionPolicy, DuplicateConfirmationFailsClosedAndNeverPopsASuccessor)
{
  Queue queue{{1, 1.0}, {2, 1.1}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 1);
  transaction.settle(queue, token, ImuDisposition::kCommitted);
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kDuplicate);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_EQ(queue.front().identity, 2);
  EXPECT_EQ(transaction.duplicateConfirmations(), 1u);
  EXPECT_THROW(transaction.borrow(queue, 1), std::logic_error);
}

TEST(ImuConsumptionPolicy, QueueClearInvalidatesTokenEvenWhenTheNextEpochHasTheSameStamp)
{
  Queue queue{{1, 1.0}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 1);
  transaction.invalidate();
  queue.clear();
  queue.push_back({2, 1.0});
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kStaleEpoch);
  EXPECT_EQ(queue.front().identity, 2);
  EXPECT_TRUE(transaction.faulted());
}

TEST(ImuConsumptionPolicy, ChangedIdentityWithAnEqualStampCannotConfirm)
{
  Queue queue{{1, 1.0}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 1);
  queue.front().identity = 2;
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kPrefixMismatch);
  EXPECT_EQ(queue.front().identity, 2);
  EXPECT_TRUE(transaction.faulted());
}

TEST(ImuConsumptionPolicy, PoisonedBatchIsExplicitlyReleasedAndNotRetried)
{
  Queue queue{{1, 1.0, false}, {2, 1.1}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 1);
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kInvalid), ImuConfirmation::kInvalidDropped);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_EQ(queue.front().identity, 2);
  EXPECT_FALSE(transaction.faulted());
}

TEST(ImuConsumptionPolicy, FatalPartialPropagationReleasesOwnedPrefixAndStopsFurtherProcessing)
{
  Queue queue{{1, 1.0}, {2, 1.1}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 1);
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kFatal), ImuConfirmation::kFatal);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_EQ(queue.front().identity, 2);
  EXPECT_THROW(transaction.borrow(queue, 1), std::logic_error);
}

TEST(ImuConsumptionPolicy, AdmissionLatchesBothEpochAndTransactionFaultsAcrossInvalidation)
{
  Queue queue{{1, 1.0}, {2, 1.1}};
  Transaction transaction;
  EXPECT_TRUE(transaction.acceptsInput(false));
  EXPECT_FALSE(transaction.acceptsInput(true));
  const auto token = transaction.borrow(queue, 1);
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kFatal), ImuConfirmation::kFatal);
  EXPECT_FALSE(transaction.acceptsInput(false));
  transaction.invalidate();
  queue.clear();
  EXPECT_FALSE(transaction.acceptsInput(false));
  EXPECT_FALSE(transaction.acceptsInput(true));
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kStaleEpoch);
  EXPECT_TRUE(queue.empty());
}

TEST(ImuConsumptionPolicy, AnEmptyPriorBatchLeavesFutureSamplesInPlace)
{
  Queue queue{{1, 2.0}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, fast_lio::imuPrefixThrough(queue, 1.0, stamp));
  EXPECT_TRUE(token.prefix.empty());
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kReleased);
  EXPECT_EQ(queue.size(), 1u);
}

TEST(ImuConsumptionPolicy, ValidityChecksRunBeforeDereferencingOrRetainingPoisonedInputs)
{
  EXPECT_TRUE(fast_lio::canRetainImuBatch(Queue{}, stamp, valid));
  EXPECT_TRUE(fast_lio::canRetainImuBatch(Queue{{1, 0.0}, {2, 0.0}}, stamp, valid));
  for (const double bad : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    EXPECT_FALSE(fast_lio::canRetainImuBatch(Queue{{1, bad}}, stamp, valid));
  EXPECT_FALSE(fast_lio::canRetainImuBatch(Queue{{1, 2.0}, {2, 1.0}}, stamp, valid));
  EXPECT_FALSE(fast_lio::canRetainImuBatch(Queue{{1, 1.0, false}}, stamp, valid));
}

TEST(ImuConsumptionPolicy, BacklogKeepsTheExistingStrictBudgetAndInvalidatesDiscardedPrefix)
{
  Queue queue{{1, 1.0}, {2, 1.5}, {3, 2.0}};
  Transaction transaction;
  const auto token = transaction.borrow(queue, 2);
  transaction.settle(queue, token, ImuDisposition::kUncommitted);
  EXPECT_EQ(
    fast_lio::imuBacklogExcess(
      queue.size(), [&](std::size_t i) { return queue[i].stamp; }, 2.0, 1.0, std::numeric_limits<double>::infinity()),
    0u);
  const auto dropped = fast_lio::imuBacklogExcess(
    queue.size(),
    [&](std::size_t i) { return queue[i].stamp; },
    std::nextafter(2.0, 3.0),
    1.0,
    std::numeric_limits<double>::infinity());
  ASSERT_EQ(dropped, 1u);
  transaction.invalidate();
  queue.pop_front();
  EXPECT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kStaleEpoch);
  EXPECT_EQ(queue.front().identity, 2);
}

TEST(ImuConsumptionPolicy, EpochGuardRetainsStrictOneSecondRollbackAndLatchesRestart)
{
  fast_lio::InputEpochGuard boundary;
  EXPECT_TRUE(boundary.observe(2.0, 1.0, 1.0));
  EXPECT_TRUE(boundary.observe(2.0, 1.9, 1.0));
  fast_lio::InputEpochGuard restart;
  EXPECT_FALSE(restart.observe(2.0, 0.99, 1.0));
  EXPECT_TRUE(restart.faulted());
  EXPECT_FALSE(restart.observe(2.0, 2.1, 1.0));
  fast_lio::ScanConsumption scans;
  scans.commit(2.0);
  EXPECT_FALSE(scans.canProcess(2.0));
}

TEST(ImuConsumptionPolicy, IntegrationLedgerSeparatesDuplicateIntervalsFromDeliberateReanchors)
{
  fast_lio::ImuIntegrationLedger ledger;
  ledger.anchor(1.0);
  ledger.observe(1.0, 1.1);
  ledger.observe(1.1, 1.1);
  ledger.anchor(1.1);
  ledger.anchor(2.0);
  ledger.observe(2.0, 2.1);
  EXPECT_EQ(ledger.duplicateIntegrations(), 0u);
  EXPECT_EQ(ledger.watermarkRegressions(), 0u);
  ledger.observe(2.05, 2.2);
  EXPECT_EQ(ledger.duplicateIntegrations(), 1u);
  ledger.anchor(1.0);
  EXPECT_EQ(ledger.watermarkRegressions(), 1u);
}

TEST(ImuConsumptionPolicy, InvalidOrNestedSelectionFailsClosed)
{
  Queue queue{{1, 1.0}};
  Transaction invalid;
  EXPECT_THROW(invalid.borrow(queue, 2), std::logic_error);
  Transaction nested;
  nested.borrow(queue, 1);
  EXPECT_THROW(nested.borrow(queue, 1), std::logic_error);
  EXPECT_EQ(queue.size(), 1u);
}

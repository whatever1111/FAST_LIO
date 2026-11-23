#include <deque>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <vector>

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
  const auto stampAt = [&](std::size_t i) {
    return queue[i].stamp;
  };
  EXPECT_EQ(fast_lio::imuBacklogExcess(queue.size(), stampAt, 2.0, 1.0, std::numeric_limits<double>::infinity()), 0u);
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

TEST(ImuConsumptionPolicy, SampleAfterTheScanEndIsReadAndConsumedOnlyByTheNextScan)
{
  // Stamps in ms: 200 Hz to 95, a 25 ms dropout across the scan end at 100, 200 Hz again from 120; scans end at 100,
  // 200 (on a sample) and 300.
  Queue queue;
  int identity = 0;
  for (int i = 0; i < 20; ++i)
    queue.push_back({++identity, i * 5.0});
  for (int i = 0; i <= 40; ++i)
    queue.push_back({++identity, 120.0 + i * 5.0});
  const int total = identity;
  Transaction transaction;
  std::map<int, int> released;
  std::vector<std::size_t> prefixes;
  std::vector<int> read;
  for (const double end : {100.0, 200.0, 300.0}) {
    const auto count = fast_lio::imuPrefixThrough(queue, end, stamp);
    const auto token = transaction.borrow(queue, count);
    const auto after = fast_lio::imuSampleAfterPrefix(queue, count, end, stamp, valid);
    ASSERT_LT(after, queue.size());
    EXPECT_EQ(after, count);
    EXPECT_GT(queue[after].stamp, end);
    read.push_back(queue[after].identity);
    // Reading it changes neither the borrowed prefix nor the queue.
    ASSERT_EQ(token.prefix.size(), count);
    for (const auto & sample : token.prefix) {
      EXPECT_LE(sample.stamp, end);
      EXPECT_NE(sample.identity, queue[after].identity);
    }
    prefixes.push_back(count);
    ASSERT_EQ(transaction.settle(queue, token, ImuDisposition::kCommitted), ImuConfirmation::kReleased);
    for (const auto & sample : token.prefix)
      ++released[sample.identity];
    // The read sample is the next scan's first.
    ASSERT_FALSE(queue.empty());
    EXPECT_EQ(queue.front().identity, read.back());
  }
  EXPECT_EQ(prefixes, (std::vector<std::size_t>{20, 17, 20}));
  EXPECT_EQ(read, (std::vector<int>{21, 38, 58}));
  for (int id = 1; id <= 57; ++id)
    EXPECT_EQ(released[id], 1) << "sample " << id;
  EXPECT_EQ(released.count(58), 0u);
  EXPECT_EQ(queue.size(), static_cast<std::size_t>(total - 57));
  EXPECT_EQ(transaction.duplicateConfirmations(), 0u);
}

TEST(ImuConsumptionPolicy, NoUsableSampleAfterTheScanEndIsReported)
{
  // None queued after the prefix (the newest sample lands on the end), an invalid one, a non-finite stamp.
  EXPECT_EQ(fast_lio::imuSampleAfterPrefix(Queue{{1, 0.095}, {2, 0.1}}, 2, 0.1, stamp, valid), 2u);
  EXPECT_EQ(fast_lio::imuSampleAfterPrefix(Queue{{1, 0.095}, {2, 0.12, false}}, 1, 0.1, stamp, valid), 2u);
  EXPECT_EQ(fast_lio::imuSampleAfterPrefix(
              Queue{{1, 0.095}, {2, std::numeric_limits<double>::quiet_NaN()}}, 1, 0.1, stamp, valid),
            2u);
  EXPECT_EQ(fast_lio::imuSampleAfterPrefix(Queue{}, 0, 0.1, stamp, valid), 0u);
}

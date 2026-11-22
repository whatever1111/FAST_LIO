#include <gtest/gtest.h>

#include "lidar_custody_receiver.hpp"

namespace
{
using namespace fast_lio;
const LidarInstance kSender{1};
const LidarInstance kReceiver{2};

TEST(LidarCustodyReceiver, PrefixAdvancesOnlyAfterActualIngressDisposition)
{
  LidarCustodyReceiver receiver(kReceiver);
  ASSERT_TRUE(receiver.bind(kSender, {}));
  const LidarWindowHeader first{kSender, kReceiver, 1, 20, 10};
  ASSERT_TRUE(receiver.validate(first));
  EXPECT_EQ(receiver.sequence(), 0U);
  EXPECT_TRUE(receiver.complete(first, LidarIngress::kEnqueued));
  EXPECT_EQ(receiver.sequence(), 1U);
  const LidarWindowHeader second{kSender, kReceiver, 2, 33, 4};
  EXPECT_TRUE(receiver.complete(second, LidarIngress::kLateRejected));
  EXPECT_EQ(receiver.points(), 14U);
  EXPECT_EQ(receiver.windowId(), 33);
  EXPECT_EQ(receiver.counts()[0].points, 10U);
  EXPECT_EQ(receiver.counts()[static_cast<std::size_t>(LidarIngress::kLateRejected)].points, 4U);
  EXPECT_TRUE(receiver.bind(kSender, kReceiver));  // Same live instance may answer a status query.
  EXPECT_EQ(receiver.sequence(), 2U);
}

TEST(LidarCustodyReceiver, LostMiddleOrDuplicateNeverReachesIngress)
{
  for (const auto seq : {1U, 3U}) {
    LidarCustodyReceiver receiver(kReceiver);
    ASSERT_TRUE(receiver.bind(kSender, {}));
    ASSERT_TRUE(receiver.complete({kSender, kReceiver, 1, 20, 10}, LidarIngress::kEnqueued));
    EXPECT_FALSE(receiver.validate({kSender, kReceiver, seq, 21, 4}));
    EXPECT_STREQ(receiver.failure(), "window_sequence");
    EXPECT_EQ(receiver.sequence(), 1U);
    EXPECT_FALSE(receiver.validate({kSender, kReceiver, 2, 21, 4}));
  }
}

TEST(LidarCustodyReceiver, MissingFirstAndWrongInstancesFailClosed)
{
  LidarCustodyReceiver missingFirst(kReceiver);
  ASSERT_TRUE(missingFirst.bind(kSender, {}));
  EXPECT_FALSE(missingFirst.validate({kSender, kReceiver, 2, 21, 4}));
  EXPECT_EQ(missingFirst.sequence(), 0U);

  LidarCustodyReceiver wrongSender(kReceiver);
  ASSERT_TRUE(wrongSender.bind(kSender, {}));
  EXPECT_FALSE(wrongSender.validate({{3}, kReceiver, 1, 20, 4}));
  EXPECT_STREQ(wrongSender.failure(), "window_instance");
  LidarCustodyReceiver restarted({3});
  EXPECT_FALSE(restarted.bind(kSender, kReceiver));
  EXPECT_STREQ(restarted.failure(), "bind_instance");
  LidarCustodyReceiver ambiguous(kReceiver);
  ASSERT_TRUE(ambiguous.bind(kSender, {}));
  EXPECT_FALSE(ambiguous.bind({4}, {}));
}

TEST(LidarCustodyReceiver, AnUnresolvedCallbackCannotAcknowledge)
{
  LidarCustodyReceiver receiver(kReceiver);
  ASSERT_TRUE(receiver.bind(kSender, {}));
  const LidarWindowHeader first{kSender, kReceiver, 1, 20, 4};
  ASSERT_TRUE(receiver.validate(first));
  EXPECT_FALSE(receiver.complete(first, LidarIngress::kUnresolved));
  EXPECT_STREQ(receiver.failure(), "ingress_unresolved");
  EXPECT_EQ(receiver.sequence(), 0U);
  EXPECT_EQ(receiver.points(), 0U);
}

TEST(LidarCustodyReceiver, EachExistingEarlyReturnHasItsOwnPartition)
{
  LidarCustodyReceiver receiver(kReceiver);
  ASSERT_TRUE(receiver.bind(kSender, {}));
  std::uint64_t seq = 0;
  for (auto condition :
       {LidarIngress::kEpochRejected, LidarIngress::kLeadRejected, LidarIngress::kTransactionRejected}) {
    ++seq;
    ASSERT_TRUE(receiver.complete({kSender, kReceiver, seq, static_cast<std::int64_t>(seq), seq}, condition));
    EXPECT_EQ(receiver.counts()[static_cast<std::size_t>(condition)].points, seq);
  }
  EXPECT_EQ(receiver.sequence(), 3U);
  EXPECT_EQ(receiver.points(), 6U);
}
}  // namespace

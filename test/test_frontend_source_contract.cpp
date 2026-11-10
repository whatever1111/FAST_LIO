#include <gtest/gtest.h>

#include "frontend_source_contract.hpp"

namespace fastlio::joint_source
{
namespace
{

FactorIdentity factor(const std::string & key, const std::string & raw)
{
  return {key, "numeric-payload", {raw}, {}, key + "/eps", "synthetic independent realization"};
}

TEST(FrontendSourceContract, I03CanonicalAliasIsExactNoOpAndConflictIsAtomic)
{
  FactorLedger ledger;
  const auto first = factor("canonical", "raw1");
  ASSERT_EQ(ledger.consume({first, first}).status, Status::kOk);
  const auto snapshot = ledger.entries();
  EXPECT_EQ(snapshot.size(), 1U);
  EXPECT_EQ(ledger.consume({first}).status, Status::kNoOp);
  auto changed = first;
  changed.payload += "changed";
  EXPECT_EQ(ledger.consume({factor("fresh", "raw2"), changed}).status, Status::kIdentityConflict);
  EXPECT_EQ(ledger.entries(), snapshot);
}

TEST(FrontendSourceContract, I03SameRawDifferentGeometryRequiresSharedSource)
{
  FactorLedger ledger;
  auto first = factor("geometry-a", "raw");
  first.shared = {{"raw", "source"}};
  ASSERT_EQ(ledger.consume({first}).status, Status::kOk);
  auto next = factor("geometry-b", "raw");
  EXPECT_EQ(ledger.consume({next}).status, Status::kUnsupportedNoiseStructure);
  next.shared = first.shared;
  EXPECT_EQ(ledger.consume({next}).status, Status::kOk);
  auto falseIndependent = factor("geometry-c", "raw-new");
  falseIndependent.remainderKey = first.remainderKey;
  EXPECT_EQ(ledger.consume({falseIndependent}).status, Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(ledger.entries().size(), 2U);
}

TEST(FrontendSourceContract, I17EqualCoordinatesAreNotAnIdentity)
{
  FactorLedger ledger;
  auto first = factor("query-a", "raw-a");
  auto second = factor("query-b", "raw-b");
  EXPECT_EQ(first.payload, second.payload);
  EXPECT_EQ(ledger.consume({first, second}).fresh.size(), 2U);
}

TEST(FrontendSourceContract, I23LedgerBoundaryAndLateRejectLeaveAllEntriesIntact)
{
  FactorLedger ledger;
  std::vector<FactorIdentity> factors;
  for (std::size_t i = 0; i < kMaxLedger; ++i) {
    const auto key = std::to_string(i);
    factors.push_back(factor(key, key));
  }
  ASSERT_EQ(ledger.consume(factors).status, Status::kOk);
  const auto snapshot = ledger.entries();
  EXPECT_EQ(ledger.consume({factors.front()}).status, Status::kNoOp);
  EXPECT_EQ(ledger.consume({factor("overflow", "overflow")}).status, Status::kModelCapacityExceeded);
  EXPECT_EQ(ledger.entries(), snapshot);
  factors.push_back(factor("overflow", "overflow"));
  EXPECT_EQ(ledger.consume(factors).status, Status::kModelCapacityExceeded);
  EXPECT_EQ(ledger.entries(), snapshot);
}

TEST(FrontendSourceContract, MissingAndMalformedIdentityAreRejected)
{
  FactorLedger ledger;
  EXPECT_EQ(ledger.consume({}).status, Status::kNoOp);
  auto invalid = factor("a", "raw");
  invalid.rawAncestry = {"raw", "raw"};
  EXPECT_EQ(ledger.consume({invalid}).status, Status::kInvalidIdentity);
  invalid = factor("a", "raw");
  invalid.shared = {{"absent", "source"}};
  EXPECT_EQ(ledger.consume({invalid}).status, Status::kInvalidIdentity);
  invalid.payload.clear();
  EXPECT_EQ(ledger.consume({invalid}).status, Status::kInvalidIdentity);
  EXPECT_TRUE(ledger.entries().empty());
}

TEST(FrontendSourceContract, SourceDictionaryDimensionsUnitsEpochAndAliases)
{
  Source source{"canonical",
                "raw_lidar_return",
                {"recording/device/message/return"},
                {"packet-alias"},
                {{"range"}, {"m"}, "L", "range", "clock/epoch"},
                "synthetic",
                "retained"};
  EXPECT_EQ(validateSources({source}), Status::kOk);
  auto conflict = source;
  conflict.key = "packet-alias";
  EXPECT_EQ(validateSources({source, conflict}), Status::kIdentityConflict);
  source.coordinates.units.clear();
  EXPECT_EQ(validateSources({source}), Status::kInvalidIdentity);
  source.coordinates.units = {"m"};
  source.coordinates.epoch.clear();
  EXPECT_EQ(validateSources({source}), Status::kInvalidIdentity);
  EXPECT_EQ(JointMoments{}.knowledge, MomentKnowledge::kUnknown);
}

}  // namespace
}  // namespace fastlio::joint_source

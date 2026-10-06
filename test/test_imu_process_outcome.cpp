#include <gtest/gtest.h>

#include "imu_process_outcome.hpp"

TEST(ImuProcessOutcome, CoverageReanchorsCommitButMalformedCoverageDoesNot)
{
  using fast_lio::ImuCoverageStatus;
  using fast_lio::ImuDisposition;
  for (const auto status : {ImuCoverageStatus::kGap, ImuCoverageStatus::kStart, ImuCoverageStatus::kEnd})
    EXPECT_EQ(fast_lio::imuCoverageOutcome(status).disposition, ImuDisposition::kCommitted);
  for (const auto status : {ImuCoverageStatus::kInvalid, ImuCoverageStatus::kUnordered})
    EXPECT_EQ(fast_lio::imuCoverageOutcome(status).disposition, ImuDisposition::kInvalid);
  EXPECT_EQ(fast_lio::imuCoverageOutcome(ImuCoverageStatus::kEmpty).disposition, ImuDisposition::kUncommitted);
}

TEST(ImuProcessOutcome, EveryReasonHasAStableName)
{
  for (std::size_t i = 0; i < static_cast<std::size_t>(fast_lio::ImuProcessReason::kCount); ++i)
    EXPECT_STRNE(fast_lio::imuProcessReasonName(static_cast<fast_lio::ImuProcessReason>(i)), "unknown");
}

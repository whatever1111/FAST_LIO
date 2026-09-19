#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "imu_coverage_policy.hpp"

namespace
{
auto coverage(double begin, double end, const std::vector<double> & stamps)
{
  return fast_lio::imuCoverage(begin, end, stamps, {0.1, 0.01});
}
}  // namespace
TEST(ImuCoverage, ChecksPreviousSampleAndBothEndpoints)
{
  EXPECT_TRUE(coverage(10.0, 10.1, {9.999, 10.005, 10.05, 10.095}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {9.0, 10.005, 10.095}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {10.02, 10.095}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {9.999, 10.005, 10.05}).covered());
}
TEST(ImuCoverage, DetectsFourSecondGapBeforePropagation)
{
  const auto result = coverage(10.0, 14.1, {9.999, 10.0, 14.0, 14.095});
  EXPECT_EQ(result.status, fast_lio::ImuCoverageStatus::kGap);
  EXPECT_DOUBLE_EQ(result.max_gap_s, 4.0);
}
TEST(ImuCoverage, RejectsMalformedWindowsAndInvalidConfiguration)
{
  EXPECT_FALSE(coverage(10.0, 10.1, {}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {10.0, 9.99, 10.1}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {10.0, 11.0}).covered());
  EXPECT_FALSE(coverage(10.0, 10.1, {10.0, std::numeric_limits<double>::quiet_NaN()}).covered());
  EXPECT_FALSE(coverage(10.1, 10.0, {10.0}).covered());
  EXPECT_FALSE(fast_lio::imuCoverage(10.0, 10.1, std::vector<double>{10.0}, {-1.0, 0.0}).covered());
}
TEST(ImuCoverage, ZeroBoundsPreserveDeviceCompatibilityButNotInvalidOrder)
{
  EXPECT_TRUE(fast_lio::imuCoverage(10.0, 14.1, std::vector<double>{9.999, 14.0}, {}).covered());
  EXPECT_FALSE(fast_lio::imuCoverage(10.0, 14.1, std::vector<double>{14.0, 10.0}, {}).covered());
}

TEST(ImuCoverage, ExactLimitIsInclusiveAndStartReasonIsDistinct)
{
  EXPECT_EQ(coverage(0.0, 0.03, {0.02, 0.03}).status, fast_lio::ImuCoverageStatus::kStart);
  EXPECT_TRUE(fast_lio::imuCoverage(0.0, 0.1, std::vector<double>{0.0}, {0.1, 0.1}).covered());
  EXPECT_EQ(fast_lio::imuCoverage(0.0, 0.100001, std::vector<double>{0.0}, {0.1, 0.1}).status,
            fast_lio::ImuCoverageStatus::kEnd);
}

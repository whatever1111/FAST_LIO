#include <gtest/gtest.h>
#include <limits>

#include "scan_hole_policy.hpp"

TEST(ScanHolePolicy, ZeroDisablesVerification)
{
  for (double gap : {-1.0, 0.0, 8.0, 26.1, 60.2, std::numeric_limits<double>::max()}) {
    EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(gap, 0.0, false));
    EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(gap, 0.0, true));
  }
}

TEST(ScanHolePolicy, ThresholdIsStrictlyGreater)
{
  EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(8.0, 10.0, false));
  EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(10.0, 10.0, false));
  EXPECT_TRUE(fast_lio::scanHoleRequiresVerification(std::nextafter(10.0, 11.0), 10.0, false));
  EXPECT_TRUE(fast_lio::scanHoleRequiresVerification(26.1, 10.0, false));
}

TEST(ScanHolePolicy, BlindGateKeepsItsVerificationProgress)
{
  EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(26.1, 10.0, true));
  EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(60.2, 10.0, true));
}

TEST(ScanHolePolicy, RejectsNonfiniteGapsAndInvalidParameters)
{
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double invalid : {nan, inf, -inf}) {
    EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(invalid, 10.0, false));
    EXPECT_FALSE(fast_lio::scanHoleRequiresVerification(invalid, 0.0, false));
    EXPECT_FALSE(fast_lio::validScanHoleVerifySec(invalid));
  }
  EXPECT_FALSE(fast_lio::validScanHoleVerifySec(-1.0));
  EXPECT_TRUE(fast_lio::validScanHoleVerifySec(0.0));
  EXPECT_TRUE(fast_lio::validScanHoleVerifySec(10.0));
}

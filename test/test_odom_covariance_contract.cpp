#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

#include "odom_covariance_contract.hpp"

namespace
{
using Matrix6 = Eigen::Matrix<double, 6, 6>;

TEST(OdomCovarianceContract, PermutesAxesAndCrossTerms)
{
  Matrix6 source = Matrix6::Zero();
  source(0, 0) = 1.25;
  source(0, 3) = -2.5;
  source(3, 0) = 3.75;
  source(5, 2) = 4.5;
  const auto output = fast_lio::makeOdomCovariance(source, false);
  EXPECT_DOUBLE_EQ(output[0], 0.0);
  EXPECT_DOUBLE_EQ(output[3], 3.75);
  EXPECT_DOUBLE_EQ(output[18], -2.5);
  EXPECT_DOUBLE_EQ(output[17], 4.5);
  EXPECT_DOUBLE_EQ(output[21], 1.25);
  EXPECT_DOUBLE_EQ(output[35], 0.0);
}

TEST(OdomCovarianceContract, IdentityAndZeroRemainUnchangedWhenHealthy)
{
  Matrix6 identity = Matrix6::Identity();
  const auto output = fast_lio::makeOdomCovariance(identity, false);
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_DOUBLE_EQ(output[i], identity(static_cast<int>(i / 6), static_cast<int>(i % 6)));
  }
  const std::array<double, 36> zero{};
  EXPECT_EQ(fast_lio::makeOdomCovariance(Matrix6::Zero(), false), zero);
}

TEST(OdomCovarianceContract, DegradedAddsOnlyWirePositionVariances)
{
  Matrix6 source = Matrix6::Constant(2.0);
  const auto healthy = fast_lio::makeOdomCovariance(source, false);
  const auto degraded = fast_lio::makeOdomCovariance(source, true);
  for (std::size_t i = 0; i < degraded.size(); ++i) {
    const double expected = (i == 21 || i == 28 || i == 35) ? healthy[i] + 100.0 : healthy[i];
    EXPECT_DOUBLE_EQ(degraded[i], expected);
  }
}

TEST(OdomCovarianceContract, NoAccumulationAndUsesCurrentInput)
{
  Matrix6 source = Matrix6::Identity();
  source(0, 3) = 4.0;
  const auto first = fast_lio::makeOdomCovariance(source, true);
  Matrix6 changed = Matrix6::Identity();
  changed(0, 0) = 9.0;
  const auto second = fast_lio::makeOdomCovariance(changed, true);
  EXPECT_NE(first, second);
  EXPECT_DOUBLE_EQ(second[21], 109.0);
  EXPECT_DOUBLE_EQ(second[0], 1.0);
}

TEST(OdomCovarianceContract, PassesNonFiniteValuesThroughForBackendSanitization)
{
  Matrix6 source = Matrix6::Zero();
  source(0, 0) = std::numeric_limits<double>::quiet_NaN();
  source(4, 5) = std::numeric_limits<double>::infinity();
  const auto output = fast_lio::makeOdomCovariance(source, false);
  EXPECT_TRUE(std::isnan(output[21]));
  EXPECT_TRUE(std::isinf(output[8]));
}
}  // namespace

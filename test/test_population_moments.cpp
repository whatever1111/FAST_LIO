#include <Eigen/Core>

#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "population_moments.hpp"

namespace
{
using Vector = Eigen::Vector3d;

TEST(PopulationMoments, EmptyAndTwoSamplePopulationVariance)
{
  Vector mean = Vector::Zero(), variance = Vector::Zero();
  const Vector first(0, 2, 4), second(2, 4, 8);
  EXPECT_THROW(fast_lio::updatePopulationMoments(first, 0, mean, variance), std::invalid_argument);
  fast_lio::updatePopulationMoments(first, 1, mean, variance);
  EXPECT_EQ(mean, first);
  EXPECT_EQ(variance, Vector::Zero());
  fast_lio::updatePopulationMoments(second, 2, mean, variance);
  EXPECT_EQ(mean, Vector(1, 3, 6));
  EXPECT_EQ(variance, Vector(1, 1, 4));
}

TEST(PopulationMoments, ConstantAlternatingAndBatchBoundariesDoNotChangeStatistics)
{
  Vector oneMean = Vector::Zero(), oneVariance = Vector::Zero();
  Vector batchMean = Vector::Zero(), batchVariance = Vector::Zero();
  const Vector center(0, 0, 9.81), deviation(0.32, 0.2, 0);
  std::vector<Vector> samples;
  for (int i = 0; i < 20; ++i)
    samples.push_back(center + (i % 2 ? -deviation : deviation));
  for (std::size_t i = 0; i < samples.size(); ++i)
    fast_lio::updatePopulationMoments(samples[i], i + 1, oneMean, oneVariance);
  std::size_t count = 0;
  for (const std::size_t batch : {3U, 5U, 12U}) {
    for (std::size_t i = 0; i < batch; ++i) {
      ++count;
      fast_lio::updatePopulationMoments(samples[count - 1], count, batchMean, batchVariance);
    }
  }
  EXPECT_LT((oneMean - center).norm(), 1e-15);
  EXPECT_LT((oneVariance - Vector(0.1024, 0.04, 0)).norm(), 1e-15);
  EXPECT_EQ(oneMean, batchMean);
  EXPECT_EQ(oneVariance, batchVariance);
  for (std::size_t i = 1; i < 5; ++i)
    fast_lio::updatePopulationMoments(center, i, oneMean, oneVariance);
  EXPECT_EQ(oneMean, center);
  EXPECT_EQ(oneVariance, Vector::Zero());
}

TEST(PopulationMoments, LargeCountHasNoIntegerSquareAndOverflowRemainsDetectable)
{
  Vector mean(0, 0, 9.81), variance(0.1, 0.1, 0.1);
  const Vector sample = mean;
  fast_lio::updatePopulationMoments(sample, 100000, mean, variance);
  EXPECT_TRUE(mean.allFinite());
  EXPECT_TRUE(variance.allFinite());
  fast_lio::updatePopulationMoments(Vector(1e200, 0, 9.81), 1, mean, variance);
  fast_lio::updatePopulationMoments(Vector(-1e200, 0, 9.81), 2, mean, variance);
  EXPECT_FALSE(variance.allFinite());
}
}  // namespace

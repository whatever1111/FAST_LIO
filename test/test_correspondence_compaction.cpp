#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "correspondence_compaction.hpp"

namespace
{
using Cloud = pcl::PointCloud<pcl::PointXYZINormal>;

class CorrespondenceCompaction : public ::testing::Test
{
protected:
  Cloud source, sourceNormals, points, normals;
  std::vector<double> sourceWeights, weights;
  std::vector<bool> selected;
  std::vector<std::size_t> visited;

  void prepare(std::size_t count)
  {
    source.resize(count);
    sourceNormals.resize(count);
    sourceWeights.resize(count);
    selected.assign(count, true);
    for (std::size_t i = 0; i < count; ++i) {
      source[i].x = static_cast<float>(i);
      source[i].intensity = static_cast<float>(i + 10);
      sourceNormals[i].z = -static_cast<float>(i + 1);
      sourceNormals[i].intensity = static_cast<float>(i + 20);
      sourceWeights[i] = 0.25 * (i + 1);
    }
  }

  std::size_t compact(bool weighted = true)
  {
    visited.clear();
    return fast_lio::compactSelectedCorrespondences(source.size(),
                                                    source,
                                                    sourceNormals,
                                                    selected,
                                                    sourceWeights,
                                                    weighted,
                                                    points,
                                                    normals,
                                                    weights,
                                                    [&](std::size_t i) { visited.push_back(i); });
  }

  void expectRows(const std::vector<std::size_t> & indices, bool weighted = true)
  {
    ASSERT_EQ(points.size(), indices.size());
    ASSERT_EQ(normals.size(), indices.size());
    ASSERT_EQ(weights.size(), indices.size());
    EXPECT_EQ(visited, indices);
    EXPECT_EQ(points.width, indices.size());
    EXPECT_EQ(normals.width, indices.size());
    EXPECT_EQ(points.height, indices.empty() ? 0U : 1U);
    EXPECT_EQ(normals.height, indices.empty() ? 0U : 1U);
    for (std::size_t row = 0; row < indices.size(); ++row) {
      const auto i = indices[row];
      EXPECT_EQ(points[row].x, static_cast<float>(i));
      EXPECT_EQ(points[row].intensity, static_cast<float>(i + 10));
      EXPECT_EQ(normals[row].z, -static_cast<float>(i + 1));
      EXPECT_EQ(normals[row].intensity, static_cast<float>(i + 20));
      EXPECT_DOUBLE_EQ(weights[row], weighted ? 0.25 * (i + 1) : 1.0);
    }
  }
};

TEST_F(CorrespondenceCompaction, EmptyAndNoMatchesHaveNoLiveRows)
{
  prepare(0);
  EXPECT_EQ(compact(), 0U);
  expectRows({});
  prepare(3);
  selected.assign(3, false);
  EXPECT_EQ(compact(), 0U);
  expectRows({});
}

TEST_F(CorrespondenceCompaction, SingleAndSparseMatchesKeepPairsAndOrder)
{
  prepare(6);
  selected = {false, false, true, false, false, false};
  EXPECT_EQ(compact(), 1U);
  expectRows({2});
  selected = {true, false, true, false, false, true};
  EXPECT_EQ(compact(), 3U);
  expectRows({0, 2, 5});
}

TEST_F(CorrespondenceCompaction, DisabledKernelUsesUnitWeightsWithoutReadingSourceWeights)
{
  prepare(3);
  sourceWeights.clear();
  EXPECT_EQ(compact(false), 3U);
  expectRows({0, 1, 2}, false);
}

TEST_F(CorrespondenceCompaction, RepeatedCallsShrinkClearAndGrowWithoutStaleRows)
{
  prepare(5);
  compact();
  expectRows({0, 1, 2, 3, 4});
  prepare(1);
  compact();
  expectRows({0});
  prepare(0);
  compact();
  expectRows({});
  prepare(8);
  for (int repeat = 0; repeat < 3; ++repeat) {
    compact();
    expectRows({0, 1, 2, 3, 4, 5, 6, 7});
  }
}

TEST_F(CorrespondenceCompaction, GrowsBeyondLegacyCloudCapacity)
{
  // This tests output compaction, not the callback's separate fixed input arrays.
  constexpr std::size_t kLegacyCapacity = 100000;
  points.resize(kLegacyCapacity);
  normals.resize(kLegacyCapacity);
  prepare(kLegacyCapacity + 1);
  ASSERT_EQ(compact(), kLegacyCapacity + 1);
  EXPECT_EQ(points.width, kLegacyCapacity + 1);
  EXPECT_EQ(normals.width, kLegacyCapacity + 1);
  ASSERT_EQ(weights.size(), kLegacyCapacity + 1);
  EXPECT_EQ(points.back().x, static_cast<float>(kLegacyCapacity));
  EXPECT_EQ(normals.back().z, -static_cast<float>(kLegacyCapacity + 1));
  EXPECT_DOUBLE_EQ(weights.back(), 0.25 * (kLegacyCapacity + 1));
  ASSERT_EQ(visited.size(), kLegacyCapacity + 1);
  EXPECT_EQ(visited.back(), kLegacyCapacity);
}

TEST_F(CorrespondenceCompaction, ObserverPreservesResidualAdditionAndDiagnosticOrder)
{
  prepare(5);
  selected = {true, false, true, true, true};
  const double residuals[] = {1e16, 999.0, 1.0, 1.0, 2.0};
  double total = 0.0;
  std::vector<double> diagnostics;
  fast_lio::compactSelectedCorrespondences(
    source.size(), source, sourceNormals, selected, sourceWeights, true, points, normals, weights, [&](std::size_t i) {
      total += residuals[i];
      diagnostics.push_back(residuals[i]);
    });
  EXPECT_EQ(diagnostics, (std::vector<double>{1e16, 1.0, 1.0, 2.0}));
  EXPECT_DOUBLE_EQ(total, ((1e16 + 1.0) + 1.0) + 2.0);
}

TEST_F(CorrespondenceCompaction, DoesNotIntroduceFiniteValueFiltering)
{
  prepare(1);
  source[0].x = std::numeric_limits<float>::quiet_NaN();
  sourceNormals[0].z = std::numeric_limits<float>::infinity();
  EXPECT_EQ(compact(), 1U);
  EXPECT_TRUE(std::isnan(points[0].x));
  EXPECT_TRUE(std::isinf(normals[0].z));
}

TEST_F(CorrespondenceCompaction, InvalidSizesAndAliasingFailBeforeClearingOutputs)
{
  prepare(2);
  compact();
  const auto invoke = [&](std::size_t count) {
    return fast_lio::compactSelectedCorrespondences(
      count, source, sourceNormals, selected, sourceWeights, true, points, normals, weights, [](std::size_t) {});
  };
  EXPECT_THROW(invoke(3), std::invalid_argument);
  sourceNormals.resize(1);
  EXPECT_THROW(invoke(2), std::invalid_argument);
  sourceNormals.resize(2);
  selected.resize(1);
  EXPECT_THROW(invoke(2), std::invalid_argument);
  selected.resize(2);
  sourceWeights.resize(1);
  EXPECT_THROW(invoke(2), std::invalid_argument);
  sourceWeights.resize(2);
  EXPECT_THROW(fast_lio::compactSelectedCorrespondences(
                 2, source, sourceNormals, selected, sourceWeights, true, source, normals, weights, [](std::size_t) {}),
               std::invalid_argument);
  EXPECT_EQ(points.size(), 2U);
  EXPECT_EQ(normals.size(), 2U);
  EXPECT_EQ(weights.size(), 2U);
  EXPECT_EQ(source.size(), 2U);
}
}  // namespace

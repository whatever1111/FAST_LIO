#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "scan_history_policy.hpp"
#include "scan_time_policy.hpp"

namespace
{
struct Point
{
  double curvature;
  int sourceIndex;
  std::uint32_t payload;
};

double offset(const Point & point)
{
  return point.curvature;
}

void expectSamePoint(const Point & actual, const Point & expected)
{
  EXPECT_EQ(std::memcmp(&actual.curvature, &expected.curvature, sizeof(double)), 0);
  EXPECT_EQ(actual.sourceIndex, expected.sourceIndex);
  EXPECT_EQ(actual.payload, expected.payload);
}
}  // namespace

TEST(ScanHistoryPolicy, ZeroOverlapKeepsBoundaryAndEveryFieldWithoutCopying)
{
  std::vector<Point> points{{0.0, 0, 17}, {100.0, 4, 18}, {50.0, 8, 19}};
  const auto original = points;
  const auto * storage = points.data();
  ASSERT_TRUE(fast_lio::scanTime(10.0, points, offset).valid());
  EXPECT_EQ(fast_lio::scanHistoryAction(points, 0.0, offset), fast_lio::ScanHistoryAction::kKeepAll);
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, 0.0, offset), 0u);
  EXPECT_EQ(points.data(), storage);
  ASSERT_EQ(points.size(), original.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    expectSamePoint(points[i], original[i]);
}

TEST(ScanHistoryPolicy, ExactHistoryBoundaryIsRetainedWithoutEpsilon)
{
  const double boundary = 125.0;
  std::vector<Point> points{{std::nextafter(boundary, 0.0), 0, 1},
                            {boundary, 4, 2},
                            {std::nextafter(boundary, std::numeric_limits<double>::infinity()), 8, 3}};
  const auto original = points;
  const double historyOffset = 0.125;
  EXPECT_EQ(fast_lio::scanHistoryAction(points, historyOffset, offset), fast_lio::ScanHistoryAction::kDiscardEarly);
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, historyOffset, offset), 1u);
  ASSERT_EQ(points.size(), 2u);
  expectSamePoint(points[0], original[1]);
  expectSamePoint(points[1], original[2]);
}

TEST(ScanHistoryPolicy, RepresentableOneNanosecondOverlapDropsOnlyEarlyPoints)
{
  const double begin = 0.0;
  const double lastEnd = 1e-9;
  std::vector<Point> points{{0.0, 0, 1}, {2e-6, 4, 2}, {100.0, 8, 3}};
  ASSERT_EQ(fast_lio::scanHistoryAction(points, lastEnd - begin, offset), fast_lio::ScanHistoryAction::kDiscardEarly);
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, lastEnd - begin, offset), 1u);
  ASSERT_EQ(points.size(), 2u);
  EXPECT_EQ(points.front().sourceIndex, 4);
}

TEST(ScanHistoryPolicy, UnixEpochNextafterUsesBinary64Boundary)
{
  const double begin = 1790000000.0;
  const double lastEnd = std::nextafter(begin, std::numeric_limits<double>::infinity());
  const double historyOffset = lastEnd - begin;
  ASSERT_GT(historyOffset, 1e-9);
  EXPECT_EQ(begin + 1e-9, begin);  // This epoch cannot express every nanosecond.
  const double boundary = historyOffset * 1000.0;
  std::vector<Point> points{{std::nextafter(boundary, 0.0), 0, 1}, {boundary, 4, 2}, {1.0, 8, 3}};
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, historyOffset, offset), 1u);
  ASSERT_EQ(points.size(), 2u);
  EXPECT_DOUBLE_EQ(points.front().curvature, boundary);
}

TEST(ScanHistoryPolicy, Observed511MicrosecondOverlapsUseTheOriginalComparison)
{
  for (double overlap : {0.000511, 0.000511169433594}) {
    SCOPED_TRACE(overlap);
    std::vector<Point> points{{0.0, 0, 1}, {0.510, 4, 2}, {0.512, 8, 3}, {100.0, 12, 4}};
    EXPECT_EQ(fast_lio::scanHistoryAction(points, overlap, offset), fast_lio::ScanHistoryAction::kDiscardEarly);
    EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, overlap, offset), 2u);
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[0].sourceIndex, 8);
    EXPECT_EQ(points[1].sourceIndex, 12);
    for (const auto & point : points)
      EXPECT_GE(double(point.curvature) / 1000.0, overlap);
  }
}

TEST(ScanHistoryPolicy, AllOldAndEmptyPointSetsRemainRejected)
{
  std::vector<Point> points{{0.0, 0, 1}, {99.0, 4, 2}};
  EXPECT_EQ(fast_lio::scanHistoryAction(points, 0.1, offset), fast_lio::ScanHistoryAction::kReject);
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, 0.1, offset), 2u);
  EXPECT_TRUE(points.empty());
  EXPECT_EQ(fast_lio::scanHistoryAction(points, 0.1, offset), fast_lio::ScanHistoryAction::kReject);
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, 0.1, offset), 0u);
}

TEST(ScanHistoryPolicy, InterleavedUnsortedPointsRetainOriginalOrderAndSensorStride)
{
  std::vector<Point> points{
    {0.0, 0, 11}, {0.9, 4, 12}, {0.1, 8, 13}, {0.5, 12, 14}, {0.5, 16, 15}, {0.2, 20, 16}, {1.0, 24, 17}};
  const auto original = points;
  EXPECT_EQ(fast_lio::discardPointsBeforeHistory(points, 0.0005, offset), 3u);
  ASSERT_EQ(points.size(), 4u);
  const std::size_t retainedIndices[]{1, 3, 4, 6};
  for (std::size_t i = 0; i < points.size(); ++i)
    expectSamePoint(points[i], original[retainedIndices[i]]);
}

TEST(ScanHistoryPolicy, OriginalScanTimeRejectsInvalidOffsetsBeforeAnyCompaction)
{
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
    std::vector<Point> points{{invalid, 0, 1}, {100.0, 4, 2}};
    const auto timing = fast_lio::scanTime(10.0, points, offset);
    EXPECT_EQ(timing.status, fast_lio::ScanTimeStatus::kInvalidOffset);
    EXPECT_EQ(points.size(), 2u);
  }
}

TEST(ScanHistoryPolicy, RetainedOffsetsHaveNonnegativeDeskewDeltaAndNegativeSeedRemainsLegal)
{
  std::vector<Point> overlapping{{0.0, 0, 1}, {62.5, 4, 2}, {75.0, 8, 3}, {100.0, 12, 4}};
  const double historyOffset = 0.0625;
  ASSERT_EQ(fast_lio::discardPointsBeforeHistory(overlapping, historyOffset, offset), 1u);
  for (const auto & point : overlapping)
    EXPECT_GE(point.curvature / 1000.0 - historyOffset, 0.0);
  EXPECT_DOUBLE_EQ(overlapping.front().curvature / 1000.0 - historyOffset, 0.0);
  const std::vector<Point> afterGap{{0.0, 0, 1}, {100.0, 4, 2}};
  EXPECT_EQ(fast_lio::scanHistoryAction(afterGap, -0.025, offset), fast_lio::ScanHistoryAction::kKeepAll);
  for (const auto & point : afterGap)
    EXPECT_GT(point.curvature / 1000.0 - (-0.025), 0.0);
}

TEST(ScanHistoryPolicy, ANewlyCommittedEndChangesTheNextScanHistoryDecision)
{
  const std::vector<Point> first{{0.0, 0, 1}, {100.0, 4, 2}};
  const std::vector<Point> successor{{0.0, 0, 3}, {100.0, 4, 4}};
  const double initialEnd = 10.1;
  const double firstBegin = initialEnd - 0.000511169433594;
  EXPECT_EQ(fast_lio::scanHistoryAction(first, initialEnd - firstBegin, offset),
            fast_lio::ScanHistoryAction::kDiscardEarly);
  const double successorBegin = 10.15;
  EXPECT_EQ(fast_lio::scanHistoryAction(successor, initialEnd - successorBegin, offset),
            fast_lio::ScanHistoryAction::kKeepAll);
  EXPECT_EQ(fast_lio::scanHistoryAction(successor, firstBegin + 0.1 - successorBegin, offset),
            fast_lio::ScanHistoryAction::kDiscardEarly);
}

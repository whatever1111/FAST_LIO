#include <rclcpp/rclcpp.hpp>

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "common_lib.h"
#include "plane_fit.hpp"

namespace
{
struct Point
{
  double x, y, z;
};
using Points = std::vector<Point>;
Points plane(double height = 0.0)
{
  return {{-1, -1, height}, {-1, 1, height}, {1, -1, height}, {1, 1, height}, {0, 0, height}};
}
void expectRejected(const Points & points, double threshold)
{
  const Eigen::Vector4d sentinel(1, 2, 3, 4);
  Eigen::Vector4d result = sentinel;
  EXPECT_FALSE(fast_lio::fitPlane<5>(points, threshold, result));
  EXPECT_EQ(result, sentinel);
}
}  // namespace

TEST(PlaneFit, OriginAndOrdinaryOffsetPlanesHaveFiniteUnitNormals)
{
  for (double height : {0.0, 2.0, -2.0}) {
    Eigen::Vector4d result;
    ASSERT_TRUE(fast_lio::fitPlane<5>(plane(height), 0.0, result));
    EXPECT_TRUE(result.allFinite());
    EXPECT_DOUBLE_EQ(result.head<3>().norm(), 1.0);
    EXPECT_DOUBLE_EQ(result[3], std::abs(height));
    EXPECT_DOUBLE_EQ(result[2], height > 0.0 ? -1.0 : 1.0);
  }
}

TEST(PlaneFit, RigidCoordinateChangesPreserveTheGeometricPlane)
{
  const Eigen::Matrix3d rotation =
    (Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(-0.7, Eigen::Vector3d::UnitY()))
      .toRotationMatrix();
  for (const Eigen::Vector3d translation :
       {Eigen::Vector3d::Zero().eval(), Eigen::Vector3d(2, -3, 4), Eigen::Vector3d(1e6, -2e6, 3e6)}) {
    Points transformed;
    for (const auto & p : plane()) {
      const Eigen::Vector3d q = rotation * Eigen::Vector3d(p.x, p.y, p.z) + translation;
      transformed.push_back({q.x(), q.y(), q.z()});
    }
    Eigen::Vector4d fit;
    ASSERT_TRUE(fast_lio::fitPlane<5>(transformed, 1e-8, fit));
    const Eigen::Vector3d normal = rotation.col(2);
    EXPECT_NEAR(std::abs(fit.head<3>().dot(normal)), 1.0, 1e-10);
    EXPECT_NEAR(fit.head<3>().dot(translation) + fit[3], 0.0, 1e-8);
  }
}

TEST(PlaneFit, NoisyOrthogonalFitAndDistanceBoundary)
{
  auto points = plane();
  points[0].z = points[3].z = 0.125;
  points[1].z = points[2].z = -0.125;
  Eigen::Vector4d fit;
  ASSERT_TRUE(fast_lio::fitPlane<5>(points, 0.125, fit));
  EXPECT_NEAR((fit - Eigen::Vector4d(0, 0, 1, 0)).norm(), 0.0, 1e-14);
  expectRejected(points, std::nextafter(0.125, 0.0));
  // Beyond the five selected neighbours is deliberately outside this fit.
  points.push_back({0, 0, std::numeric_limits<double>::quiet_NaN()});
  EXPECT_TRUE(fast_lio::fitPlane<5>(points, 0.125, fit));
}

TEST(PlaneFit, NumericalRankIsRelativeToNeighbourScale)
{
  for (double scale : {1e-3, 1.0, 1e3}) {
    for (double aspect : {1e-9, 1e-6}) {
      auto points = plane();
      for (auto & p : points) {
        p.x *= scale;
        p.y *= scale * aspect;
      }
      Eigen::Vector4d fit = Eigen::Vector4d::Constant(7);
      if (aspect == 1e-9) {
        expectRejected(points, 0.1);
      } else {
        EXPECT_TRUE(fast_lio::fitPlane<5>(points, 0.0, fit));
        EXPECT_EQ(fit, Eigen::Vector4d(0, 0, 1, 0));
      }
    }
  }
}

TEST(PlaneFit, NoisyOffsetPlaneUsesOrthogonalRatherThanFixedOffsetAlgebraicFit)
{
  auto points = plane(2.0);
  points[0].z += 0.125;
  points[3].z += 0.125;
  points[1].z -= 0.125;
  points[2].z -= 0.125;
  Eigen::Vector4d fit;
  ASSERT_TRUE(fast_lio::fitPlane<5>(points, 0.125, fit));
  EXPECT_NEAR((fit - Eigen::Vector4d(0, 0, -1, 2)).norm(), 0.0, 1e-14);
}

TEST(PlaneFit, RejectsInvalidInputWithoutPublishingPartialCoefficients)
{
  expectRejected({}, 0.1);
  auto fewer = plane();
  fewer.pop_back();
  expectRejected(fewer, 0.1);
  expectRejected(Points(5, {1, 2, 3}), 0.1);
  expectRejected({{0, 0, 0}, {1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}, 0.1);
  expectRejected(plane(), -0.1);
  for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()}) {
    expectRejected(plane(), bad);
    for (int axis = 0; axis < 3; ++axis) {
      auto points = plane();
      if (axis == 0)
        points[2].x = bad;
      if (axis == 1)
        points[2].y = bad;
      if (axis == 2)
        points[2].z = bad;
      expectRejected(points, 0.1);
    }
  }
  auto huge = plane();
  huge[0].x = std::numeric_limits<double>::max();
  expectRejected(huge, 0.1);
}

TEST(PlaneFit, ProductionWrapperUsesTheSameFivePointFitAndPreservesOutputOnFailure)
{
  PointVector points;
  for (const auto & p : plane()) {
    PointType point{};
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    points.push_back(point);
  }
  Eigen::Vector4f fit = Eigen::Vector4f::Constant(42.0f);
  ASSERT_TRUE(esti_plane(fit, points, 0.1f));
  EXPECT_EQ(fit, Eigen::Vector4f(0, 0, 1, 0));
  const auto before = fit;
  points[0].x = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(esti_plane(fit, points, 0.1f));
  EXPECT_EQ(fit, before);
}

TEST(PlaneFit, RejectsFiniteCoefficientsOutsideOutputRangeBeforeNarrowing)
{
  Points points = plane();
  for (auto & point : points) {
    point.z = point.x;
    point.x = 1e40;
  }
  const Eigen::Vector4f sentinel = Eigen::Vector4f::Constant(42.0f);
  Eigen::Vector4f fit = sentinel;
  EXPECT_FALSE(fast_lio::fitPlane<5>(points, 0.1, fit));
  EXPECT_EQ(fit, sentinel);
  Eigen::Vector4d wide;
  ASSERT_TRUE(fast_lio::fitPlane<5>(points, 0.1, wide));
  EXPECT_DOUBLE_EQ(wide[3], 1e40);
}

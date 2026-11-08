#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "cached_plane.hpp"

namespace
{
struct Point
{
  float x, y, z;
};
std::vector<Point> floorPoints()
{
  return {{-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0}, {0, 0, 0}};
}
}  // namespace

TEST(CachedPlane, RejectedResidualCanRecoverWithoutSearchingAgain)
{
  fast_lio::CachedPlane cache;
  ASSERT_TRUE(cache.fit<5>(floorPoints(), 0.1f, 1.0f));
  for (float z : {0.12f, 0.10f, 0.12f, 0.0f}) {
    fast_lio::CachedPlane fresh;
    ASSERT_TRUE(fresh.fit<5>(floorPoints(), 0.1f, 1.0f));
    const auto cached = cache.evaluate(0.0f, 0.0f, z);
    const auto searched = fresh.evaluate(0.0f, 0.0f, z);
    EXPECT_EQ(cached.accepted, searched.accepted);
    EXPECT_FLOAT_EQ(cached.distance, searched.distance);
    EXPECT_EQ(cached.accepted, z < 0.11f);
    EXPECT_TRUE(cache.valid());
  }
}

TEST(CachedPlane, FailedSearchInvalidatesTheFormerPlaneUntilAnotherSuccessfulFit)
{
  fast_lio::CachedPlane cache;
  ASSERT_TRUE(cache.fit<5>(floorPoints(), 0.1f, 1.0f));
  cache.invalidate();  // the production path does this before neighbour search
  EXPECT_FALSE(cache.evaluate(0, 0, 0).accepted);
  EXPECT_FALSE(cache.fit<5>(std::vector<Point>(5, {0, 0, 0}), 0.1f, 1.0f));
  EXPECT_FALSE(cache.valid());
  EXPECT_FALSE(cache.evaluate(0, 0, 0).accepted);
  ASSERT_TRUE(cache.fit<5>(floorPoints(), 0.1f, 1.0f));
  EXPECT_TRUE(cache.evaluate(0, 0, 0).accepted);
}

TEST(CachedPlane, CallerGatesCannotChangeTheCachedNeighbourhood)
{
  fast_lio::CachedPlane cache;
  ASSERT_TRUE(cache.fit<5>(floorPoints(), 0.1f, 1.0f));
  auto row = cache.evaluate(0, 0, 0.05f);
  ASSERT_TRUE(row.accepted);
  row.accepted = false;  // e.g. innovation or ground-selection pass this iteration
  EXPECT_TRUE(cache.evaluate(0, 0, 0.05f).accepted);
  cache.invalidate();  // a new scan must not inherit the old point's neighbourhood
  EXPECT_FALSE(cache.evaluate(0, 0, 0.05f).accepted);
}

TEST(CachedPlane, InvalidInputsCannotBecomeAdmittedRows)
{
  fast_lio::CachedPlane cache;
  EXPECT_FALSE(cache.evaluate(0, 0, 0).accepted);
  EXPECT_FALSE(cache.fit<5>(floorPoints(), 0.1f, 0.0f));
  EXPECT_FALSE(cache.fit<5>(floorPoints(), 0.1f, -1.0f));
  ASSERT_TRUE(cache.fit<5>(floorPoints(), 0.1f, 1.0f));
  EXPECT_FALSE(cache.evaluate(0, 0, std::numeric_limits<float>::quiet_NaN()).accepted);
  EXPECT_FALSE(cache.evaluate(0, 0, std::numeric_limits<float>::infinity()).accepted);
  EXPECT_TRUE(cache.evaluate(0, 0, 0).accepted);
}

TEST(CachedPlane, KeepsTheFitsEigenDataExactlyAsLongAsThePlane)
{
  fast_lio::CachedPlane cache;
  EXPECT_EQ(cache.eigen(), nullptr);  // nothing cached yet
  auto points = floorPoints();
  points[4].z = 0.1f;  // centre raised: lambda0 = 0.008, the in-plane pair 4 and 4
  ASSERT_TRUE(cache.fit<5>(points, 0.1f, 1.0f));
  ASSERT_NE(cache.eigen(), nullptr);
  Eigen::Vector4f coefficients;
  fast_lio::PlaneFitEigen direct;
  ASSERT_TRUE(fast_lio::fitPlane<5>(points, 0.1f, coefficients, &direct));
  EXPECT_EQ(cache.coefficients(), coefficients);
  EXPECT_EQ(cache.eigen()->lambda0, direct.lambda0);  // the same fit, not a recomputation
  EXPECT_EQ(cache.eigen()->lambda1, direct.lambda1);
  EXPECT_EQ(cache.eigen()->lambda2, direct.lambda2);
  EXPECT_EQ(cache.eigen()->e1, direct.e1);
  EXPECT_EQ(cache.eigen()->e2, direct.e2);
  EXPECT_EQ(cache.eigen()->count, 5);
  EXPECT_NEAR(cache.eigen()->lambda0, 0.008, 1e-8);  // float coordinates evaluated in double
  cache.invalidate();
  EXPECT_EQ(cache.eigen(), nullptr);
  ASSERT_TRUE(cache.fit<5>(points, 0.1f, 1.0f));
  EXPECT_FALSE(cache.fit<5>(points, 0.05f, 1.0f));  // a failed fit drops the old plane and its eigen data
  EXPECT_EQ(cache.eigen(), nullptr);
}

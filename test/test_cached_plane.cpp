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

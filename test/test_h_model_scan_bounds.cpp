#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "cached_plane.hpp"
#include "common_lib.h"
#include "use-ikfom.hpp"

// These are the actual component's scan buffers and model, not a copied model.
extern bool prepareScanCorrespondenceStorage(std::size_t count);
extern void h_share_model(state_ikfom &, esekfom::dyn_share_datastruct<double> &);
extern int feats_down_size, effct_feat_num, ekf_iters;
extern std::vector<std::uint8_t> point_selected_surf;
extern std::vector<float> res_last;
extern PointCloudXYZI::Ptr feats_down_body, feats_down_world, normvec, laserCloudOri, corr_normvect;
extern std::vector<PointVector> Nearest_Points;
extern std::vector<fast_lio::CachedPlane, Eigen::aligned_allocator<fast_lio::CachedPlane>> plane_cache;

namespace
{
class HModelScanBounds : public ::testing::TestWithParam<std::size_t>
{
protected:
  void prepare(std::size_t count, bool rows)
  {
    // The same production boundary called immediately after scan downsampling.
    // In particular, the test does not resize either formerly fixed array.
    ASSERT_TRUE(prepareScanCorrespondenceStorage(count));
    ASSERT_EQ(feats_down_size, static_cast<int>(count));
    ASSERT_EQ(point_selected_surf.size(), count);
    ASSERT_EQ(res_last.size(), count);
    feats_down_body->resize(count);
    feats_down_world->resize(count);
    normvec->resize(count);
    Nearest_Points.resize(count);
    PointVector plane;
    for (const auto & xy : std::vector<std::pair<float, float>>{
           {9.9f, -0.1f}, {10.1f, -0.1f}, {9.9f, 0.1f}, {10.1f, 0.1f}, {10.0f, 0.0f}}) {
      PointType point{};
      point.x = xy.first;
      point.y = xy.second;
      point.z = 1.0f;
      plane.push_back(point);
    }
    fast_lio::CachedPlane fitted;
    ASSERT_TRUE(fitted.fit<NUM_MATCH_POINTS>(plane, 0.1f, std::sqrt(std::sqrt(101.0f))));
    for (std::size_t i = 0; i < count; ++i) {
      PointType point{};
      point.x = 10.0f;
      point.z = 1.0f;
      point.intensity = static_cast<float>(i);
      (*feats_down_body)[i] = point;
      plane_cache[i] = fitted;
      if (!rows)
        plane_cache[i].invalidate();
    }
    ekf_iters = 0;
  }

  void evaluate(std::size_t count, bool rows)
  {
    state_ikfom state;
    esekfom::dyn_share_datastruct<double> measurement;
    measurement.converge = false;  // exercise the production cached-neighbour model
    measurement.valid = true;
    h_share_model(state, measurement);
    EXPECT_EQ(effct_feat_num, rows ? static_cast<int>(count) : 0);
    ASSERT_EQ(laserCloudOri->size(), rows ? count : 0u);
    ASSERT_EQ(corr_normvect->size(), rows ? count : 0u);
    EXPECT_EQ(point_selected_surf.back(), rows ? 1u : 0u);
    if (rows) {
      EXPECT_EQ(laserCloudOri->back().intensity, static_cast<float>(count - 1));
      EXPECT_NEAR(res_last.back(), 0.0f, 1e-5f);
      EXPECT_EQ(measurement.h_x.rows(), static_cast<int>(count));
      EXPECT_TRUE(measurement.h_x.allFinite());
      EXPECT_TRUE(measurement.h.allFinite());
    } else {
      EXPECT_FALSE(measurement.valid);
    }
  }
};

TEST_P(HModelScanBounds, RowsAndSecondIteration)
{
  const auto count = GetParam();
  prepare(count, true);
  evaluate(count, true);
  evaluate(count, true);
  EXPECT_EQ(ekf_iters, 2);
}

TEST_P(HModelScanBounds, NoRowsAfterPreviousPositiveScan)
{
  const auto count = GetParam();
  prepare(count, true);
  evaluate(count, true);
  prepare(count, false);
  evaluate(count, false);
  evaluate(count, false);
}

TEST_F(HModelScanBounds, LargeSmallLargeUsesProductionPreparation)
{
  for (const std::size_t count : {100001u, 99999u, 100000u, 100001u}) {
    prepare(count, true);
    evaluate(count, true);
  }
}

INSTANTIATE_TEST_SUITE_P(FormerFixedArrayBoundary, HModelScanBounds, ::testing::Values(99999u, 100000u, 100001u));
}  // namespace

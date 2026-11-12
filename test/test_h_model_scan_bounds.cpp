#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>

#include "frontend_measurement.hpp"

#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

// The component's measurement model on a map, a scan and per-scan values this fixture owns. A scan's first
// evaluation searches the map, as the filter's first iteration does; a repeated one reuses the planes it cached.
namespace
{
class HModelScanBounds : public ::testing::TestWithParam<std::size_t>
{
protected:
  HModelScanBounds()
  {
    // A flat patch at z = 1 around (10, 0), 0.1 m apart: the neighbours of (10, 0, 1) fit a plane through it.
    PointVector patch;
    for (int i = -10; i <= 10; ++i) {
      for (int j = -10; j <= 10; ++j) {
        PointType point{};
        point.x = 10.0f + 0.1f * static_cast<float>(i);
        point.y = 0.1f * static_cast<float>(j);
        point.z = 1.0f;
        patch.push_back(point);
      }
    }
    map_.set_downsample_param(0.05f);
    map_.Build(patch);
    prior_.setIdentity();
  }

  // The node's order for a new scan: the scan reset, then the storage sized from the downsampled count. The scan's
  // points lie on the patch, or (rows = false) far from every map point. The floor is off: no wheel fit is asked.
  void prepare(std::size_t count, bool rows)
  {
    measurement_.beginScan(false);
    ASSERT_TRUE(measurement_.prepareScan(count, point_count_));
    ASSERT_EQ(point_count_, static_cast<int>(count));
    body_.resize(count);
    world_.resize(count);
    measurement_.resizeNormals(count);
    measurement_.resizeNeighbours(count);
    ASSERT_EQ(measurement_.nearestPoints().size(), count);
    for (std::size_t i = 0; i < count; ++i) {
      PointType point{};
      point.x = rows ? 10.0f : 1000.0f;
      point.z = 1.0f;
      point.intensity = static_cast<float>(i);
      body_[i] = point;
    }
    iterations_ = 0;
  }

  void evaluate(std::size_t count, bool rows, bool search)
  {
    esekfom::dyn_share_datastruct<double> data;
    data.converge = search;
    data.valid = true;
    const fast_lio::MeasurementInputs inputs{
      prior_, map_, body_, world_, point_count_, scan_end_time_, 0.0, false, false, false};
    const fast_lio::MeasurementOutputs outputs{
      effective_rows_, iterations_, mean_residual_, false, dump_iteration_, dump_rows_, dump_iterations_};
    measurement_.evaluate(state_, data, inputs, outputs);
    EXPECT_EQ(effective_rows_, rows ? static_cast<int>(count) : 0);
    ASSERT_EQ(measurement_.effectCloud().size(), rows ? count : 0u);
    if (rows) {
      EXPECT_EQ(measurement_.effectCloud().back().intensity, static_cast<float>(count - 1));
      EXPECT_NEAR(mean_residual_, 0.0, 1e-5);
      EXPECT_EQ(data.h_x.rows(), static_cast<int>(count));
      EXPECT_TRUE(data.h_x.allFinite());
      EXPECT_TRUE(data.h.allFinite());
    } else {
      EXPECT_FALSE(data.valid);
    }
  }

  const fast_lio::FrontendParams params_{};  // the defaults, as a node that read no parameter holds them
  fast_lio::FrontendMeasurement measurement_{params_};
  MapBackend map_;
  fast_lio::PriorCovariance prior_;
  PointCloudXYZI body_, world_;
  state_ikfom state_;
  int point_count_ = 0, effective_rows_ = 0, iterations_ = 0, dump_iteration_ = 0;
  double mean_residual_ = 0.0, scan_end_time_ = 1.0;
  std::ofstream dump_rows_, dump_iterations_;
};

TEST_P(HModelScanBounds, RowsAndSecondIteration)
{
  const auto count = GetParam();
  prepare(count, true);
  evaluate(count, true, true);
  evaluate(count, true, false);
  EXPECT_EQ(iterations_, 2);
}

TEST_P(HModelScanBounds, NoRowsAfterPreviousPositiveScan)
{
  const auto count = GetParam();
  prepare(count, true);
  evaluate(count, true, true);
  prepare(count, false);
  evaluate(count, false, true);
  evaluate(count, false, false);
}

TEST_F(HModelScanBounds, LargeSmallLargeUsesProductionPreparation)
{
  for (const std::size_t count : {100001u, 99999u, 100000u, 100001u}) {
    prepare(count, true);
    evaluate(count, true, true);
  }
}

INSTANTIATE_TEST_SUITE_P(FormerFixedArrayBoundary, HModelScanBounds, ::testing::Values(99999u, 100000u, 100001u));
}  // namespace

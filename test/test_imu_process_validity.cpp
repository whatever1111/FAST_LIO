#include <gtest/gtest.h>
#include <omp.h>

#include "IMU_Processing.hpp"
#include "reanchor_gate.hpp"

namespace
{
MeasureGroup scan(double begin)
{
  MeasureGroup result;
  result.lidar_beg_time = begin;
  result.lidar_end_time = begin + 0.1;
  result.lidar = std::make_shared<PointCloudXYZI>();
  PointType point{};
  point.x = 1.0f;
  point.curvature = 100.0f;
  result.lidar->push_back(point);
  return result;
}
void addImu(MeasureGroup & scan, double stamp)
{
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(stamp * 1e9)));
  imu->linear_acceleration.z = 9.81;
  scan.imu.push_back(imu);
}
}  // namespace

TEST(ImuProcessValidity, EmptyInputClearsPreviouslyPopulatedOutputWithoutChangingState)
{
  ImuProcess process;
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  auto output = std::make_shared<PointCloudXYZI>();
  output->push_back(PointType{});
  const auto before = filter.get_x().pos;
  EXPECT_FALSE(process.Process(scan(10.0), filter, output));
  EXPECT_TRUE(output->empty());
  EXPECT_EQ(filter.get_x().pos, before);
}

TEST(ImuProcessValidity, InitializationCommitsTimeAndEqualEndNeverReusesOutput)
{
  ImuProcess process;
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  auto input = scan(10.0);
  for (int i = 0; i <= 20; ++i)
    addImu(input, 10.0 + i * 0.005);
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), input.lidar_end_time);
  output->push_back(PointType{});
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_TRUE(output->empty());
  process.Reset();
  EXPECT_LT(process.lastProcessedEnd(), 0.0);
}

TEST(ImuProcessValidity, SuccessfulScanThenDuplicateAndGapCannotReuseOrPropagateOldOutput)
{
  ImuProcess process;
  process.coverage_params = {0.1, 0.01};
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  double epsilon[23];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  filter.init_dyn_share(get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  auto output = std::make_shared<PointCloudXYZI>();
  auto init = scan(10.0);
  for (int i = 0; i < 20; ++i)
    addImu(init, 10.0 + i * 0.005);
  EXPECT_FALSE(process.Process(init, filter, output));
  auto good = scan(10.1);
  for (int i = 0; i < 20; ++i)
    addImu(good, 10.1 + i * 0.005);
  ASSERT_TRUE(process.Process(good, filter, output));
  ASSERT_FALSE(output->empty());
  const auto state_before = filter.get_x();
  good.imu.clear();
  EXPECT_FALSE(process.Process(good, filter, output));
  EXPECT_TRUE(output->empty());
  EXPECT_EQ(filter.get_x().pos, state_before.pos);

  auto gap = scan(14.1);
  for (int i = 0; i < 20; ++i)
    addImu(gap, 14.1 + i * 0.005);
  EXPECT_FALSE(process.Process(gap, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kCoverageGap);
  EXPECT_TRUE(output->empty());
  EXPECT_EQ(filter.get_x().pos, state_before.pos);
  EXPECT_EQ(filter.get_x().vel, state_before.vel);
  EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), gap.lidar_end_time);

  auto recovery = scan(14.2);
  for (int i = 0; i < 20; ++i)
    addImu(recovery, 14.2 + i * 0.005);
  EXPECT_TRUE(process.Process(recovery, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kProcessed);
  EXPECT_TRUE(filter.get_x().pos.allFinite());
  EXPECT_NEAR(filter.get_x().pos.norm(), 0.0, 0.0001);
}

TEST(ImuProcessValidity, CoverageRecoveryRequiresThreeMatchesAgainstFrozenMap)
{
  fast_lio::ReanchorGate gate;
  fast_lio::ReanchorParams params;
  const auto dropped = fast_lio::updateReanchorGate(&gate, true, 0, 0.0, 14.2, params);
  EXPECT_TRUE(dropped.map_frozen);
  EXPECT_TRUE(dropped.degraded);
  for (int i = 0; i < 2; ++i) {
    const auto confirming = fast_lio::updateReanchorGate(&gate, false, 300, 0.01, 14.3 + i * 0.1, params);
    EXPECT_TRUE(confirming.map_frozen);
    EXPECT_TRUE(confirming.degraded);
  }
  const auto recovered = fast_lio::updateReanchorGate(&gate, false, 300, 0.01, 14.5, params);
  EXPECT_TRUE(recovered.just_reanchored);
  EXPECT_FALSE(recovered.degraded);
}

TEST(ImuProcessValidity, ShortScanWithOnlyOldImuDoesNotIntegrateCommittedTimeTwice)
{
  ImuProcess process;
  process.coverage_params = {0.1, 0.01};
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  double epsilon[23];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  filter.init_dyn_share(get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  auto output = std::make_shared<PointCloudXYZI>();
  auto init = scan(10.0);
  for (int i = 0; i < 20; ++i)
    addImu(init, 10.0 + i * 0.005);
  EXPECT_FALSE(process.Process(init, filter, output));
  auto state = filter.get_x();
  state.vel = V3D(1.0, 0.0, 0.0);
  filter.change_x(state);
  auto short_scan = scan(10.1);
  short_scan.lidar->points[0].curvature = 2.0f;
  short_scan.lidar_end_time = short_scan.lidar_beg_time + 0.002;
  addImu(short_scan, 10.095);  // repeated retained sample, older than committed end
  ASSERT_TRUE(process.Process(short_scan, filter, output));
  EXPECT_NEAR(filter.get_x().pos.x(), state.pos.x() + 0.002, 1e-12);
}

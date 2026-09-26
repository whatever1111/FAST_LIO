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
void addImu(MeasureGroup & scan, double stamp, const V3D & acc, const V3D & gyro)
{
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(stamp * 1e9)));
  imu->linear_acceleration.x = acc.x();
  imu->linear_acceleration.y = acc.y();
  imu->linear_acceleration.z = acc.z();
  imu->angular_velocity.x = gyro.x();
  imu->angular_velocity.y = gyro.y();
  imu->angular_velocity.z = gyro.z();
  scan.imu.push_back(imu);
}

// An initialised filter at rest (world frame = IMU frame), then moving at 1 m/s along x with a small covariance:
// the state an IMU gap interrupts.
struct GapFixture
{
  using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;
  ImuProcess process;
  Filter filter;
  PointCloudXYZI::Ptr output = std::make_shared<PointCloudXYZI>();

  explicit GapFixture(bool prior)
  {
    process.coverage_params = {2.0, 0.1};
    process.gap_params.enabled = prior;
    process.gap_params.min_gap_s = 0.05;  // the 80 ms stretches below are gaps
    process.set_acc_cov(V3D::Constant(0.1));
    process.set_gyr_cov(V3D::Constant(0.1));
    double epsilon[23];
    std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
    filter.init_dyn_share(
      get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
    auto init = scan(10.0);
    for (int i = 0; i < 20; ++i)
      addImu(init, 10.0 + i * 0.005);
    EXPECT_FALSE(process.Process(init, filter, output));
    auto state = filter.get_x();
    state.vel = V3D(1.0, 0.0, 0.0);
    filter.change_x(state);
    Filter::cov P = Filter::cov::Identity() * 1e-4;
    filter.change_P(P);
  }
};

// The scan (10.6, 10.7] after an IMU gap from the last initialisation sample (10.095) to 10.6: the sample that ends
// the gap carries a 5 m/s^2 gait impulse along x and a 1 rad/s yaw rate, the rest of the scan is at rest.
MeasureGroup scanAfterGap()
{
  auto after = scan(10.6);
  addImu(after, 10.6, V3D(5.0, 0.0, 9.81), V3D(0.0, 0.0, 1.0));
  for (int i = 1; i <= 20; ++i)
    addImu(after, 10.6 + i * 0.005);
  return after;
}

double yawOf(const state_ikfom & s)
{
  const M3D R = s.rot.toRotationMatrix();
  return std::atan2(R(1, 0), R(0, 0));
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

TEST(ImuProcessValidity, GapPriorHoldsTheVelocityRotatesWithTheGyroAndInflatesTheCovariance)
{
  GapFixture f(true);
  ASSERT_TRUE(f.process.Process(scanAfterGap(), f.filter, f.output));
  const auto x = f.filter.get_x();
  // The gait impulse is not integrated over the 0.5 s gap: only its own 5 ms segment adds 2.5 * 0.005 m/s.
  EXPECT_NEAR((x.vel - V3D(1.0, 0.0, 0.0)).norm(), 0.0, 0.02);
  // ...while the gyro average (0.5 rad/s over 0.5 s, then 5 ms more) still turns the state.
  EXPECT_NEAR(yawOf(x), 0.5 * 0.5 + 0.5 * 0.005, 1e-3);
  // Constant velocity from 10.1 to 10.7.
  EXPECT_NEAR(x.pos.x(), 0.6, 0.01);
  EXPECT_NEAR(x.pos.z(), 0.0, 1e-4);  // S2 gravity is 9.809 against G_m_s2 9.81: 1 mm/s^2 at rest
  const auto & summary = f.process.gap_summary;
  EXPECT_EQ(summary.gaps, 1);
  EXPECT_NEAR(summary.longest_s, 0.5, 1e-6);
  // P carries the gap: (a T)^2 = 0.25 horizontally, (0.2 T)^2 = 0.01 vertically, (0.5 a T^2)^2 on position and the
  // yaw rate's (10 deg/s T)^2 = 0.0076, on top of IKFoM's T^2 Q = 0.025 and the 1e-4 start (without the prior:
  // about 0.028 / 0.025 / 0.0003 / 0.025). The 0.1 s after the gap adds a few 0.01 to the horizontal velocity
  // through this test's large gyro noise (0.025 rad^2 of attitude times g).
  const auto P = f.filter.get_P();
  EXPECT_GT(P(12, 12), 0.25);
  EXPECT_LT(P(12, 12), 0.35);
  EXPECT_GT(P(14, 14), 0.03);
  EXPECT_LT(P(14, 14), 0.04);
  EXPECT_GT(P(0, 0), 0.0156);
  EXPECT_GT(P(5, 5), 0.03);
  EXPECT_TRUE(P.allFinite());
}

TEST(ImuProcessValidity, WithoutTheGapPriorTheBracketingImpulseBecomesVelocity)
{
  // The upstream single-step integration this prior replaces: the mean of the two bracketing samples (2.5 m/s^2)
  // is integrated for the whole 0.5 s, the 0826 failure in miniature.
  GapFixture f(false);
  ASSERT_TRUE(f.process.Process(scanAfterGap(), f.filter, f.output));
  EXPECT_NEAR(f.filter.get_x().vel.x(), 1.0 + 1.25, 0.05);
  EXPECT_EQ(f.process.gap_summary.gaps, 0);
}

TEST(ImuProcessValidity, GapPriorDeskewsFromWhereThePlatformWasAtTheScanStart)
{
  // The IMU resumes 50 ms into the scan (10.65), so nothing covers 10.1..10.65: the first deskew pose must be the
  // state carried to the scan start (x = 0.5), not the one left at the previous scan's end (x = 0).
  const auto make_scan = []() {
    auto late = scan(10.6);
    late.lidar->points[0].curvature = 20.0f;  // one point at 20 ms, 1 m ahead
    for (int i = 0; i <= 10; ++i)
      addImu(late, 10.65 + i * 0.005);
    return late;
  };
  GapFixture with(true);
  ASSERT_TRUE(with.process.Process(make_scan(), with.filter, with.output));
  ASSERT_EQ(with.output->size(), 1u);
  EXPECT_NEAR(with.filter.get_x().pos.x(), 0.6, 1e-3);
  // observed at x = 0.52 + 1, expressed in the scan-end frame at x = 0.6
  EXPECT_NEAR(with.output->points[0].x, 0.92, 2e-3);
  EXPECT_NEAR(with.process.gap_summary.longest_s, 0.55, 1e-6);

  GapFixture without(false);
  ASSERT_TRUE(without.process.Process(make_scan(), without.filter, without.output));
  ASSERT_EQ(without.output->size(), 1u);
  EXPECT_NEAR(without.output->points[0].x, 0.42, 2e-3);  // the point is placed 0.5 m behind where it was seen
}

TEST(ImuProcessValidity, GapPriorHoldsTheVelocityPastTheLastSampleOfAScan)
{
  GapFixture f(true);
  auto early = scan(10.1);
  for (int i = 0; i < 4; ++i)
    addImu(early, 10.1 + i * 0.005);
  addImu(early, 10.12, V3D(3.0, 0.0, 9.81), V3D::Zero());  // then nothing until after the scan end (80 ms)
  ASSERT_TRUE(f.process.Process(early, f.filter, f.output));
  // Only the 5 ms segment ending at the impulse sees it (1.5 m/s^2 mean); upstream would extrapolate it 80 ms more.
  EXPECT_NEAR(f.filter.get_x().vel.x(), 1.0 + 1.5 * 0.005, 1e-3);
  EXPECT_EQ(f.process.gap_summary.gaps, 1);
  EXPECT_NEAR(f.process.gap_summary.longest_s, 0.08, 1e-6);
}

TEST(ImuProcessValidity, GapBeyondTheCoverageLimitIsStillSkippedWithThePriorOn)
{
  GapFixture f(true);
  const auto before = f.filter.get_x();
  auto late = scan(13.2);
  for (int i = 0; i < 20; ++i)
    addImu(late, 13.2 + i * 0.005);  // 3.1 s after the last sample, beyond imu_coverage.max_gap_s = 2
  EXPECT_FALSE(f.process.Process(late, f.filter, f.output));
  EXPECT_EQ(f.process.lastStatus(), ImuProcess::ProcessStatus::kCoverageGap);
  EXPECT_EQ(f.filter.get_x().pos, before.pos);
  EXPECT_EQ(f.filter.get_x().vel, before.vel);
  EXPECT_EQ(f.process.gap_summary.gaps, 0);
}

TEST(ImuProcessValidity, GapPriorCarriesAScanWithoutImuSamplesWhenAsked)
{
  // The lidar keeps recording through an IMU dropout (m20 0826: nine scans between 352.1 and 353.0 s had no IMU
  // sample). With imu_gap_empty_scans such a scan is propagated at the held velocity and the last angular rate
  // instead of being dropped unregistered.
  for (const bool carry : {true, false}) {
    GapFixture f(true);
    f.process.gap_params.empty_scans = carry;
    auto turning = scan(10.1);
    for (int i = 0; i < 20; ++i)
      addImu(turning, 10.1 + i * 0.005, V3D(0.0, 0.0, 9.81), V3D(0.0, 0.0, 0.5));
    ASSERT_TRUE(f.process.Process(turning, f.filter, f.output));
    EXPECT_NEAR(yawOf(f.filter.get_x()), 0.05, 1e-3);
    const auto before = f.filter.get_x();
    EXPECT_EQ(f.process.bridgesEmptyScan(10.3), carry);
    const bool processed = f.process.Process(scan(10.2), f.filter, f.output);
    ASSERT_EQ(processed, carry);
    if (!carry) {
      EXPECT_EQ(f.filter.get_x().pos, before.pos);  // dropped as before
      EXPECT_EQ(f.process.gap_summary.gaps, 0);
      continue;
    }
    const auto x = f.filter.get_x();
    EXPECT_FALSE(f.output->empty());
    EXPECT_NEAR((x.vel - V3D(1.0, 0.0, 0.0)).norm(), 0.0, 1e-3);  // held
    EXPECT_NEAR(x.pos.x(), 0.2, 2e-3);                            // 10.1 -> 10.3 at 1 m/s
    EXPECT_NEAR(yawOf(x), 0.10, 1e-3);                            // the last rate, held for 0.1 s
    EXPECT_EQ(f.process.gap_summary.gaps, 1);
    EXPECT_NEAR(f.process.gap_summary.longest_s, 0.1, 1e-6);
    EXPECT_NEAR(f.process.gap_summary.longest_interval_s, 0.105, 1e-6);  // since the last sample at 10.195
  }
}

TEST(ImuProcessValidity, EmptyScansAreCarriedOnlyInsideTheCoverageLimit)
{
  GapFixture f(true);
  f.process.gap_params.empty_scans = true;
  EXPECT_TRUE(f.process.bridgesEmptyScan(10.095 + 1.9));
  EXPECT_FALSE(f.process.bridgesEmptyScan(10.095 + 2.1));  // imu_coverage.max_gap_s = 2
  EXPECT_FALSE(f.process.bridgesEmptyScan(10.0));          // not after the last sample
  const auto before = f.filter.get_x();
  EXPECT_FALSE(f.process.Process(scan(12.2), f.filter, f.output));
  EXPECT_EQ(f.filter.get_x().pos, before.pos);
  ImuProcess uninitialised;
  uninitialised.gap_params.enabled = true;
  uninitialised.gap_params.empty_scans = true;
  EXPECT_FALSE(uninitialised.bridgesEmptyScan(1.0));
}

TEST(ImuProcessValidity, GapPriorExtrapolatesOnlyTheRotationAboutGravity)
{
  // The last samples roll at 0.5 rad/s and turn at 0.5 rad/s, then the IMU stops 80 ms before the scan end: the turn
  // is carried on, the gait's roll rate is not.
  GapFixture f(true);
  auto rocking = scan(10.1);
  for (int i = 0; i <= 4; ++i)
    addImu(rocking, 10.1 + i * 0.005, V3D(0.0, 0.0, 9.81), V3D(0.5, 0.0, 0.5));
  ASSERT_TRUE(f.process.Process(rocking, f.filter, f.output));
  const M3D R = f.filter.get_x().rot.toRotationMatrix();
  EXPECT_NEAR(std::atan2(R(2, 1), R(2, 2)), 0.5 * 0.02, 1e-3);  // rolled only while samples were there
  EXPECT_NEAR(yawOf(f.filter.get_x()), 0.5 * 0.1, 2e-3);        // turned for the whole scan
  EXPECT_NEAR(f.process.gap_summary.longest_s, 0.08, 1e-6);
}

#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <omp.h>
#include <stdexcept>

#include "IMU_Processing.hpp"
#include "imu_initialization_validity.hpp"
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

  explicit GapFixture(bool prior, double initializationBegin = 10.0)
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
    auto init = scan(initializationBegin);
    for (int i = 0; i < 20; ++i)
      addImu(init, initializationBegin + i * 0.005);
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

struct StaticInitFixture
{
  ImuProcess process;
  GapFixture::Filter filter;
  PointCloudXYZI::Ptr output = std::make_shared<PointCloudXYZI>();

  StaticInitFixture()
  {
    process.init_require_still = true;
    process.init_static_params.enabled = true;
    process.init_still_timeout_s = 0.5;
    process.set_acc_cov(V3D::Constant(0.1));
    process.set_gyr_cov(V3D::Constant(0.1));
    double epsilon[23];
    std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
    filter.init_dyn_share(
      get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  }

  MeasureGroup input(int frame, bool moving = false, double origin = 10.0)
  {
    // Match the IMU stamp's integer-nanosecond conversion. begin + 0.1 can
    // otherwise put the scan end one ULP before its final IMU (frame 12).
    constexpr int64_t kFrameNs = 100000000;
    const int64_t beginNs = static_cast<int64_t>(std::llround(origin * 1e9)) + frame * kFrameNs;
    auto result = scan(rclcpp::Time(beginNs).seconds());
    result.lidar_end_time = rclcpp::Time(beginNs + kFrameNs).seconds();
    for (int i = frame == 0 ? 0 : 1; i <= 20; ++i) {
      const double x = moving ? (i % 2 == 0 ? 1.0 : -1.0) : 0.0;
      addImu(result, origin + (frame * 20 + i) * 0.005, V3D(x, 0.0, 9.81), V3D(0.0, 0.0, moving ? 0.2 : 0.0));
    }
    return result;
  }
};
}  // namespace

TEST(ImuProcessValidity, T13SustainedStillInitializesAtFirstOneSecondFrameEnd)
{
  StaticInitFixture f;
  f.process.init_still_timeout_s = 90.0;
  int completeFrame = -1;
  for (int frame = 0; frame < 15; ++frame) {
    const bool processed = f.process.Process(f.input(frame), f.filter, f.output);
    if (frame < 9) {
      EXPECT_FALSE(processed);
      EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationAccumulated);
    } else if (frame == 9) {
      EXPECT_FALSE(processed);
      EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationComplete);
      completeFrame = frame;
    } else {
      EXPECT_TRUE(processed) << "frame=" << frame
                             << " reason=" << fast_lio::imuProcessReasonName(f.process.lastOutcome().reason);
    }
    EXPECT_FALSE(f.process.initializationDegraded());
  }
  EXPECT_EQ(completeFrame, 9);
}

TEST(ImuProcessValidity, T14MovingTimeoutCompletesAndDegradationSurvivesReset)
{
  StaticInitFixture f;
  for (int frame = 0; frame <= 6; ++frame) {
    EXPECT_FALSE(f.process.Process(f.input(frame, true), f.filter, f.output));
    EXPECT_EQ(f.process.lastOutcome().reason,
              frame <= 5 ? fast_lio::ImuProcessReason::kInitializationMotion
                         : fast_lio::ImuProcessReason::kInitializationComplete);
    EXPECT_EQ(f.process.initializationDegraded(), frame == 6);
  }
  // Timeout retains the current batch's legacy mean/bias and covariance calculation.
  EXPECT_DOUBLE_EQ(f.filter.get_x().bg.z(), 0.2);
  const S2 expectedGravity(V3D(0.0, 0.0, -G_m_s2));
  EXPECT_DOUBLE_EQ(f.filter.get_x().grav.get_vect().z(), expectedGravity.get_vect().z());
  EXPECT_EQ(f.process.cov_acc, V3D::Constant(0.1));
  f.process.Reset();
  EXPECT_TRUE(f.process.initializationDegraded());
  f.process.init_still_timeout_s = 90.0;
  for (int frame = 0; frame < 10; ++frame)
    EXPECT_FALSE(f.process.Process(f.input(frame, false, 20.0), f.filter, f.output));
  EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationComplete);
  EXPECT_TRUE(f.process.initializationDegraded());
}

TEST(ImuProcessValidity, T15DisabledGateRetainsLegacyMovingStreamStateBitwise)
{
  // Both disabled configurations run the old accumulator and timeout path.
  // With require_still=false the first moving batch completes, as at b460bc35.
  StaticInitFixture a, b;
  a.process.init_require_still = b.process.init_require_still = false;
  a.process.init_static_params.enabled = false;
  b.process.init_static_params.enabled = true;
  for (int frame = 0; frame < 3; ++frame) {
    auto input = a.input(frame, true);
    EXPECT_EQ(a.process.Process(input, a.filter, a.output), b.process.Process(input, b.filter, b.output));
    EXPECT_EQ(a.process.lastOutcome().reason, b.process.lastOutcome().reason);
    const auto ax = a.filter.get_x(), bx = b.filter.get_x();
    const auto ap = a.filter.get_P(), bp = b.filter.get_P();
    EXPECT_EQ(std::memcmp(ax.pos.data(), bx.pos.data(), sizeof(double) * 3), 0);
    const M3D ar = ax.rot.toRotationMatrix(), br = bx.rot.toRotationMatrix();
    EXPECT_EQ(std::memcmp(ar.data(), br.data(), sizeof(double) * 9), 0);
    EXPECT_EQ(std::memcmp(ax.vel.data(), bx.vel.data(), sizeof(double) * 3), 0);
    EXPECT_EQ(std::memcmp(ax.bg.data(), bx.bg.data(), sizeof(double) * 3), 0);
    EXPECT_EQ(std::memcmp(ap.data(), bp.data(), sizeof(double) * ap.size()), 0);
    EXPECT_EQ(std::memcmp(a.process.cov_acc.data(), b.process.cov_acc.data(), sizeof(double) * 3), 0);
    EXPECT_FALSE(a.process.initializationDegraded());
    EXPECT_FALSE(b.process.initializationDegraded());
    if (frame == 0) {
      EXPECT_EQ(a.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationComplete);
      EXPECT_DOUBLE_EQ(ax.bg.z(), 0.2);
    }
  }
  StaticInitFixture legacy;
  legacy.process.init_static_params.enabled = false;
  for (int frame = 0; frame <= 6; ++frame) {
    EXPECT_FALSE(legacy.process.Process(legacy.input(frame, true), legacy.filter, legacy.output));
    EXPECT_EQ(legacy.process.lastOutcome().reason,
              frame <= 5 ? fast_lio::ImuProcessReason::kInitializationMotion
                         : fast_lio::ImuProcessReason::kInitializationComplete);
    EXPECT_FALSE(legacy.process.initializationDegraded());
  }
}

TEST(ImuProcessValidity, SustainedInvalidBatchDiscardsTheSameGravityWindow)
{
  StaticInitFixture f;
  f.process.init_still_timeout_s = 90.0;
  EXPECT_FALSE(f.process.Process(f.input(0), f.filter, f.output));
  auto bad = f.input(1);
  auto corrupt = std::make_shared<sensor_msgs::msg::Imu>(*bad.imu.front());
  corrupt->linear_acceleration.x = std::numeric_limits<double>::quiet_NaN();
  bad.imu.front() = corrupt;
  EXPECT_FALSE(f.process.Process(bad, f.filter, f.output));
  EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInvalidImu);
  for (int frame = 2; frame <= 11; ++frame) {
    EXPECT_FALSE(f.process.Process(f.input(frame), f.filter, f.output));
    EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationAccumulated);
  }
  EXPECT_FALSE(f.process.Process(f.input(12), f.filter, f.output));
  EXPECT_EQ(f.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationComplete);
}

TEST(ImuProcessValidity, EmptyInputClearsPreviouslyPopulatedOutputWithoutChangingState)
{
  ImuProcess process;
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  auto output = std::make_shared<PointCloudXYZI>();
  output->push_back(PointType{});
  const auto before = filter.get_x().pos;
  EXPECT_FALSE(process.Process(scan(10.0), filter, output));
  EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kNoImu);
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
  EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationComplete);
  output->push_back(PointType{});
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_TRUE(output->empty());
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kNonadvancing);
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
  EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kProcessed);
  EXPECT_TRUE(process.lastOutcome().producedOutput);
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
  EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kCoverageGap);
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
  // The ordinary integrator now timestamps its seed at the real previous
  // state epoch too. Constant velocity is deskewed correctly in either mode.
  EXPECT_NEAR(without.output->points[0].x, 0.92, 2e-3);
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

namespace
{
using InitFilter = esekfom::esekf<state_ikfom, 12, input_ikfom>;

void expectUnchangedInitialization(const InitFilter & filter,
                                   const state_ikfom & before,
                                   const InitFilter::cov & covariance)
{
  const auto after = filter.get_x();
  EXPECT_TRUE(fast_lio::finiteInitializationState(after));
  EXPECT_EQ(after.pos, before.pos);
  EXPECT_EQ(after.vel, before.vel);
  EXPECT_EQ(after.bg, before.bg);
  EXPECT_EQ(after.ba, before.ba);
  EXPECT_EQ(after.offset_T_L_I, before.offset_T_L_I);
  EXPECT_EQ(after.rot.toRotationMatrix(), before.rot.toRotationMatrix());
  EXPECT_EQ(after.offset_R_L_I.toRotationMatrix(), before.offset_R_L_I.toRotationMatrix());
  EXPECT_EQ(after.grav.get_vect(), before.grav.get_vect());
  EXPECT_TRUE(filter.get_P().allFinite());
  EXPECT_EQ(filter.get_P(), covariance);
}

void configureInitialization(ImuProcess & process, InitFilter & filter)
{
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  process.coverage_params = {0.2, 0.01};
  double epsilon[23];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  filter.init_dyn_share(get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  // A nondefault covariance detects an otherwise silent partial init commit.
  InitFilter::cov covariance = InitFilter::cov::Identity() * 0.123;
  filter.change_P(covariance);
}
}  // namespace

TEST(ImuProcessValidity, InvalidGravityWindowCannotCommitOrCompleteRegardlessOfStillnessAndTimeout)
{
  for (const bool still : {false, true}) {
    for (const double timeout : {20.0, -1.0}) {
      for (const double acceleration : {0.0, 1e-9, 1e-6, 1e308}) {
        SCOPED_TRACE(::testing::Message() << still << "/" << timeout << "/" << acceleration);
        ImuProcess process;
        InitFilter filter;
        configureInitialization(process, filter);
        process.init_require_still = still;
        process.init_still_timeout_s = timeout;
        const auto before = filter.get_x();
        const auto covariance = filter.get_P();
        auto output = std::make_shared<PointCloudXYZI>();
        auto bad = scan(10.0);
        for (int i = 0; i < 20; ++i)
          addImu(bad, 10.0 + i * 0.005, V3D(0, 0, acceleration), V3D::Zero());
        EXPECT_FALSE(process.Process(bad, filter, output));
        EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
        EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
        EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationInvalidMean);
        EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), bad.lidar_end_time);
        EXPECT_TRUE(output->empty());
        expectUnchangedInitialization(filter, before, covariance);
        // Invalid windows are consumed, and neither replay nor an older scan may
        // be counted towards a replacement window.
        EXPECT_FALSE(process.Process(bad, filter, output));
        EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
        EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), bad.lidar_end_time);
        auto older = scan(9.0);
        addImu(older, 9.0);
        EXPECT_FALSE(process.Process(older, filter, output));
        expectUnchangedInitialization(filter, before, covariance);

        auto one = scan(10.1);
        addImu(one, 10.195);
        EXPECT_FALSE(process.Process(one, filter, output));
        EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
        EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), one.lidar_end_time);
        // A second single sample must still initialize. If the 20 rejected
        // samples survived in N, the first good sample would finish init and
        // this call would propagate instead.
        auto second = scan(10.2);
        addImu(second, 10.295);
        EXPECT_FALSE(process.Process(second, filter, output));
        EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
        auto fresh = scan(10.3);
        for (int i = 0; i < 20; ++i)
          addImu(fresh, 10.3 + i * 0.005);
        EXPECT_FALSE(process.Process(fresh, filter, output));
        EXPECT_TRUE(fast_lio::finiteInitializationState(filter.get_x()));
        EXPECT_TRUE(filter.get_P().allFinite());
        auto propagation = scan(10.4);
        for (int i = 0; i < 20; ++i)
          addImu(propagation, 10.4 + i * 0.005);
        EXPECT_TRUE(process.Process(propagation, filter, output));
        EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kProcessed);
        EXPECT_FALSE(output->empty());
        EXPECT_TRUE(fast_lio::finiteInitializationState(filter.get_x()));
        EXPECT_TRUE(filter.get_P().allFinite());
      }
    }
  }
}

TEST(ImuProcessValidity, OppositeFiniteAccelerationsCannotSupplyAGravityDirection)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  const auto before = filter.get_x();
  const auto covariance = filter.get_P();
  auto input = scan(10.0);
  addImu(input, 10.0, V3D(0, 0, 9.81), V3D::Zero());
  addImu(input, 10.05, V3D(0, 0, -9.81), V3D::Zero());
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  expectUnchangedInitialization(filter, before, covariance);
}

TEST(ImuProcessValidity, NonFiniteMessagesRemainRejectedWithoutChangingFilter)
{
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    for (const bool gyro : {false, true}) {
      ImuProcess process;
      InitFilter filter;
      configureInitialization(process, filter);
      const auto before = filter.get_x();
      const auto covariance = filter.get_P();
      auto input = scan(10.0);
      addImu(input, 10.0, gyro ? V3D(0, 0, 9.81) : V3D(0, 0, bad), gyro ? V3D(bad, 0, 0) : V3D::Zero());
      auto output = std::make_shared<PointCloudXYZI>();
      EXPECT_FALSE(process.Process(input, filter, output));
      EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
      expectUnchangedInitialization(filter, before, covariance);
    }
  }
}

TEST(ImuProcessValidity, NonFiniteConstructedCandidateDoesNotPartiallyCommit)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  const auto before = filter.get_x();
  const auto covariance = filter.get_P();
  process.set_extrinsic(V3D(std::numeric_limits<double>::quiet_NaN(), 0, 0));
  auto input = scan(10.0);
  for (int i = 0; i < 20; ++i)
    addImu(input, 10.0 + i * 0.005);
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_EQ(process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kInitializationInvalidState);
  expectUnchangedInitialization(filter, before, covariance);
}

TEST(ImuProcessValidity, FiniteGyroSamplesWithOverflowingAccumulationCannotCommit)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  const auto before = filter.get_x();
  const auto covariance = filter.get_P();
  auto input = scan(10.0);
  addImu(input, 10.0, V3D(0, 0, 9.81), V3D(1e308, 0, 0));
  addImu(input, 10.05, V3D(0, 0, 9.81), V3D(-1e308, 0, 0));
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(input, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  expectUnchangedInitialization(filter, before, covariance);
}

TEST(ImuInitializationValidity, NumericalNormBoundaryAndNonFiniteMeans)
{
  const V3D gyro = V3D::Zero();
  EXPECT_FALSE(fast_lio::validInitializationMean(V3D::Zero().eval(), gyro));
  EXPECT_FALSE(fast_lio::validInitializationMean(V3D(0, 0, 1e-6), gyro));
  EXPECT_TRUE(fast_lio::validInitializationMean(V3D(0, 0, 2e-6), gyro));
  EXPECT_FALSE(fast_lio::validInitializationMean(V3D(0, 0, 1e308), gyro));
  EXPECT_FALSE(fast_lio::validInitializationMean(V3D(0, 0, 9.81), V3D(std::numeric_limits<double>::quiet_NaN(), 0, 0)));
}


TEST(ImuProcessValidity, TiltedStationaryInitializationKeepsGyroBiasInTheImuFrame)
{
  for (const V3D tilt : {V3D(0.2, -0.3, 0.0), V3D(-0.4, 0.1, 0.0), V3D(0.0, 0.0, 0.0)}) {
    ImuProcess process;
    InitFilter filter;
    configureInitialization(process, filter);
    const M3D mounting = (Eigen::AngleAxisd(tilt.x(), V3D::UnitX()) *
                          Eigen::AngleAxisd(tilt.y(), V3D::UnitY())).toRotationMatrix();
    const V3D acceleration = mounting.transpose() * V3D(0.0, 0.0, 9.81);
    const V3D gyroBias(0.03, -0.02, 0.01);
    const auto samples = [&](double begin, const V3D& gyro) {
      auto input = scan(begin);
      input.lidar->points[0].curvature = 50.0f;
      for (int i = 0; i < 20; ++i)
        addImu(input, begin + i * 0.005, acceleration, gyro);
      return input;
    };
    auto output = std::make_shared<PointCloudXYZI>();
    EXPECT_FALSE(process.Process(samples(10.0, gyroBias), filter, output));
    const M3D rotation = filter.get_x().rot.toRotationMatrix();
    EXPECT_LT((filter.get_x().bg - gyroBias).norm(), 1e-12);
    ASSERT_TRUE(process.Process(samples(10.1, gyroBias), filter, output));
    EXPECT_LT((filter.get_x().rot.toRotationMatrix() - rotation).norm(), 1e-12);
    ASSERT_EQ(output->size(), 1u);
    EXPECT_NEAR(output->points[0].x, 1.0, 1e-4);
    EXPECT_NEAR(output->points[0].y, 0.0, 1e-4);
    EXPECT_NEAR(output->points[0].z, 0.0, 1e-4);
    // Subsequent estimator bias corrections use that same body-frame contract.
    const V3D updatedBias(-0.02, 0.04, 0.015);
    auto updated = filter.get_x();
    updated.bg = updatedBias;
    filter.change_x(updated);
    auto next = scan(10.2);
    addImu(next, 10.195, acceleration, updatedBias); // refreshed boundary sample
    for (int i = 0; i < 20; ++i)
      addImu(next, 10.2 + i * 0.005, acceleration, updatedBias);
    ASSERT_TRUE(process.Process(next, filter, output));
    EXPECT_LT((filter.get_x().rot.toRotationMatrix() - rotation).norm(), 1e-12);
    EXPECT_LT((filter.get_x().bg - updatedBias).norm(), 1e-12);
    EXPECT_TRUE(filter.get_P().allFinite());
  }
}

TEST(ImuProcessValidity, PopulationScatterRejectsAMovingWindowThenAcceptsFreshStillSamples)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  process.init_require_still = true;
  process.init_still_tol = 0.03;
  process.init_still_timeout_s = 90.0;
  auto output = std::make_shared<PointCloudXYZI>();
  auto moving = scan(10.0);
  for (int i = 0; i < 20; ++i)
    addImu(moving, 10.0 + i * 0.005, V3D(i % 2 ? 0.32 : -0.32, 0, 9.81), V3D::Zero());
  EXPECT_FALSE(process.Process(moving, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_NEAR(process.cov_acc.x(), 0.1024, 1e-12);
  auto one = scan(10.1);
  addImu(one, 10.195);
  EXPECT_FALSE(process.Process(one, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  auto still = scan(10.2);
  for (int i = 0; i < 20; ++i) addImu(still, 10.2 + i * 0.005);
  EXPECT_FALSE(process.Process(still, filter, output));
  auto next = scan(10.3);
  for (int i = 0; i < 20; ++i) addImu(next, 10.3 + i * 0.005);
  EXPECT_TRUE(process.Process(next, filter, output));
  EXPECT_TRUE(fast_lio::finiteInitializationState(filter.get_x()));
}

TEST(ImuProcessValidity, OverflowingPopulationScatterCannotBypassInitializationValidity)
{
  for (const bool requireStill : {false, true}) {
    ImuProcess process;
    InitFilter filter;
    configureInitialization(process, filter);
    process.init_require_still = requireStill;
    process.init_still_timeout_s = -1.0;
    const auto before = filter.get_x();
    const auto covariance = filter.get_P();
    auto output = std::make_shared<PointCloudXYZI>();
    auto input = scan(10.0);
    for (int i = 0; i < 20; ++i)
      addImu(input, 10.0 + i * 0.005, V3D(0, 0, 9.81), V3D(i % 2 ? 1e200 : -1e200, 0, 0));
    EXPECT_FALSE(process.Process(input, filter, output));
    expectUnchangedInitialization(filter, before, covariance);
    auto one = scan(10.1);
    addImu(one, 10.195);
    EXPECT_FALSE(process.Process(one, filter, output));
    auto second = scan(10.2);
    addImu(second, 10.295);
    EXPECT_FALSE(process.Process(second, filter, output));
    EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
    EXPECT_TRUE(process.cov_acc.allFinite());
    EXPECT_TRUE(process.cov_gyr.allFinite());
  }
}

namespace {
MeasureGroup initializationWindow(double begin, bool moving, bool zeroMean = false)
{
  auto input = scan(begin);
  input.lidar->points[0].curvature = 50.0f;
  for (int i = 0; i < 20; ++i) {
    const V3D acceleration = zeroMean ? V3D::Zero().eval() : V3D(moving ? (i % 2 ? 0.32 : -0.32) : 0, 0, 9.81);
    addImu(input, begin + i * 0.005, acceleration, V3D::Zero());
  }
  return input;
}
}  // namespace

TEST(ImuProcessValidity, StillnessTimeoutSpansRetriesFromZeroAndIsStrict)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  process.init_require_still = true;
  process.init_still_timeout_s = 90.0;
  auto output = std::make_shared<PointCloudXYZI>();
  for (double begin : {0.0, 30.0, 60.0, 90.0}) {
    auto moving = initializationWindow(begin, true);
    EXPECT_FALSE(process.Process(moving, filter, output));
    EXPECT_NEAR(process.cov_acc.x(), 0.1024, 1e-12); // not completed (configured noise is 0.1)
    EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), moving.lidar_end_time);
    EXPECT_FALSE(process.Process(moving, filter, output));
    EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
  }
  auto one = scan(90.1);
  addImu(one, 90.195);
  EXPECT_FALSE(process.Process(one, filter, output)); // exactly 90 did not complete
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_FALSE(process.Process(initializationWindow(90.25, true), filter, output));
  EXPECT_TRUE(process.Process(initializationWindow(90.35, false), filter, output));
}

TEST(ImuProcessValidity, ExplicitResetStartsANewAttemptAndStillnessCanRecoverEarly)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  process.init_require_still = true;
  process.init_still_timeout_s = 90.0;
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(initializationWindow(0.0, true), filter, output));
  process.Reset();
  EXPECT_FALSE(process.Process(initializationWindow(100.0, true), filter, output));
  EXPECT_NEAR(process.cov_acc.x(), 0.1024, 1e-12); // old attempt would already have expired
  auto one = scan(100.1);
  addImu(one, 100.195);
  EXPECT_FALSE(process.Process(one, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_FALSE(process.Process(initializationWindow(100.2, false), filter, output));
  EXPECT_TRUE(process.Process(initializationWindow(100.3, false), filter, output));
  process.Reset();
  EXPECT_FALSE(process.Process(initializationWindow(0.0, false), filter, output)); // Reset also clears scan cursor
  EXPECT_TRUE(process.Process(initializationWindow(0.1, false), filter, output));
}

TEST(ImuProcessValidity, ExpiredAttemptNeverAcceptsAnInvalidMeanAndRetainsConsumedScan)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  process.init_require_still = true;
  process.init_still_timeout_s = 90.0;
  auto output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(initializationWindow(0.0, true), filter, output));
  const auto before = filter.get_x();
  const auto covariance = filter.get_P();
  auto invalid = initializationWindow(100.0, false, true);
  EXPECT_FALSE(process.Process(invalid, filter, output));
  expectUnchangedInitialization(filter, before, covariance);
  EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), invalid.lidar_end_time);
  auto one = scan(100.1);
  addImu(one, 100.195);
  EXPECT_FALSE(process.Process(one, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_FALSE(process.Process(invalid, filter, output)); // reseeding must not erase consumption
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
  EXPECT_FALSE(process.Process(initializationWindow(100.2, true), filter, output));
  EXPECT_TRUE(process.Process(initializationWindow(100.3, false), filter, output));
}

namespace {
void deskewPoints(MeasureGroup& input, const std::vector<float>& offsets)
{
  input.lidar->clear();
  for (float offset : offsets) {
    PointType p{};
    p.x = 8.0f;
    p.curvature = offset;
    input.lidar->push_back(p);
  }
}
} // namespace

TEST(ImuProcessValidity, DeskewIncludesEarliestPointAndInternalBoundaryNeighbors)
{
  GapFixture fixture(false);
  auto input = scan(10.1);
  deskewPoints(input, {0.0f, 49.999f, 50.0f, 50.001f, 100.0f});
  for (int i = 0; i <= 20; ++i) addImu(input, 10.1 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  for (const auto& p : *fixture.output)
    EXPECT_NEAR(p.x, 7.9 + p.curvature / 1000.0, 2e-6);
}

TEST(ImuProcessValidity, DeskewRetainsLegalOverlapPointsAndExcludesPointsWithoutHistory)
{
  GapFixture fixture(false), reference(false);
  const double begin = fixture.process.lastProcessedEnd() - 0.0625;
  auto input = scan(begin);
  deskewPoints(input, {0.0f, 62.5f, 51.0f, 100.0f, 75.0f});
  input.lidar->header.stamp = 123456;
  for (std::size_t i = 0; i < input.lidar->size(); ++i)
    input.lidar->points[i].intensity = static_cast<float>(i + 1);
  for (int i = 0; i < 9; ++i)
    addImu(input, 10.095 + i * 0.005);
  const auto original = *input.lidar;
  auto selected = input;
  selected.lidar = std::make_shared<PointCloudXYZI>(original);
  selected.lidar->erase(selected.lidar->begin());
  selected.lidar->erase(selected.lidar->begin() + 1);
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  ASSERT_TRUE(reference.process.Process(selected, reference.filter, reference.output));
  EXPECT_EQ(fixture.process.lastStatus(), ImuProcess::ProcessStatus::kProcessed);
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), input.lidar_end_time);
  ASSERT_EQ(fixture.output->size(), 3u);
  EXPECT_EQ(fixture.output->width, 3u);
  EXPECT_EQ(fixture.output->height, 1u);
  EXPECT_EQ(fixture.output->header.stamp, original.header.stamp);
  for (std::size_t i = 0; i < fixture.output->size(); ++i) {
    const auto & point = fixture.output->points[i];
    EXPECT_GE(double(point.curvature) / 1000.0, 0.0625);
    EXPECT_NEAR(point.x, 8.0 - (input.lidar_end_time - begin - double(point.curvature) / 1000.0), 2e-6);
    EXPECT_EQ(point.x, reference.output->points[i].x);
    EXPECT_EQ(point.curvature, reference.output->points[i].curvature);
    EXPECT_EQ(point.intensity, reference.output->points[i].intensity);
  }
  EXPECT_FLOAT_EQ(fixture.output->front().curvature, 62.5f);
  EXPECT_TRUE(fixture.filter.get_x().pos.isApprox(reference.filter.get_x().pos, 0.0));
  EXPECT_TRUE(fixture.filter.get_x().vel.isApprox(reference.filter.get_x().vel, 0.0));
  EXPECT_EQ(fixture.filter.get_P(), reference.filter.get_P());
  ASSERT_EQ(input.lidar->size(), original.size());
  for (std::size_t i = 0; i < original.size(); ++i) {
    EXPECT_FLOAT_EQ(input.lidar->points[i].x, original.points[i].x);
    EXPECT_FLOAT_EQ(input.lidar->points[i].curvature, original.points[i].curvature);
    EXPECT_FLOAT_EQ(input.lidar->points[i].intensity, original.points[i].intensity);
  }
  EXPECT_DOUBLE_EQ(input.lidar_beg_time, begin);
  EXPECT_DOUBLE_EQ(input.lidar_end_time, begin + 0.1);
  EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(fixture.process.integrationLedger().watermarkRegressions(), 0u);
}

TEST(ImuProcessValidity, AllOldHistoryPointsStillRejectWithoutAdvancingOrReusingOutput)
{
  GapFixture fixture(false);
  auto rejected = scan(10.05);
  deskewPoints(rejected, {0.0f, 49.0f});
  for (int i = 0; i < 11; ++i)
    addImu(rejected, 10.095 + i * 0.005);
  const auto state = fixture.filter.get_x();
  const auto covariance = fixture.filter.get_P();
  const double watermark = fixture.process.lastProcessedEnd();
  fixture.output->push_back(rejected.lidar->front());
  EXPECT_FALSE(fixture.process.Process(rejected, fixture.filter, fixture.output));
  EXPECT_EQ(fixture.process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
  EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kHistory);
  EXPECT_TRUE(fixture.output->empty());
  expectUnchangedInitialization(fixture.filter, state, covariance);
  EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), watermark);
  EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(fixture.process.integrationLedger().watermarkRegressions(), 0u);
  EXPECT_EQ(fixture.process.staleOutputReuseCount(), 0u);
  auto accepted = scan(10.05);
  deskewPoints(accepted, {51.0f, 100.0f});
  // Retained IMU is 10.095; a valid successor must not prepend older samples.
  for (int i = 9; i <= 20; ++i) addImu(accepted, 10.05 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(accepted, fixture.filter, fixture.output));
  EXPECT_NEAR(fixture.filter.get_x().pos.x(), 0.05, 1e-10);
  EXPECT_NEAR(fixture.output->front().x, 8.0 - 0.049, 2e-6);
}

TEST(ImuProcessValidity, NanosecondAndObservedOverlapsDeskewOnlyLegalPoints)
{
  for (double overlap : {1e-9, 0.000511, 0.000511169433594}) {
    SCOPED_TRACE(overlap);
    GapFixture fixture(false);
    const double history = fixture.process.lastProcessedEnd();
    auto input = scan(history - overlap);
    deskewPoints(input, {0.0f, 0.510f, 0.512f, 100.0f});
    for (int i = 0; i < 20; ++i)
      addImu(input, 10.095 + i * 0.005);
    const double begin = input.lidar_beg_time;
    const double end = input.lidar_end_time;
    ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
    EXPECT_EQ(fixture.output->size(), overlap == 1e-9 ? 3u : 2u);
    for (const auto & point : *fixture.output) {
      EXPECT_GE(double(point.curvature) / 1000.0, history - begin);
      EXPECT_NEAR(point.x, 8.0 - (end - begin - double(point.curvature) / 1000.0), 2e-6);
    }
    EXPECT_DOUBLE_EQ(input.lidar_beg_time, begin);
    EXPECT_DOUBLE_EQ(input.lidar_end_time, end);
    EXPECT_EQ(input.lidar->size(), 4u);
    EXPECT_TRUE(fixture.filter.get_x().pos.allFinite());
    EXPECT_TRUE(fixture.filter.get_P().allFinite());
    EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
    EXPECT_EQ(fixture.process.integrationLedger().watermarkRegressions(), 0u);
  }
}

TEST(ImuProcessValidity, HistoryCompactionCannotHideInvalidOriginalScanTimes)
{
  GapFixture fixture(false);
  for (float invalid :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.0f, 101.0f}) {
    auto input = scan(10.1 - 0.000511169433594);
    deskewPoints(input, {invalid, 100.0f});
    addImu(input, 10.095);
    const auto state = fixture.filter.get_x();
    const auto covariance = fixture.filter.get_P();
    const double watermark = fixture.process.lastProcessedEnd();
    fixture.output->push_back(input.lidar->back());
    EXPECT_FALSE(fixture.process.Process(input, fixture.filter, fixture.output));
    EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInvalidScanTime);
    EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
    EXPECT_TRUE(fixture.output->empty());
    EXPECT_EQ(input.lidar->size(), 2u);
    expectUnchangedInitialization(fixture.filter, state, covariance);
    EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), watermark);
  }
}

TEST(ImuProcessValidity, RetainedHistoryPointsMatchAnalyticUniformTranslationAndRotation)
{
  GapFixture fixture(false);
  const double history = fixture.process.lastProcessedEnd();
  auto input = scan(history - 0.0625);
  deskewPoints(input, {0.0f, 62.5f, 70.0f, 90.0f, 100.0f});
  constexpr double rate = 0.2;
  for (int i = 0; i < 9; ++i)
    addImu(input, 10.095 + i * 0.005, V3D(0, 0, 9.81), V3D(0, 0, rate));
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  ASSERT_EQ(fixture.output->size(), 4u);
  const M3D endRotation = Eigen::AngleAxisd(rate * (input.lidar_end_time - history), V3D::UnitZ()).toRotationMatrix();
  for (const auto & point : *fixture.output) {
    const double offset = double(point.curvature) / 1000.0;
    const double sinceHistory = input.lidar_beg_time - history + offset;
    ASSERT_GE(sinceHistory, 0.0);
    const M3D pointRotation = Eigen::AngleAxisd(rate * sinceHistory, V3D::UnitZ()).toRotationMatrix();
    const V3D expected = endRotation.transpose() * (pointRotation * V3D(8, 0, 0) -
                                                    V3D(input.lidar_end_time - input.lidar_beg_time - offset, 0, 0));
    EXPECT_NEAR(point.x, expected.x(), 2e-6);
    EXPECT_NEAR(point.y, expected.y(), 2e-6);
    EXPECT_NEAR(point.z, expected.z(), 2e-6);
  }
  EXPECT_TRUE(fixture.filter.get_x().rot.toRotationMatrix().isApprox(endRotation, 1e-10));
  EXPECT_TRUE(fixture.filter.get_P().allFinite());
  EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(fixture.process.integrationLedger().watermarkRegressions(), 0u);
}

TEST(ImuProcessValidity, RecoveredScanCommitMakesSuccessorUseItsNewHistoryBoundary)
{
  GapFixture fixture(false);
  const double initialHistory = fixture.process.lastProcessedEnd();
  auto first = scan(initialHistory - 0.000511169433594);
  deskewPoints(first, {0.0f, 0.512f, 100.0f});
  for (int i = 0; i < 20; ++i)
    addImu(first, 10.095 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(first, fixture.filter, fixture.output));
  ASSERT_EQ(fixture.output->size(), 2u);
  EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), first.lidar_end_time);

  auto successor = scan(10.15);  // Its earliest point was legal against the pre-B watermark.
  deskewPoints(successor, {0.0f, 25.0f, 50.0f, 100.0f});
  ASSERT_GT(successor.lidar_beg_time, initialHistory);
  ASSERT_LT(successor.lidar_beg_time, first.lidar_end_time);
  for (int i = 0; i < 11; ++i)
    addImu(successor, 10.19 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(successor, fixture.filter, fixture.output));
  ASSERT_EQ(fixture.output->size(), 2u);
  EXPECT_FLOAT_EQ(fixture.output->front().curvature, 50.0f);
  EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), successor.lidar_end_time);

  auto ordinary = scan(successor.lidar_end_time);
  deskewPoints(ordinary, {0.0f, 100.0f});
  for (int i = 0; i < 22; ++i)
    addImu(ordinary, 10.24 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(ordinary, fixture.filter, fixture.output));
  ASSERT_EQ(fixture.output->size(), 2u);
  EXPECT_FLOAT_EQ(fixture.output->front().curvature, 0.0f);
  EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(fixture.process.integrationLedger().watermarkRegressions(), 0u);
  EXPECT_EQ(fixture.process.staleOutputReuseCount(), 0u);
}

TEST(ImuProcessValidity, DeskewUsesNegativeSeedOffsetAcrossScanGap)
{
  GapFixture fixture(false);
  // Exact binary endpoints avoid making the final nanosecond IMU stamp a
  // few ulps later than a separately added decimal scan end. This case tests
  // the negative seed offset, not a change to the existing coverage clock.
  auto input = scan(10.125);
  input.lidar_end_time = 10.25;
  deskewPoints(input, {0.0f, 125.0f});
  for (int i = 0; i <= 30; ++i) addImu(input, 10.1 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  EXPECT_NEAR(fixture.output->front().x, 7.875, 2e-6);
  EXPECT_NEAR(fixture.output->back().x, 8.0, 2e-6);
}

TEST(ImuProcessValidity, DeskewSynthesizesOrdinaryTailForTwoMillisecondScan)
{
  GapFixture fixture(false);
  auto input = scan(10.1);
  input.lidar_end_time = 10.102;
  deskewPoints(input, {0.0f, 2.0f});
  addImu(input, 10.095); // no newer IMU knot than retained filter state
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  EXPECT_NEAR(fixture.output->front().x, 7.998, 2e-6);
  EXPECT_NEAR(fixture.output->back().x, 8.0, 2e-6);
}

TEST(ImuProcessValidity, InitializationCompletesAtExactlyTenSamples)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  auto output = std::make_shared<PointCloudXYZI>();
  auto nine = scan(10.0);
  for (int i = 0; i < 9; ++i) addImu(nine, 10.0 + i * 0.01);
  EXPECT_FALSE(process.Process(nine, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  auto tenth = scan(10.1);
  addImu(tenth, 10.195);
  EXPECT_FALSE(process.Process(tenth, filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kInitializing);
  EXPECT_TRUE(process.Process(initializationWindow(10.2, false), filter, output));
  EXPECT_EQ(process.lastStatus(), ImuProcess::ProcessStatus::kProcessed);
}

TEST(ImuProcessValidity, DeskewConstantYawIncludesStartWithNonzeroExtrinsics)
{
  GapFixture fixture(false);
  auto state = fixture.filter.get_x();
  const M3D extrinsic = Eigen::AngleAxisd(0.3, V3D::UnitY()).toRotationMatrix();
  const V3D translation(0.2, -0.1, 0.4);
  state.offset_R_L_I = SO3(extrinsic);
  state.offset_T_L_I = translation;
  fixture.filter.change_x(state);
  // Match the actual S2 gravity magnitude, so this is truly constant velocity.
  const V3D restAcceleration = -state.grav.get_vect() * fixture.process.gravity_norm() / G_m_s2;
  auto input = scan(10.1);
  deskewPoints(input, {0.0f, 49.999f, 50.0f, 50.001f, 100.0f});
  addImu(input, 10.095, restAcceleration, V3D(0, 0, 1));
  for (int i = 0; i <= 20; ++i)
    addImu(input, 10.1 + i * 0.005, restAcceleration, V3D(0, 0, 1));
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  const M3D end = Eigen::AngleAxisd(0.1, V3D::UnitZ()).toRotationMatrix();
  for (const auto& point : *fixture.output) {
    const double t = point.curvature / 1000.0;
    const M3D rotation = Eigen::AngleAxisd(t, V3D::UnitZ()).toRotationMatrix();
    const V3D expected = extrinsic.transpose() *
      (end.transpose() * (rotation * (extrinsic * V3D(8, 0, 0) + translation) + V3D(t - 0.1, 0, 0)) - translation);
    EXPECT_NEAR(point.x, expected.x(), 2e-6);
    EXPECT_NEAR(point.y, expected.y(), 2e-6);
    EXPECT_NEAR(point.z, expected.z(), 2e-6);
  }
}

TEST(ImuProcessValidity, SyntheticTailUsesActualInputAfterAttitudeCorrection)
{
  GapFixture fixture(false);
  auto warmup = initializationWindow(10.1, false);
  ASSERT_TRUE(fixture.process.Process(warmup, fixture.filter, fixture.output));
  auto state = fixture.filter.get_x();
  state.rot = SO3(Eigen::AngleAxisd(0.2, V3D::UnitY()).toRotationMatrix());
  state.ba = V3D(0.4, 0.0, 0.0);
  fixture.filter.change_x(state);
  auto input = scan(10.2);
  input.lidar_end_time = 10.202;
  deskewPoints(input, {0.0f, 2.0f});
  addImu(input, 10.195);
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  const auto tail = fixture.filter.get_x();
  // Reconstructed input holds the previous world acceleration (zero), even
  // after a state correction; the synthetic segment must use that actual input.
  const V3D expected = tail.rot.inverse() * (state.rot * V3D(8, 0, 0) + state.pos - tail.pos);
  EXPECT_NEAR(fixture.output->front().x, expected.x(), 2e-6);
  EXPECT_NEAR(fixture.output->front().z, expected.z(), 2e-6);
  EXPECT_NEAR(fixture.output->back().x, 8.0, 2e-6);
  EXPECT_NEAR(fixture.output->back().z, 0.0, 2e-6);
}

TEST(ImuProcessValidity, DeskewBoundaryAtLargeEpoch)
{
  ImuProcess process;
  InitFilter filter;
  configureInitialization(process, filter);
  auto output = std::make_shared<PointCloudXYZI>();
  constexpr double epoch = 1700000000.0;
  EXPECT_FALSE(process.Process(initializationWindow(epoch, false), filter, output));
  auto state = filter.get_x();
  state.vel = V3D(1, 0, 0);
  filter.change_x(state);
  auto input = initializationWindow(epoch + 0.1, false);
  deskewPoints(input, {0.0f, 50.0f, 100.0f});
  ASSERT_TRUE(process.Process(input, filter, output));
  EXPECT_NEAR(output->front().x, 7.9, 2e-6);
  EXPECT_NEAR(output->back().x, 8.0, 2e-6);
}

TEST(ImuProcessValidity, DeskewInternalKnotBelongsToPrecedingMotionSegment)
{
  GapFixture fixture(false, 10.025);
  ASSERT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), 10.125);
  auto input = scan(10.125);
  // Binary-exact 62.5 ms knot avoids conflating segment ownership with stamp rounding.
  input.lidar_end_time = 10.25;
  deskewPoints(input, {0.0f, 62.499f, 62.5f, 62.501f, 125.0f});
  addImu(input, 10.125, V3D(0, 0, 9.81), V3D::Zero());
  addImu(input, 10.1875, V3D(2, 0, 9.81), V3D(0, 0, 2));
  addImu(input, 10.25, V3D(6, 0, 9.81), V3D(0, 0, 6));
  const double knot = rclcpp::Time(input.imu[1]->header.stamp).seconds() - input.lidar_beg_time;
  ASSERT_DOUBLE_EQ(knot, 0.0625);
  ASSERT_TRUE(fixture.process.Process(input, fixture.filter, fixture.output));
  const M3D r1 = Eigen::AngleAxisd(knot, V3D::UnitZ()).toRotationMatrix();
  const auto end = fixture.filter.get_x();
  for (const auto& point : *fixture.output) {
    const double t = point.curvature / 1000.0;
    const bool first = t <= knot;
    const double d = first ? t : t - knot;
    const M3D headR = first ? M3D::Identity().eval() : r1;
    const M3D tailR = first ? r1 : end.rot.toRotationMatrix();
    const V3D headP = first ? V3D::Zero().eval() : V3D(knot, 0, 0);
    const V3D headV = first ? V3D(1, 0, 0) : V3D(1 + knot, 0, 0);
    const double rate = first ? 1.0 : 4.0;
    const V3D worldPoint = headR * Eigen::AngleAxisd(rate * d, V3D::UnitZ()) * V3D(8, 0, 0) +
      headP + headV * d + 0.5 * tailR * V3D(rate, 0, 0) * d * d;
    const V3D expected = end.rot.inverse() * (worldPoint - end.pos);
    EXPECT_NEAR(point.x, expected.x(), 2e-6);
    EXPECT_NEAR(point.y, expected.y(), 2e-6);
  }
}

TEST(ImuProcessValidity, RetainedRejectionsThenSuccessMatchOneCombinedBatchIncludingAuxiliaryState)
{
  GapFixture retained(false), combined(false);
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> queue;
  fast_lio::ImuFifoTransaction<sensor_msgs::msg::Imu::ConstSharedPtr> transaction;
  auto final = scan(10.2);
  for (int i = 0; i < 40; ++i)
    addImu(final, 10.1 + i * 0.005, V3D(0.3, 0.2, 9.81), V3D(0, 0, 0.2));
  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto count = static_cast<std::size_t>(10 + attempt * 5);
    while (queue.size() < count)
      queue.push_back(final.imu[queue.size()]);
    const auto token = transaction.borrow(queue, count);
    auto rejected = scan(10.1 - 0.000511169433594);
    deskewPoints(rejected, {0.0f, 0.5f});  // All points lack history; retain A's negative control.
    rejected.imu.assign(token.prefix.begin(), token.prefix.end());
    EXPECT_FALSE(retained.process.Process(rejected, retained.filter, retained.output));
    EXPECT_EQ(retained.process.lastOutcome().reason, fast_lio::ImuProcessReason::kHistory);
    EXPECT_EQ(transaction.settle(queue, token, retained.process.lastOutcome().disposition),
              fast_lio::ImuConfirmation::kRetained);
    EXPECT_EQ(queue.size(), count);
  }
  queue.assign(final.imu.begin(), final.imu.end());
  const auto token = transaction.borrow(queue, queue.size());
  ASSERT_TRUE(retained.process.Process(final, retained.filter, retained.output));
  ASSERT_TRUE(combined.process.Process(final, combined.filter, combined.output));
  EXPECT_EQ(transaction.settle(queue, token, retained.process.lastOutcome().disposition),
            fast_lio::ImuConfirmation::kReleased);
  EXPECT_TRUE(queue.empty());
  EXPECT_TRUE(retained.filter.get_x().pos.isApprox(combined.filter.get_x().pos, 0.0));
  EXPECT_TRUE(retained.filter.get_x().vel.isApprox(combined.filter.get_x().vel, 0.0));
  EXPECT_TRUE(
    retained.filter.get_x().rot.toRotationMatrix().isApprox(combined.filter.get_x().rot.toRotationMatrix(), 0.0));
  EXPECT_TRUE(retained.filter.get_P().isApprox(combined.filter.get_P(), 0.0));
  EXPECT_TRUE(retained.process.lastDeskewAcceleration().isApprox(combined.process.lastDeskewAcceleration(), 0.0));
  EXPECT_TRUE(retained.process.lastDeskewAngularVelocity().isApprox(combined.process.lastDeskewAngularVelocity(), 0.0));
  EXPECT_DOUBLE_EQ(retained.process.lastProcessedEnd(), combined.process.lastProcessedEnd());
  EXPECT_EQ(retained.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(retained.process.integrationLedger().watermarkRegressions(), 0u);
}

TEST(ImuProcessValidity, PrevalidationRejectionsRetainOnlyFiniteOrderedSamples)
{
  GapFixture fixture(false);
  auto input = scan(10.1);
  addImu(input, 10.1);
  EXPECT_FALSE(fixture.process.Process(input, fixture.filter, nullptr));
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
  EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kNoOutput);
  input.lidar.reset();
  EXPECT_FALSE(fixture.process.Process(input, fixture.filter, fixture.output));
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
  input.imu.push_back(nullptr);
  EXPECT_FALSE(fixture.process.Process(input, fixture.filter, fixture.output));
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kInvalid);
  EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInvalidImu);
  EXPECT_TRUE(fixture.output->empty());
}

TEST(ImuProcessValidity, InvalidMessageStampsAndNonFiniteComponentsAreExplicitlyInvalid)
{
  for (int malformed = 0; malformed < 5; ++malformed) {
    GapFixture fixture(false);
    auto input = scan(10.1);
    auto imu = std::make_shared<sensor_msgs::msg::Imu>();
    imu->header.stamp = rclcpp::Time(static_cast<std::int64_t>(10100000000LL));
    imu->linear_acceleration.z = 9.81;
    switch (malformed) {
      case 0:
        imu->header.stamp.sec = -1;
        break;
      case 1:
        imu->header.stamp.nanosec = 1000000000u;
        break;
      case 2:
        imu->linear_acceleration.x = std::numeric_limits<double>::quiet_NaN();
        break;
      case 3:
        imu->angular_velocity.z = std::numeric_limits<double>::infinity();
        break;
      case 4:
        imu.reset();
        break;
    }
    input.imu.push_back(imu);
    EXPECT_FALSE(fixture.process.Process(input, fixture.filter, fixture.output));
    EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kInvalid);
    EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kInvalidImu);
    EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), 10.1);
  }
}

TEST(ImuProcessValidity, EndCoverageReanchorsButUnorderedOrInvalidCoverageDoesNot)
{
  for (int path = 0; path < 3; ++path) {
    GapFixture fixture(false);
    auto input = scan(10.1);
    addImu(input, 10.1);
    addImu(input, 10.11);
    if (path == 0)
      fixture.process.coverage_params.max_extrapolation_s = 0.01;
    else if (path == 1)
      addImu(input, 10.105);
    else
      fixture.process.coverage_params.max_gap_s = -1.0;
    EXPECT_FALSE(fixture.process.Process(input, fixture.filter, fixture.output));
    EXPECT_EQ(fixture.process.lastOutcome().disposition,
              path == 0 ? fast_lio::ImuDisposition::kCommitted : fast_lio::ImuDisposition::kInvalid);
    EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), path == 0 ? input.lidar_end_time : 10.1);
    EXPECT_TRUE(fixture.output->empty());
  }
}

TEST(ImuProcessValidity, PartiallyPropagatedExceptionIsFatalInsteadOfAnUncommittedRetry)
{
  GapFixture fixture(false);
  static int calls;
  calls = 0;
  double epsilon[23];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  fixture.filter.init_dyn_share(
    [](state_ikfom & state, const input_ikfom & input) -> Eigen::Matrix<double, 24, 1> {
      if (++calls == 2)
        throw std::runtime_error("injected partial propagation");
      return get_f(state, input);
    },
    df_dx,
    df_dw,
    [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {},
    5,
    epsilon);
  auto input = scan(10.1);
  for (int i = 0; i < 20; ++i)
    addImu(input, 10.1 + i * 0.005);
  const auto before = fixture.filter.get_x().pos;
  EXPECT_THROW(fixture.process.Process(input, fixture.filter, fixture.output), std::runtime_error);
  EXPECT_EQ(calls, 2);
  EXPECT_FALSE(fixture.filter.get_x().pos.isApprox(before, 0.0));
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kFatal);
  EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kPartialPropagation);
}

TEST(ImuProcessValidity, RepeatedUnavailableHistoryRejectionsDoNotConsumeOrMoveTheWatermark)
{
  GapFixture fixture(false);
  auto rejected = scan(10.1 - 0.000511169433594);
  deskewPoints(rejected, {0.0f, 0.5f});  // Repeated all-old rejection must still retain IMU.
  for (int i = 0; i < 20; ++i)
    addImu(rejected, 10.1 + i * 0.005);
  auto queue = rejected.imu;
  fast_lio::ImuFifoTransaction<sensor_msgs::msg::Imu::ConstSharedPtr> transaction;
  int rejectedScans = 0;
  for (int i = 0; i < 161; ++i) {
    const auto token = transaction.borrow(queue, queue.size());
    EXPECT_FALSE(fixture.process.Process(rejected, fixture.filter, fixture.output));
    ASSERT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kHistory);
    EXPECT_EQ(transaction.settle(queue, token, fixture.process.lastOutcome().disposition),
              fast_lio::ImuConfirmation::kRetained);
    EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), 10.1);
    EXPECT_TRUE(fixture.output->empty());
    ++rejectedScans;
  }
  EXPECT_EQ(rejectedScans, 161);
  auto successor = scan(10.1);
  deskewPoints(successor, {0.0f, 100.0f});
  successor.imu = queue;
  const auto token = transaction.borrow(queue, queue.size());
  ASSERT_TRUE(fixture.process.Process(successor, fixture.filter, fixture.output));
  EXPECT_EQ(transaction.settle(queue, token, fixture.process.lastOutcome().disposition),
            fast_lio::ImuConfirmation::kReleased);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(fixture.process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(fixture.process.staleOutputReuseCount(), 0u);
}

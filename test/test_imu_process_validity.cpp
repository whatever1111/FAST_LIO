#include <gtest/gtest.h>
#include <omp.h>

#include <limits>

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
}  // namespace

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

TEST(ImuProcessValidity, DeskewRejectsUnavailableOverlapWithoutAdvancingHistory)
{
  GapFixture fixture(false);
  auto rejected = scan(10.05);
  deskewPoints(rejected, {0.0f, 100.0f});
  for (int i = 0; i <= 20; ++i) addImu(rejected, 10.05 + i * 0.005);
  const auto state = fixture.filter.get_x();
  const auto covariance = fixture.filter.get_P();
  const double watermark = fixture.process.lastProcessedEnd();
  EXPECT_FALSE(fixture.process.Process(rejected, fixture.filter, fixture.output));
  EXPECT_EQ(fixture.process.lastStatus(), ImuProcess::ProcessStatus::kRejected);
  EXPECT_EQ(fixture.process.lastOutcome().disposition, fast_lio::ImuDisposition::kUncommitted);
  EXPECT_EQ(fixture.process.lastOutcome().reason, fast_lio::ImuProcessReason::kHistory);
  EXPECT_TRUE(fixture.output->empty());
  expectUnchangedInitialization(fixture.filter, state, covariance);
  EXPECT_DOUBLE_EQ(fixture.process.lastProcessedEnd(), watermark);
  auto accepted = scan(10.05);
  deskewPoints(accepted, {51.0f, 100.0f});
  // Retained IMU is 10.095; a valid successor must not prepend older samples.
  for (int i = 9; i <= 20; ++i) addImu(accepted, 10.05 + i * 0.005);
  ASSERT_TRUE(fixture.process.Process(accepted, fixture.filter, fixture.output));
  EXPECT_NEAR(fixture.filter.get_x().pos.x(), 0.05, 1e-10);
  EXPECT_NEAR(fixture.output->front().x, 8.0 - 0.049, 2e-6);
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

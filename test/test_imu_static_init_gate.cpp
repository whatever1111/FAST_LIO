#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "imu_static_init_gate.hpp"

namespace
{
using Decision = fast_lio::ImuStaticInitDecision;
using Sample = fast_lio::ImuStaticInitSample;

std::vector<Sample> stillBatch(double begin, double duration, int count, double gyro = 0.0, double gravity = 9.81)
{
  std::vector<Sample> batch;
  for (int i = 0; i < count; ++i) {
    const double stamp = begin + (count > 1 ? duration * i / (count - 1) : 0.0);
    batch.push_back({stamp, Eigen::Vector3d(0.0, 0.0, gravity), Eigen::Vector3d(0.0, 0.0, gyro)});
  }
  return batch;
}

struct GateFixture
{
  fast_lio::ImuStaticInitGate gate;
  fast_lio::ImuStaticInitParams params;
  GateFixture() { params.enabled = true; }
  Decision observe(const std::vector<Sample> & batch, double elapsed = 0.0, bool require = true)
  {
    return gate.observe(batch, [](const Sample & sample) { return sample; }, require, params, 0.03, elapsed, 20.0);
  }
};
}  // namespace

TEST(ImuStaticInitGate, T01At200HzWaitsUntilOneSecond)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 0.99, 199)), Decision::kAccumulating);
  EXPECT_EQ(f.observe(stillBatch(0.995, 0.005, 2)), Decision::kReady);
  EXPECT_EQ(f.gate.samples(), 201u);
}

TEST(ImuStaticInitGate, T02DurationAndCountBoundariesAreInclusive)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100)), Decision::kReady);
  f.gate.reset();
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.2, 99)), Decision::kAccumulating);
}

TEST(ImuStaticInitGate, T03ScatterAboveRestartsBelowPasses)
{
  for (const double scatter : {0.0299, 0.0301}) {
    GateFixture f;
    auto batch = stillBatch(0.0, 1.0, 100);
    for (std::size_t i = 0; i < batch.size(); ++i)
      batch[i].acceleration.x() = (i % 2 == 0 ? 1.0 : -1.0) * scatter * 9.81;
    EXPECT_EQ(f.observe(batch), scatter < 0.03 ? Decision::kReady : Decision::kRestart);
  }
}

TEST(ImuStaticInitGate, T04RotatingWithQuietAccelerationRestarts)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100, 0.11)), Decision::kRestart);
  EXPECT_EQ(f.observe(stillBatch(2.0, 1.0, 100, 0.09)), Decision::kReady);
  f.gate.reset();
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100, 0.1)), Decision::kReady);
}

TEST(ImuStaticInitGate, T05RawGravityMagnitudeIsChecked)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100, 0.0, 1.06 * 9.81)), Decision::kRestart);
  EXPECT_EQ(f.observe(stillBatch(2.0, 1.0, 100, 0.0, 1.04 * 9.81)), Decision::kReady);
}

TEST(ImuStaticInitGate, T06GapAboveLimitRestartsBelowPasses)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100)), Decision::kReady);
  EXPECT_EQ(f.observe(stillBatch(1.11, 0.0, 1)), Decision::kRestart);
  EXPECT_EQ(f.observe(stillBatch(2.0, 1.0, 100)), Decision::kReady);
  EXPECT_EQ(f.observe(stillBatch(3.09, 0.0, 1)), Decision::kReady);
}

TEST(ImuStaticInitGate, T07NonfiniteOrNonincreasingSamplesNeverInitialize)
{
  for (int kind = 0; kind < 7; ++kind) {
    GateFixture f;
    auto batch = stillBatch(0.0, 1.0, 100);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (kind == 0)
      batch[50].stamp = nan;
    if (kind == 1)
      batch[50].stamp = std::numeric_limits<double>::infinity();
    if (kind == 2)
      batch[50].stamp = batch[49].stamp;
    if (kind == 3)
      batch[50].stamp = batch[49].stamp - 0.01;
    if (kind == 4)
      batch[50].acceleration.x() = nan;
    if (kind == 5)
      batch[50].gyro.x() = nan;
    if (kind == 6)
      batch[50].gyro.x() = std::numeric_limits<double>::infinity();
    EXPECT_EQ(f.observe(batch, 21.0), Decision::kRestart);
    EXPECT_EQ(f.gate.samples(), 0u);
  }
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 0.5, 101)), Decision::kAccumulating);
  EXPECT_EQ(f.observe(stillBatch(0.5, 0.5, 101)), Decision::kRestart);
  EXPECT_EQ(f.observe({}), Decision::kRestart);
}

TEST(ImuStaticInitGate, T08ShortMovingBatchCannotBeDilutedByFiveSeconds)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 5.0, 1001)), Decision::kReady);
  EXPECT_EQ(f.observe(stillBatch(5.005, 0.095, 20, 0.11)), Decision::kRestart);
  EXPECT_EQ(f.gate.samples(), 0u);
  EXPECT_EQ(f.observe(stillBatch(6.0, 5.0, 1001)), Decision::kReady);
  auto moving = stillBatch(11.005, 0.095, 20);
  for (std::size_t i = 0; i < moving.size(); ++i)
    moving[i].acceleration.x() = (i % 2 == 0 ? 1.0 : -1.0) * 0.04 * 9.81;
  EXPECT_EQ(f.observe(moving), Decision::kRestart);
}

TEST(ImuStaticInitGate, T09TimeoutUsesStrictlyGreaterAndDegradesFirstLateBatch)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 1.0, 100, 0.11), 20.0), Decision::kRestart);
  EXPECT_EQ(f.observe(stillBatch(2.0, 1.0, 100, 0.11), 20.01), Decision::kDegraded);
}

TEST(ImuStaticInitGate, T10DisabledGateDoesNotReadSamplesOrParticipate)
{
  GateFixture f;
  f.params.enabled = false;
  EXPECT_EQ(f.observe({}, 100.0), Decision::kDisabled);
  f.params.enabled = true;
  EXPECT_EQ(f.observe({}, 100.0, false), Decision::kDisabled);
  EXPECT_EQ(f.gate.samples(), 0u);
}

TEST(ImuStaticInitGate, T11MovingThenStillBeforeTimeoutIsNormal)
{
  GateFixture f;
  EXPECT_EQ(f.observe(stillBatch(0.0, 0.1, 21, 0.11), 0.1), Decision::kRestart);
  EXPECT_EQ(f.observe(stillBatch(2.0, 0.99, 199), 2.99), Decision::kAccumulating);
  EXPECT_EQ(f.observe(stillBatch(2.995, 0.005, 2), 3.0), Decision::kReady);
}

TEST(ImuStaticInitGate, T12InvalidParametersAreRejected)
{
  fast_lio::ImuStaticInitParams valid;
  ASSERT_TRUE(fast_lio::validImuStaticInitParams(valid));
  for (double fast_lio::ImuStaticInitParams::* field : {&fast_lio::ImuStaticInitParams::minDurationS,
                                                        &fast_lio::ImuStaticInitParams::gyroRmsMax,
                                                        &fast_lio::ImuStaticInitParams::gravityRef,
                                                        &fast_lio::ImuStaticInitParams::gravityTol,
                                                        &fast_lio::ImuStaticInitParams::maxGapS}) {
    for (double invalid :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
      auto params = valid;
      params.*field = invalid;
      EXPECT_FALSE(fast_lio::validImuStaticInitParams(params));
    }
  }
  valid.minSamples = 0;
  EXPECT_FALSE(fast_lio::validImuStaticInitParams(valid));
  valid.minSamples = -1;
  EXPECT_FALSE(fast_lio::validImuStaticInitParams(valid));
}

TEST(ImuStaticInitGate, T16DiagnosticCellRequiresHealthActiveGateAndDegradedLatch)
{
  for (bool health : {false, true})
    for (bool active : {false, true})
      for (bool degraded : {false, true})
        EXPECT_DOUBLE_EQ(fast_lio::imuInitDegradedCell(health, active, degraded),
                         health && active && degraded ? 1.0 : 0.0);
}

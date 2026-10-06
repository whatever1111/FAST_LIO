#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "IMU_Processing.hpp"

namespace
{
float floatBits(const std::string & hex)
{
  const auto bits = static_cast<std::uint32_t>(std::stoul(hex, nullptr, 16));
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
}  // namespace

TEST(Imu0906Timing, SequentialProcessMatchesIndependentGoldenHistoryRejections)
{
  std::ifstream fixture(FASTLIO_0906_FIXTURE_PATH);
  ASSERT_TRUE(fixture.is_open());
  std::int64_t headerNs, beginNs;
  ASSERT_TRUE(static_cast<bool>(fixture >> headerNs >> beginNs));
  std::string line;
  std::getline(fixture, line);
  ImuProcess process;
  process.coverage_params = {0.1, 0.03};
  process.init_require_still = false;
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
  double epsilon[23];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  filter.init_dyn_share(get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  auto output = std::make_shared<PointCloudXYZI>();
  using Imu = sensor_msgs::msg::Imu::ConstSharedPtr;
  std::deque<Imu> queue;
  Imu latestImu;
  fast_lio::ImuFifoTransaction<Imu> transaction;
  std::int64_t nextImuNs = beginNs - 10000000;
  std::vector<std::int64_t> expectedRejected, actualRejected;
  std::size_t frames = 0, successful = 0, nonadvancing = 0, successfulAfterRejection = 0;
  bool retainedSinceSuccess = false;
  while (std::getline(fixture, line)) {
    std::int64_t rawId, headerDelta, beginDelta;
    std::string firstBits, lastBits;
    char label;
    std::istringstream row(line);
    ASSERT_TRUE(static_cast<bool>(row >> rawId >> headerDelta >> beginDelta >> firstBits >> lastBits >> label));
    row >> std::ws;
    ASSERT_TRUE(row.eof());
    SCOPED_TRACE("raw_scan_id=" + std::to_string(rawId) + " expected=" + label);
    headerNs += headerDelta;
    beginNs += beginDelta;
    ASSERT_GE(headerNs, 0);
    const float firstMs = floatBits(firstBits), lastMs = floatBits(lastBits);
    ASSERT_TRUE(std::isfinite(firstMs) && std::isfinite(lastMs));
    ASSERT_GE(firstMs, 0.0f);
    ASSERT_GE(lastMs, firstMs);
    MeasureGroup input;
    input.lidar_beg_time = static_cast<double>(beginNs) / 1e9;
    input.lidar_end_time = input.lidar_beg_time + static_cast<double>(lastMs) / 1000.0;
    input.lidar = std::make_shared<PointCloudXYZI>();
    for (const float offset : {firstMs, lastMs}) {
      PointType point{};
      point.x = 1.0f;
      point.curvature = offset;
      input.lidar->push_back(point);
    }
    // Real absolute epochs and exact float32 point offsets; sparse, continuous
    // rest IMU isolates the temporal guard from the recorded motion values.
    const std::int64_t imuPeriodNs = frames < 2 ? 5000000 : 25000000;
    while (static_cast<double>(nextImuNs) / 1e9 <= input.lidar_end_time) {
      auto message = std::make_shared<sensor_msgs::msg::Imu>();
      message->header.stamp = rclcpp::Time(nextImuNs);
      message->linear_acceleration.z = 9.81;
      queue.push_back(message);
      latestImu = message;
      nextImuNs += imuPeriodNs;
    }
    if (label == 'N') {
      // Production sync rejects these before borrowing the FIFO. Exercise the
      // direct Process guard too, with a finite sample and no FIFO consumption.
      ASSERT_TRUE(latestImu);
      input.imu.push_back(latestImu);
      const double previousEnd = process.lastProcessedEnd();
      EXPECT_FALSE(process.Process(input, filter, output));
      EXPECT_EQ(process.lastOutcome().reason, fast_lio::ImuProcessReason::kNonadvancing);
      EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), previousEnd);
      EXPECT_TRUE(output->empty());
      ++nonadvancing;
      ++frames;
      continue;
    }
    const auto count = fast_lio::imuPrefixThrough(
      queue, input.lidar_end_time, [](const Imu & imu) { return rclcpp::Time(imu->header.stamp).seconds(); });
    const auto token = transaction.borrow(queue, count);
    input.imu.assign(token.prefix.begin(), token.prefix.end());
    if (label == 'B') {
      ASSERT_EQ(frames, 0u);
      ASSERT_EQ(transaction.settle(queue, token, fast_lio::ImuDisposition::kCommitted),
                fast_lio::ImuConfirmation::kReleased);
      ++frames;
      continue;  // the real node's explicit bootstrap also bypasses Process
    }
    ASSERT_FALSE(input.imu.empty());
    const double previousEnd = process.lastProcessedEnd();
    const bool processed = process.Process(input, filter, output);
    const auto outcome = process.lastOutcome();
    if (outcome.reason == fast_lio::ImuProcessReason::kHistory)
      actualRejected.push_back(rawId);
    if (label == 'H') {
      expectedRejected.push_back(rawId);
      EXPECT_FALSE(processed);
      EXPECT_EQ(outcome.reason, fast_lio::ImuProcessReason::kHistory);
      EXPECT_EQ(outcome.disposition, fast_lio::ImuDisposition::kUncommitted);
      EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), previousEnd);
      EXPECT_TRUE(output->empty());
      retainedSinceSuccess = true;
    } else if (label == 'I') {
      EXPECT_FALSE(processed);
      EXPECT_EQ(outcome.reason, fast_lio::ImuProcessReason::kInitializationComplete);
      EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), input.lidar_end_time);
    } else {
      ASSERT_TRUE(label == 'M' || label == 'O');
      EXPECT_TRUE(processed);
      EXPECT_EQ(outcome.reason, fast_lio::ImuProcessReason::kProcessed);
      EXPECT_GT(process.lastProcessedEnd(), previousEnd);
      EXPECT_DOUBLE_EQ(process.lastProcessedEnd(), input.lidar_end_time);
      if (retainedSinceSuccess)
        ++successfulAfterRejection;
      retainedSinceSuccess = false;
      if (label == 'O')
        ++successful;
    }
    const auto confirmation = transaction.settle(queue, token, outcome.disposition);
    EXPECT_EQ(confirmation,
              outcome.disposition == fast_lio::ImuDisposition::kCommitted ? fast_lio::ImuConfirmation::kReleased
                                                                          : fast_lio::ImuConfirmation::kRetained);
    ++frames;
  }
  EXPECT_TRUE(fixture.eof());
  EXPECT_EQ(frames, 8968u);
  ASSERT_EQ(expectedRejected.size(), 161u);
  EXPECT_EQ(actualRejected, expectedRejected);
  EXPECT_EQ(successful, 8799u);
  EXPECT_EQ(nonadvancing, 5u);
  EXPECT_EQ(successfulAfterRejection, 161u);
  EXPECT_EQ(process.integrationLedger().duplicateIntegrations(), 0u);
  EXPECT_EQ(process.integrationLedger().watermarkRegressions(), 0u);
  EXPECT_EQ(process.staleOutputReuseCount(), 0u);
}

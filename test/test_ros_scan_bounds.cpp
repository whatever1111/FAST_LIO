#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <algorithm>
#include <chrono>
#include <class_loader/class_loader.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "imu_consumption_policy.hpp"
#include "imu_process_outcome.hpp"
#include "laser_mapping_test_hooks.hpp"
#include "preprocess.h"

namespace
{
// The class name RCLCPP_COMPONENTS_REGISTER_NODE(LaserMappingNode) registers.
constexpr const char * kLaserMappingFactory = "rclcpp_components::NodeFactoryTemplate<LaserMappingNode>";

struct ScanEvidence
{
  std::size_t count;
  double stamp;
  int effective;
};

struct ImuBatchEvidence
{
  std::uint64_t token;
  fast_lio::ImuDisposition disposition;
  fast_lio::ImuProcessReason reason;
  std::size_t samples;
  std::size_t pending;
  std::uint64_t firstIdentity;
};

constexpr std::size_t kPreparedEvidenceCapacity = 64;
constexpr std::size_t kModelEvidenceCapacity = 128;
constexpr std::size_t kOdomEvidenceCapacity = 256;
constexpr std::size_t kTargetEvidenceCapacity = 3;
constexpr double kTargetWaitSeconds = 60.0;

struct TargetEvidence
{
  std::size_t count;
  double stamp;
  double elapsedSeconds;
  bool prepared;
  bool modeled;
  bool odom;
  double lastOdom;
  bool lastPoseFinite;
};

class RosScanBounds : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    loader = std::make_unique<class_loader::ClassLoader>(FASTLIO_TEST_COMPONENT_PATH);
    // Every component library in the process lists its factory here, not only this one
    // (a link that keeps tf2_ros's static_transform_broadcaster_node adds a second), so
    // the front end's is taken by the name RCLCPP_COMPONENTS_REGISTER_NODE gives it.
    const auto names = loader->getAvailableClasses<rclcpp_components::NodeFactory>();
    ASSERT_NE(std::find(names.begin(), names.end(), kLaserMappingFactory), names.end());
    factory = loader->createInstance<rclcpp_components::NodeFactory>(kLaserMappingFactory);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    options.parameter_overrides(
      {rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0}),
       rclcpp::Parameter("mapping.extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
       rclcpp::Parameter("common.lid_topic", "/test_scan_bounds/points"),
       rclcpp::Parameter("common.imu_topic", "/test_scan_bounds/imu"),
       rclcpp::Parameter("preprocess.lidar_type", 3),
       rclcpp::Parameter("preprocess.timestamp_unit", 3),
       rclcpp::Parameter("point_filter_num", 1),
       rclcpp::Parameter("feature_extract_enable", false),
       rclcpp::Parameter("filter_size_surf", 0.001),
       rclcpp::Parameter("filter_size_map", 0.05),
       rclcpp::Parameter("imu_init_require_still", false),
#ifdef FASTLIO_INPUT_FAULT_TEST
       rclcpp::Parameter("max_imu_backlog_sec", 1.0),
#endif
       rclcpp::Parameter("publish.scan_publish_en", false),
       rclcpp::Parameter("publish.path_en", false),
       rclcpp::Parameter("runtime_pos_log_enable", false)});
    subject.emplace(factory->create_node_instance(options));
    rclcpp::NodeOptions observerOptions;
    observerOptions.use_intra_process_comms(true);
    observer = std::make_shared<rclcpp::Node>("scan_bounds_observer", observerOptions);
    imu = observer->create_publisher<sensor_msgs::msg::Imu>("/test_scan_bounds/imu",
                                                            rclcpp::QoS(rclcpp::KeepLast(1000)).reliable());
    lidar = observer->create_publisher<sensor_msgs::msg::PointCloud2>("/test_scan_bounds/points",
                                                                      rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    odom = observer->create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", rclcpp::QoS(rclcpp::KeepLast(20)).reliable(), [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        lastOdom = rclcpp::Time(msg->header.stamp).seconds();
        lastPoseFinite = std::isfinite(msg->pose.pose.position.x) && std::isfinite(msg->pose.pose.position.y) &&
                         std::isfinite(msg->pose.pose.position.z);
        if (odomStamps.size() < kOdomEvidenceCapacity)
          odomStamps.push_back(lastOdom);
        else
          ++omittedOdomStamps;
      });
    fast_lio_test::afterScanPrepared = [this](std::size_t count, double stamp) {
      if (prepared.size() < kPreparedEvidenceCapacity)
        prepared.push_back({count, stamp, -1});
    };
    fast_lio_test::afterHModelRows = [this](std::size_t count, double stamp, int effective) {
      if (modeled.size() < kModelEvidenceCapacity)
        modeled.push_back({count, stamp, effective});
    };
    fast_lio_test::afterImuConsumption = [this](std::uint64_t token,
                                                int disposition,
                                                int reason,
                                                std::size_t count,
                                                std::size_t pending,
                                                std::uint64_t firstIdentity) {
      if (imuBatches.size() < kModelEvidenceCapacity)
        imuBatches.push_back({token,
                              static_cast<fast_lio::ImuDisposition>(disposition),
                              static_cast<fast_lio::ImuProcessReason>(reason),
                              count,
                              pending,
                              firstIdentity});
    };
    executor->add_node(subject->get_node_base_interface());
    executor->add_node(observer);
    ASSERT_TRUE(spinUntil([&] { return imu->get_subscription_count() == 1 && lidar->get_subscription_count() == 1; }));
  }

  void TearDown() override
  {
    // These records live under /ci_ws/build, not the disappearing container /tmp.
    if (subject)
      executor->remove_node(subject->get_node_base_interface());
    if (observer)
      executor->remove_node(observer);
    subject.reset();  // production destructor drains and joins the real map worker
    fast_lio_test::afterScanPrepared = {};
    fast_lio_test::afterHModelRows = {};
    fast_lio_test::afterImuConsumption = {};
    fast_lio_test::beforeImuProcess = {};
    odom.reset();
    imu.reset();
    lidar.reset();
    observer.reset();
    factory.reset();
    loader.reset();
    executor.reset();
    rclcpp::shutdown();
    std::ofstream receipt(FASTLIO_SCAN_RECEIPT_PATH);
    receipt << "{\"shutdown_completed\":true,\"case_failed\":" << (HasFailure() ? "true" : "false")
            << ",\"prepared_capacity\":" << kPreparedEvidenceCapacity
            << ",\"h_model_capacity\":" << kModelEvidenceCapacity << ",\"prepared\":[";
    writeEvidence(receipt, prepared);
    receipt << "],\"h_model\":[";
    writeEvidence(receipt, modeled);
    receipt << "],\"odom_capacity\":" << kOdomEvidenceCapacity << ",\"odom_omitted\":" << omittedOdomStamps
            << ",\"odom_stamps\":[";
    for (std::size_t i = 0; i < odomStamps.size(); ++i) {
      if (i)
        receipt << ',';
      receipt << odomStamps[i];
    }
    receipt << "],\"target_capacity\":" << kTargetEvidenceCapacity
            << ",\"target_wait_budget_seconds\":" << kTargetWaitSeconds << ",\"targets\":[";
    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (i)
        receipt << ',';
      const auto & target = targets[i];
      receipt << "{\"count\":" << target.count << ",\"stamp\":" << target.stamp
              << ",\"steady_elapsed_seconds\":" << target.elapsedSeconds
              << ",\"prepared\":" << (target.prepared ? "true" : "false")
              << ",\"modeled\":" << (target.modeled ? "true" : "false")
              << ",\"odom\":" << (target.odom ? "true" : "false") << ",\"last_odom\":" << target.lastOdom
              << ",\"last_pose_finite\":" << (target.lastPoseFinite ? "true" : "false") << '}';
    }
    receipt << "],\"imu_batches\":[";
    for (std::size_t i = 0; i < imuBatches.size(); ++i) {
      if (i)
        receipt << ',';
      const auto & batch = imuBatches[i];
      receipt << "{\"token\":" << batch.token << ",\"disposition\":" << static_cast<int>(batch.disposition)
              << ",\"reason\":\"" << fast_lio::imuProcessReasonName(batch.reason) << "\",\"samples\":" << batch.samples
              << ",\"pending\":" << batch.pending << ",\"first_id\":" << batch.firstIdentity << '}';
    }
    receipt << "],\"fault_snapshots\":[";
    for (std::size_t i = 0; i < faultSnapshots.size(); ++i) {
      if (i)
        receipt << ',';
      const auto & state = faultSnapshots[i];
      receipt << "{\"lidar_queued\":" << state.lidarQueued << ",\"imu_queued\":" << state.imuQueued
              << ",\"rejected_lidar\":" << state.rejectedLidar << ",\"rejected_imu\":" << state.rejectedImu
              << ",\"discarded_lidar\":" << state.discardedLidar << ",\"discarded_imu\":" << state.discardedImu
              << ",\"faulted\":" << (state.faulted ? "true" : "false") << ",\"filter_end\":" << state.filterEnd << '}';
    }
    receipt << "]}\n";
  }

  static void writeEvidence(std::ofstream & out, const std::vector<ScanEvidence> & values)
  {
    out.precision(17);
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i)
        out << ',';
      const auto & v = values[i];
      out << "{\"count\":" << v.count << ",\"stamp\":" << v.stamp << ",\"effective\":" << v.effective << '}';
    }
  }

  template<class Predicate>
  bool spinUntil(Predicate ready, double seconds = 8.0)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!ready() && std::chrono::steady_clock::now() < end)
      executor->spin_once(std::chrono::milliseconds(2));
    return ready();
  }

  void sendScan(std::size_t count, double start)
  {
    // Continuous finite IMU coverage; no writes to the estimator or initialization state.
    sendImuThrough(start + 0.06);
    sendLidar(count, start);
  }

  void sendImuThrough(double end)
  {
    const auto through = static_cast<std::int64_t>(std::llround(end * 1e9));
    for (; nextImu <= through; nextImu += 5000000) {
      sensor_msgs::msg::Imu message;
      message.header.stamp = rclcpp::Time(nextImu);
      message.linear_acceleration.z = 9.81;
      imu->publish(message);
      executor->spin_some();
    }
  }

  void sendLidar(std::size_t count, double start)
  {
    pcl::PointCloud<ouster_ros::Point> cloud;
    cloud.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto & point = cloud[i];
      point = ouster_ros::Point{};
      point.x = 10.0f + static_cast<float>(i % 317) * 0.01f;
      point.y = -1.6f + static_cast<float>(i / 317) * 0.01f;
      point.z = 1.0f;
      point.intensity = 1.0f;
      point.t = static_cast<std::uint32_t>(i * 50000000ULL / (count - 1));
    }
    auto message = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(cloud, *message);
    message->header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(start * 1e9)));
    message->header.frame_id = "lidar";
    lidar->publish(std::move(message));
  }

  bool saw(const std::vector<ScanEvidence> & values, std::size_t count, double stamp) const
  {
    return std::any_of(values.begin(), values.end(), [&](const auto & value) {
      return value.count == count && std::abs(value.stamp - stamp) < 1e-6;
    });
  }

  std::unique_ptr<class_loader::ClassLoader> loader;
  std::shared_ptr<rclcpp_components::NodeFactory> factory;
  std::optional<rclcpp_components::NodeInstanceWrapper> subject;
  rclcpp::Node::SharedPtr observer;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom;
  std::vector<ScanEvidence> prepared, modeled;
  std::vector<ImuBatchEvidence> imuBatches;
  std::vector<double> odomStamps;
  std::size_t omittedOdomStamps = 0;
  std::vector<TargetEvidence> targets;
  std::vector<fast_lio_test::ImuInputSnapshot> faultSnapshots;
  std::int64_t nextImu = 990000000;
  double lastOdom = -1.0;
  bool lastPoseFinite = false;
};

#ifndef FASTLIO_INPUT_FAULT_TEST
TEST_F(RosScanBounds, RealSubscriptionsDownsampleAndModelAllThreeBounds)
{
  double start = 1.0;
  double warmupEnd = -1.0;
  // First scan, real IMU initialization, map seeding, then a real lidar update.
  for (int i = 0; i < 6 && modeled.empty(); ++i, start += 0.1) {
    sendScan(2000, start);
    warmupEnd = start + 0.05;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    spinUntil([&] { return std::chrono::steady_clock::now() >= until; }, 1.0);
  }
  ASSERT_FALSE(modeled.empty()) << "real IMU/map initialization did not reach h_model";
  // h_model precedes observer dispatch. Bind the boundary to the last successful
  // warmup scan, rather than accepting a finite odometry from an earlier scan.
  ASSERT_TRUE(spinUntil([&] {
    return saw(modeled, 2000, warmupEnd) && std::isfinite(lastOdom) && lastOdom >= 0.0 && lastPoseFinite &&
           std::abs(lastOdom - warmupEnd) < 1e-6;
  }))
    << "last successful warmup scan did not publish its finite odometry";
  ASSERT_TRUE(std::isfinite(lastOdom));
  ASSERT_GE(lastOdom, 0.0);
  ASSERT_NEAR(lastOdom, warmupEnd, 1e-6);
  ASSERT_FALSE(imuBatches.empty());
  EXPECT_EQ(imuBatches.front().reason, fast_lio::ImuProcessReason::kBootstrap);
  EXPECT_EQ(imuBatches.front().disposition, fast_lio::ImuDisposition::kCommitted);
  const std::size_t beforeOverlap = imuBatches.size();
  const double overlapBegin = lastOdom - 0.000511169433594;
  const double overlapEnd = overlapBegin + 0.05;
  sendScan(2000, overlapBegin);
  ASSERT_TRUE(spinUntil([&] { return imuBatches.size() > beforeOverlap; }));
  const auto recovered = imuBatches.back();
  ASSERT_EQ(recovered.reason, fast_lio::ImuProcessReason::kProcessed);
  EXPECT_EQ(recovered.disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_GT(recovered.samples, 0u);
  ASSERT_TRUE(spinUntil([&] { return std::abs(lastOdom - overlapEnd) < 1e-6 && lastPoseFinite; }));
  sendScan(2000, start);
  ASSERT_TRUE(spinUntil([&] { return imuBatches.back().token > recovered.token; }));
  const auto consumed = imuBatches.back();
  EXPECT_EQ(consumed.disposition, fast_lio::ImuDisposition::kCommitted);
  EXPECT_GT(consumed.samples, 0u);
  start += 0.1;
  for (const std::size_t count : {99999u, 100000u, 100001u}) {
    sendScan(count, start);
    const double end = start + 0.05;
    // Execution allowance for the production callback and subsequent observer dispatch;
    // this wall-clock budget is not a performance acceptance threshold.
    const auto waitStart = std::chrono::steady_clock::now();
    const bool ready = spinUntil(
      [&] { return saw(prepared, count, end) && saw(modeled, count, end) && std::abs(lastOdom - end) < 1e-6; },
      kTargetWaitSeconds);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count();
    if (targets.size() < kTargetEvidenceCapacity)
      targets.push_back({count,
                         end,
                         elapsed,
                         saw(prepared, count, end),
                         saw(modeled, count, end),
                         std::abs(lastOdom - end) < 1e-6,
                         lastOdom,
                         lastPoseFinite});
    ASSERT_TRUE(ready) << "production prepare/h_model/Odometry missing exact downsampled count=" << count;
    EXPECT_TRUE(lastPoseFinite);
    start += 0.1;
  }
  EXPECT_GT(imuBatches.back().pending, 0u);  // EOF keeps the future, unselected tail explicit.
}
#else
TEST_F(RosScanBounds, FatalTransactionRejectsRealInputsAndBoundsBothQueues)
{
  double start = 1.0;
  double warmupEnd = -1.0;
  for (int i = 0; i < 6 && modeled.empty(); ++i, start += 0.1) {
    sendScan(2000, start);
    warmupEnd = start + 0.05;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    spinUntil([&] { return std::chrono::steady_clock::now() >= until; }, 1.0);
  }
  ASSERT_TRUE(spinUntil([&] {
    return saw(modeled, 2000, warmupEnd) && lastPoseFinite && std::isfinite(lastOdom) &&
           std::abs(lastOdom - warmupEnd) < 1e-6;
  }));
  const auto before = fast_lio_test::inspectImuInput();
  const auto published = odomStamps.size();
  const auto modeledCount = modeled.size();
  const auto batches = imuBatches.size();
  int processCalls = 0;
  fast_lio_test::beforeImuProcess = [&] {
    ++processCalls;
    throw std::runtime_error("injected IMU processing fault");
  };
  sendScan(2000, start);
  ASSERT_TRUE(spinUntil([&] { return imuBatches.size() > batches; }));
  ASSERT_EQ(imuBatches.back().disposition, fast_lio::ImuDisposition::kFatal);
  const auto latched = fast_lio_test::inspectImuInput();
  faultSnapshots.push_back(latched);
  EXPECT_TRUE(latched.faulted);
  EXPECT_TRUE(latched.mapFrozen);
  EXPECT_TRUE(latched.degradedOdom);
  EXPECT_EQ(latched.lidarQueued, 0u);
  EXPECT_EQ(latched.imuQueued, 0u);
  EXPECT_GT(latched.discardedImu, 0u);  // the unselected future tail is accounted
  EXPECT_DOUBLE_EQ(latched.filterEnd, before.filterEnd);

  // A leaves this scan protected forever. Subsequent IMU-only traffic is thirty
  // times the configured backlog window, followed by more real LiDAR inputs.
  sendLidar(2000, start + 0.1);
  executor->spin_some();
  sendImuThrough(start + 30.0);
  const auto afterImu = fast_lio_test::inspectImuInput();
  faultSnapshots.push_back(afterImu);
  EXPECT_EQ(afterImu.lidarQueued, 0u);
  EXPECT_LE(afterImu.imuQueued, 1u);
  EXPECT_GT(afterImu.rejectedImu, latched.rejectedImu + 1000);
  for (int i = 1; i <= 8; ++i) {
    sendLidar(2000, start + 30.0 + i * 0.1);
    executor->spin_some();
  }
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  spinUntil([&] { return std::chrono::steady_clock::now() >= until; }, 1.0);
  const auto final = fast_lio_test::inspectImuInput();
  faultSnapshots.push_back(final);
  EXPECT_EQ(final.lidarQueued, 0u);
  EXPECT_LE(final.imuQueued, 1u);
  EXPECT_GE(final.rejectedLidar, latched.rejectedLidar + 9);
  EXPECT_DOUBLE_EQ(final.filterEnd, before.filterEnd);
  EXPECT_EQ(processCalls, 1);
  EXPECT_EQ(imuBatches.size(), batches + 1);
  EXPECT_EQ(modeled.size(), modeledCount);
  EXPECT_EQ(odomStamps.size(), published);
  const auto confirmation = static_cast<fast_lio::ImuConfirmation>(fast_lio_test::reconfirmLastImuBatch());
  EXPECT_EQ(confirmation, fast_lio::ImuConfirmation::kStaleEpoch);
  EXPECT_NE(confirmation, fast_lio::ImuConfirmation::kReleased);
  EXPECT_NE(confirmation, fast_lio::ImuConfirmation::kRetained);
  EXPECT_EQ(fast_lio_test::inspectImuInput().imuQueued, final.imuQueued);
}
#endif
}  // namespace

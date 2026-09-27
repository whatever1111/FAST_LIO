#include <nav_msgs/msg/odometry.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <algorithm>
#include <chrono>
#include <class_loader/class_loader.hpp>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

#include "laser_mapping_test_hooks.hpp"
#include "preprocess.h"

namespace
{
struct ScanEvidence
{
  std::size_t count;
  double stamp;
  int effective;
};

class RosScanBounds : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    executor = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    loader = std::make_unique<class_loader::ClassLoader>(FASTLIO_TEST_COMPONENT_PATH);
    const auto names = loader->getAvailableClasses<rclcpp_components::NodeFactory>();
    ASSERT_EQ(names.size(), 1u);
    factory = loader->createInstance<rclcpp_components::NodeFactory>(names.front());
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    options.parameter_overrides({rclcpp::Parameter("common.lid_topic", "/test_scan_bounds/points"),
                                 rclcpp::Parameter("common.imu_topic", "/test_scan_bounds/imu"),
                                 rclcpp::Parameter("preprocess.lidar_type", 3),
                                 rclcpp::Parameter("preprocess.timestamp_unit", 3),
                                 rclcpp::Parameter("point_filter_num", 1),
                                 rclcpp::Parameter("feature_extract_enable", false),
                                 rclcpp::Parameter("filter_size_surf", 0.001),
                                 rclcpp::Parameter("filter_size_map", 0.05),
                                 rclcpp::Parameter("imu_init_require_still", false),
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
      });
    fast_lio_test::afterScanPrepared = [this](std::size_t count, double stamp) {
      if (prepared.size() < 64)
        prepared.push_back({count, stamp, -1});
    };
    fast_lio_test::afterHModelRows = [this](std::size_t count, double stamp, int effective) {
      if (modeled.size() < 128)
        modeled.push_back({count, stamp, effective});
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
            << ",\"prepared\":[";
    writeEvidence(receipt, prepared);
    receipt << "],\"h_model\":[";
    writeEvidence(receipt, modeled);
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
    const auto through = static_cast<std::int64_t>(std::llround((start + 0.06) * 1e9));
    for (; nextImu <= through; nextImu += 5000000) {
      sensor_msgs::msg::Imu message;
      message.header.stamp = rclcpp::Time(nextImu).to_msg();
      message.linear_acceleration.z = 9.81;
      imu->publish(message);
      executor->spin_some();
    }
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
    message->header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(start * 1e9))).to_msg();
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
  std::int64_t nextImu = 990000000;
  double lastOdom = -1.0;
  bool lastPoseFinite = false;
};

TEST_F(RosScanBounds, RealSubscriptionsDownsampleAndModelAllThreeBounds)
{
  double start = 1.0;
  // First scan, real IMU initialization, map seeding, then a real lidar update.
  for (int i = 0; i < 6 && modeled.empty(); ++i, start += 0.1) {
    sendScan(2000, start);
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    spinUntil([&] { return std::chrono::steady_clock::now() >= until; }, 1.0);
  }
  ASSERT_FALSE(modeled.empty()) << "real IMU/map initialization did not reach h_model";
  for (const std::size_t count : {99999u, 100000u, 100001u}) {
    sendScan(count, start);
    const double end = start + 0.05;
    ASSERT_TRUE(spinUntil(
      [&] { return saw(prepared, count, end) && saw(modeled, count, end) && std::abs(lastOdom - end) < 1e-6; }, 15.0))
      << "production prepare/h_model/Odometry missing exact downsampled count=" << count;
    EXPECT_TRUE(lastPoseFinite);
    start += 0.1;
  }
}
}  // namespace

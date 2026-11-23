#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <chrono>
#include <condition_variable>
#include <fast_lio_interfaces/msg/lidar_window.hpp>
#include <mutex>
#include <stdexcept>

#include "laser_mapping_test_hooks.hpp"

// Load this test probe before constructing LaserMappingNode so the hooks are ready.
// Release the barriers and destroy LaserMappingNode before destroying the probe.
class LidarCustodyTestProbe : public rclcpp::Node
{
public:
  using Window = fast_lio_interfaces::msg::LidarWindow;
  using Trigger = std_srvs::srv::Trigger;
  using SetBool = std_srvs::srv::SetBool;

  explicit LidarCustodyTestProbe(const rclcpp::NodeOptions & options) : Node("lidar_custody_test_probe", options)
  {
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(64)).reliable();
    windows_ = create_publisher<Window>("~/received", qos);
    dispositions_ = create_publisher<std_msgs::msg::UInt64>("~/disposition", qos);
    occupied_ = create_publisher<std_msgs::msg::String>("~/occupied", qos);
    arm_ =
      create_service<SetBool>("~/arm_imu", [this](SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res) {
        std::lock_guard lock(mutex_);
        armImu_ = req->data;
        released_ = !armImu_;
        res->success = true;
        cv_.notify_all();
      });
    release_ =
      create_service<Trigger>("~/release", [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr res) {
        std::lock_guard lock(mutex_);
        released_ = true;
        res->success = true;
        cv_.notify_all();
      });
    fast_lio_test::installCustodyTestHold = [this](rclcpp::Node & front) {
      return front.create_service<Trigger>("~/custody_test_hold",
                                           [this](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr res) {
                                             std::unique_lock lock(mutex_);
                                             released_ = false;
                                             res->success = pause(lock, "group");
                                           });
    };
    fast_lio_test::beforeImuProcess = [this] {
      std::unique_lock lock(mutex_);
      if (armImu_) {
        armImu_ = false;
        if (!pause(lock, "imu"))
          throw std::runtime_error("test IMU barrier expired");
      }
    };
    fast_lio_test::beforeCustodyIngress = [this](const auto & header, const auto & cloud) {
      Window observation;
      observation.sender = header.sender;
      observation.receiver = header.receiver;
      observation.sequence = header.sequence;
      observation.window_id = header.windowId;
      observation.cloud = cloud;
      windows_->publish(observation);
    };
    fast_lio_test::afterCustodyIngress = [this](const auto & header, auto outcome) {
      std_msgs::msg::UInt64 result;
      result.data = (header.sequence << 8U) | static_cast<std::uint8_t>(outcome);
      dispositions_->publish(result);
    };
  }

  ~LidarCustodyTestProbe() override
  {
    fast_lio_test::installCustodyTestHold = {};
    fast_lio_test::beforeImuProcess = {};
    fast_lio_test::beforeCustodyIngress = {};
    fast_lio_test::afterCustodyIngress = {};
  }

private:
  bool pause(std::unique_lock<std::mutex> & lock, const char * tag)
  {
    std_msgs::msg::String entered;
    entered.data = tag;
    occupied_->publish(entered);
    return cv_.wait_for(lock, std::chrono::seconds(20), [this] { return released_; });
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool armImu_{};
  bool released_{};
  rclcpp::Publisher<Window>::SharedPtr windows_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr dispositions_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr occupied_;
  rclcpp::Service<SetBool>::SharedPtr arm_;
  rclcpp::Service<Trigger>::SharedPtr release_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(LidarCustodyTestProbe)

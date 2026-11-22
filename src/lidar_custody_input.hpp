#pragma once

#include <rclcpp/rclcpp.hpp>

#include <exception>
#include <fast_lio_interfaces/lidar_custody_ros.hpp>
#include <fast_lio_interfaces/msg/lidar_window.hpp>
#include <fast_lio_interfaces/msg/lidar_window_ack.hpp>
#include <fast_lio_interfaces/srv/bind_lidar_custody.hpp>
#include <functional>
#include <sstream>
#include <utility>

#include "lidar_custody_receiver.hpp"

namespace fast_lio
{
// All entry points use LaserMappingNode's default mutually exclusive group.
class LidarCustodyInput
{
public:
  using Cloud = sensor_msgs::msg::PointCloud2;
  using Window = fast_lio_interfaces::msg::LidarWindow;
  using Ack = fast_lio_interfaces::msg::LidarWindowAck;
  using Bind = fast_lio_interfaces::srv::BindLidarCustody;
  using Ingress = std::function<LidarIngress(const LidarWindowHeader &, Cloud::UniquePtr)>;

  LidarCustodyInput(rclcpp::Node & node, std::string endpoint, bool compatible, Ingress ingress)
      : node_(node),
        endpoint_(std::move(endpoint)),
        state_(fast_lio_interfaces::newLidarInstance()),
        ingress_(std::move(ingress))
  {
    using fast_lio_interfaces::isLidarCustodyQos;
    using fast_lio_interfaces::kLidarAckDepth;
    using fast_lio_interfaces::kLidarDataDepth;
    using fast_lio_interfaces::lidarCustodyQos;
    static_assert(kLidarDataDepth == kLidarCustodyDepth);
    ack_ = node_.create_publisher<Ack>(endpoint_ + "/ack", lidarCustodyQos(kLidarAckDepth));
    if (compatible) {
      input_ = node_.create_subscription<Window>(endpoint_ + "/window",
                                                 lidarCustodyQos(kLidarDataDepth),
                                                 [this](Window::UniquePtr msg) { receive(std::move(msg)); });
    }
    if (!input_ || !isLidarCustodyQos(input_->get_actual_qos(), kLidarDataDepth) ||
        !isLidarCustodyQos(ack_->get_actual_qos(), kLidarAckDepth)) {
      state_.fail("incompatible_fast_input_qos_or_type");
    }
    binder_ = node_.create_service<Bind>(
      endpoint_ + "/bind",
      [this](const Bind::Request::SharedPtr req, Bind::Response::SharedPtr res) { bind(*req, *res); });
    logFailure();
  }

  ~LidarCustodyInput()
  {
    std::ostringstream partitions;
    for (std::size_t i = 0; i < state_.counts().size(); ++i) {
      partitions << ' ' << kLidarIngressNames[i] << '=' << state_.counts()[i].windows << ' ' << kLidarIngressNames[i]
                 << "_points=" << state_.counts()[i].points;
    }
    RCLCPP_INFO(node_.get_logger(),
                "lidar_custody_fast (final) sender=%s receiver=%s r=%llu r_points=%llu last_id=%lld failure=%s%s",
                fast_lio_interfaces::lidarInstanceText(state_.sender()).c_str(),
                fast_lio_interfaces::lidarInstanceText(state_.receiver()).c_str(),
                static_cast<unsigned long long>(state_.sequence()),
                static_cast<unsigned long long>(state_.points()),
                static_cast<long long>(state_.windowId()),
                state_.failure() ? state_.failure() : "none",
                partitions.str().c_str());
  }

  bool failed() const { return state_.failure() != nullptr; }

private:
  Ack receipt() const
  {
    Ack ack;
    ack.sender = state_.sender();
    ack.receiver = state_.receiver();
    ack.sequence = state_.sequence();
    ack.points = state_.points();
    ack.window_id = state_.windowId();
    if (state_.failure())
      ack.failure = state_.failure();
    return ack;
  }

  bool matched()
  {
    const auto writers = node_.count_publishers(input_->get_topic_name());
    const auto readers = node_.count_subscribers(ack_->get_topic_name());
    if (writers > 1 || readers > 1)
      state_.fail("ambiguous_fast_pair");
    return writers == 1 && readers == 1 && input_->get_publisher_count() == 1 && ack_->get_subscription_count() == 1;
  }

  void bind(const Bind::Request & request, Bind::Response & response)
  {
    response.challenge = request.challenge;
    if (!failed() && matched()) {
      const bool first = state_.sender() == LidarInstance{};
      response.ready = state_.bind(request.sender, request.receiver);
      if (response.ready && first) {
        RCLCPP_INFO(node_.get_logger(),
                    "lidar_custody_fast BOUND endpoint=%s sender=%s receiver=%s data_depth=5 ack_depth=1",
                    endpoint_.c_str(),
                    fast_lio_interfaces::lidarInstanceText(state_.sender()).c_str(),
                    fast_lio_interfaces::lidarInstanceText(state_.receiver()).c_str());
      }
    }
    response.prefix = receipt();
    logFailure();
  }

  void receive(Window::UniquePtr window)
  {
    const LidarWindowHeader header{window->sender,
                                   window->receiver,
                                   window->sequence,
                                   window->window_id,
                                   std::uint64_t(window->cloud.width) * window->cloud.height};
    if (!failed() && !matched())
      state_.fail("fast_pair_disconnected");
    if (state_.validate(header)) {
      try {
        auto cloud = std::make_unique<Cloud>(std::move(window->cloud));
        state_.complete(header, ingress_(header, std::move(cloud)));
      } catch (const std::exception & e) {
        state_.fail("fast_ingress_exception");
        RCLCPP_ERROR(node_.get_logger(), "lidar_custody_fast ingress exception: %s", e.what());
      }
    }
    try {
      ack_->publish(receipt());
    } catch (const std::exception & e) {
      state_.fail("fast_ack_publish_unknown");
      RCLCPP_ERROR(node_.get_logger(), "lidar_custody_fast ACK publication exception: %s", e.what());
    }
    logFailure();
  }

  void logFailure()
  {
    if (failed() && !loggedFailure_) {
      loggedFailure_ = true;
      RCLCPP_ERROR(node_.get_logger(),
                   "lidar_custody_fast FAILED endpoint=%s r=%llu reason=%s",
                   endpoint_.c_str(),
                   static_cast<unsigned long long>(state_.sequence()),
                   state_.failure());
    }
  }

  rclcpp::Node & node_;
  std::string endpoint_;
  LidarCustodyReceiver state_;
  Ingress ingress_;
  bool loggedFailure_{};
  rclcpp::Publisher<Ack>::SharedPtr ack_;
  rclcpp::Subscription<Window>::SharedPtr input_;
  rclcpp::Service<Bind>::SharedPtr binder_;
};
}  // namespace fast_lio

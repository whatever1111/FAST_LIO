#pragma once

#include <rclcpp/qos.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <string>

namespace fast_lio_interfaces
{
inline constexpr std::size_t kLidarDataDepth = 5;
inline constexpr std::size_t kLidarAckDepth = 1;

inline rclcpp::QoS lidarCustodyQos(std::size_t depth)
{
  return rclcpp::QoS(rclcpp::KeepLast(depth)).reliable().durability_volatile();
}

inline bool isLidarCustodyQos(const rclcpp::QoS & qos, std::size_t depth)
{
  const auto & profile = qos.get_rmw_qos_profile();
  return profile.history == RMW_QOS_POLICY_HISTORY_KEEP_LAST && profile.depth == depth &&
         profile.reliability == RMW_QOS_POLICY_RELIABILITY_RELIABLE &&
         profile.durability == RMW_QOS_POLICY_DURABILITY_VOLATILE;
}

inline std::array<std::uint8_t, 16> newLidarInstance()
{
  std::random_device random;
  std::array<std::uint8_t, 16> id;
  for (auto & byte : id)
    byte = static_cast<std::uint8_t>(random());
  // Reserve the all-zero sentinel, including on implementations with UUID bits.
  id[6] = static_cast<std::uint8_t>((id[6] & 0x0fU) | 0x40U);
  id[8] = static_cast<std::uint8_t>((id[8] & 0x3fU) | 0x80U);
  return id;
}

inline std::string lidarInstanceText(const std::array<std::uint8_t, 16> & id)
{
  constexpr char kHex[] = "0123456789abcdef";
  std::string value;
  value.reserve(id.size() * 2);
  for (const auto byte : id) {
    value.push_back(kHex[byte >> 4U]);
    value.push_back(kHex[byte & 0x0fU]);
  }
  return value;
}
}  // namespace fast_lio_interfaces

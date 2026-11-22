#pragma once

#include <array>
#include <cstdint>
#include <limits>

namespace fast_lio
{
using LidarInstance = std::array<std::uint8_t, 16>;
inline constexpr std::size_t kLidarCustodyDepth = 5;

enum class LidarIngress
{
  kEnqueued,
  kEpochRejected,
  kLeadRejected,
  kTransactionRejected,
  kLateRejected,
  kUnresolved,
};
inline constexpr std::array kLidarIngressNames{
  "enqueued", "epoch_rejected", "lead_rejected", "transaction_rejected", "late_rejected"};

struct LidarIngressCount
{
  std::uint64_t windows{};
  std::uint64_t points{};
};

struct LidarWindowHeader
{
  LidarInstance sender{};
  LidarInstance receiver{};
  std::uint64_t sequence{};
  std::int64_t windowId{};
  std::uint64_t points{};
};

// Owned by the same mutually exclusive group as standard_pcl_cbk.
class LidarCustodyReceiver
{
public:
  explicit LidarCustodyReceiver(LidarInstance instance) : receiver_(instance) {}

  bool bind(const LidarInstance & sender, const LidarInstance & expectedReceiver)
  {
    if (failure_)
      return false;
    if (sender == LidarInstance{} || (expectedReceiver != LidarInstance{} && expectedReceiver != receiver_) ||
        (sender_ != LidarInstance{} && sender_ != sender) ||
        (sender_ == LidarInstance{} && expectedReceiver != LidarInstance{})) {
      fail("bind_instance");
      return false;
    }
    sender_ = sender;
    return true;
  }

  bool validate(const LidarWindowHeader & window)
  {
    if (failure_)
      return false;
    if (sender_ == LidarInstance{} || window.sender != sender_ || window.receiver != receiver_) {
      fail("window_instance");
    } else if (sequence_ == std::numeric_limits<std::uint64_t>::max() || window.sequence != sequence_ + 1) {
      fail("window_sequence");
    } else if (window.points == 0 || window.points > std::numeric_limits<std::uint64_t>::max() - points_ ||
               (sequence_ != 0 && window.windowId <= windowId_)) {
      fail("window_metadata");
    }
    return !failure_;
  }

  bool complete(const LidarWindowHeader & window, LidarIngress disposition)
  {
    if (!validate(window))
      return false;
    const auto index = static_cast<std::size_t>(disposition);
    if (index >= counts_.size()) {
      fail("ingress_unresolved");
      return false;
    }
    ++counts_[index].windows;
    counts_[index].points += window.points;
    sequence_ = window.sequence;
    points_ += window.points;
    windowId_ = window.windowId;
    return true;
  }

  void fail(const char * reason)
  {
    if (!failure_)
      failure_ = reason;
  }

  const LidarInstance & sender() const { return sender_; }
  const LidarInstance & receiver() const { return receiver_; }
  std::uint64_t sequence() const { return sequence_; }
  std::uint64_t points() const { return points_; }
  std::int64_t windowId() const { return windowId_; }
  const auto & counts() const { return counts_; }
  const char * failure() const { return failure_; }

private:
  LidarInstance sender_{};
  LidarInstance receiver_;
  std::uint64_t sequence_{};
  std::uint64_t points_{};
  std::int64_t windowId_{};
  std::array<LidarIngressCount, kLidarIngressNames.size()> counts_{};
  const char * failure_{};
};
}  // namespace fast_lio

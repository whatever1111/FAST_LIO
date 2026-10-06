#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "imu_process_outcome.hpp"

namespace fast_lio
{
template<class Samples, class Stamp, class Valid>
bool canRetainImuBatch(const Samples & samples, const Stamp & stamp, const Valid & valid)
{
  double previous = 0.0;
  for (const auto & sample : samples) {
    if (!valid(sample))
      return false;
    const double current = stamp(sample);
    if (!std::isfinite(current) || current < 0.0 || current < previous)
      return false;
    previous = current;
  }
  return true;
}

template<class Queue, class Stamp>
std::size_t imuPrefixThrough(const Queue & queue, double end, const Stamp & stamp)
{
  std::size_t count = 0;
  while (count < queue.size() && stamp(queue[count]) <= end)
    ++count;
  return count;
}

enum class ImuConfirmation
{
  kRetained,
  kReleased,
  kInvalidDropped,
  kFatal,
  kDuplicate,
  kStaleEpoch,
  kPrefixMismatch
};

// One mutually exclusive callback group owns selection through settlement. The
// caller holds its FIFO mutex only inside borrow/settle/invalidate, never while
// propagating the filter. Identity equality is supplied by the queue element.
template<class Sample>
class ImuFifoTransaction
{
public:
  struct Token
  {
    std::uint64_t epoch = 0;
    std::uint64_t sequence = 0;
    std::vector<Sample> prefix;
  };

  template<class Queue>
  Token borrow(const Queue & queue, std::size_t count)
  {
    if (faulted_ || active_ != 0 || count > queue.size()) {
      faulted_ = true;
      throw std::logic_error("invalid IMU transaction selection");
    }
    Token token{epoch_, ++sequence_, {}};
    token.prefix.assign(queue.begin(), queue.begin() + count);
    active_ = token.sequence;
    return token;
  }

  template<class Queue>
  ImuConfirmation settle(Queue & queue, const Token & token, ImuDisposition disposition)
  {
    if (token.epoch != epoch_) {
      faulted_ = true;
      return ImuConfirmation::kStaleEpoch;
    }
    if (token.sequence == 0 || token.sequence <= settled_) {
      ++duplicateConfirmations_;
      faulted_ = true;
      return ImuConfirmation::kDuplicate;
    }
    if (faulted_ || token.sequence != active_ || token.prefix.size() > queue.size() ||
        !std::equal(token.prefix.begin(), token.prefix.end(), queue.begin())) {
      faulted_ = true;
      return ImuConfirmation::kPrefixMismatch;
    }
    settled_ = token.sequence;
    active_ = 0;
    if (disposition == ImuDisposition::kUncommitted)
      return ImuConfirmation::kRetained;
    for (std::size_t i = 0; i < token.prefix.size(); ++i)
      queue.pop_front();
    if (disposition == ImuDisposition::kFatal) {
      faulted_ = true;
      return ImuConfirmation::kFatal;
    }
    return disposition == ImuDisposition::kInvalid ? ImuConfirmation::kInvalidDropped : ImuConfirmation::kReleased;
  }

  void invalidate()
  {
    ++epoch_;
    active_ = 0;
  }
  void failClosed() { faulted_ = true; }
  bool faulted() const { return faulted_; }
  std::uint64_t duplicateConfirmations() const { return duplicateConfirmations_; }

private:
  std::uint64_t epoch_ = 1;
  std::uint64_t sequence_ = 0;
  std::uint64_t active_ = 0;
  std::uint64_t settled_ = 0;
  std::uint64_t duplicateConfirmations_ = 0;
  bool faulted_ = false;
};

// Observe actual positive predict intervals and deliberate cursor commits.
// The ledger never changes their boundaries or the filter's numerical order.
class ImuIntegrationLedger
{
public:
  void observe(double begin, double end)
  {
    if (!std::isfinite(begin) || !std::isfinite(end) || end < begin) {
      ++watermarkRegressions_;
      return;
    }
    if (end == begin)
      return;
    if (hasIntegrated_ && begin < integratedEnd_)
      ++duplicateIntegrations_;
    integratedEnd_ = end;
    hasIntegrated_ = true;
  }

  void anchor(double end)
  {
    if (!std::isfinite(end) || (hasCommitted_ && end < committedEnd_))
      ++watermarkRegressions_;
    committedEnd_ = end;
    integratedEnd_ = end;
    hasCommitted_ = hasIntegrated_ = true;
  }
  std::uint64_t duplicateIntegrations() const { return duplicateIntegrations_; }
  std::uint64_t watermarkRegressions() const { return watermarkRegressions_; }

private:
  double committedEnd_ = 0.0;
  double integratedEnd_ = 0.0;
  std::uint64_t duplicateIntegrations_ = 0;
  std::uint64_t watermarkRegressions_ = 0;
  bool hasCommitted_ = false;
  bool hasIntegrated_ = false;
};
}  // namespace fast_lio

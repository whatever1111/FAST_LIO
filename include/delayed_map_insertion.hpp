#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <utility>

namespace fast_lio
{

struct NoProvenance
{};

template<typename Cloud, typename Provenance>
struct DelayedMapBatch
{
  Cloud first;
  Cloud second;
  Provenance provenance;
};

// Single-owner queue of already transformed world-frame points. Reset it when
// that live-map history is discarded, after joining any previously dispatched
// add. It owns no worker and does not synchronize the map backend.
template<typename Cloud, typename Provenance = NoProvenance>
class DelayedMapInsertion
{
public:
  using Batch = DelayedMapBatch<Cloud, Provenance>;  // downsampled, non-downsampled, staging provenance

  std::optional<Batch> stage(Cloud downsampled, Cloud direct, int delayScans, Provenance provenance = Provenance{})
  {
    Batch batch{std::move(downsampled), std::move(direct), std::move(provenance)};
    if (delayScans <= 0)
      return batch;
    pending_.push_back(std::move(batch));
    if (pending_.size() <= static_cast<std::size_t>(delayScans))
      return std::nullopt;
    Batch ready = std::move(pending_.front());
    pending_.pop_front();
    return ready;
  }

  void reset() { pending_.clear(); }
  std::size_t pendingSize() const { return pending_.size(); }
  // Owner-only diagnostic view; never associates an old payload with the current scan's timestamp.
  const std::deque<Batch> & pending() const { return pending_; }

private:
  std::deque<Batch> pending_;
};

}  // namespace fast_lio

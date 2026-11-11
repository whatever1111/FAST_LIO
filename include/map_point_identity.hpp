#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace fast_lio
{

enum class MapPointOrigin : std::uint8_t
{
  kUnlabelled = 0,
  kScanSeed = 1,
  kScanIncremental = 2,
  kPrior = 3,
};

// Scan frame_seq is the staging scan's lifecycle_scan_seq; point_index is its downsampled-cloud index.
// Seed poses are keyed by (frame_seq, kScanSeed, generation), incremental poses by
// (frame_seq, kScanIncremental). Multiple rebuilds in one scan can use different seed poses.
// Empty-map builds need not be the first scan: the seed pose is the state used at the transform.
// Scan (frame_seq, point_index) names one echo, shared by its seed and incremental copies;
// the four fields identify a map point, not an independent noise source.
// For kPrior, frame_seq is the installation batch and point_index is the prior-cloud index.
// Origin describes provenance, not voxel pinning. Generation is fixed at admission, even
// for scan points surviving clearLive() in pinned voxels. No state snapshot is stored here.
struct MapPointIdentity
{
  std::uint32_t frame_seq = 0;
  std::uint32_t point_index = 0;
  std::uint32_t generation = 0;
  MapPointOrigin origin = MapPointOrigin::kUnlabelled;
  std::uint8_t reserved[3] = {};
};
static_assert(sizeof(MapPointIdentity) == 16);
static_assert(std::is_standard_layout_v<MapPointIdentity> && std::is_trivially_copyable_v<MapPointIdentity>);

enum class MapIdentityStatus : std::uint8_t
{
  kOk = 0,
  kDisabled = 1,
  kUnsupportedBackend = 2,
};

// The optional index array is borrowed for this insertion call only.
struct MapInsertLabel
{
  MapPointOrigin origin = MapPointOrigin::kUnlabelled;
  std::uint32_t frame_seq = 0;
  const std::vector<std::uint32_t> * index = nullptr;
  std::size_t prior_prefix = 0;
};

// Owned by its staging batch, including while delayed or handed to the map worker.
// The default (disabled) value never allocates.
struct MapInsertProvenance
{
  std::uint32_t frame_seq = 0;
  std::vector<std::uint32_t> downsampled_index;
  std::vector<std::uint32_t> direct_index;

  void noteDownsampled(int i)
  {
    if (frame_seq != 0)
      downsampled_index.push_back(static_cast<std::uint32_t>(i));
  }
  void noteDirect(int i)
  {
    if (frame_seq != 0)
      direct_index.push_back(static_cast<std::uint32_t>(i));
  }
};

inline std::optional<std::uint32_t> mapFrameSeq(std::uint64_t lifecycle_scan_seq)
{
  if (lifecycle_scan_seq == 0 || lifecycle_scan_seq > std::numeric_limits<std::uint32_t>::max())
    return std::nullopt;
  return static_cast<std::uint32_t>(lifecycle_scan_seq);
}

inline MapInsertProvenance beginScanProvenance(bool enabled, std::uint64_t lifecycle_scan_seq, std::size_t reserve)
{
  MapInsertProvenance provenance;
  const auto frame = mapFrameSeq(lifecycle_scan_seq);
  if (enabled && frame) {
    provenance.frame_seq = *frame;
    provenance.downsampled_index.reserve(reserve);
    provenance.direct_index.reserve(reserve);
  }
  return provenance;
}

inline MapInsertLabel seedLabel(std::uint64_t lifecycle_scan_seq, std::size_t prior_prefix = 0)
{
  const auto frame = mapFrameSeq(lifecycle_scan_seq);
  return {frame ? MapPointOrigin::kScanSeed : MapPointOrigin::kUnlabelled, frame.value_or(0), nullptr, prior_prefix};
}

inline MapInsertLabel incrementalLabel(const MapInsertProvenance & provenance, bool downsampled)
{
  return {provenance.frame_seq != 0 ? MapPointOrigin::kScanIncremental : MapPointOrigin::kUnlabelled,
          provenance.frame_seq,
          downsampled ? &provenance.downsampled_index : &provenance.direct_index,
          0};
}

inline MapPointIdentity identityOf(const MapInsertLabel & label,
                                   std::size_t position,
                                   std::size_t batch_size,
                                   std::uint32_t generation,
                                   std::uint32_t prior_batch)
{
  MapPointIdentity id;
  id.generation = generation;
  if (position > std::numeric_limits<std::uint32_t>::max())
    return id;
  if (label.origin == MapPointOrigin::kPrior || position < label.prior_prefix)
    return {prior_batch, static_cast<std::uint32_t>(position), generation, MapPointOrigin::kPrior, {}};
  if ((label.origin != MapPointOrigin::kScanSeed && label.origin != MapPointOrigin::kScanIncremental) ||
      label.frame_seq == 0)
    return id;
  if (label.index != nullptr && label.index->size() != batch_size)
    return id;
  const auto index =
    label.index != nullptr ? (*label.index)[position] : static_cast<std::uint32_t>(position - label.prior_prefix);
  return {label.frame_seq, index, generation, label.origin, {}};
}

}  // namespace fast_lio

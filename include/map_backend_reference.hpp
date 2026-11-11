// map_backend_reference.hpp — the node's local map backend and whether it holds points.
//
// The front end's first local reference is its normal initial map build only when that build returned with the
// map backend holding points. Calling the build proves nothing: iVox clears itself and then skips every point it
// cannot place (a non-finite coordinate, or one whose voxel index leaves the int range), so a build of points it
// rejects all leaves it as empty as it was. The test is the one the node already uses to decide that the map still
// needs its initial build: iVox empty(), ikd-Tree a null root.
//
// No ROS. Over the two backends' own headers and the front end's point types.

#pragma once

#include "frontend_point_types.hpp"
#include "ikd-Tree/ikd_Tree.h"
#include "ivox/ivox.hpp"

// The backend the node compiles: iVox when the target defines USE_IVOX, ikd-Tree otherwise. Every translation unit
// of one target that names MapBackend sees the same definition, because USE_IVOX is set per target.
#ifdef USE_IVOX
using MapBackend = lio_ivox::IVox<PointType>;
#else
using MapBackend = KD_TREE<PointType>;
#endif

namespace fast_lio
{

template<typename PointType>
bool mapBackendHoldsPoints(const lio_ivox::IVox<PointType> & map)
{
  return !map.empty();
}

template<typename PointType>
bool mapBackendHoldsPoints(const KD_TREE<PointType> & map)
{
  return map.Root_Node != nullptr;
}

template<typename PointType>
MapIdentityStatus enableMapPointIdentity(lio_ivox::IVox<PointType> & map, bool requested)
{
  return requested && map.enablePointIdentity() ? MapIdentityStatus::kOk : MapIdentityStatus::kDisabled;
}
template<typename PointType>
MapIdentityStatus enableMapPointIdentity(KD_TREE<PointType> &, bool)
{
  return MapIdentityStatus::kUnsupportedBackend;
}
template<typename PointType>
MapIdentityStatus mapPointIdentityStatus(const lio_ivox::IVox<PointType> & map)
{
  return map.pointIdentityEnabled() ? MapIdentityStatus::kOk : MapIdentityStatus::kDisabled;
}
template<typename PointType>
MapIdentityStatus mapPointIdentityStatus(const KD_TREE<PointType> &)
{
  return MapIdentityStatus::kUnsupportedBackend;
}

// These counters are maintained even with identity disabled. Only the scan thread resets
// the map or installs priors, so reads on that thread need no join. ikd has neither counter.
template<typename PointType>
std::optional<std::uint32_t> mapGeneration(const lio_ivox::IVox<PointType> & map)
{
  return map.mapGeneration();
}
template<typename PointType>
std::optional<std::uint32_t> mapGeneration(const KD_TREE<PointType> &)
{
  return std::nullopt;
}
// Each pinned installation passing preflight and each merged Build with a nonempty prior
// prefix advances the batch, including unpinned priors. clearPinned() does not advance it.
template<typename PointType>
std::optional<std::uint32_t> priorBatchSeq(const lio_ivox::IVox<PointType> & map)
{
  return map.priorBatchSeq();
}
template<typename PointType>
std::optional<std::uint32_t> priorBatchSeq(const KD_TREE<PointType> &)
{
  return std::nullopt;
}

template<typename PointType, typename Points>
void buildScanMap(lio_ivox::IVox<PointType> & map,
                  Points & points,
                  std::uint64_t lifecycle_scan_seq,
                  std::size_t prior_prefix = 0)
{
  map.Build(points, seedLabel(lifecycle_scan_seq, prior_prefix));
}
template<typename PointType, typename Points>
void buildScanMap(KD_TREE<PointType> & map, Points & points, std::uint64_t, std::size_t = 0)
{
  map.Build(points);
}
template<typename PointType, typename Points>
void addScanSeed(lio_ivox::IVox<PointType> & map, Points & points, std::uint64_t lifecycle_scan_seq)
{
  map.Add_Points(points, true, seedLabel(lifecycle_scan_seq));
}
template<typename PointType, typename Points>
void addScanSeed(KD_TREE<PointType> & map, Points & points, std::uint64_t)
{
  map.Add_Points(points, true);
}
template<typename PointType>
void stageMapProvenance(lio_ivox::IVox<PointType> & map, MapInsertProvenance && provenance)
{
  map.stagePointProvenance(std::move(provenance));
}
template<typename PointType>
void stageMapProvenance(KD_TREE<PointType> &, MapInsertProvenance &&)
{}
template<typename PointType, typename Points>
int addStagedPoints(lio_ivox::IVox<PointType> & map, Points & downsampled, Points & direct)
{
  return map.addStagedPoints(downsampled, direct);
}
template<typename PointType, typename Points>
int addStagedPoints(KD_TREE<PointType> & map, Points & downsampled, Points & direct)
{
  const int added = map.Add_Points(downsampled, true);
  map.Add_Points(direct, false);
  return added;
}
template<typename PointType, typename Points>
MapIdentityStatus nearestSearchIdentified(lio_ivox::IVox<PointType> & map,
                                          const PointType & point,
                                          int k,
                                          Points & nearest,
                                          std::vector<float> & sqdist,
                                          std::vector<MapPointIdentity> & identities)
{
  return map.Nearest_Search_Identified(point, k, nearest, sqdist, identities);
}
template<typename PointType, typename Points>
MapIdentityStatus nearestSearchIdentified(KD_TREE<PointType> & map,
                                          const PointType & point,
                                          int k,
                                          Points & nearest,
                                          std::vector<float> & sqdist,
                                          std::vector<MapPointIdentity> & identities)
{
  identities.clear();
  map.Nearest_Search(point, k, nearest, sqdist);
  return MapIdentityStatus::kUnsupportedBackend;
}

}  // namespace fast_lio

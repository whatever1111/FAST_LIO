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

}  // namespace fast_lio

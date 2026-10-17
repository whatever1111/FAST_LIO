// map_backend_reference.hpp — whether the local map backend holds points.
//
// The front end's first local reference is its normal initial map build only when that build returned with the
// map backend holding points. Calling the build proves nothing: iVox clears itself and then skips every point it
// cannot place (a non-finite coordinate, or one whose voxel index leaves the int range), so a build of points it
// rejects all leaves it as empty as it was. The test is the one the node already uses to decide that the map still
// needs its initial build: iVox empty(), ikd-Tree a null root.
//
// No ROS. Over the two backends' own headers only.

#pragma once

#include "ikd-Tree/ikd_Tree.h"
#include "ivox/ivox.hpp"

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

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

// Same types as common_lib.h, without importing its storage definitions or
// non-inline functions into an executable that links the real node component.
using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

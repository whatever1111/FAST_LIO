// frontend_point_types.hpp — the front end's point, cloud and vector aliases and its correspondence constants.
//
// common_lib.h includes this and adds its file-scope objects and using-directives; a second translation unit that
// needs only these names includes this header instead. The correspondence constants have this one definition, so
// the measurement model and the recovery view match read the same neighbour count and point covariance.
#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
using V3F = Eigen::Vector3f;
using M3F = Eigen::Matrix3f;

#define NUM_MATCH_POINTS (5)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define LASER_POINT_COV (0.001)

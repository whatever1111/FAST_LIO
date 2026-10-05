cmake_minimum_required(VERSION 3.5)
project(shm_msgs)

# Interfaces-only build: only the messages needed to decode /pre/all_point
# (shm_msgs/PointCloud8mAndPose) for the giant 260612-allpoint pipeline.
# The package name MUST stay "shm_msgs" so the type matches the recorded bag.
# Upstream shared-memory transport demos (lib/intra/cuda/src/tool, iceoryx,
# opencv, pcl) are intentionally not built here.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/String.msg"
  "msg/Header.msg"
  "msg/PointField.msg"
  "msg/PointCloud8mAndPose.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs
)

ament_package()

cmake_minimum_required(VERSION 3.5)
project(fast_lio_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrontendObservation.msg"
  "msg/ReferenceHealth.msg"
  "msg/LidarWindow.msg"
  "msg/LidarWindowAck.msg"
  "srv/BindLidarCustody.srv"
  "srv/LidarCustodyStatus.srv"
  DEPENDENCIES geometry_msgs sensor_msgs)
add_library(lidar_custody_ros INTERFACE)
target_include_directories(lidar_custody_ros INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
install(TARGETS lidar_custody_ros EXPORT export_lidar_custody_ros)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(export_lidar_custody_ros)
ament_export_dependencies(rosidl_default_runtime)
ament_package()

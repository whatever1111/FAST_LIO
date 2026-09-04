cmake_minimum_required(VERSION 3.8)
project(fast_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

ADD_COMPILE_OPTIONS(-std=c++14)
ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

include(ProcessorCount)
ProcessorCount(N)
message("Processer number:  ${N}")

# MP_PROC_NUM is overridable from cache (e.g. -DMP_PROC_NUM_VAL=6 for Jetson Orin)
if(NOT DEFINED MP_PROC_NUM_VAL)
  if(N GREATER 8)
    set(MP_PROC_NUM_VAL 8)
  elseif(N GREATER 4)
    set(MP_PROC_NUM_VAL 4)
  elseif(N GREATER 3)
    set(MP_PROC_NUM_VAL 2)
  else()
    set(MP_PROC_NUM_VAL 1)
  endif()
endif()

if(MP_PROC_NUM_VAL GREATER 1)
  add_definitions(-DMP_EN)
endif()
add_definitions(-DMP_PROC_NUM=${MP_PROC_NUM_VAL})
message("core for MP: ${MP_PROC_NUM_VAL}")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs QUIET)

# ROS dependencies
find_package(ament_cmake REQUIRED)
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
endif()
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(fixposition_driver_msgs REQUIRED)
find_package(shm_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_conversions
  tf2_ros
  livox_ros_driver2
  fixposition_driver_msgs
  shm_msgs
)

# Thirdparty libraries
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

add_executable(fastlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} Eigen3::Eigen)
if(PythonLibs_FOUND)
  target_link_libraries(fastlio_mapping ${PYTHON_LIBRARIES})
  target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
endif()

# Option B: iVox voxel-grid map backend (drop-in for ikd-Tree). Default ON for the GPS
# deployment (iekf -31/34%, drops halved, 2D-better, no divergence). Build -DUSE_IVOX=OFF
# to fall back to ikd-Tree (e.g. GPS-denied, where iVox's raw-Z is weaker — GPS masks it).
option(USE_IVOX "Use the iVox voxel-grid map backend instead of ikd-Tree" ON)
if(USE_IVOX)
  target_compile_definitions(fastlio_mapping PRIVATE USE_IVOX)
  message(STATUS "fast_lio: map backend = iVox (default; -DUSE_IVOX=OFF for ikd-Tree)")
else()
  message(STATUS "fast_lio: map backend = ikd-Tree (USE_IVOX=OFF)")
endif()

list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")

if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # Custommsg to support foxy & galactic
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})

# ─────────── Composable component (intra-process zero-copy) ───────────────────
# Built from the SAME sources as the standalone executable but with
# FASTLIO_AS_COMPONENT, which excludes main() and registers LaserMappingNode as an
# rclcpp component. This lets FAST-LIO load into the shared input_adapters MT
# container, where the lidar_adapter hands it /lio/points via an intra-process
# unique_ptr move instead of an ~8 MB cross-process DDS copy. The node's file-scope
# globals are safe because exactly ONE LaserMappingNode is ever loaded.
add_library(fastlio_mapping_component SHARED
  src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_compile_definitions(fastlio_mapping_component PRIVATE FASTLIO_AS_COMPONENT)
target_include_directories(fastlio_mapping_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping_component ${PCL_LIBRARIES} Eigen3::Eigen)
if(PythonLibs_FOUND)
  target_link_libraries(fastlio_mapping_component ${PYTHON_LIBRARIES})
  target_include_directories(fastlio_mapping_component PRIVATE ${PYTHON_INCLUDE_DIRS})
endif()
if(USE_IVOX)
  target_compile_definitions(fastlio_mapping_component PRIVATE USE_IVOX)
endif()
if(cpp_typesupport_target)  # set by rosidl_get_typesupport_target above (non-EOL distros)
  target_link_libraries(fastlio_mapping_component ${cpp_typesupport_target})
else()
  # EOL distros (foxy/galactic): same own-msgs wiring as the executable —
  # rosidl_target_interfaces also carries the generated-header include dirs,
  # without which common_lib.h can't find fast_lio/msg/pose6_d.hpp.
  rosidl_target_interfaces(fastlio_mapping_component
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
endif()
ament_target_dependencies(fastlio_mapping_component ${dependencies})
rclcpp_components_register_nodes(fastlio_mapping_component "LaserMappingNode")

if(BUILD_TESTING)
  ament_add_pytest_test(test_analyze_kalman_channels
    test/test_analyze_kalman_channels.py
  )
  # Pure header, no ROS: the static-hold verdict that decides whether the iEKF
  # may be pinned during a blind stretch.
  ament_add_gtest(test_static_evidence test/test_static_evidence.cpp)
  if(TARGET test_static_evidence)
    target_include_directories(test_static_evidence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # Pure header, no ROS: the gate that decides when the frozen map may be
  # trusted (and grown) again after a blind stretch.
  ament_add_gtest(test_reanchor_gate test/test_reanchor_gate.cpp)
  if(TARGET test_reanchor_gate)
    target_include_directories(test_reanchor_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # Pure header, no ROS: the only check that sees drift the scan keeps hiding.
  ament_add_gtest(test_runaway_watchdog test/test_runaway_watchdog.cpp)
  if(TARGET test_runaway_watchdog)
    target_include_directories(test_runaway_watchdog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  ament_add_gtest(test_adaptive_downsample test/test_adaptive_downsample.cpp)
  if(TARGET test_adaptive_downsample)
    target_include_directories(test_adaptive_downsample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # Pure header, no PCL: one representative per occupied voxel, in one pass.
  ament_add_gtest(test_voxel_downsample test/test_voxel_downsample.cpp)
  if(TARGET test_voxel_downsample)
    target_include_directories(test_voxel_downsample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS fastlio_mapping_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
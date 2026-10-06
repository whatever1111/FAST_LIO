cmake_minimum_required(VERSION 3.16)
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

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")

include(ProcessorCount)
ProcessorCount(N)
message(STATUS "Processor number: ${N}")

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
message(STATUS "core for MP: ${MP_PROC_NUM_VAL}")

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
find_package(fast_lio_interfaces REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
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
  fast_lio_interfaces
)

# Thirdparty libraries
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")
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
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} Eigen3::Eigen OpenSSL::Crypto)
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
target_link_libraries(fastlio_mapping_component ${PCL_LIBRARIES} Eigen3::Eigen OpenSSL::Crypto)
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
  # Link the real production model: do not compile a copied h-model into the test.
  ament_add_gtest(test_h_model_scan_bounds test/test_h_model_scan_bounds.cpp TIMEOUT 120)
  target_include_directories(test_h_model_scan_bounds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS})
  target_link_libraries(test_h_model_scan_bounds fastlio_mapping_component ${PCL_LIBRARIES} Eigen3::Eigen)
  ament_target_dependencies(test_h_model_scan_bounds ${dependencies})
  add_dependencies(test_h_model_scan_bounds fastlio_mapping)
  if(cpp_typesupport_target)
    target_link_libraries(test_h_model_scan_bounds ${cpp_typesupport_target})
  else()
    rosidl_target_interfaces(test_h_model_scan_bounds ${PROJECT_NAME} "rosidl_typesupport_cpp")
  endif()
  if(USE_IVOX)
    # Non-installed real node TU with test-only worker barriers. Serialize this
    # third heavy compilation after BOTH normal production targets finish.
    add_library(fastlio_mapping_test_component SHARED
      src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
    target_compile_definitions(fastlio_mapping_test_component PRIVATE
      FASTLIO_AS_COMPONENT FASTLIO_TEST_HOOKS USE_IVOX)
    target_include_directories(fastlio_mapping_test_component PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/test ${PCL_INCLUDE_DIRS})
    target_link_libraries(fastlio_mapping_test_component ${PCL_LIBRARIES} Eigen3::Eigen OpenSSL::Crypto)
    ament_target_dependencies(fastlio_mapping_test_component ${dependencies})
    if(PythonLibs_FOUND)
      target_link_libraries(fastlio_mapping_test_component ${PYTHON_LIBRARIES})
      target_include_directories(fastlio_mapping_test_component PRIVATE ${PYTHON_INCLUDE_DIRS})
    endif()
    if(cpp_typesupport_target)
      target_link_libraries(fastlio_mapping_test_component ${cpp_typesupport_target})
    else()
      rosidl_target_interfaces(fastlio_mapping_test_component ${PROJECT_NAME} "rosidl_typesupport_cpp")
    endif()
    add_dependencies(fastlio_mapping_test_component fastlio_mapping fastlio_mapping_component)
    ament_add_gtest(test_async_map_reset test/test_async_map_reset.cpp TIMEOUT 30)
    target_include_directories(test_async_map_reset PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/test ${PCL_INCLUDE_DIRS})
    target_link_libraries(test_async_map_reset fastlio_mapping_test_component ${PCL_LIBRARIES} Eigen3::Eigen)
    ament_target_dependencies(test_async_map_reset ${dependencies})
    if(cpp_typesupport_target)
      target_link_libraries(test_async_map_reset ${cpp_typesupport_target})
    else()
      rosidl_target_interfaces(test_async_map_reset ${PROJECT_NAME} "rosidl_typesupport_cpp")
    endif()
    find_package(class_loader REQUIRED)
    # Each executable creates one component; production globals outlive a node.
    foreach(scan_test test_ros_scan_bounds test_ros_imu_fault)
      string(REGEX REPLACE "^test_" "" scan_receipt "${scan_test}")
      ament_add_gtest(${scan_test} test/test_ros_scan_bounds.cpp TIMEOUT 240)
      target_include_directories(${scan_test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/test ${PCL_INCLUDE_DIRS})
      file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_results/${PROJECT_NAME})
      target_compile_definitions(${scan_test} PRIVATE
        FASTLIO_TEST_COMPONENT_PATH="$<TARGET_FILE:fastlio_mapping_test_component>"
        FASTLIO_SCAN_RECEIPT_PATH="${CMAKE_CURRENT_BINARY_DIR}/test_results/${PROJECT_NAME}/${scan_receipt}_receipt.json")
      target_link_libraries(${scan_test} fastlio_mapping_test_component ${PCL_LIBRARIES} Eigen3::Eigen)
      ament_target_dependencies(${scan_test} ${dependencies} class_loader)
      if(cpp_typesupport_target)
        target_link_libraries(${scan_test} ${cpp_typesupport_target})
      else()
        rosidl_target_interfaces(${scan_test} ${PROJECT_NAME} "rosidl_typesupport_cpp")
      endif()
    endforeach()
    target_compile_definitions(test_ros_imu_fault PRIVATE FASTLIO_INPUT_FAULT_TEST)
    # One actual component activation per process: production globals are not reset between nodes.
    foreach(capacity_case negative zero small)
      # Direct linking and ClassLoader must resolve the same physical build DSO.
      # APPEND_LIBRARY_DIRS would leave the installed copy ahead of this directory.
      ament_add_gtest(test_ivox_node_capacity_${capacity_case} test/test_ivox_node_capacity.cpp TIMEOUT 30
        ENV "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:fastlio_mapping_component>:$ENV{LD_LIBRARY_PATH}")
      target_include_directories(test_ivox_node_capacity_${capacity_case} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS})
      target_compile_definitions(test_ivox_node_capacity_${capacity_case} PRIVATE
        FASTLIO_CAPACITY_${capacity_case} FASTLIO_COMPONENT_PATH="$<TARGET_FILE:fastlio_mapping_component>")
      target_link_libraries(test_ivox_node_capacity_${capacity_case}
        fastlio_mapping_component ${PCL_LIBRARIES} Eigen3::Eigen)
      ament_target_dependencies(test_ivox_node_capacity_${capacity_case} ${dependencies} class_loader)
      add_dependencies(test_ivox_node_capacity_${capacity_case} fastlio_mapping)
      if(cpp_typesupport_target)
        target_link_libraries(test_ivox_node_capacity_${capacity_case} ${cpp_typesupport_target})
      else()
        rosidl_target_interfaces(test_ivox_node_capacity_${capacity_case} ${PROJECT_NAME} "rosidl_typesupport_cpp")
      endif()
    endforeach()
  endif()
  ament_add_gtest(test_cached_plane test/test_cached_plane.cpp)
  target_include_directories(test_cached_plane PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_cached_plane Eigen3::Eigen)
  ament_add_gtest(test_correspondence_compaction test/test_correspondence_compaction.cpp)
  target_include_directories(test_correspondence_compaction PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                             ${PCL_INCLUDE_DIRS})
  target_link_libraries(test_correspondence_compaction ${PCL_LIBRARIES})
  ament_add_pytest_test(test_frontend_observation_node test/test_frontend_observation_node.py TIMEOUT 90)
  ament_add_pytest_test(test_frontend_observation_scan_node test/test_frontend_observation_scan_node.py
    TIMEOUT 120
    ENV FASTLIO_SCAN_CONTRACT_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/test_results/fast_lio/scan_contract)
  ament_add_gtest(test_frontend_observation test/test_frontend_observation.cpp)
  target_include_directories(test_frontend_observation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_frontend_observation OpenSSL::Crypto)
  ament_add_gtest(test_imu_coverage_policy test/test_imu_coverage_policy.cpp)
  target_include_directories(test_imu_coverage_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_imu_process_outcome test/test_imu_process_outcome.cpp)
  target_include_directories(test_imu_process_outcome PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_imu_consumption_policy test/test_imu_consumption_policy.cpp)
  target_include_directories(test_imu_consumption_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_scan_time_policy test/test_scan_time_policy.cpp)
  target_include_directories(test_scan_time_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ament_add_gtest(test_population_moments test/test_population_moments.cpp)
  target_include_directories(test_population_moments PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  ament_add_gtest(test_plane_fit test/test_plane_fit.cpp)
  ament_add_gtest(test_simple_update_tangent test/test_simple_update_tangent.cpp)
  ament_add_gtest(test_imu_process_validity test/test_imu_process_validity.cpp)
  ament_add_gtest(test_imu_0906_timing test/test_imu_0906_timing.cpp TIMEOUT 240)
  target_compile_definitions(test_imu_0906_timing PRIVATE
    FASTLIO_0906_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test/m20_0906_scan_timing.txt")
  ament_add_gtest(test_preprocess_timing test/test_preprocess_timing.cpp src/preprocess.cpp)
  # the consider update on the real estimator (use-ikfom.hpp pulls in the same dependencies)
  ament_add_gtest(test_rp_consider_filter test/test_rp_consider_filter.cpp)
  foreach(timing_test test_plane_fit test_simple_update_tangent test_imu_process_validity test_imu_0906_timing test_preprocess_timing test_rp_consider_filter)
    target_include_directories(${timing_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${CMAKE_CURRENT_SOURCE_DIR}/src ${PCL_INCLUDE_DIRS})
    target_link_libraries(${timing_test} ${PCL_LIBRARIES} Eigen3::Eigen)
    ament_target_dependencies(${timing_test} ${dependencies})
    if(cpp_typesupport_target)
      target_link_libraries(${timing_test} ${cpp_typesupport_target})
    else()
      rosidl_target_interfaces(${timing_test} ${PROJECT_NAME} "rosidl_typesupport_cpp")
    endif()
  endforeach()
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
  # Pure header, Eigen only: which directions of the scan's position information are
  # worth keeping, and by how much to hold back the rest.
  ament_add_gtest(test_degeneracy_policy test/test_degeneracy_policy.cpp)
  if(TARGET test_degeneracy_policy)
    target_include_directories(test_degeneracy_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()
  # Pure header, no PCL: one representative per occupied voxel, in one pass.
  ament_add_gtest(test_voxel_downsample test/test_voxel_downsample.cpp)
  if(TARGET test_voxel_downsample)
    target_include_directories(test_voxel_downsample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # The map backend's prior-map tier: what the live run may take away from it (nothing) and
  # what the guard's recovery drops (only what this run learned).
  ament_add_gtest(test_delayed_map_insertion test/test_delayed_map_insertion.cpp)
  if(TARGET test_delayed_map_insertion)
    target_include_directories(test_delayed_map_insertion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                               ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_delayed_map_insertion ${PCL_LIBRARIES})
  endif()
  ament_add_gtest(test_ivox_layering test/test_ivox_layering.cpp)
  if(TARGET test_ivox_layering)
    target_include_directories(test_ivox_layering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS}
                                                          ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_ivox_layering ${PCL_LIBRARIES})
  endif()
  # The map backend's per-voxel occupancy: the dedup-grid relation the per-voxel reserve
  # is sized from, and the two things that are allowed to exceed it.
  ament_add_gtest(test_ivox_voxel_occupancy test/test_ivox_voxel_occupancy.cpp)
  if(TARGET test_ivox_voxel_occupancy)
    target_include_directories(test_ivox_voxel_occupancy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                                 ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_ivox_voxel_occupancy ${PCL_LIBRARIES})
  endif()
  # Pure header, Eigen only: what a prior map and its initial pose have to prove before
  # either of them reaches the map backend.
  ament_add_gtest(test_prior_map_load test/test_prior_map_load.cpp)
  if(TARGET test_prior_map_load)
    target_include_directories(test_prior_map_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()
  # Pure header, no ROS: the decision table laserMapping.cpp's two scan branches share
  # for "the platform is parked — what may that do to the state?".
  ament_add_gtest(test_zupt_policy test/test_zupt_policy.cpp)
  if(TARGET test_zupt_policy)
    target_include_directories(test_zupt_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  ament_add_gtest(test_parked_hold test/test_parked_hold.cpp)
  if(TARGET test_parked_hold)
    target_include_directories(test_parked_hold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()
  # Pure header, no dependencies: the ring the timing log's ever-growing counters index,
  # written by laserMapping.cpp's callbacks and read back by its shutdown dump.
  ament_add_gtest(test_timing_log_ring test/test_timing_log_ring.cpp)
  if(TARGET test_timing_log_ring)
    target_include_directories(test_timing_log_ring PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # Pure header, no dependencies: how much of the IMU queue a LiDAR outage may leave behind,
  # trimmed by laserMapping.cpp's IMU callback after every push.
  ament_add_gtest(test_imu_backlog_policy test/test_imu_backlog_policy.cpp)
  if(TARGET test_imu_backlog_policy)
    target_include_directories(test_imu_backlog_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
  # Pure header, Eigen only: the window-mean linear acceleration the levelling prior
  # subtracts from the specific-force mean, and when it must be withheld
  # (docs/PGO_LOOP_TUNING.md section 7.3).
  ament_add_gtest(test_gravity_align_kinematics test/test_gravity_align_kinematics.cpp)
  if(TARGET test_gravity_align_kinematics)
    target_include_directories(test_gravity_align_kinematics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()
  # Pure header, Eigen only: whether a scan's lidar update may rotate roll/pitch, and the
  # Jacobian projection that enforces it (docs/PGO_LOOP_TUNING.md section 7.5).
  ament_add_gtest(test_attitude_hold test/test_attitude_hold.cpp)
  if(TARGET test_attitude_hold)
    target_include_directories(test_attitude_hold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  # Pure header, Eigen only: the covariance side of a state assignment made outside an update (the
  # velocity block after a release, ZUPT or hole clamp) — docs/PGO_LOOP_TUNING.md section 7.7.
  ament_add_gtest(test_covariance_reset test/test_covariance_reset.cpp)
  if(TARGET test_covariance_reset)
    target_include_directories(test_covariance_reset PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  # Pure header, Eigen only: the innovation test a point-to-plane correspondence has to pass
  # (docs/PGO_LOOP_TUNING.md section 7.7).
  ament_add_gtest(test_correspondence_gate test/test_correspondence_gate.cpp)
  if(TARGET test_correspondence_gate)
    target_include_directories(test_correspondence_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  # Pure header, Eigen only: carrying the state across an IMU dropout — gyro average, held velocity and the
  # covariance that says so (the m20 0826 data hole, docs/PGO_LOOP_TUNING.md section 7.8).
  ament_add_gtest(test_imu_gap_prior test/test_imu_gap_prior.cpp)
  if(TARGET test_imu_gap_prior)
    target_include_directories(test_imu_gap_prior PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  ament_add_gtest(test_engulf_release test/test_engulf_release.cpp)
  if(TARGET test_engulf_release)
    target_include_directories(test_engulf_release PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  # Pure header, Eigen only: the lidar update as a consider (Schmidt) update for roll/pitch — policy, projector and
  # the Joseph-form covariance checked against the explicit form (docs/PGO_LOOP_TUNING.md section 7.8).
  ament_add_gtest(test_rp_consider test/test_rp_consider.cpp)
  if(TARGET test_rp_consider)
    target_include_directories(test_rp_consider PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
  endif()

  ament_add_gtest(test_odom_covariance_contract test/test_odom_covariance_contract.cpp)
  if(TARGET test_odom_covariance_contract)
    target_include_directories(test_odom_covariance_contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_odom_covariance_contract Eigen3::Eigen)
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

#!/usr/bin/env bash
# Stage the message packages fast_lio compiles against into a colcon src/ dir, as
# interface-only packages generated from the .msg files of the commits LIO-SLAM pins.
#
# fast_lio only includes these packages' generated message headers. The packages
# themselves also build a Livox driver against Livox-SDK2 and a Fixposition library
# against fpsdk, which this CI does not need, so only their msg/ trees are fetched
# and wrapped in a plain rosidl package of the same name. shm_msgs lives in the
# private LIO-SLAM repo, so an interfaces-only copy of it is kept next to this
# script (copied from LIO-SLAM drivers/ros2_shm_msgs).
#
# Keep the refs equal to LIO-SLAM's third_party/ gitlinks; the superproject's
# "FAST_LIO Build & Test" job builds against the real packages and stays the gate
# for a gitlink bump.
#
# Usage: stage_interface_deps.sh <colcon src dir>
set -euo pipefail

LIVOX_ROS_DRIVER2_REF=6b9356cadf77084619ba406e6a0eb41163b08039
FIXPOSITION_DRIVER_REF=3f75c6280bd6119cd5e933099b2bad80ed4983ab

src_dir=${1:?usage: $0 <colcon src dir>}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${src_dir}"

# fetch_msgs <owner/repo> <ref> <msg dir inside the repo> <destination>
fetch_msgs() {
  local repo=$1 ref=$2 msg_dir=$3 dest=$4 tmp
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 "https://codeload.github.com/${repo}/tar.gz/${ref}" \
    | tar -xz -C "${tmp}" --wildcards "*/${msg_dir}/*.msg"
  mkdir -p "${dest}/msg"
  cp "${tmp}"/*/"${msg_dir}"/*.msg "${dest}/msg/"
  rm -rf "${tmp}"
  ls "${dest}"/msg/*.msg > /dev/null
}

# write_interface_pkg <dir> <name> <message dependency>...
write_interface_pkg() {
  local dir=$1 name=$2 dep
  shift 2
  {
    echo '<?xml version="1.0"?>'
    echo '<package format="3">'
    echo "  <name>${name}</name>"
    echo '  <version>0.0.0</version>'
    echo "  <description>CI-only interface build of ${name}.</description>"
    echo '  <maintainer email="noreply@example.com">ci</maintainer>'
    echo '  <license>see upstream</license>'
    echo '  <buildtool_depend>ament_cmake</buildtool_depend>'
    echo '  <buildtool_depend>rosidl_default_generators</buildtool_depend>'
    for dep in "$@"; do echo "  <depend>${dep}</depend>"; done
    echo '  <exec_depend>rosidl_default_runtime</exec_depend>'
    echo '  <member_of_group>rosidl_interface_packages</member_of_group>'
    echo '  <export><build_type>ament_cmake</build_type></export>'
    echo '</package>'
  } > "${dir}/package.xml"
  {
    echo 'cmake_minimum_required(VERSION 3.8)'
    echo "project(${name})"
    echo 'find_package(ament_cmake REQUIRED)'
    echo 'find_package(rosidl_default_generators REQUIRED)'
    for dep in "$@"; do echo "find_package(${dep} REQUIRED)"; done
    # shellcheck disable=SC2016 # CMake variables, written literally
    echo 'file(GLOB MSG_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg)'
    echo "rosidl_generate_interfaces(\${PROJECT_NAME} \${MSG_FILES} DEPENDENCIES $*)"
    echo 'ament_export_dependencies(rosidl_default_runtime)'
    echo 'ament_package()'
  } > "${dir}/CMakeLists.txt"
}

fetch_msgs Livox-SDK/livox_ros_driver2 "${LIVOX_ROS_DRIVER2_REF}" msg \
  "${src_dir}/livox_ros_driver2"
write_interface_pkg "${src_dir}/livox_ros_driver2" livox_ros_driver2 std_msgs

fetch_msgs fixposition/fixposition_driver "${FIXPOSITION_DRIVER_REF}" \
  fixposition_driver_msgs/msg "${src_dir}/fixposition_driver_msgs"
write_interface_pkg "${src_dir}/fixposition_driver_msgs" fixposition_driver_msgs \
  builtin_interfaces std_msgs geometry_msgs nav_msgs sensor_msgs

cp -r "${here}/shm_msgs" "${src_dir}/shm_msgs"

echo "staged into ${src_dir}: livox_ros_driver2 fixposition_driver_msgs shm_msgs"

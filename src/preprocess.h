// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <shm_msgs/msg/point_cloud8m_and_pose.hpp>
#include <cstdlib>

// Runtime OMP thread budget for the preprocess loops. The compile-time
// MP_PROC_NUM is derived from the BUILD host's core count (8 on a 16-core dev
// box, 4 on the CI runner), so `num_threads(MP_PROC_NUM)` made the binary's
// behaviour depend on where it was compiled: an in-image 8-thread build
// measured +0.1 m 2D mean on giant 0509 (0.62-0.65 vs 0.49-0.53 for the CI
// build and for the same source built with MP_PROC_NUM=4) — oversubscription
// against the eval's OMP_NUM_THREADS=4 budget. Honour OMP_NUM_THREADS when set,
// exactly like the laserMapping match loop does; fall back to MP_PROC_NUM.
inline int fastlio_preprocess_omp_threads()
{
  static const int n = []() {
    const char * e = std::getenv("OMP_NUM_THREADS");
    const int v = (e != nullptr) ? std::atoi(e) : 0;
#ifdef MP_PROC_NUM
    return v > 0 ? v : static_cast<int>(MP_PROC_NUM);
#else
    return v > 0 ? v : 1;
#endif
  }();
  return n;
}

using namespace std;

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
  AVIA = 1,
  VELO16,
  OUST64,
  MID360,
  HESAI32,
  ALLPOINT   // shm_msgs/PointCloud8mAndPose on /pre/all_point (RoboSense layout, fixed 8MB buffer)
};  // {1, 2, 3, 4, 5, 6}
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};

namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(float, time, time)(std::uint16_t, ring,
                                                                                                        ring))

namespace ouster_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

namespace livox_ros
{
typedef struct {
  float x;            /**< X axis, Unit:m */
  float y;            /**< Y axis, Unit:m */
  float z;            /**< Z axis, Unit:m */
  float intensity;    /**< Intensity   */
  uint8_t tag;        /**< Livox point tag   */
  uint8_t line;       /**< Laser line id     */
  double timestamp;   /**< Timestamp of point */
} LivoxPointXyzrtl;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_ros::LivoxPointXyzrtl,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint8_t, tag, tag)
    (std::uint8_t, line, line)
    (double, timestamp, timestamp)
)

namespace hesai_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t ring;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace hesai_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(std::uint16_t, ring,
                                                                                      ring)(double, timestamp,
                                                                                                  timestamp))

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_surf;
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool given_offset_time;
  // ros::Publisher pub_full, pub_surf, pub_corn;

private:
  void avia_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg);
  void oust64_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void mid360_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void hesai_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
  void allpoint_handler(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr &msg);
  void default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);
};

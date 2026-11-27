#include "preprocess.h"

#include <pcl/common/common.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#define RETURN0 0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess() : lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  N_SCANS = 6;
  SCAN_RATE = 10;
  given_offset_time = false;
}

Preprocess::~Preprocess()
{
}

void Preprocess::set(int lid_type, double bld, int pfilt_num)
{
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

void Preprocess::process(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  avia_handler(msg);
  *pcl_out = pl_surf;
}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  switch (lidar_type)
  {
    case OUST64:
      oust64_handler(msg);
      break;

    case VELO16:
      velodyne_handler(msg);
      break;

    case MID360:
      mid360_handler(msg);
      break;

    case HESAI32:
      hesai_handler(msg);
      break;

    default:
      default_handler(msg);
      break;
  }
  *pcl_out = pl_surf;
}

void Preprocess::process(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr &msg, PointCloudXYZI::Ptr& pcl_out)
{
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  allpoint_handler(msg);
  *pcl_out = pl_surf;
}

// Decode a shm_msgs/PointCloud8mAndPose (/pre/all_point) in place. The buffer is
// a fixed 8MB blob carrying a RoboSense-layout cloud (x,y,z,intensity float32,
// ring uint16, timestamp float64 absolute, point_step=32) — only the first
// width*point_step bytes are valid. Mirrors hesai_handler: resolve field offsets
// once, then read msg->data directly with point_filter_num stride + blind gate,
// computing per-point relative offset time (ms) from the absolute timestamp.
void Preprocess::allpoint_handler(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();

  const int plsize = static_cast<int>(msg->width * msg->height);
  if (plsize == 0)
    return;

  int off_x = -1, off_y = -1, off_z = -1, off_i = -1, off_ts = -1;
  for (const auto &f : msg->fields) {
    const std::string fname(reinterpret_cast<const char *>(f.name.data.data()),
                            std::min<size_t>(f.name.size, f.name.data.size()));
    if (fname == "x") off_x = f.offset;
    else if (fname == "y") off_y = f.offset;
    else if (fname == "z") off_z = f.offset;
    else if (fname == "intensity") off_i = f.offset;
    else if (fname == "timestamp") off_ts = f.offset;
  }
  if (off_x < 0 || off_y < 0 || off_z < 0 || off_ts < 0)
    return;

  const uint8_t * base = msg->data.data();
  const uint32_t step = msg->point_step;
  if (step == 0)
    return;
  const double scan_start_time = rclcpp::Time(msg->header.stamp).seconds();

  double last_ts;
  std::memcpy(&last_ts, base + (plsize - 1) * step + off_ts, sizeof(double));
  const double ts_origin = (last_ts > 1e6) ? scan_start_time : 0.0;

  const float blind_sq = static_cast<float>(blind * blind);
  const int stride = std::max(1, point_filter_num);
  const int n_candidates = (plsize + stride - 1) / stride;

  pl_surf.resize(n_candidates);
  constexpr float kDropMarker = std::numeric_limits<float>::quiet_NaN();

#ifdef MP_EN
  #pragma omp parallel for num_threads(fastlio_preprocess_omp_threads()) schedule(static)
#endif
  for (int j = 0; j < n_candidates; ++j) {
    const int i = j * stride;
    if (i >= plsize) {
      pl_surf.points[j].x = kDropMarker;
      continue;
    }
    const uint8_t * p = base + i * step;

    float x, y, z, intensity = 0.0f;
    double ts;
    std::memcpy(&x, p + off_x, sizeof(float));
    std::memcpy(&y, p + off_y, sizeof(float));
    std::memcpy(&z, p + off_z, sizeof(float));
    if (off_i >= 0) std::memcpy(&intensity, p + off_i, sizeof(float));
    std::memcpy(&ts, p + off_ts, sizeof(double));

    if (x * x + y * y + z * z <= blind_sq) {
      pl_surf.points[j].x = kDropMarker;
      continue;
    }

    PointType & pt = pl_surf.points[j];
    pt.x = x;
    pt.y = y;
    pt.z = z;
    pt.intensity = intensity;
    pt.normal_x = 0.0f;
    pt.normal_y = 0.0f;
    pt.normal_z = 0.0f;
    pt.curvature = static_cast<float>((ts - ts_origin) * time_unit_scale);
  }

  size_t out = 0;
  for (size_t k = 0; k < pl_surf.points.size(); ++k) {
    const auto & pt = pl_surf.points[k];
    if (!std::isnan(pt.x)) {
      if (out != k) pl_surf.points[out] = pt;
      ++out;
    }
  }
  pl_surf.points.resize(out);
  pl_surf.width = out;
  pl_surf.height = 1;
}

void Preprocess::avia_handler(const livox_ros_driver2::msg::CustomMsg::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  // cout<<"plsie: "<<plsize<<endl;

  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  uint valid_num = 0;

  for (uint i = 1; i < plsize; i++) {
    if ((msg->points[i].line < N_SCANS) &&
        ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
      valid_num++;
      if (valid_num % point_filter_num == 0) {
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time /
                               float(1000000);  // use curvature as time of each laser points, curvature unit: ms

        if (((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
             (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7)) &&
            (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z >
             (blind * blind))) {
          pl_surf.push_back(pl_full[i]);
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_surf.reserve(plsize);
  double time_stamp = rclcpp::Time(msg->header.stamp).seconds();
  // cout << "===================================" << endl;
  // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (i % point_filter_num != 0)
      continue;

    double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                   pl_orig.points[i].z * pl_orig.points[i].z;

    if (range < (blind * blind))
      continue;

    Eigen::Vector3d pt_vec;
    PointType added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.curvature = pl_orig.points[i].t * time_unit_scale;  // curvature unit: ms

    pl_surf.points.push_back(added_pt);
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::velodyne_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
  /*****************************************************************/

  // Field presence is the contract; the last point may legally have time zero
  // and OEM groups need not be in time order. A malformed present field must
  // not silently switch to synthesized yaw timestamps.
  const auto time_field =
    std::find_if(msg->fields.begin(), msg->fields.end(), [](const auto & field) { return field.name == "time"; });
  given_offset_time = time_field != msg->fields.end();
  if (given_offset_time) {
    if (time_field->datatype != sensor_msgs::msg::PointField::FLOAT32 || time_field->count != 1 ||
        time_field->offset > msg->point_step || msg->point_step - time_field->offset < sizeof(float))
      return;
    for (const auto & point : pl_orig.points) {
      if (!std::isfinite(point.time) || point.time < 0.0f)
        return;
    }
  }

  for (int i = 0; i < plsize; i++) {
    PointType added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature =
      pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

    if (!given_offset_time) {
      int layer = pl_orig.points[i].ring;
      if (layer >= N_SCANS)
        continue;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

      if (is_first[layer]) {
        yaw_fp[layer] = yaw_angle;
        is_first[layer] = false;
        added_pt.curvature = 0.0;
        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
        continue;
      }

      // compute offset time
      if (yaw_angle <= yaw_fp[layer]) {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
      } else {
        added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
      }

      if (added_pt.curvature < time_last[layer])
        added_pt.curvature += 360.0 / omega_l;

      yaw_last[layer] = yaw_angle;
      time_last[layer] = added_pt.curvature;
    }

    if (i % point_filter_num == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind)) {
        pl_surf.points.push_back(added_pt);
      }
    }
  }
}

void Preprocess::mid360_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<livox_ros::LivoxPointXyzrtl> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  /*** These variables only works when no point timestamps given ***/
  double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);    // yaw of first scan point
  std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
  std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
  /*****************************************************************/

  given_offset_time = false;
  double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
  double yaw_end = yaw_first;
  int layer_first = pl_orig.points[0].line;
  for (uint i = plsize - 1; i > 0; i--)
  {
    if (pl_orig.points[i].line == layer_first)
    {
      yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
      break;
    }
  }

  for (uint i = 0; i < plsize; ++i)
  {
    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = 0.;

    int layer = pl_orig.points[i].line;
    if (layer < 0 || layer >= N_SCANS)
      continue;

    double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

    if (is_first[layer])
    {
      // printf("layer: %d; is first: %d", layer, is_first[layer]);
      yaw_fp[layer] = yaw_angle;
      is_first[layer] = false;
      added_pt.curvature = 0.0;
      yaw_last[layer] = yaw_angle;
      time_last[layer] = added_pt.curvature;
      continue;
    }

    // compute offset time
    if (yaw_angle <= yaw_fp[layer])
    {
      added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
    }
    else
    {
      added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
    }

    if (added_pt.curvature < time_last[layer])
      added_pt.curvature += 360.0 / omega_l;

    yaw_last[layer] = yaw_angle;
    time_last[layer] = added_pt.curvature;

    if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
    {
      pl_surf.push_back(std::move(added_pt));
    }
  }
}

void Preprocess::hesai_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();

  const int plsize = static_cast<int>(msg->width * msg->height);
  if (plsize == 0)
    return;

  // Resolve field byte offsets once per scan; avoids pcl::fromROSMsg full copy
  // (~64K points × 32B = 2 MB) and lets us read msg->data in place.
  int off_x = -1, off_y = -1, off_z = -1, off_i = -1, off_ts = -1;
  for (const auto &f : msg->fields) {
    if (f.name == "x") off_x = f.offset;
    else if (f.name == "y") off_y = f.offset;
    else if (f.name == "z") off_z = f.offset;
    else if (f.name == "intensity") off_i = f.offset;
    else if (f.name == "timestamp") off_ts = f.offset;
  }
  if (off_x < 0 || off_y < 0 || off_z < 0 || off_ts < 0)
    return;

  const uint8_t * base = msg->data.data();
  const uint32_t step = msg->point_step;
  const double scan_start_time = rclcpp::Time(msg->header.stamp).seconds();

  double last_ts;
  std::memcpy(&last_ts, base + (plsize - 1) * step + off_ts, sizeof(double));
  const double ts_origin = (last_ts > 1e6) ? scan_start_time : 0.0;

  const float blind_sq = static_cast<float>(blind * blind);
  const int stride = std::max(1, point_filter_num);
  const int n_candidates = (plsize + stride - 1) / stride;

  // Worst case: all candidates pass blind filter. Pre-size to skip per-point grow.
  pl_surf.resize(n_candidates);

  // Parallel scan: each thread runs a contiguous chunk of candidate indices,
  // writes pl_surf[idx] only for kept points, marks dropped points by NaN, then
  // we compact in one serial pass.
  constexpr float kDropMarker = std::numeric_limits<float>::quiet_NaN();

#ifdef MP_EN
  #pragma omp parallel for num_threads(fastlio_preprocess_omp_threads()) schedule(static)
#endif
  for (int j = 0; j < n_candidates; ++j) {
    const int i = j * stride;
    if (i >= plsize) {
      pl_surf.points[j].x = kDropMarker;
      continue;
    }
    const uint8_t * p = base + i * step;

    float x, y, z, intensity = 0.0f;
    double ts;
    std::memcpy(&x, p + off_x, sizeof(float));
    std::memcpy(&y, p + off_y, sizeof(float));
    std::memcpy(&z, p + off_z, sizeof(float));
    if (off_i >= 0) std::memcpy(&intensity, p + off_i, sizeof(float));
    std::memcpy(&ts, p + off_ts, sizeof(double));

    if (x * x + y * y + z * z <= blind_sq) {
      pl_surf.points[j].x = kDropMarker;
      continue;
    }

    PointType & pt = pl_surf.points[j];
    pt.x = x;
    pt.y = y;
    pt.z = z;
    pt.intensity = intensity;
    pt.normal_x = 0.0f;
    pt.normal_y = 0.0f;
    pt.normal_z = 0.0f;
    pt.curvature = static_cast<float>((ts - ts_origin) * time_unit_scale);
  }

  // Compact: drop NaN-marked entries. Single pass, sequential, cache-friendly.
  size_t out = 0;
  for (size_t k = 0; k < pl_surf.points.size(); ++k) {
    const auto & pt = pl_surf.points[k];
    if (!std::isnan(pt.x)) {
      if (out != k) pl_surf.points[out] = pt;
      ++out;
    }
  }
  pl_surf.points.resize(out);
  pl_surf.width = out;
  pl_surf.height = 1;
}

void Preprocess::default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg)
{
  pl_surf.clear();
  pl_full.clear();

  pcl::PointCloud<pcl::PointXYZI> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0)
    return;
  pl_surf.reserve(plsize);

  for(uint i = 0; i < plsize; ++i)
  {
    PointType added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity = pl_orig.points[i].intensity;
    added_pt.curvature = 0.;

    if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind * blind))
    {
      pl_surf.push_back(std::move(added_pt));
    }
  }
}

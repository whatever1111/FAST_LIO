// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <fixposition_driver_msgs/msg/fpa_imu.hpp>
#include <fixposition_driver_msgs/msg/fpa_imubias.hpp>
#include <shm_msgs/msg/point_cloud8m_and_pose.hpp>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <ivox/ivox.hpp>   // option B: drop-in voxel-grid map backend (compile with -DUSE_IVOX=ON)

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   ikd_profile = false;  // [ikd-profile] per-scan rebuild-split line, gated by env FLIO_IKD_PROFILE
// Mapping pause/resume gate. Owned by PGO node (lio_slam_fastLioPGO publishes
// /lio_slam/mapping_enabled with transient_local QoS); we mirror it here.
// When false: skip ikd-tree updates and skip pcl_wait_save accumulation, so the
// saved PCD doesn't contain frames captured while operator had mapping paused.
// EKF prediction, ICP and /Odometry publishing keep running so pose continuity
// is preserved.
std::atomic<bool> mapping_enabled{true};
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;
double imu_time_offset = 0.0;
constexpr int kFixpositionBiasStatusThresh = 2;
Eigen::Vector3d fixposition_bias_acc = Eigen::Vector3d::Zero();
Eigen::Vector3d fixposition_bias_gyr = Eigen::Vector3d::Zero();
bool fixposition_has_bias = false;
int fixposition_bias_status = -1;
std::mutex mtx_fixposition_bias;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = string(ROOT_DIR);
string map_file_path, lid_topic, imu_topic, imu_input_type = "sensor_msgs", imu_bias_topic = "/fixposition/fpa/imubias";

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
// Position-observability metric, refreshed each h_share_model call: eigenvalues
// of the normal information matrix M = sum(n n^T) over the matched plane normals.
// pos_obs_z_weak = |Z component| of the weakest eigenvector; →1 means Z is the
// least-constrained direction (degenerate Z), the root trigger of the bag2
// divergence (see memory bag2-divergence-is-flio2-frontend).
double pos_obs_eig_min = 0.0, pos_obs_eig_max = 0.0, pos_obs_z_weak = 0.0;
bool   degeneracy_debug = false;     // verbose per-frame [DEGEN] diagnostics

// Planar-motion (zero body-vertical-velocity) soft constraint — the A'-2 fix.
//   When LiDAR loses Z observability (pos_obs_z_weak high), the iEKF dumps the
//   unexplained residual into the accel bias and the state flies off. This adds a
//   soft pseudo-measurement "body-frame vertical velocity ≈ 0" (a ground-robot
//   non-holonomic prior, like the wheel-odom update) that supplies the missing
//   vertical information and bounds the runaway at its source.
//   BODY frame → it does NOT fight legitimate slope motion: world-frame vertical
//   velocity on a ramp comes from the robot's pitch (rotation state), not from
//   body-frame vz. Gated on z_weak so it has zero effect when Z is well observed.
bool   planar_constraint_en = false;
double planar_constraint_noise = 0.1;          // m/s, soft (larger = weaker prior)
double planar_constraint_z_weak_thresh = 0.5;  // apply only when pos_obs_z_weak exceeds this

// Gravity-alignment leveling prior (A1) — the ROOT-CAUSE fix for the iVox pitch-
// coupled Z drift. The planar (body-vz≈0) constraint above pins a R↔vel CONSISTENCY
// relation the iEKF already satisfies, so it carries no absolute-pitch information and
// cannot catch a biased pitch (world-Z drift = forward_distance × sin(pitch_err)) —
// proven by zero effect from 4× strengthening it. This supplies the missing ABSOLUTE
// reference: when LiDAR Z is degenerate AND the IMU is in low-linear-accel motion
// (|‖a‖−g| small), the accelerometer's specific-force direction IS world-up expressed
// in the body frame. A soft pseudo-measurement pulls the grav-state up-axis onto it,
// constraining roll/pitch only — yaw lies in the measurement null space (gravity is
// yaw-invariant), so the LiDAR-excellent heading is never touched.
bool   gravity_align_en = false;
double gravity_align_noise = 0.05;            // unit-vector (≈rad) std of the leveling prior
double gravity_align_z_weak_thresh = 0.3;     // engage only when pos_obs_z_weak exceeds this
double gravity_align_accel_tol = 0.05;        // |‖a‖−g|/g must be below this (low-linear-accel gate)
double gravity_align_gyro_tol = 0.35;         // rad/s; skip during fast turns (lever-arm safety)

// Ground-return reweighting (A'-3) — the LiDAR-geometry fix, complementary to the
// IMU/motion priors above. In open, planar scenes the few, far, grazing-incidence
// GROUND returns are out-voted in the iEKF information matrix by abundant near-
// VERTICAL wall returns, so the vertical (z/roll/pitch) DOFs are weakly constrained
// and Z random-walks (the metastable "balloon"). Unlike the planar/gravity priors —
// which add a motion/IMU pseudo-measurement and were proven NEUTRAL here because the
// drift is a metastable random walk, not a systematic bias — this redistributes
// weight among the REAL point-to-plane measurements: surfaces whose matched normal
// is near-vertical (|n_z| ≥ ground_constraint_normal_z_min, i.e. horizontal ground)
// get their information scaled by ground_constraint_weight². Both the h_x row and
// the residual are scaled — R is a uniform scalar in the IKFoM update, so this is an
// effective per-measurement noise of R/w². Gated on pos_obs_z_weak so it is inert
// when Z is well observed, and only ground-normal points are touched, so horizontal
// (x/y/yaw) constraints from walls are never weakened. Default OFF.
bool   ground_constraint_en = false;
double ground_constraint_weight = 6.0;          // info scales by weight^2 (>1 = stronger)
double ground_constraint_normal_z_min = 0.966;  // |n_z| ≥ cos(15°) ⇒ treat as ground/horizontal
double ground_constraint_z_weak_thresh = 0.5;   // apply only when pos_obs_z_weak exceeds this
int    ground_constraint_boosted = 0;           // diag: # boosted ground points last h_share call

// Divergence guard (P1): when LiDAR correspondences collapse (scan starvation
// under CPU/IO load, or feature-poor geometry), the iEKF runs on IMU
// dead-reckoning and the state — especially Z — runs away to ±km and never
// re-locks (scan leaves the map → permanent "No Effective Points"). This guard
// detects the sustained collapse and BOUNDS the runaway: it pins body-vz≈0,
// clamps an implausible body speed, freezes the map so dead-reckoned scans do
// not corrupt it, and flags the odometry degraded. It is INERT in normal
// operation (only fires after divergence_guard_streak consecutive scans whose
// effective-point count falls below divergence_guard_min_eff). Recovery is
// automatic: once correspondences return, the streak resets and the front end
// re-locks against the (uncorrupted) map.
bool   divergence_guard_en = true;
int    divergence_guard_min_eff = 50;      // effct_feat_num below this = degenerate scan
int    divergence_guard_streak = 5;        // consecutive degenerate scans → enter degraded
double divergence_guard_max_speed = 30.0;  // m/s; body-speed clamp while degraded (headroom over any platform speed)
int    consec_low_eff = 0;                 // running degenerate-scan streak counter
bool   flio_map_frozen = false;            // degraded: skip map_incremental (no garbage)
bool   flio_degraded_odom = false;         // degraded: inflate published pose covariance

bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;
bool   diag_first_lidar_cb_logged = false;
bool   diag_first_imu_cb_logged = false;
bool   diag_first_fixposition_bias_logged = false;
bool   diag_first_fixposition_bias_applied_logged = false;
bool   diag_first_sync_logged = false;
bool   diag_first_valid_scan_logged = false;
bool   diag_first_no_point_logged = false;
bool   diag_first_odom_pub_logged = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
using SteadyClock = std::chrono::steady_clock;
using SteadyTimePoint = std::chrono::time_point<SteadyClock>;

struct LatencyStats
{
    int count = 0;
    double sum_ms = 0.0;
    double min_ms = 1e18;
    double max_ms = 0.0;

    void add(double ms)
    {
        ++count;
        sum_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    double mean() const { return count > 0 ? sum_ms / static_cast<double>(count) : 0.0; }
};

deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<SteadyTimePoint>            lidar_receive_time_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

bool latency_diag_en = true;
double latency_log_period_sec = 1.0;
SteadyTimePoint current_lidar_receive_time;
SteadyTimePoint last_latency_log_time;
LatencyStats sensor_to_odom_latency_stats;

bool   wheel_odom_en = false;
string wheel_topic = "/lio/twist";
double wheel_speed_scale = 1.0;
double wheel_vel_noise_vx = 0.10;
double wheel_vel_noise_vy = 0.05;
double wheel_vel_noise_vz = 0.10;
M3D    R_robot_to_imu = Eye3d;
deque<Eigen::Vector3d> twist_buffer;
mutex  mtx_twist;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

// Map backend: ikd-Tree by default; iVox (option B) when built with -DUSE_IVOX=ON. iVox is a
// drop-in for the hot-path API (Add_Points / Nearest_Search / Build / size / validnum /
// set_downsample_param / Delete_Point_Boxes-as-noop), so the call sites below are unchanged;
// only the init check, the dead flatten block, and the constructor Init are #ifdef'd.
#ifdef USE_IVOX
using MapBackend = lio_ivox::IVox<PointType>;
#else
using MapBackend = KD_TREE<PointType>;
#endif
MapBackend ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    const auto receive_time = SteadyClock::now();
    if (!diag_first_lidar_cb_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first lidar cb (PointCloud2) stamp=%.3f",
                    get_time_sec(msg->header.stamp));
        diag_first_lidar_cb_logged = true;
    }
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        lidar_receive_time_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    lidar_receive_time_buffer.push_back(receive_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// Direct subscriber for shm_msgs/PointCloud8mAndPose (/pre/all_point). Decodes the
// fixed 8MB buffer in place (no relay node, no extra DDS hop) — same buffering path
// as standard_pcl_cbk, just a different message type handed to Preprocess.
void shm_allpoint_cbk(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr msg)
{
    const auto receive_time = SteadyClock::now();
    if (!diag_first_lidar_cb_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first lidar cb (shm PointCloud8mAndPose) stamp=%.3f",
                    get_time_sec(msg->header.stamp));
        diag_first_lidar_cb_logged = true;
    }
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        lidar_receive_time_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    lidar_receive_time_buffer.push_back(receive_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    const auto receive_time = SteadyClock::now();
    if (!diag_first_lidar_cb_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first lidar cb (Livox) stamp=%.3f",
                    get_time_sec(msg->header.stamp));
        diag_first_lidar_cb_logged = true;
    }
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        lidar_receive_time_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    lidar_receive_time_buffer.push_back(receive_time);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void enqueue_imu_msg(sensor_msgs::msg::Imu::SharedPtr msg, double debug_raw_stamp)
{
    publish_count ++;
    if (!diag_first_imu_cb_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first imu cb raw_stamp=%.3f adjusted_stamp=%.3f",
                    debug_raw_stamp,
                    get_time_sec(msg->header.stamp));
        diag_first_imu_cb_logged = true;
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }
    enqueue_imu_msg(msg, get_time_sec(msg_in->header.stamp));
}

void imu_bias_cbk(const fixposition_driver_msgs::msg::FpaImubias::UniquePtr msg_in)
{
    std::lock_guard<std::mutex> lock(mtx_fixposition_bias);
    fixposition_bias_acc = Eigen::Vector3d(msg_in->bias_acc.x, msg_in->bias_acc.y, msg_in->bias_acc.z);
    fixposition_bias_gyr = Eigen::Vector3d(msg_in->bias_gyr.x, msg_in->bias_gyr.y, msg_in->bias_gyr.z);
    fixposition_bias_status = msg_in->imu_status;
    fixposition_has_bias = true;
    if (!diag_first_fixposition_bias_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first bias status=%d acc=[%.5f %.5f %.5f] gyr=[%.6f %.6f %.6f]",
                    fixposition_bias_status,
                    fixposition_bias_acc.x(),
                    fixposition_bias_acc.y(),
                    fixposition_bias_acc.z(),
                    fixposition_bias_gyr.x(),
                    fixposition_bias_gyr.y(),
                    fixposition_bias_gyr.z());
        diag_first_fixposition_bias_logged = true;
    }
}

void imu_fpa_cbk(const fixposition_driver_msgs::msg::FpaImu::UniquePtr msg_in)
{
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(msg_in->data));
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->data.header.stamp) + imu_time_offset);

    bool apply_bias = false;
    {
        std::lock_guard<std::mutex> lock(mtx_fixposition_bias);
        apply_bias = fixposition_has_bias && fixposition_bias_status >= kFixpositionBiasStatusThresh;
        if (apply_bias) {
            msg->linear_acceleration.x -= fixposition_bias_acc.x();
            msg->linear_acceleration.y -= fixposition_bias_acc.y();
            msg->linear_acceleration.z -= fixposition_bias_acc.z();
            msg->angular_velocity.x -= fixposition_bias_gyr.x();
            msg->angular_velocity.y -= fixposition_bias_gyr.y();
            msg->angular_velocity.z -= fixposition_bias_gyr.z();
        }
    }
    if (apply_bias && !diag_first_fixposition_bias_applied_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first bias-corrected FPA imu raw_stamp=%.3f adjusted_stamp=%.3f",
                    get_time_sec(msg_in->data.header.stamp),
                    get_time_sec(msg->header.stamp));
        diag_first_fixposition_bias_applied_logged = true;
    }
    enqueue_imu_msg(msg, get_time_sec(msg_in->data.header.stamp));
}

void twist_cbk(const geometry_msgs::msg::TwistStamped::UniquePtr msg_in)
{
    Eigen::Vector3d v(msg_in->twist.linear.x, msg_in->twist.linear.y, msg_in->twist.linear.z);
    std::lock_guard<std::mutex> lk(mtx_twist);
    twist_buffer.push_back(v);
    while (twist_buffer.size() > 100) twist_buffer.pop_front();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        if (!diag_first_sync_logged) {
            RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                        "[STARTUP][FAST_LIO] first sync lidar_beg=%.3f imu_buf=%zu lidar_buf=%zu",
                        time_buffer.front(),
                        imu_buffer.size(),
                        lidar_buffer.size());
            diag_first_sync_logged = true;
        }
        meas.lidar_beg_time = time_buffer.front();
        if (!lidar_receive_time_buffer.empty()) {
            current_lidar_receive_time = lidar_receive_time_buffer.front();
        }
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    if (!lidar_receive_time_buffer.empty()) lidar_receive_time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
// --- Asynchronous ikd-tree map update (off the LiDAR processing thread) -----------------
// The ikd-tree is single-operator: at most ONE thread may mutate/read it at a time (its
// background rebuild thread is the only sanctioned exception, guarded internally by
// search_flag_mutex). map_incremental()'s Add_Points is the second-largest per-scan tree
// cost and ~94% of its spikes are background-rebuild lock-wait (measured). We move it off
// the spin thread WITHOUT breaking the single-operator invariant by TIME-SLICING:
//
//   scan N:   [main] fov -> search -> publish -> build add-lists -> dispatchMapAdd(N) --,
//             [main] publish clouds, return from timer_callback                         | worker
//             [exec] services IMU/LiDAR cbks (buffer push only, NO tree access)         |  runs
//   scan N+1: [main] joinMapAdd()  <-- blocks iff Add(N) not yet done -------------------'  Add(N)
//             [main] fov -> search -> ...                                                    here
//
// Add(N) overlaps only the main thread's NON-tree work (wait-for-scan + IMU undistort), so
// it never runs concurrently with Nearest_Search. The background rebuild it may trigger
// runs concurrently exactly as in the synchronous path. Net: the Add cost (~16ms steady,
// spikes to ~78ms) leaves the per-scan critical path and hides in the inter-scan idle gap.
bool                    async_map_en = true;   // gated by env FLIO_ASYNC_MAP ("0" disables)
std::mutex              map_add_mtx;
std::condition_variable map_add_cv;
PointVector             map_add_ds_;            // handoff: points to add WITH downsample
PointVector             map_add_nods_;          // handoff: points to add WITHOUT downsample
bool                    map_add_ready_   = false;  // a job is queued, not yet taken
bool                    map_add_running_ = false;  // worker is executing a job
bool                    map_worker_stop_ = false;
std::thread             map_worker_thread;
std::atomic<uint64_t>   g_async_add_us{0};      // last completed worker Add() wall time (us)
double                  g_map_join_wait_ms = 0; // last joinMapAdd() block time (main-only)

void mapWorkerLoop()
{
    std::unique_lock<std::mutex> lk(map_add_mtx);
    while (true)
    {
        map_add_cv.wait(lk, [] { return map_add_ready_ || map_worker_stop_; });
        if (map_worker_stop_ && !map_add_ready_) break;
        PointVector toAdd, noDS;
        toAdd.swap(map_add_ds_);
        noDS.swap(map_add_nods_);
        map_add_ready_   = false;
        map_add_running_ = true;
        lk.unlock();

        const double st = omp_get_wtime();
        ikdtree.Add_Points(toAdd, true);
        ikdtree.Add_Points(noDS, false);
        g_async_add_us.store((uint64_t)((omp_get_wtime() - st) * 1e6), std::memory_order_relaxed);

        lk.lock();
        map_add_running_ = false;
        map_add_cv.notify_all();
    }
}

void startMapWorker()
{
    map_worker_stop_ = false;
    if (!map_worker_thread.joinable())
        map_worker_thread = std::thread(mapWorkerLoop);
}

void stopMapWorker()  // drain any in-flight Add + join; idempotent
{
    {
        std::lock_guard<std::mutex> lk(map_add_mtx);
        map_worker_stop_ = true;
    }
    map_add_cv.notify_all();
    if (map_worker_thread.joinable()) map_worker_thread.join();
}

// Hand the two add-lists to the worker. Precondition: worker idle (caller joined this
// scan's predecessor). Moves the vectors so the hot path does no point copy.
void dispatchMapAdd(PointVector &toAdd, PointVector &noDS)
{
    {
        std::lock_guard<std::mutex> lk(map_add_mtx);
        map_add_ds_.swap(toAdd);
        map_add_nods_.swap(noDS);
        map_add_ready_ = true;
    }
    map_add_cv.notify_one();
}

// Block until the worker has finished the in-flight Add. Called before the next scan's
// first tree op (lasermap_fov_segment) to restore the single-operator invariant.
void joinMapAdd()
{
    const double t0 = omp_get_wtime();
    std::unique_lock<std::mutex> lk(map_add_mtx);
    map_add_cv.wait(lk, [] { return !map_add_ready_ && !map_add_running_; });
    g_map_join_wait_ms = (omp_get_wtime() - t0) * 1000.0;
}

void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    if (async_map_en)
    {
        // Off-thread: the worker calls Add_Points during the inter-scan gap; joined before
        // the next scan's first tree op. add_point_size is known here (sizes); the Add wall
        // time lands in g_async_add_us (see mapWorkerLoop). kdtree_incremental_time is left
        // untouched in this path — it is a main-thread debug stat and the worker must not
        // write plain globals the main thread reads.
        add_point_size = (int)(PointToAdd.size() + PointNoNeedDownsample.size());
        dispatchMapAdd(PointToAdd, PointNoNeedDownsample);
    }
    else
    {
        double st_time = omp_get_wtime();
        add_point_size = ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);
        add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
        kdtree_incremental_time = omp_get_wtime() - st_time;
    }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

// --- Asynchronous PCD accumulator (off the LiDAR processing thread) ---
// `*pcl_wait_save += *scan` reallocates the whole (multi-hundred-MB, growing) cloud at
// std::vector capacity doublings; profiling showed this stalled the single processing
// thread for up to ~750ms as the map grew, overflowing the best-effort LiDAR buffer and
// dropping bursts of input scans. The hot path now only transforms the scan and ENQUEUES
// it; this dedicated writer thread owns the accumulation, so the realloc/copy cost no
// longer blocks scan intake. Output is byte-identical (same full-res scans.pcd at shutdown).
// pcl_wait_save is touched ONLY by the writer thread and by readers (save_to_pcd / final
// save) — guarded by pcl_wait_save_mtx since those readers run on the executor thread.
std::mutex pcl_wait_save_mtx;
std::mutex pcd_queue_mtx;
std::condition_variable pcd_queue_cv;
std::deque<PointCloudXYZI::Ptr> pcd_queue;
std::atomic<bool> pcd_writer_running{false};
std::atomic<uint64_t> pcd_enqueued{0};  // scans handed to the writer
std::atomic<uint64_t> pcd_accumulated{0};  // scans folded into pcl_wait_save
std::thread pcd_writer_thread;

void pcdWriterLoop()
{
    while (true) {
        PointCloudXYZI::Ptr cloud;
        {
            std::unique_lock<std::mutex> lk(pcd_queue_mtx);
            pcd_queue_cv.wait(lk, [] { return !pcd_queue.empty() || !pcd_writer_running.load(); });
            if (pcd_queue.empty() && !pcd_writer_running.load()) break;  // drained + asked to stop
            cloud = pcd_queue.front();
            pcd_queue.pop_front();
        }
        if (!cloud || cloud->empty()) { pcd_accumulated.fetch_add(1); continue; }
        {
            std::lock_guard<std::mutex> save_lk(pcl_wait_save_mtx);
            *pcl_wait_save += *cloud;  // realloc/copy happens HERE, off the processing thread
            if (pcd_save_interval > 0) {
                static int scan_wait_num = 0;
                if (++scan_wait_num >= pcd_save_interval && !pcl_wait_save->empty()) {
                    pcd_index++;
                    string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
                    pcl::PCDWriter pcd_writer;
                    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                    pcl_wait_save->clear();
                    scan_wait_num = 0;
                }
            }
        }
        pcd_accumulated.fetch_add(1);  // signal flushPcdQueue() that this scan is folded in
    }
}

// Block until the writer has folded in every scan enqueued so far. Used by the readers
// (map_save service / save_to_pcd) so a mid-run save can't miss still-queued scans.
void flushPcdQueue()
{
    const uint64_t target = pcd_enqueued.load();
    while (pcd_writer_running.load() && pcd_accumulated.load() < target) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void startPcdWriter()
{
    bool expected = false;
    if (pcd_writer_running.compare_exchange_strong(expected, true)) {
        pcd_writer_thread = std::thread(pcdWriterLoop);
    }
}

void stopPcdWriter()  // drain queue + join; idempotent, call once at shutdown
{
    if (!pcd_writer_running.exchange(false)) return;
    pcd_queue_cv.notify_all();
    if (pcd_writer_thread.joinable()) pcd_writer_thread.join();
}
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    // Same pause gate: skip accumulation when mapping is disabled, so the
    // exported scans.pcd doesn't contain operator-suppressed frames.
    if (pcd_save_en && mapping_enabled.load(std::memory_order_relaxed))
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        // Hand the transformed scan to the writer thread; accumulation + any interval
        // flush now run there (see pcdWriterLoop), off this LiDAR processing thread.
        {
            std::lock_guard<std::mutex> lk(pcd_queue_mtx);
            pcd_queue.push_back(laserCloudWorld);
        }
        pcd_enqueued.fetch_add(1);
        pcd_queue_cv.notify_one();
    }
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    // Voxel-downsample + hard cap so the accumulated /Laser_map stays bounded and
    // renderable. publish_map otherwise just appends a scan per tick and grows
    // without limit. Start at a 0.3 m leaf and coarsen until under ~100k points.
    {
        constexpr size_t kMapPubMaxPoints = 100000;
        float leaf = 0.3f;
        for (int pass = 0; pass < 8; ++pass)
        {
            pcl::VoxelGrid<PointType> vg;
            vg.setLeafSize(leaf, leaf, leaf);
            vg.setInputCloud(pcl_wait_pub);
            PointCloudXYZI::Ptr ds(new PointCloudXYZI());
            vg.filter(*ds);
            pcl_wait_pub = ds;
            if (pcl_wait_pub->size() <= kMapPubMaxPoints) break;
            leaf *= 1.5f;
        }
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;
    // pcl_wait_save accumulates full-resolution feats_undistort every scan (when pcd_save_en=true).
    // pcl_wait_pub only has ~1/10 frames from the 1Hz publish_map timer with downsampled points.
    // Make sure every enqueued scan is folded in, then lock against the async writer
    // thread (the other accessor of pcl_wait_save) for the write itself.
    flushPcdQueue();
    std::lock_guard<std::mutex> lk(pcl_wait_save_mtx);
    pcd_writer.writeBinary(map_file_path, *pcl_wait_save);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    if (!diag_first_odom_pub_logged) {
        RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                    "[STARTUP][FAST_LIO] first /Odometry publish stamp=%.3f pos=[%.3f %.3f %.3f]",
                    lidar_end_time,
                    odomAftMapped.pose.pose.position.x,
                    odomAftMapped.pose.pose.position.y,
                    odomAftMapped.pose.pose.position.z);
        diag_first_odom_pub_logged = true;
    }

    if (latency_diag_en && current_lidar_receive_time.time_since_epoch().count() != 0) {
        const auto now = SteadyClock::now();
        const double latency_ms = std::chrono::duration<double, std::milli>(now - current_lidar_receive_time).count();
        sensor_to_odom_latency_stats.add(latency_ms);
        if (last_latency_log_time.time_since_epoch().count() == 0 ||
            std::chrono::duration<double>(now - last_latency_log_time).count() >= latency_log_period_sec) {
            RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                        "[LATENCY][FAST_LIO][SENSOR_TO_ODOM] count=%d mean=%.2fms min=%.2fms max=%.2fms",
                        sensor_to_odom_latency_stats.count,
                        sensor_to_odom_latency_stats.mean(),
                        sensor_to_odom_latency_stats.min_ms,
                        sensor_to_odom_latency_stats.max_ms);
            last_latency_log_time = now;
        }
    }
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    // Divergence guard (P1): while the front end is dead-reckoning (LiDAR
    // starved), advertise large position variance on the ROS pose covariance
    // (x,y,z diagonal) so the PGO back-end down-weights these poses instead of
    // trusting a bounded-but-uncertain dead-reckoned estimate.
    if (flio_degraded_odom)
    {
        constexpr double kDegradedPosVar = 100.0;  // (10 m)^2
        odomAftMapped.pose.covariance[0]  += kDegradedPosVar;   // x
        odomAftMapped.pose.covariance[7]  += kDegradedPosVar;   // y
        odomAftMapped.pose.covariance[14] += kDegradedPosVar;   // z
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.header.stamp = get_ros_time(lidar_end_time);
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        // Respect OMP_NUM_THREADS when set so the runtime CPU budget (cgroup
        // --cpus) can be honoured; the compile-time MP_PROC_NUM (8 on x86) would
        // otherwise spawn 8 OMP threads and oversubscribe a 4-CPU container,
        // starving the PGO node it shares cores with.
        {
            static const int omp_threads = []() {
                const char * e = std::getenv("OMP_NUM_THREADS");
                return (e && std::atoi(e) > 0) ? std::atoi(e) : MP_PROC_NUM;
            }();
            omp_set_num_threads(omp_threads);
        }
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    // Position-observability metric (degeneracy detector): eigenvalues of the
    // normal information matrix M = sum(n n^T) over effective points. The min
    // eigenvalue is the weakest-constrained position direction; if its
    // eigenvector points along Z, the Z update is ill-conditioned.
    {
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
        for (int i = 0; i < effct_feat_num; i++)
        {
            const PointType &np = corr_normvect->points[i];
            Eigen::Vector3d n(np.x, np.y, np.z);
            M.noalias() += n * n.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
        pos_obs_eig_min = es.eigenvalues()(0);
        pos_obs_eig_max = es.eigenvalues()(2);
        pos_obs_z_weak = std::abs(es.eigenvectors()(2, 0));  // |Z| of weakest eigenvector
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);
    ground_constraint_boosted = 0;

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;

        /*** Ground-return reweighting (A'-3): when Z is weakly observable, boost
         *   near-horizontal (ground) surfaces so they are not out-voted by the
         *   abundant vertical-wall returns. Scaling this row of h_x AND its residual
         *   by w scales the measurement's information by w² (R is a uniform scalar in
         *   the IKFoM update — see update_iterated_dyn_share_modified). Only
         *   ground-normal points are touched, so wall-borne x/y/yaw is untouched. ***/
        if (ground_constraint_en && pos_obs_z_weak > ground_constraint_z_weak_thresh &&
            std::fabs(norm_p.z) >= ground_constraint_normal_z_min)
        {
            ekfom_data.h_x.block<1, 12>(i, 0) *= ground_constraint_weight;
            ekfom_data.h(i) *= ground_constraint_weight;
            ++ground_constraint_boosted;
        }
    }

    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<double>("publish.map_period_sec", 5.0);  // /Laser_map publish period (s); 5s = 0.2Hz
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<bool>("degeneracy_debug", false);
        this->declare_parameter<bool>("planar_constraint_en", false);
        this->declare_parameter<double>("planar_constraint_noise", 0.1);
        this->declare_parameter<double>("planar_constraint_z_weak_thresh", 0.5);
        this->declare_parameter<bool>("gravity_align_en", false);
        this->declare_parameter<double>("gravity_align_noise", 0.05);
        this->declare_parameter<double>("gravity_align_z_weak_thresh", 0.3);
        this->declare_parameter<double>("gravity_align_accel_tol", 0.05);
        this->declare_parameter<double>("gravity_align_gyro_tol", 0.35);
        this->declare_parameter<bool>("ground_constraint_en", false);
        this->declare_parameter<double>("ground_constraint_weight", 6.0);
        this->declare_parameter<double>("ground_constraint_normal_z_min", 0.966);
        this->declare_parameter<double>("ground_constraint_z_weak_thresh", 0.5);
        this->declare_parameter<bool>("divergence_guard_en", true);
        this->declare_parameter<int>("divergence_guard_min_eff", 50);
        this->declare_parameter<int>("divergence_guard_streak", 5);
        this->declare_parameter<double>("divergence_guard_max_speed", 30.0);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("data_dir", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<string>("common.imu_input_type", "sensor_msgs");
        this->declare_parameter<string>("common.imu_bias_topic", "/fixposition/fpa/imubias");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("common.imu_time_offset", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("diagnostics.latency_enable", true);
        this->declare_parameter<double>("diagnostics.latency_log_period_sec", 1.0);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<string>("prior_map_pcd", "");
        this->declare_parameter<vector<double>>("initial_pose", vector<double>());
        // initial_pose_full_rpy_override:
        //   false (default) — inject only XYZ + yaw from initial_pose; keep
        //     roll/pitch from the IMU-derived gravity-aligned current state.
        //   true            — fully overwrite roll/pitch from initial_pose. The
        //     caller MUST guarantee these match gravity (accel along -Z),
        //     otherwise point cloud transforms will be incorrect.
        this->declare_parameter<bool>("initial_pose_full_rpy_override", false);
        // Deprecated alias for initial_pose_full_rpy_override. Kept for
        // backward compatibility; will emit a warning if set to true.
        this->declare_parameter<bool>("initial_pose_apply_roll_pitch", false);
        this->declare_parameter<bool>("wheel_odom_en", false);
        this->declare_parameter<string>("wheel_topic", "/lio/twist");
        this->declare_parameter<double>("wheel_speed_scale", 1.0);
        this->declare_parameter<double>("wheel_vel_noise_vx", 0.10);
        this->declare_parameter<double>("wheel_vel_noise_vy", 0.05);
        this->declare_parameter<double>("wheel_vel_noise_vz", 0.10);
        this->declare_parameter<vector<double>>("wheel_extrinsic_R", vector<double>());
        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<bool>("degeneracy_debug", degeneracy_debug, false);
        this->get_parameter_or<bool>("planar_constraint_en", planar_constraint_en, false);
        this->get_parameter_or<double>("planar_constraint_noise", planar_constraint_noise, 0.1);
        this->get_parameter_or<double>("planar_constraint_z_weak_thresh", planar_constraint_z_weak_thresh, 0.5);
        this->get_parameter_or<bool>("gravity_align_en", gravity_align_en, false);
        this->get_parameter_or<double>("gravity_align_noise", gravity_align_noise, 0.05);
        this->get_parameter_or<double>("gravity_align_z_weak_thresh", gravity_align_z_weak_thresh, 0.3);
        this->get_parameter_or<double>("gravity_align_accel_tol", gravity_align_accel_tol, 0.05);
        this->get_parameter_or<double>("gravity_align_gyro_tol", gravity_align_gyro_tol, 0.35);
        this->get_parameter_or<bool>("ground_constraint_en", ground_constraint_en, false);
        this->get_parameter_or<double>("ground_constraint_weight", ground_constraint_weight, 6.0);
        this->get_parameter_or<double>("ground_constraint_normal_z_min", ground_constraint_normal_z_min, 0.966);
        this->get_parameter_or<double>("ground_constraint_z_weak_thresh", ground_constraint_z_weak_thresh, 0.5);
        this->get_parameter_or<bool>("divergence_guard_en", divergence_guard_en, true);
        this->get_parameter_or<int>("divergence_guard_min_eff", divergence_guard_min_eff, 50);
        this->get_parameter_or<int>("divergence_guard_streak", divergence_guard_streak, 5);
        this->get_parameter_or<double>("divergence_guard_max_speed", divergence_guard_max_speed, 30.0);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        string data_dir_param;
        this->get_parameter_or<string>("data_dir", data_dir_param, "");
        if (!data_dir_param.empty()) {
            root_dir = data_dir_param;
            if (root_dir.back() != '/') root_dir += '/';
        }
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<string>("common.imu_input_type", imu_input_type, string("sensor_msgs"));
        this->get_parameter_or<string>("common.imu_bias_topic", imu_bias_topic, string("/fixposition/fpa/imubias"));
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("common.imu_time_offset", imu_time_offset, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        p_imu->runtime_log_en = runtime_pos_log;  // mirror to ImuProcess so Log/imu.txt is written
        ikd_profile = (getenv("FLIO_IKD_PROFILE") != nullptr);  // [ikd-profile] opt-in via env
        { const char *am = getenv("FLIO_ASYNC_MAP"); if (am && std::string(am) == "0") async_map_en = false; }
#ifdef USE_IVOX
        {   // option B: iVox map backend. nearby_type MUST be >=18 (Phase-0: NEARBY6 inadequate).
            // res=0.5 NN voxels give the best probe NN-agreement (99.9%, tightest normals);
            // a GLOBAL fine-grid dedup (= filter_size_map) keeps the map at ikd-Tree's 542k.
            const double ivox_res    = this->declare_parameter<double>("ivox_grid_resolution", 0.5);
            const int    ivox_nearby = this->declare_parameter<int>("ivox_nearby_type", 26);
            const int    ivox_vcap   = this->declare_parameter<int>("ivox_voxel_capacity", 50);
            const int    ivox_maxvox = this->declare_parameter<int>("ivox_max_voxels", 5000000);
            ikdtree.Init(static_cast<float>(ivox_res), ivox_nearby, ivox_vcap,
                         static_cast<std::size_t>(ivox_maxvox));
            RCLCPP_INFO(this->get_logger(),
                "[IVOX] map backend = iVox (res=%.2f nearby=%d voxel_cap=%d max_voxels=%d)",
                ivox_res, ivox_nearby, ivox_vcap, ivox_maxvox);
        }
#endif
        this->get_parameter_or<bool>("diagnostics.latency_enable", latency_diag_en, true);
        this->get_parameter_or<double>("diagnostics.latency_log_period_sec", latency_log_period_sec, 1.0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<string>("prior_map_pcd", prior_map_pcd_, string(""));
        this->get_parameter_or<vector<double>>("initial_pose", initial_pose_vec_, vector<double>());
        {
            bool full_rpy_override = false;
            bool legacy_apply_rp = false;
            this->get_parameter_or<bool>("initial_pose_full_rpy_override", full_rpy_override, false);
            this->get_parameter_or<bool>("initial_pose_apply_roll_pitch", legacy_apply_rp, false);
            if (legacy_apply_rp) {
                RCLCPP_WARN(this->get_logger(),
                    "Parameter 'initial_pose_apply_roll_pitch' is deprecated; "
                    "use 'initial_pose_full_rpy_override' instead.");
            }
            initial_pose_full_rpy_override_ = full_rpy_override || legacy_apply_rp;
            if (initial_pose_full_rpy_override_) {
                RCLCPP_WARN(this->get_logger(),
                    "initial_pose_full_rpy_override=true: overriding IMU-aligned "
                    "roll/pitch with values from initial_pose. Caller must ensure "
                    "these match gravity (accel along -Z), else point cloud "
                    "transforms will be incorrect.");
            }
        }
        this->get_parameter_or<bool>("wheel_odom_en", wheel_odom_en, false);
        this->get_parameter_or<string>("wheel_topic", wheel_topic, string("/lio/twist"));
        this->get_parameter_or<double>("wheel_speed_scale", wheel_speed_scale, 1.0);
        this->get_parameter_or<double>("wheel_vel_noise_vx", wheel_vel_noise_vx, 0.10);
        this->get_parameter_or<double>("wheel_vel_noise_vy", wheel_vel_noise_vy, 0.05);
        this->get_parameter_or<double>("wheel_vel_noise_vz", wheel_vel_noise_vz, 0.10);
        vector<double> wext;
        this->get_parameter_or<vector<double>>("wheel_extrinsic_R", wext, vector<double>());
        if (wext.size() == 9)
            R_robot_to_imu = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(wext.data());
        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        fout_jump.open(DEBUG_FILE_DIR("jump_diag.txt"), ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else if (p_pre->lidar_type == ALLPOINT)
        {
            // /pre/all_point is published RELIABLE with fixed 8MB messages; a small
            // reliable KeepLast queue matches the bag publisher and bounds memory
            // (a few buffered scans) while avoiding silent drops of whole scans.
            rclcpp::QoS shm_qos(rclcpp::KeepLast(5));
            sub_pcl_shm_ = this->create_subscription<shm_msgs::msg::PointCloud8mAndPose>(lid_topic, shm_qos, shm_allpoint_cbk);
            RCLCPP_INFO(this->get_logger(), "shm allpoint mode: subscribing %s (shm_msgs/PointCloud8mAndPose)", lid_topic.c_str());
        }
        else
        {
            // Diagnostic knob (env FLIO_LIDAR_QDEPTH): a deep best-effort LiDAR queue so
            // the single-thread executor's processing spikes buffer scans instead of
            // dropping them. Eliminating real-time scan drops makes the front-end process
            // an identical scan sequence every run — required to turn the metastable bag2
            // raw-Z into a repeatable, measurable metric. Unset → SensorDataQoS (depth 5),
            // the production default. best_effort stays compatible with the bag publisher.
            const char* qd = std::getenv("FLIO_LIDAR_QDEPTH");
            const int qdepth = (qd && std::atoi(qd) > 0) ? std::atoi(qd) : 0;
            // Default RELIABLE KeepLast(5) to MATCH the lidar_adapter publisher
            // (it publishes canonical /lio/points reliable so no 8MB scan is silently
            // dropped). The matched reliable+volatile+keep_last pair is also what lets
            // the intra-process manager do the zero-copy unique_ptr handoff when this
            // node is co-composed with the adapter. FLIO_LIDAR_QDEPTH keeps the
            // best-effort deep-queue diagnostic override.
            rclcpp::QoS pc_qos = qdepth > 0
                ? rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qdepth))).best_effort()
                : rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, pc_qos, standard_pcl_cbk);
        }
        if (imu_input_type == "fixposition_fpa")
        {
            auto imu_qos = rclcpp::SensorDataQoS();
            sub_imu_fpa_ = this->create_subscription<fixposition_driver_msgs::msg::FpaImu>(imu_topic, imu_qos, imu_fpa_cbk);
            sub_imu_bias_ = this->create_subscription<fixposition_driver_msgs::msg::FpaImubias>(imu_bias_topic, imu_qos, imu_bias_cbk);
            RCLCPP_INFO(this->get_logger(), "Fixposition IMU direct mode: topic=%s bias_topic=%s imu_time_offset=%.3f", imu_topic.c_str(), imu_bias_topic.c_str(), imu_time_offset);
        }
        else
        {
            sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 1000, imu_cbk);
            RCLCPP_INFO(this->get_logger(), "Standard IMU mode: topic=%s", imu_topic.c_str());
        }
        if (wheel_odom_en)
        {
            sub_twist_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(wheel_topic, 2000, twist_cbk);
            RCLCPP_INFO(this->get_logger(), "Wheel odom enabled, topic: %s, scale: %.3f", wheel_topic.c_str(), wheel_speed_scale);
        }
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        // The voxel-capped /Laser_map is a coarse global map, not a live feed, so it
        // publishes slowly (default 0.2 Hz). Period is configurable via publish.map_period_sec.
        double map_pub_period_sec = 5.0;
        this->get_parameter_or<double>("publish.map_period_sec", map_pub_period_sec, 5.0);
        if (map_pub_period_sec <= 0.0) map_pub_period_sec = 5.0;
        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(map_pub_period_sec * 1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        // Spin up the async PCD writer so the hot path never blocks on accumulation.
        if (pcd_save_en) startPcdWriter();

        // Spin up the off-thread ikd-tree map-update worker (option A). Idempotent.
        if (async_map_en) {
            startMapWorker();
            RCLCPP_INFO(this->get_logger(),
                        "[ASYNC_MAP] off-thread ikd-tree Add enabled (FLIO_ASYNC_MAP!=0).");
        }

        // Mapping pause/resume: latched (transient_local) subscription so we
        // pick up the most recent state even if PGO published before we
        // started.  The PGO node is the sole publisher; we only read.
        auto mapping_qos = rclcpp::QoS(1).transient_local();
        // transient_local durability is incompatible with intra-process comms, which
        // this node enables for the point-cloud zero-copy path. This latched gate is a
        // tiny, low-rate Bool from the (separate-process) PGO node, so force it onto the
        // normal inter-process path; node-level intra-process stays on for /lio/points.
        rclcpp::SubscriptionOptions mapping_sub_opts;
        mapping_sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
        mapping_enabled_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/lio_slam/mapping_enabled", mapping_qos,
            [this](std_msgs::msg::Bool::SharedPtr msg) {
                const bool prev = mapping_enabled.exchange(msg->data, std::memory_order_relaxed);
                if (prev != msg->data) {
                    RCLCPP_WARN(this->get_logger(),
                                "[MAPPING] %s — ikd-tree updates %s.",
                                msg->data ? "RESUMED" : "PAUSED",
                                msg->data ? "re-enabled" : "suspended");
                }
            },
            mapping_sub_opts);

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fout_jump.close();
        fclose(fp);
    }

private:
    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                if (!diag_first_no_point_logged) {
                    RCLCPP_WARN(this->get_logger(),
                                "[STARTUP][FAST_LIO] first no-point scan lidar_beg=%.3f",
                                Measures.lidar_beg_time);
                    diag_first_no_point_logged = true;
                }
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            if (!diag_first_valid_scan_logged) {
                RCLCPP_INFO(this->get_logger(),
                            "[STARTUP][FAST_LIO] first valid scan lidar_beg=%.3f points=%zu",
                            Measures.lidar_beg_time,
                            feats_undistort->size());
                diag_first_valid_scan_logged = true;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Restore the ikd-tree single-operator invariant: wait for the previous
             *   scan's off-thread map Add to finish before this scan touches the tree. ***/
            if (async_map_en) joinMapAdd();

            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
#ifdef USE_IVOX
            if(ikdtree.empty())
#else
            if(ikdtree.Root_Node == nullptr)
#endif
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    // --- Prior map preload (Scheme 1) ---
                    if (!prior_map_pcd_.empty())
                    {
                        // 1. Inject initial pose into EKF if provided
                        if (initial_pose_vec_.size() >= 6)
                        {
                            state_ikfom new_state = kf.get_x();
                            new_state.pos = Eigen::Vector3d(initial_pose_vec_[0],
                                                        initial_pose_vec_[1],
                                                        initial_pose_vec_[2]);
                            double roll_rad  = initial_pose_vec_[3] * M_PI / 180.0;
                            double pitch_rad = initial_pose_vec_[4] * M_PI / 180.0;
                            double yaw_rad   = initial_pose_vec_[5] * M_PI / 180.0;
                            Eigen::Quaterniond q_current(new_state.rot.matrix());
                            Eigen::Vector3d current_rpy = q_current.toRotationMatrix().eulerAngles(0, 1, 2);
                            double applied_roll = initial_pose_full_rpy_override_ ? roll_rad : current_rpy.x();
                            double applied_pitch = initial_pose_full_rpy_override_ ? pitch_rad : current_rpy.y();
                            Eigen::AngleAxisd rollAngle(applied_roll, Eigen::Vector3d::UnitX());
                            Eigen::AngleAxisd pitchAngle(applied_pitch, Eigen::Vector3d::UnitY());
                            Eigen::AngleAxisd yawAngle(yaw_rad, Eigen::Vector3d::UnitZ());
                            Eigen::Quaterniond q_init = yawAngle * pitchAngle * rollAngle;
                            new_state.rot = SO3(q_init);
                            kf.change_x(new_state);
                            state_point = kf.get_x();
                            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                            RCLCPP_INFO(this->get_logger(),
                                "Injected initial pose: [%.3f, %.3f, %.3f] yaw_deg=%.1f full_rpy_override=%s",
                                initial_pose_vec_[0], initial_pose_vec_[1], initial_pose_vec_[2],
                                initial_pose_vec_[5], initial_pose_full_rpy_override_ ? "true" : "false");
                        }

                        // 2. Load prior PCD
                        PointCloudXYZI::Ptr prior_cloud(new PointCloudXYZI());
                        if (pcl::io::loadPCDFile(prior_map_pcd_, *prior_cloud) == 0)
                        {
                            RCLCPP_INFO(this->get_logger(),
                                "Loaded prior map: %s (%zu points)",
                                prior_map_pcd_.c_str(), prior_cloud->size());

                            // AABB self-check: helps diagnose wrong-frame PCDs
                            // (e.g. a UTM map loaded where ENU was expected).
                            // center→origin > ~10 km strongly suggests a global frame.
                            if (!prior_cloud->empty()) {
                                float xmin = std::numeric_limits<float>::max();
                                float ymin = xmin;
                                float zmin = xmin;
                                float xmax = -xmin;
                                float ymax = -xmin;
                                float zmax = -xmin;
                                for (const auto & pt : prior_cloud->points) {
                                    xmin = std::min(xmin, pt.x);
                                    ymin = std::min(ymin, pt.y);
                                    zmin = std::min(zmin, pt.z);
                                    xmax = std::max(xmax, pt.x);
                                    ymax = std::max(ymax, pt.y);
                                    zmax = std::max(zmax, pt.z);
                                }
                                const double cx = 0.5 * (xmin + xmax);
                                const double cy = 0.5 * (ymin + ymax);
                                const double cz = 0.5 * (zmin + zmax);
                                const double dist_to_origin =
                                    std::sqrt(cx * cx + cy * cy + cz * cz);
                                RCLCPP_INFO(this->get_logger(),
                                    "Prior map bbox: x=[%.2f..%.2f] y=[%.2f..%.2f] "
                                    "z=[%.2f..%.2f] center→origin=%.1f m%s",
                                    xmin, xmax, ymin, ymax, zmin, zmax, dist_to_origin,
                                    dist_to_origin > 10000.0
                                      ? "  (large offset — likely UTM/global frame, "
                                        "check coordinate system!)"
                                      : "");
                            }

                            // 3. Downsample prior map
                            pcl::VoxelGrid<PointType> voxel;
                            voxel.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
                            voxel.setInputCloud(prior_cloud);
                            PointCloudXYZI::Ptr prior_ds(new PointCloudXYZI());
                            voxel.filter(*prior_ds);
                            RCLCPP_INFO(this->get_logger(),
                                "Prior map downsampled: %zu → %zu points",
                                prior_cloud->size(), prior_ds->size());

                            // 4. Transform current scan to world frame
                            ikdtree.set_downsample_param(filter_size_map_min);
                            feats_down_world->resize(feats_down_size);
                            for(int i = 0; i < feats_down_size; i++)
                            {
                                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                            }

                            // 5. Merge prior + current scan → build tree
                            PointCloudXYZI::Ptr combined(new PointCloudXYZI());
                            *combined = *prior_ds;
                            *combined += *feats_down_world;
                            ikdtree.Build(combined->points);
                            Localmap_Initialized = false;

                            RCLCPP_INFO(this->get_logger(),
                                "ikd-tree built from prior map + current scan (%zu + %d = %zu points)",
                                prior_ds->size(), feats_down_size, combined->size());
                        }
                        else
                        {
                            RCLCPP_ERROR(this->get_logger(),
                                "Failed to load prior map PCD: %s, falling back to normal init",
                                prior_map_pcd_.c_str());
                            // Fall back: build from current scan only
                            ikdtree.set_downsample_param(filter_size_map_min);
                            feats_down_world->resize(feats_down_size);
                            for(int i = 0; i < feats_down_size; i++)
                            {
                                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                            }
                            ikdtree.Build(feats_down_world->points);
                        }
                        prior_map_pcd_.clear();  // Prevent re-execution
                    }
                    else
                    {
                        // Normal init: build from current scan only
                        ikdtree.set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);
                        for(int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }
                        ikdtree.Build(feats_down_world->points);
                    }
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            if (runtime_pos_log) {
                fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< "\n";
            }

#ifndef USE_IVOX
            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }
#endif

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            double t_pre_icp = omp_get_wtime();

            /*** iterated state estimation ***/
            const auto state_before_update = state_point;
            const V3D euler_before_update = SO3ToEuler(state_before_update.rot);
            const V3D pos_before_update = state_before_update.pos;
            double t_update_start = omp_get_wtime();
            // [ikd-profile] snapshot background/inline rebuild work spanning iEKF + map_incre
            const uint64_t prof_rb_us0  = g_ikd_rebuild_us.load(std::memory_order_relaxed);
            const uint64_t prof_rb_cnt0 = g_ikd_rebuild_count.load(std::memory_order_relaxed);
            const uint64_t prof_inl_us0 = g_ikd_inline_rebuild_us.load(std::memory_order_relaxed);
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

            state_point = kf.get_x();

            /*** Divergence guard (P1) — two independent divergence signals:
             *   (1) effective-correspondence collapse streak: effct_feat_num (the
             *       matched-plane count from the final iEKF iteration) staying low
             *       for several scans means the scan no longer overlaps the map
             *       (starvation / featureless) — the slow-drift onset.
             *   (2) velocity runaway: a ground vehicle cannot physically exceed a
             *       few m/s, so a body speed above divergence_guard_max_speed is
             *       unambiguous IMU-dead-reckoning blow-up — catches the fast mode
             *       (big IMU-integration jumps after dropped scans) that signal (1)
             *       can miss because the post-gap scan may still match.
             *   Either signal arms the degraded response below (force planar-Z,
             *   clamp the speed, freeze the map). Both are inert in healthy
             *   operation, so the guard is accuracy-neutral when undisturbed. ***/
            flio_map_frozen = false;
            flio_degraded_odom = false;
            bool flio_vel_runaway = false;
            if (divergence_guard_en)
            {
                if (effct_feat_num < divergence_guard_min_eff) ++consec_low_eff;
                else                                           consec_low_eff = 0;
                flio_vel_runaway = (state_point.vel.norm() > divergence_guard_max_speed);
            }
            const bool flio_in_degraded =
                divergence_guard_en &&
                (consec_low_eff >= divergence_guard_streak || flio_vel_runaway);

            /*** A1 gravity-alignment leveling prior — the ROOT-CAUSE fix for the
             *   iVox pitch-coupled world-Z drift. Supplies the ABSOLUTE pitch/roll
             *   reference the body-vz planar constraint below structurally lacks.
             *   When LiDAR Z is degenerate, and only when the IMU is in low-linear-
             *   acceleration motion (|‖a‖−g| small, so specific force ≈ gravity) and
             *   not turning hard (lever-arm safety), the bias-free accelerometer
             *   direction equals world-up expressed in the body frame. Pull the
             *   grav-state up-axis onto it with a soft pseudo-measurement. Constrains
             *   roll/pitch ONLY: yaw lies in the measurement null space (skew(g_body)
             *   sends a yaw-axis δθ to 0), so the LiDAR-excellent heading is untouched.
             *   h(x)=Rᵀ·u_world, right-perturbation R=R̂·Exp(δθ) ⇒ ∂h/∂δθ=skew(g_body),
             *   the same form as the planar/wheel updates. ***/
            if (gravity_align_en && !Measures.imu.empty() &&
                (pos_obs_z_weak > gravity_align_z_weak_thresh || flio_in_degraded))
            {
                const auto &imu_last = Measures.imu.back();
                V3D a_raw(imu_last->linear_acceleration.x,
                          imu_last->linear_acceleration.y,
                          imu_last->linear_acceleration.z);
                V3D w_raw(imu_last->angular_velocity.x,
                          imu_last->angular_velocity.y,
                          imu_last->angular_velocity.z);
                const double g_raw  = p_imu->gravity_norm();   // gravity magnitude, raw units
                state_ikfom st = kf.get_x();
                M3D Rw = st.rot.toRotationMatrix();
                // Kinematic-acceleration compensation. The measured specific force is
                // f = a_kinematic − g. On the open-area row-end turns that coincide with
                // weak-Z, a_kinematic is dominated by centripetal ω×v_body, which tilts
                // the raw accelerometer (mostly in ROLL) and, left uncorrected, drove a
                // ~5 m upward Z drift. Subtract it (transport theorem; tangential dv/dt
                // neglected, caught by the magnitude gate below) to recover clean gravity
                // even mid-turn. Scaled into the IMU's accel units so it is g-unit safe.
                V3D omega  = w_raw - V3D(st.bg[0], st.bg[1], st.bg[2]);  // true body rate
                V3D v_body = Rw.transpose() * st.vel;                    // velocity in IMU frame
                V3D a_kin  = omega.cross(v_body) * (g_raw / G_m_s2);     // centripetal, IMU units
                V3D a_grav = a_raw - a_kin;                             // specific force − motion
                const double a_norm = a_grav.norm();
                // Validity gate: compensated magnitude ≈ gravity (rejects residual
                // tangential accel) AND not spinning absurdly fast (ω×v single-sample
                // approximation breaks down; magnitude gate is the primary guard).
                if (g_raw > 1e-3 && a_norm > 1e-6 &&
                    std::abs(a_norm - g_raw) <= gravity_align_accel_tol * g_raw &&
                    w_raw.norm() <= gravity_align_gyro_tol)
                {
                    V3D grav_w(st.grav[0], st.grav[1], st.grav[2]);  // ≈ [0,0,-g], points DOWN
                    V3D u_world  = -grav_w.normalized();             // world up (unit)
                    V3D g_body   = Rw.transpose() * u_world;         // predicted up in body (unit)
                    V3D m_body   = a_grav / a_norm;                  // measured up in body (unit)
                    V3D residual = m_body - g_body;                  // z − h(x)
                    Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
                    H.block<3,3>(0, 3) = skew_sym_mat(g_body);       // ∂h/∂(rot); yaw in null space
                    const double n2 = gravity_align_noise * gravity_align_noise;
                    Eigen::Vector3d R_diag(n2, n2, n2);
                    if (degeneracy_debug)
                    {
                        // Leveling error the prior sees, BEFORE applying it: signed
                        // tilt of measured-up vs predicted-up, decomposed so we can see
                        // which way (and how hard) it will rotate pitch/roll.
                        V3D tilt = g_body.cross(m_body);   // axis*sin(angle), body frame
                        const double ang_deg = std::asin(std::min(1.0, tilt.norm())) * 57.2958;
                        V3D eul_before = SO3ToEuler(st.rot);
                        kf.update_simple(H, residual, R_diag);
                        state_point = kf.get_x();
                        V3D eul_after = SO3ToEuler(state_point.rot);
                        V3D deul = eul_after - eul_before;
                        std::cerr << "[GALIGN] z_weak=" << pos_obs_z_weak
                                  << " degr=" << flio_in_degraded
                                  << " tilt_deg=" << ang_deg
                                  << " tiltAxisB=[" << tilt.x() << "," << tilt.y() << "," << tilt.z() << "]"
                                  << " dRPY_deg=[" << deul.x()*57.2958 << "," << deul.y()*57.2958 << "," << deul.z()*57.2958 << "]"
                                  << " posZ=" << state_point.pos[2] << "m" << std::endl;
                    }
                    else
                    {
                        kf.update_simple(H, residual, R_diag);
                        state_point = kf.get_x();
                    }
                }
            }

            /*** A'-2 planar-motion soft constraint: pseudo-measurement that the
             *   BODY-frame vertical velocity is ~0 (a ground-robot non-holonomic
             *   prior, same mechanism as the wheel-odom update below). Applied only
             *   when LiDAR Z is degenerate (pos_obs_z_weak high), so it is inert
             *   during normal operation. Body-frame → it does NOT fight slope
             *   motion: world-frame vertical velocity on a ramp is produced by the
             *   robot's pitch (rotation state), not by body-frame vz. This supplies
             *   the missing vertical information that the degenerate iEKF otherwise
             *   dumps into the accel bias, flying the state off (bag2 km-runaway). ***/
            if (planar_constraint_en &&
                (pos_obs_z_weak > planar_constraint_z_weak_thresh || flio_in_degraded))
            {
                state_ikfom st = kf.get_x();
                M3D Rw = st.rot.toRotationMatrix();
                V3D v_body = Rw.transpose() * st.vel;   // velocity in IMU/body frame
                // Measurement: body velocity = 0, but only the vertical (z) row is
                // enforced — x/y get huge noise so forward/lateral motion is untouched.
                V3D residual = -v_body;
                Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
                H.block<3,3>(0, 3)  = skew_sym_mat(v_body);   // dh/d(rot)
                H.block<3,3>(0, 12) = Rw.transpose();          // dh/d(vel)
                Eigen::Vector3d R_diag(1e6, 1e6, planar_constraint_noise * planar_constraint_noise);
                kf.update_simple(H, residual, R_diag);
                state_point = kf.get_x();
            }

            /*** Divergence guard (P1) — bound the runaway while degraded. A ground
             *   vehicle cannot exceed a few m/s; a larger body speed here is
             *   IMU-dead-reckoning blow-up, so clamp it (direction preserved) to keep
             *   position growth linear and recoverable instead of exponential. Freeze
             *   the map so dead-reckoned scans do not corrupt it, and flag the
             *   odometry degraded so downstream (PGO) down-weights these poses. ***/
            if (flio_in_degraded)
            {
                state_ikfom st = kf.get_x();
                const double spd = st.vel.norm();
                if (spd > divergence_guard_max_speed)
                {
                    st.vel *= divergence_guard_max_speed / spd;
                    kf.change_x(st);
                    state_point = kf.get_x();
                }
                flio_map_frozen = true;
                flio_degraded_odom = true;
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "[DIVERGENCE-GUARD] LiDAR starved %d scans (effct=%d): planar-Z pinned, "
                    "speed clamped, map frozen; posZ=%.1fm spd=%.2fm/s",
                    consec_low_eff, effct_feat_num, state_point.pos[2], state_point.vel.norm());
            }

            /*** Optional degeneracy diagnostics ***/
            if (degeneracy_debug)
            {
                const double lidar_correction = (state_point.pos - pos_before_update).norm();
                const double body_speed = state_point.vel.norm();
                const double dvel = (state_point.vel - state_before_update.vel).norm();
                const double dba  = (state_point.ba  - state_before_update.ba ).norm();
                const double dbg  = (state_point.bg  - state_before_update.bg ).norm();
                if (lidar_correction > 0.2 || body_speed > 1.5 || dvel > 0.3 ||
                    dba > 0.05 || dbg > 0.01 || effct_feat_num < 200 || pos_obs_z_weak > 0.7)
                {
                    std::cerr << "[DEGEN] effct=" << effct_feat_num
                              << " corr=" << lidar_correction << "m speed=" << body_speed
                              << " dvel=" << dvel << " dba=" << dba << " dbg=" << dbg
                              << " z_weak=" << pos_obs_z_weak << " posZ=" << state_point.pos[2] << "m"
                              << " gnd_boost=" << ground_constraint_boosted << std::endl;
                }
            }

            /*** Wheel velocity update (after ICP, constrains along-track drift) ***/
            if (wheel_odom_en)
            {
                Eigen::Vector3d twist_v;
                bool have_twist = false;
                {
                    std::lock_guard<std::mutex> lk(mtx_twist);
                    if (!twist_buffer.empty()) {
                        twist_v = twist_buffer.back();
                        twist_buffer.clear();
                        have_twist = true;
                    }
                }
                if (have_twist)
                {
                    state_ikfom st = kf.get_x();
                    M3D Rw = st.rot.toRotationMatrix();    // R_{world←IMU}
                    M3D Re_T = R_robot_to_imu.transpose();  // R_{robot←IMU}
                    V3D v_imu = Rw.transpose() * st.vel;    // vel in IMU frame

                    // Predicted: h(x) = R_ext^T * R_w^T * vel (robot frame)
                    V3D h_pred = Re_T * v_imu;
                    // Measured: z = [vx*scale, 0, 0] (non-holonomic: vy=vz=0)
                    V3D z_meas(twist_v(0) * wheel_speed_scale, 0.0, 0.0);
                    V3D residual = z_meas - h_pred;

                    // Jacobian H (3×23): non-zero at rot(3-5) and vel(12-14)
                    Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
                    H.block<3,3>(0, 3)  = Re_T * skew_sym_mat(v_imu);  // dh/d(rot)
                    H.block<3,3>(0, 12) = Re_T * Rw.transpose();        // dh/d(vel)

                    // Diagonal noise variances
                    Eigen::Vector3d R_diag(
                        wheel_vel_noise_vx * wheel_vel_noise_vx,
                        wheel_vel_noise_vy * wheel_vel_noise_vy,
                        wheel_vel_noise_vz * wheel_vel_noise_vz);

                    kf.update_simple(H, residual, R_diag);
                    state_point = kf.get_x();  // re-read corrected state
                }
            }

            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            // Mapping pause gate: when operator has disabled mapping (via
            // /lio_slam/set_mapping_enabled in the PGO node), skip ikd-tree
            // growth so the prior map isn't polluted by stationary frames or
            // teleport segments (elevator).  EKF and odometry publishing
            // above continue normally.
            if (mapping_enabled.load(std::memory_order_relaxed) && !flio_map_frozen) {
                map_incremental();
            }
            t5 = omp_get_wtime();

            // [ikd-profile] per-scan split: iEKF (search-bound) vs map_incre Add vs transform
            // loop, correlated with concurrent background-rebuild work + inline rebuild.
            if (ikd_profile) {
                const double iekf_ms = (t_update_end - t_update_start) * 1000.0;
                // Under async, the Add runs off-thread: report its last completed wall time
                // (g_async_add_us, == previous scan's Add) and the join-wait this scan paid
                // (the residual Add cost the inter-scan gap could NOT hide). loop_ms is then
                // just the on-thread decision loop + dispatch.
                const double add_ms  = async_map_en ? (g_async_add_us.load(std::memory_order_relaxed) / 1000.0)
                                                    : (kdtree_incremental_time * 1000.0);
                const double join_ms = async_map_en ? g_map_join_wait_ms : 0.0;
                const double loop_ms = async_map_en ? ((t5 - t3) * 1000.0)
                                                    : ((t5 - t3) * 1000.0 - add_ms);
                const uint64_t d_rb_us  = g_ikd_rebuild_us.load(std::memory_order_relaxed)        - prof_rb_us0;
                const uint64_t d_rb_cnt = g_ikd_rebuild_count.load(std::memory_order_relaxed)     - prof_rb_cnt0;
                const uint64_t d_inl_us = g_ikd_inline_rebuild_us.load(std::memory_order_relaxed) - prof_inl_us0;
                // tree= uses kdtree_size_st (captured at scan start, worker idle) — calling
                // ikdtree.size() here would race the worker's in-flight Add under async.
                printf("[IKDPROF] tree=%d feats=%d iekf_ms=%.1f add_ms=%.1f join_ms=%.1f loop_ms=%.1f | "
                       "rb_dcnt=%llu rb_dus_ms=%.1f inl_dus_ms=%.1f rb_active=%d "
                       "rb_maxus_ms=%.1f rb_maxsz=%llu rootreb=%llu\n",
                       kdtree_size_st, feats_down_size, iekf_ms, add_ms, join_ms, loop_ms,
                       (unsigned long long)d_rb_cnt, d_rb_us / 1000.0, d_inl_us / 1000.0,
                       (int)g_ikd_rebuild_active.load(std::memory_order_relaxed),
                       g_ikd_rebuild_max_us.load(std::memory_order_relaxed) / 1000.0,
                       (unsigned long long)g_ikd_rebuild_max_size.load(std::memory_order_relaxed),
                       (unsigned long long)g_ikd_root_rebuild_count.load(std::memory_order_relaxed));
            }

            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                // Under async the worker may be mid-Add; reading ikdtree.size() here would
                // race it. Estimate from the scan-start size + this scan's add count.
                kdtree_size_end = async_map_en ? (kdtree_size_st + add_point_size) : ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                if ((t3 - t1) > 0.05) {
                    printf("[ SPIKE ]: t1_to_pre_icp: %0.3f  icp: %0.3f  publish: %0.3f  total(t3-t1): %0.3f  feats: %d  tree: %d\n",
                           (t_pre_icp-t1)*1000, (t_update_end-t_update_start)*1000, (t3-t_update_end)*1000, (t3-t1)*1000, feats_down_size, kdtree_size_st);
                }
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                if (time_log_counter % 500 == 0 || time_log_counter == 1) {
                    printf("[ extrinsic ]: T_L_I = [%0.5f, %0.5f, %0.5f]  R_euler = [%0.3f, %0.3f, %0.3f] deg  bg = [%0.6f, %0.6f, %0.6f]  ba = [%0.6f, %0.6f, %0.6f]  grav = [%0.5f, %0.5f, %0.5f]\n",
                        state_point.offset_T_L_I(0), state_point.offset_T_L_I(1), state_point.offset_T_L_I(2),
                        ext_euler(0), ext_euler(1), ext_euler(2),
                        state_point.bg(0), state_point.bg(1), state_point.bg(2),
                        state_point.ba(0), state_point.ba(1), state_point.ba(2),
                        state_point.grav[0], state_point.grav[1], state_point.grav[2]);
                }
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<"\n";
                fout_jump << setw(20) << Measures.lidar_beg_time - first_lidar_time
                          << " yaw_before_deg " << euler_before_update(2)
                          << " yaw_after_deg " << euler_cur(2)
                          << " yaw_step_deg " << (euler_cur(2) - euler_before_update(2))
                          << " pos_step_xy " << (state_point.pos.head<2>() - pos_before_update.head<2>()).norm()
                          << " feats_down " << feats_down_size
                          << " effct_feat_num " << effct_feat_num
                          << " res_mean_last " << res_mean_last
                          << " solve_H_ms " << solve_H_time * 1000.0
                          << " icp_ms " << (t_update_end - t_update_start) * 1000.0
                          << "\n";
                fout_jump.flush();
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        // publish_map() flattens the ikd-tree (a tree read). This 1 Hz timer shares the
        // single executor thread with timer_callback, so it can fire BETWEEN scans while
        // the off-thread Add is in flight — join first to keep the single-operator invariant.
        if (map_pub_en) {
            if (async_map_en) joinMapAdd();
            publish_map(pubLaserCloudMap_);
        }
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImu>::SharedPtr sub_imu_fpa_;
    rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImubias>::SharedPtr sub_imu_bias_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    rclcpp::Subscription<shm_msgs::msg::PointCloud8mAndPose>::SharedPtr sub_pcl_shm_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mapping_enabled_sub_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg, fout_jump;

    std::string prior_map_pcd_;
    std::vector<double> initial_pose_vec_;
    bool initial_pose_full_rpy_override_ = false;
};

#ifndef FASTLIO_AS_COMPONENT
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);
    signal(SIGTERM, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();
    // Drain any in-flight off-thread ikd-tree Add and join the worker before the final
    // map read/save, so the tree is quiescent and the worker thread exits cleanly.
    stopMapWorker();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    // Drain the async writer queue and join before reading pcl_wait_save, so the final
    // map contains every enqueued scan and there is no race with the writer thread.
    stopPcdWriter();
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
#endif  // FASTLIO_AS_COMPONENT

#ifdef FASTLIO_AS_COMPONENT
// Built as a SHARED rclcpp component (see CMakeLists) so LaserMappingNode can load
// into the shared input_adapters MT container and receive /lio/points via an
// intra-process pointer move instead of an 8 MB cross-process DDS copy. main() is
// excluded in this build; the container owns the spin loop. The node's many
// file-scope globals are safe here because exactly ONE LaserMappingNode is loaded.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(LaserMappingNode)
#endif  // FASTLIO_AS_COMPONENT

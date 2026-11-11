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
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <pcl/filters/voxel_grid.h>

#include "delayed_map_insertion.hpp"
// PCL < 1.11 (focal/Foxy ships 1.10) does not precompile
// VoxelGrid<PointXYZINormal>; pull the impl in so this TU instantiates it.
#if PCL_VERSION_COMPARE(<, 1, 11, 0)
#include <pcl/filters/impl/voxel_grid.hpp>
#endif
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Python.h>
#include <algorithm>
#include <array>
#ifdef FASTLIO_TEST_HOOKS
#include "laser_mapping_test_hooks.hpp"
#endif
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fast_lio_interfaces/msg/frontend_observation.hpp>
#include <fast_lio_interfaces/msg/reference_health.hpp>
#include <fixposition_driver_msgs/msg/fpa_imu.hpp>
#include <fixposition_driver_msgs/msg/fpa_imubias.hpp>
#include <fstream>
#include <ikd-Tree/ikd_Tree.h>
#include <iomanip>
#include <ivox/ivox.hpp>  // option B: drop-in voxel-grid map backend (compile with -DUSE_IVOX=ON)
#include <limits>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <math.h>
#include <memory>
#include <mutex>
#include <odom_covariance_contract.hpp>
#include <omp.h>
#include <shm_msgs/msg/point_cloud8m_and_pose.hpp>
#include <so3_math.h>
#include <sys/random.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>

#include "IMU_Processing.hpp"
#include "adaptive_downsample.hpp"
#include "anchor_trace.hpp"
#include "attitude_hold.hpp"
#include "cached_plane.hpp"
#include "correspondence_compaction.hpp"
#include "correspondence_gate.hpp"
#include "covariance_reset.hpp"
#include "engulf_release.hpp"
#include "frontend_measurement.hpp"
#include "frontend_observation.hpp"
#include "frontend_parameters_ros.hpp"
#include "gravity_align_kinematics.hpp"
#include "imu_backlog_policy.hpp"
#include "imu_gap_prior.hpp"
#include "lost_recovery_policy.hpp"
#include "map_backend_reference.hpp"
#include "normal_noise_floor.hpp"
#include "parked_hold.hpp"
#include "preprocess.h"
#include "prior_map_load.hpp"
#include "reanchor_gate.hpp"
#include "recovery_view_match.hpp"
#include "reference_health_contract.hpp"
#include "rp_consider.hpp"
#include "runaway_watchdog.hpp"
#include "static_evidence.hpp"
#include "stationary_anchor_policy.hpp"
#include "timing_log_ring.hpp"
#include "voxel_downsample.hpp"
#include "wheel_speed_input.hpp"
#include "zupt_policy.hpp"

#define INIT_TIME (0.1)
#define MAXN (fast_lio::kTimingLogCapacity)
#define PUBFRAME_PERIOD (20)

struct FileCloser
{
  void operator()(std::FILE * file) const noexcept
  {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FileHandle = std::unique_ptr<std::FILE, FileCloser>;

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
// A ring over scan_count and time_log_counter, which outlive these arrays: every store goes through
// fast_lio::timingLogSlot() and the dump reads back through timingLogRowSlot() (timing_log_ring.hpp).
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN],
  s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double solve_const_H_time = 0;
int ekf_iters = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false;
bool ikd_profile = false;  // [ikd-profile] per-scan rebuild-split line, gated by env FLIO_IKD_PROFILE
// Mapping pause/resume gate. Owned by PGO node (lio_slam_fastLioPGO publishes
// /lio_slam/mapping_enabled with transient_local QoS); we mirror it here.
// When false: skip ikd-tree updates and skip pcl_wait_save accumulation, so the
// saved PCD doesn't contain frames captured while operator had mapping paused.
// EKF prediction, ICP and /Odometry publishing keep running so pose continuity
// is preserved.
std::atomic<bool> mapping_enabled{true};
/**************************/

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
string map_file_path;

// Frame ids stamped on everything this node publishes. The odom frame is the
// origin of the local map: fixed at initialization, gravity-aligned, and free to
// drift. Upstream LOAM called it `camera_init` for a frame that holds no camera;
// a deployment still running a back end that expects that name must set this
// parameter back to it, because the PGO's `lidar_odom_frame` and this string are
// the two ends of one TF edge and the tree severs silently if they disagree.
string odom_frame_id = "lidar_odom";
string body_frame_id = "body";

double res_mean_last = 0.05;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double filter_size_map_min = 0;
double cube_len = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool degeneracy_debug = false;  // verbose per-frame [DEGEN] diagnostics

// Front-end health side-channel (Plan C): per-scan along-motion observability and
// quality metrics published on /Odometry twist.covariance for the PGO
// between-factor gate. Layout contract: LIO-SLAM core pgo_factor_kernels.hpp
// (kFeHealth*Cell). obs_along = d·(pos_obs_M·d)/effct_feat_num with d = the
// world-frame velocity direction: mean cos² between the effective plane normals
// and the travel direction (1 = every plane faces the motion, 0 = all planes
// parallel to it → along-track unobservable). -1 below the speed gate (direction
// undefined), 0 when a moving scan has no effective points (worst case).
bool health_pub_en = true;
constexpr double kHealthSpeedGate = 0.2;  // m/s, same gate as the [SCAN] trace obs_along
double scan_obs_along = -1.0;
double scan_body_speed = 0.0;

// [v5] Whether the last scan's levelling update ran at the degraded/held strength;
// published in the health flags so a pose-graph consumer can leave those
// keyframes' roll/pitch unpinned.
bool gravity_align_degraded_active = false;
// [v6] What the levelling compensation (FrontendParams::gravity_align_lin_accel_comp) keeps between scans.
bool gravity_align_vel_overwritten = false;   // the state velocity was assigned outside an update since the last scan
double gravity_align_prev_t = -1.0;           // previous scan's lidar_end_time (-1 = none yet)
V3D gravity_align_prev_v_body = V3D::Zero();  // previous scan's body-frame velocity (posterior)
double gravity_align_prev_var = 0.0;          // previous scan's velocity covariance trace
bool gravity_align_prev_edge_valid = false;
// [diag] the previous scan's final state, so the IMU propagation can be logged as its own channel
state_ikfom kalman_last_final_state;
bool kalman_last_final_valid = false;
std::deque<std::pair<double, double>> gravity_align_leg_hist;  // (stamp, forward speed) under mtx_twist
std::deque<std::pair<double, V3D>> gravity_align_gyro_hist;    // (stamp, raw body rate) scan-thread only
bool imu_init_degraded_warned_ = false;
struct GravWinAgg
{
  double t;
  Eigen::Vector3d sum;
  int n;
  double max_w;
  fast_lio::LevellingWindowScan edge;  // [v6] body velocity at both ends of this scan's IMU span
};
std::deque<GravWinAgg> grav_win;  // scan-thread only
double gravity_align_last_trig = -1e18;
// The correspondence dump (FrontendParams::corr_dump_en describes it): its files and its per-scan state.
std::ofstream fout_corr, fout_corr_iter, fout_corr_scan;
bool corr_dump_scan_active = false;  // this scan lies inside a dump window (set before the update)
int corr_dump_iter = 0;              // measurement-model evaluations dumped so far on this scan
// Attitude hold (FrontendParams::attitude_hold is the policy): this scan's verdicts and the telemetry.
bool lidar_attitude_hold_active = false;   // this scan's scene-rule verdict; read by the measurement model
bool lidar_attitude_hold_allowed = false;  // release conditions satisfied this scan (starvation, age, holes)
int lidar_attitude_hold_scans = 0;             // telemetry
double lidar_attitude_hold_since = -1.0;       // lidar_end_time when the current hold began (-1 = off)

// Engulfment latch; FrontendParams documents the divergence guard and the engulfment release.
int consec_engulf_clear = 0;                  // running healthy-scan streak while latched
bool engulf_latched = false;                  // sticky engulfed state (entered, not yet released)
double engulf_latch_time = -1.0;              // lidar_end_time of the scan that latched (blind duration at release)
// Release policy (engulf_release.hpp), its velocity side: what velocity leaves the blind stretch. The default
// reproduces the legacy zero velocity; FrontendParams::release_inflate holds the covariance side.
fast_lio::ReleaseVelocityParams release_vel_params;
double engulf_release_time = -1.0;  // lidar_end_time of the last engulfment release
// Consider update for roll/pitch (FrontendParams::rp_consider is the policy): verdict and telemetry.
bool rp_consider_active = false;  // this scan's verdict
double rp_consider_since = -1.0;  // start of the current considering stretch
int rp_consider_scans = 0;        // scans considered so far (telemetry)
int rp_consider_indefinite = 0;   // considered scans whose prior P had a negative eigenvalue (telemetry)
// Scan-hole guard state (FrontendParams::guard_hole_en describes the guard).
double hole_last_scan_end = -1.0;
V3D hole_vel_pre = V3D::Zero();
V3D hole_pos_pre = V3D::Zero();
// Static hold (ZUPT): while degraded there are no correspondences to bound the
// iEKF, so it runs on IMU propagation alone. Measured on dog 2 (2026-09-04):
// 9 s of effct=0 cost 3.9 m of Z and 1.4 m/s of phantom velocity, and the front
// end never came back — the map is unfrozen at release and then follows the
// wrong pose. When the IMU and the wheel speed both say the platform is parked
// there is nothing to dead-reckon: hold the last map-confirmed pose until
// correspondences return. Evidence is deliberately independent of the pose
// estimate, which is the thing under suspicion.
// The hold is switched by FrontendParams::zupt_en.
double zupt_gyro_thresh = 0.05;         // rad/s; max |omega| over the scan's IMU window
double zupt_acc_span_ratio = 0.02;      // (max-min)|a| / mean|a| — free of the IMU's unit scale
double zupt_wheel_speed_thresh = 0.05;  // m/s; wheel speed below this reads as parked
double zupt_wheel_max_age = 0.5;        // s; older wheel samples are not evidence
bool zupt_require_wheel = false;        // true = hold only when a fresh wheel sample agrees
V3D zupt_anchor_pos = V3D::Zero();      // pose of the current candidate or anchor (stationary_anchor.state says which)
SO3 zupt_anchor_rot;
bool zupt_active = false;
int zupt_hold_scans = 0;
// Anchor lifecycle (stationary_anchor_policy.hpp): an anchor is the reference of one continuous stationary
// episode, and whatever ends the episode drops it together with both holds. A later anchor is a new one.
fast_lio::StationaryAnchor stationary_anchor;
fast_lio::StationaryAnchorParams stationary_anchor_params;
bool anchor_usable = false;        // this scan's anchor step found a usable anchor; cleared at scan start and on drops
bool anchor_ever_created = false;  // an anchor has been created in this process
// Recovery trust, the platform policy (on when reanchor_total_timeout_sec > 0): how long the front end may run
// without support before its position is not trusted again in this process. recovery_state also holds the first
// local reference: the constructor configures the start from prior_map_pcd before any input, the normal initial
// map build records the reference, and a map reset, a replaced reference or a clock fault revokes a cold start.
// Every access runs in timer_callback or the IMU callback (checkAnchorEvidenceOnImu), serial only while both stay
// in the node's default mutually exclusive callback group; no other thread touches it.
fast_lio::RecoveryParams recovery_params;
fast_lio::RecoveryState recovery_state;
fast_lio::LostRecoveryParams lost_recovery_params;
fast_lio::LostRecoveryState lost_recovery;
bool lost_recovery_map_ready = false;
std::uint64_t lifecycle_scan_seq = 0;  // one per scan the scan path starts on: a matching run is consecutive in it
bool anchor_scan_evaluated = false;    // the scan being processed reached the anchor step
bool reanchor_lost_this_scan = false;  // the gate declared the scan being processed lost, whatever re-armed it since
bool anchor_idle_path = false;         // inside the check every IMU sample the front end takes runs
bool input_epoch_handled = false;
// Parked hold state (FrontendParams::parked_hold is the policy).
fast_lio::ParkedHoldState parked_hold_state;
// Stationary-anchor trace (anchor_trace.hpp specifies the format). Observation only: the estimator never reads
// any of it back. Transition records and counters are always on; scan records need zupt_anchor_trace_en.
constexpr const char * kAnchorTraceLogger = "laser_mapping.zupt_anchor";
bool zupt_anchor_trace_en = false;
fast_lio::AnchorTraceCounters anchor_trace_counters;
fast_lio::MonotonicSequence anchor_event_seq;
fast_lio::MonotonicSequence anchor_scan_seq;
fast_lio::AnchorScanRecord anchor_scan_record;  // the scan being processed
bool anchor_scan_record_open = false;
bool anchor_trace_body_was_static = false;  // the previous evaluated verdict, for motion edges
bool anchor_trace_gap_run = false;          // inside a run of scans with IMU coverage gaps
bool anchor_trace_bridge_run = false;       // inside a run of scans with bridged IMU gaps
double input_epoch_fault_stamp = std::numeric_limits<double>::quiet_NaN();  // under mtx_buffer
double last_twist_stamp = -1.0;       // newest wheel-speed sample time (s); -1 = none yet
double last_twist_speed = 0.0;        // |v| of the newest wheel sample (m/s)
V3D last_twist_v_body = V3D::Zero();  // the newest finite wheel sample, body frame (m/s), under mtx_twist
// Re-anchor gate: the guard used to unfreeze the map the moment the scene
// looked healthy again, which is where a recoverable blind stretch became a
// permanent runaway — the pose error survived the release, the map was rebuilt
// around it, and residuals stayed small while the map walked off with the
// estimate. Verify against the still-frozen map first (reanchor_gate.hpp).
fast_lio::ReanchorParams reanchor_params;
fast_lio::ReanchorGate reanchor_gate;
// What a healthy scan looks like right now: running medians of the residual and the correspondence count of
// the scans the gate called healthy, the reference of the relative verification (reanchor_max_res_ratio,
// reanchor_min_eff_ratio). A registration from a wrong prior passes the absolute 0.10 m ceiling at 1.2-2.2x
// the healthy residual (m20 0826 after the data hole).
constexpr std::size_t kReanchorHealthyMinScans = 20;
fast_lio::HealthyScanWindow reanchor_healthy_res;
fast_lio::HealthyScanWindow reanchor_healthy_eff;
// Runaway watchdog state (FrontendParams::runaway is the policy and runaway_on_trip its action).
fast_lio::RunawayState runaway_state;
double scan_far_frac = 1.0;  // per-scan telemetry: fraction of raw points beyond far_range

int consec_engulf = 0;             // running engulfed-scan streak
int consec_low_eff = 0;            // running degenerate-scan streak counter
bool flio_map_frozen = false;      // degraded: skip map_incremental (no garbage)
bool required_prior_lost = false;  // scan-owner latch; only a fresh activation can reinstall a required prior
bool flio_degraded_odom = false;   // degraded: inflate published pose covariance

bool lidar_pushed, flg_first_scan = true, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false;
bool is_first_lidar = true;
bool diag_first_lidar_cb_logged = false;
bool diag_first_imu_cb_logged = false;
bool diag_first_fixposition_bias_logged = false;
bool diag_first_fixposition_bias_applied_logged = false;
bool diag_first_sync_logged = false;
bool diag_first_valid_scan_logged = false;
bool diag_first_no_point_logged = false;
bool diag_first_odom_pub_logged = false;

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
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
    if (ms < min_ms)
      min_ms = ms;
    if (ms > max_ms)
      max_ms = ms;
  }

  double mean() const { return count > 0 ? sum_ms / static_cast<double>(count) : 0.0; }
};

deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<SteadyTimePoint> lidar_receive_time_buffer;
struct QueuedImu
{
  sensor_msgs::msg::Imu::ConstSharedPtr message;
  std::uint64_t identity = 0;
  bool retained = false;
  const sensor_msgs::msg::Imu * operator->() const { return message.get(); }
  bool operator==(const QueuedImu & other) const { return identity == other.identity && message == other.message; }
};
deque<QueuedImu> imu_buffer;
// Callback group serializes the transaction lifetime. mtx_buffer protects only
// FIFO selection/confirmation and ingestion; no filter propagation holds it.
fast_lio::ImuFifoTransaction<QueuedImu> imu_fifo_transaction;
fast_lio::ImuFifoTransaction<QueuedImu>::Token pending_imu_token;
#ifdef FASTLIO_TEST_HOOKS
fast_lio::ImuFifoTransaction<QueuedImu>::Token last_imu_token_for_test;
#endif
std::uint64_t imu_sample_sequence = 0;
struct ImuConsumptionCounters
{
  std::array<std::uint64_t, static_cast<std::size_t>(fast_lio::ImuProcessReason::kCount)> events{};
  std::array<std::uint64_t, static_cast<std::size_t>(fast_lio::ImuProcessReason::kCount)> lostValid{};
  std::uint64_t retainedBatches = 0;
  std::uint64_t retainedUniqueSamples = 0;
  std::uint64_t committedSamples = 0;
  std::uint64_t validLostOnReject = 0;
  std::uint64_t backlogValidLost = 0;
  std::uint64_t retainedBacklogLost = 0;
  std::uint64_t epochValidLost = 0;
  std::uint64_t exceptionValidLost = 0;
  std::uint64_t faultValidLost = 0;
  std::uint64_t faultDiscardedLidar = 0;
  std::uint64_t faultDiscardedImu = 0;
  std::uint64_t rejectedLidar = 0;
  std::uint64_t rejectedImu = 0;
  double maxPendingAge = 0.0;
};
ImuConsumptionCounters imu_consumption_counters;
bool imu_backlog_reanchor_required = false;

void observePendingImuAgeLocked()
{
  if (!imu_buffer.empty() && imu_buffer.front().retained)
    imu_consumption_counters.maxPendingAge =
      std::max(imu_consumption_counters.maxPendingAge,
               get_time_sec(imu_buffer.back()->header.stamp) - get_time_sec(imu_buffer.front()->header.stamp));
}

// The scan queue is the front end's only backpressure valve. Unbounded, any
// sustained shortfall — a starved core, an IMU stream stamped seconds behind
// the LiDAR — queues every scan the estimator cannot reach, and the process
// grows until the OOM killer takes it (m20 dog 2, 2026-09-03: 12.9 GB in 20
// min). Past the cap the oldest waiting scans are dropped instead, so the
// estimate stays near real time at a lower effective scan rate (the hole guard
// covers the gap) rather than going stale and then dying. 0 = unbounded.
int max_buffered_scans = 100;
// 条数上限只挡内存(100 帧 = 10 s 积压);真正要挡的是时延 —— 排队 6 s 时队列还没
// 满,BACKPRESSURE 一次没响,下游却已经在吃 6 s 前的位姿(狗 2 2026-09-04)。按最旧
// 帧的年龄丢,估计器永远处理最新的一帧:算力不够表现为有效帧率下降,而不是延迟无限
// 堆高。0 = 关闭。
// The count cap only bounds memory; latency is what has to be bounded. Dropping by
// age keeps the estimator on the newest scan — the cost of falling behind becomes a
// lower effective scan rate instead of an ever-growing lag.
double max_scan_backlog_sec = 0.5;
// Only a scan drains imu_buffer, so while the LiDAR is absent and the IMU keeps publishing nothing else bounds it, and
// the first scan afterwards would integrate the whole backlog. trim_imu_buffer() keeps the newest max_imu_backlog_sec
// of IMU sensor time and every sample from the oldest queued scan's begin on (imu_backlog_policy.hpp).
double max_imu_backlog_sec = 10.0;
std::uint64_t dropped_imu_backlog_total = 0;
// Adaptive downsampling state (FrontendParams::adaptive_ds is the policy).
fast_lio::AdaptiveDownsampleState adaptive_ds_state;
size_t dropped_scan_total = 0;

bool latency_diag_en = true;
double latency_log_period_sec = 1.0;
SteadyTimePoint current_lidar_receive_time;
SteadyTimePoint last_latency_log_time;
LatencyStats sensor_to_odom_latency_stats;

// The wheel update's samples live in the node's wheel_speed_ (wheel_speed_input.hpp); the two wheel callbacks are
// free functions and reach it through this pointer. It owns nothing. The node sets it in subscribeInputs, only with
// wheel_odom_en and before the twist subscription exists, and clears it first thing in its destructor, both under
// mtx_twist: the destructor body runs before the members go, so a callback holding mtx_twist sees either a live
// object or null. One node per process, as for the rest of this file's state.
// mtx_twist guards this pointer, what the callbacks fill through it, last_twist_* and gravity_align_leg_hist. It is
// taken alone: no other lock is acquired while it is held, and it is never acquired under another lock.
fast_lio::WheelSpeedInput * wheel_speed_input = nullptr;
mutex mtx_twist;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;

fast_lio::VoxelGridHash downSizeFilterSurf;  // scan downsample: one pass, real points
pcl::VoxelGrid<PointType> downSizeFilterMap;

// Map backend: ikd-Tree by default; iVox (option B) when built with -DUSE_IVOX=ON. iVox is a
// drop-in for the hot-path API (Add_Points / Nearest_Search / Build / size / validnum /
// set_downsample_param / Delete_Point_Boxes-as-noop), so the call sites below are unchanged;
// only the init check, the dead flatten block, and the constructor Init are #ifdef'd.
MapBackend ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;

// [v10] The covariance side of a velocity assignment (covariance_reset.hpp). Every st.vel assignment below
// goes through change_x(), which leaves P as it was: the velocity still "known" to a few cm/s and still
// correlated with roll/pitch through the gravity leak of the blind stretch, so the first position
// innovations after a release are explained as a roll error (docs/PGO_LOOP_TUNING.md §7.7). With a
// sigma > 0 the assigned velocity gets that 1-σ and no cross terms; 0 keeps the legacy behaviour bit for bit.
// The sigma of a discarded velocity (engulfment release, hole clamp, runaway/parked) is in FrontendParams.
double zupt_vel_reset_sigma = 0.0;   // m/s; a velocity zeroed on static evidence (ZUPT, parked hold)
int vel_cov_resets = 0;
inline void resetVelocityCovariance(double sigma_v)
{
  if (!(sigma_v > 0.0)) {
    return;
  }
  auto P = kf.get_P();
  if (fast_lio::resetVelocityBlock(P, sigma_v)) {
    kf.change_P(P);
    ++vel_cov_resets;
  }
}
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

inline void dump_lio_state_to_log(FILE * fp)
{
  V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                             // Angle
  fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));     // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                  // omega
  fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));     // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                  // Acc
  fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));        // Bias_g
  fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));        // Bias_a
  fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]);  // Bias_a
  fprintf(fp, "\r\n");
  fflush(fp);
}

void pointBodyToWorld_ikfom(const PointType * const pi, PointType * const po, state_ikfom & s)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void pointBodyToWorld(const PointType * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> & pi, Matrix<T, 3, 1> & po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(const PointType * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(const PointType * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

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
  if (!Localmap_Initialized) {
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
      need_move = true;
  }
  if (!need_move)
    return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist =
    max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0)
    kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// Enforce max_buffered_scans and max_scan_backlog_sec. Called from the LiDAR callbacks with mtx_buffer
// held; every callback of this node shares one mutually-exclusive callback
// group, so sync_packages is never mid-flight here. The one exception is the
// scan it has already taken (lidar_pushed) and not yet popped — that one stays.
// All observation bookkeeping shares the estimator's mutually exclusive callback group.
// Queue sizes are read under mtx_buffer; no worker accesses this telemetry.
bool frontend_observation_en = false;
bool reference_health_en = false;
std::string frontend_clock_domain;
std::int64_t frontend_idle_period_ns = 500000000;
std::array<std::uint8_t, 16> frontend_instance{};
bool frontend_has_instance = false;
bool frontend_runaway = false;
fast_lio::FrontendProgress frontend_progress;
fast_lio::FrontendPublicationGuard frontend_publication_guard;
rclcpp::Publisher<fast_lio_interfaces::msg::FrontendObservation>::SharedPtr frontend_publisher;
rclcpp::Publisher<fast_lio_interfaces::msg::ReferenceHealth>::SharedPtr reference_publisher;
fast_lio::ReferenceHealthProgress reference_progress;
fast_lio::FrontendPublicationGuard reference_publication_guard;
std::array<std::uint8_t, 32> reference_contract_id{}, reference_config_id{};
void publishFrontendObservation(bool completed, bool scanEvent = false, double eventTime = 0.0) noexcept;
void publishPendingReference() noexcept;

void frontendAdmit(const builtin_interfaces::msg::Time & stamp)
{
  if (!frontend_observation_en)
    return;
  std::int64_t ns = 0;
  const bool known = fast_lio::frontendStamp(stamp.sec, stamp.nanosec, ns);
  frontend_progress.admit(ns, known);
}

void frontendDrop(std::uint64_t count = 1)
{
  if (frontend_observation_en)
    frontend_progress.increment(frontend_progress.dropped, count);
}

static void trim_lidar_buffer(fast_lio::TimeAdmissionAccounting * observer = nullptr)
{
  const bool count_bounded = max_buffered_scans > 0;
  const bool age_bounded = max_scan_backlog_sec > 0.0;
  if (!count_bounded && !age_bounded)
    return;
  const size_t cap = count_bounded ? static_cast<size_t>(max_buffered_scans) : 0u;
  const size_t keep_front = lidar_pushed ? 1u : 0u;
  size_t dropped = 0;
  auto over_cap = [&]() {
    if (lidar_buffer.size() <= keep_front)
      return false;
    if (count_bounded && lidar_buffer.size() > cap)
      return true;
    // 队首(除去已被取走那一帧)比队尾旧过预算 → 它已经没有价值了。
    if (age_bounded && time_buffer.size() > keep_front &&
        time_buffer.back() - time_buffer[keep_front] > max_scan_backlog_sec)
      return true;
    return false;
  };
  while (over_cap()) {
    if (observer) {
      const bool count = count_bounded && lidar_buffer.size() > cap;
      const bool age = age_bounded && time_buffer.size() > keep_front &&
                       time_buffer.back() - time_buffer[keep_front] > max_scan_backlog_sec;
      observer->record(count ? fast_lio::AdmissionBucket::kTrimCount : fast_lio::AdmissionBucket::kTrimAge,
                       lidar_buffer[keep_front]->size());
      if (count && age)
        observer->add(observer->trim_also_age);
    }
    lidar_buffer.erase(lidar_buffer.begin() + keep_front);
    time_buffer.erase(time_buffer.begin() + keep_front);
    if (lidar_receive_time_buffer.size() > keep_front)
      lidar_receive_time_buffer.erase(lidar_receive_time_buffer.begin() + keep_front);
    ++dropped;
  }
  if (dropped == 0)
    return;
  dropped_scan_total += dropped;
  frontendDrop(dropped);
  static double last_warn_time = 0.0;
  const double now = omp_get_wtime();
  if (now - last_warn_time > 5.0) {
    last_warn_time = now;
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[BACKPRESSURE] scan queue over budget (%d scans / %.2fs): dropped %zu scan(s) (%zu total) — "
                "the estimator is not keeping up with the LiDAR",
                max_buffered_scans,
                max_scan_backlog_sec,
                dropped,
                dropped_scan_total);
  }
}

rclcpp::Clock input_time_log_clock(RCL_STEADY_TIME);
fast_lio::InputEpochGuard input_epoch_guard;

// Enforce max_imu_backlog_sec. Called from enqueue_imu_msg with mtx_buffer held, right after the push. The samples from
// the oldest queued scan's begin on stay whatever their age: that scan still integrates them. Dropped samples count
// into the observation's dropped counter, as dropped scans do.
struct ImuBacklogLoss
{
  std::size_t samples = 0;
  double oldest = 0.0;
  double newest = 0.0;
  std::uint64_t validTotal = 0;
  std::uint64_t retainedTotal = 0;
  std::uint64_t droppedTotal = 0;
};

static ImuBacklogLoss trim_imu_buffer()
{
  if (!(max_imu_backlog_sec > 0.0) || imu_buffer.empty())
    return {};
  const double protect_from = lidar_buffer.empty() ? std::numeric_limits<double>::infinity() : time_buffer.front();
  const std::size_t dropped = fast_lio::imuBacklogExcess(
    imu_buffer.size(),
    [](std::size_t i) { return get_time_sec(imu_buffer[i]->header.stamp); },
    get_time_sec(imu_buffer.back()->header.stamp),
    max_imu_backlog_sec,
    protect_from);
  if (dropped == 0)
    return {};
  const double oldest_dropped = get_time_sec(imu_buffer.front()->header.stamp);
  const double newest_dropped = get_time_sec(imu_buffer[dropped - 1]->header.stamp);
  for (std::size_t i = 0; i < dropped; ++i) {
    if (validImuForConsumption(imu_buffer.front().message)) {
      ++imu_consumption_counters.backlogValidLost;
      if (imu_buffer.front().retained) {
        ++imu_consumption_counters.retainedBacklogLost;
        imu_backlog_reanchor_required = true;
      }
    }
    imu_buffer.pop_front();
  }
  imu_fifo_transaction.invalidate();
  dropped_imu_backlog_total += dropped;
  frontendDrop(dropped);
  return {dropped,
          oldest_dropped,
          newest_dropped,
          imu_consumption_counters.backlogValidLost,
          imu_consumption_counters.retainedBacklogLost,
          dropped_imu_backlog_total};
}

// Snapshot under mtx_buffer, log after unlocking. Exact loss accounting lives
// in the counters and EOF summary; per-event identities use an opt-in logger.
void logImuBacklogLoss(const ImuBacklogLoss & loss)
{
  if (loss.samples == 0)
    return;
  if (reference_health_en)
    reference_progress.negative(fast_lio::kReferenceLostBatch);
  RCLCPP_DEBUG(
    rclcpp::get_logger("laser_mapping.imu_transactions"),
    "[IMU-CONSUMPTION-LOSS] reason=backlog samples=%zu oldest=%.9f newest=%.9f valid_total=%lu retained_total=%lu",
    loss.samples,
    loss.oldest,
    loss.newest,
    static_cast<unsigned long>(loss.validTotal),
    static_cast<unsigned long>(loss.retainedTotal));
  RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                       input_time_log_clock,
                       5000,
                       "[IMU-BACKLOG] IMU queue over its %.2fs window: dropped %zu sample(s) stamped %.3f..%.3f "
                       "(%.3fs) that no queued scan needs, %lu dropped in total",
                       max_imu_backlog_sec,
                       loss.samples,
                       loss.oldest,
                       loss.newest,
                       loss.newest - loss.oldest,
                       static_cast<unsigned long>(loss.droppedTotal));
}

// Caller holds mtx_buffer. Clearing references and invalidating the epoch is
// one operation with admission latching; partial integration is never retried.
void clearBufferedInputsLocked(fast_lio::TimeAdmissionAccounting * observer, fast_lio::AdmissionBucket reason)
{
  if (observer)
    for (const auto & cloud : lidar_buffer)
      observer->record(reason, cloud->size());
  imu_fifo_transaction.invalidate();
  pending_imu_token = {};
  lidar_buffer.clear();
  time_buffer.clear();
  lidar_receive_time_buffer.clear();
  imu_buffer.clear();
  lidar_pushed = false;
  Measures.imu.clear();
  Measures.lidar.reset();
  feats_undistort->clear();
  flio_map_frozen = true;
  flio_degraded_odom = true;
}

// Caller holds mtx_buffer; all ingestion/sync callbacks share one mutually
// exclusive group. A full estimator/map reset requires explicit restart, not
// an automatic gravity reinitialization on a possibly moving platform. `lead`
// is the calling stream's own filter for stamps far ahead (scan_time_policy.hpp).
fast_lio::AdmissionInputResult acceptInputEpoch(double previous,
                                                double current,
                                                const char * source,
                                                fast_lio::InputLeadFilter & lead,
                                                fast_lio::TimeAdmissionAccounting * observer = nullptr)
{
  const bool already_faulted = input_epoch_guard.faulted();
  const bool input_allowed = imu_fifo_transaction.acceptsInput(already_faulted);
  constexpr double kInputEpochRollbackSec = 1.0;
  // Longer than any hole the front end bridges on its own (the IMU backlog bound is 10 s as well), so a normal
  // stream never meets it; a stamp far in the future only costs its own sample.
  constexpr double kInputEpochLeadSec = 10.0;
  const auto result = fast_lio::classifyAdmissionInput(
    input_epoch_guard, lead, previous, current, input_allowed, kInputEpochRollbackSec, kInputEpochLeadSec);
  if (!input_allowed) {
    if (std::strcmp(source, "imu") == 0)
      ++imu_consumption_counters.rejectedImu;
    else
      ++imu_consumption_counters.rejectedLidar;
    frontendDrop();
    return result;
  }
  if (result == fast_lio::AdmissionInputResult::kLeadRejected) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] %s stamp %.9f is %.3f s past the stream's newest %.9f: dropped, and the next "
                         "sample shows whether the stream moved on",
                         source,
                         current,
                         current - previous,
                         previous);
    frontendDrop();
    return result;
  }
  if (result == fast_lio::AdmissionInputResult::kAccepted)
    return result;
  frontendDrop(lidar_buffer.size() + imu_buffer.size() + 1);
  for (const auto & sample : imu_buffer)
    if (validImuForConsumption(sample.message))
      ++imu_consumption_counters.epochValidLost;
  clearBufferedInputsLocked(observer, fast_lio::AdmissionBucket::kClearEpoch);
  if (!already_faulted) {
    input_epoch_fault_stamp = current;
    RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"),
                 "[INPUT-EPOCH] %s clock discontinuity %.9f -> %.9f: input processing and odometry stopped; "
                 "restart the node after clocks stabilize to reset estimator and map state",
                 source,
                 previous,
                 current);
  }
  return result;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg,
                      fast_lio::InputLeadFilter & lead,
                      fast_lio::TimeAdmissionAccounting * observer)
{
  fast_lio::AdmissionCallbackScope accounting{observer};
  if (observer)
    observer->record(fast_lio::AdmissionBucket::kReceived, std::uint64_t(msg->width) * msg->height);
  const auto receive_time = SteadyClock::now();
  if (!diag_first_lidar_cb_logged) {
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                "[STARTUP][FAST_LIO] first lidar cb (PointCloud2) stamp=%.3f",
                get_time_sec(msg->header.stamp));
    diag_first_lidar_cb_logged = true;
  }
  mtx_buffer.lock();
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
  if (observer) {
    observer->canonical_header_known = true;
    observer->canonical_header_bits = fast_lio::admissionDoubleBits(cur_time);
  }
  const auto input = acceptInputEpoch(is_first_lidar ? -1.0 : last_timestamp_lidar, cur_time, "lidar", lead, observer);
  if (input != fast_lio::AdmissionInputResult::kAccepted) {
    if (observer)
      observer->recordInputRejection(input, std::uint64_t(msg->width) * msg->height);
    accounting.resolve();
    mtx_buffer.unlock();
    if (frontend_observation_en)
      publishFrontendObservation(false);
    return;
  }
  double preprocess_start_time = omp_get_wtime();
  if (degeneracy_debug) {
    // Input-side telemetry for scan-stream holes: callback count + stamp step.
    static double last_cb_stamp = 0.0;
    const double step = last_cb_stamp > 0.0 ? cur_time - last_cb_stamp : 0.0;
    last_cb_stamp = cur_time;
    std::cerr << std::fixed << std::setprecision(3) << "[LIDARCB] n=" << scan_count << " t=" << cur_time
              << " step=" << step << " buf=" << lidar_buffer.size() << std::endl;
  }
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    // Late data cannot be replayed into the forward-only estimator.
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] late lidar by %.6f seconds, dropped",
                         last_timestamp_lidar - cur_time);
    frontendDrop();
    if (observer)
      observer->record(fast_lio::AdmissionBucket::kLateRejected, std::uint64_t(msg->width) * msg->height);
    accounting.resolve();
    mtx_buffer.unlock();
    return;
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  if (observer) {
    observer->record(fast_lio::AdmissionBucket::kEnqueued, ptr->size());
    observer->record(fast_lio::AdmissionBucket::kEnqueuedDeclared, std::uint64_t(msg->width) * msg->height);
  }
  accounting.resolve();
  frontendAdmit(msg->header.stamp);
  time_buffer.push_back(cur_time);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer(observer);
  last_timestamp_lidar = cur_time;
  if (fast_lio::timingLogAccepts(scan_count)) {
    s_plot11[fast_lio::timingLogSlot(scan_count)] = omp_get_wtime() - preprocess_start_time;
  }
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

// Direct subscriber for shm_msgs/PointCloud8mAndPose (/pre/all_point). Decodes the
// fixed 8MB buffer in place (no relay node, no extra DDS hop) — same buffering path
// as standard_pcl_cbk, just a different message type handed to Preprocess.
void shm_allpoint_cbk(const shm_msgs::msg::PointCloud8mAndPose::UniquePtr msg, fast_lio::InputLeadFilter & lead)
{
  const auto receive_time = SteadyClock::now();
  if (!diag_first_lidar_cb_logged) {
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                "[STARTUP][FAST_LIO] first lidar cb (shm PointCloud8mAndPose) stamp=%.3f",
                get_time_sec(msg->header.stamp));
    diag_first_lidar_cb_logged = true;
  }
  mtx_buffer.lock();
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
  if (acceptInputEpoch(is_first_lidar ? -1.0 : last_timestamp_lidar, cur_time, "lidar", lead) !=
      fast_lio::AdmissionInputResult::kAccepted) {
    mtx_buffer.unlock();
    if (frontend_observation_en)
      publishFrontendObservation(false);
    return;
  }
  double preprocess_start_time = omp_get_wtime();
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    // Late data cannot be replayed into the forward-only estimator.
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] late lidar by %.6f seconds, dropped",
                         last_timestamp_lidar - cur_time);
    frontendDrop();
    mtx_buffer.unlock();
    return;
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  frontendAdmit(msg->header.stamp);
  time_buffer.push_back(cur_time);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer();
  last_timestamp_lidar = cur_time;
  if (fast_lio::timingLogAccepts(scan_count)) {
    s_plot11[fast_lio::timingLogSlot(scan_count)] = omp_get_wtime() - preprocess_start_time;
  }
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg, fast_lio::InputLeadFilter & lead)
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
  if (acceptInputEpoch(is_first_lidar ? -1.0 : last_timestamp_lidar, cur_time, "lidar", lead) !=
      fast_lio::AdmissionInputResult::kAccepted) {
    mtx_buffer.unlock();
    if (frontend_observation_en)
      publishFrontendObservation(false);
    return;
  }
  double preprocess_start_time = omp_get_wtime();
  scan_count++;
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    // Late data cannot be replayed into the forward-only estimator.
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] late lidar by %.6f seconds, dropped",
                         last_timestamp_lidar - cur_time);
    frontendDrop();
    mtx_buffer.unlock();
    return;
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }
  last_timestamp_lidar = cur_time;

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() &&
      !lidar_buffer.empty()) {
    printf(
      "IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
  }

  if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()) {
    timediff_set_flg = true;
    timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
    printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  frontendAdmit(msg->header.stamp);
  time_buffer.push_back(last_timestamp_lidar);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer();

  if (fast_lio::timingLogAccepts(scan_count)) {
    s_plot11[fast_lio::timingLogSlot(scan_count)] = omp_get_wtime() - preprocess_start_time;
  }
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void checkAnchorEvidenceOnImu(double imu_stamp, bool scan_waiting);

void enqueue_imu_msg(sensor_msgs::msg::Imu::SharedPtr msg,
                     double debug_raw_stamp,
                     fast_lio::InputLeadFilter & lead,
                     fast_lio::TimeAdmissionAccounting * observer = nullptr)
{
  publish_count++;
  if (!diag_first_imu_cb_logged) {
    RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
                "[STARTUP][FAST_LIO] first imu cb raw_stamp=%.3f adjusted_stamp=%.3f",
                debug_raw_stamp,
                get_time_sec(msg->header.stamp));
    diag_first_imu_cb_logged = true;
  }

  double timestamp = get_time_sec(msg->header.stamp);

  mtx_buffer.lock();
  if (acceptInputEpoch(last_timestamp_imu, timestamp, "imu", lead, observer) !=
      fast_lio::AdmissionInputResult::kAccepted) {
    mtx_buffer.unlock();
    if (frontend_observation_en)
      publishFrontendObservation(false);
    return;
  }

  if (timestamp < last_timestamp_imu) {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] late IMU by %.6f seconds, dropped",
                         last_timestamp_imu - timestamp);
    frontendDrop();
    mtx_buffer.unlock();
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back({msg, ++imu_sample_sequence, false});
  frontendAdmit(msg->header.stamp);
  const auto backlogLoss = trim_imu_buffer();
  observePendingImuAgeLocked();
  const bool scan_waiting = !lidar_buffer.empty() || lidar_pushed;
  mtx_buffer.unlock();
  logImuBacklogLoss(backlogLoss);
  sig_buffer.notify_all();
  checkAnchorEvidenceOnImu(timestamp, scan_waiting);
  if (frontend_observation_en)
    publishFrontendObservation(false);
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in,
             fast_lio::InputLeadFilter & lead,
             fast_lio::TimeAdmissionAccounting * observer)
{
  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
  msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
    msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
  }
  enqueue_imu_msg(msg, get_time_sec(msg_in->header.stamp), lead, observer);
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

void imu_fpa_cbk(const fixposition_driver_msgs::msg::FpaImu::UniquePtr msg_in,
                 fast_lio::InputLeadFilter & lead,
                 fast_lio::TimeAdmissionAccounting * observer)
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
  enqueue_imu_msg(msg, get_time_sec(msg_in->data.header.stamp), lead, observer);
}

void twist_cbk(const geometry_msgs::msg::TwistStamped::UniquePtr msg_in)
{
  Eigen::Vector3d v(msg_in->twist.linear.x, msg_in->twist.linear.y, msg_in->twist.linear.z);
  std::lock_guard<std::mutex> lk(mtx_twist);
  // NaN is the wheelspeed bridge saying "cannot measure", not a number. Only a
  // real measurement may enter the buffer the iEKF update consumes; what the
  // absence of one means is decided once, in static_evidence.hpp, which is why
  // the stamp and speed below are still recorded as they arrive.
  if (fast_lio::wheelSampleIsEvidence(v(0), v(1), v(2))) {
    if (wheel_speed_input != nullptr)
      wheel_speed_input->addTwist(msg_in->header.stamp.sec, msg_in->header.stamp.nanosec, v(0));
  }
  // Stamped newest sample for the static hold: the wheel-odom fusion below
  // consumes the buffer, but the hold needs to know how OLD the evidence is.
  last_twist_stamp = get_time_sec(msg_in->header.stamp);
  last_twist_speed = v.norm();
  if (v.allFinite()) {
    last_twist_v_body = v;
  }
  // [v6] stamped forward-speed history for the levelling prior's leg-velocity edges
  if (std::isfinite(v(0))) {
    gravity_align_leg_hist.emplace_back(last_twist_stamp, v(0));
    while (!gravity_align_leg_hist.empty() &&
           (gravity_align_leg_hist.front().first < last_twist_stamp - 2.5 || gravity_align_leg_hist.size() > 500)) {
      gravity_align_leg_hist.pop_front();
    }
  }
}

// The variances of the wheel samples (wheel_cov_topic): cov[0] forward and cov[7] lateral, paired with the gated
// twist of the same stamp. Subscribed only while wheel_speed_input is set; it feeds nothing else.
void twist_cov_cbk(const geometry_msgs::msg::TwistWithCovarianceStamped::UniquePtr msg_in)
{
  std::lock_guard<std::mutex> lk(mtx_twist);
  if (wheel_speed_input != nullptr)
    wheel_speed_input->addVariances(
      msg_in->header.stamp.sec, msg_in->header.stamp.nanosec, msg_in->twist.covariance[0], msg_in->twist.covariance[7]);
}

// Independent, front-end-agnostic evidence that the platform is parked. The
// verdict itself lives in static_evidence.hpp and stationary_anchor_policy.hpp (pure, unit-tested); this only
// summarises the scan's IMU window and reads the newest wheel sample. `trace`, when given, receives the verdict and
// its inputs for the anchor trace.
fast_lio::StaticEvidenceVerdict
static_evidence_of(const MeasureGroup & meas, double now, fast_lio::AnchorScanRecord * trace = nullptr)
{
  fast_lio::ImuWindowStats imu;
  imu.acc_min = std::numeric_limits<double>::max();
  double acc_sum = 0.0;
  for (const auto & m : meas.imu) {
    const auto & w = m->angular_velocity;
    const auto & a = m->linear_acceleration;
    imu.gyr_max = std::max(imu.gyr_max, V3D(w.x, w.y, w.z).norm());
    const double acc_norm = V3D(a.x, a.y, a.z).norm();
    imu.acc_min = std::min(imu.acc_min, acc_norm);
    imu.acc_max = std::max(imu.acc_max, acc_norm);
    acc_sum += acc_norm;
    ++imu.samples;
  }
  if (imu.samples > 0)
    imu.acc_mean = acc_sum / static_cast<double>(imu.samples);

  double wheel_stamp = -1.0;
  double wheel_speed = 0.0;
  {
    std::lock_guard<std::mutex> lk(mtx_twist);
    wheel_stamp = last_twist_stamp;
    wheel_speed = last_twist_speed;
  }
  const double wheel_age = wheel_stamp > 0.0 ? std::fabs(now - wheel_stamp) : -1.0;

  fast_lio::StaticEvidenceParams params;
  params.gyro_thresh = zupt_gyro_thresh;
  params.acc_span_ratio = zupt_acc_span_ratio;
  params.wheel_speed_thresh = zupt_wheel_speed_thresh;
  params.wheel_max_age = zupt_wheel_max_age;
  params.require_wheel = zupt_require_wheel;
  fast_lio::WheelEvidence wheel;
  wheel.valid = wheel_stamp > 0.0;
  wheel.age = wheel_age;
  wheel.speed = wheel_speed;
  const fast_lio::StaticEvidenceVerdict verdict = fast_lio::classifyStaticEvidence(imu, wheel, params);
  if (trace != nullptr) {
    trace->body = verdict.stationary() ? fast_lio::AnchorTraceBody::kStatic : fast_lio::AnchorTraceBody::kMoving;
    trace->imu_n = imu.samples;
    trace->gyr_max = imu.samples > 0 ? imu.gyr_max : std::numeric_limits<double>::quiet_NaN();
    trace->acc_span = imu.samples > 0 && imu.acc_mean > 0.0 ? (imu.acc_max - imu.acc_min) / imu.acc_mean
                                                            : std::numeric_limits<double>::quiet_NaN();
    trace->wheel_age = wheel_age;
    trace->wheel_speed = wheel_stamp > 0.0 ? wheel_speed : std::numeric_limits<double>::quiet_NaN();
    trace->evidence = verdict.evidence;
    trace->evidence_reason = verdict.reason;
    trace->imu_only = verdict.imu_only;
    trace->wheel_valid = fast_lio::wheelEvidenceUsable(wheel, params);
  }
  return verdict;
}

bool body_is_static(const MeasureGroup & meas, double now)
{
  return static_evidence_of(meas, now).stationary();
}

double max_scan_duration_s = 0.0;  // 0 disables the device-specific upper bound.
std::uint64_t invalid_scan_time_count = 0;
std::uint64_t nonadvancing_scan_count = 0;
std::uint64_t empty_scan_imu_count = 0;
std::uint64_t imu_coverage_drop_count = 0;

// Callbacks and sync use the same mutually exclusive callback group.
void popLidarScan(fast_lio::TimeAdmissionAccounting * observer, fast_lio::AdmissionBucket reason)
{
  if (observer)
    observer->record(reason, lidar_buffer.front()->size());
  lidar_buffer.pop_front();
  time_buffer.pop_front();
  if (!lidar_receive_time_buffer.empty())
    lidar_receive_time_buffer.pop_front();
  lidar_pushed = false;
}

// What every encoding of the current scan reports (stationary_anchor_policy.hpp): the legacy flags, the published
// covariance inflation, the typed observation and the trace's export fields all read this one verdict. Degraded is
// the scan's own verdict, flio_degraded_odom, and is set on every scan once Unanchored; Pending adds nothing to it.
// A static hold is claimed only for a hold on a usable anchor of a LocalQualified reference, and the scan the gate
// declared lost reports the loss even when its own map reset has re-armed the gate since.
fast_lio::AnchorHealth currentAnchorHealth()
{
  fast_lio::AnchorHealthInputs in;
  in.degraded = flio_degraded_odom;
  in.hold_active = zupt_active || parked_hold_state.active;
  in.anchor_usable = anchor_usable;
  in.lost = required_prior_lost || reanchor_gate.state == fast_lio::ReanchorState::kLost || reanchor_lost_this_scan;
  in.trust = recovery_state.trust;
  return fast_lio::anchorHealth(in);
}

// Serialized with the estimator by the existing default mutually-exclusive callback group.
// This value is frozen after late guards (or after the scope's residual negative settlement).
// Every encoding reads it; none of the publishers re-reads RecoveryState.
struct FinalFrontendSnapshot
{
  fast_lio::AnchorHealth health;
  fast_lio::FrontendFacts frontend;
  fast_lio::ReferenceHealthSnapshot reference;
  std::int64_t poseStamp = 0;
  std::uint64_t coordinateEpoch = lost_recovery.coordinate_epoch;
};
FinalFrontendSnapshot anchor_final_snapshot;
bool anchor_final_frozen = false;
bool reference_pending = false;
bool anchor_pending_legacy_observation = false;

void disableReferencePublication(const char * reason)
{
  reference_health_en = false;
  reference_pending = false;
  RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"), "Reference health disabled for this activation: %s", reason);
}

FinalFrontendSnapshot captureFinalFrontend(bool completed, bool scanEvent, double eventTime) noexcept
{
  FinalFrontendSnapshot final;
  final.health = currentAnchorHealth();
  auto & facts = final.frontend;
  facts.completed = completed;
  facts.initialized = flg_EKF_inited;
  facts.lost = final.health.lost;
  facts.blind = reanchor_gate.state == fast_lio::ReanchorState::kBlind;
  facts.verifying = reanchor_gate.state == fast_lio::ReanchorState::kVerifying;
  facts.degraded = final.health.effective_degraded;
  facts.staticHold = final.health.reported_static_hold;
  facts.runaway = frontend_runaway;
  facts.levelling = gravity_align_degraded_active;
  facts.blocked = input_epoch_guard.faulted();
  facts.contextKnown = frontend_has_instance && !odom_frame_id.empty() && !body_frame_id.empty() &&
                       !frontend_clock_domain.empty() && !facts.blocked;
  if (completed) {
    facts.stampKnown = fast_lio::frontendSecondsStamp(lidar_end_time, final.poseStamp);
    facts.pose = {
      {state_point.pos[0], state_point.pos[1], state_point.pos[2], geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w}};
  }
  // Lifecycle projection is present even with R1 disabled, so trace v5's startup fact is preserved.
  auto & value = final.reference;
  value.kind = completed ? 1 : 3;
  value.hasPoseBinding = completed;
  value.pose = facts.pose;
  value.legacyFlags = fast_lio::frontendVerdict(facts).health;
  value.referenceGeneration = stationary_anchor.reference_generation;
  value.hasEvaluatedInput = frontend_progress.hasInputStamp && !facts.blocked;
  value.evaluatedInputNs = value.hasEvaluatedInput ? frontend_progress.inputStamp : 0;
  std::int64_t imuProgressNs = 0;
  if (!facts.blocked && imu_sample_sequence != 0 && fast_lio::frontendSecondsStamp(last_timestamp_imu, imuProgressNs) &&
      (!value.hasEvaluatedInput || imuProgressNs > value.evaluatedInputNs)) {
    value.hasEvaluatedInput = true;
    value.evaluatedInputNs = imuProgressNs;
  }
  std::int64_t endNs = 0;
  const bool endKnown = scanEvent && anchor_scan_record.end_ns >= 0 && fast_lio::frontendSecondsStamp(eventTime, endNs);
  value.hasEffective = fast_lio::referenceEffective(
    completed, endKnown, endNs, value.hasEvaluatedInput, value.evaluatedInputNs, value.effectiveNs);
  value.hasStateValidUntil = value.hasEvaluatedInput;
  value.stateValidUntilNs = value.hasStateValidUntil ? fast_lio::referenceUntil(value.evaluatedInputNs) : 0;
  value.hasScanSequence = scanEvent && anchor_scan_record.scan_seq != 0;
  value.scanSequence = value.hasScanSequence ? anchor_scan_record.scan_seq : 0;
  value.hasScanEnd = scanEvent && anchor_scan_record.end_ns >= 0;
  // Pose association uses the exact legacy stamp conversion, not traceNs's rounded ns.
  value.scanEndNs = completed ? final.poseStamp : value.hasScanEnd ? anchor_scan_record.end_ns : 0;
  value.hasScanBegin = scanEvent && anchor_scan_record.beg_ns >= 0;
  value.scanBeginNs =
    value.hasScanBegin ? fast_lio::referenceScanBegin(completed, anchor_scan_record.beg_ns, value.scanEndNs) : 0;
  const fast_lio::ReferenceScanFacts scan{completed,
                                          facts.initialized,
                                          flio_degraded_odom,
                                          facts.lost,
                                          facts.runaway,
                                          facts.blocked,
                                          facts.blind || facts.verifying,
                                          facts.levelling,
                                          facts.staticHold,
                                          anchor_scan_evaluated};
  value = fast_lio::projectReferenceHealth(std::move(value), recovery_state, scan);
  if (frontend_observation_en && completed) {
    frontend_publication_guard.run(
      [&] {
        auto progress = frontend_progress;
        progress.finish(fast_lio::frontendVerdict(facts));
        value.hasObservationId = fast_lio::frontendObservationId(frontend_instance,
                                                                 frontend_has_instance && progress.valid,
                                                                 progress.completion,
                                                                 final.poseStamp,
                                                                 facts.stampKnown,
                                                                 frontend_clock_domain,
                                                                 value.observationId);
      },
      [](const char * reason) {
        frontend_observation_en = false;
        disableReferencePublication(reason);
      });
  }
  if (reference_health_en) {
    reference_publication_guard.run(
      [&] {
        // Runaway and Lost settle with the scan that reached them, whether or not it completed a pose. A snapshot
        // between scans only projects the persistent facts; it records no new negative event.
        const bool runaway = scanEvent && facts.runaway;
        if ((completed && flio_degraded_odom) || runaway)
          reference_progress.negative((completed && flio_degraded_odom ? fast_lio::kReferenceRawDegraded : 0u) |
                                        (runaway ? fast_lio::kReferenceRunaway : 0u),
                                      runaway ? 8u : 0u);
        if (scanEvent && facts.lost)
          reference_progress.negative(0, 1);
        if (completed && !fast_lio::referencePoseFinite(facts.pose))
          reference_progress.negative(0, 32);
        value.hasSourceInstance = frontend_has_instance;
        value.sourceInstance = frontend_instance;
        value.clockDomain = frontend_clock_domain;
        value.frameId = odom_frame_id;
        value.childFrameId = body_frame_id;
        value.hasCoordinateEpoch = facts.contextKnown;
        value.coordinateEpoch = facts.contextKnown ? final.coordinateEpoch : 0;
        value.producerContractId = reference_contract_id;
        value.configId = reference_config_id;
        value.enabledPolicyMask = fast_lio::kReferenceImplemented;
        value = fast_lio::projectReferenceHealth(std::move(value), recovery_state, scan);
        if (!scanEvent && !reference_progress.inputDue(value))
          return;
        value = reference_progress.settle(std::move(value));
        if (!scanEvent && value.hasEvaluatedInput)
          reference_progress.noteInput(value.evaluatedInputNs);
      },
      disableReferencePublication);
  }
  return final;
}

void buildAndPublishReference(const fast_lio::ReferenceHealthSnapshot & value)
{
  if (!reference_health_en || !reference_publisher || !value.hasSnapshotSequence)
    return;
  fast_lio_interfaces::msg::ReferenceHealth message;
  message.schema_version = value.schemaVersion;
  message.identity_version = value.identityVersion;
  message.has_source_instance = value.hasSourceInstance;
  message.source_instance = value.sourceInstance;
  message.clock_domain = value.clockDomain;
  message.frame_id = value.frameId;
  message.child_frame_id = value.childFrameId;
  message.has_coordinate_epoch = value.hasCoordinateEpoch;
  message.coordinate_epoch = value.coordinateEpoch;
  message.producer_contract_id = value.producerContractId;
  message.config_id = value.configId;
  message.implemented_mask = value.implementedMask;
  message.enabled_policy_mask = value.enabledPolicyMask;
  message.has_snapshot_sequence = value.hasSnapshotSequence;
  message.snapshot_sequence = value.snapshotSequence;
  message.has_state_revision = value.hasStateRevision;
  message.state_revision = value.stateRevision;
  message.has_scan_sequence = value.hasScanSequence;
  message.scan_sequence = value.scanSequence;
  message.has_observation_id = value.hasObservationId;
  message.observation_id = value.observationId;
  message.kind = value.kind;
  message.has_evaluated_input = value.hasEvaluatedInput;
  message.evaluated_input_ns = value.evaluatedInputNs;
  message.has_effective = value.hasEffective;
  message.effective_ns = value.effectiveNs;
  message.has_state_valid_until = value.hasStateValidUntil;
  message.state_valid_until_ns = value.stateValidUntilNs;
  message.has_scan_begin = value.hasScanBegin;
  message.scan_begin_ns = value.scanBeginNs;
  message.has_scan_end = value.hasScanEnd;
  message.scan_end_ns = value.scanEndNs;
  message.has_pose_binding = value.hasPoseBinding;
  message.pose_binding = value.pose;
  message.lifecycle = value.lifecycle;
  message.startup_reference = value.startupReference;
  message.trust = value.trust;
  message.local = value.local;
  message.legacy_flags = value.legacyFlags;
  message.has_raw_registration_degraded = value.hasRawRegistrationDegraded;
  message.raw_registration_degraded = value.rawRegistrationDegraded;
  message.reference_generation = value.referenceGeneration;
  message.recovery_episode = value.recoveryEpisode;
  message.has_qualified_event_sequence = value.hasQualifiedEventSequence;
  message.qualified_event_sequence = value.qualifiedEventSequence;
  message.has_original_start = value.hasOriginalStart;
  message.original_start_ns = value.originalStartNs;
  message.has_original_deadline = value.hasOriginalDeadline;
  message.original_deadline_ns = value.originalDeadlineNs;
  message.has_last_supported_end = value.hasLastSupportedEnd;
  message.last_supported_end_ns = value.lastSupportedEndNs;
  message.present_mask = value.presentMask;
  message.true_mask = value.trueMask;
  message.has_negative_masks = value.hasNegativeMasks;
  message.cause_mask = value.causeMask;
  message.taint_mask = value.taintMask;
  message.hard_mask = value.hardMask;
  message.hr_witness_kind = value.hrWitnessKind;
  message.has_hr_valid_until = value.hasHrValidUntil;
  message.hr_valid_until_ns = value.hrValidUntilNs;
  // All RW / g-prime presence fields retain their false value-initialization.
  reference_publisher->publish(message);
}

void publishPendingReference() noexcept
{
  if (!reference_pending)
    return;
  reference_pending = false;
  reference_publication_guard.run([&] { buildAndPublishReference(anchor_final_snapshot.reference); },
                                  disableReferencePublication);
}

void publishReferenceRepeat() noexcept
{
  if (!reference_health_en)
    return;
  static SteadyClock::time_point last;
#ifdef FASTLIO_TEST_HOOKS
  const auto now = fast_lio_test::referenceRepeatClock ? fast_lio_test::referenceRepeatClock() : SteadyClock::now();
#else
  const auto now = SteadyClock::now();
#endif
  if (last.time_since_epoch().count() != 0 && now - last < std::chrono::milliseconds(100))
    return;
  last = now;
  reference_publication_guard.run([&] { buildAndPublishReference(reference_progress.repeat()); },
                                  disableReferencePublication);
}

// Stationary-anchor trace (anchor_trace.hpp). Everything from here to the anchor lifecycle helpers only reads the
// estimator and the lifecycle, and writes the trace's own state.
fast_lio::AnchorTraceGate anchorTraceGate(fast_lio::ReanchorState state)
{
  switch (state) {
    case fast_lio::ReanchorState::kHealthy:
      return fast_lio::AnchorTraceGate::kHealthy;
    case fast_lio::ReanchorState::kBlind:
      return fast_lio::AnchorTraceGate::kBlind;
    case fast_lio::ReanchorState::kVerifying:
      return fast_lio::AnchorTraceGate::kVerifying;
    case fast_lio::ReanchorState::kLost:
      return fast_lio::AnchorTraceGate::kLost;
  }
  return fast_lio::AnchorTraceGate::kNa;
}

void emitAnchorTraceLine(const std::string & line)
{
  RCLCPP_INFO(rclcpp::get_logger(kAnchorTraceLogger), "%s", line.c_str());
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterAnchorTraceLine)
    fast_lio_test::afterAnchorTraceLine(line);
#endif
}

// One transition record. Never throttled: the checker counts them.
void anchorTransition(fast_lio::AnchorTraceEvent event, fast_lio::AnchorTraceReason reason, double stamp)
{
  fast_lio::AnchorTransition transition;
  transition.event = event;
  transition.reason = reason;
  transition.event_seq = anchor_event_seq.next();
  transition.scan_seq = anchor_scan_seq.last();
  transition.stamp_ns = fast_lio::traceNs(stamp);
  transition.anchor_id = stationary_anchor.id;
  transition.anchor_valid = stationary_anchor.state == fast_lio::AnchorState::kValid;
  transition.created_ns = stationary_anchor.created_ns;
  transition.last_evidence_ns = stationary_anchor.last_evidence_end_ns;
  transition.anchor_state = stationary_anchor.state;
  transition.episode = stationary_anchor.episode;
  transition.ref_gen = stationary_anchor.reference_generation;
  transition.streak = static_cast<std::uint64_t>(std::max(parked_hold_state.parked_scans, 0));
  fast_lio::saturatingIncrement(&anchor_trace_counters.events);
  emitAnchorTraceLine(fast_lio::formatAnchorTransition(transition));
}

// One recovery record, from the same event sequence. Never throttled.
void recoveryTransition(fast_lio::RecoveryTraceEvent event, fast_lio::RecoveryCause reason, std::int64_t stamp_ns)
{
  fast_lio::RecoveryTransition transition;
  transition.event = event;
  transition.reason = reason;
  transition.event_seq = anchor_event_seq.next();
  transition.scan_seq = anchor_scan_seq.last();
  transition.stamp_ns = stamp_ns;
  transition.episode = recovery_state.episode;
  transition.start_ns = recovery_state.start_ns;
  transition.deadline_ns = recovery_state.deadline_ns;
  transition.elapsed_ns =
    stamp_ns >= 0 && recovery_state.start_ns >= 0 ? stamp_ns - recovery_state.start_ns : std::int64_t{-1};
  transition.reset_count = recovery_state.map_resets;
  transition.path = anchor_idle_path ? fast_lio::RecoveryTracePath::kIdle : fast_lio::RecoveryTracePath::kScan;
  fast_lio::saturatingIncrement(&anchor_trace_counters.events);
  auto line = fast_lio::formatRecoveryTransition(transition);
  if (event == fast_lio::RecoveryTraceEvent::kEpochCommitted)
    line += " coordinate_epoch=" + std::to_string(lost_recovery.coordinate_epoch);
  emitAnchorTraceLine(line);
}

// The parked streak counts the scans of one candidate or anchor, so whatever ends that, or breaks the evidence
// behind it, ends the streak. The count goes first and the records of the event already carry it; an engaged
// prior is released, and that recorded, where the anchor ends or by the scan's anchor step.
void endParkedStreak()
{
  parked_hold_state.parked_scans = 0;
}

// An input event that breaks the static evidence behind an anchor: counted every time, marked on the open scan
// record, and logged when `report` is set.
void anchorEvidence(
  fast_lio::AnchorTraceReason reason, std::uint32_t bit, std::uint64_t * counter, double stamp, bool report)
{
  constexpr auto breaksLostRecovery = fast_lio::kAnchorEventMapReset | fast_lio::kAnchorEventReferenceChanged |
                                      fast_lio::kAnchorEventLost | fast_lio::kAnchorEventRunawayTrip |
                                      fast_lio::kAnchorEventTimeEpoch | fast_lio::kAnchorEventScanHole |
                                      fast_lio::kAnchorEventImuCoverageGap | fast_lio::kAnchorEventImuBridge;
  if (bit & breaksLostRecovery)
    fast_lio::abandonLostRecovery(&lost_recovery, lost_recovery_params);
  if (reference_health_en) {
    std::uint32_t cause = 0, hard = 0;
    switch (reason) {
      case fast_lio::AnchorTraceReason::kScanHole:
        cause = fast_lio::kReferenceScanHole;
        break;
      case fast_lio::AnchorTraceReason::kImuCoverageGap:
        cause = fast_lio::kReferenceImuCoverage;
        break;
      case fast_lio::AnchorTraceReason::kImuBridge:
        cause = fast_lio::kReferenceImuBridge;
        break;
      case fast_lio::AnchorTraceReason::kMapReset:
        cause = fast_lio::kReferenceMapReset;
        hard = 4;
        break;
      case fast_lio::AnchorTraceReason::kTimeEpoch:
        cause = fast_lio::kReferenceSourceFault;
        hard = 64;
        break;
      case fast_lio::AnchorTraceReason::kLost:
        hard = 1;
        break;
      default:
        break;
    }
    if (cause != 0 || hard != 0)
      reference_progress.negative(cause, hard);
  }
  fast_lio::saturatingIncrement(counter);
  endParkedStreak();
  if (anchor_scan_record_open)
    anchor_scan_record.events |= bit;
  if (report)
    anchorTransition(fast_lio::AnchorTraceEvent::kEvidence, reason, stamp);
}

void beginAnchorScanRecord(double beg, double end)
{
  if (reference_health_en)
    reference_progress.beginScan(recovery_state);
  anchor_final_frozen = false;
  anchor_pending_legacy_observation = false;
  anchor_scan_record = fast_lio::AnchorScanRecord{};
  anchor_scan_record.scan_seq = anchor_scan_seq.next();
  fast_lio::saturatingIncrement(&anchor_trace_counters.scans);
  anchor_scan_record.beg_ns = fast_lio::traceNs(beg);
  anchor_scan_record.end_ns = fast_lio::traceNs(end);
  if (stationary_anchor.last_evidence_end_ns >= 0 && anchor_scan_record.beg_ns >= 0)
    anchor_scan_record.gap_ns = anchor_scan_record.beg_ns - stationary_anchor.last_evidence_end_ns;
  anchor_scan_record.gate_pre = anchorTraceGate(reanchor_gate.state);
  anchor_scan_record.trust_pre = recovery_state.trust;
  anchor_scan_record_open = true;
}

bool anchorTraceWantsCovariance()
{
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterAnchorScanRecord)
    return true;
#endif
  return zupt_anchor_trace_en;
}

template<class Matrix>
void copyPoseCovarianceBlock(const Matrix & P, double out[36])
{
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c)
      out[r * 6 + c] = P(r, c);
}

void copyTracePose(const V3D & pos, const SO3 & rot, double p[3], double q[4])
{
  const auto & coeffs = rot.coeffs();  // x, y, z, w
  q[0] = coeffs[3];
  q[1] = coeffs[0];
  q[2] = coeffs[1];
  q[3] = coeffs[2];
  for (int i = 0; i < 3; ++i)
    p[i] = pos[i];
}

void copyTraceState(const state_ikfom & x, double p[3], double q[4], double v[3])
{
  copyTracePose(x.pos, x.rot, p, q);
  for (int i = 0; i < 3; ++i)
    v[i] = x.vel[i];
}

// The fields of the scan record that only the per-scan trace prints are observed only with it on: each of the
// helpers below reads the estimator or the lifecycle and writes the record, and decides nothing.
bool anchorScanDetail()
{
  return anchor_scan_record_open && zupt_anchor_trace_en;
}

// The state right after a writer of the anchor step returned. The last writer of the scan wins.
void anchorTraceWriterReturned()
{
  if (!anchorScanDetail())
    return;
  auto & r = anchor_scan_record;
  r.have_act = true;
  copyTraceState(kf.get_x(), r.act_p, r.act_q, r.act_v);
  copyPoseCovarianceBlock(kf.get_P(), r.p_pose_act);
}

// The pose the pin set, or the parked prior pulled toward: the anchor's, as the writer read it.
void anchorTraceTarget()
{
  if (!anchorScanDetail())
    return;
  anchor_scan_record.have_target = true;
  copyTracePose(zupt_anchor_pos, zupt_anchor_rot, anchor_scan_record.target_p, anchor_scan_record.target_q);
}

// The scan's points were handed to the local map: the first build, the rebuild of a reset, the incremental update.
void anchorTraceMapUpdated()
{
  if (anchorScanDetail())
    anchor_scan_record.map_updated = true;
}

// What the scan's lidar measurement update moved: distance and rotation angle between the state before and after.
void anchorTraceMeasurementUpdate(const state_ikfom & before, const state_ikfom & after)
{
  if (!anchorScanDetail())
    return;
  auto & r = anchor_scan_record;
  r.have_update = true;
  r.upd_dp = (after.pos - before.pos).norm();
  r.upd_dq = Eigen::AngleAxisd(before.rot.toRotationMatrix().transpose() * after.rot.toRotationMatrix()).angle();
}

// The state the anchor step starts from.
void anchorTraceBeforeStep(bool guard_degraded, bool degraded)
{
  if (!anchor_scan_record_open)
    return;
  auto & r = anchor_scan_record;
  r.anchor_step = true;
  r.guard_degraded = guard_degraded;
  r.degraded = degraded;
  r.action = fast_lio::AnchorTraceAction::kNone;
  r.have_pre = true;
  copyTraceState(kf.get_x(), r.pre_p, r.pre_q, r.pre_v);
  copyPoseCovarianceBlock(kf.get_P(), r.p_pose_pre);
}

// What the anchor step wrote; `used` is the anchor the pose came from, 0 for none.
void anchorTraceAction(fast_lio::AnchorTraceAction action, std::uint64_t used)
{
  if (!anchor_scan_record_open)
    return;
  anchor_scan_record.action = action;
  if (used != 0)
    anchor_scan_record.anchor_used = used;
  anchor_scan_record.have_cov = true;
}

void anchorTraceAfterStep()
{
  if (!anchor_scan_record_open)
    return;
  auto & r = anchor_scan_record;
  r.zupt_active = zupt_active;
  r.parked = parked_hold_state.active;
  r.outcome = fast_lio::AnchorScanOutcome::kAfterAnchorStep;
  if ((r.events & fast_lio::kAnchorEventImuCoverageGap) == 0u)
    anchor_trace_gap_run = false;
  if ((r.events & fast_lio::kAnchorEventImuBridge) == 0u)
    anchor_trace_bridge_run = false;
}

// The state after the scan's last writer: what /Odometry carries.
void captureAnchorPost()
{
  if (!anchor_scan_record_open)
    return;
  auto & r = anchor_scan_record;
  r.have_post = true;
  copyTraceState(state_point, r.post_p, r.post_q, r.post_v);
  if (!r.have_cov)
    return;
  const auto P = kf.get_P();
  copyPoseCovarianceBlock(P, r.p_pose_post);
  if (!anchorTraceWantsCovariance())
    return;
  r.p_finite = P.allFinite();
  using Covariance = std::decay_t<decltype(P)>;
  const Covariance symmetric = 0.5 * (P + P.transpose());
  const Eigen::SelfAdjointEigenSolver<Covariance> eig(symmetric, Eigen::EigenvaluesOnly);
  r.p_min_eig = eig.eigenvalues().minCoeff();
  const double spectral = eig.eigenvalues().cwiseAbs().maxCoeff();
  r.p_sym = (P - P.transpose()).cwiseAbs().maxCoeff() / std::max(1.0, spectral);
}

void finishAnchorScanRecord()
{
  if (!anchor_scan_record_open)
    return;
  if (!anchor_final_frozen) {
    anchor_final_snapshot = captureFinalFrontend(false, true, static_cast<double>(anchor_scan_record.end_ns) * 1e-9);
    anchor_final_frozen = true;
    reference_pending = reference_health_en;
  }
  anchor_scan_record_open = false;
  auto & r = anchor_scan_record;
  r.anchor_id = stationary_anchor.id;
  r.anchor_valid = stationary_anchor.state == fast_lio::AnchorState::kValid;
  r.anchor_created_ns = stationary_anchor.created_ns;
  r.anchor_state = stationary_anchor.state;
  r.episode = stationary_anchor.episode;
  r.ref_gen = stationary_anchor.reference_generation;
  const auto & reference = anchor_final_snapshot.reference;
  r.trust_post = reference.trust == 1   ? fast_lio::RecoveryTrust::kLocalQualified
                 : reference.trust == 2 ? fast_lio::RecoveryTrust::kPending
                                        : fast_lio::RecoveryTrust::kUnanchored;
  r.recovery_episode = reference.recoveryEpisode;
  r.recovery_start_ns = reference.hasOriginalStart ? reference.originalStartNs : -1;
  r.recovery_deadline_ns = reference.hasOriginalDeadline ? reference.originalDeadlineNs : -1;
  r.startup_reference = reference.startupReference == 1   ? fast_lio::StartupReference::kCold
                        : reference.startupReference == 2 ? fast_lio::StartupReference::kEstablished
                                                          : fast_lio::StartupReference::kExcluded;
  r.reference_health = fast_lio::referenceHealthTrace(reference);
  const auto & health = anchor_final_snapshot.health;
  r.eff_degraded = health.effective_degraded;
  r.static_hold = health.reported_static_hold;
  r.gate_post = anchorTraceGate(reanchor_gate.state);
  if (zupt_anchor_trace_en) {
    r.streak = static_cast<std::uint64_t>(std::max(parked_hold_state.parked_scans, 0));
    r.have_anchor_pose = stationary_anchor.state != fast_lio::AnchorState::kEmpty;
    if (r.have_anchor_pose)
      copyTracePose(zupt_anchor_pos, zupt_anchor_rot, r.anchor_p, r.anchor_q);
  }
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterAnchorScanRecord)
    fast_lio_test::afterAnchorScanRecord(r);
#endif
  if (zupt_anchor_trace_en)
    emitAnchorTraceLine(fast_lio::formatAnchorScanRecord(r));
}

// The counter summary (anchor_trace.hpp section 3), once at shutdown.
void emitAnchorTraceCounters()
{
  const bool exhausted = anchor_event_seq.exhausted() || anchor_scan_seq.exhausted() || stationary_anchor.ids_exhausted;
  emitAnchorTraceLine(fast_lio::formatAnchorCounters(anchor_trace_counters, anchor_event_seq.next(), exhausted));
}

// Anchor lifecycle (stationary_anchor_policy.hpp). Every event of a scan that breaks the static evidence comes
// through here before the scan's anchor step, so the step never sees an anchor the scan has already ended.

// The holds end with their anchor, as state and in what the outputs claim: neither the static hold nor the parked
// prior or its streak carries into the next stationary episode.
void releaseAnchorHolds(fast_lio::AnchorTraceReason reason, double stamp)
{
  anchor_usable = false;
  if (zupt_active) {
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[ZUPT] static hold released after %d scans: its anchor ended (%s)",
                zupt_hold_scans,
                fast_lio::anchorTraceReasonName(reason));
    zupt_active = false;
    zupt_hold_scans = 0;
    anchorTransition(fast_lio::AnchorTraceEvent::kPinReleased, reason, stamp);
  }
  if (fast_lio::dropParkedHold(&parked_hold_state)) {
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[ZUPT] parked hold released: its anchor ended (%s)",
                fast_lio::anchorTraceReasonName(reason));
    anchorTransition(fast_lio::AnchorTraceEvent::kParkedPriorReleased, reason, stamp);
  }
}

// Records the end of whatever the lifecycle just dropped (`before` is its state before the drop) and releases the
// holds.
void noteAnchorEnded(fast_lio::AnchorState before, fast_lio::AnchorDrop drop, double stamp)
{
  const fast_lio::AnchorTraceReason reason = fast_lio::anchorTraceReasonOf(drop);
  endParkedStreak();
  if (before == fast_lio::AnchorState::kValid) {
    fast_lio::saturatingIncrement(&anchor_trace_counters.invalidated);
    fast_lio::saturatingIncrement(&anchor_trace_counters.invalidated_by[static_cast<std::size_t>(drop)]);
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[ZUPT-ANCHOR] anchor %llu invalidated (%s): no hold or prior takes a pose from it again",
                static_cast<unsigned long long>(stationary_anchor.id),
                fast_lio::anchorTraceReasonName(reason));
    anchorTransition(fast_lio::AnchorTraceEvent::kInvalidated, reason, stamp);
  } else if (before == fast_lio::AnchorState::kQualifying) {
    fast_lio::saturatingIncrement(&anchor_trace_counters.candidates_dropped);
    anchorTransition(fast_lio::AnchorTraceEvent::kCandidateDropped, reason, stamp);
  }
  releaseAnchorHolds(reason, stamp);
}

// Ends the current candidate or anchor, if any, and marks the scan so that it neither uses nor starts one.
void dropStationaryAnchor(fast_lio::AnchorDrop drop, double stamp)
{
  const fast_lio::AnchorState before = stationary_anchor.state;
  fast_lio::dropAnchor(&stationary_anchor, drop);
  noteAnchorEnded(before, drop, stamp);
}

// The reference the anchors live in was replaced (local map reset, new prior map): the next anchor belongs to the
// new generation.
void changeStationaryReference(fast_lio::AnchorDrop drop, double stamp)
{
  fast_lio::abandonLostRecovery(&lost_recovery, lost_recovery_params);
  if (lost_recovery_params.enabled)
    lost_recovery_map_ready = false;
  const fast_lio::AnchorState before = stationary_anchor.state;
  fast_lio::changeAnchorReference(&stationary_anchor, drop);
  noteAnchorEnded(before, drop, stamp);
  // A replaced or rebuilt reference is no fresh start: a cold start still open ends here.
  fast_lio::revokeRecoveryStartup(&recovery_state);
}

// What the human-readable end of a recovery adds while its entry view is open (reanchor_recovery_entry_view): the
// scans the map took in meanwhile. Empty otherwise, so those lines read as before; the structured trace line
// never carries it.
std::string entryViewNote()
{
#ifdef USE_IVOX
  if (ikdtree.entryViewActive())
    return " view_inserted=" + std::to_string(ikdtree.entryViewFrames());
#endif
  return std::string();
}

void noteRecoveryEvent(const fast_lio::RecoveryEvent & event, std::int64_t stamp_ns)
{
  const double total_s = static_cast<double>(recovery_params.total_ns) * 1e-9;
  const double stable_s = static_cast<double>(recovery_params.stable_ns) * 1e-9;
  if (event.pending) {
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[FE-RECOVERY] unsupported recovery #%llu (%s): %.1fs from the last supported scan to show %.1fs of "
                "unbroken matching on this reference, or the position stops being trusted",
                static_cast<unsigned long long>(recovery_state.episode),
                fast_lio::recoveryCauseName(event.cause),
                total_s,
                stable_s);
    recoveryTransition(fast_lio::RecoveryTraceEvent::kPending, event.cause, stamp_ns);
  }
  if (event.qualified) {
    if (reference_health_en)
      reference_progress.qualified();
    fast_lio::saturatingIncrement(&anchor_trace_counters.recovery_qualified);
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[FE-RECOVERY] recovery #%llu qualified: %.1fs of unbroken matching on the same reference%s",
                static_cast<unsigned long long>(recovery_state.episode),
                stable_s,
                entryViewNote().c_str());
    recoveryTransition(fast_lio::RecoveryTraceEvent::kQualified, event.cause, stamp_ns);
  }
  if (event.unanchored) {
    if (lost_recovery_params.enabled)
      lost_recovery_map_ready = false;
    fast_lio::saturatingIncrement(event.cause == fast_lio::RecoveryCause::kMapReset
                                    ? &anchor_trace_counters.map_reset_unanchored
                                    : &anchor_trace_counters.recovery_expired);
    if (lost_recovery_params.enabled) {
      RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"),
                   "[FE-RECOVERY] UNANCHORED (%s): this coordinate epoch stays untrusted; LOST recovery is armed%s",
                   fast_lio::recoveryCauseName(event.cause),
                   entryViewNote().c_str());
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("laser_mapping"),
                   "[FE-RECOVERY] UNANCHORED (%s): the position is not trusted again in this process and odometry "
                   "stays flagged degraded; only a restart restores normal operation%s",
                   fast_lio::recoveryCauseName(event.cause),
                   entryViewNote().c_str());
    }
    recoveryTransition(fast_lio::RecoveryTraceEvent::kUnanchored, event.cause, stamp_ns);
  }
}

void startAnchorRecovery(fast_lio::RecoveryCause cause, std::int64_t now_ns)
{
  const fast_lio::RecoveryEvent event = fast_lio::startRecovery(&recovery_state, cause, now_ns, recovery_params);
  if (event.pending || event.unanchored)
    fast_lio::saturatingIncrement(&anchor_trace_counters.recovery_started);
  noteRecoveryEvent(event, now_ns);
}

void expireAnchorRecovery(std::int64_t now_ns)
{
  noteRecoveryEvent(fast_lio::expireRecovery(&recovery_state, now_ns), now_ns);
}

// The anchor write of the scan being processed. The pin and the parked prior each note the filter state they are
// about to change; the pin runs on degraded scans and the prior on healthy ones, so a scan has at most one. When a
// later event of the same scan ends the anchor (the runaway watchdog resetting the map, or declaring the front end
// lost), takeBackAnchorWrite() restores that state: the map is rebuilt around, and the scan publishes, what it
// would have had with no usable anchor at the anchor step.
struct AnchorWrite
{
  enum class Writer : std::uint8_t
  {
    kNone,
    kPin,
    kParkedPrior,
  };
  Writer writer = Writer::kNone;
  state_ikfom x;
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P;
};
AnchorWrite anchor_write;

void noteAnchorWrite(AnchorWrite::Writer writer)
{
  anchor_write.writer = writer;
  anchor_write.x = kf.get_x();
  anchor_write.P = kf.get_P();
}

// Zero-only: parked on a degraded scan with no anchor of this episode to hold to. The velocity is all the static
// evidence speaks for; position and attitude stay where the scan has them.
void noteZeroOnlyScan()
{
  if (anchor_scan_record_open)
    anchor_scan_record.zero_only = true;
  fast_lio::saturatingIncrement(&anchor_trace_counters.zero_only_unanchored);
  if (anchor_ever_created)
    fast_lio::saturatingIncrement(&anchor_trace_counters.blocked_stale_use);
}

// Called before the event that ends the anchor is applied. A pin is only ever written on a degraded, parked scan
// with ZUPT on, and without a usable anchor that scan zeroes its velocity and nothing else, so that is what takes
// the pin's place. A healthy scan without a usable anchor applies no prior and keeps whatever else it did. The
// record and the counters follow the state: they describe what the scan publishes.
void takeBackAnchorWrite()
{
  const AnchorWrite::Writer writer = anchor_write.writer;
  if (writer == AnchorWrite::Writer::kNone)
    return;
  anchor_write.writer = AnchorWrite::Writer::kNone;
  kf.change_x(anchor_write.x);
  kf.change_P(anchor_write.P);
  if (anchor_scan_record_open) {
    anchor_scan_record.anchor_used = 0;
    anchor_scan_record.have_target = false;
    anchor_scan_record.have_act = false;
  }
  if (writer == AnchorWrite::Writer::kPin) {
    anchorTraceAction(fast_lio::AnchorTraceAction::kZeroVelocity, 0);
    noteZeroOnlyScan();
    state_ikfom st = kf.get_x();
    st.vel.setZero();
    gravity_align_vel_overwritten = true;
    kf.change_x(st);
    resetVelocityCovariance(zupt_vel_reset_sigma);
  } else if (anchor_scan_record_open) {
    anchor_scan_record.parked_applied = false;
  }
  state_point = kf.get_x();
  if (anchor_scan_record_open && anchor_scan_record.action != fast_lio::AnchorTraceAction::kNone)
    anchorTraceWriterReturned();
}

// The anchor writes that stand are counted once nothing later in the scan can take them back.
void countAnchorWrite()
{
  if (anchor_write.writer == AnchorWrite::Writer::kPin)
    fast_lio::saturatingIncrement(&anchor_trace_counters.pin_applied);
  else if (anchor_write.writer == AnchorWrite::Writer::kParkedPrior)
    fast_lio::saturatingIncrement(&anchor_trace_counters.parked_prior_applied);
}

// The start of every scan taken off the queue, before any of its processing: the scan-hole check, then the
// deadline. Nothing later in the scan can use an anchor, or a trust, that has already run out.
void beginAnchorLifecycleScan(double beg, double end)
{
  if (lifecycle_scan_seq != std::numeric_limits<std::uint64_t>::max())
    ++lifecycle_scan_seq;
  anchor_usable = false;
  anchor_scan_evaluated = false;
  reanchor_lost_this_scan = false;
  anchor_write.writer = AnchorWrite::Writer::kNone;
  const std::int64_t end_ns = fast_lio::traceNs(end);
  const fast_lio::AnchorState before = stationary_anchor.state;
  const fast_lio::AnchorScanStart start =
    fast_lio::beginAnchorScan(&stationary_anchor, fast_lio::traceNs(beg), stationary_anchor_params);
  if (start.hole) {
    anchorEvidence(fast_lio::AnchorTraceReason::kScanHole,
                   fast_lio::kAnchorEventScanHole,
                   &anchor_trace_counters.scan_hole,
                   end,
                   true);
    noteAnchorEnded(before, fast_lio::AnchorDrop::kScanHole, end);
    fast_lio::breakRecoveryRun(&recovery_state);
    startAnchorRecovery(fast_lio::RecoveryCause::kScanHole, end_ns);
  }
  expireAnchorRecovery(end_ns);
}

// A scan that was taken and ended without its static evidence being evaluated is a missing scan to a candidate,
// whose confirming scans must be consecutive (stationary_anchor_policy.hpp). An anchor is left to the scan-gap bound
// and to the IMU window the filter committed for the scan (AnchorScanScope).
void dropUnevaluatedAnchorCandidate(double end)
{
  const fast_lio::AnchorState before = stationary_anchor.state;
  if (fast_lio::dropUnevaluatedCandidate(&stationary_anchor, fast_lio::traceNs(end)))
    noteAnchorEnded(before, fast_lio::AnchorDrop::kScanNotEvaluated, end);
}

// The scan path's bracket around one scan: its record and the start-of-scan checks on the way in; on every way
// out, early returns included, what a scan that never reached the anchor step leaves behind, then the record. Such a
// scan ends a valid anchor when the IMU window the filter committed for it shows motion or cannot show the platform
// parked (stationary_anchor_policy.hpp), and a candidate in any case. The loss the gate declared belongs to this
// scan's outputs only and ends with it.
struct AnchorScanScope
{
  explicit AnchorScanScope(const MeasureGroup & measures) : measures_(measures), end_(measures.lidar_end_time)
  {
    beginAnchorScanRecord(measures.lidar_beg_time, end_);
    beginAnchorLifecycleScan(measures.lidar_beg_time, end_);
  }
  ~AnchorScanScope()
  {
    if (!anchor_scan_evaluated) {
      if (imu_committed_ && stationary_anchor.state == fast_lio::AnchorState::kValid) {
        const fast_lio::StaticEvidenceVerdict verdict = static_evidence_of(measures_, end_, nullptr);
        const fast_lio::AnchorDrop drop =
          fast_lio::unevaluatedConsumedImuDrop(stationary_anchor, imu_committed_, verdict.evidence);
        if (drop != fast_lio::AnchorDrop::kNone)
          dropStationaryAnchor(drop, end_);
      }
      dropUnevaluatedAnchorCandidate(end_);
      if (reference_health_en && imu_committed_)
        reference_progress.negative(fast_lio::kReferenceUnassessed);
    }
    finishAnchorScanRecord();
    if (anchor_pending_legacy_observation && frontend_observation_en)
      publishFrontendObservation(false, true, end_);
    publishPendingReference();
    reanchor_lost_this_scan = false;
    anchor_write.writer = AnchorWrite::Writer::kNone;
  }
  AnchorScanScope(const AnchorScanScope &) = delete;
  AnchorScanScope & operator=(const AnchorScanScope &) = delete;

  // The scan's IMU batch was settled with `outcome`. Its disposition, not whether the filter produced a cloud, says
  // whether the window is spent.
  void noteImuSettlement(const fast_lio::ImuProcessOutcome & outcome) noexcept
  {
    imu_committed_ = outcome.disposition == fast_lio::ImuDisposition::kCommitted;
  }

private:
  const MeasureGroup & measures_;  // the scan's batch; the next sync refills it only after the scope has ended
  double end_;
  bool imu_committed_ = false;
};

// A scan dropped before processing still gets its record, so the trace accounts for every scan taken.
// `unevaluated` says the scan is a new one that ended without its static evidence being evaluated; a repeated or
// old scan is not, and a scan whose drop ends candidate and anchor alike needs no second word.
void recordDroppedAnchorScan(fast_lio::AnchorScanOutcome outcome, double beg, double end, bool unevaluated)
{
  fast_lio::abandonLostRecovery(&lost_recovery, lost_recovery_params);
  beginAnchorScanRecord(beg, end);
  anchor_scan_record.outcome = outcome;
  if (unevaluated)
    dropUnevaluatedAnchorCandidate(end);
  finishAnchorScanRecord();
}

// A scan still in transit gets this long to arrive before the IMU-driven check below calls its stretch a hole.
constexpr std::int64_t kAnchorIdleSettleMinNs = 500000000;

// Every IMU sample the front end takes is input that advances, and the deadline is measured by it whether or not a
// scan is queued: the first sample at or past it latches, at that sample's time, and a scan still waiting for its
// IMU is processed under the latch. A queued scan gets no grace: the deadline holds in processing order. A scan
// hole is not known that soon, because the scan that would close it may still be in transit, so the hole is called
// only while no scan is queued, once the stretch has outlasted the gap bound by the settle allowance. The Pending it
// opens is timed from the last supported scan like any other, and has run out at once when that was long enough
// ago. The next scan re-checks both before anything uses an anchor. No pose is made up: /Odometry changes at the
// next published scan. Without a Pending episode the expiry writes nothing and changes nothing, so a sample taken
// while a scan is queued leaves no trace; with no scan queued the hole check can still open a Pending, as above.
void checkAnchorEvidenceOnImu(double imu_stamp, bool scan_waiting)
{
  const std::int64_t now_ns = fast_lio::traceNs(imu_stamp);
  if (now_ns < 0)
    return;
  anchor_idle_path = true;
  // Serial with timer_callback only while both callbacks stay in the node's default mutually exclusive group.
  expireAnchorRecovery(now_ns);
  const std::int64_t settle_ns = std::max(stationary_anchor_params.max_scan_gap_ns, kAnchorIdleSettleMinNs);
  const std::int64_t last_ns = stationary_anchor.last_evidence_end_ns;
  if (!scan_waiting && last_ns >= 0 && now_ns >= settle_ns &&
      now_ns - settle_ns - last_ns > stationary_anchor_params.max_scan_gap_ns) {
    if (stationary_anchor.state != fast_lio::AnchorState::kEmpty)
      dropStationaryAnchor(fast_lio::AnchorDrop::kScanHole, imu_stamp);
    fast_lio::breakRecoveryRun(&recovery_state);
    startAnchorRecovery(fast_lio::RecoveryCause::kScanHole, now_ns);
    expireAnchorRecovery(now_ns);
  }
  anchor_idle_path = false;
}

// The scan being processed carried a stretch the IMU did not observe. Its pose crossed that stretch on the motion
// model alone: with the platform policy on, an unsupported recovery starts at the last supported scan, and this
// scan is not one of the matching scans that could end it.
void startUnobservedImuRecovery(fast_lio::RecoveryCause cause, double end)
{
  fast_lio::excludeRecoveryScan(&recovery_state, lifecycle_scan_seq);
  startAnchorRecovery(cause, fast_lio::traceNs(end));
}

void markImuCoverageGap(double end)
{
  ++imu_coverage_drop_count;
  anchorEvidence(fast_lio::AnchorTraceReason::kImuCoverageGap,
                 fast_lio::kAnchorEventImuCoverageGap,
                 &anchor_trace_counters.imu_coverage_gap,
                 end,
                 !anchor_trace_gap_run);
  anchor_trace_gap_run = true;
  // The stretch was not observed: the evidence behind the anchor is broken whether or not the platform moved.
  dropStationaryAnchor(fast_lio::AnchorDrop::kImuCoverageGap, end);
  startUnobservedImuRecovery(fast_lio::RecoveryCause::kImuCoverageGap, end);
  fast_lio::updateReanchorGate(&reanchor_gate, true, 0, 0.0, end, reanchor_params);
  flio_map_frozen = true;
  flio_degraded_odom = true;
}

// An IMU batch whose samples were lost (dropped as invalid, or the input failing closed) leaves the scan's stretch
// unobserved, as a coverage gap does. A batch kept for the next scan is not lost: that scan's evidence covers it.
// The re-anchor gate is left as it is: the scan ends here, and the gate judges the scans that register.
void dropAnchorForLostImu(double end)
{
  dropStationaryAnchor(fast_lio::AnchorDrop::kImuCoverageGap, end);
  startUnobservedImuRecovery(fast_lio::RecoveryCause::kImuCoverageGap, end);
}
std::uint64_t imu_gap_bridged_count = 0;
void markImuGapBridged(double end)
{
  fast_lio::updateReanchorGate(&reanchor_gate, true, 0, 0.0, end, reanchor_params);
}

bool finishImuBatch(const fast_lio::ImuProcessOutcome & outcome, fast_lio::TimeAdmissionAccounting * observer = nullptr)
{
  std::unique_lock<std::mutex> lock(mtx_buffer);
  const auto sequence = pending_imu_token.sequence;
  const auto epoch = pending_imu_token.epoch;
  const auto samples = pending_imu_token.prefix.size();
  const auto firstIdentity = samples == 0 ? 0 : pending_imu_token.prefix.front().identity;
  const auto lastIdentity = samples == 0 ? 0 : pending_imu_token.prefix.back().identity;
  const auto begin = Measures.lidar_beg_time;
  const auto end = Measures.lidar_end_time;
#ifdef FASTLIO_TEST_HOOKS
  last_imu_token_for_test = pending_imu_token;
#endif
  const auto confirmation = imu_fifo_transaction.settle(imu_buffer, pending_imu_token, outcome.disposition);
  if (observer) {
    observer->add(observer->confirmations[static_cast<std::size_t>(confirmation)]);
    observer->add(observer->settle_reasons[static_cast<std::size_t>(outcome.reason)]);
    observer->add(observer->settle_dispositions[static_cast<std::size_t>(outcome.disposition)]);
    observer->add(observer->settle_samples, samples);
    observer->last_token = sequence;
    observer->last_epoch = epoch;
    observer->last_settle_known = true;
    observer->settle_fc_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
  }
  const auto reason = static_cast<std::size_t>(outcome.reason);
  ++imu_consumption_counters.events.at(reason);
  if (confirmation == fast_lio::ImuConfirmation::kRetained) {
    if (!pending_imu_token.prefix.empty())
      ++imu_consumption_counters.retainedBatches;
    for (std::size_t i = 0; i < pending_imu_token.prefix.size(); ++i) {
      if (!imu_buffer[i].retained) {
        ++imu_consumption_counters.retainedUniqueSamples;
        imu_buffer[i].retained = true;
      }
    }
    observePendingImuAgeLocked();
  } else if (confirmation == fast_lio::ImuConfirmation::kReleased) {
    imu_consumption_counters.committedSamples += pending_imu_token.prefix.size();
  } else if (confirmation == fast_lio::ImuConfirmation::kInvalidDropped ||
             confirmation == fast_lio::ImuConfirmation::kFatal) {
    for (const auto & sample : pending_imu_token.prefix) {
      if (validImuForConsumption(sample.message)) {
        ++imu_consumption_counters.lostValid.at(reason);
        if (confirmation == fast_lio::ImuConfirmation::kFatal) {
          ++imu_consumption_counters.exceptionValidLost;
          ++imu_consumption_counters.faultValidLost;
        } else
          ++imu_consumption_counters.validLostOnReject;
      }
    }
  }
  const bool settled = confirmation == fast_lio::ImuConfirmation::kRetained ||
                       confirmation == fast_lio::ImuConfirmation::kReleased ||
                       confirmation == fast_lio::ImuConfirmation::kInvalidDropped;
  if (!settled) {
    imu_fifo_transaction.failClosed();
    imu_consumption_counters.faultDiscardedLidar += lidar_buffer.size();
    imu_consumption_counters.faultDiscardedImu += imu_buffer.size();
    frontendDrop(lidar_buffer.size() + imu_buffer.size());
    for (const auto & sample : imu_buffer) {
      if (validImuForConsumption(sample.message)) {
        ++imu_consumption_counters.faultValidLost;
        ++imu_consumption_counters.lostValid.at(reason);
        if (outcome.reason == fast_lio::ImuProcessReason::kPartialPropagation)
          ++imu_consumption_counters.exceptionValidLost;
      }
    }
    clearBufferedInputsLocked(observer, fast_lio::AdmissionBucket::kClearFatal);
  }
  const auto pending = imu_buffer.size();
  const auto events = imu_consumption_counters.events.at(reason);
  const auto retained = imu_consumption_counters.retainedBatches;
  lock.unlock();
  if (outcome.reason != fast_lio::ImuProcessReason::kProcessed || !settled) {
    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[IMU-CONSUMPTION] reason=%s events=%lu retained_batches=%lu pending=%zu faulted=%d",
                         fast_lio::imuProcessReasonName(outcome.reason),
                         static_cast<unsigned long>(events),
                         static_cast<unsigned long>(retained),
                         pending,
                         !settled);
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping.imu_transactions"),
                 "[IMU-CONSUMPTION-EVENT] token=%lu epoch=%lu begin=%.9f end=%.9f disposition=%d reason=%s "
                 "confirmation=%d samples=%zu first_id=%lu last_id=%lu pending=%zu",
                 static_cast<unsigned long>(sequence),
                 static_cast<unsigned long>(epoch),
                 begin,
                 end,
                 static_cast<int>(outcome.disposition),
                 fast_lio::imuProcessReasonName(outcome.reason),
                 static_cast<int>(confirmation),
                 samples,
                 static_cast<unsigned long>(firstIdentity),
                 static_cast<unsigned long>(lastIdentity),
                 pending);
  }
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterImuConsumption)
    fast_lio_test::afterImuConsumption(sequence,
                                       static_cast<int>(outcome.disposition),
                                       static_cast<int>(outcome.reason),
                                       samples,
                                       pending,
                                       firstIdentity);
#endif
  return settled;
}

bool sync_packages(MeasureGroup & meas, fast_lio::TimeAdmissionAccounting * observer)
{
  std::unique_lock<std::mutex> lock(mtx_buffer);
  if (input_epoch_guard.faulted() && !input_epoch_handled) {
    input_epoch_handled = true;
    anchorEvidence(fast_lio::AnchorTraceReason::kTimeEpoch,
                   fast_lio::kAnchorEventTimeEpoch,
                   &anchor_trace_counters.time_epoch,
                   input_epoch_fault_stamp,
                   true);
    // A changed clock epoch ends every proof timed in the old one; input stays stopped until a restart.
    dropStationaryAnchor(fast_lio::AnchorDrop::kTimeEpoch, input_epoch_fault_stamp);
    fast_lio::breakRecoveryRun(&recovery_state);
    fast_lio::revokeRecoveryStartup(&recovery_state);
  }
  if (!imu_fifo_transaction.acceptsInput(input_epoch_guard.faulted()))
    return false;
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    if (observer && !lidar_buffer.empty() && imu_buffer.empty())
      observer->add(observer->wait_no_imu);
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed) {
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
    const auto timing = fast_lio::scanTime(
      meas.lidar_beg_time, meas.lidar->points, [](const auto & point) { return point.curvature; }, max_scan_duration_s);
    if (observer)
      observer->add(observer->scan_time_status[static_cast<std::size_t>(timing.status)]);
    if (!timing.valid()) {
      ++invalid_scan_time_count;
      RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping.imu_transactions"),
                   "[IMU-CONSUMPTION-SYNC] reason=invalid_scan_time begin=%.9f",
                   meas.lidar_beg_time);
      frontendDrop();
      RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                           input_time_log_clock,
                           5000,
                           "[INPUT-TIME] invalid scan timing reason=%d count=%lu",
                           static_cast<int>(timing.status),
                           static_cast<unsigned long>(invalid_scan_time_count));
      // Without usable timing the scan cannot be shown to be a repeat of an evaluated one: it is a new scan that
      // nothing evaluated.
      recordDroppedAnchorScan(fast_lio::AnchorScanOutcome::kDroppedInvalidTime,
                              meas.lidar_beg_time,
                              std::numeric_limits<double>::quiet_NaN(),
                              true);
      // A rejected scan is no match: a recovery has to show its stable time again without it.
      fast_lio::breakRecoveryRun(&recovery_state);
      meas.imu.clear();
      popLidarScan(observer, fast_lio::AdmissionBucket::kSyncInvalid);
      return false;
    }
    lidar_end_time = timing.end;
    if (lidar_end_time <= p_imu->lastProcessedEnd()) {
      ++nonadvancing_scan_count;
      RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping.imu_transactions"),
                   "[IMU-CONSUMPTION-SYNC] reason=nonadvancing begin=%.9f end=%.9f",
                   meas.lidar_beg_time,
                   lidar_end_time);
      frontendDrop();
      RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                           input_time_log_clock,
                           5000,
                           "[INPUT-TIME] nonadvancing scan count=%lu",
                           static_cast<unsigned long>(nonadvancing_scan_count));
      // A scan that does not advance past the processed input is a repeated or old sample, not a missing scan.
      recordDroppedAnchorScan(
        fast_lio::AnchorScanOutcome::kDroppedNonadvancing, meas.lidar_beg_time, lidar_end_time, false);
      fast_lio::breakRecoveryRun(&recovery_state);
      meas.imu.clear();
      popLidarScan(observer, fast_lio::AdmissionBucket::kSyncNonadvancing);
      return false;
    }

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time) {
    if (observer)
      observer->add(observer->wait_tail);
    return false;
  }

  // Borrow the exact front prefix; only an explicit consumption outcome releases it.
  meas.imu.clear();
  const std::size_t count = fast_lio::imuPrefixThrough(
    imu_buffer, lidar_end_time, [](const auto & sample) { return get_time_sec(sample->header.stamp); });
  pending_imu_token = imu_fifo_transaction.borrow(imu_buffer, count);
  for (const auto & sample : pending_imu_token.prefix)
    meas.imu.push_back(sample.message);
  popLidarScan(observer, fast_lio::AdmissionBucket::kBorrowed);
  if (meas.imu.empty() && p_imu->bridgesEmptyScan(meas.lidar_end_time)) {
    // No IMU sample in this scan, but the lidar kept recording through the IMU dropout: the IMU-gap prior carries
    // the state across it (imu_gap_empty_scans) and the scan is registered instead of dropped.
    return true;
  }
  if (meas.imu.empty()) {
    const auto confirmation =
      imu_fifo_transaction.settle(imu_buffer, pending_imu_token, fast_lio::ImuDisposition::kUncommitted);
    if (observer) {
      observer->record(fast_lio::AdmissionBucket::kEmptyImu, meas.lidar->size());
      observer->add(observer->confirmations[static_cast<std::size_t>(confirmation)]);
      observer->add(observer->settle_dispositions[static_cast<std::size_t>(fast_lio::ImuDisposition::kUncommitted)]);
      observer->add(observer->settle_reason_not_recorded);
      observer->last_settle_known = true;
      observer->settle_fc_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
      observer->last_token = pending_imu_token.sequence;
      observer->last_epoch = pending_imu_token.epoch;
    }
#ifdef FASTLIO_TEST_HOOKS
    const std::uint64_t emptyWindowSequence = pending_imu_token.sequence;
    const std::size_t emptyWindowSamples = pending_imu_token.prefix.size();
    const std::size_t emptyWindowPending = imu_buffer.size();
    const std::uint64_t emptyWindowFirst = imu_buffer.empty() ? 0 : imu_buffer.front().identity;
    const bool emptyWindowFaulted = imu_fifo_transaction.faulted();
#endif
    ++empty_scan_imu_count;
    frontendDrop();
    beginAnchorScanRecord(meas.lidar_beg_time, meas.lidar_end_time);
    anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kDroppedEmptyImu;
    markImuCoverageGap(meas.lidar_end_time);
    lock.unlock();
#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::afterEmptyImuWindow)
      fast_lio_test::afterEmptyImuWindow(
        emptyWindowSequence, emptyWindowSamples, emptyWindowPending, emptyWindowFirst, emptyWindowFaulted);
#endif
    finishAnchorScanRecord();
    RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping.imu_transactions"),
                 "[IMU-CONSUMPTION-SYNC] reason=empty begin=%.9f end=%.9f token=%lu",
                 meas.lidar_beg_time,
                 meas.lidar_end_time,
                 static_cast<unsigned long>(pending_imu_token.sequence));
    if (frontend_observation_en)
      publishFrontendObservation(false, true, meas.lidar_end_time);
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         input_time_log_clock,
                         5000,
                         "[INPUT-TIME] empty IMU window count=%lu",
                         static_cast<unsigned long>(empty_scan_imu_count));
    return false;
  }
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
bool async_map_en = true;  // gated by env FLIO_ASYNC_MAP ("0" disables)
// Deterministic map-insertion lag (mapping.map_insert_delay_scans, default 0):
// scan k's points enter the map only after scan k+delay has been matched. With
// the async worker the lag exists implicitly (scan k is still being inserted
// while k+1 matches) and helps in the near field — a scan matched against a
// map that already contains its immediate predecessor locks onto it (indoor
// m20: FE-only re-fix along-track −1.0…−1.4 m sync vs +0.5…+1.5 async) — but
// its length depends on scheduling (non-reproducible). This makes the lag
// explicit: same benefit, bit-reproducible.
int map_insert_delay_scans = 0;
fast_lio::DelayedMapInsertion<PointVector> map_insert_pending;
std::mutex map_add_mtx;
std::condition_variable map_add_cv;
PointVector map_add_ds_;        // handoff: points to add WITH downsample
PointVector map_add_nods_;      // handoff: points to add WITHOUT downsample
bool map_add_ready_ = false;    // a job is queued, not yet taken
bool map_add_running_ = false;  // worker is executing a job
bool map_worker_stop_ = false;
std::thread map_worker_thread;
std::atomic<uint64_t> g_async_add_us{0};  // last completed worker Add() wall time (us)
double g_map_join_wait_ms = 0;            // last joinMapAdd() block time (main-only)

#ifdef FASTLIO_TEST_HOOKS
#include <rcl/timer.h>

#include "laser_mapping_test_hooks.hpp"
namespace fast_lio_test
{
std::function<void()> beforeMapAdd;
std::function<void(const fast_lio::TimeAdmissionSnapshot &)> afterAdmissionSnapshot;
std::function<void(const char *, std::size_t)> afterAdmissionSummary;
std::function<void()> beforeResetJoin;
std::function<void()> afterPendingReset;
std::function<void(std::size_t, double)> afterScanPrepared;
std::function<void(std::size_t, double, int)> afterHModelRows;
std::function<void(std::uint64_t, int, int, std::size_t, std::size_t, std::uint64_t)> afterImuConsumption;
std::function<void()> beforeImuProcess;
std::function<int(double)> overrideGuardDegraded;
std::function<void(double, AnchorFixtureState &)> beforeAnchorStep;
std::function<bool(double)> forceRunawayTrip;
std::function<bool(double)> initialBuildTakesNoPoint;
std::function<bool(double)> resetBeforeReference;
std::function<void(bool, bool, bool, std::size_t)> afterPriorMapReset;
std::function<void(std::uint64_t, bool)> afterLostRecoveryCommit;
std::function<void(const fast_lio::AnchorScanRecord &)> afterAnchorScanRecord;
std::function<void(const std::string &)> afterAnchorTraceLine;
std::function<void(std::uint8_t, std::int64_t, const std::array<std::uint8_t, 32> &)> afterPosePublish;
std::function<void(double, int, double, int, double, std::size_t)> afterRecoveryViewMatch;
std::function<void(double, bool)> afterMapInsertion;
std::function<void(double, bool, const Eigen::MatrixXd &, const Eigen::VectorXd &)> afterHModelMatrices;
std::function<void(double, const FilterSnapshot &)> afterLidarUpdate;
std::function<void(std::uint64_t, std::size_t, std::size_t, std::uint64_t, bool)> afterEmptyImuWindow;
std::function<bool(std::array<std::uint8_t, 16> &)> overrideFrontendInstance;
std::function<std::chrono::steady_clock::time_point()> referenceRepeatClock;
std::function<void(const char *, const std::vector<std::string> &)> afterConstructionStage;
std::function<void(const std::vector<EffectiveValue> &)> afterConstruction;
std::function<void(const HModelSelection &)> afterHModelSelection;
std::function<void(const HModelExit &)> afterHModelExit;
std::string correspondenceDumpDir;

FilterSnapshot filterSnapshot()
{
  const state_ikfom & x = kf.get_x();
  const auto & P = kf.get_P();
  static_assert(std::decay<decltype(P)>::type::RowsAtCompileTime == 23 &&
                  std::decay<decltype(P)>::type::ColsAtCompileTime == 23,
                "FilterSnapshot holds a 23x23 error-state covariance");
  FilterSnapshot snapshot;
  for (int i = 0; i < 3; ++i) {
    snapshot.pos[i] = x.pos[i];
    snapshot.offset_T_L_I[i] = x.offset_T_L_I[i];
    snapshot.vel[i] = x.vel[i];
    snapshot.bg[i] = x.bg[i];
    snapshot.ba[i] = x.ba[i];
    snapshot.grav[i] = x.grav[i];
  }
  for (int i = 0; i < 4; ++i) {
    snapshot.rot_xyzw[i] = x.rot.coeffs()[i];
    snapshot.offset_R_L_I_xyzw[i] = x.offset_R_L_I.coeffs()[i];
  }
  for (int r = 0; r < 23; ++r) {
    for (int c = 0; c < 23; ++c) {
      snapshot.P[r * 23 + c] = P(r, c);
    }
  }
  return snapshot;
}

ImuInputSnapshot inspectImuInput()
{
  std::lock_guard<std::mutex> lock(mtx_buffer);
  const auto & counts = imu_consumption_counters;
  return {lidar_buffer.size(),
          imu_buffer.size(),
          counts.rejectedLidar,
          counts.rejectedImu,
          counts.faultDiscardedLidar,
          counts.faultDiscardedImu,
          imu_fifo_transaction.faulted(),
          flio_map_frozen,
          flio_degraded_odom,
          p_imu->lastProcessedEnd(),
          scan_count};
}

int reconfirmLastImuBatch()
{
  std::lock_guard<std::mutex> lock(mtx_buffer);
  return static_cast<int>(
    imu_fifo_transaction.settle(imu_buffer, last_imu_token_for_test, fast_lio::ImuDisposition::kCommitted));
}

// A correspondence dump file: in correspondenceDumpDir when it is set, in the package's Log/ directory otherwise.
static std::string correspondenceDumpFile(const char * name)
{
  return correspondenceDumpDir.empty() ? DEBUG_FILE_DIR(name) : correspondenceDumpDir + "/" + name;
}
}  // namespace fast_lio_test

// Constructor locals, recorded where the constructor last applies each; the effective record copies them at its end.
fast_lio_test::EffectiveRecorder effective_locals_for_test;
#endif

// The divergence guard's verdict for this scan. A test build may replace it to drive the anchor logic through a
// fixed sequence; a production build returns it unchanged.
bool guardVerdictForTest(bool verdict)
{
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::overrideGuardDegraded) {
    const int forced = fast_lio_test::overrideGuardDegraded(lidar_end_time);
    if (forced >= 0)
      return forced != 0;
  }
#endif
  return verdict;
}

// The runaway watchdog's verdict for this scan; a test build may force a trip to reach the late map reset.
fast_lio::RunawayVerdict runawayVerdictForTest(fast_lio::RunawayVerdict verdict)
{
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::forceRunawayTrip && fast_lio_test::forceRunawayTrip(lidar_end_time)) {
    verdict.tripped = true;
    verdict.reason = fast_lio::RunawayReason::kWheel;
  }
#endif
  return verdict;
}

// A test build may set the filter state right before the anchor step: the fixture's stand-in for what
// registration produced. A production build does nothing here.
void applyAnchorFixtureForTest()
{
#ifdef FASTLIO_TEST_HOOKS
  if (!fast_lio_test::beforeAnchorStep)
    return;
  fast_lio_test::AnchorFixtureState fixture;
  fast_lio_test::beforeAnchorStep(lidar_end_time, fixture);
  if (fixture.overwrite) {
    state_ikfom st = kf.get_x();
    st.pos = V3D(fixture.pos[0], fixture.pos[1], fixture.pos[2]);
    st.rot = SO3(fixture.rot_wxyz[0], fixture.rot_wxyz[1], fixture.rot_wxyz[2], fixture.rot_wxyz[3]);
    st.vel = V3D(fixture.vel[0], fixture.vel[1], fixture.vel[2]);
    kf.change_x(st);
  }
  if (fixture.overwrite_cov) {
    auto P = kf.get_P();
    P.setZero();
    for (int i = 0; i < P.rows() && i < 23; ++i)
      P(i, i) = fixture.cov_diag[i];
    kf.change_P(P);
  }
  state_point = kf.get_x();
#endif
}

void mapWorkerLoop()
{
  std::unique_lock<std::mutex> lk(map_add_mtx);
  while (true) {
    map_add_cv.wait(lk, [] { return map_add_ready_ || map_worker_stop_; });
    if (map_worker_stop_ && !map_add_ready_)
      break;
    PointVector toAdd, noDS;
    toAdd.swap(map_add_ds_);
    noDS.swap(map_add_nods_);
    map_add_ready_ = false;
    map_add_running_ = true;
    lk.unlock();

#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::beforeMapAdd)
      fast_lio_test::beforeMapAdd();
#endif
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

void stopMapWorker(fast_lio::TimeAdmissionAccounting * observer)  // drain + join; idempotent
{
  {
    std::lock_guard<std::mutex> lk(map_add_mtx);
    map_worker_stop_ = true;
  }
  map_add_cv.notify_all();
  if (map_worker_thread.joinable())
    map_worker_thread.join();
  if (observer)
    observer->joined();
}

// Hand the two add-lists to the worker. Precondition: worker idle (caller joined this
// scan's predecessor). Moves the vectors so the hot path does no point copy.
void dispatchMapAdd(PointVector & toAdd, PointVector & noDS, fast_lio::TimeAdmissionAccounting * observer)
{
  std::uint64_t downsampled = 0, direct = 0;
  if (observer) {
    if (observer->async_inflight)
      observer->accounting_complete = false;
    downsampled = toAdd.size();
    direct = noDS.size();
  }
  {
    std::lock_guard<std::mutex> lk(map_add_mtx);
    map_add_ds_.swap(toAdd);
    map_add_nods_.swap(noDS);
    map_add_ready_ = true;
  }
  map_add_cv.notify_one();
  if (observer) {
    observer->submit(fast_lio::AdmissionMapKind::kIncrementalDownsampled, downsampled);
    observer->submit(fast_lio::AdmissionMapKind::kIncrementalDirect, direct);
    observer->add(observer->async_dispatches);
    observer->async_inflight = true;
  }
}

// Block until the worker has finished the in-flight Add. Called before the next scan's
// first tree op (lasermap_fov_segment) to restore the single-operator invariant.
void joinMapAdd(fast_lio::TimeAdmissionAccounting * observer = nullptr)
{
  const double t0 = omp_get_wtime();
  std::unique_lock<std::mutex> lk(map_add_mtx);
  map_add_cv.wait(lk, [] { return !map_add_ready_ && !map_add_running_; });
  g_map_join_wait_ms = (omp_get_wtime() - t0) * 1000.0;
  if (observer)
    observer->joined();
}

// Point counts are read only on the opt-in path. Completion means this exact backend call returned, not that
// every presented point survived voxel selection. The original return and exception propagate unchanged.
template<class Points, class Call>
auto observeMapOperation(fast_lio::TimeAdmissionAccounting * observer,
                         fast_lio::AdmissionMapKind kind,
                         Points points,
                         Call call) -> decltype(call())
{
  if (!observer)
    return call();
  const auto sizes = points();  // current scan, prior, reused local map (map_presented units)
  observer->submit(kind, sizes[0], sizes[1], sizes[2]);
  try {
    if constexpr (std::is_void<decltype(call())>::value) {
      call();
      observer->complete(kind);
    } else {
      auto result = call();
      observer->complete(kind);
      return result;
    }
  } catch (...) {
    observer->accounting_complete = false;
    throw;
  }
}

fast_lio::AdmissionCount admissionDelayedPending(fast_lio::TimeAdmissionAccounting & observer)
{
  fast_lio::AdmissionCount pending{};
  for (const auto & batch : map_insert_pending.pending()) {
    observer.count(pending, batch.first.size());
    observer.add(pending.points, batch.second.size());
  }
  return pending;
}

// The recovery callback and the worker-ordering fixture use this exact boundary.
// Join precedes every map read/write and discarding delayed world-frame batches.
bool rebuildLocalMapStorage(int & map_before, fast_lio::TimeAdmissionAccounting * observer)
{
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::beforeResetJoin)
    fast_lio_test::beforeResetJoin();
#endif
  if (async_map_en)
    joinMapAdd(observer);
  // The worker no longer owns an old batch. Pending world-frame points
  // belong to the discarded live history and must not reappear later.
  if (observer) {
    const auto pending = admissionDelayedPending(*observer);
    observer->count(observer->reset_discarded, pending.points, pending.messages);
  }
  map_insert_pending.reset();
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterPendingReset)
    fast_lio_test::afterPendingReset();
#endif
  feats_down_world->resize(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
  }
  bool kept_prior = false;
  map_before = ikdtree.size();
#ifdef USE_IVOX
  if (ikdtree.pinnedVoxels() > 0) {
    // A pinned prior map is the one thing here that is still trustworthy: the local map
    // stopped fitting, the delivered map did not. Recovery drops what this run learned
    // and re-registers against what it was given.
    ikdtree.clearLive();
    observeMapOperation(
      observer,
      fast_lio::AdmissionMapKind::kRecoveryAdd,
      [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
      [] { return ikdtree.Add_Points(feats_down_world->points, true); });
    kept_prior = true;
  } else
#endif
  {
    observeMapOperation(
      observer,
      fast_lio::AdmissionMapKind::kRecoveryBuild,
      [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
      [] { return ikdtree.Build(feats_down_world->points); });
  }
  Localmap_Initialized = false;
  return kept_prior;
}

#ifdef FASTLIO_TEST_HOOKS
// Preserve the existing worker fixture's ABI; those calls have no node-owned observer.
void stopMapWorker()
{
  stopMapWorker(nullptr);
}
void dispatchMapAdd(PointVector & downsampled, PointVector & direct)
{
  dispatchMapAdd(downsampled, direct, nullptr);
}
bool rebuildLocalMapStorage(int & before)
{
  return rebuildLocalMapStorage(before, nullptr);
}
#endif

// nearest: each scan point's map neighbours from the measurement model's last search.
void map_incremental(const std::vector<PointVector> & nearest, fast_lio::TimeAdmissionAccounting * observer)
{
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    /* decide if need add to map */
    if (!nearest[i].empty() && flg_EKF_inited) {
      const PointVector & points_near = nearest[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point;
      mid_point.x =
        floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.y =
        floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.z =
        floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
      float dist = calc_dist(feats_down_world->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS)
          break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add)
        PointToAdd.push_back(feats_down_world->points[i]);
    } else {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  // Preserve each scan's paired add decisions and release exactly the batch
  // that is now delay scans old; the reset path discards its old-world history.
  if (observer) {
    observer->count(observer->staged, PointToAdd.size());
    observer->add(observer->staged.points, PointNoNeedDownsample.size());
  }
  auto ready =
    map_insert_pending.stage(std::move(PointToAdd), std::move(PointNoNeedDownsample), map_insert_delay_scans);
  if (!ready) {
    add_point_size = 0;
    return;
  }
  PointToAdd = std::move(ready->first);
  PointNoNeedDownsample = std::move(ready->second);
  if (observer) {
    observer->count(observer->released, PointToAdd.size());
    observer->add(observer->released.points, PointNoNeedDownsample.size());
  }
#ifdef USE_IVOX
  ikdtree.countEntryViewFrame();
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterMapInsertion)
    fast_lio_test::afterMapInsertion(lidar_end_time, ikdtree.entryViewActive());
#endif
#endif

  if (async_map_en) {
    // Off-thread: the worker calls Add_Points during the inter-scan gap; joined before
    // the next scan's first tree op. add_point_size is known here (sizes); the Add wall
    // time lands in g_async_add_us (see mapWorkerLoop). kdtree_incremental_time is left
    // untouched in this path — it is a main-thread debug stat and the worker must not
    // write plain globals the main thread reads.
    add_point_size = (int)(PointToAdd.size() + PointNoNeedDownsample.size());
    dispatchMapAdd(PointToAdd, PointNoNeedDownsample, observer);
  } else {
    double st_time = omp_get_wtime();
    add_point_size = observeMapOperation(
      observer,
      fast_lio::AdmissionMapKind::kIncrementalDownsampled,
      [&] { return std::array<std::uint64_t, 3>{PointToAdd.size(), 0, 0}; },
      [&] { return ikdtree.Add_Points(PointToAdd, true); });
    observeMapOperation(
      observer,
      fast_lio::AdmissionMapKind::kIncrementalDirect,
      [&] { return std::array<std::uint64_t, 3>{PointNoNeedDownsample.size(), 0, 0}; },
      [&] { return ikdtree.Add_Points(PointNoNeedDownsample, false); });
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
std::atomic<uint64_t> pcd_enqueued{0};     // scans handed to the writer
std::atomic<uint64_t> pcd_accumulated{0};  // scans folded into pcl_wait_save
std::thread pcd_writer_thread;

void pcdWriterLoop()
{
  while (true) {
    PointCloudXYZI::Ptr cloud;
    {
      std::unique_lock<std::mutex> lk(pcd_queue_mtx);
      pcd_queue_cv.wait(lk, [] { return !pcd_queue.empty() || !pcd_writer_running.load(); });
      if (pcd_queue.empty() && !pcd_writer_running.load())
        break;  // drained + asked to stop
      cloud = pcd_queue.front();
      pcd_queue.pop_front();
    }
    if (!cloud || cloud->empty()) {
      pcd_accumulated.fetch_add(1);
      continue;
    }
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
  {
    // Serialize this predicate change with the writer's wait to avoid a lost wakeup.
    std::lock_guard<std::mutex> lk(pcd_queue_mtx);
    if (!pcd_writer_running.exchange(false))
      return;
  }
  pcd_queue_cv.notify_all();
  if (pcd_writer_thread.joinable())
    pcd_writer_thread.join();
}
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
  if (scan_pub_en) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_frame_id;
    pubLaserCloudFull->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
  }

  /**************** save map ****************/
  // Same pause gate: skip accumulation when mapping is disabled, so the
  // exported scans.pcd doesn't contain operator-suppressed frames.
  if (pcd_save_en && mapping_enabled.load(std::memory_order_relaxed)) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
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

  for (int i = 0; i < size; i++) {
    RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
  laserCloudmsg.header.frame_id = body_frame_id;
  pubLaserCloudFull_body->publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

// effect: the measurement model's effective points, body frame.
void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect,
                          const PointCloudXYZI & effect)
{
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&effect.points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
  laserCloudFullRes3.header.frame_id = odom_frame_id;
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_wait_pub += *laserCloudWorld;

  // Voxel-downsample + hard cap so the accumulated /Laser_map stays bounded and
  // renderable. publish_map otherwise just appends a scan per tick and grows
  // without limit. Start at a 0.3 m leaf and coarsen until under ~100k points.
  {
    constexpr size_t kMapPubMaxPoints = 100000;
    float leaf = 0.3f;
    for (int pass = 0; pass < 8; ++pass) {
      pcl::VoxelGrid<PointType> vg;
      vg.setLeafSize(leaf, leaf, leaf);
      vg.setInputCloud(pcl_wait_pub);
      PointCloudXYZI::Ptr ds(new PointCloudXYZI());
      vg.filter(*ds);
      pcl_wait_pub = ds;
      if (pcl_wait_pub->size() <= kMapPubMaxPoints)
        break;
      leaf *= 1.5f;
    }
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
  // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
  laserCloudmsg.header.frame_id = odom_frame_id;
  pubLaserCloudMap->publish(laserCloudmsg);

  // sensor_msgs::msg::PointCloud2 laserCloudMap;
  // pcl::toROSMsg(*featsFromMap, laserCloudMap);
  // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
  // laserCloudMap.header.frame_id = "lidar_odom";
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

void exportLostRecoveryEpoch(const FinalFrontendSnapshot & final)
{
  if (lost_recovery_params.enabled)
    odomAftMapped.twist.covariance[6] = static_cast<double>(final.coordinateEpoch);
}

void logFloorFinal(double scan_end,
                   fast_lio::FloorMode mode,
                   const fast_lio::FloorScanRecord & record,
                   const FinalFrontendSnapshot & final,
                   std::uint64_t measurement_epoch)
{
  if (mode == fast_lio::FloorMode::kOff) {
    return;
  }
  const auto & pose = final.frontend.pose;
  const Eigen::Quaterniond rotation(pose[6], pose[3], pose[4], pose[5]);
  const bool rotation_valid =
    rotation.coeffs().allFinite() && std::isfinite(rotation.squaredNorm()) && rotation.squaredNorm() > 0.0;
  const Eigen::Vector3d direction = rotation_valid
                                      ? Eigen::Vector3d(rotation.toRotationMatrix() * Eigen::Vector3d::UnitX())
                                      : Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const bool same_frame = final.coordinateEpoch == measurement_epoch && !odom_frame_id.empty();
  const auto stamp_ns = rclcpp::Time(get_ros_time(scan_end)).nanoseconds();
  RCLCPP_INFO(rclcpp::get_logger("laser_mapping"),
              "%s",
              fast_lio::floorLine(scan_end, stamp_ns, mode, record, direction, same_frame).c_str());
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
                      std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br,
                      const FinalFrontendSnapshot & final)
{
  odomAftMapped.header.frame_id = odom_frame_id;
  odomAftMapped.child_frame_id = body_frame_id;
  odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
  const auto & pose = final.frontend.pose;
  odomAftMapped.pose.pose.position.x = pose[0];
  odomAftMapped.pose.pose.position.y = pose[1];
  odomAftMapped.pose.pose.position.z = pose[2];
  odomAftMapped.pose.pose.orientation.x = pose[3];
  odomAftMapped.pose.pose.orientation.y = pose[4];
  odomAftMapped.pose.pose.orientation.z = pose[5];
  odomAftMapped.pose.pose.orientation.w = pose[6];
  // One verdict for every encoding of this scan: the covariance inflation, the flags and the typed observation.
  const auto & anchor_health = final.health;
  auto P = kf.get_P();
  odomAftMapped.pose.covariance = fast_lio::makeOdomCovariance(P.block<6, 6>(0, 0), anchor_health.effective_degraded);
  // Front-end health side-channel (Plan C) — written BEFORE the publish so it is
  // same-scan. The twist covariance was always zero here and has no other
  // consumer. Layout contract:
  // LIO-SLAM core pgo_factor_kernels.hpp kFeHealth*Cell.
  if (health_pub_en) {
    odomAftMapped.twist.covariance[5] =
      fast_lio::imuInitDegradedCell(health_pub_en, p_imu->sustainedInitActive(), p_imu->initializationDegraded());
    odomAftMapped.twist.covariance[0] = 1.0;  // sentinel/version
    exportLostRecoveryEpoch(final);
    // Flags say what the front end did about this scan, which is what a
    // consumer must act on: a degraded pose is fresh and self-consistent, so
    // staleness gates downstream pass it through unchanged (dog 2, 2026-09-04).
    const unsigned health_flags = final.reference.legacyFlags;
    odomAftMapped.twist.covariance[1] = static_cast<double>(health_flags);
    if (anchor_scan_record_open)
      anchor_scan_record.flags = static_cast<int>(health_flags);
    odomAftMapped.twist.covariance[7] = scan_obs_along;
    odomAftMapped.twist.covariance[14] = static_cast<double>(effct_feat_num);
    odomAftMapped.twist.covariance[21] = scan_far_frac;
    odomAftMapped.twist.covariance[28] = res_mean_last;
    odomAftMapped.twist.covariance[35] = scan_body_speed;
    // Gravity side-channel (LIO-SLAM core fe_health.hpp kFeGravity*Cell): the S2
    // gravity state, world frame, m/s^2. The world frame was levelled from the
    // start-up accelerometer mean; this is where the filter now puts "down", and
    // downstream attitude references are levelled by it.
    odomAftMapped.twist.covariance[2] = state_point.grav[0];
    odomAftMapped.twist.covariance[3] = state_point.grav[1];
    odomAftMapped.twist.covariance[4] = state_point.grav[2];
  }
  pubOdomAftMapped->publish(odomAftMapped);
#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterPosePublish)
    fast_lio_test::afterPosePublish(
      0, rclcpp::Time(odomAftMapped.header.stamp).nanoseconds(), final.reference.observationId);
#endif
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
  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = odom_frame_id;
  trans.child_frame_id = body_frame_id;
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

void buildAndPublishFrontendObservation(bool completed,
                                        bool scanEvent,
                                        double eventTime,
                                        const FinalFrontendSnapshot & final)
{
  if (!frontend_publisher)
    return;
  // Bounded identifiers fail closed; never silently shorten the source identity.
  if (odom_frame_id.size() > 64 || body_frame_id.size() > 64)
    return;
  const auto & facts = final.frontend;
  const std::int64_t poseStamp = final.poseStamp;
  const auto verdict = fast_lio::frontendVerdict(facts);
  if (!completed && (!frontend_progress.hasInputStamp ||
                     !frontend_progress.idleDue(frontend_progress.inputStamp, frontend_idle_period_ns, verdict)))
    return;
  if (completed) {
    frontend_progress.finish(verdict);
    frontend_progress.hasIdleStamp = facts.stampKnown;
    frontend_progress.idleStamp = poseStamp;
    frontend_progress.previousHealth = verdict.health;
    frontend_progress.previousState = verdict.state;
    frontend_progress.previousActivity = verdict.activity;
    frontend_progress.previousSubstate = verdict.substate;
  }
  frontend_progress.increment(frontend_progress.sample);
  fast_lio_interfaces::msg::FrontendObservation message;
  message.schema_version = 1;
  message.identity_version = 1;
  message.has_source_instance = frontend_has_instance;
  message.source_instance = frontend_instance;
  message.has_sample_sequence = frontend_progress.valid;
  message.sample_sequence = frontend_progress.sample;
  message.clock_domain = frontend_clock_domain;
  message.frame_id = odom_frame_id;
  message.child_frame_id = body_frame_id;
  message.has_coordinate_epoch = facts.contextKnown;
  message.coordinate_epoch = facts.contextKnown ? final.coordinateEpoch : 0;
  message.has_input_stamp = frontend_progress.hasInputStamp;
  message.input_stamp_ns = frontend_progress.inputStamp;
  message.has_pose = completed;
  message.has_observation_stamp = completed && facts.stampKnown;
  message.observation_stamp_ns = message.has_observation_stamp ? poseStamp : 0;
  message.has_effective_observation_stamp = !facts.blocked && (completed ? facts.stampKnown : message.has_input_stamp);
  message.effective_observation_stamp_ns =
    message.has_effective_observation_stamp ? (completed ? message.observation_stamp_ns : message.input_stamp_ns) : 0;
  if (scanEvent && !facts.blocked) {
    message.has_effective_observation_stamp =
      fast_lio::frontendSecondsStamp(eventTime, message.effective_observation_stamp_ns);
  }
  if (completed) {
    message.pose.pose = odomAftMapped.pose.pose;
    message.has_observation_id = final.reference.hasObservationId;
    message.observation_id = final.reference.observationId;
  }
  // Neither legacy twist (unfilled) nor the EKF attitude-error covariance has a verified wire contract.
  message.has_pose_covariance = false;
  message.has_twist = false;
  message.has_twist_covariance = false;
  message.output_validity = verdict.validity;
  message.axis_validity_bits = verdict.axes;
  message.estimator_state = verdict.state;
  message.activity = verdict.activity;
  message.fe_substate = verdict.substate;
  message.has_guard_reason_bits = true;
  message.guard_reason_bits = verdict.guards;
  message.has_health_flags = true;
  message.health_flags = verdict.health;
  // Per-scan metrics cannot be borrowed by an idle sample.
  if (completed) {
    message.has_observability_along = fast_lio::frontendHasObservabilityAlong(ekf_iters, scan_obs_along);
    if (message.has_observability_along)
      message.observability_along = scan_obs_along;
    message.has_effective_feature_count = fast_lio::frontendHasEffectiveFeatureCount(ekf_iters, effct_feat_num);
    if (message.has_effective_feature_count)
      message.effective_feature_count = static_cast<std::uint32_t>(effct_feat_num);
    message.has_far_point_fraction = std::isfinite(scan_far_frac) && scan_far_frac >= 0.0 && scan_far_frac <= 1.0;
    if (message.has_far_point_fraction)
      message.far_point_fraction = scan_far_frac;
    message.has_mean_residual = fast_lio::frontendHasMeanResidual(ekf_iters, effct_feat_num, res_mean_last);
    if (message.has_mean_residual)
      message.mean_residual_m = res_mean_last;
    message.has_body_speed = std::isfinite(scan_body_speed) && scan_body_speed >= 0.0;
    if (message.has_body_speed)
      message.body_speed_mps = scan_body_speed;
    message.has_gravity =
      std::isfinite(state_point.grav[0]) && std::isfinite(state_point.grav[1]) && std::isfinite(state_point.grav[2]);
    if (message.has_gravity)
      message.gravity_world_mps2 = {{state_point.grav[0], state_point.grav[1], state_point.grav[2]}};
  }
  message.has_input_sequence = frontend_progress.valid;
  message.input_sequence = frontend_progress.input;
  message.has_completion_sequence = frontend_progress.valid;
  message.completion_sequence = frontend_progress.completion;
  message.has_output_sequence = frontend_progress.valid;
  message.output_sequence = frontend_progress.output;
  // The legacy preprocessing/IMU initialization paths do not expose a complete
  // per-input discard ledger. Partial node-level accounting cannot certify a total.
  message.has_dropped_count = false;
  message.dropped_count = 0;
  {
    std::lock_guard<std::mutex> lock(mtx_buffer);
    const auto pending = static_cast<std::uint64_t>(imu_buffer.size()) + lidar_buffer.size();
    message.has_queue_count = pending <= std::numeric_limits<std::uint32_t>::max();
    if (message.has_queue_count)
      message.queue_count = static_cast<std::uint32_t>(pending);
  }
  frontend_publisher->publish(message);
#ifdef FASTLIO_TEST_HOOKS
  if (completed && fast_lio_test::afterPosePublish)
    fast_lio_test::afterPosePublish(1, message.observation_stamp_ns, message.observation_id);
#endif
}

void publishFrontendObservation(bool completed, bool scanEvent, double eventTime) noexcept
{
  if (!completed && scanEvent && anchor_scan_record_open) {
    anchor_pending_legacy_observation = true;
    return;  // the scope must first settle committed-IMU residual negatives
  }
  auto publish = [&](const FinalFrontendSnapshot & final) {
    frontend_publication_guard.run([&] { buildAndPublishFrontendObservation(completed, scanEvent, eventTime, final); },
                                   [](const char * reason) {
                                     frontend_observation_en = false;
                                     disableReferencePublication(reason);
                                     RCLCPP_ERROR(
                                       rclcpp::get_logger("laser_mapping"),
                                       "Frontend observation disabled for this activation after telemetry failure: %s",
                                       reason);
                                   });
  };
  if ((completed || scanEvent) && anchor_final_frozen) {
    publish(anchor_final_snapshot);
    publishPendingReference();
  } else {
    const auto final = captureFinalFrontend(completed, scanEvent, eventTime);
    publish(final);
    reference_publication_guard.run([&] { buildAndPublishReference(final.reference); }, disableReferencePublication);
  }
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = get_ros_time(lidar_end_time);  // ros::Time().fromSec(lidar_end_time);
  msg_body_pose.header.frame_id = odom_frame_id;

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
  }
}

namespace
{
/// The filter's measurement model: context is the node's FrontendMeasurement, and the scan's values are this
/// translation unit's.
void measurementModel(void * context, state_ikfom & state, esekfom::dyn_share_datastruct<double> & data)
{
  // inputs.prior_covariance binds to the filter's own matrix only while get_P() returns exactly that type.
  static_assert(std::is_same<decltype(kf.get_P()), const fast_lio::PriorCovariance &>::value,
                "PriorCovariance must be the filter's covariance type");
  const fast_lio::MeasurementInputs inputs{kf.get_P(),
                                           ikdtree,
                                           *feats_down_body,
                                           *feats_down_world,
                                           feats_down_size,
                                           lidar_end_time,
                                           scan_far_frac,
                                           lidar_attitude_hold_active,
                                           lidar_attitude_hold_allowed,
                                           degeneracy_debug};
  const fast_lio::MeasurementOutputs outputs{
    effct_feat_num, ekf_iters, res_mean_last, corr_dump_scan_active, corr_dump_iter, fout_corr, fout_corr_iter};
  static_cast<fast_lio::FrontendMeasurement *>(context)->evaluate(state, data, inputs, outputs);
}
}  // namespace

class LaserMappingNode : public rclcpp::Node
{
public:
  LaserMappingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
  {
#ifdef FASTLIO_TEST_HOOKS
    effective_locals_for_test.clear();
#endif
    params_.declareFront(*this);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("declare_front");
#endif
    params_.declareActivation(
      *this, frontend_observation_en, reference_health_en, frontend_clock_domain, frontend_idle_period_ns);
    if (frontend_observation_en)
      startFrontendObservation();
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("typed_outputs");
#endif
    params_.declareBack(*this);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("declare_back");
#endif
    fast_lio::FrontendStartup startup;
    params_.readOutputs(*this, startup, scan_pub_en, dense_pub_en, odom_frame_id, body_frame_id);
    params_.readBacklog(*this, max_buffered_scans, max_scan_backlog_sec, max_imu_backlog_sec);
    params_.readDownsampleAndDegeneracy(*this, degeneracy_debug);
    params_.readConstraints(*this, health_pub_en);
    params_.readCorrespondence(*this);
    params_.readCorrespondenceDump(*this);
    if (params_.values().corr_dump_en)
      openCorrespondenceDump();
    params_.readAttitudeAndGravity(*this, zupt_vel_reset_sigma);
    params_.readImuInit(*this, startup);
    params_.readGuards(*this, release_vel_params);
    params_.readHoleAndZupt(*this,
                            startup,
                            zupt_gyro_thresh,
                            zupt_acc_span_ratio,
                            zupt_wheel_speed_thresh,
                            zupt_wheel_max_age,
                            zupt_require_wheel,
                            zupt_anchor_trace_en,
                            stationary_anchor_params);
    params_.readReanchor(*this, startup, reanchor_params, recovery_params);
    reanchor_healthy_res =
      fast_lio::HealthyScanWindow(static_cast<std::size_t>(startup.reanchor_healthy_window), kReanchorHealthyMinScans);
    params_.loadLostRecovery(*this, health_pub_en, startup.reanchor_total_timeout_sec, lost_recovery_params);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("lost_recovery_loaded");
#endif
    params_.loadRecoveryEntryView(*this);
    reanchor_healthy_eff =
      fast_lio::HealthyScanWindow(static_cast<std::size_t>(startup.reanchor_healthy_window), kReanchorHealthyMinScans);
    params_.readRunaway(*this, zupt_wheel_max_age, zupt_wheel_speed_thresh);
    params_.readCanopyAndGround(*this);
    params_.readDataPaths(*this, map_file_path, root_dir);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("read_policies");
#endif
    params_.readInputs(*this, startup, time_sync_en, time_diff_lidar_to_imu, imu_time_offset);
    params_.readMapAndNoise(*this, startup, filter_size_map_min, cube_len, DET_RANGE);
    params_.readPreprocess(*this, p_pre->blind, p_pre->lidar_type, p_pre->N_SCANS, p_pre->time_unit, p_pre->SCAN_RATE);
    params_.resolveAnchorGap(*this, startup, p_pre->SCAN_RATE, stationary_anchor_params, recovery_params);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("anchor_gap_resolved");
#endif
    params_.readImuCoverage(
      *this, max_scan_duration_s, p_imu->coverage_params, p_imu->gap_params, reanchor_params.enabled);
    params_.readFeatureInputs(*this, p_pre->point_filter_num, p_pre->feature_enabled, runtime_pos_log);
    p_imu->runtime_log_en = runtime_pos_log;                // mirror to ImuProcess so Log/imu.txt is written
    ikd_profile = (getenv("FLIO_IKD_PROFILE") != nullptr);  // [ikd-profile] opt-in via env
    {
      const char * am = getenv("FLIO_ASYNC_MAP");
      if (am && std::string(am) == "0")
        async_map_en = false;
    }
#ifdef USE_IVOX
    {  // option B: iVox map backend (its parameters and their notes: frontend_parameter_specs_map.hpp)
      const fast_lio::IvoxDeclaredValues ivox = params_.declareIvox(*this);
      ikdtree.Init(static_cast<float>(ivox.grid_resolution),
                   ivox.nearby_type,
                   ivox.voxel_capacity,
                   static_cast<std::size_t>(ivox.max_voxels));
      RCLCPP_INFO(this->get_logger(),
                  "[IVOX] map backend = iVox (res=%.2f nearby=%d voxel_cap=%d max_voxels=%d)",
                  ivox.grid_resolution,
                  ivox.nearby_type,
                  ivox.voxel_capacity,
                  ivox.max_voxels);
    }
#endif
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("map_backend");
#endif
    configureDiagnostics(startup);
    params_.readPriorMapPath(*this, prior_map_pcd_);
    // Recovery episodes wait for the first local reference only in a fresh process that builds its local map from
    // its own scans. Decided here, once, from the configured parameter and before any input: the prior-map paths
    // clear prior_map_pcd_ later, which grants nothing.
    fast_lio::configureRecoveryStartup(&recovery_state, prior_map_pcd_.empty());
    params_.readPriorMapAndPose(*this);
    params_.readWheel(*this, startup);
    RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("read_inputs");
#endif

    path.header.stamp = this->get_clock()->now();
    path.header.frame_id = odom_frame_id;

    // /*** variables definition ***/
    // int effect_feat_num = 0, frame_num = 0;
    // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
    // aver_time_solve = 0, aver_time_const_H_time = 0; bool flg_EKF_converged, EKF_stop_flg = 0;

    const double FOV_DEG = (startup.fov_deg + 10.0) > 179.9 ? 179.9 : (startup.fov_deg + 10.0);
    [[maybe_unused]] const double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    downSizeFilterSurf.setLeaf(params_.values().filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    V3D Lidar_T_wrt_IMU(Zero3d);
    M3D Lidar_R_wrt_IMU(Eye3d);
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(startup.extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(startup.extrinR);
    p_imu->init_require_still = startup.imu_init_require_still;
    p_imu->init_still_tol = startup.imu_init_still_tol;
    p_imu->init_still_timeout_s = startup.imu_init_still_timeout_s;
    p_imu->init_static_params = startup.imu_init_static_params;
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(startup.gyr_cov, startup.gyr_cov, startup.gyr_cov));
    p_imu->set_acc_cov(V3D(startup.acc_cov, startup.acc_cov, startup.acc_cov));
    p_imu->set_gyr_bias_cov(V3D(startup.b_gyr_cov, startup.b_gyr_cov, startup.b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(startup.b_acc_cov, startup.b_acc_cov, startup.b_acc_cov));

    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share_context(get_f, df_dx, df_dw, measurementModel, &measurement_, startup.num_max_iterations, epsi);
#ifdef FASTLIO_TEST_HOOKS
    effective_locals_for_test.value("derived:fov:FOV_DEG", FOV_DEG);
    effective_locals_for_test.value("derived:fov:HALF_FOV_COS", HALF_FOV_COS);
    effective_locals_for_test.matrix("derived:extrinsic_and_noise:Lidar_T_wrt_IMU", Lidar_T_wrt_IMU);
    effective_locals_for_test.matrix("derived:extrinsic_and_noise:Lidar_R_wrt_IMU", Lidar_R_wrt_IMU);
    constructionStageForTest("filter_init");
#endif

    openDebugRecords();
    subscribeInputs(startup);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("subscriptions");
#endif
    if (params_.values().degeneracy_floor_mode != fast_lio::FloorMode::kOff) {
      const fast_lio::FloorParams & floor_params = params_.values().degeneracy_floor;
      RCLCPP_INFO(this->get_logger(),
                  "[FLOOR] normal-noise floor %s: false keep rate %g, model scale %g, relative uncertainty %g, "
                  "Schur ridge %g",
                  fast_lio::floorModeName(params_.values().degeneracy_floor_mode),
                  floor_params.alpha,
                  floor_params.kappa,
                  floor_params.rel_uncertainty,
                  params_.values().degeneracy_params.schur_ridge);
    }
    pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
    pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
    pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
    pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("cloud_publishers");
#endif
    if (frontend_observation_en &&
        (!fast_lio::frontendIdentifier(odom_frame_id, 64) || !fast_lio::frontendIdentifier(body_frame_id, 64))) {
      throw std::invalid_argument("frontend observation frame identifiers must contain 1..64 bytes");
    }
    pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
    pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("odometry_publishers");
#endif

    //------------------------------------------------------------------------------------------------------
    auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
    // Offline deterministic replay (env FLIO_OFFLINE_STEP=1, see LIO-SLAM
    // tools/offline_replay): no free-running timer — the replay driver delivers a
    // scan and its IMU, drains the executor, then publishes one step message; each
    // step processes every scan queued at that moment (bounded, so a scan whose IMU
    // coverage is still missing simply waits for the next step). Nothing else in
    // the node consults wall time for the estimate, so the sequence is reproducible.
    {
      const char * os = getenv("FLIO_OFFLINE_STEP");
      offline_step_en_ = (os != nullptr && std::string(os) != "0");
    }
    if (offline_step_en_) {
      subOfflineStep_ = this->create_subscription<std_msgs::msg::Empty>(
        "/lio/offline_step", 100, [this](std_msgs::msg::Empty::UniquePtr) {
          size_t queued = 0;
          {
            std::lock_guard<std::mutex> lk(mtx_buffer);
            queued = lidar_buffer.size();
          }
          for (size_t i = 0; i < queued; ++i) {
            timer_callback();
          }
        });
      RCLCPP_WARN(this->get_logger(), "[OFFLINE_STEP] processing driven by /lio/offline_step (FLIO_OFFLINE_STEP set)");
    } else {
      timer_ =
        rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));
    }
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("scan_timer");
#endif

    // The voxel-capped /Laser_map is a coarse global map, not a live feed, so it
    // publishes slowly (default 0.2 Hz). Period is configurable via publish.map_period_sec.
    const double map_pub_period_sec = params_.readMapPeriod(*this);
    auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(map_pub_period_sec * 1000.0));
    map_pub_timer_ = rclcpp::create_timer(
      this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("map_timer");
#endif

    map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

    // Spin up the async PCD writer so the hot path never blocks on accumulation.
    if (pcd_save_en)
      startPcdWriter();

    // Spin up the off-thread ikd-tree map-update worker (option A). Idempotent.
    if (async_map_en) {
      startMapWorker();
      RCLCPP_INFO(this->get_logger(), "[ASYNC_MAP] off-thread ikd-tree Add enabled (FLIO_ASYNC_MAP!=0).");
    }

    subscribeMappingGates();

#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("config_id");
#endif
    if (reference_health_en) {
      reference_publication_guard.run(
        [&] {
          auto names = this->list_parameters({}, 0).names;
          std::sort(names.begin(), names.end());
          std::string config;
          for (const auto & name : names) {
            // Local observation does not change the estimator/reference contract or its existing identity.
            if (name == "diagnostics.time_admission_counts_enable")
              continue;
            const auto text = this->get_parameter(name).value_to_string();
            config += std::to_string(name.size()) + ":" + name + std::to_string(text.size()) + ":" + text;
          }
          unsigned length = 0;
          const std::string contract = "R1-OBS-1/schema1/b1-startup/trace6";
          if (EVP_Digest(
                contract.data(), contract.size(), reference_contract_id.data(), &length, EVP_sha256(), nullptr) != 1 ||
              length != reference_contract_id.size() ||
              EVP_Digest(config.data(), config.size(), reference_config_id.data(), &length, EVP_sha256(), nullptr) !=
                1 ||
              length != reference_config_id.size())
            throw std::runtime_error("reference contract/config digest unavailable");
        },
        disableReferencePublication);
    }
    RCLCPP_INFO(this->get_logger(), "Node init finished.");
#ifdef FASTLIO_TEST_HOOKS
    constructionStageForTest("constructed");
    reportEffectiveForTest();
#endif
  }

  ~LaserMappingNode()
  {
    // kf outlives the node: unbind measurement_ before any member goes.
    kf.clear_dyn_share_context();
    {
      // The wheel callbacks reach wheel_speed_ through this pointer: detach it before the member goes.
      std::lock_guard<std::mutex> lk(mtx_twist);
      if (wheel_speed_input == &wheel_speed_)
        wheel_speed_input = nullptr;
    }
    emitAnchorTraceCounters();
    {
      std::lock_guard<std::mutex> lock(mtx_buffer);
      std::uint64_t pendingValid = 0;
      std::uint64_t pendingRetained = 0;
      for (const auto & sample : imu_buffer) {
        pendingValid += validImuForConsumption(sample.message) ? 1 : 0;
        pendingRetained += sample.retained ? 1 : 0;
      }
      const auto & counts = imu_consumption_counters;
      RCLCPP_INFO(
        this->get_logger(),
        "[IMU-CONSUMPTION-SUMMARY] retained_batches=%lu retained_unique_samples=%lu committed_samples=%lu "
        "valid_lost_lidar_reject=%lu backlog_valid_lost=%lu retained_backlog_lost=%lu epoch_valid_lost=%lu "
        "exception_valid_lost=%lu duplicate_confirm=%lu duplicate_integration=%lu watermark_regressions=%lu "
        "stale_output_reuse=%lu history_rejected=%lu bridged_scans=%lu empty=%lu nonadvancing=%lu "
        "max_pending_age=%.9f pending_valid=%lu pending_retained=%lu max_imu_backlog_sec=%.9f faulted=%d "
        "fault_valid_lost=%lu fault_discarded_lidar=%lu fault_discarded_imu=%lu rejected_lidar=%lu rejected_imu=%lu",
        static_cast<unsigned long>(counts.retainedBatches),
        static_cast<unsigned long>(counts.retainedUniqueSamples),
        static_cast<unsigned long>(counts.committedSamples),
        static_cast<unsigned long>(counts.validLostOnReject),
        static_cast<unsigned long>(counts.backlogValidLost),
        static_cast<unsigned long>(counts.retainedBacklogLost),
        static_cast<unsigned long>(counts.epochValidLost),
        static_cast<unsigned long>(counts.exceptionValidLost),
        static_cast<unsigned long>(imu_fifo_transaction.duplicateConfirmations()),
        static_cast<unsigned long>(p_imu->integrationLedger().duplicateIntegrations()),
        static_cast<unsigned long>(p_imu->integrationLedger().watermarkRegressions()),
        static_cast<unsigned long>(p_imu->staleOutputReuseCount()),
        static_cast<unsigned long>(counts.events[static_cast<std::size_t>(fast_lio::ImuProcessReason::kHistory)]),
        static_cast<unsigned long>(imu_gap_bridged_count),
        static_cast<unsigned long>(empty_scan_imu_count),
        static_cast<unsigned long>(nonadvancing_scan_count),
        counts.maxPendingAge,
        static_cast<unsigned long>(pendingValid),
        static_cast<unsigned long>(pendingRetained),
        max_imu_backlog_sec,
        imu_fifo_transaction.faulted(),
        static_cast<unsigned long>(counts.faultValidLost),
        static_cast<unsigned long>(counts.faultDiscardedLidar),
        static_cast<unsigned long>(counts.faultDiscardedImu),
        static_cast<unsigned long>(counts.rejectedLidar),
        static_cast<unsigned long>(counts.rejectedImu));
      for (std::size_t i = 0; i < counts.events.size(); ++i)
        RCLCPP_INFO(this->get_logger(),
                    "[IMU-CONSUMPTION-REASON] reason=%s events=%lu valid_lost=%lu",
                    fast_lio::imuProcessReasonName(static_cast<fast_lio::ImuProcessReason>(i)),
                    static_cast<unsigned long>(counts.events[i]),
                    static_cast<unsigned long>(counts.lostValid[i]));
    }
    // Both workers are global std::threads: component destruction must join them before
    // library unload to avoid std::terminate. Match standalone shutdown order: finish map
    // updates first, then drain PCD writes. Both stops are idempotent with main's calls.
    stopMapWorker(admissionObserver());
    finishAdmissionAccounting();
    stopPcdWriter();
    frontend_publisher.reset();
    reference_publisher.reset();
    reference_health_en = false;
    frontend_observation_en = false;
    fout_out.close();
    fout_pre.close();
    fout_jump.close();
    fout_kalman.close();
  }

private:
  using EkfUpdateDiagnostics = esekfom::esekf<state_ikfom, 12, input_ikfom>::UpdateDiagnostics;

  static double
  covarianceBlockNorm(const Eigen::Matrix<double, 23, 23> & covariance, int row, int rows, int col, int cols)
  {
    return covariance.block(row, col, rows, cols).norm();
  }

  static double vectorBlockNorm(const Eigen::Matrix<double, 23, 1> & values, int row, int rows)
  {
    return values.segment(row, rows).norm();
  }

  void writeKalmanChannel(const char * channel,
                          const state_ikfom & before,
                          const state_ikfom & after,
                          const Eigen::Matrix<double, 23, 23> & covarianceBefore,
                          const EkfUpdateDiagnostics & diagnostics,
                          const V3D & measurement,
                          const V3D & prediction,
                          const fast_lio::WheelStep * wheel = nullptr)
  {
    if (!params_.values().kalman_channel_diag_en || !fout_kalman.is_open())
      return;

    const fast_lio::MeasurementTelemetry & measured = measurement_.telemetry();
    const V3D dpos = after.pos - before.pos;
    const V3D dvel = after.vel - before.vel;
    const V3D dba = after.ba - before.ba;
    const V3D dgrav(after.grav[0] - before.grav[0], after.grav[1] - before.grav[1], after.grav[2] - before.grav[2]);
    // the rotation the channel applied, body frame (rad): what the iterated update's last_dx does not carry
    const V3D drot = Log(M3D(before.rot.toRotationMatrix().transpose() * after.rot.toRotationMatrix()));
    const auto & dx = diagnostics.last_dx;
    const auto & gain = diagnostics.gain_row_norm;
    const auto & projection = diagnostics.correction_projection_row_norm;
    const auto & pBefore = diagnostics.covariance_diag_before;
    const auto & pAfter = diagnostics.covariance_diag_after;

    fout_kalman << std::setprecision(17) << 1 << ',' << time_log_counter << ','
                << Measures.lidar_beg_time - first_lidar_time << ',' << channel << ',' << diagnostics.valid << ','
                << diagnostics.converged << ',' << diagnostics.iterations << ',' << diagnostics.measurement_dim << ','
                << diagnostics.residual_norm << ',' << diagnostics.weighted_residual_squared << ',' << effct_feat_num
                << ',' << res_mean_last << ',' << measured.pos_obs_eig_min << ',' << measured.pos_obs_eig_max << ','
                << measured.pos_obs_z_weak << ',' << measurement.x() << ',' << measurement.y() << ',' << measurement.z()
                << ',' << prediction.x() << ',' << prediction.y() << ',' << prediction.z();
    for (int i = 0; i < 23; ++i)
      fout_kalman << ',' << dx(i);
    fout_kalman << ',' << dpos.x() << ',' << dpos.y() << ',' << dpos.z() << ',' << dvel.x() << ',' << dvel.y() << ','
                << dvel.z() << ',' << dba.x() << ',' << dba.y() << ',' << dba.z() << ',' << dgrav.x() << ','
                << dgrav.y() << ',' << dgrav.z() << ',' << vectorBlockNorm(gain, 3, 3) << ','
                << vectorBlockNorm(gain, 12, 3) << ',' << vectorBlockNorm(gain, 18, 3) << ','
                << vectorBlockNorm(gain, 21, 2) << ',' << vectorBlockNorm(projection, 3, 3) << ','
                << vectorBlockNorm(projection, 12, 3) << ',' << vectorBlockNorm(projection, 18, 3) << ','
                << vectorBlockNorm(projection, 21, 2) << ',' << covarianceBlockNorm(covarianceBefore, 21, 2, 0, 3)
                << ',' << covarianceBlockNorm(covarianceBefore, 21, 2, 3, 3) << ','
                << covarianceBlockNorm(covarianceBefore, 21, 2, 12, 3) << ','
                << covarianceBlockNorm(covarianceBefore, 21, 2, 18, 3) << ','
                << covarianceBlockNorm(covarianceBefore, 18, 3, 0, 3) << ','
                << covarianceBlockNorm(covarianceBefore, 18, 3, 3, 3) << ','
                << covarianceBlockNorm(covarianceBefore, 18, 3, 12, 3) << ',' << pBefore(21) << ',' << pBefore(22)
                << ',' << pAfter(21) << ',' << pAfter(22) << ',' << pBefore(18) << ',' << pBefore(19) << ','
                << pBefore(20) << ',' << pAfter(18) << ',' << pAfter(19) << ',' << pAfter(20) << ',' << drot.x() << ','
                << drot.y() << ',' << drot.z() << ',' << measured.lidar_rot_obs_rp_min;
    writeWheelColumns(wheel);
    writeVelocityCovarianceColumns();
    fout_kalman << '\n';
    fout_kalman.flush();
  }

  // The wheel channel's own columns: the fitted gated speed before the scale and the slip correction, the slip
  // regressor phi (nan when the scan had none), the samples fitted with their scatter, and R of the forward and the
  // lateral row (nan without a lateral row). nan on every other channel's row.
  void writeWheelColumns(const fast_lio::WheelStep * wheel)
  {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const bool fitted = wheel != nullptr && wheel->ring.fit.has_value();
    const bool measured = wheel != nullptr && wheel->measurement.has_value();
    fout_kalman << ',' << (fitted ? wheel->ring.fit->v : kNaN) << ','
                << (wheel != nullptr && wheel->phi ? *wheel->phi : kNaN) << ','
                << (fitted ? static_cast<double>(wheel->ring.fit->samples) : kNaN) << ','
                << (fitted ? wheel->ring.fit->residual_rms : kNaN) << ','
                << (measured ? wheel->measurement->noise.var_vx : kNaN) << ','
                << (measured && wheel->measurement->noise.lateral ? wheel->measurement->noise.var_vy : kNaN);
  }

  // The velocity block of the covariance the channel left, upper triangle, world frame, (m/s)^2: every row is written
  // right after its channel's update, so the filter's P here is that update's posterior (the propagation row's prior).
  void writeVelocityCovarianceColumns()
  {
    const auto & P = kf.get_P();
    fout_kalman << ',' << P(12, 12) << ',' << P(12, 13) << ',' << P(12, 14) << ',' << P(13, 13) << ',' << P(13, 14)
                << ',' << P(14, 14);
  }

  // The wheel update after the lidar update (design B.2): the paired samples of the scan's own span fitted at its
  // end, the slip regressor from the raw IMU, one non-iterated update, and one [WHEEL] line per scan.
  void updateWheelSpeed(const fast_lio::FrontendParams & p)
  {
    fast_lio::WheelRingRead ring;
    {
      std::lock_guard<std::mutex> lk(mtx_twist);
      ring = wheel_speed_.read(Measures.lidar_beg_time, lidar_end_time);
    }
    for (const auto & imu : Measures.imu) {
      const auto & a = imu->linear_acceleration;
      wheel_speed_.recordImu(get_time_sec(imu->header.stamp),
                             fast_lio::wheelForward(p.R_robot_to_imu, V3D(a.x, a.y, a.z)));
    }
    const auto angular = fast_lio::angularEndpointFromImu(Measures.imu, lidar_end_time, p_imu->cov_gyr);
    fast_lio::WheelModelParams model;
    model.speed_scale = p.wheel_speed_scale;
    model.slip_k = p.wheel_slip_k;
    model.slip_phi0 = p.wheel_slip_phi0;
    model.slip_window_s = p.wheel_slip_fx_window_s;
    model.sigma_vx_floor = p.wheel_vel_noise_vx;
    model.sigma_vz = p.wheel_vel_noise_vz;
    model.sigma_lever = p.wheel_lever_sigma;
    model.lever_arm = p.wheel_lever_arm;
    const state_ikfom st = kf.get_x();
    const fast_lio::WheelStep step = wheel_speed_.step(
      ring, lidar_end_time, angular, st.rot.toRotationMatrix(), st.vel, st.bg, p.R_robot_to_imu, model, kf.get_P());
    if (step.measurement) {
      // The wheel diagnostics' residual_norm is dimensionless after whitening. z/h/R stay native for the CSV.
      EkfUpdateDiagnostics wheelDiagnostics;
      Eigen::Matrix<double, 23, 23> wheelCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
      if (p.kalman_channel_diag_en)
        wheelCovarianceBefore = kf.get_P();
      kf.update_simple(step.jacobian_white,
                       step.residual_white,
                       step.variances_white,
                       p.kalman_channel_diag_en ? &wheelDiagnostics : nullptr);
      state_point = kf.get_x();  // re-read corrected state
      if (p.kalman_channel_diag_en) {
        writeKalmanChannel("wheel",
                           st,
                           state_point,
                           wheelCovarianceBefore,
                           wheelDiagnostics,
                           step.measurement->z,
                           step.measurement->h,
                           &step);
      }
    }
    RCLCPP_INFO(this->get_logger(), "%s", fast_lio::wheelLine(lidar_end_time, step, wheel_speed_.counts()).c_str());
  }

  // The typed observation outputs, once publish.frontend_observation has been declared on and validated.
  void startFrontendObservation()
  {
    frontend_progress = fast_lio::FrontendProgress{};
    frontend_publication_guard = fast_lio::FrontendPublicationGuard{};
    frontend_instance.fill(0);
    std::size_t randomBytes = 0;
    while (randomBytes < frontend_instance.size()) {
      const auto count = getrandom(frontend_instance.data() + randomBytes, frontend_instance.size() - randomBytes, 0);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        break;
      randomBytes += static_cast<std::size_t>(count);
    }
    frontend_has_instance = randomBytes == frontend_instance.size();
#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::overrideFrontendInstance)
      frontend_has_instance = fast_lio_test::overrideFrontendInstance(frontend_instance);
#endif
    if (!frontend_has_instance) {
      frontend_instance.fill(0);
      RCLCPP_ERROR(this->get_logger(), "Frontend activation UUID unavailable; typed identity remains unknown");
    }
    const auto frontendQos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    frontend_publisher = this->create_publisher<fast_lio_interfaces::msg::FrontendObservation>(
      "/lio_slam/frontend_observation", frontendQos);
    if (reference_health_en) {
      reference_progress = fast_lio::ReferenceHealthProgress{};
      reference_publication_guard = fast_lio::FrontendPublicationGuard{};
      reference_publisher =
        this->create_publisher<fast_lio_interfaces::msg::ReferenceHealth>("/lio_slam/reference_health", frontendQos);
    }
  }

#ifdef FASTLIO_TEST_HOOKS
// A fixture's correspondenceDumpDir replaces Log/: a test build may run on a read-only source tree.
#pragma push_macro("DEBUG_FILE_DIR")
#undef DEBUG_FILE_DIR
#define DEBUG_FILE_DIR(name) fast_lio_test::correspondenceDumpFile(name)
#endif
  void openCorrespondenceDump()
  {
    fout_corr.open(DEBUG_FILE_DIR("corr_dump.csv"), std::ios::out);
    fout_corr << "t,iter,i,eff,range,bx,by,bz,nx,ny,nz,pd2,ax,ay,az,nn0,nn4,n_near,plane_ok,plane_rms,lam0,lam1,lam2,"
                 "d_in,cosv,w,pz,px,py\n";
    fout_corr_iter.open(DEBUG_FILE_DIR("corr_iter.csv"), std::ios::out);
    fout_corr_iter << "t,iter,effct,held,rp_info,ux,uy,uz,searching,degen\n";
    fout_corr_scan.open(DEBUG_FILE_DIR("corr_scan.csv"), std::ios::out);
    fout_corr_scan << "t,tk,iters,effct,hold_scene,hold_info,rp_info,drot_x,drot_y,drot_z,dpos_x,dpos_y,dpos_z,"
                      "far_frac,z_weak,P_rx,P_ry,P_rz,P_px,P_py,P_pz,dvel_x,dvel_y,dvel_z";
    // the prior covariance over [pos(0-2), rot(3-5), vel(12-14)] as a 9x9 block, row-major: the exact
    // iteration-0 update of pos/rot needs the pos-rot block, the vel rows show the blind-window coupling
    for (int a = 0; a < 9; a++) {
      for (int b = 0; b < 9; b++) {
        fout_corr_scan << ",P" << a << b;
      }
    }
    fout_corr_scan << "\n";
    RCLCPP_INFO(this->get_logger(),
                "Correspondence dump enabled: %zu window(s) -> %s",
                params_.values().corr_dump_windows.size(),
                DEBUG_FILE_DIR("corr_dump.csv").c_str());
  }
#ifdef FASTLIO_TEST_HOOKS
#pragma pop_macro("DEBUG_FILE_DIR")
#endif

  void openDebugRecords()
  {
    /*** debug record ***/
    if (runtime_pos_log) {
      const string pos_log_dir = root_dir + "/Log/pos_log.txt";
      positionLog_ = FileHandle(std::fopen(pos_log_dir.c_str(), "w"));
      if (!positionLog_) {
        RCLCPP_WARN(this->get_logger(), "Position logging disabled: failed to open %s", pos_log_dir.c_str());
      }
    }

    // ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    fout_jump.open(DEBUG_FILE_DIR("jump_diag.txt"), ios::out);
    if (params_.values().kalman_channel_diag_en) {
#ifdef FASTLIO_TEST_HOOKS
      // A fixture's correspondenceDumpDir replaces Log/, as for the correspondence dump.
      const std::string kalmanFile = fast_lio_test::correspondenceDumpFile("kalman_channels.csv");
#else
      const std::string kalmanFile = DEBUG_FILE_DIR("kalman_channels.csv");
#endif
      fout_kalman.open(kalmanFile, ios::out);
      fout_kalman << "schema_version,scan_index,time,channel,valid,converged,iterations,measurement_dim,"
                  << "residual_norm,weighted_residual_squared,effct_feat_num,res_mean,pos_obs_eig_min,"
                  << "pos_obs_eig_max,pos_obs_z_weak,z0,z1,z2,h0,h1,h2,"
                  << "dx_pos_x,dx_pos_y,dx_pos_z,dx_rot_x,dx_rot_y,dx_rot_z,"
                  << "dx_ext_rot_x,dx_ext_rot_y,dx_ext_rot_z,dx_ext_t_x,dx_ext_t_y,dx_ext_t_z,"
                  << "dx_vel_x,dx_vel_y,dx_vel_z,dx_bg_x,dx_bg_y,dx_bg_z,dx_ba_x,dx_ba_y,dx_ba_z,"
                  << "dx_grav_t0,dx_grav_t1,dpos_x,dpos_y,dpos_z,dvel_x,dvel_y,dvel_z,"
                  << "dba_x,dba_y,dba_z,dgrav_x,dgrav_y,dgrav_z,"
                  << "gain_rot,gain_vel,gain_ba,gain_grav,projection_rot,projection_vel,projection_ba,projection_grav,"
                  << "p_grav_pos,p_grav_rot,p_grav_vel,p_grav_ba,p_ba_pos,p_ba_rot,p_ba_vel,"
                  << "p_grav_diag0_before,p_grav_diag1_before,p_grav_diag0_after,p_grav_diag1_after,"
                  << "p_ba_x_before,p_ba_y_before,p_ba_z_before,p_ba_x_after,p_ba_y_after,p_ba_z_after,"
                  << "drot_x,drot_y,drot_z,rot_obs_rp_min,"
                  << "wheel_raw_vx,wheel_phi,wheel_samples,wheel_fit_rms,wheel_r_vx,wheel_r_vy,"
                  << "p_vel_xx,p_vel_xy,p_vel_xz,p_vel_yy,p_vel_yz,p_vel_zz\n";
      RCLCPP_INFO(this->get_logger(), "Kalman channel diagnostics enabled: %s (schema v1)", kalmanFile.c_str());
    }
    if (fout_pre && fout_out)
      cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
      cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;
  }

  void subscribeInputs(const fast_lio::FrontendStartup & startup)
  {
    /*** ROS subscribe initialization ***/
    if (p_pre->lidar_type == AVIA) {
      sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        startup.lid_topic, 20, [this](livox_ros_driver2::msg::CustomMsg::UniquePtr msg) {
          livox_pcl_cbk(std::move(msg), lidar_input_lead_);
        });
    } else if (p_pre->lidar_type == ALLPOINT) {
      // /pre/all_point is published RELIABLE with fixed 8MB messages; a small
      // reliable KeepLast queue matches the bag publisher and bounds memory
      // (a few buffered scans) while avoiding silent drops of whole scans.
      rclcpp::QoS shm_qos(rclcpp::KeepLast(5));
      sub_pcl_shm_ = this->create_subscription<shm_msgs::msg::PointCloud8mAndPose>(
        startup.lid_topic, shm_qos, [this](shm_msgs::msg::PointCloud8mAndPose::UniquePtr msg) {
          shm_allpoint_cbk(std::move(msg), lidar_input_lead_);
        });
      RCLCPP_INFO(this->get_logger(),
                  "shm allpoint mode: subscribing %s (shm_msgs/PointCloud8mAndPose)",
                  startup.lid_topic.c_str());
    } else {
      // Diagnostic knob (env FLIO_LIDAR_QDEPTH): a deep best-effort LiDAR queue so
      // the single-thread executor's processing spikes buffer scans instead of
      // dropping them. Eliminating real-time scan drops makes the front-end process
      // an identical scan sequence every run — required to turn the metastable bag2
      // raw-Z into a repeatable, measurable metric. Unset → SensorDataQoS (depth 5),
      // the production default. best_effort stays compatible with the bag publisher.
      const char * qd = std::getenv("FLIO_LIDAR_QDEPTH");
      const int qdepth = (qd && std::atoi(qd) > 0) ? std::atoi(qd) : 0;
      // Default RELIABLE KeepLast(5) to MATCH the lidar_adapter publisher
      // (it publishes canonical /lio/points reliable so no 8MB scan is silently
      // dropped). The matched reliable+volatile+keep_last pair is also what lets
      // the intra-process manager do the zero-copy unique_ptr handoff when this
      // node is co-composed with the adapter. FLIO_LIDAR_QDEPTH keeps the
      // best-effort deep-queue diagnostic override.
      rclcpp::QoS pc_qos = qdepth > 0 ? rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(qdepth))).best_effort()
                                      : rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
      sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        startup.lid_topic, pc_qos, [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) {
          standard_pcl_cbk(std::move(msg), lidar_input_lead_, admissionObserver());
        });
    }
    if (startup.imu_input_type == "fixposition_fpa") {
      auto imu_qos = rclcpp::SensorDataQoS();
      sub_imu_fpa_ = this->create_subscription<fixposition_driver_msgs::msg::FpaImu>(
        startup.imu_topic, imu_qos, [this](fixposition_driver_msgs::msg::FpaImu::UniquePtr msg) {
          imu_fpa_cbk(std::move(msg), imu_input_lead_, admissionObserver());
        });
      sub_imu_bias_ = this->create_subscription<fixposition_driver_msgs::msg::FpaImubias>(
        startup.imu_bias_topic, imu_qos, imu_bias_cbk);
      RCLCPP_INFO(this->get_logger(),
                  "Fixposition IMU direct mode: topic=%s bias_topic=%s imu_time_offset=%.3f",
                  startup.imu_topic.c_str(),
                  startup.imu_bias_topic.c_str(),
                  imu_time_offset);
    } else {
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        startup.imu_topic, 1000, [this](sensor_msgs::msg::Imu::UniquePtr msg) {
          imu_cbk(std::move(msg), imu_input_lead_, admissionObserver());
        });
      RCLCPP_INFO(this->get_logger(), "Standard IMU mode: topic=%s", startup.imu_topic.c_str());
    }
    if (params_.values().wheel_odom_en)
      attachWheelSpeedInput(startup);
    if (params_.values().wheel_odom_en || params_.values().zupt_en || params_.values().parked_hold.enabled) {
      sub_twist_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(startup.wheel_topic, 2000, twist_cbk);
      RCLCPP_INFO(this->get_logger(),
                  "Wheel twist subscribed: topic=%s scale=%.3f (fusion %s, static hold %s, parked hold %s)",
                  startup.wheel_topic.c_str(),
                  params_.values().wheel_speed_scale,
                  params_.values().wheel_odom_en ? "on" : "off",
                  params_.values().zupt_en ? "on" : "off",
                  params_.values().parked_hold.enabled ? "on" : "off");
    }
  }

  // The wheel update's input (wheel_speed_input.hpp): where its variances come from, the pointer the wheel callbacks
  // fill it through, and with wheel_cov_topic the variance subscription. Before the twist subscription exists.
  void attachWheelSpeedInput(const fast_lio::FrontendStartup & startup)
  {
    const fast_lio::FrontendParams & p = params_.values();
    const bool variances_from_topic = !p.wheel_cov_topic.empty();
    wheel_speed_.configure(variances_from_topic, p.wheel_vel_noise_vx, p.wheel_vel_noise_vy);
    {
      std::lock_guard<std::mutex> lk(mtx_twist);
      assert(wheel_speed_input == nullptr && "one LaserMappingNode per process");
      wheel_speed_input = &wheel_speed_;
    }
    if (!variances_from_topic) {
      RCLCPP_INFO(this->get_logger(),
                  "Wheel speed variances: the parameter sigmas, vx %.3f m/s lateral %.3f m/s (no wheel_cov_topic)",
                  p.wheel_vel_noise_vx,
                  p.wheel_vel_noise_vy);
      return;
    }
    // Reliable like the bridge's publisher (rclcpp::QoS(10)), and as deep as the ring: every sample of the longest
    // stall the front end rides through (kWheelSampleCapacity), since an unpaired sample is a lost update.
    const rclcpp::QoS cov_qos = rclcpp::SensorDataQoS(rclcpp::KeepLast(fast_lio::kWheelSampleCapacity)).reliable();
    sub_twist_cov_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      p.wheel_cov_topic, cov_qos, twist_cov_cbk);
    RCLCPP_INFO(this->get_logger(),
                "Wheel speed variances subscribed: topic=%s depth=%zu, paired with %s by stamp; forward sigma "
                "floor %.3f m/s",
                p.wheel_cov_topic.c_str(),
                fast_lio::kWheelSampleCapacity,
                startup.wheel_topic.c_str(),
                p.wheel_vel_noise_vx);
  }

  void subscribeMappingGates()
  {
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
      "/lio_slam/mapping_enabled",
      mapping_qos,
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

    // Where the prior map is, in this run's frame. The relocalization node answers that
    // question and nothing else can: the map's own frame belongs to the run that built it.
    // Latched, because the answer usually predates nothing — it arrives once, seconds in,
    // and a subscriber that joined late still needs it.
    if (!prior_map_pcd_.empty() && params_.values().prior_map_align == fast_lio::PriorMapAlign::kRelocalization) {
      rclcpp::SubscriptionOptions align_sub_opts;
      align_sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
      prior_map_align_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/relocalization/result",
        rclcpp::QoS(1).transient_local(),
        [this](geometry_msgs::msg::TransformStamped::SharedPtr msg) {
          const auto & q = msg->transform.rotation;
          const auto & t = msg->transform.translation;
          Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
          T.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
          T.translation() = Eigen::Vector3d(t.x, t.y, t.z);
          if (!T.matrix().allFinite()) {
            RCLCPP_ERROR(this->get_logger(), "[PRIOR-MAP] relocalization result is not finite - ignored");
            return;
          }
          std::lock_guard<std::mutex> lk(prior_map_align_mtx_);
          prior_map_pending_T_ = T;
          prior_map_pending_ = true;
        },
        align_sub_opts);
      RCLCPP_INFO(
        this->get_logger(), "[PRIOR-MAP] waiting for /relocalization/result before loading %s", prior_map_pcd_.c_str());
    }
  }

#ifdef FASTLIO_TEST_HOOKS
  void constructionStageForTest(const char * stage)
  {
    if (fast_lio_test::afterConstructionStage)
      fast_lio_test::afterConstructionStage(stage, this->list_parameters({}, 0).names);
  }

  // Parameter storage that outlives the constructor, one function per parameter domain, in declaration order
  // (generated from the parameter inventory).
  void effectiveOutputForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("publish.path_en", params_.values().path_en);
    record.value("publish.effect_map_en", params_.values().effect_pub_en);
    record.value("publish.map_en", params_.values().map_pub_en);
    record.value("publish.scan_publish_en", scan_pub_en);
    record.value("publish.dense_publish_en", dense_pub_en);
    record.value("publish.scan_bodyframe_pub_en", params_.values().scan_body_pub_en);
    record.value("publish.odom_frame_id", odom_frame_id);
    record.value("publish.body_frame_id", body_frame_id);
    record.value("health_pub_en", health_pub_en);
    record.value("publish.frontend_observation", frontend_observation_en);
    record.value("publish.reference_health", reference_health_en);
    record.value("publish.frontend_clock_domain", frontend_clock_domain);
  }
  void effectiveMeasurementForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("degeneracy_debug", degeneracy_debug);
    record.value("degeneracy_spectrum_trace", params_.values().degeneracy_spectrum_trace);
    record.value("degeneracy_false_keep_rate", params_.values().degeneracy_floor.alpha);
    record.value("degeneracy_floor_model_scale", params_.values().degeneracy_floor.kappa);
    record.value("degeneracy_floor_rel_uncertainty", params_.values().degeneracy_floor.rel_uncertainty);
    record.value("degeneracy_schur_ridge", params_.values().degeneracy_params.schur_ridge);
    record.value("lidar_attitude_hold_en", params_.values().attitude_hold.enabled);
    record.value("corr_innov_gate_k", params_.values().corr_gate.innovation_k);
    record.value("corr_nn0_max", params_.values().corr_gate.nn0_max);
    record.value("corr_gate_mad_scale", params_.values().corr_gate.mad_scale);
    record.value("corr_gate_rot_sigma_floor_deg", params_.values().corr_gate.rot_sigma_floor_deg);
    record.value("corr_kernel_mad_scale", params_.values().corr_kernel_mad_scale);
    record.value("corr_kernel_min_scale", params_.values().corr_kernel_min_scale);
    record.value("lidar_attitude_hold_far_frac", params_.values().attitude_hold.far_frac_max);
    record.value("lidar_attitude_hold_min_effct", params_.values().attitude_hold.min_effct);
    record.value("lidar_attitude_hold_max_s", params_.values().attitude_hold.max_hold_s);
    record.value("lidar_attitude_hold_max_scan_gap_s", params_.values().attitude_hold.max_scan_gap_s);
    record.value("lidar_attitude_hold_min_rp_info", params_.values().lidar_attitude_hold_min_rp_info);
    record.value("lidar_attitude_hold_z_weak", params_.values().attitude_hold.z_weak_min);
    record.value("lidar_consider_rp_en", params_.values().rp_consider.enabled);
    record.value("lidar_consider_rp_while_engulfed", params_.values().rp_consider.while_engulfed);
    record.value("lidar_consider_rp_post_release_s", params_.values().rp_consider.post_release_s);
    record.value("lidar_consider_rp_min_info", params_.values().rp_consider.min_rp_info);
    record.value("lidar_consider_rp_z_weak_min", params_.values().rp_consider.z_weak_min);
    record.value("lidar_consider_rp_max_s", params_.values().rp_consider.max_consider_s);
    record.value("canopy_near_range", params_.values().canopy_near_range);
    record.value("canopy_ground_z", params_.values().canopy_ground_z);
    record.value("canopy_ground_halfband", params_.values().canopy_ground_halfband);
    record.value("ground_recover_en", params_.values().ground_recover_en);
    record.value("ground_recover_scan_band", params_.values().ground_recover_scan_band);
    record.value("ground_recover_min_inliers", params_.values().ground_recover_min_inliers);
    record.value("ground_recover_max_residual", params_.values().ground_recover_max_residual);
    record.value("ground_recover_sector_tol", params_.values().ground_recover_sector_tol);
    record.value("ground_recover_max_rows", params_.values().ground_recover_max_rows);
    record.value("nn_refresh_gate_en", params_.values().nn_refresh_gate_en);
    record.value("nn_refresh_min_shift", params_.values().nn_refresh_min_shift);
    record.value("nn_refresh_ref_range", params_.values().nn_refresh_ref_range);
  }
  void effectiveInputForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("max_buffered_scans", max_buffered_scans);
    record.value("max_scan_backlog_sec", max_scan_backlog_sec);
    record.value("max_imu_backlog_sec", max_imu_backlog_sec);
    record.value("common.time_sync_en", time_sync_en);
    record.value("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu);
    record.value("common.imu_time_offset", imu_time_offset);
    record.value("preprocess.blind", p_pre->blind);
    record.value("preprocess.lidar_type", p_pre->lidar_type);
    record.value("preprocess.scan_line", p_pre->N_SCANS);
    record.value("preprocess.timestamp_unit", p_pre->time_unit);
    record.value("preprocess.scan_rate", p_pre->SCAN_RATE);
    record.value("preprocess.max_scan_duration_s", max_scan_duration_s);
    record.value("point_filter_num", p_pre->point_filter_num);
    record.value("feature_extract_enable", p_pre->feature_enabled);
  }
  void effectiveDownsampleForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("adaptive_downsample_en", params_.values().adaptive_ds.enabled);
    record.value("adaptive_downsample_budget_sec", params_.values().adaptive_ds.budget_sec);
    record.value("adaptive_downsample_release_sec", params_.values().adaptive_ds.release_sec);
    record.value("adaptive_downsample_step", params_.values().adaptive_ds.step);
    record.value("adaptive_downsample_max_voxel", params_.values().adaptive_ds.max_voxel);
    record.value("adaptive_downsample_raise_after", params_.values().adaptive_ds.raise_after);
    record.value("adaptive_downsample_lower_after", params_.values().adaptive_ds.lower_after);
  }
  void effectiveConstraintsForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("planar_constraint_en", params_.values().planar_constraint_en);
    record.value("planar_constraint_noise", params_.values().planar_constraint_noise);
    record.value("planar_constraint_z_weak_thresh", params_.values().planar_constraint_z_weak_thresh);
    record.value("gravity_align_en", params_.values().gravity_align_en);
    record.value("gravity_align_noise", params_.values().gravity_align_noise);
    record.value("gravity_align_z_weak_thresh", params_.values().gravity_align_z_weak_thresh);
    record.value("gravity_align_accel_tol", params_.values().gravity_align_accel_tol);
    record.value("gravity_align_gyro_tol", params_.values().gravity_align_gyro_tol);
    record.value("gravity_align_window_s", params_.values().gravity_align_window_s);
    record.value("gravity_align_continuous", params_.values().gravity_align_continuous);
    record.value("gravity_align_hold_s", params_.values().gravity_align_hold_s);
    record.value("gravity_align_noise_degraded", params_.values().gravity_align_noise_degraded);
    record.value("gravity_align_mag_ref", params_.values().gravity_align_mag_ref);
    record.value("gravity_align_grav_leak", params_.values().gravity_align_grav_leak);
    record.value("gravity_align_grav_cap_deg", params_.values().gravity_align_grav_cap_deg);
    record.value("gravity_align_grav_freeze_degraded", params_.values().gravity_align_grav_freeze_degraded);
    record.value("gravity_align_lin_accel_comp", params_.values().gravity_align_lin_accel_comp);
    record.value("gravity_align_vel_sigma_scale", params_.values().gravity_align_vel_sigma_scale);
    record.value("gravity_align_lin_accel_source", params_.values().gravity_align_lin_accel_source);
    record.value("gravity_align_withhold_blind_moving", params_.values().gravity_align_withhold_blind_moving);
    record.value("gravity_align_sigma_policy", params_.values().gravity_align_sigma_policy);
    record.value("gravity_align_noise_static", params_.values().gravity_align_noise_static);
    record.value("gravity_align_noise_moving", params_.values().gravity_align_noise_moving);
    record.value("gravity_align_leg_vel_sigma", params_.values().gravity_align_leg_vel_sigma);
    record.value("gravity_align_leg_edge_window_s", params_.values().gravity_align_leg_edge_window_s);
    record.value("wheel_odom_en", params_.values().wheel_odom_en);
    record.value("wheel_speed_scale", params_.values().wheel_speed_scale);
    record.value("wheel_cov_topic", params_.values().wheel_cov_topic);
    record.value("wheel_vel_noise_vx", params_.values().wheel_vel_noise_vx);
    record.value("wheel_vel_noise_vy", params_.values().wheel_vel_noise_vy);
    record.value("wheel_vel_noise_vz", params_.values().wheel_vel_noise_vz);
    record.value("wheel_slip_k", params_.values().wheel_slip_k);
    record.value("wheel_slip_phi0", params_.values().wheel_slip_phi0);
    record.value("wheel_slip_fx_window_s", params_.values().wheel_slip_fx_window_s);
    record.value("wheel_lever_sigma", params_.values().wheel_lever_sigma);
  }
  void effectiveDiagnosticsForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("corr_dump_en", params_.values().corr_dump_en);
    record.value("runtime_pos_log_enable", runtime_pos_log);
    record.value("diagnostics.latency_enable", latency_diag_en);
    record.value("diagnostics.latency_log_period_sec", latency_log_period_sec);
    record.value("diagnostics.kalman_channel_enable", params_.values().kalman_channel_diag_en);
    record.value("diagnostics.time_admission_counts_enable", params_.values().time_admission_counts_enable);
  }
  void effectiveRecoveryForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("guard_vel_reset_sigma", params_.values().guard_vel_reset_sigma);
    record.value("zupt_vel_reset_sigma", zupt_vel_reset_sigma);
    record.value("divergence_guard_en", params_.values().divergence_guard_en);
    record.value("divergence_guard_min_eff", params_.values().divergence_guard_min_eff);
    record.value("guard_engulf_en", params_.values().guard_engulf_en);
    record.value("guard_engulf_far_range", params_.values().guard_engulf_far_range);
    record.value("guard_engulf_far_frac_min", params_.values().guard_engulf_far_frac_min);
    record.value("guard_engulf_streak", params_.values().guard_engulf_streak);
    record.value("guard_engulf_far_frac_release", params_.values().guard_engulf_far_frac_release);
    record.value("guard_engulf_release_scans", params_.values().guard_engulf_release_scans);
    record.value("guard_release_leg_max_age_s", release_vel_params.leg_max_age_s);
    record.value("guard_release_leg_sigma", release_vel_params.leg_sigma);
    record.value("guard_release_keep_sigma", release_vel_params.keep_sigma);
    record.value("guard_release_inflate_en", params_.values().release_inflate.enabled);
    record.value("guard_release_pos_sigma_min", params_.values().release_inflate.pos_sigma_min);
    record.value("guard_release_pos_sigma_rate", params_.values().release_inflate.pos_sigma_rate);
    record.value("guard_release_pos_sigma_max", params_.values().release_inflate.pos_sigma_max);
    record.value("guard_release_rp_sigma_deg", params_.values().release_inflate.rp_sigma_min_deg);
    record.value("guard_release_rp_sigma_rate_deg", params_.values().release_inflate.rp_sigma_rate_deg);
    record.value("guard_release_rp_sigma_max_deg", params_.values().release_inflate.rp_sigma_max_deg);
    record.value("guard_release_yaw_sigma_deg", params_.values().release_inflate.yaw_sigma_deg);
    record.value("guard_hole_en", params_.values().guard_hole_en);
    record.value("guard_hole_min_sec", params_.values().guard_hole_min_sec);
    record.value("guard_hole_max_accel", params_.values().guard_hole_max_accel);
    record.value("guard_hole_reanchor_pos", params_.values().guard_hole_reanchor_pos);
    record.value("divergence_guard_streak", params_.values().divergence_guard_streak);
    record.value("divergence_guard_max_speed", params_.values().divergence_guard_max_speed);
    record.value("zupt_en", params_.values().zupt_en);
    record.value("zupt_gyro_thresh", zupt_gyro_thresh);
    record.value("zupt_acc_span_ratio", zupt_acc_span_ratio);
    record.value("zupt_wheel_speed_thresh", zupt_wheel_speed_thresh);
    record.value("zupt_wheel_max_age", zupt_wheel_max_age);
    record.value("zupt_require_wheel", zupt_require_wheel);
    record.value("zupt_anchor_trace_en", zupt_anchor_trace_en);
    record.value("zupt_parked_hold_en", params_.values().parked_hold.enabled);
    record.value("zupt_parked_hold_scans", params_.values().parked_hold.engage_scans);
    record.value("zupt_hold_pos_noise", params_.values().parked_hold.pos_noise);
    record.value("zupt_hold_rot_noise", params_.values().parked_hold.rot_noise);
    record.value("zupt_hold_vel_noise", params_.values().parked_hold.vel_noise);
    record.value("zupt_hold_deadband_sigma", params_.values().parked_hold.deadband_sigma);
    record.value("zupt_anchor_confirm_scans", stationary_anchor_params.confirm_scans);
    record.value("reanchor_en", reanchor_params.enabled);
    record.value("reanchor_min_eff", reanchor_params.min_eff);
    record.value("reanchor_max_res", reanchor_params.max_res);
    record.value("reanchor_confirm_scans", reanchor_params.confirm_scans);
    record.value("reanchor_timeout_sec", reanchor_params.timeout_sec);
    record.value("reanchor_on_fail", params_.values().reanchor_on_fail);
    record.value("reanchor_max_res_ratio", reanchor_params.max_res_ratio);
    record.value("reanchor_res_floor", reanchor_params.res_floor);
    record.value("reanchor_min_eff_ratio", reanchor_params.min_eff_ratio);
    record.value("reanchor_recovery_entry_view", params_.values().reanchor_recovery_entry_view);
    record.value("runaway_watchdog_en", params_.values().runaway.enabled);
    record.value("runaway_parked_speed_thresh", params_.values().runaway.parked_speed_thresh);
    record.value("runaway_wheel_speed_margin", params_.values().runaway.wheel_speed_margin);
    record.value("runaway_hold_sec", params_.values().runaway.hold_sec);
    record.value("runaway_on_trip", params_.values().runaway_on_trip);
    record.value("input_epoch_on_fault", params_.values().input_epoch_on_fault);
  }
  void effectiveImuForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("imu_coverage.max_gap_s", p_imu->coverage_params.max_gap_s);
    record.value("imu_coverage.max_extrapolation_s", p_imu->coverage_params.max_extrapolation_s);
    record.value("imu_gap_prior_en", p_imu->gap_params.enabled);
    record.value("imu_gap_min_s", p_imu->gap_params.min_gap_s);
    record.value("imu_gap_accel_max", p_imu->gap_params.accel_max);
    record.value("imu_gap_accel_max_z", p_imu->gap_params.accel_max_z);
    record.value("imu_gap_verify_s", params_.values().imu_gap_verify_s);
    record.value("imu_gap_empty_scans", p_imu->gap_params.empty_scans);
    record.value("mapping.extrinsic_est_en", params_.values().extrinsic_est_en);
  }
  void effectiveMapForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("map_file_path", map_file_path);
    record.value("filter_size_surf", params_.values().filter_size_surf_min);
    record.value("filter_size_map", filter_size_map_min);
    record.value("cube_side_length", cube_len);
    record.value("mapping.det_range", DET_RANGE);
    record.value("mapping.map_insert_delay_scans", map_insert_delay_scans);
    record.value("pcd_save.pcd_save_en", pcd_save_en);
    record.value("pcd_save.interval", pcd_save_interval);
    record.value("prior_map_pcd", prior_map_pcd_);
    record.value("prior_map_pinned", params_.values().prior_map_pinned);
    record.value("prior_map_voxel", params_.values().prior_map_voxel);
    record.value("prior_map_freeze", params_.values().prior_map_freeze);
    record.value("initial_pose", params_.values().initial_pose_vec);
  }

  // The state the constructor derives from the parameters, read from where it is applied.
  void effectiveDerivedForTest(fast_lio_test::EffectiveRecorder & record)
  {
    record.value("derived:frontend_activation:frontend_idle_period_ns", frontend_idle_period_ns);
    record.value("derived:frontend_activation:frontend_observation_en", frontend_observation_en);
    record.value("derived:frontend_activation:reference_health_en", reference_health_en);
    record.value("derived:frontend_activation:frontend_clock_domain", frontend_clock_domain);
    record.bytes("derived:config_id:reference_contract_id", reference_contract_id);
    record.bytes("derived:config_id:reference_config_id", reference_config_id);
    record.value("derived:corr_kernel:corr_kernel", params_.values().corr_kernel);
    record.windows("derived:corr_dump:corr_dump_windows", params_.values().corr_dump_windows);
    record.matrix("derived:lever_arm:gravity_align_leg_lever_arm", params_.values().gravity_align_leg_lever_arm);
    record.value("derived:imu_static_limits:p_imu->init_require_still", p_imu->init_require_still);
    record.value("derived:imu_static_limits:p_imu->init_still_tol", p_imu->init_still_tol);
    record.value("derived:imu_static_limits:p_imu->init_still_timeout_s", p_imu->init_still_timeout_s);
    const fast_lio::ImuStaticInitParams & init_static = p_imu->init_static_params;
    record.value("derived:imu_static_limits:p_imu->init_static_params.enabled", init_static.enabled);
    record.value("derived:imu_static_limits:p_imu->init_static_params.minDurationS", init_static.minDurationS);
    record.value("derived:imu_static_limits:p_imu->init_static_params.minSamples", init_static.minSamples);
    record.value("derived:imu_static_limits:p_imu->init_static_params.gyroRmsMax", init_static.gyroRmsMax);
    record.value("derived:imu_static_limits:p_imu->init_static_params.gravityRef", init_static.gravityRef);
    record.value("derived:imu_static_limits:p_imu->init_static_params.gravityTol", init_static.gravityTol);
    record.value("derived:imu_static_limits:p_imu->init_static_params.maxGapS", init_static.maxGapS);
    record.value("derived:release_source:release_vel_params.source", release_vel_params.source);
    record.value("derived:reanchor_relative:reanchor_healthy_res.capacity", reanchor_healthy_res.capacityForTest());
    record.value("derived:reanchor_relative:reanchor_healthy_res.min_count", reanchor_healthy_res.minCountForTest());
    record.value("derived:reanchor_relative:reanchor_healthy_res.size", reanchor_healthy_res.size());
    record.value("derived:reanchor_relative:reanchor_healthy_eff.capacity", reanchor_healthy_eff.capacityForTest());
    record.value("derived:reanchor_relative:reanchor_healthy_eff.min_count", reanchor_healthy_eff.minCountForTest());
    record.value("derived:reanchor_relative:reanchor_healthy_eff.size", reanchor_healthy_eff.size());
    record.value("derived:recovery_budget:recovery_params.total_ns", recovery_params.total_ns);
    record.value("derived:recovery_budget:recovery_params.stable_ns", recovery_params.stable_ns);
    record.value("derived:runaway_derived:runaway_params.wheel_max_age", params_.values().runaway.wheel_max_age);
    record.value("derived:runaway_derived:runaway_params.wheel_moving_thresh",
                 params_.values().runaway.wheel_moving_thresh);
    record.value("derived:data_dir:root_dir", root_dir);
    record.value("derived:anchor_gap:stationary_anchor_params.max_scan_gap_ns",
                 stationary_anchor_params.max_scan_gap_ns);
    record.value("derived:imu_gap:p_imu->gap_params.rate_rp_max", p_imu->gap_params.rate_rp_max);
    record.value("derived:imu_gap:p_imu->gap_params.rate_yaw_max", p_imu->gap_params.rate_yaw_max);
    record.value("derived:runtime_log:p_imu->runtime_log_en", p_imu->runtime_log_en);
#ifdef USE_IVOX
    record.value("derived:ivox:ikdtree.res_", ikdtree.resolutionForTest());
    record.value("derived:ivox:ikdtree.inv_res_", ikdtree.inverseResolutionForTest());
    record.value("derived:ivox:ikdtree.per_voxel_cap_", ikdtree.voxelCapacityForTest());
    record.value("derived:ivox:ikdtree.max_voxels_", ikdtree.maxVoxels());
    record.offsets("derived:ivox:ikdtree.nbr_", ikdtree.neighborOffsetsForTest());
#endif
    record.value("derived:prior_startup:recovery_state.trust", recovery_state.trust);
    record.value("derived:prior_startup:recovery_state.cause", recovery_state.cause);
    record.value("derived:prior_startup:recovery_state.episode", recovery_state.episode);
    record.value("derived:prior_startup:recovery_state.start_ns", recovery_state.start_ns);
    record.value("derived:prior_startup:recovery_state.deadline_ns", recovery_state.deadline_ns);
    record.value("derived:prior_startup:recovery_state.last_supported_end_ns", recovery_state.last_supported_end_ns);
    record.value("derived:prior_startup:recovery_state.stable_begin_ns", recovery_state.stable_begin_ns);
    record.value("derived:prior_startup:recovery_state.stable_last_scan", recovery_state.stable_last_scan);
    record.value("derived:prior_startup:recovery_state.stable_generation", recovery_state.stable_generation);
    record.value("derived:prior_startup:recovery_state.excluded_scan", recovery_state.excluded_scan);
    record.value("derived:prior_startup:recovery_state.map_resets", recovery_state.map_resets);
    record.value("derived:prior_startup:recovery_state.startup_configured", recovery_state.startup_configured);
    record.value("derived:prior_startup:recovery_state.cold_start_eligible", recovery_state.cold_start_eligible);
    record.value("derived:prior_startup:recovery_state.reference_initialized_once",
                 recovery_state.reference_initialized_once);
    record.value("derived:prior_alignment:prior_map_align_", params_.values().prior_map_align);
    record.value("derived:prior_alignment:prior_map_pinned_", params_.values().prior_map_pinned);
    record.value("derived:initial_pose_alias:initial_pose_full_rpy_override_",
                 params_.values().initial_pose_full_rpy_override);
    record.matrix("derived:wheel_extrinsic:R_robot_to_imu", params_.values().R_robot_to_imu);
    record.matrix("derived:lever_arm:wheel_lever_arm", params_.values().wheel_lever_arm);
    record.value("derived:filter_setters:downSizeFilterSurf.leaf", downSizeFilterSurf.leaf());
    record.matrix("derived:filter_setters:downSizeFilterMap.leaf_size", downSizeFilterMap.getLeafSize());
    record.matrix("derived:extrinsic_and_noise:p_imu->Lidar_T_wrt_IMU", p_imu->lidarTranslationForTest());
    record.matrix("derived:extrinsic_and_noise:p_imu->Lidar_R_wrt_IMU", p_imu->lidarRotationForTest());
    record.matrix("derived:extrinsic_and_noise:p_imu->cov_gyr_scale", p_imu->cov_gyr_scale);
    record.matrix("derived:extrinsic_and_noise:p_imu->cov_acc_scale", p_imu->cov_acc_scale);
    record.matrix("derived:extrinsic_and_noise:p_imu->cov_bias_gyr", p_imu->cov_bias_gyr);
    record.matrix("derived:extrinsic_and_noise:p_imu->cov_bias_acc", p_imu->cov_bias_acc);
    record.value("derived:iteration_count:kf.maximum_iter", kf.maximumIterationsForTest());
    std::int64_t map_period_ns = 0;
    if (map_pub_timer_ && rcl_timer_get_period(map_pub_timer_->get_timer_handle().get(), &map_period_ns) == RCL_RET_OK)
      record.value("derived:map_timer:map_pub_timer_.period_ns", map_period_ns);
    else
      record.unavailable("derived:map_timer:map_pub_timer_.period_ns", "int64");
    record.value("derived:lost_recovery:lost_recovery_params.enabled", lost_recovery_params.enabled);
    record.value("derived:lost_recovery:lost_recovery_params.stable_ns", lost_recovery_params.stable_ns);
  }

  // Called once, as the constructor's last statement: a constructor that throws reports its stage events only.
  void reportEffectiveForTest()
  {
    if (!fast_lio_test::afterConstruction)
      return;
    fast_lio_test::EffectiveRecorder record = effective_locals_for_test;
    record.append(params_.localsForTest());
    effectiveOutputForTest(record);
    effectiveMeasurementForTest(record);
    effectiveInputForTest(record);
    effectiveDownsampleForTest(record);
    effectiveConstraintsForTest(record);
    effectiveDiagnosticsForTest(record);
    effectiveRecoveryForTest(record);
    effectiveImuForTest(record);
    effectiveMapForTest(record);
    effectiveDerivedForTest(record);
    fast_lio_test::afterConstruction(record.values());
  }
#endif
  void observeLostRecovery(bool degraded, bool match_ok, std::int64_t begin_ns, std::int64_t end_ns)
  {
    if (!lost_recovery_params.enabled)
      return;
    constexpr auto breaks = fast_lio::kAnchorEventMapReset | fast_lio::kAnchorEventLost |
                            fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventTimeEpoch;
    if (degraded || reanchor_gate.state == fast_lio::ReanchorState::kLost || (anchor_scan_record.events & breaks) ||
        input_epoch_guard.faulted() || required_prior_lost) {
      fast_lio::abandonLostRecovery(&lost_recovery, lost_recovery_params);
      return;
    }
    if (!fast_lio::lostRecoveryArmed(recovery_state, lost_recovery_params))
      return;
    if (!lost_recovery.candidate) {
      if (effct_feat_num < reanchor_params.min_eff)
        return;
      if (!lost_recovery_map_ready) {
        if (!rebuildLocalMap("LOST_RECOVERY"))
          return;
        // This scan matched the old reference. The rebuilt map must earn its own matching run.
        fast_lio::beginLostRecoveryCandidate(
          &lost_recovery, stationary_anchor.reference_generation, lifecycle_scan_seq, -1, lost_recovery_params);
        return;
      }
      fast_lio::beginLostRecoveryCandidate(
        &lost_recovery, stationary_anchor.reference_generation, lifecycle_scan_seq, begin_ns, lost_recovery_params);
    }
#ifdef FASTLIO_TEST_HOOKS
    const auto before = state_point;
    const auto covariance = kf.get_P();
#endif
    if (fast_lio::observeLostRecoveryCandidate(&lost_recovery,
                                               &recovery_state,
                                               lifecycle_scan_seq,
                                               begin_ns,
                                               end_ns,
                                               match_ok,
                                               stationary_anchor.reference_generation,
                                               lost_recovery_params)) {
      if (reference_health_en)
        reference_progress.qualified();
      RCLCPP_WARN(this->get_logger(),
                  "[FE-RECOVERY] committed local coordinate epoch %llu",
                  static_cast<unsigned long long>(lost_recovery.coordinate_epoch));
      recoveryTransition(fast_lio::RecoveryTraceEvent::kEpochCommitted, fast_lio::RecoveryCause::kStable, end_ns);
      lost_recovery_map_ready = false;
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterLostRecoveryCommit) {
        bool preserved = (covariance.array() == kf.get_P().array()).all();
        for (int i = 0; i < 3; ++i)
          preserved = preserved && before.ba[i] == state_point.ba[i] && before.bg[i] == state_point.bg[i] &&
                      before.grav[i] == state_point.grav[i];
        fast_lio_test::afterLostRecoveryCommit(lost_recovery.coordinate_epoch, preserved);
      }
#endif
    }
  }

  /// Opens the entry view once per pending recovery and closes it when the recovery is no longer pending.
  /// Called from the scan loop only, after joinMapAdd() and before this scan's map insertion, so that neither
  /// overlaps an addition. A recovery begun elsewhere (the IMU callback, the coverage checks before the join)
  /// is picked up by the next call. A recovery that begins over a pinned tier does not get a view.
  void syncRecoveryEntryView()
  {
#ifdef USE_IVOX
    const bool pending = recovery_state.trust == fast_lio::RecoveryTrust::kPending;
    if (!pending && ikdtree.entryViewActive())
      ikdtree.EndEntryView();
    if (params_.values().reanchor_recovery_entry_view && pending && entry_view_round_ != recovery_state.episode) {
      entry_view_round_ = recovery_state.episode;
      ikdtree.BeginEntryView();
    }
#endif
  }

  bool recoveryEntryViewServes() const
  {
#ifdef USE_IVOX
    return params_.values().reanchor_recovery_entry_view && recovery_state.trust == fast_lio::RecoveryTrust::kPending &&
           ikdtree.entryViewActive();
#else
    return false;
#endif
  }

  /// The re-anchor gate's match test for the recovery. While the entry view serves a pending recovery the
  /// evidence is this scan matched against the view at its final state, by the measurement model's rules
  /// (recovery_view_match.hpp) and with the update's prior covariance `prior_P`; otherwise it is
  /// the measurement model's own count.
  bool recoveryMatchOk(const fast_lio::ReanchorReference & reference,
                       const Eigen::Matrix<double, 23, 23> & prior_P) const
  {
    int effective = effct_feat_num;
    double res_mean = res_mean_last;
#ifdef USE_IVOX
    if (recoveryEntryViewServes()) {
      fast_lio::RecoveryViewPose pose;
      pose.rot = state_point.rot;
      pose.pos = state_point.pos;
      pose.rot_li = state_point.offset_R_L_I;
      pose.pos_li = state_point.offset_T_L_I;
      const auto search = [](const PointType & p, int k, PointVector & near, std::vector<float> & sqdist) {
        ikdtree.Nearest_Search_View(p, k, near, sqdist);
      };
      const fast_lio::RecoveryViewMatch match = fast_lio::matchRecoveryView<NUM_MATCH_POINTS, PointVector>(
        feats_down_body->points, pose, search, params_.values().corr_gate, prior_P.block<6, 6>(0, 0), LASER_POINT_COV);
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterRecoveryViewMatch)
        fast_lio_test::afterRecoveryViewMatch(
          lidar_end_time, effct_feat_num, res_mean_last, match.effective, match.res_mean, ikdtree.entryViewFrames());
#endif
      effective = match.effective;
      res_mean = match.res_mean;
    }
#endif
    return fast_lio::reanchorMatchOk(effective, res_mean, reanchor_params, reference);
  }

  /// Pending freezes the map unless the entry view serves the recovery and no pinned tier joined the map while
  /// it was open (a pinned tier is not a scan: the view never shows it, so the frozen map is the safe side).
  bool pendingRecoveryFreezesMap() const
  {
    if (recovery_state.trust != fast_lio::RecoveryTrust::kPending)
      return false;
#ifdef USE_IVOX
    if (recoveryEntryViewServes() && !ikdtree.entryViewPinned())
      return false;
#endif
    return true;
  }

  // The production reset owner, shared verbatim by the late guards and the test-only scheduler.
  bool rebuildLocalMap(const char * tag)
  {
    if (lost_recovery_params.enabled)
      lost_recovery_map_ready = false;
    anchorEvidence(fast_lio::AnchorTraceReason::kMapReset,
                   fast_lio::kAnchorEventMapReset,
                   &anchor_trace_counters.map_reset,
                   lidar_end_time,
                   true);
    // The rebuilt map is seeded from the very pose in question. The anchors of the old map end with it, and
    // the reference counts as rebuilt, not recovered: the gate going healthy below says the local map works
    // again, not where the platform is.
    changeStationaryReference(fast_lio::AnchorDrop::kMapReset, lidar_end_time);
    fast_lio::breakRecoveryRun(&recovery_state);
    noteRecoveryEvent(
      fast_lio::resetRecoveryReference(&recovery_state, fast_lio::traceNs(lidar_end_time), recovery_params),
      fast_lio::traceNs(lidar_end_time));
    int map_before = 0;
    const bool kept_prior = rebuildLocalMapStorage(map_before, admissionObserver());
    const auto prior =
      fast_lio::priorMapAfterReset(prior_map_installed_, kept_prior, params_.values().prior_map_freeze);
#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::afterPriorMapReset)
      fast_lio_test::afterPriorMapReset(prior_map_installed_,
                                        prior.installed_after,
                                        required_prior_lost || prior.required_prior_lost,
                                        ikdtree.pinnedVoxels());
#endif
    prior_map_installed_ = prior.installed_after;
    required_prior_lost = required_prior_lost || prior.required_prior_lost;
    anchorTraceMapUpdated();
    if (required_prior_lost) {
      fast_lio::tripReanchorLost(&reanchor_gate);
      flio_degraded_odom = flio_map_frozen = true;
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] required prior map lost during reset; installed identity invalidated, freeze request "
                   "retained. Normal pose and map growth stopped; restart explicitly to reload the prior.",
                   tag);
      return false;
    }
    if (!fast_lio::mapBackendHoldsPoints(ikdtree)) {
      fast_lio::tripReanchorLost(&reanchor_gate);
      flio_degraded_odom = flio_map_frozen = true;
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] reset completed with zero retained points; no usable local map: map %d -> %d pts "
                   "at pos=[%.2f %.2f %.2f]",
                   tag,
                   map_before,
                   ikdtree.size(),
                   state_point.pos[0],
                   state_point.pos[1],
                   state_point.pos[2]);
      return false;
    }
    fast_lio::resetReanchorGate(&reanchor_gate);
    if (lost_recovery_params.enabled)
      lost_recovery_map_ready = true;
    if (kept_prior) {
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] live map dropped and re-seeded from the current scan (%d pts) at pos=[%.2f %.2f %.2f]: "
                   "map %d -> %d pts, the prior map (%zu voxels) is still there to re-register against",
                   tag,
                   feats_down_size,
                   state_point.pos[0],
                   state_point.pos[1],
                   state_point.pos[2],
                   map_before,
                   ikdtree.size(),
                   ikdtree.pinnedVoxels());
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] map rebuilt from the current scan (%d pts) at pos=[%.2f %.2f %.2f]: map %d -> %d pts, the "
                   "estimate is live again but unanchored",
                   tag,
                   feats_down_size,
                   state_point.pos[0],
                   state_point.pos[1],
                   state_point.pos[2],
                   map_before,
                   ikdtree.size());
    }
    return true;
  }

  // The latched input-epoch guard stops input and odometry until a restart (acceptInputEpoch), and the scan path
  // calls here instead of the plain reference repeat every cycle while it is latched. With reference health on, the
  // final reference and the repeat go out first, while the context still delivers them. fault_exit then asks for that
  // restart: the node shuts its context down and the supervisor restarts the pipeline. hold keeps saying it is stopped.
  void onInputEpochFault()
  {
    if (reference_health_en) {
      const auto final = captureFinalFrontend(false, false, 0.0);
      reference_publication_guard.run([&] { buildAndPublishReference(final.reference); }, disableReferencePublication);
    }
    publishReferenceRepeat();
    if (params_.values().input_epoch_on_fault == "fault_exit") {
      RCLCPP_FATAL(
        this->get_logger(),
        "[INPUT-EPOCH] exiting so the supervisor can restart the pipeline (input_epoch_on_fault=fault_exit)");
      rclcpp::shutdown();
      return;
    }
    RCLCPP_ERROR_THROTTLE(this->get_logger(),
                          input_time_log_clock,
                          10000,
                          "[INPUT-EPOCH] still latched since the sample stamped %.9f: input and odometry stay stopped "
                          "until the node restarts (input_epoch_on_fault=hold)",
                          input_epoch_fault_stamp);
  }

  void configureDiagnostics(fast_lio::FrontendStartup & startup)
  {
    params_.readDiagnosticsAndMapSave(
      *this, startup, latency_diag_en, latency_log_period_sec, map_insert_delay_scans, pcd_save_en, pcd_save_interval);
    admission_.enabled = params_.values().time_admission_counts_enable;
    if (admission_.enabled && (p_pre->lidar_type == AVIA || p_pre->lidar_type == ALLPOINT)) {
      admission_.supported = false;
      admission_.accounting_complete = false;
    }
  }

  fast_lio::TimeAdmissionAccounting * admissionObserver() noexcept
  {
    return admission_.enabled && admission_.supported ? &admission_ : nullptr;
  }

  void accountProcessException(fast_lio::TimeAdmissionAccounting * observer) noexcept
  {
    if (!observer)
      return;
    auto & audit = observer->last_process;
    audit.fc_after_known = true;
    audit.fc_after_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
    audit.reason = fast_lio::ImuProcessReason::kPartialPropagation;
    audit.disposition = fast_lio::ImuDisposition::kFatal;
    observer->process(audit);
  }

  // Called only after stopMapWorker returned. No buffer mutex is held while joining or formatting.
  fast_lio::TimeAdmissionSnapshot admissionShutdownSnapshot()
  {
    fast_lio::TimeAdmissionSnapshot snapshot{};
    snapshot.accounting = admission_;
    snapshot.shutdown = true;
    snapshot.map_joined = !map_worker_thread.joinable() && !admission_.async_inflight;
    if (!admission_.enabled)
      return snapshot;
    // stopMapWorker has joined: these are the real worker's flags, read without mtx_buffer or a live worker.
    snapshot.map_pending_known = true;
    snapshot.map_pending_operations = (map_add_ready_ || map_add_running_) ? 2 : 0;
    {
      std::lock_guard<std::mutex> lock(mtx_buffer);
      for (const auto & cloud : lidar_buffer)
        snapshot.accounting.count(snapshot.queue_pending, cloud->size());
    }
    snapshot.delayed_pending = admissionDelayedPending(snapshot.accounting);
    snapshot.pending_known = true;
    snapshot.fc_known = true;
    snapshot.fc_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
    snapshot.imu_duplicate_confirmations = imu_fifo_transaction.duplicateConfirmations();
    snapshot.imu_duplicate_integrations = p_imu->integrationLedger().duplicateIntegrations();
    snapshot.imu_watermark_regressions = p_imu->integrationLedger().watermarkRegressions();
    fast_lio::checkAdmissionSnapshot(snapshot);
    if (!snapshot.map_joined || snapshot.map_pending_operations != 0)
      snapshot.accounting.accounting_complete = false;
    return snapshot;
  }

  void finishAdmissionAccounting() noexcept
  {
    if (!admission_.enabled) {
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterAdmissionSnapshot)
        fast_lio_test::afterAdmissionSnapshot(admissionShutdownSnapshot());
#endif
      return;
    }
    try {
      const auto snapshot = admissionShutdownSnapshot();
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterAdmissionSnapshot)
        fast_lio_test::afterAdmissionSnapshot(snapshot);
#endif
      logAdmissionSummary(snapshot);
    } catch (...) {
      admission_.accounting_complete = false;  // diagnostic trouble never stops the estimator or throws at shutdown
    }
  }

  template<class Append>
  void appendAdmissionDetails(Append append, const fast_lio::TimeAdmissionSnapshot & s)
  {
    const auto & a = s.accounting;
    const auto integers = [&](const char * name, const auto & values) {
      append("\"%s\":[", name);
      for (std::size_t i = 0; i < values.size(); ++i)
        append("%s%llu", i ? "," : "", static_cast<unsigned long long>(values[i]));
      append("],");
    };
    integers("scan_time_status", a.scan_time_status);
    integers("settle_confirmations", a.confirmations);
    integers("settle_reasons", a.settle_reasons);
    integers("settle_dispositions", a.settle_dispositions);
    integers("process_dispositions", a.process_dispositions);
    integers("caller_exits", a.exits);
    integers("map_gate_labels", a.map_gates);
    append("\"map_pending_known\":%s,\"map_pending_operations\":%llu,",
           s.map_pending_known ? "true" : "false",
           static_cast<unsigned long long>(s.map_pending_operations));
    append(
      "\"history_keep_erased_points\":%llu,\"history_keep_retained_points\":%llu,"
      "\"history_reject_erased_points\":%llu,\"history_reject_retained_points\":%llu,\"settle_fc_bits\":%llu,",
      static_cast<unsigned long long>(a.history_keep_erased),
      static_cast<unsigned long long>(a.history_keep_retained),
      static_cast<unsigned long long>(a.history_reject_erased),
      static_cast<unsigned long long>(a.history_reject_retained),
      static_cast<unsigned long long>(a.settle_fc_bits));
    append("\"last_settle_known\":%s,\"settle_reason_not_recorded\":%llu,",
           a.last_settle_known ? "true" : "false",
           static_cast<unsigned long long>(a.settle_reason_not_recorded));
    append(
      "\"async_inflight\":%s,\"async_dispatches\":%llu,\"async_joins\":%llu,"
      "\"last_settle_token\":%llu,\"last_settle_epoch\":%llu,\"settle_samples\":%llu,"
      "\"trim_also_age_label\":%llu,\"callback_exceptions\":%llu,"
      "\"imu_duplicate_confirmations\":%llu,\"imu_duplicate_integrations\":%llu,\"imu_watermark_regressions\":%llu,",
      a.async_inflight ? "true" : "false",
      static_cast<unsigned long long>(a.async_dispatches),
      static_cast<unsigned long long>(a.async_joins),
      static_cast<unsigned long long>(a.last_token),
      static_cast<unsigned long long>(a.last_epoch),
      static_cast<unsigned long long>(a.settle_samples),
      static_cast<unsigned long long>(a.trim_also_age),
      static_cast<unsigned long long>(a.callback_exceptions),
      static_cast<unsigned long long>(s.imu_duplicate_confirmations),
      static_cast<unsigned long long>(s.imu_duplicate_integrations),
      static_cast<unsigned long long>(s.imu_watermark_regressions));
    append(
      "\"canonical_header_known\":%s,\"canonical_header_bits\":%llu,\"bootstrap_fc_known\":%s,"
      "\"bootstrap_fc_before_bits\":%llu,\"bootstrap_fc_after_bits\":%llu,"
      "\"partitions\":{\"callback\":%s,\"fifo\":%s,\"borrowed\":%s,\"process\":%s,\"history\":%s,\"map\":%s},",
      a.canonical_header_known ? "true" : "false",
      static_cast<unsigned long long>(a.canonical_header_bits),
      a.bootstrap_fc_known ? "true" : "false",
      static_cast<unsigned long long>(a.bootstrap_fc_before_bits),
      static_cast<unsigned long long>(a.bootstrap_fc_after_bits),
      s.callback_partition ? "true" : "false",
      s.queue_partition ? "true" : "false",
      s.borrowed_partition ? "true" : "false",
      s.process_partition ? "true" : "false",
      s.history_partition ? "true" : "false",
      s.map_partition ? "true" : "false");
    const auto & p = a.last_process;
    append(
      "\"last_process\":{\"entered\":%s,\"completed\":%s,\"input_effective_points\":%llu,"
      "\"history_evaluated\":%s,\"history_action\":%d,\"history_erase_completed\":%s,"
      "\"history_erased_points\":%llu,\"history_retained_points\":%llu,\"deskew_completed\":%s,\"deskew_points\":%llu,"
      "\"fc_before_known\":%s,\"fc_before_bits\":%llu,\"fc_after_known\":%s,\"fc_after_bits\":%llu,"
      "\"reason\":%d,\"disposition\":%d,\"produced_output\":%s},\"delayed_count_unit\":\"batches\",",
      p.entered ? "true" : "false",
      p.completed ? "true" : "false",
      static_cast<unsigned long long>(p.input_points),
      p.history_evaluated ? "true" : "false",
      static_cast<int>(p.history_action),
      p.history_erase_completed ? "true" : "false",
      static_cast<unsigned long long>(p.history_erased_points),
      static_cast<unsigned long long>(p.history_retained_points),
      p.deskew_completed ? "true" : "false",
      static_cast<unsigned long long>(p.deskew_points),
      p.fc_before_known ? "true" : "false",
      static_cast<unsigned long long>(p.fc_before_bits),
      p.fc_after_known ? "true" : "false",
      static_cast<unsigned long long>(p.fc_after_bits),
      static_cast<int>(p.reason),
      static_cast<int>(p.disposition),
      p.produced_output ? "true" : "false");
  }

  void logAdmissionSummary(const fast_lio::TimeAdmissionSnapshot & s)
  {
    char text[32768];
    std::size_t used = 0;
    bool complete = true;
    const auto append = [&](const char * format, auto... values) {
      if (!complete)
        return;
      const int n = std::snprintf(text + used, sizeof text - used, format, values...);
      if (n < 0 || static_cast<std::size_t>(n) >= sizeof text - used) {
        complete = false;
        return;
      }
      used += static_cast<std::size_t>(n);
    };
    const auto count = [&](const char * name, const fast_lio::AdmissionCount & c) {
      append("\"%s\":{\"messages\":%llu,\"points\":%llu},",
             name,
             static_cast<unsigned long long>(c.messages),
             static_cast<unsigned long long>(c.points));
    };
    const auto & a = s.accounting;
    append(
      "{\"schema\":\"fe_local_accounting_v1\",\"scope\":\"fe_local_standard_pcl_v1\","
      "\"enabled\":true,\"supported\":%s,\"end_reason\":\"shutdown\",\"reader_eof_verified\":false,"
      "\"map_joined\":%s,\"source_identity\":\"NOT_RECORDED\",\"config_identity\":\"NOT_RECORDED\","
      "\"source_message_uid\":\"NOT_RECORDED\",\"point_uid\":\"NOT_RECORDED\",\"raw_header\":\"NOT_RECORDED\","
      "\"raw_index\":\"NOT_RECORDED\",\"bag_ordinal\":\"NOT_RECORDED\",\"physical_identity\":\"NOT_RECORDED\","
      "\"lineage_scope\":\"LOCAL_COUNTS_ONLY\",\"local_event_seq\":%llu,\"buckets\":[",
      a.supported ? "true" : "false",
      s.map_joined ? "true" : "false",
      static_cast<unsigned long long>(a.local_event_seq));
    constexpr const char * names[] = {"received",
                                      "epoch_rejected",
                                      "late_rejected",
                                      "enqueued_declared",
                                      "enqueued",
                                      "trim_count",
                                      "trim_age",
                                      "clear_epoch",
                                      "clear_fatal",
                                      "sync_invalid",
                                      "sync_nonadvancing",
                                      "borrowed",
                                      "empty_imu",
                                      "bootstrap",
                                      "process_entered",
                                      "deskew",
                                      "downsampled",
                                      "lead_rejected",
                                      "transaction_rejected"};
    constexpr const char * units[] = {
      "declared_input", "preprocessed_effective", "deskew", "downsampled", "map_presented"};
    for (std::size_t i = 0; i < a.counts.size(); ++i) {
      const auto & c = a.counts[i];
      append("%s{\"name\":\"%s\",\"point_unit\":\"%s\",\"messages\":%llu,\"points\":%llu}",
             i ? "," : "",
             names[i],
             units[static_cast<std::size_t>(fast_lio::admissionPointUnit(static_cast<fast_lio::AdmissionBucket>(i)))],
             static_cast<unsigned long long>(c.messages),
             static_cast<unsigned long long>(c.points));
    }
    append("],\"process_outcomes\":[");
    for (std::size_t i = 0; i < a.process_outcomes.size(); ++i)
      append("%s{\"reason\":\"%s\",\"messages\":%llu,\"effective_points\":%llu}",
             i ? "," : "",
             fast_lio::imuProcessReasonName(static_cast<fast_lio::ImuProcessReason>(i)),
             static_cast<unsigned long long>(a.process_outcomes[i].messages),
             static_cast<unsigned long long>(a.process_outcomes[i].points));
    append("],\"map_operations\":[");
    for (std::size_t i = 0; i < a.map.size(); ++i) {
      const auto & op = a.map[i];
      append(
        "%s{\"kind\":%zu,\"submitted\":%llu,\"completed\":%llu,\"map_presented\":%llu,"
        "\"current\":%llu,\"prior\":%llu,\"reused\":%llu}",
        i ? "," : "",
        i,
        static_cast<unsigned long long>(op.submitted),
        static_cast<unsigned long long>(op.completed),
        static_cast<unsigned long long>(op.presented),
        static_cast<unsigned long long>(op.current),
        static_cast<unsigned long long>(op.prior),
        static_cast<unsigned long long>(op.reused));
    }
    append("],");
    appendAdmissionDetails(append, s);
    count("queue_pending_effective", s.queue_pending);
    count("callback_unresolved_declared", s.callback_unresolved_declared);
    count("borrowed_unresolved_effective", s.borrowed_unresolved);
    count("history_keep_effective", a.history_keep);
    count("history_reject_effective", a.history_reject);
    count("history_mixed_effective", a.history_mixed);
    count("history_unknown_effective", a.history_unknown);
    count("history_not_reached_effective", a.history_not_reached);
    count("process_exceptions_effective", a.process_exceptions);
    count("staged_map_presented_batches", a.staged);
    count("released_map_presented_batches", a.released);
    count("reset_discarded_map_presented_batches", a.reset_discarded);
    count("pending_map_presented_batches", s.delayed_pending);
    append(
      "\"history_mixed_erased\":%llu,\"history_mixed_retained\":%llu,\"wait_no_imu_calls\":%llu,"
      "\"wait_tail_calls\":%llu,\"overflow\":%s,\"exact\":%s,\"accounting_complete\":%s,"
      "\"fc_known\":%s,\"fc_bits\":%llu,\"sink_delivery_verified\":false,\"summary_complete\":true}",
      static_cast<unsigned long long>(a.history_erased),
      static_cast<unsigned long long>(a.history_retained),
      static_cast<unsigned long long>(a.wait_no_imu),
      static_cast<unsigned long long>(a.wait_tail),
      a.overflow ? "true" : "false",
      a.exact ? "true" : "false",
      a.accounting_complete ? "true" : "false",
      s.fc_known ? "true" : "false",
      static_cast<unsigned long long>(s.fc_bits));
    if (!complete) {
      admission_.accounting_complete = false;
      return;  // no complete marker is emitted when the fixed serialization buffer is insufficient
    }
#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::afterAdmissionSummary)
      fast_lio_test::afterAdmissionSummary(text, used);
#endif
    RCLCPP_INFO(this->get_logger(), "[TIME-ADMISSION-SUMMARY] %s", text);
  }

  void bootstrapScan(fast_lio::TimeAdmissionAccounting * observer, fast_lio::AdmissionCallerScope & accounting)
  {
    accounting.finish(fast_lio::AdmissionExit::kBootstrap);
    if (observer) {
      observer->record(fast_lio::AdmissionBucket::kBootstrap, Measures.lidar->size());
      observer->bootstrap_fc_known = true;
      observer->bootstrap_fc_before_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
    }
    first_lidar_time = Measures.lidar_beg_time;
    p_imu->first_lidar_time = first_lidar_time;
    flg_first_scan = false;
    finishImuBatch({fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kBootstrap}, observer);
    if (observer)
      observer->bootstrap_fc_after_bits = fast_lio::admissionDoubleBits(p_imu->lastProcessedEnd());
    anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kBootstrap;
  }

  bool processImuScan(fast_lio::TimeAdmissionAccounting * observer,
                      fast_lio::AdmissionCallerScope & accounting,
                      AnchorScanScope & anchor_scan_scope)
  {
    bool processed = false;
    auto * audit = observer ? &observer->last_process : nullptr;
    if (audit)
      *audit = {};
    try {
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::beforeImuProcess)
        fast_lio_test::beforeImuProcess();
#endif
      processed = p_imu->Process(Measures, kf, feats_undistort, audit);
      if (observer)
        observer->process(*audit);
    } catch (const std::exception & error) {
      accountProcessException(observer);
      accounting.finish(fast_lio::AdmissionExit::kProcessException);
      feats_undistort->clear();
      finishImuBatch({fast_lio::ImuDisposition::kFatal, fast_lio::ImuProcessReason::kPartialPropagation}, observer);
      RCLCPP_ERROR(this->get_logger(), "[IMU-CONSUMPTION-FAULT] filter processing failed closed: %s", error.what());
      anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kImuFault;
      dropAnchorForLostImu(Measures.lidar_end_time);
      return false;
    } catch (...) {
      accountProcessException(observer);
      accounting.finish(fast_lio::AdmissionExit::kProcessException);
      feats_undistort->clear();
      finishImuBatch({fast_lio::ImuDisposition::kFatal, fast_lio::ImuProcessReason::kPartialPropagation}, observer);
      RCLCPP_ERROR(this->get_logger(), "[IMU-CONSUMPTION-FAULT] filter processing failed closed: unknown exception");
      anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kImuFault;
      dropAnchorForLostImu(Measures.lidar_end_time);
      return false;
    }
    if (p_imu->initializationDegraded() && !imu_init_degraded_warned_) {
      imu_init_degraded_warned_ = true;
      RCLCPP_WARN(this->get_logger(),
                  "[IMU-INIT] sustained still-gate timeout: initialized from a MOVING IMU window; "
                  "map roll/pitch may be biased (initialization degraded)");
    }
    if (!finishImuBatch(p_imu->lastOutcome(), observer)) {
      accounting.finish(fast_lio::AdmissionExit::kSettleFailed);
      anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kImuFault;
      dropAnchorForLostImu(Measures.lidar_end_time);
      return false;
    }
    anchor_scan_scope.noteImuSettlement(p_imu->lastOutcome());
    if (required_prior_lost) {
      accounting.finish(fast_lio::AdmissionExit::kRequiredPriorLost);
      fast_lio::tripReanchorLost(&reanchor_gate);
      flio_degraded_odom = flio_map_frozen = true;
      return false;
    }
    if (!processed) {
      accounting.finish(fast_lio::AdmissionExit::kNotProcessed);
      anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kImuNotProcessed;
      if (p_imu->lastOutcome().disposition == fast_lio::ImuDisposition::kInvalid)
        dropAnchorForLostImu(Measures.lidar_end_time);
      if (p_imu->lastStatus() == ImuProcess::ProcessStatus::kCoverageGap) {
        anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kImuCoverageGap;
        markImuCoverageGap(Measures.lidar_end_time);
        frontendDrop();
        if (frontend_observation_en)
          publishFrontendObservation(false, true, Measures.lidar_end_time);
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             5000,
                             "[IMU-COVERAGE] skip without propagation reason=%d gap=%.6f end_extrap=%.6f count=%lu; "
                             "map frozen until re-anchored",
                             static_cast<int>(p_imu->coverage_result.status),
                             p_imu->coverage_result.max_gap_s,
                             p_imu->coverage_result.end_extrapolation_s,
                             static_cast<unsigned long>(imu_coverage_drop_count));
      }
      return false;
    }
    return true;
  }

  void timer_callback()
  {
    const fast_lio::FrontendParams & p = params_.values();
    auto * observer = admissionObserver();
    const bool ready = sync_packages(Measures, observer);
    publishPendingReference();  // sync's drop path has released mtx_buffer
    if (!ready) {
      if (input_epoch_guard.faulted())
        onInputEpochFault();
      else
        publishReferenceRepeat();
    }
    if (ready) {
      fast_lio::AdmissionCallerScope accounting{observer};
      AnchorScanScope anchor_scan_scope(Measures);
      frontend_runaway = false;
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::resetBeforeReference && fast_lio_test::resetBeforeReference(lidar_end_time))
        rebuildLocalMap("TEST_PRE_REFERENCE");
#endif
      if (imu_backlog_reanchor_required) {
        markImuCoverageGap(Measures.lidar_end_time);
        imu_backlog_reanchor_required = false;
      }
      if (flg_first_scan) {
        bootstrapScan(observer, accounting);
        return;
      }

      double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;
      // Pre-iEKF split: [IKDPROF] covers only the iEKF and the map-update loop, so the
      // undistort / FOV-segment / downsample costs were invisible in the per-frame budget.
      double t_undist = 0.0, t_join = 0.0, t_fov = 0.0;

      measurement_.beginScan();
      const fast_lio::MeasurementTelemetry & measured = measurement_.telemetry();
      ekf_iters = 0;
      kdtree_search_time = 0.0;
      solve_const_H_time = 0;
      svd_time = 0;
      t0 = omp_get_wtime();

      if (!processImuScan(observer, accounting, anchor_scan_scope))
        return;
      t_undist = omp_get_wtime();
      state_point = kf.get_x();
      if (p_imu->gap_summary.gaps > 0) {
        const fast_lio::ImuGapSummary & gaps = p_imu->gap_summary;
        ++imu_gap_bridged_count;
        anchorEvidence(fast_lio::AnchorTraceReason::kImuBridge,
                       fast_lio::kAnchorEventImuBridge,
                       &anchor_trace_counters.imu_bridge,
                       Measures.lidar_end_time,
                       !anchor_trace_bridge_run);
        anchor_trace_bridge_run = true;
        // The motion model crossed a stretch no sample observed, however short: the static evidence behind the
        // anchor is broken, and the pose on the far side is not supported by anything that saw the crossing. Only
        // a long interval also sends the gate to blind (below).
        dropStationaryAnchor(fast_lio::AnchorDrop::kImuBridge, Measures.lidar_end_time);
        startUnobservedImuRecovery(fast_lio::RecoveryCause::kImuBridge, Measures.lidar_end_time);
        RCLCPP_DEBUG(rclcpp::get_logger("laser_mapping.imu_transactions"),
                     "[IMU-CONSUMPTION-GAP] token=%lu begin=%.9f end=%.9f gaps=%d longest_interval=%.9f total=%.9f "
                     "scans_total=%lu",
                     static_cast<unsigned long>(pending_imu_token.sequence),
                     Measures.lidar_beg_time,
                     Measures.lidar_end_time,
                     gaps.gaps,
                     gaps.longest_interval_s,
                     gaps.total_s,
                     static_cast<unsigned long>(imu_gap_bridged_count));
        // The IMU interval, not the stretch this scan bridged: a dropout carried through several empty scans is
        // bridged 0.1 s at a time.
        const bool verify = gaps.longest_interval_s > p.imu_gap_verify_s;
        if (verify) {
          markImuGapBridged(Measures.lidar_end_time);
        }
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             500,
                             "[IMU-GAP] bridged %d IMU gap(s), longest %.3fs of a %.3fs IMU interval (total "
                             "%.3fs)%s: gyro average, velocity held at %.2fm/s; the longest added 1-sigma pos %.3fm "
                             "vel %.2fm/s yaw %.2fdeg roll/pitch %.2fdeg%s (scans with gaps: %lu)",
                             gaps.gaps,
                             gaps.longest_s,
                             gaps.longest_interval_s,
                             gaps.total_s,
                             Measures.imu.empty() ? " in a scan without IMU samples" : "",
                             state_point.vel.norm(),
                             gaps.longest.sigma_pos_h,
                             gaps.longest.sigma_vel_h,
                             gaps.longest.sigma_yaw * 180.0 / M_PI,
                             gaps.longest.sigma_rp * 180.0 / M_PI,
                             verify ? "; map frozen until re-anchored" : "",
                             static_cast<unsigned long>(imu_gap_bridged_count));
      }

      /*** Scan-hole guard: bound the IMU-only excursion across a gap ***/
      if (p.guard_hole_en && flg_EKF_inited && hole_last_scan_end > 0.0) {
        const double gap = Measures.lidar_beg_time - hole_last_scan_end;
        if (gap > p.guard_hole_min_sec) {
          state_ikfom st = kf.get_x();
          const double spd_imu = st.vel.norm();
          const V3D dv = st.vel - hole_vel_pre;
          const double dv_max = p.guard_hole_max_accel * gap;
          const double dv_n = dv.norm();
          if (dv_n > dv_max) {
            st.vel = hole_vel_pre + dv * (dv_max / dv_n);
            gravity_align_vel_overwritten = true;
            resetVelocityCovariance(p.guard_vel_reset_sigma);
          }
          const V3D pos_imu = st.pos;
          if (p.guard_hole_reanchor_pos) {
            st.pos = hole_pos_pre + hole_vel_pre * gap;
          }
          kf.change_x(st);
          state_point = kf.get_x();
          // With the IMU-gap prior the part of the hole without IMU samples was already crossed at the held
          // velocity, so |dv| here is what the samples that do exist integrated and the clamp rarely engages.
          RCLCPP_WARN(this->get_logger(),
                      "[HOLE-GUARD] scan hole %.2fs: imu-only vel %.2f m/s (pre-hole %.2f) -> held %.2f; pos "
                      "re-anchored by %.2fm%s",
                      gap,
                      spd_imu,
                      hole_vel_pre.norm(),
                      state_point.vel.norm(),
                      p.guard_hole_reanchor_pos ? (pos_imu - state_point.pos).norm() : 0.0,
                      p_imu->gap_summary.gaps > 0 ? " (IMU gap bridged, velocity held across it)" : "");
        }
      }
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

      if (!feats_undistort || feats_undistort->empty()) {
        accounting.finish(fast_lio::AdmissionExit::kNoCloud);
        if (!diag_first_no_point_logged) {
          RCLCPP_WARN(
            this->get_logger(), "[STARTUP][FAST_LIO] first no-point scan lidar_beg=%.3f", Measures.lidar_beg_time);
          diag_first_no_point_logged = true;
        }
        RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
        frontendDrop();
        anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kNoCloud;
        return;
      }

      if (!diag_first_valid_scan_logged) {
        RCLCPP_INFO(this->get_logger(),
                    "[STARTUP][FAST_LIO] first valid scan lidar_beg=%.3f points=%zu",
                    Measures.lidar_beg_time,
                    feats_undistort->size());
        diag_first_valid_scan_logged = true;
      }

      flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
      /*** Restore the ikd-tree single-operator invariant: wait for the previous
       *   scan's off-thread map Add to finish before this scan touches the tree. ***/
      if (async_map_en)
        joinMapAdd(observer);
      t_join = omp_get_wtime();
      installPriorMap();
#ifdef USE_IVOX
      // Capacity is a silent failure mode: once the grid is full every new voxel costs the
      // least-recently-written one, and with a prior map loaded those are the parts of it the
      // robot has not driven through yet. Read here, where the async Add has just been joined.
      if (ikdtree.evictedVoxels() != last_reported_evicted_voxels) {
        last_reported_evicted_voxels = ikdtree.evictedVoxels();
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             30000,
                             "[IVOX] map at capacity: %zu voxel(s) evicted so far (holding %zu of "
                             "ivox_max_voxels=%zu). The map no longer keeps everything it has seen.",
                             last_reported_evicted_voxels,
                             ikdtree.numVoxels(),
                             ikdtree.maxVoxels());
      }
#endif

      /*** Segment the map in lidar FOV ***/
      lasermap_fov_segment();
      t_fov = omp_get_wtime();

      /*** downsample the feature points in a scan ***/
      downSizeFilterSurf.filter(feats_undistort->points, feats_down_body->points);
      const std::size_t downsampled_count = feats_down_body->points.size();
      if (observer)
        observer->record(fast_lio::AdmissionBucket::kDownsampled, downsampled_count);
      if (!measurement_.prepareScan(downsampled_count, feats_down_size)) {
        accounting.finish(fast_lio::AdmissionExit::kPrepareFailed);
        RCLCPP_ERROR(
          this->get_logger(), "Downsampled scan exceeds int-indexed estimator capacity: %zu points", downsampled_count);
        frontendDrop();
        return;
      }
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterScanPrepared)
        fast_lio_test::afterScanPrepared(downsampled_count, lidar_end_time);
#endif
      feats_down_body->width = static_cast<uint32_t>(downsampled_count);
      feats_down_body->height = 1;
      feats_down_body->is_dense = true;
      t1 = omp_get_wtime();
      /*** Engulfment telemetry: far-field fraction of the raw downsampled scan ***/
      if (feats_down_size > 0) {
        int far_cnt = 0;
        const double far_r2 = p.guard_engulf_far_range * p.guard_engulf_far_range;
        for (int i = 0; i < feats_down_size; i++) {
          const PointType & pb = feats_down_body->points[i];
          if (pb.x * pb.x + pb.y * pb.y > far_r2)
            ++far_cnt;
        }
        scan_far_frac = static_cast<double>(far_cnt) / feats_down_size;
      }
      /*** initialize the map kdtree ***/
#ifdef USE_IVOX
      if (ikdtree.empty())
#else
      if (ikdtree.Root_Node == nullptr)
#endif
      {
        if (observer)
          observer->add(observer->map_gates[3]);
        RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
        if (feats_down_size > 5) {
          // --- Prior map preload (Scheme 1) ---
          if (!prior_map_pcd_.empty()) {
            // 1. Inject initial pose into EKF if provided. Only on the as-delivered path:
            //    initial_pose places the filter's start inside the MAP's frame, which only
            //    means anything when that frame is this run's. On the relocalization path
            //    the map moves to the front end instead, so the start pose stays untouched.
            if (p.prior_map_align == fast_lio::PriorMapAlign::kAsDelivered) {
              state_ikfom new_state = kf.get_x();
              const Eigen::Quaterniond q_current(new_state.rot.matrix());
              const Eigen::Vector3d current_rpy = q_current.toRotationMatrix().eulerAngles(0, 1, 2);
              const auto injection = fast_lio::resolveInitialPose(
                p.initial_pose_vec, p.initial_pose_full_rpy_override, current_rpy.x(), current_rpy.y());
              if (!injection.valid && !p.initial_pose_vec.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                             "initial_pose is unusable (%zu element(s), need %zu finite: x,y,z,roll,pitch,yaw) - "
                             "loading the prior map at the filter's own pose instead",
                             p.initial_pose_vec.size(),
                             fast_lio::kInitialPoseSize);
              }
              if (injection.valid) {
                new_state.pos = injection.position;
                new_state.rot = SO3(injection.orientation);
                kf.change_x(new_state);
                state_point = kf.get_x();
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                RCLCPP_INFO(this->get_logger(),
                            "Injected initial pose: [%.3f, %.3f, %.3f] yaw_deg=%.1f full_rpy_override=%s",
                            injection.position.x(),
                            injection.position.y(),
                            injection.position.z(),
                            injection.applied_yaw_rad * 180.0 / M_PI,
                            p.initial_pose_full_rpy_override ? "true" : "false");
              }
            }

            // 2. Load prior PCD
            PointCloudXYZI::Ptr prior_cloud(new PointCloudXYZI());
            if (pcl::io::loadPCDFile(prior_map_pcd_, *prior_cloud) == 0) {
              RCLCPP_INFO(
                this->get_logger(), "Loaded prior map: %s (%zu points)", prior_map_pcd_.c_str(), prior_cloud->size());

              // AABB self-check: helps diagnose wrong-frame PCDs
              // (e.g. a UTM map loaded where ENU was expected).
              // center→origin > ~10 km strongly suggests a global frame.
              {
                const auto bbox =
                  fast_lio::computePriorMapBounds(prior_cloud->points.begin(), prior_cloud->points.end());
                if (!bbox.valid) {
                  RCLCPP_ERROR(this->get_logger(),
                               "Prior map holds no finite point (%zu read) - check the PCD",
                               prior_cloud->size());
                } else {
                  RCLCPP_INFO(this->get_logger(),
                              "Prior map bbox: x=[%.2f..%.2f] y=[%.2f..%.2f] "
                              "z=[%.2f..%.2f] center→origin=%.1f m%s",
                              bbox.xmin,
                              bbox.xmax,
                              bbox.ymin,
                              bbox.ymax,
                              bbox.zmin,
                              bbox.zmax,
                              bbox.center_to_origin_m,
                              bbox.likely_global_frame ? "  (large offset — likely UTM/global frame, "
                                                         "check coordinate system!)"
                                                       : "");
                }
              }

              // 3. Downsample prior map
              const double prior_leaf = fast_lio::priorMapLeaf(p.prior_map_voxel, filter_size_map_min);
              pcl::VoxelGrid<PointType> voxel;
              voxel.setLeafSize(
                static_cast<float>(prior_leaf), static_cast<float>(prior_leaf), static_cast<float>(prior_leaf));
              voxel.setInputCloud(prior_cloud);
              PointCloudXYZI::Ptr prior_ds(new PointCloudXYZI());
              voxel.filter(*prior_ds);
              RCLCPP_INFO(this->get_logger(),
                          "Prior map downsampled at %.2f m: %zu → %zu points",
                          prior_leaf,
                          prior_cloud->size(),
                          prior_ds->size());

              // 4. Transform current scan to world frame
              ikdtree.set_downsample_param(filter_size_map_min);
              feats_down_world->resize(feats_down_size);
              for (int i = 0; i < feats_down_size; i++) {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
              }

              // 5. Prior map + current scan → map backend. Pinned, the prior map goes into a
              //    tier of its own; unpinned, it is merged in and lives or dies with the rest.
              //    On the relocalization path nothing goes in yet: the map is held in its own
              //    frame until an alignment says where to put it (installPriorMap below).
              size_t map_points = 0;
              if (p.prior_map_align == fast_lio::PriorMapAlign::kRelocalization) {
                prior_map_ds_ = prior_ds;
                observeMapOperation(
                  observer,
                  fast_lio::AdmissionMapKind::kInitialBuild,
                  [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
                  [] { return ikdtree.Build(feats_down_world->points); });
                anchorTraceMapUpdated();
                RCLCPP_INFO(this->get_logger(),
                            "map built from the current scan (%d points); the prior map (%zu points) is held "
                            "until /relocalization/result says where it is",
                            feats_down_size,
                            prior_ds->size());
                Localmap_Initialized = false;
                prior_map_pcd_.clear();
                accounting.finish(fast_lio::AdmissionExit::kInitialMap);
                return;
              }
#ifdef USE_IVOX
              if (!ikdtree.canAddPinnedPoints(prior_ds->points)) {
                accounting.finish(fast_lio::AdmissionExit::kPriorCapacity);
                RCLCPP_FATAL(this->get_logger(),
                             "Prior map cannot fit coarse capacity (%zu) or has unrepresentable voxel coordinates",
                             ikdtree.maxVoxels());
                rclcpp::shutdown();
                return;
              }
              if (p.prior_map_pinned) {
                observeMapOperation(
                  observer,
                  fast_lio::AdmissionMapKind::kPriorAdd,
                  [&] { return std::array<std::uint64_t, 3>{0, prior_ds->size(), 0}; },
                  [&] { return ikdtree.AddPinnedPoints(prior_ds->points); });
                observeMapOperation(
                  observer,
                  fast_lio::AdmissionMapKind::kPriorAdd,
                  [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
                  [] { return ikdtree.Add_Points(feats_down_world->points, true); });
                map_points = static_cast<size_t>(ikdtree.size());
                RCLCPP_INFO(this->get_logger(),
                            "map built from prior map (PINNED, %zu voxels) + current scan (%zu + %d = %zu points)",
                            ikdtree.pinnedVoxels(),
                            prior_ds->size(),
                            feats_down_size,
                            map_points);
              } else
#endif
              {
                PointCloudXYZI::Ptr combined(new PointCloudXYZI());
                *combined = *prior_ds;
                *combined += *feats_down_world;
                observeMapOperation(
                  observer,
                  fast_lio::AdmissionMapKind::kPriorBuild,
                  [&] { return std::array<std::uint64_t, 3>{feats_down_world->size(), prior_ds->size(), 0}; },
                  [&] { return ikdtree.Build(combined->points); });
                map_points = combined->size();
                RCLCPP_INFO(this->get_logger(),
                            "ikd-tree built from prior map + current scan (%zu + %d = %zu points)",
                            prior_ds->size(),
                            feats_down_size,
                            map_points);
              }
              anchorTraceMapUpdated();
              Localmap_Initialized = false;
              prior_map_installed_ = true;
#ifdef USE_IVOX
              // The prior map has to survive the load intact. iVox bounds itself by evicting
              // the least-recently-WRITTEN voxel, so a map larger than the capacity is
              // truncated in PCD file order — an arbitrary hole in the map the robot is about
              // to localize against, with no other symptom than a degraded match later.
              if (ikdtree.evictedVoxels() > 0) {
                accounting.finish(fast_lio::AdmissionExit::kPriorCapacity);
                RCLCPP_FATAL(this->get_logger(),
                             "Prior map does not fit the map backend: %zu voxel(s) were evicted while loading "
                             "(capacity ivox_max_voxels=%zu, held=%zu). The loaded map is truncated in PCD file "
                             "order. Raise ivox_max_voxels, coarsen ivox_grid_resolution, or downsample the PCD "
                             "(tools/downsample_pcd.py).",
                             ikdtree.evictedVoxels(),
                             ikdtree.maxVoxels(),
                             ikdtree.numVoxels());
                rclcpp::shutdown();
                return;
              }
              RCLCPP_INFO(this->get_logger(),
                          "[IVOX] prior map held: %zu voxel(s) of %zu capacity (%.1f%% used)",
                          ikdtree.numVoxels(),
                          ikdtree.maxVoxels(),
                          100.0 * static_cast<double>(ikdtree.numVoxels()) / static_cast<double>(ikdtree.maxVoxels()));
#endif
            } else {
              RCLCPP_ERROR(this->get_logger(),
                           "Failed to load prior map PCD: %s, falling back to normal init",
                           prior_map_pcd_.c_str());
              // Fall back: build from current scan only
              ikdtree.set_downsample_param(filter_size_map_min);
              feats_down_world->resize(feats_down_size);
              for (int i = 0; i < feats_down_size; i++) {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
              }
              observeMapOperation(
                observer,
                fast_lio::AdmissionMapKind::kInitialBuild,
                [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
                [] { return ikdtree.Build(feats_down_world->points); });
              anchorTraceMapUpdated();
            }
            prior_map_pcd_.clear();  // Prevent re-execution
          } else {
            // Normal init: build from current scan only
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++) {
              pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
#ifdef FASTLIO_TEST_HOOKS
            if (fast_lio_test::initialBuildTakesNoPoint && fast_lio_test::initialBuildTakesNoPoint(lidar_end_time)) {
              for (auto & point : feats_down_world->points)
                point.x = std::numeric_limits<float>::quiet_NaN();
            }
#endif
            observeMapOperation(
              observer,
              fast_lio::AdmissionMapKind::kInitialBuild,
              [] { return std::array<std::uint64_t, 3>{feats_down_world->size(), 0, 0}; },
              [] { return ikdtree.Build(feats_down_world->points); });
            // The first local reference: this build returned with the map backend holding points. Not a supported
            // scan, and nothing else of the recovery state moves; a build that took no point records nothing.
            if (fast_lio::mapBackendHoldsPoints(ikdtree))
              fast_lio::commitInitialReference(&recovery_state);
            anchorTraceMapUpdated();
          }
        }
        if (recovery_state.map_resets != 0 && !fast_lio::mapBackendHoldsPoints(ikdtree)) {
          fast_lio::tripReanchorLost(&reanchor_gate);
          flio_degraded_odom = flio_map_frozen = true;
        }
        accounting.finish(fast_lio::AdmissionExit::kInitialMap);
        return;
      }
      int featsFromMapNum = ikdtree.validnum();
      kdtree_size_st = ikdtree.size();

      // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num:
      // "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

      /*** ICP and iterated Kalman filter update ***/
      if (feats_down_size < 5) {
        accounting.finish(fast_lio::AdmissionExit::kDownsampleSmall);
        RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
        frontendDrop();
        return;
      }

      measurement_.resizeNormals(feats_down_size);
      feats_down_world->resize(feats_down_size);

      V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
      if (runtime_pos_log) {
        fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " "
                 << state_point.pos.transpose() << " " << ext_euler.transpose() << " "
                 << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose() << " "
                 << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << "\n";
      }

#ifndef USE_IVOX
      if (0)  // If you need to see map point, change to "if(1)"
      {
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
      }
#endif

      pointSearchInd_surf.resize(feats_down_size);
      measurement_.resizeNeighbours(feats_down_size);
      int rematch_num = 0;
      bool nearest_search_en = true;  //

      double t_pre_icp = omp_get_wtime();

      /*** iterated state estimation ***/
      const auto state_before_update = state_point;
      if (p.kalman_channel_diag_en && kalman_last_final_valid) {
        // the IMU propagation from the previous scan's final state to this scan's prior, as its own channel
        EkfUpdateDiagnostics propagationDiagnostics;
        writeKalmanChannel("propagation",
                           kalman_last_final_state,
                           state_before_update,
                           Eigen::Matrix<double, 23, 23>::Zero(),
                           propagationDiagnostics,
                           Zero3d,
                           Zero3d);
      }
      const V3D euler_before_update = SO3ToEuler(state_before_update.rot);
      const V3D pos_before_update = state_before_update.pos;
      double t_update_start = omp_get_wtime();
      // [ikd-profile] snapshot background/inline rebuild work spanning iEKF + map_incre
      const uint64_t prof_rb_us0 = g_ikd_rebuild_us.load(std::memory_order_relaxed);
      const uint64_t prof_rb_cnt0 = g_ikd_rebuild_count.load(std::memory_order_relaxed);
      const uint64_t prof_inl_us0 = g_ikd_inline_rebuild_us.load(std::memory_order_relaxed);
      double solve_H_time = 0;
      {
        fast_lio::AttitudeHoldInputs hold_in;
        hold_in.engulf_latched = engulf_latched;
        hold_in.effct_prev = effct_feat_num;  // previous scan's count at this point
        hold_in.hold_age_s = lidar_attitude_hold_since >= 0.0 ? lidar_end_time - lidar_attitude_hold_since : 0.0;
        hold_in.scan_gap_s = hole_last_scan_end > 0.0 ? Measures.lidar_beg_time - hole_last_scan_end : 0.1;
        hold_in.far_frac = scan_far_frac;
        hold_in.z_weak_prev = measured.pos_obs_z_weak;
        lidar_attitude_hold_active = fast_lio::holdAttitudeThisScan(p.attitude_hold, hold_in);
        fast_lio::AttitudeHoldInputs allowed_in = hold_in;
        allowed_in.engulf_latched = true;  // the release conditions alone: would a latch be honoured?
        lidar_attitude_hold_allowed = fast_lio::holdAttitudeThisScan(p.attitude_hold, allowed_in);
        measurement_.clearAttitudeHoldInfo();
      }
      if (lidar_attitude_hold_active) {
        if (lidar_attitude_hold_since < 0.0) {
          lidar_attitude_hold_since = lidar_end_time;
        }
        ++lidar_attitude_hold_scans;
        RCLCPP_INFO_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "[ATT-HOLD] lidar update restricted to translation+yaw (engulfed=%d effct_prev=%d "
                             "far_frac=%.2f z_weak_prev=%.2f age=%.1fs, %d scans so far)",
                             engulf_latched,
                             effct_feat_num,
                             scan_far_frac,
                             measured.pos_obs_z_weak,
                             lidar_end_time - lidar_attitude_hold_since,
                             lidar_attitude_hold_scans);
      }
      corr_dump_scan_active = false;
      corr_dump_iter = 0;
      if (p.corr_dump_en && fout_corr.is_open()) {
        for (const auto & w : p.corr_dump_windows) {
          if (lidar_end_time >= w.first && lidar_end_time <= w.second) {
            corr_dump_scan_active = true;
            break;
          }
        }
      }
      EkfUpdateDiagnostics lidarDiagnostics;
      Eigen::Matrix<double, 23, 23> lidarCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
      if (p.kalman_channel_diag_en || corr_dump_scan_active || p.reanchor_recovery_entry_view)
        lidarCovarianceBefore = kf.get_P();
      {
        // Consider roll/pitch on this scan? Decided from what the previous scan left (its roll/pitch information
        // and z_weak), the guard's latch and the time since the last release (rp_consider.hpp).
        fast_lio::RollPitchConsiderInputs consider_in;
        consider_in.engulf_latched = engulf_latched;
        consider_in.since_release_s = engulf_release_time > 0.0 ? lidar_end_time - engulf_release_time : -1.0;
        consider_in.rp_info_prev = measured.lidar_rot_obs_rp_min;
        consider_in.z_weak_prev = measured.pos_obs_z_weak;
        consider_in.consider_age_s = rp_consider_since >= 0.0 ? lidar_end_time - rp_consider_since : 0.0;
        rp_consider_active = fast_lio::considerRollPitchThisScan(p.rp_consider, consider_in);
        if (rp_consider_active) {
          const V3D grav_w(state_point.grav[0], state_point.grav[1], state_point.grav[2]);
          const V3D up_body = state_point.rot.toRotationMatrix().transpose() * (-grav_w.normalized());
          const auto proj = fast_lio::yawOnlyProjector(up_body);
          rp_consider_active = proj.valid;
          if (proj.valid) {
            kf.setConsiderRotation(3, proj.keep);
          }
        }
        if (!rp_consider_active) {
          kf.clearConsiderRotation();
          rp_consider_since = -1.0;
        } else {
          if (rp_consider_since < 0.0) {
            rp_consider_since = lidar_end_time;
          }
          ++rp_consider_scans;
          // covariance health while considering: the prior's smallest eigenvalue and the sigmas that matter
          const auto & Pc = kf.get_P();
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 23, 23>> es(Pc);
          const double p_min = es.info() == Eigen::Success ? es.eigenvalues().minCoeff() : std::nan("");
          if (!(p_min > -1e-12)) {
            ++rp_consider_indefinite;
          }
          RCLCPP_INFO_THROTTLE(this->get_logger(),
                               *this->get_clock(),
                               2000,
                               "[RP-CONSIDER] roll/pitch left to the gyro this scan (engulfed=%d since_release=%.1fs "
                               "rp_info_prev=%.0f z_weak_prev=%.2f age=%.1fs, %d scans so far); P: pos %.3f m rot "
                               "%.2f deg vel %.2f m/s, min eig %.2e (%d indefinite so far)",
                               engulf_latched,
                               consider_in.since_release_s,
                               measured.lidar_rot_obs_rp_min,
                               measured.pos_obs_z_weak,
                               consider_in.consider_age_s,
                               rp_consider_scans,
                               std::sqrt(std::max(0.0, Pc.template block<3, 3>(0, 0).trace() / 3.0)),
                               std::sqrt(std::max(0.0, Pc.template block<3, 3>(3, 3).trace() / 3.0)) * 57.2958,
                               std::sqrt(std::max(0.0, Pc.template block<3, 3>(12, 12).trace() / 3.0)),
                               p_min,
                               rp_consider_indefinite);
        }
      }
      const auto floorCoordinateEpoch = lost_recovery.coordinate_epoch;
      kf.update_iterated_dyn_share_modified(
        LASER_POINT_COV, solve_H_time, p.kalman_channel_diag_en ? &lidarDiagnostics : nullptr);
#ifdef FASTLIO_TEST_HOOKS
      if (fast_lio_test::afterLidarUpdate)
        fast_lio_test::afterLidarUpdate(lidar_end_time, fast_lio_test::filterSnapshot());
#endif
      kf.clearConsiderRotation();  // never leaks into the gravity / planar / wheel / parked updates (update_simple)

      state_point = kf.get_x();
      anchorTraceMeasurementUpdate(state_before_update, state_point);
      if (!lidar_attitude_hold_active && !measured.lidar_attitude_hold_info_active) {
        lidar_attitude_hold_since = -1.0;
      }
      if (measured.lidar_attitude_hold_info_active && !lidar_attitude_hold_active) {
        if (lidar_attitude_hold_since < 0.0) {
          lidar_attitude_hold_since = lidar_end_time;
        }
        ++lidar_attitude_hold_scans;
        RCLCPP_INFO_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "[ATT-HOLD] information rule: roll/pitch info %.1f < %.1f, update restricted to "
                             "translation+yaw (effct=%d far_frac=%.2f, %d scans so far)",
                             measured.lidar_rot_obs_rp_min,
                             p.lidar_attitude_hold_min_rp_info,
                             effct_feat_num,
                             scan_far_frac,
                             lidar_attitude_hold_scans);
      }
      if (p.kalman_channel_diag_en) {
        writeKalmanChannel(
          "lidar", state_before_update, state_point, lidarCovarianceBefore, lidarDiagnostics, Zero3d, Zero3d);
      }
      if ((p.corr_gate.innovation_k > 0.0 || p.corr_gate.nn0_max > 0.0) && measured.corr_gate_rejected.load() > 0) {
        RCLCPP_INFO_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             5000,
                             "[CORR-GATE] refused %d matches on the last iteration (effct=%d, k=%.1f, nn0_max=%.2f, "
                             "scan sigma=%.3f m)",
                             measured.corr_gate_rejected.load(),
                             effct_feat_num,
                             p.corr_gate.innovation_k,
                             p.corr_gate.nn0_max,
                             measured.corr_gate_scan_sigma);
      }
      if (corr_dump_scan_active && fout_corr_scan.is_open()) {
        const M3D dR = state_before_update.rot.toRotationMatrix().transpose() * state_point.rot.toRotationMatrix();
        const V3D drot = Log(dR);
        const V3D dpos = state_point.pos - state_before_update.pos;
        fout_corr_scan << std::setprecision(15) << lidar_end_time << ',' << std::setprecision(9)
                       << Measures.lidar_beg_time - first_lidar_time << ',' << corr_dump_iter << ',' << effct_feat_num
                       << ',' << int(lidar_attitude_hold_active) << ',' << int(measured.lidar_attitude_hold_info_active)
                       << ',' << std::setprecision(6) << measured.lidar_rot_obs_rp_min << ',' << drot(0) << ','
                       << drot(1) << ',' << drot(2) << ',' << dpos(0) << ',' << dpos(1) << ',' << dpos(2) << ','
                       << scan_far_frac << ',' << measured.pos_obs_z_weak << ',' << lidarCovarianceBefore(3, 3) << ','
                       << lidarCovarianceBefore(4, 4) << ',' << lidarCovarianceBefore(5, 5) << ','
                       << lidarCovarianceBefore(0, 0) << ',' << lidarCovarianceBefore(1, 1) << ','
                       << lidarCovarianceBefore(2, 2);
        const V3D dvel = state_point.vel - state_before_update.vel;
        fout_corr_scan << ',' << dvel(0) << ',' << dvel(1) << ',' << dvel(2);
        const int blk[9] = {0, 1, 2, 3, 4, 5, 12, 13, 14};
        for (int a = 0; a < 9; a++) {
          for (int b = 0; b < 9; b++) {
            fout_corr_scan << ',' << lidarCovarianceBefore(blk[a], blk[b]);
          }
        }
        fout_corr_scan << '\n';
        fout_corr.flush();
        fout_corr_iter.flush();
        fout_corr_scan.flush();
      }

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
      bool flio_engulfed = false;
      if (p.divergence_guard_en) {
        if (effct_feat_num < p.divergence_guard_min_eff)
          ++consec_low_eff;
        else
          consec_low_eff = 0;
        flio_vel_runaway = (state_point.vel.norm() > p.divergence_guard_max_speed);
        if (reference_health_en && flio_vel_runaway)
          reference_progress.negative(1u << 10, 1u << 4);  // the actual pre-clamp fact
        if (reference_health_en && consec_low_eff >= p.divergence_guard_streak)
          reference_progress.negative(1u << 11, 1u << 7);
        // (3) engulfment: far field collapsed in the raw scan — leading indicator
        //     (fires before effct starves; see guard_engulf_en block comment).
        if (p.guard_engulf_en) {
          if (scan_far_frac < p.guard_engulf_far_frac_min)
            ++consec_engulf;
          else
            consec_engulf = 0;
          if (!engulf_latched) {
            if (consec_engulf >= p.guard_engulf_streak) {
              engulf_latched = true;
              consec_engulf_clear = 0;
              engulf_latch_time = lidar_end_time;
            }
          } else {
            // Latched: release only on a SUSTAINED healthy far field (hysteresis).
            if (scan_far_frac >= p.guard_engulf_far_frac_release)
              ++consec_engulf_clear;
            else
              consec_engulf_clear = 0;
            if (consec_engulf_clear >= p.guard_engulf_release_scans) {
              engulf_latched = false;
              consec_engulf = 0;
              // The blind stretch's velocity: the legacy policy discards it (see block comment); the release
              // policy (engulf_release.hpp) can take the legs or keep the estimate, each with the 1-σ the
              // assignment deserves, so the next innovations are not forced through stale cross terms.
              state_ikfom st = kf.get_x();
              V3D leg_v = V3D::Zero();
              double leg_age = -1.0;
              {
                std::lock_guard<std::mutex> lk(mtx_twist);
                leg_v = last_twist_v_body;
                leg_age = last_twist_stamp > 0.0 ? lidar_end_time - last_twist_stamp : -1.0;
              }
              release_vel_params.zero_sigma = p.guard_vel_reset_sigma;
              release_vel_params.max_speed = p.divergence_guard_max_speed;
              const auto rel =
                fast_lio::releaseVelocity(st.vel, st.rot.toRotationMatrix(), leg_v, leg_age, release_vel_params);
              st.vel = rel.velocity;
              gravity_align_vel_overwritten = true;
              kf.change_x(st);
              resetVelocityCovariance(rel.sigma);
              // The pose drifted against the frozen map for the whole blind stretch while P kept shrinking:
              // grow the position and roll/pitch variance by that duration (yaw alone) before the far field
              // gets its say and before anything is inserted again.
              const double blind_s = engulf_latch_time > 0.0 ? lidar_end_time - engulf_latch_time : 0.0;
              engulf_release_time = lidar_end_time;
              bool inflated = false;
              if (p.release_inflate.enabled) {
                auto P = kf.get_P();
                const V3D grav_w(st.grav[0], st.grav[1], st.grav[2]);
                const V3D up_body = st.rot.toRotationMatrix().transpose() * (-grav_w.normalized());
                inflated = fast_lio::inflateReleaseCovariance(P, blind_s, up_body, p.release_inflate);
                if (inflated) {
                  kf.change_P(P);
                }
              }
              state_point = kf.get_x();
              RCLCPP_WARN(this->get_logger(),
                          "[DIVERGENCE-GUARD] engulfment released (far_frac=%.2f for %d scans, blind %.1fs): "
                          "map unfrozen, velocity %s (%.2f m/s, sigma %.2f)%s",
                          scan_far_frac,
                          p.guard_engulf_release_scans,
                          blind_s,
                          rel.used == fast_lio::ReleaseVelocitySource::kLeg    ? "from legs"
                          : rel.used == fast_lio::ReleaseVelocitySource::kKeep ? "kept"
                                                                               : "reset",
                          state_point.vel.norm(),
                          rel.sigma,
                          inflated ? ", P inflated" : "");
            }
          }
          flio_engulfed = engulf_latched;
        }
      }
      const bool flio_in_degraded = guardVerdictForTest(
        p.divergence_guard_en && (consec_low_eff >= p.divergence_guard_streak || flio_vel_runaway || flio_engulfed));

      /*** A1 gravity-alignment leveling prior — the ROOT-CAUSE fix for the
       *   iVox pitch-coupled world-Z drift. Supplies the ABSOLUTE pitch/roll
       *   reference the body-vz planar constraint below structurally lacks.
       *   2026-08-31 upgrade (m20_0831 doorway pitch anatomy: dZ_rate =
       *   −v·sin(pitch_err), r=0.94; the per-sample standstill gate engaged
       *   7/2569 scans, all inside guard windows — the phase where no error
       *   integrates). Three additions, parameter-gated, defaults = legacy:
       *   [L2] gravity_align_window_s > 0: the measurement becomes the
       *        WINDOW-MEAN compensated specific force — gait linear
       *        acceleration is near zero-mean over a stride, so the mean is a
       *        clean gravity direction even while walking (measured on
       *        m20_0831 /IMU: 0.13-0.56° median at 1.2 m/s vs 1.2-2.5°
       *        per-sample). gravity_align_continuous engages every scan.
       *   [L1] gravity_align_hold_s: a z_weak/degraded trigger keeps the
       *        stronger gravity_align_noise_degraded sigma for this long
       *        after clearing — the doorway pitch twist integrates into Z on
       *        the walk AFTER the guard window, not inside it.
       *        error-state variance BEFORE this update. A confidently-wrong
       *        iEKF (measured 0.09-0.67°/scan authority against a 6-7° error)
       *        beats any honestly-weighted witness; softening P is what lets
       *        the measurement act (Schmidt/consider-state practice).
       *   Constrains roll/pitch ONLY: yaw lies in the measurement null space
       *   (skew(g_body) sends a yaw-axis δθ to 0), so the LiDAR-excellent
       *   heading is untouched. h(x)=Rᵀ·u_world, right-perturbation
       *   R=R̂·Exp(δθ) ⇒ ∂h/∂δθ=skew(g_body), same form as planar/wheel. ***/
      if (p.gravity_align_en && !Measures.imu.empty()) {
        const double g_raw = p_imu->gravity_norm();  // gravity magnitude, raw units
        state_ikfom st = kf.get_x();
        M3D Rw = st.rot.toRotationMatrix();
        const V3D bg_now(st.bg[0], st.bg[1], st.bg[2]);
        const V3D v_body_now = Rw.transpose() * st.vel;                         // velocity in IMU frame
        const bool body_static_now = body_is_static(Measures, lidar_end_time);  // one verdict for this block
        const bool sigma_by_measurement = p.gravity_align_sigma_policy == "measurement";
        // [v6] velocity covariance trace at this scan's end (the compensation sigma's input)
        const double vel_var_now = p.gravity_align_vel_sigma_scale > 0.0 ? kf.get_P().block<3, 3>(12, 12).trace() : 0.0;
        // [L2] per-scan aggregate of kinematically-compensated specific force
        // (centripetal ω×v_body subtracted per sample — transport theorem; the
        // scan-end velocity serves every sample, walking-speed error << gates).
        if (p.gravity_align_window_s > 0.0) {
          GravWinAgg agg{lidar_end_time, Eigen::Vector3d::Zero(), 0, 0.0};
          // [v6] this scan's IMU span runs from the previous scan's end to this one's;
          // the body velocity at both edges is what the linear-acceleration term needs.
          V3D v_edge = v_body_now;
          double var_edge = vel_var_now;
          bool edge_valid = !gravity_align_vel_overwritten;
          if (p.gravity_align_lin_accel_comp && p.gravity_align_lin_accel_source == "leg") {
            // trailing edge means of the leg forward speed and the body rate over the same span
            for (const auto & imu : Measures.imu) {
              gravity_align_gyro_hist.emplace_back(
                get_time_sec(imu->header.stamp),
                V3D(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z));
            }
            while (!gravity_align_gyro_hist.empty() && (gravity_align_gyro_hist.front().first < lidar_end_time - 2.5 ||
                                                        gravity_align_gyro_hist.size() > 1000)) {
              gravity_align_gyro_hist.pop_front();
            }
            const double t_lo = lidar_end_time - p.gravity_align_leg_edge_window_s;
            double vx_sum = 0.0;
            int n_leg = 0;
            {
              std::lock_guard<std::mutex> lk(mtx_twist);
              for (const auto & sample : gravity_align_leg_hist) {
                if (sample.first >= t_lo && sample.first <= lidar_end_time + 0.02) {
                  vx_sum += sample.second;
                  ++n_leg;
                }
              }
            }
            V3D w_sum = V3D::Zero();
            int n_w = 0;
            for (const auto & sample : gravity_align_gyro_hist) {
              if (sample.first >= t_lo && sample.first <= lidar_end_time) {
                w_sum += sample.second;
                ++n_w;
              }
            }
            const double vx_mean = n_leg > 0 ? vx_sum / double(n_leg) * p.wheel_speed_scale : 0.0;
            V3D w_mean = V3D::Zero();
            if (n_w > 0) {
              w_mean = w_sum / double(n_w) - bg_now;
            }
            edge_valid = fast_lio::legEdgeValid(n_leg, vx_mean, body_static_now);
            v_edge = fast_lio::legEdgeVelocity(vx_mean, p.R_robot_to_imu, w_mean, p.gravity_align_leg_lever_arm);
            var_edge = fast_lio::legEdgeVariance(vx_mean, body_static_now, p.gravity_align_leg_vel_sigma);
          }
          agg.edge.t_start = gravity_align_prev_t;
          agg.edge.t_end = lidar_end_time;
          agg.edge.v_start = gravity_align_prev_v_body;
          agg.edge.v_end = v_edge;
          agg.edge.var_start = gravity_align_prev_var;
          agg.edge.var_end = var_edge;
          agg.edge.velocity_overwritten = !edge_valid || !gravity_align_prev_edge_valid || gravity_align_prev_t < 0.0;
          gravity_align_vel_overwritten = false;
          gravity_align_prev_t = lidar_end_time;
          gravity_align_prev_v_body = v_edge;
          gravity_align_prev_var = var_edge;
          gravity_align_prev_edge_valid = edge_valid;
          for (const auto & imu : Measures.imu) {
            V3D a_i(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
            V3D w_i(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
            // per-SAMPLE gyro filter: one spike must not disqualify the scan
            // (measured: scan-level max|ω| gating killed 18-23 % of scans).
            if (w_i.norm() > p.gravity_align_gyro_tol) {
              continue;
            }
            agg.sum += a_i - (w_i - bg_now).cross(v_body_now) * (g_raw / G_m_s2);
            agg.n++;
          }
          grav_win.push_back(agg);
          while (!grav_win.empty() &&
                 (grav_win.front().t < lidar_end_time - p.gravity_align_window_s || grav_win.size() > 100)) {
            grav_win.pop_front();
          }
        }
        // Engagement: legacy trigger, [L1] hold hysteresis, [L2] continuous.
        const bool ga_trig = measured.pos_obs_z_weak > p.gravity_align_z_weak_thresh || flio_in_degraded;
        if (ga_trig) {
          gravity_align_last_trig = lidar_end_time;
        }
        const bool ga_held = p.gravity_align_hold_s > 0.0 && !ga_trig &&
                             lidar_end_time - gravity_align_last_trig <= p.gravity_align_hold_s;
        gravity_align_degraded_active =
          !sigma_by_measurement && (ga_trig || ga_held) && p.gravity_align_noise_degraded > 0.0;
        if (ga_trig || ga_held || p.gravity_align_continuous) {
          // Measurement vector: window mean ([L2]) or legacy last-sample.
          V3D f_meas = V3D::Zero();
          double win_span = 0.0;
          int win_n = 0;
          fast_lio::LinearAccelerationTerm lin_term;  // [v6] invalid unless the option is on and the window is clean
          if (p.gravity_align_window_s > 0.0) {
            double t_old = lidar_end_time;
            bool overwritten = false;
            for (const auto & a : grav_win) {
              overwritten = overwritten || a.edge.velocity_overwritten;
              if (a.n == 0) {
                continue;
              }  // scan fully rejected by the per-sample gyro filter
              f_meas += a.sum;
              win_n += a.n;
              t_old = std::min(t_old, a.t);
            }
            if (win_n > 0) {
              f_meas /= double(win_n);
            }
            win_span = lidar_end_time - t_old + 0.1;
            // [v6] window-mean linear acceleration from the velocity history: (v_b(end) − v_b(start))/span,
            // raw accelerometer units like the ω×v term. A window with an overwritten velocity has no
            // derivative — the term is invalid and the prior is withheld below.
            if (p.gravity_align_lin_accel_comp && !grav_win.empty()) {
              const bool leg_source = p.gravity_align_lin_accel_source == "leg";
              lin_term = fast_lio::linearAccelerationTerm(grav_win.front().edge,
                                                          grav_win.back().edge,
                                                          leg_source ? false : overwritten,
                                                          G_m_s2,
                                                          p.gravity_align_vel_sigma_scale);
              if (lin_term.valid) {
                f_meas -= lin_term.accel * (g_raw / G_m_s2);
              }
            }
          } else {
            const auto & imu_last = Measures.imu.back();
            V3D a_raw(
              imu_last->linear_acceleration.x, imu_last->linear_acceleration.y, imu_last->linear_acceleration.z);
            V3D w_raw(imu_last->angular_velocity.x, imu_last->angular_velocity.y, imu_last->angular_velocity.z);
            if (w_raw.norm() <= p.gravity_align_gyro_tol) {
              f_meas = a_raw - (w_raw - bg_now).cross(v_body_now) * (g_raw / G_m_s2);
              win_n = 1;
              win_span = 1.0;
            }
          }
          const double a_norm = f_meas.norm();
          // Validity gates: enough window coverage (windowed mode), compensated
          // magnitude ≈ gravity (rejects residual tangential accel). The anchor
          // is gravity_align_mag_ref when set: the IMU-init g estimate is biased
          // when the bag/robot starts IN MOTION (m20_0831: init while walking →
          // the 5 % gate rejected nearly every honest window, 9/2569 engaged).
          const double g_ref = p.gravity_align_mag_ref > 1e-3 ? p.gravity_align_mag_ref : g_raw;
          const bool span_ok =
            p.gravity_align_window_s <= 0.0 || (win_span >= 0.7 * p.gravity_align_window_s && win_n >= 20);
          // [v6] with the compensation on, a window whose velocity history is broken cannot be levelled to.
          // A platform with no leg stream at all keeps the legacy measurement (the stream, not an edge, is absent).
          const bool leg_stream_alive = p.gravity_align_lin_accel_source != "leg" ||
                                        (last_twist_stamp > 0.0 && lidar_end_time - last_twist_stamp < 1.0);
          const bool lin_ok =
            !p.gravity_align_lin_accel_comp || p.gravity_align_window_s <= 0.0 || lin_term.valid || !leg_stream_alive;
          // [v6] blind stretch with the body moving: the acceleration is unobservable, so is the vertical
          const bool blind_moving =
            (p.gravity_align_withhold_blind_moving || sigma_by_measurement) && flio_in_degraded && !body_static_now;
          if (degeneracy_debug && blind_moving) {
            std::cerr << "[GALIGN] withheld: blind stretch, body moving" << std::endl;
          }
          if (degeneracy_debug && !lin_ok) {
            std::cerr << "[GALIGN] withheld: velocity overwritten inside the window" << std::endl;
          }
          if (g_ref > 1e-3 && win_n > 0 && a_norm > 1e-6 && span_ok && lin_ok && !blind_moving &&
              std::abs(a_norm - g_ref) <= p.gravity_align_accel_tol * g_ref) {
            V3D grav_w(st.grav[0], st.grav[1], st.grav[2]);  // ≈ [0,0,-g], points DOWN
            V3D u_world = -grav_w.normalized();              // world up (unit)
            V3D g_body = Rw.transpose() * u_world;           // predicted up in body (unit)
            V3D m_body = f_meas / a_norm;                    // measured up in body (unit)
            V3D residual = m_body - g_body;                  // z − h(x)
            if (residual.norm() <= 0.5) {                    // ~30° outlier gate (never inject a wild window)
              Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
              H.block<3, 3>(0, 3) = skew_sym_mat(g_body);  // ∂h/∂(rot); yaw in null space
              // [v5] degraded/held strength is an attitude measure; the gravity
              // state neither takes the residual nor leaks while it is in force.
              const bool grav_frozen = p.gravity_align_grav_freeze_degraded && (ga_trig || ga_held);
              // [v4] gravity-state observability: h = Rᵀ·(−g/‖g‖) also depends on the
              // S2 gravity state — without these two columns a motion-biased gravity
              // INIT is permanently unobservable (m20_0831: the bag starts walking →
              // the world frame is tilted 6.7° for the entire run and the rot-only
              // update just pulls attitude onto the tilted gravity — measured as the
              // 3-5° tug-of-war [GALIGN] tilt that never converges). With the columns
              // in, the Kalman gain splits the residual by covariance: rot follows
              // the (tight) lidar, grav follows the absolute accel reference.
              // ∂h/∂δg = −(1/‖g‖)·Rᵀ·M(g), M = S2_Mx (same helper df_dx uses).
              if (!grav_frozen) {
                Eigen::Matrix<state_ikfom::scalar, 2, 1> dz0 = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
                Eigen::Matrix<state_ikfom::scalar, 3, 2> Mg;
                st.S2_Mx(Mg, dz0, 21);
                const double gnorm = grav_w.norm();
                if (gnorm > 1e-6) {
                  H.block<3, 2>(0, 21) = -(1.0 / gnorm) * Rw.transpose() * Mg;
                }
              }
              // [v4] fading memory on the gravity block: with no process noise the
              // grav variance collapses and even a correct H routes nothing into it.
              // Gentle multiplicative leak (congruence row/col scaling — correlations
              // and PSD preserved, no covariance shock) capped at (3°)².
              if (p.gravity_align_grav_leak > 1.0 && !grav_frozen) {
                const double kGravVarCap = std::pow(p.gravity_align_grav_cap_deg * M_PI / 180.0, 2);
                auto Pg = kf.get_P();
                const double lam = std::sqrt(p.gravity_align_grav_leak);
                bool leaked = false;
                for (int i = 21; i <= 22; i++) {
                  if (Pg(i, i) < kGravVarCap) {
                    Pg.row(i) *= lam;
                    Pg.col(i) *= lam;
                    leaked = true;
                  }
                }
                if (leaked) {
                  kf.change_P(Pg);
                }
              }
              const double sigma =
                fast_lio::levellingSigma(sigma_by_measurement ? fast_lio::LevellingSigmaPolicy::kMeasurement
                                                              : fast_lio::LevellingSigmaPolicy::kScene,
                                         ga_trig || ga_held,
                                         body_static_now,
                                         p.gravity_align_noise,
                                         p.gravity_align_noise_degraded,
                                         p.gravity_align_noise_static,
                                         p.gravity_align_noise_moving);
              // [v6] the compensation's own uncertainty widens the measurement (quadrature)
              const double n2 = sigma * sigma + (lin_term.valid ? lin_term.sigma * lin_term.sigma : 0.0);
              Eigen::Vector3d R_diag(n2, n2, n2);
              EkfUpdateDiagnostics gravityDiagnostics;
              Eigen::Matrix<double, 23, 23> gravityCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
              if (p.kalman_channel_diag_en)
                gravityCovarianceBefore = kf.get_P();
              if (degeneracy_debug) {
                V3D tilt = g_body.cross(m_body);  // axis*sin(angle), body frame
                const double ang_deg = std::asin(std::min(1.0, tilt.norm())) * 57.2958;
                V3D eul_before = SO3ToEuler(st.rot);  // degrees
                kf.update_simple(H, residual, R_diag, p.kalman_channel_diag_en ? &gravityDiagnostics : nullptr);
                state_point = kf.get_x();
                V3D eul_after = SO3ToEuler(state_point.rot);
                V3D deul = eul_after - eul_before;
                std::cerr << "[GALIGN] z_weak=" << measured.pos_obs_z_weak << " degr=" << flio_in_degraded
                          << " trig=" << ga_trig << " held=" << ga_held << " win_n=" << win_n << " span=" << win_span
                          << " sigma=" << sigma << " sig_eff=" << std::sqrt(n2) << " gvar=" << kf.get_P()(21, 21)
                          << " tilt_deg=" << ang_deg << " alin=" << (lin_term.valid ? lin_term.accel.norm() : 0.0)
                          << " alin_deg="
                          << (lin_term.valid ? fast_lio::leanOfAcceleration(lin_term.accel, G_m_s2) * 57.2958 : 0.0)
                          << " dRPY_deg=[" << deul.x() << "," << deul.y() << "," << deul.z() << "]"
                          << " posZ=" << state_point.pos[2] << "m" << std::endl;
              } else {
                kf.update_simple(H, residual, R_diag, p.kalman_channel_diag_en ? &gravityDiagnostics : nullptr);
                state_point = kf.get_x();
              }
              if (p.kalman_channel_diag_en) {
                writeKalmanChannel(
                  "gravity", st, state_point, gravityCovarianceBefore, gravityDiagnostics, m_body, g_body);
              }
            }  // outlier gate
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
      if (p.planar_constraint_en && (measured.pos_obs_z_weak > p.planar_constraint_z_weak_thresh || flio_in_degraded)) {
        state_ikfom st = kf.get_x();
        M3D Rw = st.rot.toRotationMatrix();
        V3D v_body = Rw.transpose() * st.vel;  // velocity in IMU/body frame
        // Measurement: body velocity = 0, but only the vertical (z) row is
        // enforced — x/y get huge noise so forward/lateral motion is untouched.
        V3D residual = -v_body;
        Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
        H.block<3, 3>(0, 3) = skew_sym_mat(v_body);  // dh/d(rot)
        H.block<3, 3>(0, 12) = Rw.transpose();       // dh/d(vel)
        Eigen::Vector3d R_diag(1e6, 1e6, p.planar_constraint_noise * p.planar_constraint_noise);
        EkfUpdateDiagnostics planarDiagnostics;
        Eigen::Matrix<double, 23, 23> planarCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
        if (p.kalman_channel_diag_en)
          planarCovarianceBefore = kf.get_P();
        kf.update_simple(H, residual, R_diag, p.kalman_channel_diag_en ? &planarDiagnostics : nullptr);
        state_point = kf.get_x();
        if (p.kalman_channel_diag_en) {
          writeKalmanChannel("planar", st, state_point, planarCovarianceBefore, planarDiagnostics, Zero3d, v_body);
        }
      }

      /*** Divergence guard (P1) — bound the runaway while degraded. A ground
       *   vehicle cannot exceed a few m/s; a larger body speed here is
       *   IMU-dead-reckoning blow-up, so clamp it (direction preserved) to keep
       *   position growth linear and recoverable instead of exponential. Freeze
       *   the map so dead-reckoned scans do not corrupt it, and flag the
       *   odometry degraded so downstream (PGO) down-weights these poses. ***/
      /*** Re-anchor gate (reanchor_gate.hpp): the guard's verdict is not the
       *   release criterion. Once the scene recovers, the pre-blind map stays
       *   frozen until THIS scan registers against it — enough correspondences
       *   with a small residual, several scans running. Releasing on the scene
       *   alone is what turned dog 2's 9 s blind stretch into 45 min of drift:
       *   the pose error survived the release and the map was rebuilt around
       *   it. A verification that never converges is a lost front end, which
       *   downstream has to hear rather than consume. ***/
      // Trade absolute anchoring for a working estimate: rebuild the local map
      // from this scan at the current pose. The jump is real, is logged, and
      // reaches consumers through the health flags.
      // The relative tests need what healthy scans look like; only a gate that is verifying asks (the medians
      // are also logged at the release), and an unsupported recovery that runs the same match test. The tests
      // themselves stay off unless their ratios are set.
      fast_lio::ReanchorReference reanchor_reference;
      if (reanchor_gate.state == fast_lio::ReanchorState::kBlind ||
          reanchor_gate.state == fast_lio::ReanchorState::kVerifying ||
          recovery_state.trust == fast_lio::RecoveryTrust::kPending) {
        reanchor_reference.res_median = reanchor_healthy_res.median();
        reanchor_reference.eff_median = reanchor_healthy_eff.median();
      }
      const auto reanchor = fast_lio::updateReanchorGate(&reanchor_gate,
                                                         flio_in_degraded,
                                                         effct_feat_num,
                                                         res_mean_last,
                                                         lidar_end_time,
                                                         reanchor_params,
                                                         reanchor_reference);
      if (!reanchor.degraded && !reanchor.just_reanchored && effct_feat_num >= reanchor_params.min_eff) {
        reanchor_healthy_res.push(res_mean_last);
        reanchor_healthy_eff.push(static_cast<double>(effct_feat_num));
      }
      if (reanchor.just_reanchored) {
        RCLCPP_WARN(this->get_logger(),
                    "[REANCHOR] re-registered against the frozen map (effct=%d res=%.3fm, healthy median %.3fm) "
                    "after %u blind stretch(es): map unfrozen",
                    effct_feat_num,
                    res_mean_last,
                    reanchor_reference.res_median,
                    reanchor_gate.reanchors);
        if (anchor_scan_record_open)
          anchor_scan_record.events |= fast_lio::kAnchorEventReanchored;
      }
      if (reanchor.just_lost) {
        // The loss is this scan's verdict and its outputs carry it, whatever the reset below does to the gate.
        reanchor_lost_this_scan = true;
        anchorEvidence(fast_lio::AnchorTraceReason::kLost,
                       fast_lio::kAnchorEventLost,
                       &anchor_trace_counters.lost,
                       lidar_end_time,
                       true);
        // Recorded as a loss before a map reset of the same scan can end the anchor under another reason.
        dropStationaryAnchor(fast_lio::AnchorDrop::kLost, lidar_end_time);
        fast_lio::breakRecoveryRun(&recovery_state);
        RCLCPP_ERROR(this->get_logger(),
                     "[REANCHOR] LOST: no re-registration within %.1fs (effct=%d res=%.3fm) — the pose no longer fits "
                     "the map; policy=%s",
                     reanchor_params.timeout_sec,
                     effct_feat_num,
                     res_mean_last,
                     p.reanchor_on_fail.c_str());
        if (p.reanchor_on_fail == "reset_map") {
          if (!rebuildLocalMap("REANCHOR")) {
            accounting.finish(fast_lio::AdmissionExit::kRecoveryFailed);
            return;
          }
        }
      }
      // Release the static hold.  The flag leaves this node as "the platform is
      // provably parked" (kFeHealthFlagStaticHold), so it has to drop the moment
      // that stops being true: on a healthy scan, and equally when a degraded
      // stretch resumes motion.  Left raised while moving, it tells the fusion
      // node to publish IMU dead reckoning as a pinned pose and to reject the
      // GNSS displacement that is the platform actually moving.
      const auto release_static_hold = [&](const char * reason, fast_lio::AnchorTraceReason trace_reason) {
        if (!zupt_active) {
          return;
        }
        RCLCPP_WARN(this->get_logger(),
                    "[ZUPT] static hold released after %d scans (%s: effct=%d far_frac=%.2f)",
                    zupt_hold_scans,
                    reason,
                    effct_feat_num,
                    scan_far_frac);
        zupt_active = false;
        zupt_hold_scans = 0;
        anchorTransition(fast_lio::AnchorTraceEvent::kPinReleased, trace_reason, lidar_end_time);
      };
      applyAnchorFixtureForTest();
      anchorTraceBeforeStep(flio_in_degraded, reanchor.degraded);
      // One verdict per scan: the anchor lifecycle, the static hold, the parked
      // ZUPT on the healthy branch and the runaway watchdog must all answer "is
      // it parked?" the same way, and the summary costs an IMU-window pass. What
      // that verdict is allowed to do to the state is the decision table in
      // zupt_policy.hpp.
      const fast_lio::StaticEvidenceVerdict static_evidence =
        static_evidence_of(Measures, lidar_end_time, anchor_scan_record_open ? &anchor_scan_record : nullptr);
      const bool body_static = static_evidence.stationary();
      if (static_evidence.evidence == fast_lio::StaticEvidence::kMoving && anchor_trace_body_was_static) {
        anchorEvidence(fast_lio::AnchorTraceReason::kMotion,
                       fast_lio::kAnchorEventMotion,
                       &anchor_trace_counters.motion,
                       lidar_end_time,
                       true);
      }
      if (static_evidence.evidence == fast_lio::StaticEvidence::kUnknown) {
        anchorEvidence(fast_lio::AnchorTraceReason::kEvidenceUnknown,
                       fast_lio::kAnchorEventEvidenceUnknown,
                       &anchor_trace_counters.evidence_unknown,
                       lidar_end_time,
                       anchor_trace_body_was_static);
      }
      anchor_trace_body_was_static = body_static;
      // The anchor step (stationary_anchor_policy.hpp). Every event of this scan
      // that broke the evidence has already ended the anchor, so whatever the
      // step finds usable belongs to the current stationary episode.
      fast_lio::AnchorStepInput anchor_in;
      anchor_in.end_ns = fast_lio::traceNs(lidar_end_time);
      anchor_in.evidence = static_evidence.evidence;
      anchor_in.degraded = reanchor.degraded;
      anchor_in.reference_trusted = fast_lio::recoveryTrusted(recovery_state);
      anchor_in.lost = reanchor_gate.state == fast_lio::ReanchorState::kLost;
      const fast_lio::AnchorState anchor_before = stationary_anchor.state;
      const fast_lio::AnchorStep anchor_step =
        fast_lio::stepAnchor(&stationary_anchor, anchor_in, stationary_anchor_params);
      anchor_scan_evaluated = true;
      if (anchor_step.dropped) {
        noteAnchorEnded(anchor_before, anchor_step.drop, lidar_end_time);
      }
      // The parked hold counts the scans of the current episode only: its
      // streak and its engagement end with the anchor (releaseAnchorHolds), and
      // the prior below needs the anchor usable on this very scan. Counted
      // before the records of this step are written, so they include the scan.
      const auto parked = fast_lio::updateParkedHold(&parked_hold_state, anchor_step.parked_evidence, p.parked_hold);
      if (anchor_step.started) {
        // A candidate carries the pose of its first scan: the platform has not
        // moved since, so a later scan adds no information, only the chance of a
        // slow registration bias.
        zupt_anchor_pos = state_point.pos;
        zupt_anchor_rot = state_point.rot;
        fast_lio::saturatingIncrement(&anchor_trace_counters.candidates);
        anchorTransition(
          fast_lio::AnchorTraceEvent::kCandidateStarted, fast_lio::AnchorTraceReason::kNone, lidar_end_time);
      }
      if (anchor_step.committed) {
        anchor_ever_created = true;
        fast_lio::saturatingIncrement(&anchor_trace_counters.created);
        anchorTransition(fast_lio::AnchorTraceEvent::kCreated, fast_lio::AnchorTraceReason::kNone, lidar_end_time);
      }
      anchor_usable = anchor_step.usable;
      if (anchor_scan_record_open)
        anchor_scan_record.usable = anchor_step.usable;
      fast_lio::ZuptPolicyInputs zupt_in;
      zupt_in.scan_degraded = reanchor.degraded;
      zupt_in.zupt_enabled = p.zupt_en;
      zupt_in.anchor_valid = anchor_step.usable;
      zupt_in.body_static = body_static;
      zupt_in.speed = kf.get_x().vel.norm();
      zupt_in.max_speed = p.divergence_guard_max_speed;
      const fast_lio::ZuptDecision zupt_verdict = fast_lio::decideZupt(zupt_in);
      if (parked.just_engaged) {
        anchorTransition(
          fast_lio::AnchorTraceEvent::kParkedPriorEngaged, fast_lio::AnchorTraceReason::kNone, lidar_end_time);
      }
      if (parked.just_released) {
        RCLCPP_WARN(this->get_logger(), "[ZUPT] parked hold released: the scan is no parked evidence");
        anchorTransition(
          fast_lio::AnchorTraceEvent::kParkedPriorReleased, fast_lio::AnchorTraceReason::kNone, lidar_end_time);
      }
      if (reanchor.degraded) {
        state_ikfom st = kf.get_x();
        // Static hold takes precedence over the speed reset: with independent
        // proof that the platform is parked, and an anchor taken in this same
        // stationary episode, the blind stretch has nothing to integrate, while
        // zeroing only the velocity would leave the position and attitude
        // wherever propagation put them.
        if (zupt_verdict.hold) {
          noteAnchorWrite(AnchorWrite::Writer::kPin);
          st.pos = zupt_anchor_pos;
          st.rot = zupt_anchor_rot;
          st.vel.setZero();
          gravity_align_vel_overwritten = true;
          kf.change_x(st);
          resetVelocityCovariance(zupt_vel_reset_sigma);
          state_point = kf.get_x();
          ++zupt_hold_scans;
          anchorTraceAction(fast_lio::AnchorTraceAction::kPin, stationary_anchor.id);
          anchorTraceTarget();
          anchorTraceWriterReturned();
          if (!zupt_active) {
            zupt_active = true;
            RCLCPP_WARN(this->get_logger(),
                        "[ZUPT] static hold engaged (effct=%d far_frac=%.2f): pose pinned at [%.2f %.2f %.2f]",
                        effct_feat_num,
                        scan_far_frac,
                        zupt_anchor_pos[0],
                        zupt_anchor_pos[1],
                        zupt_anchor_pos[2]);
            anchorTransition(
              fast_lio::AnchorTraceEvent::kPinEngaged, fast_lio::AnchorTraceReason::kNone, lidar_end_time);
          }
        } else {
          // No hold on an anchor of this episode, and the scan still cannot
          // confirm the pose: this is the stretch no hold may survive into.
          release_static_hold("no hold on a degraded scan", fast_lio::AnchorTraceReason::kNone);
          if (zupt_verdict.action == fast_lio::ZuptAction::kZeroVelocity) {
            anchorTraceAction(fast_lio::AnchorTraceAction::kZeroVelocity, 0);
            if (zupt_in.zupt_enabled && zupt_in.body_static)
              noteZeroOnlyScan();
            // A body speed above the platform's physical maximum is a
            // GUARANTEED-WRONG state, not a bound to enforce. Clamping it to
            // max_speed (the old behaviour) turned the clamp into an
            // equilibrium: IMU propagation re-inflated |vel| every scan and the
            // callback re-clamped it — measured on m20_0814_degrade as spd=3.00
            // on every scan for 60 s, dragging the state ~3 m/s out of the
            // frozen map until no re-lock was geometrically possible. The
            // decisive response is the same one engulfment release uses: ZERO
            // the velocity so the next correspondences re-estimate it. Costs at
            // most one scan of re-convergence on a healthy scene; the clamp
            // guaranteed max_speed of position error per second.
            st.vel.setZero();
            gravity_align_vel_overwritten = true;
            kf.change_x(st);
            resetVelocityCovariance(zupt_vel_reset_sigma);
            state_point = kf.get_x();
            anchorTraceWriterReturned();
          }
        }
        flio_map_frozen = true;
        flio_degraded_odom = true;
        // Report only the defenses that are actually engaged: the planar-Z and
        // gravity-align priors are config-gated (both default OFF), and a log
        // line that claimed "planar-Z pinned" unconditionally hid a disabled
        // defense through the m20 vegetation-engulfment km-runaway (2026-08-14).
        RCLCPP_WARN_THROTTLE(this->get_logger(),
                             *this->get_clock(),
                             2000,
                             "[DIVERGENCE-GUARD] starved %d scans (effct=%d) engulfed %d scans (far_frac=%.2f) "
                             "reanchor=%s: vel reset if >%.1fm/s, map frozen, planar-Z %s, gravity-align %s; "
                             "posZ=%.1fm spd=%.2fm/s",
                             consec_low_eff,
                             effct_feat_num,
                             consec_engulf,
                             scan_far_frac,
                             reanchor_gate.state == fast_lio::ReanchorState::kVerifying ? "verifying"
                             : reanchor_gate.state == fast_lio::ReanchorState::kLost    ? "LOST"
                             : reanchor_gate.state == fast_lio::ReanchorState::kBlind   ? "blind"
                                                                                        : "ok",
                             p.divergence_guard_max_speed,
                             p.planar_constraint_en ? "pinned" : "OFF",
                             p.gravity_align_en ? "on" : "OFF",
                             state_point.pos[2],
                             state_point.vel.norm());
      } else {
        release_static_hold("scan healthy", fast_lio::AnchorTraceReason::kScanHealthy);
        // A parked platform has zero velocity, and a healthy-looking scan does not
        // enforce that: with far_frac 0.84 the correspondences pin rotation and
        // leave translation nearly free, so the velocity state rides the IMU bias
        // while effct and the residual stay good. Measured on dog 2 (2026-09-05):
        // 21 of 22 runaway trips fired with the OEM body velocity at 0.03 m/s and
        // the estimate claiming 0.4-2.6 m/s — every one of them on this branch,
        // where nothing consulted the static evidence at all. Zero the velocity
        // only: the healthy scan owns the pose, so a false "parked" costs one scan
        // of re-convergence rather than a pin at the wrong place.
        if (zupt_verdict.action == fast_lio::ZuptAction::kZeroVelocity) {
          anchorTraceAction(fast_lio::AnchorTraceAction::kZeroVelocity, 0);
          state_ikfom st = kf.get_x();
          st.vel.setZero();
          gravity_align_vel_overwritten = true;
          kf.change_x(st);
          resetVelocityCovariance(zupt_vel_reset_sigma);
          state_point = kf.get_x();
          anchorTraceWriterReturned();
          if (zupt_verdict.report) {
            RCLCPP_WARN_THROTTLE(this->get_logger(),
                                 *this->get_clock(),
                                 5000,
                                 "[ZUPT] parked with a healthy scan: zeroed %.2fm/s of phantom velocity "
                                 "(effct=%d far_frac=%.2f)",
                                 zupt_in.speed,
                                 effct_feat_num,
                                 scan_far_frac);
          }
        }

        /*** Parked hold (A'-3): a healthy scan that keeps moving the estimate
         *   while the IMU and the wheels both say parked is being moved by
         *   something that is not the robot. The blind-stretch hold above
         *   cannot answer it — the scan is not degraded — and no residual or
         *   correspondence-count test can see it either, because the map and
         *   the mover agree with each other (effct=1308, res=0.047 m during
         *   the 2026-09-04 drag).
         *
         *   Answered as a MEASUREMENT, not an override: pose pulled toward the
         *   anchor, velocity toward zero, with a covariance that says how much
         *   the prior is worth. A hard pin here would be wrong — the scan is
         *   healthy, so the lidar still deserves a vote, and a robot that is
         *   pushed or slides has to be able to out-vote the prior. That is
         *   also why the hold is slow to engage and instant to release. The
         *   anchor it pulls toward must be usable on this scan, the same test
         *   the blind-stretch pin uses. ***/
        if (fast_lio::parkedHoldPriorApplies(parked, anchor_step.usable)) {
          const state_ikfom stp = kf.get_x();
          const M3D est_rot = stp.rot.toRotationMatrix();
          const M3D anchor_rot = zupt_anchor_rot.toRotationMatrix();
          if (fast_lio::parkedHoldShouldApply(zupt_anchor_pos, anchor_rot, stp.pos, est_rot, p.parked_hold)) {
            const auto hold =
              fast_lio::parkedHoldMeasurement(zupt_anchor_pos, anchor_rot, stp.pos, est_rot, stp.vel, p.parked_hold);
            if (anchor_scan_record_open) {
              anchor_scan_record.parked_applied = true;
              anchorTraceAction(anchor_scan_record.action, stationary_anchor.id);
            }
            noteAnchorWrite(AnchorWrite::Writer::kParkedPrior);
            kf.update_simple(hold.H, hold.residual, hold.R_diag);
            state_point = kf.get_x();
            anchorTraceTarget();
            anchorTraceWriterReturned();
            RCLCPP_WARN_THROTTLE(this->get_logger(),
                                 *this->get_clock(),
                                 5000,
                                 "[ZUPT] parked hold applied on a healthy scan (effct=%d res=%.3fm): pulling %.2fm "
                                 "back to the anchor [%.2f %.2f %.2f]",
                                 effct_feat_num,
                                 res_mean_last,
                                 (zupt_anchor_pos - stp.pos).norm(),
                                 zupt_anchor_pos[0],
                                 zupt_anchor_pos[1],
                                 zupt_anchor_pos[2]);
          }
        }
      }

      /*** Runaway watchdog (runaway_watchdog.hpp): the guard above keys off the
       *   scan, and the drift that matters most keeps the scan looking healthy.
       *   Compare the estimate against the only sources outside it — the wheels
       *   and the IMU's static verdict — and act when they contradict it for
       *   long enough that noise cannot explain it. ***/
      if (p.runaway.enabled) {
        double wheel_stamp = -1.0;
        double wheel_speed = 0.0;
        {
          std::lock_guard<std::mutex> lk(mtx_twist);
          wheel_stamp = last_twist_stamp;
          wheel_speed = last_twist_speed;
        }
        const double wheel_age = wheel_stamp > 0.0 ? std::fabs(lidar_end_time - wheel_stamp) : -1.0;
        const auto runaway = runawayVerdictForTest(fast_lio::updateRunawayWatchdog(
          &runaway_state, lidar_end_time, state_point.vel.norm(), body_static, wheel_age, wheel_speed, p.runaway));
        if (runaway.tripped) {
          frontend_runaway = true;
          if (anchor_scan_record_open)
            anchor_scan_record.events |= fast_lio::kAnchorEventRunawayTrip;
          RCLCPP_ERROR(this->get_logger(),
                       "[RUNAWAY] estimate contradicted for %.1fs (%s): speed=%.2fm/s wheel=%.2fm/s effct=%d "
                       "res=%.3fm — trip #%u, policy=%s",
                       runaway.bad_for,
                       fast_lio::runawayReasonName(runaway.reason),
                       state_point.vel.norm(),
                       wheel_age >= 0.0 ? wheel_speed : -1.0,
                       effct_feat_num,
                       res_mean_last,
                       runaway_state.trips,
                       p.runaway_on_trip.c_str());
          // A reset and a hold both end the anchor below, within the scan whose anchor step may just have pinned
          // to it or pulled toward it. That write goes first: what is cleared, rebuilt around and published is
          // the state the scan has without the anchor. The verdict stands as it was reached.
          if (p.runaway_on_trip != "fault_exit")
            takeBackAnchorWrite();
          // Clear what tripped it. A kParked trip says the velocity is phantom,
          // and rebuilding the map around a state that still carries it re-arms
          // the identical contradiction: dog 2 (2026-09-05) tripped 91 times, 54
          // of the 90 intervals landing on hold_sec exactly. The IMU and the
          // wheels both say zero — the map is not the part that is wrong.
          if (runaway.reason == fast_lio::RunawayReason::kParked) {
            state_ikfom st = kf.get_x();
            st.vel.setZero();
            gravity_align_vel_overwritten = true;
            kf.change_x(st);
            resetVelocityCovariance(p.guard_vel_reset_sigma);
            state_point = kf.get_x();
          }
          if (p.runaway_on_trip == "reset_map") {
            if (!rebuildLocalMap("RUNAWAY")) {
              accounting.finish(fast_lio::AdmissionExit::kRecoveryFailed);
              return;
            }
          } else if (p.runaway_on_trip == "fault_exit") {
            accounting.finish(fast_lio::AdmissionExit::kFaultExit);
            RCLCPP_FATAL(this->get_logger(),
                         "[RUNAWAY] exiting so the supervisor can restart the pipeline (runaway_on_trip=fault_exit)");
            rclcpp::shutdown();
            return;
          } else {
            // hold: freeze the map and let the health flags stop downstream from
            // consuming a pose we no longer believe.
            fast_lio::tripReanchorLost(&reanchor_gate);
            anchorEvidence(fast_lio::AnchorTraceReason::kLost,
                           fast_lio::kAnchorEventLost,
                           &anchor_trace_counters.lost,
                           lidar_end_time,
                           true);
            dropStationaryAnchor(fast_lio::AnchorDrop::kLost, lidar_end_time);
            fast_lio::breakRecoveryRun(&recovery_state);
          }
        }
      }
      countAnchorWrite();

      /*** Recovery trust (stationary_anchor_policy.hpp, on with
       *   reanchor_total_timeout_sec > 0). A degraded scan held on a usable
       *   anchor is supported: the hold is the reference. Any other degraded or
       *   lost scan starts an unsupported-recovery episode, timed from the last
       *   supported scan, and so did a stretch of this scan the IMU did not
       *   observe. While it runs, the only way back is the re-anchor
       *   gate's own match test, passed by consecutive scans of one reference
       *   for the stable time, and the scans must not vouch for themselves with
       *   points they added: by default the map stays frozen meanwhile. With
       *   reanchor_recovery_entry_view the scans the guard did not call
       *   degraded keep going into the map and the test counts only the points
       *   it held when the episode began (iVox's entry view); a pinned tier
       *   brings the freeze back. ***/
      const bool scan_held = anchor_write.writer == AnchorWrite::Writer::kPin;  // a pin that was not taken back
      const bool scan_supported = (!reanchor.degraded || scan_held) &&
                                  reanchor_gate.state != fast_lio::ReanchorState::kLost &&
                                  !fast_lio::recoveryScanExcluded(recovery_state, lifecycle_scan_seq);
      if (anchor_scan_record_open)
        anchor_scan_record.supported = scan_supported;
      if (scan_supported) {
        fast_lio::noteSupportedScan(&recovery_state, anchor_in.end_ns);
      } else {
        startAnchorRecovery(fast_lio::RecoveryCause::kDegraded, anchor_in.end_ns);
      }
      syncRecoveryEntryView();
      const bool recovery_match = !flio_in_degraded && reanchor_gate.state != fast_lio::ReanchorState::kLost &&
                                  recoveryMatchOk(reanchor_reference, lidarCovarianceBefore);
      noteRecoveryEvent(fast_lio::observeRecoveryScan(&recovery_state,
                                                      lifecycle_scan_seq,
                                                      fast_lio::traceNs(Measures.lidar_beg_time),
                                                      anchor_in.end_ns,
                                                      recovery_match,
                                                      stationary_anchor.reference_generation,
                                                      recovery_params),
                        anchor_in.end_ns);
      syncRecoveryEntryView();
      if (pendingRecoveryFreezesMap())
        flio_map_frozen = true;
      observeLostRecovery(
        flio_in_degraded, recovery_match, fast_lio::traceNs(Measures.lidar_beg_time), anchor_in.end_ns);
      anchorTraceAfterStep();

      /*** Optional degeneracy diagnostics ***/
      if (degeneracy_debug) {
        const double lidar_correction = (state_point.pos - pos_before_update).norm();
        const double body_speed = state_point.vel.norm();
        const double dvel = (state_point.vel - state_before_update.vel).norm();
        const double dba = (state_point.ba - state_before_update.ba).norm();
        const double dbg = (state_point.bg - state_before_update.bg).norm();
        // Rotation applied by the lidar update (world frame): total angle and its yaw part.
        const Eigen::AngleAxisd upd_rot(state_point.rot.toRotationMatrix() *
                                        state_before_update.rot.toRotationMatrix().transpose());
        const double drot_deg = upd_rot.angle() * 180.0 / M_PI;
        const double dyaw_deg = upd_rot.angle() * upd_rot.axis()(2) * 180.0 / M_PI;
        // Online lidar→IMU extrinsic estimate (deg) — to see whether it converges and where.
        const V3D ex_eul = SO3ToEuler(state_point.offset_R_L_I);
        // Per-scan trace (every scan): lidar end time, IMU samples covering the
        // scan and their max stamp gap, correspondence count, far-field
        // fraction, and the update's state deltas — for correlating FE
        // instabilities against the input stream offline.
        {
          double imu_gap_max = 0.0;
          for (size_t k = 1; k < Measures.imu.size(); ++k) {
            const double dt =
              get_time_sec(Measures.imu[k]->header.stamp) - get_time_sec(Measures.imu[k - 1]->header.stamp);
            if (dt > imu_gap_max)
              imu_gap_max = dt;
          }
          // Translation observability along the travel direction: mean cos² between the
          // effective plane normals and the world-frame velocity direction (1 = every
          // plane faces the motion, 0 = all planes parallel to it → along-track unobservable);
          // plus normalized min/max eigenvalues of M and the weakest horizontal direction.
          double obs_along = -1.0, obs_emin = 0.0, obs_emax = 0.0, obs_hmin = 0.0;
          if (effct_feat_num > 0) {
            const double n = static_cast<double>(effct_feat_num);
            const Eigen::Vector3d vw(state_point.vel[0], state_point.vel[1], state_point.vel[2]);
            if (vw.norm() > 0.2) {
              const Eigen::Vector3d d = vw.normalized();
              obs_along = d.dot(measured.pos_obs_M * d) / n;
            }
            obs_emin = measured.pos_obs_eig_min / n;
            obs_emax = measured.pos_obs_eig_max / n;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es2(measured.pos_obs_M.topLeftCorner<2, 2>());
            obs_hmin = es2.eigenvalues()(0) / n;
          }
          std::cerr << std::fixed << std::setprecision(3) << "[SCAN] t=" << lidar_end_time
                    << " imu_n=" << Measures.imu.size() << " imu_gap=" << imu_gap_max << " pts=" << feats_down_size
                    << " effct=" << effct_feat_num << " far=" << scan_far_frac << " speed=" << body_speed
                    << " dvel=" << dvel << " corr=" << lidar_correction << " posZ=" << state_point.pos[2]
                    << " deg=" << (flio_in_degraded ? 1 : 0) << " along=" << obs_along << " emin=" << obs_emin
                    << " emax=" << obs_emax << " hmin=" << obs_hmin << " drot=" << drot_deg << " dyaw=" << dyaw_deg
                    << " bgz=" << state_point.bg[2] << " exT=" << state_point.offset_T_L_I[0] << ","
                    << state_point.offset_T_L_I[1] << "," << state_point.offset_T_L_I[2] << " exR=" << ex_eul[0] << ","
                    << ex_eul[1] << "," << ex_eul[2] << " res=";
          for (int b = 0; b < fast_lio::MeasurementTelemetry::kResBins; ++b)
            std::cerr << (b ? "," : "") << measured.res_bin_n[b] << ":"
                      << (measured.res_bin_n[b] ? std::sqrt(measured.res_bin_ss[b] / measured.res_bin_n[b]) * 1000.0
                                                : 0.0);
          std::cerr << std::endl;
        }
        if (lidar_correction > 0.2 || body_speed > 1.5 || dvel > 0.3 || dba > 0.05 || dbg > 0.01 ||
            effct_feat_num < 200 || measured.pos_obs_z_weak > 0.7) {
          std::cerr << "[DEGEN] effct=" << effct_feat_num << " corr=" << lidar_correction << "m speed=" << body_speed
                    << " dvel=" << dvel << " dba=" << dba << " dbg=" << dbg << " z_weak=" << measured.pos_obs_z_weak
                    << " posZ=" << state_point.pos[2] << "m"
                    << " rec=" << measured.canopy_ground_recovered << std::endl;
        }
      }

      /*** Wheel velocity update (after ICP, constrains along-track drift) ***/
      if (p.wheel_odom_en)
        updateWheelSpeed(p);

      euler_cur = SO3ToEuler(state_point.rot);
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
      geoQuat.x = state_point.rot.coeffs()[0];
      geoQuat.y = state_point.rot.coeffs()[1];
      geoQuat.z = state_point.rot.coeffs()[2];
      geoQuat.w = state_point.rot.coeffs()[3];

      double t_update_end = omp_get_wtime();

      /*** Scan-hole guard bookkeeping: this scan's posterior is the next hole's anchor ***/
      hole_last_scan_end = lidar_end_time;
      kalman_last_final_state = state_point;
      kalman_last_final_valid = true;
      hole_vel_pre = state_point.vel;
      hole_pos_pre = state_point.pos;

      /*** Front-end health for the PGO between-factor gate (Plan C): along-motion
           observability of this scan's final (post-update) state ***/
      {
        const Eigen::Vector3d vw(state_point.vel[0], state_point.vel[1], state_point.vel[2]);
        scan_body_speed = vw.norm();
        scan_obs_along = -1.0;
        if (scan_body_speed > kHealthSpeedGate) {
          if (effct_feat_num > 0) {
            const Eigen::Vector3d d = vw / scan_body_speed;
            scan_obs_along = d.dot(measured.pos_obs_M * d) / effct_feat_num;
          } else {
            scan_obs_along = 0.0;  // moving with no effective points: worst case
          }
        }
      }

      /******* Publish odometry *******/
      captureAnchorPost();
      anchor_final_snapshot = captureFinalFrontend(true, true, lidar_end_time);
      anchor_final_frozen = true;
      reference_pending = reference_health_en;
      const auto & final = anchor_final_snapshot;
      logFloorFinal(lidar_end_time, p.degeneracy_floor_mode, measured.floor, final, floorCoordinateEpoch);
      publish_odometry(pubOdomAftMapped_, tf_broadcaster_, final);
      accounting.finish(fast_lio::AdmissionExit::kPublished);
      if (anchor_scan_record_open) {
        anchor_scan_record.published = true;
        anchor_scan_record.outcome = fast_lio::AnchorScanOutcome::kPublished;
      }
      if (frontend_observation_en)
        publishFrontendObservation(true);

      /*** add the feature points to map kdtree ***/
      t3 = omp_get_wtime();
      // Mapping pause gate: when operator has disabled mapping (via
      // /lio_slam/set_mapping_enabled in the PGO node), skip ikd-tree
      // growth so the prior map isn't polluted by stationary frames or
      // teleport segments (elevator).  EKF and odometry publishing
      // above continue normally.
      // prior_map_freeze: with a prior map installed the operator can ask for pure
      // localization — the delivered map is the only map, and live scans stop being added
      // to it. Without it the map is neither: the prior owns the cells it covers and live
      // scans fill the gaps between them.
      const bool frozen_by_prior = p.prior_map_freeze && prior_map_installed_;
      const bool map_enabled = mapping_enabled.load(std::memory_order_relaxed);
      if (observer) {
        if (!map_enabled)
          observer->add(observer->map_gates[0]);
        if (flio_map_frozen)
          observer->add(observer->map_gates[1]);
        if (frozen_by_prior)
          observer->add(observer->map_gates[2]);
      }
      if (map_enabled && !flio_map_frozen && !frozen_by_prior) {
        map_incremental(measurement_.nearestPoints(), observer);
        anchorTraceMapUpdated();
      }
      t5 = omp_get_wtime();

      // 自适应降采样:t5-t0 是这一帧的计算耗时(不含排队 —— 排队归 max_scan_backlog_sec 管)。
      // 超预算就把扫描体素调粗:点少了、精度略降,但成本回到扫描周期以内。
      // Per-scan compute (queue wait excluded) drives the voxel: over budget costs points,
      // not latency.
      if (p.adaptive_ds.enabled) {
        const double scan_cost = t5 - t0;
        if (fast_lio::updateAdaptiveDownsample(&adaptive_ds_state, scan_cost, p.filter_size_surf_min, p.adaptive_ds)) {
          const float leaf = static_cast<float>(adaptive_ds_state.voxel);
          downSizeFilterSurf.setLeaf(leaf);
          RCLCPP_WARN(this->get_logger(),
                      "[ADAPTIVE-DS] scan voxel %.2f m (cost %.0f ms vs budget %.0f ms; %u up / %u down)",
                      adaptive_ds_state.voxel,
                      scan_cost * 1e3,
                      p.adaptive_ds.budget_sec * 1e3,
                      adaptive_ds_state.raises,
                      adaptive_ds_state.lowers);
        }
      }

      // [ikd-profile] per-scan split: iEKF (search-bound) vs map_incre Add vs transform
      // loop, correlated with concurrent background-rebuild work + inline rebuild.
      if (ikd_profile) {
        const double iekf_ms = (t_update_end - t_update_start) * 1000.0;
        // Under async, the Add runs off-thread: report its last completed wall time
        // (g_async_add_us, == previous scan's Add) and the join-wait this scan paid
        // (the residual Add cost the inter-scan gap could NOT hide). loop_ms is then
        // just the on-thread decision loop + dispatch.
        const double add_ms =
          async_map_en ? (g_async_add_us.load(std::memory_order_relaxed) / 1000.0) : (kdtree_incremental_time * 1000.0);
        const double join_ms = async_map_en ? g_map_join_wait_ms : 0.0;
        const double loop_ms = async_map_en ? ((t5 - t3) * 1000.0) : ((t5 - t3) * 1000.0 - add_ms);
        const uint64_t d_rb_us = g_ikd_rebuild_us.load(std::memory_order_relaxed) - prof_rb_us0;
        const uint64_t d_rb_cnt = g_ikd_rebuild_count.load(std::memory_order_relaxed) - prof_rb_cnt0;
        const uint64_t d_inl_us = g_ikd_inline_rebuild_us.load(std::memory_order_relaxed) - prof_inl_us0;
        // tree= uses kdtree_size_st (captured at scan start, worker idle) — calling
        // ikdtree.size() here would race the worker's in-flight Add under async.
        printf(
          "[IKDPROF] tree=%d feats=%d undist_ms=%.1f fov_ms=%.1f ds_ms=%.1f "
          "iekf_ms=%.1f its=%d/%d effct=%d msrch_ms=%.1f mfit_ms=%.1f solve_ms=%.1f "
          "nnshift_mm=%.1f nnskip=%d "
          "add_ms=%.1f join_ms=%.1f loop_ms=%.1f | "
          "rb_dcnt=%llu rb_dus_ms=%.1f inl_dus_ms=%.1f rb_active=%d "
          "rb_maxus_ms=%.1f rb_maxsz=%llu rootreb=%llu\n",
          kdtree_size_st,
          feats_down_size,
          (t_undist - t0) * 1000.0,
          (t_fov - t_join) * 1000.0,
          (t1 - t_fov) * 1000.0,
          iekf_ms,
          measured.ekf_search_iters,
          ekf_iters,
          effct_feat_num,
          measured.match_search_time * 1000.0,
          measured.match_nosearch_time * 1000.0,
          (measured.solve_time + solve_H_time) * 1000.0,
          measured.nn_max_shift * 1000.0,
          measured.nn_skips,
          add_ms,
          join_ms,
          loop_ms,
          (unsigned long long)d_rb_cnt,
          d_rb_us / 1000.0,
          d_inl_us / 1000.0,
          (int)g_ikd_rebuild_active.load(std::memory_order_relaxed),
          g_ikd_rebuild_max_us.load(std::memory_order_relaxed) / 1000.0,
          (unsigned long long)g_ikd_rebuild_max_size.load(std::memory_order_relaxed),
          (unsigned long long)g_ikd_root_rebuild_count.load(std::memory_order_relaxed));
      }

      /******* Publish points *******/
      if (p.path_en)
        publish_path(pubPath_);
      if (scan_pub_en)
        publish_frame_world(pubLaserCloudFull_);
      if (scan_pub_en && p.scan_body_pub_en)
        publish_frame_body(pubLaserCloudFull_body_);
      if (p.effect_pub_en)
        publish_effect_world(pubLaserCloudEffect_, measurement_.effectCloud());
      // if (map_pub_en) publish_map(pubLaserCloudMap_);

      /*** Debug variables ***/
      if (runtime_pos_log) {
        frame_num++;
        // Under async the worker may be mid-Add; reading ikdtree.size() here would
        // race it. Estimate from the scan-start size + this scan's add count.
        kdtree_size_end = async_map_en ? (kdtree_size_st + add_point_size) : ikdtree.size();
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (measured.match_time) / frame_num;
        aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
        aver_time_solve =
          aver_time_solve * (frame_num - 1) / frame_num + (measured.solve_time + solve_H_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + measured.solve_time / frame_num;
        if (fast_lio::timingLogAccepts(time_log_counter)) {
          const std::size_t slot = fast_lio::timingLogSlot(time_log_counter);
          T1[slot] = Measures.lidar_beg_time;
          s_plot[slot] = t5 - t0;
          s_plot2[slot] = feats_undistort->points.size();
          s_plot3[slot] = kdtree_incremental_time;
          s_plot4[slot] = kdtree_search_time;
          s_plot5[slot] = kdtree_delete_counter;
          s_plot6[slot] = kdtree_delete_time;
          s_plot7[slot] = kdtree_size_st;
          s_plot8[slot] = kdtree_size_end;
          s_plot9[slot] = aver_time_consu;
          s_plot10[slot] = add_point_size;
        }
        time_log_counter++;
        printf(
          "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  "
          "map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
          t1 - t0,
          aver_time_match,
          aver_time_solve,
          t3 - t1,
          t5 - t3,
          aver_time_consu,
          aver_time_icp,
          aver_time_const_H_time);
        if ((t3 - t1) > 0.05) {
          printf(
            "[ SPIKE ]: t1_to_pre_icp: %0.3f  icp: %0.3f  publish: %0.3f  total(t3-t1): %0.3f  feats: %d  tree: %d\n",
            (t_pre_icp - t1) * 1000,
            (t_update_end - t_update_start) * 1000,
            (t3 - t_update_end) * 1000,
            (t3 - t1) * 1000,
            feats_down_size,
            kdtree_size_st);
        }
        ext_euler = SO3ToEuler(state_point.offset_R_L_I);
        if (time_log_counter % 500 == 0 || time_log_counter == 1) {
          printf(
            "[ extrinsic ]: T_L_I = [%0.5f, %0.5f, %0.5f]  R_euler = [%0.3f, %0.3f, %0.3f] deg  bg = [%0.6f, %0.6f, "
            "%0.6f]  ba = [%0.6f, %0.6f, %0.6f]  grav = [%0.5f, %0.5f, %0.5f]\n",
            state_point.offset_T_L_I(0),
            state_point.offset_T_L_I(1),
            state_point.offset_T_L_I(2),
            ext_euler(0),
            ext_euler(1),
            ext_euler(2),
            state_point.bg(0),
            state_point.bg(1),
            state_point.bg(2),
            state_point.ba(0),
            state_point.ba(1),
            state_point.ba(2),
            state_point.grav[0],
            state_point.grav[1],
            state_point.grav[2]);
        }
        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " "
                 << state_point.pos.transpose() << " " << ext_euler.transpose() << " "
                 << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose() << " "
                 << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " "
                 << feats_undistort->points.size() << "\n";
        fout_jump << setw(20) << Measures.lidar_beg_time - first_lidar_time << " yaw_before_deg "
                  << euler_before_update(2) << " yaw_after_deg " << euler_cur(2) << " yaw_step_deg "
                  << (euler_cur(2) - euler_before_update(2)) << " pos_step_xy "
                  << (state_point.pos.head<2>() - pos_before_update.head<2>()).norm() << " feats_down "
                  << feats_down_size << " effct_feat_num " << effct_feat_num << " res_mean_last " << res_mean_last
                  << " solve_H_ms " << solve_H_time * 1000.0 << " icp_ms " << (t_update_end - t_update_start) * 1000.0
                  << "\n";
        fout_jump.flush();
        if (positionLog_) {
          dump_lio_state_to_log(positionLog_.get());
        }
      }
    }
  }

  /// Put the held prior map into the map backend at the alignment relocalization found.
  ///
  /// The map is delivered in its own frame; the alignment (T_map_lidar_odom) says where
  /// that frame sits in this run's. Transforming the MAP is what keeps the estimate
  /// continuous: moving the filter instead would teleport /Odometry, and every consumer
  /// downstream reads that as the platform having jumped. A later alignment replaces the
  /// tier from the delivered points again, so corrections never compound.
  ///
  /// Called from the scan loop only, after joinMapAdd(): the backend has a single-operator
  /// invariant and the subscription callback is not that operator.
  void installPriorMap()
  {
#ifdef USE_IVOX
    if (!prior_map_ds_ || prior_map_ds_->empty()) {
      return;
    }
    Eigen::Isometry3d T_map_lidar_odom;
    {
      std::lock_guard<std::mutex> lk(prior_map_align_mtx_);
      if (!prior_map_pending_) {
        return;
      }
      prior_map_pending_ = false;
      T_map_lidar_odom = prior_map_pending_T_;
    }
    if (!fast_lio::priorMapReinstallNeeded(prior_map_installed_, prior_map_installed_T_, T_map_lidar_odom)) {
      return;  // the producer re-confirming its own answer
    }

    // Transform by hand rather than through pcl::transformPointCloud: the point type carries
    // normal fields that esti_plane rewrites per iEKF iteration anyway, and rotating stale
    // normals into the new frame would only make them look meaningful.
    const Eigen::Isometry3d T_lidar_odom_map = T_map_lidar_odom.inverse();
    const Eigen::Matrix3f R = T_lidar_odom_map.linear().cast<float>();
    const Eigen::Vector3f t_off = T_lidar_odom_map.translation().cast<float>();
    PointCloudXYZI::Ptr in_frame(new PointCloudXYZI());
    in_frame->points.reserve(prior_map_ds_->points.size());
    for (const auto & p : prior_map_ds_->points) {
      PointType q = p;
      const Eigen::Vector3f v = R * Eigen::Vector3f(p.x, p.y, p.z) + t_off;
      q.x = v.x();
      q.y = v.y();
      q.z = v.z();
      in_frame->points.push_back(q);
    }
    in_frame->width = static_cast<uint32_t>(in_frame->points.size());
    in_frame->height = 1;

    if (!ikdtree.canReplacePinnedPoints(in_frame->points)) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(),
                            *this->get_clock(),
                            5000,
                            "[PRIOR-MAP] replacement exceeds pinned coarse capacity (%zu) or voxel coordinate range; "
                            "keeping the installed prior and transform",
                            ikdtree.maxVoxels());
      return;
    }
    const std::size_t evicted_before = ikdtree.evictedVoxels();
    const int live_before = ikdtree.size();
    ikdtree.clearPinned();
    // What comes back is what actually went in: the map backend's dedup grid is
    // first-point-wins, so where this run has already put a point the delivered map does not
    // displace it. Expected, and small early in a run — but a large shortfall means the live
    // map already covers the area and the prior map is adding little.
    const int added = observeMapOperation(
      admissionObserver(),
      fast_lio::AdmissionMapKind::kPriorInstall,
      [&] { return std::array<std::uint64_t, 3>{0, in_frame->size(), 0}; },
      [&] { return ikdtree.AddPinnedPoints(in_frame->points); });
    prior_map_installed_T_ = T_map_lidar_odom;
    prior_map_installed_ = true;
    ++prior_map_installs_;
    anchorEvidence(fast_lio::AnchorTraceReason::kReferenceChanged,
                   fast_lio::kAnchorEventReferenceChanged,
                   &anchor_trace_counters.reference_changed,
                   lidar_end_time,
                   true);
    // An anchor is a pose in the reference it was taken against; the replaced one takes its anchors with it.
    changeStationaryReference(fast_lio::AnchorDrop::kReferenceChanged, lidar_end_time);

    const Eigen::Vector3d t = T_map_lidar_odom.translation();
    const Eigen::Matrix3d & r = T_map_lidar_odom.linear();
    RCLCPP_WARN(this->get_logger(),
                "[PRIOR-MAP] install #%u at T_map_lidar_odom=[%.2f %.2f %.2f] yaw=%.1f deg: %d of %zu point(s) "
                "taken into %zu pinned voxel(s) (the rest fell where this run's map already had one), "
                "map %d -> %d pts",
                prior_map_installs_,
                t.x(),
                t.y(),
                t.z(),
                std::atan2(r(1, 0), r(0, 0)) * 180.0 / M_PI,
                added,
                in_frame->size(),
                ikdtree.pinnedVoxels(),
                live_before,
                ikdtree.size());
    if (ikdtree.evictedVoxels() > evicted_before) {
      RCLCPP_ERROR(this->get_logger(),
                   "[PRIOR-MAP] the install evicted %zu live voxel(s): the map backend is at capacity "
                   "(ivox_max_voxels=%zu). Raise it, or the prior map is competing with this run's own map.",
                   ikdtree.evictedVoxels() - evicted_before,
                   ikdtree.maxVoxels());
    }
#endif
  }

  void map_publish_callback()
  {
    // publish_map() flattens the ikd-tree (a tree read). This 1 Hz timer shares the
    // single executor thread with timer_callback, so it can fire BETWEEN scans while
    // the off-thread Add is in flight — join first to keep the single-operator invariant.
    if (params_.values().map_pub_en) {
      if (async_map_en)
        joinMapAdd(admissionObserver());
      publish_map(pubLaserCloudMap_);
    }
  }

  void map_save_callback(std_srvs::srv::Trigger::Request::SharedPtr req,
                         std_srvs::srv::Trigger::Response::SharedPtr res)
  {
    RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
    if (pcd_save_en) {
      save_to_pcd();
      res->success = true;
      res->message = "Map saved.";
    } else {
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
  // Each input stream's filter for stamps far ahead (acceptInputEpoch), declared before the subscriptions whose
  // callbacks use it, under mtx_buffer.
  fast_lio::InputLeadFilter lidar_input_lead_;
  fast_lio::InputLeadFilter imu_input_lead_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImu>::SharedPtr sub_imu_fpa_;
  rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImubias>::SharedPtr sub_imu_bias_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
  rclcpp::Subscription<shm_msgs::msg::PointCloud8mAndPose>::SharedPtr sub_pcl_shm_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_twist_cov_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool offline_step_en_ = false;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr subOfflineStep_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fe_z_drift_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mapping_enabled_sub_;

  fast_lio::FrontendParameters params_;  // the configuration the constructor read; read through values()
  // Only the default mutually-exclusive FE callback group writes this POD. The map worker never receives it.
  fast_lio::TimeAdmissionAccounting admission_{};
  fast_lio::FrontendMeasurement measurement_{params_.values()};  // the filter's measurement model; kf holds its address
  // The wheel update's samples and counts. The callbacks fill it through wheel_speed_input under mtx_twist; the scan
  // thread reads it under mtx_twist and keeps its slip-window IMU history and counts without a lock.
  fast_lio::WheelSpeedInput wheel_speed_;
  int effect_feat_num = 0, frame_num = 0;
  double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
                         aver_time_solve = 0, aver_time_const_H_time = 0;
  double epsi[23] = {0.001};

  FileHandle positionLog_;
  ofstream fout_pre, fout_out, fout_dbg, fout_jump, fout_kalman;

  std::string prior_map_pcd_;
  std::size_t last_reported_evicted_voxels = 0;  // last value the capacity WARN reported
  /// Downsampled prior map, held in ITS OWN (delivered) frame so a later, better alignment
  /// can re-transform the same points instead of compounding onto the last one. Only filled
  /// on the relocalization path; empty and freed on the as-delivered path.
  PointCloudXYZI::Ptr prior_map_ds_;
  /// Alignment currently installed, and the one waiting to be. Written by the
  /// /relocalization/result callback, read and cleared by the scan loop: the map backend has
  /// a single-operator invariant and a subscription callback is not that operator.
  std::mutex prior_map_align_mtx_;
  Eigen::Isometry3d prior_map_pending_T_ = Eigen::Isometry3d::Identity();  // T_map_lidar_odom
  bool prior_map_pending_ = false;
  Eigen::Isometry3d prior_map_installed_T_ = Eigen::Isometry3d::Identity();
  bool prior_map_installed_ = false;
  unsigned prior_map_installs_ = 0;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr prior_map_align_sub_;
  std::uint64_t entry_view_round_ = 0;  // the recovery episode the entry view was last opened for
};

#ifndef FASTLIO_AS_COMPONENT
namespace
{
void relayStandaloneSigterm(int) noexcept
{
  const int savedErrno = errno;
  ::kill(::getpid(), SIGINT);
  errno = savedErrno;
}
}  // namespace

int main(int argc, char ** argv)
{
  // Preserve the pre-init action, not rclcpp's subsequently uninstalled SIGTERM handler.
  struct sigaction previousTermAction{};
  if (::sigaction(SIGTERM, nullptr, &previousTermAction) != 0) {
    const int signalError = errno;
    RCLCPP_ERROR(rclcpp::get_logger("fastlio_mapping"), "read pre-init SIGTERM action failed (errno=%d)", signalError);
    return 1;
  }

  // Keep rclcpp's deferred SIGINT shutdown path on Foxy and Humble.
  rclcpp::init(argc, argv);
  struct sigaction relayAction{};
  relayAction.sa_handler = relayStandaloneSigterm;
  if (::sigemptyset(&relayAction.sa_mask) != 0) {
    const int signalError = errno;
    RCLCPP_ERROR(rclcpp::get_logger("fastlio_mapping"), "initialize SIGTERM relay mask failed (errno=%d)", signalError);
    rclcpp::shutdown();
    return 1;
  }
  if (::sigaction(SIGTERM, &relayAction, nullptr) != 0) {
    const int signalError = errno;
    RCLCPP_ERROR(
      rclcpp::get_logger("fastlio_mapping"), "install standalone SIGTERM relay failed (errno=%d)", signalError);
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(std::make_shared<LaserMappingNode>());

  if (rclcpp::ok())
    rclcpp::shutdown();
  // Drain any in-flight off-thread ikd-tree Add and join the worker before the final
  // map read/save, so the tree is quiescent and the worker thread exits cleanly.
  stopMapWorker(nullptr);
  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. pcd save will largely influence the real-time performences **/
  // Drain the async writer queue and join before reading pcl_wait_save, so the final
  // map contains every enqueued scan and there is no race with the writer thread.
  stopPcdWriter();
  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    string file_name = string("scans.pcd");
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    cout << "current scan saved to /PCD/" << file_name << endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
  }

  if (runtime_pos_log) {
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    const string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    FileHandle timingLog(std::fopen(log_dir.c_str(), "w"));
    if (!timingLog) {
      RCLCPP_WARN(rclcpp::get_logger("fastlio_mapping"), "Timing log not written: failed to open %s", log_dir.c_str());
    } else {
      fprintf(timingLog.get(),
              "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree "
              "size st, tree size end, add point size, preprocess time\n");
      const std::size_t rows = fast_lio::timingLogRows(time_log_counter);
      for (std::size_t row = 0; row < rows; row++) {
        const std::size_t i = fast_lio::timingLogRowSlot(time_log_counter, row);
        fprintf(timingLog.get(),
                "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",
                T1[i],
                s_plot[i],
                int(s_plot2[i]),
                s_plot3[i],
                s_plot4[i],
                int(s_plot5[i]),
                s_plot6[i],
                int(s_plot7[i]),
                int(s_plot8[i]),
                int(s_plot10[i]),
                s_plot11[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot9[i]);
        s_vec2.push_back(s_plot3[i] + s_plot6[i]);
        s_vec3.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot[i]);
      }
    }
  }

  // Do not overwrite an action installed by another owner or rclcpp shutdown.
  struct sigaction currentTermAction{};
  if (::sigaction(SIGTERM, nullptr, &currentTermAction) != 0) {
    const int signalError = errno;
    RCLCPP_ERROR(rclcpp::get_logger("fastlio_mapping"), "read exit SIGTERM action failed (errno=%d)", signalError);
    return 1;
  }
  if (!(currentTermAction.sa_flags & SA_SIGINFO) && currentTermAction.sa_handler == relayStandaloneSigterm &&
      ::sigaction(SIGTERM, &previousTermAction, nullptr) != 0) {
    const int signalError = errno;
    RCLCPP_ERROR(
      rclcpp::get_logger("fastlio_mapping"), "restore pre-init SIGTERM action failed (errno=%d)", signalError);
    return 1;
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

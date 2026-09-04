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
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <fixposition_driver_msgs/msg/fpa_imu.hpp>
#include <fixposition_driver_msgs/msg/fpa_imubias.hpp>
#include <fstream>
#include <ikd-Tree/ikd_Tree.h>
#include <iomanip>
#include <ivox/ivox.hpp>  // option B: drop-in voxel-grid map backend (compile with -DUSE_IVOX=ON)
#include <limits>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <shm_msgs/msg/point_cloud8m_and_pose.hpp>
#include <so3_math.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "IMU_Processing.hpp"
#include "reanchor_gate.hpp"
#include "adaptive_downsample.hpp"
#include "runaway_watchdog.hpp"
#include "static_evidence.hpp"
#include "preprocess.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN],
  s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool ikd_profile = false;  // [ikd-profile] per-scan rebuild-split line, gated by env FLIO_IKD_PROFILE
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
constexpr double kLidarLoopBackSec = 1.0;  // stamp jump back beyond this = real loop back (bag restart), else a stray sample
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1,
    pcd_index = 0;
// Position-observability metric, refreshed each h_share_model call: eigenvalues
// of the normal information matrix M = sum(n n^T) over the matched plane normals.
// pos_obs_z_weak = |Z component| of the weakest eigenvector; →1 means Z is the
// least-constrained direction (degenerate Z), the root trigger of the bag2
// divergence (see memory bag2-divergence-is-flio2-frontend).
double pos_obs_eig_min = 0.0, pos_obs_eig_max = 0.0, pos_obs_z_weak = 0.0;
Eigen::Matrix3d pos_obs_M = Eigen::Matrix3d::Zero();  // Σ n nᵀ of the last update (world frame), for the [SCAN] trace
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

// Planar-motion (zero body-vertical-velocity) soft constraint — the A'-2 fix.
//   When LiDAR loses Z observability (pos_obs_z_weak high), the iEKF dumps the
//   unexplained residual into the accel bias and the state flies off. This adds a
//   soft pseudo-measurement "body-frame vertical velocity ≈ 0" (a ground-robot
//   non-holonomic prior, like the wheel-odom update) that supplies the missing
//   vertical information and bounds the runaway at its source.
//   BODY frame → it does NOT fight legitimate slope motion: world-frame vertical
//   velocity on a ramp comes from the robot's pitch (rotation state), not from
//   body-frame vz. Gated on z_weak so it has zero effect when Z is well observed.
bool planar_constraint_en = false;
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
bool gravity_align_en = false;
double gravity_align_noise = 0.05;         // unit-vector (≈rad) std of the leveling prior
double gravity_align_z_weak_thresh = 0.3;  // engage only when pos_obs_z_weak exceeds this
double gravity_align_accel_tol = 0.05;     // |‖a‖−g|/g must be below this (low-linear-accel gate)
double gravity_align_gyro_tol = 0.35;      // rad/s; skip during fast turns (lever-arm safety)
// 2026-08-31 additions (m20_0831 doorway pitch anatomy) — all default-off:
double gravity_align_window_s = 0.0;       // [L2] >0: measurement = window-mean specific force
bool gravity_align_continuous = false;     // [L2] engage every scan (gates still apply)
double gravity_align_hold_s = 0.0;         // [L1] hold the degraded-strength prior after a trigger
double gravity_align_noise_degraded = 0.0; // [L1] tighter sigma while triggered/held (0 = use base)
double attitude_cov_floor_deg = 0.0;       // [L3] roll/pitch error-state std floor while triggered
double gravity_align_mag_ref = 0.0;        // magnitude-gate anchor; 0 = IMU-init g (biased if init in motion)
double gravity_align_grav_leak = 1.0;      // [v4] per-engaged-scan multiplicative variance leak on the S2
                                           // gravity block (fading memory); 1.0 = off
double gravity_align_grav_cap_deg = 3.0;   // [v4] leak cap: grav tangent std ceiling (deg) — bounds how far
                                           // the gravity state may be steered per run
bool imu_init_require_still_ = false;      // quasi-static IMU-init gate (see IMU_Processing.hpp)
double imu_init_still_tol_ = 0.03;
double imu_init_still_timeout_s_ = 20.0;
struct GravWinAgg {
  double t;
  Eigen::Vector3d sum;
  int n;
  double max_w;
};
std::deque<GravWinAgg> grav_win;           // scan-thread only
double gravity_align_last_trig = -1e18;

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
bool divergence_guard_en = true;
int divergence_guard_min_eff = 50;         // effct_feat_num below this = degenerate scan
int divergence_guard_streak = 5;           // consecutive degenerate scans → enter degraded
double divergence_guard_max_speed = 30.0;  // m/s; body-speed clamp while degraded (headroom over any platform speed)
// Engulfment leading indicator (2026-08-14, m20_0814_degrade forensics): when a
// low-mounted lidar wades into knee-high vegetation the FAR field vanishes from
// the raw scan (healthy: ~50% of returns beyond guard_engulf_far_range; engulfed:
// ~5%) BEFORE effct collapses. The last few "valid" updates on sub-metre grass
// inject velocity/tilt (measured 2.14 m/s vs 0.5 m/s ground truth, 6° tilt)
// and, worse, insert drifting garbage into the map for divergence_guard_streak
// scans before the lagging effct-streak criterion freezes it — after which
// re-lock is impossible (map poisoned) and survival is a scheduling coin-flip
// (identical runs: clean vs km-runaway). Freezing the map on the SCAN-level
// signal makes the guard leading instead of lagging. Computed on the raw
// downsampled scan (a scene property, independent of matching outcome).
bool guard_engulf_en = true;               // arm the far-field-collapse trigger
double guard_engulf_far_range = 2.0;       // m; horizontal radius separating near/far
double guard_engulf_far_frac_min = 0.15;   // enter when far-field fraction < this
int guard_engulf_streak = 2;               // consecutive engulfed scans → degraded
// Release hysteresis + velocity reset (m20_0814_degrade far_frac time series):
// the brush is FOUR separate plunges (t≈146,165,175,191) with far_frac
// hovering 0.04–0.39 on every shoulder. Instant release on one scan ≥ min
// flaps on those shoulders and unfreezes the map for half-grass scans; and
// whatever velocity the iEKF holds at release is unobserved IMU integration
// (gravity-align saw 6–7° tilt), which then contaminates the inter-plunge
// map before the next plunge — that is the run-to-run coin flip. So: release
// only after guard_engulf_release_scans consecutive scans with far_frac ≥
// guard_engulf_far_frac_release, and ZERO velocity at release so the next
// good scans re-estimate it from correspondences instead of trusting the
// blind stretch. Position/attitude/biases are kept.
double guard_engulf_far_frac_release = 0.35;  // release needs far field this healthy
int guard_engulf_release_scans = 5;           // ...for this many consecutive scans
int consec_engulf_clear = 0;                  // running healthy-scan streak while latched
bool engulf_latched = false;                  // sticky engulfed state (entered, not yet released)
// Scan-hole guard: when the gap between consecutive processed scans exceeds
// guard_hole_min_sec, the IMU-only propagation across the hole is replaced by
// a bounded constant-velocity hold from the pre-hole state before the lidar
// update runs. Measured on m20_0814_degrade with injected holes over open
// ground: <=2 s holes recover cleanly, but a 5 s hole let the IMU-only
// velocity reach 4.1 m/s (walking 1.8) and the position drift metres off the
// map (effct 3400 -> 400) before the runaway reset — quadratic IMU divergence
// that ground-only geometry cannot re-observe. Default OFF.
bool guard_hole_en = false;
double guard_hole_min_sec = 0.3;    // s; scan gap that counts as a hole (nominal 0.1)
double guard_hole_max_accel = 1.0;  // m/s^2; |vel - vel_pre_hole| is capped at this * gap
bool guard_hole_reanchor_pos = true;  // also reset pos to pos_pre + vel_pre * gap
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
bool zupt_en = true;
double zupt_gyro_thresh = 0.05;         // rad/s; max |omega| over the scan's IMU window
double zupt_acc_span_ratio = 0.02;      // (max-min)|a| / mean|a| — free of the IMU's unit scale
double zupt_wheel_speed_thresh = 0.05;  // m/s; wheel speed below this reads as parked
double zupt_wheel_max_age = 0.5;        // s; older wheel samples are not evidence
bool zupt_require_wheel = false;        // true = hold only when a fresh wheel sample agrees
V3D zupt_anchor_pos = V3D::Zero();      // last pose a healthy scan confirmed
SO3 zupt_anchor_rot;
bool zupt_anchor_valid = false;
bool zupt_active = false;
int zupt_hold_scans = 0;
double last_twist_stamp = -1.0;  // newest wheel-speed sample time (s); -1 = none yet
double last_twist_speed = 0.0;   // |v| of the newest wheel sample (m/s)
// Re-anchor gate: the guard used to unfreeze the map the moment the scene
// looked healthy again, which is where a recoverable blind stretch became a
// permanent runaway — the pose error survived the release, the map was rebuilt
// around it, and residuals stayed small while the map walked off with the
// estimate. Verify against the still-frozen map first (reanchor_gate.hpp).
fast_lio::ReanchorParams reanchor_params;
fast_lio::ReanchorGate reanchor_gate;
// What to do when verification never converges: "hold" keeps the map frozen and
// the odometry flagged (an external relocalisation has to solve it), while
// "reset_map" rebuilds the local map at the current pose — the robot keeps
// working, having traded absolute anchoring for a live estimate.
std::string reanchor_on_fail = "hold";
// Runaway watchdog: the guard and the re-anchor gate both key off the scan, and
// the worst drift keeps the scan looking healthy — the map is rebuilt around
// the wrong pose every frame, so correspondences and residuals stay good while
// the estimate walks away (dog 2: effct 2000-3500, res 0.026 m, 45 min at
// 0.5 m/s, parked). Only evidence from outside the estimator can see that.
fast_lio::RunawayParams runaway_params;
fast_lio::RunawayState runaway_state;
// hold = flag it lost (map freezes, downstream stops consuming the pose),
// reset_map = rebuild the local map here, fault_exit = die and let the
// supervisor restart the pipeline.
std::string runaway_on_trip = "hold";
double scan_far_frac = 1.0;                // per-scan telemetry: fraction of raw points beyond far_range
// [RES] telemetry: converged point-to-plane residual by body range bin (0-2,2-5,5-10,10-20,20-40,40+ m):
// count and sum of squares from the last h_share_model call of the scan (degeneracy_debug only).
constexpr int kResBins = 6;
constexpr double kResBinEdges[kResBins] = {2.0, 5.0, 10.0, 20.0, 40.0, 1e9};
int res_bin_n[kResBins] = {0};
double res_bin_ss[kResBins] = {0.0};

int consec_engulf = 0;                     // running engulfed-scan streak
int consec_low_eff = 0;                    // running degenerate-scan streak counter
bool flio_map_frozen = false;              // degraded: skip map_incremental (no garbage)
bool flio_degraded_odom = false;           // degraded: inflate published pose covariance

// Canopy detector — a SCENE detector, orthogonal to pos_obs_z_weak. Inside tall
// crop the matched normals stay directionally diverse (the canopy is volumetric
// and self-similar), so the Σnnᵀ observability metric reads HEALTHY while the Z
// constraint is physically false: each scan's small Z error is baked into the
// map and the next scan matches that freshly-laid layer (a map-mediated ratchet
// — the measured −60 m in-canopy dive is R²≈0.997 smooth). The reliable
// signature is the collapse of NEAR-FIELD GROUND returns (measured on the giant
// 0702 bag: 79–99 % healthy → 42–60 % in canopy) plus elevated near-field point
// height. Both statistics use the LiDAR/cloud BODY frame so they are immune to
// the very Z drift they must detect. Hysteresis + debounce below.
bool canopy_detect_en = false;
bool canopy_debug = false;            // per-scan [CANOPY] stderr line
double canopy_near_range = 20.0;      // near-field horizontal radius, cloud frame (m)
double canopy_ground_z = -0.47;       // ground height in the CLOUD frame (giant rig measured)
double canopy_ground_halfband = 0.8;  // ± band around canopy_ground_z counted as ground (m)
double canopy_normal_z_min = 0.966;   // |n_z| ≥ cos(15°) ⇒ horizontal surface
double canopy_frac_enter = 0.65;      // ground fraction below this (AND zp95 above) → enter vote
double canopy_frac_exit = 0.75;       // ground fraction above this → exit vote
double canopy_zp95_enter = 1.5;       // near-field body-z p95 must exceed this to enter (m)
int canopy_min_near = 50;             // fewer near-field points than this = neutral scan
int canopy_enter_scans = 10;          // consecutive enter votes to switch on (~1 s at 10 Hz)
int canopy_exit_scans = 30;           // consecutive exit votes to switch off (~3 s)
// per-scan raw statistics (written by h_share_model, consumed once per scan in the main loop)
double canopy_ground_frac = 1.0, canopy_zp95 = 0.0;
int canopy_near_count = 0;
bool canopy_mode = false;
int canopy_enter_streak = 0, canopy_exit_streak = 0;

// Ground recovery (selection-layer repair): re-admit true-ground returns that
// the 5-NN plane gates reject under ground-hugging clutter (grass, crops,
// debris — content-agnostic) — see the block comment in h_share_model.
// SELF-GATED by per-sector fit quality + the sensor-height sanity gate; no
// scene classifier involved (the canopy detector is telemetry only). Default OFF.
bool ground_recover_en = false;
double ground_recover_scan_band = 0.15;    // |dist to sector ground plane| accept band (m)
int ground_recover_min_inliers = 100;      // PER-SECTOR fit support gate (8 sectors)
double ground_recover_max_residual = 1.0;  // |z - map lower envelope| accept gate (m)
double ground_recover_sector_tol = 0.25;   // sector-vs-global plane height agreement (m)
int ground_recover_max_rows = 4000;        // per-scan cap on recovered rows
int canopy_ground_recovered = 0;           // diag: rows recovered last h_share call

// TEM (Terrain Elevation Memory): the ground-recovery rows above still measure
// their residual against the SELF-BUILT map's lower envelope — a reference that
// drifts with the vehicle, so the map-mediated Z ratchet is only slowed, never
// stopped (measured: −54 m class remains). TEM swaps that reference for a
// world-frame terrain height grid that is LEARNED while Z observability is
// healthy and FROZEN while it is degraded, exploiting the one physical
// invariant available without external aiding: terrain does not move within a
// session. Continuous quality weighting (no scene classifier): the learning
// rate is a dead-banded ramp of the effective-set vertical-normal ground share
// (egf — the very statistic whose collapse IS the pathology, computed before
// the recovery pass so recovered rows cannot vouch for themselves). The dead
// band is a physical requirement, not a tuning nicety: canopy dwell is ~700 s
// at ~0.1 m/s drift, so any nonzero learning rate there eventually hands the
// memory to the drift. Cells store a height variance fed by intra-cell spread
// AND frame-to-frame disagreement — slopes and residual drift inflate it, and
// the row weight collapses with it (online slope statistics; no fixed slope
// prior, no site-specific extents: hashed grid). Rows keep every existing
// recovery gate; only the residual reference and the row weight change, and
// rows fall back to the lower envelope wherever TEM has no served coverage.
bool tem_en = false;
double tem_cell = 5.0;        // grid cell size (m); terrain is smooth at this scale
double tem_tau = 2.0;         // s, per-cell EMA time constant at full health (q=1)
double tem_egf_floor = 0.30;  // egf at/below this → learning fully frozen
double tem_egf_ref = 0.60;    // egf at/above this → full learning rate
double tem_sigma0 = 0.15;     // m, row weight = sigma0/(sigma0 + cell std)
struct TemCell
{
  double z;
  double var;
  double w;
};
std::unordered_map<int64_t, TemCell> tem_grid;
std::vector<Eigen::Vector3d> tem_samples;              // this scan's fit-band inliers (body frame)
double tem_egf_ema = 1.0;                              // smoothed egf (starts healthy: cold start must not freeze)
int tem_rows = 0, tem_ext_rows = 0, tem_env_rows = 0;  // diag: rows per residual reference
// Slope-clamped extrapolation (measured first-visit failure: a mission that
// enters a degraded area it never covered while healthy has no served cells
// there — every row fell back to the drifting envelope and the −55 m ratchet
// survived TEM v1 untouched). The honest information terrain smoothness still
// carries: height at distance d from the anchored network lies within
// slope_hi * d of the anchor, where slope_hi is an ONLINE statistic of the
// anchored cells' own gradients (no fixed site prior). A small fraction of
// recovered rows uses that envelope as a weak absolute reference with a
// trust-region clamp (degraded-RTK lesson: unclamped absolute pulls run away)
// while the rest keep the envelope reference for local Z observability — the
// absolute channel fights secular drift, it must not cannibalize the local
// restoration that is the recovery channel's primary job.
double tem_slope_mean = 0.01, tem_slope_var = 1e-4;  // online |dz|/dxy statistic (healthy pairs)
bool tem_ext_ok = false;                             // per-scan extrapolation anchor (nearest served cells)
double tem_ext_z = 0.0, tem_ext_sigma = 1.0;
// Graph-side drift feedback (terrain anchoring, experimental — default OFF).
// The graph (GPS-pinned) publishes d = how far this front-end's Z has drifted
// since its epoch (/lio_slam/fe_z_drift, gated on GPS-factor freshness). With a
// FRESH d the TEM learns from DRIFT-CORRECTED samples (z + d) at a high fixed
// rate even where egf says the scene is degraded — the reason egf had to
// freeze learning (drift poisoning) is being measured and removed upstream.
// This is what closes the first-visit gap the self-memory cannot: cells get
// anchored along the vehicle's own trail in real time. Model anchoring only —
// the state is still constrained exclusively through observed ground returns.
// Stale d (graph following the front-end in GPS-degraded stretches) degrades
// to the plain egf-gated self-learning.
bool tem_anchor_en = false;
double tem_anchor_max_age = 3.0;  // s, drift note older than this = stale ("no anchor")
std::mutex tem_anchor_mtx;        // handler (executor thread) vs main-loop consumer
double tem_anchor_d = 0.0, tem_anchor_sigma = 0.2, tem_anchor_stamp = -1.0;

// Constraint-set cleaning (clutter deweight; EXPERIMENTAL, default OFF, needs
// ground_recover_en for the sector ground planes). The Z ratchet lives in the
// ACCEPTED pseudo-plane rows: 5 map points cannot measure planarity (measured
// at map density: 5-NN thickness p50 0.004/0.005 ground vs clutter — blind;
// 15-NN separates 12x, 0.011 vs 0.139-0.165). For accepted near-field rows in
// the CLUTTER BAND above the fitted ground plane (the band itself is the
// classifier — the canopy-area ground band is a ~0.16 m grass layer where a
// thickness test would hurt true ground, so ground-band rows are never
// touched), a one-off k=15 map query measures neighborhood thickness and the
// row weight collapses continuously with it. Two protections from the
// measured dose-response failures: (a) |n_z| taper — horizontal-normal rows
// (hedge walls) carry the lateral information that keeps clutter scenes
// conditioned, so only vertical-normal rows (the Z-bias carriers) take the
// full cut; (b) a weight floor — rows are never removed, and the scaled
// normals make the pos-observability metric quality-weighted for free.
bool clutter_deweight_en = false;
double clutter_th_lo = 0.05;        // m, 15-NN thickness below this = real plane, no cut
double clutter_th_hi = 0.20;        // m, thickness at/above this = full cut
double clutter_min_weight = 0.15;   // floor: never remove a row outright (conditioning)
int clut_tested = 0, clut_cut = 0;  // diag, last h_share call
double clut_msum = 0.0;

inline int64_t temKeyIdx(int64_t kx, int64_t ky)
{
  return (kx << 32) ^ (ky & 0xffffffff);
}

inline int64_t temKey(double x, double y)
{
  return temKeyIdx(static_cast<int64_t>(std::floor(x / tem_cell)), static_cast<int64_t>(std::floor(y / tem_cell)));
}

// Recover the cell indices from a key (valid for |k| < 2^31; low half carries
// ky's low 32 bits, high half kx's — the XOR never mixes them).
inline void temKeyToIdx(int64_t key, int64_t & kx, int64_t & ky)
{
  ky = static_cast<int32_t>(key & 0xffffffff);
  kx = key >> 32;
}

// Bilinear height lookup over the 2x2 cell-center neighborhood; cells below the
// serving weight are skipped and the coefficients renormalized. Requires at
// least half the interpolation mass to be served — partial coverage degrades
// gracefully instead of extrapolating.
inline bool temLookup(double x, double y, double & z_out, double & std_out)
{
  constexpr double kTemMinServeW = 2.0;  // ~2 healthy scans before a cell serves
  const double u = x / tem_cell - 0.5, v = y / tem_cell - 0.5;
  const double bu = std::floor(u), bv = std::floor(v);
  const double fx = u - bu, fy = v - bv;
  double csum = 0.0, zsum = 0.0, vsum = 0.0;
  for (int di = 0; di < 2; di++)
    for (int dj = 0; dj < 2; dj++) {
      const double cx = (bu + di + 0.5) * tem_cell, cy = (bv + dj + 0.5) * tem_cell;
      const auto it = tem_grid.find(temKey(cx, cy));
      if (it == tem_grid.end() || it->second.w < kTemMinServeW)
        continue;
      const double c = (di ? fx : 1.0 - fx) * (dj ? fy : 1.0 - fy);
      csum += c;
      zsum += c * it->second.z;
      vsum += c * it->second.var;
    }
  if (csum < 0.5)
    return false;
  z_out = zsum / csum;
  std_out = std::sqrt(std::max(vsum / csum, 0.0));
  return true;
}

// P2 VoxelMap-lite: anisotropic per-voxel surface statistics (EXPERIMENTAL,
// default OFF, content-agnostic — no scene gate, no band membership). Successor
// to the clutter-band deweight after its measured verdict: the remaining Z bias
// COHABITS with the load-bearing Z constraints in the 0.15-0.5 m layer above
// ground, so any per-row keep/cut decision keyed on layer membership either
// misses the bias (band lo 0.5) or guts the constraint (band lo 0.25 -> FE
// 796 m). This decomposes WITHIN the constraint instead: a world-frame voxel
// hash accumulates first/second moments of the matched cloud across scans
// (online replica of the offline probe's 20-scan accumulation — the density at
// which thickness IS measurable, 12x ground/clutter separation), and each
// accepted row is reweighted by the measured surface spread ALONG ITS OWN
// FITTED NORMAL: sigma_dir^2 = n^T Cov(voxel) n,
//   m = sigma_r / sqrt(sigma_r^2 + sigma_dir^2)   (information ~ m^2)
// i.e. the implicit per-row noise is inflated to the surface's measured
// roughness in the constraint direction. Thin ground keeps m~1; a fluffy
// grass/canopy voxel collapses its VERTICAL-normal rows toward the honest
// ~0.15 m sigma while HORIZONTAL-normal rows through the same voxel keep their
// lateral information (n^T Cov n is small sideways) — the direction-resolved
// split that row-level granularity could not express. Safety properties vs the
// measured failures: evidence-based per voxel per direction (not layer
// membership), continuous with a floor (rows never removed), voxels below
// vxl_min_pts observations are untouched (first visit = baseline behavior),
// recovery rows are never rescaled (their vertical channel is the load-bearing
// counterweight). Cell moments freeze at vxl_freeze_pts so a later secular
// drift cannot smear the statistics it is being judged against.
struct VxlCell
{
  float mean[3];  // world-frame mean (float ok: ~1e-4 m precision at 1 km)
  float m2[6];    // centered comoment sums, upper triangle xx,xy,xz,yy,yz,zz
  float zq;       // SGD 10th-percentile tracker of z (cell lower mode)
  int32_t n;
};
std::unordered_map<int64_t, VxlCell> vxl_grid;
bool vxl_en = false;
double vxl_size_xy = 1.0;         // m; XY footprint pools enough returns per cell
double vxl_size_z = 0.25;         // m; thin Z slabs separate soil from the grass layer above
double vxl_sigma_r = 0.05;        // m; nominal point-to-plane noise a clean row carries
double vxl_min_weight = 0.10;     // floor: never remove a row outright (conditioning)
double vxl_max_range = 30.0;      // m body range for both learning and weighting (bounds memory)
int vxl_min_pts = 8;              // below this: no verdict, row untouched
int vxl_freeze_pts = 300;         // cell moments freeze here (drift-smear cap)
int vxl_tested = 0, vxl_cut = 0;  // diag, last h_share call
double vxl_msum = 0.0, vxl_sd_sum = 0.0;
double vxl_vert_msum = 0.0;  // diag: m over |n_z|>0.8 rows (the Z-bias carriers)
int vxl_vert_cnt = 0;

// P3 within-voxel decomposition (NEGATIVE — measured; keep OFF, needs vxl_en):
// residual-reference swap for ground-class rows onto the voxel column's lower
// envelope (zq = 10th-pct of the lowest mature cell). Design intent was to
// break the per-scan re-manufacturing of the ratchet's reference. MEASURED
// (0702): the swap MANUFACTURES A STRONGER RATCHET INSTEAD — with ~1500
// near-field rows/scan a cell reaches min_pts within <1 s, so the envelope
// follows the drift by cell turnover with near-zero lag, and a LOWER-quantile
// estimator has a built-in positive residual offset against the ground return
// population (telemetry: h_avg +0.03..0.06 every scan, never decaying) =
// constant downward pull. P2's natural post-dive recovery (-47 -> +3) was
// replaced by a monotonic second-half grind (-20 -> -33), tail divergence to
// -156 (3x worse than doing nothing), /odom breached 16 m. The dilemma is
// structural, not a tuning miss: serve-fast = co-drifting reference with a
// quantile offset (this failure); serve-frozen-only = absolute injection =
// the closed B3 line, at ~6x tema's row authority (do NOT try); symmetric
// median = no offset but no resistance either (residual ~0 against a
// co-drifting median by construction). No reprocessing of the self-built data
// stream can create the independent information the ratchet requires.
bool vxl_env_en = false;
int vxl_env_min_pts = 20;     // cell mass before it can serve as envelope
double vxl_env_band = 0.12;   // m, |z - env| within this = ground-class row
double vxl_env_nz_min = 0.5;  // fitted |n_z| below this = lateral carrier, never swapped
int vxl_env_cnt = 0;          // diag, last h_share call
double vxl_env_habs = 0.0;

inline int64_t vxlKeyIdx(int64_t kx, int64_t ky, int64_t kz)
{
  return ((kx & 0x1FFFFF) << 42) | ((ky & 0x1FFFFF) << 21) | (kz & 0x1FFFFF);
}

// 21 bits per axis (±2^20 cells; at 0.25 m Z that is ±262 km — ample).
inline int64_t vxlKey(double x, double y, double z)
{
  return vxlKeyIdx(static_cast<int64_t>(std::floor(x / vxl_size_xy)),
                   static_cast<int64_t>(std::floor(y / vxl_size_xy)),
                   static_cast<int64_t>(std::floor(z / vxl_size_z)));
}

bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
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
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
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
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

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
// 自适应降采样:单帧计算越过预算就把扫描体素调粗,让精度降级而不是延迟降级。
// Adaptive downsampling: quality degrades instead of latency when a scan costs too much.
fast_lio::AdaptiveDownsampleParams adaptive_ds_params;
fast_lio::AdaptiveDownsampleState adaptive_ds_state;
size_t dropped_scan_total = 0;

bool latency_diag_en = true;
double latency_log_period_sec = 1.0;
bool kalman_channel_diag_en = false;
SteadyTimePoint current_lidar_receive_time;
SteadyTimePoint last_latency_log_time;
LatencyStats sensor_to_odom_latency_stats;

bool wheel_odom_en = false;
string wheel_topic = "/lio/twist";
double wheel_speed_scale = 1.0;
double wheel_vel_noise_vx = 0.10;
double wheel_vel_noise_vy = 0.05;
double wheel_vel_noise_vz = 0.10;
M3D R_robot_to_imu = Eye3d;
deque<Eigen::Vector3d> twist_buffer;
mutex mtx_twist;

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

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom & s)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
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

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
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
static void trim_lidar_buffer()
{
  const bool count_bounded = max_buffered_scans > 0;
  const bool age_bounded = max_scan_backlog_sec > 0.0;
  if (!count_bounded && !age_bounded) return;
  const size_t cap = count_bounded ? static_cast<size_t>(max_buffered_scans) : 0u;
  const size_t keep_front = lidar_pushed ? 1u : 0u;
  size_t dropped = 0;
  auto over_cap = [&]() {
    if (lidar_buffer.size() <= keep_front) return false;
    if (count_bounded && lidar_buffer.size() > cap) return true;
    // 队首(除去已被取走那一帧)比队尾旧过预算 → 它已经没有价值了。
    if (age_bounded && time_buffer.size() > keep_front &&
        time_buffer.back() - time_buffer[keep_front] > max_scan_backlog_sec)
      return true;
    return false;
  };
  while (over_cap()) {
    lidar_buffer.erase(lidar_buffer.begin() + keep_front);
    time_buffer.erase(time_buffer.begin() + keep_front);
    if (lidar_receive_time_buffer.size() > keep_front)
      lidar_receive_time_buffer.erase(lidar_receive_time_buffer.begin() + keep_front);
    ++dropped;
  }
  if (dropped == 0) return;
  dropped_scan_total += dropped;
  static double last_warn_time = 0.0;
  const double now = omp_get_wtime();
  if (now - last_warn_time > 5.0) {
    last_warn_time = now;
    RCLCPP_WARN(rclcpp::get_logger("laser_mapping"),
                "[BACKPRESSURE] scan queue over budget (%d scans / %.2fs): dropped %zu scan(s) (%zu total) — "
                "the estimator is not keeping up with the LiDAR",
                max_buffered_scans, max_scan_backlog_sec, dropped, dropped_scan_total);
  }
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
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
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
    // A stray out-of-order scan (7 per m20 bag) must not clear the queue —
    // that throws away the scan waiting for its IMU coverage (a self-inflicted
    // scan hole). Drop the stray one; only a real jump back clears.
    if (last_timestamp_lidar - cur_time > kLidarLoopBackSec) {
      std::cerr << "lidar loop back (" << (last_timestamp_lidar - cur_time) << "s), clear buffer" << std::endl;
      lidar_buffer.clear();
      lidar_receive_time_buffer.clear();
    } else {
      std::cerr << "lidar scan out of order by " << (last_timestamp_lidar - cur_time) * 1e3 << " ms, dropped"
                << std::endl;
      mtx_buffer.unlock();
      return;
    }
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(cur_time);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer();
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
  scan_count++;
  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    // A stray out-of-order scan (7 per m20 bag) must not clear the queue —
    // that throws away the scan waiting for its IMU coverage (a self-inflicted
    // scan hole). Drop the stray one; only a real jump back clears.
    if (last_timestamp_lidar - cur_time > kLidarLoopBackSec) {
      std::cerr << "lidar loop back (" << (last_timestamp_lidar - cur_time) << "s), clear buffer" << std::endl;
      lidar_buffer.clear();
      lidar_receive_time_buffer.clear();
    } else {
      std::cerr << "lidar scan out of order by " << (last_timestamp_lidar - cur_time) * 1e3 << " ms, dropped"
                << std::endl;
      mtx_buffer.unlock();
      return;
    }
  }
  if (is_first_lidar) {
    is_first_lidar = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(cur_time);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer();
  last_timestamp_lidar = cur_time;
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
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
  scan_count++;
  if (!is_first_lidar && cur_time < last_timestamp_lidar) {
    // A stray out-of-order scan (7 per m20 bag) must not clear the queue —
    // that throws away the scan waiting for its IMU coverage (a self-inflicted
    // scan hole). Drop the stray one; only a real jump back clears.
    if (last_timestamp_lidar - cur_time > kLidarLoopBackSec) {
      std::cerr << "lidar loop back (" << (last_timestamp_lidar - cur_time) << "s), clear buffer" << std::endl;
      lidar_buffer.clear();
      lidar_receive_time_buffer.clear();
    } else {
      std::cerr << "lidar scan out of order by " << (last_timestamp_lidar - cur_time) * 1e3 << " ms, dropped"
                << std::endl;
      mtx_buffer.unlock();
      return;
    }
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
  time_buffer.push_back(last_timestamp_lidar);
  lidar_receive_time_buffer.push_back(receive_time);
  trim_lidar_buffer();

  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void enqueue_imu_msg(sensor_msgs::msg::Imu::SharedPtr msg, double debug_raw_stamp)
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

  if (timestamp < last_timestamp_imu) {
    // A single out-of-order IMU sample (DDS reordering, driver hiccup; 7 per
    // m20 bag) must not wipe the buffer — that removes the coverage of the scan
    // waiting in lidar_buffer and the next propagation runs over a hole
    // (measured 0.9 m jump). Drop the stray sample; only a real jump backwards
    // (bag restart / clock reset, > kImuLoopBackSec) clears the buffer.
    constexpr double kImuLoopBackSec = 1.0;
    if (last_timestamp_imu - timestamp > kImuLoopBackSec) {
      std::cerr << "imu loop back (" << (last_timestamp_imu - timestamp) << "s), clear buffer" << std::endl;
      imu_buffer.clear();
    } else {
      static int stray_count = 0;
      if (++stray_count <= 20) {
        std::cerr << "imu sample out of order by " << (last_timestamp_imu - timestamp) * 1e3 << " ms, dropped"
                  << std::endl;
      }
      mtx_buffer.unlock();
      return;
    }
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
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
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
  while (twist_buffer.size() > 100)
    twist_buffer.pop_front();
  // Stamped newest sample for the static hold: the wheel-odom fusion below
  // consumes the buffer, but the hold needs to know how OLD the evidence is.
  last_twist_stamp = get_time_sec(msg_in->header.stamp);
  last_twist_speed = v.norm();
}

// Independent, front-end-agnostic evidence that the platform is parked. The
// verdict itself lives in static_evidence.hpp (pure, unit-tested); this only
// summarises the scan's IMU window and reads the newest wheel sample.
bool body_is_static(const MeasureGroup & meas, double now)
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
  return fast_lio::bodyIsStatic(imu, wheel_age, wheel_speed, params);
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup & meas)
{
  if (lidar_buffer.empty() || imu_buffer.empty()) {
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
    if (meas.lidar->points.size() <= 1)  // time too little
    {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
      std::cerr << "Too few input point cloud!\n";
    } else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    } else {
      scan_num++;
      lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
    }

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time) {
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
    imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    if (imu_time > lidar_end_time)
      break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  if (!lidar_receive_time_buffer.empty())
    lidar_receive_time_buffer.pop_front();
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
std::deque<std::pair<PointVector, PointVector>> map_insert_pending;
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
  if (map_worker_thread.joinable())
    map_worker_thread.join();
}

// Hand the two add-lists to the worker. Precondition: worker idle (caller joined this
// scan's predecessor). Moves the vectors so the hot path does no point copy.
void dispatchMapAdd(PointVector & toAdd, PointVector & noDS)
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
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points[i].empty() && flg_EKF_inited) {
      const PointVector & points_near = Nearest_Points[i];
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

  if (map_insert_delay_scans > 0) {
    // Hold this scan's (world-frame) points; insert the scan that is now
    // `delay` scans old. The add/no-add decisions above were made against the
    // map as it was when the scan was matched — good enough for a lag of a few
    // scans, and identical every run.
    map_insert_pending.emplace_back(std::move(PointToAdd), std::move(PointNoNeedDownsample));
    if (map_insert_pending.size() <= static_cast<size_t>(map_insert_delay_scans)) {
      add_point_size = 0;
      return;
    }
    PointToAdd = std::move(map_insert_pending.front().first);
    PointNoNeedDownsample = std::move(map_insert_pending.front().second);
    map_insert_pending.pop_front();
  }

  if (async_map_en) {
    // Off-thread: the worker calls Add_Points during the inter-scan gap; joined before
    // the next scan's first tree op. add_point_size is known here (sizes); the Add wall
    // time lands in g_async_add_us (see mapWorkerLoop). kdtree_incremental_time is left
    // untouched in this path — it is a main-thread debug stat and the worker must not
    // write plain globals the main thread reads.
    add_point_size = (int)(PointToAdd.size() + PointNoNeedDownsample.size());
    dispatchMapAdd(PointToAdd, PointNoNeedDownsample);
  } else {
    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
  }
}

// P2 VoxelMap-lite: fold the converged scan into the per-voxel Welford moments
// (see the vxl_en block comment). Called from the main thread once per scan,
// right after map_incremental — the same "converged pose, map grew" event, so
// the statistics stay consistent with what the matcher will see. Transforms
// into a local point instead of trusting feats_down_world (under async map the
// world cloud may be rewritten off-thread). Cells past vxl_freeze_pts stop
// updating; delta arithmetic in double, storage in float.
void vxl_update()
{
  if (!vxl_en)
    return;
  if (vxl_grid.empty())
    vxl_grid.reserve(1 << 20);
  for (int i = 0; i < feats_down_size; i++) {
    const PointType & pb = feats_down_body->points[i];
    if (std::hypot(pb.x, pb.y) > vxl_max_range)
      continue;
    PointType pw;
    pointBodyToWorld(&feats_down_body->points[i], &pw);
    VxlCell & c = vxl_grid[vxlKey(pw.x, pw.y, pw.z)];
    if (c.n >= vxl_freeze_pts)
      continue;
    if (c.n == 0) {
      c.mean[0] = pw.x;
      c.mean[1] = pw.y;
      c.mean[2] = pw.z;
      c.zq = pw.z;
      c.n = 1;
      continue;
    }
    ++c.n;
    // SGD quantile step toward the 10th percentile: down-moves 9x faster
    // than up-moves, so zq settles on the cell's lower mode.
    constexpr float kVxlZqEta = 0.02f;
    c.zq += kVxlZqEta * (0.1f - (pw.z < c.zq ? 1.0f : 0.0f));
    const double d0 = pw.x - c.mean[0], d1 = pw.y - c.mean[1], d2 = pw.z - c.mean[2];
    const double inv_n = 1.0 / static_cast<double>(c.n);
    c.mean[0] += static_cast<float>(d0 * inv_n);
    c.mean[1] += static_cast<float>(d1 * inv_n);
    c.mean[2] += static_cast<float>(d2 * inv_n);
    const double e0 = pw.x - c.mean[0], e1 = pw.y - c.mean[1], e2 = pw.z - c.mean[2];
    c.m2[0] += static_cast<float>(d0 * e0);
    c.m2[1] += static_cast<float>(d0 * e1);
    c.m2[2] += static_cast<float>(d0 * e2);
    c.m2[3] += static_cast<float>(d1 * e1);
    c.m2[4] += static_cast<float>(d1 * e2);
    c.m2[5] += static_cast<float>(d2 * e2);
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
  if (!pcd_writer_running.exchange(false))
    return;
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
    laserCloudmsg.header.frame_id = "camera_init";
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
  laserCloudmsg.header.frame_id = "body";
  pubLaserCloudFull_body->publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
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

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped,
                      std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
  set_posestamp(odomAftMapped.pose);
  // Front-end health side-channel (Plan C) — written BEFORE the publish so it is
  // same-scan (unlike pose.covariance below, which fills the persistent global
  // message after publishing and therefore rides out one scan late). The twist
  // covariance was always zero here and has no other consumer. Layout contract:
  // LIO-SLAM core pgo_factor_kernels.hpp kFeHealth*Cell.
  if (health_pub_en) {
    odomAftMapped.twist.covariance[0] = 1.0;  // sentinel/version
    // Flags say what the front end did about this scan, which is what a
    // consumer must act on: a degraded pose is fresh and self-consistent, so
    // staleness gates downstream pass it through unchanged (dog 2, 2026-09-04).
    unsigned health_flags = 0u;
    if (flio_degraded_odom)
      health_flags |= 1u << 0;  // kFeHealthFlagDegraded
    if (zupt_active)
      health_flags |= 1u << 1;  // kFeHealthFlagStaticHold
    if (reanchor_gate.state == fast_lio::ReanchorState::kLost)
      health_flags |= 1u << 2;  // kFeHealthFlagLost
    odomAftMapped.twist.covariance[1] = static_cast<double>(health_flags);
    odomAftMapped.twist.covariance[7] = scan_obs_along;
    odomAftMapped.twist.covariance[14] = static_cast<double>(effct_feat_num);
    odomAftMapped.twist.covariance[21] = scan_far_frac;
    odomAftMapped.twist.covariance[28] = res_mean_last;
    odomAftMapped.twist.covariance[35] = scan_body_speed;
  }
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
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  // Divergence guard (P1): while the front end is dead-reckoning (LiDAR
  // starved), advertise large position variance on the ROS pose covariance
  // (x,y,z diagonal) so the PGO back-end down-weights these poses instead of
  // trusting a bounded-but-uncertain dead-reckoned estimate.
  if (flio_degraded_odom) {
    constexpr double kDegradedPosVar = 100.0;              // (10 m)^2
    odomAftMapped.pose.covariance[0] += kDegradedPosVar;   // x
    odomAftMapped.pose.covariance[7] += kDegradedPosVar;   // y
    odomAftMapped.pose.covariance[14] += kDegradedPosVar;  // z
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
  msg_body_pose.header.stamp = get_ros_time(lidar_end_time);  // ros::Time().fromSec(lidar_end_time);
  msg_body_pose.header.frame_id = "camera_init";

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
  }
}

void h_share_model(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
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
  for (int i = 0; i < feats_down_size; i++) {
    PointType & point_body = feats_down_body->points[i];
    PointType & point_world = feats_down_world->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto & points_near = Nearest_Points[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS        ? false
                               : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                            : true;
    }

    if (!point_selected_surf[i])
      continue;

    VF(4) pabcd;
    point_selected_surf[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_surf[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;
        res_last[i] = abs(pd2);
      }
    }
  }

  // Canopy/clutter TELEMETRY (does not gate any behavior): near-field effective
  // ground fraction + height p95, both in the LiDAR/cloud BODY frame
  // (drift-immune). MUST run BEFORE the ground recovery pass below so the
  // statistic reflects the unmodified selection, not the recovery's own output.
  // The debounced mode switch in the main loop is telemetry/diagnostics too.
  // TEM consumes canopy_ground_frac as its anchor-quality signal (egf), so the
  // statistic is also computed when tem_en.
  if (canopy_detect_en || tem_en) {
    static std::vector<float> near_z;  // h_share_model is single-threaded here
    near_z.clear();
    int ground_cnt = 0;
    for (int i = 0; i < feats_down_size; i++) {
      if (!point_selected_surf[i])
        continue;
      const PointType & pb = feats_down_body->points[i];
      if (std::hypot(pb.x, pb.y) > canopy_near_range)
        continue;
      near_z.push_back(pb.z);
      if (std::fabs(normvec->points[i].z) >= canopy_normal_z_min &&
          std::fabs(pb.z - canopy_ground_z) <= canopy_ground_halfband)
        ++ground_cnt;
    }
    canopy_near_count = static_cast<int>(near_z.size());
    if (canopy_near_count > 0) {
      canopy_ground_frac = static_cast<double>(ground_cnt) / canopy_near_count;
      const size_t k = static_cast<size_t>(0.95 * (near_z.size() - 1));
      std::nth_element(near_z.begin(), near_z.begin() + k, near_z.end());
      canopy_zp95 = near_z[k];
    }
  }

  // P2 VoxelMap-lite anisotropic deweight (see the flag's block comment).
  // Runs on the ACCEPTED rows BEFORE ground recovery, so recovery rows (added
  // below with fabricated vertical normals — the load-bearing counterweight)
  // are never rescaled. esti_plane rewrites the organic normals every iEKF
  // iteration, so the scaling is applied fresh each call — no compounding.
  // The grid is only written from the main thread between scans; this pass
  // is read-only and safe under OMP.
  vxl_tested = vxl_cut = vxl_vert_cnt = vxl_env_cnt = 0;
  vxl_msum = vxl_sd_sum = vxl_vert_msum = vxl_env_habs = 0.0;
  if (vxl_en) {
    // P3 envelope cache, per scan per row (the column's mature-cell set only
    // moves with map growth, not with iEKF iterations; the residual itself
    // is recomputed each iteration from the fresh world point).
    static double env_time = -1.0;
    static int8_t env_state[100000];  // 0 = not computed, 1 = served, 2 = none
    static float env_z[100000], env_sig[100000];
    if (vxl_env_en && lidar_end_time != env_time) {
      env_time = lidar_end_time;
      std::fill_n(env_state, feats_down_size, static_cast<int8_t>(0));
    }
    int tested = 0, cut = 0, vert_cnt = 0, env_cnt = 0;
    double msum = 0.0, sdsum = 0.0, vert_msum = 0.0, env_habs = 0.0;
#ifdef MP_EN
#pragma omp parallel for reduction(+ : tested, cut, vert_cnt, msum, sdsum, vert_msum, env_cnt, env_habs)
#endif
    for (int i = 0; i < feats_down_size; i++) {
      if (!point_selected_surf[i])
        continue;
      const PointType & pb = feats_down_body->points[i];
      if (std::hypot(pb.x, pb.y) > vxl_max_range)
        continue;
      const PointType & pw = feats_down_world->points[i];
      const double nx = normvec->points[i].x, ny = normvec->points[i].y,
                   nz = normvec->points[i].z;  // unit normal from esti_plane
      // P3 residual-reference swap for ground-class rows (see the
      // vxl_env_en block comment). Evidence is per point: its height
      // against the column's lower envelope, not the (possibly bent)
      // fitted normal — but strong lateral carriers are never swapped.
      if (vxl_env_en && std::fabs(nz) >= vxl_env_nz_min) {
        if (env_state[i] == 0) {
          // Lowest mature cell in this column, from one slab above the
          // point down to 2 m below (kVxlEnvScanDown slabs).
          constexpr int kVxlEnvScanDown = 8;
          const int64_t kx = static_cast<int64_t>(std::floor(pw.x / vxl_size_xy));
          const int64_t ky = static_cast<int64_t>(std::floor(pw.y / vxl_size_xy));
          const int64_t kz0 = static_cast<int64_t>(std::floor(pw.z / vxl_size_z));
          env_state[i] = 2;
          for (int64_t kz = kz0 - kVxlEnvScanDown; kz <= kz0 + 1; kz++) {
            const auto ite = vxl_grid.find(vxlKeyIdx(kx, ky, kz));
            if (ite == vxl_grid.end() || ite->second.n < vxl_env_min_pts)
              continue;
            env_state[i] = 1;
            env_z[i] = ite->second.zq;
            env_sig[i] = std::max(std::sqrt(std::max(ite->second.m2[5] / ite->second.n, 0.0f)), 0.05f);
            break;  // lowest first: kz ascends from the bottom
          }
        }
        if (env_state[i] == 1 && std::fabs(pw.z - env_z[i]) <= vxl_env_band) {
          // Same row shape as a recovery row: forced vertical normal,
          // envelope residual, trust-region clamp (never hard-gated),
          // honest weight from the envelope cell's own thickness.
          const double sig = env_sig[i];
          const double pd2 = std::clamp(static_cast<double>(pw.z - env_z[i]), -3.0 * sig, 3.0 * sig);
          const double w = vxl_sigma_r / std::sqrt(vxl_sigma_r * vxl_sigma_r + sig * sig);
          normvec->points[i].x = 0.0f;
          normvec->points[i].y = 0.0f;
          normvec->points[i].z = static_cast<float>(w);
          normvec->points[i].intensity = static_cast<float>(w * pd2);
          res_last[i] = std::fabs(pd2);
          ++env_cnt;
          env_habs += std::fabs(pd2);
          continue;  // re-referenced: the P2 plane deweight no longer applies
        }
      }
      const auto it = vxl_grid.find(vxlKey(pw.x, pw.y, pw.z));
      if (it == vxl_grid.end() || it->second.n < vxl_min_pts)
        continue;
      const VxlCell & c = it->second;
      const double inv_n = 1.0 / static_cast<double>(c.n);
      // sigma_dir^2 = n^T Cov n from the upper-triangle comoment sums
      const double sd2 = std::max(0.0,
                                  (nx * nx * c.m2[0] + ny * ny * c.m2[3] + nz * nz * c.m2[5] +
                                   2.0 * (nx * ny * c.m2[1] + nx * nz * c.m2[2] + ny * nz * c.m2[4])) *
                                    inv_n);
      const double m = std::clamp(vxl_sigma_r / std::sqrt(vxl_sigma_r * vxl_sigma_r + sd2), vxl_min_weight, 1.0);
      ++tested;
      msum += m;
      sdsum += std::sqrt(sd2);
      if (std::fabs(nz) > 0.8) {
        ++vert_cnt;
        vert_msum += m;
      }
      if (m < 0.999) {
        ++cut;
        normvec->points[i].x *= static_cast<float>(m);
        normvec->points[i].y *= static_cast<float>(m);
        normvec->points[i].z *= static_cast<float>(m);
        normvec->points[i].intensity *= static_cast<float>(m);
      }
    }
    vxl_tested = tested;
    vxl_cut = cut;
    vxl_msum = msum;
    vxl_sd_sum = sdsum;
    vxl_vert_cnt = vert_cnt;
    vxl_vert_msum = vert_msum;
    vxl_env_cnt = env_cnt;
    vxl_env_habs = env_habs;
  }

  /*** Ground recovery (selection-layer repair, 2026-07-06 forensics): ANY
   *   ground-hugging clutter (grass, crops, debris — content-agnostic) pollutes
   *   the 5-NN esti_plane + s>0.9 gates: true-ground returns get REJECTED while
   *   clutter micro-patches get accepted, depleting the vertical-normal share of
   *   the effective set even though the raw scan-to-scan Z valley stays sharp
   *   (measured: raw ground 62-70 % → ~10 % effective on the giant 0702 bag).
   *   Repair at the same layer, SELF-GATED (no scene classifier): (1) fit robust
   *   ground planes HYBRID global+sector in the body frame (shrinking-band LS —
   *   anything above the dominant surface is trimmed regardless of what it is).
   *   The whole-scan GLOBAL fit pools every candidate: maximum support, hardest
   *   to bias (in deep clutter a per-sector fit can latch onto a locally
   *   consistent clutter shelf that the pooled fit rejects — measured: sector-
   *   only was ~2x weaker than whole-scan in the 0702 canopy). Per-SECTOR fits
   *   refine it for crowned/rutted/partially-occupied terrain, but a sector
   *   plane is used only where it AGREES with the global plane (normal dot +
   *   height at the sector centroid); a sector that fails its own gates or
   *   disagrees falls back to the global plane, and without a valid global fit
   *   the surviving sector fits stand alone. The height-sanity gate (plane at
   *   the known sensor mount height) guarantees the dominant surface IS what
   *   the vehicle stands on, without scene semantics.
   *   (2) re-admit rejected near-field returns lying on their sector's plane,
   *   with an imposed vertical world normal and the residual measured against
   *   the LOWER ENVELOPE of their map neighbors (mean of the two lowest z —
   *   robust to clutter fuzz in the map). Rows flow into the standard
   *   point-to-plane H. Only points the normal path rejected are touched.
   *   NOTE: distinct from the removed A'-3 boost, which amplified already-
   *   matched rows (NEGATIVE result); this restores SELECTION of discarded
   *   true ground. ***/
  canopy_ground_recovered = 0;
  tem_rows = tem_ext_rows = tem_env_rows = 0;
  if (ground_recover_en) {
    // Per-sector ground planes, cached per scan (body cloud is fixed across
    // iEKF iterations). n·p + d = 0 with unit n, n_z > 0.
    constexpr int kGndSectors = 8;
    static double fit_time = -1.0;
    static bool sec_ok[kGndSectors];
    static Eigen::Vector3d sec_n[kGndSectors];
    static double sec_d[kGndSectors];
    if (lidar_end_time != fit_time) {
      fit_time = lidar_end_time;
      static std::vector<Eigen::Vector3d> cand[kGndSectors];
      for (int s = 0; s < kGndSectors; s++) {
        cand[s].clear();
        sec_ok[s] = false;
      }
      for (int i = 0; i < feats_down_size; i++) {
        const PointType & pb = feats_down_body->points[i];
        if (std::hypot(pb.x, pb.y) > canopy_near_range)
          continue;
        if (std::fabs(pb.z - canopy_ground_z) > canopy_ground_halfband)
          continue;
        const int s =
          std::min(kGndSectors - 1, static_cast<int>((std::atan2(pb.y, pb.x) + M_PI) * kGndSectors / (2.0 * M_PI)));
        cand[s].emplace_back(pb.x, pb.y, pb.z);
      }
      // Shrinking-band iterative LS over one or more candidate sets.
      // Gates: enough support, near-horizontal (terrain + vehicle tilt stay
      // within a few degrees), sane height under the sensor. The height gate
      // is what makes always-on safe: the accepted surface must sit where
      // the vehicle's own ground sits.
      auto fit_ground = [](const std::vector<Eigen::Vector3d> * sets,
                           int nsets,
                           int min_in,
                           Eigen::Vector3d & n_out,
                           double & d_out) -> bool {
        static std::vector<float> zs;
        zs.clear();
        for (int k = 0; k < nsets; k++)
          for (const auto & p : sets[k])
            zs.push_back(static_cast<float>(p.z()));
        if (static_cast<int>(zs.size()) < min_in)
          return false;
        // Init: horizontal plane at the pooled median candidate height.
        std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
        Eigen::Vector3d n(0, 0, 1);
        double d = -static_cast<double>(zs[zs.size() / 2]);
        int n_in = 0;
        const double bands[3] = {0.4, 0.25, 0.15};
        for (double band : bands) {
          Eigen::Vector3d mean = Eigen::Vector3d::Zero();
          Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
          n_in = 0;
          for (int k = 0; k < nsets; k++)
            for (const auto & p : sets[k])
              if (std::fabs(n.dot(p) + d) <= band) {
                mean += p;
                ++n_in;
              }
          if (n_in < min_in)
            break;
          mean /= n_in;
          for (int k = 0; k < nsets; k++)
            for (const auto & p : sets[k])
              if (std::fabs(n.dot(p) + d) <= band) {
                const Eigen::Vector3d q = p - mean;
                cov.noalias() += q * q.transpose();
              }
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
          n = es.eigenvectors().col(0);
          if (n.z() < 0)
            n = -n;
          d = -n.dot(mean);
        }
        n_out = n;
        d_out = d;
        return n_in >= min_in && n.z() >= 0.95 &&
               std::fabs(-d / std::max(n.z(), 1e-6) - canopy_ground_z) <= canopy_ground_halfband;
      };
      // Global whole-scan fit: pooled support is the hardest to bias — the
      // prior every sector must agree with, and the fallback plane.
      Eigen::Vector3d gn(0, 0, 1);
      double gd = 0.0;
      const bool glob_ok = fit_ground(cand, kGndSectors, ground_recover_min_inliers, gn, gd);
      for (int s = 0; s < kGndSectors; s++) {
        Eigen::Vector3d n;
        double d;
        bool own_ok = fit_ground(&cand[s], 1, ground_recover_min_inliers, n, d);
        if (own_ok && glob_ok) {
          // A sector plane may only REFINE the global one: normals must
          // agree and the plane heights at the sector centroid must match
          // within ground_recover_sector_tol; otherwise the sector fit
          // latched onto something the pooled fit rejected.
          Eigen::Vector3d c = Eigen::Vector3d::Zero();
          for (const auto & p : cand[s])
            c += p;
          c /= static_cast<double>(cand[s].size());
          const double z_sec = -(d + n.x() * c.x() + n.y() * c.y()) / std::max(n.z(), 1e-6);
          const double z_glob = -(gd + gn.x() * c.x() + gn.y() * c.y()) / std::max(gn.z(), 1e-6);
          constexpr double kSectorNormalDotMin = 0.985;  // ~10 deg relative tilt
          own_ok = n.dot(gn) >= kSectorNormalDotMin && std::fabs(z_sec - z_glob) <= ground_recover_sector_tol;
        }
        if (own_ok) {
          sec_ok[s] = true;
          sec_n[s] = n;
          sec_d[s] = d;
        } else if (glob_ok) {
          sec_ok[s] = true;
          sec_n[s] = gn;
          sec_d[s] = gd;
        }
      }
      // TEM observation samples: this scan's fit-band inliers (raw geometry
      // only — recovered rows are downstream of TEM and must never feed it).
      // Consumed once per scan in the main loop with the CONVERGED pose.
      if (tem_en) {
        tem_samples.clear();
        size_t total = 0;
        for (int s = 0; s < kGndSectors; s++)
          if (sec_ok[s])
            total += cand[s].size();
        const size_t stride = std::max<size_t>(1, total / 2000);
        size_t idx = 0;
        for (int s = 0; s < kGndSectors; s++) {
          if (!sec_ok[s])
            continue;
          for (const auto & p : cand[s]) {
            if (std::fabs(sec_n[s].dot(p) + sec_d[s]) > ground_recover_scan_band)
              continue;
            if (idx++ % stride == 0)
              tem_samples.push_back(p);
          }
        }
      }
    }
    // Clutter deweight pass (see the flag's block comment). Runs on the
    // ACCEPTED rows before recovery; the expensive k=15 thickness is cached
    // per scan (band membership is body-frame = iteration-invariant, the
    // organic normals are rewritten by esti_plane every iteration so the
    // scaling below is applied fresh each call — no compounding).
    if (clutter_deweight_en) {
      // Band lower edge 0.5 m is a HARD boundary: the 0.15-0.5 m grass
      // layer just above the fitted plane carries Z bias BUT is also the
      // bulk of the effective Z constraint in deep clutter (half of it is
      // true ground in mixed grass) — lowering the edge to 0.25 m
      // re-opened the raw pathology worse than baseline (FE 2D 796 m,
      // Z -105, /odom breached 8.8 m). Do not lower it again.
      constexpr double kClutBandLo = 0.5, kClutBandHi = 3.0;  // m above the sector plane
      constexpr int kClutK = 15;                              // measured: 5 is blind, 15 separates 12x
      constexpr int kClutMinNeighbors = 12;                   // sparse map neighborhood = no verdict
      constexpr double kClutNzLo = 0.3, kClutNzHi = 0.8;      // |n_z| taper (lateral carriers spared)
      static double clut_time = -1.0;
      static float clut_thick[100000];
      if (lidar_end_time != clut_time) {
        clut_time = lidar_end_time;
        std::fill_n(clut_thick, feats_down_size, -1.0f);
      }
      int tested = 0, cut = 0;
      double msum = 0.0;
#ifdef MP_EN
#pragma omp parallel for reduction(+ : tested, cut, msum)
#endif
      for (int i = 0; i < feats_down_size; i++) {
        if (!point_selected_surf[i])
          continue;
        const PointType & pb = feats_down_body->points[i];
        if (std::hypot(pb.x, pb.y) > canopy_near_range)
          continue;
        const int s =
          std::min(kGndSectors - 1, static_cast<int>((std::atan2(pb.y, pb.x) + M_PI) * kGndSectors / (2.0 * M_PI)));
        if (!sec_ok[s])
          continue;
        const double h = sec_n[s].dot(Eigen::Vector3d(pb.x, pb.y, pb.z)) + sec_d[s];
        if (h < kClutBandLo || h > kClutBandHi)
          continue;
        ++tested;
        if (clut_thick[i] < 0.0f) {
          PointVector nb;
          std::vector<float> sq;
          ikdtree.Nearest_Search(feats_down_world->points[i], kClutK, nb, sq);
          if (static_cast<int>(nb.size()) < kClutMinNeighbors) {
            clut_thick[i] = 0.0f;  // no verdict — leave the row alone
          } else {
            Eigen::Vector3d mean = Eigen::Vector3d::Zero();
            for (const auto & p : nb)
              mean += Eigen::Vector3d(p.x, p.y, p.z);
            mean /= static_cast<double>(nb.size());
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (const auto & p : nb) {
              const Eigen::Vector3d q = Eigen::Vector3d(p.x, p.y, p.z) - mean;
              cov.noalias() += q * q.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov / static_cast<double>(nb.size()));
            clut_thick[i] = static_cast<float>(std::sqrt(std::max(es.eigenvalues()(0), 0.0)));
          }
        }
        // f: planar at map scale -> 1, fuzzy -> 0 (linear ramp).
        const double f =
          std::clamp((clutter_th_hi - clut_thick[i]) / std::max(clutter_th_hi - clutter_th_lo, 1e-6), 0.0, 1.0);
        // g: only vertical-normal rows (Z-bias carriers) take the cut.
        const double nz = std::fabs(normvec->points[i].z);
        const double g = std::clamp((nz - kClutNzLo) / (kClutNzHi - kClutNzLo), 0.0, 1.0);
        const double m = std::max(1.0 - (1.0 - f) * g, clutter_min_weight);
        msum += m;
        if (m < 0.999) {
          ++cut;
          normvec->points[i].x *= static_cast<float>(m);
          normvec->points[i].y *= static_cast<float>(m);
          normvec->points[i].z *= static_cast<float>(m);
          normvec->points[i].intensity *= static_cast<float>(m);
        }
      }
      clut_tested = tested;
      clut_cut = cut;
      clut_msum = msum;
    }
    for (int i = 0; i < feats_down_size; i++) {
      if (point_selected_surf[i])
        continue;
      if (canopy_ground_recovered >= ground_recover_max_rows)
        break;
      const PointType & pb = feats_down_body->points[i];
      if (std::hypot(pb.x, pb.y) > canopy_near_range)
        continue;
      const int s =
        std::min(kGndSectors - 1, static_cast<int>((std::atan2(pb.y, pb.x) + M_PI) * kGndSectors / (2.0 * M_PI)));
      if (!sec_ok[s])
        continue;
      // Scan side: on this sector's ground plane (clutter trimmed by the fit band).
      if (std::fabs(sec_n[s].dot(Eigen::Vector3d(pb.x, pb.y, pb.z)) + sec_d[s]) > ground_recover_scan_band)
        continue;
      auto & pn = Nearest_Points[i];
      if (static_cast<int>(pn.size()) < NUM_MATCH_POINTS)
        continue;
      const PointType & pw = feats_down_world->points[i];
      // Map neighborhood must actually be nearby (same criterion class as
      // the normal path's 5th-neighbor sqdist gate).
      const double dx0 = pn[0].x - pw.x, dy0 = pn[0].y - pw.y, dz0 = pn[0].z - pw.z;
      if (dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > 5.0)
        continue;
      // Residual reference, best available first: TEM terrain memory
      // (drift-independent — learned while healthy, frozen while not),
      // weighted down by the interpolated cell std so slopes / disagreeing
      // observations pull weakly; else the map-neighbor lower envelope
      // (mean of the two lowest z — robust to clutter fuzz, but it DRIFTS
      // with the vehicle: ratchet-slowing only).
      // Reference split by JOB, not by availability. 3/4 of the rows keep
      // the map lower envelope at weight 1 — local Z observability
      // restoration, the validated hybrid behavior, untouched. Every 4th
      // row is the ABSOLUTE channel: residual against the TEM (direct
      // where served, slope-envelope extrapolation otherwise) with a
      // trust-region clamp and a sigma-collapsed weight. The absolute
      // channel is NEVER gated by max_residual — a multi-metre secular
      // drift is exactly what it exists to fight (measured failure: hard-
      // gating TEM rows at 1 m gutted the whole recovery channel once the
      // drift passed 1 m, and the raw pathology returned worse, -64.6 m).
      constexpr double kTemClampSigma = 3.0;
      constexpr double kTemStdFloor = 0.1;  // weight floor: never trust a cell below 10 cm
      // AUTHORITY IS PINNED AT THE SAFE-NEUTRAL LEVEL (~0.07 m/s counter-
      // drift) — the full dose-response of pushing harder is measured and
      // closed. (a) A 2.5 m clamp floor + sigma0 0.5 reaches parity with
      // the worst dive (~0.5 m/s) and does cut the Z ratchet (-55 -> -43),
      // but the point-plane Jacobian's attitude columns turn the pull into
      // torque and the error reroutes into XY in attitude-degenerate
      // clutter (FE 2D 150 -> 327 m, alignment bent 8 deg). (b) Zeroing
      // the attitude/extrinsic columns (translation-only rows) makes the
      // filter's measurement model inconsistent and Z diverges
      // exponentially (measured -12 km; healthy scenes also damaged).
      // The B3 lesson holds at every level tried: while degraded clutter
      // still pollutes the constraint set, absolute-Z injection strong
      // enough to matter always finds another channel to blow. Do NOT
      // raise the authority again — fix the constraint set instead.
      double pd2, w_row = 1.0;
      int ref_kind = 2;  // 0 = TEM direct, 1 = TEM extrapolated, 2 = envelope
      double z_tem, std_tem;
      const bool abs_slot = tem_en && (canopy_ground_recovered % 4) == 3;
      if (abs_slot && temLookup(pw.x, pw.y, z_tem, std_tem)) {
        const double sig = std::max(std_tem, kTemStdFloor);
        const double lim = kTemClampSigma * sig;
        pd2 = std::clamp(pw.z - z_tem, -lim, lim);
        w_row = tem_sigma0 / (tem_sigma0 + sig);
        ref_kind = 0;
      } else if (abs_slot && tem_ext_ok) {
        const double sig = std::max(tem_ext_sigma, kTemStdFloor);
        const double lim = kTemClampSigma * sig;
        pd2 = std::clamp(pw.z - tem_ext_z, -lim, lim);
        w_row = tem_sigma0 / (tem_sigma0 + sig);
        ref_kind = 1;
      } else {
        float z1 = pn[0].z, z2 = std::numeric_limits<float>::max();
        for (size_t k = 1; k < pn.size(); k++) {
          const float z = pn[k].z;
          if (z < z1) {
            z2 = z1;
            z1 = z;
          } else if (z < z2) {
            z2 = z;
          }
        }
        const double z_ground_map = 0.5 * (z1 + (z2 == std::numeric_limits<float>::max() ? z1 : z2));
        pd2 = pw.z - z_ground_map;
        if (std::fabs(pd2) > ground_recover_max_residual)
          continue;
      }
      point_selected_surf[i] = true;
      normvec->points[i].x = 0.0f;
      normvec->points[i].y = 0.0f;
      // Row weight rides on the normal + residual scaling (information
      // scales as w_row^2 in the point-to-plane H) — same mechanism as an
      // ordinary row, no new plumbing.
      normvec->points[i].z = static_cast<float>(w_row);
      normvec->points[i].intensity = static_cast<float>(w_row * pd2);
      res_last[i] = std::fabs(pd2);
      ++(ref_kind == 0 ? tem_rows : ref_kind == 1 ? tem_ext_rows : tem_env_rows);
      ++canopy_ground_recovered;
    }
  }

  effct_feat_num = 0;
  if (degeneracy_debug) {
    std::fill(std::begin(res_bin_n), std::end(res_bin_n), 0);
    std::fill(std::begin(res_bin_ss), std::end(res_bin_ss), 0.0);
  }

  for (int i = 0; i < feats_down_size; i++) {
    if (point_selected_surf[i]) {
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      total_residual += res_last[i];
      if (degeneracy_debug) {
        const PointType & pb = feats_down_body->points[i];
        const double rng = std::sqrt(double(pb.x) * pb.x + double(pb.y) * pb.y + double(pb.z) * pb.z);
        int b = 0;
        while (b < kResBins - 1 && rng >= kResBinEdges[b]) ++b;
        ++res_bin_n[b];
        res_bin_ss[b] += double(res_last[i]) * res_last[i];
      }
      effct_feat_num++;
    }
  }

  // Position-observability metric (degeneracy detector): eigenvalues of the
  // normal information matrix M = sum(n n^T) over effective points. The min
  // eigenvalue is the weakest-constrained position direction; if its
  // eigenvector points along Z, the Z update is ill-conditioned.
  {
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (int i = 0; i < effct_feat_num; i++) {
      const PointType & np = corr_normvect->points[i];
      Eigen::Vector3d n(np.x, np.y, np.z);
      M.noalias() += n * n.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
    pos_obs_M = M;
    pos_obs_eig_min = es.eigenvalues()(0);
    pos_obs_eig_max = es.eigenvalues()(2);
    pos_obs_z_weak = std::abs(es.eigenvectors()(2, 0));  // |Z| of weakest eigenvector
  }

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    std::cerr << "No Effective Points!" << std::endl;
    return;
  }

  res_mean_last = total_residual / effct_feat_num;
  match_time += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23
  ekfom_data.h.resize(effct_feat_num);

  for (int i = 0; i < effct_feat_num; i++) {
    const PointType & laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType & norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B),
        VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }

  solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
  LaserMappingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
  {
    this->declare_parameter<bool>("publish.path_en", true);
    this->declare_parameter<bool>("publish.effect_map_en", false);
    this->declare_parameter<bool>("publish.map_en", false);
    this->declare_parameter<double>("publish.map_period_sec", 5.0);  // /Laser_map publish period (s); 5s = 0.2Hz
    this->declare_parameter<bool>("publish.scan_publish_en", true);
    this->declare_parameter<bool>("publish.dense_publish_en", true);
    this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
    this->declare_parameter<int>("max_iteration", 4);
    this->declare_parameter<int>("max_buffered_scans", 100);
    this->declare_parameter<double>("max_scan_backlog_sec", 0.5);
    this->declare_parameter<bool>("adaptive_downsample_en", false);
    this->declare_parameter<double>("adaptive_downsample_budget_sec", 0.06);
    this->declare_parameter<double>("adaptive_downsample_release_sec", 0.035);
    this->declare_parameter<double>("adaptive_downsample_step", 0.05);
    this->declare_parameter<double>("adaptive_downsample_max_voxel", 0.5);
    this->declare_parameter<int>("adaptive_downsample_raise_after", 5);
    this->declare_parameter<int>("adaptive_downsample_lower_after", 100);
    this->declare_parameter<bool>("degeneracy_debug", false);
    this->declare_parameter<bool>("health_pub_en", true);
    this->declare_parameter<bool>("planar_constraint_en", false);
    this->declare_parameter<double>("planar_constraint_noise", 0.1);
    this->declare_parameter<double>("planar_constraint_z_weak_thresh", 0.5);
    this->declare_parameter<bool>("gravity_align_en", false);
    this->declare_parameter<double>("gravity_align_noise", 0.05);
    this->declare_parameter<double>("gravity_align_z_weak_thresh", 0.3);
    this->declare_parameter<double>("gravity_align_accel_tol", 0.05);
    this->declare_parameter<double>("gravity_align_gyro_tol", 0.35);
    this->declare_parameter<double>("gravity_align_window_s", 0.0);
    this->declare_parameter<bool>("gravity_align_continuous", false);
    this->declare_parameter<double>("gravity_align_hold_s", 0.0);
    this->declare_parameter<double>("gravity_align_noise_degraded", 0.0);
    this->declare_parameter<double>("attitude_cov_floor_deg", 0.0);
    this->declare_parameter<double>("gravity_align_mag_ref", 0.0);
    this->declare_parameter<double>("gravity_align_grav_leak", 1.0);
    this->declare_parameter<double>("gravity_align_grav_cap_deg", 3.0);
    this->declare_parameter<bool>("imu_init_require_still", false);
    this->declare_parameter<double>("imu_init_still_tol", 0.03);
    this->declare_parameter<double>("imu_init_still_timeout_s", 20.0);
    this->declare_parameter<bool>("divergence_guard_en", true);
    this->declare_parameter<int>("divergence_guard_min_eff", 50);
    this->declare_parameter<bool>("guard_engulf_en", true);
    this->declare_parameter<double>("guard_engulf_far_range", 2.0);
    this->declare_parameter<double>("guard_engulf_far_frac_min", 0.15);
    this->declare_parameter<int>("guard_engulf_streak", 2);
    this->declare_parameter<double>("guard_engulf_far_frac_release", 0.35);
    this->declare_parameter<int>("guard_engulf_release_scans", 5);
    this->declare_parameter<bool>("guard_hole_en", false);
    this->declare_parameter<double>("guard_hole_min_sec", 0.3);
    this->declare_parameter<double>("guard_hole_max_accel", 1.0);
    this->declare_parameter<bool>("guard_hole_reanchor_pos", true);
    this->declare_parameter<int>("divergence_guard_streak", 5);
    this->declare_parameter<double>("divergence_guard_max_speed", 30.0);
    this->declare_parameter<bool>("zupt_en", true);
    this->declare_parameter<double>("zupt_gyro_thresh", 0.05);
    this->declare_parameter<double>("zupt_acc_span_ratio", 0.02);
    this->declare_parameter<double>("zupt_wheel_speed_thresh", 0.05);
    this->declare_parameter<double>("zupt_wheel_max_age", 0.5);
    this->declare_parameter<bool>("zupt_require_wheel", false);
    this->declare_parameter<bool>("reanchor_en", true);
    this->declare_parameter<int>("reanchor_min_eff", 200);
    this->declare_parameter<double>("reanchor_max_res", 0.10);
    this->declare_parameter<int>("reanchor_confirm_scans", 3);
    this->declare_parameter<double>("reanchor_timeout_sec", 5.0);
    this->declare_parameter<std::string>("reanchor_on_fail", "hold");
    this->declare_parameter<bool>("runaway_watchdog_en", true);
    this->declare_parameter<double>("runaway_parked_speed_thresh", 0.3);
    this->declare_parameter<double>("runaway_wheel_speed_margin", 1.0);
    this->declare_parameter<double>("runaway_hold_sec", 3.0);
    this->declare_parameter<std::string>("runaway_on_trip", "hold");
    this->declare_parameter<bool>("canopy_detect_en", false);
    this->declare_parameter<bool>("canopy_debug", false);
    this->declare_parameter<double>("canopy_near_range", 20.0);
    this->declare_parameter<double>("canopy_ground_z", -0.47);
    this->declare_parameter<double>("canopy_ground_halfband", 0.8);
    this->declare_parameter<double>("canopy_normal_z_min", 0.966);
    this->declare_parameter<double>("canopy_frac_enter", 0.65);
    this->declare_parameter<double>("canopy_frac_exit", 0.75);
    this->declare_parameter<double>("canopy_zp95_enter", 1.5);
    this->declare_parameter<int>("canopy_min_near", 50);
    this->declare_parameter<int>("canopy_enter_scans", 10);
    this->declare_parameter<int>("canopy_exit_scans", 30);
    this->declare_parameter<bool>("ground_recover_en", false);
    this->declare_parameter<double>("ground_recover_scan_band", 0.15);
    this->declare_parameter<int>("ground_recover_min_inliers", 100);
    this->declare_parameter<double>("ground_recover_max_residual", 1.0);
    this->declare_parameter<double>("ground_recover_sector_tol", 0.25);
    this->declare_parameter<int>("ground_recover_max_rows", 4000);
    this->declare_parameter<bool>("tem_en", false);
    this->declare_parameter<double>("tem_cell", 5.0);
    this->declare_parameter<double>("tem_tau", 2.0);
    this->declare_parameter<double>("tem_egf_floor", 0.30);
    this->declare_parameter<double>("tem_egf_ref", 0.60);
    this->declare_parameter<double>("tem_sigma0", 0.15);
    this->declare_parameter<bool>("tem_anchor_en", false);
    this->declare_parameter<double>("tem_anchor_max_age", 3.0);
    this->declare_parameter<bool>("clutter_deweight_en", false);
    this->declare_parameter<double>("clutter_th_lo", 0.05);
    this->declare_parameter<double>("clutter_th_hi", 0.20);
    this->declare_parameter<double>("clutter_min_weight", 0.15);
    this->declare_parameter<bool>("vxl_en", false);
    this->declare_parameter<double>("vxl_size_xy", 1.0);
    this->declare_parameter<double>("vxl_size_z", 0.25);
    this->declare_parameter<double>("vxl_sigma_r", 0.05);
    this->declare_parameter<double>("vxl_min_weight", 0.10);
    this->declare_parameter<double>("vxl_max_range", 30.0);
    this->declare_parameter<int>("vxl_min_pts", 8);
    this->declare_parameter<int>("vxl_freeze_pts", 300);
    this->declare_parameter<bool>("vxl_env_en", false);
    this->declare_parameter<int>("vxl_env_min_pts", 20);
    this->declare_parameter<double>("vxl_env_band", 0.12);
    this->declare_parameter<double>("vxl_env_nz_min", 0.5);
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
    this->declare_parameter<bool>("diagnostics.kalman_channel_enable", false);
    this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
    this->declare_parameter<int>("mapping.map_insert_delay_scans", 0);
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
    this->get_parameter_or<int>("max_buffered_scans", max_buffered_scans, 100);
    this->get_parameter_or<double>("max_scan_backlog_sec", max_scan_backlog_sec, 0.5);
    this->get_parameter_or<bool>("adaptive_downsample_en", adaptive_ds_params.enabled, false);
    this->get_parameter_or<double>("adaptive_downsample_budget_sec", adaptive_ds_params.budget_sec, 0.06);
    this->get_parameter_or<double>("adaptive_downsample_release_sec", adaptive_ds_params.release_sec, 0.035);
    this->get_parameter_or<double>("adaptive_downsample_step", adaptive_ds_params.step, 0.05);
    this->get_parameter_or<double>("adaptive_downsample_max_voxel", adaptive_ds_params.max_voxel, 0.5);
    this->get_parameter_or<int>("adaptive_downsample_raise_after", adaptive_ds_params.raise_after, 5);
    this->get_parameter_or<int>("adaptive_downsample_lower_after", adaptive_ds_params.lower_after, 100);
    this->get_parameter_or<bool>("degeneracy_debug", degeneracy_debug, false);
    this->get_parameter_or<bool>("health_pub_en", health_pub_en, true);
    this->get_parameter_or<bool>("planar_constraint_en", planar_constraint_en, false);
    this->get_parameter_or<double>("planar_constraint_noise", planar_constraint_noise, 0.1);
    this->get_parameter_or<double>("planar_constraint_z_weak_thresh", planar_constraint_z_weak_thresh, 0.5);
    this->get_parameter_or<bool>("gravity_align_en", gravity_align_en, false);
    this->get_parameter_or<double>("gravity_align_noise", gravity_align_noise, 0.05);
    this->get_parameter_or<double>("gravity_align_z_weak_thresh", gravity_align_z_weak_thresh, 0.3);
    this->get_parameter_or<double>("gravity_align_accel_tol", gravity_align_accel_tol, 0.05);
    this->get_parameter_or<double>("gravity_align_gyro_tol", gravity_align_gyro_tol, 0.35);
    this->get_parameter_or<double>("gravity_align_window_s", gravity_align_window_s, 0.0);
    this->get_parameter_or<bool>("gravity_align_continuous", gravity_align_continuous, false);
    this->get_parameter_or<double>("gravity_align_hold_s", gravity_align_hold_s, 0.0);
    this->get_parameter_or<double>("gravity_align_noise_degraded", gravity_align_noise_degraded, 0.0);
    this->get_parameter_or<double>("attitude_cov_floor_deg", attitude_cov_floor_deg, 0.0);
    this->get_parameter_or<double>("gravity_align_mag_ref", gravity_align_mag_ref, 0.0);
    this->get_parameter_or<double>("gravity_align_grav_leak", gravity_align_grav_leak, 1.0);
    this->get_parameter_or<double>("gravity_align_grav_cap_deg", gravity_align_grav_cap_deg, 3.0);
    this->get_parameter_or<bool>("imu_init_require_still", imu_init_require_still_, false);
    this->get_parameter_or<double>("imu_init_still_tol", imu_init_still_tol_, 0.03);
    this->get_parameter_or<double>("imu_init_still_timeout_s", imu_init_still_timeout_s_, 20.0);
    this->get_parameter_or<bool>("divergence_guard_en", divergence_guard_en, true);
    this->get_parameter_or<int>("divergence_guard_min_eff", divergence_guard_min_eff, 50);
    this->get_parameter_or<bool>("guard_engulf_en", guard_engulf_en, true);
    this->get_parameter_or<double>("guard_engulf_far_range", guard_engulf_far_range, 2.0);
    this->get_parameter_or<double>("guard_engulf_far_frac_min", guard_engulf_far_frac_min, 0.15);
    this->get_parameter_or<int>("guard_engulf_streak", guard_engulf_streak, 2);
    this->get_parameter_or<double>("guard_engulf_far_frac_release", guard_engulf_far_frac_release, 0.35);
    this->get_parameter_or<int>("guard_engulf_release_scans", guard_engulf_release_scans, 5);
    this->get_parameter_or<bool>("guard_hole_en", guard_hole_en, false);
    this->get_parameter_or<double>("guard_hole_min_sec", guard_hole_min_sec, 0.3);
    this->get_parameter_or<double>("guard_hole_max_accel", guard_hole_max_accel, 1.0);
    this->get_parameter_or<bool>("guard_hole_reanchor_pos", guard_hole_reanchor_pos, true);
    this->get_parameter_or<int>("divergence_guard_streak", divergence_guard_streak, 5);
    this->get_parameter_or<bool>("zupt_en", zupt_en, true);
    this->get_parameter_or<double>("zupt_gyro_thresh", zupt_gyro_thresh, 0.05);
    this->get_parameter_or<double>("zupt_acc_span_ratio", zupt_acc_span_ratio, 0.02);
    this->get_parameter_or<double>("zupt_wheel_speed_thresh", zupt_wheel_speed_thresh, 0.05);
    this->get_parameter_or<double>("zupt_wheel_max_age", zupt_wheel_max_age, 0.5);
    this->get_parameter_or<bool>("zupt_require_wheel", zupt_require_wheel, false);
    this->get_parameter_or<bool>("reanchor_en", reanchor_params.enabled, true);
    this->get_parameter_or<int>("reanchor_min_eff", reanchor_params.min_eff, 200);
    this->get_parameter_or<double>("reanchor_max_res", reanchor_params.max_res, 0.10);
    this->get_parameter_or<int>("reanchor_confirm_scans", reanchor_params.confirm_scans, 3);
    this->get_parameter_or<double>("reanchor_timeout_sec", reanchor_params.timeout_sec, 5.0);
    this->get_parameter_or<std::string>("reanchor_on_fail", reanchor_on_fail, std::string("hold"));
    this->get_parameter_or<bool>("runaway_watchdog_en", runaway_params.enabled, true);
    this->get_parameter_or<double>("runaway_parked_speed_thresh", runaway_params.parked_speed_thresh, 0.3);
    this->get_parameter_or<double>("runaway_wheel_speed_margin", runaway_params.wheel_speed_margin, 1.0);
    this->get_parameter_or<double>("runaway_hold_sec", runaway_params.hold_sec, 3.0);
    this->get_parameter_or<std::string>("runaway_on_trip", runaway_on_trip, std::string("hold"));
    runaway_params.wheel_max_age = zupt_wheel_max_age;
    // 轮速要先自己报出运动,才有资格指认跑飞:恒 0 的速度源(m20 的 /lio/twist 是 GNSS
    // 代理,拒止段整段读 0)否则会把正常行走判成跑飞。
    runaway_params.wheel_moving_thresh = zupt_wheel_speed_thresh;
    this->get_parameter_or<double>("divergence_guard_max_speed", divergence_guard_max_speed, 30.0);
    this->get_parameter_or<bool>("canopy_detect_en", canopy_detect_en, false);
    this->get_parameter_or<bool>("canopy_debug", canopy_debug, false);
    this->get_parameter_or<double>("canopy_near_range", canopy_near_range, 20.0);
    this->get_parameter_or<double>("canopy_ground_z", canopy_ground_z, -0.47);
    this->get_parameter_or<double>("canopy_ground_halfband", canopy_ground_halfband, 0.8);
    this->get_parameter_or<double>("canopy_normal_z_min", canopy_normal_z_min, 0.966);
    this->get_parameter_or<double>("canopy_frac_enter", canopy_frac_enter, 0.65);
    this->get_parameter_or<double>("canopy_frac_exit", canopy_frac_exit, 0.75);
    this->get_parameter_or<double>("canopy_zp95_enter", canopy_zp95_enter, 1.5);
    this->get_parameter_or<int>("canopy_min_near", canopy_min_near, 50);
    this->get_parameter_or<int>("canopy_enter_scans", canopy_enter_scans, 10);
    this->get_parameter_or<int>("canopy_exit_scans", canopy_exit_scans, 30);
    this->get_parameter_or<bool>("ground_recover_en", ground_recover_en, false);
    this->get_parameter_or<double>("ground_recover_scan_band", ground_recover_scan_band, 0.15);
    this->get_parameter_or<int>("ground_recover_min_inliers", ground_recover_min_inliers, 100);
    this->get_parameter_or<double>("ground_recover_max_residual", ground_recover_max_residual, 1.0);
    this->get_parameter_or<double>("ground_recover_sector_tol", ground_recover_sector_tol, 0.25);
    this->get_parameter_or<int>("ground_recover_max_rows", ground_recover_max_rows, 4000);
    this->get_parameter_or<bool>("tem_en", tem_en, false);
    this->get_parameter_or<double>("tem_cell", tem_cell, 5.0);
    this->get_parameter_or<double>("tem_tau", tem_tau, 2.0);
    this->get_parameter_or<double>("tem_egf_floor", tem_egf_floor, 0.30);
    this->get_parameter_or<double>("tem_egf_ref", tem_egf_ref, 0.60);
    this->get_parameter_or<double>("tem_sigma0", tem_sigma0, 0.15);
    this->get_parameter_or<bool>("tem_anchor_en", tem_anchor_en, false);
    this->get_parameter_or<double>("tem_anchor_max_age", tem_anchor_max_age, 3.0);
    this->get_parameter_or<bool>("clutter_deweight_en", clutter_deweight_en, false);
    this->get_parameter_or<double>("clutter_th_lo", clutter_th_lo, 0.05);
    this->get_parameter_or<double>("clutter_th_hi", clutter_th_hi, 0.20);
    this->get_parameter_or<double>("clutter_min_weight", clutter_min_weight, 0.15);
    this->get_parameter_or<bool>("vxl_en", vxl_en, false);
    this->get_parameter_or<double>("vxl_size_xy", vxl_size_xy, 1.0);
    this->get_parameter_or<double>("vxl_size_z", vxl_size_z, 0.25);
    this->get_parameter_or<double>("vxl_sigma_r", vxl_sigma_r, 0.05);
    this->get_parameter_or<double>("vxl_min_weight", vxl_min_weight, 0.10);
    this->get_parameter_or<double>("vxl_max_range", vxl_max_range, 30.0);
    this->get_parameter_or<int>("vxl_min_pts", vxl_min_pts, 8);
    this->get_parameter_or<int>("vxl_freeze_pts", vxl_freeze_pts, 300);
    this->get_parameter_or<bool>("vxl_env_en", vxl_env_en, false);
    this->get_parameter_or<int>("vxl_env_min_pts", vxl_env_min_pts, 20);
    this->get_parameter_or<double>("vxl_env_band", vxl_env_band, 0.12);
    this->get_parameter_or<double>("vxl_env_nz_min", vxl_env_nz_min, 0.5);
    this->get_parameter_or<string>("map_file_path", map_file_path, "");
    string data_dir_param;
    this->get_parameter_or<string>("data_dir", data_dir_param, "");
    if (!data_dir_param.empty()) {
      root_dir = data_dir_param;
      if (root_dir.back() != '/')
        root_dir += '/';
    }
    this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
    this->get_parameter_or<string>("common.imu_topic", imu_topic, "/livox/imu");
    this->get_parameter_or<string>("common.imu_input_type", imu_input_type, string("sensor_msgs"));
    this->get_parameter_or<string>("common.imu_bias_topic", imu_bias_topic, string("/fixposition/fpa/imubias"));
    this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
    this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    this->get_parameter_or<double>("common.imu_time_offset", imu_time_offset, 0.0);
    this->get_parameter_or<double>("filter_size_corner", filter_size_corner_min, 0.5);
    this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);
    this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);
    this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);
    this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);
    this->get_parameter_or<double>("mapping.fov_degree", fov_deg, 180.f);
    this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);
    this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);
    this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
    this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
    this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
    this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
    this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
    this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
    this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
    this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
    this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    p_imu->runtime_log_en = runtime_pos_log;                // mirror to ImuProcess so Log/imu.txt is written
    ikd_profile = (getenv("FLIO_IKD_PROFILE") != nullptr);  // [ikd-profile] opt-in via env
    {
      const char * am = getenv("FLIO_ASYNC_MAP");
      if (am && std::string(am) == "0")
        async_map_en = false;
    }
#ifdef USE_IVOX
    {  // option B: iVox map backend. nearby_type MUST be >=18 (Phase-0: NEARBY6 inadequate).
      // res=0.5 NN voxels give the best probe NN-agreement (99.9%, tightest normals);
      // a GLOBAL fine-grid dedup (= filter_size_map) keeps the map at ikd-Tree's 542k.
      const double ivox_res = this->declare_parameter<double>("ivox_grid_resolution", 0.5);
      const int ivox_nearby = this->declare_parameter<int>("ivox_nearby_type", 26);
      const int ivox_vcap = this->declare_parameter<int>("ivox_voxel_capacity", 50);
      const int ivox_maxvox = this->declare_parameter<int>("ivox_max_voxels", 5000000);
      ikdtree.Init(static_cast<float>(ivox_res), ivox_nearby, ivox_vcap, static_cast<std::size_t>(ivox_maxvox));
      RCLCPP_INFO(this->get_logger(),
                  "[IVOX] map backend = iVox (res=%.2f nearby=%d voxel_cap=%d max_voxels=%d)",
                  ivox_res,
                  ivox_nearby,
                  ivox_vcap,
                  ivox_maxvox);
    }
#endif
    this->get_parameter_or<bool>("diagnostics.latency_enable", latency_diag_en, true);
    this->get_parameter_or<double>("diagnostics.latency_log_period_sec", latency_log_period_sec, 1.0);
    this->get_parameter_or<bool>("diagnostics.kalman_channel_enable", kalman_channel_diag_en, false);
    this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
    this->get_parameter_or<int>("mapping.map_insert_delay_scans", map_insert_delay_scans, 0);
    if (map_insert_delay_scans > 0) {
      RCLCPP_WARN(this->get_logger(), "[MAP] deterministic insertion lag: %d scan(s)", map_insert_delay_scans);
    }
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
      R_robot_to_imu = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(wext.data());
    RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

    path.header.stamp = this->get_clock()->now();
    path.header.frame_id = "camera_init";

    // /*** variables definition ***/
    // int effect_feat_num = 0, frame_num = 0;
    // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
    // aver_time_solve = 0, aver_time_const_H_time = 0; bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->init_require_still = imu_init_require_still_;
    p_imu->init_still_tol = imu_init_still_tol_;
    p_imu->init_still_timeout_s = imu_init_still_timeout_s_;
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    fill(epsi, epsi + 23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    // FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    // ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    fout_jump.open(DEBUG_FILE_DIR("jump_diag.txt"), ios::out);
    if (kalman_channel_diag_en) {
      fout_kalman.open(DEBUG_FILE_DIR("kalman_channels.csv"), ios::out);
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
                  << "p_ba_x_before,p_ba_y_before,p_ba_z_before,p_ba_x_after,p_ba_y_after,p_ba_z_after\n";
      RCLCPP_INFO(this->get_logger(),
                  "Kalman channel diagnostics enabled: %s (schema v1)",
                  DEBUG_FILE_DIR("kalman_channels.csv").c_str());
    }
    if (fout_pre && fout_out)
      cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
      cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    if (p_pre->lidar_type == AVIA) {
      sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
    } else if (p_pre->lidar_type == ALLPOINT) {
      // /pre/all_point is published RELIABLE with fixed 8MB messages; a small
      // reliable KeepLast queue matches the bag publisher and bounds memory
      // (a few buffered scans) while avoiding silent drops of whole scans.
      rclcpp::QoS shm_qos(rclcpp::KeepLast(5));
      sub_pcl_shm_ =
        this->create_subscription<shm_msgs::msg::PointCloud8mAndPose>(lid_topic, shm_qos, shm_allpoint_cbk);
      RCLCPP_INFO(
        this->get_logger(), "shm allpoint mode: subscribing %s (shm_msgs/PointCloud8mAndPose)", lid_topic.c_str());
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
      sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, pc_qos, standard_pcl_cbk);
    }
    if (imu_input_type == "fixposition_fpa") {
      auto imu_qos = rclcpp::SensorDataQoS();
      sub_imu_fpa_ = this->create_subscription<fixposition_driver_msgs::msg::FpaImu>(imu_topic, imu_qos, imu_fpa_cbk);
      sub_imu_bias_ =
        this->create_subscription<fixposition_driver_msgs::msg::FpaImubias>(imu_bias_topic, imu_qos, imu_bias_cbk);
      RCLCPP_INFO(this->get_logger(),
                  "Fixposition IMU direct mode: topic=%s bias_topic=%s imu_time_offset=%.3f",
                  imu_topic.c_str(),
                  imu_bias_topic.c_str(),
                  imu_time_offset);
    } else {
      sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 1000, imu_cbk);
      RCLCPP_INFO(this->get_logger(), "Standard IMU mode: topic=%s", imu_topic.c_str());
    }
    if (wheel_odom_en || zupt_en) {
      sub_twist_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(wheel_topic, 2000, twist_cbk);
      RCLCPP_INFO(this->get_logger(),
                  "Wheel twist subscribed: topic=%s scale=%.3f (fusion %s, static hold %s)",
                  wheel_topic.c_str(),
                  wheel_speed_scale,
                  wheel_odom_en ? "on" : "off",
                  zupt_en ? "on" : "off");
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

    // The voxel-capped /Laser_map is a coarse global map, not a live feed, so it
    // publishes slowly (default 0.2 Hz). Period is configurable via publish.map_period_sec.
    double map_pub_period_sec = 5.0;
    this->get_parameter_or<double>("publish.map_period_sec", map_pub_period_sec, 5.0);
    if (map_pub_period_sec <= 0.0)
      map_pub_period_sec = 5.0;
    auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(map_pub_period_sec * 1000.0));
    map_pub_timer_ = rclcpp::create_timer(
      this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

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

    // Graph-side FE Z-drift feedback for the TEM (terrain anchoring). Tiny
    // low-rate scalar stream from the PGO node; inter-process only.
    if (tem_en && tem_anchor_en) {
      rclcpp::SubscriptionOptions drift_opts;
      drift_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
      fe_z_drift_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/lio_slam/fe_z_drift",
        rclcpp::QoS(1).best_effort(),
        [](nav_msgs::msg::Odometry::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(tem_anchor_mtx);
          tem_anchor_d = msg->pose.pose.position.z;
          tem_anchor_sigma = std::sqrt(std::max(msg->pose.covariance[14], 1e-4));
          tem_anchor_stamp = rclcpp::Time(msg->header.stamp).seconds();
        },
        drift_opts);
      RCLCPP_INFO(
        this->get_logger(), "[TEM] graph drift anchoring ON (/lio_slam/fe_z_drift, max_age=%.1fs)", tem_anchor_max_age);
    }

    RCLCPP_INFO(this->get_logger(), "Node init finished.");
  }

  ~LaserMappingNode()
  {
    // The off-thread map worker is a global std::thread; when the node is used as a
    // component (offline replay, composed pipelines) nobody else joins it before the
    // library unloads → std::terminate. Idempotent with the standalone main's call.
    stopMapWorker();
    fout_out.close();
    fout_pre.close();
    fout_jump.close();
    fout_kalman.close();
    fclose(fp);
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
                          const V3D & prediction)
  {
    if (!kalman_channel_diag_en || !fout_kalman.is_open())
      return;

    const V3D dpos = after.pos - before.pos;
    const V3D dvel = after.vel - before.vel;
    const V3D dba = after.ba - before.ba;
    const V3D dgrav(after.grav[0] - before.grav[0], after.grav[1] - before.grav[1], after.grav[2] - before.grav[2]);
    const auto & dx = diagnostics.last_dx;
    const auto & gain = diagnostics.gain_row_norm;
    const auto & projection = diagnostics.correction_projection_row_norm;
    const auto & pBefore = diagnostics.covariance_diag_before;
    const auto & pAfter = diagnostics.covariance_diag_after;

    fout_kalman << std::setprecision(17) << 1 << ',' << time_log_counter << ','
                << Measures.lidar_beg_time - first_lidar_time << ',' << channel << ',' << diagnostics.valid << ','
                << diagnostics.converged << ',' << diagnostics.iterations << ',' << diagnostics.measurement_dim << ','
                << diagnostics.residual_norm << ',' << diagnostics.weighted_residual_squared << ',' << effct_feat_num
                << ',' << res_mean_last << ',' << pos_obs_eig_min << ',' << pos_obs_eig_max << ',' << pos_obs_z_weak
                << ',' << measurement.x() << ',' << measurement.y() << ',' << measurement.z() << ',' << prediction.x()
                << ',' << prediction.y() << ',' << prediction.z();
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
                << pBefore(20) << ',' << pAfter(18) << ',' << pAfter(19) << ',' << pAfter(20) << '\n';
    fout_kalman.flush();
  }

  void timer_callback()
  {
    if (sync_packages(Measures)) {
      if (flg_first_scan) {
        first_lidar_time = Measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        flg_first_scan = false;
        return;
      }

      double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;
      // Pre-iEKF split: [IKDPROF] covers only the iEKF and the map-update loop, so the
      // undistort / FOV-segment / downsample costs were invisible in the per-frame budget.
      double t_undist = 0.0, t_join = 0.0, t_fov = 0.0;

      match_time = 0;
      kdtree_search_time = 0.0;
      solve_time = 0;
      solve_const_H_time = 0;
      svd_time = 0;
      t0 = omp_get_wtime();

      p_imu->Process(Measures, kf, feats_undistort);
      t_undist = omp_get_wtime();
      state_point = kf.get_x();

      /*** Scan-hole guard: bound the IMU-only excursion across a gap ***/
      if (guard_hole_en && flg_EKF_inited && hole_last_scan_end > 0.0) {
        const double gap = Measures.lidar_beg_time - hole_last_scan_end;
        if (gap > guard_hole_min_sec) {
          state_ikfom st = kf.get_x();
          const double spd_imu = st.vel.norm();
          const V3D dv = st.vel - hole_vel_pre;
          const double dv_max = guard_hole_max_accel * gap;
          const double dv_n = dv.norm();
          if (dv_n > dv_max) {
            st.vel = hole_vel_pre + dv * (dv_max / dv_n);
          }
          const V3D pos_imu = st.pos;
          if (guard_hole_reanchor_pos) {
            st.pos = hole_pos_pre + hole_vel_pre * gap;
          }
          kf.change_x(st);
          state_point = kf.get_x();
          RCLCPP_WARN(this->get_logger(),
                      "[HOLE-GUARD] scan hole %.2fs: imu-only vel %.2f m/s (pre-hole %.2f) -> held %.2f; pos re-anchored by %.2fm",
                      gap,
                      spd_imu,
                      hole_vel_pre.norm(),
                      state_point.vel.norm(),
                      guard_hole_reanchor_pos ? (pos_imu - state_point.pos).norm() : 0.0);
        }
      }
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

      if (feats_undistort->empty() || (feats_undistort == NULL)) {
        if (!diag_first_no_point_logged) {
          RCLCPP_WARN(
            this->get_logger(), "[STARTUP][FAST_LIO] first no-point scan lidar_beg=%.3f", Measures.lidar_beg_time);
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

      flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
      /*** Restore the ikd-tree single-operator invariant: wait for the previous
       *   scan's off-thread map Add to finish before this scan touches the tree. ***/
      if (async_map_en)
        joinMapAdd();
      t_join = omp_get_wtime();

      /*** Segment the map in lidar FOV ***/
      lasermap_fov_segment();
      t_fov = omp_get_wtime();

      /*** downsample the feature points in a scan ***/
      downSizeFilterSurf.setInputCloud(feats_undistort);
      downSizeFilterSurf.filter(*feats_down_body);
      t1 = omp_get_wtime();
      feats_down_size = feats_down_body->points.size();
      /*** Engulfment telemetry: far-field fraction of the raw downsampled scan ***/
      if (feats_down_size > 0) {
        int far_cnt = 0;
        const double far_r2 = guard_engulf_far_range * guard_engulf_far_range;
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
        RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
        if (feats_down_size > 5) {
          // --- Prior map preload (Scheme 1) ---
          if (!prior_map_pcd_.empty()) {
            // 1. Inject initial pose into EKF if provided
            if (initial_pose_vec_.size() >= 6) {
              state_ikfom new_state = kf.get_x();
              new_state.pos = Eigen::Vector3d(initial_pose_vec_[0], initial_pose_vec_[1], initial_pose_vec_[2]);
              double roll_rad = initial_pose_vec_[3] * M_PI / 180.0;
              double pitch_rad = initial_pose_vec_[4] * M_PI / 180.0;
              double yaw_rad = initial_pose_vec_[5] * M_PI / 180.0;
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
                          initial_pose_vec_[0],
                          initial_pose_vec_[1],
                          initial_pose_vec_[2],
                          initial_pose_vec_[5],
                          initial_pose_full_rpy_override_ ? "true" : "false");
            }

            // 2. Load prior PCD
            PointCloudXYZI::Ptr prior_cloud(new PointCloudXYZI());
            if (pcl::io::loadPCDFile(prior_map_pcd_, *prior_cloud) == 0) {
              RCLCPP_INFO(
                this->get_logger(), "Loaded prior map: %s (%zu points)", prior_map_pcd_.c_str(), prior_cloud->size());

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
                const double dist_to_origin = std::sqrt(cx * cx + cy * cy + cz * cz);
                RCLCPP_INFO(this->get_logger(),
                            "Prior map bbox: x=[%.2f..%.2f] y=[%.2f..%.2f] "
                            "z=[%.2f..%.2f] center→origin=%.1f m%s",
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            zmin,
                            zmax,
                            dist_to_origin,
                            dist_to_origin > 10000.0 ? "  (large offset — likely UTM/global frame, "
                                                       "check coordinate system!)"
                                                     : "");
              }

              // 3. Downsample prior map
              pcl::VoxelGrid<PointType> voxel;
              voxel.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
              voxel.setInputCloud(prior_cloud);
              PointCloudXYZI::Ptr prior_ds(new PointCloudXYZI());
              voxel.filter(*prior_ds);
              RCLCPP_INFO(
                this->get_logger(), "Prior map downsampled: %zu → %zu points", prior_cloud->size(), prior_ds->size());

              // 4. Transform current scan to world frame
              ikdtree.set_downsample_param(filter_size_map_min);
              feats_down_world->resize(feats_down_size);
              for (int i = 0; i < feats_down_size; i++) {
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
                          prior_ds->size(),
                          feats_down_size,
                          combined->size());
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
              ikdtree.Build(feats_down_world->points);
            }
            prior_map_pcd_.clear();  // Prevent re-execution
          } else {
            // Normal init: build from current scan only
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++) {
              pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
          }
        }
        return;
      }
      int featsFromMapNum = ikdtree.validnum();
      kdtree_size_st = ikdtree.size();

      // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num:
      // "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

      /*** ICP and iterated Kalman filter update ***/
      if (feats_down_size < 5) {
        RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
        return;
      }

      normvec->resize(feats_down_size);
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
      Nearest_Points.resize(feats_down_size);
      int rematch_num = 0;
      bool nearest_search_en = true;  //

      double t_pre_icp = omp_get_wtime();

      /*** iterated state estimation ***/
      const auto state_before_update = state_point;
      const V3D euler_before_update = SO3ToEuler(state_before_update.rot);
      const V3D pos_before_update = state_before_update.pos;
      double t_update_start = omp_get_wtime();
      // [ikd-profile] snapshot background/inline rebuild work spanning iEKF + map_incre
      const uint64_t prof_rb_us0 = g_ikd_rebuild_us.load(std::memory_order_relaxed);
      const uint64_t prof_rb_cnt0 = g_ikd_rebuild_count.load(std::memory_order_relaxed);
      const uint64_t prof_inl_us0 = g_ikd_inline_rebuild_us.load(std::memory_order_relaxed);
      double solve_H_time = 0;
      EkfUpdateDiagnostics lidarDiagnostics;
      Eigen::Matrix<double, 23, 23> lidarCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
      if (kalman_channel_diag_en)
        lidarCovarianceBefore = kf.get_P();
      kf.update_iterated_dyn_share_modified(
        LASER_POINT_COV, solve_H_time, kalman_channel_diag_en ? &lidarDiagnostics : nullptr);

      state_point = kf.get_x();
      if (kalman_channel_diag_en) {
        writeKalmanChannel(
          "lidar", state_before_update, state_point, lidarCovarianceBefore, lidarDiagnostics, Zero3d, Zero3d);
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
      if (divergence_guard_en) {
        if (effct_feat_num < divergence_guard_min_eff)
          ++consec_low_eff;
        else
          consec_low_eff = 0;
        flio_vel_runaway = (state_point.vel.norm() > divergence_guard_max_speed);
        // (3) engulfment: far field collapsed in the raw scan — leading indicator
        //     (fires before effct starves; see guard_engulf_en block comment).
        if (guard_engulf_en) {
          if (scan_far_frac < guard_engulf_far_frac_min)
            ++consec_engulf;
          else
            consec_engulf = 0;
          if (!engulf_latched) {
            if (consec_engulf >= guard_engulf_streak) {
              engulf_latched = true;
              consec_engulf_clear = 0;
            }
          } else {
            // Latched: release only on a SUSTAINED healthy far field (hysteresis).
            if (scan_far_frac >= guard_engulf_far_frac_release)
              ++consec_engulf_clear;
            else
              consec_engulf_clear = 0;
            if (consec_engulf_clear >= guard_engulf_release_scans) {
              engulf_latched = false;
              consec_engulf = 0;
              // Discard the blind stretch's velocity (see block comment).
              state_ikfom st = kf.get_x();
              st.vel.setZero();
              kf.change_x(st);
              state_point = kf.get_x();
              RCLCPP_WARN(this->get_logger(),
                          "[DIVERGENCE-GUARD] engulfment released (far_frac=%.2f for %d scans): "
                          "map unfrozen, velocity reset",
                          scan_far_frac, guard_engulf_release_scans);
            }
          }
          flio_engulfed = engulf_latched;
        }
      }
      const bool flio_in_degraded =
        divergence_guard_en && (consec_low_eff >= divergence_guard_streak || flio_vel_runaway || flio_engulfed);

      /*** Canopy mode debounce (once per scan, on the h_share_model raw
       *   stats). Enter needs BOTH signals (ground fraction collapsed AND
       *   near-field points tall) for canopy_enter_scans consecutive scans;
       *   exit needs the ground fraction healthy for canopy_exit_scans.
       *   Scans with too few near-field points are neutral: hold state,
       *   reset both streaks (no drift toward either decision). ***/
      if (canopy_detect_en) {
        const bool measurable = canopy_near_count >= canopy_min_near;
        const bool vote_enter = measurable && canopy_ground_frac < canopy_frac_enter && canopy_zp95 > canopy_zp95_enter;
        const bool vote_exit = measurable && canopy_ground_frac > canopy_frac_exit;
        if (!canopy_mode) {
          canopy_enter_streak = vote_enter ? canopy_enter_streak + 1 : 0;
          if (canopy_enter_streak >= canopy_enter_scans) {
            canopy_mode = true;
            canopy_enter_streak = 0;
            RCLCPP_WARN(this->get_logger(),
                        "[CANOPY] ENTER ground_frac=%.2f zp95=%.2f near=%d posZ=%.2f",
                        canopy_ground_frac,
                        canopy_zp95,
                        canopy_near_count,
                        state_point.pos[2]);
          }
        } else {
          canopy_exit_streak = vote_exit ? canopy_exit_streak + 1 : 0;
          if (canopy_exit_streak >= canopy_exit_scans) {
            canopy_mode = false;
            canopy_exit_streak = 0;
            RCLCPP_WARN(this->get_logger(),
                        "[CANOPY] EXIT ground_frac=%.2f zp95=%.2f near=%d posZ=%.2f",
                        canopy_ground_frac,
                        canopy_zp95,
                        canopy_near_count,
                        state_point.pos[2]);
          }
        }
        if (canopy_debug)
          std::cerr << "[CANOPY] t=" << std::fixed << std::setprecision(3) << lidar_end_time
                    << " frac=" << canopy_ground_frac << " zp95=" << canopy_zp95 << " near=" << canopy_near_count
                    << " mode=" << canopy_mode << " rec=" << canopy_ground_recovered << " posZ=" << state_point.pos[2]
                    << std::endl;
      }

      /*** TEM update (once per scan, CONVERGED pose — h_share iterations
       *   only read the grid). Learning rate = alpha0(tem_tau) x q where
       *   q is the dead-banded, squared ramp of the smoothed egf: fully
       *   frozen at/below tem_egf_floor (any nonzero rate would hand the
       *   memory to the drift over a ~700 s canopy dwell), full rate
       *   at/above tem_egf_ref. Samples are the scan's fit-band inliers —
       *   never recovered rows, so TEM cannot vouch for itself. ***/
      if (tem_en) {
        static double tem_last_t = -1.0;
        const double dt =
          (tem_last_t > 0.0 && lidar_end_time > tem_last_t) ? std::min(lidar_end_time - tem_last_t, 0.5) : 0.1;
        tem_last_t = lidar_end_time;
        // Smoothed egf (~1 s at 10 Hz); unmeasurable scans hold the estimate.
        if (canopy_near_count >= canopy_min_near)
          tem_egf_ema += 0.1 * (canopy_ground_frac - tem_egf_ema);
        double q = (tem_egf_ema - tem_egf_floor) / std::max(tem_egf_ref - tem_egf_floor, 1e-6);
        q = std::clamp(q, 0.0, 1.0);
        q *= q;
        // Graph-side drift note: while FRESH, samples are drift-corrected
        // (z + d) and learning runs at a high fixed rate regardless of
        // egf — the poisoning egf guarded against is measured and removed
        // upstream. Stale note = plain egf-gated self-learning.
        double d_use = 0.0, d_sig2 = 0.0;
        bool anchored = false;
        if (tem_anchor_en) {
          std::lock_guard<std::mutex> lk(tem_anchor_mtx);
          if (tem_anchor_stamp > 0.0 && std::fabs(lidar_end_time - tem_anchor_stamp) < tem_anchor_max_age) {
            d_use = tem_anchor_d;
            d_sig2 = tem_anchor_sigma * tem_anchor_sigma;
            anchored = true;
          }
        }
        constexpr double kTemAnchorQ = 0.8;
        const double q_eff = anchored ? std::max(q, kTemAnchorQ) : q;
        if (q_eff > 0.0 && !tem_samples.empty()) {
          struct TemAcc
          {
            double n = 0.0, sz = 0.0, szz = 0.0;
          };
          static std::unordered_map<int64_t, TemAcc> acc;
          acc.clear();
          for (const auto & pb : tem_samples) {
            const V3D pw =
              state_point.rot * (state_point.offset_R_L_I * pb + state_point.offset_T_L_I) + state_point.pos;
            const double zc = pw.z() + d_use;  // drift-corrected (epoch-frame) height
            TemAcc & a = acc[temKey(pw.x(), pw.y())];
            a.n += 1.0;
            a.sz += zc;
            a.szz += zc * zc;
          }
          const double alpha0 = 1.0 - std::exp(-dt / std::max(tem_tau, 1e-3));
          constexpr double kTemMinCellSamples = 5.0;  // one-point cells carry no spread info
          constexpr double kTemWMax = 30.0;           // confidence cap (~30 healthy scans)
          constexpr double kTemVarPrior = 0.01;       // (0.1 m)^2 until frames disagree
          for (const auto & kv : acc) {
            const TemAcc & a = kv.second;
            if (a.n < kTemMinCellSamples)
              continue;
            const double mean_s = a.sz / a.n;
            const double var_s = std::max(a.szz / a.n - mean_s * mean_s, 0.0) + d_sig2;
            auto it = tem_grid.find(kv.first);
            if (it == tem_grid.end()) {
              tem_grid.emplace(kv.first, TemCell{mean_s, var_s + kTemVarPrior, q_eff});
              continue;
            }
            TemCell & c = it->second;
            const double alpha = alpha0 * q_eff;
            const double dz = mean_s - c.z;
            c.z += alpha * dz;
            // Variance sees intra-cell spread AND frame-to-frame
            // disagreement: slopes and residual drift inflate it, and
            // the recovery row weight collapses with it.
            c.var += alpha * (dz * dz + var_s - c.var);
            c.w = std::min(c.w + q_eff, kTemWMax);
            // Online slope statistic from served-neighbor gradients:
            // the extrapolation envelope's only terrain prior, learned
            // on site. HIGH-QUALITY updates only — at low q a cell
            // freshly touched by a drifting estimate sits metres from
            // its pre-drift neighbor and the gradient samples poison
            // the statistic (measured: 0.014 → 0.05-0.11 through the
            // 0702 canopy edge, blowing sigma_ext to 20-35 m).
            if (q_eff < 0.5)
              continue;
            constexpr double kTemMinServeW = 2.0;
            int64_t ckx, cky;
            temKeyToIdx(kv.first, ckx, cky);
            const int64_t nkeys[4] = {
              temKeyIdx(ckx - 1, cky), temKeyIdx(ckx + 1, cky), temKeyIdx(ckx, cky - 1), temKeyIdx(ckx, cky + 1)};
            for (const int64_t nk : nkeys) {
              const auto nit = tem_grid.find(nk);
              if (nit == tem_grid.end() || nit->second.w < kTemMinServeW)
                continue;
              const double slope = std::fabs(c.z - nit->second.z) / tem_cell;
              const double ds = slope - tem_slope_mean;
              tem_slope_mean += 0.02 * ds;
              tem_slope_var += 0.02 * (ds * ds - tem_slope_var);
            }
          }
        }
        // Per-scan extrapolation anchor: nearest served cells to the
        // vehicle (expanding ring search). Runs precisely when q == 0 too
        // — first-visit degraded terrain is where it is needed.
        tem_ext_ok = false;
        if (!tem_grid.empty()) {
          constexpr double kTemMinServeW = 2.0;
          constexpr int kTemExtMaxRing = 60;  // x tem_cell = 300 m reach
          const int64_t vx = static_cast<int64_t>(std::floor(state_point.pos(0) / tem_cell));
          const int64_t vy = static_cast<int64_t>(std::floor(state_point.pos(1) / tem_cell));
          for (int r = 0; r <= kTemExtMaxRing && !tem_ext_ok; r++) {
            double zsum = 0.0, vsum = 0.0, n = 0.0;
            for (int dx = -r; dx <= r; dx++)
              for (int dy = -r; dy <= r; dy++) {
                if (std::max(std::abs(dx), std::abs(dy)) != r)
                  continue;  // ring only
                const auto it = tem_grid.find(temKeyIdx(vx + dx, vy + dy));
                if (it == tem_grid.end() || it->second.w < kTemMinServeW)
                  continue;
                zsum += it->second.z;
                vsum += it->second.var;
                n += 1.0;
              }
            if (n < 1.0)
              continue;
            const double slope_hi = std::max(tem_slope_mean + 2.0 * std::sqrt(std::max(tem_slope_var, 0.0)), 0.003);
            const double dist = (r + 4.0) * tem_cell;  // +4 cells: row spread allowance (~20 m)
            tem_ext_z = zsum / n;
            tem_ext_sigma = std::sqrt(std::max(vsum / n, 0.0)) + slope_hi * dist;
            tem_ext_ok = true;
          }
        }
        if (canopy_debug)
          std::cerr << "[TEM] t=" << std::fixed << std::setprecision(3) << lidar_end_time << " egf=" << tem_egf_ema
                    << " q=" << q_eff << " cells=" << tem_grid.size() << " rows=" << tem_rows << "/" << tem_ext_rows
                    << "/" << tem_env_rows << " ext=" << (tem_ext_ok ? tem_ext_sigma : -1.0)
                    << " slope=" << tem_slope_mean << " d=" << (anchored ? d_use : std::nan("")) << std::endl;
      }

      if (clutter_deweight_en && canopy_debug)
        std::cerr << "[CLUT] t=" << std::fixed << std::setprecision(3) << lidar_end_time << " tested=" << clut_tested
                  << " cut=" << clut_cut << " m_avg=" << (clut_tested > 0 ? clut_msum / clut_tested : 1.0)
                  << " eigmin=" << pos_obs_eig_min << " zweak=" << pos_obs_z_weak << std::endl;

      if (vxl_en && canopy_debug)
        std::cerr << "[VXL] t=" << std::fixed << std::setprecision(3) << lidar_end_time << " cells=" << vxl_grid.size()
                  << " tested=" << vxl_tested << " cut=" << vxl_cut
                  << " m_avg=" << (vxl_tested > 0 ? vxl_msum / vxl_tested : 1.0)
                  << " mv_avg=" << (vxl_vert_cnt > 0 ? vxl_vert_msum / vxl_vert_cnt : 1.0)
                  << " sd_avg=" << (vxl_tested > 0 ? vxl_sd_sum / vxl_tested : 0.0) << " env=" << vxl_env_cnt
                  << " h_avg=" << (vxl_env_cnt > 0 ? vxl_env_habs / vxl_env_cnt : 0.0) << " eigmin=" << pos_obs_eig_min
                  << " zweak=" << pos_obs_z_weak << std::endl;

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
       *   [L3] attitude_cov_floor_deg: while triggered, floor the roll/pitch
       *        error-state variance BEFORE this update. A confidently-wrong
       *        iEKF (measured 0.09-0.67°/scan authority against a 6-7° error)
       *        beats any honestly-weighted witness; softening P is what lets
       *        the measurement act (Schmidt/consider-state practice).
       *   Constrains roll/pitch ONLY: yaw lies in the measurement null space
       *   (skew(g_body) sends a yaw-axis δθ to 0), so the LiDAR-excellent
       *   heading is untouched. h(x)=Rᵀ·u_world, right-perturbation
       *   R=R̂·Exp(δθ) ⇒ ∂h/∂δθ=skew(g_body), same form as planar/wheel. ***/
      if (gravity_align_en && !Measures.imu.empty()) {
        const double g_raw = p_imu->gravity_norm();  // gravity magnitude, raw units
        state_ikfom st = kf.get_x();
        M3D Rw = st.rot.toRotationMatrix();
        const V3D bg_now(st.bg[0], st.bg[1], st.bg[2]);
        const V3D v_body_now = Rw.transpose() * st.vel;  // velocity in IMU frame
        // [L2] per-scan aggregate of kinematically-compensated specific force
        // (centripetal ω×v_body subtracted per sample — transport theorem; the
        // scan-end velocity serves every sample, walking-speed error << gates).
        if (gravity_align_window_s > 0.0) {
          GravWinAgg agg{lidar_end_time, Eigen::Vector3d::Zero(), 0, 0.0};
          for (const auto & imu : Measures.imu) {
            V3D a_i(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
            V3D w_i(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
            // per-SAMPLE gyro filter: one spike must not disqualify the scan
            // (measured: scan-level max|ω| gating killed 18-23 % of scans).
            if (w_i.norm() > gravity_align_gyro_tol) { continue; }
            agg.sum += a_i - (w_i - bg_now).cross(v_body_now) * (g_raw / G_m_s2);
            agg.n++;
          }
          grav_win.push_back(agg);
          while (!grav_win.empty() &&
                 (grav_win.front().t < lidar_end_time - gravity_align_window_s || grav_win.size() > 100)) {
            grav_win.pop_front();
          }
        }
        // Engagement: legacy trigger, [L1] hold hysteresis, [L2] continuous.
        const bool ga_trig = pos_obs_z_weak > gravity_align_z_weak_thresh || flio_in_degraded;
        if (ga_trig) { gravity_align_last_trig = lidar_end_time; }
        const bool ga_held = gravity_align_hold_s > 0.0 && !ga_trig &&
                             lidar_end_time - gravity_align_last_trig <= gravity_align_hold_s;
        // [L3] roll/pitch covariance floor while the lidar may be twisting attitude.
        if (ga_trig && attitude_cov_floor_deg > 0.0) {
          // Multiplicative inflation (congruence P' = D·P·D with diagonal D):
          // raises the roll/pitch variance to the floor while scaling the
          // cross-covariances consistently — correlations and PSD preserved.
          // A naive diagonal bump leaves stale attitude-vel/pos cross terms
          // against an inflated variance and the next update drags the whole
          // state through them (measured 2026-08-31: 100°+ Euler jumps, 8 m
          // runaways on m20_0814). This is why the literature uses consider/
          // Schmidt treatments instead of covariance surgery.
          const double floor2 = std::pow(attitude_cov_floor_deg * M_PI / 180.0, 2);
          auto P = kf.get_P();
          bool bumped = false;
          for (int i = 3; i <= 4; i++) {  // error-state δθ x,y = roll/pitch; 5 = yaw, untouched
            if (P(i, i) > 1e-12 && P(i, i) < floor2) {
              const double lam = std::sqrt(floor2 / P(i, i));
              P.row(i) *= lam;
              P.col(i) *= lam;
              bumped = true;
            }
          }
          if (bumped) { kf.change_P(P); }
        }
        if (ga_trig || ga_held || gravity_align_continuous) {
          // Measurement vector: window mean ([L2]) or legacy last-sample.
          V3D f_meas = V3D::Zero();
          double win_span = 0.0;
          int win_n = 0;
          if (gravity_align_window_s > 0.0) {
            double t_old = lidar_end_time;
            for (const auto & a : grav_win) {
              if (a.n == 0) { continue; }  // scan fully rejected by the per-sample gyro filter
              f_meas += a.sum;
              win_n += a.n;
              t_old = std::min(t_old, a.t);
            }
            if (win_n > 0) { f_meas /= double(win_n); }
            win_span = lidar_end_time - t_old + 0.1;
          } else {
            const auto & imu_last = Measures.imu.back();
            V3D a_raw(imu_last->linear_acceleration.x, imu_last->linear_acceleration.y,
                      imu_last->linear_acceleration.z);
            V3D w_raw(imu_last->angular_velocity.x, imu_last->angular_velocity.y, imu_last->angular_velocity.z);
            if (w_raw.norm() <= gravity_align_gyro_tol) {
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
          const double g_ref = gravity_align_mag_ref > 1e-3 ? gravity_align_mag_ref : g_raw;
          const bool span_ok = gravity_align_window_s <= 0.0 ||
                               (win_span >= 0.7 * gravity_align_window_s && win_n >= 20);
          if (g_ref > 1e-3 && win_n > 0 && a_norm > 1e-6 && span_ok &&
              std::abs(a_norm - g_ref) <= gravity_align_accel_tol * g_ref) {
            V3D grav_w(st.grav[0], st.grav[1], st.grav[2]);  // ≈ [0,0,-g], points DOWN
            V3D u_world = -grav_w.normalized();              // world up (unit)
            V3D g_body = Rw.transpose() * u_world;           // predicted up in body (unit)
            V3D m_body = f_meas / a_norm;                    // measured up in body (unit)
            V3D residual = m_body - g_body;                  // z − h(x)
            if (residual.norm() <= 0.5) {  // ~30° outlier gate (never inject a wild window)
            Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
            H.block<3, 3>(0, 3) = skew_sym_mat(g_body);  // ∂h/∂(rot); yaw in null space
            // [v4] gravity-state observability: h = Rᵀ·(−g/‖g‖) also depends on the
            // S2 gravity state — without these two columns a motion-biased gravity
            // INIT is permanently unobservable (m20_0831: the bag starts walking →
            // the world frame is tilted 6.7° for the entire run and the rot-only
            // update just pulls attitude onto the tilted gravity — measured as the
            // 3-5° tug-of-war [GALIGN] tilt that never converges). With the columns
            // in, the Kalman gain splits the residual by covariance: rot follows
            // the (tight) lidar, grav follows the absolute accel reference.
            // ∂h/∂δg = −(1/‖g‖)·Rᵀ·M(g), M = S2_Mx (same helper df_dx uses).
            {
              Eigen::Matrix<state_ikfom::scalar, 2, 1> dz0 = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
              Eigen::Matrix<state_ikfom::scalar, 3, 2> Mg;
              st.S2_Mx(Mg, dz0, 21);
              const double gnorm = grav_w.norm();
              if (gnorm > 1e-6) { H.block<3, 2>(0, 21) = -(1.0 / gnorm) * Rw.transpose() * Mg; }
            }
            // [v4] fading memory on the gravity block: with no process noise the
            // grav variance collapses and even a correct H routes nothing into it.
            // Gentle multiplicative leak (congruence row/col scaling — correlations
            // and PSD preserved, no covariance shock) capped at (3°)².
            if (gravity_align_grav_leak > 1.0) {
              const double kGravVarCap = std::pow(gravity_align_grav_cap_deg * M_PI / 180.0, 2);
              auto Pg = kf.get_P();
              const double lam = std::sqrt(gravity_align_grav_leak);
              bool leaked = false;
              for (int i = 21; i <= 22; i++) {
                if (Pg(i, i) < kGravVarCap) {
                  Pg.row(i) *= lam;
                  Pg.col(i) *= lam;
                  leaked = true;
                }
              }
              if (leaked) { kf.change_P(Pg); }
            }
            const double sigma = (ga_trig || ga_held) && gravity_align_noise_degraded > 0.0
                                     ? gravity_align_noise_degraded
                                     : gravity_align_noise;
            const double n2 = sigma * sigma;
            Eigen::Vector3d R_diag(n2, n2, n2);
            if (degeneracy_debug) {
              V3D tilt = g_body.cross(m_body);  // axis*sin(angle), body frame
              const double ang_deg = std::asin(std::min(1.0, tilt.norm())) * 57.2958;
              V3D eul_before = SO3ToEuler(st.rot);
              kf.update_simple(H, residual, R_diag);
              state_point = kf.get_x();
              V3D eul_after = SO3ToEuler(state_point.rot);
              V3D deul = eul_after - eul_before;
              std::cerr << "[GALIGN] z_weak=" << pos_obs_z_weak << " degr=" << flio_in_degraded
                        << " trig=" << ga_trig << " held=" << ga_held << " win_n=" << win_n
                        << " span=" << win_span << " sigma=" << sigma << " gvar=" << kf.get_P()(21, 21)
                        << " tilt_deg=" << ang_deg
                        << " dRPY_deg=[" << deul.x() * 57.2958 << "," << deul.y() * 57.2958 << ","
                        << deul.z() * 57.2958 << "]"
                        << " posZ=" << state_point.pos[2] << "m" << std::endl;
            } else {
              kf.update_simple(H, residual, R_diag);
              state_point = kf.get_x();
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
      if (planar_constraint_en && (pos_obs_z_weak > planar_constraint_z_weak_thresh || flio_in_degraded)) {
        state_ikfom st = kf.get_x();
        M3D Rw = st.rot.toRotationMatrix();
        V3D v_body = Rw.transpose() * st.vel;  // velocity in IMU/body frame
        // Measurement: body velocity = 0, but only the vertical (z) row is
        // enforced — x/y get huge noise so forward/lateral motion is untouched.
        V3D residual = -v_body;
        Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
        H.block<3, 3>(0, 3) = skew_sym_mat(v_body);  // dh/d(rot)
        H.block<3, 3>(0, 12) = Rw.transpose();       // dh/d(vel)
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
      auto rebuild_local_map = [&](const char * tag) {
        if (async_map_en)
          joinMapAdd();
        feats_down_world->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        }
        ikdtree.Build(feats_down_world->points);
        Localmap_Initialized = false;
        fast_lio::resetReanchorGate(&reanchor_gate);
        RCLCPP_ERROR(this->get_logger(),
                     "[%s] map rebuilt from the current scan (%d pts) at pos=[%.2f %.2f %.2f]: the estimate is live "
                     "again but unanchored",
                     tag,
                     feats_down_size,
                     state_point.pos[0],
                     state_point.pos[1],
                     state_point.pos[2]);
      };

      const auto reanchor = fast_lio::updateReanchorGate(
        &reanchor_gate, flio_in_degraded, effct_feat_num, res_mean_last, lidar_end_time, reanchor_params);
      if (reanchor.just_reanchored) {
        RCLCPP_WARN(this->get_logger(),
                    "[REANCHOR] re-registered against the frozen map (effct=%d res=%.3fm) after %u blind stretch(es): "
                    "map unfrozen",
                    effct_feat_num,
                    res_mean_last,
                    reanchor_gate.reanchors);
      }
      if (reanchor.just_lost) {
        RCLCPP_ERROR(this->get_logger(),
                     "[REANCHOR] LOST: no re-registration within %.1fs (effct=%d res=%.3fm) — the pose no longer fits "
                     "the map; policy=%s",
                     reanchor_params.timeout_sec,
                     effct_feat_num,
                     res_mean_last,
                     reanchor_on_fail.c_str());
        if (reanchor_on_fail == "reset_map") {
          rebuild_local_map("REANCHOR");
        }
      }
      if (reanchor.degraded) {
        state_ikfom st = kf.get_x();
        const double spd = st.vel.norm();
        // Static hold takes precedence over the speed reset: with independent
        // proof that the platform is parked, the blind stretch has nothing to
        // integrate, and zeroing only the velocity still leaves the position and
        // attitude wherever propagation put them (dog 2, 2026-09-04: −3.9 m of Z
        // in 9 s, unrecoverable once the map unfroze around the wrong pose).
        const bool static_hold = zupt_en && zupt_anchor_valid && body_is_static(Measures, lidar_end_time);
        if (static_hold) {
          st.pos = zupt_anchor_pos;
          st.rot = zupt_anchor_rot;
          st.vel.setZero();
          kf.change_x(st);
          state_point = kf.get_x();
          ++zupt_hold_scans;
          if (!zupt_active) {
            zupt_active = true;
            RCLCPP_WARN(this->get_logger(),
                        "[ZUPT] static hold engaged (effct=%d far_frac=%.2f): pose pinned at [%.2f %.2f %.2f]",
                        effct_feat_num,
                        scan_far_frac,
                        zupt_anchor_pos[0],
                        zupt_anchor_pos[1],
                        zupt_anchor_pos[2]);
          }
        } else if (spd > divergence_guard_max_speed) {
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
          kf.change_x(st);
          state_point = kf.get_x();
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
                             reanchor_gate.state == fast_lio::ReanchorState::kVerifying   ? "verifying"
                               : reanchor_gate.state == fast_lio::ReanchorState::kLost    ? "LOST"
                               : reanchor_gate.state == fast_lio::ReanchorState::kBlind   ? "blind"
                                                                                          : "ok",
                             divergence_guard_max_speed,
                             planar_constraint_en ? "pinned" : "OFF",
                             gravity_align_en ? "on" : "OFF",
                             state_point.pos[2],
                             state_point.vel.norm());
      } else {
        // Healthy scan: the map just confirmed this pose, so it becomes the
        // anchor the next blind stretch holds on to.
        zupt_anchor_pos = state_point.pos;
        zupt_anchor_rot = state_point.rot;
        zupt_anchor_valid = true;
        if (zupt_active) {
          RCLCPP_WARN(this->get_logger(),
                      "[ZUPT] static hold released after %d scans (effct=%d far_frac=%.2f)",
                      zupt_hold_scans,
                      effct_feat_num,
                      scan_far_frac);
          zupt_active = false;
          zupt_hold_scans = 0;
        }
      }

      /*** Runaway watchdog (runaway_watchdog.hpp): the guard above keys off the
       *   scan, and the drift that matters most keeps the scan looking healthy.
       *   Compare the estimate against the only sources outside it — the wheels
       *   and the IMU's static verdict — and act when they contradict it for
       *   long enough that noise cannot explain it. ***/
      if (runaway_params.enabled) {
        double wheel_stamp = -1.0;
        double wheel_speed = 0.0;
        {
          std::lock_guard<std::mutex> lk(mtx_twist);
          wheel_stamp = last_twist_stamp;
          wheel_speed = last_twist_speed;
        }
        const double wheel_age = wheel_stamp > 0.0 ? std::fabs(lidar_end_time - wheel_stamp) : -1.0;
        const auto runaway = fast_lio::updateRunawayWatchdog(&runaway_state,
                                                             lidar_end_time,
                                                             state_point.vel.norm(),
                                                             body_is_static(Measures, lidar_end_time),
                                                             wheel_age,
                                                             wheel_speed,
                                                             runaway_params);
        if (runaway.tripped) {
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
                       runaway_on_trip.c_str());
          if (runaway_on_trip == "reset_map") {
            rebuild_local_map("RUNAWAY");
          } else if (runaway_on_trip == "fault_exit") {
            RCLCPP_FATAL(this->get_logger(),
                         "[RUNAWAY] exiting so the supervisor can restart the pipeline (runaway_on_trip=fault_exit)");
            rclcpp::shutdown();
          } else {
            // hold: freeze the map and let the health flags stop downstream from
            // consuming a pose we no longer believe.
            fast_lio::tripReanchorLost(&reanchor_gate);
          }
        }
      }

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
            const double dt = get_time_sec(Measures.imu[k]->header.stamp) - get_time_sec(Measures.imu[k - 1]->header.stamp);
            if (dt > imu_gap_max) imu_gap_max = dt;
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
              obs_along = d.dot(pos_obs_M * d) / n;
            }
            obs_emin = pos_obs_eig_min / n;
            obs_emax = pos_obs_eig_max / n;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es2(pos_obs_M.topLeftCorner<2, 2>());
            obs_hmin = es2.eigenvalues()(0) / n;
          }
          std::cerr << std::fixed << std::setprecision(3) << "[SCAN] t=" << lidar_end_time << " imu_n=" << Measures.imu.size()
                    << " imu_gap=" << imu_gap_max << " pts=" << feats_down_size << " effct=" << effct_feat_num
                    << " far=" << scan_far_frac << " speed=" << body_speed << " dvel=" << dvel
                    << " corr=" << lidar_correction << " posZ=" << state_point.pos[2]
                    << " deg=" << (flio_in_degraded ? 1 : 0)
                    << " along=" << obs_along << " emin=" << obs_emin << " emax=" << obs_emax << " hmin=" << obs_hmin
                    << " drot=" << drot_deg << " dyaw=" << dyaw_deg << " bgz=" << state_point.bg[2]
                    << " exT=" << state_point.offset_T_L_I[0] << "," << state_point.offset_T_L_I[1] << ","
                    << state_point.offset_T_L_I[2] << " exR=" << ex_eul[0] << "," << ex_eul[1] << "," << ex_eul[2]
                    << " res=";
          for (int b = 0; b < kResBins; ++b)
            std::cerr << (b ? "," : "") << res_bin_n[b] << ":"
                      << (res_bin_n[b] ? std::sqrt(res_bin_ss[b] / res_bin_n[b]) * 1000.0 : 0.0);
          std::cerr << std::endl;
        }
        if (lidar_correction > 0.2 || body_speed > 1.5 || dvel > 0.3 || dba > 0.05 || dbg > 0.01 ||
            effct_feat_num < 200 || pos_obs_z_weak > 0.7) {
          std::cerr << "[DEGEN] effct=" << effct_feat_num << " corr=" << lidar_correction << "m speed=" << body_speed
                    << " dvel=" << dvel << " dba=" << dba << " dbg=" << dbg << " z_weak=" << pos_obs_z_weak
                    << " posZ=" << state_point.pos[2] << "m"
                    << " rec=" << canopy_ground_recovered << std::endl;
        }
      }

      /*** Wheel velocity update (after ICP, constrains along-track drift) ***/
      if (wheel_odom_en) {
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
        if (have_twist) {
          state_ikfom st = kf.get_x();
          M3D Rw = st.rot.toRotationMatrix();     // R_{world←IMU}
          M3D Re_T = R_robot_to_imu.transpose();  // R_{robot←IMU}
          V3D v_imu = Rw.transpose() * st.vel;    // vel in IMU frame

          // Predicted: h(x) = R_ext^T * R_w^T * vel (robot frame)
          V3D h_pred = Re_T * v_imu;
          // Measured: z = [vx*scale, 0, 0] (non-holonomic: vy=vz=0)
          V3D z_meas(twist_v(0) * wheel_speed_scale, 0.0, 0.0);
          V3D residual = z_meas - h_pred;

          // Jacobian H (3×23): non-zero at rot(3-5) and vel(12-14)
          Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();
          H.block<3, 3>(0, 3) = Re_T * skew_sym_mat(v_imu);  // dh/d(rot)
          H.block<3, 3>(0, 12) = Re_T * Rw.transpose();      // dh/d(vel)

          // Diagonal noise variances
          Eigen::Vector3d R_diag(wheel_vel_noise_vx * wheel_vel_noise_vx,
                                 wheel_vel_noise_vy * wheel_vel_noise_vy,
                                 wheel_vel_noise_vz * wheel_vel_noise_vz);

          EkfUpdateDiagnostics wheelDiagnostics;
          Eigen::Matrix<double, 23, 23> wheelCovarianceBefore = Eigen::Matrix<double, 23, 23>::Zero();
          if (kalman_channel_diag_en)
            wheelCovarianceBefore = kf.get_P();
          kf.update_simple(H, residual, R_diag, kalman_channel_diag_en ? &wheelDiagnostics : nullptr);
          state_point = kf.get_x();  // re-read corrected state
          if (kalman_channel_diag_en) {
            writeKalmanChannel("wheel", st, state_point, wheelCovarianceBefore, wheelDiagnostics, z_meas, h_pred);
          }
        }
      }

      euler_cur = SO3ToEuler(state_point.rot);
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
      geoQuat.x = state_point.rot.coeffs()[0];
      geoQuat.y = state_point.rot.coeffs()[1];
      geoQuat.z = state_point.rot.coeffs()[2];
      geoQuat.w = state_point.rot.coeffs()[3];

      double t_update_end = omp_get_wtime();

      /*** Scan-hole guard bookkeeping: this scan's posterior is the next hole's anchor ***/
      hole_last_scan_end = lidar_end_time;
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
            scan_obs_along = d.dot(pos_obs_M * d) / effct_feat_num;
          } else {
            scan_obs_along = 0.0;  // moving with no effective points: worst case
          }
        }
      }

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
        // Same gate as map growth: a frozen/paused map means "don't
        // learn from this state" — that applies to the voxel surface
        // statistics as much as to the ikd-tree.
        vxl_update();
      }
      t5 = omp_get_wtime();

      // 自适应降采样:t5-t0 是这一帧的计算耗时(不含排队 —— 排队归 max_scan_backlog_sec 管)。
      // 超预算就把扫描体素调粗:点少了、精度略降,但成本回到扫描周期以内。
      // Per-scan compute (queue wait excluded) drives the voxel: over budget costs points,
      // not latency.
      if (adaptive_ds_params.enabled) {
        const double scan_cost = t5 - t0;
        if (fast_lio::updateAdaptiveDownsample(
              &adaptive_ds_state, scan_cost, filter_size_surf_min, adaptive_ds_params)) {
          const float leaf = static_cast<float>(adaptive_ds_state.voxel);
          downSizeFilterSurf.setLeafSize(leaf, leaf, leaf);
          RCLCPP_WARN(this->get_logger(),
                      "[ADAPTIVE-DS] scan voxel %.2f m (cost %.0f ms vs budget %.0f ms; %u up / %u down)",
                      adaptive_ds_state.voxel,
                      scan_cost * 1e3,
                      adaptive_ds_params.budget_sec * 1e3,
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
          "iekf_ms=%.1f add_ms=%.1f join_ms=%.1f loop_ms=%.1f | "
          "rb_dcnt=%llu rb_dus_ms=%.1f inl_dus_ms=%.1f rb_active=%d "
          "rb_maxus_ms=%.1f rb_maxsz=%llu rootreb=%llu\n",
          kdtree_size_st,
          feats_down_size,
          (t_undist - t0) * 1000.0,
          (t_fov - t_join) * 1000.0,
          (t1 - t_fov) * 1000.0,
          iekf_ms,
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
      if (path_en)
        publish_path(pubPath_);
      if (scan_pub_en)
        publish_frame_world(pubLaserCloudFull_);
      if (scan_pub_en && scan_body_pub_en)
        publish_frame_body(pubLaserCloudFull_body_);
      if (effect_pub_en)
        publish_effect_world(pubLaserCloudEffect_);
      // if (map_pub_en) publish_map(pubLaserCloudMap_);

      /*** Debug variables ***/
      if (runtime_pos_log) {
        frame_num++;
        // Under async the worker may be mid-Add; reading ikdtree.size() here would
        // race it. Estimate from the scan-start size + this scan's add count.
        kdtree_size_end = async_map_en ? (kdtree_size_st + add_point_size) : ikdtree.size();
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
        aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
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
      if (async_map_en)
        joinMapAdd();
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
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImu>::SharedPtr sub_imu_fpa_;
  rclcpp::Subscription<fixposition_driver_msgs::msg::FpaImubias>::SharedPtr sub_imu_bias_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
  rclcpp::Subscription<shm_msgs::msg::PointCloud8mAndPose>::SharedPtr sub_pcl_shm_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool offline_step_en_ = false;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr subOfflineStep_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fe_z_drift_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mapping_enabled_sub_;

  bool effect_pub_en = false, map_pub_en = false;
  int effect_feat_num = 0, frame_num = 0;
  double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
                         aver_time_solve = 0, aver_time_const_H_time = 0;
  bool flg_EKF_converged, EKF_stop_flg = 0;
  double epsi[23] = {0.001};

  FILE * fp;
  ofstream fout_pre, fout_out, fout_dbg, fout_jump, fout_kalman;

  std::string prior_map_pcd_;
  std::vector<double> initial_pose_vec_;
  bool initial_pose_full_rpy_override_ = false;
};

#ifndef FASTLIO_AS_COMPONENT
int main(int argc, char ** argv)
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
  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    string file_name = string("scans.pcd");
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    pcl::PCDWriter pcd_writer;
    cout << "current scan saved to /PCD/" << file_name << endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
  }

  if (runtime_pos_log) {
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    FILE * fp2;
    string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(fp2,
            "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree "
            "size st, tree size end, add point size, preprocess time\n");
    for (int i = 0; i < time_log_counter; i++) {
      fprintf(fp2,
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

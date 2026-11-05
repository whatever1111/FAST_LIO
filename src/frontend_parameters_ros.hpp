#pragma once

#include <rclcpp/node.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "correspondence_gate.hpp"
#include "degeneracy_policy.hpp"
#include "engulf_release.hpp"
#include "frontend_params.hpp"
#include "imu_coverage_policy.hpp"
#include "imu_gap_prior.hpp"
#include "imu_static_init_gate.hpp"
#include "lost_recovery_policy.hpp"
#include "reanchor_gate.hpp"
#include "stationary_anchor_policy.hpp"
#ifdef FASTLIO_TEST_HOOKS
#include "laser_mapping_test_hooks.hpp"
#endif

namespace fast_lio
{
/// Values one parameter block of the constructor reads and a later startup step applies. It lives only as long as
/// the constructor; nothing after construction reads it.
struct FrontendStartup
{
  int num_max_iterations = 0;
  bool imu_init_require_still = false;  // quasi-static IMU-init gate (see IMU_Processing.hpp)
  double imu_init_still_tol = 0.03;
  double imu_init_still_timeout_s = 20.0;
  fast_lio::ImuStaticInitParams imu_init_static_params;
  double zupt_anchor_max_scan_gap_s = 0.0;
  int reanchor_healthy_window = 100;
  double reanchor_total_timeout_sec = 0.0;
  std::string lid_topic, imu_topic, imu_input_type = "sensor_msgs", imu_bias_topic = "/fixposition/fpa/imubias";
  double fov_deg = 0;
  double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
  std::vector<double> extrinT = std::vector<double>(3, 0.0);
  std::vector<double> extrinR = std::vector<double>(9, 0.0);
  std::string wheel_topic = "/lio/twist";
};

/// The four iVox parameters as declared; the node validates nothing more and initialises its map backend with them.
struct IvoxDeclaredValues
{
  double grid_resolution = 0.0;
  int nearby_type = 0;
  int voxel_capacity = 0;
  int max_voxels = 0;
};

/// The front end's parameters: it declares and reads them, converts and validates them, in the constructor's
/// original order, and afterwards hands the result out read-only through values(). Every call takes the node and the
/// storage that still lives elsewhere as explicit references for that call only; it keeps neither.
class FrontendParameters
{
public:
  const FrontendParams & values() const { return values_; }

  // Declarations: the front block (ending with the LOST recovery pair), the typed-output activation with its checks,
  // then the back block. The iVox declarations come later, among the reads.
  void declareFront(rclcpp::Node & node);
  void declareActivation(rclcpp::Node & node,
                         bool & observation_en,
                         bool & reference_health_en,
                         std::string & clock_domain,
                         std::int64_t & idle_period_ns);
  void declareBack(rclcpp::Node & node);

  // Reads with their conversions and checks, in the constructor's order.
  void readOutputs(const rclcpp::Node & node,
                   FrontendStartup & startup,
                   bool & scan_pub_en,
                   bool & dense_pub_en,
                   std::string & odom_frame_id,
                   std::string & body_frame_id);
  void readBacklog(const rclcpp::Node & node,
                   int & max_buffered_scans,
                   double & max_scan_backlog_sec,
                   double & max_imu_backlog_sec);
  void readDownsampleAndDegeneracy(const rclcpp::Node & node,
                                   bool & degeneracy_debug,
                                   bool & degeneracy_spectrum_trace,
                                   DegeneracyParams & degeneracy_params);
  void readConstraints(const rclcpp::Node & node, bool & health_pub_en);
  void readCorrespondence(const rclcpp::Node & node,
                          CorrespondenceGateParams & corr_gate,
                          int & corr_kernel,
                          double & corr_kernel_mad_scale,
                          double & corr_kernel_min_scale);
  void readCorrespondenceDump(const rclcpp::Node & node);
  void readAttitudeAndGravity(const rclcpp::Node & node,
                              double & lidar_attitude_hold_min_rp_info,
                              double & zupt_vel_reset_sigma);
  void readImuInit(const rclcpp::Node & node, FrontendStartup & startup);
  void readGuards(const rclcpp::Node & node, ReleaseVelocityParams & release_vel_params);
  void readHoleAndZupt(const rclcpp::Node & node,
                       FrontendStartup & startup,
                       double & zupt_gyro_thresh,
                       double & zupt_acc_span_ratio,
                       double & zupt_wheel_speed_thresh,
                       double & zupt_wheel_max_age,
                       bool & zupt_require_wheel,
                       bool & zupt_anchor_trace_en,
                       StationaryAnchorParams & stationary_anchor_params);
  void readReanchor(const rclcpp::Node & node,
                    FrontendStartup & startup,
                    ReanchorParams & reanchor_params,
                    RecoveryParams & recovery_params);
  void loadLostRecovery(const rclcpp::Node & node,
                        bool health_pub_en,
                        double total_s,
                        LostRecoveryParams & lost_recovery_params);
  void loadRecoveryEntryView(const rclcpp::Node & node);
  void readRunaway(const rclcpp::Node & node, double zupt_wheel_max_age, double zupt_wheel_speed_thresh);
  void readCanopyAndGround(const rclcpp::Node & node,
                           double & canopy_near_range,
                           double & canopy_ground_z,
                           double & canopy_ground_halfband,
                           bool & ground_recover_en,
                           double & ground_recover_scan_band,
                           int & ground_recover_min_inliers,
                           double & ground_recover_max_residual,
                           double & ground_recover_sector_tol,
                           int & ground_recover_max_rows);
  void readDataPaths(const rclcpp::Node & node, std::string & map_file_path, std::string & root_dir);
  void readInputs(const rclcpp::Node & node,
                  FrontendStartup & startup,
                  bool & time_sync_en,
                  double & time_diff_lidar_to_imu,
                  double & imu_time_offset);
  void readMapAndNoise(const rclcpp::Node & node,
                       FrontendStartup & startup,
                       double & filter_size_map_min,
                       double & cube_len,
                       float & det_range);
  void readPreprocess(
    const rclcpp::Node & node, double & blind, int & lidar_type, int & scan_lines, int & time_unit, int & scan_rate);
  void resolveAnchorGap(const rclcpp::Node & node,
                        const FrontendStartup & startup,
                        int scan_rate,
                        StationaryAnchorParams & stationary_anchor_params,
                        const RecoveryParams & recovery_params);
  void readImuCoverage(const rclcpp::Node & node,
                       double & max_scan_duration_s,
                       ImuCoverageParams & coverage,
                       ImuGapPriorParams & gap,
                       bool reanchor_en);
  void readFeatureInputs(const rclcpp::Node & node,
                         int & point_filter_num,
                         bool & nn_refresh_gate_en,
                         double & nn_refresh_min_shift,
                         double & nn_refresh_ref_range,
                         bool & feature_enabled,
                         bool & runtime_pos_log);
#ifdef USE_IVOX
  IvoxDeclaredValues declareIvox(rclcpp::Node & node);
#endif
  void readDiagnosticsAndMapSave(const rclcpp::Node & node,
                                 FrontendStartup & startup,
                                 bool & latency_diag_en,
                                 double & latency_log_period_sec,
                                 bool & extrinsic_est_en,
                                 int & map_insert_delay_scans,
                                 bool & pcd_save_en,
                                 int & pcd_save_interval);
  void readPriorMapPath(const rclcpp::Node & node, std::string & prior_map_pcd);
  void readPriorMapAndPose(const rclcpp::Node & node);
  void readWheel(const rclcpp::Node & node, FrontendStartup & startup);
  double readMapPeriod(const rclcpp::Node & node);

#ifdef FASTLIO_TEST_HOOKS
  /// The constructor-local values this object read, under the names the effective-configuration fixture records.
  const fast_lio_test::EffectiveRecorder & localsForTest() const { return locals_for_test_; }
#endif

private:
  void declareOutputsAndMeasurement(rclcpp::Node & node);
  void declareGravityToConsider(rclcpp::Node & node);
  void declareHoleToLostRecovery(rclcpp::Node & node);
  void declareRunawayToPreprocess(rclcpp::Node & node);
  void declareImuGapToWheel(rclcpp::Node & node);

  FrontendParams values_;
#ifdef FASTLIO_TEST_HOOKS
  fast_lio_test::EffectiveRecorder locals_for_test_;
#endif
};
}  // namespace fast_lio

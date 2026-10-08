#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Eigen>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <common_lib.h>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <math.h>
#include <mutex>
#include <so3_math.h>
#include <thread>

#include "imu_consumption_policy.hpp"
#include "imu_coverage_policy.hpp"
#include "imu_gap_prior.hpp"
#include "imu_initialization_validity.hpp"
#include "imu_process_outcome.hpp"
#include "population_moments.hpp"
#include "scan_history_policy.hpp"
#include "scan_time_policy.hpp"
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

inline bool validImuForConsumption(const sensor_msgs::msg::Imu::ConstSharedPtr & imu)
{
  return imu && imu->header.stamp.sec >= 0 && imu->header.stamp.nanosec < 1000000000u &&
         std::isfinite(imu->linear_acceleration.x) && std::isfinite(imu->linear_acceleration.y) &&
         std::isfinite(imu->linear_acceleration.z) && std::isfinite(imu->angular_velocity.x) &&
         std::isfinite(imu->angular_velocity.y) && std::isfinite(imu->angular_velocity.z);
}

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  bool Process(const MeasureGroup & meas,
               esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state,
               PointCloudXYZI::Ptr pcl_un_);

  enum class ProcessStatus
  {
    kRejected,
    kInitializing,
    kProcessed,
    kCoverageGap
  };
  ProcessStatus lastStatus() const { return last_status_; }
  const fast_lio::ImuProcessOutcome & lastOutcome() const { return last_outcome_; }
  const fast_lio::ImuIntegrationLedger & integrationLedger() const { return integration_ledger_; }
  std::uint64_t staleOutputReuseCount() const { return stale_output_reuse_count_; }
  const V3D & lastDeskewAcceleration() const { return acc_s_last; }
  const V3D & lastDeskewAngularVelocity() const { return angvel_last; }
  fast_lio::ImuCoverageParams coverage_params;
  fast_lio::ImuCoverageResult coverage_result;
  // IMU-gap prior (imu_gap_prior.hpp): intervals longer than gap_params.min_gap_s inside the coverage limit are
  // bridged with the gyro average, a held velocity and an inflated P. gap_summary says what the last Process() call
  // bridged (empty unless it returned true).
  fast_lio::ImuGapPriorParams gap_params;
  fast_lio::ImuGapSummary gap_summary;

  double lastProcessedEnd() const { return scan_consumption_.lastEnd(); }
  /// Whether the IMU-gap prior carries a scan with no IMU sample at all (gap_params.empty_scans): the filter is
  /// initialised, and the IMU silence up to the scan end is inside the coverage limit.
  bool bridgesEmptyScan(double lidar_end_time) const
  {
    if (!gap_params.enabled || !gap_params.empty_scans || imu_need_init_ || !last_imu_)
      return false;
    const double last_stamp = rclcpp::Time(last_imu_->header.stamp).seconds();
    const double silence = lidar_end_time - last_stamp;
    return last_stamp > 0.0 && std::isfinite(silence) && silence > 0.0 &&
           (!(coverage_params.max_gap_s > 0.0) || silence <= coverage_params.max_gap_s);
  }

  ofstream fout_imu;
  // When true, dump per-sample IMU (bias-corrected, as fed to the filter) to
  // Log/imu.txt: "t angvel(xyz) acc(xyz)". Mirrors laserMapping runtime_pos_log
  // so the raw IMU stream is available for offline drift/gravity analysis. Set by
  // laserMapping from the runtime_pos_log_enable param before the first Process().
  bool runtime_log_en = false;
  // Quasi-static init gate (2026-08-31, m20_0831): initializing gravity from a
  // WALKING accel mean tilted the world frame 6.5-7 deg for the entire run —
  // every later observation is frame-relative, nothing can re-level the map
  // post hoc (measured: plane-fit tilt 6.5 deg, residual Z RMS 4 cm). When
  // enabled, the init accumulator restarts until the accel-norm scatter says
  // the platform is still; after init_still_timeout_s it proceeds with a WARN.
  bool   init_require_still = false;
  double init_still_tol = 0.03;        // sqrt(tr(cov_acc))/|mean_acc| threshold
  double init_still_timeout_s = 20.0;  // fall back to legacy init after this
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

  // Magnitude of gravity in the raw IMU-message accel units (||mean_acc|| frozen
  // after init). Used by the gravity-alignment leveling prior in laserMapping to
  // gate "linear acceleration ≈ 0" (|‖a_raw‖ − g_raw| small) in a scale-agnostic
  // way, regardless of whether the IMU reports in m/s² or g.
  double gravity_norm() const { return mean_acc.norm(); }

 private:
  // Discard only the invalid initialization window. In particular, keep the
  // consumed-scan guard and the initialization attempt clock.
  void discardInitializationWindow()
  {
    b_first_frame_ = true;
    init_iter_num = 0;
    mean_acc.setZero();
    mean_gyr.setZero();
    cov_acc.setZero();
    cov_gyr.setZero();
  }
  fast_lio::ImuProcessOutcome
  IMU_init(const MeasureGroup & meas, esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state, std::size_t & N);
  void UndistortPcl(const MeasureGroup & meas,
                    esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state,
                    PointCloudXYZI & pcl_in_out,
                    const PointCloudXYZI & working_cloud);

  PointCloudXYZI::Ptr cur_pcl_un_;
  // sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double initialization_attempt_start_ = 0.0;
  bool has_initialization_attempt_start_ = false;
  double last_lidar_end_time_ = -1.0;
  fast_lio::ScanConsumption scan_consumption_;
  ProcessStatus last_status_ = ProcessStatus::kRejected;
  fast_lio::ImuProcessOutcome last_outcome_;
  fast_lio::ImuIntegrationLedger integration_ledger_;
  std::uint64_t stale_output_reuse_count_ = 0;
  std::size_t init_iter_num = 0;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 0;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  acc_s_last      = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_ = std::make_shared<sensor_msgs::msg::Imu>();
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  initialization_attempt_start_ = 0.0;
  has_initialization_attempt_start_ = false;
  init_iter_num     = 0;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_ = std::make_shared<sensor_msgs::msg::Imu>();
  cur_pcl_un_ = PointCloudXYZI{}.makeShared();
  last_lidar_end_time_ = -1.0;
  scan_consumption_.reset();
  last_status_ = ProcessStatus::kRejected;
  last_outcome_ = {};
  integration_ledger_ = {};
  stale_output_reuse_count_ = 0;
  b_first_frame_ = true;
  acc_s_last = Zero3d;
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

fast_lio::ImuProcessOutcome ImuProcess::IMU_init(const MeasureGroup & meas,
                                                 esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state,
                                                 std::size_t & N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    // A new statistics window is not a new startup attempt. In particular,
    // retain the timeout origin and the consumed-scan cursor across retries.
    discardInitializationWindow();
    N = 0;
    b_first_frame_ = false;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    ++N; // inclusive sample count for both independent vector statistics
    fast_lio::updatePopulationMoments(cur_acc, N, mean_acc, cov_acc);
    fast_lio::updatePopulationMoments(cur_gyr, N, mean_gyr, cov_gyr);
  }
  // Finite components alone do not make a usable gravity direction. Reject
  // zero/cancelling/tiny means before quaternion construction or normalization.
  if (!fast_lio::validInitializationMean(mean_acc, mean_gyr) || !cov_acc.allFinite() || !cov_gyr.allFinite() ||
      !std::isfinite(cov_acc.sum()) || !std::isfinite(cov_gyr.sum()))
    return {fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kInitializationInvalidMean};
  state_ikfom init_state = kf_state.get_x();

  // Gravity alignment: rotate world frame so Z-axis aligns with gravity.
  // This prevents gravity-accelerometer bias coupling instability when
  // IMU is mounted at a tilt angle relative to the horizon.
  Eigen::Quaterniond gravity_align = Eigen::Quaterniond::FromTwoVectors(
      mean_acc, Eigen::Vector3d::UnitZ());
  V3D aligned_acc = gravity_align * mean_acc;

  init_state.rot  = SO3(gravity_align);
  init_state.grav = S2(- aligned_acc / aligned_acc.norm() * G_m_s2);
  // get_f subtracts bg directly from the incoming IMU/body-frame gyro.
  // Gravity alignment changes the world orientation, not the gyro input frame.
  init_state.bg   = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  if (!fast_lio::finiteInitializationState(init_state) || !init_P.allFinite())
    return {fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kInitializationInvalidState};
  kf_state.change_x(init_state);
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
  return {fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kInitializationAccumulated};
}

void ImuProcess::UndistortPcl(const MeasureGroup & meas,
                              esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state,
                              PointCloudXYZI & pcl_out,
                              const PointCloudXYZI & working_cloud)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = working_cloud;
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  in.gyro = angvel_last + imu_state.bg;
  in.acc = imu_state.rot.inverse() * (acc_s_last - imu_state.grav.get_vect()) + imu_state.ba;

  // IMU-gap prior (imu_gap_prior.hpp). Across an interval longer than gap_params.min_gap_s the state is rotated with
  // the bracketing gyro average, its world-frame velocity is held, and P is inflated for the whole gap once it ends.
  // With the prior off (the default) no branch below differs from the upstream integration.
  double prop_time = last_lidar_end_time_;  // the state is propagated up to here
  double gap_run = 0.0;                     // length of the gap being bridged; inflated when it ends
  double gap_interval = 0.0;                // the IMU interval that gap belongs to (it may start in an earlier scan)
  const auto gap_input = [&kf_state](const V3D & gyro) {
    const state_ikfom s = kf_state.get_x();
    input_ikfom gin;
    gin.gyro = gyro;
    gin.acc = fast_lio::zeroAccelerationInput(s.rot.toRotationMatrix(), s.ba, s.grav.get_vect());
    return gin;
  };
  const auto close_gap = [&]() {
    if (gap_run <= 0.0)
      return;
    const state_ikfom s = kf_state.get_x();
    const V3D up = -s.grav.get_vect();
    const fast_lio::ImuGapInflation inflation =
      fast_lio::imuGapInflation(gap_run, up, s.rot.toRotationMatrix(), gap_params);
    auto P = kf_state.get_P();
    if (fast_lio::applyImuGapInflation(P, inflation))
      kf_state.change_P(P);
    fast_lio::recordImuGap(&gap_summary, inflation, gap_interval);
    gap_run = 0.0;
    gap_interval = 0.0;
  };
  // A gap that covers the start of this scan (the scans in between had an empty IMU window and were dropped): carry
  // the state to the scan start first, so that the first deskew pose is where the platform was at pcl_beg_time and
  // not where it was when the last processed scan ended (0826: 0.9 s and 1.1 m earlier).
  if (gap_params.enabled && prop_time > 0.0 && prop_time < pcl_beg_time)
  {
    for (size_t i = 1; i < v_imu.size(); ++i)
    {
      const double first_stamp = rclcpp::Time(v_imu[i]->header.stamp).seconds();
      if (first_stamp < prop_time)
        continue;
      const double before_stamp = rclcpp::Time(v_imu[i - 1]->header.stamp).seconds();
      if (first_stamp > pcl_beg_time && fast_lio::isImuGap(first_stamp - before_stamp, gap_params))
      {
        const auto & w0 = v_imu[i - 1]->angular_velocity;
        const auto & w1 = v_imu[i]->angular_velocity;
        const V3D gyro_avr(0.5 * (w0.x + w1.x), 0.5 * (w0.y + w1.y), 0.5 * (w0.z + w1.z));
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
        in = gap_input(gyro_avr);
        double dt_start = pcl_beg_time - prop_time;
        integration_ledger_.observe(prop_time, pcl_beg_time);
        kf_state.predict(dt_start, Q, in);
        gap_run += dt_start;
        gap_interval = std::max(gap_interval, first_stamp - before_stamp);
        prop_time = pcl_beg_time;
        imu_state = kf_state.get_x();
        angvel_last = gyro_avr - imu_state.bg;
        acc_s_last = Zero3d;
      }
      break;
    }
  }
  IMUpose.push_back(set_pose6d(prop_time - pcl_beg_time, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp <= prop_time)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    if (runtime_log_en && fout_imu.is_open()) {
      fout_imu << setw(10) << rclcpp::Time(head->header.stamp).seconds() - first_lidar_time
               << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    }

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head_stamp < prop_time)
    {
      dt = tail_stamp - prop_time;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail_stamp - head_stamp;
    }
    
    const bool gap = fast_lio::isImuGap(tail_stamp - head_stamp, gap_params);
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    if (gap)
    {
      // Gyro average as upstream; the bracketing accelerometer mean is NOT integrated over the gap.
      in = gap_input(angvel_avr);
      integration_ledger_.observe(std::max(prop_time, head_stamp), tail_stamp);
      kf_state.predict(dt, Q, in);
      gap_run += std::max(dt, 0.0);
      gap_interval = std::max(gap_interval, tail_stamp - head_stamp);
    }
    else
    {
      close_gap();
      in.acc = acc_avr;
      in.gyro = angvel_avr;
      integration_ledger_.observe(std::max(prop_time, head_stamp), tail_stamp);
      kf_state.predict(dt, Q, in);
    }

    prop_time = tail_stamp;
    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    if (gap)
    {
      acc_s_last = Zero3d;  // the held velocity: no world acceleration inside the gap, for the deskew as well
    }
    else
    {
      acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
      for(int i=0; i<3; i++)
      {
        acc_s_last[i] += imu_state.grav[i];
      }
    }
    double &&offs_t = tail_stamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // A short advancing scan can contain only duplicate old IMU samples.
  // They were already integrated up to last_lidar_end_time_; never integrate
  // that old interval a second time while extrapolating to this scan's end.
  dt = pcl_end_time - prop_time;
  if (fast_lio::isImuGap(dt, gap_params))
  {
    // The IMU stops before the scan ends: hold the velocity to the end as well, carry on only the rotation about
    // gravity (imu_gap_prior.hpp: rotationAboutGravityInput), and give the points past the last sample a deskew pose
    // that moves the way the state did.
    const state_ikfom s_end = kf_state.get_x();
    const V3D gyro_last = fast_lio::rotationAboutGravityInput(
      V3D(in.gyro), s_end.bg, s_end.rot.toRotationMatrix(), s_end.grav.get_vect());
    in = gap_input(gyro_last);
    integration_ledger_.observe(prop_time, pcl_end_time);
    kf_state.predict(dt, Q, in);
    gap_run += dt;
    gap_interval = std::max(gap_interval, pcl_end_time - imu_end_time);
    close_gap();
    imu_state = kf_state.get_x();
    angvel_last = gyro_last - imu_state.bg;
    acc_s_last = Zero3d;
    IMUpose.push_back(set_pose6d(pcl_end_time - pcl_beg_time, acc_s_last, angvel_last, imu_state.vel, imu_state.pos,
                                 imu_state.rot.toRotationMatrix()));
  }
  else
  {
    close_gap();
    integration_ledger_.observe(prop_time, pcl_end_time);
    kf_state.predict(dt, Q, in);
    if (IMUpose.size() == 1) {
      // No new IMU knot: the actual tail prediction still defines one deskew segment.
      const state_ikfom tail_state = kf_state.get_x();
      angvel_last = in.gyro - tail_state.bg;
      acc_s_last = tail_state.rot * (in.acc - tail_state.ba) + tail_state.grav.get_vect();
      IMUpose.push_back(set_pose6d(pcl_end_time - pcl_beg_time, acc_s_last, angvel_last, tail_state.vel,
                                   tail_state.pos, tail_state.rot.toRotationMatrix()));
    }
  }

  imu_state = kf_state.get_x();
  if (!meas.imu.empty())
    last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  // Deskew every retained point into the frame-end pose. Internal knots belong
  // to the preceding segment (head < t <= tail); only the earliest head is
  // included. The final segment retains unbounded extrapolation past its tail.
  const int n_pts = static_cast<int>(pcl_out.points.size());
  if (n_pts == 0) return;
  const int n_seg = static_cast<int>(IMUpose.size()) - 1;
  if (n_seg < 1) return;
  std::vector<double> seg_head_t(IMUpose.size());
  for (size_t j = 0; j < IMUpose.size(); j++) seg_head_t[j] = IMUpose[j].offset_time;
  std::vector<M3D, Eigen::aligned_allocator<M3D>> seg_R(n_seg);
  std::vector<V3D, Eigen::aligned_allocator<V3D>> seg_vel(n_seg), seg_pos(n_seg), seg_acc(n_seg), seg_gyr(n_seg);
  for (int j = 0; j < n_seg; j++)
  {
    const auto &head = IMUpose[j];
    const auto &tail = IMUpose[j + 1];
    seg_R[j] << MAT_FROM_ARRAY(head.rot);
    seg_vel[j] << VEC_FROM_ARRAY(head.vel);
    seg_pos[j] << VEC_FROM_ARRAY(head.pos);
    seg_acc[j] << VEC_FROM_ARRAY(tail.acc);
    seg_gyr[j] << VEC_FROM_ARRAY(tail.gyr);
  }
  // Knots carry their actual propagation times, including negative offsets
  // before scan begin. The backward sweep preserves internal boundary ownership.
  // Per-segment affine folding. The compensation below is affine in the point and every
  // factor except the intra-segment rotation is constant over a scan, and Rodrigues on a
  // fixed axis is exact as R(dt) = R0 + sin(w dt) R1 + (1-cos(w dt)) R2. Precomputing the
  // three matrices per segment turns the per-point work from an Exp() plus four rotations
  // into three mat-vecs, which is what the scan-rate budget is actually spent on.
  const M3D R_li = imu_state.offset_R_L_I.toRotationMatrix();
  const V3D t_li = imu_state.offset_T_L_I;
  const M3D M_fold = R_li.transpose() * imu_state.rot.conjugate().toRotationMatrix();
  const V3D c_li = R_li.transpose() * t_li;
  std::vector<M3D, Eigen::aligned_allocator<M3D>> segA0(n_seg), segA1(n_seg), segA2(n_seg);
  std::vector<V3D, Eigen::aligned_allocator<V3D>> segd0(n_seg), segd1(n_seg), segd2(n_seg),
    segU(n_seg), segV(n_seg), segAcc(n_seg);
  std::vector<double> seg_w(n_seg);
  for (int j = 0; j < n_seg; j++)
  {
    const double w = seg_gyr[j].norm();
    seg_w[j] = w;
    M3D K = M3D::Zero();
    if (w > 1e-7)
    {
      const V3D axis = seg_gyr[j] / w;
      K << SKEW_SYM_MATRX(axis);
    }
    const M3D MR0 = M_fold * seg_R[j];
    const M3D MR1 = MR0 * K;
    const M3D MR2 = MR1 * K;
    segA0[j] = MR0 * R_li;
    segA1[j] = MR1 * R_li;
    segA2[j] = MR2 * R_li;
    segd0[j] = MR0 * t_li - c_li;
    segd1[j] = MR1 * t_li;
    segd2[j] = MR2 * t_li;
    segU[j] = M_fold * (seg_pos[j] - imu_state.pos);
    segV[j] = M_fold * seg_vel[j];
    segAcc[j] = 0.5 * (M_fold * seg_acc[j]);
  }

  std::vector<int> seg_of(n_pts);
  {
    int p = n_pts - 1;
    for (int j = n_seg; j >= 1 && p >= 0; j--)
    {
      const double head_t = seg_head_t[j - 1];
      while (p >= 0 && (pcl_out.points[p].curvature / double(1000) > head_t ||
                        (j == 1 && pcl_out.points[p].curvature / double(1000) == head_t)))
      {
        seg_of[p] = j - 1;
        --p;
      }
    }
    for (; p >= 0; --p) seg_of[p] = -1;  // earlier than available history (rejected by Process)
  }
#ifdef MP_EN
  #pragma omp parallel for
#endif
  for (int i = 0; i < n_pts; i++)
  {
    auto &pt = pcl_out.points[i];
    const int s = seg_of[i];
    if (s < 0) continue;
    const double t = pt.curvature / double(1000);
    const double dt_i = t - seg_head_t[s];

    /* Compensate to the 'end' frame (direction is INVERSE of the frame's motion), as the
     * per-segment affine folded above: P' = (A0 + sin*A1 + vers*A2) P + d0 + sin*d1 + vers*d2
     * + U + V dt + Acc dt^2. Intra-segment angles are milliradians, where the series is exact
     * to double precision and far cheaper than sin/cos; the branch keeps a pathological
     * gyro/dt honest. */
    const double th = seg_w[s] * dt_i;
    double sn, vers;
    if (std::fabs(th) < 0.1)
    {
      const double th2 = th * th;
      sn = th * (1.0 - th2 * (1.0 / 6.0 - th2 * (1.0 / 120.0)));
      vers = th2 * (0.5 - th2 * (1.0 / 24.0 - th2 * (1.0 / 720.0)));
    }
    else
    {
      sn = std::sin(th);
      vers = 1.0 - std::cos(th);
    }

    const V3D P_i(pt.x, pt.y, pt.z);
    const V3D P_compensate = segA0[s] * P_i + sn * (segA1[s] * P_i) + vers * (segA2[s] * P_i)
                           + segd0[s] + sn * segd1[s] + vers * segd2[s]
                           + segU[s] + dt_i * segV[s] + (dt_i * dt_i) * segAcc[s];

    pt.x = P_compensate(0);
    pt.y = P_compensate(1);
    pt.z = P_compensate(2);
  }
}

bool ImuProcess::Process(const MeasureGroup & meas,
                         esekfom::esekf<state_ikfom, 12, input_ikfom> & kf_state,
                         PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  last_status_ = ProcessStatus::kRejected;
  last_outcome_ = {};
  const auto finish = [this, &meas, &cur_pcl_un_](
                        fast_lio::ImuDisposition disposition, fast_lio::ImuProcessReason reason, bool output = false) {
    if (disposition == fast_lio::ImuDisposition::kUncommitted &&
        !fast_lio::canRetainImuBatch(
          meas.imu,
          [](const auto & imu) { return rclcpp::Time(imu->header.stamp).seconds(); },
          validImuForConsumption)) {
      disposition = fast_lio::ImuDisposition::kInvalid;
      reason = fast_lio::ImuProcessReason::kInvalidImu;
    }
    if (!output && cur_pcl_un_ && !cur_pcl_un_->empty())
      ++stale_output_reuse_count_;
    last_outcome_ = {disposition, reason, output};
    return output;
  };
  gap_summary = fast_lio::ImuGapSummary{};
  // Always invalidate the caller's previous cloud before any early return.
  if (!cur_pcl_un_)
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kNoOutput);
  cur_pcl_un_->clear();
  const bool bridged_empty = meas.imu.empty() && bridgesEmptyScan(meas.lidar_end_time);
  if (!meas.lidar || meas.lidar->empty())
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kNoLidar);
  if (meas.imu.empty() && !bridged_empty)
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kNoImu);
  if (!scan_consumption_.canProcess(meas.lidar_end_time))
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kNonadvancing);
  const auto timing =
    fast_lio::scanTime(meas.lidar_beg_time, meas.lidar->points, [](const auto & point) { return point.curvature; });
  if (!timing.valid() || timing.end > meas.lidar_end_time)
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kInvalidScanTime);

  for (const auto & imu : meas.imu) {
    if (!validImuForConsumption(imu))
      return finish(fast_lio::ImuDisposition::kInvalid, fast_lio::ImuProcessReason::kInvalidImu);
  }

  if (imu_need_init_)
  {
    if (!has_initialization_attempt_start_) {
      initialization_attempt_start_ = meas.lidar_beg_time;
      has_initialization_attempt_start_ = true;
    }
    const auto initialization = IMU_init(meas, kf_state, init_iter_num);
    const bool valid_init = initialization.reason == fast_lio::ImuProcessReason::kInitializationAccumulated;

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();
    last_lidar_end_time_ = meas.lidar_end_time;
    scan_consumption_.commit(meas.lidar_end_time);
    integration_ledger_.anchor(meas.lidar_end_time);
    last_status_ = ProcessStatus::kInitializing;
    if (!valid_init) {
      discardInitializationWindow();
      return finish(initialization.disposition, initialization.reason);
    }

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num >= MAX_INI_COUNT)
    {
      const double still_ratio = mean_acc.norm() > 1e-6 ? std::sqrt(cov_acc.sum()) / mean_acc.norm() : 1e9;
      const bool timed_out = (meas.lidar_beg_time - initialization_attempt_start_) > init_still_timeout_s;
      if (init_require_still && still_ratio > init_still_tol && !timed_out)
      {
        // platform is moving: restart the accumulation window and keep waiting
        std::cerr << "[IMU-INIT] motion detected during init (scatter " << still_ratio
                  << " > " << init_still_tol << ") — restarting init window" << std::endl;
        b_first_frame_ = true;   // next IMU_init call re-seeds mean/cov from fresh samples
        init_iter_num = 0;
        return finish(fast_lio::ImuDisposition::kCommitted, fast_lio::ImuProcessReason::kInitializationMotion);
      }
      if (init_require_still && timed_out && still_ratio > init_still_tol)
        std::cerr << "[IMU-INIT] WARN still-gate timeout after " << init_still_timeout_s
                  << " s — initializing from a MOVING mean (gravity may be tilted; scatter "
                  << still_ratio << ")" << std::endl;
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      std::cout << "IMU Initial Done" << std::endl;
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      if (runtime_log_en) fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return finish(fast_lio::ImuDisposition::kCommitted,
                  imu_need_init_ ? fast_lio::ImuProcessReason::kInitializationAccumulated
                                 : fast_lio::ImuProcessReason::kInitializationComplete);
  }

  // Original scanTime validation above must precede any removal. Only the
  // overlapping branch copies points; all timestamps and the IMU window stay
  // unchanged. Points without history never enter deskew or the caller's map.
  const double history_offset = last_lidar_end_time_ - meas.lidar_beg_time;
  const auto point_offset = [](const auto & point) {
    return point.curvature;
  };
  const auto history = fast_lio::scanHistoryAction(meas.lidar->points, history_offset, point_offset);
  if (history == fast_lio::ScanHistoryAction::kReject)
    return finish(fast_lio::ImuDisposition::kUncommitted, fast_lio::ImuProcessReason::kHistory);
  PointCloudXYZI::Ptr history_cloud;
  if (history == fast_lio::ScanHistoryAction::kDiscardEarly) {
    history_cloud.reset(new PointCloudXYZI(*meas.lidar));
    fast_lio::discardPointsBeforeHistory(history_cloud->points, history_offset, point_offset);
    history_cloud->width = static_cast<std::uint32_t>(history_cloud->size());
    history_cloud->height = 1;
  }

  std::vector<double> imu_stamps;
  imu_stamps.reserve(meas.imu.size() + 1);
  imu_stamps.push_back(rclcpp::Time(last_imu_->header.stamp).seconds());
  for (const auto & imu : meas.imu)
    imu_stamps.push_back(rclcpp::Time(imu->header.stamp).seconds());
  // With empty scans carried by the prior, the stretch past the last sample is a gap like any other: bounded by
  // max_gap_s, not by the one-scan extrapolation limit.
  fast_lio::ImuCoverageParams limits = coverage_params;
  if (gap_params.enabled && gap_params.empty_scans && limits.max_gap_s > 0.0 && limits.max_extrapolation_s > 0.0)
    limits.max_extrapolation_s = std::max(limits.max_extrapolation_s, limits.max_gap_s);
  coverage_result = fast_lio::imuCoverage(last_lidar_end_time_, meas.lidar_end_time, imu_stamps, limits);
  if (!coverage_result.covered()) {
    // Missing motion cannot be recovered by integrating across the gap. Hold
    // the state, rebase ONLY the input cursor and require caller-side map
    // re-anchoring before claiming healthy output again. Do not relearn gravity.
    // Malformed / unordered samples cannot establish a new temporal anchor.
    if (coverage_result.status == fast_lio::ImuCoverageStatus::kGap ||
        coverage_result.status == fast_lio::ImuCoverageStatus::kStart ||
        coverage_result.status == fast_lio::ImuCoverageStatus::kEnd) {
      if (!meas.imu.empty())
        last_imu_ = meas.imu.back();
      last_lidar_end_time_ = meas.lidar_end_time;
      scan_consumption_.commit(meas.lidar_end_time);
      integration_ledger_.anchor(meas.lidar_end_time);
      IMUpose.clear();
    }
    last_status_ = ProcessStatus::kCoverageGap;
    const auto outcome = fast_lio::imuCoverageOutcome(coverage_result.status);
    return finish(outcome.disposition, outcome.reason);
  }

  last_outcome_ = {fast_lio::ImuDisposition::kFatal, fast_lio::ImuProcessReason::kPartialPropagation};
  UndistortPcl(meas, kf_state, *cur_pcl_un_, history_cloud ? *history_cloud : *meas.lidar);
  scan_consumption_.commit(meas.lidar_end_time);
  integration_ledger_.anchor(meas.lidar_end_time);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
  last_status_ = ProcessStatus::kProcessed;
  const bool output = !cur_pcl_un_->empty();
  return finish(fast_lio::ImuDisposition::kCommitted,
                output ? fast_lio::ImuProcessReason::kProcessed : fast_lio::ImuProcessReason::kProcessedNoCloud,
                output);
}

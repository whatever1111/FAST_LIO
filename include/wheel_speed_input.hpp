// wheel_speed_input.hpp — what the front end keeps of the wheel speed between scans, and what a scan makes of it.
//
// Two subscription callbacks fill one WheelSampleRing: the gated twist (wheel_topic, the stream the static paths
// read) and, when the node has a variance topic, the variances of the same samples. Without a variance topic every
// gated sample carries the parameter sigmas, the wheel update's noise model before the bridge published variances.
// After the lidar update the scan thread reads the ring once: the paired samples of the scan's own span, fitted at
// its end (wheel_measurement.hpp). Before the lidar update an applying normal-noise floor asks whether that read will
// find a speed (hasFit): the callbacks wait for the scan timer, so the ring cannot change in between. The scan thread
// alone keeps the slip regressor's IMU history and the per-scan counts.
//
// Locking is the owner's. The node guards addTwist, addVariances, read and hasFit with its mtx_twist; recordImu, step
// and counts belong to the scan thread and take no lock.
//
// Pure C++ + Eigen. No ROS.

#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "wheel_measurement.hpp"

namespace fast_lio
{

/// Hz, the wheelspeed bridge's publication rate.
constexpr std::size_t kWheelBridgeRateHz = 50;
/// s. Every callback of the node shares one mutually exclusive group with the scan timer, so while a scan is
/// processed the bridge's samples wait in the subscription queues and arrive together afterwards. 5 s is the longest
/// such wait the front end rides through without losing IMU input: its IMU subscription keeps 1000 samples, 5 s at
/// 200 Hz. A longer one loses IMU as well, and what it pushes out of the ring unpaired is counted.
constexpr std::size_t kWheelStallSeconds = 5;
/// Samples the ring holds, and the depth of the variance subscription: every sample of the longest stall.
constexpr std::size_t kWheelSampleCapacity = kWheelBridgeRateHz * kWheelStallSeconds;

/// The header stamp as the pairing key: seconds times 1e9 plus nanoseconds, an integer. nullopt for a nanosecond
/// field of a second or more, which no well-formed stamp has.
inline std::optional<std::int64_t> wheelStampNs(std::int32_t sec, std::uint32_t nanosec)
{
  constexpr std::int64_t kNanosPerSecond = 1000000000;
  if (nanosec >= kNanosPerSecond) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(sec) * kNanosPerSecond + static_cast<std::int64_t>(nanosec);
}

/// What a scan's wheel step did.
enum class WheelOutcome
{
  kUpdate,                  ///< the measurement went into the filter
  kNoSample,                ///< no paired, valid sample in the scan's own span
  kNoImu,                   ///< the scan carried no IMU sample
  kPhiUnavailable,          ///< k != 0 and no IMU sample in the slip window
  kInvalid,                 ///< the slip correction is undefined or the measurement is not finite
  kUnidentifiableEndpoint,  ///< one wheel sample time strictly before the endpoint
  kEndpointUnavailable,     ///< the scan's IMU times cannot identify its endpoint
};

inline const char * wheelOutcomeName(WheelOutcome outcome)
{
  switch (outcome) {
    case WheelOutcome::kUpdate:
      return "update";
    case WheelOutcome::kNoSample:
      return "no_sample";
    case WheelOutcome::kNoImu:
      return "no_imu";
    case WheelOutcome::kPhiUnavailable:
      return "phi_unavailable";
    case WheelOutcome::kInvalid:
      return "invalid";
    case WheelOutcome::kUnidentifiableEndpoint:
      return "unidentifiable_endpoint";
    case WheelOutcome::kEndpointUnavailable:
      return "endpoint_unavailable";
  }
  return "unknown";
}

/// The ring's part of a scan, read under the node's lock. The counters run since construction.
struct WheelRingRead
{
  std::optional<WheelFit> fit;
  WheelFitStatus fit_status = WheelFitStatus::kNoSample;
  int unpaired = 0;                       ///< valid samples in the span still without their variances
  std::uint64_t unmatched_samples = 0;    ///< gated samples pushed out without their variances
  std::uint64_t unmatched_variances = 0;  ///< variances pushed out without their gated sample
  std::uint64_t rejected = 0;             ///< duplicate halves, unusable variances and malformed stamps
};

/// The scan thread's counts since construction, one per outcome.
struct WheelScanCounts
{
  std::uint64_t updates = 0;
  std::uint64_t no_sample = 0;
  std::uint64_t no_imu = 0;
  std::uint64_t phi_unavailable = 0;
  std::uint64_t invalid = 0;
};

/// One scan's wheel step: what it read and the measurement, when there is one.
struct WheelStep
{
  WheelOutcome outcome = WheelOutcome::kNoSample;
  WheelRingRead ring;
  std::optional<double> phi;  ///< the scan's slip regressor, whether or not k needs it
  std::optional<WheelMeasurement> measurement;
  WheelAngularEndpoint angular;
  const char * reason = "no_sample";
  Eigen::Matrix<double, Eigen::Dynamic, kWheelStateDim> jacobian_white;
  Eigen::VectorXd residual_white;                         ///< dimensionless, after noise whitening
  Eigen::VectorXd variances_white;                        ///< ones
  double nis = std::numeric_limits<double>::quiet_NaN();  ///< diagnostic only
  int nis_dof = 0;
};

class WheelSpeedInput
{
public:
  WheelSpeedInput() : ring_(kWheelSampleCapacity) {}

  /// Where the variances come from: the variance topic, or the parameter sigmas (m/s) for every gated sample. Called
  /// before any callback can reach this object.
  void configure(bool variances_from_topic, double sigma_vx, double sigma_lat)
  {
    variances_from_topic_ = variances_from_topic;
    var_vx_ = sigma_vx * sigma_vx;
    var_lat_ = sigma_lat * sigma_lat;
  }

  // The callbacks' part: the caller holds the node's mtx_twist.

  /// A gated sample the caller has found to be a measurement (wheelSampleIsEvidence). Without a variance topic it is
  /// paired at once with the parameter variances.
  void addTwist(std::int32_t sec, std::uint32_t nanosec, double v)
  {
    const std::optional<std::int64_t> stamp = wheelStampNs(sec, nanosec);
    if (!stamp) {
      ++malformed_stamps_;
      return;
    }
    if (ring_.pushSample(*stamp, v, true) && !variances_from_topic_) {
      ring_.attachSigma(*stamp, var_vx_, var_lat_);
    }
  }

  /// The variance half: cov[0] and cov[7] of the twist_cov message with the same stamp.
  void addVariances(std::int32_t sec, std::uint32_t nanosec, double var_vx, double var_lat)
  {
    const std::optional<std::int64_t> stamp = wheelStampNs(sec, nanosec);
    if (!stamp) {
      ++malformed_stamps_;
      return;
    }
    ring_.attachSigma(*stamp, var_vx, var_lat);
  }

  /// The speed over the scan's span [t_begin, t_end], fitted at t_end, and the ring's counters.
  WheelRingRead read(double t_begin, double t_end) const
  {
    // t_end - (t_end - t_begin) is t_begin exactly: two stamps of one scan are within a factor of two of each other,
    // so their difference is exact, and the subtraction back has an exact, representable result.
    const double span = t_end - t_begin;
    WheelRingRead out;
    out.fit = ring_.fitAt(t_end, span, &out.fit_status);
    out.unpaired = ring_.unpairedIn(t_end, span);
    out.unmatched_samples = ring_.unmatchedSamples();
    out.unmatched_variances = ring_.unmatchedSigmas();
    out.rejected = ring_.rejected() + malformed_stamps_;
    return out;
  }

  /// Whether read(t_begin, t_end) fits a speed: the scan's wheel update has a measurement to make only then. A span
  /// without a paired valid sample, an endpoint one sample time cannot identify and an invalid fit leave it without.
  bool hasFit(double t_begin, double t_end) const { return read(t_begin, t_end).fit.has_value(); }

  // The scan thread's part: no lock.

  /// One raw IMU sample of a scan, in order: t [s] and the forward specific force in the wheel frame [m/s^2]
  /// (wheelForward of the raw a_m). A sample older than the last one kept means the input started again; the history
  /// then starts again with it.
  void recordImu(double t, double a_forward)
  {
    if (!imu_forward_.empty() && t < imu_forward_.back().first) {
      imu_forward_.clear();
    }
    imu_forward_.emplace_back(t, a_forward);
  }

  /// The wheel step after this scan's IMU samples were recorded, using its timestamped raw angular endpoint.
  /// prior is the filter covariance immediately before the wheel update, used only for diagnostic NIS.
  WheelStep step(const WheelRingRead & ring,
                 double t_end,
                 const WheelAngularEndpoint & angular,
                 const Eigen::Matrix3d & R_world_imu,
                 const Eigen::Vector3d & velocity_world,
                 const Eigen::Vector3d & gyro_bias,
                 const Eigen::Matrix3d & R_robot_to_imu,
                 const WheelModelParams & params,
                 const Eigen::Matrix<double, kWheelStateDim, kWheelStateDim> & prior)
  {
    const double t_begin = t_end - params.slip_window_s;
    while (!imu_forward_.empty() && imu_forward_.front().first < t_begin) {
      imu_forward_.pop_front();
    }
    WheelStep out;
    out.ring = ring;
    out.phi = slipPhi(imu_forward_, t_end, params.slip_window_s, params.slip_phi0);
    if (!ring.fit) {
      if (ring.fit_status == WheelFitStatus::kNoSample) {
        ++counts_.no_sample;
        out.outcome = WheelOutcome::kNoSample;
      } else {
        ++counts_.invalid;
        out.outcome = ring.fit_status == WheelFitStatus::kUnidentifiableEndpoint ? WheelOutcome::kUnidentifiableEndpoint
                                                                                 : WheelOutcome::kInvalid;
        out.reason =
          ring.fit_status == WheelFitStatus::kUnidentifiableEndpoint ? "unidentifiable_endpoint" : "invalid_fit";
      }
      return out;
    }
    out.angular = angular;
    if (angular.status != WheelAngularStatus::kValid) {
      out.reason = wheelAngularStatusName(angular.status);
      if (angular.status == WheelAngularStatus::kNoImu) {
        ++counts_.no_imu;
        out.outcome = WheelOutcome::kNoImu;
      } else {
        ++counts_.invalid;
        out.outcome = angular.status == WheelAngularStatus::kEndpointUnavailable ? WheelOutcome::kEndpointUnavailable
                                                                                 : WheelOutcome::kInvalid;
      }
      return out;
    }
    if (params.slip_k != 0.0 && !out.phi) {
      ++counts_.phi_unavailable;
      out.outcome = WheelOutcome::kPhiUnavailable;
      out.reason = "phi_unavailable";
      return out;
    }
    out.measurement =
      wheelMeasurement(*ring.fit, out.phi, R_world_imu, velocity_world, gyro_bias, angular, R_robot_to_imu, params);
    if (!out.measurement) {
      ++counts_.invalid;
      out.outcome = WheelOutcome::kInvalid;
      out.reason = "invalid_measurement";
      return out;
    }
    if (!out.measurement->pack(out.jacobian_white, out.residual_white, out.variances_white)) {
      out.measurement.reset();
      ++counts_.invalid;
      out.outcome = WheelOutcome::kInvalid;
      out.reason = "invalid_active_covariance";
      return out;
    }
    const auto nis = out.measurement->nis(prior);
    if (nis) {
      out.nis = *nis;
    }
    out.nis_dof = static_cast<int>(out.residual_white.size());
    ++counts_.updates;
    out.outcome = WheelOutcome::kUpdate;
    out.reason = "valid";
    return out;
  }

  const WheelScanCounts & counts() const { return counts_; }
  std::size_t imuHistorySize() const { return imu_forward_.size(); }

private:
  // Shared with the callbacks, under the node's mtx_twist.
  WheelSampleRing ring_;
  std::uint64_t malformed_stamps_ = 0;
  // Written before the callbacks can reach the object, read-only afterwards.
  bool variances_from_topic_ = false;
  double var_vx_ = 0.0;
  double var_lat_ = 0.0;
  // The scan thread's own.
  std::deque<std::pair<double, double>> imu_forward_;  ///< (t [s], a_x [m/s^2]) within the slip window
  WheelScanCounts counts_;
};

/// One scan's [WHEEL] line: lidar_end_time, the outcome, what the step used, and every counter since construction.
/// raw_vx is the fitted gated speed before the scale and the slip correction, z the corrected one; a value the step
/// did not reach is nan, r_vy also when the bridge does not observe the lateral axis.
inline std::string wheelLine(double t_end, const WheelStep & step, const WheelScanCounts & counts)
{
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  const bool fitted = step.ring.fit.has_value();
  const bool measured = step.measurement.has_value();
  const bool lateral = measured && step.measurement->noise.lateral;
  char text[3072];
  std::snprintf(text,
                sizeof(text),
                "[WHEEL] t=%.6f outcome=%s samples=%d unpaired=%d raw_vx=%.9g fit_rms=%.9g phi=%.9g z=%.9g r_vx=%.9g "
                "r_vy=%.9g r_vz=%.9g updates=%llu no_sample=%llu no_imu=%llu phi_unavailable=%llu invalid=%llu "
                "unmatched_samples=%llu unmatched_variances=%llu rejected=%llu schema=2 "
                "omega_end=[%.17g %.17g %.17g] omega_t0=%.17g omega_t1=%.17g omega_weights=[%.17g %.17g] "
                "Rgyro=[%.17g %.17g %.17g %.17g %.17g %.17g] fit_order=%d c_l1=%.17g c_l2_sq=%.17g "
                "Rfit_x=%.17g Rfit_y=%.17g nis=%.17g nis_dof=%d reason=%s",
                t_end,
                wheelOutcomeName(step.outcome),
                fitted ? step.ring.fit->samples : 0,
                step.ring.unpaired,
                fitted ? step.ring.fit->v : kNaN,
                fitted ? step.ring.fit->residual_rms : kNaN,
                step.phi ? *step.phi : kNaN,
                measured ? step.measurement->z(0) : kNaN,
                measured ? step.measurement->noise.var_vx : kNaN,
                lateral ? step.measurement->noise.var_vy : kNaN,
                measured ? step.measurement->noise.var_vz : kNaN,
                static_cast<unsigned long long>(counts.updates),
                static_cast<unsigned long long>(counts.no_sample),
                static_cast<unsigned long long>(counts.no_imu),
                static_cast<unsigned long long>(counts.phi_unavailable),
                static_cast<unsigned long long>(counts.invalid),
                static_cast<unsigned long long>(step.ring.unmatched_samples),
                static_cast<unsigned long long>(step.ring.unmatched_variances),
                static_cast<unsigned long long>(step.ring.rejected),
                step.angular.raw(0),
                step.angular.raw(1),
                step.angular.raw(2),
                step.angular.t0,
                step.angular.t1,
                step.angular.weights(0),
                step.angular.weights(1),
                measured ? step.measurement->noise.gyro(0, 0) : kNaN,
                measured ? step.measurement->noise.gyro(0, 1) : kNaN,
                measured ? step.measurement->noise.gyro(0, 2) : kNaN,
                measured ? step.measurement->noise.gyro(1, 1) : kNaN,
                measured ? step.measurement->noise.gyro(1, 2) : kNaN,
                measured ? step.measurement->noise.gyro(2, 2) : kNaN,
                fitted ? step.ring.fit->fit_order : -1,
                fitted ? step.ring.fit->c_l1 : kNaN,
                fitted ? step.ring.fit->c_l2_sq : kNaN,
                measured ? step.measurement->noise.fit_vx : kNaN,
                lateral ? step.measurement->noise.fit_vy : kNaN,
                step.nis,
                step.nis_dof,
                step.reason);
  return text;
}

}  // namespace fast_lio

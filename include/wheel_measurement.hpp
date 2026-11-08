// wheel_measurement.hpp — the legs' forward speed as an iEKF measurement.
//
// The wheelspeed bridge publishes two messages per sample with the same header
// stamp: the gated speed on /wheel_odom/twist, NaN where it cannot measure, and the
// speed with its variances on /wheel_odom/twist_cov (cov[0] the forward variance,
// cov[7] the lateral one, 1e6 for an axis it does not observe). Which arrives first is
// not defined. A sample enters the update only when both halves met: the gated speed
// is finite and its variances were attached by the exact stamp.
//
// Per scan, every such sample in [t_end - window, t_end] is used: a line weighted by
// 1/sigma^2 is evaluated at the scan end, which removes the lag of taking the newest
// sample (up to one bridge period) without the half-window lag of a mean. The fit's
// scatter is for logging only; the measurement variance stays the per-sample one,
// floored at the honest per-frame sigma of the 5 s averaging horizon, because the
// speed error is correlated over seconds and averaging the frame's samples does not
// shrink it.
//
// On a slope a fraction of the stride slips. With phi the window mean of the raw
// forward specific force in g, minus the platform's level intercept phi0, the speed is
// corrected as v / (1 + k phi); k = 0 returns v itself, bit for bit, and then needs no
// phi at all.
//
// The legs measure their reference point O, the state is the IMU point I, and
// v_O = v_I - omega x r with r = p_I - p_O in the IMU frame (the inverse of
// legEdgeVelocity in gravity_align_kinematics.hpp). The term enters h and, through
// omega = omega_m - b_g, the gyro-bias columns of H.
//
// Pure C++ + Eigen. No ROS. Not synchronized: the owner guards a ring shared between
// the callback and the update.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>  // cross()

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <vector>

namespace fast_lio
{

/// (m/s)^2; the bridge's "this axis is not observed" variance (fslam
/// joints_to_wheelspeed kUnobservedVariance). A contract marker, not a threshold.
constexpr double kBridgeUnobservedVariance = 1e6;
/// m/s^2; the g that expresses the slip regressor phi in units of g.
constexpr double kSlipGravity = 9.81;

/// Error-state layout of state_ikfom (use-ikfom.hpp) that the Jacobian is written for.
constexpr int kWheelStateDim = 23;
constexpr int kWheelRotIndex = 3;
constexpr int kWheelVelIndex = 12;
constexpr int kWheelGyroBiasIndex = 15;

/// One bridge sample as the ring holds it. Either half may still be missing.
struct WheelSample
{
  std::int64_t stamp_ns = 0;  ///< header stamp, the pairing key
  double t = 0.0;             ///< s, the stamp as rclcpp::Time::seconds() reads it
  double v = 0.0;             ///< m/s, gated forward speed (linear.x)
  bool has_sample = false;    ///< the gated speed arrived
  bool valid = false;         ///< ... and is a measurement: finite and gated valid
  bool has_sigma = false;     ///< the variances arrived
  double var_vx = 0.0;        ///< (m/s)^2, twist_cov cov[0]
  double var_lat = 0.0;       ///< (m/s)^2, twist_cov cov[7]
};

/// A scan's speed: the weighted line through the window's paired samples at t_end.
struct WheelFit
{
  double v = 0.0;             ///< m/s at t_end
  int samples = 0;            ///< paired samples used
  double residual_rms = 0.0;  ///< m/s about the line (0 for a single sample); logging only
  double var_vx = 0.0;        ///< (m/s)^2, largest forward variance among the samples used
  double var_lat = 0.0;       ///< (m/s)^2, largest lateral variance among the samples used
};

/// Bridge samples by exact stamp, oldest first. Capacity is the window times the
/// sample rate; whatever is pushed out unpaired is counted and gone.
class WheelSampleRing
{
public:
  explicit WheelSampleRing(std::size_t capacity) : capacity_(capacity)
  {
    if (capacity == 0) {
      throw std::invalid_argument("WheelSampleRing capacity must be positive");
    }
    entries_.reserve(capacity + 1);
  }

  /// The gated half. `valid` is the caller's gate; a non-finite speed is never valid.
  /// A second gated sample for a stamp is refused and counted.
  bool pushSample(std::int64_t stamp_ns, double v, bool valid)
  {
    WheelSample & entry = entryFor(stamp_ns);
    if (entry.has_sample) {
      ++rejected_;
      return false;
    }
    entry.has_sample = true;
    entry.v = v;
    entry.valid = valid && std::isfinite(v);
    trim();
    return true;
  }

  /// The variance half: cov[0] and cov[7] of the same stamp. A forward variance that is
  /// not finite and positive, a lateral one that is not, or a second variance message
  /// for a stamp is refused and counted.
  bool attachSigma(std::int64_t stamp_ns, double var_vx, double var_lat)
  {
    if (!std::isfinite(var_vx) || !(var_vx > 0.0) || !std::isfinite(var_lat) || !(var_lat > 0.0)) {
      ++rejected_;
      return false;
    }
    WheelSample & entry = entryFor(stamp_ns);
    if (entry.has_sigma) {
      ++rejected_;
      return false;
    }
    entry.has_sigma = true;
    entry.var_vx = var_vx;
    entry.var_lat = var_lat;
    trim();
    return true;
  }

  /// The speed at t_end from the paired, valid samples in [t_end - window_s, t_end]:
  /// a 1/var_vx-weighted line for two or more, the sample itself for one, nullopt for
  /// none or for a non-finite t_end or window. Samples whose stamps coincide in double
  /// seconds leave the slope undetermined; the weighted mean is then the answer.
  std::optional<WheelFit> fitAt(double t_end, double window_s) const
  {
    if (!std::isfinite(t_end) || !std::isfinite(window_s) || window_s < 0.0) {
      return std::nullopt;
    }
    const double t_begin = t_end - window_s;
    // Times and speeds relative to the newest usable sample. Its own offsets are exactly
    // zero, so one sample, or samples at one double time, give exactly zero time spread
    // and no slope. Centred on a weighted mean instead, the mean's rounding leaves a
    // spread of an ulp, and the speeds' rounding divided by it is an arbitrary slope.
    const WheelSample * reference = nullptr;
    for (const WheelSample & e : entries_) {
      if (usable(e, t_begin, t_end)) {
        reference = &e;
      }
    }
    if (reference == nullptr) {
      return std::nullopt;
    }
    WheelFit fit;
    double sw = 0.0;
    double swt = 0.0;
    double swv = 0.0;
    for (const WheelSample & e : entries_) {
      if (!usable(e, t_begin, t_end)) {
        continue;
      }
      const double w = 1.0 / e.var_vx;
      sw += w;
      swt += w * (e.t - reference->t);
      swv += w * (e.v - reference->v);
      fit.var_vx = std::max(fit.var_vx, e.var_vx);
      fit.var_lat = std::max(fit.var_lat, e.var_lat);
      ++fit.samples;
    }
    const double tau_mean = swt / sw;
    const double dv_mean = swv / sw;
    double stt = 0.0;
    double stv = 0.0;
    for (const WheelSample & e : entries_) {
      if (!usable(e, t_begin, t_end)) {
        continue;
      }
      const double w = 1.0 / e.var_vx;
      const double dt = (e.t - reference->t) - tau_mean;
      stt += w * dt * dt;
      stv += w * dt * ((e.v - reference->v) - dv_mean);
    }
    const double slope = stt > 0.0 ? stv / stt : 0.0;
    fit.v = reference->v + dv_mean + slope * ((t_end - reference->t) - tau_mean);
    double squares = 0.0;
    for (const WheelSample & e : entries_) {
      if (!usable(e, t_begin, t_end)) {
        continue;
      }
      const double residual = (e.v - reference->v) - (dv_mean + slope * ((e.t - reference->t) - tau_mean));
      squares += residual * residual;
    }
    fit.residual_rms = std::sqrt(squares / fit.samples);
    return fit;
  }

  /// Valid gated samples in [t_end - window_s, t_end] that have no variances yet: they
  /// do not enter this scan's update.
  int unpairedIn(double t_end, double window_s) const
  {
    int count = 0;
    const double t_begin = t_end - window_s;
    for (const WheelSample & e : entries_) {
      if (e.has_sample && e.valid && !e.has_sigma && e.t >= t_begin && e.t <= t_end) {
        ++count;
      }
    }
    return count;
  }

  std::size_t size() const { return entries_.size(); }
  std::size_t capacity() const { return capacity_; }
  const std::vector<WheelSample> & entries() const { return entries_; }
  /// Gated samples pushed out without their variances.
  std::uint64_t unmatchedSamples() const { return unmatched_samples_; }
  /// Variances pushed out without their gated sample.
  std::uint64_t unmatchedSigmas() const { return unmatched_sigmas_; }
  /// Inputs refused at entry: duplicate halves, unusable variances.
  std::uint64_t rejected() const { return rejected_; }

private:
  static bool usable(const WheelSample & e, double t_begin, double t_end)
  {
    // Written so that a NaN time fails the window test.
    return e.has_sample && e.valid && e.has_sigma && e.t >= t_begin && e.t <= t_end;
  }

  WheelSample & entryFor(std::int64_t stamp_ns)
  {
    // Arrivals are nearly in stamp order: search from the newest end.
    auto it = entries_.end();
    while (it != entries_.begin() && std::prev(it)->stamp_ns > stamp_ns) {
      --it;
    }
    if (it != entries_.begin() && std::prev(it)->stamp_ns == stamp_ns) {
      return *std::prev(it);
    }
    WheelSample fresh;
    fresh.stamp_ns = stamp_ns;
    // rclcpp::Time::seconds(): the nanosecond count divided as a double.
    fresh.t = static_cast<double>(stamp_ns) / 1e9;
    return *entries_.insert(it, fresh);
  }

  void trim()
  {
    while (entries_.size() > capacity_) {
      const WheelSample & oldest = entries_.front();
      if (oldest.has_sample && !oldest.has_sigma) {
        ++unmatched_samples_;
      } else if (oldest.has_sigma && !oldest.has_sample) {
        ++unmatched_sigmas_;
      }
      entries_.erase(entries_.begin());
    }
  }

  std::size_t capacity_;
  std::vector<WheelSample> entries_;
  std::uint64_t unmatched_samples_ = 0;
  std::uint64_t unmatched_sigmas_ = 0;
  std::uint64_t rejected_ = 0;
};

/// The forward component, in the wheel (robot) frame, of an IMU-frame vector:
/// (R_robot_to_imu^T a)_x, the rotation the wheel update uses.
inline double wheelForward(const Eigen::Matrix3d & R_robot_to_imu, const Eigen::Vector3d & imu_vector)
{
  return R_robot_to_imu.col(0).dot(imu_vector);
}

/// phi = mean over t in [t_m - window_s, t_m] of a_x(t) / g - phi0, with a_x the raw
/// (bias not removed) forward specific force in the wheel frame: `samples` is a range of
/// (t [s], a_x [m/s^2]) pairs, equally weighted. nullopt when the window holds no
/// sample, when a sample in it is not finite, or for non-finite arguments.
template<typename Samples>
std::optional<double> slipPhi(const Samples & samples, double t_m, double window_s, double phi0)
{
  if (!std::isfinite(t_m) || !std::isfinite(window_s) || window_s < 0.0 || !std::isfinite(phi0)) {
    return std::nullopt;
  }
  const double t_begin = t_m - window_s;
  double sum = 0.0;
  int count = 0;
  for (const auto & sample : samples) {
    if (!(sample.first >= t_begin && sample.first <= t_m)) {
      continue;
    }
    if (!std::isfinite(sample.second)) {
      return std::nullopt;
    }
    sum += sample.second;
    ++count;
  }
  if (count == 0) {
    return std::nullopt;
  }
  return sum / count / kSlipGravity - phi0;
}

/// v / (1 + k phi). k == 0 returns v itself and needs no phi; otherwise nullopt when phi
/// is missing or 1 + k phi is not a finite positive number.
inline std::optional<double> slipCorrect(double v, std::optional<double> phi, double k)
{
  if (k == 0.0) {
    return v;
  }
  if (!phi) {
    return std::nullopt;
  }
  const double denominator = 1.0 + k * *phi;
  if (!std::isfinite(denominator) || !(denominator > 0.0)) {
    return std::nullopt;
  }
  return v / denominator;
}

/// The measurement model's parameters.
struct WheelModelParams
{
  double speed_scale = 1.0;       ///< wheel_speed_scale, applied before the slip correction
  double slip_k = 0.0;            ///< slip coefficient k; 0 = no correction
  double slip_phi0 = 0.0;         ///< the platform's level intercept of phi
  double slip_window_s = 1.0;     ///< s, window of the phi mean
  double sigma_vx_floor = 0.075;  ///< m/s, per-frame sigma of the 5 s averaging horizon
  double sigma_vz = 0.10;         ///< m/s, vertical non-holonomic pseudo-measurement
  double sigma_lever = 0.0;       ///< m, geometric uncertainty of the lever arm
  /// m, IMU frame: the IMU point minus the legs' reference point (as
  /// gravity_align_leg_lever_arm).
  Eigen::Vector3d lever_arm = Eigen::Vector3d::Zero();
};

/// Diagonal measurement variances. The lateral row exists only where the bridge
/// observes the lateral axis.
struct WheelNoise
{
  double var_vx = 0.0;
  double var_vy = 0.0;
  double var_vz = 0.0;
  bool lateral = false;
};

/// R_vx = max(var_vx, floor^2) + (|omega| sigma_lever)^2; R_vy = var_lat + the same lever
/// term when var_lat is below the bridge's unobserved marker, no lateral row otherwise;
/// R_vz = sigma_vz^2. omega_norm is the scan's mean bias-corrected angular rate.
inline WheelNoise composeR(double var_vx, double var_lat, double omega_norm, const WheelModelParams & params)
{
  const double lever = omega_norm * params.sigma_lever;
  WheelNoise noise;
  noise.var_vx = std::max(var_vx, params.sigma_vx_floor * params.sigma_vx_floor) + lever * lever;
  noise.lateral = var_lat < kBridgeUnobservedVariance;
  noise.var_vy = noise.lateral ? var_lat + lever * lever : 0.0;
  noise.var_vz = params.sigma_vz * params.sigma_vz;
  return noise;
}

/// z, h(x), H and R of one wheel update, in the robot frame, rows (vx, vy, vz).
struct WheelMeasurement
{
  Eigen::Vector3d z = Eigen::Vector3d::Zero();  ///< (corrected forward speed, 0, 0)
  Eigen::Vector3d h = Eigen::Vector3d::Zero();  ///< predicted velocity of the legs' reference point
  Eigen::Matrix<double, 3, kWheelStateDim> H = Eigen::Matrix<double, 3, kWheelStateDim>::Zero();
  WheelNoise noise;

  /// The rows the update uses, (vx, vy, vz) or (vx, vz), as update_simple takes them.
  void pack(Eigen::Matrix<double, Eigen::Dynamic, kWheelStateDim> & jacobian,
            Eigen::VectorXd & residual,
            Eigen::VectorXd & variances) const
  {
    const int rows = noise.lateral ? 3 : 2;
    jacobian.resize(rows, kWheelStateDim);
    residual.resize(rows);
    variances.resize(rows);
    int row = 0;
    const auto put = [&](int axis, double variance) {
      jacobian.row(row) = H.row(axis);
      residual(row) = z(axis) - h(axis);
      variances(row) = variance;
      ++row;
    };
    put(0, noise.var_vx);
    if (noise.lateral) {
      put(1, noise.var_vy);
    }
    put(2, noise.var_vz);
  }
};

namespace wheel_detail
{
inline Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return m;
}
}  // namespace wheel_detail

/// One wheel update. h(x) = R_robot<-imu (R_world<-imu^T v - (omega_m - b_g) x r);
/// H has the rotation and velocity columns of the plain model and the gyro-bias columns
/// of the lever term. The forward speed is fit.v * speed_scale, slip-corrected with phi
/// (which k = 0 does not need). nullopt when the slip correction is not available or
/// anything in the result is not finite.
inline std::optional<WheelMeasurement> wheelMeasurement(const WheelFit & fit,
                                                        std::optional<double> phi,
                                                        const Eigen::Matrix3d & R_world_imu,
                                                        const Eigen::Vector3d & velocity_world,
                                                        const Eigen::Vector3d & gyro_bias,
                                                        const Eigen::Vector3d & omega_mean,
                                                        const Eigen::Matrix3d & R_robot_to_imu,
                                                        const WheelModelParams & params)
{
  const std::optional<double> forward = slipCorrect(fit.v * params.speed_scale, phi, params.slip_k);
  if (!forward) {
    return std::nullopt;
  }
  const Eigen::Matrix3d R_wheel_imu = R_robot_to_imu.transpose();
  const Eigen::Vector3d v_imu = R_world_imu.transpose() * velocity_world;
  const Eigen::Vector3d omega = omega_mean - gyro_bias;

  WheelMeasurement m;
  m.z = Eigen::Vector3d(*forward, 0.0, 0.0);
  m.h = R_wheel_imu * (v_imu - omega.cross(params.lever_arm));
  m.H.block<3, 3>(0, kWheelRotIndex) = R_wheel_imu * wheel_detail::skew(v_imu);
  m.H.block<3, 3>(0, kWheelVelIndex) = R_wheel_imu * R_world_imu.transpose();
  m.H.block<3, 3>(0, kWheelGyroBiasIndex) = -R_wheel_imu * wheel_detail::skew(params.lever_arm);
  m.noise = composeR(fit.var_vx, fit.var_lat, omega.norm(), params);
  if (!m.z.allFinite() || !m.h.allFinite() || !m.H.allFinite() || !std::isfinite(m.noise.var_vx) ||
      !std::isfinite(m.noise.var_vy) || !std::isfinite(m.noise.var_vz)) {
    return std::nullopt;
  }
  return m;
}

}  // namespace fast_lio

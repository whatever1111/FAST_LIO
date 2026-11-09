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
// scatter is for logging only. Endpoint coefficients propagate a worst-correlation
// variance bound, with the effective per-frame noise budget added once. The speed
// error is correlated over seconds, so averaging does not shrink that budget.
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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>  // cross()

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
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

enum class WheelAngularStatus
{
  kValid,
  kNoImu,
  kEndpointUnavailable,
  kInvalidTime,
  kInvalidRate,
  kInvalidCovariance,
  kNonFiniteEndpoint
};

inline const char * wheelAngularStatusName(WheelAngularStatus status)
{
  switch (status) {
    case WheelAngularStatus::kValid:
      return "valid";
    case WheelAngularStatus::kNoImu:
      return "no_imu";
    case WheelAngularStatus::kEndpointUnavailable:
      return "endpoint_unavailable";
    case WheelAngularStatus::kInvalidTime:
      return "invalid_imu_time";
    case WheelAngularStatus::kInvalidRate:
      return "invalid_gyro";
    case WheelAngularStatus::kInvalidCovariance:
      return "invalid_gyro_covariance";
    case WheelAngularStatus::kNonFiniteEndpoint:
      return "nonfinite_angular_endpoint";
  }
  return "invalid";
}

struct WheelAngularSample
{
  double t = 0.0;                                 ///< seconds
  Eigen::Vector3d raw = Eigen::Vector3d::Zero();  ///< rad/s, bias included
};

struct WheelAngularEndpoint
{
  WheelAngularStatus status = WheelAngularStatus::kNoImu;
  Eigen::Vector3d raw = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  double t0 = std::numeric_limits<double>::quiet_NaN();
  double t1 = std::numeric_limits<double>::quiet_NaN();
  Eigen::Vector2d weights = Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN());
};

/// Endpoint of a locally affine raw rate, using only this scan's samples. gyro_variance is cov_gyr's
/// diagonal in (rad/s)^2, shared by the independent samples. Equal timestamps use the last raw sample.
inline WheelAngularEndpoint
angularRateAtEnd(const std::vector<WheelAngularSample> & samples, double t_end, const Eigen::Vector3d & gyro_variance)
{
  WheelAngularEndpoint out;
  if (samples.empty()) {
    return out;
  }
  out.status = WheelAngularStatus::kInvalidTime;
  if (!std::isfinite(t_end)) {
    return out;
  }
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (!std::isfinite(samples[i].t) || (i > 0 && samples[i].t < samples[i - 1].t)) {
      return out;
    }
    if (!samples[i].raw.allFinite()) {
      out.status = WheelAngularStatus::kInvalidRate;
      return out;
    }
  }
  if (!gyro_variance.allFinite() || (gyro_variance.array() < 0.0).any()) {
    out.status = WheelAngularStatus::kInvalidCovariance;
    return out;
  }
  std::size_t tail = samples.size();
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].t <= t_end) {
      tail = i;
    }
  }
  out.status = WheelAngularStatus::kEndpointUnavailable;
  if (tail == samples.size()) {
    return out;  // no sample on the earlier side, so no backward extrapolation
  }
  if (samples[tail].t == t_end) {
    out.raw = samples[tail].raw;
    out.t0 = out.t1 = t_end;
    out.weights = Eigen::Vector2d(0.0, 1.0);
    out.covariance = gyro_variance.asDiagonal();
  } else {
    std::size_t first = tail;
    std::size_t second = tail + 1;
    if (second == samples.size()) {
      while (first > 0 && samples[first - 1].t == samples[tail].t) {
        --first;
      }
      if (first == 0) {
        return out;
      }
      --first;
      second = tail;
    }
    out.t0 = samples[first].t;
    out.t1 = samples[second].t;
    const double span = out.t1 - out.t0;
    if (!std::isfinite(span) || !(span > 0.0)) {
      out.status = WheelAngularStatus::kNonFiniteEndpoint;
      return out;
    }
    const double beta = (t_end - out.t0) / span;
    out.weights = Eigen::Vector2d(1.0 - beta, beta);
    out.raw = out.weights(0) * samples[first].raw + out.weights(1) * samples[second].raw;
    out.covariance = (out.weights.squaredNorm() * gyro_variance).asDiagonal();
  }
  if (!out.raw.allFinite() || !out.weights.allFinite() || !out.covariance.allFinite()) {
    out.status = WheelAngularStatus::kNonFiniteEndpoint;
    return out;
  }
  out.status = WheelAngularStatus::kValid;
  return out;
}

/// Pure structural conversion of the current MeasureGroup's IMU messages, without a ROS dependency.
/// The node and its timestamped-input contract test use this same conversion.
template<typename ImuSamples>
WheelAngularEndpoint
angularEndpointFromImu(const ImuSamples & imu_samples, double t_end, const Eigen::Vector3d & gyro_variance)
{
  std::vector<WheelAngularSample> samples;
  samples.reserve(imu_samples.size());
  for (const auto & imu : imu_samples) {
    const auto & stamp = imu->header.stamp;
    const auto & rate = imu->angular_velocity;
    const std::int64_t stamp_ns = static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    const double t = static_cast<double>(stamp_ns) / 1e9;  // rclcpp::Time::seconds()
    samples.push_back({t, Eigen::Vector3d(rate.x, rate.y, rate.z)});
  }
  return angularRateAtEnd(samples, t_end, gyro_variance);
}

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
  double sample_max_vx = 0.0;       ///< (m/s)^2, largest forward input variance
  double sample_max_lat = 0.0;      ///< (m/s)^2, largest lateral input variance
  double endpoint_bound_vx = 0.0;   ///< (m/s)^2, max(sample_max_vx, U_x)
  double endpoint_bound_lat = 0.0;  ///< (m/s)^2, max(sample_max_lat, U_y), when observed
  bool lateral_observed = false;    ///< all input lateral variances are below the bridge marker
  int fit_order = 0;                ///< 0 = identifiable constant at t_end, 1 = affine endpoint
  double c_l1 = 0.0;                ///< sum |c_i| of the forward fit, dimensionless
  double c_l2_sq = 0.0;             ///< sum c_i^2 of the forward fit, dimensionless
};

enum class WheelFitStatus
{
  kValid,
  kNoSample,
  kUnidentifiableEndpoint,
  kInvalid
};

namespace wheel_detail
{
struct EndpointFit
{
  double value = 0.0;
  double residual_rms = 0.0;
  double bound = 0.0;
  double c_l1 = 0.0;
  double c_l2_sq = 0.0;
  int order = 0;
};

// The caller supplies paired samples in time order, with positive finite variances.
// min(variance)/variance is (1/variance)/max(1/variance) without overflowing the reciprocal.
inline std::optional<EndpointFit>
fitEndpoint(const std::vector<const WheelSample *> & samples, double t_end, bool lateral)
{
  const auto variance = [lateral](const WheelSample & sample) {
    return lateral ? sample.var_lat : sample.var_vx;
  };
  const WheelSample & reference = *samples.back();
  double min_variance = variance(reference);
  double max_variance = min_variance;
  for (const auto * sample : samples) {
    min_variance = std::min(min_variance, variance(*sample));
    max_variance = std::max(max_variance, variance(*sample));
  }
  double sw = 0.0;
  double swt = 0.0;
  double swv = 0.0;
  for (const auto * sample : samples) {
    const double w = min_variance / variance(*sample);
    sw += w;
    swt += w * (sample->t - reference.t);
    swv += w * (lateral ? 0.0 : sample->v - reference.v);
  }
  const double mean_t = swt / sw;
  const double mean_v = swv / sw;
  double spread = 0.0;
  double cross = 0.0;
  for (const auto * sample : samples) {
    const double w = min_variance / variance(*sample);
    const double dt = (sample->t - reference.t) - mean_t;
    spread += w * dt * dt;
    cross += w * dt * ((lateral ? 0.0 : sample->v - reference.v) - mean_v);
  }
  if (spread == 0.0 && (samples.front()->t != reference.t || t_end != reference.t)) {
    return std::nullopt;  // distinguish numerical loss of time spread from an identified constant
  }
  const double slope = spread > 0.0 ? cross / spread : 0.0;
  const double endpoint_dt = (t_end - reference.t) - mean_t;
  EndpointFit out;
  out.order = spread > 0.0 ? 1 : 0;
  out.value = (lateral ? 0.0 : reference.v) + mean_v + slope * endpoint_dt;
  double std_bound = 0.0;
  double squares = 0.0;
  for (const auto * sample : samples) {
    const double w = min_variance / variance(*sample);
    const double dt = (sample->t - reference.t) - mean_t;
    const double coefficient = w / sw + (spread > 0.0 ? endpoint_dt * w * dt / spread : 0.0);
    out.c_l1 += std::abs(coefficient);
    out.c_l2_sq += coefficient * coefficient;
    std_bound += std::abs(coefficient) * std::sqrt(variance(*sample));
    const double residual = (lateral ? 0.0 : sample->v - reference.v) - (mean_v + slope * dt);
    squares += residual * residual;
  }
  out.bound = spread > 0.0 ? std::max(max_variance, std_bound * std_bound) : max_variance;
  out.residual_rms = std::sqrt(squares / samples.size());
  if (!std::isfinite(sw) || !std::isfinite(spread) || !std::isfinite(cross) || !std::isfinite(out.value) ||
      !std::isfinite(out.bound) || !std::isfinite(out.residual_rms) || !std::isfinite(out.c_l1) ||
      !std::isfinite(out.c_l2_sq)) {
    return std::nullopt;
  }
  return out;
}
}  // namespace wheel_detail

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
  /// a 1/var_vx-weighted line. One distinct time identifies only that time's endpoint.
  /// status distinguishes an empty window, an unidentifiable endpoint and invalid arithmetic.
  std::optional<WheelFit> fitAt(double t_end, double window_s, WheelFitStatus * status = nullptr) const
  {
    const auto set_status = [status](WheelFitStatus value) {
      if (status != nullptr) {
        *status = value;
      }
    };
    set_status(WheelFitStatus::kInvalid);
    if (!std::isfinite(t_end) || !std::isfinite(window_s) || window_s < 0.0) {
      return std::nullopt;
    }
    const double t_begin = t_end - window_s;
    std::vector<const WheelSample *> samples;
    WheelFit fit;
    fit.lateral_observed = true;
    for (const WheelSample & e : entries_) {
      if (usable(e, t_begin, t_end)) {
        samples.push_back(&e);
        fit.sample_max_vx = std::max(fit.sample_max_vx, e.var_vx);
        fit.sample_max_lat = std::max(fit.sample_max_lat, e.var_lat);
        fit.lateral_observed = fit.lateral_observed && e.var_lat < kBridgeUnobservedVariance;
      }
    }
    if (samples.empty()) {
      set_status(WheelFitStatus::kNoSample);
      return std::nullopt;
    }
    if (samples.front()->t == samples.back()->t && samples.back()->t != t_end) {
      set_status(WheelFitStatus::kUnidentifiableEndpoint);
      return std::nullopt;
    }
    const auto forward = wheel_detail::fitEndpoint(samples, t_end, false);
    if (!forward) {
      return std::nullopt;
    }
    fit.samples = static_cast<int>(samples.size());
    fit.v = forward->value;
    fit.residual_rms = forward->residual_rms;
    fit.endpoint_bound_vx = forward->bound;
    fit.fit_order = forward->order;
    fit.c_l1 = forward->c_l1;
    fit.c_l2_sq = forward->c_l2_sq;
    if (fit.lateral_observed) {
      const auto lateral = wheel_detail::fitEndpoint(samples, t_end, true);
      if (!lateral) {
        return std::nullopt;
      }
      fit.endpoint_bound_lat = lateral->bound;
    }
    set_status(WheelFitStatus::kValid);
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
  double fit_vx = 0.0;                             ///< (m/s)^2, Rfit_x before the lever and gyro terms
  double fit_vy = 0.0;                             ///< (m/s)^2, Rfit_y when observed
  Eigen::Matrix3d gyro = Eigen::Matrix3d::Zero();  ///< A_r C_omega A_r^T, (m/s)^2
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();     ///< native covariance in the robot frame, (m/s)^2
};

/// Add the effective common noise budget once, then the lever's geometric uncertainty.
/// Lateral validity comes from the input marker, independently of the propagated bound.
inline WheelNoise composeR(const WheelFit & fit, double omega_norm, const WheelModelParams & params)
{
  const double lever = omega_norm * params.sigma_lever;
  WheelNoise noise;
  noise.fit_vx = std::max(fit.sample_max_vx, params.sigma_vx_floor * params.sigma_vx_floor) +
                 (fit.endpoint_bound_vx - fit.sample_max_vx);
  noise.fit_vy = fit.endpoint_bound_lat;
  noise.var_vx = noise.fit_vx + lever * lever;
  noise.lateral = fit.lateral_observed;
  noise.var_vy = noise.lateral ? fit.endpoint_bound_lat + lever * lever : 0.0;
  noise.var_vz = params.sigma_vz * params.sigma_vz;
  noise.R.diagonal() << noise.var_vx, noise.var_vy, noise.var_vz;
  return noise;
}

/// z, h(x), H and R of one wheel update, in the robot frame, rows (vx, vy, vz).
struct WheelMeasurement
{
  Eigen::Vector3d z = Eigen::Vector3d::Zero();  ///< (corrected forward speed, 0, 0)
  Eigen::Vector3d h = Eigen::Vector3d::Zero();  ///< predicted velocity of the legs' reference point
  Eigen::Matrix<double, 3, kWheelStateDim> H = Eigen::Matrix<double, 3, kWheelStateDim>::Zero();
  WheelNoise noise;

  /// Select native rows before any factorization. Inactive y never conditions x/z.
  void nativeRows(Eigen::Matrix<double, Eigen::Dynamic, kWheelStateDim> & jacobian,
                  Eigen::VectorXd & residual,
                  Eigen::MatrixXd & covariance) const
  {
    const int rows = noise.lateral ? 3 : 2;
    const int axes[] = {0, noise.lateral ? 1 : 2, 2};
    jacobian.resize(rows, kWheelStateDim);
    residual.resize(rows);
    covariance.resize(rows, rows);
    for (int i = 0; i < rows; ++i) {
      jacobian.row(i) = H.row(axes[i]);
      residual(i) = z(axes[i]) - h(axes[i]);
      for (int j = 0; j < rows; ++j) {
        covariance(i, j) = noise.R(axes[i], axes[j]);
      }
    }
  }

  /// update_simple receives L^-1 H and L^-1(z-h), with unit R_diag. Residuals are dimensionless.
  /// A singular or non-finite active R is refused, with no jitter or additional floor.
  bool pack(Eigen::Matrix<double, Eigen::Dynamic, kWheelStateDim> & jacobian,
            Eigen::VectorXd & residual,
            Eigen::VectorXd & variances) const
  {
    Eigen::MatrixXd covariance;
    nativeRows(jacobian, residual, covariance);
    if (!covariance.allFinite()) {
      return false;
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
      return false;
    }
    jacobian = llt.matrixL().solve(jacobian).eval();
    residual = llt.matrixL().solve(residual).eval();
    variances = Eigen::VectorXd::Ones(residual.size());
    return jacobian.allFinite() && residual.allFinite();
  }

  /// Native active-row NIS at the wheel update's prior, diagnostic only.
  std::optional<double> nis(const Eigen::Matrix<double, kWheelStateDim, kWheelStateDim> & prior) const
  {
    Eigen::Matrix<double, Eigen::Dynamic, kWheelStateDim> jacobian;
    Eigen::VectorXd residual;
    Eigen::MatrixXd covariance;
    nativeRows(jacobian, residual, covariance);
    const Eigen::MatrixXd innovation = jacobian * prior * jacobian.transpose() + covariance;
    if (!innovation.allFinite() || !residual.allFinite()) {
      return std::nullopt;
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(innovation);
    if (llt.info() != Eigen::Success) {
      return std::nullopt;
    }
    const double value = residual.dot(llt.solve(residual));
    return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
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
                                                        const WheelAngularEndpoint & endpoint,
                                                        const Eigen::Matrix3d & R_robot_to_imu,
                                                        const WheelModelParams & params)
{
  if (endpoint.status != WheelAngularStatus::kValid || !endpoint.raw.allFinite() || !endpoint.covariance.allFinite()) {
    return std::nullopt;
  }
  const std::optional<double> forward = slipCorrect(fit.v * params.speed_scale, phi, params.slip_k);
  if (!forward) {
    return std::nullopt;
  }
  const Eigen::Matrix3d R_wheel_imu = R_robot_to_imu.transpose();
  const Eigen::Vector3d v_imu = R_world_imu.transpose() * velocity_world;
  const Eigen::Vector3d omega = endpoint.raw - gyro_bias;

  WheelMeasurement m;
  m.z = Eigen::Vector3d(*forward, 0.0, 0.0);
  m.h = R_wheel_imu * (v_imu - omega.cross(params.lever_arm));
  m.H.block<3, 3>(0, kWheelRotIndex) = R_wheel_imu * wheel_detail::skew(v_imu);
  m.H.block<3, 3>(0, kWheelVelIndex) = R_wheel_imu * R_world_imu.transpose();
  m.H.block<3, 3>(0, kWheelGyroBiasIndex) = -R_wheel_imu * wheel_detail::skew(params.lever_arm);
  m.noise = composeR(fit, omega.norm(), params);
  const Eigen::Matrix3d A = R_wheel_imu * wheel_detail::skew(params.lever_arm);
  m.noise.gyro = A * endpoint.covariance * A.transpose();
  m.noise.R += m.noise.gyro;
  m.noise.var_vx = m.noise.R(0, 0);
  m.noise.var_vy = m.noise.R(1, 1);
  m.noise.var_vz = m.noise.R(2, 2);
  if (!m.z.allFinite() || !m.h.allFinite() || !m.H.allFinite() || !std::isfinite(m.noise.var_vx) ||
      !std::isfinite(m.noise.var_vy) || !std::isfinite(m.noise.var_vz) || !m.noise.R.allFinite()) {
    return std::nullopt;
  }
  return m;
}

}  // namespace fast_lio

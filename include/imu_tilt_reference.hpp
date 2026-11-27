#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>

#include "frontend_gravity.hpp"
#include "gravity_align_kinematics.hpp"

namespace fast_lio
{
struct ImuTiltReferenceParams
{
  double window_s = 0.0;
  double gyr_cov = 0.0;
  double b_gyr_cov = 0.0;
  double noise_static = 0.0;
  double noise_moving = 0.0;
  double motion_init_sigma = 0.0;
  double accel_tol = 0.05;
  double gravity_raw = 0.0;
  double gravity_m_s2 = G_m_s2;
  double mag_ref = 0.0;
  double vel_sigma_scale = 0.0;
};

struct ImuTiltLegEdge
{
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double variance = 0.0;
  bool valid = false;
  bool stream_alive = false;
};

enum class ImuTiltWindowRejection
{
  kNone,
  kCoverage,
  kLegEdge,
  kMagnitude,
  kOutlier,
  kCovariance,
};

struct ImuTiltWindowDiagnostics
{
  int windows = 0;
  int samples = 0;
  double residual_angle = 0.0;
  double nis = 0.0;
  bool updated = false;
  ImuTiltWindowRejection rejection = ImuTiltWindowRejection::kNone;
};

// Scan-thread state, with no observations from the main filter or LiDAR.
class ImuTiltReference
{
public:
  void initialise(const Eigen::Vector3d & mean_acc,
                  const Eigen::Vector3d & mean_gyr,
                  bool still,
                  const ImuTiltReferenceParams & params,
                  double last_sample_time = 0.0)
  {
    *this = ImuTiltReference{};
    const double acc_norm = mean_acc.stableNorm();
    if (!(params.window_s > 0.0) || !std::isfinite(acc_norm) || !(acc_norm > 0.0) || (still && !mean_gyr.allFinite())) {
      return;
    }
    params_ = params;
    last_sample_time_ = last_sample_time;
    down_ = -mean_acc / acc_norm;
    if (still)
      bias_ = mean_gyr;
    const double sigma = still ? params.noise_static : params.motion_init_sigma;
    covariance_.topLeftCorner<3, 3>() = (sigma * sigma) * tiltProjector();
    covariance_.bottomRightCorner<3, 3>().diagonal().setConstant(1e-4);
    initialised_ = true;
  }

  void propagate(const Eigen::Vector3d & gyro_prev,
                 const Eigen::Vector3d & gyro_cur,
                 const Eigen::Vector3d & accel_cur,
                 double dt)
  {
    if (!initialised_ || !std::isfinite(dt) || !(dt > 0.0))
      return;
    last_sample_time_ += dt;
    last_period_ = 0.0;
    if (!gyro_prev.allFinite() || !gyro_cur.allFinite() || !accel_cur.allFinite())
      return;

    last_rate_ = 0.5 * (gyro_prev + gyro_cur) - bias_;
    last_period_ = dt;
    const Eigen::Matrix3d rotation = rotationBack(last_rate_, dt).toRotationMatrix();
    down_ = (rotation * down_).normalized();
    sum_ = rotation * sum_;
    start_edge_.velocity = rotation * start_edge_.velocity;
    // The force average and the leg velocity difference span the same window.
    sum_ += accel_cur;
    if (samples_++ == 0)
      first_sample_ = last_sample_time_;
    last_sample_ = last_sample_time_;

    const Eigen::Matrix3d pi = tiltProjector();
    Matrix6 phi = Matrix6::Identity();
    phi.topLeftCorner<3, 3>() = rotation;
    Eigen::Matrix3d cross;
    cross << 0.0, -down_.z(), down_.y(), down_.z(), 0.0, -down_.x(), -down_.y(), down_.x(), 0.0;
    phi.topRightCorner<3, 3>() = -dt * cross;
    covariance_ = (phi * covariance_ * phi.transpose()).eval();
    // Same per-sample gyr_cov and b_gyr_cov as the main filter.
    covariance_.topLeftCorner<3, 3>() += (params_.gyr_cov * dt * dt) * pi;
    covariance_.bottomRightCorner<3, 3>().diagonal().array() += params_.b_gyr_cov * dt * dt;
    projectCovariance(pi);
  }

  void closeWindow(const ImuTiltLegEdge & edge_now, bool scan_static, double t)
  {
    if (!initialised_)
      return;
    if (!has_window_start_) {
      resetWindow(edge_now, t);
      if (scan_static)
        start_edge_.velocity.setZero();
      has_window_start_ = true;
      return;
    }
    window_static_ = window_static_ && scan_static;
    if (t - window_start_ < params_.window_s)
      return;

    ++diagnostics_.windows;
    diagnostics_.samples = samples_;
    diagnostics_.residual_angle = 0.0;
    diagnostics_.nis = 0.0;
    diagnostics_.rejection = updateWindow(edge_now, t);
    diagnostics_.updated = diagnostics_.rejection == ImuTiltWindowRejection::kNone;
    resetWindow(edge_now, t);
  }

  Eigen::Vector3d downBodyAt(double t) const
  {
    const double dt = std::clamp(t - last_sample_time_, 0.0, last_period_);
    return rotationBack(last_rate_, dt) * down_;
  }
  bool initialised() const { return initialised_; }
  const Eigen::Vector3d & bias() const { return bias_; }
  Eigen::Matrix<double, 6, 1> covarianceDiagonal() const { return covariance_.diagonal(); }
  const ImuTiltWindowDiagnostics & diagnostics() const { return diagnostics_; }

private:
  using Matrix6 = Eigen::Matrix<double, 6, 6>;

  static Eigen::Quaterniond rotationBack(const Eigen::Vector3d & rate, double dt)
  {
    const Eigen::Vector3d radians = -dt * rate;
    const double angle = radians.norm();
    if (angle == 0.0)
      return Eigen::Quaterniond::Identity();
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, radians / angle));
  }

  Eigen::Matrix3d tiltProjector() const { return Eigen::Matrix3d::Identity() - down_ * down_.transpose(); }

  void projectCovariance(const Eigen::Matrix3d & pi)
  {
    covariance_.topLeftCorner<3, 3>() = (pi * covariance_.topLeftCorner<3, 3>() * pi).eval();
    covariance_.topRightCorner<3, 3>() = (pi * covariance_.topRightCorner<3, 3>()).eval();
    covariance_.bottomLeftCorner<3, 3>() = covariance_.topRightCorner<3, 3>().transpose();
  }

  void resetWindow(const ImuTiltLegEdge & edge_now, double t)
  {
    sum_.setZero();
    samples_ = 0;
    start_edge_ = edge_now;
    window_start_ = t;
    window_static_ = true;
  }

  ImuTiltWindowRejection updateWindow(const ImuTiltLegEdge & edge_now, double t)
  {
    using Rejection = ImuTiltWindowRejection;
    if (samples_ < 20 || last_sample_ - first_sample_ < 0.7 * params_.window_s)
      return Rejection::kCoverage;

    Eigen::Vector3d force = sum_ / static_cast<double>(samples_);
    double sigma_lin = 0.0;
    if (!window_static_) {
      if (!start_edge_.valid || !edge_now.valid || !edge_now.stream_alive)
        return Rejection::kLegEdge;
      LevellingWindowScan edges;
      edges.t_start = window_start_;
      edges.t_end = t;
      edges.v_start = start_edge_.velocity;
      edges.v_end = edge_now.velocity;
      edges.var_start = start_edge_.variance;
      edges.var_end = edge_now.variance;
      const auto term = linearAccelerationTerm(edges, edges, false, params_.gravity_m_s2, params_.vel_sigma_scale);
      if (!term.valid)
        return Rejection::kLegEdge;
      force -= term.accel * (params_.gravity_raw / params_.gravity_m_s2);
      sigma_lin = term.sigma;
    }

    const double norm = force.norm();
    const double g_ref = params_.mag_ref > 1e-3 ? params_.mag_ref : params_.gravity_raw;
    if (!(g_ref > 1e-3 && norm > 1e-6 && std::abs(norm - g_ref) <= params_.accel_tol * g_ref))
      return Rejection::kMagnitude;
    const Eigen::Vector3d measurement = -force / norm;
    diagnostics_.residual_angle = std::atan2(measurement.cross(down_).norm(), measurement.dot(down_));
    if ((measurement - down_).norm() > 0.5)
      return Rejection::kOutlier;

    const double sigma = levellingSigma(
      LevellingSigmaPolicy::kMeasurement, false, window_static_, 0.0, 0.0, params_.noise_static, params_.noise_moving);
    const double noise = sigma * sigma + sigma_lin * sigma_lin;
    const Eigen::Matrix3d pi = tiltProjector();
    const Eigen::Vector3d residual = pi * (measurement - down_);
    Eigen::Matrix3d innovation_cov = pi * covariance_.topLeftCorner<3, 3>() * pi;
    innovation_cov.diagonal().array() += noise;
    const Eigen::LLT<Eigen::Matrix3d> solve(innovation_cov);
    if (solve.info() != Eigen::Success)
      return Rejection::kCovariance;

    diagnostics_.nis = residual.dot(solve.solve(residual));
    const Eigen::Matrix<double, 6, 3> pht = covariance_.leftCols<3>() * pi;
    const Eigen::Matrix<double, 6, 3> gain = solve.solve(pht.transpose()).transpose();
    const Eigen::Matrix<double, 6, 1> delta = gain * residual;
    down_ = (down_ + delta.head<3>()).normalized();
    bias_ += delta.tail<3>();
    Matrix6 update = Matrix6::Identity();
    update.leftCols<3>() -= gain * pi;
    covariance_ = (update * covariance_ * update.transpose() + noise * gain * gain.transpose()).eval();
    projectCovariance(tiltProjector());
    return Rejection::kNone;
  }

  ImuTiltReferenceParams params_;
  Eigen::Vector3d down_ = -Eigen::Vector3d::UnitZ();
  Eigen::Vector3d bias_ = Eigen::Vector3d::Zero();
  Matrix6 covariance_ = Matrix6::Zero();
  bool initialised_ = false;
  Eigen::Vector3d last_rate_ = Eigen::Vector3d::Zero();
  double last_sample_time_ = 0.0;
  double last_period_ = 0.0;

  Eigen::Vector3d sum_ = Eigen::Vector3d::Zero();
  ImuTiltLegEdge start_edge_;
  bool has_window_start_ = false;
  bool window_static_ = true;
  double window_start_ = 0.0;
  double first_sample_ = 0.0;
  double last_sample_ = 0.0;
  int samples_ = 0;
  ImuTiltWindowDiagnostics diagnostics_;
};
}  // namespace fast_lio

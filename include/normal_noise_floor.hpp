// normal_noise_floor.hpp — how much position information a scan's plane normals can
// fake on their own, and how much of each direction is left after taking it out.
//
// Every scan point is matched to a plane fitted to its nearest map points
// (plane_fit.hpp). The fitted normal is noisy: a 2-3 cm residual over a 0.3-0.4 m
// patch tilts it by 6-10 degrees. In a corridor the true along-corridor information is
// close to zero, yet each tilted normal carries a little of it, and over hundreds of
// points those bits add up to several normal equivalents per frame. Because the same
// map patches answer every frame, that "measurement" does not average out: it holds the
// along-corridor position, and with it the velocity, at the map. That is the noise
// floor, and the information a direction has must be judged against it.
//
// The first-order covariance of the total-least-squares normal of a Count-point patch:
//
//     Cov(n) = sigma^2 · sum_{m=1,2} e_m e_m^T / lambda_m,   sigma^2 = lambda_0 / (Count - 3)
//
// lambda_0 <= lambda_1 <= lambda_2 are the eigenvalues of the patch's scatter matrix,
// e_1 and e_2 its in-plane eigenvectors, and Count - 3 the degrees of freedom left after
// fitting a plane's three parameters. For each eigenvector v_k of the Schur-complemented
// position information M_eff = V diag(lambda) V^T the floor is expected to contribute
//
//     f_k = kappa · sum_i w_i^2 · v_k^T Cov(n_i) v_k
//
// with w_i the point's row weight and kappa the model-check factor. Deducting the floor
// and a margin z_alpha · sigma_rel for kappa's own uncertainty leaves the keep gain
//
//     g_k = sqrt( max(lambda_k - f_k (1 + z_alpha sigma_rel), 0) / lambda_k ),
//
// so the kept information g_k^2 lambda_k is the deducted one. g_k is exactly zero up to
// the margin and tends to one where lambda_k >> f_k. G = V diag(g) V^T applies the gains
// to the position columns of the measurement Jacobian: a zero gain removes the direction
// from every row exactly, not down to some epsilon.
//
// Pure C++ + Eigen. No ROS.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "plane_fit.hpp"

namespace fast_lio
{

/// Parameters of the floor decision.
struct FloorParams
{
  /// Rate at which a direction made only of floor is kept anyway, per direction and
  /// frame. Each false keep can pin up to one frame of motion: 1e-3 keeps the expected
  /// cost of a 60 s window at about 5 cm, inside the 7-10 cm budget.
  double alpha = 1e-3;
  /// Scale of the floor model; nominal until measured on the O-arm replay (design F0).
  double kappa = 1.0;
  /// Relative uncertainty of kappa; the margin is z(alpha) times this.
  double rel_uncertainty = 0.3;
};

/// First-order covariance of the unit normal of an accepted plane fit. nullopt marks a
/// fit that cannot carry one: at most three points (no residual degree of freedom),
/// non-finite data, eigenvalues out of order, or no in-plane spread (lambda1 <= 0).
/// Coplanar points (lambda0 = 0) give the zero matrix. A negative lambda0 within the
/// fit's own rank tolerance is the rounding of a positive semi-definite matrix and reads
/// as coplanar; a larger one is not a scatter matrix and is refused.
inline std::optional<Eigen::Matrix3d> planeNormalCovariance(const PlaneFitEigen & fit)
{
  if (fit.count <= 3 || !std::isfinite(fit.lambda0) || !std::isfinite(fit.lambda1) || !std::isfinite(fit.lambda2) ||
      !fit.e1.allFinite() || !fit.e2.allFinite()) {
    return std::nullopt;
  }
  if (!(fit.lambda1 > 0.0) || fit.lambda1 > fit.lambda2 || fit.lambda0 > fit.lambda1) {
    return std::nullopt;
  }
  // The same numerical rank tolerance fitPlane applies to lambda1.
  const double rounding = std::numeric_limits<double>::epsilon() * fit.count * fit.lambda2;
  if (fit.lambda0 < -rounding) {
    return std::nullopt;
  }
  const double sigma2 = std::max(fit.lambda0, 0.0) / static_cast<double>(fit.count - 3);
  return Eigen::Matrix3d(sigma2 *
                         (fit.e1 * fit.e1.transpose() / fit.lambda1 + fit.e2 * fit.e2.transpose() / fit.lambda2));
}

/// Accumulates the floor f_k of one scan over its points, for the eigenvectors (the
/// columns of V) of that scan's M_eff. A point whose contribution is unknown, because
/// its plane carries no covariance or its weight is not finite, is not guessed: it is
/// counted apart and left to the caller.
class DirectionFloorAccumulator
{
public:
  explicit DirectionFloorAccumulator(const Eigen::Matrix3d & eigenvectors) : eigenvectors_(eigenvectors) {}

  void add(double weight, const std::optional<Eigen::Matrix3d> & normal_covariance)
  {
    ++points_;
    if (!normal_covariance || !normal_covariance->allFinite() || !std::isfinite(weight)) {
      ++points_without_covariance_;
      return;
    }
    const double w2 = weight * weight;
    for (int k = 0; k < 3; ++k) {
      const Eigen::Vector3d v = eigenvectors_.col(k);
      sum_(k) += w2 * v.dot(*normal_covariance * v);
    }
  }

  /// f_k = kappa · sum_i w_i^2 v_k^T Cov(n_i) v_k over the points with a covariance.
  Eigen::Vector3d floor(double kappa) const { return kappa * sum_; }
  int points() const { return points_; }
  int pointsWithoutCovariance() const { return points_without_covariance_; }

private:
  Eigen::Matrix3d eigenvectors_;
  Eigen::Vector3d sum_ = Eigen::Vector3d::Zero();
  int points_ = 0;
  int points_without_covariance_ = 0;
};

/// g_k = sqrt(max(lambda_k - f_k (1 + z_alpha sigma_rel), 0) / lambda_k), evaluated as
/// sqrt(max(1 - c_k / lambda_k, 0)): every operation in that order is monotone under
/// correct rounding, so g is non-decreasing in lambda and non-increasing in f exactly,
/// not just up to an ulp. A direction with no information (lambda_k <= 0), a non-finite
/// lambda_k, or a floor that is not a finite non-negative number keeps nothing; so does
/// every direction when the margin is not a finite non-negative number.
inline Eigen::Vector3d floorKeepGains(const Eigen::Vector3d & lambda,
                                      const Eigen::Vector3d & noise_floor,
                                      double z_alpha,
                                      double rel_uncertainty)
{
  Eigen::Vector3d gains = Eigen::Vector3d::Zero();
  const double margin = z_alpha * rel_uncertainty;
  if (!std::isfinite(margin) || !(margin >= 0.0)) {
    return gains;
  }
  for (int k = 0; k < 3; ++k) {
    if (!std::isfinite(lambda(k)) || !(lambda(k) > 0.0) || !std::isfinite(noise_floor(k)) || !(noise_floor(k) >= 0.0)) {
      continue;
    }
    const double deducted = noise_floor(k) * (1.0 + margin);
    gains(k) = std::sqrt(std::max(1.0 - deducted / lambda(k), 0.0));
  }
  return gains;
}

/// G = V diag(g) V^T, built as sum_k g_k v_k v_k^T: exactly symmetric, and a zero gain
/// adds nothing at all, so G v_k = g_k v_k up to the orthonormality of V.
inline Eigen::Matrix3d projectionFromGains(const Eigen::Matrix3d & eigenvectors, const Eigen::Vector3d & gains)
{
  Eigen::Matrix3d projection = Eigen::Matrix3d::Zero();
  for (int k = 0; k < 3; ++k) {
    if (gains(k) != 0.0) {
      const Eigen::Vector3d v = eigenvectors.col(k);
      // Evaluated before scaling: Eigen rewrites g * (v v^T) as (g v) v^T, whose (i, j)
      // and (j, i) round differently.
      const Eigen::Matrix3d outer = v * v.transpose();
      projection += gains(k) * outer;
    }
  }
  return projection;
}

/// One-sided standard normal quantile: the z with P(Z > z) = alpha, for alpha in
/// (0, 0.5); nullopt outside. Abramowitz & Stegun 26.2.23 (|error| < 4.5e-4) seeds
/// Newton steps on the exact tail 0.5 erfc(z / sqrt 2); four steps reach rounding.
inline std::optional<double> zFromFalseKeepRate(double alpha)
{
  if (!(alpha > 0.0 && alpha < 0.5)) {
    return std::nullopt;
  }
  constexpr double kPi = 3.14159265358979323846;
  const double t = std::sqrt(-2.0 * std::log(alpha));
  double z =
    t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
  for (int step = 0; step < 4; ++step) {
    const double tail = 0.5 * std::erfc(z / std::sqrt(2.0));
    const double density = std::exp(-0.5 * z * z) / std::sqrt(2.0 * kPi);
    z += (tail - alpha) / density;
  }
  if (!std::isfinite(z)) {
    return std::nullopt;
  }
  return z;
}

}  // namespace fast_lio

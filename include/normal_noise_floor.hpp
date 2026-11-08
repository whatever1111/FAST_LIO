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
// The residual has to follow. Rows whose position columns are H_t G but whose residual
// still answers to H_t make the iterated update correct the mean with g_k lambda_k while
// the covariance takes g_k^2 lambda_k, and with a wide prior the iterations oscillate
// apart. So the residual is first shifted by what the removed information reads: the
// scan's least-squares position reading, rotation profiled out, times (I - G). The rows
// then form a linear model of their own, which the update solves in one step and absorbs
// with g_k^2 lambda_k for mean and covariance alike (applyFloorProjection).
//
// Pure C++ + Eigen. No ROS.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "degeneracy_policy.hpp"
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

/// The residual shift d that goes with the projection G = V diag(g) V^T:
///
///     d = V coef,  coef_k = (1 - g_k) beta_k / lambda_k,  beta = V^T b_eff,
///     b_eff = b_t - M_tr (M_rr + (ridge trace(M_rr) / 3) I)^-1 b_r,
///
/// with b_t = H_t^T h and b_r = H_r^T h from the unprojected position columns and the
/// unshifted residual, M_tr and M_rr the blocks given to schurPositionInformation, and
/// ridge as given to it (b_eff = b_t when trace(M_rr) <= 0, as M_eff = H_tt there). V and
/// lambda are M_eff's eigenvectors and ascending eigenvalues, g the keep gains, rows the
/// scan's row count N. h - H_t d then has the profiled information vector
/// b_eff - M_eff d = V diag(g) beta: nothing along a removed direction, g_k beta_k along a
/// partly kept one, so the update corrects with the information it is told about.
///
/// A kept direction (g_k == 1.0) is not shifted, so a scan without floor keeps its
/// residual bit for bit; neither is a direction with no information above the rank
/// tolerance eps N lambda_max (fitPlane's rule), where beta_k / lambda_k is rounding over
/// rounding. Any non-finite input, or rows < 0, gives the zero vector: the residual stays.
inline Eigen::Vector3d floorResidualShift(const Eigen::Matrix3d & eigenvectors,
                                          const Eigen::Vector3d & lambda,
                                          const Eigen::Vector3d & gains,
                                          const Eigen::Vector3d & b_t,
                                          const Eigen::Vector3d & b_r,
                                          const Eigen::Matrix3d & M_tr,
                                          const Eigen::Matrix3d & M_rr,
                                          double ridge,
                                          int rows)
{
  const Eigen::Vector3d none = Eigen::Vector3d::Zero();
  if (!eigenvectors.allFinite() || !lambda.allFinite() || !gains.allFinite() || !b_t.allFinite() || !b_r.allFinite() ||
      !M_tr.allFinite() || !M_rr.allFinite() || !std::isfinite(ridge) || rows < 0) {
    return none;
  }
  Eigen::Vector3d b_eff = b_t;
  const double trace_rr = M_rr.trace();
  if (trace_rr > 0.0) {
    const Eigen::Matrix3d rr = M_rr + std::max(ridge, 0.0) * trace_rr / 3.0 * Eigen::Matrix3d::Identity();
    const Eigen::Vector3d coupled = M_tr * rr.ldlt().solve(b_r);
    // schurPositionInformation leaves M_eff = H_tt when its product is not finite; so does this.
    if (coupled.allFinite()) {
      b_eff -= coupled;
    }
  }
  const Eigen::Vector3d beta = eigenvectors.transpose() * b_eff;
  const double rank_tolerance = std::numeric_limits<double>::epsilon() * rows * lambda.maxCoeff();
  Eigen::Vector3d coef = Eigen::Vector3d::Zero();
  for (int k = 0; k < 3; ++k) {
    if (gains(k) == 1.0 || !(lambda(k) > 0.0) || !(lambda(k) > rank_tolerance)) {
      continue;
    }
    coef(k) = (1.0 - gains(k)) * beta(k) / lambda(k);
  }
  if ((coef.array() == 0.0).all()) {
    return none;
  }
  const Eigen::Vector3d shift = eigenvectors * coef;
  return shift.allFinite() ? shift : none;
}

/// What applyFloorProjection did to one linearisation of the scan's rows. With valid
/// false nothing was done, and the fields say so: gains one, shift zero.
struct FloorProjectionResult
{
  bool valid = false;
  Eigen::Vector3d lambda = Eigen::Vector3d::Zero();            ///< M_eff's eigenvalues, ascending
  Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();  ///< column k belongs to lambda(k)
  Eigen::Vector3d noise_floor = Eigen::Vector3d::Zero();       ///< f, kappa included
  Eigen::Vector3d gains = Eigen::Vector3d::Ones();             ///< g
  Eigen::Vector3d shift = Eigen::Vector3d::Zero();             ///< d, in the position frame
  int points = 0;
  int points_without_covariance = 0;
};

/// The floor projection on one linearisation of the scan's point-to-plane rows. From
/// the rows as built (h_x N x 12: position columns 0-2, rotation 3-5, extrinsics 6-11):
///
///   1. M_tt, M_tr, M_rr from the position and rotation columns;
///   2. M_eff = schurPositionInformation(M_tt, M_tr, M_rr, ridge), its V and lambda;
///   3. the floor f from each row's weight and normal covariance, g, G;
///   4. d = floorResidualShift(...) from H_t^T h and H_r^T h;
///   5. h <- h - H_t d, then h_x.leftCols(3) <- H_t G.
///
/// The shift comes first because both use the position columns the rows came with; the
/// rotation and extrinsic columns are not touched. The rows and the linearisation point
/// move between iterations of the update, so this runs on every one of them. weights[i]
/// is the weight row i already carries (applied to its h_x row and h); a row whose weight
/// or covariance is missing or not finite stays out of the floor and is counted.
///
/// h_x must be N x 12 with h, weights and normal_cov of length N; anything else is a
/// programming error and throws std::invalid_argument. Non-finite rows, a parameter set
/// outside its range (alpha in (0, 0.5); kappa, rel_uncertainty and ridge finite, kappa
/// and rel_uncertainty >= 0) or a failed eigen decomposition leave the rows as they are
/// and return valid false. When every gain is exactly one (no floor anywhere) h and h_x
/// stay bit for bit.
inline FloorProjectionResult applyFloorProjection(Eigen::Ref<Eigen::MatrixXd> h_x,
                                                  Eigen::Ref<Eigen::VectorXd> h,
                                                  const std::vector<double> & weights,
                                                  const std::vector<std::optional<Eigen::Matrix3d>> & normal_cov,
                                                  const FloorParams & params,
                                                  double ridge)
{
  const Eigen::Index rows = h_x.rows();
  if (h_x.cols() != 12 || h.size() != rows || static_cast<Eigen::Index>(weights.size()) != rows ||
      static_cast<Eigen::Index>(normal_cov.size()) != rows) {
    throw std::invalid_argument("applyFloorProjection: h_x must be N x 12, h, weights and normal_cov of length N");
  }
  FloorProjectionResult result;
  const std::optional<double> z_alpha = zFromFalseKeepRate(params.alpha);
  if (!z_alpha || !std::isfinite(params.kappa) || !(params.kappa >= 0.0) || !std::isfinite(params.rel_uncertainty) ||
      !(params.rel_uncertainty >= 0.0) || !std::isfinite(ridge) || !h_x.allFinite() || !h.allFinite()) {
    return result;
  }
  const Eigen::MatrixXd H_t = h_x.leftCols(3);
  const Eigen::MatrixXd H_r = h_x.middleCols(3, 3);
  const Eigen::Matrix3d M_tt = H_t.transpose() * H_t;
  const Eigen::Matrix3d M_tr = H_t.transpose() * H_r;
  const Eigen::Matrix3d M_rr = H_r.transpose() * H_r;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(schurPositionInformation(M_tt, M_tr, M_rr, ridge));
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() || !solver.eigenvectors().allFinite()) {
    return result;
  }
  DirectionFloorAccumulator accumulator(solver.eigenvectors());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    accumulator.add(weights[i], normal_cov[i]);
  }
  const Eigen::Vector3d noise_floor = accumulator.floor(params.kappa);
  const Eigen::Vector3d gains = floorKeepGains(solver.eigenvalues(), noise_floor, *z_alpha, params.rel_uncertainty);
  const Eigen::Vector3d b_t = H_t.transpose() * h;
  const Eigen::Vector3d b_r = H_r.transpose() * h;
  const Eigen::Vector3d shift = floorResidualShift(
    solver.eigenvectors(), solver.eigenvalues(), gains, b_t, b_r, M_tr, M_rr, ridge, static_cast<int>(rows));
  if (!(shift.array() == 0.0).all()) {
    h -= H_t * shift;
  }
  if (!(gains.array() == 1.0).all()) {
    h_x.leftCols(3) = H_t * projectionFromGains(solver.eigenvectors(), gains);
  }
  result.valid = true;
  result.lambda = solver.eigenvalues();
  result.eigenvectors = solver.eigenvectors();
  result.noise_floor = noise_floor;
  result.gains = gains;
  result.shift = shift;
  result.points = accumulator.points();
  result.points_without_covariance = accumulator.pointsWithoutCovariance();
  return result;
}

}  // namespace fast_lio

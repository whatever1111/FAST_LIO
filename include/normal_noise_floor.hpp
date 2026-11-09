// normal_noise_floor.hpp — TLS normal noise and the joint-row information floor.
// S0 sums complete weighted normal covariances. On the exact rotation complement,
// H_perp = Q_p R_p, F_q = sqrt((I - a R_p^-T kappa S0 R_p^-1)_+).
// Every Jacobian column and the residual receive the same bounded row map
// L = I - Q_p (I - F_q) Q_p^T. Only thin QR factors are formed, never an N x N L
// or a position-coordinate G. Thus the entire measurement Gram can only decrease.
// Removed modes combine position and rotation. Pure-position information can remain.
// Pure C++ + Eigen. No ROS.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "degeneracy_policy.hpp"
#include "plane_fit.hpp"

namespace fast_lio
{

/// What the lidar update does with the floor (degeneracy_floor_mode).
enum class FloorMode
{
  kOff,      ///< "off": nothing is computed; the rows stay as built
  kObserve,  ///< "observe": the decision is computed and reported; the rows stay as built
  kApply,    ///< "apply": the decision is applied to the rows (applyFloorProjection)
};

/// The mode a parameter value names; nullopt for anything but "off", "observe" and "apply".
inline std::optional<FloorMode> floorModeFromName(const std::string & name)
{
  if (name == "off") {
    return FloorMode::kOff;
  }
  if (name == "observe") {
    return FloorMode::kObserve;
  }
  if (name == "apply") {
    return FloorMode::kApply;
  }
  return std::nullopt;
}

inline const char * floorModeName(FloorMode mode)
{
  switch (mode) {
    case FloorMode::kObserve:
      return "observe";
    case FloorMode::kApply:
      return "apply";
    case FloorMode::kOff:
      break;
  }
  return "off";
}

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

/// Complete S0 = sum_i w_i^2 Cov(n_i), before kappa or the uncertainty margin.
/// Missing/non-finite contributions are counted separately, without guessing a floor.
class FloorAccumulator
{
public:
  void add(double weight, const std::optional<Eigen::Matrix3d> & normal_covariance)
  {
    ++points_;
    if (!normal_covariance || !normal_covariance->allFinite() || !std::isfinite(weight)) {
      ++points_without_covariance_;
      return;
    }
    const double w2 = weight * weight;
    sum_ += w2 * *normal_covariance;
  }

  const Eigen::Matrix3d & matrix() const { return sum_; }
  Eigen::Matrix3d floor(double kappa) const { return kappa * sum_; }
  int points() const { return points_; }
  int pointsWithoutCovariance() const { return points_without_covariance_; }

private:
  Eigen::Matrix3d sum_ = Eigen::Matrix3d::Zero();
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

/// Result retained for telemetry. QR storage belongs only to the current decision.
/// Neutral lambda/gains preserve the floor-off h_exit.v1 contract.
struct FloorProjectionResult
{
  bool valid = false;
  const char * rule = "joint_row_v1";
  const char * reason = "bad_input";
  Eigen::Vector3d lambda = Eigen::Vector3d::Zero();
  Eigen::Vector3d gains = Eigen::Vector3d::Ones();
  Eigen::Vector3d tau = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  int spectral_count = 0;
  int rank_m = 0;
  int rank_r = 0;
  int points = 0;
  int points_without_covariance = 0;
  double kappa = std::numeric_limits<double>::quiet_NaN();
  double sigma_rel = std::numeric_limits<double>::quiet_NaN();
  double z = std::numeric_limits<double>::quiet_NaN();
  double a = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix3d M0 = Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d M_ridge = M0;
  Eigen::Matrix3d S0 = M0;
  Eigen::Matrix3d K = M0;
  Eigen::Matrix3d F_sym = M0;
  Eigen::Matrix3d A = M0;
  Eigen::Matrix3d B = M0;
  Eigen::Matrix3d C = M0;
  Eigen::Matrix3d A_keep = M0;
  Eigen::Matrix3d B_keep = M0;
  Eigen::Matrix3d C_keep = M0;
  bool shift_coords_valid = false;
  Eigen::Vector3d d_t = Eigen::Vector3d::Zero();
  Eigen::Vector3d d_r = Eigen::Vector3d::Zero();
  double dh_norm = 0.0;
};

/// Iteration-local factors. No factor is stored in the measurement object's state.
struct FloorDecision
{
  FloorProjectionResult result;
  Eigen::MatrixXd Q_p;
  Eigen::MatrixXd rotation_basis;
  Eigen::Vector3d rotation_values = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_p = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d F_q = Eigen::Matrix3d::Identity();
};

inline Eigen::Matrix3d floorSymmetric(const Eigen::Matrix3d & matrix)
{
  return 0.5 * matrix + 0.5 * matrix.transpose();
}

/// Numerical support and rounding-only PSD correction use epsilon * N * lambda_max.
inline bool floorSpectrum(const Eigen::Matrix3d & matrix,
                          Eigen::Index rows,
                          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> & solver,
                          int & rank,
                          const char *& reason)
{
  if (!matrix.allFinite()) {
    reason = "nonfinite";
    return false;
  }
  solver.compute(floorSymmetric(matrix));
  if (solver.info() != Eigen::Success) {
    reason = "decomposition";
    return false;
  }
  if (!solver.eigenvalues().allFinite() || !solver.eigenvectors().allFinite()) {
    reason = "nonfinite";
    return false;
  }
  const double tolerance = std::numeric_limits<double>::epsilon() * rows * solver.eigenvalues().cwiseAbs().maxCoeff();
  if (solver.eigenvalues().minCoeff() < -tolerance) {
    reason = "not_psd";
    return false;
  }
  rank = static_cast<int>((solver.eigenvalues().array() > tolerance).count());
  return true;
}

/// Exact rotation support, thin QR complement, and the diagnostic ridge Schur.
inline bool floorGeometry(const Eigen::Ref<const Eigen::MatrixXd> & H,
                          double ridge,
                          FloorDecision & decision,
                          Eigen::MatrixXd & perpendicular,
                          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> & position_solver)
{
  auto & r = decision.result;
  const Eigen::MatrixXd H_t = H.leftCols(3);
  const Eigen::MatrixXd H_r = H.middleCols(3, 3);
  r.A = H_t.transpose() * H_t;
  r.B = H_t.transpose() * H_r;
  r.C = H_r.transpose() * H_r;
  if (!r.A.allFinite() || !r.B.allFinite()) {
    r.reason = "nonfinite";
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rotation_solver;
  if (!floorSpectrum(r.C, H.rows(), rotation_solver, r.rank_r, r.reason)) {
    return false;
  }
  decision.rotation_basis = rotation_solver.eigenvectors().rightCols(r.rank_r);
  decision.rotation_values = rotation_solver.eigenvalues();
  perpendicular = H_t;
  if (r.rank_r > 0) {
    const Eigen::MatrixXd supported = H_r * decision.rotation_basis;
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(supported);
    const Eigen::MatrixXd Q_r = qr.householderQ() * Eigen::MatrixXd::Identity(H.rows(), r.rank_r);
    perpendicular -= Q_r * (Q_r.transpose() * H_t);
  }
  r.M0 = floorSymmetric(perpendicular.transpose() * perpendicular);
  r.M_ridge = schurPositionInformation(r.A, r.B, r.C, ridge);
  if (!r.M_ridge.allFinite() || !floorSpectrum(r.M0, H.rows(), position_solver, r.rank_m, r.reason)) {
    if (!r.M_ridge.allFinite()) {
      r.reason = "nonfinite";
    }
    return false;
  }
  r.lambda = position_solver.eigenvalues().cwiseMax(0.0);
  return true;
}

/// Evaluate the candidate without touching the caller's rows or computing a shift.
inline FloorDecision decideFloor(const Eigen::Ref<const Eigen::MatrixXd> & H,
                                 const Eigen::Ref<const Eigen::VectorXd> & h,
                                 const std::vector<double> & weights,
                                 const std::vector<std::optional<Eigen::Matrix3d>> & normal_cov,
                                 const FloorParams & params,
                                 double ridge)
{
  if (H.cols() != 12 || h.size() != H.rows() || static_cast<Eigen::Index>(weights.size()) != H.rows() ||
      static_cast<Eigen::Index>(normal_cov.size()) != H.rows()) {
    throw std::invalid_argument("decideFloor: H must be N x 12, h, weights and normal_cov of length N");
  }
  FloorDecision decision;
  auto & r = decision.result;
  r.lambda.setConstant(std::numeric_limits<double>::quiet_NaN());
  r.gains = r.lambda;
  r.d_t = r.lambda;
  r.d_r = r.lambda;
  FloorAccumulator accumulator;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    accumulator.add(weights[i], normal_cov[i]);
  }
  r.S0 = floorSymmetric(accumulator.matrix());
  r.points = accumulator.points();
  r.points_without_covariance = accumulator.pointsWithoutCovariance();
  const auto z = zFromFalseKeepRate(params.alpha);
  if (!z || !std::isfinite(params.kappa) || params.kappa < 0.0 || !std::isfinite(params.rel_uncertainty) ||
      params.rel_uncertainty < 0.0 || !std::isfinite(ridge)) {
    return decision;
  }
  r.kappa = params.kappa;
  r.sigma_rel = params.rel_uncertainty;
  r.z = *z;
  r.a = 1.0 + *z * params.rel_uncertainty;
  if (!H.allFinite() || !h.allFinite() || !std::isfinite(r.a)) {
    r.reason = "nonfinite";
    return decision;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> floor_solver;
  int floor_rank = 0;
  if (!floorSpectrum(r.S0, H.rows(), floor_solver, floor_rank, r.reason)) {
    return decision;
  }
  const bool zero_floor = params.kappa == 0.0 || (r.S0.array() == 0.0).all();
  Eigen::MatrixXd perpendicular;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> position_solver;
  const bool geometry_valid = floorGeometry(H, ridge, decision, perpendicular, position_solver);
  // A Gram matrix cannot have more supported columns than measurement rows.
  r.rank_m = static_cast<int>(std::min<Eigen::Index>(r.rank_m, H.rows()));
  if (zero_floor) {
    r.valid = true;
    r.reason = "zero_floor";
    r.K = r.M0;
    r.F_sym.setIdentity();
    r.A_keep = r.A;
    r.B_keep = r.B;
    r.C_keep = r.C;
    r.spectral_count = geometry_valid ? r.rank_m : 0;
    r.tau.setZero();
    r.gains.setOnes();
    return decision;
  }
  if (!geometry_valid) {
    return decision;
  }
  if (r.rank_m < 3) {
    r.reason = "rank_m";
    return decision;
  }
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(perpendicular);
  decision.Q_p = qr.householderQ() * Eigen::MatrixXd::Identity(H.rows(), 3);
  decision.R_p = qr.matrixQR().topLeftCorner<3, 3>().template triangularView<Eigen::Upper>();
  // R^-T S R^-1 using two triangular solves, without a coordinate inverse.
  const Eigen::Matrix3d S = params.kappa * r.S0;
  const Eigen::Matrix3d left = decision.R_p.transpose().triangularView<Eigen::Lower>().solve(S);
  const Eigen::Matrix3d T = decision.R_p.transpose().triangularView<Eigen::Lower>().solve(left.transpose()).transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> tau_solver;
  int tau_rank = 0;
  if (!S.allFinite() || !left.allFinite() || !floorSpectrum(T, H.rows(), tau_solver, tau_rank, r.reason)) {
    if (!S.allFinite() || !left.allFinite()) {
      r.reason = "nonfinite";
    }
    return decision;
  }
  r.tau = tau_solver.eigenvalues().cwiseMax(0.0);
  const Eigen::Vector3d deducted = r.a * r.tau;
  if (!deducted.allFinite()) {
    r.reason = "nonfinite";
    return decision;
  }
  r.gains = (Eigen::Vector3d::Ones() - deducted).cwiseMax(0.0).cwiseSqrt();
  decision.F_q = projectionFromGains(tau_solver.eigenvectors(), r.gains);
  r.K = floorSymmetric(decision.R_p.transpose() * decision.F_q * decision.F_q * decision.R_p);
  const Eigen::Matrix3d inverse_root = position_solver.eigenvectors() *
                                       r.lambda.cwiseSqrt().cwiseInverse().asDiagonal() *
                                       position_solver.eigenvectors().transpose();
  const Eigen::Matrix3d basis = decision.R_p * inverse_root;
  r.F_sym = floorSymmetric(basis.transpose() * decision.F_q * basis);
  const Eigen::MatrixXd candidate = H.leftCols(6) - decision.Q_p * ((Eigen::Matrix3d::Identity() - decision.F_q) *
                                                                    (decision.Q_p.transpose() * H.leftCols(6)));
  if (!r.K.allFinite() || !r.F_sym.allFinite() || !candidate.allFinite()) {
    r.reason = "nonfinite";
    return decision;
  }
  r.A_keep = candidate.leftCols(3).transpose() * candidate.leftCols(3);
  r.B_keep = candidate.leftCols(3).transpose() * candidate.rightCols(3);
  r.C_keep = candidate.rightCols(3).transpose() * candidate.rightCols(3);
  if (!r.A_keep.allFinite() || !r.B_keep.allFinite() || !r.C_keep.allFinite()) {
    r.reason = "nonfinite";
    return decision;
  }
  r.valid = true;
  r.reason = "ok";
  r.spectral_count = 3;
  return decision;
}

/// Coordinate-only diagnostics. Overflow here never invalidates a finite row map.
inline void
floorShiftCoordinates(const FloorDecision & decision, const Eigen::Vector3d & bh, FloorProjectionResult & result)
{
  result.d_t = decision.R_p.triangularView<Eigen::Upper>().solve(bh);
  const Eigen::VectorXd rhs = decision.rotation_basis.transpose() * result.B.transpose() * result.d_t;
  const Eigen::VectorXd supported = rhs.array() / decision.rotation_values.tail(result.rank_r).array();
  result.d_r = -decision.rotation_basis * supported;
  result.shift_coords_valid = result.d_t.allFinite() && result.d_r.allFinite();
  if (!result.shift_coords_valid) {
    result.d_t.setConstant(std::numeric_limits<double>::quiet_NaN());
    result.d_r = result.d_t;
  }
}

struct FloorRowUpdate
{
  Eigen::MatrixXd H;
  Eigen::VectorXd h;
  Eigen::Vector3d bh = Eigen::Vector3d::Zero();
  double dh_norm = 0.0;
  bool valid = false;
};

/// Both modes validate the same candidate rows, using only the original inputs.
inline FloorRowUpdate stageFloorRows(const FloorDecision & decision,
                                     const Eigen::Ref<const Eigen::MatrixXd> & H,
                                     const Eigen::Ref<const Eigen::VectorXd> & h)
{
  FloorRowUpdate update;
  const Eigen::Matrix3d removed = Eigen::Matrix3d::Identity() - decision.F_q;
  const Eigen::MatrixXd BH = removed * (decision.Q_p.transpose() * H);
  update.bh = removed * (decision.Q_p.transpose() * h);
  if (!BH.allFinite() || !update.bh.allFinite()) {
    return update;
  }
  update.H = H - decision.Q_p * BH;
  const Eigen::VectorXd delta_h = decision.Q_p * update.bh;
  update.h = h - delta_h;
  update.dh_norm = delta_h.stableNorm();
  update.valid = update.H.allFinite() && update.h.allFinite() && std::isfinite(update.dh_norm);
  return update;
}

inline FloorProjectionResult refuseFloorUpdate(FloorProjectionResult result)
{
  result.valid = false;
  result.reason = "nonfinite";
  result.spectral_count = 0;
  result.K.setConstant(std::numeric_limits<double>::quiet_NaN());
  result.F_sym = result.K;
  result.A_keep = result.K;
  result.B_keep = result.K;
  result.C_keep = result.K;
  return result;
}

/// Stage both outputs from original rows and commit only when all are finite.
inline FloorProjectionResult applyFloorProjection(Eigen::Ref<Eigen::MatrixXd> H,
                                                  Eigen::Ref<Eigen::VectorXd> h,
                                                  const std::vector<double> & weights,
                                                  const std::vector<std::optional<Eigen::Matrix3d>> & normal_cov,
                                                  const FloorParams & params,
                                                  double ridge)
{
  const FloorDecision decision = decideFloor(H, h, weights, normal_cov, params, ridge);
  auto result = decision.result;
  if (!result.valid) {
    return result;
  }
  if ((result.gains.array() == 1.0).all()) {
    result.shift_coords_valid = true;
    result.d_t.setZero();
    result.d_r.setZero();
    return result;
  }
  const FloorRowUpdate update = stageFloorRows(decision, H, h);
  if (!update.valid) {
    return refuseFloorUpdate(result);
  }
  result.dh_norm = update.dh_norm;
  floorShiftCoordinates(decision, update.bh, result);
  H = update.H;
  h = update.h;
  return result;
}

inline FloorProjectionResult observeFloor(const Eigen::Ref<const Eigen::MatrixXd> & H,
                                          const Eigen::Ref<const Eigen::VectorXd> & h,
                                          const std::vector<double> & weights,
                                          const std::vector<std::optional<Eigen::Matrix3d>> & normal_cov,
                                          const FloorParams & params,
                                          double ridge)
{
  const FloorDecision decision = decideFloor(H, h, weights, normal_cov, params, ridge);
  if (decision.result.valid && !(decision.result.gains.array() == 1.0).all() && !stageFloorRows(decision, H, h).valid) {
    return refuseFloorUpdate(decision.result);
  }
  return decision.result;
}

struct FloorScanRecord
{
  FloorProjectionResult last;
  int evaluations = 0;
  int invalid_evaluations = 0;
  int evaluation_iter = 0;
  bool projected = false;
};

inline void floorStep(FloorMode mode,
                      Eigen::Ref<Eigen::MatrixXd> H,
                      Eigen::Ref<Eigen::VectorXd> h,
                      const std::vector<double> & weights,
                      const std::vector<std::optional<Eigen::Matrix3d>> & normal_cov,
                      const FloorParams & params,
                      double ridge,
                      FloorScanRecord & record)
{
  if (mode == FloorMode::kOff) {
    return;
  }
  record.last = mode == FloorMode::kApply ? applyFloorProjection(H, h, weights, normal_cov, params, ridge)
                                          : observeFloor(H, h, weights, normal_cov, params, ridge);
  ++record.evaluations;
  record.evaluation_iter = record.evaluations;
  if (!record.last.valid) {
    ++record.invalid_evaluations;
  }
  record.projected = mode == FloorMode::kApply && record.last.valid && !(record.last.gains.array() == 1.0).all();
}

/// Calibration validity is independent of whether the floor could be applied.
struct FloorCalibration
{
  double numerator = std::numeric_limits<double>::quiet_NaN();
  double denominator = std::numeric_limits<double>::quiet_NaN();
  double ratio = std::numeric_limits<double>::quiet_NaN();
  bool direction_valid = false;
  bool valid = false;
  int zero_denominator = 0;  ///< per-frame count, summed by the calibration reader
  const char * reason = "nonfinite";
};

inline FloorCalibration
floorCalibration(const FloorProjectionResult & result, const Eigen::Vector3d & direction, bool frame_matches)
{
  FloorCalibration calibration;
  if (!frame_matches) {
    calibration.reason = "frame_mismatch";
    return calibration;
  }
  calibration.direction_valid =
    direction.allFinite() && std::isfinite(direction.squaredNorm()) && direction.squaredNorm() > 0.0;
  if (!calibration.direction_valid || !result.M0.allFinite() || !result.S0.allFinite()) {
    return calibration;
  }
  calibration.numerator = direction.dot(result.M0 * direction);
  calibration.denominator = direction.dot(result.S0 * direction);
  if (!std::isfinite(calibration.numerator) || !std::isfinite(calibration.denominator)) {
    return calibration;
  }
  if (calibration.denominator == 0.0) {
    calibration.zero_denominator = 1;
    calibration.reason = "zero_denominator";
    return calibration;
  }
  if (calibration.denominator < 0.0) {
    calibration.reason = "not_psd";
    return calibration;
  }
  calibration.ratio = calibration.numerator / calibration.denominator;
  calibration.valid = std::isfinite(calibration.ratio);
  calibration.reason = calibration.valid ? "ok" : "nonfinite";
  return calibration;
}

inline std::string floorNumber(double value)
{
  if (!std::isfinite(value)) {
    return "nan";
  }
  char text[64];
  std::snprintf(text, sizeof text, "%.17g", value);
  return text;
}

template<typename Derived>
inline std::string floorArray(const Eigen::MatrixBase<Derived> & values, int count)
{
  std::string text = "[";
  for (int i = 0; i < count; ++i) {
    if (i > 0) {
      text += ' ';
    }
    text += floorNumber(values(i));
  }
  return text + ']';
}

inline std::string floorMatrix(const Eigen::Matrix3d & matrix, bool symmetric = true)
{
  Eigen::Matrix<double, 9, 1> values;
  int count = 0;
  for (int row = 0; row < 3; ++row) {
    for (int col = symmetric ? row : 0; col < 3; ++col) {
      values(count++) = matrix(row, col);
    }
  }
  return floorArray(values, count);
}

/// Schema 2, one final evaluation per frame, raw normal-equivalent row units.
/// u/stamp come from the same final snapshot and conversion used by /Odometry.
inline std::string floorLine(double lidar_end_time,
                             std::int64_t stamp_ns,
                             FloorMode mode,
                             const FloorScanRecord & record,
                             const Eigen::Vector3d & direction,
                             bool frame_matches)
{
  if (mode == FloorMode::kOff) {
    return {};
  }
  const auto & r = record.last;
  const auto calibration = floorCalibration(r, direction, frame_matches);
  char text[768];
  std::snprintf(text,
                sizeof text,
                "[FLOOR] schema=2 rule=%s mode=%s t=%.17g stamp_ns=%lld evaluation_iter=%d valid=%d reason=%s "
                "points=%d points_without_covariance=%d iterations=%d invalid_iterations=%d projected=%d "
                "rank_m=%d rank_r=%d",
                r.rule,
                floorModeName(mode),
                lidar_end_time,
                static_cast<long long>(stamp_ns),
                record.evaluation_iter,
                r.valid ? 1 : 0,
                r.reason,
                r.points,
                r.points_without_covariance,
                record.evaluations,
                record.invalid_evaluations,
                record.projected ? 1 : 0,
                r.rank_m,
                r.rank_r);
  std::string line(text);
  line += " kappa=" + floorNumber(r.kappa) + " sigma_rel=" + floorNumber(r.sigma_rel);
  line += " z=" + floorNumber(r.z) + " a=" + floorNumber(r.a);
  line += " m_definition=exact_rotation_schur units=normal_equivalent";
  line += " m_source=last_measurement_linearization";
  line += " M=" + floorMatrix(r.M0) + " M_ridge=" + floorMatrix(r.M_ridge) + " S0=" + floorMatrix(r.S0);
  line += " lambda=" + floorArray(r.lambda, r.M0.allFinite() ? 3 : 0);
  line += " tau=" + floorArray(r.tau, r.valid ? r.spectral_count : 0);
  line += " g_gen=" + floorArray(r.gains, r.valid ? r.spectral_count : 0);
  line += " K=" + floorMatrix(r.K) + " F_sym=" + floorMatrix(r.F_sym) + " transform_kind=row_contraction";
  const Eigen::Vector3d u =
    calibration.direction_valid ? direction : Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  line += " u=" + floorArray(u, 3) + " u_source=odom_frame_end u_valid=" + (calibration.direction_valid ? "1" : "0");
  line += " rD_num=" + floorNumber(calibration.numerator) + " rD_den=" + floorNumber(calibration.denominator);
  line += " rD=" + floorNumber(calibration.ratio) + " rD_valid=" + (calibration.valid ? "1" : "0");
  line += " rD_zero_denominator=" + std::to_string(calibration.zero_denominator);
  line += " rD_reason=" + std::string(calibration.reason);
  line += " A=" + floorMatrix(r.A) + " B=" + floorMatrix(r.B, false) + " C=" + floorMatrix(r.C);
  line += " A_keep=" + floorMatrix(r.A_keep) + " B_keep=" + floorMatrix(r.B_keep, false);
  line += " C_keep=" + floorMatrix(r.C_keep);
  if (mode == FloorMode::kApply) {
    line += " shift_kind=joint_schur shift_coords_valid=" + std::string(r.shift_coords_valid ? "1" : "0");
    line += " d_t=" + floorArray(r.d_t, 3) + " d_r=" + floorArray(r.d_r, 3) + " dh_norm=" + floorNumber(r.dh_norm);
  }
  return line;
}

}  // namespace fast_lio

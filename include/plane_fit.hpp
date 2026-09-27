#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstddef>
#include <limits>

namespace fast_lio
{

// Orthogonal least-squares fit to exactly the first Count points. Fixed-size
// storage avoids allocations per correspondence; coordinates are evaluated in
// double even when the point and output types use float. Failure preserves out.
template<int Count, typename Points, typename Scalar>
bool fitPlane(const Points & points, double threshold, Eigen::Matrix<Scalar, 4, 1> & out)
{
  static_assert(Count >= 3, "A plane needs at least three points");
  if (points.size() < static_cast<std::size_t>(Count) || !std::isfinite(threshold) || threshold < 0.0)
    return false;

  Eigen::Matrix<double, Count, 3> coordinates;
  for (int i = 0; i < Count; ++i) {
    coordinates.row(i) << static_cast<double>(points[i].x), static_cast<double>(points[i].y),
      static_cast<double>(points[i].z);
  }
  if (!coordinates.allFinite())
    return false;
  // Subtract a local reference before averaging to limit cancellation after a
  // large translation. Check overflow instead of admitting an unreadable fit.
  const Eigen::Vector3d reference = coordinates.row(0).transpose();
  coordinates.rowwise() -= reference.transpose();
  const Eigen::Vector3d localMean = coordinates.colwise().mean().transpose();
  const Eigen::Vector3d center = reference + localMean;
  coordinates.rowwise() -= localMean.transpose();
  const Eigen::Matrix3d scatter = coordinates.transpose() * coordinates;
  if (!center.allFinite() || !scatter.allFinite())
    return false;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
    return false;
  const auto & values = solver.eigenvalues();
  // Accumulation of Count double outer products resolves directions only above
  // epsilon*Count relative to the largest eigenvalue. This is a numerical rank
  // test, not a tunable planarity/noise test; distances below retain that role.
  const double rankTolerance = std::numeric_limits<double>::epsilon() * Count * values[2];
  if (!(values[2] > 0.0) || !(values[1] > rankTolerance))
    return false;

  Eigen::Vector4d candidate;
  candidate.head<3>() = solver.eigenvectors().col(0).normalized();
  candidate[3] = -candidate.head<3>().dot(center);
  if (!candidate.allFinite())
    return false;
  Eigen::Index dominant;
  candidate.head<3>().cwiseAbs().maxCoeff(&dominant);
  if (candidate[3] < 0.0 || (candidate[3] == 0.0 && candidate[dominant] < 0.0))
    candidate = -candidate;
  if ((candidate.cwiseAbs().array() > static_cast<double>(std::numeric_limits<Scalar>::max())).any())
    return false;
  const Eigen::Matrix<Scalar, 4, 1> result = candidate.template cast<Scalar>();
  if (!result.allFinite())
    return false;
  // Validate the coefficients actually returned, including output precision.
  const Eigen::Vector4d returned = result.template cast<double>();
  for (int i = 0; i < Count; ++i) {
    const double distance =
      returned[0] * points[i].x + returned[1] * points[i].y + returned[2] * points[i].z + returned[3];
    if (!std::isfinite(distance) || std::abs(distance) > threshold)
      return false;
  }
  out = result;
  return true;
}

}  // namespace fast_lio

// recovery_view_match.hpp — the re-anchor gate's match evidence for one scan against a map view.
//
// While a recovery is pending, laserMapping can qualify it against the points the map held when the
// recovery began (iVox's entry view) instead of freezing the whole map: the scans being qualified then go
// into the map as usual and still cannot vouch for themselves with points they put there. The evidence is
// what reanchorMatchOk judges, the effective correspondences and their mean absolute residual, counted by
// h_share_model's own rules at one state, the scan's final one:
//   - Count nearest neighbours, the farthest within sqrt(5) m (h_share_model: NUM_MATCH_POINTS, <= 5);
//   - a plane through them (CachedPlane::fit: threshold 0.1, the square root of the body range);
//   - the plane's distance/range test (CachedPlane::evaluate);
//   - with the correspondence gate on, its innovation and first-neighbour tests (correspondenceAdmitted)
//     against the update's prior [pos, rot] covariance, floored as h_share_model floors it, and the scan's
//     own robust residual scale.
// h_share_model's ground recovery (ground_recover_en) re-admits rejected ground returns; it is not repeated
// here, so with it on a scan counts fewer correspondences here than in the update.
//
// Pure C++ + Eigen. No ROS, no estimator types.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "cached_plane.hpp"
#include "correspondence_gate.hpp"

namespace fast_lio
{

/// The estimator's pose of a scan: world <- IMU (rot, pos) and IMU <- lidar (rot_li, pos_li).
struct RecoveryViewPose
{
  Eigen::Quaterniond rot = Eigen::Quaterniond::Identity();
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot_li = Eigen::Quaterniond::Identity();
  Eigen::Vector3d pos_li = Eigen::Vector3d::Zero();
};

/// reanchorMatchOk's inputs for one scan.
struct RecoveryViewMatch
{
  int effective = 0;
  double res_mean = std::numeric_limits<double>::quiet_NaN();  ///< mean |residual|; NaN without a correspondence
};

/// The match of the body points `body` at `pose` against what `search(world_point, Count, neighbours,
/// sqdist)` returns (neighbours ascending, as IVox::Nearest_Search_View fills them). `prior_P6` is the
/// update's prior [pos, rot] covariance block, `point_cov` the per-point measurement variance
/// (LASER_POINT_COV). Every point is judged on its own scratch, so `search` may run concurrently.
template<int Count, class Neighbours, class BodyPoints, class Search>
RecoveryViewMatch matchRecoveryView(const BodyPoints & body,
                                    const RecoveryViewPose & pose,
                                    const Search & search,
                                    const CorrespondenceGateParams & gate,
                                    const Eigen::Matrix<double, 6, 6> & prior_P6,
                                    double point_cov)
{
  static_assert(Count > 0, "a plane needs neighbours");
  struct Row
  {
    Neighbours near;
    std::vector<float> sqdist;
    bool selected = false;
    float residual = 0.0f;
    double variance = 0.0;
    double nn0 = 0.0;
  };
  const bool gate_on = gate.innovation_k > 0.0 || gate.nn0_max > 0.0;
  Eigen::Matrix<double, 6, 6> gate_P6 = Eigen::Matrix<double, 6, 6>::Zero();
  if (gate.innovation_k > 0.0)
    gate_P6 = flooredPriorBlock(prior_P6, gate.rot_sigma_floor_deg * M_PI / 180.0);
  const auto count = static_cast<std::ptrdiff_t>(body.size());
  std::vector<Row> rows(static_cast<std::size_t>(count));
  for (Row & row : rows) {  // no allocation inside the parallel loop
    row.near.reserve(Count);
    row.sqdist.reserve(Count);
  }

#ifdef MP_EN
#pragma omp parallel for
#endif
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    Row & row = rows[static_cast<std::size_t>(i)];
    auto world = body[static_cast<std::size_t>(i)];
    const Eigen::Vector3d p_body(world.x, world.y, world.z);
    const Eigen::Vector3d p_imu = pose.rot_li * p_body + pose.pos_li;
    const Eigen::Vector3d p_global = pose.rot * p_imu + pose.pos;
    world.x = static_cast<float>(p_global(0));
    world.y = static_cast<float>(p_global(1));
    world.z = static_cast<float>(p_global(2));
    search(world, Count, row.near, row.sqdist);
    if (static_cast<int>(row.near.size()) < Count || !(row.sqdist[Count - 1] <= 5))
      continue;
    CachedPlane plane;
    if (!plane.fit<Count>(row.near, 0.1f, static_cast<float>(std::sqrt(p_body.norm()))))
      continue;
    const PlaneResidual residual = plane.evaluate(world.x, world.y, world.z);
    if (!residual.accepted)
      continue;
    row.selected = true;
    row.residual = residual.distance;
    if (gate_on) {
      const Eigen::Vector4f & abcd = plane.coefficients();
      const Eigen::Vector3d n_w(abcd(0), abcd(1), abcd(2));
      const Eigen::Vector3d A = p_imu.cross(Eigen::Vector3d(pose.rot.conjugate() * n_w));
      row.variance = gate.innovation_k > 0.0 ? residualVariance(n_w, A, gate_P6, point_cov) : 0.0;
      if (gate.nn0_max > 0.0)
        row.nn0 = (Eigen::Vector3d(row.near[0].x, row.near[0].y, row.near[0].z) - p_global).norm();
    }
  }

  // The gate's decision waits for the scan's residual scale, as in h_share_model; sums run in point order.
  double scan_floor = 0.0;
  if (gate_on && gate.mad_scale > 0.0) {
    std::vector<double> residuals;
    residuals.reserve(rows.size());
    for (const Row & row : rows) {
      if (row.selected)
        residuals.push_back(row.residual);
    }
    scan_floor = gate.mad_scale * robustResidualScale(residuals);
  }
  RecoveryViewMatch match;
  double total = 0.0;
  for (const Row & row : rows) {
    if (!row.selected)
      continue;
    if (gate_on && !correspondenceAdmitted(gate, row.residual, row.variance, row.nn0, scan_floor))
      continue;
    ++match.effective;
    total += std::fabs(row.residual);
  }
  if (match.effective > 0)
    match.res_mean = total / match.effective;
  return match;
}

}  // namespace fast_lio

// frontend_measurement.cpp — the front end's lidar measurement model: the phases of one evaluation, and the object
// that keeps their state between evaluations.
#include "frontend_measurement.hpp"

#include <rclcpp/logging.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <omp.h>
#include <so3_math.h>
#include <vector>

#include "attitude_hold.hpp"
#include "cached_plane.hpp"
#include "correspondence_compaction.hpp"
#include "correspondence_gate.hpp"
#include "degeneracy_policy.hpp"

#ifdef FASTLIO_TEST_HOOKS
#include "laser_mapping_test_hooks.hpp"

// The node's roll/pitch consider verdict for its scan, which the H-model exit record carries.
extern bool rp_consider_active;

namespace fast_lio_test
{
// The H-model call in progress, right after compaction.
static void reportHModelSelection(const fast_lio::MeasurementScratch & scratch,
                                  const fast_lio::MeasurementTelemetry & telemetry,
                                  const fast_lio::MeasurementInputs & inputs,
                                  const fast_lio::MeasurementOutputs & outputs,
                                  bool search_requested,
                                  bool searched)
{
  HModelSelection selection;
  selection.scan_end = inputs.scan_end_time;
  selection.call = outputs.iterations;
  selection.search_requested = search_requested;
  selection.searched = searched;
  selection.refresh_skips = telemetry.nn_skips;
  selection.refresh_max_shift = telemetry.nn_max_shift;
  selection.points = inputs.point_count;
  selection.selected = scratch.point_selected_surf.data();
  selection.residuals = scratch.res_last.data();
  selection.gate_rejected = telemetry.corr_gate_rejected.load(std::memory_order_relaxed);
  selection.gate_scan_sigma = telemetry.corr_gate_scan_sigma;
  selection.ground_rows = telemetry.canopy_ground_recovered;
  selection.rows.resize(static_cast<std::size_t>(std::max(outputs.effective_rows, 0)));
  for (std::size_t i = 0; i < selection.rows.size(); ++i) {
    const PointType & point = scratch.laserCloudOri.points[i];
    const PointType & normal = scratch.corr_normvect.points[i];
    HModelRow & row = selection.rows[i];
    row.point[0] = point.x;
    row.point[1] = point.y;
    row.point[2] = point.z;
    row.normal[0] = normal.x;
    row.normal[1] = normal.y;
    row.normal[2] = normal.z;
    row.residual = normal.intensity;
    row.weight = scratch.corr_weight[i];
  }
  afterHModelSelection(selection);
}

// The H-model call's exit. `hold_rows`: the call restricted its rotation rows to yaw (false without rows).
static HModelExit hModelExit(const fast_lio::MeasurementTelemetry & telemetry,
                             const fast_lio::MeasurementInputs & inputs,
                             const fast_lio::MeasurementOutputs & outputs,
                             bool valid,
                             bool hold_rows)
{
  HModelExit out;
  out.scan_end = inputs.scan_end_time;
  out.call = outputs.iterations;
  out.valid = valid;
  out.effective = outputs.effective_rows;
  out.mean_residual = outputs.mean_residual;
  out.pos_eig_min = telemetry.pos_obs_eig_min;
  out.pos_eig_max = telemetry.pos_obs_eig_max;
  out.pos_z_weak = telemetry.pos_obs_z_weak;
  out.hold_scene = inputs.attitude_hold_active;
  out.hold_allowed = inputs.attitude_hold_allowed;
  out.hold_rows = hold_rows;
  out.hold_info = telemetry.lidar_attitude_hold_info_active;
  out.rp_info_min = telemetry.lidar_rot_obs_rp_min;
  out.rp_consider = rp_consider_active;
  out.degeneracy_engaged = telemetry.degeneracy_last.engaged;
  for (int i = 0; i < 3; ++i) {
    out.degeneracy_shares[i] = telemetry.degeneracy_last.lambda_norm(i);
    out.degeneracy_lambdas[i] = telemetry.degeneracy_last.lambda_abs(i);
    out.degeneracy_weights[i] = telemetry.degeneracy_last.weights(i);
  }
  out.degeneracy_weighted_points = telemetry.degeneracy_point_weighted;
  out.degeneracy_weight_sum = telemetry.degeneracy_point_weight_sum;
  out.dump_written = valid && outputs.dump_scan_active && outputs.dump_rows.is_open();
  return out;
}
}  // namespace fast_lio_test
#endif

namespace
{
// Sectors of the ground-recovery planes.
constexpr int kGndSectors = fast_lio::MeasurementScratch::kGndSectors;
// Upper body-range edges of the [RES] residual bins (m).
constexpr int kResBins = fast_lio::MeasurementTelemetry::kResBins;
constexpr double kResBinEdges[kResBins] = {2.0, 5.0, 10.0, 20.0, 40.0, 1e9};

bool decideSearch(fast_lio::MeasurementScratch & scratch,
                  fast_lio::MeasurementTelemetry & telemetry,
                  const fast_lio::FrontendParams & params,
                  const fast_lio::MeasurementOutputs & outputs,
                  state_ikfom & s,
                  const esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  bool searching = ekfom_data.converge;
  if (searching && scratch.nn_have_last_search) {
    // A point at range r moves by at most |dt| + r*|dtheta| between two states, so that
    // bound against the map resolution decides whether a refresh can change anything.
    const V3D dt = s.pos - scratch.nn_last_search_pos;
    const M3D dR = scratch.nn_last_search_rot.transpose() * s.rot.toRotationMatrix();
    const V3D dr = Log(dR);
    const double shift = dt.norm() + params.nn_refresh_ref_range * dr.norm();
    telemetry.nn_max_shift = std::max(telemetry.nn_max_shift, shift);
    if (params.nn_refresh_gate_en && shift < params.nn_refresh_min_shift) {
      searching = false;
      ++telemetry.nn_skips;
    }
  }
  if (searching) {
    scratch.nn_last_search_pos = s.pos;
    scratch.nn_last_search_rot = s.rot.toRotationMatrix();
    scratch.nn_have_last_search = true;
  }
  ++outputs.iterations;
  telemetry.ekf_search_iters += searching ? 1 : 0;
  scratch.total_residual = 0.0;
  return searching;
}

// [v11] the gate judges every residual against the prior's [pos, rot] covariance, which IKFoM keeps
// unchanged through the iterations; read it once per call.
bool gatePrior(fast_lio::MeasurementTelemetry & telemetry,
               const fast_lio::FrontendParams & params,
               const fast_lio::MeasurementInputs & inputs,
               Eigen::Matrix<double, 6, 6> & gate_P6)
{
  const bool gate_on = params.corr_gate.innovation_k > 0.0 || params.corr_gate.nn0_max > 0.0;
  if (params.corr_gate.innovation_k > 0.0) {
    gate_P6 = fast_lio::flooredPriorBlock(inputs.prior_covariance.block<6, 6>(0, 0),
                                          params.corr_gate.rot_sigma_floor_deg * M_PI / 180.0);
  }
  telemetry.corr_gate_rejected.store(0, std::memory_order_relaxed);
  return gate_on;
}

/** closest surface search and residual computation **/
void matchPoints(fast_lio::MeasurementScratch & scratch,
                 const fast_lio::FrontendParams & params,
                 const fast_lio::MeasurementInputs & inputs,
                 state_ikfom & s,
                 bool searching,
                 bool gate_on,
                 const Eigen::Matrix<double, 6, 6> & gate_P6)
{
#ifdef MP_EN
  // Respect OMP_NUM_THREADS when set so the runtime CPU budget (cgroup
  // --cpus) can be honoured; the compile-time MP_PROC_NUM (8 on x86) would
  // otherwise spawn 8 OMP threads and oversubscribe a 4-CPU container,
  // starving the PGO node it shares cores with.
  {
    static const int omp_threads = []() {
      const char * e = std::getenv("OMP_NUM_THREADS");
      const int n = (e != nullptr) ? static_cast<int>(std::strtol(e, nullptr, 10)) : 0;
      return n > 0 ? n : MP_PROC_NUM;
    }();
    omp_set_num_threads(omp_threads);
  }
#pragma omp parallel for
#endif
  for (int i = 0; i < inputs.point_count; i++) {
    PointType & point_body = inputs.body.points[i];
    PointType & point_world = inputs.world.points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto & points_near = scratch.Nearest_Points[i];

    auto & cached_plane = scratch.plane_cache[i];
    if (searching) {
      // A failed search/fit must not leave the previous neighbourhood usable.
      cached_plane.invalidate();
      inputs.map.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      if (points_near.size() >= NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 5) {
        cached_plane.fit<NUM_MATCH_POINTS>(points_near, 0.1f, std::sqrt(p_body.norm()));
      }
    }

    // Per-iteration selection is independent of the cached neighbours. A row
    // rejected by residual/innovation/ground gates can recover at the new pose.
    scratch.point_selected_surf[i] = false;
    if (!cached_plane.valid())
      continue;

    const Eigen::Matrix<float, 4, 1> & pabcd = cached_plane.coefficients();
    {
      const auto residual = cached_plane.evaluate(point_world.x, point_world.y, point_world.z);
      const float pd2 = residual.distance;
      // [v11] the gate's inputs, computed while `s` still names the state; the decision waits for the
      // whole scan's residual scale (below the loop)
      if (gate_on) {
        const V3D n_w(pabcd(0), pabcd(1), pabcd(2));
        const V3D point_this = s.offset_R_L_I * p_body + s.offset_T_L_I;
        const V3D A = point_this.cross(V3D(s.rot.conjugate() * n_w));
        scratch.gate_var[i] =
          params.corr_gate.innovation_k > 0.0 ? fast_lio::residualVariance(n_w, A, gate_P6, LASER_POINT_COV) : 0.0;
        scratch.gate_nn0[i] = 0.0;
        if (params.corr_gate.nn0_max > 0.0 && !points_near.empty()) {
          scratch.gate_nn0[i] = (V3D(points_near[0].x, points_near[0].y, points_near[0].z) - p_global).norm();
        }
      }
      if (residual.accepted) {
        scratch.point_selected_surf[i] = true;
        scratch.normvec.points[i].x = pabcd(0);
        scratch.normvec.points[i].y = pabcd(1);
        scratch.normvec.points[i].z = pabcd(2);
        scratch.normvec.points[i].intensity = pd2;
        scratch.res_last[i] = std::abs(pd2);
      }
    }
  }
}

// [v11] gate decision and kernel weights: the scan's robust residual scale is known only now.
void applyGateAndKernels(fast_lio::MeasurementScratch & scratch,
                         fast_lio::MeasurementTelemetry & telemetry,
                         const fast_lio::FrontendParams & params,
                         const fast_lio::MeasurementInputs & inputs,
                         bool gate_on)
{
  telemetry.corr_gate_scan_sigma = 0.0;
  if (gate_on || params.corr_kernel != 0) {
    if (params.corr_gate.mad_scale > 0.0 || params.corr_kernel != 0) {
      std::vector<double> res;
      res.reserve(inputs.point_count);
      for (int i = 0; i < inputs.point_count; i++) {
        if (scratch.point_selected_surf[i]) {
          res.push_back(scratch.normvec.points[i].intensity);
        }
      }
      telemetry.corr_gate_scan_sigma = fast_lio::robustResidualScale(res);
    }
    const double scan_floor = params.corr_gate.mad_scale * telemetry.corr_gate_scan_sigma;
    const double kernel_c =
      params.corr_kernel != 0
        ? std::max(params.corr_kernel_mad_scale * telemetry.corr_gate_scan_sigma, params.corr_kernel_min_scale)
        : 0.0;
#ifdef MP_EN
#pragma omp parallel for
#endif
    for (int i = 0; i < inputs.point_count; i++) {
      scratch.gate_w[i] = 1.0;
      if (!scratch.point_selected_surf[i]) {
        continue;
      }
      const double pd2 = scratch.normvec.points[i].intensity;
      if (gate_on && !fast_lio::correspondenceAdmitted(
                       params.corr_gate, pd2, scratch.gate_var[i], scratch.gate_nn0[i], scan_floor)) {
        scratch.point_selected_surf[i] = false;
        telemetry.corr_gate_rejected.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (params.corr_kernel == 1) {
        scratch.gate_w[i] = fast_lio::gemanMcClureWeight(pd2, kernel_c);
      } else if (params.corr_kernel == 2) {
        scratch.gate_w[i] = fast_lio::cauchyWeight(pd2, kernel_c);
      }
    }
  }
}

// Shrinking-band iterative LS over one or more candidate sets.
// Gates: enough support, near-horizontal (terrain + vehicle tilt stay
// within a few degrees), sane height under the sensor. The height gate
// is what makes always-on safe: the accepted surface must sit where
// the vehicle's own ground sits.
bool fitGround(fast_lio::MeasurementScratch & scratch,
               const fast_lio::FrontendParams & params,
               const std::vector<Eigen::Vector3d> * sets,
               int nsets,
               int min_in,
               Eigen::Vector3d & n_out,
               double & d_out)
{
  std::vector<float> & zs = scratch.zs;
  zs.clear();
  for (int k = 0; k < nsets; k++)
    for (const auto & p : sets[k])
      zs.push_back(static_cast<float>(p.z()));
  if (static_cast<int>(zs.size()) < min_in)
    return false;
  // Init: horizontal plane at the pooled median candidate height.
  std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
  Eigen::Vector3d n(0, 0, 1);
  double d = -static_cast<double>(zs[zs.size() / 2]);
  int n_in = 0;
  const double bands[3] = {0.4, 0.25, 0.15};
  for (double band : bands) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    n_in = 0;
    for (int k = 0; k < nsets; k++)
      for (const auto & p : sets[k])
        if (std::fabs(n.dot(p) + d) <= band) {
          mean += p;
          ++n_in;
        }
    if (n_in < min_in)
      break;
    mean /= n_in;
    for (int k = 0; k < nsets; k++)
      for (const auto & p : sets[k])
        if (std::fabs(n.dot(p) + d) <= band) {
          const Eigen::Vector3d q = p - mean;
          cov.noalias() += q * q.transpose();
        }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    n = es.eigenvectors().col(0);
    if (n.z() < 0)
      n = -n;
    d = -n.dot(mean);
  }
  n_out = n;
  d_out = d;
  return n_in >= min_in && n.z() >= 0.95 &&
         std::fabs(-d / std::max(n.z(), 1e-6) - params.canopy_ground_z) <= params.canopy_ground_halfband;
}

void fitGroundSectors(fast_lio::MeasurementScratch & scratch,
                      const fast_lio::FrontendParams & params,
                      const fast_lio::MeasurementInputs & inputs)
{
  auto & cand = scratch.cand;
  for (int s = 0; s < kGndSectors; s++) {
    cand[s].clear();
    scratch.sec_ok[s] = false;
  }
  for (int i = 0; i < inputs.point_count; i++) {
    const PointType & pb = inputs.body.points[i];
    if (std::hypot(pb.x, pb.y) > params.canopy_near_range)
      continue;
    if (std::fabs(pb.z - params.canopy_ground_z) > params.canopy_ground_halfband)
      continue;
    const int s =
      std::min(kGndSectors - 1, static_cast<int>((std::atan2(pb.y, pb.x) + M_PI) * kGndSectors / (2.0 * M_PI)));
    cand[s].emplace_back(pb.x, pb.y, pb.z);
  }
  // Global whole-scan fit: pooled support is the hardest to bias — the
  // prior every sector must agree with, and the fallback plane.
  Eigen::Vector3d gn(0, 0, 1);
  double gd = 0.0;
  const bool glob_ok = fitGround(scratch, params, cand, kGndSectors, params.ground_recover_min_inliers, gn, gd);
  for (int s = 0; s < kGndSectors; s++) {
    Eigen::Vector3d n;
    double d;
    bool own_ok = fitGround(scratch, params, &cand[s], 1, params.ground_recover_min_inliers, n, d);
    if (own_ok && glob_ok) {
      // A sector plane may only REFINE the global one: normals must
      // agree and the plane heights at the sector centroid must match
      // within ground_recover_sector_tol; otherwise the sector fit
      // latched onto something the pooled fit rejected.
      Eigen::Vector3d c = Eigen::Vector3d::Zero();
      for (const auto & p : cand[s])
        c += p;
      c /= static_cast<double>(cand[s].size());
      const double z_sec = -(d + n.x() * c.x() + n.y() * c.y()) / std::max(n.z(), 1e-6);
      const double z_glob = -(gd + gn.x() * c.x() + gn.y() * c.y()) / std::max(gn.z(), 1e-6);
      constexpr double kSectorNormalDotMin = 0.985;  // ~10 deg relative tilt
      own_ok = n.dot(gn) >= kSectorNormalDotMin && std::fabs(z_sec - z_glob) <= params.ground_recover_sector_tol;
    }
    if (own_ok) {
      scratch.sec_ok[s] = true;
      scratch.sec_n[s] = n;
      scratch.sec_d[s] = d;
    } else if (glob_ok) {
      scratch.sec_ok[s] = true;
      scratch.sec_n[s] = gn;
      scratch.sec_d[s] = gd;
    }
  }
}

void recoverGroundRows(fast_lio::MeasurementScratch & scratch,
                       fast_lio::MeasurementTelemetry & telemetry,
                       const fast_lio::FrontendParams & params,
                       const fast_lio::MeasurementInputs & inputs)
{
  for (int i = 0; i < inputs.point_count; i++) {
    if (scratch.point_selected_surf[i])
      continue;
    if (telemetry.canopy_ground_recovered >= params.ground_recover_max_rows)
      break;
    const PointType & pb = inputs.body.points[i];
    if (std::hypot(pb.x, pb.y) > params.canopy_near_range)
      continue;
    const int s =
      std::min(kGndSectors - 1, static_cast<int>((std::atan2(pb.y, pb.x) + M_PI) * kGndSectors / (2.0 * M_PI)));
    if (!scratch.sec_ok[s])
      continue;
    // Scan side: on this sector's ground plane (clutter trimmed by the fit band).
    if (std::fabs(scratch.sec_n[s].dot(Eigen::Vector3d(pb.x, pb.y, pb.z)) + scratch.sec_d[s]) >
        params.ground_recover_scan_band)
      continue;
    auto & pn = scratch.Nearest_Points[i];
    if (static_cast<int>(pn.size()) < NUM_MATCH_POINTS)
      continue;
    const PointType & pw = inputs.world.points[i];
    // Map neighborhood must actually be nearby (same criterion class as
    // the normal path's 5th-neighbor sqdist gate).
    const double dx0 = pn[0].x - pw.x, dy0 = pn[0].y - pw.y, dz0 = pn[0].z - pw.z;
    if (dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > 5.0)
      continue;
    // Residual reference: the map-neighbour lower envelope (mean of the two
    // lowest z — robust to clutter fuzz in the map). It DRIFTS with the
    // vehicle, so this channel slows the Z ratchet, it does not anchor it.
    // An absolute reference was tried and closed: every level of authority
    // strong enough to matter rerouted the error into another channel
    // (attitude torque in degenerate clutter, or an inconsistent
    // measurement model when the attitude columns were zeroed). Fix the
    // constraint set instead of injecting absolute Z here.
    const double w_row = 1.0;
    float z1 = pn[0].z, z2 = std::numeric_limits<float>::max();
    for (size_t k = 1; k < pn.size(); k++) {
      const float z = pn[k].z;
      if (z < z1) {
        z2 = z1;
        z1 = z;
      } else if (z < z2) {
        z2 = z;
      }
    }
    const double z_ground_map = 0.5 * (z1 + (z2 == std::numeric_limits<float>::max() ? z1 : z2));
    const double pd2 = pw.z - z_ground_map;
    if (std::fabs(pd2) > params.ground_recover_max_residual)
      continue;
    scratch.point_selected_surf[i] = true;
    scratch.normvec.points[i].x = 0.0f;
    scratch.normvec.points[i].y = 0.0f;
    // Row weight rides on the normal + residual scaling (information
    // scales as w_row^2 in the point-to-plane H) — same mechanism as an
    // ordinary row, no new plumbing.
    scratch.normvec.points[i].z = static_cast<float>(w_row);
    scratch.normvec.points[i].intensity = static_cast<float>(w_row * pd2);
    scratch.res_last[i] = std::fabs(pd2);
    ++telemetry.canopy_ground_recovered;
  }
}

// P2 VoxelMap-lite anisotropic deweight (see the flag's block comment).
// Runs on the ACCEPTED rows BEFORE ground recovery, so recovery rows (added
// below with fabricated vertical normals — the load-bearing counterweight)
// are never rescaled. esti_plane rewrites the organic normals every iEKF
// iteration, so the scaling is applied fresh each call — no compounding.
// The grid is only written from the main thread between scans; this pass
// is read-only and safe under OMP.

/*** Ground recovery (selection-layer repair, 2026-07-06 forensics): ANY
 *   ground-hugging clutter (grass, crops, debris — content-agnostic) pollutes
 *   the 5-NN esti_plane + s>0.9 gates: true-ground returns get REJECTED while
 *   clutter micro-patches get accepted, depleting the vertical-normal share of
 *   the effective set even though the raw scan-to-scan Z valley stays sharp
 *   (measured: raw ground 62-70 % → ~10 % effective on the giant 0702 bag).
 *   Repair at the same layer, SELF-GATED (no scene classifier): (1) fit robust
 *   ground planes HYBRID global+sector in the body frame (shrinking-band LS —
 *   anything above the dominant surface is trimmed regardless of what it is).
 *   The whole-scan GLOBAL fit pools every candidate: maximum support, hardest
 *   to bias (in deep clutter a per-sector fit can latch onto a locally
 *   consistent clutter shelf that the pooled fit rejects — measured: sector-
 *   only was ~2x weaker than whole-scan in the 0702 canopy). Per-SECTOR fits
 *   refine it for crowned/rutted/partially-occupied terrain, but a sector
 *   plane is used only where it AGREES with the global plane (normal dot +
 *   height at the sector centroid); a sector that fails its own gates or
 *   disagrees falls back to the global plane, and without a valid global fit
 *   the surviving sector fits stand alone. The height-sanity gate (plane at
 *   the known sensor mount height) guarantees the dominant surface IS what
 *   the vehicle stands on, without scene semantics.
 *   (2) re-admit rejected near-field returns lying on their sector's plane,
 *   with an imposed vertical world normal and the residual measured against
 *   the LOWER ENVELOPE of their map neighbors (mean of the two lowest z —
 *   robust to clutter fuzz in the map). Rows flow into the standard
 *   point-to-plane H. Only points the normal path rejected are touched.
 *   NOTE: distinct from the removed A'-3 boost, which amplified already-
 *   matched rows (NEGATIVE result); this restores SELECTION of discarded
 *   true ground. ***/
void recoverGround(fast_lio::MeasurementScratch & scratch,
                   fast_lio::MeasurementTelemetry & telemetry,
                   const fast_lio::FrontendParams & params,
                   const fast_lio::MeasurementInputs & inputs)
{
  telemetry.canopy_ground_recovered = 0;
  if (params.ground_recover_en) {
    // Per-sector ground planes, cached per scan (body cloud is fixed across
    // iEKF iterations). n·p + d = 0 with unit n, n_z > 0.
    if (inputs.scan_end_time != scratch.fit_time) {
      scratch.fit_time = inputs.scan_end_time;
      fitGroundSectors(scratch, params, inputs);
    }
    recoverGroundRows(scratch, telemetry, params, inputs);
  }
}

void compactRows(fast_lio::MeasurementScratch & scratch,
                 fast_lio::MeasurementTelemetry & telemetry,
                 const fast_lio::FrontendParams & params,
                 const fast_lio::MeasurementInputs & inputs,
                 const fast_lio::MeasurementOutputs & outputs)
{
  outputs.effective_rows = 0;
  if (inputs.degeneracy_debug) {
    std::fill(std::begin(telemetry.res_bin_n), std::end(telemetry.res_bin_n), 0);
    std::fill(std::begin(telemetry.res_bin_ss), std::end(telemetry.res_bin_ss), 0.0);
  }

  // res_last and point_selected_surf have the same extent; the compactor checks
  // that extent and the point/normal/weight bounds before invoking this observer.
  outputs.effective_rows = static_cast<int>(fast_lio::compactSelectedCorrespondences(
    static_cast<std::size_t>(inputs.point_count),
    inputs.body,
    scratch.normvec,
    scratch.point_selected_surf,
    scratch.gate_w,
    params.corr_kernel != 0,
    scratch.laserCloudOri,
    scratch.corr_normvect,
    scratch.corr_weight,
    [&](std::size_t i) {
      scratch.total_residual += scratch.res_last[i];
      if (inputs.degeneracy_debug) {
        const PointType & pb = inputs.body.points[i];
        const double rng = std::sqrt(double(pb.x) * pb.x + double(pb.y) * pb.y + double(pb.z) * pb.z);
        int b = 0;
        while (b < kResBins - 1 && rng >= kResBinEdges[b])
          ++b;
        ++telemetry.res_bin_n[b];
        telemetry.res_bin_ss[b] += double(scratch.res_last[i]) * scratch.res_last[i];
      }
    }));
}

// Position-observability metric (degeneracy detector): eigenvalues of the
// normal information matrix M = sum(n n^T) over effective points. The min
// eigenvalue is the weakest-constrained position direction; if its
// eigenvector points along Z, the Z update is ill-conditioned.
void observePosition(const fast_lio::MeasurementScratch & scratch,
                     fast_lio::MeasurementTelemetry & telemetry,
                     const fast_lio::MeasurementOutputs & outputs)
{
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  for (int i = 0; i < outputs.effective_rows; i++) {
    const PointType & np = scratch.corr_normvect.points[i];
    Eigen::Vector3d n(np.x, np.y, np.z);
    M.noalias() += n * n.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
  telemetry.pos_obs_M = M;
  telemetry.pos_obs_eig_min = es.eigenvalues()(0);
  telemetry.pos_obs_eig_max = es.eigenvalues()(2);
  telemetry.pos_obs_z_weak = std::abs(es.eigenvectors()(2, 0));  // |Z| of weakest eigenvector
}

// [v8] attitude hold: the vertical in the body frame (the gravity state's own up), recomputed per
// iteration; every rotation row below is projected onto it so the update cannot move roll/pitch.
// [v9] the information rule decides from this iteration's own rows: Σ AAᵀ restricted to the
// roll/pitch plane, its weakest eigenvalue against lidar_attitude_hold_min_rp_info.
bool prepareAttitudeHold(const fast_lio::MeasurementScratch & scratch,
                         fast_lio::MeasurementTelemetry & telemetry,
                         const fast_lio::FrontendParams & params,
                         const fast_lio::MeasurementInputs & inputs,
                         const fast_lio::MeasurementOutputs & outputs,
                         state_ikfom & s,
                         V3D & hold_up)
{
  {
    V3D grav_w(s.grav[0], s.grav[1], s.grav[2]);
    const V3D up_w = grav_w.norm() > 1e-6 ? V3D(-grav_w.normalized()) : V3D(0.0, 0.0, 1.0);
    hold_up = (s.rot.conjugate() * up_w).normalized();
  }
  bool hold_rows = inputs.attitude_hold_active;
  {
    M3D rot_info = M3D::Zero();
    for (int i = 0; i < outputs.effective_rows; i++) {
      const PointType & laser_p = scratch.laserCloudOri.points[i];
      const V3D point_this = s.offset_R_L_I * V3D(laser_p.x, laser_p.y, laser_p.z) + s.offset_T_L_I;
      const PointType & norm_p = scratch.corr_normvect.points[i];
      const V3D C(s.rot.conjugate() * V3D(norm_p.x, norm_p.y, norm_p.z));
      const V3D A(point_this.cross(C));
      rot_info += A * A.transpose();
    }
    telemetry.lidar_rot_obs_rp_min = fast_lio::rollPitchInformation(rot_info, hold_up);
    if (inputs.attitude_hold_allowed && params.lidar_attitude_hold_min_rp_info > 0.0 &&
        telemetry.lidar_rot_obs_rp_min < params.lidar_attitude_hold_min_rp_info) {
      hold_rows = true;
      telemetry.lidar_attitude_hold_info_active = true;
    }
  }
  return hold_rows;
}

void buildRows(const fast_lio::MeasurementScratch & scratch,
               const fast_lio::FrontendParams & params,
               const fast_lio::MeasurementOutputs & outputs,
               state_ikfom & s,
               esekfom::dyn_share_datastruct<double> & ekfom_data,
               const V3D & hold_up,
               bool hold_rows)
{
  for (int i = 0; i < outputs.effective_rows; i++) {
    const PointType & laser_p = scratch.laserCloudOri.points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType & norm_p = scratch.corr_normvect.points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (hold_rows) {
      A = fast_lio::projectRotationRowToYaw(A, hold_up);
    }
    if (params.extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B),
        VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
    if (params.corr_kernel != 0 && scratch.corr_weight[i] < 1.0) {
      const double sw = std::sqrt(std::max(scratch.corr_weight[i], 0.0));
      ekfom_data.h_x.block<1, 12>(i, 0) *= sw;
      ekfom_data.h(i) *= sw;
    }
  }
}

/*** Subspace weighting. Judged on the position information LEFT once rotation has
 *   been solved for (the Schur complement of H^T H over the rotation columns), taken
 *   straight off the Jacobian we just built, then applied per point: this plane only
 *   constrains its own normal, so weight it by how much the policy trusts that
 *   direction. Row AND residual together — that is what makes it R_i <- R / s^2, a
 *   weight on the measurement, rather than a claim that the plane faces elsewhere.
 *   Recomputed every iEKF iteration: the correspondences move with the state. ***/
void degeneracyWeights(const fast_lio::MeasurementScratch & scratch,
                       fast_lio::MeasurementTelemetry & telemetry,
                       const fast_lio::FrontendParams & params,
                       rclcpp::Clock & spectrum_clock,
                       const fast_lio::MeasurementInputs & inputs,
                       const fast_lio::MeasurementOutputs & outputs,
                       esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  const auto H_t = ekfom_data.h_x.leftCols(3);
  const auto H_r = ekfom_data.h_x.middleCols(3, 3);
  const Eigen::Matrix3d H_tt = H_t.transpose() * H_t;
  const Eigen::Matrix3d H_tr = H_t.transpose() * H_r;
  const Eigen::Matrix3d H_rr = H_r.transpose() * H_r;
  telemetry.degeneracy_last =
    fast_lio::computePositionDegeneracy(H_tt, H_tr, H_rr, outputs.effective_rows, params.degeneracy_params);
  if (params.degeneracy_spectrum_trace) {
    // Both spectra side by side: the gap between them is what rotation explains away,
    // and it is the only way to see the coupling the thresholds now have to live with.
    const Eigen::Vector3d tt = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(H_tt).eigenvalues();
    const Eigen::Vector3d & sh = telemetry.degeneracy_last.lambda_norm;
    const Eigen::Vector3d & ab = telemetry.degeneracy_last.lambda_abs;
    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("laser_mapping"),
                         spectrum_clock,
                         2000,
                         "[DEGEN-SPECTRUM] shares [%.3f %.3f %.3f] sum=%.3f | judged raw [%.1f %.1f %.1f] | "
                         "H_tt raw [%.1f %.1f %.1f] | effct=%d far_frac=%.2f",
                         sh(0),
                         sh(1),
                         sh(2),
                         sh.sum(),
                         ab(0),
                         ab(1),
                         ab(2),
                         tt(0),
                         tt(1),
                         tt(2),
                         outputs.effective_rows,
                         inputs.far_fraction);
  }
  if (telemetry.degeneracy_last.engaged) {
    for (int i = 0; i < outputs.effective_rows; i++) {
      const PointType & np = scratch.corr_normvect.points[i];
      const double point_w = fast_lio::pointWeight(telemetry.degeneracy_last, Eigen::Vector3d(np.x, np.y, np.z));
      if (point_w < 1.0) {
        ekfom_data.h_x.block<1, 12>(i, 0) *= point_w;
        ekfom_data.h(i) *= point_w;
        telemetry.degeneracy_point_weight_sum += point_w;
        ++telemetry.degeneracy_point_weighted;
      }
    }
  }
}

void writeDump(const fast_lio::MeasurementScratch & scratch,
               const fast_lio::MeasurementTelemetry & telemetry,
               const fast_lio::MeasurementInputs & inputs,
               const fast_lio::MeasurementOutputs & outputs,
               state_ikfom & s,
               bool searching,
               const V3D & hold_up,
               bool hold_rows)
{
  const int iter = outputs.dump_iteration++;
  const V3D sensor_w = s.rot * s.offset_T_L_I + s.pos;
  outputs.dump_iterations << std::setprecision(15) << inputs.scan_end_time << ',' << iter << ','
                          << outputs.effective_rows << ',' << int(hold_rows) << ',' << std::setprecision(6)
                          << telemetry.lidar_rot_obs_rp_min << ',' << hold_up(0) << ',' << hold_up(1) << ','
                          << hold_up(2) << ',' << int(searching) << ',' << int(telemetry.degeneracy_last.engaged)
                          << '\n';
  for (int i = 0; i < inputs.point_count; i++) {
    const bool eff = scratch.point_selected_surf[i];
    if (!eff && iter > 0) {
      continue;
    }
    const auto & pb = inputs.body.points[i];
    const V3D p_body(pb.x, pb.y, pb.z);
    const V3D point_this = s.offset_R_L_I * p_body + s.offset_T_L_I;
    const auto & pwp = inputs.world.points[i];
    const V3D pw(pwp.x, pwp.y, pwp.z);
    const auto & near = scratch.Nearest_Points[i];
    const int n_near = static_cast<int>(near.size());
    double nn0 = -1.0, nn4 = -1.0, lam0 = -1.0, lam1 = -1.0, lam2 = -1.0, d_in = -1.0, plane_rms = -1.0;
    if (n_near > 0) {
      nn0 = (V3D(near[0].x, near[0].y, near[0].z) - pw).norm();
      if (n_near >= NUM_MATCH_POINTS) {
        const auto & q = near[NUM_MATCH_POINTS - 1];
        nn4 = (V3D(q.x, q.y, q.z) - pw).norm();
      }
      V3D c = V3D::Zero();
      for (const auto & q : near) {
        c += V3D(q.x, q.y, q.z);
      }
      c /= double(n_near);
      M3D S = M3D::Zero();
      for (const auto & q : near) {
        const V3D d = V3D(q.x, q.y, q.z) - c;
        S += d * d.transpose();
      }
      S /= double(n_near);
      Eigen::SelfAdjointEigenSolver<M3D> es(S);
      lam0 = es.eigenvalues()(0);
      lam1 = es.eigenvalues()(1);
      lam2 = es.eigenvalues()(2);
      const V3D nrm = es.eigenvectors().col(0);
      const V3D d = pw - c;
      d_in = (d - nrm * nrm.dot(d)).norm();
      if (scratch.plane_cache[i].valid()) {
        const Eigen::Matrix<float, 4, 1> & pl = scratch.plane_cache[i].coefficients();
        double ss = 0.0;
        for (const auto & q : near) {
          const double dd = pl(0) * q.x + pl(1) * q.y + pl(2) * q.z + pl(3);
          ss += dd * dd;
        }
        plane_rms = std::sqrt(ss / double(n_near));
      }
    }
    double pd2 = 0.0, nx = 0.0, ny = 0.0, nz = 0.0, ax = 0.0, ay = 0.0, az = 0.0, cosv = 0.0, w = 1.0;
    if (eff) {
      const auto & np = scratch.normvec.points[i];
      nx = np.x;
      ny = np.y;
      nz = np.z;
      pd2 = np.intensity;
      const V3D n(nx, ny, nz);
      const V3D C = s.rot.conjugate() * n;
      const V3D A = point_this.cross(C);
      ax = A(0);
      ay = A(1);
      az = A(2);
      const V3D dv = pw - sensor_w;
      cosv = dv.norm() > 1e-6 ? std::abs(n.dot(dv)) / dv.norm() : 0.0;
      if (telemetry.degeneracy_last.engaged) {
        w = fast_lio::pointWeight(telemetry.degeneracy_last, n);
      }
    }
    outputs.dump_rows << std::setprecision(15) << inputs.scan_end_time << ',' << iter << ',' << i << ',' << int(eff)
                      << std::setprecision(5) << ',' << p_body.norm() << ',' << point_this(0) << ',' << point_this(1)
                      << ',' << point_this(2) << ',' << nx << ',' << ny << ',' << nz << ',' << pd2 << ',' << ax << ','
                      << ay << ',' << az << ',' << nn0 << ',' << nn4 << ',' << n_near << ','
                      << int(scratch.plane_cache[i].valid()) << ',' << plane_rms << ',' << lam0 << ',' << lam1 << ','
                      << lam2 << ',' << d_in << ',' << cosv << ',' << w << ',' << pw(2) << ',' << pw(0) << ',' << pw(1)
                      << '\n';
  }
}
}  // namespace

void fast_lio::FrontendMeasurement::beginScan()
{
  telemetry_.match_time = 0;
  telemetry_.match_search_time = 0;
  telemetry_.match_nosearch_time = 0;
  telemetry_.ekf_search_iters = 0;
  scratch_.nn_have_last_search = false;
  for (auto & cached_plane : scratch_.plane_cache)
    cached_plane.invalidate();
  telemetry_.nn_max_shift = 0.0;
  telemetry_.nn_skips = 0;
  telemetry_.solve_time = 0;
}

// Called once at the real downsampled-scan boundary, before any int-indexed
// scan loop. No scratch allocation occurs in the parallel measurement loop.
bool fast_lio::FrontendMeasurement::prepareScan(std::size_t count, int & point_count)
{
  if (!fast_lio::prepareCorrespondenceScratch(count, scratch_.point_selected_surf, scratch_.res_last, point_count))
    return false;
  if (scratch_.plane_cache.size() < count) {
    scratch_.plane_cache.resize(count);
    scratch_.gate_var.resize(count);
    scratch_.gate_nn0.resize(count);
    scratch_.gate_w.resize(count);
  }
  return true;
}

void fast_lio::FrontendMeasurement::evaluate(state_ikfom & state,
                                             esekfom::dyn_share_datastruct<double> & data,
                                             const MeasurementInputs & inputs,
                                             const MeasurementOutputs & outputs)
{
  double match_start = omp_get_wtime();
  const bool searching = decideSearch(scratch_, telemetry_, params_, outputs, state, data);
  Eigen::Matrix<double, 6, 6> gate_P6 = Eigen::Matrix<double, 6, 6>::Zero();
  const bool gate_on = gatePrior(telemetry_, params_, inputs, gate_P6);
  matchPoints(scratch_, params_, inputs, state, searching, gate_on, gate_P6);
  applyGateAndKernels(scratch_, telemetry_, params_, inputs, gate_on);
  recoverGround(scratch_, telemetry_, params_, inputs);
  compactRows(scratch_, telemetry_, params_, inputs, outputs);

#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterHModelRows)
    fast_lio_test::afterHModelRows(
      static_cast<std::size_t>(inputs.point_count), inputs.scan_end_time, outputs.effective_rows);
  if (fast_lio_test::afterHModelSelection)
    fast_lio_test::reportHModelSelection(scratch_, telemetry_, inputs, outputs, data.converge, searching);
#endif

  observePosition(scratch_, telemetry_, outputs);

  if (outputs.effective_rows < 1) {
    data.valid = false;
    std::cerr << "No Effective Points!" << '\n';
#ifdef FASTLIO_TEST_HOOKS
    if (fast_lio_test::afterHModelMatrices)
      fast_lio_test::afterHModelMatrices(inputs.scan_end_time, data.valid, Eigen::MatrixXd(), Eigen::VectorXd());
    if (fast_lio_test::afterHModelExit)
      fast_lio_test::afterHModelExit(fast_lio_test::hModelExit(telemetry_, inputs, outputs, data.valid, false));
#endif
    return;
  }

  outputs.mean_residual = scratch_.total_residual / outputs.effective_rows;
  const double match_dt = omp_get_wtime() - match_start;
  telemetry_.match_time += match_dt;
  (searching ? telemetry_.match_search_time : telemetry_.match_nosearch_time) += match_dt;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  data.h_x = Eigen::MatrixXd::Zero(outputs.effective_rows, 12);  // 23
  data.h.resize(outputs.effective_rows);
  V3D hold_up = V3D::Zero();
  const bool hold_rows = prepareAttitudeHold(scratch_, telemetry_, params_, inputs, outputs, state, hold_up);
  buildRows(scratch_, params_, outputs, state, data, hold_up, hold_rows);
  degeneracyWeights(scratch_, telemetry_, params_, spectrum_clock_, inputs, outputs, data);
  if (outputs.dump_scan_active && outputs.dump_rows.is_open()) {
    writeDump(scratch_, telemetry_, inputs, outputs, state, searching, hold_up, hold_rows);
  }

#ifdef FASTLIO_TEST_HOOKS
  if (fast_lio_test::afterHModelMatrices)
    fast_lio_test::afterHModelMatrices(inputs.scan_end_time, data.valid, data.h_x, data.h);
  if (fast_lio_test::afterHModelExit)
    fast_lio_test::afterHModelExit(fast_lio_test::hModelExit(telemetry_, inputs, outputs, data.valid, hold_rows));
#endif
  telemetry_.solve_time += omp_get_wtime() - solve_start_;
}

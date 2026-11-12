// frontend_measurement.hpp — the front end's lidar measurement model, as the object the filter calls through its
// context entry, with the state it keeps between evaluations and the telemetry it leaves for the node.
#pragma once

#include <rclcpp/clock.hpp>

#include <Eigen/Core>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <vector>

#include "cached_plane.hpp"
#include "frontend_params.hpp"
#include "frontend_point_types.hpp"
#include "ikfom_types.hpp"
#include "map_backend_reference.hpp"
#include "normal_noise_floor.hpp"
#include "plane_fit.hpp"

namespace fast_lio
{
/// The filter's prior covariance, as esekf::get_P() returns it.
using PriorCovariance = Eigen::Matrix<double, state_ikfom::DOF, state_ikfom::DOF>;

/// What one evaluation reads of the node's scan. The caller binds it for that call only; nothing stores it.
struct MeasurementInputs
{
  const PriorCovariance & prior_covariance;  // the filter's covariance, read at each call
  MapBackend & map;                          // the local map the correspondences are searched in
  PointCloudXYZI & body;                     // the downsampled scan, body frame
  PointCloudXYZI & world;                    // the same points in the world frame, written at each call
  int point_count;                           // the downsampled scan's point count
  double scan_end_time;                      // the scan's end time; it keys the ground-plane cache
  double far_fraction;                       // fraction of the raw points beyond far_range, for the spectrum trace
  bool attitude_hold_active;                 // this scan's attitude-hold scene verdict
  bool attitude_hold_allowed;                // the attitude hold's release conditions hold this scan
  bool degeneracy_debug;                     // fill the residual range bins
};

/// Where one evaluation writes the node's per-scan values. The references write through at the statements that wrote
/// these values before, so an evaluation that stops early leaves them as it always did.
struct MeasurementOutputs
{
  int & effective_rows;             // the rows of this evaluation
  int & iterations;                 // evaluations so far on this scan
  double & mean_residual;           // the effective rows' mean residual
  bool dump_scan_active;            // this scan lies inside a correspondence dump window
  int & dump_iteration;             // dumped evaluations so far on this scan
  std::ofstream & dump_rows;        // the per-point correspondence dump
  std::ofstream & dump_iterations;  // the per-evaluation correspondence dump
};

/// What the evaluations leave for the node's decisions and traces, read through FrontendMeasurement::telemetry().
struct MeasurementTelemetry
{
  /// Body-range bins of the [RES] trace: 0-2, 2-5, 5-10, 10-20, 20-40 and 40+ m.
  static constexpr int kResBins = 6;

  // The scan's time in matching and in building the rows, summed over its evaluations. A correspondence search only
  // runs on the evaluations the filter declares converged, so the matching time of the two kinds is also kept apart:
  // that split says whether the cost is the search or the residual and plane work around it.
  double match_time = 0.0;
  double match_search_time = 0.0;
  double match_nosearch_time = 0.0;
  double solve_time = 0.0;
  int ekf_search_iters = 0;  // the scan's evaluations that searched
  // Position-observability metric of the last evaluation: eigenvalues of the normal information matrix
  // M = sum(n n^T) over the matched plane normals. pos_obs_z_weak = |Z component| of the weakest eigenvector; →1
  // means Z is the least-constrained direction (degenerate Z).
  double pos_obs_eig_min = 0.0;
  double pos_obs_eig_max = 0.0;
  double pos_obs_z_weak = 0.0;
  Eigen::Matrix3d pos_obs_M = Eigen::Matrix3d::Zero();  // M itself (world frame), for the [SCAN] trace
  // The normal-noise floor (FrontendParams::degeneracy_floor_mode is the policy): on the current scan, the mode it
  // runs in, the last evaluation's decision and how many evaluations ran it and were refused, for the [FLOOR] line;
  // over the node's life, the rows of the evaluations whose projection changed them and the sum of the position
  // information each of those rows kept (the actual transformed position-row norm), so the dose stays measurable, and
  // the scans whose floor observed because apply found no wheel speed (fast_lio::scanFloorMode).
  fast_lio::FloorScanRecord floor;
  long floor_rows_projected = 0;
  double floor_kept_sum = 0.0;
  long floor_no_wheel_scans = 0;
  // The correspondence gate and robust kernel (FrontendParams::corr_gate and corr_kernel are the policy) on the last
  // evaluation: the matches the gate refused, and the scan's robust residual scale (1.4826 MAD).
  std::atomic<int> corr_gate_rejected{0};
  double corr_gate_scan_sigma = 0.0;
  // Attitude hold (FrontendParams::attitude_hold is the policy): whether the information rule held on an evaluation
  // since the node last cleared it, and the roll/pitch information of the last evaluation that built rows.
  bool lidar_attitude_hold_info_active = false;
  double lidar_rot_obs_rp_min = 0.0;
  // [RES] trace: the converged point-to-plane residual by body-range bin, count and sum of squares, from the last
  // evaluation (only while the node's degeneracy_debug is on).
  int res_bin_n[kResBins] = {0};
  double res_bin_ss[kResBins] = {0.0};
  int canopy_ground_recovered = 0;  // rows the ground recovery added on the last evaluation
  // Neighbour refresh (FrontendParams::nn_refresh_gate_en is the policy) over the scan: the largest would-be refresh
  // shift (m) and the searches the gate skipped.
  double nn_max_shift = 0.0;
  int nn_skips = 0;
};

/// What the evaluations keep for each other: the per-point correspondence storage, the neighbour cache and the
/// ground-plane cache. Only the measurement model reads it.
struct MeasurementScratch
{
  /// Sectors of the ground-recovery planes.
  static constexpr int kGndSectors = 8;

  MeasurementScratch()
  {
    for (Eigen::Vector3d & normal : sec_n)
      normal.setZero();
  }

  // Per point of the downsampled scan, sized before the parallel loops, which only index them.
  std::vector<float> res_last;
  std::vector<std::uint8_t> point_selected_surf;
  std::vector<double> gate_var, gate_nn0, gate_w;  // the gate's inputs and the kernel weight
  std::vector<double> corr_weight;                 // per effective row
  double total_residual = 0.0;                     // the effective rows' residual sum on this evaluation
  std::vector<PointVector> Nearest_Points;         // each point's neighbours from its last search
  std::vector<std::vector<fast_lio::MapPointIdentity>> Nearest_Identities;
  fast_lio::MapIdentityStatus identity_status = fast_lio::MapIdentityStatus::kDisabled;
  // The neighbour set only moves on the iterations that search, so the 5x3 plane fit and the body-range scale in the
  // residual gate are invariant in between and are kept per point.
  std::vector<fast_lio::CachedPlane, Eigen::aligned_allocator<fast_lio::CachedPlane>> plane_cache;
  // Per point, the scatter of the plane its row was accepted against (plane_cache's own eigen data, written with the
  // selection; null for a ground-recovery row, whose normal is imposed rather than fitted). Per effective row, the
  // same pointer, compacted with the rows while the floor is on, and the floor's inputs: the weight the row was
  // scaled by and its normal's covariance.
  std::vector<const fast_lio::PlaneFitEigen *> point_plane;
  std::vector<const fast_lio::PlaneFitEigen *> corr_plane;
  std::vector<double> floor_weights;
  std::vector<std::optional<Eigen::Matrix3d>> floor_cov;
  // Neighbour refresh: the pose of the last search.
  V3D nn_last_search_pos = V3D::Zero();
  M3D nn_last_search_rot = M3D::Identity();
  bool nn_have_last_search = false;
  // Per point, the plane normal with the residual in intensity; per effective row, the point and its normal.
  PointCloudXYZI normvec{100000, 1};
  PointCloudXYZI laserCloudOri{100000, 1};
  PointCloudXYZI corr_normvect{100000, 1};
  // Ground recovery: per-sector planes n·p + d = 0 (unit n, n_z > 0), fitted for the scan that ends at fit_time,
  // with the candidates of each sector and the heights the fit pools.
  double fit_time = -1.0;
  bool sec_ok[kGndSectors] = {};
  Eigen::Vector3d sec_n[kGndSectors];
  double sec_d[kGndSectors] = {};
  std::vector<Eigen::Vector3d> cand[kGndSectors];
  std::vector<float> zs;
};

/// The lidar measurement model of the iterated update. The node holds one by value and binds it with
/// init_dyn_share_context; the filter keeps its address until the node clears that binding, so it is neither copied
/// nor moved.
class FrontendMeasurement
{
public:
  /// params is the node's configuration, read from the first evaluation on: the node passes it before reading its
  /// parameters, and it must outlive this object.
  explicit FrontendMeasurement(const FrontendParams & params) : params_(params) {}
  ~FrontendMeasurement() = default;
  FrontendMeasurement(const FrontendMeasurement &) = delete;
  FrontendMeasurement & operator=(const FrontendMeasurement &) = delete;
  FrontendMeasurement(FrontendMeasurement &&) = delete;
  FrontendMeasurement & operator=(FrontendMeasurement &&) = delete;

  /// A new scan: clears the scan's timings, search count, neighbour-refresh telemetry and floor record, drops every
  /// cached neighbour plane and the pose of the last search, and sets the mode the scan runs the floor in. wheel_fit:
  /// the scan's wheel update finds a speed, which an applying floor needs (fast_lio::scanFloorMode).
  void beginScan(bool wheel_fit);
  /// Sizes the per-point storage for a downsampled scan of count points and sets point_count to count. A count the
  /// evaluation's int-indexed loops cannot hold returns false and changes neither.
  bool prepareScan(std::size_t count, int & point_count);
  void resizeNormals(std::size_t count) { scratch_.normvec.resize(count); }
  void resizeNeighbours(std::size_t count) { scratch_.Nearest_Points.resize(count); }
  /// Clears the attitude hold's information verdict before a scan's update.
  void clearAttitudeHoldInfo() { telemetry_.lidar_attitude_hold_info_active = false; }

  /// One evaluation at the filter's current iterate: fills data's rows, or sets data.valid to false when no point
  /// is usable.
  void evaluate(state_ikfom & state,
                esekfom::dyn_share_datastruct<double> & data,
                const MeasurementInputs & inputs,
                const MeasurementOutputs & outputs);

  /// Condition already-weighted rows and record their actual dose. evaluate() calls this after row construction.
  void applyFloorRows(esekfom::dyn_share_datastruct<double> & data,
                      const std::vector<double> & weights,
                      const std::vector<std::optional<Eigen::Matrix3d>> & covariances,
                      int evaluation_iteration);

  const MeasurementTelemetry & telemetry() const { return telemetry_; }
  /// Each point's neighbours from its last search, for the map update.
  const std::vector<PointVector> & nearestPoints() const { return scratch_.Nearest_Points; }
  fast_lio::MapIdentityStatus identityStatus() const { return scratch_.identity_status; }
  /// kOk: [i][j] identifies nearestPoints()[i][j], in the same search order. Cached searches
  /// reuse both arrays. Disabled/unsupported: empty, without allocating identity storage.
  /// Value copies, readable on the scan thread even during worker additions, valid until
  /// the next scan's first evaluate(). Ground-recovery neighbours are not fitted supports.
  /// Any unlabelled fitted support makes the entire update unavailable to ancestry consumers.
  const std::vector<std::vector<fast_lio::MapPointIdentity>> & nearestIdentities() const
  {
    return scratch_.Nearest_Identities;
  }
  /// The last evaluation's effective points, body frame.
  const PointCloudXYZI & effectCloud() const { return scratch_.laserCloudOri; }

private:
  const FrontendParams & params_;
  MeasurementScratch scratch_;
  MeasurementTelemetry telemetry_;
  rclcpp::Clock spectrum_clock_{RCL_STEADY_TIME};  // throttles the [DEGEN-SPECTRUM] trace
};
}  // namespace fast_lio

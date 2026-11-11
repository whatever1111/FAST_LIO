#pragma once

#include <Eigen/Core>

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "anchor_trace.hpp"
#include "map_point_identity.hpp"
#include "time_admission_accounting.hpp"

// Only the non-installed test component defines or references these callbacks.
// Install before starting the worker/reset threads, clear only after joining both.
namespace fast_lio_test
{
/// State a fixture writes into the filter right before the anchor step: its
/// stand-in for what registration produced on that scan.
struct AnchorFixtureState
{
  bool overwrite = false;
  double pos[3] = {0.0, 0.0, 0.0};
  double rot_wxyz[4] = {1.0, 0.0, 0.0, 0.0};
  double vel[3] = {0.0, 0.0, 0.0};
  bool overwrite_cov = false;
  double cov_diag[23] = {};
};

extern std::function<void()> beforeMapAdd;
/// Owner snapshot after the existing shutdown join, outside mtx_buffer. No worker accesses this hook.
extern std::function<void(const fast_lio::TimeAdmissionSnapshot &)> afterAdmissionSnapshot;
extern std::function<void(const char *, std::size_t)> afterAdmissionSummary;
extern std::function<void()> beforeResetJoin;
extern std::function<void()> afterPendingReset;
extern std::function<void(std::size_t, double)> afterScanPrepared;
extern std::function<void(std::size_t, double, int)> afterHModelRows;
extern std::function<void(std::uint64_t, int, int, std::size_t, std::size_t, std::uint64_t)> afterImuConsumption;
extern std::function<void()> beforeImuProcess;
/// Scan end time -> -1 keep the divergence guard's verdict, 0 healthy, 1 degraded.
extern std::function<int(double)> overrideGuardDegraded;
extern std::function<void(double, AnchorFixtureState &)> beforeAnchorStep;
/// Scan end time -> true to trip the runaway watchdog this scan (reason: wheel).
extern std::function<bool(double)> forceRunawayTrip;
/// Scan end time -> true to hand the normal initial map build of that scan only points the map backend skips
/// (non-finite coordinates): the build runs and leaves the backend empty.
extern std::function<bool(double)> initialBuildTakesNoPoint;
/// Schedule the production map-reset owner before the first reference is built.
extern std::function<bool(double)> resetBeforeReference;
/// Read-only reset receipt: installed before/after, required-prior latch, retained pinned voxels.
extern std::function<void(bool, bool, bool, std::size_t)> afterPriorMapReset;
/// Commit receipt: epoch, exact preservation of ba/bg/grav and estimator covariance across the epoch transition.
extern std::function<void(std::uint64_t, bool)> afterLostRecoveryCommit;
/// Every scan record, whether or not zupt_anchor_trace_en prints it.
extern std::function<void(const fast_lio::AnchorScanRecord &)> afterAnchorScanRecord;
/// Every anchor trace line actually logged.
extern std::function<void(const std::string &)> afterAnchorTraceLine;
/// Successful publish return: encoding 0 Odometry, 1 pose FrontendObservation.
extern std::function<void(std::uint8_t, std::int64_t, const std::array<std::uint8_t, 32> &)> afterPosePublish;
/// A pending recovery's re-anchor match that its entry view served (reanchor_recovery_entry_view): scan end time,
/// h_share_model's own (effct, res_mean) on the whole map, the entry view's, and the scans the map took while the
/// view was open. Neither count reaches an output: only whether the recovery qualified does.
extern std::function<void(double, int, double, int, double, std::size_t)> afterRecoveryViewMatch;
/// A scan's points went into the map (map_incremental): scan end time and whether an entry view was open then.
extern std::function<void(double, bool)> afterMapInsertion;
/// h_share_model's exit: scan end time, validity, and the Jacobian rows and residuals handed to the filter (empty when
/// no correspondence is left).
extern std::function<void(double, bool, const Eigen::MatrixXd &, const Eigen::VectorXd &)> afterHModelMatrices;

/// One row the H-model hands to the filter, in the filter's order: the body-frame point, the world-frame plane
/// normal, the signed point-to-plane residual the row is built from, and the kernel weight (1 without a kernel).
struct HModelRow
{
  float point[3];
  float normal[3];
  float residual;
  double weight;
};

/// One H-model call's correspondences right after compaction: the search decision, the selection in point order,
/// what the gate, the kernel and the ground recovery did, and the rows.
struct HModelSelection
{
  double scan_end = 0.0;
  int call = 0;                             ///< H-model calls on this scan so far, this one included
  bool search_requested = false;            ///< the filter asked for a neighbour search
  bool searched = false;                    ///< this call searched; false: it used the cached planes
  int refresh_skips = 0;                    ///< requested searches the refresh gate skipped on this scan so far
  double refresh_max_shift = 0.0;           ///< the largest refresh shift on this scan so far (m)
  int points = 0;                           ///< downsampled scan points
  const std::uint8_t * selected = nullptr;  ///< `points` flags: the point has a row
  const float * residuals = nullptr;        ///< `points` values, read where the point has a row: |residual|
  int gate_rejected = 0;                    ///< points the correspondence gate refused
  double gate_scan_sigma = 0.0;             ///< the scan's robust residual scale (0 when no gate or kernel uses it)
  int ground_rows = 0;                      ///< rows the ground recovery added
  std::vector<HModelRow> rows;
};

/// One H-model call's exit, with rows or without them. The values a call without rows does not compute are the
/// ones the previous call left.
struct HModelExit
{
  double scan_end = 0.0;
  int call = 0;
  bool valid = false;               ///< rows went to the filter
  int effective = 0;                ///< rows
  double mean_residual = 0.0;       ///< mean |residual| of the rows
  double pos_eig_min = 0.0;         ///< the rows' position information sum(n n^T): smallest eigenvalue
  double pos_eig_max = 0.0;         ///< largest eigenvalue
  double pos_z_weak = 0.0;          ///< |z| of the weakest direction
  bool hold_scene = false;          ///< the scan's attitude hold verdict from its scene
  bool hold_allowed = false;        ///< the hold's release conditions alone
  bool hold_rows = false;           ///< this call restricted its rotation rows to yaw (false without rows)
  bool hold_info = false;           ///< the information rule has held the attitude on this scan
  double rp_info_min = 0.0;         ///< roll/pitch information of the rows
  bool rp_consider = false;         ///< the scan's update leaves roll/pitch to the gyro (lidar_consider_rp_en)
  // The normal-noise floor's decision of the scan's last evaluation that ran it; neutral (1, 0, 1, 0, 0) while the
  // floor is off.
  bool degeneracy_engaged = false;                 ///< the floor's projection changed this call's rows
  double degeneracy_shares[3] = {0.0, 0.0, 0.0};   ///< M_eff's eigenvalues over the rows, ascending (1 undecided)
  double degeneracy_lambdas[3] = {0.0, 0.0, 0.0};  ///< M_eff's eigenvalues
  double degeneracy_weights[3] = {0.0, 0.0, 0.0};  ///< the keep gain per direction
  long degeneracy_weighted_points = 0;             ///< rows the floor projected since construction
  double degeneracy_weight_sum = 0.0;              ///< the sum of what each of them kept, |G n|
  bool dump_written = false;                       ///< this call appended to the correspondence dump
};
/// Every H-model call right after afterHModelRows.
extern std::function<void(const HModelSelection &)> afterHModelSelection;
/// The same evaluation's neighbour identities: end, call, lifecycle scan, map generation, [query][neighbour].
extern std::function<void(
  double, int, std::uint64_t, std::uint32_t, const std::vector<std::vector<fast_lio::MapPointIdentity>> &)>
  afterHModelIdentities;

// TEST_PRE_REFERENCE uses the previous scan's cloud. A returned seed must never claim the
// current scan; surviving points in pinned voxels may still name an earlier scan or prior.
template<typename Map, typename Points>
void assertPreReferenceSeedUnlabelled(const Map & map, const Points & seeds, std::uint64_t current_scan)
{
  if (!map.pointIdentityEnabled())
    return;
  typename Map::PointVector nearest;
  std::vector<float> distances;
  std::vector<fast_lio::MapPointIdentity> identities;
  for (const auto & point : seeds) {
    map.Nearest_Search_Identified(point, 1, nearest, distances, identities);
    for (const auto & id : identities)
      assert(id.origin == fast_lio::MapPointOrigin::kUnlabelled || id.origin == fast_lio::MapPointOrigin::kPrior ||
             id.frame_seq < current_scan);
  }
}
/// Every H-model exit, right after afterHModelMatrices.
extern std::function<void(const HModelExit &)> afterHModelExit;
/// The directory the correspondence dump writes its files to; empty: the package's Log/ directory. The constructor
/// opens the files: set it before.
extern std::string correspondenceDumpDir;

/// The filter right after the iterated lidar update: the full state and the error-state covariance (row-major).
struct FilterSnapshot
{
  double pos[3];
  double rot_xyzw[4];
  double offset_R_L_I_xyzw[4];
  double offset_T_L_I[3];
  double vel[3];
  double bg[3];
  double ba[3];
  double grav[3];
  double P[23 * 23];
};
extern std::function<void(double, const FilterSnapshot &)> afterLidarUpdate;
/// An empty IMU window settled uncommitted: batch sequence, retained prefix size, queued samples, identity of the
/// first queued sample (0 when none), FIFO fault latch.
extern std::function<void(std::uint64_t, std::size_t, std::size_t, std::uint64_t, bool)> afterEmptyImuWindow;
/// Replaces the instance bytes the constructor read from the system; returns whether an instance is present.
extern std::function<bool(std::array<std::uint8_t, 16> &)> overrideFrontendInstance;
/// Clock read by the reference health repeat throttle.
extern std::function<std::chrono::steady_clock::time_point()> referenceRepeatClock;
/// A constructor stage boundary: its name and the names of the parameters declared so far.
extern std::function<void(const char *, const std::vector<std::string> &)> afterConstructionStage;

/// One value of the configuration in effect when the constructor returns. Floating values are IEEE-754 bit patterns,
/// integers decimal with their width and signedness, enums their underlying value; text is the raw string, which
/// the fixture escapes and gives its length.
struct EffectiveValue
{
  std::string name;
  std::string type;
  std::string value;
  bool text = false;
};
/// The constructor's last call: every effective value, parameter storage first, then the derived state.
extern std::function<void(const std::vector<EffectiveValue> &)> afterConstruction;

/// Encodes values from their static C++ type, so a recorded type cannot disagree with the storage it was read from.
class EffectiveRecorder
{
public:
  template<typename T>
  void value(const std::string & name, const T & v)
  {
    if constexpr (std::is_same<T, bool>::value) {
      add(name, "bool", v ? "1" : "0");
    } else if constexpr (std::is_enum<T>::value) {
      using Underlying = typename std::underlying_type<T>::type;
      add(name, "enum<" + integerType<Underlying>() + ">", integerText(static_cast<Underlying>(v)));
    } else if constexpr (std::is_integral<T>::value) {
      add(name, integerType<T>(), integerText(v));
    } else if constexpr (std::is_same<T, double>::value || std::is_same<T, float>::value) {
      add(name, realType<T>(), bits(v));
    } else if constexpr (std::is_same<T, std::string>::value) {
      entries_.push_back(EffectiveValue{name, "string", v, true});
    } else if constexpr (std::is_same<T, std::vector<double>>::value) {
      std::string list;
      for (const double element : v) {
        list += (list.empty() ? "" : ",") + bits(element);
      }
      add(name, "float64[" + std::to_string(v.size()) + "]", "[" + list + "]");
    } else {
      static_assert(sizeof(T) == 0, "no effective-value encoding for this type");
    }
  }

  /// Row-major elements with the dimensions in the type.
  template<typename Derived>
  void matrix(const std::string & name, const Eigen::MatrixBase<Derived> & m)
  {
    using Scalar = typename Derived::Scalar;
    static_assert(std::is_same<Scalar, double>::value || std::is_same<Scalar, float>::value, "real matrices only");
    std::string list;
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
      for (Eigen::Index c = 0; c < m.cols(); ++c) {
        list += (list.empty() ? "" : ",") + bits(static_cast<Scalar>(m(r, c)));
      }
    }
    add(name,
        realType<Scalar>() + "[" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + "]",
        "[" + list + "]");
  }

  template<std::size_t N>
  void bytes(const std::string & name, const std::array<std::uint8_t, N> & b)
  {
    std::string hex;
    char digits[3];
    for (const std::uint8_t byte : b) {
      std::snprintf(digits, sizeof digits, "%02x", static_cast<unsigned>(byte));
      hex += digits;
    }
    add(name, "bytes[" + std::to_string(N) + "]", hex);
  }

  /// Pairs as N rows of two columns.
  void windows(const std::string & name, const std::vector<std::pair<double, double>> & pairs)
  {
    std::string list;
    for (const auto & pair : pairs) {
      list += (list.empty() ? "" : ",") + bits(pair.first) + "," + bits(pair.second);
    }
    add(name, "float64[" + std::to_string(pairs.size()) + "x2]", "[" + list + "]");
  }

  /// Integer offsets as N rows of three columns.
  void offsets(const std::string & name, const std::vector<std::array<int, 3>> & rows)
  {
    std::string list;
    for (const auto & row : rows) {
      for (const int element : row) {
        list += (list.empty() ? "" : ",") + std::to_string(element);
      }
    }
    add(name, integerType<int>() + "[" + std::to_string(rows.size()) + "x3]", "[" + list + "]");
  }

  /// A value the test component could not read back.
  void unavailable(const std::string & name, const std::string & type) { add(name, type, "unavailable"); }

  const std::vector<EffectiveValue> & values() const { return entries_; }
  void clear() { entries_.clear(); }
  /// Appends another recorder's entries; the fixture sorts by name, so the order does not matter.
  void append(const EffectiveRecorder & other)
  {
    entries_.insert(entries_.end(), other.entries_.begin(), other.entries_.end());
  }

private:
  void add(const std::string & name, const std::string & type, const std::string & value)
  {
    entries_.push_back(EffectiveValue{name, type, value, false});
  }

  template<typename T>
  static std::string integerType()
  {
    return std::string(std::is_signed<T>::value ? "int" : "uint") + std::to_string(sizeof(T) * 8);
  }

  template<typename T>
  static std::string integerText(T v)
  {
    if constexpr (std::is_signed<T>::value) {
      return std::to_string(static_cast<long long>(v));
    } else {
      return std::to_string(static_cast<unsigned long long>(v));
    }
  }

  template<typename T>
  static std::string realType()
  {
    return std::is_same<T, double>::value ? "float64" : "float32";
  }

  static std::string bits(double v)
  {
    std::uint64_t pattern = 0;
    std::memcpy(&pattern, &v, sizeof pattern);
    char text[19];
    std::snprintf(text, sizeof text, "0x%016llx", static_cast<unsigned long long>(pattern));
    return text;
  }

  static std::string bits(float v)
  {
    std::uint32_t pattern = 0;
    std::memcpy(&pattern, &v, sizeof pattern);
    char text[11];
    std::snprintf(text, sizeof text, "0x%08x", static_cast<unsigned>(pattern));
    return text;
  }

  std::vector<EffectiveValue> entries_;
};

struct ImuInputSnapshot
{
  std::size_t lidarQueued;
  std::size_t imuQueued;
  std::uint64_t rejectedLidar;
  std::uint64_t rejectedImu;
  std::uint64_t discardedLidar;
  std::uint64_t discardedImu;
  bool faulted;
  bool mapFrozen;
  bool degradedOdom;
  double filterEnd;
  int scanCount;
};
ImuInputSnapshot inspectImuInput();
int reconfirmLastImuBatch();
}  // namespace fast_lio_test

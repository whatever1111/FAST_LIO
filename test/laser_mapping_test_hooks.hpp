#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "anchor_trace.hpp"

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
};
ImuInputSnapshot inspectImuInput();
int reconfirmLastImuBatch();
}  // namespace fast_lio_test

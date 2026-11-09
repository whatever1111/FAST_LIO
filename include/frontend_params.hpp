#pragma once

#include <Eigen/Core>

#include <string>
#include <utility>
#include <vector>

#include "adaptive_downsample.hpp"
#include "attitude_hold.hpp"
#include "correspondence_gate.hpp"
#include "degeneracy_policy.hpp"
#include "engulf_release.hpp"
#include "normal_noise_floor.hpp"
#include "parked_hold.hpp"
#include "prior_map_load.hpp"
#include "rp_consider.hpp"
#include "runaway_watchdog.hpp"

namespace fast_lio
{
/// The front end's configuration as the constructor leaves it: the parameters, and the values derived from them,
/// that afterwards only node methods and the lidar measurement model (FrontendMeasurement) read. FrontendParameters
/// fills it while the node is constructed and then only hands it out const. Values the scan path still writes, and
/// values that other free functions, workers or other owners read, stay in laserMapping.cpp until their own owners
/// take them.
struct FrontendParams
{
  bool path_en = true;
  bool effect_pub_en = false, map_pub_en = false;
  bool scan_body_pub_en = false;

  // Planar-motion (zero body-vertical-velocity) soft constraint — the A'-2 fix.
  //   When LiDAR loses Z observability (pos_obs_z_weak high), the iEKF dumps the
  //   unexplained residual into the accel bias and the state flies off. This adds a
  //   soft pseudo-measurement "body-frame vertical velocity ≈ 0" (a ground-robot
  //   non-holonomic prior, like the wheel-odom update) that supplies the missing
  //   vertical information and bounds the runaway at its source.
  //   BODY frame → it does NOT fight legitimate slope motion: world-frame vertical
  //   velocity on a ramp comes from the robot's pitch (rotation state), not from
  //   body-frame vz. Gated on z_weak so it has zero effect when Z is well observed.
  bool planar_constraint_en = false;
  double planar_constraint_noise = 0.1;          // m/s, soft (larger = weaker prior)
  double planar_constraint_z_weak_thresh = 0.5;  // apply only when pos_obs_z_weak exceeds this

  // Gravity-alignment leveling prior (A1) — the ROOT-CAUSE fix for the iVox pitch-
  // coupled Z drift. The planar (body-vz≈0) constraint above pins a R↔vel CONSISTENCY
  // relation the iEKF already satisfies, so it carries no absolute-pitch information and
  // cannot catch a biased pitch (world-Z drift = forward_distance × sin(pitch_err)) —
  // proven by zero effect from 4× strengthening it. This supplies the missing ABSOLUTE
  // reference: when LiDAR Z is degenerate AND the IMU is in low-linear-accel motion
  // (|‖a‖−g| small), the accelerometer's specific-force direction IS world-up expressed
  // in the body frame. A soft pseudo-measurement pulls the grav-state up-axis onto it,
  // constraining roll/pitch only — yaw lies in the measurement null space (gravity is
  // yaw-invariant), so the LiDAR-excellent heading is never touched.
  bool gravity_align_en = false;
  double gravity_align_noise = 0.05;         // unit-vector (≈rad) std of the leveling prior
  double gravity_align_z_weak_thresh = 0.3;  // engage only when pos_obs_z_weak exceeds this
  double gravity_align_accel_tol = 0.05;     // |‖a‖−g|/g must be below this (low-linear-accel gate)
  double gravity_align_gyro_tol = 0.35;      // rad/s; skip during fast turns (lever-arm safety)
  // 2026-08-31 additions (m20_0831 doorway pitch anatomy) — all default-off:
  double gravity_align_window_s = 0.0;        // [L2] >0: measurement = window-mean specific force
  bool gravity_align_continuous = false;      // [L2] engage every scan (gates still apply)
  double gravity_align_hold_s = 0.0;          // [L1] hold the degraded-strength prior after a trigger
  double gravity_align_noise_degraded = 0.0;  // [L1] tighter sigma while triggered/held (0 = use base)
  double gravity_align_mag_ref = 0.0;         // magnitude-gate anchor; 0 = IMU-init g (biased if init in motion)
  double gravity_align_grav_leak = 1.0;       // [v4] per-engaged-scan multiplicative variance leak on the S2
                                              // gravity block (fading memory); 1.0 = off
  // [v5] Freeze the S2 gravity state while the levelling prior runs at its
  // degraded strength (trigger or hold): only attitude takes the update then.
  // The leak makes the gravity block a fading-memory filter whose steady-state
  // gain is ~(leak-1) per engaged scan whatever the sigma (P* = (leak-1)·σ²), so
  // at 1.05 the gravity state is a ~2 s low-pass of the accelerometer direction
  // and inside blind re-anchor windows it swings 1.5-2° with the walking
  // dynamics while the vendor INS attitude stays put (m20 0825 133/151/164 s,
  // 0826 111/131 s; docs/PGO_LOOP_TUNING.md §7). The true vertical does not
  // change in those seconds; the attitude is what the degraded prior is for.
  bool gravity_align_grav_freeze_degraded = false;
  // [v6] Kinematic compensation of the levelling measurement (gravity_align_kinematics.hpp):
  // f = dv_b/dt + ω×v_b − Rᵀg + b_a, and only the ω×v_b term was subtracted. The window mean
  // of dv_b/dt is (v_b(end) − v_b(start))/span — 4-7° of lean at a doorway stop/start on the
  // m20, taken as a 0.46° measurement (docs/PGO_LOOP_TUNING.md §7.3). With the option on the
  // term comes from the filter's own per-scan body-velocity history, the prior is withheld
  // while a velocity overwrite (reset/pin/clamp) lies inside the window, and the velocity
  // covariance at the window edges widens the measurement sigma (scale 0 = off).
  bool gravity_align_lin_accel_comp = false;
  double gravity_align_vel_sigma_scale = 0.0;
  // [v6] velocity source for the term: "leg" (default) takes the leg-odometry forward speed
  // (wheel_topic) averaged over the last gravity_align_leg_edge_window_s at each scan end,
  // moved to the IMU point over gravity_align_leg_lever_arm; "filter" takes the iEKF's own
  // velocity — which drifts with the attitude error where the LiDAR does not observe it and
  // then cancels the gravity leakage the prior exists to correct (m20 0831: +3° frame tilt).
  std::string gravity_align_lin_accel_source = "leg";
  double gravity_align_leg_vel_sigma = 0.15;     // m/s; gait residual of the edge mean while moving
  double gravity_align_leg_edge_window_s = 0.2;  // s; trailing mean at each scan end
  Eigen::Vector3d gravity_align_leg_lever_arm =
    Eigen::Vector3d::Zero();  // m, IMU frame; IMU point minus the legs' reference point
  // [v6] While the map is frozen (blind stretch) and the body moves, nothing observes the body's
  // acceleration: not the frozen-map near-field registration, not the legs at a threshold, not the
  // FP. The 1 s specific-force mean then leans 5-12° with the stop/start and no compensation can
  // remove it (docs/PGO_LOOP_TUNING.md §7.3-7.4). With this on the prior is withheld there; the
  // attitude rides on the gyro (0.1-1° over such a window) instead of being levelled onto acceleration.
  bool gravity_align_withhold_blind_moving = false;
  // [v7] What sigma describes (gravity_align_kinematics.hpp LevellingSigmaPolicy): "scene" is the legacy
  // rule above; "measurement" makes sigma the measurement's own error — noise_static at rest,
  // noise_moving while moving — and withholds the prior where the acceleration is unobservable
  // (map frozen and body moving); the attitude covariance decides the rest.
  std::string gravity_align_sigma_policy = "scene";
  double gravity_align_noise_static =
    0.008;  // rad-equivalent; the specific-force mean is exact at rest (0.06° measured)
  double gravity_align_noise_moving = 0.045;  // ≈2.6°: the gait floor of a compensated 1 s mean while walking
  double gravity_align_grav_cap_deg = 3.0;    // [v4] leak cap: grav tangent std ceiling (deg) — bounds how far
                                              // the gravity state may be steered per run

  // [v8] Attitude hold (attitude_hold.hpp): scans registered against a frozen map (or, optionally,
  // near-field-only scans with no horizontal surface in view) may update translation and yaw but not
  // roll/pitch — the point-to-plane residuals carry no vertical information there and were tilting
  // the state 0.4-1.7° per scan (docs/PGO_LOOP_TUNING.md §7.5). Decided once per scan before the
  // update from the previous scan's guard verdicts and this scan's far-field share.
  fast_lio::AttitudeHoldParams attitude_hold;
  // [diag] per-correspondence dump (corr_dump_en, corr_dump_windows "t1:t2,t3:t4" in lidar_end_time seconds):
  // inside a window every iEKF iteration writes one row per effective correspondence (every point on the
  // first iteration) to corr_dump.csv: the point in the IMU frame, the matched plane's normal, the signed
  // residual, the rotation Jacobian row before any hold projection, the nearest-map distances, the 5-NN
  // patch scatter eigenvalues, the point's in-plane offset from the patch, the view cosine and the
  // degeneracy weight. corr_iter.csv carries one row per iteration (hold verdict, roll/pitch information,
  // the vertical in the body frame), corr_scan.csv one row per scan (the applied rotation and translation,
  // the prior covariance diagonals) — the data a per-correspondence admission rule is designed from
  // (docs/PGO_LOOP_TUNING.md §7.7).
  bool corr_dump_en = false;
  std::vector<std::pair<double, double>> corr_dump_windows;

  // Divergence guard (P1): when LiDAR correspondences collapse (scan starvation
  // under CPU/IO load, or feature-poor geometry), the iEKF runs on IMU
  // dead-reckoning and the state — especially Z — runs away to ±km and never
  // re-locks (scan leaves the map → permanent "No Effective Points"). This guard
  // detects the sustained collapse and BOUNDS the runaway: it pins body-vz≈0,
  // clamps an implausible body speed, freezes the map so dead-reckoned scans do
  // not corrupt it, and flags the odometry degraded. It is INERT in normal
  // operation (only fires after divergence_guard_streak consecutive scans whose
  // effective-point count falls below divergence_guard_min_eff). Recovery is
  // automatic: once correspondences return, the streak resets and the front end
  // re-locks against the (uncorrupted) map.
  bool divergence_guard_en = true;
  int divergence_guard_min_eff = 50;         // effct_feat_num below this = degenerate scan
  int divergence_guard_streak = 5;           // consecutive degenerate scans → enter degraded
  double divergence_guard_max_speed = 30.0;  // m/s; body-speed clamp while degraded (headroom over any platform speed)
  // Engulfment leading indicator (2026-08-14, m20_0814_degrade forensics): when a
  // low-mounted lidar wades into knee-high vegetation the FAR field vanishes from
  // the raw scan (healthy: ~50% of returns beyond guard_engulf_far_range; engulfed:
  // ~5%) BEFORE effct collapses. The last few "valid" updates on sub-metre grass
  // inject velocity/tilt (measured 2.14 m/s vs 0.5 m/s ground truth, 6° tilt)
  // and, worse, insert drifting garbage into the map for divergence_guard_streak
  // scans before the lagging effct-streak criterion freezes it — after which
  // re-lock is impossible (map poisoned) and survival is a scheduling coin-flip
  // (identical runs: clean vs km-runaway). Freezing the map on the SCAN-level
  // signal makes the guard leading instead of lagging. Computed on the raw
  // downsampled scan (a scene property, independent of matching outcome).
  bool guard_engulf_en = true;              // arm the far-field-collapse trigger
  double guard_engulf_far_range = 2.0;      // m; horizontal radius separating near/far
  double guard_engulf_far_frac_min = 0.15;  // enter when far-field fraction < this
  int guard_engulf_streak = 2;              // consecutive engulfed scans → degraded
  // Release hysteresis + velocity reset (m20_0814_degrade far_frac time series):
  // the brush is FOUR separate plunges (t≈146,165,175,191) with far_frac
  // hovering 0.04–0.39 on every shoulder. Instant release on one scan ≥ min
  // flaps on those shoulders and unfreezes the map for half-grass scans; and
  // whatever velocity the iEKF holds at release is unobserved IMU integration
  // (gravity-align saw 6–7° tilt), which then contaminates the inter-plunge
  // map before the next plunge — that is the run-to-run coin flip. So: release
  // only after guard_engulf_release_scans consecutive scans with far_frac ≥
  // guard_engulf_far_frac_release, and ZERO velocity at release so the next
  // good scans re-estimate it from correspondences instead of trusting the
  // blind stretch. Position/attitude/biases are kept.
  double guard_engulf_far_frac_release = 0.35;  // release needs far field this healthy
  int guard_engulf_release_scans = 5;           // ...for this many consecutive scans
  // Release policy (engulf_release.hpp), its covariance side: how much the pose's covariance grows when the blind
  // stretch is released. The default leaves P untouched, as the legacy release did.
  fast_lio::ReleaseInflationParams release_inflate;
  // Consider update for roll/pitch (rp_consider.hpp): on the scans the policy names, the lidar update leaves the
  // horizontal rotation at the propagated (gyro) value and updates everything else with the full-P gain.
  fast_lio::RollPitchConsiderParams rp_consider;
  // Scan-hole guard: when the gap between consecutive processed scans exceeds
  // guard_hole_min_sec, the IMU-only propagation across the hole is replaced by
  // a bounded constant-velocity hold from the pre-hole state before the lidar
  // update runs. Measured on m20_0814_degrade with injected holes over open
  // ground: <=2 s holes recover cleanly, but a 5 s hole let the IMU-only
  // velocity reach 4.1 m/s (walking 1.8) and the position drift metres off the
  // map (effct 3400 -> 400) before the runaway reset — quadratic IMU divergence
  // that ground-only geometry cannot re-observe. Default OFF.
  bool guard_hole_en = false;
  double guard_hole_min_sec = 0.3;      // s; scan gap that counts as a hole (nominal 0.1)
  double guard_hole_max_accel = 1.0;    // m/s^2; |vel - vel_pre_hole| is capped at this * gap
  bool guard_hole_reanchor_pos = true;  // also reset pos to pos_pre + vel_pre * gap
  // Static hold (ZUPT) on parked evidence; its evidence thresholds (zupt_*) stay beside the anchor state in
  // laserMapping.cpp, which documents the hold.
  bool zupt_en = true;
  // Parked hold (parked_hold.hpp): the same evidence, applied while the scan is
  // HEALTHY.
  fast_lio::ParkedHoldParams parked_hold;
  // What to do when verification never converges: "hold" keeps the map frozen and
  // the odometry flagged (an external relocalisation has to solve it), while
  // "reset_map" rebuilds the local map at the current pose — the robot keeps
  // working, having traded absolute anchoring for a live estimate.
  std::string reanchor_on_fail = "hold";
  // A pending recovery is qualified against the map it began with, iVox's entry view, and its scans keep going into
  // the map; off, the map stays frozen until the recovery ends.
  bool reanchor_recovery_entry_view = false;
  // Runaway watchdog: the guard and the re-anchor gate both key off the scan, and
  // the worst drift keeps the scan looking healthy — the map is rebuilt around
  // the wrong pose every frame, so correspondences and residuals stay good while
  // the estimate walks away (dog 2: effct 2000-3500, res 0.026 m, 45 min at
  // 0.5 m/s, parked). Only evidence from outside the estimator can see that.
  fast_lio::RunawayParams runaway;
  // hold = flag it lost (map freezes, downstream stops consuming the pose),
  // reset_map = rebuild the local map here, fault_exit = die and let the
  // supervisor restart the pipeline.
  std::string runaway_on_trip = "hold";
  // [v10] The covariance side of a velocity assignment (covariance_reset.hpp); laserMapping.cpp documents it beside
  // zupt_vel_reset_sigma.
  double guard_vel_reset_sigma = 0.0;  // m/s; a discarded velocity: engulfment release, hole clamp, runaway/parked
  // IMU-gap prior (imu_gap_prior.hpp, p_imu->gap_params): the scan is propagated across the gap instead of
  // skipped, and an IMU interval longer than imu_gap_verify_s still sends the re-anchor gate to blind, so the map
  // stays frozen until the scans after the gap have re-registered against it — the protection the coverage skip
  // gave, without the lost motion.
  double imu_gap_verify_s = 0.3;

  // 自适应降采样:单帧计算越过预算就把扫描体素调粗,让精度降级而不是延迟降级。
  // Adaptive downsampling: quality degrades instead of latency when a scan costs too much.
  fast_lio::AdaptiveDownsampleParams adaptive_ds;
  double filter_size_surf_min = 0;
  bool kalman_channel_diag_en = false;

  bool wheel_odom_en = false;
  double wheel_speed_scale = 1.0;
  // The topic of the wheel samples' variances (TwistWithCovarianceStamped, paired with wheel_topic by stamp); empty:
  // every sample carries the parameter sigmas below.
  std::string wheel_cov_topic;
  // With wheel_cov_topic the forward sigma floor, without it every sample's forward sigma (m/s).
  double wheel_vel_noise_vx = 0.10;
  // Without wheel_cov_topic every sample's lateral sigma (m/s); with it unused.
  double wheel_vel_noise_vy = 0.05;
  double wheel_vel_noise_vz = 0.10;
  Eigen::Matrix3d R_robot_to_imu = Eigen::Matrix3d::Identity();
  // The wheel measurement model (wheel_measurement.hpp): the slip coefficient k with the platform's level intercept
  // phi0 and the window of the slip regressor (s); the legs' lever arm, the IMU point minus their reference point in
  // the IMU frame (m), and its uncertainty (m).
  double wheel_slip_k = 0.0;
  double wheel_slip_phi0 = 0.0;
  double wheel_slip_fx_window_s = 1.0;
  Eigen::Vector3d wheel_lever_arm = Eigen::Vector3d::Zero();
  double wheel_lever_sigma = 0.0;

  bool prior_map_pinned = false;  // prior map goes into its own non-evictable tier
  double prior_map_voxel = 0.0;   // leaf the prior map is downsampled to (0 = filter_size_map)
  bool prior_map_freeze = false;  // stop growing the map once a prior map is installed
  fast_lio::PriorMapAlign prior_map_align = fast_lio::PriorMapAlign::kAsDelivered;
  std::vector<double> initial_pose_vec;
  bool initial_pose_full_rpy_override = false;

  // The lidar measurement model's parameters: the model reads them at every evaluation, the node only reports them.
  // The lidar update also estimates the lidar-IMU extrinsic: its rows carry the extrinsic rotation and translation
  // columns.
  bool extrinsic_est_en = false;
  // The spectrum trace alone, without the per-scan cerr traffic degeneracy_debug carries:
  // throttled, so it costs nothing and can run on a scoring replay.
  bool degeneracy_spectrum_trace = false;
  // The normal-noise floor (normal_noise_floor.hpp): every evaluation judges the position information its rows leave
  // once rotation is solved for (the Schur complement, regularised by degeneracy_params.schur_ridge) against what the
  // plane normals' own noise can fake, and keeps of each direction only what is left above that floor. off computes
  // nothing; observe computes the decision and logs it ([FLOOR]) without touching the rows; apply also projects the
  // rows' position columns and shifts their residuals to match. The released directions fall to IMU propagation
  // until another measurement (wheel speed, ZUPT, planar-Z) fills them.
  fast_lio::FloorMode degeneracy_floor_mode = fast_lio::FloorMode::kOff;
  fast_lio::FloorParams degeneracy_floor;
  fast_lio::DegeneracyParams degeneracy_params;
  // [v11] correspondence gate (correspondence_gate.hpp): a match is admitted only when its residual lies
  // within corr_innov_gate_k sigma of its own predicted variance H P Hᵀ + R (and, separately, when its
  // nearest map point is within corr_nn0_max). Both 0 = the legacy s > 0.9 test alone.
  fast_lio::CorrespondenceGateParams corr_gate;
  // [v11] optional robust kernel on the admitted rows (corr_kernel none|gm|cauchy): IRLS weight on the
  // squared residual with scale c = max(corr_kernel_mad_scale * scan scale, corr_kernel_min_scale); the row
  // and its residual are scaled by sqrt(w), i.e. R_i <- R / w.
  int corr_kernel = 0;  // 0 none, 1 Geman-McClure, 2 Cauchy
  double corr_kernel_mad_scale = 3.0, corr_kernel_min_scale = 0.03;
  // [v9] Attitude hold's information rule: hold when the scan's own roll/pitch information (attitude_hold.hpp
  // rollPitchInformation) is below this; 0 = off.
  double lidar_attitude_hold_min_rp_info = 0.0;

  // Near-field ground geometry, in the LiDAR/cloud BODY frame so it is immune to
  // the Z drift it helps repair. Consumed by the ground recovery below; the
  // canopy_ prefix is historical (these were shared with a canopy scene detector
  // that measured as telemetry-only and was removed) and is kept so the giant
  // profiles keep working.
  double canopy_near_range = 20.0;      // near-field horizontal radius, cloud frame (m)
  double canopy_ground_z = -0.47;       // ground height in the CLOUD frame (giant rig measured)
  double canopy_ground_halfband = 0.8;  // ± band around canopy_ground_z counted as ground (m)

  // Ground recovery (selection-layer repair): re-admit true-ground returns that
  // the 5-NN plane gates reject under ground-hugging clutter (grass, crops,
  // debris — content-agnostic) — see the block comment above the measurement
  // model's recoverGround. SELF-GATED by per-sector fit quality + the
  // sensor-height sanity gate; no scene classifier involved. Default OFF.
  bool ground_recover_en = false;
  double ground_recover_scan_band = 0.15;    // |dist to sector ground plane| accept band (m)
  int ground_recover_min_inliers = 100;      // PER-SECTOR fit support gate (8 sectors)
  double ground_recover_max_residual = 1.0;  // |z - map lower envelope| accept gate (m)
  double ground_recover_sector_tol = 0.25;   // sector-vs-global plane height agreement (m)
  int ground_recover_max_rows = 4000;        // per-scan cap on recovered rows

  // Neighbour refresh: the filter searches again once it declares itself converged, but if the
  // state barely moved since the last search the neighbour sets cannot have changed, and that
  // second 5-NN pass over the scan is the single most expensive item in the front end.
  bool nn_refresh_gate_en = false;
  double nn_refresh_min_shift = 0.0;
  double nn_refresh_ref_range = 20.0;
};
}  // namespace fast_lio

// runaway_watchdog.hpp — catch the drift that never trips the guard.
//
// The divergence guard only fires when the scan stops constraining the state.
// The worst failure mode has the opposite signature: after a bad re-anchor the
// map is rebuilt around the wrong pose every scan, so correspondences stay
// plentiful and residuals small while the estimate walks away from the world.
// Dog 2 held effct 2000-3500 and res 0.026 m for 45 minutes at 0.5 m/s — parked.
//
// Nothing inside the estimator can see that: the map agrees with itself. Only a
// source outside it can, and there are two — the wheels, and the IMU's own
// "am I moving at all" verdict. When the state claims motion that neither
// supports, and keeps claiming it, the estimate is gone and something has to
// say so.
//
// Pure C++. No ROS, no Eigen.

#pragma once

#include <cmath>
#include <cstdint>

namespace fast_lio
{

struct RunawayParams
{
  bool enabled = true;
  double parked_speed_thresh = 0.3;  ///< m/s of estimated speed that contradicts "parked"
  double wheel_speed_margin = 1.0;   ///< m/s the estimate may exceed the wheels by
  double hold_sec = 3.0;             ///< the contradiction must persist this long
  double wheel_max_age = 0.5;        ///< s; older wheel samples cannot contradict anything
  double wheel_moving_thresh = 0.05; ///< m/s the wheels must report before they may contradict
};

enum class RunawayReason : std::uint8_t
{
  kNone,
  kParked,  ///< independent evidence says parked, the estimate says otherwise
  kWheel,   ///< the estimate outruns the wheels by more than the margin
};

struct RunawayState
{
  double bad_since = -1.0;    ///< when the current contradiction started; <0 = none
  std::uint32_t trips = 0;    ///< diagnostics
  RunawayReason reason = RunawayReason::kNone;
};

struct RunawayVerdict
{
  bool tripped = false;                        ///< edge: the contradiction just outlasted hold_sec
  double bad_for = 0.0;                        ///< how long it has been contradicted (s)
  RunawayReason reason = RunawayReason::kNone;
};

/// One scan. `evidence_parked` is the independent static verdict
/// (static_evidence.hpp); `wheel_age` < 0 means no wheel sample at all.
inline RunawayVerdict updateRunawayWatchdog(RunawayState * state,
                                            double now,
                                            double state_speed,
                                            bool evidence_parked,
                                            double wheel_age,
                                            double wheel_speed,
                                            const RunawayParams & params)
{
  RunawayVerdict verdict;
  if (state == nullptr || !params.enabled || !std::isfinite(now) || !std::isfinite(state_speed)) {
    return verdict;
  }

  const bool wheel_fresh =
    wheel_age >= 0.0 && std::isfinite(wheel_age) && wheel_age <= params.wheel_max_age && std::isfinite(wheel_speed);
  RunawayReason reason = RunawayReason::kNone;
  if (evidence_parked && state_speed > params.parked_speed_thresh) {
    reason = RunawayReason::kParked;
  } else if (wheel_fresh && wheel_speed > params.wheel_moving_thresh &&
             state_speed > wheel_speed + params.wheel_speed_margin) {
    // Covers the moving case, where "parked" says nothing: a front end sliding
    // along its own map outruns the wheels even while the robot really drives.
    // The wheels must report motion of their own to contradict anything — a
    // source stuck at zero (m20 feeds a GNSS velocity proxy here, which reads
    // 0.00 for the whole GNSS-denied stretch) would otherwise accuse a healthy
    // walk of running away. A speed of zero is the parked branch's business,
    // and that one needs the IMU to agree.
    reason = RunawayReason::kWheel;
  }

  if (reason == RunawayReason::kNone) {
    state->bad_since = -1.0;
    state->reason = RunawayReason::kNone;
    return verdict;
  }
  // A clock that jumps backwards (bag restart) restarts the window rather than
  // instantly satisfying it.
  if (state->bad_since < 0.0 || now < state->bad_since) {
    state->bad_since = now;
  }
  state->reason = reason;
  verdict.reason = reason;
  verdict.bad_for = now - state->bad_since;
  if (verdict.bad_for >= params.hold_sec) {
    verdict.tripped = true;
    ++state->trips;
    state->bad_since = -1.0;  // re-arm: the caller acts once per episode
  }
  return verdict;
}

inline const char * runawayReasonName(RunawayReason reason)
{
  switch (reason) {
    case RunawayReason::kParked:
      return "parked-but-moving";
    case RunawayReason::kWheel:
      return "outruns-wheels";
    default:
      return "none";
  }
}

}  // namespace fast_lio

// reanchor_gate.hpp — do not trust the map again until the scan proves it.
//
// The divergence guard freezes the map while the scan cannot constrain the
// state, and releases as soon as the scene looks healthy again. That release
// was unconditional, and it is where a recoverable blind stretch turns into a
// permanent runaway: whatever pose error the blind stretch produced is still
// there, the map unfreezes around it, the next scans are inserted at the wrong
// pose, and from then on the map moves with the estimate — residuals stay small
// while the robot's own map walks away from the world (dog 2, 2026-09-04: 9 s
// blind, then 45 min of 0.5 m/s drift while parked, residual 0.026 m).
//
// So the release becomes a verification: keep the pre-blind map frozen and
// require the scan to register against IT — enough correspondences, small
// residual, for several consecutive scans — before the map may grow again. A
// verification that never converges is a lost front end, which is a fact the
// rest of the system needs to hear rather than a state to paper over.
//
// Pure C++. No ROS, no Eigen.

#pragma once

#include <cmath>
#include <cstdint>

namespace fast_lio
{

enum class ReanchorState : std::uint8_t
{
  kHealthy,    ///< the map is trusted and growing
  kBlind,      ///< the guard is degraded: no usable correspondences
  kVerifying,  ///< scene looks healthy again; proving the pose still fits the frozen map
  kLost,       ///< verification timed out — the pose is not recoverable from this map
};

struct ReanchorParams
{
  bool enabled = true;
  int min_eff = 200;         ///< effective correspondences that count as re-registered
  double max_res = 0.10;     ///< m; mean point-to-plane residual of a trustworthy match
  int confirm_scans = 3;     ///< consecutive good scans before the map may grow again
  double timeout_sec = 5.0;  ///< give up verifying after this long
};

struct ReanchorGate
{
  ReanchorState state = ReanchorState::kHealthy;
  int ok_streak = 0;
  double verify_start = -1.0;
  std::uint32_t reanchors = 0;  ///< successful verifications (diagnostics)
  std::uint32_t losses = 0;     ///< timed-out verifications (diagnostics)
};

struct ReanchorDecision
{
  bool map_frozen = false;       ///< keep the map frozen this scan
  bool degraded = false;         ///< report the odometry degraded downstream
  bool just_reanchored = false;  ///< edge: verification passed this scan
  bool just_lost = false;        ///< edge: verification timed out this scan
};

/// One scan of gate bookkeeping. `guard_degraded` is the divergence guard's own
/// verdict (starvation / engulfment / velocity runaway); `effct` and `res_mean`
/// describe the match this scan achieved against the (still frozen) map.
inline ReanchorDecision updateReanchorGate(ReanchorGate * gate,
                                           bool guard_degraded,
                                           int effct,
                                           double res_mean,
                                           double now,
                                           const ReanchorParams & params)
{
  ReanchorDecision decision;
  if (gate == nullptr) {
    return decision;
  }
  if (!params.enabled) {
    // Pre-gate behaviour: the guard's verdict is the whole story.
    gate->state = guard_degraded ? ReanchorState::kBlind : ReanchorState::kHealthy;
    gate->ok_streak = 0;
    gate->verify_start = -1.0;
    decision.map_frozen = guard_degraded;
    decision.degraded = guard_degraded;
    return decision;
  }

  if (guard_degraded) {
    // Any fresh degradation restarts the whole recovery: a verification that was
    // already under way tells us nothing about the map after another blind run.
    gate->state = ReanchorState::kBlind;
    gate->ok_streak = 0;
    gate->verify_start = -1.0;
    decision.map_frozen = true;
    decision.degraded = true;
    return decision;
  }

  switch (gate->state) {
    case ReanchorState::kHealthy:
      break;

    case ReanchorState::kBlind:
      // The scene recovered. The pose has not been proven yet, so the map stays
      // frozen: this scan is the first piece of evidence, not a licence to grow.
      gate->state = ReanchorState::kVerifying;
      gate->ok_streak = 0;
      gate->verify_start = now;
      [[fallthrough]];

    case ReanchorState::kVerifying: {
      const bool match_ok =
        effct >= params.min_eff && std::isfinite(res_mean) && res_mean >= 0.0 && res_mean <= params.max_res;
      gate->ok_streak = match_ok ? gate->ok_streak + 1 : 0;
      if (gate->ok_streak >= params.confirm_scans) {
        gate->state = ReanchorState::kHealthy;
        gate->ok_streak = 0;
        gate->verify_start = -1.0;
        ++gate->reanchors;
        decision.just_reanchored = true;
        break;  // trusted again from this scan on
      }
      // A clock that jumps backwards (bag restart) must not expire the window.
      const bool timed_out = params.timeout_sec > 0.0 && gate->verify_start >= 0.0 && now >= gate->verify_start &&
                             (now - gate->verify_start) > params.timeout_sec;
      if (timed_out) {
        gate->state = ReanchorState::kLost;
        gate->ok_streak = 0;
        ++gate->losses;
        decision.just_lost = true;
      }
      decision.map_frozen = true;
      decision.degraded = true;
      break;
    }

    case ReanchorState::kLost:
      // Nothing here recovers on its own: the caller either rebuilds the map or
      // waits for an external relocalisation, and calls resetReanchorGate().
      decision.map_frozen = true;
      decision.degraded = true;
      break;
  }
  return decision;
}

/// Declare the map trustworthy again — after a caller-side map rebuild or an
/// external relocalisation.
inline void resetReanchorGate(ReanchorGate * gate)
{
  if (gate == nullptr) {
    return;
  }
  gate->state = ReanchorState::kHealthy;
  gate->ok_streak = 0;
  gate->verify_start = -1.0;
}

}  // namespace fast_lio

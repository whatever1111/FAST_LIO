// zupt_policy.hpp — what one scan may do to the velocity state when the
// evidence outside the estimator says the platform is parked.
//
// Two branches used to answer this question in laserMapping.cpp and only one of
// them ever asked. The static hold (degraded scan) consulted bodyIsStatic(); the
// healthy branch consulted nothing, so a parked platform whose scan still looked
// good got no zero-velocity update at all. That is the state the front end runs
// in most of the time, and it is exactly where the failure lives: with far_frac
// 0.84 the correspondences pin rotation and leave translation nearly free, so
// the velocity rides the IMU bias while effct and the residual stay healthy.
// Dog 2 (2026-09-05) had 21 of 22 runaway trips fire from that branch, the OEM
// body velocity at 0.03 m/s against an estimate claiming 0.4-2.6 m/s.
//
// The two branches also have to answer "is it parked?" the SAME way as the
// runaway watchdog, or the guard and the watchdog spend the run contradicting
// each other. One verdict per scan, one decision table, one place to test it.
//
// What the verdict may cost is deliberately asymmetric:
//   - blind + parked  → pin pose AND attitude back to the anchor. Zeroing only
//     the velocity leaves position and attitude wherever propagation put them
//     (dog 2, 2026-09-04: -3.9 m of Z in 9 s, unrecoverable once the map
//     unfroze around the wrong pose).
//   - healthy + parked → zero the velocity only. The scan owns the pose here,
//     so a false "parked" costs one scan of re-convergence rather than a pin at
//     the wrong place.
//
// Pure C++. No ROS, no Eigen.

#pragma once

#include <cstdint>

namespace fast_lio
{

/// m/s; a parked correction that shaves off less than this is filter noise, not news.
constexpr double kZuptPhantomReportSpeed = 0.1;

enum class ZuptAction : std::uint8_t
{
  kNone,          ///< leave the state alone
  kPinToAnchor,   ///< pos and rot back to the anchor, velocity zeroed
  kZeroVelocity,  ///< velocity zeroed; pose and attitude left where they are
};

struct ZuptPolicyInputs
{
  bool scan_degraded = false;  ///< the divergence guard says this scan cannot confirm the pose
  bool zupt_enabled = false;   ///< zupt_en
  bool anchor_valid = false;   ///< a healthy scan has left an anchor to pin to
  bool body_static = false;    ///< bodyIsStatic() for this scan (static_evidence.hpp)
  double speed = 0.0;          ///< |v| of the current state (m/s)
  double max_speed = 0.0;      ///< divergence_guard_max_speed (m/s)
};

struct ZuptDecision
{
  ZuptAction action = ZuptAction::kNone;
  bool hold = false;    ///< the static hold is engaged this scan (release it whenever false)
  bool report = false;  ///< the correction is large enough to be worth a log line
};

/// One scan's verdict. Non-finite speeds compare false everywhere, so they fall
/// through to kNone rather than triggering a correction on a number nobody trusts
/// — except on the pin branch, which does not look at the speed at all: with
/// independent proof of "parked" a blind stretch has nothing to integrate,
/// whatever the velocity claims.
inline ZuptDecision decideZupt(const ZuptPolicyInputs & in)
{
  ZuptDecision decision;
  if (in.scan_degraded) {
    if (in.zupt_enabled && in.anchor_valid && in.body_static) {
      decision.action = ZuptAction::kPinToAnchor;
      decision.hold = true;
      return decision;
    }
    // Not parked, and the scan still cannot confirm the pose: this is the
    // stretch the hold must NOT survive into. A body speed above the platform's
    // physical maximum is a GUARANTEED-WRONG state, not a bound to enforce —
    // clamping it to max_speed made the clamp an equilibrium (m20_0814_degrade:
    // spd=3.00 on every scan for 60 s), so zero it and let the next
    // correspondences re-estimate it.
    if (in.speed > in.max_speed) {
      decision.action = ZuptAction::kZeroVelocity;
    }
    return decision;
  }

  if (in.zupt_enabled && in.body_static && in.speed > 0.0) {
    decision.action = ZuptAction::kZeroVelocity;
    decision.report = in.speed > kZuptPhantomReportSpeed;
  }
  return decision;
}

}  // namespace fast_lio

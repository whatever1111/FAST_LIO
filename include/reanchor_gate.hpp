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
// The absolute bounds are not enough on their own. A registration started from
// a wrong prior converges to a self-consistent wrong minimum whose residual is
// well under reanchor_max_res and still clearly worse than what the same scene
// gives when healthy (m20 0826 after the data hole: res 0.045-0.083 m against a
// healthy 0.03-0.04). The optional relative test judges the residual (and the
// correspondence count) against running medians of recent healthy scans, the
// way KISS-ICP adapts its threshold to the running statistics of its own
// corrections, with a floor so that a very clean scene does not demand the
// impossible.
//
// The divergence guard debounces its own verdict: it counts a scan whose far field collapsed (or whose
// correspondences starved) at once, and declares the scene degraded only after a streak of them. The first scan of
// such a stretch is therefore not degraded yet, but the guard has already called it bad, and a bad scan proves
// nothing about the frozen map. With require_guard_clear the verification takes as evidence only scans the guard
// counted nothing against.
//
// Pure C++. No ROS, no Eigen.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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
  /// > 0: a verification scan must also have res_mean <= max(res_floor, max_res_ratio x the healthy median
  /// residual). 0 = off, the absolute bounds above decide alone.
  double max_res_ratio = 0.0;
  double res_floor = 0.04;  ///< m; the relative residual bound never tightens below this
  /// > 0: a verification scan must also have effct >= min_eff_ratio x the healthy median effct. 0 = off.
  double min_eff_ratio = 0.0;
};

/// What a healthy scan looks like right now: running medians the caller keeps (HealthyScanWindow). NaN means not
/// known yet, and the relative tests are then skipped.
struct ReanchorReference
{
  double res_median = std::numeric_limits<double>::quiet_NaN();
  double eff_median = std::numeric_limits<double>::quiet_NaN();
};

/// Whether one scan's match against the frozen map counts as re-registration evidence.
inline bool reanchorMatchOk(int effct,
                            double res_mean,
                            const ReanchorParams & params,
                            const ReanchorReference & reference = ReanchorReference{})
{
  if (effct < params.min_eff || !std::isfinite(res_mean) || res_mean < 0.0 || res_mean > params.max_res) {
    return false;
  }
  if (params.max_res_ratio > 0.0 && std::isfinite(reference.res_median) && reference.res_median > 0.0) {
    const double bound = std::max(params.res_floor, params.max_res_ratio * reference.res_median);
    if (res_mean > bound) {
      return false;
    }
  }
  if (params.min_eff_ratio > 0.0 && std::isfinite(reference.eff_median) && reference.eff_median > 0.0 &&
      static_cast<double>(effct) < params.min_eff_ratio * reference.eff_median) {
    return false;
  }
  return true;
}

/// Median of the last `capacity` finite values pushed; NaN until `min_count` of them have been seen. The caller
/// pushes the res_mean / effct of scans the gate called healthy.
class HealthyScanWindow
{
public:
  explicit HealthyScanWindow(std::size_t capacity = 100, std::size_t min_count = 20)
      : capacity_(std::max<std::size_t>(capacity, 1)), min_count_(std::max<std::size_t>(min_count, 1))
  {
    values_.reserve(capacity_);
  }

  void push(double value)
  {
    if (!std::isfinite(value)) {
      return;
    }
    if (values_.size() < capacity_) {
      values_.push_back(value);
    } else {
      values_[next_] = value;
    }
    next_ = (next_ + 1) % capacity_;
  }

  double median() const
  {
    if (values_.size() < min_count_) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> sorted = values_;
    const auto mid = sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2);
    std::nth_element(sorted.begin(), mid, sorted.end());
    return *mid;
  }

  std::size_t size() const { return values_.size(); }
#ifdef FASTLIO_TEST_HOOKS
  std::size_t capacityForTest() const { return capacity_; }
  std::size_t minCountForTest() const { return min_count_; }
#endif

  void clear()
  {
    values_.clear();
    next_ = 0;
  }

private:
  std::size_t capacity_;
  std::size_t min_count_;
  std::size_t next_ = 0;
  std::vector<double> values_;
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

/// What one scan brings to the gate.
struct ReanchorScan
{
  int effct = 0;          ///< effective correspondences of the match against the (still frozen) map
  double res_mean = 0.0;  ///< m; their mean point-to-plane residual
  double now = 0.0;       ///< s; the scan's time
  /// The divergence guard counted nothing against this scan: no engulfment or starvation vote, whether or not a
  /// streak of such votes has made the scene degraded.
  bool guard_clear = true;
  /// true: a verification scan must also be guard_clear (reanchor_require_guard_clear). The caller keeps this switch
  /// with its other parameters and hands it over with every scan. false = the match alone decides.
  bool require_guard_clear = false;
};

/// One scan of gate bookkeeping. `guard_degraded` is the divergence guard's own
/// verdict (starvation / engulfment / velocity runaway); `scan` is the match this
/// scan achieved against the (still frozen) map and whether the guard counted
/// against it; `reference` is what healthy scans look like (only read by the
/// relative tests).
inline ReanchorDecision updateReanchorGate(ReanchorGate * gate,
                                           bool guard_degraded,
                                           const ReanchorScan & scan,
                                           const ReanchorParams & params,
                                           const ReanchorReference & reference = ReanchorReference{})
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
      gate->verify_start = scan.now;
      [[fallthrough]];

    case ReanchorState::kVerifying: {
      // Under require_guard_clear a scan the guard counted against is a failed match: it breaks the run, and the
      // timeout runs on.
      const bool match_ok = reanchorMatchOk(scan.effct, scan.res_mean, params, reference) &&
                            (scan.guard_clear || !scan.require_guard_clear);
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
      const bool timed_out = params.timeout_sec > 0.0 && gate->verify_start >= 0.0 && scan.now >= gate->verify_start &&
                             (scan.now - gate->verify_start) > params.timeout_sec;
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

/// The same with the match as separate values and the guard's count left out: the match alone decides.
inline ReanchorDecision updateReanchorGate(ReanchorGate * gate,
                                           bool guard_degraded,
                                           int effct,
                                           double res_mean,
                                           double now,
                                           const ReanchorParams & params,
                                           const ReanchorReference & reference = ReanchorReference{})
{
  return updateReanchorGate(gate, guard_degraded, ReanchorScan{effct, res_mean, now, true, false}, params, reference);
}

/// Declare the front end lost from outside the gate — the runaway watchdog sees
/// drift the scan itself never reveals, and its verdict has the same
/// consequences: freeze the map and stop claiming the pose is usable.
inline void tripReanchorLost(ReanchorGate * gate)
{
  if (gate == nullptr || gate->state == ReanchorState::kLost) {
    return;
  }
  gate->state = ReanchorState::kLost;
  gate->ok_streak = 0;
  gate->verify_start = -1.0;
  ++gate->losses;
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

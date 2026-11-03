#pragma once

#include "stationary_anchor_policy.hpp"

namespace fast_lio
{

constexpr std::uint64_t kMaxExactCoordinateEpoch = 1ULL << 53;

struct LostRecoveryParams
{
  bool enabled = false;
  std::int64_t stable_ns = 2000000000;
};

inline bool resolveLostRecoveryParams(bool enabled, double stable_s, double total_s, LostRecoveryParams * out)
{
  if (!std::isfinite(stable_s) || !(stable_s > 0.0) || (enabled && (!std::isfinite(total_s) || !(total_s > 0.0)))) {
    return false;
  }
  const double ns = std::round(stable_s * 1e9);
  if (!(ns >= 1.0) || ns > kAnchorMaxNs) {
    return false;
  }
  *out = {enabled, static_cast<std::int64_t>(ns)};
  return true;
}

struct LostRecoveryState
{
  std::uint64_t coordinate_epoch = 1;
  bool candidate = false;
  std::int64_t candidate_begin_ns = -1;
  std::uint64_t candidate_generation = 0;
  std::uint64_t candidate_last_scan = 0;
  std::uint64_t epochs_committed = 0;
  std::uint64_t attempts = 0;
};

inline bool lostRecoveryArmed(const RecoveryState & recovery, const LostRecoveryParams & params)
{
  return params.enabled && recovery.trust == RecoveryTrust::kUnanchored;
}

inline void abandonLostRecovery(LostRecoveryState * state, const LostRecoveryParams & params)
{
  if (params.enabled) {
    state->candidate = false;
    state->candidate_begin_ns = -1;
  }
}

inline void beginLostRecoveryCandidate(LostRecoveryState * state,
                                       std::uint64_t generation,
                                       std::uint64_t scan,
                                       std::int64_t begin_ns,
                                       const LostRecoveryParams & params)
{
  if (!params.enabled) {
    return;
  }
  state->candidate = true;
  state->candidate_begin_ns = begin_ns;
  state->candidate_generation = generation;
  state->candidate_last_scan = scan;
  if (state->attempts != std::numeric_limits<std::uint64_t>::max()) {
    ++state->attempts;
  }
}

// A new epoch qualifies only local consistency; it never restores the old epoch's trust.
inline bool commitLostRecovery(LostRecoveryState * state,
                               RecoveryState * recovery,
                               std::int64_t end_ns,
                               const LostRecoveryParams & params)
{
  if (!lostRecoveryArmed(*recovery, params) || !state->candidate || state->candidate_begin_ns < 0 ||
      end_ns < state->candidate_begin_ns || end_ns - state->candidate_begin_ns < params.stable_ns ||
      state->coordinate_epoch == kMaxExactCoordinateEpoch) {
    return false;
  }
  ++state->coordinate_epoch;
  ++state->epochs_committed;
  abandonLostRecovery(state, params);
  recovery->trust = RecoveryTrust::kLocalQualified;
  recovery->cause = RecoveryCause::kStable;
  recovery->last_supported_end_ns = end_ns;
  recovery->stable_begin_ns = -1;
  return true;
}

inline bool observeLostRecoveryCandidate(LostRecoveryState * state,
                                         RecoveryState * recovery,
                                         std::uint64_t scan,
                                         std::int64_t begin_ns,
                                         std::int64_t end_ns,
                                         bool match_ok,
                                         std::uint64_t generation,
                                         const LostRecoveryParams & params)
{
  if (!lostRecoveryArmed(*recovery, params) || !state->candidate) {
    return false;
  }
  if (!match_ok || scan == 0 || begin_ns < 0 || end_ns < begin_ns || recoveryScanExcluded(*recovery, scan)) {
    state->candidate_begin_ns = -1;
    return false;
  }
  const bool continues = state->candidate_begin_ns >= 0 && state->candidate_last_scan + 1 == scan &&
                         state->candidate_generation == generation;
  if (!continues) {
    state->candidate_begin_ns = begin_ns;
    state->candidate_generation = generation;
  }
  state->candidate_last_scan = scan;
  return commitLostRecovery(state, recovery, end_ns, params);
}

}  // namespace fast_lio

#include "reanchor_gate.hpp"

#include <gtest/gtest.h>

#include <cmath>

using fast_lio::ReanchorGate;
using fast_lio::ReanchorParams;
using fast_lio::ReanchorState;
using fast_lio::resetReanchorGate;
using fast_lio::updateReanchorGate;

namespace
{
constexpr int kGoodEff = 2500;   // a healthy indoor scan on the M20
constexpr double kGoodRes = 0.011;
constexpr int kBlindEff = 0;     // effct during the dog-2 engulfment
constexpr double kBlindRes = 0.0;
}  // namespace

TEST(ReanchorGate, HealthyScansNeverFreeze)
{
  ReanchorGate g;
  ReanchorParams p;
  for (int i = 0; i < 10; ++i) {
    const auto d = updateReanchorGate(&g, false, kGoodEff, kGoodRes, i * 0.1, p);
    EXPECT_FALSE(d.map_frozen) << i;
    EXPECT_FALSE(d.degraded) << i;
  }
  EXPECT_EQ(g.state, ReanchorState::kHealthy);
  EXPECT_EQ(g.reanchors, 0u);
}

TEST(ReanchorGate, ReleaseIsVerifiedBeforeTheMapMayGrow)
{
  ReanchorGate g;
  ReanchorParams p;  // confirm_scans = 3
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p);
  EXPECT_EQ(g.state, ReanchorState::kBlind);

  // The scene is healthy again, but the match must prove the pose first.
  for (int i = 1; i <= 2; ++i) {
    const auto d = updateReanchorGate(&g, false, kGoodEff, kGoodRes, i * 0.1, p);
    EXPECT_TRUE(d.map_frozen) << i;
    EXPECT_TRUE(d.degraded) << i;
    EXPECT_FALSE(d.just_reanchored) << i;
    EXPECT_EQ(g.state, ReanchorState::kVerifying) << i;
  }
  const auto d = updateReanchorGate(&g, false, kGoodEff, kGoodRes, 0.3, p);
  EXPECT_TRUE(d.just_reanchored);
  EXPECT_FALSE(d.map_frozen);
  EXPECT_FALSE(d.degraded);
  EXPECT_EQ(g.state, ReanchorState::kHealthy);
  EXPECT_EQ(g.reanchors, 1u);
}

TEST(ReanchorGate, PartialMatchesDoNotAccumulate)
{
  // The failure mode the streak exists for: a scan that half-matches the frozen
  // map every other frame must never add up to a release.
  ReanchorGate g;
  ReanchorParams p;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p);
  for (int i = 1; i <= 20; ++i) {
    const bool good = (i % 2) == 0;
    const auto d = updateReanchorGate(&g, false, good ? kGoodEff : 40, good ? kGoodRes : 0.5, i * 0.1, p);
    EXPECT_FALSE(d.just_reanchored) << i;
  }
  EXPECT_NE(g.state, ReanchorState::kHealthy);
}

TEST(ReanchorGate, VerificationTimesOutIntoLost)
{
  ReanchorGate g;
  ReanchorParams p;
  p.timeout_sec = 1.0;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 10.0, p);
  auto d = updateReanchorGate(&g, false, 10, 0.9, 10.1, p);  // verification starts
  EXPECT_EQ(g.state, ReanchorState::kVerifying);
  EXPECT_FALSE(d.just_lost);
  d = updateReanchorGate(&g, false, 10, 0.9, 11.5, p);
  EXPECT_TRUE(d.just_lost);
  EXPECT_EQ(g.state, ReanchorState::kLost);
  EXPECT_EQ(g.losses, 1u);
  // Lost stays lost — and stays frozen — until the caller acts.
  for (int i = 0; i < 5; ++i) {
    const auto held = updateReanchorGate(&g, false, kGoodEff, kGoodRes, 12.0 + i, p);
    EXPECT_TRUE(held.map_frozen) << i;
    EXPECT_TRUE(held.degraded) << i;
    EXPECT_FALSE(held.just_reanchored) << i;
  }
  resetReanchorGate(&g);
  EXPECT_EQ(g.state, ReanchorState::kHealthy);
  EXPECT_FALSE(updateReanchorGate(&g, false, kGoodEff, kGoodRes, 20.0, p).map_frozen);
}

TEST(ReanchorGate, NewBlindStretchRestartsTheVerification)
{
  ReanchorGate g;
  ReanchorParams p;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p);
  updateReanchorGate(&g, false, kGoodEff, kGoodRes, 0.1, p);
  updateReanchorGate(&g, false, kGoodEff, kGoodRes, 0.2, p);
  EXPECT_EQ(g.ok_streak, 2);
  const auto d = updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.3, p);
  EXPECT_EQ(g.state, ReanchorState::kBlind);
  EXPECT_EQ(g.ok_streak, 0);
  EXPECT_TRUE(d.map_frozen);
  // Two more good scans must NOT be enough now: the streak restarted.
  updateReanchorGate(&g, false, kGoodEff, kGoodRes, 0.4, p);
  EXPECT_FALSE(updateReanchorGate(&g, false, kGoodEff, kGoodRes, 0.5, p).just_reanchored);
}

TEST(ReanchorGate, NonFiniteResidualIsNeverEvidence)
{
  ReanchorGate g;
  ReanchorParams p;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p);
  for (int i = 1; i <= 5; ++i) {
    const auto d = updateReanchorGate(&g, false, kGoodEff, std::nan(""), i * 0.1, p);
    EXPECT_FALSE(d.just_reanchored) << i;
  }
  EXPECT_EQ(g.ok_streak, 0);
}

TEST(ReanchorGate, BackwardsClockDoesNotExpireTheWindow)
{
  // Offline replay restarts the bag: `now` jumps backwards mid-verification.
  ReanchorGate g;
  ReanchorParams p;
  p.timeout_sec = 1.0;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 100.0, p);
  updateReanchorGate(&g, false, 10, 0.9, 100.1, p);
  const auto d = updateReanchorGate(&g, false, 10, 0.9, 0.5, p);
  EXPECT_FALSE(d.just_lost);
  EXPECT_EQ(g.state, ReanchorState::kVerifying);
}

TEST(ReanchorGate, DisabledGateReproducesThePreGateRelease)
{
  ReanchorGate g;
  ReanchorParams p;
  p.enabled = false;
  EXPECT_TRUE(updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p).map_frozen);
  const auto d = updateReanchorGate(&g, false, 10, 0.9, 0.1, p);  // still a bad match
  EXPECT_FALSE(d.map_frozen);                                     // ...released anyway, as before
  EXPECT_FALSE(d.degraded);
  EXPECT_EQ(g.state, ReanchorState::kHealthy);
}

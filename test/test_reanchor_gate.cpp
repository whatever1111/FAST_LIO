#include "reanchor_gate.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

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

// ---- relative verification: the residual judged against what this scene gives when healthy ----

namespace
{
fast_lio::ReanchorReference healthyReference(double res_median, double eff_median)
{
  fast_lio::ReanchorReference ref;
  ref.res_median = res_median;
  ref.eff_median = eff_median;
  return ref;
}
}  // namespace

TEST(ReanchorGate, RelativeTestsAreOffByDefault)
{
  ReanchorParams p;
  const auto ref = healthyReference(0.03, 1500.0);
  EXPECT_TRUE(fast_lio::reanchorMatchOk(300, 0.09, p, ref));  // 3x the healthy residual, still under 0.10
  EXPECT_FALSE(fast_lio::reanchorMatchOk(300, 0.11, p, ref));
  EXPECT_FALSE(fast_lio::reanchorMatchOk(150, 0.03, p, ref));  // min_eff 200
}

TEST(ReanchorGate, RelativeResidualRejectsASelfConsistentWrongMinimum)
{
  // m20 0826 after the data hole: healthy median ~0.035 m, the wrong minimum 0.045-0.083 m.
  ReanchorParams p;
  p.max_res_ratio = 1.5;
  p.res_floor = 0.04;
  const auto ref = healthyReference(0.035, 1500.0);
  EXPECT_TRUE(fast_lio::reanchorMatchOk(800, 0.045, p, ref));  // bound 0.0525
  EXPECT_FALSE(fast_lio::reanchorMatchOk(800, 0.060, p, ref));
  EXPECT_FALSE(fast_lio::reanchorMatchOk(800, 0.083, p, ref));

  ReanchorGate g;
  updateReanchorGate(&g, true, kBlindEff, kBlindRes, 0.0, p, ref);
  for (int i = 1; i <= 5; ++i) {
    const auto d = updateReanchorGate(&g, false, 800, 0.070, i * 0.1, p, ref);
    EXPECT_FALSE(d.just_reanchored) << i;
    EXPECT_TRUE(d.map_frozen) << i;
  }
  EXPECT_EQ(g.state, ReanchorState::kVerifying);
  // The same gate without a reference (not enough healthy scans yet) would have released on the absolute bound.
  ReanchorGate g_abs;
  updateReanchorGate(&g_abs, true, kBlindEff, kBlindRes, 0.0, p);
  for (int i = 1; i <= 3; ++i) {
    updateReanchorGate(&g_abs, false, 800, 0.070, i * 0.1, p);
  }
  EXPECT_EQ(g_abs.state, ReanchorState::kHealthy);
  // Registrations as good as the healthy scene release as before.
  for (int i = 6; i <= 8; ++i) {
    updateReanchorGate(&g, false, 800, 0.036, i * 0.1, p, ref);
  }
  EXPECT_EQ(g.state, ReanchorState::kHealthy);
}

TEST(ReanchorGate, RelativeBoundKeepsItsFloorAndTheAbsoluteCeiling)
{
  ReanchorParams p;
  p.max_res_ratio = 1.5;
  p.res_floor = 0.04;
  // A very clean scene (median 1 cm) does not demand 1.5 cm: the floor holds.
  EXPECT_TRUE(fast_lio::reanchorMatchOk(800, 0.039, p, healthyReference(0.01, 1500.0)));
  EXPECT_FALSE(fast_lio::reanchorMatchOk(800, 0.041, p, healthyReference(0.01, 1500.0)));
  // A rough scene (median 0.2 m) never relaxes the absolute 0.10 m ceiling.
  EXPECT_FALSE(fast_lio::reanchorMatchOk(800, 0.11, p, healthyReference(0.2, 1500.0)));
  EXPECT_TRUE(fast_lio::reanchorMatchOk(800, 0.09, p, healthyReference(0.2, 1500.0)));
}

TEST(ReanchorGate, UnknownOrBrokenReferenceFallsBackToTheAbsoluteTest)
{
  ReanchorParams p;
  p.max_res_ratio = 1.5;
  p.min_eff_ratio = 0.5;
  const double nan = std::nan("");
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(fast_lio::reanchorMatchOk(300, 0.09, p, healthyReference(nan, nan)));
  EXPECT_TRUE(fast_lio::reanchorMatchOk(300, 0.09, p, healthyReference(inf, inf)));
  EXPECT_TRUE(fast_lio::reanchorMatchOk(300, 0.09, p, healthyReference(0.0, 0.0)));
  EXPECT_TRUE(fast_lio::reanchorMatchOk(300, 0.09, p, healthyReference(-1.0, -5.0)));
  EXPECT_FALSE(fast_lio::reanchorMatchOk(300, nan, p, healthyReference(0.03, 1000.0)));
  EXPECT_FALSE(fast_lio::reanchorMatchOk(300, inf, p, healthyReference(0.03, 1000.0)));
}

TEST(ReanchorGate, RelativeCorrespondenceCount)
{
  ReanchorParams p;
  p.min_eff_ratio = 0.5;
  const auto ref = healthyReference(0.03, 1500.0);
  EXPECT_FALSE(fast_lio::reanchorMatchOk(700, 0.03, p, ref));                            // under half the healthy count
  EXPECT_TRUE(fast_lio::reanchorMatchOk(750, 0.03, p, ref));                             // exactly half: inclusive
  EXPECT_FALSE(fast_lio::reanchorMatchOk(199, 0.03, p, healthyReference(0.03, 100.0)));  // min_eff still applies
}

TEST(ReanchorGate, HealthyScanWindowKeepsTheMedianOfTheRecentPast)
{
  fast_lio::HealthyScanWindow w(5, 3);
  EXPECT_TRUE(std::isnan(w.median()));
  w.push(0.03);
  w.push(std::nan(""));                             // ignored
  w.push(std::numeric_limits<double>::infinity());  // ignored
  w.push(0.05);
  EXPECT_EQ(w.size(), 2u);
  EXPECT_TRUE(std::isnan(w.median()));  // fewer than min_count
  w.push(0.04);
  EXPECT_DOUBLE_EQ(w.median(), 0.04);
  // The ring buffer forgets the oldest values once full.
  for (int i = 0; i < 5; ++i) {
    w.push(0.10);
  }
  EXPECT_EQ(w.size(), 5u);
  EXPECT_DOUBLE_EQ(w.median(), 0.10);
  w.clear();
  EXPECT_EQ(w.size(), 0u);
  EXPECT_TRUE(std::isnan(w.median()));
  // Degenerate sizes are clamped to one value.
  fast_lio::HealthyScanWindow one(0, 0);
  one.push(0.02);
  one.push(0.07);
  EXPECT_EQ(one.size(), 1u);
  EXPECT_DOUBLE_EQ(one.median(), 0.07);
}

#include "zupt_policy.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using fast_lio::decideZupt;
using fast_lio::kZuptPhantomReportSpeed;
using fast_lio::ZuptAction;
using fast_lio::ZuptDecision;
using fast_lio::ZuptPolicyInputs;

namespace
{
// The deployed configuration: ZUPT on, an anchor already laid down by an earlier
// healthy scan, and the divergence guard's speed ceiling at its default.
ZuptPolicyInputs deployed()
{
  ZuptPolicyInputs in;
  in.zupt_enabled = true;
  in.anchor_valid = true;
  in.max_speed = 30.0;
  return in;
}
}  // namespace

// ── The bug this table exists for ────────────────────────────────────────────
// Dog 2, 2026-09-05: parked, the scan healthy (far_frac 0.84 pins rotation and
// leaves translation nearly free), and the velocity state riding the IMU bias up
// to 2.6 m/s while nothing on that branch ever consulted the static evidence.
TEST(ZuptPolicy, ParkedWithHealthyScanZeroesPhantomVelocity)
{
  ZuptPolicyInputs in = deployed();
  in.scan_degraded = false;
  in.body_static = true;
  in.speed = 2.6;

  const ZuptDecision d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kZeroVelocity);
  EXPECT_FALSE(d.hold);  // the scan owns the pose here — nothing is pinned
  EXPECT_TRUE(d.report);
}

TEST(ZuptPolicy, HealthyScanLeavesRealMotionAlone)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = false;
  in.speed = 1.2;

  EXPECT_EQ(decideZupt(in).action, ZuptAction::kNone);
}

// A false "parked" must cost one scan of re-convergence, never a pose pin: the
// healthy branch may zero the velocity and nothing else.
TEST(ZuptPolicy, HealthyBranchNeverPinsThePose)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = true;
  for (const double speed : {0.0, 0.02, 0.4, 2.6, 40.0}) {
    in.speed = speed;
    const ZuptDecision d = decideZupt(in);
    EXPECT_NE(d.action, ZuptAction::kPinToAnchor) << "speed " << speed;
    EXPECT_FALSE(d.hold) << "speed " << speed;
  }
}

// Below the reporting speed the correction still happens — it just does not get
// a log line. A 5 cm/s shave is filter noise; 0.4 m/s is the failure.
TEST(ZuptPolicy, SmallParkedCorrectionsAreSilent)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = true;

  in.speed = kZuptPhantomReportSpeed / 2.0;
  ZuptDecision d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kZeroVelocity);
  EXPECT_FALSE(d.report);

  in.speed = kZuptPhantomReportSpeed * 4.0;
  d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kZeroVelocity);
  EXPECT_TRUE(d.report);
}

// Nothing to correct: an already-zero velocity must not be "zeroed" again, or
// every parked scan reports a correction it did not make.
TEST(ZuptPolicy, AlreadyStoppedIsLeftAlone)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = true;
  in.speed = 0.0;

  const ZuptDecision d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kNone);
  EXPECT_FALSE(d.report);
}

TEST(ZuptPolicy, DisabledZuptTouchesNothingOnAHealthyScan)
{
  ZuptPolicyInputs in = deployed();
  in.zupt_enabled = false;
  in.body_static = true;
  in.speed = 2.6;

  EXPECT_EQ(decideZupt(in).action, ZuptAction::kNone);
}

// ── The blind branch ─────────────────────────────────────────────────────────
// Parked with a scan that cannot confirm the pose: pin pose AND attitude. Zeroing
// only the velocity left propagation free to walk the state (dog 2, 2026-09-04:
// -3.9 m of Z in 9 s, unrecoverable once the map unfroze around the wrong pose).
TEST(ZuptPolicy, BlindAndParkedPinsToTheAnchor)
{
  ZuptPolicyInputs in = deployed();
  in.scan_degraded = true;
  in.body_static = true;
  in.speed = 1.4;

  const ZuptDecision d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kPinToAnchor);
  EXPECT_TRUE(d.hold);
}

// The pin does not depend on the speed: with independent proof of "parked" a
// blind stretch has nothing to integrate, whatever the velocity claims.
TEST(ZuptPolicy, BlindAndParkedPinsEvenAtZeroSpeed)
{
  ZuptPolicyInputs in = deployed();
  in.scan_degraded = true;
  in.body_static = true;
  in.speed = 0.0;

  EXPECT_EQ(decideZupt(in).action, ZuptAction::kPinToAnchor);
}

// No anchor yet (no healthy scan since startup) — there is nothing to pin to, so
// the hold must not engage.
TEST(ZuptPolicy, BlindAndParkedWithoutAnAnchorDoesNotHold)
{
  ZuptPolicyInputs in = deployed();
  in.anchor_valid = false;
  in.scan_degraded = true;
  in.body_static = true;
  in.speed = 1.4;

  const ZuptDecision d = decideZupt(in);
  EXPECT_FALSE(d.hold);
  EXPECT_EQ(d.action, ZuptAction::kNone);  // 1.4 m/s is under the guard ceiling
}

// Moving while blind: the hold must not survive into this, and a speed above the
// platform's physical maximum is zeroed rather than clamped — clamping made the
// ceiling an equilibrium (m20_0814_degrade: spd=3.00 every scan for 60 s).
TEST(ZuptPolicy, BlindAndMovingZeroesOnlyImpossibleSpeeds)
{
  ZuptPolicyInputs in = deployed();
  in.scan_degraded = true;
  in.body_static = false;

  in.speed = 1.4;
  ZuptDecision d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kNone);
  EXPECT_FALSE(d.hold);

  in.speed = in.max_speed + 1.0;
  d = decideZupt(in);
  EXPECT_EQ(d.action, ZuptAction::kZeroVelocity);
  EXPECT_FALSE(d.hold);
}

TEST(ZuptPolicy, DisabledZuptStillGuardsImpossibleSpeedsWhileBlind)
{
  ZuptPolicyInputs in = deployed();
  in.zupt_enabled = false;
  in.scan_degraded = true;
  in.body_static = true;  // ignored: the hold needs zupt_en
  in.speed = 45.0;

  const ZuptDecision d = decideZupt(in);
  EXPECT_FALSE(d.hold);
  EXPECT_EQ(d.action, ZuptAction::kZeroVelocity);
}

// ── Degenerate inputs ────────────────────────────────────────────────────────
// A non-finite speed compares false everywhere, so it must not trigger a
// correction on a number nobody trusts — but it must not block the pin either.
TEST(ZuptPolicy, NonFiniteSpeedDoesNotDriveTheHealthyBranch)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = true;
  for (const double speed : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity()}) {
    in.speed = speed;
    in.scan_degraded = false;
    const ZuptDecision healthy = decideZupt(in);
    if (std::isnan(speed)) {
      EXPECT_EQ(healthy.action, ZuptAction::kNone);
    } else {
      EXPECT_EQ(healthy.action, ZuptAction::kZeroVelocity);
    }

    in.scan_degraded = true;
    EXPECT_EQ(decideZupt(in).action, ZuptAction::kPinToAnchor);
  }
}

// The guard and the watchdog have to agree about "parked", so the policy must
// read the verdict it is handed and nothing else — same inputs, same answer.
TEST(ZuptPolicy, IsAPureFunctionOfItsInputs)
{
  ZuptPolicyInputs in = deployed();
  in.body_static = true;
  in.speed = 0.7;

  const ZuptDecision first = decideZupt(in);
  const ZuptDecision second = decideZupt(in);
  EXPECT_EQ(first.action, second.action);
  EXPECT_EQ(first.hold, second.hold);
  EXPECT_EQ(first.report, second.report);
}

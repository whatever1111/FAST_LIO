#include "runaway_watchdog.hpp"

#include <gtest/gtest.h>

#include <cmath>

using fast_lio::RunawayParams;
using fast_lio::RunawayReason;
using fast_lio::RunawayState;
using fast_lio::updateRunawayWatchdog;

namespace
{
// Dog 2's silent runaway: parked, but the estimate slid at ~0.5 m/s for 45 min
// with a healthy-looking map.
constexpr double kDriftSpeed = 0.5;
constexpr double kNoWheel = -1.0;

// Drive `scans` scans of 0.1 s at the given estimated speed.
int runScans(RunawayState & s,
             const RunawayParams & p,
             int scans,
             double speed,
             bool parked,
             double t0 = 0.0,
             double wheel_age = 0.05,
             double wheel_speed = 0.0)
{
  int trips = 0;
  for (int i = 0; i < scans; ++i) {
    if (updateRunawayWatchdog(&s, t0 + i * 0.1, speed, parked, wheel_age, wheel_speed, p).tripped) {
      ++trips;
    }
  }
  return trips;
}
}  // namespace

TEST(RunawayWatchdog, HealthyParkedEstimateNeverTrips)
{
  RunawayState s;
  RunawayParams p;
  EXPECT_EQ(runScans(s, p, 600, 0.002, true), 0);  // the measured parked speed
  EXPECT_EQ(s.trips, 0u);
}

TEST(RunawayWatchdog, ParkedButMovingTripsOnceTheContradictionPersists)
{
  RunawayState s;
  RunawayParams p;  // hold_sec = 3.0
  EXPECT_EQ(runScans(s, p, 29, kDriftSpeed, true), 0);  // 2.9 s: not yet
  const auto v = updateRunawayWatchdog(&s, 3.0, kDriftSpeed, true, 0.05, 0.0, p);
  EXPECT_TRUE(v.tripped);
  EXPECT_EQ(v.reason, RunawayReason::kParked);
  EXPECT_GE(v.bad_for, p.hold_sec);
  EXPECT_EQ(s.trips, 1u);
}

TEST(RunawayWatchdog, OneHonestScanClearsTheWindow)
{
  // A single slip or a momentary wheel dropout must not accumulate towards a trip.
  RunawayState s;
  RunawayParams p;
  for (int i = 0; i < 100; ++i) {
    const bool contradicted = (i % 10) != 0;  // 9 bad scans, then one clean one
    const auto v = updateRunawayWatchdog(&s, i * 0.1, contradicted ? kDriftSpeed : 0.0, true, 0.05, 0.0, p);
    EXPECT_FALSE(v.tripped) << i;
  }
  EXPECT_EQ(s.trips, 0u);
}

TEST(RunawayWatchdog, EstimateOutrunningTheWheelsTripsWhileDriving)
{
  // Not parked, so the static verdict says nothing: only the wheels can object.
  RunawayState s;
  RunawayParams p;  // margin = 1.0 m/s
  EXPECT_EQ(runScans(s, p, 100, 1.4, false, 0.0, 0.05, 1.0), 0);  // within the margin
  RunawayState s2;
  // 3.5 s of being 2 m/s over the wheels: one episode, one trip.
  EXPECT_EQ(runScans(s2, p, 35, 3.0, false, 0.0, 0.05, 1.0), 1);
  EXPECT_EQ(s2.reason, RunawayReason::kWheel);
}

TEST(RunawayWatchdog, WithoutEvidenceNothingIsContradicted)
{
  RunawayState s;
  RunawayParams p;
  // Moving fast, no static verdict, no wheel sample: the watchdog has no case.
  EXPECT_EQ(runScans(s, p, 200, 5.0, false, 0.0, kNoWheel, 0.0), 0);
  // A stale wheel sample is not evidence either.
  RunawayState s2;
  EXPECT_EQ(runScans(s2, p, 200, 5.0, false, 0.0, 3.0 /*s old*/, 0.0), 0);
  EXPECT_EQ(s2.trips, 0u);
}

TEST(RunawayWatchdog, TripsOncePerEpisodeThenReArms)
{
  RunawayState s;
  RunawayParams p;
  // 10 s of continuous contradiction: the caller acts once, not on every scan.
  const int trips = runScans(s, p, 100, kDriftSpeed, true);
  EXPECT_EQ(trips, 3);  // 3 s, 6 s, 9 s
  EXPECT_EQ(s.trips, 3u);
}

TEST(RunawayWatchdog, BackwardsClockRestartsTheWindow)
{
  RunawayState s;
  RunawayParams p;
  updateRunawayWatchdog(&s, 100.0, kDriftSpeed, true, 0.05, 0.0, p);
  const auto v = updateRunawayWatchdog(&s, 0.5, kDriftSpeed, true, 0.05, 0.0, p);
  EXPECT_FALSE(v.tripped);
  EXPECT_DOUBLE_EQ(s.bad_since, 0.5);
}

TEST(RunawayWatchdog, NonFiniteInputsAndDisabledAreInert)
{
  RunawayState s;
  RunawayParams p;
  EXPECT_FALSE(updateRunawayWatchdog(&s, std::nan(""), kDriftSpeed, true, 0.05, 0.0, p).tripped);
  EXPECT_FALSE(updateRunawayWatchdog(&s, 1.0, std::nan(""), true, 0.05, 0.0, p).tripped);
  EXPECT_FALSE(updateRunawayWatchdog(nullptr, 1.0, kDriftSpeed, true, 0.05, 0.0, p).tripped);
  RunawayParams off;
  off.enabled = false;
  RunawayState s2;
  EXPECT_EQ(runScans(s2, off, 200, kDriftSpeed, true), 0);
}

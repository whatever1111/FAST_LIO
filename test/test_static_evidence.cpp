#include "static_evidence.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using fast_lio::bodyIsStatic;
using fast_lio::ImuWindowStats;
using fast_lio::StaticEvidenceParams;
using fast_lio::wheelSampleIsEvidence;

namespace
{
// A parked M20 as measured on dog 2: |omega| 0.003 rad/s, |a| 9.834 +- 0.013.
ImuWindowStats parkedImu()
{
  ImuWindowStats s;
  s.samples = 20;
  s.gyr_max = 0.003;
  s.acc_min = 9.821;
  s.acc_max = 9.847;
  s.acc_mean = 9.834;
  return s;
}
}  // namespace

TEST(StaticEvidence, ParkedImuAndWheelAgree)
{
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 0.05, 0.0, StaticEvidenceParams{}));
}

TEST(StaticEvidence, RotationVetoesTheHold)
{
  ImuWindowStats s = parkedImu();
  s.gyr_max = 0.2;  // 11 deg/s — the robot is turning
  EXPECT_FALSE(bodyIsStatic(s, 0.05, 0.0, StaticEvidenceParams{}));
}

TEST(StaticEvidence, AccelSpanVetoesTheHold)
{
  ImuWindowStats s = parkedImu();
  s.acc_max = 10.6;  // 8% spread: walking, not standing
  EXPECT_FALSE(bodyIsStatic(s, 0.05, 0.0, StaticEvidenceParams{}));
}

TEST(StaticEvidence, AccelSpanTestIsScaleFree)
{
  // Same platform, a driver publishing g instead of m/s^2.
  ImuWindowStats s = parkedImu();
  s.acc_min /= 9.81;
  s.acc_max /= 9.81;
  s.acc_mean /= 9.81;
  EXPECT_TRUE(bodyIsStatic(s, 0.05, 0.0, StaticEvidenceParams{}));
}

TEST(StaticEvidence, MovingWheelVetoesTheHoldEvenWithAQuietImu)
{
  // A dog being pushed on a trolley: the IMU can look quiet, the wheels do not.
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.05, 0.4, StaticEvidenceParams{}));
}

TEST(StaticEvidence, StaleOrAbsentWheelFallsBackToTheImuPolicy)
{
  StaticEvidenceParams permissive;  // require_wheel = false
  EXPECT_TRUE(bodyIsStatic(parkedImu(), -1.0, 0.0, permissive));   // never received one
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 5.0, 0.0, permissive));    // 5 s old
  StaticEvidenceParams strict;
  strict.require_wheel = true;
  EXPECT_FALSE(bodyIsStatic(parkedImu(), -1.0, 0.0, strict));
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 5.0, 0.0, strict));
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 0.1, 0.0, strict));
}

TEST(StaticEvidence, EmptyOrNonFiniteWindowIsNeverStatic)
{
  ImuWindowStats empty;
  EXPECT_FALSE(bodyIsStatic(empty, 0.05, 0.0, StaticEvidenceParams{}));
  ImuWindowStats nan_stats = parkedImu();
  nan_stats.gyr_max = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(bodyIsStatic(nan_stats, 0.05, 0.0, StaticEvidenceParams{}));
  ImuWindowStats zero_mean = parkedImu();
  zero_mean.acc_mean = 0.0;  // free fall / dead accelerometer
  EXPECT_FALSE(bodyIsStatic(zero_mean, 0.05, 0.0, StaticEvidenceParams{}));
}

TEST(StaticEvidence, NonFiniteWheelSampleIsTreatedAsNoEvidence)
{
  // NaN is the wheelspeed bridge saying "cannot measure": while its legs work, and
  // permanently if its wheel signs never lock or a sign mismatch latches. It is the
  // absence of a measurement, so it must land on the same policy as an absent or a
  // stale sample — not on "not parked", which would silently disable the hold, the
  // parked-velocity zeroing and the runaway watchdog for the whole run.
  const double nan_speed = std::numeric_limits<double>::quiet_NaN();
  StaticEvidenceParams permissive;  // require_wheel = false: the IMU alone may decide
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 0.05, nan_speed, permissive));
  EXPECT_EQ(bodyIsStatic(parkedImu(), 0.05, nan_speed, permissive),
            bodyIsStatic(parkedImu(), -1.0, 0.0, permissive));

  StaticEvidenceParams strict;
  strict.require_wheel = true;  // no wheel evidence means "not proven"
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.05, nan_speed, strict));

  // A moving IMU still vetoes it: dropping the wheel term never invents a hold.
  ImuWindowStats turning = parkedImu();
  turning.gyr_max = 0.2;
  EXPECT_FALSE(bodyIsStatic(turning, 0.05, nan_speed, permissive));

  // A non-finite AGE is the same story: we cannot tell how old the sample is, so
  // there is no usable evidence and require_wheel decides.
  EXPECT_TRUE(bodyIsStatic(parkedImu(), nan_speed, 0.0, permissive));
  EXPECT_FALSE(bodyIsStatic(parkedImu(), nan_speed, 0.0, strict));
}

TEST(StaticEvidence, WheelSampleIsEvidenceRejectsAnyNonFiniteComponent)
{
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  const double inf_v = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(wheelSampleIsEvidence(0.0, 0.0, 0.0));      // standstill IS a measurement
  EXPECT_TRUE(wheelSampleIsEvidence(-1.25, 0.0, 0.0));
  EXPECT_FALSE(wheelSampleIsEvidence(nan_v, 0.0, 0.0));
  EXPECT_FALSE(wheelSampleIsEvidence(0.0, nan_v, 0.0));
  EXPECT_FALSE(wheelSampleIsEvidence(0.0, 0.0, nan_v));
  EXPECT_FALSE(wheelSampleIsEvidence(inf_v, 0.0, 0.0));
}

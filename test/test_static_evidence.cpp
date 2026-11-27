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

TEST(StaticEvidence, AParkedImuWithoutAFreshWheelSampleIsNotStatic)
{
  // A quiet IMU alone cannot tell parked from a slow, smooth motion: without a wheel sample that testifies,
  // nothing is static, however still the IMU reads.
  const StaticEvidenceParams params;
  EXPECT_FALSE(bodyIsStatic(parkedImu(), -1.0, 0.0, params));  // never received one
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 5.0, 0.0, params));   // 5 s old
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.51, 0.0, params));  // just past wheel_max_age
  EXPECT_FALSE(bodyIsStatic(parkedImu(), -0.2, 0.0, params));  // stamped after the scan: time runs backwards
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 0.5, 0.0, params));    // exactly wheel_max_age old still testifies
  EXPECT_TRUE(bodyIsStatic(parkedImu(), 0.1, 0.0, params));
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
  // absence of a measurement: not static, exactly like an absent or a stale sample.
  const double nan_speed = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const StaticEvidenceParams params;
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.05, nan_speed, params));
  EXPECT_EQ(bodyIsStatic(parkedImu(), 0.05, nan_speed, params), bodyIsStatic(parkedImu(), -1.0, 0.0, params));
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.05, inf, params));

  // A non-finite AGE is the same story: nobody can tell how old the sample is.
  EXPECT_FALSE(bodyIsStatic(parkedImu(), nan_speed, 0.0, params));
  EXPECT_FALSE(bodyIsStatic(parkedImu(), inf, 0.0, params));
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

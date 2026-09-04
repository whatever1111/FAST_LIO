#include "static_evidence.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using fast_lio::bodyIsStatic;
using fast_lio::ImuWindowStats;
using fast_lio::StaticEvidenceParams;

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
  EXPECT_FALSE(bodyIsStatic(parkedImu(), 0.05, std::numeric_limits<double>::quiet_NaN(), StaticEvidenceParams{}));
}

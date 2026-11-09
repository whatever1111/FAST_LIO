// test_wheel_measurement.cpp — the bridge-sample ring, the per-scan speed fit, the slip
// correction and the measurement model of the wheel update (wheel_measurement.hpp), and the
// front end's holder of the wheel samples with its per-scan step (wheel_speed_input.hpp).

#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gravity_align_kinematics.hpp"
#include "wheel_measurement.hpp"
#include "wheel_speed_input.hpp"

using fast_lio::composeR;
using fast_lio::slipCorrect;
using fast_lio::slipPhi;
using fast_lio::WheelFit;
using fast_lio::wheelForward;
using fast_lio::WheelMeasurement;
using fast_lio::wheelMeasurement;
using fast_lio::WheelModelParams;
using fast_lio::WheelNoise;
using fast_lio::WheelSampleRing;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kUnobserved = 1e6;  // the bridge's cov[7] for an axis it does not observe
// A 2025 header stamp: large enough that double seconds resolve only ~240 ns.
constexpr std::int64_t kEpochNs = 1762590000LL * 1000000000LL;

double seconds(std::int64_t stamp_ns)
{
  return static_cast<double>(stamp_ns) / 1e9;  // rclcpp::Time::seconds()
}

std::uint64_t bits(double value)
{
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

Eigen::Matrix3d rotation(double angle, const Eigen::Vector3d & axis)
{
  return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d & w)
{
  const double angle = w.norm();
  return angle > 0.0 ? rotation(angle, w) : Eigen::Matrix3d(Eigen::Matrix3d::Identity());
}

struct ModelState
{
  Eigen::Matrix3d R_world_imu = rotation(0.7, Eigen::Vector3d::UnitZ()) * rotation(0.1, Eigen::Vector3d::UnitY());
  Eigen::Matrix3d R_robot_to_imu = rotation(0.05, Eigen::Vector3d::UnitX()) * rotation(-0.02, Eigen::Vector3d::UnitY());
  Eigen::Vector3d velocity_world = Eigen::Vector3d(0.6, -0.4, 0.05);
  Eigen::Vector3d gyro_bias = Eigen::Vector3d(0.01, -0.02, 0.005);
  Eigen::Vector3d omega_mean = Eigen::Vector3d(0.11, -0.22, 0.805);  // raw gyro mean, bias included
};

WheelFit singleFit(double v, double var_vx, double var_lat)
{
  WheelFit fit;
  fit.v = v;
  fit.samples = 1;
  fit.sample_max_vx = fit.endpoint_bound_vx = var_vx;
  fit.sample_max_lat = fit.endpoint_bound_lat = var_lat;
  fit.lateral_observed = var_lat < kUnobserved;
  fit.c_l1 = fit.c_l2_sq = 1.0;
  return fit;
}

WheelNoise sampleNoise(double var_vx, double var_lat, double omega_norm, const WheelModelParams & params)
{
  return composeR(singleFit(0.0, var_vx, var_lat), omega_norm, params);
}

std::optional<WheelMeasurement> modelAt(const ModelState & s, const WheelModelParams & params, double v = 0.9)
{
  const WheelFit fit = singleFit(v, 0.002, kUnobserved);
  return wheelMeasurement(
    fit, std::nullopt, s.R_world_imu, s.velocity_world, s.gyro_bias, s.omega_mean, s.R_robot_to_imu, params);
}
}  // namespace

TEST(WheelSampleRing, FitIsExactOnALinearRampAtTheScanEnd)
{
  // 50 Hz samples on v(t) = 0.8 + 0.25 (t - t_end) with uneven variances; the scan ends 5 ms after the last
  // sample. Speeds are written in the ring's own time coordinates, so the samples lie exactly on the line.
  WheelSampleRing ring(128);
  const std::int64_t end_ns = kEpochNs + 60 * 20000000LL + 5000000LL;
  const double t_end = seconds(end_ns);
  for (int k = 0; k <= 60; ++k) {
    const std::int64_t stamp = kEpochNs + k * 20000000LL;
    ASSERT_TRUE(ring.pushSample(stamp, 0.8 + 0.25 * (seconds(stamp) - t_end), true));
    ASSERT_TRUE(ring.attachSigma(stamp, 0.002 + 0.001 * (k % 3), kUnobserved));
  }
  const auto fit = ring.fitAt(t_end, 1.0);
  ASSERT_TRUE(fit.has_value());
  EXPECT_NEAR(fit->v, 0.8, 1e-12);  // the line's value at t_end, not the newest sample's 0.79875
  EXPECT_EQ(fit->samples, 50);      // k = 11 ... 60 lie in [t_end - 1 s, t_end]
  EXPECT_LT(fit->residual_rms, 1e-12);
  EXPECT_EQ(fit->sample_max_vx, 0.002 + 0.001 * 2);
  EXPECT_EQ(fit->sample_max_lat, kUnobserved);
  // numerical-design.txt:L39, the same double-epoch ramp fixture.
  EXPECT_NEAR(fit->endpoint_bound_vx, 0.007297929312071825, 1e-10);
  EXPECT_NEAR(composeR(*fit, 0.0, WheelModelParams{}).var_vx, 0.008922929312071825, 1e-10);

  // numerical-design.txt:L35: an early single time cannot identify the endpoint.
  WheelSampleRing single(4);
  ASSERT_TRUE(single.pushSample(kEpochNs, 0.42, true));
  ASSERT_TRUE(single.attachSigma(kEpochNs, 0.003, 0.004));
  fast_lio::WheelFitStatus status;
  EXPECT_FALSE(single.fitAt(seconds(kEpochNs) + 0.05, 1.0, &status));
  EXPECT_EQ(status, fast_lio::WheelFitStatus::kUnidentifiableEndpoint);
  const auto one = single.fitAt(seconds(kEpochNs), 1.0);
  ASSERT_TRUE(one.has_value());
  EXPECT_EQ(one->v, 0.42);
  EXPECT_EQ(one->samples, 1);
  EXPECT_EQ(one->residual_rms, 0.0);
}

TEST(WheelSampleRing, OnlyTheWindowCountsAndAnEmptyWindowHasNoSpeed)
{
  // Stamps of a few seconds, where double seconds are exact to the nanosecond.
  WheelSampleRing ring(16);
  const auto pair = [&](std::int64_t stamp, double v) {
    ASSERT_TRUE(ring.pushSample(stamp, v, true));
    ASSERT_TRUE(ring.attachSigma(stamp, 0.002, kUnobserved));
  };
  pair(1000000000LL, 5.0);  // 1.000 s: stale
  pair(1499000000LL, 5.0);  // 1.499 s: 1 ms before the window
  pair(1500000000LL, 1.0);  // 1.500 s: the window's first instant, included
  pair(1750000000LL, 1.0);
  pair(2000000000LL, 1.0);  // t_end itself, included
  pair(2001000000LL, 9.0);  // after the scan end
  const auto fit = ring.fitAt(2.0, 0.5);
  ASSERT_TRUE(fit.has_value());
  EXPECT_EQ(fit->samples, 3);
  EXPECT_NEAR(fit->v, 1.0, 1e-15);
  EXPECT_FALSE(ring.fitAt(0.9, 0.5).has_value());  // nothing that old
  EXPECT_FALSE(ring.fitAt(1.2, 0.1).has_value());  // a gap between samples
  EXPECT_FALSE(ring.fitAt(kNaN, 0.5).has_value());
  EXPECT_FALSE(ring.fitAt(2.0, kInf).has_value());
  EXPECT_FALSE(ring.fitAt(2.0, -0.5).has_value());
  EXPECT_FALSE(WheelSampleRing(4).fitAt(2.0, 0.5).has_value());  // no samples at all
}

TEST(WheelSampleRing, PairsByExactStampInEitherArrivalOrder)
{
  // The gated speed first, or its variances first: the same entry.
  WheelSampleRing speed_first(8);
  WheelSampleRing sigma_first(8);
  ASSERT_TRUE(speed_first.pushSample(kEpochNs, 0.7, true));
  ASSERT_TRUE(speed_first.attachSigma(kEpochNs, 0.003, 0.004));
  ASSERT_TRUE(sigma_first.attachSigma(kEpochNs, 0.003, 0.004));
  ASSERT_TRUE(sigma_first.pushSample(kEpochNs, 0.7, true));
  ASSERT_EQ(speed_first.size(), 1u);
  ASSERT_EQ(sigma_first.size(), 1u);
  const auto & a = speed_first.entries().front();
  const auto & b = sigma_first.entries().front();
  EXPECT_EQ(a.stamp_ns, b.stamp_ns);
  EXPECT_TRUE(a.has_sample && a.valid && a.has_sigma);
  EXPECT_TRUE(b.has_sample && b.valid && b.has_sigma);
  EXPECT_EQ(a.v, b.v);
  EXPECT_EQ(a.var_vx, b.var_vx);
  EXPECT_EQ(a.var_lat, b.var_lat);

  // The key is the integer stamp: 1 ns apart is another sample, although double seconds cannot tell them
  // apart at this epoch. Out-of-order halves still find their partners.
  WheelSampleRing ring(8);
  ASSERT_TRUE(ring.pushSample(kEpochNs + 2, 3.0, true));
  ASSERT_TRUE(ring.attachSigma(kEpochNs + 1, 0.002, kUnobserved));
  ASSERT_TRUE(ring.pushSample(kEpochNs, 1.0, true));
  ASSERT_TRUE(ring.attachSigma(kEpochNs + 2, 0.002, kUnobserved));
  ASSERT_TRUE(ring.pushSample(kEpochNs + 1, 2.0, true));
  ASSERT_TRUE(ring.attachSigma(kEpochNs, 0.002, kUnobserved));
  ASSERT_EQ(ring.size(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(ring.entries()[i].stamp_ns, kEpochNs + static_cast<std::int64_t>(i));  // kept in stamp order
  }
  fast_lio::WheelFitStatus status;
  EXPECT_FALSE(ring.fitAt(seconds(kEpochNs) + 0.01, 0.1, &status));
  EXPECT_EQ(status, fast_lio::WheelFitStatus::kUnidentifiableEndpoint);
  const auto fit = ring.fitAt(seconds(kEpochNs), 0.1);
  ASSERT_TRUE(fit.has_value());
  EXPECT_EQ(fit->samples, 3);
  EXPECT_DOUBLE_EQ(fit->v, 2.0);  // equal times: no slope, the weighted mean
}

TEST(WheelSampleRing, SamplesAtOneDoubleTimeIdentifyOnlyThatEndpoint)
{
  // Stamps 1 ns apart are one double time at this epoch: zero time spread, so no slope. Centred on the
  // weighted mean time instead, the mean's rounding leaves a spread of an ulp, and the speeds' rounding
  // divided by it is an arbitrary slope: these three samples then gave 0.33 m/s, the lone 0.1 one an ulp off.
  WheelSampleRing ring(8);
  const double speeds[] = {0.5, 0.7, 0.9};
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(ring.pushSample(kEpochNs + i, speeds[i], true));
    ASSERT_TRUE(ring.attachSigma(kEpochNs + i, 0.003, kUnobserved));
  }
  EXPECT_FALSE(ring.fitAt(seconds(kEpochNs) + 0.015, 0.1));
  const auto fit = ring.fitAt(seconds(kEpochNs), 0.1);
  ASSERT_TRUE(fit.has_value());
  EXPECT_EQ(fit->samples, 3);
  EXPECT_NEAR(fit->v, 0.7, 1e-15);

  WheelSampleRing single(4);
  ASSERT_TRUE(single.pushSample(kEpochNs, 0.1, true));
  ASSERT_TRUE(single.attachSigma(kEpochNs, 0.003, kUnobserved));
  for (int ms = 1; ms <= 20; ++ms) {
    EXPECT_FALSE(single.fitAt(seconds(kEpochNs) + ms / 1000.0, 0.1)) << ms;
  }
  const auto one = single.fitAt(seconds(kEpochNs), 0.1);
  ASSERT_TRUE(one);
  EXPECT_EQ(bits(one->v), bits(0.1));
}

TEST(WheelSampleRing, UnpairedSamplesStayOutOfTheUpdateAndAreCountedWhenDropped)
{
  WheelSampleRing ring(5);
  ASSERT_TRUE(ring.pushSample(1000000000LL, 7.0, true));            // never gets its variances
  ASSERT_TRUE(ring.attachSigma(1010000000LL, 0.002, kUnobserved));  // never gets its speed
  ASSERT_TRUE(ring.pushSample(1020000000LL, kNaN, true));           // gated invalid: NaN
  ASSERT_TRUE(ring.attachSigma(1020000000LL, 0.002, kUnobserved));
  ASSERT_TRUE(ring.pushSample(1030000000LL, 1.0, false));  // gated invalid by the caller
  ASSERT_TRUE(ring.attachSigma(1030000000LL, 0.002, kUnobserved));
  EXPECT_FALSE(ring.fitAt(1.05, 0.1).has_value());  // nothing usable yet
  EXPECT_EQ(ring.unpairedIn(1.05, 0.1), 1);         // only the valid speed waiting for variances

  ASSERT_TRUE(ring.pushSample(1040000000LL, 1.0, true));
  ASSERT_TRUE(ring.attachSigma(1040000000LL, 0.002, kUnobserved));
  ASSERT_EQ(ring.size(), 5u);  // the unpaired 7.0 is still held, inside the window
  EXPECT_FALSE(ring.fitAt(1.05, 0.1));
  const auto fit = ring.fitAt(1.04, 0.1);
  ASSERT_TRUE(fit.has_value());
  EXPECT_EQ(fit->samples, 1);
  EXPECT_EQ(fit->v, 1.0);  // ... and not in the fit
  EXPECT_EQ(ring.unmatchedSamples(), 0u);
  EXPECT_EQ(ring.unmatchedSigmas(), 0u);

  // Capacity 5: two more samples push out the speed and the variances that never met.
  ASSERT_TRUE(ring.pushSample(1050000000LL, 1.0, true));
  ASSERT_TRUE(ring.pushSample(1060000000LL, 1.0, true));
  EXPECT_EQ(ring.size(), 5u);
  EXPECT_EQ(ring.unmatchedSamples(), 1u);
  EXPECT_EQ(ring.unmatchedSigmas(), 1u);
  EXPECT_EQ(ring.rejected(), 0u);
}

TEST(WheelSampleRing, RefusesDuplicatesAndUnusableVariances)
{
  WheelSampleRing ring(8);
  ASSERT_TRUE(ring.pushSample(kEpochNs, 1.0, true));
  EXPECT_FALSE(ring.pushSample(kEpochNs, 2.0, true));
  ASSERT_TRUE(ring.attachSigma(kEpochNs, 0.002, kUnobserved));
  EXPECT_FALSE(ring.attachSigma(kEpochNs, 0.004, kUnobserved));
  EXPECT_EQ(ring.entries().front().v, 1.0);  // the first half stands
  EXPECT_EQ(ring.entries().front().var_vx, 0.002);
  for (const auto & bad : std::vector<std::pair<double, double>>{
         {0.0, 0.01}, {-0.002, 0.01}, {kNaN, 0.01}, {kInf, 0.01}, {0.002, 0.0}, {0.002, kNaN}, {0.002, kInf}}) {
    EXPECT_FALSE(ring.attachSigma(kEpochNs + 1, bad.first, bad.second)) << bad.first << " " << bad.second;
  }
  EXPECT_EQ(ring.rejected(), 9u);
  EXPECT_EQ(ring.size(), 1u);  // a refused half leaves no entry behind
  EXPECT_THROW(WheelSampleRing(0), std::invalid_argument);
}

TEST(WheelFit, TwoEarlySamplesAmplifyVariance)
{
  // numerical-design.txt:L26-L29. c=(-3,4), U=.0025*(3+4)^2.
  WheelSampleRing ring(8);
  ASSERT_TRUE(ring.pushSample(20000000, 0.775, true));
  ASSERT_TRUE(ring.attachSigma(20000000, 0.0025, kUnobserved));
  ASSERT_TRUE(ring.pushSample(40000000, 0.825, true));
  ASSERT_TRUE(ring.attachSigma(40000000, 0.0025, kUnobserved));
  const auto fit = ring.fitAt(0.1, 0.1);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->v, 0.975, 1e-12);
  EXPECT_LT(fit->residual_rms, 1e-12);
  EXPECT_NEAR(fit->c_l1, 7.0, 1e-12);
  EXPECT_NEAR(fit->c_l2_sq, 25.0, 1e-12);
  EXPECT_EQ(fit->fit_order, 1);
  EXPECT_NEAR(fit->endpoint_bound_vx, 0.1225, 1e-12);
  EXPECT_NEAR(composeR(*fit, 0.0, WheelModelParams{}).var_vx, 0.125625, 1e-12);
}

TEST(WheelFit, CommonAndWhitePartsHaveDifferentGains)
{
  // numerical-design.txt:L26-L29. Explicit covariance oracles, with no runtime rho input.
  const Eigen::Vector2d c(-3.0, 4.0);
  const double rho[] = {0.0, 0.5, 0.58, 1.0};
  const double expected[] = {0.0625, 0.0325, 0.0277, 0.0025};
  for (int i = 0; i < 4; ++i) {
    const Eigen::Matrix2d covariance =
      0.0025 * ((1.0 - rho[i]) * Eigen::Matrix2d::Identity() + rho[i] * Eigen::Matrix2d::Ones());
    EXPECT_NEAR(c.dot(covariance * c), expected[i], 1e-12);
    EXPECT_NEAR(c.dot(covariance * c), 0.0025 * ((1.0 - rho[i]) * 25.0 + rho[i]), 1e-12);
  }
}

TEST(WheelFit, UnknownCorrelationBound)
{
  // numerical-design.txt:L38,L40. All PSD correlations lie below U, and the signed rank-one case attains it.
  WheelSampleRing ring(8);
  ring.pushSample(20000000, 0.7, true);
  ring.attachSigma(20000000, 0.003, kUnobserved);
  ring.pushSample(40000000, 0.74, true);
  ring.attachSigma(40000000, 0.004, kUnobserved);
  const auto fit = ring.fitAt(0.1, 0.1);
  ASSERT_TRUE(fit);
  const Eigen::Vector2d c(-3.0, 4.0);
  const Eigen::Vector2d sigma(std::sqrt(0.003), std::sqrt(0.004));
  for (double rho : {-1.0, -0.58, 0.0, 0.5, 1.0}) {
    Eigen::Matrix2d covariance;
    covariance << 0.003, rho * sigma.prod(), rho * sigma.prod(), 0.004;
    EXPECT_LE(c.dot(covariance * c), fit->endpoint_bound_vx + 1e-12);
  }
  const Eigen::Vector2d signed_sigma(-sigma(0), sigma(1));
  EXPECT_NEAR(c.dot(signed_sigma * signed_sigma.transpose() * c), fit->endpoint_bound_vx, 1e-12);
  EXPECT_NEAR(composeR(*fit, 0.0, WheelModelParams{}).var_vx, 0.17576343876330622, 1e-12);
}

TEST(WheelFit, LateralCovarianceAndMarkerAreSeparate)
{
  // numerical-design.txt:L34 and D3.2's input-marker contract.
  const auto fitWith = [](double first_lat, double second_lat) {
    WheelSampleRing ring(8);
    ring.pushSample(20000000, 0.8, true);
    ring.attachSigma(20000000, 0.0025, first_lat);
    ring.pushSample(40000000, 0.8, true);
    ring.attachSigma(40000000, 0.0025, second_lat);
    return ring.fitAt(0.1, 0.1);
  };
  const auto observed = fitWith(0.046 * 0.046, 0.046 * 0.046);
  ASSERT_TRUE(observed);
  EXPECT_TRUE(observed->lateral_observed);
  EXPECT_NEAR(composeR(*observed, 0.0, WheelModelParams{}).var_vy, 0.103684, 1e-12);
  const auto large = fitWith(999999.0, 999999.0);
  ASSERT_TRUE(large);
  EXPECT_TRUE(large->lateral_observed);
  EXPECT_GT(composeR(*large, 0.0, WheelModelParams{}).var_vy, kUnobserved);
  for (const auto & pair : {std::pair<double, double>{kUnobserved, 0.002}, {0.002, kUnobserved}}) {
    const auto marker = fitWith(pair.first, pair.second);
    ASSERT_TRUE(marker);
    EXPECT_FALSE(marker->lateral_observed);
    EXPECT_FALSE(composeR(*marker, 0.0, WheelModelParams{}).lateral);
  }
}

TEST(WheelFit, RankOneEndpointIsNotExtrapolation)
{
  // numerical-design.txt:L35-L36. One double time identifies a constant only at that time.
  for (int count : {1, 3}) {
    WheelSampleRing ring(8);
    for (int i = 0; i < count; ++i) {
      ring.pushSample(kEpochNs + i, 0.7, true);
      ring.attachSigma(kEpochNs + i, 0.003, kUnobserved);
    }
    fast_lio::WheelFitStatus status;
    EXPECT_FALSE(ring.fitAt(seconds(kEpochNs) + 0.06, 0.1, &status));
    EXPECT_EQ(status, fast_lio::WheelFitStatus::kUnidentifiableEndpoint);
    const auto fit = ring.fitAt(seconds(kEpochNs), 0.1, &status);
    ASSERT_TRUE(fit);
    EXPECT_EQ(status, fast_lio::WheelFitStatus::kValid);
    EXPECT_EQ(fit->fit_order, 0);
    EXPECT_NEAR(fit->c_l1, 1.0, 1e-12);
    EXPECT_NEAR(fit->c_l2_sq, 1.0 / count, 1e-12);
    EXPECT_DOUBLE_EQ(composeR(*fit, 0.0, WheelModelParams{}).var_vx, 0.075 * 0.075);
  }
}

TEST(WheelFit, FullWindowKeepsTheApprovedFloorCombination)
{
  // numerical-design.txt:L37. The ordinary five-sample endpoint still has negative coefficients.
  WheelSampleRing ring(8);
  for (int i = 1; i <= 5; ++i) {
    ring.pushSample(i * 20000000, 0.8, true);
    ring.attachSigma(i * 20000000, 0.0025, 0.046 * 0.046);
  }
  const auto fit = ring.fitAt(0.1, 0.1);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->c_l1, 1.4, 1e-12);
  EXPECT_NEAR(composeR(*fit, 0.0, WheelModelParams{}).var_vx, 0.008025, 1e-12);
}

TEST(WheelFit, NormalizedWeightsAvoidReciprocalOverflow)
{
  // D3.3: scaling every variance must leave the line and endpoint coefficients unchanged.
  WheelSampleRing ring(8);
  ring.pushSample(20000000, 0.775, true);
  ring.attachSigma(20000000, 1e-320, 1e-320);
  ring.pushSample(40000000, 0.825, true);
  ring.attachSigma(40000000, 1e-320, 1e-320);
  const auto fit = ring.fitAt(0.1, 0.1);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->v, 0.975, 1e-12);
  EXPECT_NEAR(fit->c_l1, 7.0, 1e-12);
  EXPECT_TRUE(std::isfinite(fit->endpoint_bound_vx));
}

TEST(SlipCorrection, ZeroCoefficientReturnsTheSpeedBitForBit)
{
  for (double v : {0.7351, -0.0, 5e-324, 1e300, std::nextafter(1.0, 2.0), -2.5}) {
    for (const std::optional<double> phi : {std::optional<double>(), std::optional<double>(0.37)}) {
      const auto corrected = slipCorrect(v, phi, 0.0);
      ASSERT_TRUE(corrected.has_value());
      EXPECT_EQ(bits(*corrected), bits(v)) << v;
    }
  }
  // Through the whole model: k = 0 and scale 1 put the fitted speed into z untouched, with no phi at all.
  const ModelState state;
  const auto m = modelAt(state, WheelModelParams{}, 0.7351);
  ASSERT_TRUE(m.has_value());
  EXPECT_EQ(bits(m->z(0)), bits(0.7351));
}

TEST(SlipCorrection, DividesByOnePlusKPhiAndNeedsPhiOtherwise)
{
  const auto corrected = slipCorrect(1.0, 0.1, 0.13);
  ASSERT_TRUE(corrected.has_value());
  EXPECT_DOUBLE_EQ(*corrected, 1.0 / 1.013);
  EXPECT_FALSE(slipCorrect(1.0, std::nullopt, 0.13).has_value());  // no IMU window: skip, do not guess
  EXPECT_FALSE(slipCorrect(1.0, -1.0, 1.0).has_value());           // 1 + k phi = 0
  EXPECT_FALSE(slipCorrect(1.0, -2.0, 1.0).has_value());           // negative
  EXPECT_FALSE(slipCorrect(1.0, kInf, 0.13).has_value());
  EXPECT_FALSE(slipCorrect(1.0, 0.1, kNaN).has_value());

  WheelModelParams params;
  params.speed_scale = 0.973;
  params.slip_k = 0.13;
  const ModelState s;
  const WheelFit fit = singleFit(1.0, 0.002, kUnobserved);
  const auto m =
    wheelMeasurement(fit, 0.2, s.R_world_imu, s.velocity_world, s.gyro_bias, s.omega_mean, s.R_robot_to_imu, params);
  ASSERT_TRUE(m.has_value());
  EXPECT_DOUBLE_EQ(m->z(0), 0.973 / (1.0 + 0.13 * 0.2));  // the scale first, then the slip
  EXPECT_EQ(m->z(1), 0.0);
  EXPECT_EQ(m->z(2), 0.0);
  EXPECT_FALSE(
    wheelMeasurement(
      fit, std::nullopt, s.R_world_imu, s.velocity_world, s.gyro_bias, s.omega_mean, s.R_robot_to_imu, params)
      .has_value());
}

TEST(SlipPhi, WindowMeanOfTheRawForwardSpecificForceInGMinusTheIntercept)
{
  // 100 Hz from t = 10 s to 12 s, a_x = 9.81 (0.05 + 0.001 i): over [11 s, 12 s] (i = 100 ... 200) the mean is
  // 0.2 g. Times are built from integers so both window edges fall exactly on samples.
  std::vector<std::pair<double, double>> samples;
  for (int i = 0; i <= 200; ++i) {
    samples.emplace_back((1000 + i) / 100.0, fast_lio::kSlipGravity * (0.05 + 0.001 * i));
  }
  samples[50].second = kNaN;  // outside the window: ignored
  const auto phi = slipPhi(samples, 12.0, 1.0, 0.03);
  ASSERT_TRUE(phi.has_value());
  EXPECT_NEAR(*phi, 0.2 - 0.03, 1e-12);
  const auto level = slipPhi(samples, 12.0, 1.0, 0.0);
  ASSERT_TRUE(level.has_value());
  EXPECT_NEAR(*phi - *level, -0.03, 1e-15);

  samples[150].second = kNaN;  // inside: the window has no defined mean
  EXPECT_FALSE(slipPhi(samples, 12.0, 1.0, 0.03).has_value());
  EXPECT_FALSE(slipPhi(samples, 5.0, 1.0, 0.03).has_value());  // no IMU sample in the window
  EXPECT_FALSE(slipPhi(std::vector<std::pair<double, double>>{}, 12.0, 1.0, 0.0).has_value());
  EXPECT_FALSE(slipPhi(samples, kNaN, 1.0, 0.0).has_value());
  EXPECT_FALSE(slipPhi(samples, 12.0, -1.0, 0.0).has_value());
  EXPECT_FALSE(slipPhi(samples, 12.0, 1.0, kNaN).has_value());

  // The forward component in the wheel frame of an IMU-frame specific force.
  const Eigen::Matrix3d R_robot_to_imu =
    rotation(0.3, Eigen::Vector3d::UnitZ()) * rotation(-0.1, Eigen::Vector3d::UnitY());
  const Eigen::Vector3d a_robot(1.2, -0.4, 9.7);
  EXPECT_NEAR(wheelForward(R_robot_to_imu, R_robot_to_imu * a_robot), 1.2, 1e-14);
}

TEST(WheelMeasurementModel, LeverArmSignMatchesLegEdgeVelocity)
{
  // gravity_align_kinematics: the IMU point moves at R (vx, 0, 0) + omega x r. A state moving exactly so must
  // predict the legs' measured speed (vx, 0, 0) with zero residual.
  ModelState s;
  WheelModelParams params;
  params.lever_arm = Eigen::Vector3d(-0.25, 0.05, 0.12);
  const double vx = 0.9;
  const Eigen::Vector3d omega = s.omega_mean - s.gyro_bias;
  const Eigen::Vector3d v_imu = fast_lio::legEdgeVelocity(vx, s.R_robot_to_imu, omega, params.lever_arm);
  s.velocity_world = s.R_world_imu * v_imu;
  const auto m = modelAt(s, params, vx);
  ASSERT_TRUE(m.has_value());
  EXPECT_LT((m->h - Eigen::Vector3d(vx, 0.0, 0.0)).norm(), 1e-14);
  EXPECT_LT((m->z - m->h).norm(), 1e-14);
  // The lever term is not negligible here, so the opposite sign could not pass.
  EXPECT_GT((s.R_robot_to_imu.transpose() * omega.cross(params.lever_arm)).norm(), 0.1);
}

TEST(WheelMeasurementModel, JacobianMatchesCentralDifferences)
{
  // Rotation perturbed on the right (MTK SO3 boxplus), velocity and gyro bias additively; every other column
  // of the 23-dimensional error state is zero. Central differences with step 1e-6: truncation O(1e-12),
  // rounding O(1e-10).
  const ModelState base;
  WheelModelParams params;
  params.lever_arm = Eigen::Vector3d(-0.25, 0.05, 0.12);
  const auto m = modelAt(base, params);
  ASSERT_TRUE(m.has_value());
  const double step = 1e-6;
  const auto column = [&](int index) {
    ModelState plus = base;
    ModelState minus = base;
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(index % 3) = step;
    if (index < fast_lio::kWheelVelIndex) {
      plus.R_world_imu = base.R_world_imu * expSO3(delta);
      minus.R_world_imu = base.R_world_imu * expSO3(-delta);
    } else if (index < fast_lio::kWheelGyroBiasIndex) {
      plus.velocity_world += delta;
      minus.velocity_world -= delta;
    } else {
      plus.gyro_bias += delta;
      minus.gyro_bias -= delta;
    }
    return Eigen::Vector3d((modelAt(plus, params)->h - modelAt(minus, params)->h) / (2.0 * step));
  };
  for (int block : {fast_lio::kWheelRotIndex, fast_lio::kWheelVelIndex, fast_lio::kWheelGyroBiasIndex}) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_LT((m->H.col(block + k) - column(block + k)).norm(), 1e-8) << "column " << block + k;
    }
  }
  Eigen::Matrix<double, 3, fast_lio::kWheelStateDim> others = m->H;
  others.middleCols<3>(fast_lio::kWheelRotIndex).setZero();
  others.middleCols<3>(fast_lio::kWheelVelIndex).setZero();
  others.middleCols<3>(fast_lio::kWheelGyroBiasIndex).setZero();
  EXPECT_EQ(others, (Eigen::Matrix<double, 3, fast_lio::kWheelStateDim>::Zero()));
  // Without a lever arm the gyro-bias columns vanish: the model is laserMapping's plain one.
  const auto plain = modelAt(base, WheelModelParams{});
  ASSERT_TRUE(plain.has_value());
  EXPECT_EQ(plain->H.middleCols<3>(fast_lio::kWheelGyroBiasIndex), Eigen::Matrix3d::Zero());
  EXPECT_LT((plain->h - base.R_robot_to_imu.transpose() * base.R_world_imu.transpose() * base.velocity_world).norm(),
            1e-14);  // the same product in another order
}

TEST(WheelMeasurementModel, NoiseHasTheFloorTheLeverTermAndTheLateralMarker)
{
  WheelModelParams params;
  params.sigma_lever = 0.02;
  const double lever2 = (0.5 * 0.02) * (0.5 * 0.02);  // |omega| = 0.5 rad/s
  const WheelNoise below = sampleNoise(0.002, kUnobserved, 0.5, params);
  EXPECT_DOUBLE_EQ(below.var_vx, 0.075 * 0.075 + lever2);  // the per-frame floor of design C.1
  EXPECT_FALSE(below.lateral);                             // cov[7] at the bridge's marker
  EXPECT_EQ(below.var_vy, 0.0);
  EXPECT_DOUBLE_EQ(below.var_vz, 0.10 * 0.10);
  const WheelNoise above = sampleNoise(0.01, 4e-4, 0.5, params);
  EXPECT_DOUBLE_EQ(above.var_vx, 0.01 + lever2);
  EXPECT_TRUE(above.lateral);
  EXPECT_DOUBLE_EQ(above.var_vy, 4e-4 + lever2);
  EXPECT_FALSE(sampleNoise(0.01, 2e6, 0.5, params).lateral);
  EXPECT_TRUE(sampleNoise(0.01, 999999.0, 0.5, params).lateral);
  EXPECT_DOUBLE_EQ(sampleNoise(0.002, kUnobserved, 0.0, params).var_vx, 0.075 * 0.075);

  // pack(): (vx, vz) without the lateral axis, (vx, vy, vz) with it, residual z - h.
  const ModelState s;
  auto m = modelAt(s, params);
  ASSERT_TRUE(m.has_value());
  Eigen::Matrix<double, Eigen::Dynamic, fast_lio::kWheelStateDim> H;
  Eigen::VectorXd residual;
  Eigen::VectorXd variances;
  m->pack(H, residual, variances);
  ASSERT_EQ(H.rows(), 2);
  EXPECT_EQ(H.row(0), m->H.row(0));
  EXPECT_EQ(H.row(1), m->H.row(2));
  EXPECT_EQ(residual(0), m->z(0) - m->h(0));
  EXPECT_EQ(residual(1), -m->h(2));
  EXPECT_EQ(variances(0), m->noise.var_vx);
  EXPECT_EQ(variances(1), m->noise.var_vz);
  m->noise = sampleNoise(0.01, 4e-4, 0.5, params);
  m->pack(H, residual, variances);
  ASSERT_EQ(H.rows(), 3);
  EXPECT_EQ(H.row(1), m->H.row(1));
  EXPECT_EQ(residual(1), -m->h(1));
  EXPECT_EQ(variances(1), m->noise.var_vy);
}

TEST(WheelMeasurementModel, NonFiniteInputsGiveNoMeasurement)
{
  ModelState s;
  s.velocity_world(1) = kNaN;
  EXPECT_FALSE(modelAt(s, WheelModelParams{}).has_value());
  const ModelState fine;
  WheelModelParams params;
  params.lever_arm(0) = kInf;
  EXPECT_FALSE(modelAt(fine, params).has_value());
  EXPECT_FALSE(modelAt(fine, WheelModelParams{}, kNaN).has_value());
  EXPECT_TRUE(modelAt(fine, WheelModelParams{}).has_value());
}

// The front end's holder of the wheel samples (wheel_speed_input.hpp): the two callbacks' halves paired by the
// integer header stamp, the scan's own span fitted at its end, the slip regressor from the raw IMU, and why a scan
// makes no update.
namespace
{
using fast_lio::WheelOutcome;
using fast_lio::WheelSpeedInput;

constexpr std::int64_t kNsPerSecond = 1000000000;

std::int32_t stampSec(std::int64_t stamp_ns)
{
  return static_cast<std::int32_t>(stamp_ns / kNsPerSecond);
}

std::uint32_t stampNanosec(std::int64_t stamp_ns)
{
  return static_cast<std::uint32_t>(stamp_ns % kNsPerSecond);
}

void addTwistAt(WheelSpeedInput & input, std::int64_t stamp_ns, double v)
{
  input.addTwist(stampSec(stamp_ns), stampNanosec(stamp_ns), v);
}

void addVariancesAt(WheelSpeedInput & input, std::int64_t stamp_ns, double var_vx, double var_lat)
{
  input.addVariances(stampSec(stamp_ns), stampNanosec(stamp_ns), var_vx, var_lat);
}

// One 10 Hz scan starting at the epoch: [kEpochNs, kEpochNs + 100 ms].
const double kScanBegin = seconds(kEpochNs);
const double kScanEnd = seconds(kEpochNs + 100000000);

fast_lio::WheelStep stepOf(WheelSpeedInput & input,
                           const WheelModelParams & params,
                           const std::optional<Eigen::Vector3d> & omega_mean = ModelState().omega_mean)
{
  const ModelState s;
  return input.step(input.read(kScanBegin, kScanEnd),
                    kScanEnd,
                    omega_mean,
                    s.R_world_imu,
                    s.velocity_world,
                    s.gyro_bias,
                    s.R_robot_to_imu,
                    params);
}

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return m;
}
}  // namespace

TEST(WheelSpeedInput, PairsTheTwoTopicsByStampInEitherArrivalOrder)
{
  WheelSpeedInput input;
  input.configure(true, 0.075, 0.05);
  const std::int64_t first = kEpochNs + 20000000;
  const std::int64_t second = kEpochNs + 40000000;
  const std::int64_t third = kEpochNs + 60000000;
  addTwistAt(input, first, 0.70);  // the gated speed first
  addVariancesAt(input, first, 0.003, kUnobserved);
  addVariancesAt(input, second, 0.004, kUnobserved);  // the variances first
  addTwistAt(input, second, 0.74);
  addTwistAt(input, third, 0.78);  // its variances have not arrived

  const fast_lio::WheelRingRead ring = input.read(kScanBegin, kScanEnd);
  ASSERT_TRUE(ring.fit.has_value());
  EXPECT_EQ(ring.fit->samples, 2);  // only the paired samples enter the update
  EXPECT_EQ(ring.unpaired, 1);
  EXPECT_EQ(ring.fit->sample_max_vx, 0.004);
  // numerical-design.txt:L38; epoch rounding changes the two double time differences.
  EXPECT_NEAR(composeR(*ring.fit, 0.0, WheelModelParams{}).var_vx, 0.17576343876330622, 1e-6 * 0.17576343876330622);
  EXPECT_EQ(ring.unmatched_samples, 0u);
  EXPECT_EQ(ring.rejected, 0u);

  addVariancesAt(input, third, 0.003, kUnobserved);  // late, but in time for the next read
  const fast_lio::WheelRingRead later = input.read(kScanBegin, kScanEnd);
  ASSERT_TRUE(later.fit.has_value());
  EXPECT_EQ(later.fit->samples, 3);
  EXPECT_EQ(later.unpaired, 0);
}

TEST(WheelSpeedInput, HalvesPushedOutUnpairedAreCountedAndGone)
{
  WheelSpeedInput input;
  input.configure(true, 0.075, 0.05);
  const std::int64_t period = kNsPerSecond / static_cast<std::int64_t>(fast_lio::kWheelBridgeRateHz);
  const std::size_t extra = 3;
  for (std::size_t i = 0; i < fast_lio::kWheelSampleCapacity + extra; ++i) {
    addTwistAt(input, kEpochNs + static_cast<std::int64_t>(i) * period, 0.5);  // no variance ever arrives
  }
  fast_lio::WheelRingRead ring = input.read(kScanBegin, kScanEnd);
  EXPECT_EQ(ring.unmatched_samples, extra);
  EXPECT_FALSE(ring.fit.has_value());  // nothing of the scan's span is paired

  // Variances arriving after their samples were pushed out are older than everything held: counted and gone.
  for (std::size_t i = 0; i < extra; ++i) {
    addVariancesAt(input, kEpochNs + static_cast<std::int64_t>(i) * period, 0.003, kUnobserved);
  }
  ring = input.read(kScanBegin, kScanEnd);
  EXPECT_EQ(ring.unmatched_variances, extra);
  EXPECT_EQ(ring.unmatched_samples, extra);
  EXPECT_EQ(ring.rejected, 0u);
}

TEST(WheelSpeedInput, MalformedStampsAndDuplicateHalvesAreRefusedAndCounted)
{
  WheelSpeedInput input;
  input.configure(true, 0.075, 0.05);
  input.addTwist(stampSec(kEpochNs), 1000000000u, 0.5);  // a nanosecond field of a whole second
  input.addVariances(stampSec(kEpochNs), 1000000000u, 0.003, kUnobserved);
  addTwistAt(input, kEpochNs + 50000000, 0.5);
  addTwistAt(input, kEpochNs + 50000000, 0.6);  // the same stamp twice
  const fast_lio::WheelRingRead ring = input.read(kScanBegin, kScanEnd);
  EXPECT_EQ(ring.rejected, 3u);
  EXPECT_EQ(ring.unpaired, 1);
}

TEST(WheelSpeedInput, WithoutAVarianceTopicAnEndpointSamplePreservesTodaysUnits)
{
  // Today's update: the gated sample with R = diag(vx^2, vy^2, vz^2) of the parameter sigmas, z = v * scale,
  // h = R_robot<-imu R_world<-imu^T v and H on the rotation and velocity columns. Without a variance topic, a zero
  // lever arm and no lever uncertainty, one sample in the scan gives the same R, bit for bit, and the same model.
  const double sigma_vx = 0.10;
  const double sigma_vy = 0.05;
  const double sigma_vz = 0.20;
  const double scale = 1.02;
  const double v = 0.83;
  WheelSpeedInput input;
  input.configure(false, sigma_vx, sigma_vy);
  addTwistAt(input, kEpochNs + 40000000, v);
  WheelModelParams params;
  params.speed_scale = scale;
  params.sigma_vx_floor = sigma_vx;
  params.sigma_vz = sigma_vz;
  const auto early = stepOf(input, params);
  EXPECT_EQ(early.outcome, WheelOutcome::kUnidentifiableEndpoint);
  EXPECT_FALSE(early.measurement);
  addTwistAt(input, kEpochNs + 100000000, v);
  // Exclude the earlier sample to exercise the exact single-endpoint R14 contract.
  const ModelState endpoint_state;
  const fast_lio::WheelStep step = input.step(input.read(kScanEnd, kScanEnd),
                                              kScanEnd,
                                              endpoint_state.omega_mean,
                                              endpoint_state.R_world_imu,
                                              endpoint_state.velocity_world,
                                              endpoint_state.gyro_bias,
                                              endpoint_state.R_robot_to_imu,
                                              params);
  ASSERT_EQ(step.outcome, WheelOutcome::kUpdate);
  ASSERT_TRUE(step.measurement.has_value());
  EXPECT_EQ(step.ring.fit->samples, 1);
  EXPECT_EQ(step.ring.unpaired, 0);

  Eigen::Matrix<double, Eigen::Dynamic, fast_lio::kWheelStateDim> H;
  Eigen::VectorXd residual;
  Eigen::VectorXd variances;
  step.measurement->pack(H, residual, variances);
  ASSERT_EQ(variances.size(), 3);
  EXPECT_EQ(bits(variances(0)), bits(sigma_vx * sigma_vx));
  EXPECT_EQ(bits(variances(1)), bits(sigma_vy * sigma_vy));
  EXPECT_EQ(bits(variances(2)), bits(sigma_vz * sigma_vz));
  EXPECT_EQ(bits(step.measurement->z(0)), bits(v * scale));

  const ModelState s;
  const Eigen::Matrix3d Re_T = s.R_robot_to_imu.transpose();
  const Eigen::Vector3d v_imu = s.R_world_imu.transpose() * s.velocity_world;
  EXPECT_TRUE(step.measurement->h.isApprox(Re_T * v_imu, 1e-14));
  Eigen::Matrix<double, 3, fast_lio::kWheelStateDim> today = Eigen::Matrix<double, 3, fast_lio::kWheelStateDim>::Zero();
  today.block<3, 3>(0, fast_lio::kWheelRotIndex) = Re_T * skew(v_imu);
  today.block<3, 3>(0, fast_lio::kWheelVelIndex) = Re_T * s.R_world_imu.transpose();
  EXPECT_TRUE(H.isApprox(today, 1e-14)) << H << "\n\n" << today;
  EXPECT_TRUE(H.middleCols(fast_lio::kWheelGyroBiasIndex, 3).isZero(0.0));
}

TEST(WheelSpeedInput, AScanWithoutAPairedSampleInItsSpanMakesNoUpdate)
{
  WheelSpeedInput input;
  input.configure(true, 0.075, 0.05);
  addTwistAt(input, kEpochNs - 20000000, 0.5);  // before the scan
  addVariancesAt(input, kEpochNs - 20000000, 0.003, kUnobserved);
  addTwistAt(input, kEpochNs + 50000000, 0.5);  // in the scan, never paired
  const fast_lio::WheelStep step = stepOf(input, WheelModelParams{});
  EXPECT_EQ(step.outcome, WheelOutcome::kNoSample);
  EXPECT_FALSE(step.measurement.has_value());
  EXPECT_EQ(step.ring.unpaired, 1);
  EXPECT_EQ(input.counts().no_sample, 1u);
  EXPECT_EQ(input.counts().updates, 0u);
}

TEST(WheelSpeedInput, AScanWithoutImuSamplesMakesNoUpdate)
{
  WheelSpeedInput input;
  input.configure(false, 0.10, 0.05);
  addTwistAt(input, kEpochNs + 50000000, 0.5);
  EXPECT_EQ(stepOf(input, WheelModelParams{}).outcome, WheelOutcome::kUnidentifiableEndpoint);
  addTwistAt(input, kEpochNs + 100000000, 0.5);
  const fast_lio::WheelStep step = stepOf(input, WheelModelParams{}, std::nullopt);
  EXPECT_EQ(step.outcome, WheelOutcome::kNoImu);
  EXPECT_FALSE(step.measurement.has_value());
  EXPECT_EQ(input.counts().no_imu, 1u);
}

TEST(WheelSpeedInput, TheSlipNeedsPhiOnlyWhenKIsNotZero)
{
  const double v = 0.9;
  WheelModelParams slip;
  slip.slip_k = 0.128;
  slip.slip_phi0 = -0.011;
  slip.slip_window_s = 1.0;

  // No IMU sample in the slip window: k != 0 makes no update and counts it, k = 0 updates without phi.
  WheelSpeedInput without_imu;
  without_imu.configure(false, 0.10, 0.05);
  addTwistAt(without_imu, kEpochNs + 50000000, v);
  EXPECT_EQ(stepOf(without_imu, slip).outcome, WheelOutcome::kUnidentifiableEndpoint);
  addTwistAt(without_imu, kEpochNs + 100000000, v);
  const fast_lio::WheelStep refused = stepOf(without_imu, slip);
  EXPECT_EQ(refused.outcome, WheelOutcome::kPhiUnavailable);
  EXPECT_FALSE(refused.phi.has_value());
  EXPECT_FALSE(refused.measurement.has_value());
  EXPECT_EQ(without_imu.counts().phi_unavailable, 1u);
  const fast_lio::WheelStep plain = stepOf(without_imu, WheelModelParams{});
  EXPECT_EQ(plain.outcome, WheelOutcome::kUpdate);
  EXPECT_FALSE(plain.phi.has_value());
  ASSERT_TRUE(plain.measurement.has_value());
  EXPECT_EQ(bits(plain.measurement->z(0)), bits(v));

  // 200 Hz over the 1.5 s before the scan end, a_x = 0.5 m/s^2: phi is the window mean in g minus phi0, and the
  // samples older than the window are forgotten.
  WheelSpeedInput with_imu;
  with_imu.configure(false, 0.10, 0.05);
  addTwistAt(with_imu, kEpochNs + 50000000, v);
  EXPECT_EQ(stepOf(with_imu, slip).outcome, WheelOutcome::kUnidentifiableEndpoint);
  addTwistAt(with_imu, kEpochNs + 100000000, v);
  const std::int64_t imu_period = 5000000;
  for (std::int64_t i = 300; i >= 0; --i) {
    with_imu.recordImu(seconds(kEpochNs + 100000000 - i * imu_period), 0.5);
  }
  const fast_lio::WheelStep corrected = stepOf(with_imu, slip);
  ASSERT_EQ(corrected.outcome, WheelOutcome::kUpdate);
  ASSERT_TRUE(corrected.phi.has_value());
  const double phi = 0.5 / fast_lio::kSlipGravity + 0.011;
  EXPECT_NEAR(*corrected.phi, phi, 1e-12);
  EXPECT_NEAR(corrected.measurement->z(0), v / (1.0 + 0.128 * phi), 1e-12);
  EXPECT_LE(with_imu.imuHistorySize(), 201u);
  EXPECT_GE(with_imu.imuHistorySize(), 200u);

  // k = 0 with phi at hand: the speed itself, bit for bit, and phi still reported.
  const fast_lio::WheelStep level = stepOf(with_imu, WheelModelParams{});
  ASSERT_TRUE(level.measurement.has_value());
  EXPECT_EQ(bits(level.measurement->z(0)), bits(v));
  EXPECT_TRUE(level.phi.has_value());
}

TEST(WheelSpeedInput, ImuHistoryStartsAgainWhenTimeGoesBack)
{
  WheelSpeedInput input;
  input.recordImu(10.0, 0.1);
  input.recordImu(10.0, 0.2);  // a repeated stamp is kept
  input.recordImu(10.005, 0.3);
  EXPECT_EQ(input.imuHistorySize(), 3u);
  input.recordImu(5.0, 0.4);  // the input started again
  EXPECT_EQ(input.imuHistorySize(), 1u);
}

TEST(WheelSpeedInput, WheelLineCarriesTheOutcomeAndEveryCounter)
{
  WheelSpeedInput input;
  input.configure(true, 0.075, 0.05);
  addTwistAt(input, kEpochNs + 50000000, 0.5);
  addVariancesAt(input, kEpochNs + 50000000, 0.003, 0.004);
  const auto early = stepOf(input, WheelModelParams{});
  EXPECT_EQ(early.outcome, WheelOutcome::kUnidentifiableEndpoint);
  EXPECT_NE(fast_lio::wheelLine(kScanEnd, early, input.counts()).find("outcome=unidentifiable_endpoint"),
            std::string::npos);
  const ModelState state;
  const auto endpoint_ring = input.read(seconds(kEpochNs + 50000000), seconds(kEpochNs + 50000000));
  const fast_lio::WheelStep updated = input.step(endpoint_ring,
                                                 seconds(kEpochNs + 50000000),
                                                 state.omega_mean,
                                                 state.R_world_imu,
                                                 state.velocity_world,
                                                 state.gyro_bias,
                                                 state.R_robot_to_imu,
                                                 WheelModelParams{});
  ASSERT_EQ(updated.outcome, WheelOutcome::kUpdate);
  const std::string line = fast_lio::wheelLine(kScanEnd, updated, input.counts());
  EXPECT_EQ(line.rfind("[WHEEL] t=", 0), 0u) << line;
  for (const char * field : {" outcome=update ",
                             " samples=1 ",
                             " unpaired=0 ",
                             " phi=nan ",
                             " updates=1 ",
                             " no_sample=0 ",
                             " no_imu=0 ",
                             " phi_unavailable=0 ",
                             " invalid=1 ",
                             " unmatched_samples=0 ",
                             " unmatched_variances=0 "}) {
    EXPECT_NE(line.find(field), std::string::npos) << field << " in " << line;
  }
  EXPECT_NE(line.find(" rejected=0"), std::string::npos) << line;
  EXPECT_EQ(line.find("r_vy=nan"), std::string::npos) << line;  // the lateral axis was observed

  WheelSpeedInput empty;
  empty.configure(true, 0.075, 0.05);
  const fast_lio::WheelStep none = stepOf(empty, WheelModelParams{});
  const std::string skipped = fast_lio::wheelLine(kScanEnd, none, empty.counts());
  EXPECT_NE(skipped.find(" outcome=no_sample samples=0 unpaired=0 raw_vx=nan fit_rms=nan "), std::string::npos)
    << skipped;
  EXPECT_NE(skipped.find(" z=nan r_vx=nan r_vy=nan r_vz=nan "), std::string::npos) << skipped;
}

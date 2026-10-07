#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>

#include "population_moments.hpp"

namespace fast_lio
{

struct ImuStaticInitParams
{
  bool enabled = false;
  double minDurationS = 1.0;
  int minSamples = 100;
  double gyroRmsMax = 0.1;   // rad/s
  double gravityRef = 9.81;  // raw IMU acceleration units
  double gravityTol = 0.05;  // relative to gravityRef
  double maxGapS = 0.1;
};

inline bool validImuStaticInitParams(const ImuStaticInitParams & params)
{
  const auto positive = [](double value) {
    return std::isfinite(value) && value > 0.0;
  };
  return params.minSamples > 0 && positive(params.minDurationS) && positive(params.gyroRmsMax) &&
         positive(params.gravityRef) && positive(params.gravityTol) && positive(params.maxGapS);
}

inline bool imuStaticInitActive(bool requireStill, const ImuStaticInitParams & params)
{
  return requireStill && params.enabled;
}

struct ImuStaticInitSample
{
  double stamp = 0.0;
  Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

enum class ImuStaticInitDecision
{
  kDisabled,
  kAccumulating,
  kRestart,
  kReady,
  kDegraded
};

class ImuStaticInitGate
{
public:
  void reset()
  {
    window_ = {};
    firstStamp_ = lastStamp_ = 0.0;
  }

  std::size_t samples() const { return window_.count; }

  // The caller feeds exactly the batch used by its gravity moments, and restarts
  // those moments whenever this returns kRestart. The attempt clock belongs to
  // the caller and is deliberately not reset with this window.
  template<class Range, class SampleOf>
  ImuStaticInitDecision observe(const Range & batch,
                                SampleOf sampleOf,
                                bool requireStill,
                                const ImuStaticInitParams & params,
                                double scatterTol,
                                double elapsedS,
                                double timeoutS)
  {
    if (!imuStaticInitActive(requireStill, params)) {
      return ImuStaticInitDecision::kDisabled;
    }
    Moments batchMoments;
    bool continuous = true;
    for (const auto & entry : batch) {
      const ImuStaticInitSample sample = sampleOf(entry);
      if (!std::isfinite(sample.stamp) || !sample.acceleration.allFinite() || !sample.gyro.allFinite() ||
          !std::isfinite(sample.gyro.squaredNorm())) {
        return restart();
      }
      if (window_.count != 0) {
        if (!(sample.stamp > lastStamp_)) {
          return restart();
        }
        continuous = continuous && sample.stamp - lastStamp_ <= params.maxGapS;
      } else {
        firstStamp_ = sample.stamp;
      }
      lastStamp_ = sample.stamp;
      window_.add(sample);
      batchMoments.add(sample);
    }
    if (batchMoments.count == 0 || !window_.finite() || !batchMoments.finite()) {
      return restart();
    }
    const bool still =
      continuous && window_.still(scatterTol, params.gyroRmsMax) && batchMoments.still(scatterTol, params.gyroRmsMax) &&
      std::abs(window_.meanAcceleration.norm() - params.gravityRef) <= params.gravityTol * params.gravityRef;
    if (still && lastStamp_ - firstStamp_ >= params.minDurationS &&
        window_.count >= static_cast<std::size_t>(params.minSamples)) {
      return ImuStaticInitDecision::kReady;
    }
    // A valid but moving/incomplete window uses the same strictly-greater
    // timeout as legacy initialization. Invalid or unordered data never passes.
    if (elapsedS > timeoutS) {
      return ImuStaticInitDecision::kDegraded;
    }
    return still ? ImuStaticInitDecision::kAccumulating : restart();
  }

private:
  struct Moments
  {
    std::size_t count = 0;
    Eigen::Vector3d meanAcceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d varianceAcceleration = Eigen::Vector3d::Zero();
    double meanGyroSquared = 0.0;

    void add(const ImuStaticInitSample & sample)
    {
      ++count;
      updatePopulationMoments(sample.acceleration, count, meanAcceleration, varianceAcceleration);
      meanGyroSquared += (sample.gyro.squaredNorm() - meanGyroSquared) / static_cast<double>(count);
    }

    bool finite() const
    {
      return meanAcceleration.allFinite() && varianceAcceleration.allFinite() &&
             std::isfinite(meanAcceleration.norm()) && std::isfinite(varianceAcceleration.sum()) &&
             std::isfinite(meanGyroSquared) && varianceAcceleration.sum() >= 0.0 && meanGyroSquared >= 0.0;
    }

    bool still(double scatterTol, double gyroRmsMax) const
    {
      const double norm = meanAcceleration.norm();
      return norm > 0.0 && std::sqrt(varianceAcceleration.sum()) / norm <= scatterTol &&
             std::sqrt(meanGyroSquared) <= gyroRmsMax;
    }
  };

  ImuStaticInitDecision restart()
  {
    reset();
    return ImuStaticInitDecision::kRestart;
  }

  Moments window_;
  double firstStamp_ = 0.0;
  double lastStamp_ = 0.0;
};

}  // namespace fast_lio

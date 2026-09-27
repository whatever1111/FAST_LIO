#pragma once

#include <cstddef>
#include <stdexcept>

namespace fast_lio
{

// Welford's population variance update. count includes the incoming sample;
// count==1 starts a fresh window. No integer products or sample-variance correction.
template<class Vector>
void updatePopulationMoments(const Vector & sample, std::size_t count, Vector & mean, Vector & variance)
{
  if (count == 0) {
    throw std::invalid_argument("population moment count must include the sample");
  }
  if (count == 1) {
    mean = sample;
    variance.setZero();
    return;
  }
  const double n = static_cast<double>(count);
  const Vector meanStep = (sample - mean) / n;
  mean += meanStep;
  variance = variance * ((n - 1.0) / n) + meanStep.cwiseProduct(sample - mean);
}

}  // namespace fast_lio

#pragma once

#include "plane_fit.hpp"

namespace fast_lio
{

struct PlaneResidual
{
  float distance = 0.0f;
  bool accepted = false;
};

// Neighbours and their fitted plane may survive an iEKF iteration. Whether the
// current transformed point passes the residual gate never does.
class CachedPlane
{
public:
  void invalidate() { valid_ = false; }
  bool valid() const { return valid_; }
  const Eigen::Vector4f & coefficients() const { return coefficients_; }

  template<int Count, typename Points>
  bool fit(const Points & points, float threshold, float body_range_sqrt)
  {
    invalidate();
    if (!std::isfinite(body_range_sqrt) || body_range_sqrt <= 0.0f ||
        !fitPlane<Count>(points, threshold, coefficients_)) {
      return false;
    }
    bodyRangeSqrt_ = body_range_sqrt;
    valid_ = true;
    return true;
  }

  PlaneResidual evaluate(float x, float y, float z) const
  {
    PlaneResidual result;
    if (!valid_)
      return result;
    result.distance = coefficients_[0] * x + coefficients_[1] * y + coefficients_[2] * z + coefficients_[3];
    // Keep the original FAST-LIO distance/range gate, including its float score.
    const float score = 1 - 0.9 * std::fabs(result.distance) / bodyRangeSqrt_;
    result.accepted = std::isfinite(result.distance) && score > 0.9;
    return result;
  }

private:
  Eigen::Vector4f coefficients_ = Eigen::Vector4f::Zero();
  float bodyRangeSqrt_ = 0.0f;
  bool valid_ = false;
};

}  // namespace fast_lio

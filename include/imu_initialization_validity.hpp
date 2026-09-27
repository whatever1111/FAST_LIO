#pragma once

#include <cmath>

namespace fast_lio
{
// Numerical normalization bound in raw acceleration units, not a stillness gate.
constexpr double kMinInitializationAccelNorm = 1e-6;

template<class Vector>
bool validInitializationMean(const Vector & acceleration, const Vector & gyro)
{
  const double norm = acceleration.norm();
  return acceleration.allFinite() && gyro.allFinite() && std::isfinite(norm) && norm > kMinInitializationAccelNorm;
}

template<class State>
bool finiteInitializationState(const State & state)
{
  return state.pos.allFinite() && state.rot.toRotationMatrix().allFinite() &&
         state.offset_R_L_I.toRotationMatrix().allFinite() && state.offset_T_L_I.allFinite() && state.vel.allFinite() &&
         state.bg.allFinite() && state.ba.allFinite() && state.grav.get_vect().allFinite();
}
}  // namespace fast_lio

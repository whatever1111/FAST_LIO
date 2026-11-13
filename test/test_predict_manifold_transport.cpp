// Covariance transport of esekf::predict on the front end's state manifold.
//
// With Q = 0 the predicted covariance is J P J^T, where J is the error-state transition of the nominal step
// x+ = x oplus f(x, u) dt:
//   J = d[((x boxplus e) oplus f(x boxplus e, u) dt) boxminus (x oplus f(x, u) dt)] / de  at e = 0.
// The oracle differentiates that map by central differences through the manifold's own boxplus, oplus and
// boxminus, so it shares no expression with the filter's analytic transition. The prior is dense, anisotropic and
// correlated across components: a covariance that is not carried through the step's rotation differs from J P J^T.
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iterator>

#include "use-ikfom.hpp"

namespace
{
using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;
using Cov = Filter::cov;
using Delta = Filter::vectorized_state;
using Rate = Filter::flatted_state;
using RateJacobian = Eigen::Matrix<double, state_ikfom::DIM, state_ikfom::DOF>;
using NoiseJacobian = Eigen::Matrix<double, state_ikfom::DIM, 12>;

// Offsets in the rate (DIM) layout of state_ikfom.
constexpr int kRotRate = 3;
constexpr int kExtrinsicRotRate = 6;
constexpr int kGravityRate = 21;
// Offset of the body rotation in the error-state (DOF) layout.
constexpr int kRotError = 3;

// Central-difference step of the oracle and the bound on its error in P (truncation and rounding stay far below).
constexpr double kStep = 1e-6;
constexpr double kTolerance = 1e-8;

// Fixture process models whose rates come from the input and not from the state, so f_x = 0 and J is the transport
// of oplus alone. rotationRates turns the body (rot) by u.gyro and the lidar extrinsic (offset_R_L_I) by u.acc;
// directionRate turns the gravity direction (S2) by u.acc. FAST-LIO's own model never moves the S2 component, so
// only a fixture like this one reaches the S2 transport.
Rate rotationRates(state_ikfom &, const input_ikfom & input)
{
  Rate rate = Rate::Zero();
  rate.segment<3>(kRotRate) = input.gyro;
  rate.segment<3>(kExtrinsicRotRate) = input.acc;
  return rate;
}

Rate directionRate(state_ikfom &, const input_ikfom & input)
{
  Rate rate = Rate::Zero();
  rate.segment<3>(kGravityRate) = input.acc;
  return rate;
}

RateJacobian noStateDependence(state_ikfom &, const input_ikfom &)
{
  return RateJacobian::Zero();
}

NoiseJacobian noNoiseInput(state_ikfom &, const input_ikfom &)
{
  return NoiseJacobian::Zero();
}

// Dense SPD prior with unequal variances and nonzero covariances between every pair of components.
Cov correlatedPrior()
{
  Cov lower = Cov::Zero();
  for (int row = 0; row < lower.rows(); ++row) {
    lower(row, row) = 0.3 + 0.01 * row;
    for (int col = 0; col < row; ++col) {
      lower(row, col) = 0.035 * std::sin(1.0 + row + 2.0 * col);
    }
  }
  return lower * lower.transpose();
}

state_ikfom initialState()
{
  state_ikfom state;
  state.pos = Eigen::Vector3d(1.5, -0.7, 0.4);
  state.rot = SO3(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1.0, -2.0, 3.0).normalized()).toRotationMatrix());
  state.offset_R_L_I = SO3(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()).toRotationMatrix());
  state.offset_T_L_I = Eigen::Vector3d(0.05, -0.02, 0.1);
  state.vel = Eigen::Vector3d(1.0, 0.2, -0.1);
  state.bg = Eigen::Vector3d(0.01, -0.02, 0.015);
  state.ba = Eigen::Vector3d(0.02, 0.01, -0.03);
  state.grav = S2(Eigen::Vector3d(2.0, -1.0, -9.0));
  return state;
}

// The nominal step from the perturbed state before boxplus error.
state_ikfom propagate(
  const state_ikfom & before, const Delta & error, const input_ikfom & input, double dt, Filter::processModel * rate)
{
  state_ikfom state = before;
  state.boxplus(error);
  const Rate increment = rate(state, input);
  state.oplus(increment, dt);
  return state;
}

Cov numericalTransition(const state_ikfom & before, const input_ikfom & input, double dt, Filter::processModel * rate)
{
  const state_ikfom after = propagate(before, Delta::Zero(), input, dt, rate);
  Cov transition;
  for (int col = 0; col < transition.cols(); ++col) {
    const Delta plus_error = kStep * Delta::Unit(col);
    const Delta minus_error = -plus_error;
    Delta plus;
    Delta minus;
    propagate(before, plus_error, input, dt, rate).boxminus(plus, after);
    propagate(before, minus_error, input, dt, rate).boxminus(minus, after);
    transition.col(col) = (plus - minus) / (2.0 * kStep);
  }
  return transition;
}

struct Prediction
{
  state_ikfom state;
  Cov covariance;
};

// One esekf::predict with Q = 0 from the given state and covariance.
Prediction predictOnce(const state_ikfom & before,
                       const Cov & prior,
                       const input_ikfom & input,
                       double dt,
                       Filter::processModel * rate,
                       Filter::processMatrix1 * rate_jacobian,
                       Filter::processMatrix2 * noise_jacobian)
{
  Filter filter;
  double epsilon[state_ikfom::DOF];
  std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
  filter.init_dyn_share(
    rate, rate_jacobian, noise_jacobian, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 5, epsilon);
  state_ikfom state = before;
  filter.change_x(state);
  Cov covariance = prior;
  filter.change_P(covariance);
  Filter::processnoisecovariance no_noise = Filter::processnoisecovariance::Zero();
  double step = dt;
  filter.predict(step, no_noise, input);
  return {filter.get_x(), filter.get_P()};
}

// predict still integrates the nominal state as x oplus f(x, u) dt and carries P by the oracle's J.
void expectTransported(const Prediction & actual,
                       const state_ikfom & before,
                       const input_ikfom & input,
                       double dt,
                       Filter::processModel * rate,
                       const Cov & expected)
{
  Delta state_error;
  actual.state.boxminus(state_error, propagate(before, Delta::Zero(), input, dt, rate));
  EXPECT_LT(state_error.cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((actual.covariance - expected).cwiseAbs().maxCoeff(), kTolerance);
}
}  // namespace

TEST(PredictManifoldTransport, RotationRatesCarryTheCovarianceIntoBothTurnedFrames)
{
  const state_ikfom before = initialState();
  const Cov prior = correlatedPrior();
  input_ikfom input;
  input.gyro = Eigen::Vector3d(0.9, -0.6, 1.2);  // the body's rate, about all three axes
  input.acc = Eigen::Vector3d(-0.4, 0.7, 0.3);   // the extrinsic rotation's rate in this fixture
  constexpr double kDt = 0.05;

  const Cov transition = numericalTransition(before, input, kDt, rotationRates);
  const Cov expected = transition * prior * transition.transpose();
  // Leaving both rotation blocks unturned would return the prior itself.
  ASSERT_GT((expected - prior).cwiseAbs().maxCoeff(), 1e-4);

  const Prediction actual = predictOnce(before, prior, input, kDt, rotationRates, noStateDependence, noNoiseInput);
  expectTransported(actual, before, input, kDt, rotationRates, expected);
}

TEST(PredictManifoldTransport, DirectionRateCarriesTheCovarianceIntoTheTurnedTangentPlane)
{
  const state_ikfom before = initialState();
  const Cov prior = correlatedPrior();
  const Eigen::Vector3d direction = before.grav.get_vect().normalized();
  struct Case
  {
    const char * name;
    Eigen::Vector3d rate;
    bool direction_stays;
  };
  // About the direction itself the direction stays and only its tangent plane turns, so a covariance left unturned
  // would be the prior. Across it the direction moves along a great circle: on a circle through the pole of this S2
  // type's tangent basis (+x) the basis turns with the direction and J is the identity, so a step that only projects
  // the old tangent plane onto the new one shrinks the covariance; off the pole J also turns the tangent coordinates.
  // The general rate does both.
  const std::array<Case, 4> cases = {{
    {"about the direction", 0.8 * direction, true},
    {"great circle through the basis pole", 1.1 * direction.cross(Eigen::Vector3d::UnitX()).normalized(), false},
    {"great circle off the basis pole", 1.1 * direction.cross(Eigen::Vector3d::UnitZ()).normalized(), false},
    {"general", Eigen::Vector3d(0.5, -0.9, 0.35), false},
  }};
  constexpr double kDt = 0.1;
  for (const auto & test_case : cases) {
    SCOPED_TRACE(test_case.name);
    input_ikfom input;
    input.acc = test_case.rate;  // the gravity direction's rate in this fixture
    const Cov transition = numericalTransition(before, input, kDt, directionRate);
    const Cov expected = transition * prior * transition.transpose();
    if (test_case.direction_stays) {
      ASSERT_GT((expected - prior).cwiseAbs().maxCoeff(), 1e-4);
    }

    const Prediction actual = predictOnce(before, prior, input, kDt, directionRate, noStateDependence, noNoiseInput);
    expectTransported(actual, before, input, kDt, directionRate, expected);
  }
}

TEST(PredictManifoldTransport, ImuModelTurnsTheCovarianceOfARotatingBody)
{
  const state_ikfom before = initialState();
  const Cov prior = correlatedPrior();
  input_ikfom input;
  input.gyro = Eigen::Vector3d(0.9, -0.6, 1.2);
  input.acc = Eigen::Vector3d(0.3, -0.2, 9.9);
  constexpr double kDt = 0.005;  // one 200 Hz IMU interval

  const Cov transition = numericalTransition(before, input, kDt, get_f);
  // The rotation error turns by (gyro - bg) dt within one step; keep that turn far above the tolerance.
  ASSERT_GT((transition.block<3, 3>(kRotError, kRotError) - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-3);
  const Cov expected = transition * prior * transition.transpose();

  const Prediction actual = predictOnce(before, prior, input, kDt, get_f, df_dx, df_dw);
  expectTransported(actual, before, input, kDt, get_f, expected);
}

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <limits>
#include <vector>

#include "correspondence_gate.hpp"

using fast_lio::cauchyWeight;
using fast_lio::correspondenceAdmitted;
using fast_lio::CorrespondenceGateParams;
using fast_lio::flooredPriorBlock;
using fast_lio::gateSigma;
using fast_lio::gemanMcClureWeight;
using fast_lio::residualVariance;
using fast_lio::robustResidualScale;

namespace
{
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPointCov = 0.001;  // LASER_POINT_COV: 3.2 cm

CorrespondenceGateParams gate(double k, double nn0 = 0.0)
{
  CorrespondenceGateParams p;
  p.innovation_k = k;
  p.nn0_max = nn0;
  return p;
}
}  // namespace

// Off by default: whatever the residual, the legacy test alone decides.
TEST(CorrespondenceGate, DisabledNeverRejectsAFiniteResidual)
{
  const CorrespondenceGateParams p;
  EXPECT_TRUE(correspondenceAdmitted(p, 0.25, kPointCov, 0.4));
  EXPECT_TRUE(correspondenceAdmitted(p, -0.5, kNaN, kNaN));
  EXPECT_FALSE(correspondenceAdmitted(p, kNaN, kPointCov, 0.1));
}

// A certain pose: the test is 3 sigma of the point noise. A healthy 3 cm residual passes, the 20 cm
// residual of a point matched to the boundary of the mapped region does not.
TEST(CorrespondenceGate, CertainPoseIsAThreeSigmaTestOfThePointNoise)
{
  const Eigen::Matrix<double, 6, 6> P6 = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;
  const Eigen::Vector3d n(0.0, 0.0, 1.0);
  const Eigen::Vector3d A = Eigen::Vector3d(5.0, 0.0, -0.5).cross(n);  // floor point 5 m ahead
  const double var = residualVariance(n, A, P6, kPointCov);
  EXPECT_NEAR(var, kPointCov, 1e-6);
  const double sigma = std::sqrt(var);
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0), 0.03, var, 0.1));
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0), 2.9 * sigma, var, 0.1));
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 3.1 * sigma, var, 0.1));
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), -0.20, var, 0.1));
}

// An uncertain pose opens the gate by the residual that pose error could produce: 1 deg of attitude
// on a point 5 m out is 8.7 cm of plane distance, so a 15 cm residual is admitted there and rejected
// when the attitude is known to 0.1 deg.
TEST(CorrespondenceGate, UncertainPoseOpensTheGateByItsProjection)
{
  Eigen::Matrix<double, 6, 6> P6 = Eigen::Matrix<double, 6, 6>::Zero();
  const double one_deg = M_PI / 180.0;
  P6.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * one_deg * one_deg;
  const Eigen::Vector3d n(0.0, 0.0, 1.0);
  const Eigen::Vector3d A = Eigen::Vector3d(5.0, 0.0, -0.5).cross(n);  // |A| = 5 m lever about y
  const double var = residualVariance(n, A, P6, kPointCov);
  EXPECT_NEAR(std::sqrt(var), std::sqrt(0.0872 * 0.0872 + kPointCov), 2e-3);
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0), 0.15, var, 0.1));
  P6.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.1 * one_deg * 0.1 * one_deg;
  const double var_tight = residualVariance(n, A, P6, kPointCov);
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.15, var_tight, 0.1));
  // position uncertainty along the normal opens it as well: 10 cm of z uncertainty admits 0.3 m
  P6.setZero();
  P6(2, 2) = 0.1 * 0.1;
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0), 0.29, residualVariance(n, A, P6, kPointCov), 0.1));
  // ...but not along a direction the normal does not see
  P6.setZero();
  P6(0, 0) = 0.1 * 0.1;
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.29, residualVariance(n, A, P6, kPointCov), 0.1));
}

// The first-neighbour bound is independent of the innovation test.
TEST(CorrespondenceGate, NearestNeighbourBoundIsSeparate)
{
  EXPECT_TRUE(correspondenceAdmitted(gate(0.0, 0.2), 0.5, kNaN, 0.19));
  EXPECT_FALSE(correspondenceAdmitted(gate(0.0, 0.2), 0.01, kPointCov, 0.21));
  EXPECT_FALSE(correspondenceAdmitted(gate(0.0, 0.2), 0.01, kPointCov, kNaN));
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0, 0.2), 0.03, kPointCov, 0.1));
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0, 0.2), 0.03, kPointCov, 0.3));
}

// Degenerate variance inputs reject rather than admit.
TEST(CorrespondenceGate, BadVarianceRejects)
{
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.01, kNaN, 0.1));
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.01, -1.0, 0.1));
  Eigen::Matrix<double, 6, 6> P6 = Eigen::Matrix<double, 6, 6>::Zero();
  P6(0, 0) = kNaN;
  EXPECT_FALSE(std::isfinite(residualVariance(Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), P6, kPointCov)));
}

// The scan's robust scale: the MAD of a Gaussian core, untouched by a large minority of outliers.
TEST(CorrespondenceGate, RobustScaleIgnoresOutliersAndDegenerateInput)
{
  std::vector<double> v;
  for (int i = -50; i <= 50; ++i) {
    v.push_back(0.001 * i);  // uniform core, MAD = 0.025
  }
  for (int i = 0; i < 20; ++i) {
    v.push_back(0.25);  // a coherent 25 cm minority (17 %)
  }
  const double sigma = robustResidualScale(v);
  EXPECT_NEAR(sigma, 1.4826 * 0.025, 0.01);
  std::vector<double> tiny = {0.01, kNaN};
  EXPECT_DOUBLE_EQ(robustResidualScale(tiny), 0.0);
  std::vector<double> nan_only = {kNaN, kNaN, kNaN};
  EXPECT_DOUBLE_EQ(robustResidualScale(nan_only), 0.0);
}

// The floor: with an overconfident model every residual of a mispositioned scan would be rejected;
// the scan's own scale keeps the gate open in proportion.
TEST(CorrespondenceGate, ScanScaleFloorOpensTheGateWhenTheModelIsOverconfident)
{
  const double model_var = kPointCov;  // 3.2 cm
  EXPECT_NEAR(gateSigma(model_var, 0.0), std::sqrt(kPointCov), 1e-12);
  EXPECT_NEAR(gateSigma(model_var, 0.08), 0.08, 1e-12);
  EXPECT_NEAR(gateSigma(model_var, kNaN), std::sqrt(kPointCov), 1e-12);
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.15, model_var, 0.1, 0.0));
  EXPECT_TRUE(correspondenceAdmitted(gate(3.0), 0.15, model_var, 0.1, 0.06));
  EXPECT_FALSE(correspondenceAdmitted(gate(3.0), 0.25, model_var, 0.1, 0.06));
}

TEST(CorrespondenceGate, AttitudeFloorRaisesOnlyTheRotationVariances)
{
  Eigen::Matrix<double, 6, 6> P6 = Eigen::Matrix<double, 6, 6>::Identity() * 1e-8;
  P6(0, 3) = P6(3, 0) = 2e-9;
  const double floor_rad = 0.3 * M_PI / 180.0;
  const Eigen::Matrix<double, 6, 6> F = flooredPriorBlock(P6, floor_rad);
  for (int k = 0; k < 3; ++k) {
    EXPECT_DOUBLE_EQ(F(k, k), 1e-8);
    EXPECT_DOUBLE_EQ(F(3 + k, 3 + k), floor_rad * floor_rad);
  }
  EXPECT_DOUBLE_EQ(F(0, 3), 2e-9);
  EXPECT_DOUBLE_EQ((flooredPriorBlock(P6, 0.0) - P6).norm(), 0.0);
  EXPECT_DOUBLE_EQ((flooredPriorBlock(P6, kNaN) - P6).norm(), 0.0);
  // a floor already below the prior changes nothing
  P6(4, 4) = 1e-2;
  EXPECT_DOUBLE_EQ(flooredPriorBlock(P6, floor_rad)(4, 4), 1e-2);
}

TEST(CorrespondenceGate, KernelWeightsAreOneAtZeroAndDecayWithTheResidual)
{
  EXPECT_DOUBLE_EQ(gemanMcClureWeight(0.0, 0.05), 1.0);
  EXPECT_DOUBLE_EQ(cauchyWeight(0.0, 0.05), 1.0);
  EXPECT_NEAR(gemanMcClureWeight(0.05, 0.05), 0.25, 1e-12);
  EXPECT_NEAR(cauchyWeight(0.05, 0.05), 0.5, 1e-12);
  EXPECT_LT(gemanMcClureWeight(0.20, 0.05), cauchyWeight(0.20, 0.05));  // GM cuts harder in the tail
  EXPECT_DOUBLE_EQ(gemanMcClureWeight(0.3, 0.0), 1.0);  // no kernel
  EXPECT_DOUBLE_EQ(cauchyWeight(kNaN, 0.05), 1.0);
}

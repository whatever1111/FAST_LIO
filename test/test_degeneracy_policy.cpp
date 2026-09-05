// test_degeneracy_policy.cpp — unit tests for fast_lio::computePositionDegeneracy.

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "degeneracy_policy.hpp"

using fast_lio::computePositionDegeneracy;
using fast_lio::DegeneracyParams;
using fast_lio::DegeneracyResult;
using fast_lio::degeneracyWeight;
using fast_lio::pointWeight;
using fast_lio::schurPositionInformation;

namespace
{
/// Shape-only params: the absolute test is off, so the verdict is the share alone.
DegeneracyParams shapeOnlyParams(double bad = 0.02, double good = 0.10)
{
  DegeneracyParams p;
  p.enable = true;
  p.lambda_bad = bad;
  p.lambda_good = good;
  p.lambda_abs_bad = 0.0;  // good <= bad -> absolute test disabled
  p.lambda_abs_good = 0.0;
  return p;
}

/// Both tests live, as they run on the robot.
DegeneracyParams enabledParams(double bad = 0.02, double good = 0.10, double abs_bad = 2.0, double abs_good = 20.0)
{
  DegeneracyParams p = shapeOnlyParams(bad, good);
  p.lambda_abs_bad = abs_bad;
  p.lambda_abs_good = abs_good;
  return p;
}

/// Stack unit normals into the position columns of a measurement Jacobian, exactly as
/// h_share_model does (the position columns ARE the normals).
Eigen::MatrixXd jacobianFromNormals(const std::vector<Eigen::Vector3d> & normals)
{
  Eigen::MatrixXd h(static_cast<int>(normals.size()), 3);
  for (size_t i = 0; i < normals.size(); ++i) {
    h.row(static_cast<int>(i)) = normals[i].transpose();
  }
  return h;
}

Eigen::Matrix3d informationOf(const std::vector<Eigen::Vector3d> & normals)
{
  Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
  for (const auto & n : normals) {
    m.noalias() += n * n.transpose();
  }
  return m;
}

/// A corridor along +y: walls contribute +-x normals, floor/ceiling +-z, nothing
/// faces the travel direction.
std::vector<Eigen::Vector3d> corridorNormals(int walls, int floors)
{
  std::vector<Eigen::Vector3d> n;
  n.reserve(static_cast<size_t>(walls + floors));
  for (int i = 0; i < walls; ++i) {
    n.emplace_back((i % 2 == 0) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d(-1.0, 0.0, 0.0));
  }
  for (int i = 0; i < floors; ++i) {
    n.emplace_back((i % 2 == 0) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d(0.0, 0.0, -1.0));
  }
  return n;
}

/// An open scene: normals spread evenly over the three axes.
std::vector<Eigen::Vector3d> isotropicNormals(int per_axis)
{
  std::vector<Eigen::Vector3d> n;
  for (int i = 0; i < per_axis; ++i) {
    n.emplace_back(Eigen::Vector3d::UnitX());
    n.emplace_back(Eigen::Vector3d::UnitY());
    n.emplace_back(Eigen::Vector3d::UnitZ());
  }
  return n;
}

/// A scene whose weakest direction (+y) carries a fixed SHARE of the normals, at a
/// point count the caller chooses — the same shape starved or rich.
std::vector<Eigen::Vector3d> wedgeNormals(int weak, int strong_per_axis)
{
  std::vector<Eigen::Vector3d> n;
  for (int i = 0; i < weak; ++i) {
    n.emplace_back(Eigen::Vector3d::UnitY());
  }
  for (int i = 0; i < strong_per_axis; ++i) {
    n.emplace_back(Eigen::Vector3d::UnitX());
    n.emplace_back(Eigen::Vector3d::UnitZ());
  }
  return n;
}

/// The iEKF's position solve in information form, as esekfom.hpp runs it:
///   dx = (H^T H + R * P^-1)^-1 H^T z
/// `prior` is R * P^-1, i.e. how much the propagated prior outweighs one plane.
Eigen::Vector3d solve(const Eigen::MatrixXd & h, const Eigen::VectorXd & z, double prior)
{
  const Eigen::Matrix3d a = h.transpose() * h + prior * Eigen::Matrix3d::Identity();
  return a.ldlt().solve(h.transpose() * z);
}

/// The three blocks of H^T H that the policy judges, built the way h_share_model does:
/// position column = the plane normal n, rotation column = the lever arm p x n.
struct JacobianBlocks
{
  Eigen::Matrix3d tt = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d tr = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d rr = Eigen::Matrix3d::Zero();
};
JacobianBlocks blocksOf(const std::vector<Eigen::Vector3d> & normals, const std::vector<Eigen::Vector3d> & points)
{
  JacobianBlocks b;
  for (size_t i = 0; i < normals.size(); ++i) {
    const Eigen::Vector3d & n = normals[i];
    const Eigen::Vector3d a = points[i].cross(n);
    b.tt.noalias() += n * n.transpose();
    b.tr.noalias() += n * a.transpose();
    b.rr.noalias() += a * a.transpose();
  }
  return b;
}
}  // namespace

// --- weight ramp ------------------------------------------------------------

TEST(DegeneracyWeight, ShareRampClampsToUnitInterval)
{
  const DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.0, 0.0, p), 0.0);
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.02, 0.0, p), 0.0);  // at bad -> 0
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.10, 0.0, p), 1.0);  // at good -> 1
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.5, 0.0, p), 1.0);
  EXPECT_NEAR(degeneracyWeight(0.06, 0.0, p), 0.5, 1e-12);
}

TEST(DegeneracyWeight, DegenerateBoundsMeanFullTrust)
{
  DegeneracyParams p = shapeOnlyParams(0.10, 0.10);  // good <= bad -> both ramps disabled
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.0, 0.0, p), 1.0);
}

TEST(DegeneracyWeight, FloorKeepsATrickle)
{
  DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  p.min_weight = 0.1;
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.0, 0.0, p), 0.1);  // never fully blind
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.5, 0.0, p), 1.0);  // floor does not cap the top
}

TEST(DegeneracyWeight, NonFiniteInputIsFullTrust)
{
  const DegeneracyParams p = shapeOnlyParams();
  EXPECT_DOUBLE_EQ(degeneracyWeight(std::nan(""), 0.0, p), 1.0);
  EXPECT_DOUBLE_EQ(degeneracyWeight(std::numeric_limits<double>::infinity(), 0.0, p), 1.0);
}

// A direction is held back only when BOTH tests condemn it: either one alone means
// full trust. Absolute strength beats a small share (a narrow field of view), and a
// healthy share beats a starved scan (a job for the scalar guards, not a subspace).
TEST(DegeneracyWeight, EitherTestAloneMeansFullTrust)
{
  const DegeneracyParams p = enabledParams(0.02, 0.10, 2.0, 20.0);
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.0, 100.0, p), 1.0);   // blind share, plenty of information
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.33, 0.0, p), 1.0);    // isotropic share, no information
  EXPECT_NEAR(degeneracyWeight(0.06, 2.0, p), 0.5, 1e-12);  // both weak -> the share's verdict
  EXPECT_NEAR(degeneracyWeight(0.0, 11.0, p), 0.5, 1e-12);  // both weak -> the absolute verdict
}

TEST(DegeneracyWeight, HardProjectionCollapsesTheRampToAStep)
{
  DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  p.hard_projection = true;
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.03, 0.0, p), 0.0);  // below the midpoint
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.05, 0.0, p), 0.0);
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.08, 0.0, p), 1.0);  // above the midpoint
  EXPECT_DOUBLE_EQ(degeneracyWeight(0.09, 0.0, p), 1.0);
}

// --- inert paths: the update must run exactly as it does today --------------

TEST(DegeneracyPolicy, DisabledPolicyIsIdentity)
{
  DegeneracyParams p = enabledParams();
  p.enable = false;
  const auto normals = corridorNormals(60, 40);
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  EXPECT_FALSE(r.engaged);
  EXPECT_TRUE(r.weight_matrix.isIdentity(0.0));
  EXPECT_DOUBLE_EQ(pointWeight(r, Eigen::Vector3d::UnitY()), 1.0);
}

TEST(DegeneracyPolicy, EmptyOrNonFiniteInformationIsIdentity)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = corridorNormals(60, 40);
  const Eigen::Matrix3d m = informationOf(normals);

  EXPECT_TRUE(computePositionDegeneracy(m, 0, p).weight_matrix.isIdentity(0.0));   // no points
  EXPECT_TRUE(computePositionDegeneracy(m, -3, p).weight_matrix.isIdentity(0.0));  // nonsense count

  Eigen::Matrix3d bad = m;
  bad(1, 1) = std::nan("");
  const auto r = computePositionDegeneracy(bad, 100, p);
  EXPECT_FALSE(r.engaged);
  EXPECT_TRUE(r.weight_matrix.isIdentity(0.0));
}

// An open scene must come back exactly untouched — this is the accuracy-neutrality
// bar: the policy may not cost anything when nothing is wrong.
TEST(DegeneracyPolicy, IsotropicSceneIsNotEngaged)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = isotropicNormals(40);
  const auto r = computePositionDegeneracy(informationOf(normals), static_cast<int>(normals.size()), p);
  EXPECT_FALSE(r.engaged);
  EXPECT_TRUE(r.weight_matrix.isIdentity(0.0));
  // an ideal scene splits the information three ways; the thresholds must sit below it
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(r.lambda_norm(i), 1.0 / 3.0, 1e-12);
  }
}

// The normalised eigenvalues are shares of one scan's information, so they sum to 1
// whatever the scene — the fact that caps the weakest direction at 1/3, sets where the
// share thresholds have to live, and is exactly why an absolute test is needed too.
TEST(DegeneracyPolicy, NormalisedEigenvaluesSumToOne)
{
  const DegeneracyParams p = enabledParams();
  for (const auto & normals : {corridorNormals(60, 40), isotropicNormals(10), corridorNormals(97, 3)}) {
    const auto r = computePositionDegeneracy(informationOf(normals), static_cast<int>(normals.size()), p);
    EXPECT_NEAR(r.lambda_norm.sum(), 1.0, 1e-12);
  }
}

// --- the case this exists for ----------------------------------------------

// A corridor sees nothing along its own axis. That direction must be released while
// the two the walls and floor do constrain are kept at full strength.
TEST(DegeneracyPolicy, CorridorReleasesTheTravelDirectionOnly)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = corridorNormals(60, 40);
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);

  // y (the corridor axis) is unobservable; x and z are well above both thresholds
  EXPECT_NEAR(r.lambda_norm(0), 0.0, 1e-12);
  EXPECT_NEAR(r.weights(0), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(r.weights(1), 1.0);
  EXPECT_DOUBLE_EQ(r.weights(2), 1.0);

  // a plane facing along the corridor is dropped; the walls and floor are kept whole
  EXPECT_NEAR(pointWeight(r, Eigen::Vector3d::UnitY()), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(pointWeight(r, Eigen::Vector3d::UnitX()), 1.0);
  EXPECT_DOUBLE_EQ(pointWeight(r, Eigen::Vector3d::UnitZ()), 1.0);
}

// A scan swallowed by one near wall keeps only that wall's normal direction — the
// doorway/engulfment shape. Two of three directions go.
TEST(DegeneracyPolicy, SingleWallReleasesBothDirectionsAlongIt)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = corridorNormals(200, 0);  // every normal on +-x
  const auto r = computePositionDegeneracy(informationOf(normals), 200, p);
  ASSERT_TRUE(r.engaged);
  EXPECT_NEAR(r.weights(0), 0.0, 1e-12);
  EXPECT_NEAR(r.weights(1), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(r.weights(2), 1.0);
  EXPECT_NEAR(r.lambda_norm(2), 1.0, 1e-12);
}

// Partial degeneracy is graded, not binary: a direction between the thresholds keeps
// a proportional share of the update.
TEST(DegeneracyPolicy, PartialDegeneracyIsGraded)
{
  const DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  // 6 of 100 normals face y -> lambda_norm = 0.06, halfway up the ramp
  auto normals = corridorNormals(54, 40);
  for (int i = 0; i < 6; ++i) {
    normals.emplace_back(Eigen::Vector3d::UnitY());
  }
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);
  EXPECT_NEAR(r.lambda_norm(0), 0.06, 1e-12);
  EXPECT_NEAR(r.weights(0), 0.5, 1e-12);
  // a plane facing the released direction keeps sqrt(0.5) of its row and residual
  EXPECT_NEAR(pointWeight(r, Eigen::Vector3d::UnitY()), std::sqrt(0.5), 1e-12);
}

// --- the absolute test ------------------------------------------------------

// The same shape, the same share, opposite verdicts: a narrow field of view holds the
// weakest share structurally low, so the share alone would hold back a direction that
// hundreds of planes are looking straight at.
TEST(DegeneracyPolicy, AbsoluteStrengthRescuesAWeakShareInARichScan)
{
  const DegeneracyParams p = enabledParams(0.05, 0.15, 2.0, 20.0);
  const auto rich = wedgeNormals(117, 590);  // 1297 normals, ~9% face y -> lambda_y = 117
  const auto starved = wedgeNormals(3, 15);  // 33 normals, ~9% face y -> lambda_y = 3

  const auto r_rich = computePositionDegeneracy(informationOf(rich), static_cast<int>(rich.size()), p);
  const auto r_starved = computePositionDegeneracy(informationOf(starved), static_cast<int>(starved.size()), p);

  EXPECT_NEAR(r_rich.lambda_norm(0), r_starved.lambda_norm(0), 1e-3);  // same share
  EXPECT_NEAR(r_rich.lambda_abs(0), 117.0, 1e-9);
  EXPECT_NEAR(r_starved.lambda_abs(0), 3.0, 1e-9);

  EXPECT_FALSE(r_rich.engaged);  // plenty of planes face it; the narrow FOV is not degeneracy
  ASSERT_TRUE(r_starved.engaged);
  EXPECT_LT(r_starved.weights(0), 0.5);
}

// Turning the absolute test off must restore the share-only verdict exactly — the
// fallback has to be the old behaviour, not a third one.
TEST(DegeneracyPolicy, DisablingTheAbsoluteTestRestoresTheShareOnlyVerdict)
{
  const auto rich = wedgeNormals(117, 590);
  const auto r_both =
    computePositionDegeneracy(informationOf(rich), static_cast<int>(rich.size()), enabledParams(0.05, 0.15, 2.0, 20.0));
  const auto r_share =
    computePositionDegeneracy(informationOf(rich), static_cast<int>(rich.size()), shapeOnlyParams(0.05, 0.15));
  EXPECT_FALSE(r_both.engaged);
  ASSERT_TRUE(r_share.engaged);
  EXPECT_LT(r_share.weights(0), 1.0);
}

// The share is a property of the scene's shape; the absolute strength is not. Doubling
// the surviving points must leave the shares untouched and may change the verdict —
// that asymmetry is the whole reason both tests exist.
TEST(DegeneracyPolicy, SharesAreInvariantToPointCountButStrengthIsNot)
{
  const auto small = corridorNormals(30, 3);
  auto large = small;
  for (int i = 0; i < 19; ++i) {
    large.insert(large.end(), small.begin(), small.end());  // same scene, 20x the points
  }

  const DegeneracyParams shape = shapeOnlyParams();
  const auto s_small = computePositionDegeneracy(informationOf(small), static_cast<int>(small.size()), shape);
  const auto s_large = computePositionDegeneracy(informationOf(large), static_cast<int>(large.size()), shape);
  EXPECT_TRUE(s_small.lambda_norm.isApprox(s_large.lambda_norm, 1e-12));
  EXPECT_TRUE(s_small.weight_matrix.isApprox(s_large.weight_matrix, 1e-12));

  const DegeneracyParams both = enabledParams(0.02, 0.10, 2.0, 20.0);
  const auto b_small = computePositionDegeneracy(informationOf(small), static_cast<int>(small.size()), both);
  const auto b_large = computePositionDegeneracy(informationOf(large), static_cast<int>(large.size()), both);
  EXPECT_LT(b_small.weights(1), b_large.weights(1));  // the floor: 3 planes starved, 60 not
}

// --- the algebraic contract the design rests on -----------------------------

// The per-point weight is a weight on the MEASUREMENT: multiplying the row and the
// residual by s is exactly R -> R/s^2, so the solve must equal the one that inflates
// the measurement covariance instead.
TEST(DegeneracyPolicy, PointWeightIsExactlyAnInflatedMeasurementCovariance)
{
  const DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  auto normals = corridorNormals(54, 40);
  for (int i = 0; i < 6; ++i) {
    normals.emplace_back(Eigen::Vector3d::UnitY());
  }
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);

  const Eigen::MatrixXd h = jacobianFromNormals(normals);
  Eigen::VectorXd z(h.rows());
  for (int i = 0; i < h.rows(); ++i) {
    z(i) = 0.01 * std::sin(0.7 * i) + 0.02;
  }
  const double prior = 0.5;

  // (a) scale row and residual
  Eigen::MatrixXd hs = h;
  Eigen::VectorXd zs = z;
  for (int i = 0; i < h.rows(); ++i) {
    const double s = pointWeight(r, normals[static_cast<size_t>(i)]);
    hs.row(i) *= s;
    zs(i) *= s;
  }
  const Eigen::Vector3d dx_weighted = solve(hs, zs, prior);

  // (b) leave the model alone and inflate R by 1/s^2 per row
  Eigen::Matrix3d hth = Eigen::Matrix3d::Zero();
  Eigen::Vector3d htz = Eigen::Vector3d::Zero();
  for (int i = 0; i < h.rows(); ++i) {
    const double s = pointWeight(r, normals[static_cast<size_t>(i)]);
    const double inv_r = s * s;  // R_i = R / s^2
    hth.noalias() += inv_r * normals[static_cast<size_t>(i)] * normals[static_cast<size_t>(i)].transpose();
    htz.noalias() += inv_r * normals[static_cast<size_t>(i)] * z(i);
  }
  const Eigen::Vector3d dx_inflated = (hth + prior * Eigen::Matrix3d::Identity()).ldlt().solve(htz);

  EXPECT_TRUE(dx_weighted.isApprox(dx_inflated, 1e-12)) << dx_weighted.transpose() << " vs " << dx_inflated.transpose();
}

// Holding a direction back must SHRINK the correction along it, monotonically, all the
// way to zero. Scaling the position columns of H instead — leaving the residual alone
// — does not: it rewrites the plane normal rather than the weight, so explaining the
// same point-to-plane distance takes a LARGER step and the correction grows. This test
// pins the difference; it is the reason the policy hands back a per-point scalar and
// not a matrix to multiply into H.
TEST(DegeneracyPolicy, ColumnScalingAmplifiesWhatPointWeightingSuppresses)
{
  auto normals = corridorNormals(54, 40);
  for (int i = 0; i < 6; ++i) {
    normals.emplace_back(Eigen::Vector3d::UnitY());
  }
  const Eigen::MatrixXd h = jacobianFromNormals(normals);
  Eigen::VectorXd z = Eigen::VectorXd::Constant(h.rows(), 0.0);
  for (size_t i = 0; i < normals.size(); ++i) {
    z(static_cast<int>(i)) = 0.03 * normals[i].y();  // a residual that pulls along y
  }
  const double prior = 0.5;  // R * P^-1: the scan outweighs the prior 12:1 along y
  const double dx_off = solve(h, z, prior).y();

  double prev_weighted = std::abs(dx_off);
  bool column_scaling_ever_amplified = false;
  // sweep the weight on y from full trust down to blind
  for (const double good : {0.061, 0.07, 0.09, 0.12, 0.20, 0.40}) {
    const auto r = computePositionDegeneracy(informationOf(normals), 100, shapeOnlyParams(0.02, good));
    ASSERT_TRUE(r.engaged);
    ASSERT_LT(r.weights(0), 1.0);

    Eigen::MatrixXd hs = h;
    Eigen::VectorXd zs = z;
    for (size_t i = 0; i < normals.size(); ++i) {
      const double s = pointWeight(r, normals[i]);
      hs.row(static_cast<int>(i)) *= s;
      zs(static_cast<int>(i)) *= s;
    }
    const double dx_weighted = std::abs(solve(hs, zs, prior).y());

    // what scaling the position columns by sqrt(W) would have done
    Eigen::Vector3d root_w = r.weights.cwiseSqrt();
    const Eigen::Matrix3d scale = r.basis * root_w.asDiagonal() * r.basis.transpose();
    const double dx_columns = std::abs(solve(h * scale, z, prior).y());

    EXPECT_LE(dx_weighted, prev_weighted + 1e-12) << "weight " << r.weights(0) << " must not raise the correction";
    EXPECT_LT(dx_weighted, std::abs(dx_off));
    if (dx_columns > std::abs(dx_off) * 1.05) {
      column_scaling_ever_amplified = true;
    }
    prev_weighted = dx_weighted;
  }
  EXPECT_TRUE(column_scaling_ever_amplified) << "column scaling should overshoot the unweighted solve somewhere";

  // and at the blind end of the ramp the planes facing y drop out entirely
  const auto blind = computePositionDegeneracy(informationOf(normals), 100, shapeOnlyParams(0.07, 0.10));
  ASSERT_TRUE(blind.engaged);
  ASSERT_DOUBLE_EQ(blind.weights(0), 0.0);
  Eigen::MatrixXd hb = h;
  Eigen::VectorXd zb = z;
  for (size_t i = 0; i < normals.size(); ++i) {
    const double s = pointWeight(blind, normals[i]);
    hb.row(static_cast<int>(i)) *= s;
    zb(static_cast<int>(i)) *= s;
  }
  EXPECT_NEAR(solve(hb, zb, prior).y(), 0.0, 1e-12);
}

TEST(DegeneracyPolicy, WeightMatrixIsSymmetricPositiveSemiDefinite)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = corridorNormals(60, 40);
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);
  EXPECT_TRUE(r.weight_matrix.isApprox(r.weight_matrix.transpose(), 1e-12));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(r.weight_matrix);
  EXPECT_GE(es.eigenvalues().minCoeff(), -1e-12);
  EXPECT_LE(es.eigenvalues().maxCoeff(), 1.0 + 1e-12);
}

// Every per-point weight is a weight: in [0, 1], whatever the normal.
TEST(DegeneracyPolicy, PointWeightStaysInTheUnitInterval)
{
  const DegeneracyParams p = enabledParams();
  const auto normals = corridorNormals(60, 40);
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);
  for (int a = 0; a < 24; ++a) {
    const double t = a * M_PI / 12.0;
    for (int b = 0; b < 12; ++b) {
      const double u = b * M_PI / 12.0;
      const Eigen::Vector3d n(std::sin(u) * std::cos(t), std::sin(u) * std::sin(t), std::cos(u));
      const double s = pointWeight(r, n);
      EXPECT_GE(s, 0.0);
      EXPECT_LE(s, 1.0);
    }
  }
}

// Hard projection keeps only whole directions, which is the one setting that is also an
// exact constrained solve.
TEST(DegeneracyPolicy, HardProjectionYieldsBinaryWeightsAndAProjector)
{
  DegeneracyParams p = shapeOnlyParams(0.02, 0.10);
  p.hard_projection = true;
  auto normals = corridorNormals(54, 40);
  for (int i = 0; i < 6; ++i) {
    normals.emplace_back(Eigen::Vector3d::UnitY());  // share 0.06, mid-ramp
  }
  const auto r = computePositionDegeneracy(informationOf(normals), 100, p);
  ASSERT_TRUE(r.engaged);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(r.weights(i) == 0.0 || r.weights(i) == 1.0) << "weight " << i << " = " << r.weights(i);
  }
  EXPECT_TRUE((r.weight_matrix * r.weight_matrix).isApprox(r.weight_matrix, 1e-12));  // idempotent
}

// --- the Schur complement -----------------------------------------------------------

// One plane, every point at the SAME lever arm: a small rotation about the
// perpendicular axis slides the sensor along that normal by exactly as much as a
// translation does, so the scan cannot tell the two apart. H_tt counts all N normals at
// full strength and sees nothing wrong; the Schur complement sees the direction is gone.
TEST(SchurComplement, ConstantLeverArmMakesTranslationIndistinguishableFromRotation)
{
  const int n = 200;
  std::vector<Eigen::Vector3d> normals(n, Eigen::Vector3d::UnitX());
  std::vector<Eigen::Vector3d> points(n, Eigen::Vector3d(0.0, 0.0, 4.0));
  const JacobianBlocks b = blocksOf(normals, points);

  EXPECT_NEAR(b.tt(0, 0), double(n), 1e-9);  // H_tt: fully observed, it thinks
  const Eigen::Matrix3d m = schurPositionInformation(b.tt, b.tr, b.rr, 1e-6);
  EXPECT_NEAR(m(0, 0), 0.0, 1e-3) << "rotation explains this direction away entirely";
}

// Vary the lever arm and the confusion is only partial: by Cauchy-Schwarz the rotation
// can absorb (sum z)^2 / sum z^2 <= N of the N normals' worth of information.
TEST(SchurComplement, SpreadLeverArmsLeaveInformationBehind)
{
  std::vector<Eigen::Vector3d> normals, points;
  for (int i = 0; i < 100; ++i) {
    normals.emplace_back(Eigen::Vector3d::UnitX());
    points.emplace_back(0.0, 0.0, (i % 2 == 0) ? 1.0 : 4.0);
  }
  const JacobianBlocks b = blocksOf(normals, points);
  const Eigen::Matrix3d m = schurPositionInformation(b.tt, b.tr, b.rr, 1e-6);

  const double sum_z = 50 * 1.0 + 50 * 4.0;
  const double sum_zz = 50 * 1.0 + 50 * 16.0;
  EXPECT_NEAR(m(0, 0), 100.0 - sum_z * sum_z / sum_zz, 1e-3);
  EXPECT_GT(m(0, 0), 0.0);
  EXPECT_LT(m(0, 0), 100.0);
}

// Points at the sensor origin have no lever arm, so rotation is unobservable and can
// explain nothing away: the complement must be the translation block itself.
TEST(SchurComplement, NoRotationInformationLeavesTheTranslationBlockUntouched)
{
  const int n = 60;
  std::vector<Eigen::Vector3d> normals, points(n, Eigen::Vector3d::Zero());
  for (int i = 0; i < n; ++i) {
    normals.emplace_back(isotropicNormals(1)[i % 3]);
  }
  const JacobianBlocks b = blocksOf(normals, points);
  EXPECT_TRUE(b.rr.isZero(1e-12));
  EXPECT_TRUE(schurPositionInformation(b.tt, b.tr, b.rr, 1e-6).isApprox(b.tt, 1e-12));
}

// Never larger than H_tt, and still a valid information matrix.
TEST(SchurComplement, IsPositiveSemiDefiniteAndNoLargerThanTheTranslationBlock)
{
  std::vector<Eigen::Vector3d> normals, points;
  for (int i = 0; i < 120; ++i) {
    const double t = 0.11 * i;
    normals.emplace_back(Eigen::Vector3d(std::cos(t), std::sin(t), std::cos(2 * t)).normalized());
    points.emplace_back(3.0 * std::sin(t), 2.0 * std::cos(t), 1.0 + 0.5 * std::sin(3 * t));
  }
  const JacobianBlocks b = blocksOf(normals, points);
  const Eigen::Matrix3d m = schurPositionInformation(b.tt, b.tr, b.rr, 1e-6);

  EXPECT_TRUE(m.isApprox(m.transpose(), 1e-9));
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(m).eigenvalues().minCoeff(), -1e-9);
  // H_tt - M_eff is the information rotation absorbs: also PSD
  EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(b.tt - m).eigenvalues().minCoeff(), -1e-9);
}

TEST(SchurComplement, NonFiniteBlocksFallBackToTheTranslationBlock)
{
  Eigen::Matrix3d tt = Eigen::Matrix3d::Identity() * 10.0;
  Eigen::Matrix3d tr = Eigen::Matrix3d::Ones();
  Eigen::Matrix3d rr = Eigen::Matrix3d::Identity() * 5.0;
  rr(1, 1) = std::nan("");
  EXPECT_TRUE(schurPositionInformation(tt, tr, rr, 1e-6).isApprox(tt, 1e-12));
}

// The flag decides which matrix is judged, and off must reproduce the old verdict bit
// for bit — the fallback has to be the previous behaviour, not a third one.
TEST(SchurComplement, TheFlagDecidesWhichSpectrumIsJudged)
{
  const int n = 200;
  std::vector<Eigen::Vector3d> normals(n, Eigen::Vector3d::UnitX());
  std::vector<Eigen::Vector3d> points(n, Eigen::Vector3d(0.0, 0.0, 4.0));
  const JacobianBlocks b = blocksOf(normals, points);

  DegeneracyParams off = enabledParams(0.05, 0.30, 3.0, 30.0);
  DegeneracyParams on = off;
  on.schur_en = true;

  const auto r_off = computePositionDegeneracy(b.tt, b.tr, b.rr, n, off);
  const auto r_on = computePositionDegeneracy(b.tt, b.tr, b.rr, n, on);
  const auto r_direct = computePositionDegeneracy(b.tt, n, off);

  EXPECT_TRUE(r_off.lambda_abs.isApprox(r_direct.lambda_abs, 1e-12));
  EXPECT_TRUE(r_off.weights.isApprox(r_direct.weights, 1e-12));

  // H_tt says the wall direction carries all N normals; the Schur complement says none
  EXPECT_NEAR(r_off.lambda_abs(2), double(n), 1e-6);
  EXPECT_NEAR(r_on.lambda_abs(2), 0.0, 1e-3);
  EXPECT_LE(r_on.lambda_norm.sum(), 1.0 + 1e-12) << "shares sum to at most 1 under Schur";
  EXPECT_NEAR(r_off.lambda_norm.sum(), 1.0, 1e-12) << "and to exactly 1 without it";
}

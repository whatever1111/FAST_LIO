#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <random>
#include <sys/resource.h>

#include "IMU_Processing.hpp"
#include "frontend_lidar_joint_adapter.hpp"
#include "plane_fit.hpp"

namespace fastlio::lidar_joint
{
namespace
{
namespace js = joint_source;

// Frozen before execution: seeded dense fixtures have <=200 rows, well-scaled
// SPD remainders/prior. 2^22 eps covers their independent dense factorizations
// and sufficient-statistic eigendecomposition, without data-dependent fitting.
constexpr double kOracleTolerance = 4194304.0 * std::numeric_limits<double>::epsilon();
constexpr double kHalf = 0.5, kQuarter = 0.25;
constexpr Eigen::Index kVelocityIndex = 12, kGyroIndex = 15, kAccelIndex = 18, kGravityIndex = 21;

void expectMatrix(const Eigen::MatrixXd & actual, const Eigen::MatrixXd & expected)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  ASSERT_TRUE(actual.allFinite());
  EXPECT_LE((actual - expected).norm(), kOracleTolerance * std::max(1.0, expected.norm()));
}

Eigen::MatrixXd dense(std::mt19937 & generator, Eigen::Index rows, Eigen::Index columns)
{
  std::uniform_real_distribution<double> distribution(-kHalf, kHalf);
  Eigen::MatrixXd result(rows, columns);
  for (Eigen::Index column = 0; column < columns; ++column)
    for (Eigen::Index row = 0; row < rows; ++row)
      result(row, column) = distribution(generator);
  return result;
}

Eigen::MatrixXd mirror(const Eigen::MatrixXd & matrix)
{
  return matrix.selfadjointView<Eigen::Lower>();
}

js::JointMoments moments(Eigen::Index active)
{
  js::JointMoments result;
  result.active = {{}, {}, "synthetic", "Euclidean", "single-frame"};
  for (Eigen::Index axis = 0; axis < active; ++axis) {
    result.active.components.push_back("x/" + std::to_string(axis));
    result.active.units.push_back("m");
  }
  result.knowledge = js::MomentKnowledge::kKnown;
  result.evidence = "independent analytic oracle prior";
  result.mean = Eigen::VectorXd::Zero(active);
  result.covariance = Eigen::MatrixXd::Identity(active, active);
  return result;
}

js::LinearFactor factor(const js::JointMoments & prior, const CompressedFrame & compressed)
{
  js::LinearFactor result;
  result.domain = js::ModelDomain::kLinearFixedModel;
  result.identity = {"test-frame", "compressed fixed model", {"test-raw"}, {}, "test-eps", "independent unit noise"};
  result.model = "analytic-test";
  result.linearization = "fixed-test";
  result.active = prior.active;
  result.rowUnits = std::vector<std::string>(static_cast<std::size_t>(compressed.h.rows()), "dimensionless");
  result.remainderKnowledge = js::MomentKnowledge::kKnown;
  result.independentRemainder = true;
  result.h = compressed.h;
  result.a.resize(compressed.h.rows(), 0);
  result.observation = compressed.observation;
  result.remainderMean = Eigen::VectorXd::Zero(compressed.h.rows());
  result.remainderVariance = Eigen::VectorXd::Ones(compressed.h.rows());
  return result;
}

void checkK2(const js::JointMoments & prior,
             const CompressedFrame & compressed,
             const Eigen::MatrixXd & expectedOptimal,
             const Eigen::MatrixXd & expectedGain,
             const Eigen::VectorXd & expectedMean,
             const Eigen::VectorXd & expectedGainMean)
{
  js::JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(prior), js::Status::kOk);
  const auto measurement = factor(prior, compressed);
  const auto proposal = filter.propose({measurement});
  ASSERT_EQ(proposal.status(), js::Status::kOk);
  ASSERT_NE(proposal.posterior(), nullptr);
  expectMatrix(proposal.posterior()->covariance, expectedOptimal);
  expectMatrix(proposal.posterior()->mean, expectedMean);
  const auto applied = filter.appliedGain({measurement}, compressed.gain);
  ASSERT_EQ(applied.status, js::Status::kOk);
  expectMatrix(applied.covariance, expectedGain);
  expectMatrix(applied.mean, expectedGainMean);
  EXPECT_TRUE((filter.moments()->covariance.array() == prior.covariance.array()).all());
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

Row crossRow()
{
  Row row;
  row.support = {{{-1.0F, 0.0F, 1.0F, 0.0F},
                  {1.0F, 0.0F, 1.0F, 0.0F},
                  {0.0F, -1.0F, 1.0F, 0.0F},
                  {0.0F, 1.0F, 1.0F, 0.0F},
                  {0.0F, 0.0F, 1.0F, 0.0F}}};
  row.plane = Plane{Eigen::Vector3d::UnitZ(),
                    -1.0,
                    Eigen::Vector3d(0.0, 2.0, 2.0),
                    Eigen::Vector3d::UnitX(),
                    Eigen::Vector3d::UnitY(),
                    kSupportCount};
  row.point = Eigen::Vector3d(0.0, 0.0, 1.0);
  return row;
}

Eigen::Vector4d fitted(const std::array<SupportPoint, kSupportCount> & support,
                       fast_lio::PlaneFitEigen * eigen = nullptr)
{
  Eigen::Vector4d result = Eigen::Vector4d::Zero();
  EXPECT_TRUE(fast_lio::fitPlane<5>(support, 1.0, result, eigen));
  return result;
}

Eigen::Matrix<double, 1, kProductionColumns> productionRow(const Row & row, const State & state, bool extrinsics)
{
  const Eigen::Matrix3d r = state.rotation.toRotationMatrix(), e = state.extrinsicRotation.toRotationMatrix();
  const Eigen::Vector3d n = r.transpose() * row.plane->normal;
  Eigen::Matrix<double, 1, kProductionColumns> result = Eigen::Matrix<double, 1, kProductionColumns>::Zero();
  result.head<3>() = row.plane->normal.transpose();
  result.segment<3>(3) = (e * row.point + state.extrinsicTranslation).cross(n).transpose();
  if (extrinsics) {
    result.segment<3>(6) = row.point.cross(e.transpose() * n).transpose();
    result.tail<3>() = n.transpose();
  }
  return result;
}

void closeFixture(FrameSnapshot & snapshot)
{
  const auto count = static_cast<Eigen::Index>(snapshot.rows.size());
  snapshot.preFloorH.resize(count, kProductionColumns);
  for (Eigen::Index i = 0; i < count; ++i) {
    const auto & row = snapshot.rows[static_cast<std::size_t>(i)];
    snapshot.preFloorH.row(i) = row.weight * productionRow(row, *snapshot.linearization, snapshot.extrinsicEstimated);
    if (snapshot.holdActive)
      snapshot.preFloorH.block<1, 2>(i, 3).setZero();
  }
  snapshot.productionH = snapshot.preFloorH;
  snapshot.preFloorObservation = snapshot.productionObservation = Eigen::VectorXd::Zero(count);
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(count, kStateDof);
  h.leftCols(kProductionColumns) = snapshot.productionH;
  const Matrix23 prior = *snapshot.priorCovariance;
  const Matrix23 information = h.transpose() * h + kProductionVariance * prior.inverse();
  Eigen::MatrixXd gain = information.ldlt().solve(h.transpose());
  if (snapshot.considerKeep) {
    Matrix23 keep = Matrix23::Identity();
    keep.block<3, 3>(3, 3) = *snapshot.considerKeep;
    gain = (keep * gain).eval();
    const Matrix23 error = Matrix23::Identity() - gain * h;
    snapshot.posteriorCovariance =
      mirror(error * prior * error.transpose() + kProductionVariance * gain * gain.transpose());
  } else {
    snapshot.posteriorCovariance = prior - gain * h * prior;
  }
  snapshot.posterior = snapshot.linearization;
  snapshot.diagnostics = {true, true, 2, snapshot.rows.size(), Vector23::Zero()};
}

FrameSnapshot frame(std::size_t count = 36, std::size_t reuse = 1)
{
  constexpr int kScanEndNs = 1000000000;
  constexpr double kFrameDuration = 0.1, kPointTimeStep = 0.001;
  constexpr double kQueryOffset = 0.1, kQueryStride = 0.01, kQueryLateral = 0.2;
  FrameSnapshot result;
  result.identity = FrameIdentity{1, kScanEndNs, 2, "odom"};
  result.prior = result.linearization = result.posterior = State{};
  result.priorCovariance = Matrix23::Identity() * kHalf;
  result.scanDuration = kFrameDuration;
  result.angularVelocity = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t group = i / reuse;
    const double angle = static_cast<double>(group % 19) * 0.1;
    const Eigen::Vector3d normal = Eigen::Vector3d(std::cos(angle), std::sin(angle), kHalf).normalized();
    const Eigen::Vector3d u = normal.unitOrthogonal(), v = normal.cross(u);
    const Eigen::Vector3d center(
      1.0 + static_cast<double>(group % 11) * 0.1, kQuarter + static_cast<double>(group % 7) * 0.1, 2.0);
    Row row;
    row.queryOrdinal = i;
    row.relativeTime = -static_cast<double>(i % 100) * kPointTimeStep;
    row.point = center + u * (kQueryOffset + static_cast<double>(i % reuse) * kQueryStride) + v * kQueryLateral;
    row.beamDirection = row.point.normalized();  // Static query fixture, no deskew translation.
    const std::array<Eigen::Vector2d, kSupportCount> grid{Eigen::Vector2d(-1.0, 0.0),
                                                          Eigen::Vector2d(1.0, 0.0),
                                                          Eigen::Vector2d(0.0, -1.0),
                                                          Eigen::Vector2d(0.0, 1.0),
                                                          Eigen::Vector2d::Zero()};
    for (std::size_t a = 0; a < kSupportCount; ++a) {
      const double scatter = a + 1 == kSupportCount ? 0.016 : -0.004;
      const Eigen::Vector3d point = center + u * grid[a][0] + v * grid[a][1] + normal * scatter;
      row.support[a] = {static_cast<float>(point.x()),
                        static_cast<float>(point.y()),
                        static_cast<float>(point.z()),
                        static_cast<float>(group)};
    }
    fast_lio::PlaneFitEigen spectrum;
    const Eigen::Vector4d fit = fitted(row.support, &spectrum);
    row.plane = Plane{fit.head<3>(),
                      fit[3],
                      Eigen::Vector3d(spectrum.lambda0, spectrum.lambda1, spectrum.lambda2),
                      spectrum.e1,
                      spectrum.e2,
                      kSupportCount};
    result.rows.push_back(row);
  }
  closeFixture(result);
  return result;
}

TEST(FrontendLidarJointAdapter, O1HatWeightsMatchAffineAnalyticPlane)
{
  const Row row = crossRow();
  std::array<double, kSupportCount> weights{};
  ASSERT_EQ(hatWeights(row, Eigen::Vector3d(0.0, 0.0, 1.0), weights), FrameStatus::kOkFrameConditional);
  for (const double weight : weights)
    EXPECT_DOUBLE_EQ(weight, 0.2);
  ASSERT_EQ(hatWeights(row, Eigen::Vector3d(kHalf, kQuarter, 1.0), weights), FrameStatus::kOkFrameConditional);
  constexpr std::array<double, kSupportCount> kExpected{-0.05, 0.45, 0.075, 0.325, 0.2};
  Eigen::Vector3d affine = Eigen::Vector3d::Zero();
  for (std::size_t a = 0; a < kSupportCount; ++a) {
    EXPECT_NEAR(weights[a], kExpected[a], kOracleTolerance);
    affine += weights[a] * Eigen::Vector3d(row.support[a].x, row.support[a].y, row.support[a].z);
  }
  expectMatrix(affine, Eigen::Vector3d(kHalf, kQuarter, 1.0));
}

TEST(FrontendLidarJointAdapter, O2TlsDerivativeMatchesProductionPlaneFitFiniteDifference)
{
  const auto row = crossRow();
  const Eigen::Vector4d baseline = fitted(row.support);
  constexpr double kStep = 1.0 / 4096.0;
  constexpr double kDifferenceBound = 8.0 * kStep * kStep + 256.0 * std::numeric_limits<double>::epsilon() / kStep;
  const Eigen::Vector3d query(kHalf, kQuarter, 1.0);
  std::array<double, kSupportCount> weights{};
  ASSERT_EQ(hatWeights(row, query, weights), FrameStatus::kOkFrameConditional);
  for (std::size_t a = 0; a < kSupportCount; ++a) {
    auto positive = row.support, negative = row.support;
    positive[a].z += static_cast<float>(kStep);
    negative[a].z -= static_cast<float>(kStep);
    auto high = fitted(positive), low = fitted(negative);
    if (high.head<3>().dot(baseline.head<3>()) < 0.0)
      high = -high;
    if (low.head<3>().dot(baseline.head<3>()) < 0.0)
      low = -low;
    const Eigen::Vector3d derivative = (high.head<3>() - low.head<3>()) / (2.0 * kStep);
    const Eigen::Vector3d expected(row.support[a].x * kHalf, row.support[a].y * kHalf, 0.0);
    EXPECT_LE((derivative - expected).norm(), kDifferenceBound);
    const double residualDerivative =
      ((high.head<3>().dot(query) + high[3]) - (low.head<3>().dot(query) + low[3])) / (2.0 * kStep);
    EXPECT_NEAR(residualDerivative, weights[a], kDifferenceBound);
  }
}

TEST(FrontendLidarJointAdapter, O3SharedTwoRowsMatchRationalOracle)
{
  constexpr double kThree = 3.0, kFour = 4.0;
  Eigen::MatrixXd b(2, 3);
  b << 1.0, 1.0, 0.0, 1.0, 0.0, 1.0;
  const Eigen::MatrixXd h = Eigen::MatrixXd::Ones(2, 1), gain = Eigen::MatrixXd::Constant(1, 2, kQuarter);
  CompressedFrame compressed;
  Counts counts;
  ASSERT_EQ(compressRows(
              h, Eigen::VectorXd::Ones(2), gain, b.sparseView(), Eigen::VectorXd::Ones(2), 1.0, {}, compressed, counts),
            FrameStatus::kOkFrameConditional);
  EXPECT_NEAR(compressed.h.squaredNorm(), kHalf, kOracleTolerance);
  EXPECT_NEAR(compressed.gain.squaredNorm(), kHalf, kOracleTolerance);
  checkK2(moments(1),
          compressed,
          Eigen::MatrixXd::Constant(1, 1, 2.0 / kThree),
          Eigen::MatrixXd::Constant(1, 1, kThree / kFour),
          Eigen::VectorXd::Constant(1, 1.0 / kThree),
          Eigen::VectorXd::Constant(1, kHalf));
}

TEST(FrontendLidarJointAdapter, O4RepeatedRowsSaturateWithoutInformationCeiling)
{
  for (const Eigen::Index count : {1, 10, 1000}) {
    const Eigen::MatrixXd h = Eigen::MatrixXd::Ones(count, 1);
    const Eigen::MatrixXd gain = Eigen::MatrixXd::Constant(1, count, 1.0 / static_cast<double>(count + 1));
    const SparseMatrix b = h.sparseView();
    CompressedFrame compressed;
    Counts counts;
    ASSERT_EQ(compressRows(h,
                           Eigen::VectorXd::Ones(count),
                           gain,
                           b,
                           Eigen::VectorXd::Constant(count, kHalf),
                           kQuarter,
                           {},
                           compressed,
                           counts),
              FrameStatus::kOkFrameConditional);
    const double information = static_cast<double>(count) / (kHalf + static_cast<double>(count) * kQuarter);
    EXPECT_NEAR(compressed.h.squaredNorm(), information, kOracleTolerance);
    EXPECT_LT(information, 4.0);
    const double sumGain = static_cast<double>(count) / static_cast<double>(count + 1);
    const double pg =
      (1.0 - sumGain) * (1.0 - sumGain) + sumGain * sumGain * (kQuarter + kHalf / static_cast<double>(count));
    checkK2(moments(1),
            compressed,
            Eigen::MatrixXd::Constant(1, 1, 1.0 / (1.0 + information)),
            Eigen::MatrixXd::Constant(1, 1, pg),
            Eigen::VectorXd::Constant(1, information / (1.0 + information)),
            Eigen::VectorXd::Constant(1, sumGain));
  }
}

TEST(FrontendLidarJointAdapter, O5SparseEliminationMatchesIndependentFullJointPosterior)
{
  constexpr unsigned kSeed = 20261110U;
  constexpr Eigen::Index kLoads = 15, kRowStride = 7, kLoadStride = 11;
  constexpr double kLoadDenominator = 32.0;
  std::mt19937 generator(kSeed);  // NOLINT(bugprone-random-generator-seed): frozen reproducible oracle.
  constexpr Eigen::Index kRows = 200, kActive = 4, kSources = 180;
  auto prior = moments(kActive);
  prior.mean = dense(generator, kActive, 1);
  const Eigen::MatrixXd h = dense(generator, kRows, kActive), observation = dense(generator, kRows, 1);
  const Eigen::MatrixXd gain = dense(generator, kActive, kRows) / static_cast<double>(kRows);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(kRows, kSources);
  for (Eigen::Index row = 0; row < kRows; ++row)
    for (Eigen::Index load = 0; load < kLoads; ++load)
      b(row, (row * kRowStride + load * kLoadStride) % kSources) = static_cast<double>(load + 1) / kLoadDenominator;
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(kRows, kHalf);
  Eigen::MatrixXd jointH(kRows, kActive + kSources);
  jointH << h, b;
  Eigen::MatrixXd jointPrior = Eigen::MatrixXd::Identity(kActive + kSources, kActive + kSources);
  jointPrior.bottomRightCorner(kSources, kSources) *= kQuarter;
  const Eigen::MatrixXd jointPrecision =
    jointPrior.inverse() + jointH.transpose() * diagonal.cwiseInverse().asDiagonal() * jointH;
  const Eigen::MatrixXd jointPosterior =
    jointPrecision.ldlt().solve(Eigen::MatrixXd::Identity(kActive + kSources, kActive + kSources));
  Eigen::VectorXd jointMean = Eigen::VectorXd::Zero(kActive + kSources);
  jointMean.head(kActive) = prior.mean;
  const Eigen::VectorXd updated =
    jointPosterior *
    (jointPrior.inverse() * jointMean + jointH.transpose() * diagonal.cwiseInverse().asDiagonal() * observation);
  const Eigen::MatrixXd r = kQuarter * b * b.transpose() + diagonal.asDiagonal().toDenseMatrix();
  const Eigen::MatrixXd error = Eigen::MatrixXd::Identity(kActive, kActive) - gain * h;
  const Eigen::MatrixXd pg = error * prior.covariance * error.transpose() + gain * r * gain.transpose();
  CompressedFrame compressed;
  Counts counts;
  ASSERT_EQ(compressRows(h, observation, gain, b.sparseView(), diagonal, kQuarter, {}, compressed, counts),
            FrameStatus::kOkFrameConditional);
  checkK2(prior,
          compressed,
          mirror(jointPosterior.topLeftCorner(kActive, kActive)),
          mirror(pg),
          updated.head(kActive),
          prior.mean + gain * (observation - h * prior.mean));
  EXPECT_LE(*counts.compressedRows, 2 * kActive);
  EXPECT_LT(*counts.accountedPeakBytes, Budgets{}.bytes);
}

TEST(FrontendLidarJointAdapter, O7CompressionPreservesRankSixAndTwelveArbitraryGains)
{
  constexpr unsigned kSeed = 71U;
  constexpr Eigen::Index kSources = 90, kLoadStride = 7;
  constexpr double kLoad = 0.1;
  std::mt19937 generator(kSeed);  // NOLINT(bugprone-random-generator-seed): frozen reproducible oracle.
  constexpr Eigen::Index kRows = 200;
  Eigen::MatrixXd h = dense(generator, kRows, kStateDof);
  h.col(4) = h.col(3);  // Hold-like rank deficiency is kept, never jittered.
  const Eigen::VectorXd observation = dense(generator, kRows, 1);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(kRows, kSources);
  for (Eigen::Index row = 0; row < kRows; ++row)
    for (Eigen::Index load = 0; load < 9; ++load)
      b(row, (row + load * kLoadStride) % b.cols()) = kLoad;
  const Eigen::VectorXd diagonal = Eigen::VectorXd::Constant(kRows, 0.75);
  const Eigen::MatrixXd r = kQuarter * b * b.transpose() + diagonal.asDiagonal().toDenseMatrix();
  const Eigen::MatrixXd information = h.transpose() * r.ldlt().solve(h);
  const Matrix23 p = mirror((Matrix23::Identity() + information).ldlt().solve(Matrix23::Identity()));
  for (const Eigen::Index rank : {6, 12}) {
    const Eigen::MatrixXd gain =
      dense(generator, kStateDof, rank) * dense(generator, rank, kRows) / static_cast<double>(kRows);
    CompressedFrame compressed;
    Counts counts;
    ASSERT_EQ(compressRows(h, observation, gain, b.sparseView(), diagonal, kQuarter, {}, compressed, counts),
              FrameStatus::kOkFrameConditional);
    EXPECT_EQ(*counts.rankGain, rank);
    EXPECT_LE(*counts.compressedRows, kStateDof + rank);
    expectMatrix(compressed.gain * compressed.h, gain * h);
    expectMatrix(compressed.gain * compressed.observation, gain * observation);
    expectMatrix(compressed.gain * compressed.gain.transpose(), gain * r * gain.transpose());
    const Matrix23 error = Matrix23::Identity() - gain * h;
    checkK2(moments(kStateDof),
            compressed,
            p,
            mirror(error * error.transpose() + gain * r * gain.transpose()),
            p * h.transpose() * r.ldlt().solve(observation),
            gain * observation);
    const double expectedChi = observation.dot(r.ldlt().solve(observation)) - compressed.observation.squaredNorm();
    EXPECT_NEAR(compressed.chi2Perpendicular, expectedChi, kOracleTolerance * std::max(1.0, expectedChi));
  }
}

TEST(FrontendLidarJointAdapter, O9ContractiveRowTransformMatchesCovarianceDifference)
{
  Eigen::Matrix<double, 3, 2> h;
  h << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
  Eigen::Matrix3d l = Eigen::Vector3d(1.0, kHalf, kQuarter).asDiagonal();
  const Eigen::Matrix<double, 3, 2> transformed = l * h;
  const Eigen::Matrix<double, 2, 3> k =
    (Eigen::Matrix2d::Identity() + transformed.transpose() * transformed).ldlt().solve(transformed.transpose());
  const Eigen::Matrix<double, 2, 3> actual = k * l;
  const Eigen::Matrix2d produced = Eigen::Matrix2d::Identity() - k * transformed;
  const Eigen::Matrix2d error = Eigen::Matrix2d::Identity() - actual * h;
  const Eigen::Matrix2d honest = error * error.transpose() + actual * actual.transpose();
  expectMatrix(produced - honest, k * (Eigen::Matrix3d::Identity() - l * l.transpose()) * k.transpose());
  EXPECT_GT(Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(produced - honest).eigenvalues()[0], 0.0);
  auto snapshot = frame();
  snapshot.floorMode = FloorMode::kApply;
  const auto result = observeFrame(snapshot);
  EXPECT_EQ(result.status, FrameStatus::kUnsupportedFloorApply);
  EXPECT_FALSE(result.readings);
}

TEST(FrontendLidarJointAdapter, O11SeededInformationBoundsAndOptimalCovarianceOrdering)
{
  constexpr unsigned kSeed = 11011U;
  constexpr int kTrials = 20;
  std::mt19937 generator(kSeed);  // NOLINT(bugprone-random-generator-seed): frozen reproducible oracle.
  for (int trial = 0; trial < kTrials; ++trial) {
    constexpr Eigen::Index kRows = 12, kActive = 4;
    const Eigen::MatrixXd h = dense(generator, kRows, kActive), b = dense(generator, kRows, 6);
    const Eigen::MatrixXd gain = dense(generator, kActive, kRows) / static_cast<double>(kRows);
    for (const double rho : {kQuarter, 1.0, 4.0}) {
      const Eigen::MatrixXd r = b * b.transpose() + rho * Eigen::MatrixXd::Identity(kRows, kRows);
      const Eigen::MatrixXd information = h.transpose() * r.ldlt().solve(h);
      CompressedFrame compressed;
      Counts counts;
      ASSERT_EQ(compressRows(h,
                             Eigen::VectorXd::Zero(kRows),
                             gain,
                             b.sparseView(),
                             Eigen::VectorXd::Constant(kRows, rho),
                             1.0,
                             {},
                             compressed,
                             counts),
                FrameStatus::kOkFrameConditional);
      expectMatrix(compressed.h.transpose() * compressed.h, information);
      for (Eigen::Index axis = 0; axis < kActive; ++axis) {
        const double gamma = rho * information(axis, axis) / h.col(axis).squaredNorm();
        EXPECT_GT(gamma, 0.0);
        EXPECT_LE(gamma, 1.0 + kOracleTolerance);
      }
      const Eigen::MatrixXd p = (Eigen::MatrixXd::Identity(kActive, kActive) + information).inverse();
      const Eigen::MatrixXd error = Eigen::MatrixXd::Identity(kActive, kActive) - gain * h;
      const Eigen::MatrixXd pg = error * error.transpose() + gain * r * gain.transpose();
      EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(mirror(pg - p)).eigenvalues()[0], -kOracleTolerance);
      checkK2(moments(kActive),
              compressed,
              mirror(p),
              mirror(pg),
              Eigen::VectorXd::Zero(kActive),
              Eigen::VectorXd::Zero(kActive));
    }
  }
}

State adapterState(const state_ikfom & state)
{
  State result;
  result.position = state.pos;
  result.rotation = state.rot;
  result.extrinsicRotation = state.offset_R_L_I;
  result.extrinsicTranslation = state.offset_T_L_I;
  result.velocity = state.vel;
  result.gyroBias = state.bg;
  result.accelBias = state.ba;
  result.gravity = state.grav.get_vect();
  return result;
}

struct DeskewResult
{
  state_ikfom state;
  Eigen::Vector3d point, worldPoint;
};

MeasureGroup imuScan(double begin, bool initialization)
{
  constexpr double kFrameDuration = 0.1, kSamplePeriod = 0.005, kNanosecondsPerSecond = 1e9;
  constexpr float kPointX = 1.25F, kPointY = 2.5F, kPointZ = 3.75F;
  constexpr int kInitializationSamples = 19, kMotionSamples = 20;
  MeasureGroup result;
  result.lidar_beg_time = begin;
  result.lidar_end_time = begin + kFrameDuration;
  result.lidar = std::make_shared<PointCloudXYZI>();
  for (const float time : {50.0F, 100.0F}) {
    PointType point{};
    point.x = kPointX;
    point.y = kPointY;
    point.z = kPointZ;
    point.curvature = time;
    result.lidar->push_back(point);
  }
  for (int i = 0; i <= (initialization ? kInitializationSamples : kMotionSamples); ++i) {
    auto sample = std::make_shared<sensor_msgs::msg::Imu>();
    sample->header.stamp = rclcpp::Time(static_cast<std::int64_t>(
      std::llround((begin + static_cast<double>(i) * kSamplePeriod) * kNanosecondsPerSecond)));
    sample->linear_acceleration.z = initialization ? 9.81 : 0.0;
    result.imu.push_back(sample);
  }
  return result;
}

DeskewResult actualDeskew(const Vector23 & perturbation)
{
  constexpr int kProcessNoiseDof = 12, kIterations = 5;
  constexpr double kInputVariance = 0.1, kEpsilon = 0.001;
  using Filter = esekfom::esekf<state_ikfom, kProcessNoiseDof, input_ikfom>;
  ImuProcess process;
  Filter filter;
  process.coverage_params = {2.0, kInputVariance};
  process.set_acc_cov(V3D::Constant(kInputVariance));
  process.set_gyr_cov(V3D::Constant(kInputVariance));
  std::array<double, kStateDof> epsilon{};
  epsilon.fill(kEpsilon);
  filter.init_dyn_share(
    get_f, df_dx, df_dw, [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, kIterations, epsilon.data());
  PointCloudXYZI::Ptr output = std::make_shared<PointCloudXYZI>();
  EXPECT_FALSE(process.Process(imuScan(10.0, true), filter, output));
  state_ikfom state;
  state.grav = S2(Eigen::Vector3d(0.0, 0.0, -kGravityMagnitude));
  state.boxplus(perturbation);
  filter.change_x(state);
  Filter::cov covariance = Filter::cov::Identity() * kEpsilon;
  filter.change_P(covariance);
  EXPECT_TRUE(process.Process(imuScan(10.1, false), filter, output));
  EXPECT_EQ(output->size(), 2U);
  const auto end = filter.get_x();
  const auto & point = output->front();
  const Eigen::Vector3d body(point.x, point.y, point.z);
  const Eigen::Vector3d world = end.rot * (end.offset_R_L_I * body + end.offset_T_L_I) + end.pos;
  return {end, body, world};
}

TEST(FrontendLidarJointAdapter, O6aRigidDeskewColumnsMatchIndependentMotionFiniteDifference)
{
  constexpr double kThirdCoordinate = 3.0, kPointX = 1.25, kPointY = 2.5, kPointZ = 3.75, kRelativeTime = 0.075;
  State state;
  state.rotation = Eigen::AngleAxisd(kQuarter, Eigen::Vector3d(1.0, 2.0, kThirdCoordinate).normalized());
  auto row = crossRow();
  row.point = Eigen::Vector3d(kPointX, kPointY, kPointZ);
  row.relativeTime = -kRelativeTime;
  row.plane->normal = Eigen::Vector3d(2.0, -1.0, kThirdCoordinate).normalized();
  const Eigen::Vector3d omega(kQuarter, -kHalf, 0.75);
  StateRowJacobian analytic;
  ASSERT_EQ(geometricRow(row, state, true, omega, analytic), FrameStatus::kOkFrameConditional);
  constexpr double kStep = 1e-5;
  constexpr double kCentralBound = 128.0 * std::numeric_limits<double>::epsilon() / kStep + 64.0 * kStep * kStep;
  const auto motion = [&](Eigen::Index index, double delta) {
    const double time = *row.relativeTime;
    Eigen::Vector3d point = row.point;
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    if (index >= kVelocityIndex && index < kGyroIndex)
      displacement[index - kVelocityIndex] = time * delta;
    if (index >= kGyroIndex && index < kAccelIndex) {
      Eigen::Vector3d bias = Eigen::Vector3d::Zero();
      bias[index - kGyroIndex] = delta;
      const Eigen::Vector3d high = (omega - bias) * time, low = -omega * time;
      point =
        Eigen::AngleAxisd(high.norm(), high.normalized()) * (Eigen::AngleAxisd(low.norm(), low.normalized()) * point);
    }
    if (index >= kAccelIndex && index < kGravityIndex)
      point[index - kAccelIndex] -= kHalf * time * time * delta;
    if (index >= kGravityIndex) {
      S2 gravity(state.gravity);
      MTK::vect<2, double> tangent = Eigen::Vector2d::Zero();
      tangent[index - kGravityIndex] = delta;
      gravity.boxplus(tangent);
      displacement += kHalf * time * time * (gravity.get_vect() - state.gravity);
    }
    return row.plane->normal.dot(state.rotation * point + displacement);
  };
  for (Eigen::Index index = kVelocityIndex; index < kStateDof; ++index)
    EXPECT_NEAR(analytic[index], (motion(index, kStep) - motion(index, -kStep)) / (2.0 * kStep), kCentralBound);
  const auto expected = productionRow(row, state, true);
  expectMatrix(analytic.head<kProductionColumns>(), expected);
}

TEST(FrontendLidarJointAdapter, O6bColumnsPredictActualImuProcessDeskewDifferentials)
{
  constexpr double kRelativeTime = 0.05;
  const auto nominal = actualDeskew(Vector23::Zero());
  Row row = crossRow();
  row.point = nominal.point;
  row.relativeTime = -kRelativeTime;
  for (Eigen::Index index = kVelocityIndex; index < kStateDof; ++index) {
    const bool gyro = index >= kGyroIndex && index < kAccelIndex;
    const double step = gyro || index >= kGravityIndex ? 0.01 : 0.1;
    Vector23 perturbation = Vector23::Zero();
    perturbation[index] = step;
    const auto positive = actualDeskew(perturbation), negative = actualDeskew(-perturbation);
    Vector23 high, low;
    positive.state.boxminus(high, nominal.state);
    negative.state.boxminus(low, nominal.state);
    const Vector23 endDerivative = (high - low) / (2.0 * step);
    const Eigen::Vector3d pointDerivative = (positive.worldPoint - negative.worldPoint) / (2.0 * step);
    // The z-normal gyro case cancels the free-fall translation lever. Other
    // directions validate v/ba/S2 gravity. General segmented motion is K3-2.
    const std::vector<Eigen::Index> axes = gyro ? std::vector<Eigen::Index>{2} : std::vector<Eigen::Index>{0, 1, 2};
    // Binary32 point output plus the declared central-trig truncation bound.
    const double truncation = gyro ? 4.0 * step * step : (index >= kGravityIndex ? 0.1 * step * step : 0.0);
    const double tolerance = 16.0 * std::numeric_limits<float>::epsilon() * 16.0 / step + truncation;
    for (const auto axis : axes) {
      row.plane->normal = Eigen::Vector3d::Unit(axis);
      StateRowJacobian h;
      ASSERT_EQ(geometricRow(row, adapterState(nominal.state), false, Eigen::Vector3d::Zero(), h),
                FrameStatus::kOkFrameConditional);
      EXPECT_NEAR(h.dot(endDerivative), pointDerivative[axis], tolerance) << "state " << index << " normal " << axis;
    }
  }
}

TEST(FrontendLidarJointAdapter, ClosureReadingsAndConditionalFlags)
{
  constexpr std::size_t kRows = 40;
  auto snapshot = frame(kRows);
  snapshot.holdActive = true;
  snapshot.considerKeep = Eigen::Vector3d(kHalf, kQuarter, 1.0).asDiagonal();
  snapshot.floorMode = FloorMode::kObserve;
  snapshot.nearestNeighborsRefreshed = false;
  closeFixture(snapshot);
  snapshot.diagnostics.converged = false;
  const auto result = observeFrame(snapshot);
  ASSERT_EQ(result.status, FrameStatus::kOkFrameConditional) << result.rejectReason;
  ASSERT_TRUE(result.readings);
  ASSERT_TRUE(result.readings->amplitude);
  EXPECT_EQ(result.k2Status, js::Status::kOk);
  EXPECT_EQ(result.readings->poolDof, 80U);
  EXPECT_GT(*result.readings->pooledMapVariance, 0.0);
  EXPECT_GE(result.readings->amplitude->gainMinusOptimalMinimum, -kOracleTolerance);
  for (const auto value : {FrameFlag::kHoldActive,
                           FrameFlag::kConsiderActive,
                           FrameFlag::kFloorObserve,
                           FrameFlag::kNnNotRefreshed,
                           FrameFlag::kLinearizationNotConverged,
                           FrameFlag::kQuerySideOnly,
                           FrameFlag::kAmplitudeAssumed,
                           FrameFlag::kDeskewColumnsRigidApprox})
    EXPECT_NE(result.flags & static_cast<std::uint32_t>(value), 0U);
  EXPECT_TRUE((snapshot.priorCovariance->array() == (Matrix23::Identity() * kHalf).array()).all());
  EXPECT_TRUE(result.timings.closureNs);
  EXPECT_TRUE(result.timings.k2Ns);
}

TEST(FrontendLidarJointAdapter, LowPoolDofAndUnavailableColumnsRemainExplicit)
{
  auto snapshot = frame(10);
  snapshot.rows[0].relativeTime.reset();
  snapshot.angularVelocity.reset();
  const auto result = observeFrame(snapshot);
  ASSERT_EQ(result.status, FrameStatus::kOkFrameConditional);
  ASSERT_TRUE(result.readings);
  EXPECT_FALSE(result.readings->amplitude);
  EXPECT_FALSE(result.readings->pooledMapVariance);
  EXPECT_FALSE(result.readings->clockSensitivity);
  EXPECT_FALSE(result.readings->gyroWhiteSensitivity);
  EXPECT_FALSE(result.readings->accelWhiteSensitivity);
  EXPECT_FALSE(result.timings.k2Ns);
  EXPECT_NE(result.flags & static_cast<std::uint32_t>(FrameFlag::kSigmaPoolDofLow), 0U);
  EXPECT_NE(result.flags & static_cast<std::uint32_t>(FrameFlag::kDeskewColumnsUnavailable), 0U);
}

TEST(FrontendLidarJointAdapter, ManufacturerRangeAmplitudeUsesBeamProjectionAndProvenance)
{
  constexpr std::size_t kRows = 36;
  constexpr double kThirdCoordinate = 3.0, kRotationAngle = 0.2, kExtrinsicAngle = 0.1;
  constexpr Eigen::Index kAnalyticSupportCount = 5, kAnalyticLoadsPerRow = 15;
  constexpr double kSupportDivisor = 5.0, kWebsiteSigma = 0.01;
  auto snapshot = frame(kRows);
  snapshot.rows[0].point = Eigen::Vector3d(1.0, 2.0, kThirdCoordinate);
  snapshot.rows[0].beamDirection = Eigen::Vector3d(-1.0, 2.0, kThirdCoordinate).normalized();
  snapshot.linearization->rotation = Eigen::AngleAxisd(kRotationAngle, Eigen::Vector3d::UnitY());
  snapshot.linearization->extrinsicRotation = Eigen::AngleAxisd(-kExtrinsicAngle, Eigen::Vector3d::UnitZ());
  snapshot.prior = snapshot.linearization;
  snapshot.queryAmplitude.source = QueryAmplitudeSource::kVendorRange;
  closeFixture(snapshot);
  const auto typical = observeFrame(snapshot);
  ASSERT_EQ(typical.status, FrameStatus::kOkFrameConditional);
  ASSERT_TRUE(typical.readings->amplitude);
  EXPECT_DOUBLE_EQ(*typical.readings->vendorRangeSigma, 0.015);
  EXPECT_NE(typical.flags & static_cast<std::uint32_t>(FrameFlag::kAmplitudeRangeOnly), 0U);
  EXPECT_NE(typical.flags & static_cast<std::uint32_t>(FrameFlag::kAmplitudeAssumed), 0U);
  snapshot.queryAmplitude.specification = VendorRangeSpecification::kWebsite10mm;
  const auto website = observeFrame(snapshot);
  ASSERT_EQ(website.status, FrameStatus::kOkFrameConditional);
  EXPECT_DOUBLE_EQ(*website.readings->vendorRangeSigma, 0.01);
  EXPECT_DOUBLE_EQ(*website.readings->pooledMapVariance, *typical.readings->pooledMapVariance);
  EXPECT_LT(website.readings->amplitude->optimalCovariance.trace(),
            typical.readings->amplitude->optimalCovariance.trace());
  // Independent dense construction of the query-only diagonal plus TLS map
  // loads tests world-frame beam projection, not just the exposed sigma tag.
  const auto count = static_cast<Eigen::Index>(snapshot.rows.size());
  Eigen::MatrixXd h(count, kStateDof),
    b = Eigen::MatrixXd::Zero(count, Eigen::Index{3} * static_cast<Eigen::Index>(kSupportCount) * count);
  Eigen::VectorXd d(count);
  const auto & state = *snapshot.linearization;
  for (Eigen::Index i = 0; i < count; ++i) {
    const auto & row = snapshot.rows[static_cast<std::size_t>(i)];
    h.row(i).setZero();
    h.block<1, kProductionColumns>(i, 0) = productionRow(row, state, false);
    const double time = *row.relativeTime;
    const Eigen::Vector3d normalBody = state.rotation.inverse() * row.plane->normal;
    h.block<1, 3>(i, kVelocityIndex) = time * row.plane->normal.transpose();
    h.block<1, 3>(i, kGyroIndex) = -time * h.block<1, 3>(i, 3);  // Fixture omega is exactly zero.
    h.block<1, 3>(i, kAccelIndex) = -kHalf * time * time * normalBody.transpose();
    S2 direction(state.gravity);
    Eigen::Matrix<double, 3, 2> gravityDerivative;
    MTK::vect<2, double> zero = Eigen::Vector2d::Zero();
    direction.S2_Mx(gravityDerivative, zero);
    h.block<1, 2>(i, kGravityIndex) = kHalf * time * time * row.plane->normal.transpose() * gravityDerivative;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto & point : row.support)
      center += Eigen::Vector3d(point.x, point.y, point.z) / kSupportDivisor;
    const Eigen::Vector3d world =
      state.rotation * (state.extrinsicRotation * row.point + state.extrinsicTranslation) + state.position;
    for (Eigen::Index a = 0; a < kAnalyticSupportCount; ++a) {
      const auto & point = row.support[static_cast<std::size_t>(a)];
      const Eigen::Vector3d local = Eigen::Vector3d(point.x, point.y, point.z) - center;
      const double weight =
        0.2 + (world - center).dot(row.plane->axis1) * local.dot(row.plane->axis1) / row.plane->eigenvalues[1] +
        (world - center).dot(row.plane->axis2) * local.dot(row.plane->axis2) / row.plane->eigenvalues[2];
      b.block<1, 3>(i, kAnalyticLoadsPerRow * i + 3 * a) = weight * row.plane->normal.transpose();
    }
    const Eigen::Vector3d beam = state.rotation * (state.extrinsicRotation * row.beamDirection->normalized());
    d[i] = kWebsiteSigma * kWebsiteSigma * std::pow(row.plane->normal.dot(beam), 2);
  }
  const Eigen::MatrixXd r = *website.readings->pooledMapVariance * b * b.transpose() + d.asDiagonal().toDenseMatrix();
  const Matrix23 expected = (snapshot.priorCovariance->inverse() + h.transpose() * r.ldlt().solve(h)).inverse();
  expectMatrix(website.readings->amplitude->optimalCovariance, expected);
}

TEST(FrontendLidarJointAdapter, ReviewClosureRejectsSmallScaleRelativeMismatch)
{
  constexpr double kPoseVariance = 1.0e-6, kPositionIncrement = 1.0e-3, kRelativePerturbation = 1.0e-3;
  constexpr Eigen::Index kPositionDof = 3;
  auto snapshot = frame();
  snapshot.priorCovariance->topLeftCorner<kPoseDof, kPoseDof>() =
    Eigen::Matrix<double, kPoseDof, kPoseDof>::Identity() * kPoseVariance;
  snapshot.considerKeep = Eigen::Matrix3d::Zero();
  closeFixture(snapshot);
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(snapshot.productionH.rows(), kStateDof);
  h.leftCols(kProductionColumns) = snapshot.productionH;
  const Matrix23 prior = *snapshot.priorCovariance;
  const Matrix23 information = h.transpose() * h + kProductionVariance * prior.inverse();
  const Eigen::MatrixXd gain = information.ldlt().solve(h.transpose());
  const Eigen::MatrixXd positionGain = gain.topRows(kPositionDof);
  const Eigen::Vector3d target(kPositionIncrement, 0.0, 0.0);
  snapshot.productionObservation =
    positionGain.transpose() * (positionGain * positionGain.transpose()).ldlt().solve(target);
  snapshot.preFloorObservation = snapshot.productionObservation;
  const Eigen::Vector3d increment = positionGain * snapshot.productionObservation;
  snapshot.posterior->position = increment;
  snapshot.diagnostics.lastDx.head<kPositionDof>() = increment;
  const auto baseline = observeFrame(snapshot);
  ASSERT_EQ(baseline.status, FrameStatus::kOkFrameConditional) << baseline.rejectReason;
  EXPECT_NEAR(increment[0], kPositionIncrement, kOracleTolerance * kPositionIncrement);
  auto covarianceMismatch = snapshot;
  (*covarianceMismatch.posteriorCovariance)(0, 0) *= 1.0 + kRelativePerturbation;
  const auto covarianceResult = observeFrame(covarianceMismatch);
  EXPECT_EQ(covarianceResult.status, FrameStatus::kIdentityMismatch) << covarianceResult.rejectReason;
  EXPECT_FALSE(covarianceResult.readings);
  auto incrementMismatch = snapshot;
  incrementMismatch.diagnostics.lastDx[0] *= 1.0 + kRelativePerturbation;
  const auto incrementResult = observeFrame(incrementMismatch);
  EXPECT_EQ(incrementResult.status, FrameStatus::kIdentityMismatch) << incrementResult.rejectReason;
  EXPECT_FALSE(incrementResult.readings);
}

TEST(FrontendLidarJointAdapter, O10RejectionsAreAtomicAndNumericalReadingsAreAbsent)
{
  constexpr double kDifferentPointTime = 0.01, kIdentityPerturbation = 0.1;
  const auto original = frame();
  const auto check = [&](FrameSnapshot snapshot, FrameStatus expected, Budgets budgets = {}) {
    const auto prior = snapshot.priorCovariance;
    const auto result = observeFrame(snapshot, budgets);
    EXPECT_EQ(result.status, expected) << result.rejectReason;
    EXPECT_FALSE(result.readings);
    if (prior)
      EXPECT_TRUE((snapshot.priorCovariance->array() == prior->array()).all());
  };
  auto changed = original;
  changed.selected = false;
  check(changed, FrameStatus::kSkippedDecimated);
  changed = original;
  changed.diagnostics.valid = false;
  check(changed, FrameStatus::kNoUpdate);
  changed = original;
  changed.posterior.reset();
  check(changed, FrameStatus::kNoUpdate);
  changed = original;
  changed.rows[0].point.x() = std::numeric_limits<double>::quiet_NaN();
  check(changed, FrameStatus::kNonfiniteInput);
  changed = original;
  changed.kernel = Kernel::kRobust;
  check(changed, FrameStatus::kUnsupportedKernel);
  changed = original;
  changed.rows[0].kind = RowKind::kGroundRecovery;
  check(changed, FrameStatus::kUnsupportedRowKind);
  changed = original;
  changed.rows[0].plane.reset();
  check(changed, FrameStatus::kUnsupportedRowKind);
  changed = original;
  changed.rows[0].weight = 0.0;
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  changed.rows[0].relativeTime = kDifferentPointTime;
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  changed.rows[1].queryOrdinal = changed.rows[0].queryOrdinal;
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  changed.diagnostics.lastDx[0] = kIdentityPerturbation;
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  (*changed.posteriorCovariance)(0, 0) += kIdentityPerturbation;
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  changed.rows[0].plane->eigenvalues[1] = 0.0;
  check(changed, FrameStatus::kRowLeverageUnbounded);
  Budgets small;
  small.rows = 1;
  check(original, FrameStatus::kFrameRowsOverBudget, small);
  small = {};
  small.bytes = 1;
  check(original, FrameStatus::kMemoryBudget, small);
  small = {};
  small.compressedRows = 0;
  check(original, FrameStatus::kRowCapacity, small);
  changed = original;
  changed.queryAmplitude.source = QueryAmplitudeSource::kVendorRange;
  changed.rows[0].beamDirection = Eigen::Vector3d::Zero();
  closeFixture(changed);
  check(changed, FrameStatus::kNumericalFailure);
  changed = original;
  changed.queryAmplitude.source = QueryAmplitudeSource::kVendorRange;
  changed.rows[0].beamDirection.reset();
  check(changed, FrameStatus::kIdentityMismatch);
  changed = original;
  changed.queryAmplitude.source = QueryAmplitudeSource::kVendorRange;
  changed.rows[0] = crossRow();
  changed.rows[0].beamDirection = Eigen::Vector3d::UnitX();
  closeFixture(changed);
  check(changed, FrameStatus::kNumericalFailure);  // Grazing range-only D=0 is never jittered.
  // A finite, invertible but indefinite unused nuisance prior passes the
  // production identity calculation and is then rejected by K2, not repaired.
  changed = original;
  (*changed.priorCovariance)(kStateDof - 1, kStateDof - 1) = -kHalf;
  closeFixture(changed);
  const auto rejected = observeFrame(changed);
  EXPECT_EQ(rejected.status, FrameStatus::kK2Rejected);
  EXPECT_EQ(rejected.k2Status, js::Status::kInvalidCovariance);
  EXPECT_FALSE(rejected.readings);
}

TEST(FrontendLidarJointAdapter, CompressionFailureLeavesCallerOutputUnchanged)
{
  constexpr double kSentinel = 123.0;
  CompressedFrame output;
  output.h = Eigen::MatrixXd::Constant(1, 1, kSentinel);
  Counts counts;
  const SparseMatrix b(2, 0);
  EXPECT_EQ(compressRows(Eigen::MatrixXd::Ones(2, 1),
                         Eigen::VectorXd::Zero(2),
                         Eigen::MatrixXd::Ones(1, 2),
                         b,
                         Eigen::VectorXd::Zero(2),
                         0.0,
                         {},
                         output,
                         counts),
            FrameStatus::kNumericalFailure);
  EXPECT_DOUBLE_EQ(output.h(0, 0), kSentinel);
}

TEST(FrontendLidarJointAdapter, ReviewInnerFactorUsesFillReducingEliminationOrder)
{
  // One scalar map source per node of a 24x24 grid and one row per grid edge,
  // so the Woodbury inner matrix has the connected grid pattern. Source columns
  // are scrambled: neither the natural order nor the inverse of a fill-reducing
  // order is then fill-reducing. The reference is Eigen's own AMD factorization
  // of the same inner matrix. Equality is expected; the factor 2 only absorbs
  // ordering tie-breaks, while an inverted permutation fills several times more.
  constexpr Eigen::Index kGrid = 24, kSources = kGrid * kGrid, kScramble = 77;
  const auto source = [&](Eigen::Index row, Eigen::Index column) {
    return ((row * kGrid + column) * kScramble) % kSources;
  };
  std::vector<Eigen::Triplet<double>> loads;
  Eigen::Index rows = 0;
  const auto edge = [&](Eigen::Index first, Eigen::Index second) {
    loads.emplace_back(rows, first, 1.0);
    loads.emplace_back(rows, second, 1.0);
    ++rows;
  };
  for (Eigen::Index row = 0; row < kGrid; ++row)
    for (Eigen::Index column = 0; column < kGrid; ++column) {
      if (column + 1 < kGrid)
        edge(source(row, column), source(row, column + 1));
      if (row + 1 < kGrid)
        edge(source(row, column), source(row + 1, column));
    }
  SparseMatrix b(rows, kSources);
  b.setFromTriplets(loads.begin(), loads.end());
  SparseMatrix inner = b.transpose() * b;
  for (Eigen::Index i = 0; i < kSources; ++i)
    inner.coeffRef(i, i) += 1.0;
  inner.makeCompressed();
  const Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower, Eigen::AMDOrdering<int>> reference(inner);
  ASSERT_EQ(reference.info(), Eigen::Success);
  const auto referenceFill = static_cast<std::size_t>(reference.matrixL().nestedExpression().nonZeros());
  ASSERT_GT(referenceFill, static_cast<std::size_t>(kSources));
  CompressedFrame compressed;
  Counts counts;
  ASSERT_EQ(compressRows(Eigen::MatrixXd::Ones(rows, 1),
                         Eigen::VectorXd::Zero(rows),
                         Eigen::MatrixXd::Constant(1, rows, 1.0 / static_cast<double>(rows)),
                         b,
                         Eigen::VectorXd::Ones(rows),
                         1.0,
                         {},
                         compressed,
                         counts),
            FrameStatus::kOkFrameConditional);
  ASSERT_TRUE(counts.nnzFactor);
  EXPECT_LE(*counts.nnzFactor, 2 * referenceFill);
}

TEST(FrontendLidarJointAdapter, DuplicateSupportIdentityUsesAllFloatBits)
{
  constexpr std::size_t kRows = 40;
  auto snapshot = frame(kRows, 4);
  auto result = observeFrame(snapshot);
  ASSERT_EQ(result.status, FrameStatus::kOkFrameConditional);
  EXPECT_EQ(*result.counts.supportPoints, 50U);
  EXPECT_EQ(*result.counts.duplicateSupportReferences, 150U);
  EXPECT_EQ(result.readings->poolDof, 20U);
  snapshot.rows[1].support[0].intensity += 1.0F;
  result = observeFrame(snapshot);
  ASSERT_EQ(result.status, FrameStatus::kOkFrameConditional);
  EXPECT_EQ(*result.counts.supportPoints, 51U);
}

TEST(FrontendLidarJointAdapter, RowIdentityDigestIsOrderedCanonicalSha256)
{
  constexpr int kSecondOrdinal = 7, kThirdOrdinal = 42;
  std::vector<Row> rows(3);
  rows[0].queryOrdinal = 0;
  rows[1].queryOrdinal = kSecondOrdinal;
  rows[2].queryOrdinal = kThirdOrdinal;
  // SHA256 of little-endian uint64 [3,0,7,42], independently generated.
  EXPECT_EQ(rowSetDigest(rows), "d04da9bd6e532de288ee63d5d8adc4358fdf2a1e7d3d3fbc614a83465048c2f0");
  std::swap(rows[0], rows[1]);
  EXPECT_NE(rowSetDigest(rows), rowSetDigest(std::vector<Row>(3)));
}

TEST(FrontendLidarJointAdapter, DISABLED_SyntheticFrameResourceDistribution)
{
  constexpr int kSamples = 5;
  for (const std::size_t rows : {1118U, 1599U, 2374U}) {
    const auto snapshot = frame(rows, 8);
    for (int sample = 0; sample < kSamples; ++sample) {
      const auto result = observeFrame(snapshot);
      ASSERT_EQ(result.status, FrameStatus::kOkFrameConditional) << result.rejectReason;
      struct rusage usage{};
      ASSERT_EQ(getrusage(RUSAGE_SELF, &usage), 0);
      std::cout << "K3_RESOURCE rows=" << rows << " sample=" << sample << " total_ns=" << *result.timings.totalNs
                << " accounted_bytes=" << *result.counts.accountedPeakBytes << " process_peak_kib=" << usage.ru_maxrss
                << " support=" << *result.counts.supportPoints << " factor_nnz=" << *result.counts.nnzFactor << '\n';
    }
  }
}

}  // namespace
}  // namespace fastlio::lidar_joint

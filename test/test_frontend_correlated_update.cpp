#include <Eigen/Cholesky>

#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <sstream>

#include "frontend_correlated_update.hpp"

namespace fastlio::joint_source
{
namespace
{

// Frozen before execution. Rational fixtures have innovation conditioning <=129,
// QR/products have <=110 terms. 8192 eps bounds accumulated arithmetic error for
// these fixtures. Matrix assertions scale this by the expected norm.
constexpr double kTolerance = 8192.0 * std::numeric_limits<double>::epsilon();

std::string numericValues(const Eigen::MatrixXd & value)
{
  std::ostringstream text;
  text << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (Eigen::Index i = 0; i < value.size(); ++i)
    text << (i == 0 ? "" : ",") << value.data()[i];
  return text.str();
}

void recordNumeric(const Eigen::MatrixXd & actual,
                   const Eigen::MatrixXd & expected,
                   double tolerance,
                   const std::string & metric)
{
  const auto * result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
  const std::string key = "sample_" + std::to_string(result->test_property_count());
  ::testing::Test::RecordProperty(key + "_actual", numericValues(actual));
  ::testing::Test::RecordProperty(key + "_expected", numericValues(expected));
  ::testing::Test::RecordProperty(key + "_tolerance", numericValues(Eigen::MatrixXd::Constant(1, 1, tolerance)));
  ::testing::Test::RecordProperty(key + "_metric", metric);
}

// The original GTest predicate and its three operands are unchanged; recording
// only exposes successful comparisons for the required cross-architecture table.
::testing::AssertionResult recordedNear(const char * actualText,
                                        const char * expectedText,
                                        const char * toleranceText,
                                        double actual,
                                        double expected,
                                        double tolerance)
{
  recordNumeric(Eigen::MatrixXd::Constant(1, 1, actual), Eigen::MatrixXd::Constant(1, 1, expected), tolerance, "abs");
  return ::testing::internal::DoubleNearPredFormat(
    actualText, expectedText, toleranceText, actual, expected, tolerance);
}

Coordinates coordinates(Eigen::Index count, const std::string & prefix)
{
  Coordinates result{{}, {}, "synthetic-frame", "linear tangent", "epoch-1"};
  for (Eigen::Index i = 0; i < count; ++i) {
    result.components.push_back(prefix + std::to_string(i));
    result.units.push_back("m");
  }
  return result;
}

Source source(const std::string & key, Eigen::Index count = 1)
{
  return {key, "synthetic", {key + "/raw"}, {key + "/alias"}, coordinates(count, key), "analytic", "retained"};
}

JointMoments prior(Eigen::Index active = 1, Eigen::Index sources = 1)
{
  JointMoments result;
  result.active = coordinates(active, "x");
  if (sources > 0)
    result.sources = {source("z", sources)};
  result.knowledge = MomentKnowledge::kKnown;
  result.evidence = "explicit analytic joint moments";
  result.mean = Eigen::VectorXd::Zero(active + sources);
  result.covariance = Eigen::MatrixXd::Identity(active + sources, active + sources);
  return result;
}

LinearFactor measurement(const JointMoments & moments, const std::string & key = "query", Eigen::Index rows = 1)
{
  LinearFactor factor;
  factor.domain = ModelDomain::kLinearFixedModel;
  factor.identity = {key, "fixed likelihood v1", {key + "/raw"}, {}, key + "/eps", "independent analytic eps"};
  factor.model = "synthetic fixed H/A";
  factor.linearization = "fixed nominal";
  factor.active = moments.active;
  for (const auto & block : moments.sources) {
    factor.sourceKeys.push_back(block.key);
    factor.sourceCoordinates.push_back(block.coordinates);
  }
  factor.rowUnits = std::vector<std::string>(rows, "m");
  factor.remainderKnowledge = MomentKnowledge::kKnown;
  factor.independentRemainder = true;
  factor.h = Eigen::MatrixXd::Zero(rows, moments.active.components.size());
  factor.h.col(0).setOnes();
  factor.a = Eigen::MatrixXd::Ones(rows, sourceDof(moments.sources));
  factor.observation = Eigen::VectorXd::Ones(rows);
  factor.remainderMean = Eigen::VectorXd::Zero(rows);
  factor.remainderVariance = Eigen::VectorXd::Ones(rows);
  return factor;
}

ProcessNoise zeroNoise(Eigen::Index priorDof, Eigen::Index count)
{
  return {MomentKnowledge::kKnown,
          "explicit structural zero",
          Eigen::VectorXd::Zero(count),
          Eigen::MatrixXd::Zero(count, count),
          Eigen::MatrixXd::Zero(priorDof, count)};
}

void expectMatrix(const Eigen::MatrixXd & actual, const Eigen::MatrixXd & expected)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  recordNumeric(actual, expected, kTolerance * std::max(1.0, expected.norm()), "norm");
  EXPECT_LE((actual - expected).norm(), kTolerance * std::max(1.0, expected.norm()));
}

void expectExact(const JointMoments & actual, const JointMoments & expected)
{
  ASSERT_EQ(actual.mean.size(), expected.mean.size());
  ASSERT_EQ(actual.covariance.rows(), expected.covariance.rows());
  EXPECT_TRUE((actual.mean.array() == expected.mean.array()).all());
  EXPECT_TRUE((actual.covariance.array() == expected.covariance.array()).all());
  EXPECT_EQ(actual.active.epoch, expected.active.epoch);
  ASSERT_EQ(actual.sources.size(), expected.sources.size());
  for (std::size_t i = 0; i < actual.sources.size(); ++i) {
    EXPECT_EQ(actual.sources[i].key, expected.sources[i].key);
    EXPECT_EQ(actual.sources[i].rawAncestry, expected.sources[i].rawAncestry);
  }
}

void update(JointSourceFilter & filter, const std::vector<LinearFactor> & factors)
{
  const auto proposal = filter.propose(factors);
  ASSERT_EQ(proposal.status(), Status::kOk);
  ASSERT_EQ(filter.commit(proposal), Status::kOk);
}

TEST(FrontendCorrelatedUpdate, I01SharedSaturationAndIdentifiableDifferentLoading)
{
  for (const Eigen::Index count : {1, 2, 8, 64}) {
    const auto initial = prior();
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    update(filter, {measurement(initial, "batch", count)});
    const double expected = static_cast<double>(count + 1) / static_cast<double>(2 * count + 1);
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), expected, kTolerance);
    EXPECT_GT(expected, 0.5);
    if (count > 1)
      EXPECT_GT(std::abs(expected - 2.0 / (2.0 + count)), 0.09);
  }
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto positive = measurement(initial, "plus");
  auto negative = measurement(initial, "minus");
  negative.a(0, 0) = -1.0;
  update(filter, {positive, negative});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
  EXPECT_LT(filter.moments()->covariance(0, 0), 0.5);  // no universal ceiling
}

TEST(FrontendCorrelatedUpdate, I02ExactDuplicateAliasesAndChangedNumericPayload)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial);
  update(filter, {factor, factor});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 2.0 / 3.0, kTolerance);
  const auto snapshot = *filter.moments();
  const auto entries = filter.ledger()->entries();
  factor.sourceKeys[0] = "z/alias";
  const auto alias = filter.propose({factor});
  ASSERT_EQ(alias.status(), Status::kNoOp);
  EXPECT_EQ(filter.commit(alias), Status::kNoOp);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries(), entries);
  factor.observation(0) += 1.0;
  EXPECT_EQ(filter.propose({factor}).status(), Status::kIdentityConflict);
  EXPECT_GT(std::abs(snapshot.covariance(0, 0) - 0.5), 0.1);
  expectExact(*filter.moments(), snapshot);
}

TEST(FrontendCorrelatedUpdate, I03RawGeometryReuseSharesCanonicalSource)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"shared-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto first = measurement(initial, "geometry1");
  first.identity.rawAncestry = {"shared-raw"};
  first.identity.shared = {{"shared-raw", "z/alias"}};
  update(filter, {first});
  auto second = measurement(initial, "geometry2");
  second.identity.rawAncestry = {"shared-raw"};
  EXPECT_EQ(filter.propose({second}).status(), Status::kUnsupportedNoiseStructure);
  second.identity.shared = {{"shared-raw", "z"}};
  second.a(0, 0) = -1.0;
  update(filter, {second});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
  second.identity.key = "geometry3";
  second.identity.remainderKey = "eps3";
  second.a.setZero();
  EXPECT_EQ(filter.propose({second}).status(), Status::kUnsupportedNoiseStructure);
}

TEST(FrontendCorrelatedUpdate, I04ConditionalEquivalenceHasIndependentRationalOracle)
{
  auto initial = prior();
  initial.covariance << 2.0, 1.0, 1.0, 3.0;
  auto factor = measurement(initial);
  factor.a(0, 0) = 2.0;
  factor.observation(0) = 3.0;
  // B=1/2, Cxi=5/2, Hc=2, Rc=11, S=19, E_x=4, E_z=7.
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  update(filter, {factor});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->mean(0), 12.0 / 19.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 22.0 / 19.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 1), -9.0 / 19.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(1, 1), 8.0 / 19.0, kTolerance);
  EXPECT_GT(std::abs(22.0 / 19.0 - (2.0 - 4.0 / 15.0)), 0.5);    // deleting K
  EXPECT_GT(std::abs(22.0 / 19.0 - (2.0 - 16.0 / 21.0)), 0.07);  // HP H' in Rc
}

TEST(FrontendCorrelatedUpdate, I05PerfectCorrelationCancellationAndDoubling)
{
  for (const double sign : {-1.0, 1.0}) {
    auto initial = prior();
    initial.covariance.setOnes();
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    auto factor = measurement(initial);
    factor.a(0, 0) = sign;
    update(filter, {factor});
    const double expected = sign < 0.0 ? 1.0 : 1.0 / 5.0;
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), expected, kTolerance);
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 1), expected, kTolerance);
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(1, 1), expected, kTolerance);
    EXPECT_GT(std::abs(expected - 2.0 / 3.0), 0.3);
  }
}

TEST(FrontendCorrelatedUpdate, I06SequentialBatchAndBothOrdersPreservePosterior)
{
  const auto initial = prior();
  const auto first = measurement(initial, "one");
  auto second = measurement(initial, "two");
  second.observation(0) = 2.0;
  Eigen::Matrix2d expected;
  expected << 3.0 / 5.0, -2.0 / 5.0, -2.0 / 5.0, 3.0 / 5.0;
  for (int order = 0; order < 3; ++order) {
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    if (order == 0)
      update(filter, {first, second});
    else {
      update(filter, {order == 1 ? first : second});
      EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 1), -1.0 / 3.0, kTolerance);
      update(filter, {order == 1 ? second : first});
    }
    expectMatrix(filter.moments()->covariance, expected);
    expectMatrix(filter.moments()->mean, Eigen::Vector2d::Constant(3.0 / 5.0));
  }
  EXPECT_GT(3.0 / 5.0 - 10.0 / 21.0, 0.12);
  EXPECT_GT(3.0 / 5.0 - 1.0 / 2.0, 0.09);
}

TEST(FrontendCorrelatedUpdate, I07ProposalsShareImmutablePriorAndCommitOnlyOnce)
{
  const auto initial = prior();
  JointSourceFilter filter;
  JointSourceFilter other;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  ASSERT_EQ(other.initialize(initial), Status::kOk);
  const auto factor = measurement(initial);
  const auto first = filter.propose({factor});
  const auto second = filter.propose({factor});
  ASSERT_NE(first.posterior(), nullptr);
  ASSERT_NE(second.posterior(), nullptr);
  expectExact(*first.posterior(), *second.posterior());
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
  EXPECT_EQ(other.commit(first), Status::kStaleProposal);
  EXPECT_EQ(filter.commit(first), Status::kOk);
  const auto snapshot = *filter.moments();
  EXPECT_EQ(filter.commit(first), Status::kStaleProposal);
  EXPECT_EQ(filter.commit(second), Status::kStaleProposal);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries().size(), 1U);
}

TEST(FrontendCorrelatedUpdate, I10GeneratedMomentsHaveSignedCrossAndDeterministicSupport)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  Eigen::Matrix2d generator;
  generator << 0.0, -2.0, 0.0, 1.0;
  ASSERT_EQ(filter.augment(source("h-and-n", 2), generator, Eigen::Vector2d::Zero(), zeroNoise(2, 2)), Status::kOk);
  const auto & covariance = filter.moments()->covariance;
  EXPECT_PRED_FORMAT3(recordedNear, covariance(2, 2), 4.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, covariance(2, 3), -2.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, covariance(3, 3), 1.0, kTolerance);
  const Eigen::Vector4d null(0.0, 0.0, 1.0, 2.0);
  EXPECT_LE((covariance * null).norm(), kTolerance);
  EXPECT_GT(std::abs(covariance(2, 3) - 2.0), 3.9);
  auto invalid = initial;
  invalid.covariance << 1.0, 2.0, 2.0, 1.0;
  JointSourceFilter bad;
  EXPECT_EQ(bad.initialize(invalid), Status::kInvalidCovariance);
}

TEST(FrontendCorrelatedUpdate, I14UnknownZeroNonPSDAndIllegalCrossAreDistinct)
{
  auto initial = prior();
  JointSourceFilter filter;
  initial.knowledge = MomentKnowledge::kUnknown;
  EXPECT_EQ(filter.initialize(initial), Status::kUnknownMoment);
  EXPECT_EQ(filter.moments(), nullptr);
  initial.knowledge = MomentKnowledge::kKnown;
  initial.covariance.setZero();
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  update(filter, {measurement(initial)});
  expectMatrix(filter.moments()->covariance, Eigen::Matrix2d::Zero());
  expectMatrix(filter.moments()->mean, Eigen::Vector2d::Zero());
  for (const double cross : {-2.0, 2.0}) {
    initial = prior();
    initial.covariance(0, 1) = initial.covariance(1, 0) = cross;
    JointSourceFilter invalid;
    EXPECT_EQ(invalid.initialize(initial), Status::kInvalidCovariance);
  }
  initial.covariance << -1.0, 0.0, 0.0, 1.0;
  JointSourceFilter negative;
  EXPECT_EQ(negative.initialize(initial), Status::kInvalidCovariance);
}

TEST(FrontendCorrelatedUpdate, I14SingularObservationIsRejectedIncludingImpossibleResidual)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial, "singular", 2);
  factor.remainderVariance.setZero();
  factor.observation << 1.0, 2.0;
  EXPECT_EQ(filter.propose({factor}).status(), Status::kUnsupportedNoiseStructure);
  factor.observation.setOnes();
  EXPECT_EQ(filter.propose({factor}).status(), Status::kUnsupportedNoiseStructure);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, I15PlaneNullspaceUsesAllPoseColumns)
{
  const auto initial = prior(6, 0);
  auto factor = measurement(initial, "plane", 3);
  factor.h << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  update(filter, {factor});
  for (const Eigen::Index column : {1, 2, 3}) {
    const auto direction = Eigen::VectorXd::Unit(6, column).eval();
    EXPECT_EQ((factor.h * direction).norm(), 0.0);
    expectMatrix(filter.moments()->covariance * direction, direction);
  }
  EXPECT_LT(filter.moments()->covariance(0, 0), 1.0);
  auto wrong = factor.h;
  wrong(0, 1) = 0.1;
  EXPECT_GT((wrong * Eigen::VectorXd::Unit(6, 1)).norm(), 0.09);
}

TEST(FrontendCorrelatedUpdate, I16PriorCrossLegitimatelyReducesNullCoordinateVariance)
{
  for (const double rho : {0.0, 0.5}) {
    auto initial = prior(2, 0);
    initial.covariance(0, 1) = initial.covariance(1, 0) = rho;
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    const auto factor = measurement(initial);
    update(filter, {factor});
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(1, 1), 1.0 - rho * rho / 2.0, kTolerance);
    EXPECT_EQ((factor.h.transpose() * factor.h).col(1).norm(), 0.0);
    if (rho != 0.0)
      EXPECT_LT(filter.moments()->covariance(1, 1), 0.9);
  }
}

TEST(FrontendCorrelatedUpdate, I17IndependentEqualCoordinatesReduceP)
{
  const auto initial = prior(1, 0);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  update(filter, {measurement(initial, "first")});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 0.5, kTolerance);
  update(filter, {measurement(initial, "second")});
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
  EXPECT_LT(filter.moments()->covariance(0, 0), 0.5);
}

TEST(FrontendCorrelatedUpdate, I18ActualGainReportsFiveEighthsAndCanIncreaseP)
{
  const auto initial = prior(1, 0);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  const auto factor = measurement(initial);
  for (const double gain : {0.0, 0.25, 0.5, 2.0}) {
    const auto result = filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, gain));
    ASSERT_EQ(result.status, Status::kOk);
    const double expected = 1.0 - 2.0 * gain + 2.0 * gain * gain;
    EXPECT_PRED_FORMAT3(recordedNear, result.covariance(0, 0), expected, kTolerance);
    EXPECT_PRED_FORMAT3(recordedNear, result.mean(0), gain, kTolerance);
    EXPECT_GE(expected, 0.5);
  }
  const auto quarter = filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, 0.25));
  EXPECT_PRED_FORMAT3(recordedNear, quarter.covariance(0, 0), 5.0 / 8.0, kTolerance);
  EXPECT_GT(quarter.covariance(0, 0) - 0.5, 0.12);
  const auto amplified = filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, 2.0));
  EXPECT_GT(amplified.covariance(0, 0), 1.0);
  expectExact(*filter.moments(), initial);
}

TEST(FrontendCorrelatedUpdate, I21ResetTransportsCrossAndPreservesPredictions)
{
  const auto initial = prior();
  JointSourceFilter original;
  JointSourceFilter transformed;
  ASSERT_EQ(original.initialize(initial), Status::kOk);
  ASSERT_EQ(transformed.initialize(initial), Status::kOk);
  const auto first = measurement(initial, "first");
  update(original, {first});
  update(transformed, {first});
  Eigen::Matrix2d transform;
  transform << 2.0, 0.0, 0.0, -3.0;
  auto active = initial.active;
  auto sourceCoordinates = initial.sources[0].coordinates;
  active.epoch = sourceCoordinates.epoch = "epoch-2";
  ASSERT_EQ(transformed.reset(transform, Eigen::Vector2d::Zero(), active, {sourceCoordinates}), Status::kOk);
  Eigen::Matrix2d expected;
  expected << 8.0 / 3.0, 2.0, 2.0, 6.0;
  expectMatrix(transformed.moments()->covariance, expected);
  EXPECT_GT(std::abs(transformed.moments()->covariance(0, 1) + 1.0 / 3.0), 2.0);
  auto nextOriginal = measurement(*original.moments(), "next");
  auto nextTransformed = measurement(*transformed.moments(), "next");
  nextTransformed.h(0, 0) = 0.5;
  nextTransformed.a(0, 0) = -1.0 / 3.0;
  update(original, {nextOriginal});
  update(transformed, {nextTransformed});
  expectMatrix(transformed.moments()->mean, transform * original.moments()->mean);
  expectMatrix(transformed.moments()->covariance, transform * original.moments()->covariance * transform.transpose());
  EXPECT_EQ(transformed.moments()->sources[0].key, "z");
  EXPECT_EQ(transformed.ledger()->entries().size(), 2U);
}

TEST(FrontendCorrelatedUpdate, I22CorrelatedProcessAndGeneratedSourceKeepFullCross)
{
  auto initial = prior();
  initial.covariance(0, 1) = initial.covariance(1, 0) = 0.25;
  initial.mean << 1.0, 2.0;
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  Eigen::Matrix2d transition = Eigen::Vector2d(2.0, 1.0).asDiagonal();
  auto noise = zeroNoise(2, 2);
  noise.covariance(0, 0) = 0.5;
  noise.priorCross(1, 0) = 0.25;
  noise.mean(0) = 0.5;
  ASSERT_EQ(
    filter.propagate(transition, Eigen::Vector2d::Zero(), noise, initial.active, {initial.sources[0].coordinates}),
    Status::kOk);
  Eigen::Matrix2d expected;
  expected << 4.5, 0.75, 0.75, 1.0;
  expectMatrix(filter.moments()->covariance, expected);
  expectMatrix(filter.moments()->mean, Eigen::Vector2d(2.5, 2.0));
  EXPECT_GT(std::abs(filter.moments()->covariance(0, 1) - 0.5), 0.24);
  Eigen::RowVector2d generator(3.0, -2.0);
  auto sourceNoise = zeroNoise(2, 1);
  sourceNoise.covariance(0, 0) = 0.25;
  sourceNoise.mean(0) = 0.25;
  ASSERT_EQ(filter.augment(source("map"), generator, Eigen::VectorXd::Ones(1), sourceNoise), Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 2), 12.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(1, 2), 0.25, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(2, 2), 35.75, kTolerance * 36.0);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->mean(2), 4.75, kTolerance * 5.0);
}

TEST(FrontendCorrelatedUpdate, I23Full23ColumnsAndMaximumDimensionsAreUsable)
{
  const auto initial = prior(23, 32);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial, "max", 64);
  factor.h.setZero();
  factor.a.setZero();
  for (const Eigen::Index column : {12, 15, 18, 21, 22})
    factor.h(column, column) = 1.0;
  update(filter, {factor});
  for (const Eigen::Index column : {12, 15, 18, 21, 22})
    EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(column, column), 0.5, kTolerance);
  EXPECT_EQ(filter.moments()->mean.size(), 55);
  const auto snapshot = *filter.moments();
  const auto entries = filter.ledger()->entries();
  EXPECT_EQ(filter.propose({measurement(initial, "too-many", 65)}).status(), Status::kModelCapacityExceeded);
  EXPECT_EQ(filter.augment(source("extra"), Eigen::MatrixXd::Zero(1, 55), Eigen::VectorXd::Zero(1), zeroNoise(55, 1)),
            Status::kModelCapacityExceeded);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries(), entries);
  JointSourceFilter oversized;
  EXPECT_EQ(oversized.initialize(prior(24, 0)), Status::kModelCapacityExceeded);
  EXPECT_EQ(oversized.initialize(prior(1, 33)), Status::kModelCapacityExceeded);
}

TEST(FrontendCorrelatedUpdate, I23LedgerOverflowIsAtomic)
{
  const auto initial = prior(1, 0);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  for (std::size_t i = 0; i < kMaxLedger; ++i)
    update(filter, {measurement(initial, std::to_string(i))});
  const auto snapshot = *filter.moments();
  const auto entries = filter.ledger()->entries();
  const auto overflow = filter.propose({measurement(initial, "overflow")});
  EXPECT_EQ(overflow.status(), Status::kModelCapacityExceeded);
  EXPECT_EQ(filter.commit(overflow), Status::kModelCapacityExceeded);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries(), entries);
}

TEST(FrontendCorrelatedUpdate, I24NonzeroMeansAreCenteredExactlyOnce)
{
  auto initial = prior();
  initial.mean << 2.0, 3.0;
  auto factor = measurement(initial);
  factor.remainderMean(0) = 4.0;
  factor.observation(0) = 12.0;
  JointSourceFilter shifted;
  ASSERT_EQ(shifted.initialize(initial), Status::kOk);
  update(shifted, {factor});
  expectMatrix(shifted.moments()->mean, Eigen::Vector2d(3.0, 4.0));
  const auto centered = prior();
  JointSourceFilter zero;
  ASSERT_EQ(zero.initialize(centered), Status::kOk);
  auto centeredFactor = measurement(centered);
  centeredFactor.observation(0) = 3.0;
  update(zero, {centeredFactor});
  expectMatrix(shifted.moments()->mean - initial.mean, zero.moments()->mean);
  expectMatrix(shifted.moments()->covariance, zero.moments()->covariance);
  EXPECT_GT(std::abs(shifted.moments()->mean(0) - (2.0 + 12.0 / 3.0)), 2.9);
  factor.identity.key = "unknown";
  factor.remainderKnowledge = MomentKnowledge::kUnknown;
  EXPECT_EQ(shifted.propose({factor}).status(), Status::kUnknownMoment);
}

TEST(FrontendCorrelatedUpdate, I26ExactMarginalAndRenamedAncestryReentry)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  update(filter, {measurement(initial)});
  const auto snapshot = *filter.moments();
  const auto entries = filter.ledger()->entries();
  EXPECT_EQ(filter.marginalize({"z"}, false, "may reuse"), Status::kSourceHistoryUnavailable);
  expectExact(*filter.moments(), snapshot);
  ASSERT_EQ(filter.marginalize({"z/alias"}, true, "finished source"), Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 0), 2.0 / 3.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->mean(0), 1.0 / 3.0, kTolerance);
  EXPECT_TRUE(filter.moments()->sources.empty());
  EXPECT_EQ(filter.retiredSources(), 1U);
  EXPECT_EQ(filter.ledger()->entries(), entries);
  auto renamed = initial.sources[0];
  const auto repeated = filter.propose({measurement(initial)});
  EXPECT_EQ(repeated.status(), Status::kNoOp);
  EXPECT_EQ(filter.commit(repeated), Status::kNoOp);
  renamed.key = "renamed";
  renamed.coordinates.epoch = "new-epoch";
  EXPECT_EQ(filter.augment(renamed, Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Zero(1), zeroNoise(1, 1)),
            Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.marginalize({"unknown"}, true, "unused"), Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.retiredSources(), 1U);
}

TEST(FrontendCorrelatedUpdate, I02ExactResendSurvivesResetAndAugmentation)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  const auto factor = measurement(initial);
  update(filter, {factor});
  auto changed = initial.active;
  changed.epoch = "epoch-2";
  ASSERT_EQ(
    filter.reset(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), changed, {initial.sources[0].coordinates}),
    Status::kOk);
  ASSERT_EQ(filter.augment(source("new"), Eigen::MatrixXd::Zero(1, 2), Eigen::VectorXd::Zero(1), zeroNoise(2, 1)),
            Status::kOk);
  const auto snapshot = *filter.moments();
  const auto repeated = filter.propose({factor});
  EXPECT_EQ(repeated.status(), Status::kNoOp);
  EXPECT_EQ(filter.commit(repeated), Status::kNoOp);
  expectExact(*filter.moments(), snapshot);
  auto newLikelihood = factor;
  newLikelihood.identity.key = "new-key";
  newLikelihood.identity.remainderKey = "new-eps";
  newLikelihood.identity.rawAncestry = {"new-raw"};
  EXPECT_EQ(filter.propose({newLikelihood}).status(), Status::kDimensionMismatch);
}

TEST(FrontendCorrelatedUpdate, InvalidDimensionsNonFiniteAndLateBatchFailureAreAtomic)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  const auto fresh = measurement(initial, "fresh");
  auto invalid = measurement(initial, "invalid");
  invalid.a.resize(0, 0);
  EXPECT_EQ(filter.propose({fresh, invalid}).status(), Status::kDimensionMismatch);
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    invalid = measurement(initial, "invalid");
    invalid.h(0, 0) = bad;
    EXPECT_EQ(filter.propose({fresh, invalid}).status(), Status::kNonFinite);
    auto badPrior = initial;
    badPrior.mean(0) = bad;
    JointSourceFilter badFilter;
    EXPECT_EQ(badFilter.initialize(badPrior), Status::kNonFinite);
    badPrior = initial;
    badPrior.covariance(0, 0) = bad;
    EXPECT_EQ(badFilter.initialize(badPrior), Status::kNonFinite);
  }
  invalid = measurement(initial, "invalid");
  invalid.remainderVariance(0) = -1.0;
  EXPECT_EQ(filter.propose({invalid}).status(), Status::kUnsupportedNoiseStructure);
  invalid.domain = ModelDomain::kUnknown;
  EXPECT_EQ(filter.propose({invalid}).status(), Status::kUnsupportedModel);
  EXPECT_EQ(filter.commit(filter.propose({})), Status::kNoOp);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, InvalidProcessCrossResetAndAugmentationAreAtomic)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto noise = zeroNoise(2, 2);
  noise.priorCross(0, 0) = 1.0;
  EXPECT_EQ(
    filter.propagate(
      Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), noise, initial.active, {initial.sources[0].coordinates}),
    Status::kInvalidCovariance);
  noise = zeroNoise(2, 1);
  noise.knowledge = MomentKnowledge::kUnknown;
  EXPECT_EQ(filter.augment(source("fresh"), Eigen::MatrixXd::Ones(1, 2), Eigen::VectorXd::Zero(1), noise),
            Status::kUnknownMoment);
  EXPECT_EQ(
    filter.reset(
      Eigen::MatrixXd::Identity(3, 3), Eigen::Vector2d::Zero(), initial.active, {initial.sources[0].coordinates}),
    Status::kDimensionMismatch);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, EmptyMissingAndAsymmetricPriorCannotBeInitialized)
{
  JointSourceFilter filter;
  EXPECT_EQ(filter.propose({}).status(), Status::kUnknownMoment);
  EXPECT_EQ(filter.initialize(JointMoments{}), Status::kInvalidIdentity);
  auto invalid = prior();
  invalid.mean.resize(0);
  EXPECT_EQ(filter.initialize(invalid), Status::kDimensionMismatch);
  invalid = prior();
  invalid.evidence.clear();
  EXPECT_EQ(filter.initialize(invalid), Status::kUnknownMoment);
  invalid = prior();
  invalid.covariance(0, 1) = 0.1;
  EXPECT_EQ(filter.initialize(invalid), Status::kInvalidCovariance);
  invalid = prior();
  invalid.sources[0].coordinates.units.clear();
  EXPECT_EQ(filter.initialize(invalid), Status::kInvalidIdentity);
  EXPECT_EQ(filter.moments(), nullptr);
}

TEST(FrontendCorrelatedUpdate, I22IndependentProcessAndCorrelatedAugmentationNoise)
{
  auto initial = prior();
  initial.covariance(0, 1) = initial.covariance(1, 0) = 0.25;
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto independent = zeroNoise(2, 2);
  independent.covariance(0, 0) = 1.0;
  Eigen::Matrix2d transition = Eigen::Vector2d(2.0, 1.0).asDiagonal();
  ASSERT_EQ(filter.propagate(
              transition, Eigen::Vector2d::Zero(), independent, initial.active, {initial.sources[0].coordinates}),
            Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 1), 0.5, kTolerance);
  auto correlated = zeroNoise(2, 1);
  correlated.covariance(0, 0) = 1.0;
  correlated.priorCross(1, 0) = 0.5;
  ASSERT_EQ(filter.augment(source("generated"), Eigen::RowVector2d(1.0, 0.0), Eigen::VectorXd::Zero(1), correlated),
            Status::kOk);
  // q=x+w, Var(x)=5, Cov(w,z)=1/2. Cross with z is 1/2+1/2.
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(0, 2), 5.0, kTolerance * 5.0);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(1, 2), 1.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, filter.moments()->covariance(2, 2), 6.0, kTolerance * 6.0);
  EXPECT_GT(std::abs(filter.moments()->covariance(1, 2) - 0.5), 0.49);
}

TEST(FrontendCorrelatedUpdate, I18AppliedGainRejectsDimensionAliasesAndNonFinite)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  const auto factor = measurement(initial);
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(2, 1)).status, Status::kDimensionMismatch);
  EXPECT_EQ(filter.appliedGain({factor, factor}, Eigen::MatrixXd::Zero(1, 2)).status,
            Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(
    filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, std::numeric_limits<double>::infinity())).status,
    Status::kNonFinite);
  update(filter, {factor});
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kUnsupportedNoiseStructure);
}

TEST(FrontendCorrelatedUpdate, I26RetiredIdentityCapacityCannotBeClearedByEpoch)
{
  const auto initial = prior(1, 0);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  for (std::size_t i = 0; i < kMaxLedger; ++i) {
    const auto block = source("retired" + std::to_string(i));
    ASSERT_EQ(filter.augment(block, Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Zero(1), zeroNoise(1, 1)),
              Status::kOk);
    ASSERT_EQ(filter.marginalize({block.key}, true, "never reused"), Status::kOk);
  }
  auto aliasesOld = source("different");
  aliasesOld.aliases = {"retired0"};
  EXPECT_EQ(filter.augment(aliasesOld, Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Zero(1), zeroNoise(1, 1)),
            Status::kSourceHistoryUnavailable);
  ASSERT_EQ(filter.augment(source("extra"), Eigen::MatrixXd::Zero(1, 1), Eigen::VectorXd::Zero(1), zeroNoise(1, 1)),
            Status::kOk);
  const auto snapshot = *filter.moments();
  EXPECT_EQ(filter.marginalize({"extra"}, true, "never reused"), Status::kModelCapacityExceeded);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.retiredSources(), kMaxLedger);
}

TEST(FrontendCorrelatedUpdate, I26MarginalPreservesOtherSourcesCrossWithoutConditioning)
{
  auto initial = prior(1, 2);
  initial.sources = {source("a"), source("b")};
  initial.mean << 1.0, 2.0, 3.0;
  initial.covariance << 2.0, 0.5, 0.25, 0.5, 1.0, 0.125, 0.25, 0.125, 1.0;
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  ASSERT_EQ(filter.marginalize({"a"}, true, "completed a"), Status::kOk);
  Eigen::Matrix2d expected;
  expected << 2.0, 0.25, 0.25, 1.0;
  expectMatrix(filter.moments()->covariance, expected);
  expectMatrix(filter.moments()->mean, Eigen::Vector2d(1.0, 3.0));
  ASSERT_EQ(filter.moments()->sources.size(), 1U);
  EXPECT_EQ(filter.moments()->sources[0].key, "b");
  EXPECT_GT(std::abs(filter.moments()->covariance(0, 0) - 1.75), 0.24);  // conditioning is different
}

// Independent covariance-form batch oracle. The product updates roots one row at
// a time. This fixture solves the three-row innovation system directly instead.
TEST(FrontendCorrelatedUpdate, ReviewF1DensePosteriorSurvivesEveryLifecycleStep)
{
  for (const Eigen::Index sources : {1, 3, 9, 10, 21, 31}) {
    const Eigen::Index dimension = 23 + sources;
    auto initial = prior(23, sources);
    for (Eigen::Index row = 0; row < dimension; ++row) {
      for (Eigen::Index column = 0; column < dimension; ++column)
        initial.covariance(row, column) = std::ldexp(1.0, -static_cast<int>(std::abs(row - column)));
    }
    auto factor = measurement(initial, "dense", 3);
    for (Eigen::Index row = 0; row < 3; ++row) {
      for (Eigen::Index column = 0; column < 23; ++column)
        factor.h(row, column) = 1.0 / static_cast<double>(1 + row + column);
      for (Eigen::Index column = 0; column < sources; ++column)
        factor.a(row, column) = 1.0 / static_cast<double>(2 + row + 2 * column);
    }
    Eigen::MatrixXd loading(3, dimension);
    loading << factor.h, factor.a;
    const Eigen::MatrixXd cross = initial.covariance * loading.transpose();
    const Eigen::Matrix3d innovation = loading * cross + Eigen::Matrix3d::Identity();
    const Eigen::MatrixXd expectedPosterior = initial.covariance - cross * innovation.ldlt().solve(cross.transpose());
    const Eigen::VectorXd expectedMean = cross * innovation.ldlt().solve(Eigen::Vector3d::Ones());
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    update(filter, {factor});
    const Eigen::MatrixXd posterior = filter.moments()->covariance;
    EXPECT_TRUE((posterior.array() == posterior.transpose().array()).all());
    expectMatrix(posterior, expectedPosterior);
    expectMatrix(filter.moments()->mean, expectedMean);
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(dimension, dimension);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(dimension);
    std::vector<Coordinates> blocks{initial.sources[0].coordinates};
    ASSERT_EQ(filter.reset(identity, zero, initial.active, blocks), Status::kOk);
    expectMatrix(filter.moments()->covariance, expectedPosterior);
    auto correlated = zeroNoise(dimension, dimension);
    correlated.covariance.setIdentity();
    correlated.priorCross(0, 0) = 0.125;
    ASSERT_EQ(filter.propagate(identity, zero, correlated, initial.active, blocks), Status::kOk);
    Eigen::MatrixXd expected = expectedPosterior + identity;
    expected(0, 0) += 0.25;
    expectMatrix(filter.moments()->covariance, expected);
    const Eigen::MatrixXd generator = Eigen::MatrixXd::Constant(1, dimension, 0.5);
    ASSERT_EQ(filter.augment(source("derived"), generator, Eigen::VectorXd::Zero(1), zeroNoise(dimension, 1)),
              Status::kOk);
    Eigen::VectorXd null = Eigen::VectorXd::Constant(dimension + 1, 0.5);
    null(dimension) = -1.0;
    EXPECT_LE((filter.moments()->covariance * null).norm(), kTolerance * expected.norm());
    blocks.push_back(source("derived").coordinates);
    ASSERT_EQ(filter.reset(Eigen::MatrixXd::Identity(dimension + 1, dimension + 1),
                           Eigen::VectorXd::Zero(dimension + 1),
                           initial.active,
                           blocks),
              Status::kOk);
    expectMatrix(filter.moments()->covariance.topLeftCorner(dimension, dimension), expected);
    auto independent = zeroNoise(dimension + 1, dimension + 1);
    independent.covariance(0, 0) = 1.0;
    ASSERT_EQ(filter.propagate(Eigen::MatrixXd::Identity(dimension + 1, dimension + 1),
                               Eigen::VectorXd::Zero(dimension + 1),
                               independent,
                               initial.active,
                               blocks),
              Status::kOk);
    EXPECT_PRED_FORMAT3(
      recordedNear, filter.moments()->covariance(0, 0), expected(0, 0) + 1.0, kTolerance * expected.norm());
  }
}

TEST(FrontendCorrelatedUpdate, ReviewF2DeterministicSourcesSurviveRepeatedLifecycleSteps)
{
  for (const Eigen::Index seed : {1, 2, 3, 4, 5, 6, 7, 8}) {
    auto initial = prior(2, 1);
    initial.covariance << 2.0, 0.5, 0.25, 0.5, 1.0, 0.125, 0.25, 0.125, 1.0;
    auto factor = measurement(initial);
    factor.h << 1.0, 0.2;
    factor.a << 0.3;
    const Eigen::Vector3d loading(1.0, 0.2, 0.3);
    const Eigen::Vector3d cross = initial.covariance * loading;
    const Eigen::Matrix3d posterior = initial.covariance - cross * cross.transpose() / (1.0 + loading.dot(cross));
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    update(filter, {factor});
    Eigen::RowVector3d generator(1.0 / static_cast<double>(seed + 2), 0.4, -1.0 / 7.0);
    ASSERT_EQ(filter.augment(source("deterministic"), generator, Eigen::VectorXd::Zero(1), zeroNoise(3, 1)),
              Status::kOk);
    Eigen::Matrix4d expected;
    expected << posterior, posterior * generator.transpose(), generator * posterior,
      generator * posterior * generator.transpose();
    std::vector<Coordinates> blocks{initial.sources[0].coordinates, source("deterministic").coordinates};
    ASSERT_EQ(filter.reset(Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(), initial.active, blocks), Status::kOk);
    expectMatrix(filter.moments()->covariance, expected);
    auto independent = zeroNoise(4, 4);
    independent.covariance(0, 0) = 0.25;
    ASSERT_EQ(
      filter.propagate(Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(), independent, initial.active, blocks),
      Status::kOk);
    expected(0, 0) += 0.25;
    expectMatrix(filter.moments()->covariance, expected);
  }
}

TEST(FrontendCorrelatedUpdate, ReviewF3FreshFactorsCheckLiveAndRetiredRawAncestry)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"carried-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto unshared = measurement(initial, "unshared");
  unshared.identity.rawAncestry = {"carried-raw"};
  unshared.a.setZero();
  EXPECT_EQ(filter.propose({unshared}).status(), Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(filter.appliedGain({unshared}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kUnsupportedNoiseStructure);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
  auto valid = measurement(initial, "shared");
  valid.identity.rawAncestry = {"carried-raw"};
  valid.identity.shared = {{"carried-raw", "z/alias"}};
  update(filter, {valid});
  const auto beforeRetirement = filter.ledger()->entries();
  ASSERT_EQ(filter.marginalize({"z"}, true, "finished source"), Status::kOk);
  const auto snapshot = *filter.moments();
  auto reentry = measurement(snapshot, "reentry");
  reentry.identity.rawAncestry = {"carried-raw"};
  EXPECT_EQ(filter.propose({reentry}).status(), Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.appliedGain({reentry}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kSourceHistoryUnavailable);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries(), beforeRetirement);
  EXPECT_EQ(filter.commit(filter.propose({valid})), Status::kNoOp);
  expectExact(*filter.moments(), snapshot);
}

TEST(FrontendCorrelatedUpdate, ReviewF4AppliedGainPreservesLedgerErrorReasons)
{
  const auto initial = prior(1, 0);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  const auto first = measurement(initial, "first");
  update(filter, {first});
  auto conflict = first;
  conflict.observation(0) += 1.0;
  EXPECT_EQ(filter.appliedGain({conflict}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kIdentityConflict);
  EXPECT_EQ(filter.appliedGain({first}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(filter.appliedGain({first, measurement(initial, "new")}, Eigen::MatrixXd::Zero(1, 2)).status,
            Status::kUnsupportedNoiseStructure);
  for (std::size_t i = 1; i < kMaxLedger; ++i)
    update(filter, {measurement(initial, std::to_string(i))});
  const auto snapshot = *filter.moments();
  const auto entries = filter.ledger()->entries();
  EXPECT_EQ(filter.appliedGain({measurement(initial, "overflow")}, Eigen::MatrixXd::Zero(1, 1)).status,
            Status::kModelCapacityExceeded);
  expectExact(*filter.moments(), snapshot);
  EXPECT_EQ(filter.ledger()->entries(), entries);
}

TEST(FrontendCorrelatedUpdate, ReviewF6DistinctLoadingsAcrossSeveralSourceBlocks)
{
  auto initial = prior(1, 3);
  initial.sources = {source("a"), source("b", 2)};
  initial.sources[1].rawAncestry = {"second-block-raw"};
  auto factor = measurement(initial, "multiblock", 2);
  factor.identity.rawAncestry = {"second-block-raw"};
  factor.identity.shared = {{"second-block-raw", "b/alias"}};
  factor.h << 1.0, 2.0;
  factor.a << 1.0, 2.0, -1.0, 0.0, 1.0, 3.0;
  factor.observation << 1.0, 2.0;
  // Independent exact batch algebra. S=[[8,1],[1,15]], det(S)=119.
  // J' S^-1 y = [43,13,41,32]/119, P+=I-J'S^-1J.
  Eigen::Vector4d expectedMean(43.0, 13.0, 41.0, 32.0);
  expectedMean /= 119.0;
  Eigen::Matrix4d expected;
  expected << 76.0, -13.0, -41.0, -32.0, -13.0, 104.0, -29.0, 18.0, -41.0, -29.0, 55.0, 11.0, -32.0, 18.0, 11.0, 26.0;
  expected /= 119.0;
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  Eigen::RowVector2d gain(0.25, -0.125);
  const auto applied = filter.appliedGain({factor}, gain);
  ASSERT_EQ(applied.status, Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, applied.covariance(0, 0), 107.0 / 64.0, kTolerance);
  EXPECT_PRED_FORMAT3(recordedNear, applied.mean(0), 0.0, kTolerance);
  auto wrongBlock = factor;
  wrongBlock.a.rightCols(2).setZero();
  EXPECT_EQ(filter.propose({wrongBlock}).status(), Status::kUnsupportedNoiseStructure);
  update(filter, {factor});
  expectMatrix(filter.moments()->mean, expectedMean);
  expectMatrix(filter.moments()->covariance, expected);
}

TEST(FrontendCorrelatedUpdate, ReviewF9CoordinateIdentityMismatchHasIdentityStatus)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  for (const bool active : {true, false}) {
    auto mismatched = measurement(initial);
    if (active)
      mismatched.active.epoch = "wrong-epoch";
    else
      mismatched.sourceCoordinates[0].frame = "wrong-frame";
    EXPECT_EQ(filter.propose({mismatched}).status(), Status::kIdentityConflict);
    EXPECT_EQ(filter.appliedGain({mismatched}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kIdentityConflict);
    expectExact(*filter.moments(), initial);
    EXPECT_TRUE(filter.ledger()->entries().empty());
  }
}

TEST(FrontendCorrelatedUpdate, RecheckR1LiveUnloadedSharedRemainsUnsupported)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"carried-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial, "unloaded-live");
  factor.identity.rawAncestry = {"carried-raw"};
  factor.identity.shared = {{"carried-raw", "z/alias"}};
  factor.a.setZero();
  EXPECT_EQ(filter.propose({factor}).status(), Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kUnsupportedNoiseStructure);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, RecheckR1RetiredUnloadedSharedIsUnavailable)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"carried-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  ASSERT_EQ(filter.marginalize({"z"}, true, "source lifetime ended"), Status::kOk);
  const auto snapshot = *filter.moments();
  auto factor = measurement(snapshot, "unloaded-retired");
  factor.identity.rawAncestry = {"carried-raw"};
  factor.identity.shared = {{"carried-raw", "z/alias"}};
  EXPECT_EQ(filter.propose({factor}).status(), Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kSourceHistoryUnavailable);
  expectExact(*filter.moments(), snapshot);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, RecheckR1RetiredAndLiveCarrierRemainUnavailable)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"carried-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto child = source("derived");
  child.rawAncestry = {"carried-raw"};
  Eigen::RowVector2d generator(0.0, 1.0);
  ASSERT_EQ(filter.augment(child, generator, Eigen::VectorXd::Zero(1), zeroNoise(2, 1)), Status::kOk);
  ASSERT_EQ(filter.marginalize({"derived"}, true, "derived lifetime ended"), Status::kOk);
  const auto snapshot = *filter.moments();
  auto factor = measurement(snapshot, "live-with-retired-carrier");
  factor.identity.rawAncestry = {"carried-raw"};
  factor.identity.shared = {{"carried-raw", "z"}};
  EXPECT_EQ(filter.propose({factor}).status(), Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kSourceHistoryUnavailable);
  expectExact(*filter.moments(), snapshot);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

TEST(FrontendCorrelatedUpdate, RecheckR2AppliedGainIsBitwiseSymmetric)
{
  for (const Eigen::Index dimension : {7, 12, 23}) {
    auto initial = prior(dimension, 0);
    Eigen::MatrixXd generator(dimension, dimension);
    for (Eigen::Index row = 0; row < dimension; ++row)
      for (Eigen::Index col = 0; col < dimension; ++col)
        generator(row, col) = static_cast<double>((row * 11 + col * 7) % 19 - 9) / 23.0;
    const Eigen::MatrixXd gram = generator * generator.transpose();
    initial.covariance = gram.selfadjointView<Eigen::Lower>();
    initial.covariance.diagonal().array() += 1.0;
    JointSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial), Status::kOk);
    auto factor = measurement(initial, "dense-gain", kMaxRows);
    Eigen::MatrixXd gain(dimension, kMaxRows);
    for (Eigen::Index row = 0; row < kMaxRows; ++row)
      for (Eigen::Index col = 0; col < dimension; ++col) {
        factor.h(row, col) = static_cast<double>((row * 13 + col * 3) % 17 - 8) / 19.0;
        gain(col, row) = static_cast<double>((row * 5 + col * 11) % 29 - 14) / 37.0;
      }
    const auto result = filter.appliedGain({factor}, gain);
    ASSERT_EQ(result.status, Status::kOk);
    EXPECT_TRUE((result.covariance.array() == result.covariance.transpose().array()).all()) << dimension;
    expectExact(*filter.moments(), initial);
    EXPECT_TRUE(filter.ledger()->entries().empty());
  }
}

TEST(FrontendCorrelatedUpdate, B1bX8LegacyDefaultRetainsAnyCarrier)
{
  auto initial = prior(1, 0);
  initial.sources = {source("first"), source("second")};
  for (auto & block : initial.sources)
    block.rawAncestry = {"shared-raw"};
  initial.mean = Eigen::VectorXd::Zero(3);
  initial.covariance = Eigen::MatrixXd::Ones(3, 3);
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial, "one-of-two-carriers");
  factor.identity.rawAncestry = {"shared-raw"};
  factor.identity.shared = {{"shared-raw", "first"}};
  factor.a(0, 1) = 0.0;
  const auto proposal = filter.propose({factor});
  ASSERT_EQ(proposal.status(), Status::kOk);
  // y=2e+d, Cov(e)=Cov(d)=1: optimal P=1-4/5=1/5.
  EXPECT_PRED_FORMAT3(recordedNear, proposal.posterior()->covariance(0, 0), 1.0 / 5.0, kTolerance);
  const auto applied = filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, 0.25));
  ASSERT_EQ(applied.status, Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, applied.covariance(0, 0), 5.0 / 16.0, kTolerance);
}

TEST(FrontendCorrelatedUpdate, B1bX8ExplicitDerivedRetirementDoesNotRouteRaw)
{
  auto initial = prior();
  initial.sources[0].rawAncestry = {"shared-raw"};
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto derived = source("derived");
  derived.rawAncestry = {"shared-raw"};
  derived.rawRoles = {{"shared-raw", RawRole::kDerived}};
  const Eigen::RowVector2d selector(0.0, 1.0);
  ASSERT_EQ(filter.augment(derived, selector, Eigen::VectorXd::Zero(1), zeroNoise(2, 1)), Status::kOk);
  ASSERT_EQ(filter.marginalize({"derived"}, true, "derived-only lifetime ended"), Status::kOk);
  auto factor = measurement(*filter.moments(), "retired-derived");
  factor.identity.rawAncestry = {"shared-raw"};
  factor.identity.shared = {{"shared-raw", "z"}};
  const auto proposal = filter.propose({factor});
  ASSERT_EQ(proposal.status(), Status::kOk);
  EXPECT_PRED_FORMAT3(recordedNear, proposal.posterior()->covariance(0, 0), 2.0 / 3.0, kTolerance);
}

TEST(FrontendCorrelatedUpdate, B1bX10UpperBoundIsIsolatedAtAllThreeEntries)
{
  auto initial = prior();
  auto upper = initial;
  upper.knowledge = MomentKnowledge::kUpperBound;
  JointSourceFilter filter;
  EXPECT_EQ(filter.initialize(upper), Status::kUnknownMoment);
  EXPECT_EQ(filter.moments(), nullptr);
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial, "upper-remainder");
  factor.remainderKnowledge = MomentKnowledge::kUpperBound;
  EXPECT_EQ(filter.propose({factor}).status(), Status::kUnknownMoment);
  EXPECT_EQ(filter.appliedGain({factor}, Eigen::MatrixXd::Zero(1, 1)).status, Status::kUnknownMoment);
  expectExact(*filter.moments(), initial);
  EXPECT_TRUE(filter.ledger()->entries().empty());
}

}  // namespace
}  // namespace fastlio::joint_source

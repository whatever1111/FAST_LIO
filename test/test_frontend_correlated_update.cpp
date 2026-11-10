#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "frontend_correlated_update.hpp"

namespace fastlio::joint_source
{
namespace
{

// Frozen before execution. Rational fixtures have innovation conditioning <=129,
// QR/products have <=110 terms. 8192 eps bounds accumulated arithmetic error for
// these fixtures. Matrix assertions scale this by the expected norm.
constexpr double kTolerance = 8192.0 * std::numeric_limits<double>::epsilon();

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
    EXPECT_NEAR(filter.moments()->covariance(0, 0), expected, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
  EXPECT_LT(filter.moments()->covariance(0, 0), 0.5);  // no universal ceiling
}

TEST(FrontendCorrelatedUpdate, I02ExactDuplicateAliasesAndChangedNumericPayload)
{
  const auto initial = prior();
  JointSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial), Status::kOk);
  auto factor = measurement(initial);
  update(filter, {factor, factor});
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 2.0 / 3.0, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
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
  EXPECT_NEAR(filter.moments()->mean(0), 12.0 / 19.0, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 22.0 / 19.0, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(0, 1), -9.0 / 19.0, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(1, 1), 8.0 / 19.0, kTolerance);
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
    EXPECT_NEAR(filter.moments()->covariance(0, 0), expected, kTolerance);
    EXPECT_NEAR(filter.moments()->covariance(0, 1), expected, kTolerance);
    EXPECT_NEAR(filter.moments()->covariance(1, 1), expected, kTolerance);
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
      EXPECT_NEAR(filter.moments()->covariance(0, 1), -1.0 / 3.0, kTolerance);
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
  EXPECT_NEAR(covariance(2, 2), 4.0, kTolerance);
  EXPECT_NEAR(covariance(2, 3), -2.0, kTolerance);
  EXPECT_NEAR(covariance(3, 3), 1.0, kTolerance);
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
    EXPECT_NEAR(filter.moments()->covariance(1, 1), 1.0 - rho * rho / 2.0, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 0.5, kTolerance);
  update(filter, {measurement(initial, "second")});
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 1.0 / 3.0, kTolerance);
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
    EXPECT_NEAR(result.covariance(0, 0), expected, kTolerance);
    EXPECT_NEAR(result.mean(0), gain, kTolerance);
    EXPECT_GE(expected, 0.5);
  }
  const auto quarter = filter.appliedGain({factor}, Eigen::MatrixXd::Constant(1, 1, 0.25));
  EXPECT_NEAR(quarter.covariance(0, 0), 5.0 / 8.0, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 2), 12.0, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(1, 2), 0.25, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(2, 2), 35.75, kTolerance * 36.0);
  EXPECT_NEAR(filter.moments()->mean(2), 4.75, kTolerance * 5.0);
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
    EXPECT_NEAR(filter.moments()->covariance(column, column), 0.5, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 0), 2.0 / 3.0, kTolerance);
  EXPECT_NEAR(filter.moments()->mean(0), 1.0 / 3.0, kTolerance);
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
  EXPECT_NEAR(filter.moments()->covariance(0, 1), 0.5, kTolerance);
  auto correlated = zeroNoise(2, 1);
  correlated.covariance(0, 0) = 1.0;
  correlated.priorCross(1, 0) = 0.5;
  ASSERT_EQ(filter.augment(source("generated"), Eigen::RowVector2d(1.0, 0.0), Eigen::VectorXd::Zero(1), correlated),
            Status::kOk);
  // q=x+w, Var(x)=5, Cov(w,z)=1/2. Cross with z is 1/2+1/2.
  EXPECT_NEAR(filter.moments()->covariance(0, 2), 5.0, kTolerance * 5.0);
  EXPECT_NEAR(filter.moments()->covariance(1, 2), 1.0, kTolerance);
  EXPECT_NEAR(filter.moments()->covariance(2, 2), 6.0, kTolerance * 6.0);
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

}  // namespace
}  // namespace fastlio::joint_source

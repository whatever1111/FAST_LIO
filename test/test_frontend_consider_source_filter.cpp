#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/resource.h>

#include "frontend_consider_source_filter.hpp"

namespace fastlio::joint_source
{
namespace detail
{

// GNU ld wrapping observes the real matrix at each call site. The return address
// is relative to the ELF image; addr2line identifies processCopyAudit offline.
// Only this GTest target links the wrapper, and the original audit always runs.
Status realAudit(const Eigen::MatrixXd & covariance) asm(
  "__real__ZN7fastlio12joint_source6detail15auditCovarianceERKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEE");
Status observedAudit(const Eigen::MatrixXd & covariance) asm(
  "__wrap__ZN7fastlio12joint_source6detail15auditCovarianceERKN5Eigen6MatrixIdLin1ELin1ELi0ELin1ELin1EEE");

Status observedAudit(const Eigen::MatrixXd & covariance)
{
  const auto * test = ::testing::UnitTest::GetInstance()->current_test_info();
  Dl_info image{};
  const auto * caller = __builtin_return_address(0);
  // The timing fixture measures entries without XML observation overhead.
  if (test != nullptr && std::strcmp(test->name(), "DISABLED_FullBudgetSingleEntryTimings") != 0 &&
      dladdr(caller, &image) != 0) {
    const auto offset = reinterpret_cast<std::uintptr_t>(caller) - reinterpret_cast<std::uintptr_t>(image.dli_fbase);
    const std::string key = "audit_rows_at_" + std::to_string(offset);
    auto maximum = covariance.rows();
    for (int i = 0; i < test->result()->test_property_count(); ++i) {
      const auto & property = test->result()->GetTestProperty(i);
      if (key == property.key())
        maximum = std::max(maximum, static_cast<Eigen::Index>(std::stoll(property.value())));
    }
    ::testing::Test::RecordProperty(key, std::to_string(maximum));
  }
  return realAudit(covariance);
}

}  // namespace detail
namespace
{

// Frozen before the implementation: the rational scalar fixtures are dyadic;
// the dense reference uses at most 55 coordinates and 128 products per entry.
// 32768 eps allows their accumulated double arithmetic, on either architecture.
constexpr double kTolerance = 32768.0 * std::numeric_limits<double>::epsilon();
constexpr std::size_t kMemoryBytes = static_cast<std::size_t>(256U) * 1024U * 1024U;

Coordinates coordinates(Eigen::Index n, const std::string & prefix = "x")
{
  Coordinates result{{}, {}, "world", "linear", "epoch-1"};
  for (Eigen::Index i = 0; i < n; ++i) {
    result.components.push_back(prefix + std::to_string(i));
    result.units.push_back("m");
  }
  return result;
}

Budget budget()
{
  return {64, 64, 64, 128, 128, kMemoryBytes};
}

JointMoments prior(Eigen::Index n = 1)
{
  return {coordinates(n),
          {},
          MomentKnowledge::kKnown,
          "analytic independent prior",
          Eigen::VectorXd::Zero(n),
          Eigen::MatrixXd::Identity(n, n)};
}

Source source(const std::string & key, const std::string & raw = "map-raw", Eigen::Index n = 1)
{
  return {key, "clone", {raw}, {}, coordinates(n, key), "fixed selector", "until last use"};
}

LinearFactor factor(const Coordinates & active, const std::vector<Source> & sources, const std::string & key)
{
  LinearFactor result;
  result.domain = ModelDomain::kLinearFixedModel;
  result.identity = {key, "analytic fixed likelihood", {key + "/raw"}, {}, key + "/eps", "fresh independent scalar"};
  result.model = "fixed scalar model";
  result.linearization = "given nominal";
  result.active = active;
  for (const auto & block : sources) {
    result.sourceKeys.push_back(block.key);
    result.sourceCoordinates.push_back(block.coordinates);
  }
  result.rowUnits = {"m"};
  result.remainderKnowledge = MomentKnowledge::kKnown;
  result.independentRemainder = true;
  result.h = Eigen::MatrixXd::Ones(1, active.components.size());
  result.a = Eigen::MatrixXd::Zero(1, sourceDof(sources));
  result.observation = Eigen::VectorXd::Zero(1);
  result.remainderMean = Eigen::VectorXd::Zero(1);
  result.remainderVariance = Eigen::VectorXd::Ones(1);
  return result;
}

ProcessNoise noise(Eigen::Index joint, Eigen::Index active, double variance = 0.0)
{
  return {MomentKnowledge::kKnown,
          "explicit complete old joint independence",
          Eigen::VectorXd::Zero(active),
          variance * Eigen::MatrixXd::Identity(active, active),
          Eigen::MatrixXd::Zero(joint, active)};
}

std::string numbers(const Eigen::MatrixXd & matrix)
{
  std::ostringstream text;
  text << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (Eigen::Index col = 0; col < matrix.cols(); ++col)
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
      text << (col == 0 && row == 0 ? "" : ",") << matrix(row, col);
  return text.str();
}

void recordComparison(const Eigen::MatrixXd & actual, const Eigen::MatrixXd & expected)
{
  const auto * result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
  const std::string key = "sample_" + std::to_string(result->test_property_count());
  ::testing::Test::RecordProperty(key + "_actual", numbers(actual));
  ::testing::Test::RecordProperty(key + "_expected", numbers(expected));
  const Eigen::MatrixXd tolerance = Eigen::MatrixXd::Constant(1, 1, kTolerance * std::max(1.0, expected.norm()));
  ::testing::Test::RecordProperty(key + "_tolerance", numbers(tolerance));
  ::testing::Test::RecordProperty(key + "_metric", "max_abs");
}

void expectMatrix(const Eigen::MatrixXd & actual, const Eigen::MatrixXd & expected)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  const double error = actual.size() == 0 ? 0.0 : (actual - expected).cwiseAbs().maxCoeff();
  recordComparison(actual, expected);
  EXPECT_LE(error, kTolerance * std::max(1.0, expected.norm()));
}

Eigen::MatrixXd joint(const ConsiderSourceFilter & filter, const std::vector<Source> & sources)
{
  const auto n = filter.activeCovariance()->rows();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n + sourceDof(sources), n + sourceDof(sources));
  result.topLeftCorner(n, n) = *filter.activeCovariance();
  Eigen::Index row = n;
  for (const auto & first : sources) {
    const auto d = static_cast<Eigen::Index>(first.coordinates.components.size());
    Eigen::MatrixXd block;
    EXPECT_EQ(filter.cross(first.key, block), Status::kOk);
    result.block(0, row, n, d) = block;
    result.block(row, 0, d, n) = block.transpose();
    Eigen::Index col = n;
    for (const auto & second : sources) {
      EXPECT_EQ(filter.sourceCovariance(first.key, second.key, block), Status::kOk);
      result.block(row, col, block.rows(), block.cols()) = block;
      col += block.cols();
    }
    row += d;
  }
  return result;
}

void apply(ConsiderSourceFilter & filter,
           const LinearFactor & input,
           double gain,
           const std::vector<ReuseNoise> & copies = {},
           const std::vector<SensitivityLoading> & sensitivities = {})
{
  auto candidate = filter.proposeAppliedGain(
    {input}, Eigen::MatrixXd::Constant(input.h.cols(), input.h.rows(), gain), copies, sensitivities);
  ASSERT_EQ(candidate.status(), Status::kOk);
  EXPECT_EQ(filter.commit(std::move(candidate)), Status::kOk);
}

TEST(ConsiderSourceFilter, X1DenseCommonDomainSequence)
{
  auto initial = prior(2);
  ConsiderSourceFilter filter;
  JointSourceFilter reference;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  ASSERT_EQ(reference.initialize(initial), Status::kOk);
  const Eigen::Matrix2d phi = (Eigen::Matrix2d() << 1, 0.5, 0, 1).finished();
  ASSERT_EQ(filter.propagate(phi, noise(2, 2, 0.25)), Status::kOk);
  ASSERT_EQ(reference.propagate(phi, Eigen::Vector2d::Zero(), noise(2, 2, 0.25), initial.active, {}), Status::kOk);
  auto z = source("z", "z/raw", 2);
  ASSERT_EQ(filter.augment(z, Eigen::Matrix2d::Identity()), Status::kOk);
  ASSERT_EQ(reference.augment(z, Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), noise(2, 2)), Status::kOk);
  auto input = factor(initial.active, {z}, "first");
  input.h << 1, 0.5;
  input.a << -0.25, 0.5;
  const Eigen::Vector2d gain(0.25, -0.125);
  auto candidate = filter.proposeAppliedGain({input}, gain);
  ASSERT_EQ(candidate.status(), Status::kOk);
  ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<2, 2>() -= gain * input.h;
  transform.topRightCorner<2, 2>() = -gain * input.a;
  auto process = noise(4, 4);
  process.covariance.topLeftCorner<2, 2>() = gain * gain.transpose();
  ASSERT_EQ(reference.propagate(transform, Eigen::Vector4d::Zero(), process, initial.active, {z.coordinates}),
            Status::kOk);
  expectMatrix(joint(filter, {z}), reference.moments()->covariance);
  Eigen::Matrix2d reset = Eigen::Matrix2d::Identity();
  reset(0, 1) = -0.25;
  ASSERT_EQ(filter.reset(reset), Status::kOk);
  transform.setIdentity();
  transform.topLeftCorner<2, 2>() = reset;
  ASSERT_EQ(reference.reset(transform, Eigen::Vector4d::Zero(), initial.active, {z.coordinates}), Status::kOk);
  expectMatrix(joint(filter, {z}), reference.moments()->covariance);
}

TEST(ConsiderSourceFilter, X8DefaultCoverageBoundaryAndExplicitDerived)
{
  const auto initial = prior();
  auto z1 = source("z1");
  auto z2 = source("z2");
  for (const bool derived : {false, true}) {
    if (derived)
      z2.rawRoles = {{"map-raw", RawRole::kDerived}};
    ConsiderSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
    ASSERT_EQ(filter.augment(z1, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    ASSERT_EQ(filter.augment(z2, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    auto input = factor(initial.active, {z1, z2}, "query");
    input.identity.rawAncestry = {"map-raw"};
    input.identity.shared = {{"map-raw", "z1"}};
    input.a << 1, 0;
    auto candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.25));
    EXPECT_EQ(candidate.status(), derived ? Status::kOk : Status::kUnsupportedNoiseStructure);
    if (derived) {
      ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
      Eigen::Matrix3d expected;
      expected << 5.0 / 16.0, 0.5, 0.5, 0.5, 1, 1, 0.5, 1, 1;
      expectMatrix(joint(filter, {z1, z2}), expected);
    } else {
      input.identity.shared.push_back({"map-raw", "z2"});
      input.a << 1, 1;
      apply(filter, input, 0.25);
      EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 0.125);
    }
  }
}

TEST(ConsiderSourceFilter, X2TwoFramesRationalPosteriorAndCross)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  apply(filter, factor(initial.active, {}, "query"), 0.5);
  const auto z = source("frame");
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto support = factor(initial.active, {z}, "support");
  support.a(0, 0) = -1;
  apply(filter, support, 0.5);
  Eigen::Matrix2d expected;
  expected << 0.75, 0.5, 0.5, 0.5;
  expectMatrix(joint(filter, {z}), expected);
  const Eigen::RowVector2d delta(1, -1);
  EXPECT_DOUBLE_EQ((delta * expected * delta.transpose())(0), 0.25);
}

TEST(ConsiderSourceFilter, X3CapturedGainsPreserveCommonPartAndAddRemainder)
{
  const auto initial = prior();
  const auto z = source("z");
  for (const double gain : {0.0, 0.5, 1.0}) {
    ConsiderSourceFilter filter;
    ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
    ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    auto input = factor(initial.active, {z}, "common");
    input.a(0, 0) = -1;
    apply(filter, input, gain);
    Eigen::Matrix2d expected;
    expected << 1 + gain * gain, 1, 1, 1;
    expectMatrix(joint(filter, {z}), expected);
  }
  // H=-A alone does not imply E=0. Phi=0 with Q=2 after the clone leaves the
  // active error independent of the frame source: P=2, K=0, C=3, E=HP+AK'=2.
  // Primitives e0~(0,3), w~(0,2), eps~(0,1): z=e0, e=w, e+=w/2+e0/2-eps/2.
  // The common-mode shortcut P+G*D*G'=9/4, K unchanged=0 fails this fixture.
  auto wide = prior();
  wide.covariance.setConstant(3);
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(wide, budget()), Status::kOk);
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  ASSERT_EQ(filter.propagate(Eigen::MatrixXd::Zero(1, 1), noise(2, 1, 2.0)), Status::kOk);
  Eigen::Matrix2d noncommon;
  noncommon << 2, 0, 0, 3;
  expectMatrix(joint(filter, {z}), noncommon);
  auto input = factor(wide.active, {z}, "noncommon");
  input.a(0, 0) = -1;
  apply(filter, input, 0.5);
  Eigen::Matrix2d expected;
  expected << 1.5, 1.5, 1.5, 3;
  expectMatrix(joint(filter, {z}), expected);
}

ReuseNoise reuse(LinearFactor & input, const std::string & raw, RawId frame, std::uint64_t total)
{
  input.identity.rawAncestry = {raw};
  input.identity.typedRaw = {{raw, {7, 1, 0}}};
  ReuseUse use{{7, 1, 0}, frame, total, {"frame", RawId{9, 1, 0}}};
  input.identity.reuse = {use};
  return {use,
          coordinates(1, "echo"),
          MomentKnowledge::kKnown,
          "independent inflated copy at first query",
          true,
          Eigen::MatrixXd::Ones(1, 1),
          {{input.identity.key, Eigen::MatrixXd::Ones(1, 1)}}};
}

TEST(ConsiderSourceFilter, X4ReuseFromFirstQueryBoundsCompleteJoint)
{
  auto initial = prior();
  initial.covariance.setZero();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto query = factor(initial.active, {}, "query");
  auto first = reuse(query, "echo", {8, 1, 0}, 2);
  apply(filter, query, -1, {first});
  auto z = source("frame", "echo");
  z.sequence = RawId{9, 1, 0};
  z.typedRaw = query.identity.typedRaw;
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto support = factor(initial.active, {z}, "support");
  support.a(0, 0) = -1;
  auto second = reuse(support, "echo", {8, 2, 0}, 2);
  support.identity.shared = {{"echo", "frame"}};
  apply(filter, support, -1, {second});
  Eigen::Matrix2d bound;
  bound << 6, 3, 3, 3;
  Eigen::Matrix2d exact;
  exact << 6, 3, 3, 2;
  expectMatrix(joint(filter, {z}), bound);
  EXPECT_GE(Eigen::LDLT<Eigen::Matrix2d>(bound - exact).vectorD().minCoeff(), 0);
  const Eigen::RowVector2d delta(1, -1);
  EXPECT_GE((delta * (bound - exact) * delta.transpose())(0), 0);
  EXPECT_EQ(filter.knowledge(), MomentKnowledge::kUpperBound);
}

TEST(ConsiderSourceFilter, X4SameFrameAcrossCallsRetainsBothCrossBlocks)
{
  auto initial = prior();
  initial.covariance.setZero();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto query = factor(initial.active, {}, "query");
  auto copy = reuse(query, "echo", {8, 1, 0}, 1);
  apply(filter, query, -1, {copy});
  auto z = source("frame", "echo");
  z.sequence = RawId{9, 1, 0};
  z.typedRaw = query.identity.typedRaw;
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto support = factor(initial.active, {z}, "same-frame-support");
  support.a(0, 0) = -1;
  copy = reuse(support, "echo", {8, 1, 0}, 1);
  support.identity.shared = {{"echo", "frame"}};
  apply(filter, support, -1, {copy});
  Eigen::Matrix2d expected;
  expected << 6, 3, 3, 2;
  expectMatrix(joint(filter, {z}), expected);
  EXPECT_EQ(filter.reuseCount({7, 1, 0}), 1U);
}

TEST(ConsiderSourceFilter, X4ProcessCopyCrossUsesTheSameAuditRule)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto query = factor(initial.active, {}, "query");
  auto copy = reuse(query, "echo", {8, 1, 0}, 1);
  apply(filter, query, -1, {copy});
  // Independent unit primitives e0, eta, epsQuery, epsSupport. Query gives
  // e = 2 e0 + eta + epsQuery. Choose w = eta - e0/2, so Cov(e,w)=0,
  // Var(w)=5/4 and Cov(w,eta)=1. The positive fresh remainders stay independent.
  expectMatrix(*filter.activeCovariance(), Eigen::MatrixXd::Constant(1, 1, 6));
  const auto process = noise(1, 1, 5.0 / 4.0);
  CopyNoiseCross cross{{7, 1, 0}, {8, 1, 0}, Eigen::MatrixXd::Constant(1, 1, 2)};
  const auto ledger = filter.ledgerIdentity();
  EXPECT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), process, {cross}), Status::kInvalidCovariance);
  EXPECT_EQ(filter.ledgerIdentity(), ledger);
  expectMatrix(*filter.activeCovariance(), Eigen::MatrixXd::Constant(1, 1, 6));
  cross.covariance.setOnes();
  ASSERT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), process, {cross}), Status::kOk);
  // e' = 3 e0/2 + 2 eta + epsQuery; the next update is
  // e'' = 3 e0/4 + eta/2 + epsQuery/2 - epsSupport/2.
  expectMatrix(*filter.activeCovariance(), Eigen::MatrixXd::Constant(1, 1, 29.0 / 4.0));
  auto support = factor(initial.active, {}, "support");
  copy = reuse(support, "echo", {8, 1, 0}, 1);
  apply(filter, support, 0.5, {copy});
  expectMatrix(*filter.activeCovariance(), Eigen::MatrixXd::Constant(1, 1, 21.0 / 16.0));
}

TEST(ConsiderSourceFilter, X5YoungOracleDoesNotAuthorizeReentry)
{
  Eigen::Matrix2d exact;
  exact << 0, 0, 0, 1;
  Eigen::Matrix2d bound;
  bound << 4, 2, 2, 2;
  // v=w, M=[1,1]^T and N=[-1,0]^T, beta=1.
  const Eigen::Vector2d m(1, 1), n(-1, 0);
  expectMatrix(2 * m * m.transpose() + 2 * n * n.transpose(), bound);
  EXPECT_GE(Eigen::LDLT<Eigen::Matrix2d>(bound - exact).vectorD().minCoeff(), 0);
  Eigen::Matrix2d onlyP;
  onlyP << 4, 1, 1, 1;
  EXPECT_LT((onlyP - exact).determinant(), 0);
}

TEST(ConsiderSourceFilter, X6RecentNoOpConflictAndCompressedRejectionAreAtomic)
{
  auto limits = budget();
  limits.recentLedger = 1;
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  auto first = factor(initial.active, {}, "first");
  first.identity.sequence = RawId{1, 1, 0};
  first.identity.typedRaw = {{"first/raw", {2, 1, 0}}};
  apply(filter, first, 0.5);
  const auto before = *filter.activeCovariance();
  const auto revision = filter.revision();
  auto again = filter.proposeAppliedGain({first}, Eigen::MatrixXd::Constant(1, 1, 0.5));
  EXPECT_EQ(again.status(), Status::kNoOp);
  EXPECT_EQ(filter.commit(std::move(again)), Status::kNoOp);
  first.remainderVariance(0) = 2;
  EXPECT_EQ(filter.proposeAppliedGain({first}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(),
            Status::kIdentityConflict);
  EXPECT_EQ(filter.revision(), revision);
  expectMatrix(*filter.activeCovariance(), before);
  first.remainderVariance(0) = 1;
  auto next = factor(initial.active, {}, "second");
  next.identity.sequence = RawId{1, 2, 0};
  next.identity.typedRaw = {{"second/raw", {2, 2, 0}}};
  apply(filter, next, 0.5);
  const auto after = *filter.activeCovariance();
  const auto ledger = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({first}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(),
            Status::kLedgerCompacted);
  EXPECT_EQ(filter.ledgerIdentity(), ledger);
  expectMatrix(*filter.activeCovariance(), after);
}

TEST(ConsiderSourceFilter, X7FinalRetirementPreservesArbitraryRawHistory)
{
  ConsiderSourceFilter filter;
  const auto initial = prior();
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto z = source("old", "unrelated-raw-ordinal");
  z.sequence = RawId{3, 1, 0};
  z.typedRaw = {{z.rawAncestry.front(), {4, 99, 7}}};
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  ASSERT_EQ(filter.retire({"old"}, true, "last use verified"), Status::kOk);
  const auto ledger = filter.ledgerIdentity();
  EXPECT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kSourceHistoryUnavailable);
  auto input = factor(initial.active, {}, "needs-old");
  input.identity.rawAncestry = z.rawAncestry;
  input.identity.typedRaw = z.typedRaw;
  EXPECT_EQ(filter.proposeAppliedGain({input}, Eigen::MatrixXd::Ones(1, 1)).status(),
            Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.ledgerIdentity(), ledger);
}

TEST(ConsiderSourceFilter, X8DerivedRetirementMatchesNeverCreatedControl)
{
  const auto initial = prior();
  for (const bool derived : {true, false}) {
    ConsiderSourceFilter filter, control;
    ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
    ASSERT_EQ(control.initialize(initial, budget()), Status::kOk);
    auto live = source("live");
    auto old = source("old");
    if (derived)
      old.rawRoles = {{"map-raw", RawRole::kDerived}};
    ASSERT_EQ(filter.augment(live, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    ASSERT_EQ(control.augment(live, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    ASSERT_EQ(filter.augment(old, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    ASSERT_EQ(filter.retire({"old"}, true, "last derived use"), Status::kOk);
    auto input = factor(initial.active, {live}, "uses-live");
    input.identity.rawAncestry = {"map-raw"};
    input.identity.shared = {{"map-raw", "live"}};
    input.a.setOnes();
    auto candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.25));
    EXPECT_EQ(candidate.status(), derived ? Status::kOk : Status::kSourceHistoryUnavailable);
    if (derived) {
      ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
      apply(control, input, 0.25);
      expectMatrix(joint(filter, {live}), joint(control, {live}));
    }
  }
}

TEST(ConsiderSourceFilter, X9BudgetExcessAndKnowledgeStickAcrossLifecycle)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto input = factor(initial.active, {}, "one");
  auto copy = reuse(input, "echo", {8, 1, 0}, 1);
  apply(filter, input, 0.25, {copy});
  auto z = source("frame", "echo");
  z.sequence = RawId{9, 1, 0};
  z.typedRaw = input.identity.typedRaw;
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto next = factor(initial.active, {z}, "two");
  next.a.setOnes();
  next.identity.shared = {{"echo", "frame"}};
  copy = reuse(next, "echo", {8, 2, 0}, 1);
  const auto ledger = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({next}, Eigen::MatrixXd::Ones(1, 1), {copy}).status(),
            Status::kReuseBudgetExceeded);
  EXPECT_EQ(filter.ledgerIdentity(), ledger);
  const CopyNoiseCross explicitZero{{7, 1, 0}, {8, 1, 0}, Eigen::MatrixXd::Zero(1, 1)};
  EXPECT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), noise(2, 1), {explicitZero}), Status::kOk);
  EXPECT_EQ(filter.reset(Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  Eigen::MatrixXd block;
  EXPECT_EQ(filter.cross("frame", block), Status::kOk);
  EXPECT_EQ(filter.sourceCovariance("frame", "frame", block), Status::kOk);
  EXPECT_EQ(filter.retire({"frame"}, true, "final use"), Status::kOk);
  EXPECT_EQ(filter.knowledge(), MomentKnowledge::kUpperBound);
}

TEST(ConsiderSourceFilter, X1ProcessIndependenceIsCompleteAndExplicit)
{
  const auto initial = prior(2);
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto z = source("z");
  const Eigen::RowVector2d selector(0, 1);
  ASSERT_EQ(filter.augment(z, selector), Status::kOk);
  auto process = noise(3, 2);
  process.covariance(0, 0) = 2;
  process.priorCross.col(0).setOnes();
  const auto before = joint(filter, {z});
  const auto ledger = filter.ledgerIdentity();
  EXPECT_EQ(filter.propagate(Eigen::Matrix2d::Identity(), process), Status::kUnsupportedNoiseStructure);
  expectMatrix(joint(filter, {z}), before);
  EXPECT_EQ(filter.ledgerIdentity(), ledger);
  JointSourceFilter dense;
  ASSERT_EQ(dense.initialize(initial), Status::kOk);
  ASSERT_EQ(dense.augment(z, selector, Eigen::VectorXd::Zero(1), noise(2, 1)), Status::kOk);
  auto general = noise(3, 3);
  general.covariance(0, 0) = 2;
  general.priorCross.col(0).setOnes();
  ASSERT_EQ(
    dense.propagate(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), general, initial.active, {z.coordinates}),
    Status::kOk);
  EXPECT_DOUBLE_EQ(dense.moments()->covariance(0, 0), 5);
  EXPECT_DOUBLE_EQ(dense.moments()->covariance(0, 2), 1);
}

TEST(ConsiderSourceFilter, X11NearSingularAndAsymmetry)
{
  auto initial = prior(2);
  initial.covariance << 1, 1, 1, 1 + std::ldexp(1.0, -36);
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto z = source("z", "z/raw", 2);
  ASSERT_EQ(filter.augment(z, Eigen::Matrix2d::Identity()), Status::kOk);
  for (unsigned i = 0; i < 8; ++i) {
    auto input = factor(initial.active, {z}, "near" + std::to_string(i));
    input.h << 1, -1;
    input.a << -1, 1;
    const Eigen::Vector2d gain(0.125, 0.125);
    auto candidate = filter.proposeAppliedGain({input}, gain);
    ASSERT_EQ(candidate.status(), Status::kOk);
    ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
    // h*G = 0 keeps e0-e1 = z0-z1 exactly: every joint has rank 3, and its
    // entries are dyadic, so the reference is exact.
    const Eigen::Matrix2d p = initial.covariance;
    Eigen::Matrix4d expected;
    expected << p + Eigen::Matrix2d::Constant((i + 1) / 64.0), p, p, p;
    expectMatrix(joint(filter, {z}), expected);
    EXPECT_EQ(detail::auditCovariance(joint(filter, {z})), Status::kOk);
  }
  // Symmetric and indefinite beyond the arithmetic budget: smallest eigenvalue -2^-36.
  initial.covariance << 1, 1 + std::ldexp(1.0, -36), 1 + std::ldexp(1.0, -36), 1;
  ConsiderSourceFilter indefinite;
  EXPECT_EQ(indefinite.initialize(initial, budget()), Status::kInvalidCovariance);
  EXPECT_EQ(indefinite.activeCovariance(), nullptr);
  initial.covariance << 1, 1, 1, 1 + std::ldexp(1.0, -36);
  initial.covariance(0, 1) = std::nextafter(1.0, 2.0);
  ConsiderSourceFilter invalid;
  EXPECT_EQ(invalid.initialize(initial, budget()), Status::kInvalidCovariance);
  EXPECT_EQ(invalid.activeCovariance(), nullptr);
}

TEST(ConsiderSourceFilter, X12UnknownAmplitudeSensitivityMatchesExplicitConstantSource)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto calibration = source("calibration", "cal/raw");
  ASSERT_EQ(filter.registerSensitivity(calibration, Eigen::MatrixXd::Zero(1, 1)), Status::kOk);
  auto query = factor(initial.active, {}, "query");
  apply(filter, query, 0.5, {}, {{"query", "calibration", Eigen::MatrixXd::Ones(1, 1)}});
  auto z = source("frame");
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto support = factor(initial.active, {z}, "support");
  support.a.setConstant(-1);
  apply(filter, support, 0.5, {}, {{"support", "calibration", Eigen::MatrixXd::Ones(1, 1)}});
  Eigen::MatrixXd sensitivity, sourceSensitivity;
  ASSERT_EQ(filter.sensitivity("calibration", sensitivity), Status::kOk);
  ASSERT_EQ(filter.sourceSensitivity("frame", "calibration", sourceSensitivity), Status::kOk);
  EXPECT_DOUBLE_EQ(sensitivity(0, 0), -1);
  EXPECT_DOUBLE_EQ(sourceSensitivity(0, 0), -0.5);
  Eigen::Vector2d complete(sensitivity(0, 0), sourceSensitivity(0, 0));
  Eigen::Matrix2d expected;
  expected << 1, 5.0 / 8.0, 5.0 / 8.0, 9.0 / 16.0;
  expectMatrix(joint(filter, {z}) + 0.25 * complete * complete.transpose(), expected);
  EXPECT_EQ(filter.sensitivitySources().size(), 1U);  // Unknown amplitude remains labelled.

  JointSourceFilter dense;
  auto explicitPrior = initial;
  explicitPrior.sources = {calibration};
  explicitPrior.mean = Eigen::Vector2d::Zero();
  explicitPrior.covariance = (Eigen::Matrix2d() << 1, 0, 0, 0.25).finished();
  ASSERT_EQ(dense.initialize(explicitPrior), Status::kOk);
  Eigen::Matrix2d first;
  first << 0.5, -0.5, 0, 1;
  auto fresh = noise(2, 2);
  fresh.covariance(0, 0) = 0.25;
  ASSERT_EQ(dense.propagate(first, Eigen::Vector2d::Zero(), fresh, initial.active, {calibration.coordinates}),
            Status::kOk);
  ASSERT_EQ(dense.augment(z, Eigen::RowVector2d(1, 0), Eigen::VectorXd::Zero(1), noise(2, 1)), Status::kOk);
  Eigen::Matrix3d second = Eigen::Matrix3d::Identity();
  second.row(0) << 0.5, -0.5, 0.5;
  auto secondFresh = noise(3, 3);
  secondFresh.covariance(0, 0) = 0.25;
  ASSERT_EQ(dense.propagate(
              second, Eigen::Vector3d::Zero(), secondFresh, initial.active, {calibration.coordinates, z.coordinates}),
            Status::kOk);
  Eigen::Matrix2d retained;
  retained << dense.moments()->covariance(0, 0), dense.moments()->covariance(0, 2), dense.moments()->covariance(2, 0),
    dense.moments()->covariance(2, 2);
  expectMatrix(retained, expected);
}

TEST(ConsiderSourceFilter, X12ProcessSensitivityIsExplicitThroughCloneResetAndUpdate)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  const auto calibration = source("cal", "cal/raw");
  ASSERT_EQ(filter.registerSensitivity(calibration, Eigen::MatrixXd::Zero(1, 1)), Status::kOk);
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), noise(1, 1)), Status::kUnknownMoment);
  EXPECT_EQ(filter.ledgerIdentity(), before);
  ASSERT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), noise(1, 1), {}, {{"cal", Eigen::MatrixXd::Ones(1, 1)}}),
            Status::kOk);
  const auto z = source("frame");
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  ASSERT_EQ(filter.reset(Eigen::MatrixXd::Constant(1, 1, 2)), Status::kOk);
  auto input = factor(initial.active, {z}, "support");
  input.a.setConstant(-1);
  apply(filter, input, 0.5, {}, {{"support", "cal", Eigen::MatrixXd::Zero(1, 1)}});
  Eigen::MatrixXd activeLoading, sourceLoading;
  ASSERT_EQ(filter.sensitivity("cal", activeLoading), Status::kOk);
  ASSERT_EQ(filter.sourceSensitivity("frame", "cal", sourceLoading), Status::kOk);
  expectMatrix(activeLoading, Eigen::MatrixXd::Constant(1, 1, 1.5));
  expectMatrix(sourceLoading, Eigen::MatrixXd::Ones(1, 1));
  const Eigen::Vector2d loading(activeLoading(0, 0), sourceLoading(0, 0));
  // Independent primitives: e0~(0,1), cal~(0,1/4), eps~(0,1).
  // z=e0+cal; e+=3/2*(e0+cal)-eps/2. No amplitude enters the random P/K/C.
  Eigen::Matrix2d expected;
  expected << 49.0 / 16.0, 15.0 / 8.0, 15.0 / 8.0, 5.0 / 4.0;
  expectMatrix(joint(filter, {z}) + 0.25 * loading * loading.transpose(), expected);
}

TEST(ConsiderSourceFilter, IndependentInformationReducesPAndCommitIsSingleUse)
{
  ConsiderSourceFilter filter, other;
  const auto initial = prior();
  const auto input = factor(initial.active, {}, "fresh");
  EXPECT_EQ(filter.activeCovariance(), nullptr);
  EXPECT_EQ(filter.reset(Eigen::MatrixXd::Ones(1, 1)), Status::kUnknownMoment);
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  ASSERT_EQ(other.initialize(initial, budget()), Status::kOk);
  auto first = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.5));
  auto stale = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.5));
  // commit takes an rvalue reference and retains the object for stale rejection.
  const auto commitFirst = [&first](ConsiderSourceFilter & target) {
    return target.commit(std::move(first));
  };
  EXPECT_EQ(commitFirst(other), Status::kStaleProposal);
  EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 1);
  ASSERT_EQ(commitFirst(filter), Status::kOk);
  EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 0.5);
  EXPECT_EQ(commitFirst(filter), Status::kStaleProposal);
  EXPECT_EQ(filter.commit(std::move(stale)), Status::kStaleProposal);
}

TEST(ConsiderSourceFilter, RejectedBatchDoesNotConsumeAnyPrefix)
{
  ConsiderSourceFilter filter;
  const auto initial = prior();
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto one = factor(initial.active, {}, "one"), two = factor(initial.active, {}, "two");
  two.identity.remainderKey = one.identity.remainderKey;
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({one, two}, Eigen::MatrixXd::Constant(1, 2, 0.25)).status(),
            Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(filter.ledgerIdentity(), before);
  EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 1);
  apply(filter, one, 0.5);
}

TEST(ConsiderSourceFilter, NoOpCandidatesRetainAndReleaseTheirMemoryCharge)
{
  ConsiderSourceFilter filter;
  const auto initial = prior();
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  const auto input = factor(initial.active, {}, "one");
  apply(filter, input, 0.5);
  const auto before = filter.memoryBytes();
  const auto ledger = filter.ledgerIdentity();
  {
    auto noOp = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.5));
    ASSERT_EQ(noOp.status(), Status::kNoOp);
    EXPECT_GT(filter.memoryBytes(), before);
    EXPECT_EQ(filter.commit(std::move(noOp)), Status::kNoOp);
    EXPECT_GT(filter.memoryBytes(), before);
    EXPECT_EQ(filter.ledgerIdentity(), ledger);
  }
  EXPECT_EQ(filter.memoryBytes(), before);
}

TEST(ConsiderSourceFilter, X4SameFrameBatchUsesOneSharedCopy)
{
  auto initial = prior();
  initial.covariance.setZero();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto one = factor(initial.active, {}, "one"), two = factor(initial.active, {}, "two");
  auto copy = reuse(one, "echo", {8, 1, 0}, 1);
  reuse(two, "echo", {8, 1, 0}, 1);
  copy.factors.push_back({"two", Eigen::MatrixXd::Ones(1, 1)});
  auto candidate = filter.proposeAppliedGain({one, two}, Eigen::MatrixXd::Constant(1, 2, -1), {copy});
  ASSERT_EQ(candidate.status(), Status::kOk);
  ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 6);
  EXPECT_EQ(filter.reuseCount({7, 1, 0}), 1U);
}

TEST(ConsiderSourceFilter, X4LateBudgetCannotRepairConsumedPrefix)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto query = factor(initial.active, {}, "query");
  query.identity.rawAncestry = {"echo"};
  query.identity.typedRaw = {{"echo", {7, 1, 0}}};
  apply(filter, query, 0.5);
  auto z = source("frame", "echo");
  z.sequence = RawId{9, 1, 0};
  z.typedRaw = query.identity.typedRaw;
  ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  auto support = factor(initial.active, {z}, "support");
  support.a.setOnes();
  support.identity.shared = {{"echo", "frame"}};
  auto copy = reuse(support, "echo", {8, 2, 0}, 2);
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({support}, Eigen::MatrixXd::Constant(1, 1, 0.5), {copy}).status(),
            Status::kSourceHistoryUnavailable);
  EXPECT_EQ(filter.ledgerIdentity(), before);
}

TEST(ConsiderSourceFilter, X6TypedRawCannotChangeNamesOrReturnAsFresh)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto first = factor(initial.active, {}, "first");
  first.identity.typedRaw = {{"first/raw", {2, 10, 0}}};
  apply(filter, first, 0.5);
  auto old = factor(initial.active, {}, "old");
  old.identity.typedRaw = {{"old/raw", {2, 9, 0}}};
  EXPECT_EQ(filter.proposeAppliedGain({old}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(), Status::kLedgerCompacted);
  old.identity.typedRaw = {{"old/raw", {2, 10, 0}}};
  EXPECT_EQ(filter.proposeAppliedGain({old}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(), Status::kIdentityConflict);
  old.identity.rawAncestry = {"first/raw"};
  old.identity.typedRaw.clear();
  EXPECT_EQ(filter.proposeAppliedGain({old}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(), Status::kIdentityConflict);
}

TEST(ConsiderSourceFilter, X6FirstReuseRegistrationCannotBypassRawWatermark)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto first = factor(initial.active, {}, "first");
  first.identity.typedRaw = {{"first/raw", {7, 10, 0}}};
  apply(filter, first, 0.5);
  auto old = factor(initial.active, {}, "old");
  auto copy = reuse(old, "echo", {8, 11, 0}, 1);
  // reuse() names ordinal 1: it is not fresh and has no earlier budget record.
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({old}, Eigen::MatrixXd::Constant(1, 1, 0.5), {copy}).status(),
            Status::kLedgerCompacted);
  EXPECT_EQ(filter.ledgerIdentity(), before);
  EXPECT_EQ(filter.reuseCount(copy.use.raw), 0U);
}

TEST(ConsiderSourceFilter, X6CompactionFollowsTheWholeTypedBatch)
{
  const auto initial = prior();
  auto limits = budget();
  limits.recentLedger = 1;
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  std::vector<LinearFactor> inputs;
  constexpr std::uint32_t kFactors = 3;
  for (std::uint32_t sub = 0; sub < kFactors; ++sub) {
    inputs.push_back(factor(initial.active, {}, "factor" + std::to_string(sub)));
    inputs.back().identity.sequence = RawId{10, 1, sub};
    inputs.back().identity.typedRaw = {{inputs.back().identity.rawAncestry.front(), {20, 1, sub}}};
  }
  auto candidate = filter.proposeAppliedGain(inputs, Eigen::MatrixXd::Constant(1, kFactors, 0.25));
  ASSERT_EQ(candidate.status(), Status::kOk);
  ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  EXPECT_DOUBLE_EQ((*filter.activeCovariance())(0, 0), 0.25);
  EXPECT_EQ(filter.proposeAppliedGain({inputs.back()}, Eigen::MatrixXd::Constant(1, 1, 0.25)).status(), Status::kNoOp);
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({inputs.front()}, Eigen::MatrixXd::Constant(1, 1, 0.25)).status(),
            Status::kLedgerCompacted);
  EXPECT_EQ(filter.ledgerIdentity(), before);
}

TEST(ConsiderSourceFilter, X6OpaqueHistoryCannotEvictTheNewestTypedPayload)
{
  const auto initial = prior();
  auto limits = budget();
  limits.recentLedger = 1;
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  apply(filter, factor(initial.active, {}, "opaque"), 0.5);
  auto input = factor(initial.active, {}, "typed");
  input.identity.sequence = RawId{1, 1, 0};
  input.identity.typedRaw = {{"typed/raw", {2, 1, 0}}};
  const auto before = filter.ledgerIdentity();
  EXPECT_EQ(filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, 0.5)).status(),
            Status::kModelCapacityExceeded);
  EXPECT_EQ(filter.ledgerIdentity(), before);
}

TEST(ConsiderSourceFilter, X7TypedRetirementHasNoPerSource128TombstoneCap)
{
  ConsiderSourceFilter filter;
  const auto initial = prior();
  auto limits = budget();
  limits.sourceDof = 1;
  limits.liveSources = 1;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  constexpr std::uint64_t kRetirements = kMaxLedger + 1;
  for (std::uint64_t i = 1; i <= kRetirements; ++i) {
    auto z = source("z" + std::to_string(i), "raw" + std::to_string(i));
    z.sequence = RawId{1, i, 0};
    z.typedRaw = {{z.rawAncestry.front(), {2, i + kRetirements, 0}}};
    ASSERT_EQ(filter.augment(z, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
    ASSERT_EQ(filter.retire({z.key}, true, "last use verified"), Status::kOk);
  }
  auto reused = source("changed-key", "raw1");
  reused.sequence = RawId{1, kRetirements + 1, 0};
  reused.typedRaw = {{"raw1", {2, kRetirements + 1, 0}}};
  EXPECT_EQ(filter.augment(reused, Eigen::MatrixXd::Ones(1, 1)), Status::kSourceHistoryUnavailable);
}

TEST(ConsiderSourceFilter, BudgetExtendsCumulativeDofAndRowsWithoutChangingV1)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  auto first = source("first", "first/raw", kMaxSourceDof);
  auto second = source("second", "second/raw");
  ASSERT_EQ(filter.augment(first, Eigen::MatrixXd::Ones(kMaxSourceDof, 1)), Status::kOk);
  ASSERT_EQ(filter.augment(second, Eigen::MatrixXd::Ones(1, 1)), Status::kOk);
  constexpr Eigen::Index kRows = kMaxRows + 1;
  auto input = factor(initial.active, {first, second}, "wide");
  input.h = Eigen::MatrixXd::Ones(kRows, 1);
  input.a = Eigen::MatrixXd::Zero(kRows, kMaxSourceDof + 1);
  input.observation = Eigen::VectorXd::Zero(kRows);
  input.remainderMean = Eigen::VectorXd::Zero(kRows);
  input.remainderVariance = Eigen::VectorXd::Ones(kRows);
  input.rowUnits = std::vector<std::string>(kRows, "m");
  auto candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Zero(1, kRows));
  ASSERT_EQ(candidate.status(), Status::kOk);
  EXPECT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  EXPECT_EQ(kMaxSourceDof, 32);
  EXPECT_EQ(kMaxRows, 64);
  EXPECT_EQ(kMaxLedger, 128U);
}

TEST(ConsiderSourceFilter, UnknownAndBudgetRejectionsAreAtomic)
{
  const auto initial = prior();
  ConsiderSourceFilter filter;
  auto limits = budget();
  limits.sourceDof = 1;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  const auto before = filter.ledgerIdentity();
  auto missing = noise(1, 1);
  missing.knowledge = MomentKnowledge::kUnknown;
  EXPECT_EQ(filter.propagate(Eigen::MatrixXd::Ones(1, 1), missing), Status::kUnknownMoment);
  EXPECT_EQ(filter.augment(source("wide", "wide/raw", 2), Eigen::MatrixXd::Ones(2, 1)), Status::kModelCapacityExceeded);
  auto invalid = factor(initial.active, {}, "invalid");
  invalid.remainderVariance(0) = 0;
  EXPECT_EQ(filter.proposeAppliedGain({invalid}, Eigen::MatrixXd::Ones(1, 1)).status(),
            Status::kUnsupportedNoiseStructure);
  EXPECT_EQ(filter.ledgerIdentity(), before);
  ConsiderSourceFilter tooSmall;
  limits.memoryBytes = 1;
  EXPECT_EQ(tooSmall.initialize(initial, limits), Status::kModelCapacityExceeded);
  EXPECT_EQ(tooSmall.activeCovariance(), nullptr);
}

TEST(ConsiderSourceFilter, ConstructiveAuditHandlesInteriorZeroPivotsWithoutClipping)
{
  Eigen::Matrix3d supported;
  supported << 5.0 / 16.0, 0.5, 0.5, 0.5, 1, 1, 0.5, 1, 1;
  const Eigen::LDLT<Eigen::Matrix3d> eigen(supported);
  ::testing::Test::RecordProperty("eigen_ldlt_info", static_cast<int>(eigen.info()));
  EXPECT_GE(eigen.vectorD().minCoeff(), 0);
  EXPECT_EQ(detail::auditCovariance(supported), Status::kOk);
  Eigen::Matrix3d invalid = Eigen::Matrix3d::Zero();
  invalid(0, 0) = 1;
  invalid(1, 2) = invalid(2, 1) = std::numeric_limits<double>::epsilon();
  EXPECT_EQ(detail::auditCovariance(invalid), Status::kInvalidCovariance);
  invalid.setIdentity();
  invalid(2, 2) = -std::numeric_limits<double>::epsilon();
  EXPECT_EQ(detail::auditCovariance(invalid), Status::kInvalidCovariance);
}

// The stored joint of a valid recursion with a deterministic relation is the
// rounding of a singular matrix: its smallest eigenvalue is zero up to a few
// ulp of either sign. X1 with non-dyadic numbers makes every product inexact.
TEST(ConsiderSourceFilter, X1NonDyadicSequenceKeepsDeterministicSupport)
{
  auto initial = prior(2);
  ConsiderSourceFilter filter;
  JointSourceFilter reference;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  ASSERT_EQ(reference.initialize(initial), Status::kOk);
  const Eigen::Matrix2d phi = (Eigen::Matrix2d() << 1, 0.3, 0, 1).finished();
  ASSERT_EQ(filter.propagate(phi, noise(2, 2, 0.1)), Status::kOk);
  ASSERT_EQ(reference.propagate(phi, Eigen::Vector2d::Zero(), noise(2, 2, 0.1), initial.active, {}), Status::kOk);
  auto z = source("z", "z/raw", 2);
  ASSERT_EQ(filter.augment(z, Eigen::Matrix2d::Identity()), Status::kOk);
  ASSERT_EQ(reference.augment(z, Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(), noise(2, 2)), Status::kOk);
  auto input = factor(initial.active, {z}, "first");
  input.h << 1, 0.7;
  input.a << -0.3, 0.45;
  const Eigen::Vector2d gain(0.3, -0.1);
  auto candidate = filter.proposeAppliedGain({input}, gain);
  ASSERT_EQ(candidate.status(), Status::kOk);
  ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<2, 2>() -= gain * input.h;
  transform.topRightCorner<2, 2>() = -gain * input.a;
  auto process = noise(4, 4);
  process.covariance.topLeftCorner<2, 2>() = gain * gain.transpose();
  ASSERT_EQ(reference.propagate(transform, Eigen::Vector4d::Zero(), process, initial.active, {z.coordinates}),
            Status::kOk);
  expectMatrix(joint(filter, {z}), reference.moments()->covariance);
}

// h*G = 0 keeps e0-e1 = 0.99^k (z0-z1) through every update and propagation,
// with no process noise to remove it: rounding in that null direction only
// accumulates. 256 consecutive steps stay far inside the arithmetic budget.
TEST(ConsiderSourceFilter, PersistentDeterministicRelationStaysWithinArithmeticBudget)
{
  auto initial = prior(2);
  initial.covariance << 1.3, 0.4, 0.4, 0.9;
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, budget()), Status::kOk);
  const auto z = source("z", "z/raw", 2);
  ASSERT_EQ(filter.augment(z, Eigen::Matrix2d::Identity()), Status::kOk);
  for (std::uint64_t i = 0; i < 256; ++i) {
    ASSERT_EQ(filter.propagate(0.99 * Eigen::Matrix2d::Identity(), noise(4, 2)), Status::kOk);
    auto input = factor(initial.active, {z}, "step" + std::to_string(i));
    input.identity.sequence = RawId{1, i + 1, 0};
    input.identity.typedRaw = {{input.identity.rawAncestry.front(), {2, i + 1, 0}}};
    input.h << 1, -1;
    input.a << -1, 1;
    auto candidate = filter.proposeAppliedGain({input}, Eigen::Vector2d(0.1, 0.1));
    ASSERT_EQ(candidate.status(), Status::kOk) << "step " << i;
    ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  }
  const Eigen::MatrixXd value = joint(filter, {z});
  // e0-e1 - 0.99^256 (z0-z1) has zero variance.
  const double decay = std::pow(0.99, 256);
  const Eigen::Vector4d null(1, -1, -decay, decay);
  const double error = std::abs(null.dot(value * null));
  const double tolerance = kTolerance * value.cwiseAbs().maxCoeff();
  ::testing::Test::RecordProperty("null_direction_actual", numbers(Eigen::MatrixXd::Constant(1, 1, error)));
  ::testing::Test::RecordProperty("null_direction_expected", "0");
  ::testing::Test::RecordProperty("null_direction_tolerance", numbers(Eigen::MatrixXd::Constant(1, 1, tolerance)));
  ::testing::Test::RecordProperty("null_direction_metric", "max_abs");
  EXPECT_LE(error, tolerance);
}

TEST(ConsiderSourceFilter, AuditAcceptsSingularSupportWithinArithmeticBudget)
{
  // Exact rational X1 joint after the update: rank 3, null vector [1, 2, -1, -2].
  Eigen::Matrix4d support;
  support << 475.0 / 512, 181.0 / 1024, 35.0 / 32, 3.0 / 32, 181.0 / 1024, 3515.0 / 2048, 45.0 / 64, 93.0 / 64,
    35.0 / 32, 45.0 / 64, 1.5, 0.5, 3.0 / 32, 93.0 / 64, 0.5, 1.25;
  EXPECT_LE((support * Eigen::Vector4d(1, 2, -1, -2)).cwiseAbs().maxCoeff(), 0.0);
  // The budget is relative to maxAbs: a change of units changes no verdict.
  for (const double unit : {1.0, 1e-12, 1e12})
    EXPECT_EQ(detail::auditCovariance(unit * support), Status::kOk) << unit;
  // Rounded rank one: 0.1, 0.3 and 0.7 have no exact products.
  const Eigen::Vector3d direction(0.1, 0.3, 0.7);
  EXPECT_EQ(detail::auditCovariance(direction * direction.transpose()), Status::kOk);
  // Largest sampled window, 23 + 32: one variable repeated 55 times, and 27 exact clone pairs.
  EXPECT_EQ(detail::auditCovariance(Eigen::MatrixXd::Constant(55, 55, 0.3)), Status::kOk);
  Eigen::MatrixXd pairs = Eigen::MatrixXd::Identity(55, 55);
  for (Eigen::Index i = 0; i + 1 < 55; i += 2)
    pairs(i, i + 1) = pairs(i + 1, i) = 1;
  EXPECT_EQ(detail::auditCovariance(pairs), Status::kOk);
  // Known zero covariance is deterministic support.
  EXPECT_EQ(detail::auditCovariance(Eigen::Matrix3d::Zero()), Status::kOk);
  Eigen::Matrix3d partial = Eigen::Matrix3d::Zero();
  partial(1, 1) = 2;
  EXPECT_EQ(detail::auditCovariance(partial), Status::kOk);
}

TEST(ConsiderSourceFilter, AuditRejectsIndefiniteBeyondArithmeticBudget)
{
  const double eps = std::numeric_limits<double>::epsilon();
  Eigen::Matrix2d invalid;
  invalid << 1, 2, 2, 1;
  EXPECT_EQ(detail::auditCovariance(invalid), Status::kInvalidCovariance);
  // Smallest eigenvalue -2^-36 * maxAbs at every scale.
  invalid << 1, 1 + std::ldexp(1.0, -36), 1 + std::ldexp(1.0, -36), 1;
  for (const double unit : {1.0, 1e-12, 1e12})
    EXPECT_EQ(detail::auditCovariance(unit * invalid), Status::kInvalidCovariance) << unit;
  // Smallest eigenvalue -1024 eps, eight times the n = 2 budget of 128 eps.
  invalid << 1, 1 + 1024 * eps, 1 + 1024 * eps, 1;
  EXPECT_EQ(detail::auditCovariance(invalid), Status::kInvalidCovariance);
  // The X1 joint with one variance lowered by 1e-9.
  Eigen::Matrix4d lowered;
  lowered << 475.0 / 512, 181.0 / 1024, 35.0 / 32, 3.0 / 32, 181.0 / 1024, 3515.0 / 2048, 45.0 / 64, 93.0 / 64,
    35.0 / 32, 45.0 / 64, 1.5, 0.5, 3.0 / 32, 93.0 / 64, 0.5, 1.25;
  lowered(0, 0) -= 1e-9;
  EXPECT_EQ(detail::auditCovariance(lowered), Status::kInvalidCovariance);
  // Largest sampled window: 26 exact clone pairs and one pair with correlation
  // 1 + 2^-36. The budget is 3520 eps and the error bound 1541 eps.
  Eigen::MatrixXd pairs = Eigen::MatrixXd::Identity(55, 55);
  for (Eigen::Index i = 0; i + 1 < 55; i += 2)
    pairs(i, i + 1) = pairs(i + 1, i) = 1;
  pairs(52, 53) = pairs(53, 52) = 1 + std::ldexp(1.0, -36);
  EXPECT_EQ(detail::auditCovariance(pairs), Status::kInvalidCovariance);
  Eigen::Matrix2d nonFinite = Eigen::Matrix2d::Identity();
  nonFinite(1, 1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(detail::auditCovariance(nonFinite), Status::kNonFinite);
  nonFinite(1, 1) = std::numeric_limits<double>::infinity();
  EXPECT_EQ(detail::auditCovariance(nonFinite), Status::kNonFinite);
  Eigen::Matrix2d asymmetric = Eigen::Matrix2d::Identity();
  asymmetric(0, 1) = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(detail::auditCovariance(asymmetric), Status::kInvalidCovariance);
}

// The budget is where the verdict changes, not an endorsement of indefinite
// input: a symmetric matrix that is indefinite within it is the same kind of
// double matrix a valid singular joint rounds to, and no judgement of the stored
// numbers can reject one and keep the other. [[1, 1 + d], [1 + d, 1]] has the
// smallest eigenvalue -d exactly. Acceptance is certain above -(budget - c) and
// rejection below -(budget + c), with c the factorization error bound.
TEST(ConsiderSourceFilter, AuditVerdictChangesAtTheArithmeticBudget)
{
  const double eps = std::numeric_limits<double>::epsilon();
  const auto pair = [](double excess) {
    Eigen::Matrix2d value;
    value << 1, 1 + excess, 1 + excess, 1;
    return value;
  };
  EXPECT_LT(pair(eps).determinant(), 0);
  EXPECT_EQ(detail::auditCovariance(pair(eps)), Status::kOk);
  // n = 2: budget 128 eps, c = 4 eps.
  EXPECT_EQ(detail::auditCovariance(pair(112 * eps)), Status::kOk);
  EXPECT_EQ(detail::auditCovariance(pair(144 * eps)), Status::kInvalidCovariance);
  // n = 55, the largest sampled window: budget 3520 eps, c = 1541 eps.
  const auto window = [&pair](double excess) {
    Eigen::MatrixXd value = Eigen::MatrixXd::Identity(55, 55);
    value.topLeftCorner(2, 2) = pair(excess);
    return value;
  };
  EXPECT_EQ(detail::auditCovariance(window(1900 * eps)), Status::kOk);
  EXPECT_EQ(detail::auditCovariance(window(5200 * eps)), Status::kInvalidCovariance);
}

TEST(ConsiderSourceFilter, AllSourcesReferencedKeepsUpdateWorkspaceLinear)
{
  constexpr Eigen::Index kBlocks = 32, kBlockDof = 32, kDof = kBlocks * kBlockDof;
  constexpr std::size_t kLimit = static_cast<std::size_t>(22U) * 1024U * 1024U;
  auto limits = budget();
  limits.sourceDof = kDof;
  limits.liveSources = kBlocks;
  limits.memoryBytes = kLimit;
  const auto initial = prior();
  ConsiderSourceFilter filter;
  ASSERT_EQ(filter.initialize(initial, limits), Status::kOk);
  std::vector<Source> sources;
  for (Eigen::Index i = 0; i < kBlocks; ++i) {
    sources.push_back(source("block" + std::to_string(i), "raw" + std::to_string(i), kBlockDof));
    ASSERT_EQ(filter.augment(sources.back(), Eigen::MatrixXd::Ones(kBlockDof, 1)), Status::kOk);
  }
  auto input = factor(initial.active, sources, "all-sources");
  input.a.setOnes();
  const double gain = 1.0 / static_cast<double>(kDof);
  auto candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Constant(1, 1, gain));
  ASSERT_EQ(candidate.status(), Status::kOk);
  EXPECT_LT(filter.memoryBytes(), kLimit);
  ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  // Every clone is exactly the original scalar e; e+ = -e/1024 - eps/1024.
  expectMatrix(*filter.activeCovariance(), Eigen::MatrixXd::Constant(1, 1, 2 * gain * gain));
  Eigen::MatrixXd cross;
  ASSERT_EQ(filter.cross("block0", cross), Status::kOk);
  expectMatrix(cross, Eigen::MatrixXd::Constant(1, kBlockDof, -gain));
}

template<class Function>
void measure(const std::string & entry, Function && operation, const ConsiderSourceFilter & filter)
{
  const auto start = std::chrono::steady_clock::now();
  operation();
  const auto ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  struct rusage usage{};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &usage), 0);
  std::cout << "B1B_ENTRY entry=" << entry << " ns=" << ns << " accounted_bytes=" << filter.memoryBytes()
            << " process_peak_kib=" << usage.ru_maxrss << '\n';
}

TEST(ConsiderSourceFilter, DISABLED_FullBudgetSingleEntryTimings)
{
  auto limits = budget();
  limits.sourceDof = kMaxSourceDof;
  limits.rowsPerBatch = kMaxRows;
  const auto initial = prior(kMaxActiveDof);
  ConsiderSourceFilter filter;
  measure(
    "initialize_active23_empty_sources", [&] { ASSERT_EQ(filter.initialize(initial, limits), Status::kOk); }, filter);
  auto first = source("first", "first/raw", kMaxSourceDof - 1);
  Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(kMaxSourceDof - 1, kMaxActiveDof);
  for (Eigen::Index i = 0; i < selector.rows(); ++i)
    selector(i, i % kMaxActiveDof) = 1;
  ASSERT_EQ(filter.augment(first, selector), Status::kOk);
  for (std::size_t i = 0; i < kMaxLedger; ++i) {
    auto input = factor(initial.active, {first}, "ledger" + std::to_string(i));
    input.identity.sequence = RawId{1, i + 1, 0};
    input.identity.typedRaw = {{input.identity.rawAncestry.front(), {2, i + 1, 0}}};
    auto candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Zero(kMaxActiveDof, 1));
    ASSERT_EQ(candidate.status(), Status::kOk);
    ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk);
  }
  const auto last = source("last", "last/raw");
  Eigen::MatrixXd lastSelector = Eigen::MatrixXd::Zero(1, kMaxActiveDof);
  lastSelector(0, 0) = 1;
  measure(
    "augment_31_to_32_source_dof_ledger128",
    [&] { ASSERT_EQ(filter.augment(last, lastSelector), Status::kOk); },
    filter);
  auto input = factor(initial.active, {first, last}, "full");
  input.identity.sequence = RawId{1, kMaxLedger + 1, 0};
  input.identity.typedRaw = {{"full/raw", {2, kMaxLedger + 1, 0}}};
  input.h = Eigen::MatrixXd::Ones(kMaxRows, kMaxActiveDof);
  input.a = Eigen::MatrixXd::Zero(kMaxRows, kMaxSourceDof);
  input.a.col(0).setOnes();
  input.observation = Eigen::VectorXd::Zero(kMaxRows);
  input.remainderMean = Eigen::VectorXd::Zero(kMaxRows);
  input.remainderVariance = Eigen::VectorXd::Ones(kMaxRows);
  input.rowUnits = std::vector<std::string>(kMaxRows, "m");
  ConsiderSourceFilter::Candidate candidate;
  measure(
    "propose_active23_source32_rows64_ledger128",
    [&] {
      candidate = filter.proposeAppliedGain({input}, Eigen::MatrixXd::Zero(kMaxActiveDof, kMaxRows));
      ASSERT_EQ(candidate.status(), Status::kOk);
    },
    filter);
  measure(
    "commit_active23_source32_rows64_ledger128",
    [&] { ASSERT_EQ(filter.commit(std::move(candidate)), Status::kOk); },
    filter);
  measure(
    "propagate_active23_source32_ledger128",
    [&] {
      ASSERT_EQ(filter.propagate(Eigen::MatrixXd::Identity(kMaxActiveDof, kMaxActiveDof),
                                 noise(kMaxActiveDof + kMaxSourceDof, kMaxActiveDof)),
                Status::kOk);
    },
    filter);
  measure(
    "reset_active23_source32_ledger128",
    [&] { ASSERT_EQ(filter.reset(Eigen::MatrixXd::Identity(kMaxActiveDof, kMaxActiveDof)), Status::kOk); },
    filter);
  Eigen::MatrixXd block;
  measure("cross_active23_source32_ledger128", [&] { ASSERT_EQ(filter.cross("first", block), Status::kOk); }, filter);
  measure(
    "sourceCovariance_active23_source32_ledger128",
    [&] { ASSERT_EQ(filter.sourceCovariance("first", "last", block), Status::kOk); },
    filter);
  const auto calibration = source("cal", "cal/raw");
  measure(
    "registerSensitivity_active23_source32_ledger128",
    [&] {
      ASSERT_EQ(filter.registerSensitivity(calibration, Eigen::MatrixXd::Zero(kMaxActiveDof + kMaxSourceDof, 1)),
                Status::kOk);
    },
    filter);
  measure(
    "sensitivity_active23_source32_ledger128",
    [&] { ASSERT_EQ(filter.sensitivity("cal", block), Status::kOk); },
    filter);
  measure(
    "sourceSensitivity_active23_source32_ledger128",
    [&] { ASSERT_EQ(filter.sourceSensitivity("first", "cal", block), Status::kOk); },
    filter);
  measure(
    "retire_32_to_31_source_dof_ledger128",
    [&] { ASSERT_EQ(filter.retire({"last"}, true, "last use"), Status::kOk); },
    filter);
}

}  // namespace
}  // namespace fastlio::joint_source

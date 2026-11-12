#include <Eigen/Geometry>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "lost_recovery_policy.hpp"
#include "rebuild_relevel.hpp"

namespace
{
using namespace fast_lio;

class LostRecovery : public ::testing::Test
{
protected:
  LostRecoveryParams params{true, 200};
  LostRecoveryState lost;
  RecoveryState recovery;

  void SetUp() override
  {
    recovery.trust = RecoveryTrust::kUnanchored;
    recovery.cause = RecoveryCause::kDeadline;
    recovery.episode = 7;
    recovery.map_resets = 3;
    recovery.reference_initialized_once = true;
  }

  bool scan(std::uint64_t seq, std::int64_t begin, std::int64_t end, bool match = true, std::uint64_t gen = 4)
  {
    return observeLostRecoveryCandidate(&lost, &recovery, seq, begin, end, match, gen, params);
  }

  void begin() { beginLostRecoveryCandidate(&lost, 4, 1, 0, params); }
};

TEST_F(LostRecovery, DisabledOperationsPreserveAnExistingCandidateAndLatch)
{
  begin();
  params.enabled = false;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
  beginLostRecoveryCandidate(&lost, 9, 9, 999, params);
  abandonLostRecovery(&lost, params);
  EXPECT_FALSE(scan(2, 100, 300));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 300, params));
  EXPECT_TRUE(lost.candidate);
  EXPECT_EQ(lost.candidate_begin_ns, 0);
  EXPECT_EQ(lost.candidate_generation, 4u);
  EXPECT_EQ(lost.candidate_last_scan, 1u);
  EXPECT_EQ(lost.attempts, 1u);
  EXPECT_EQ(lost.coordinate_epoch, 1u);
  EXPECT_EQ(lost.epochs_committed, 0u);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
  EXPECT_EQ(recovery.cause, RecoveryCause::kDeadline);
}

TEST_F(LostRecovery, OnlyAnEnabledLatchArmsRecovery)
{
  EXPECT_TRUE(lostRecoveryArmed(recovery, params));
  recovery.trust = RecoveryTrust::kPending;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
  recovery.trust = RecoveryTrust::kLocalQualified;
  EXPECT_FALSE(lostRecoveryArmed(recovery, params));
}

TEST_F(LostRecovery, StableBoundaryCommitsOnceAndPreservesHistory)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 99));
  EXPECT_FALSE(scan(2, 100, 199));
  EXPECT_TRUE(scan(3, 199, 200));
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(recovery.cause, RecoveryCause::kStable);
  EXPECT_EQ(recovery.last_supported_end_ns, 200);
  EXPECT_EQ(recovery.stable_begin_ns, -1);
  EXPECT_EQ(recovery.episode, 7u);
  EXPECT_EQ(recovery.map_resets, 3u);
  EXPECT_TRUE(recovery.reference_initialized_once);
  EXPECT_EQ(lost.coordinate_epoch, 2u);
  EXPECT_EQ(lost.epochs_committed, 1u);
  EXPECT_FALSE(lost.candidate);
  EXPECT_FALSE(scan(4, 200, 400));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 400, params));
  EXPECT_EQ(lost.coordinate_epoch, 2u);
}

TEST_F(LostRecovery, MissingScanRestartsTheStableWindow)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(3, 100, 200));
  EXPECT_TRUE(scan(4, 200, 300));
}

TEST_F(LostRecovery, ChangedGenerationRestartsTheStableWindow)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(2, 100, 200, true, 5));
  EXPECT_TRUE(scan(3, 200, 300, true, 5));
}

TEST_F(LostRecovery, FailedMatchCannotSupplyStableTime)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  EXPECT_FALSE(scan(2, 100, 300, false));
  EXPECT_FALSE(scan(3, 300, 400));
  EXPECT_TRUE(scan(4, 400, 500));
}

TEST_F(LostRecovery, ExcludedAndInvalidScansCannotQualify)
{
  begin();
  recovery.excluded_scan = 1;
  EXPECT_FALSE(scan(1, 0, 300));
  EXPECT_FALSE(scan(0, 0, 300));
  EXPECT_FALSE(scan(2, -1, 300));
  EXPECT_FALSE(scan(3, 300, 200));
  EXPECT_FALSE(scan(4, 300, 400));
  EXPECT_TRUE(scan(5, 400, 500));
}

TEST_F(LostRecovery, AbandonRequiresANewCandidate)
{
  begin();
  EXPECT_FALSE(scan(1, 0, 100));
  abandonLostRecovery(&lost, params);
  EXPECT_FALSE(scan(2, 100, 400));
  EXPECT_FALSE(commitLostRecovery(&lost, &recovery, 400, params));
  EXPECT_EQ(lost.coordinate_epoch, 1u);
  beginLostRecoveryCandidate(&lost, 5, 3, 400, params);
  EXPECT_FALSE(scan(3, 400, 500, true, 5));
  EXPECT_TRUE(scan(4, 500, 600, true, 5));
  EXPECT_EQ(lost.attempts, 2u);
}

TEST_F(LostRecovery, RepeatedLossCommitsEpochsTwoThenThree)
{
  begin();
  EXPECT_TRUE(scan(1, 0, 200));
  EXPECT_EQ(lost.coordinate_epoch, 2u);
  recovery.trust = RecoveryTrust::kUnanchored;
  beginLostRecoveryCandidate(&lost, 6, 2, 200, params);
  EXPECT_TRUE(scan(2, 200, 400, true, 6));
  EXPECT_EQ(lost.coordinate_epoch, 3u);
  EXPECT_EQ(lost.epochs_committed, 2u);
  EXPECT_EQ(recovery.episode, 7u);
  EXPECT_EQ(recovery.map_resets, 3u);
}

TEST_F(LostRecovery, LastExactlyRepresentableEpochCanCommit)
{
  lost.coordinate_epoch = (std::uint64_t{1} << 53) - 1;
  begin();
  EXPECT_TRUE(scan(1, 0, 200));
  EXPECT_EQ(lost.coordinate_epoch, std::uint64_t{1} << 53);
  EXPECT_EQ(static_cast<std::uint64_t>(static_cast<double>(lost.coordinate_epoch)), lost.coordinate_epoch);
  EXPECT_EQ(recovery.trust, RecoveryTrust::kLocalQualified);
  EXPECT_EQ(lost.epochs_committed, 1u);
}

TEST_F(LostRecovery, EpochExhaustionNeverWrapsOrQualifies)
{
  lost.coordinate_epoch = std::uint64_t{1} << 53;
  begin();
  EXPECT_FALSE(scan(1, 0, 200));
  EXPECT_EQ(recovery.trust, RecoveryTrust::kUnanchored);
  EXPECT_EQ(recovery.cause, RecoveryCause::kDeadline);
  EXPECT_EQ(lost.coordinate_epoch, std::uint64_t{1} << 53);
  EXPECT_EQ(lost.epochs_committed, 0u);
  EXPECT_EQ(lost.attempts, 1u);
}

TEST(LostRecoveryParameters, ValidatesDurationAndRequiresTheLatchPolicy)
{
  LostRecoveryParams params;
  EXPECT_TRUE(resolveLostRecoveryParams(false, 2.0, 0.0, &params));
  EXPECT_FALSE(params.enabled);
  EXPECT_EQ(params.stable_ns, 2000000000);
  EXPECT_TRUE(resolveLostRecoveryParams(true, 2.0, 10.0, &params));
  EXPECT_TRUE(params.enabled);
  EXPECT_FALSE(resolveLostRecoveryParams(true, 2.0, 0.0, &params));
  EXPECT_FALSE(resolveLostRecoveryParams(true, 2.0, -1.0, &params));
  for (double invalid :
       {0.0, -1.0, 1e-12, 1e20, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(resolveLostRecoveryParams(false, invalid, 0.0, &params));
    EXPECT_FALSE(resolveLostRecoveryParams(true, invalid, 10.0, &params));
  }
}

// ── The rebuild relevel (rebuild_relevel.hpp) ────────────────────────────────
// A rebuild that founds a new coordinate epoch first levels the frame its map is seeded in.

constexpr double kGravity = 9.809;
constexpr double kDeg = M_PI / 180.0;

Eigen::Matrix3d turn(double deg, const Eigen::Vector3d & axis)
{
  return Eigen::AngleAxisd(deg * kDeg, axis.normalized()).toRotationMatrix();
}

LevellingWindowReference parkedWindow(const Eigen::Vector3d & up_body)
{
  LevellingWindowReference window;
  window.valid = true;
  window.parked = true;
  window.up_body = up_body.normalized();
  window.variance = 0.008 * 0.008;
  return window;
}

LevellingWindowReference walkingWindow(const Eigen::Vector3d & up_body)
{
  LevellingWindowReference window;
  window.valid = true;
  window.up_body = up_body.normalized();
  return window;
}

TEST(RebuildRelevel, AllowedOnlyWhenEveryConditionHolds)
{
  EXPECT_TRUE(relevelAllowed(true, true, false, true));
  EXPECT_FALSE(relevelAllowed(false, true, false, true));  // the switch
  EXPECT_FALSE(relevelAllowed(true, false, false, true));  // LOST recovery off: the rebuilt map founds no epoch
  EXPECT_FALSE(relevelAllowed(true, true, true, true));    // a pinned prior map defines the frame
  EXPECT_FALSE(relevelAllowed(true, true, false, false));  // no first local reference yet
  int allowed = 0;
  for (int bits = 0; bits < 16; ++bits) {
    allowed += relevelAllowed((bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0, (bits & 8) != 0) ? 1 : 0;
  }
  EXPECT_EQ(allowed, 1);
}

// Parked on a 3° slope with the estimate 8.7° off about a horizontal axis: the window sees the true up, and the
// plan turns the estimate's view of it onto +Z, about a horizontal axis. With a pure tilt error that is the true
// attitude again. The gravity state does not enter.
TEST(RebuildRelevel, AParkedWindowLevelsTheAttitudeOntoItsVertical)
{
  const Eigen::Matrix3d R_true = turn(3.0, Eigen::Vector3d::UnitX()) * turn(30.0, Eigen::Vector3d::UnitZ());
  const Eigen::Matrix3d R = turn(8.7, Eigen::Vector3d(1.0, 2.0, 0.0)) * R_true;
  const Eigen::Vector3d u_b = R_true.transpose() * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d g = -kGravity * (turn(-4.0, Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitZ());
  const RelevelPlan plan = planRelevel(R, g, parkedWindow(u_b));
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.source, RelevelSource::kWindow);
  EXPECT_STREQ(relevelSourceName(plan.source), "window");
  EXPECT_NEAR((plan.L * R * u_b - Eigen::Vector3d::UnitZ()).norm(), 0.0, 1e-12);
  EXPECT_NEAR(plan.angle, 8.7 * kDeg, 1e-12);
  const Eigen::AngleAxisd L(plan.L);
  EXPECT_NEAR(L.angle(), 8.7 * kDeg, 1e-12);
  EXPECT_NEAR(L.axis().z(), 0.0, 1e-12);  // nothing turns about the vertical
  EXPECT_NEAR((plan.L * R - R_true).norm(), 0.0, 1e-12);
  // a world-frame velocity turns with the frame: the body-frame velocity the legs and the window edges see stays
  const Eigen::Vector3d v(0.4, -0.3, 0.05);
  EXPECT_NEAR(((plan.L * R).transpose() * (plan.L * v) - R.transpose() * v).norm(), 0.0, 1e-12);
}

// Not parked: the gravity state alone. L takes it onto −Z, and the attitude reference PGO levels with,
// levelingRotation(g)·R, is the same before and after: a change of coordinates, not a measurement.
TEST(RebuildRelevel, WithoutAParkedWindowTheGravityStateLevelsTheFrame)
{
  const Eigen::Matrix3d R = turn(8.7, Eigen::Vector3d(1.0, 2.0, 0.0)) * turn(30.0, Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d g = -kGravity * (turn(8.7, Eigen::Vector3d(1.0, 2.0, 0.0)) * Eigen::Vector3d::UnitZ());
  const LevellingWindowReference walking =
    walkingWindow(turn(5.0, Eigen::Vector3d::UnitX()) * Eigen::Vector3d::UnitZ());
  const RelevelPlan plan = planRelevel(R, g, walking);
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.source, RelevelSource::kGravity);
  EXPECT_STREQ(relevelSourceName(plan.source), "gravity");
  EXPECT_NEAR((plan.L * g.normalized() + Eigen::Vector3d::UnitZ()).norm(), 0.0, 1e-12);
  EXPECT_NEAR(plan.angle, 8.7 * kDeg, 1e-12);
  EXPECT_NEAR(Eigen::AngleAxisd(plan.L).axis().z(), 0.0, 1e-12);
  const auto levelling = [](const Eigen::Vector3d & gravity) {
    return Eigen::Quaterniond::FromTwoVectors(gravity, -Eigen::Vector3d::UnitZ()).toRotationMatrix();
  };
  EXPECT_NEAR((levelling(plan.L * g) * (plan.L * R) - levelling(g) * R).norm(), 0.0, 1e-12);
  // the window's direction never enters: another walking window, or none, gives the same rotation
  EXPECT_TRUE(planRelevel(R, g, walkingWindow(Eigen::Vector3d(0.3, -0.2, 1.0))).L == plan.L);
  EXPECT_TRUE(planRelevel(R, g, LevellingWindowReference{}).L == plan.L);
}

TEST(RebuildRelevel, ALevelFrameIsLeftAsItIs)
{
  const Eigen::Matrix3d yawed = turn(30.0, Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d level(0.0, 0.0, -kGravity);
  const RelevelPlan by_gravity = planRelevel(yawed, level, LevellingWindowReference{});
  ASSERT_TRUE(by_gravity.valid);
  EXPECT_TRUE(by_gravity.L == Eigen::Matrix3d::Identity());
  EXPECT_EQ(by_gravity.angle, 0.0);
  const RelevelPlan by_window = planRelevel(yawed, level, parkedWindow(yawed.transpose() * Eigen::Vector3d::UnitZ()));
  ASSERT_TRUE(by_window.valid);
  EXPECT_NEAR((by_window.L - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-15);
  EXPECT_NEAR(by_window.angle, 0.0, 1e-15);
}

// The reference is absolute, not an increment: levelling the levelled frame again turns nothing. After a plan the
// node sets the gravity state vertical.
TEST(RebuildRelevel, RelevellingTwiceDoesNotAccumulate)
{
  const Eigen::Matrix3d R = turn(8.7, Eigen::Vector3d(1.0, 2.0, 0.0)) * turn(30.0, Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d g = -kGravity * (turn(6.0, Eigen::Vector3d(-1.0, 1.0, 0.0)) * Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d level(0.0, 0.0, -kGravity);
  const RelevelPlan gravity_first = planRelevel(R, g, LevellingWindowReference{});
  ASSERT_TRUE(gravity_first.valid);
  const RelevelPlan gravity_second = planRelevel(gravity_first.L * R, level, LevellingWindowReference{});
  ASSERT_TRUE(gravity_second.valid);
  EXPECT_EQ(gravity_second.angle, 0.0);
  const LevellingWindowReference window = parkedWindow(Eigen::Vector3d(0.05, -0.02, 1.0));
  const RelevelPlan window_first = planRelevel(R, g, window);
  ASSERT_TRUE(window_first.valid);
  EXPECT_GT(window_first.angle, 1.0 * kDeg);
  const RelevelPlan window_second = planRelevel(window_first.L * R, level, window);
  ASSERT_TRUE(window_second.valid);
  EXPECT_NEAR(window_second.angle, 0.0, 1e-12);
}

TEST(RebuildRelevel, AReferenceAtOrBelowTheHorizonGivesNoPlan)
{
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const LevellingWindowReference none;
  EXPECT_FALSE(planRelevel(I, Eigen::Vector3d(0.0, 0.0, kGravity), none).valid);  // gravity pointing up
  EXPECT_FALSE(planRelevel(I, Eigen::Vector3d(kGravity, 0.0, 0.0), none).valid);  // horizontal
  EXPECT_FALSE(planRelevel(I, Eigen::Vector3d(0.0, 0.0, std::numeric_limits<double>::quiet_NaN()), none).valid);
  // upside down, the parked window's up points down in the world
  const Eigen::Matrix3d flipped = turn(180.0, Eigen::Vector3d::UnitX());
  EXPECT_FALSE(
    planRelevel(flipped, Eigen::Vector3d(0.0, 0.0, -kGravity), parkedWindow(Eigen::Vector3d::UnitZ())).valid);
  // no plan turns nothing
  const RelevelPlan up = planRelevel(I, Eigen::Vector3d(0.0, 0.0, kGravity), none);
  EXPECT_TRUE(up.L == Eigen::Matrix3d::Identity());
  EXPECT_EQ(up.angle, 0.0);
}
}  // namespace

// The consider update on the real estimator: a synthetic point-to-plane scene (floor + walls) whose residuals ask
// for a pitch correction. Plain, the iterated update recovers the pose; with the rotation considered, roll/pitch
// stay at the propagated value, the position still updates, and the roll/pitch variance is kept.
// The same scene drives the filter's two dyn_share entries side by side: a model bound with a context
// (init_dyn_share_context) must be called, and leave the filter, exactly as the plain entry does.
//
// Further down, the same estimator in a corridor with noisy plane normals: the floor projection
// (normal_noise_floor.hpp: the residual shifted by what the removed information reads, then the position
// columns projected) at every iteration, and the wheel update (wheel_measurement.hpp) after the lidar, for
// design D.5 and F2 (a)-(c).
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <omp.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "degeneracy_policy.hpp"
#include "normal_noise_floor.hpp"
#include "rp_consider.hpp"
#include "use-ikfom.hpp"
#include "wheel_measurement.hpp"

// The wheel Jacobian is written for this error-state layout.
static_assert(state_ikfom::DOF == fast_lio::kWheelStateDim, "wheel Jacobian width");
static_assert(decltype(state_ikfom::rot)::IDX == fast_lio::kWheelRotIndex, "wheel rotation columns");
static_assert(decltype(state_ikfom::vel)::IDX == fast_lio::kWheelVelIndex, "wheel velocity columns");
static_assert(decltype(state_ikfom::bg)::IDX == fast_lio::kWheelGyroBiasIndex, "wheel gyro-bias columns");

namespace
{
using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;

struct Plane
{
  Eigen::Vector3d n;
  double d;  // n . x + d = 0
};

// world points observed from the TRUE pose, expressed in the body frame
struct Scene
{
  std::vector<Eigen::Vector3d> body;
  std::vector<Plane> plane;
};

Scene makeScene(const Eigen::Matrix3d & R_true, const Eigen::Vector3d & t_true)
{
  Scene s;
  const Plane floor{Eigen::Vector3d(0, 0, 1), 0.0};
  const Plane wall_x{Eigen::Vector3d(1, 0, 0), -5.0};
  const Plane wall_y{Eigen::Vector3d(0, 1, 0), -4.0};
  for (int i = -3; i <= 3; ++i) {
    for (int j = -3; j <= 3; ++j) {
      s.body.push_back(R_true.transpose() * (Eigen::Vector3d(1.5 * i, 1.5 * j, 0.0) - t_true));
      s.plane.push_back(floor);
    }
  }
  for (int k = 0; k < 8; ++k) {
    s.body.push_back(R_true.transpose() * (Eigen::Vector3d(5.0, -2.0 + 0.6 * k, 0.3 + 0.2 * k) - t_true));
    s.plane.push_back(wall_x);
    s.body.push_back(R_true.transpose() * (Eigen::Vector3d(-2.0 + 0.6 * k, 4.0, 0.2 + 0.25 * k) - t_true));
    s.plane.push_back(wall_y);
  }
  return s;
}

Scene * g_scene = nullptr;

// FAST-LIO's h_share_model conventions: h = -(signed distance), rows [n_world, (p_body x (R^T n)), 0...]
void sceneModel(state_ikfom & s, esekfom::dyn_share_datastruct<double> & data)
{
  const int m = static_cast<int>(g_scene->body.size());
  data.h_x = Eigen::MatrixXd::Zero(m, 12);
  data.h.resize(m);
  const Eigen::Matrix3d R = s.rot.toRotationMatrix();
  for (int i = 0; i < m; ++i) {
    const Eigen::Vector3d p_b = g_scene->body[i];
    const Eigen::Vector3d p_w = R * p_b + s.pos;
    const Plane & pl = g_scene->plane[i];
    const double pd2 = pl.n.dot(p_w) + pl.d;
    const Eigen::Vector3d C = R.transpose() * pl.n;
    const Eigen::Vector3d A = p_b.cross(C);
    data.h_x.block<1, 3>(i, 0) = pl.n.transpose();
    data.h_x.block<1, 3>(i, 3) = A.transpose();
    data.h(i) = -pd2;
  }
  data.valid = true;
}

struct ModelFailure : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

// The calls a recorded scene model answered, and the faults it injects.
struct ModelCalls
{
  struct Call
  {
    bool converge;  // the filter's flag on entry
    bool valid;     // the model's answer
    bool operator==(const Call & other) const { return converge == other.converge && valid == other.valid; }
  };
  std::vector<Call> calls;
  bool full_form = false;  // answer in update_iterated_dyn_share's form: z - h, h_x over the whole state, h_v, R
  int invalid_call = -1;   // this call answers valid = false
  int throwing_call = -1;  // this call throws ModelFailure instead
};

void recordedModel(ModelCalls & record, state_ikfom & s, esekfom::dyn_share_datastruct<double> & data)
{
  const int call = static_cast<int>(record.calls.size());
  if (call == record.throwing_call) {
    throw ModelFailure("scene model failed on call " + std::to_string(call));
  }
  sceneModel(s, data);
  if (record.full_form) {
    const Eigen::Index rows = data.h.size();
    Eigen::MatrixXd h_x = Eigen::MatrixXd::Zero(rows, state_ikfom::DOF);
    h_x.leftCols(data.h_x.cols()) = data.h_x;
    data.h_x = h_x;
    data.z = data.h;
    data.h = Eigen::VectorXd::Zero(rows);
    data.h_v = Eigen::MatrixXd::Identity(rows, rows);
    data.R = Eigen::MatrixXd::Identity(rows, rows) * 0.001;
  }
  data.valid = call != record.invalid_call;
  record.calls.push_back({data.converge, data.valid});
}

ModelCalls * g_plain_calls = nullptr;  // the plain entry passes no context: its model finds the record here

void plainRecordedModel(state_ikfom & s, esekfom::dyn_share_datastruct<double> & data)
{
  recordedModel(*g_plain_calls, s, data);
}

void contextRecordedModel(void * context, state_ikfom & s, esekfom::dyn_share_datastruct<double> & data)
{
  recordedModel(*static_cast<ModelCalls *>(context), s, data);
}

// Every double of the filter's state and covariance, in a fixed order.
std::vector<double> filterBits(const Filter & filter)
{
  const state_ikfom & x = filter.get_x();
  std::vector<double> bits;
  const auto append = [&bits](const double * values, long count) {
    bits.insert(bits.end(), values, values + count);
  };
  append(x.pos.data(), 3);
  append(x.rot.coeffs().data(), 4);
  append(x.offset_R_L_I.coeffs().data(), 4);
  append(x.offset_T_L_I.data(), 3);
  append(x.vel.data(), 3);
  append(x.bg.data(), 3);
  append(x.ba.data(), 3);
  append(x.grav.get_vect().data(), 3);
  append(filter.get_P().data(), filter.get_P().size());
  return bits;
}

::testing::AssertionResult sameBits(const std::vector<double> & a, const std::vector<double> & b)
{
  if (a.size() != b.size()) {
    return ::testing::AssertionFailure() << "sizes " << a.size() << " and " << b.size();
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) {
      return ::testing::AssertionFailure() << "value " << i << " differs: " << a[i] << " and " << b[i];
    }
  }
  return ::testing::AssertionSuccess();
}

Eigen::Vector3d rollPitchYaw(const Eigen::Matrix3d & R)
{
  return Eigen::Vector3d(std::atan2(R(2, 1), R(2, 2)), -std::asin(R(2, 0)), std::atan2(R(1, 0), R(0, 0)));
}

// How a fixture binds the scene to its filter.
enum class Binding
{
  kScene,            // init_dyn_share(sceneModel)
  kRecordedPlain,    // init_dyn_share(plainRecordedModel), recording into the fixture's calls
  kRecordedContext,  // init_dyn_share_context(contextRecordedModel) with the fixture's calls as the context
};

struct Fixture
{
  Filter filter;
  Eigen::Matrix3d R_true = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_true = Eigen::Vector3d(0.0, 0.0, 1.0);
  Eigen::Vector3d prior_rpy;
  Scene scene;
  double pitch_prior = 0.02;
  Eigen::Vector3d up_body = Eigen::Vector3d::UnitZ();  // the body-frame vertical of the prior attitude
  ModelCalls calls;
  double epsilon[23];

  explicit Fixture(Binding binding = Binding::kScene)
  {
    scene = makeScene(R_true, t_true);
    g_scene = &scene;
    std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
    if (binding == Binding::kScene) {
      filter.init_dyn_share(get_f, df_dx, df_dw, sceneModel, 6, epsilon);
    } else if (binding == Binding::kRecordedPlain) {
      g_plain_calls = &calls;
      filter.init_dyn_share(get_f, df_dx, df_dw, plainRecordedModel, 6, epsilon);
    } else {
      filter.init_dyn_share_context(get_f, df_dx, df_dw, contextRecordedModel, &calls, 6, epsilon);
    }
    state_ikfom x = filter.get_x();
    x.pos = t_true + Eigen::Vector3d(0.05, -0.03, 0.02);
    x.rot = SO3(Eigen::AngleAxisd(pitch_prior, Eigen::Vector3d::UnitY()).toRotationMatrix());
    x.grav = S2(Eigen::Vector3d(0.0, 0.0, -9.81));
    filter.change_x(x);
    Filter::cov P = Filter::cov::Identity() * 1e-8;
    for (int k = 0; k < 3; ++k) {
      P(k, k) = 1e-2;  // position sigma 0.1 m: the 16 wall points (8e3 of information per axis) own x and y
    }
    for (int k = 3; k < 6; ++k) {
      P(k, k) = 1e-4;  // attitude sigma 0.57 deg: the 49 floor points own roll and pitch
    }
    filter.change_P(P);
    prior_rpy = rollPitchYaw(x.rot.toRotationMatrix());
    up_body = x.rot.toRotationMatrix().transpose() * Eigen::Vector3d::UnitZ();
  }
};
}  // namespace

TEST(ConsiderFilter, PlainUpdateRecoversThePoseIncludingPitch)
{
  Fixture f;
  double solve_time = 0.0;
  f.filter.update_iterated_dyn_share_modified(0.001, solve_time);
  const state_ikfom x = f.filter.get_x();
  EXPECT_LT((x.pos - f.t_true).norm(), 5e-3);
  const Eigen::Vector3d rpy = rollPitchYaw(x.rot.toRotationMatrix());
  EXPECT_LT(std::abs(rpy(1)), 2e-3);  // the pitch offset is corrected
  EXPECT_LT(f.filter.get_P()(4, 4), 1e-4 * 0.5);  // and the pitch variance shrinks
  EXPECT_LT(f.filter.get_P()(0, 0), 1e-2 * 0.1);  // so does the position variance
}

TEST(ConsiderFilter, ConsideredRollPitchStayAtThePriorWhilePositionUpdates)
{
  Fixture f;
  const Filter::cov P0 = f.filter.get_P();
  const auto proj = fast_lio::yawOnlyProjector(f.up_body);  // the rotation error lives in the body frame
  ASSERT_TRUE(proj.valid);
  f.filter.setConsiderRotation(3, proj.keep);
  EXPECT_TRUE(f.filter.considerRotationActive());
  double solve_time = 0.0;
  f.filter.update_iterated_dyn_share_modified(0.001, solve_time);
  const state_ikfom x = f.filter.get_x();
  const Eigen::Vector3d rpy = rollPitchYaw(x.rot.toRotationMatrix());
  EXPECT_NEAR(rpy(1), f.prior_rpy(1), 1e-6);  // pitch untouched
  EXPECT_NEAR(rpy(0), f.prior_rpy(0), 1e-6);  // roll untouched
  EXPECT_LT(std::abs(x.pos.x() - f.t_true.x()), 1e-2);  // the walls still fix x and y
  EXPECT_LT(std::abs(x.pos.y() - f.t_true.y()), 1e-2);
  EXPECT_GT((x.pos - (f.t_true + Eigen::Vector3d(0.05, -0.03, 0.02))).norm(), 0.04);  // it did move
  const Filter::cov P = f.filter.get_P();
  EXPECT_TRUE(P.allFinite());
  EXPECT_NEAR(P(3, 3), P0(3, 3), 1e-4 * 1e-3);  // roll/pitch variance kept
  EXPECT_NEAR(P(4, 4), P0(4, 4), 1e-4 * 1e-3);
  EXPECT_LT(P(0, 0), P0(0, 0) * 0.1);  // position variance shrinks
  EXPECT_LT((P - P.transpose()).norm(), 1e-9 * P.norm());
  f.filter.clearConsiderRotation();
  EXPECT_FALSE(f.filter.considerRotationActive());
}

TEST(ConsiderFilter, ClearingRestoresThePlainUpdate)
{
  Fixture f;
  const auto proj = fast_lio::yawOnlyProjector(f.up_body);
  f.filter.setConsiderRotation(3, proj.keep);
  f.filter.clearConsiderRotation();
  double solve_time = 0.0;
  f.filter.update_iterated_dyn_share_modified(0.001, solve_time);
  EXPECT_LT(std::abs(rollPitchYaw(f.filter.get_x().rot.toRotationMatrix())(1)), 2e-3);
}

TEST(ConsiderFilter, InvalidIndexOrProjectorIsIgnored)
{
  Fixture f;
  Eigen::Matrix3d bad = Eigen::Matrix3d::Identity();
  bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
  f.filter.setConsiderRotation(3, bad);
  EXPECT_FALSE(f.filter.considerRotationActive());
  f.filter.setConsiderRotation(21, Eigen::Matrix3d::Identity());
  EXPECT_FALSE(f.filter.considerRotationActive());
}

TEST(ConsiderFilter, HundredsOfConsideredUpdatesKeepTheCovarianceDefiniteAndTheStateBounded)
{
  // Considered updates in a row, with a propagation step between them: the m20 doorway windows are 25-40 scans
  // of it. The first implementation (the P - M A - A^T M + M A M shortcut mixed with the tangent-space Jacobians)
  // left P indefinite after a few dozen scans and the state ran away 12 m in one scan; the Joseph form must not.
  Fixture f;
  const auto proj = fast_lio::yawOnlyProjector(f.up_body);
  ASSERT_TRUE(proj.valid);
  input_ikfom in;
  in.acc = Eigen::Vector3d(0.0, 0.0, 9.81);  // the specific force of a body at rest, world = IMU frame
  in.gyro = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity() * 1e-4;
  double solve_time = 0.0;
  double min_eig = 1.0;
  double dt = 0.005;
  for (int cycle = 0; cycle < 300; ++cycle) {
    // a little gait-like noise in the input keeps the linearisation point moving
    in.gyro = Eigen::Vector3d(0.02 * std::sin(0.3 * cycle), 0.03 * std::cos(0.2 * cycle), 0.05);
    for (int step = 0; step < 20; ++step) {
      f.filter.predict(dt, Q, in);
    }
    f.filter.setConsiderRotation(3, proj.keep);
    f.filter.update_iterated_dyn_share_modified(0.001, solve_time);
    f.filter.clearConsiderRotation();
    const Filter::cov P = f.filter.get_P();
    ASSERT_TRUE(P.allFinite()) << "cycle " << cycle;
    ASSERT_LT((P - P.transpose()).norm(), 1e-9 * (1.0 + P.norm())) << "cycle " << cycle;
    Eigen::SelfAdjointEigenSolver<Filter::cov> es(P);
    ASSERT_EQ(es.info(), Eigen::Success);
    min_eig = std::min(min_eig, es.eigenvalues().minCoeff());
    ASSERT_GT(es.eigenvalues().minCoeff(), -1e-12) << "cycle " << cycle;
    const state_ikfom x = f.filter.get_x();
    ASSERT_LT((x.pos - f.t_true).norm(), 0.5) << "cycle " << cycle;
    ASSERT_TRUE(x.vel.allFinite());
  }
  EXPECT_GT(min_eig, -1e-12);
  // roll/pitch never estimated: their variance only grew with the process noise, the position's did not run away
  const Filter::cov P = f.filter.get_P();
  EXPECT_GT(P(4, 4), 1e-4 * 0.9);
  EXPECT_LT(P(0, 0), 1.0);
}

// Runs one dyn_share update: update_iterated_dyn_share for a full-form model, the modified one otherwise.
void dynShareUpdate(Fixture & f)
{
  double solve_time = 0.0;
  if (f.calls.full_form) {
    f.filter.update_iterated_dyn_share();
  } else {
    f.filter.update_iterated_dyn_share_modified(0.001, solve_time);
  }
}

TEST(DynShareContext, CallsTheModelInThePlainEntrysSequence)
{
  for (const bool full_form : {false, true}) {
    Fixture plain(Binding::kRecordedPlain);
    Fixture context(Binding::kRecordedContext);
    for (Fixture * f : {&plain, &context}) {
      f->calls.full_form = full_form;
      f->calls.invalid_call = 1;  // the filter skips this iteration and evaluates the same state again
      dynShareUpdate(*f);
    }
    ASSERT_GE(plain.calls.calls.size(), 3u) << "full form " << full_form;
    EXPECT_FALSE(plain.calls.calls[1].valid);
    EXPECT_EQ(plain.calls.calls, context.calls.calls) << "full form " << full_form;
  }
}

TEST(DynShareContext, LeavesTheStateAndCovarianceBitForBit)
{
  Fixture plain(Binding::kRecordedPlain);
  Fixture context(Binding::kRecordedContext);
  const std::vector<double> start = filterBits(plain.filter);
  ASSERT_TRUE(sameBits(start, filterBits(context.filter)));
  input_ikfom in;
  in.acc = Eigen::Vector3d(0.0, 0.0, 9.81);
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Identity() * 1e-4;
  double dt = 0.005;
  for (int cycle = 0; cycle < 6; ++cycle) {
    in.gyro = Eigen::Vector3d(0.02 * std::sin(0.3 * cycle), 0.03 * std::cos(0.2 * cycle), 0.05);
    for (Fixture * f : {&plain, &context}) {
      f->calls.full_form = cycle % 2 == 1;  // the two dyn_share updates in turn
      for (int step = 0; step < 20; ++step) {
        f->filter.predict(dt, Q, in);
      }
      dynShareUpdate(*f);
    }
    ASSERT_TRUE(plain.filter.get_P().allFinite()) << "cycle " << cycle;
    ASSERT_TRUE(sameBits(filterBits(plain.filter), filterBits(context.filter))) << "cycle " << cycle;
  }
  EXPECT_EQ(plain.calls.calls, context.calls.calls);
  EXPECT_FALSE(sameBits(start, filterBits(plain.filter)));  // the updates did move the filter
}

TEST(DynShareContext, PassesAModelExceptionThroughUnchanged)
{
  for (const bool full_form : {false, true}) {
    Fixture plain(Binding::kRecordedPlain);
    Fixture context(Binding::kRecordedContext);
    std::vector<const std::type_info *> types;
    std::vector<std::string> messages;
    for (Fixture * f : {&plain, &context}) {
      f->calls.full_form = full_form;
      f->calls.throwing_call = 1;  // after one evaluated iteration
      try {
        dynShareUpdate(*f);
        ADD_FAILURE() << "no exception, full form " << full_form;
      } catch (const std::exception & e) {
        types.push_back(&typeid(e));
        messages.push_back(e.what());
      }
    }
    ASSERT_EQ(types.size(), 2u);
    EXPECT_TRUE(*types[0] == typeid(ModelFailure)) << types[0]->name();
    EXPECT_TRUE(*types[1] == typeid(ModelFailure)) << types[1]->name();
    EXPECT_EQ(messages[0], "scene model failed on call 1");
    EXPECT_EQ(messages[1], messages[0]);
    EXPECT_EQ(plain.calls.calls, context.calls.calls);
    EXPECT_TRUE(sameBits(filterBits(plain.filter), filterBits(context.filter))) << "full form " << full_form;
  }
}

TEST(DynShareContext, SwitchesBindingsAndRejectsANullContext)
{
  Fixture f(Binding::kRecordedPlain);
  ModelCalls first;
  ModelCalls second;
  dynShareUpdate(f);
  const std::size_t plain_calls = f.calls.calls.size();
  EXPECT_GT(plain_calls, 0u);

  f.filter.init_dyn_share_context(get_f, df_dx, df_dw, contextRecordedModel, &first, 6, f.epsilon);
  dynShareUpdate(f);
  const std::size_t first_calls = first.calls.size();
  EXPECT_GT(first_calls, 0u);
  EXPECT_EQ(f.calls.calls.size(), plain_calls);  // the context entry replaced the plain model

  f.filter.clear_dyn_share_context();
  f.filter.init_dyn_share_context(get_f, df_dx, df_dw, contextRecordedModel, &second, 6, f.epsilon);
  dynShareUpdate(f);
  const std::size_t second_calls = second.calls.size();
  EXPECT_GT(second_calls, 0u);
  EXPECT_EQ(first.calls.size(), first_calls);  // a cleared context is not called again

  f.filter.init_dyn_share(get_f, df_dx, df_dw, plainRecordedModel, 6, f.epsilon);
  dynShareUpdate(f);
  EXPECT_GT(f.calls.calls.size(), plain_calls);  // the plain entry replaced the context
  EXPECT_EQ(second.calls.size(), second_calls);

  EXPECT_THROW(f.filter.init_dyn_share_context(get_f, df_dx, df_dw, contextRecordedModel, nullptr, 6, f.epsilon),
               std::invalid_argument);
  EXPECT_THROW(f.filter.init_dyn_share_context(get_f, df_dx, df_dw, nullptr, &first, 6, f.epsilon),
               std::invalid_argument);
  const std::size_t before_rejected = f.calls.calls.size();
  dynShareUpdate(f);
  EXPECT_GT(f.calls.calls.size(), before_rejected);  // a rejected bind leaves the bound model in place
  EXPECT_EQ(first.calls.size(), first_calls);
}

namespace
{
// A 40 m corridor along x seen from (0, 0, 1): a floor of 160 points and two walls of 80. Each point lies on a
// five-point patch with a 2 cm residual, so its fitted normal tilts by sqrt(0.00625) (4.5 degrees) along the
// corridor, the signs balanced over the scan. The scan then holds f = 1.5 normals of fake along-corridor
// information and no real one; the floor model sees exactly that and keeps g = 0 along x. End walls at
// x = +-21 m add real along-corridor information in proportion to the square of their weight.
constexpr double kAccCov = 0.003993957088823881;  // m20 acc_cov
constexpr double kImuDt = 0.005;
constexpr int kImuStepsPerScan = 20;
constexpr double kScanPeriod = kImuDt * kImuStepsPerScan;
constexpr double kLidarR = 0.001;
constexpr double kQ = kAccCov * kImuDt;  // m^2/s^3 per axis: the predict adds acc_cov dt^2 per IMU step
constexpr int kVelX = fast_lio::kWheelVelIndex;

struct CorridorPoint
{
  Eigen::Vector3d body = Eigen::Vector3d::Zero();  // seen from the true pose
  Eigen::Vector3d n = Eigen::Vector3d::UnitZ();    // the fitted (tilted) unit normal
  double d = 0.0;
  double weight = 1.0;
  std::optional<Eigen::Matrix3d> cov;
};

struct Corridor
{
  std::vector<CorridorPoint> points;
  bool project = true;
  // The last projection, for the assertions.
  Eigen::Vector3d lambda = Eigen::Vector3d::Zero();
  Eigen::Vector3d noise_floor = Eigen::Vector3d::Zero();
  Eigen::Vector3d gains = Eigen::Vector3d::Ones();
  Eigen::Matrix3d V = Eigen::Matrix3d::Identity();
  double zeroed_information = 0.0;  // largest sum_i (row_i . v_k)^2 over directions with g_k = 0, any call
};

Corridor makeCorridor(double tilt_scale, double end_weight, bool noisy = true)
{
  Corridor corridor;
  const Eigen::Vector3d p_true(0.0, 0.0, 1.0);
  const Eigen::Vector3d X = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d Y = Eigen::Vector3d::UnitY();
  const Eigen::Vector3d Z = Eigen::Vector3d::UnitZ();
  const auto add = [&](const Eigen::Vector3d & p_w,
                       const Eigen::Vector3d & n_true,
                       const Eigen::Vector3d & e1,
                       const Eigen::Vector3d & e2,
                       double sign,
                       double weight) {
    fast_lio::PlaneFitEigen fit;
    fit.lambda0 = 0.002;  // 2 cm rms over five points
    fit.lambda1 = 0.16;
    fit.lambda2 = 0.25;
    fit.e1 = e1;
    fit.e2 = e2;
    fit.count = 5;
    CorridorPoint point;
    point.cov = fast_lio::planeNormalCovariance(fit);
    point.n = n_true;
    if (noisy) {
      point.n += tilt_scale * sign * std::sqrt(e1.dot(*point.cov * e1)) * e1;
      point.n.normalize();
    }
    point.d = -point.n.dot(p_w);
    point.body = p_w - p_true;
    point.weight = weight;
    corridor.points.push_back(point);
  };
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 4; ++j) {
      add(Eigen::Vector3d(-19.5 + i, -1.5 + j, 0.0), Z, X, Y, (i + j) % 2 == 0 ? 1.0 : -1.0, 1.0);
    }
  }
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        const double y = k == 0 ? -2.0 : 2.0;
        add(Eigen::Vector3d(-19.0 + 2.0 * i, y, 0.5 + j),
            Eigen::Vector3d(0.0, -y / 2.0, 0.0),
            X,
            Z,
            (i + j + k) % 2 == 0 ? 1.0 : -1.0,
            1.0);
      }
    }
  }
  if (end_weight > 0.0) {
    for (double x : {-21.0, 21.0}) {
      for (double y : {-1.0, 1.0}) {
        for (double z : {0.5, 1.5}) {
          add(Eigen::Vector3d(x, y, z), Eigen::Vector3d(-x / 21.0, 0.0, 0.0), Y, Z, 1.0, end_weight);
        }
      }
    }
  }
  return corridor;
}

Corridor * g_corridor = nullptr;

// The corridor's point weights and normal covariances, row for row, as the floor projection takes them.
std::pair<std::vector<double>, std::vector<std::optional<Eigen::Matrix3d>>> floorInputs(const Corridor & corridor)
{
  std::pair<std::vector<double>, std::vector<std::optional<Eigen::Matrix3d>>> inputs;
  for (const CorridorPoint & point : corridor.points) {
    inputs.first.push_back(point.weight);
    inputs.second.push_back(point.cov);
  }
  return inputs;
}

// sceneModel's conventions, with the floor projection as part B is to run it when the corridor asks for it:
// applyFloorProjection with the FloorParams defaults and the front end's Schur ridge, which shifts the residual
// by what the removed information reads and then applies G to the position columns, at every iteration.
void corridorModel(state_ikfom & s, esekfom::dyn_share_datastruct<double> & data)
{
  Corridor & corridor = *g_corridor;
  const int m = static_cast<int>(corridor.points.size());
  data.h_x = Eigen::MatrixXd::Zero(m, 12);
  data.h.resize(m);
  const Eigen::Matrix3d R = s.rot.toRotationMatrix();
  for (int i = 0; i < m; ++i) {
    const CorridorPoint & point = corridor.points[i];
    const Eigen::Vector3d p_w = R * point.body + s.pos;
    const Eigen::Vector3d A = point.body.cross(R.transpose() * point.n);
    data.h_x.block<1, 3>(i, 0) = point.weight * point.n.transpose();
    data.h_x.block<1, 3>(i, 3) = point.weight * A.transpose();
    data.h(i) = -point.weight * (point.n.dot(p_w) + point.d);
  }
  if (corridor.project) {
    const auto inputs = floorInputs(corridor);
    const fast_lio::FloorProjectionResult projection = fast_lio::applyFloorProjection(
      data.h_x, data.h, inputs.first, inputs.second, fast_lio::FloorParams{}, fast_lio::DegeneracyParams{}.schur_ridge);
    if (!projection.valid) {
      ADD_FAILURE() << "the floor projection refused the corridor's rows";
    }
    corridor.V = projection.eigenvectors;
    corridor.lambda = projection.lambda;
    corridor.noise_floor = projection.noise_floor;
    corridor.gains = projection.gains;
    for (int k = 0; k < 3; ++k) {
      if (corridor.gains(k) == 0.0) {
        corridor.zeroed_information =
          std::max(corridor.zeroed_information, (data.h_x.leftCols(3) * corridor.V.col(k)).squaredNorm());
      }
    }
  }
  data.valid = true;
}

state_ikfom restingState(const Eigen::Vector3d & position_offset = Eigen::Vector3d::Zero(),
                         double pitch = 0.0,
                         const Eigen::Vector3d & velocity = Eigen::Vector3d::Zero())
{
  state_ikfom x;
  x.pos = Eigen::Vector3d(0.0, 0.0, 1.0) + position_offset;
  x.rot = SO3(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix());
  x.vel = velocity;
  x.grav = S2(Eigen::Vector3d(0.0, 0.0, -9.81));
  return x;
}

// The fixture's position and attitude priors; the extrinsics at the fixture's 1e-8; velocity and the
// bias/gravity states as given.
Filter::cov corridorPrior(double velocity, double biases)
{
  Filter::cov P = Filter::cov::Identity() * biases;
  for (int k = 0; k < 3; ++k) {
    P(k, k) = 1e-2;
    P(3 + k, 3 + k) = 1e-4;
    P(6 + k, 6 + k) = 1e-8;
    P(9 + k, 9 + k) = 1e-8;
    P(kVelX + k, kVelX + k) = velocity;
  }
  return P;
}

// The estimator at rest in a corridor: acc-only process noise (the q of design D), a lidar update per scan,
// and the wheel update as part B is to call it: the wheel_measurement model, packed for update_simple.
class CorridorEstimator
{
public:
  CorridorEstimator(Corridor corridor, const Filter::cov & P0, const state_ikfom & x0) : corridor_(std::move(corridor))
  {
    double epsilon[23];
    std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
    filter_.init_dyn_share(get_f, df_dx, df_dw, corridorModel, 6, epsilon);
    state_ikfom x = x0;
    filter_.change_x(x);
    Filter::cov P = P0;
    filter_.change_P(P);
    Q_.setZero();
    Q_.block<3, 3>(3, 3) = kAccCov * Eigen::Matrix3d::Identity();
    // At rest the specific force is the state's own gravity reversed (S2 keeps |g| = 9.809), so nothing drifts.
    in_.acc = Eigen::Vector3d(-x0.grav.get_vect());
    in_.gyro = Eigen::Vector3d::Zero();
  }

  void propagate()
  {
    for (int step = 0; step < kImuStepsPerScan; ++step) {
      double dt = kImuDt;
      filter_.predict(dt, Q_, in_);
    }
  }

  void lidar()
  {
    g_corridor = &corridor_;
    double solve_time = 0.0;
    filter_.update_iterated_dyn_share_modified(kLidarR, solve_time);
  }

  // The legs report standing still with the bridge's 0.05 m/s (below the 0.075 floor), lateral not observed.
  // Returns the update's gain on the along-corridor velocity, 1 - P+/P-.
  double wheel()
  {
    const state_ikfom x = filter_.get_x();
    fast_lio::WheelFit fit;
    fit.v = 0.0;
    fit.samples = 1;
    fit.sample_max_vx = fit.endpoint_bound_vx = 0.05 * 0.05;
    fit.sample_max_lat = fast_lio::kBridgeUnobservedVariance;
    fit.lateral_observed = false;
    const auto measurement = fast_lio::wheelMeasurement(
      fit,
      std::nullopt,
      x.rot.toRotationMatrix(),
      x.vel,
      x.bg,
      fast_lio::angularRateAtEnd({{0.0, Eigen::Vector3d::Zero()}}, 0.0, Eigen::Vector3d::Zero()),
      Eigen::Matrix3d::Identity(),
      fast_lio::WheelModelParams{});
    if (!measurement) {
      ADD_FAILURE() << "no wheel measurement";
      return 0.0;
    }
    Eigen::Matrix<double, Eigen::Dynamic, fast_lio::kWheelStateDim> H;
    Eigen::VectorXd residual;
    Eigen::VectorXd variances;
    if (!measurement->pack(H, residual, variances)) {
      ADD_FAILURE() << "wheel covariance could not be whitened";
      return 0.0;
    }
    const double before = filter_.get_P()(kVelX, kVelX);
    filter_.update_simple(H, residual, variances);
    return 1.0 - filter_.get_P()(kVelX, kVelX) / before;
  }

  Filter & filter() { return filter_; }
  Corridor & corridor() { return corridor_; }

private:
  Corridor corridor_;
  Filter filter_;
  Eigen::Matrix<double, 12, 12> Q_;
  input_ikfom in_;
};

// The along-corridor information and floor at the true pose, for placing lambda / f with end walls.
std::pair<double, double> alongCorridorInformation(double tilt_scale, double end_weight)
{
  Corridor corridor = makeCorridor(tilt_scale, end_weight);
  g_corridor = &corridor;
  state_ikfom truth = restingState();
  esekfom::dyn_share_datastruct<double> data;
  corridorModel(truth, data);
  g_corridor = nullptr;
  return {corridor.lambda(0), corridor.noise_floor(0)};
}

// End-wall weight that puts lambda_u at `ratio` times the floor (the Schur coupling makes it nearly, not
// exactly, linear in the weight squared; the tests read the realised ratio back).
double endWeightFor(double tilt_scale, double ratio)
{
  const auto bare = alongCorridorInformation(tilt_scale, 0.0);
  const auto unit = alongCorridorInformation(tilt_scale, 1.0);
  return std::sqrt(std::max(ratio * bare.second - bare.first, 0.0) / (unit.first - bare.first));
}

// The largest |x_b - x_a| / sigma_a over the error state; a non-finite state counts as infinitely far.
std::pair<double, int> worstSigmaDistance(const state_ikfom & a, const state_ikfom & b, const Filter::cov & P_a)
{
  Filter::vectorized_state difference = Filter::vectorized_state::Zero();
  b.boxminus(difference, a);
  double worst = 0.0;
  int index = -1;
  for (int i = 0; i < difference.size(); ++i) {
    const double distance = std::abs(difference(i)) / std::sqrt(P_a(i, i));
    if (!std::isfinite(distance)) {
      return {std::numeric_limits<double>::infinity(), i};
    }
    if (distance > worst) {
      worst = distance;
      index = i;
    }
  }
  return {worst, index};
}

std::string text(double value)
{
  std::ostringstream out;
  out.precision(6);
  out << value;
  return out.str();
}
}  // namespace

TEST(FloorProjection, WheelSetsTheAlongCorridorVelocityAndTheOtherDirectionsKeepTheirs)
{
  // Design D.5 (F6): 200 scans with the wheel update after each lidar update. With q = acc_cov dt and
  // r = 0.075^2 T the along-corridor velocity sigma must converge to (q r)^(1/4) = 10.3 mm/s within 10 %, the
  // wheel must take >= 1.5 % per scan, and the directions across the corridor keep the variance they have
  // without the projection, up to the information the floor deduction takes from them: a variance ratio in
  // [1, 1 / g^2]. Bias and gravity priors are tight because D.5's model has no unknown constant acceleration.
  const Filter::cov P0 = corridorPrior(1e-8, 1e-12);
  CorridorEstimator projected(makeCorridor(1.0, 0.0), P0, restingState());
  CorridorEstimator plain(makeCorridor(1.0, 0.0), P0, restingState());
  plain.corridor().project = false;
  std::vector<double> gains;
  double plain_gain = 0.0;
  for (int scan = 0; scan < 200; ++scan) {
    projected.propagate();
    projected.lidar();
    gains.push_back(projected.wheel());
    plain.propagate();
    plain.lidar();
    plain_gain = plain.wheel();
  }
  const Corridor & corridor = projected.corridor();
  for (int k = 0; k < 3; ++k) {
    ASSERT_GT(std::abs(corridor.V(k, k)), 1.0 - 1e-9) << "eigenvector " << k << " is not the axis";
  }
  EXPECT_EQ(corridor.gains(0), 0.0);  // only the floor along the corridor
  EXPECT_GT(corridor.gains(1), 0.99);
  EXPECT_GT(corridor.gains(2), 0.99);

  const double target = std::pow(kQ * 0.075 * 0.075 * kScanPeriod, 0.25);
  const Filter::cov & P = projected.filter().get_P();
  const Filter::cov & P_plain = plain.filter().get_P();
  const double ratio = std::sqrt(P(kVelX, kVelX)) / target;
  const double min_gain = *std::min_element(gains.end() - 50, gains.end());
  RecordProperty("sigma_v_over_target", text(ratio));
  RecordProperty("min_wheel_gain_last50", text(min_gain));
  std::cout << "D.5: sigma_v " << std::sqrt(P(kVelX, kVelX)) << " m/s, (q r)^(1/4) " << target << ", ratio " << ratio
            << ", min wheel gain over the last 50 scans " << min_gain << "\n";
  EXPECT_NEAR(ratio, 1.0, 0.10);
  EXPECT_GE(min_gain, 0.015);
  for (int axis = 1; axis < 3; ++axis) {
    const double bound = 1.0 / (corridor.gains(axis) * corridor.gains(axis));
    for (int index : {axis, kVelX + axis}) {
      const double variance_ratio = P(index, index) / P_plain(index, index);
      RecordProperty("perpendicular_ratio_" + std::to_string(index), text(variance_ratio));
      EXPECT_GE(variance_ratio, 1.0 - 1e-9) << "state " << index;
      EXPECT_LE(variance_ratio, bound) << "state " << index;
    }
  }
  // Without the projection the floor pins the along-corridor velocity and D.5 does not hold.
  EXPECT_LT(std::sqrt(P_plain(kVelX, kVelX)) / target, 0.9);
  EXPECT_LT(plain_gain, 0.015);
}

TEST(FloorProjection, ExactZeroTheLidarAddsNothingAlongTheCorridor)
{
  // F2 (c), without the wheel. (1) At every iteration of every scan the projected rows carry no information
  // along the zeroed direction: sum_i (row_i . u)^2 <= 3.2e-9 normals, the level that would already pin the
  // velocity (calc.out:17); what is left is the rounding of V's orthonormality. (2) So the covariance does not
  // shrink along u: the along-corridor position and velocity variances follow, scan for scan, those of the same
  // corridor with exact normals, where the lidar has no along-corridor information by construction. Both grow
  // by q per scan apart from the tilt-gravity coupling they share; what differs is the second-order change of
  // the tilt information from the tilted normals (simulation: 1.4e-4 relative), within the 1e-3 bound.
  // Without the projection the same noisy corridor pins them to a tenth.
  const Filter::cov P0 = corridorPrior(1e-8, 1e-12);
  CorridorEstimator projected(makeCorridor(1.0, 0.0), P0, restingState());
  CorridorEstimator exact(makeCorridor(1.0, 0.0, false), P0, restingState());
  exact.corridor().project = false;
  CorridorEstimator plain(makeCorridor(1.0, 0.0), P0, restingState());
  plain.corridor().project = false;
  double worst_velocity = 0.0;
  double worst_position = 0.0;
  double previous = 0.0;
  double growth = 0.0;
  for (int scan = 0; scan < 200; ++scan) {
    for (CorridorEstimator * estimator : {&projected, &exact, &plain}) {
      estimator->propagate();
      estimator->lidar();
    }
    const Filter::cov & P = projected.filter().get_P();
    const Filter::cov & P_exact = exact.filter().get_P();
    worst_velocity = std::max(worst_velocity, std::abs(P(kVelX, kVelX) / P_exact(kVelX, kVelX) - 1.0));
    worst_position = std::max(worst_position, std::abs(P(0, 0) / P_exact(0, 0) - 1.0));
    growth = (P(kVelX, kVelX) - previous) / (kQ * kScanPeriod);
    previous = P(kVelX, kVelX);
  }
  RecordProperty("zeroed_information", text(projected.corridor().zeroed_information));
  RecordProperty("worst_velocity_variance_vs_exact", text(worst_velocity));
  RecordProperty("worst_position_variance_vs_exact", text(worst_position));
  RecordProperty("last_scan_growth_over_qT", text(growth));
  std::cout << "F2(c): zeroed information " << projected.corridor().zeroed_information << " normals, P_vv vs exact "
            << worst_velocity << ", P_pp vs exact " << worst_position << ", last-scan growth / qT " << growth << "\n";
  EXPECT_EQ(projected.corridor().gains(0), 0.0);
  EXPECT_LE(projected.corridor().zeroed_information, 3.2e-9);
  EXPECT_LE(worst_velocity, 1e-3);
  EXPECT_LE(worst_position, 1e-3);
  EXPECT_LT(plain.filter().get_P()(kVelX, kVelX) / exact.filter().get_P()(kVelX, kVelX), 0.5);
}

TEST(FloorProjection, WheelBeforeOrAfterTheLidarAgreesWithinOnePercentSigma)
{
  // F2 (a) where the along-corridor gain is partial: end walls lift lambda_u to 1.95, 2, 3 and 5 times the floor
  // (g = 0.11, 0.19, 0.60, 0.78; 1.95 is just past the onset at 1.927). The fixture's priors and offsets plus a
  // 3 cm/s velocity error for the wheel to correct; 50 scans with the wheel after the lidar and 50 with it
  // before; every error-state component of the two runs within 1 % of its sigma at every scan. Two linear
  // updates commute exactly; what is left is the wheel model relinearised on the rotation the lidar moved, about
  // |dv| |dtheta| K_w / sigma_v = 1e-3 sigma here, whatever the gain.
  const double tilt = std::sqrt(0.5);
  for (double ratio : {1.95, 2.0, 3.0, 5.0}) {
    const double weight = endWeightFor(tilt, ratio);
    const Filter::cov P0 = corridorPrior(1e-4, 1e-8);
    const state_ikfom x0 = restingState(Eigen::Vector3d(0.05, -0.03, 0.02), 0.02, Eigen::Vector3d(0.03, 0.0, 0.0));
    CorridorEstimator lidar_first(makeCorridor(tilt, weight), P0, x0);
    CorridorEstimator wheel_first(makeCorridor(tilt, weight), P0, x0);
    double worst = 0.0;
    int worst_scan = -1;
    int worst_index = -1;
    for (int scan = 0; scan < 50; ++scan) {
      lidar_first.propagate();
      lidar_first.lidar();
      lidar_first.wheel();
      wheel_first.propagate();
      wheel_first.wheel();
      wheel_first.lidar();
      const auto distance =
        worstSigmaDistance(lidar_first.filter().get_x(), wheel_first.filter().get_x(), lidar_first.filter().get_P());
      if (!(distance.first <= worst)) {
        worst = distance.first;
        worst_scan = scan;
        worst_index = distance.second;
      }
    }
    const double realised = lidar_first.corridor().lambda(0) / lidar_first.corridor().noise_floor(0);
    RecordProperty("worst_sigma_fraction_at_ratio_" + text(ratio), text(worst));
    std::cout << "F2(a): lambda/f " << realised << ", g " << lidar_first.corridor().gains(0) << ": worst |dx|/sigma "
              << worst << " at scan " << worst_scan << ", state " << worst_index << "\n";
    EXPECT_LT(worst, 0.01) << "lambda/f " << realised << ", scan " << worst_scan << ", state " << worst_index;
  }
}

TEST(FloorProjection, PosteriorMatchesTheClosedFormAsTheInformationRisesThroughTheFloor)
{
  // F2 (b): lambda_u swept from 0.5 f to 5 f in 450 steps of 0.01 f by end walls of growing weight, one lidar
  // update each from the same propagated prior (the fixture's priors and position offset, one scan of IMU). With
  // no attitude offset the point-to-plane residuals are linear in the error, so the update must land on the
  // linear-Gaussian posterior of the projected model linearised at the prior: it absorbs g^2 lambda in mean and
  // covariance alike, is the prior (plus its Kalman coupling) along u wherever g = 0, and is continuous in lambda.
  // Every error-state component within 1e-4 of the reference's sigma. The reference, from the unprojected rows
  // H, h at the prior, with no residual shift in it:
  //   J = H^T H, b = H^T h over position and rotation (H is zero elsewhere); V, lambda, f, g, G as the projection
  //   takes them; e = J^-1 b; D = diag(G, I); J' = D J D, b' = J' e;
  //   Lambda = R P^-1 + J', delta = Lambda^-1 b', x_ref = x_prior [+] delta, P_ref = R Lambda^-1.
  const double tilt = std::sqrt(0.5);
  CorridorEstimator propagated(
    makeCorridor(tilt, 0.0), corridorPrior(1e-8, 1e-8), restingState(Eigen::Vector3d(0.05, -0.03, 0.02), 0.0));
  propagated.propagate();
  const state_ikfom x_prior = propagated.filter().get_x();
  const Filter::cov P_prior = propagated.filter().get_P();
  const Filter::cov prior_information = kLidarR * P_prior.inverse();
  const auto bare = alongCorridorInformation(tilt, 0.0);
  const auto unit = alongCorridorInformation(tilt, 1.0);
  const fast_lio::FloorParams params;
  const double z_alpha = *fast_lio::zFromFalseKeepRate(params.alpha);
  const double ridge = fast_lio::DegeneracyParams{}.schur_ridge;
  const int steps = 450;
  std::vector<double> realised;
  double worst = 0.0;
  int worst_step = -1;
  int worst_index = -1;
  int zero_gain_steps = 0;
  for (int j = 0; j <= steps; ++j) {
    const double ratio = 0.5 + 4.5 * j / steps;
    const double weight = std::sqrt(std::max(ratio * bare.second - bare.first, 0.0) / (unit.first - bare.first));

    Corridor unprojected = makeCorridor(tilt, weight);
    unprojected.project = false;
    g_corridor = &unprojected;
    state_ikfom at_prior = x_prior;
    esekfom::dyn_share_datastruct<double> rows;
    corridorModel(at_prior, rows);
    g_corridor = nullptr;
    const Eigen::MatrixXd H = rows.h_x.leftCols(6);
    const Eigen::Matrix<double, 6, 6> J = H.transpose() * H;
    const Eigen::Matrix<double, 6, 1> b = H.transpose() * rows.h;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(
      fast_lio::schurPositionInformation(J.block<3, 3>(0, 0), J.block<3, 3>(0, 3), J.block<3, 3>(3, 3), ridge));
    fast_lio::DirectionFloorAccumulator accumulator(solver.eigenvectors());
    for (const CorridorPoint & point : unprojected.points) {
      accumulator.add(point.weight, point.cov);
    }
    const Eigen::Vector3d g =
      fast_lio::floorKeepGains(solver.eigenvalues(), accumulator.floor(params.kappa), z_alpha, params.rel_uncertainty);
    Eigen::Matrix<double, 6, 6> D = Eigen::Matrix<double, 6, 6>::Identity();
    D.block<3, 3>(0, 0) = fast_lio::projectionFromGains(solver.eigenvectors(), g);
    const Eigen::Matrix<double, 6, 6> J_projected = D * J * D;
    const Eigen::Matrix<double, 6, 1> b_projected = J_projected * J.ldlt().solve(b);
    Filter::cov information = prior_information;
    information.block<6, 6>(0, 0) += J_projected;
    Filter::vectorized_state rhs = Filter::vectorized_state::Zero();
    rhs.head<6>() = b_projected;
    const Filter::vectorized_state delta = information.ldlt().solve(rhs);
    state_ikfom x_ref = x_prior;
    x_ref.boxplus(delta);
    const Filter::cov P_ref = kLidarR * information.inverse();

    CorridorEstimator estimator(makeCorridor(tilt, weight), P_prior, x_prior);
    estimator.lidar();
    const auto distance = worstSigmaDistance(x_ref, estimator.filter().get_x(), P_ref);
    if (worst_step < 0 || !(distance.first <= worst)) {
      worst = distance.first;
      worst_step = j;
      worst_index = distance.second;
    }
    realised.push_back(estimator.corridor().lambda(0) / estimator.corridor().noise_floor(0));
    zero_gain_steps += g(0) == 0.0 ? 1 : 0;
  }
  RecordProperty("worst_sigma_fraction_vs_closed_form", text(worst));
  RecordProperty("zero_gain_steps", std::to_string(zero_gain_steps));
  std::cout << "F2(b): worst |x - x_ref| / sigma_ref " << worst << " at lambda/f " << realised[worst_step] << ", state "
            << worst_index << "; " << zero_gain_steps << " of " << steps + 1
            << " steps keep nothing along the corridor\n";
  EXPECT_LE(realised.front(), 0.51);
  EXPECT_GE(realised.back(), 4.95);
  EXPECT_LE(worst, 1e-4) << "lambda/f " << realised[worst_step] << ", state " << worst_index;
}

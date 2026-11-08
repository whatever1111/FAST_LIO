// The consider update on the real estimator: a synthetic point-to-plane scene (floor + walls) whose residuals ask
// for a pitch correction. Plain, the iterated update recovers the pose; with the rotation considered, roll/pitch
// stay at the propagated value, the position still updates, and the roll/pitch variance is kept.
// The same scene drives the filter's two dyn_share entries side by side: a model bound with a context
// (init_dyn_share_context) must be called, and leave the filter, exactly as the plain entry does.
#include <Eigen/Dense>

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include "rp_consider.hpp"
#include "use-ikfom.hpp"

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

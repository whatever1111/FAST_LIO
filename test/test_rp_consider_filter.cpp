// The consider update on the real estimator: a synthetic point-to-plane scene (floor + walls) whose residuals ask
// for a pitch correction. Plain, the iterated update recovers the pose; with the rotation considered, roll/pitch
// stay at the propagated value, the position still updates, and the roll/pitch variance is kept.
#include <gtest/gtest.h>
#include <omp.h>

#include <Eigen/Dense>

#include <cmath>
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

Eigen::Vector3d rollPitchYaw(const Eigen::Matrix3d & R)
{
  return Eigen::Vector3d(std::atan2(R(2, 1), R(2, 2)), -std::asin(R(2, 0)), std::atan2(R(1, 0), R(0, 0)));
}

struct Fixture
{
  Filter filter;
  Eigen::Matrix3d R_true = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_true = Eigen::Vector3d(0.0, 0.0, 1.0);
  Eigen::Vector3d prior_rpy;
  Scene scene;
  double pitch_prior = 0.02;
  Eigen::Vector3d up_body = Eigen::Vector3d::UnitZ();  // the body-frame vertical of the prior attitude

  Fixture()
  {
    scene = makeScene(R_true, t_true);
    g_scene = &scene;
    double epsilon[23];
    std::fill(std::begin(epsilon), std::end(epsilon), 0.001);
    filter.init_dyn_share(get_f, df_dx, df_dw, sceneModel, 6, epsilon);
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

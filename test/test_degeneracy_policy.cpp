// test_degeneracy_policy.cpp — unit tests for fast_lio::schurPositionInformation, the position information left once
// rotation has been solved for.

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "degeneracy_policy.hpp"

using fast_lio::schurPositionInformation;

namespace
{
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

/// The three blocks of H^T H the complement is built from, the way the measurement model builds its rows:
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

// rebuild_relevel.hpp — level the local frame of a rebuilt map that founds a new coordinate epoch.
//
// A local-map rebuild seeds the new map from the current pose. Whatever roll/pitch error that pose carries goes
// into the map, and from then on the attitude, the gravity state and the map agree on it: the tilt is frozen into
// the new epoch, and the next rebuild adds whatever grew on top of it. Before the map is seeded, the frame is turned
// by the minimal rotation L about the current position that brings the reference vertical onto +Z. Its axis is
// horizontal, so nothing turns about the vertical (yaw stays) and the position does not move.
//
// The reference is the levelling window's specific force when the legs and the IMU call the body parked at both
// window edges: the window's mean linear acceleration is then exactly zero, so its mean is gravity whatever happened
// in between, the same contract as the static IMU initialisation. Otherwise the reference is the gravity state
// alone, a change of coordinates that leaves every belief of the filter as it was. A single walking window is never
// the reference: it is off by degrees in its tail, and levelling onto it is a gravity initialisation in motion,
// which tilts the frame it founds.
//
// The state follows the frame: R ← L·R, v ← L·v, g vertical; the biases and the extrinsics are body-frame and stay.
// The covariance follows it too: its world-frame rows (position, velocity) and the gravity tangent are carried into
// the new frame, and a window reference, which assigns g, gives the gravity block the measurement's variance.
//
// Pure C++ + Eigen. No ROS, no estimator types: the caller passes the 23x23 IKFoM covariance and the S2 tangent
// bases of the gravity state before and after.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

#include "covariance_reset.hpp"
#include "gravity_align_kinematics.hpp"
#include "imu_gap_prior.hpp"

namespace fast_lio
{

/// FAST-LIO error-state rows: position at kStatePosIndex (imu_gap_prior.hpp), velocity at kStateVelIndex
/// (covariance_reset.hpp), the gravity tangent at 21-22.
constexpr int kStateGravIndex = 21;

/// The FAST-LIO error-state covariance.
using StateCovariance = Eigen::Matrix<double, 23, 23>;

/// Whether a rebuild re-levels the frame. All four must hold:
///  - the switch is on;
///  - LOST recovery is on: only then is the rebuilt map the one a new coordinate epoch is committed on;
///  - no prior map is pinned: the delivered map defines the frame, and the rebuild re-registers against it;
///  - the first local reference was built: a rebuild before it founds nothing.
inline bool relevelAllowed(bool relevel_enabled, bool lost_recovery_enabled, bool prior_pinned, bool reference_built)
{
  return relevel_enabled && lost_recovery_enabled && !prior_pinned && reference_built;
}

enum class RelevelSource
{
  kWindow,   ///< the levelling window, parked at both edges
  kGravity,  ///< the gravity state
};

inline const char * relevelSourceName(RelevelSource source)
{
  return source == RelevelSource::kWindow ? "window" : "gravity";
}

struct RelevelPlan
{
  bool valid = false;  ///< false: the reference does not point above the horizon, and no rotation is defined
  RelevelSource source = RelevelSource::kGravity;
  Eigen::Matrix3d L = Eigen::Matrix3d::Identity();  ///< world-frame rotation; its axis is horizontal
  double angle = 0.0;                               ///< rad, the tilt L removes
};

/// The rotation that levels the frame. R is the attitude (body to world), gravity the gravity state (world frame,
/// pointing down), window the levelling window read as a reference: its body-frame up is used only when it is
/// parked. The minimal rotation is undefined at the antipode, and a reference at or below the horizon is no
/// vertical: either gives no plan.
inline RelevelPlan
planRelevel(const Eigen::Matrix3d & R, const Eigen::Vector3d & gravity, const LevellingWindowReference & window)
{
  RelevelPlan plan;
  plan.source = window.parked ? RelevelSource::kWindow : RelevelSource::kGravity;
  const Eigen::Vector3d up = window.parked ? Eigen::Vector3d(R * window.up_body) : Eigen::Vector3d(-gravity);
  if (!up.allFinite() || !(up.z() > 0.0)) {
    return plan;
  }
  plan.valid = true;
  plan.L = Eigen::Quaterniond::FromTwoVectors(up, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  plan.angle = std::atan2(up.head<2>().norm(), up.z());
  return plan;
}

/// Carries P into the frame turned by L: P ← J·P·Jᵀ with J the identity except for the position and velocity
/// blocks (L: world-frame vectors) and the gravity tangent (B_newᵀ·L·B_old, from the old tangent basis to the new
/// one; both bases orthonormal and tangent to their gravity). The rotation block (a right perturbation), the
/// extrinsics and the biases are body-frame and keep every element.
inline void rotateWorldCovariance(StateCovariance & P,
                                  const Eigen::Matrix3d & L,
                                  const Eigen::Matrix<double, 3, 2> & B_old,
                                  const Eigen::Matrix<double, 3, 2> & B_new)
{
  StateCovariance J = StateCovariance::Identity();
  J.block<3, 3>(kStatePosIndex, kStatePosIndex) = L;
  J.block<3, 3>(kStateVelIndex, kStateVelIndex) = L;
  J.block<2, 2>(kStateGravIndex, kStateGravIndex) = B_new.transpose() * L * B_old;
  P = J * P * J.transpose();
}

/// The gravity state was assigned from a measurement: its tangent block takes that measurement's variance (rad², finite
/// and not negative) on both axes and loses every cross term, as covariance_reset.hpp does for an assigned velocity.
inline void assignGravityBlock(StateCovariance & P, double variance)
{
  P.middleRows<2>(kStateGravIndex).setZero();
  P.middleCols<2>(kStateGravIndex).setZero();
  P(kStateGravIndex, kStateGravIndex) = variance;
  P(kStateGravIndex + 1, kStateGravIndex + 1) = variance;
}

}  // namespace fast_lio

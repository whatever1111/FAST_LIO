// The re-anchor gate's evidence against the entry view (recovery_view_match.hpp): h_share_model's
// correspondence rules over what iVox's entry view shows, so that a pending recovery cannot be qualified
// by points its own scans put into the map.

#include <Eigen/Geometry>
#include <pcl/point_types.h>

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "correspondence_gate.hpp"
#include "ivox/ivox.hpp"
#include "reanchor_gate.hpp"
#include "recovery_view_match.hpp"

namespace
{
using Ivox = lio_ivox::IVox<pcl::PointXYZ>;
using Cloud = Ivox::PointVector;
constexpr int kNeighbours = 5;       // h_share_model's NUM_MATCH_POINTS
constexpr double kPointCov = 0.001;  // laserMapping's LASER_POINT_COV

pcl::PointXYZ pointAt(float x, float y, float z)
{
  pcl::PointXYZ p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

// The map the recovery began with: a floor, z = 0 over 4 m x 4 m, a point every 0.1 m.
Ivox floorMap()
{
  Ivox map;
  map.Init(0.5f, 26, 50, 100000);
  map.set_downsample_param(0.1f);
  Cloud floor;
  for (int i = 0; i < 40; ++i)
    for (int j = 0; j < 40; ++j)
      floor.push_back(pointAt(0.1f * static_cast<float>(i) + 0.05f, 0.1f * static_cast<float>(j) + 0.05f, 0.0f));
  map.Add_Points(floor, true);
  return map;
}

// One scan of that floor, 1-3 m ahead of the sensor and to its left, in the body frame.
Cloud floorScan()
{
  Cloud scan;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      scan.push_back(pointAt(1.0f + 0.2f * static_cast<float>(i), 1.0f + 0.2f * static_cast<float>(j), 0.0f));
  return scan;
}

// The M20 profile's correspondence gate (corr_innov_gate_k, corr_gate_mad_scale, corr_gate_rot_sigma_floor_deg).
fast_lio::CorrespondenceGateParams m20Gate()
{
  fast_lio::CorrespondenceGateParams gate;
  gate.innovation_k = 2.5;
  gate.mad_scale = 1.0;
  gate.rot_sigma_floor_deg = 0.3;
  return gate;
}

Eigen::Matrix<double, 6, 6> priorBlock()
{
  Eigen::Matrix<double, 6, 1> diagonal;
  diagonal << 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5;
  return diagonal.asDiagonal();
}

fast_lio::RecoveryViewMatch matchScan(const Ivox & map,
                                      const Cloud & body,
                                      const fast_lio::RecoveryViewPose & pose,
                                      bool in_view,
                                      const fast_lio::CorrespondenceGateParams & gate)
{
  const auto search = [&map, in_view](const pcl::PointXYZ & p, int k, Cloud & near, std::vector<float> & sqdist) {
    if (in_view)
      map.Nearest_Search_View(p, k, near, sqdist);
    else
      map.Nearest_Search(p, k, near, sqdist);
  };
  return fast_lio::matchRecoveryView<kNeighbours, Cloud>(body, pose, search, gate, priorBlock(), kPointCov);
}

// The scan's points where `pose` puts them: what the scan path inserts.
Cloud inWorld(const Cloud & body, const fast_lio::RecoveryViewPose & pose)
{
  Cloud world;
  for (const auto & p : body) {
    const Eigen::Vector3d w = pose.rot * (pose.rot_li * Eigen::Vector3d(p.x, p.y, p.z) + pose.pos_li) + pose.pos;
    world.push_back(pointAt(static_cast<float>(w.x()), static_cast<float>(w.y()), static_cast<float>(w.z())));
  }
  return world;
}

fast_lio::ReanchorParams gateParams()
{
  fast_lio::ReanchorParams params;
  params.min_eff = 50;
  params.max_res = 0.10;
  return params;
}
}  // namespace

TEST(RecoveryViewMatch, WithNothingAddedSinceTheOpeningTheViewMatchesAsTheMapDoes)
{
  Ivox map = floorMap();
  ASSERT_TRUE(map.BeginEntryView());
  // Close to the floor, not on it: every residual is different and well inside both gates.
  fast_lio::RecoveryViewPose pose;
  pose.pos = Eigen::Vector3d(0.02, -0.01, 0.01);
  pose.rot = Eigen::Quaterniond(Eigen::AngleAxisd(0.5 * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
                                Eigen::AngleAxisd(0.2 * M_PI / 180.0, Eigen::Vector3d::UnitX()));
  pose.pos_li = Eigen::Vector3d(0.05, 0.0, 0.0);
  const Cloud scan = floorScan();
  for (const auto & gate : {fast_lio::CorrespondenceGateParams{}, m20Gate()}) {
    const fast_lio::RecoveryViewMatch whole = matchScan(map, scan, pose, false, gate);
    const fast_lio::RecoveryViewMatch view = matchScan(map, scan, pose, true, gate);
    ASSERT_GT(whole.effective, 0);
    EXPECT_EQ(view.effective, whole.effective);
    EXPECT_EQ(view.res_mean, whole.res_mean);
  }
}

TEST(RecoveryViewMatch, PointsTheScanPutIntoTheMapCannotQualifyIt)
{
  Ivox map = floorMap();
  ASSERT_TRUE(map.BeginEntryView());
  const Cloud scan = floorScan();
  // The recovery's estimate is 0.5 m above the floor, and its scans went into the map where it put them.
  fast_lio::RecoveryViewPose lifted;
  lifted.pos = Eigen::Vector3d(0.0, 0.0, 0.5);
  Cloud own = inWorld(scan, lifted);
  ASSERT_EQ(map.Add_Points(own, true), static_cast<int>(own.size()));

  for (const auto & gate : {fast_lio::CorrespondenceGateParams{}, m20Gate()}) {
    const fast_lio::RecoveryViewMatch whole = matchScan(map, scan, lifted, false, gate);
    EXPECT_TRUE(fast_lio::reanchorMatchOk(whole.effective, whole.res_mean, gateParams()))
      << "against the whole map its own points vouch for it: effct=" << whole.effective << " res=" << whole.res_mean;
    const fast_lio::RecoveryViewMatch view = matchScan(map, scan, lifted, true, gate);
    EXPECT_EQ(view.effective, 0) << "the view holds only the floor, 0.5 m below every point";
    EXPECT_TRUE(std::isnan(view.res_mean));
    EXPECT_FALSE(fast_lio::reanchorMatchOk(view.effective, view.res_mean, gateParams()));

    // Where the floor is, the view qualifies the same scan.
    const fast_lio::RecoveryViewMatch placed = matchScan(map, scan, fast_lio::RecoveryViewPose{}, true, gate);
    EXPECT_EQ(placed.effective, static_cast<int>(scan.size()));
    EXPECT_TRUE(fast_lio::reanchorMatchOk(placed.effective, placed.res_mean, gateParams()));
  }
}

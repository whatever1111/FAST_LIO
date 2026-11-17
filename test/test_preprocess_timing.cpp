#include <pcl_conversions/pcl_conversions.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "preprocess.h"

TEST(PreprocessTiming, PresentTimeWithZeroTailPreservesStrideAndTimes)
{
  pcl::PointCloud<velodyne_ros::Point> points;
  for (float time : {0.02f, 0.09f, 0.04f, 0.08f, 0.0f}) {
    velodyne_ros::Point point{};
    point.x = 1.0f;
    point.time = time;
    points.push_back(point);
  }
  auto message = std::make_unique<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(points, *message);
  Preprocess preprocess;
  preprocess.set(false, VELO16, 0.1, 2);
  preprocess.time_unit = SEC;
  preprocess.N_SCANS = 16;
  auto output = std::make_shared<PointCloudXYZI>();
  preprocess.process(message, output);
  ASSERT_EQ(output->size(), 3u);
  EXPECT_NEAR(output->points[0].curvature, 20.0, 0.00001);
  EXPECT_NEAR(output->points[1].curvature, 40.0, 0.00001);
  EXPECT_FLOAT_EQ(output->points[2].curvature, 0.0f);
}

// Rings against scan_line: a Velodyne scan whose points name rings scan_line - 1 and 0, which
// exist, and scan_line and 65535, which do not.

namespace
{

constexpr int kScanLine = 16;

struct ScanPoint
{
  std::uint16_t ring;
  int tag;  // bearing -tag degrees, intensity tag, and with a time field (tag + 1) * 10 ms
};

const std::vector<ScanPoint> kScan{{kScanLine - 1, 0}, {kScanLine, 1}, {0, 2}, {65535, 3}, {kScanLine - 1, 4},
                                   {kScanLine, 5}, {0, 6}, {65535, 7}, {kScanLine - 1, 8}};

std::unique_ptr<sensor_msgs::msg::PointCloud2> velodyneScan(const std::vector<ScanPoint> & scan, bool with_time)
{
  pcl::PointCloud<velodyne_ros::Point> points;
  for (const ScanPoint & scan_point : scan) {
    const double bearing = -scan_point.tag * M_PI / 180.0;
    velodyne_ros::Point point{};
    point.x = static_cast<float>(5.0 * std::cos(bearing));
    point.y = static_cast<float>(5.0 * std::sin(bearing));
    point.intensity = static_cast<float>(scan_point.tag);
    point.time = 0.01f * static_cast<float>(scan_point.tag + 1);
    point.ring = scan_point.ring;
    points.push_back(point);
  }
  auto message = std::make_unique<sensor_msgs::msg::PointCloud2>();
  pcl::toROSMsg(points, *message);
  if (!with_time) {
    auto & fields = message->fields;
    fields.erase(std::remove_if(fields.begin(), fields.end(), [](const auto & field) { return field.name == "time"; }),
                 fields.end());
  }
  return message;
}

// Velodyne without feature extraction, with a filter that keeps every point.
PointCloudXYZI preprocessed(const std::unique_ptr<sensor_msgs::msg::PointCloud2> & message)
{
  Preprocess preprocess;
  preprocess.set(false, VELO16, 0.1, 1);
  preprocess.time_unit = SEC;
  preprocess.N_SCANS = kScanLine;
  auto output = std::make_shared<PointCloudXYZI>();
  preprocess.process(message, output);
  return *output;
}

}  // namespace

// Without a time field a point's time comes from its bearing against the first point of its ring,
// kept per ring. A ring outside scan_line has no such slot: its points are dropped, and the points
// of the rings that exist come out as from the same scan without them, in order and with the same
// times.
TEST(PreprocessTiming, WithoutTimeFieldPointsOfRingsOutsideScanLineAreDropped)
{
  std::vector<ScanPoint> in_range;
  std::copy_if(kScan.begin(), kScan.end(), std::back_inserter(in_range),
               [](const ScanPoint & point) { return point.ring < kScanLine; });
  const PointCloudXYZI output = preprocessed(velodyneScan(kScan, false));
  const PointCloudXYZI reference = preprocessed(velodyneScan(in_range, false));

  // The first point of each ring anchors its bearing and is not emitted.
  ASSERT_EQ(output.size(), 3u);
  EXPECT_EQ(output.points[0].intensity, 4.0f);
  EXPECT_EQ(output.points[1].intensity, 6.0f);
  EXPECT_EQ(output.points[2].intensity, 8.0f);
  ASSERT_EQ(reference.size(), output.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output.points[i].x, reference.points[i].x);
    EXPECT_EQ(output.points[i].y, reference.points[i].y);
    EXPECT_EQ(output.points[i].intensity, reference.points[i].intensity);
    EXPECT_EQ(output.points[i].curvature, reference.points[i].curvature);
    EXPECT_GT(output.points[i].curvature, 0.0f);
  }
}

// With a time field each point carries its own time and the ring indexes nothing: every point is
// kept, in order, with its field time.
TEST(PreprocessTiming, WithTimeFieldEveryPointKeepsItsOwnTime)
{
  const PointCloudXYZI output = preprocessed(velodyneScan(kScan, true));

  ASSERT_EQ(output.size(), kScan.size());
  for (std::size_t i = 0; i < kScan.size(); ++i) {
    EXPECT_EQ(output.points[i].intensity, static_cast<float>(kScan[i].tag));
    EXPECT_NEAR(output.points[i].curvature, 10.0 * (kScan[i].tag + 1), 0.0001);
  }
}

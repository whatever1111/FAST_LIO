#include <pcl_conversions/pcl_conversions.h>

#include <gtest/gtest.h>

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

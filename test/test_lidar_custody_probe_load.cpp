#include <rclcpp_components/node_factory.hpp>

#include <class_loader/class_loader.hpp>
#include <gtest/gtest.h>

TEST(LidarCustodyProbeLoad, LoaderHasProbeAndMappingFactories)
{
  class_loader::ClassLoader loader(FASTLIO_CUSTODY_PROBE_PATH);
  using Factory = rclcpp_components::NodeFactory;
  EXPECT_TRUE(loader.isClassAvailable<Factory>("rclcpp_components::NodeFactoryTemplate<LidarCustodyTestProbe>"));
  EXPECT_TRUE(loader.isClassAvailable<Factory>("rclcpp_components::NodeFactoryTemplate<LaserMappingNode>"));
}

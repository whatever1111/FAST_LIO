#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <class_loader/class_loader.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "ivox/ivox.hpp"
#include "laser_mapping_test_types.hpp"

extern lio_ivox::IVox<PointType> ikdtree;

namespace
{
// Each case has its own CTest process: the production node owns file-scope state.
class IVoxNodeCapacity : public ::testing::Test
{
protected:
  void SetUp() override { rclcpp::init(0, nullptr); }
  void TearDown() override { rclcpp::shutdown(); }

  void constructAndCheck(int requested, std::size_t expected)
  {
    class_loader::ClassLoader loader(FASTLIO_COMPONENT_PATH);
    const auto names = loader.getAvailableClasses<rclcpp_components::NodeFactory>();
    ASSERT_EQ(names.size(), 1u);
    auto factory = loader.createInstance<rclcpp_components::NodeFactory>(names.front());
    rclcpp::NodeOptions options;
    options.parameter_overrides(
      {rclcpp::Parameter("ivox_max_voxels", requested),
       rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0}),
       rclcpp::Parameter("mapping.extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})});
    if (requested < 0) {
      // The refusal has to name the parameter an operator must fix, not just fail.
      try {
        factory->create_node_instance(options);
        ADD_FAILURE() << "a negative ivox_max_voxels was accepted";
      } catch (const std::invalid_argument & error) {
        EXPECT_NE(std::string(error.what()).find("ivox_max_voxels"), std::string::npos) << error.what();
      }
      return;
    }
    auto node = factory->create_node_instance(options);
    ASSERT_NE(node.get_node_base_interface(), nullptr);
    EXPECT_EQ(ikdtree.maxVoxels(), expected);
    if (requested > 0) {
      PointVector points;
      for (int i = 0; i < requested + 2; ++i) {
        PointType point{};
        point.x = static_cast<float>(i * 10);
        points.push_back(point);
      }
      ikdtree.Add_Points(points, false);
      EXPECT_EQ(ikdtree.numVoxels(), expected);
      EXPECT_EQ(ikdtree.evictedVoxels(), 2u);
    }
  }
};
#ifdef FASTLIO_CAPACITY_negative
TEST_F(IVoxNodeCapacity, NegativeRejected)
{
  constructAndCheck(-1, 0);
}
#endif
#ifdef FASTLIO_CAPACITY_zero
TEST_F(IVoxNodeCapacity, ZeroSelectsDefault)
{
  constructAndCheck(0, 5000000u);
}
#endif
#ifdef FASTLIO_CAPACITY_small
TEST_F(IVoxNodeCapacity, SmallCapacityIsEnforced)
{
  constructAndCheck(2, 2u);
}
#endif
}  // namespace

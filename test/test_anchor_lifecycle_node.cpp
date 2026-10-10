// The stationary anchor on the production call chain.
//
// The real LaserMappingNode, fed through its real IMU and point-cloud
// subscriptions, takes every scan through the production scan path, the map
// reset in rebuild_local_map and the /Odometry publisher. FASTLIO_TEST_HOOKS
// fix only what a synthetic stream cannot produce on demand: the divergence
// guard's verdict, the state registration hands to the anchor step, and a
// runaway trip for the late map reset. The static verdict is the real one,
// computed from the IMU windows the fixture sends.
//
// The anchor and pre-event positions are the three frozen cases F1-F3.
// Attitude, velocity and covariance are fixed fixture values (identity, a
// nonzero velocity below the platform's maximum, a positive-definite
// diagonal), not a recorded filter state.
//
// One process can activate the component once (its globals outlive a node), so
// every scenario runs in a forked child that reports what the node did; the
// parent makes the assertions.

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <class_loader/class_loader.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include "anchor_trace.hpp"
#include "laser_mapping_test_hooks.hpp"
#include "preprocess.h"

namespace
{

constexpr const char * kLaserMappingFactory = "rclcpp_components::NodeFactoryTemplate<LaserMappingNode>";
constexpr const char * kImuTopic = "/test_anchor_lifecycle/imu";
constexpr const char * kLidarTopic = "/test_anchor_lifecycle/points";
constexpr double kFirstScan = 1.0;    // s, start of the first scan
constexpr double kScanPeriod = 0.1;   // s
constexpr double kScanLength = 0.05;  // s, last point time within a scan
constexpr std::size_t kScanPoints = 2000;
constexpr std::int64_t kImuPeriodNs = 5000000;
constexpr std::int64_t kImuFirstNs = 992500000;  // 2.5 ms off the scan grid: no sample sits on a scan end
constexpr double kMovingYawRate = 0.3;           // rad/s, well above zupt_gyro_thresh
constexpr double kGravity = 9.81;
constexpr double kGapMargin = 0.010;  // s of IMU kept at each end of a gap scan's window
constexpr int kWarmupPublished = 2;
constexpr int kWarmupMaxScans = 12;
constexpr double kScanWaitSeconds = 20.0;
constexpr unsigned kChildAlarmSeconds = 300;

constexpr std::array<double, 3> kFixtureVelocity = {0.4, -0.3, 0.05};  // m/s, below divergence_guard_max_speed
constexpr std::array<double, 23> kFixtureCovDiag = {1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6,
                                                    1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2, 1e-6,
                                                    1e-6, 1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4};

constexpr unsigned kFlagDegraded = 1u << 0;
constexpr unsigned kFlagStaticHold = 1u << 1;

struct FieldCase
{
  const char * name;
  std::array<double, 3> old_anchor;
  std::array<double, 3> pre_event;
};

// Frozen cases: the old anchor and the position of the scan before the event.
constexpr FieldCase kFieldCases[] = {
  {"F1", {28.41, 8.25, -5.21}, {36.33, 25.37, -39.25}},
  {"F2", {33.49, 7.60, -13.59}, {35.15, 24.09, -103.88}},
  {"F3", {36.48, -1.34, -27.41}, {8.62, 10.77, -56.78}},
};

enum class Imu : std::uint8_t
{
  kStatic,
  kMoving,
  kGap,
};

enum class Guard : std::int8_t
{
  kNatural = -1,
  kHealthy = 0,
  kDegraded = 1,
};

enum class Phase : int
{
  kWarmup,
  kAnchorMove,    // healthy, moving: the scan whose pose becomes the anchor
  kAnchorStatic,  // healthy, parked at the anchor
  kMove,          // degraded and moving: the event that ends the stationary stretch
  kMoveDegraded,  // degraded and moving on
  kStatic,        // parked again (degraded or healthy per scenario), at the pre-event pose
  kGap,           // IMU coverage gap: the scan path returns before the anchor step
  kHold,          // degraded and parked at the anchor: a legitimate pin
  kHoldReset,     // the same, with a runaway map reset after the anchor step
  kAfterReset,    // degraded and parked after the reset
  kNatural,       // no hooks beyond the record
};

struct Step
{
  Phase phase = Phase::kNatural;
  Imu imu = Imu::kStatic;
  Guard guard = Guard::kNatural;
  bool inject = false;
  std::array<double, 3> pos = {0.0, 0.0, 0.0};
  bool trip = false;
};

struct Scenario
{
  std::string name;
  std::vector<Step> steps;
  std::map<std::string, rclcpp::Parameter> params;
  bool trace = false;
};

struct Observation
{
  int step = -1;  // index into Scenario::steps; -1 = warm-up
  int phase = 0;
  fast_lio::AnchorScanRecord record;
  int odom_seen = 0;
  double odom_p[3] = {0.0, 0.0, 0.0};
  double odom_q[4] = {0.0, 0.0, 0.0, 0.0};  // w, x, y, z
  double odom_cov[36] = {};
  double odom_twist_cov[36] = {};
};
static_assert(std::is_trivially_copyable<Observation>::value, "observations cross the pipe as bytes");

struct ChildResult
{
  int status = -1;
  std::string error;
  std::vector<Observation> observations;
  std::vector<std::string> lines;

  bool ok() const { return WIFEXITED(status) && WEXITSTATUS(status) == 0 && error.empty(); }
  std::string describe() const
  {
    std::ostringstream out;
    out << "child status=" << status;
    if (WIFSIGNALED(status))
      out << " signal=" << WTERMSIG(status);
    out << " error='" << error << "' observations=" << observations.size() << " trace lines=" << lines.size();
    return out.str();
  }
  std::vector<const Observation *> phase(Phase wanted) const
  {
    std::vector<const Observation *> out;
    for (const auto & o : observations)
      if (o.phase == static_cast<int>(wanted))
        out.push_back(&o);
    return out;
  }
};

std::int64_t msKey(double seconds)
{
  return static_cast<std::int64_t>(std::llround(seconds * 1000.0));
}

bool sameBits(double a, double b)
{
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool samePosition(const double p[3], const std::array<double, 3> & expected)
{
  return sameBits(p[0], expected[0]) && sameBits(p[1], expected[1]) && sameBits(p[2], expected[2]);
}

double distance(const double p[3], const std::array<double, 3> & q)
{
  return std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
}

double distance(const std::array<double, 3> & p, const std::array<double, 3> & q)
{
  const double a[3] = {p[0], p[1], p[2]};
  return distance(a, q);
}

std::array<double, 3> lerp(const std::array<double, 3> & a, const std::array<double, 3> & b, double t)
{
  return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
}

std::string formatPosition(const double p[3])
{
  std::ostringstream out;
  out.precision(17);
  out << '[' << p[0] << ' ' << p[1] << ' ' << p[2] << ']';
  return out.str();
}

unsigned odomFlags(const Observation & o)
{
  return o.odom_seen ? static_cast<unsigned>(o.odom_twist_cov[1]) : 0xffffffffu;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

Step step(Phase phase, Imu imu, Guard guard, const std::array<double, 3> & pos, bool trip = false)
{
  Step s;
  s.phase = phase;
  s.imu = imu;
  s.guard = guard;
  s.inject = true;
  s.pos = pos;
  s.trip = trip;
  return s;
}

void setParam(Scenario & scenario, const rclcpp::Parameter & param)
{
  scenario.params[param.get_name()] = param;
}

Scenario baseParams(const std::string & name, bool trace)
{
  Scenario s;
  s.name = name;
  s.trace = trace;
  for (const auto & p :
       {rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0}),
        rclcpp::Parameter("mapping.extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
        rclcpp::Parameter("common.lid_topic", kLidarTopic),
        rclcpp::Parameter("common.imu_topic", kImuTopic),
        rclcpp::Parameter("preprocess.lidar_type", 3),
        rclcpp::Parameter("preprocess.timestamp_unit", 3),
        rclcpp::Parameter("point_filter_num", 1),
        rclcpp::Parameter("feature_extract_enable", false),
        rclcpp::Parameter("filter_size_surf", 0.001),
        rclcpp::Parameter("filter_size_map", 0.05),
        rclcpp::Parameter("imu_init_require_still", false),
        rclcpp::Parameter("publish.scan_publish_en", false),
        rclcpp::Parameter("publish.path_en", false),
        rclcpp::Parameter("runtime_pos_log_enable", false),
        // The re-anchor gate would hold a recovered scan in verification against a map the
        // fixture never matches; with it off the anchor logic sees the guard's verdict directly.
        rclcpp::Parameter("reanchor_en", false),
        rclcpp::Parameter("runaway_watchdog_en", false),
        rclcpp::Parameter("zupt_en", true),
        rclcpp::Parameter("zupt_parked_hold_en", false),
        rclcpp::Parameter("zupt_anchor_trace_en", trace)}) {
    setParam(s, p);
  }
  return s;
}

// Healthy and moving at the old anchor, parked there, then a degraded moving stretch toward the pre-event pose.
void appendAnchorThenMove(Scenario & s, const FieldCase & c)
{
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 4; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  s.steps.push_back(step(Phase::kMove, Imu::kMoving, Guard::kDegraded, lerp(c.old_anchor, c.pre_event, 0.2)));
  for (int i = 2; i <= 4; ++i)
    s.steps.push_back(
      step(Phase::kMoveDegraded, Imu::kMoving, Guard::kDegraded, lerp(c.old_anchor, c.pre_event, 0.2 * i)));
}

Scenario parkedAgainDegraded(const FieldCase & c, bool trace = false)
{
  Scenario s = baseParams(std::string(c.name) + "_degraded", trace);
  appendAnchorThenMove(s, c);
  for (int i = 0; i < 3; ++i)
    s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kDegraded, c.pre_event));
  return s;
}

// The same, with a runaway trip on the first parked-again scan: rebuild_local_map runs after the anchor step.
Scenario parkedAgainWithLateReset(const FieldCase & c, bool trace = false)
{
  Scenario s = baseParams(std::string(c.name) + "_reset", trace);
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  appendAnchorThenMove(s, c);
  s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kDegraded, c.pre_event, true));
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kDegraded, c.pre_event));
  return s;
}

// Parked again on HEALTHY scans with the parked prior on: the healthy branch's soft pull.
Scenario parkedAgainHealthy(const FieldCase & c)
{
  Scenario s = baseParams(std::string(c.name) + "_parked", false);
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_scans", 3));
  appendAnchorThenMove(s, c);
  for (int i = 0; i < 5; ++i)
    s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kHealthy, c.pre_event));
  return s;
}

// No motion the IMU could see, but an IMU coverage gap: the scan path returns before the anchor step, and the
// pose that comes back is elsewhere.
Scenario coverageGapThenParked(const FieldCase & c)
{
  Scenario s = baseParams(std::string(c.name) + "_coverage", false);
  setParam(s, rclcpp::Parameter("reanchor_en", true));  // imu_coverage.max_gap_s requires it
  setParam(s, rclcpp::Parameter("imu_coverage.max_gap_s", 0.02));
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 4; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  Step gap;
  gap.phase = Phase::kGap;
  gap.imu = Imu::kGap;
  gap.guard = Guard::kHealthy;
  s.steps.push_back(gap);
  for (int i = 0; i < 3; ++i)
    s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kDegraded, c.pre_event));
  return s;
}

// A legitimate pin (same stationary stretch), then a runaway map reset after the anchor step of a pinned scan.
Scenario pinThenLateReset(bool trace = false)
{
  const FieldCase & c = kFieldCases[0];
  Scenario s = baseParams("pin_late_reset", trace);
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 4; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  const std::array<double, 3> nudged = {c.old_anchor[0] + 0.3, c.old_anchor[1] + 0.2, c.old_anchor[2]};
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kHold, Imu::kStatic, Guard::kDegraded, nudged));
  s.steps.push_back(step(Phase::kHoldReset, Imu::kStatic, Guard::kDegraded, nudged, true));
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kAfterReset, Imu::kStatic, Guard::kDegraded, nudged));
  return s;
}

// Nothing forced after the warm-up: the guard's own verdict, the filter's own state.
Scenario natural(bool trace)
{
  Scenario s = baseParams("natural", trace);
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  for (const Imu imu : {Imu::kStatic, Imu::kMoving, Imu::kStatic}) {
    for (int i = 0; i < 8; ++i) {
      Step st;
      st.phase = Phase::kNatural;
      st.imu = imu;
      st.guard = Guard::kNatural;
      s.steps.push_back(st);
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// The child: one component activation, one scenario
// ---------------------------------------------------------------------------

class ScenarioRunner
{
public:
  explicit ScenarioRunner(const Scenario & scenario) : scenario_(scenario) {}

  std::string run()
  {
    setenv("FLIO_ASYNC_MAP", "0", 1);  // map insertion on the scan thread: a deterministic map
    rclcpp::init(0, nullptr);
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    loader_ = std::make_unique<class_loader::ClassLoader>(FASTLIO_TEST_COMPONENT_PATH);
    const auto names = loader_->getAvailableClasses<rclcpp_components::NodeFactory>();
    if (std::find(names.begin(), names.end(), kLaserMappingFactory) == names.end())
      return "LaserMappingNode factory not found";
    factory_ = loader_->createInstance<rclcpp_components::NodeFactory>(kLaserMappingFactory);
    installHooks();

    std::vector<rclcpp::Parameter> overrides;
    for (const auto & entry : scenario_.params)
      overrides.push_back(entry.second);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    options.parameter_overrides(overrides);
    subject_.emplace(factory_->create_node_instance(options));
    rclcpp::NodeOptions observer_options;
    observer_options.use_intra_process_comms(true);
    observer_ = std::make_shared<rclcpp::Node>("anchor_lifecycle_observer", observer_options);
    imu_ =
      observer_->create_publisher<sensor_msgs::msg::Imu>(kImuTopic, rclcpp::QoS(rclcpp::KeepLast(1000)).reliable());
    lidar_ = observer_->create_publisher<sensor_msgs::msg::PointCloud2>(kLidarTopic,
                                                                        rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    odom_ = observer_->create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", rclcpp::QoS(rclcpp::KeepLast(50)).reliable(), [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        odom_by_ms_[msKey(rclcpp::Time(msg->header.stamp).seconds())] = *msg;
      });
    executor_->add_node(subject_->get_node_base_interface());
    executor_->add_node(observer_);
    if (!spinUntil([&] { return imu_->get_subscription_count() == 1 && lidar_->get_subscription_count() == 1; }))
      return "front end subscriptions never matched";

    std::string error = warmUp();
    for (std::size_t i = 0; error.empty() && i < scenario_.steps.size(); ++i)
      error = runScan(static_cast<int>(i));
    teardown();
    return error;
  }

  void report(int fd, const std::string & error) const
  {
    writeString(fd, error);
    const auto count = static_cast<std::uint32_t>(observations_.size());
    writeAll(fd, &count, sizeof(count));
    if (count > 0)
      writeAll(fd, observations_.data(), sizeof(Observation) * observations_.size());
    const auto lines = static_cast<std::uint32_t>(lines_.size());
    writeAll(fd, &lines, sizeof(lines));
    for (const auto & line : lines_)
      writeString(fd, line);
  }

private:
  static void writeAll(int fd, const void * data, std::size_t size)
  {
    const auto * bytes = static_cast<const char *>(data);
    while (size > 0) {
      const ssize_t written = ::write(fd, bytes, size);
      if (written <= 0)
        return;
      bytes += written;
      size -= static_cast<std::size_t>(written);
    }
  }

  static void writeString(int fd, const std::string & text)
  {
    const auto size = static_cast<std::uint32_t>(text.size());
    writeAll(fd, &size, sizeof(size));
    writeAll(fd, text.data(), text.size());
  }

  void installHooks()
  {
    fast_lio_test::overrideGuardDegraded = [this](double end) {
      const Step * s = stepAt(end);
      return s == nullptr ? static_cast<int>(Guard::kHealthy) : static_cast<int>(s->guard);
    };
    fast_lio_test::beforeAnchorStep = [this](double end, fast_lio_test::AnchorFixtureState & fixture) {
      const Step * s = stepAt(end);
      if (s == nullptr || !s->inject)
        return;
      fixture.overwrite = true;
      for (int i = 0; i < 3; ++i) {
        fixture.pos[i] = s->pos[i];
        fixture.vel[i] = kFixtureVelocity[i];
      }
      fixture.rot_wxyz[0] = 1.0;
      fixture.rot_wxyz[1] = fixture.rot_wxyz[2] = fixture.rot_wxyz[3] = 0.0;
      fixture.overwrite_cov = true;
      for (std::size_t i = 0; i < kFixtureCovDiag.size(); ++i)
        fixture.cov_diag[i] = kFixtureCovDiag[i];
    };
    fast_lio_test::forceRunawayTrip = [this](double end) {
      const Step * s = stepAt(end);
      return s != nullptr && s->trip;
    };
    fast_lio_test::afterAnchorScanRecord = [this](const fast_lio::AnchorScanRecord & record) {
      records_.push_back(record);
    };
    fast_lio_test::afterAnchorTraceLine = [this](const std::string & line) {
      lines_.push_back(line);
    };
  }

  static void clearHooks()
  {
    fast_lio_test::overrideGuardDegraded = {};
    fast_lio_test::beforeAnchorStep = {};
    fast_lio_test::forceRunawayTrip = {};
    fast_lio_test::afterAnchorScanRecord = {};
    fast_lio_test::afterAnchorTraceLine = {};
  }

  const Step * stepAt(double end) const
  {
    const auto it = step_by_end_ms_.find(msKey(end));
    if (it == step_by_end_ms_.end() || it->second < 0)
      return nullptr;
    return &scenario_.steps[static_cast<std::size_t>(it->second)];
  }

  Imu imuForScan(int scan) const
  {
    const auto it = step_by_scan_.find(scan);
    if (it == step_by_scan_.end() || it->second < 0)
      return Imu::kStatic;
    return scenario_.steps[static_cast<std::size_t>(it->second)].imu;
  }

  double scanStart(int scan) const { return kFirstScan + kScanPeriod * scan; }
  double scanEnd(int scan) const { return scanStart(scan) + kScanLength; }

  // The scan whose IMU window (previous end, this end] holds time t.
  int scanOfImuTime(double t) const
  {
    const double k = std::ceil((t - kFirstScan - kScanLength) / kScanPeriod - 1e-9);
    return static_cast<int>(std::max(0.0, k));
  }

  void sendImuThrough(double end)
  {
    const auto through = static_cast<std::int64_t>(std::llround(end * 1e9));
    for (; next_imu_ns_ <= through; next_imu_ns_ += kImuPeriodNs) {
      const double t = static_cast<double>(next_imu_ns_) * 1e-9;
      const int scan = scanOfImuTime(t);
      const Imu pattern = imuForScan(scan);
      if (pattern == Imu::kGap && t > scanEnd(scan - 1) + kGapMargin && t < scanEnd(scan) - kGapMargin)
        continue;
      sensor_msgs::msg::Imu message;
      message.header.stamp = rclcpp::Time(next_imu_ns_);
      message.linear_acceleration.z = kGravity;
      message.angular_velocity.z = pattern == Imu::kMoving ? kMovingYawRate : 0.0;
      imu_->publish(message);
      executor_->spin_some();
    }
  }

  void sendLidar(double start)
  {
    pcl::PointCloud<ouster_ros::Point> cloud;
    cloud.resize(kScanPoints);
    for (std::size_t i = 0; i < kScanPoints; ++i) {
      auto & point = cloud[i];
      point = ouster_ros::Point{};
      point.x = 10.0f + static_cast<float>(i % 317) * 0.01f;
      point.y = -1.6f + static_cast<float>(i / 317) * 0.01f;
      point.z = 1.0f;
      point.intensity = 1.0f;
      point.t = static_cast<std::uint32_t>(i * 50000000ULL / (kScanPoints - 1));
    }
    auto message = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(cloud, *message);
    message->header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(start * 1e9)));
    message->header.frame_id = "lidar";
    lidar_->publish(std::move(message));
  }

  template<class Predicate>
  bool spinUntil(Predicate ready, double seconds = kScanWaitSeconds)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!ready() && std::chrono::steady_clock::now() < end)
      executor_->spin_once(std::chrono::milliseconds(2));
    return ready();
  }

  const fast_lio::AnchorScanRecord * recordAt(double end) const
  {
    const std::int64_t key = msKey(end);
    for (auto it = records_.rbegin(); it != records_.rend(); ++it)
      if (it->end_ns >= 0 && msKey(static_cast<double>(it->end_ns) * 1e-9) == key)
        return &*it;
    return nullptr;
  }

  // Sends scan number `scan_` (step `index`, -1 for warm-up) and waits until the node has finished it.
  std::string sendAndWait(int index, const fast_lio::AnchorScanRecord ** record)
  {
    const int scan = scan_++;
    step_by_scan_[scan] = index;
    step_by_end_ms_[msKey(scanEnd(scan))] = index;
    // Sync needs one IMU sample past this scan's end, and that sample belongs to the next scan's window: give it
    // the pattern of the step that runs next (a warm-up scan cannot know, and stays static).
    const bool has_next = index >= 0 && static_cast<std::size_t>(index) + 1 < scenario_.steps.size();
    step_by_scan_[scan + 1] = has_next ? index + 1 : -1;
    sendImuThrough(scanEnd(scan) + 0.006);
    sendLidar(scanStart(scan));
    const double end = scanEnd(scan);
    const bool done = spinUntil([&] {
      const auto * r = recordAt(end);
      return r != nullptr && (!r->published || odom_by_ms_.count(msKey(end)) > 0);
    });
    if (!done) {
      std::ostringstream out;
      out << "scan " << scan << " (step " << index << ", end " << end << ") produced no complete record";
      return out.str();
    }
    *record = recordAt(end);
    return "";
  }

  void observe(int index, const fast_lio::AnchorScanRecord & record)
  {
    Observation o;
    std::memset(static_cast<void *>(&o), 0, sizeof(o));
    o.step = index;
    o.phase = index < 0 ? static_cast<int>(Phase::kWarmup)
                        : static_cast<int>(scenario_.steps[static_cast<std::size_t>(index)].phase);
    o.record = record;
    const auto odom = odom_by_ms_.find(msKey(static_cast<double>(record.end_ns) * 1e-9));
    if (record.published && odom != odom_by_ms_.end()) {
      const auto & m = odom->second;
      o.odom_seen = 1;
      o.odom_p[0] = m.pose.pose.position.x;
      o.odom_p[1] = m.pose.pose.position.y;
      o.odom_p[2] = m.pose.pose.position.z;
      o.odom_q[0] = m.pose.pose.orientation.w;
      o.odom_q[1] = m.pose.pose.orientation.x;
      o.odom_q[2] = m.pose.pose.orientation.y;
      o.odom_q[3] = m.pose.pose.orientation.z;
      for (int i = 0; i < 36; ++i) {
        o.odom_cov[i] = m.pose.covariance[static_cast<std::size_t>(i)];
        o.odom_twist_cov[i] = m.twist.covariance[static_cast<std::size_t>(i)];
      }
    }
    observations_.push_back(o);
  }

  std::string warmUp()
  {
    int published = 0;
    for (int i = 0; i < kWarmupMaxScans && published < kWarmupPublished; ++i) {
      const fast_lio::AnchorScanRecord * record = nullptr;
      const std::string error = sendAndWait(-1, &record);
      if (!error.empty())
        return "warm-up: " + error;
      observe(-1, *record);
      published += record->published ? 1 : 0;
    }
    return published >= kWarmupPublished ? "" : "warm-up never published odometry";
  }

  std::string runScan(int index)
  {
    const fast_lio::AnchorScanRecord * record = nullptr;
    const std::string error = sendAndWait(index, &record);
    if (!error.empty())
      return error;
    observe(index, *record);
    return "";
  }

  void teardown()
  {
    if (subject_)
      executor_->remove_node(subject_->get_node_base_interface());
    if (observer_)
      executor_->remove_node(observer_);
    subject_.reset();  // the destructor logs the counter summary through the trace hook
    clearHooks();
    odom_.reset();
    imu_.reset();
    lidar_.reset();
    observer_.reset();
    factory_.reset();
    loader_.reset();
    executor_.reset();
    rclcpp::shutdown();
  }

  const Scenario & scenario_;
  std::unique_ptr<class_loader::ClassLoader> loader_;
  std::shared_ptr<rclcpp_components::NodeFactory> factory_;
  std::optional<rclcpp_components::NodeInstanceWrapper> subject_;
  rclcpp::Node::SharedPtr observer_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_;
  std::map<std::int64_t, nav_msgs::msg::Odometry> odom_by_ms_;
  std::map<std::int64_t, int> step_by_end_ms_;
  std::map<int, int> step_by_scan_;
  std::vector<fast_lio::AnchorScanRecord> records_;
  std::vector<Observation> observations_;
  std::vector<std::string> lines_;
  std::int64_t next_imu_ns_ = kImuFirstNs;
  int scan_ = 0;
};

// ---------------------------------------------------------------------------
// The parent: fork, collect, assert
// ---------------------------------------------------------------------------

bool readAll(int fd, std::string * out)
{
  char buffer[65536];
  for (;;) {
    const ssize_t got = ::read(fd, buffer, sizeof(buffer));
    if (got == 0)
      return true;
    if (got < 0)
      return false;
    out->append(buffer, static_cast<std::size_t>(got));
  }
}

class Reader
{
public:
  explicit Reader(const std::string & bytes) : bytes_(bytes) {}
  bool take(void * out, std::size_t size)
  {
    if (offset_ + size > bytes_.size())
      return false;
    std::memcpy(out, bytes_.data() + offset_, size);
    offset_ += size;
    return true;
  }
  bool takeString(std::string * out)
  {
    std::uint32_t size = 0;
    if (!take(&size, sizeof(size)) || offset_ + size > bytes_.size())
      return false;
    out->assign(bytes_.data() + offset_, size);
    offset_ += size;
    return true;
  }

private:
  const std::string & bytes_;
  std::size_t offset_ = 0;
};

ChildResult runScenario(const Scenario & scenario)
{
  ChildResult result;
  int fds[2];
  if (::pipe(fds) != 0) {
    result.error = "pipe failed";
    return result;
  }
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    result.error = "fork failed";
    return result;
  }
  if (pid == 0) {
    ::close(fds[0]);
    ::alarm(kChildAlarmSeconds);
    std::string error;
    ScenarioRunner runner(scenario);
    try {
      error = runner.run();
    } catch (const std::exception & e) {
      error = std::string("exception: ") + e.what();
    } catch (...) {
      error = "unknown exception";
    }
    runner.report(fds[1], error);
    ::close(fds[1]);
    ::_exit(error.empty() ? 0 : 3);
  }
  ::close(fds[1]);
  std::string bytes;
  const bool read_ok = readAll(fds[0], &bytes);
  ::close(fds[0]);
  ::waitpid(pid, &result.status, 0);
  if (!read_ok) {
    result.error = "pipe read failed";
    return result;
  }
  Reader reader(bytes);
  std::string child_error;
  std::uint32_t count = 0;
  if (!reader.takeString(&child_error) || !reader.take(&count, sizeof(count))) {
    result.error = "child report truncated";
    return result;
  }
  result.error = child_error;
  result.observations.resize(count);
  if (count > 0 && !reader.take(result.observations.data(), sizeof(Observation) * count)) {
    result.error = "child observations truncated";
    return result;
  }
  std::uint32_t lines = 0;
  if (!reader.take(&lines, sizeof(lines))) {
    result.error = "child trace lines truncated";
    return result;
  }
  result.lines.resize(lines);
  for (auto & line : result.lines) {
    if (!reader.takeString(&line)) {
      result.error = "child trace line truncated";
      return result;
    }
  }
  return result;
}

std::string valueOf(const std::string & line, const std::string & key)
{
  std::istringstream in(line);
  std::string token;
  while (in >> token)
    if (token.compare(0, key.size() + 1, key + "=") == 0)
      return token.substr(key.size() + 1);
  return "<missing>";
}

// The anchor the parked-at-the-old-anchor scans ended with: the ID a stale pin would reuse.
std::uint64_t oldAnchorId(const ChildResult & r)
{
  const auto anchored = r.phase(Phase::kAnchorStatic);
  return anchored.empty() ? 0u : anchored.back()->record.anchor_id;
}

bool anchorWasValidAtTheOldPose(const ChildResult & r, const FieldCase & c)
{
  const auto anchored = r.phase(Phase::kAnchorStatic);
  return !anchored.empty() && anchored.back()->record.anchor_valid && anchored.back()->odom_seen &&
         samePosition(anchored.back()->odom_p, c.old_anchor);
}

// Field by field, doubles by bits: what the node published and wrote for one scan.
bool sameObservation(const Observation & a, const Observation & b, std::string * why)
{
  const auto & x = a.record;
  const auto & y = b.record;
  std::ostringstream diff;
  const auto real = [&](const char * name, double u, double v) {
    if (!sameBits(u, v))
      diff << name << ' ';
  };
  const auto integer = [&](const char * name, long long u, long long v) {
    if (u != v)
      diff << name << ' ';
  };
  integer("step", a.step, b.step);
  integer("scan_seq", static_cast<long long>(x.scan_seq), static_cast<long long>(y.scan_seq));
  integer("outcome", static_cast<int>(x.outcome), static_cast<int>(y.outcome));
  integer("beg_ns", x.beg_ns, y.beg_ns);
  integer("end_ns", x.end_ns, y.end_ns);
  integer("gap_ns", x.gap_ns, y.gap_ns);
  integer("anchor_id", static_cast<long long>(x.anchor_id), static_cast<long long>(y.anchor_id));
  integer("anchor_valid", x.anchor_valid, y.anchor_valid);
  integer("anchor_created_ns", x.anchor_created_ns, y.anchor_created_ns);
  integer("anchor_step", x.anchor_step, y.anchor_step);
  integer("body", static_cast<int>(x.body), static_cast<int>(y.body));
  integer("imu_n", x.imu_n, y.imu_n);
  real("gyr_max", x.gyr_max, y.gyr_max);
  real("acc_span", x.acc_span, y.acc_span);
  integer("guard_degraded", x.guard_degraded, y.guard_degraded);
  integer("degraded", x.degraded, y.degraded);
  integer("gate_pre", static_cast<int>(x.gate_pre), static_cast<int>(y.gate_pre));
  integer("gate_post", static_cast<int>(x.gate_post), static_cast<int>(y.gate_post));
  integer("action", static_cast<int>(x.action), static_cast<int>(y.action));
  integer("anchor_used", static_cast<long long>(x.anchor_used), static_cast<long long>(y.anchor_used));
  integer("parked", x.parked, y.parked);
  integer("parked_applied", x.parked_applied, y.parked_applied);
  integer("zupt_active", x.zupt_active, y.zupt_active);
  integer("events", x.events, y.events);
  integer("have_pre", x.have_pre, y.have_pre);
  integer("have_post", x.have_post, y.have_post);
  integer("have_cov", x.have_cov, y.have_cov);
  for (int i = 0; i < 3; ++i) {
    real("pre_p", x.pre_p[i], y.pre_p[i]);
    real("pre_v", x.pre_v[i], y.pre_v[i]);
    real("post_p", x.post_p[i], y.post_p[i]);
    real("post_v", x.post_v[i], y.post_v[i]);
    real("odom_p", a.odom_p[i], b.odom_p[i]);
  }
  for (int i = 0; i < 4; ++i) {
    real("pre_q", x.pre_q[i], y.pre_q[i]);
    real("post_q", x.post_q[i], y.post_q[i]);
    real("odom_q", a.odom_q[i], b.odom_q[i]);
  }
  for (int i = 0; i < 36; ++i) {
    real("p_pose_pre", x.p_pose_pre[i], y.p_pose_pre[i]);
    real("p_pose_post", x.p_pose_post[i], y.p_pose_post[i]);
    real("odom_cov", a.odom_cov[i], b.odom_cov[i]);
    real("odom_twist_cov", a.odom_twist_cov[i], b.odom_twist_cov[i]);
  }
  integer("p_finite", x.p_finite, y.p_finite);
  real("p_sym", x.p_sym, y.p_sym);
  real("p_min_eig", x.p_min_eig, y.p_min_eig);
  integer("published", x.published, y.published);
  integer("flags", x.flags, y.flags);
  integer("odom_seen", a.odom_seen, b.odom_seen);
  *why = diff.str();
  return why->empty();
}

void expectSameRun(const ChildResult & on, const ChildResult & off, const std::string & name)
{
  ASSERT_EQ(on.observations.size(), off.observations.size()) << name;
  for (std::size_t i = 0; i < on.observations.size(); ++i) {
    std::string why;
    EXPECT_TRUE(sameObservation(on.observations[i], off.observations[i], &why))
      << name << " scan " << i << " differs with the trace on: " << why;
  }
}

std::vector<std::string> linesTagged(const ChildResult & r, const std::string & tag)
{
  std::vector<std::string> out;
  for (const auto & line : r.lines)
    if (line.rfind(tag + " ", 0) == 0)
      out.push_back(line);
  return out;
}

// K7 on the node: one scan record per scan taken, sequences without gaps or repeats, counters that reconcile.
void expectTraceAccountsForEveryScan(const ChildResult & r, bool trace_on, const std::string & name)
{
  const auto scans = linesTagged(r, "[ZUPT-ANCHOR-SCAN]");
  const auto transitions = linesTagged(r, "[ZUPT-ANCHOR]");
  ASSERT_FALSE(transitions.empty()) << name << ": no counter summary";
  const std::string counters = transitions.back();
  EXPECT_EQ(valueOf(counters, "event"), "counters") << name;
  EXPECT_EQ(valueOf(counters, "scans"), std::to_string(r.observations.size())) << name;
  EXPECT_EQ(valueOf(counters, "events"), std::to_string(transitions.size() - 1)) << name;
  for (std::size_t i = 0; i < transitions.size(); ++i) {
    EXPECT_EQ(valueOf(transitions[i], "v"), "1") << name;
    EXPECT_EQ(valueOf(transitions[i], "event_seq"), std::to_string(i + 1)) << name << ": " << transitions[i];
  }
  if (!trace_on) {
    EXPECT_TRUE(scans.empty()) << name << ": scan records without zupt_anchor_trace_en";
    return;
  }
  ASSERT_EQ(scans.size(), r.observations.size()) << name;
  for (std::size_t i = 0; i < scans.size(); ++i) {
    const auto & o = r.observations[i];
    EXPECT_EQ(valueOf(scans[i], "scan_seq"), std::to_string(i + 1)) << name;
    EXPECT_EQ(valueOf(scans[i], "scan_seq"), std::to_string(o.record.scan_seq)) << name;
    EXPECT_EQ(scans[i], fast_lio::formatAnchorScanRecord(o.record)) << name;
    if (o.odom_seen) {
      // The printed state round-trips to exactly what /Odometry carried.
      const std::string post_p = valueOf(scans[i], "post_p");
      std::istringstream in(post_p);
      std::string part;
      for (int axis = 0; axis < 3 && std::getline(in, part, ','); ++axis) {
        EXPECT_TRUE(sameBits(std::strtod(part.c_str(), nullptr), o.odom_p[axis])) << name << ": " << post_p;
      }
    }
  }
}

}  // namespace

// The stale anchor, characterised: parked again after a moving degraded stretch, the real write-back pins position
// and attitude to the anchor taken before the motion and publishes StaticHold, in all three frozen cases.
TEST(AnchorLifecycleNode, F1F2F3DegradedParkedAgainPinsToTheOldAnchor)
{
  int positive = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainDegraded(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    ASSERT_TRUE(anchorWasValidAtTheOldPose(r, c)) << c.name;
    const auto moved = r.phase(Phase::kMove);
    ASSERT_EQ(moved.size(), 1u) << c.name;
    EXPECT_NE(moved.front()->record.events & fast_lio::kAnchorEventMotion, 0u) << c.name << ": motion not traced";
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 3u) << c.name;
    bool all = true;
    for (const auto * o : parked) {
      const bool pinned = o->record.action == fast_lio::AnchorTraceAction::kPin && o->record.anchor_used == old_id &&
                          samePosition(o->record.pre_p, c.pre_event) && samePosition(o->record.post_p, c.old_anchor) &&
                          o->odom_seen && samePosition(o->odom_p, c.old_anchor) && sameBits(o->record.post_v[0], 0.0) &&
                          odomFlags(*o) == (kFlagDegraded | kFlagStaticHold);
      EXPECT_TRUE(pinned) << c.name << " step " << o->step << ": published " << formatPosition(o->odom_p) << " flags "
                          << odomFlags(*o) << " anchor_used " << o->record.anchor_used << " (old " << old_id << ")";
      all = all && pinned;
    }
    EXPECT_GT(distance(c.pre_event, c.old_anchor), 30.0) << c.name;
    positive += all ? 1 : 0;
  }
  EXPECT_EQ(positive, 3) << "A positive in " << positive << "/3 frozen cases";
}

// The late reset: rebuild_local_map runs after the anchor step pinned the scan, and the scan still publishes the
// old anchor's pose with StaticHold.
TEST(AnchorLifecycleNode, F1F2F3LateResetStillPublishesTheOldAnchorAsStaticHold)
{
  int positive = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainWithLateReset(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 3u) << c.name;
    const auto & reset = *parked.front();
    const std::uint32_t reset_events = fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventMapReset;
    EXPECT_EQ(reset.record.events & reset_events, reset_events) << c.name;
    bool all = true;
    for (const auto * o : parked) {
      const bool stale = o->record.action == fast_lio::AnchorTraceAction::kPin && o->record.anchor_used == old_id &&
                         o->odom_seen && samePosition(o->odom_p, c.old_anchor) &&
                         odomFlags(*o) == (kFlagDegraded | kFlagStaticHold);
      EXPECT_TRUE(stale) << c.name << " step " << o->step << ": flags " << odomFlags(*o);
      all = all && stale;
    }
    positive += all ? 1 : 0;
  }
  EXPECT_EQ(positive, 3);
}

// Parked again on healthy scans: the parked prior pulls the estimate back toward the anchor taken before the
// motion and reports StaticHold.
TEST(AnchorLifecycleNode, F1F2F3HealthyParkedPriorPullsTowardTheOldAnchor)
{
  int positive = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainHealthy(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 5u) << c.name;
    bool pulled = false;
    for (const auto * o : parked) {
      if (!o->record.parked_applied)
        continue;
      const bool toward_old = o->record.anchor_used == old_id && o->odom_seen &&
                              distance(o->odom_p, c.old_anchor) < 0.5 * distance(c.pre_event, c.old_anchor) &&
                              (odomFlags(*o) & kFlagStaticHold) != 0u;
      EXPECT_TRUE(toward_old) << c.name << " step " << o->step << ": published " << formatPosition(o->odom_p);
      pulled = pulled || toward_old;
    }
    EXPECT_TRUE(pulled) << c.name << ": the parked prior never applied";
    positive += pulled ? 1 : 0;
  }
  EXPECT_EQ(positive, 3);
}

// An IMU coverage gap returns from the scan path before the anchor step; the anchor taken before it is pinned to
// afterwards although the pose came back elsewhere.
TEST(AnchorLifecycleNode, CoverageGapBeforeTheAnchorStepLeavesTheOldAnchorInUse)
{
  const FieldCase & c = kFieldCases[0];
  const ChildResult r = runScenario(coverageGapThenParked(c));
  ASSERT_TRUE(r.ok()) << r.describe();
  const std::uint64_t old_id = oldAnchorId(r);
  ASSERT_NE(old_id, 0u);
  const auto gap = r.phase(Phase::kGap);
  ASSERT_EQ(gap.size(), 1u);
  EXPECT_EQ(gap.front()->record.outcome, fast_lio::AnchorScanOutcome::kImuCoverageGap);
  EXPECT_NE(gap.front()->record.events & fast_lio::kAnchorEventImuCoverageGap, 0u);
  EXPECT_FALSE(gap.front()->record.anchor_step);
  for (const auto * o : r.phase(Phase::kStatic)) {
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin) << o->step;
    EXPECT_EQ(o->record.anchor_used, old_id) << o->step;
    EXPECT_TRUE(o->odom_seen && samePosition(o->odom_p, c.old_anchor)) << formatPosition(o->odom_p);
    EXPECT_EQ(odomFlags(*o), kFlagDegraded | kFlagStaticHold) << o->step;
  }
}

// A legitimate pin, then a runaway map reset after the anchor step of a pinned scan: the reset scan still
// reports StaticHold, and the next scan pins to the same anchor again.
TEST(AnchorLifecycleNode, LateResetAfterAPinKeepsTheStaticHoldFlag)
{
  const ChildResult r = runScenario(pinThenLateReset());
  ASSERT_TRUE(r.ok()) << r.describe();
  const std::uint64_t old_id = oldAnchorId(r);
  ASSERT_NE(old_id, 0u);
  for (const auto * o : r.phase(Phase::kHold)) {
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin);
    EXPECT_EQ(odomFlags(*o), kFlagDegraded | kFlagStaticHold);
  }
  const auto reset = r.phase(Phase::kHoldReset);
  ASSERT_EQ(reset.size(), 1u);
  EXPECT_NE(reset.front()->record.events & fast_lio::kAnchorEventMapReset, 0u);
  EXPECT_EQ(odomFlags(*reset.front()), kFlagDegraded | kFlagStaticHold);
  for (const auto * o : r.phase(Phase::kAfterReset)) {
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin);
    EXPECT_EQ(o->record.anchor_used, old_id);
  }
}

// K8: the observation switch changes no estimate, timestamp or flag. Each pair runs the same fixture with the
// per-scan trace on and off; every published and written value must match bit for bit.
TEST(AnchorLifecycleNode, TraceOnAndOffPublishBitIdenticalStatesAndFlags)
{
  const std::vector<std::pair<Scenario, Scenario>> pairs = {
    {parkedAgainWithLateReset(kFieldCases[0], true), parkedAgainWithLateReset(kFieldCases[0], false)},
    {pinThenLateReset(true), pinThenLateReset(false)},
    {natural(true), natural(false)},
  };
  for (const auto & pair : pairs) {
    const ChildResult on = runScenario(pair.first);
    const ChildResult off = runScenario(pair.second);
    ASSERT_TRUE(on.ok()) << pair.first.name << " (trace on): " << on.describe();
    ASSERT_TRUE(off.ok()) << pair.second.name << " (trace off): " << off.describe();
    expectSameRun(on, off, pair.first.name);
    expectTraceAccountsForEveryScan(on, true, pair.first.name + " (trace on)");
    expectTraceAccountsForEveryScan(off, false, pair.second.name + " (trace off)");
    EXPECT_EQ(linesTagged(on, "[ZUPT-ANCHOR]"), linesTagged(off, "[ZUPT-ANCHOR]"))
      << pair.first.name << ": transition records differ with the per-scan trace on";
  }
}

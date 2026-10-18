// The stationary anchor and the recovery trust on the production call chain.
//
// The real LaserMappingNode, fed through its real IMU and point-cloud
// subscriptions, takes every scan through the production scan path, the map
// reset in rebuild_local_map and the /Odometry publisher. FASTLIO_TEST_HOOKS
// fix only what a synthetic stream cannot produce on demand: the divergence
// guard's verdict, the state registration hands to the anchor step, a runaway
// trip for the watchdog's late map reset or hold, and a normal initial map
// build whose points the map backend all skips. The static verdict is the
// real one, computed from the IMU windows the fixture sends.
//
// The anchor and pre-event positions are the three frozen cases F1-F3.
// Attitude, velocity and covariance are fixed fixture values (identity, a
// nonzero velocity below the platform's maximum, a positive-definite
// diagonal), not a recorded filter state.
//
// One process can activate the component once (its globals outlive a node), so
// every scenario runs in a forked child that reports what the node did; the
// parent makes the assertions.
//
// The suites are registered as separate CTest entries, each with its own
// timeout (CMakeLists.txt, FASTLIO_ANCHOR_NODE_SUITES); AnchorNodeEntries runs
// in every entry and fails when a suite has none.

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

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
#include <fast_lio_interfaces/msg/frontend_observation.hpp>
#include <fast_lio_interfaces/msg/reference_health.hpp>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <rcl/error_handling.h>
#include <rcl/timer.h>
#include <rcl/wait.h>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#include "anchor_trace.hpp"
#include "frontend_observation.hpp"
#include "imu_process_outcome.hpp"
#include "laser_mapping_test_hooks.hpp"
#include "odom_covariance_contract.hpp"
#include "preprocess.h"

namespace
{

constexpr const char * kLaserMappingFactory = "rclcpp_components::NodeFactoryTemplate<LaserMappingNode>";
constexpr const char * kImuTopic = "/test_anchor_lifecycle/imu";
constexpr const char * kLidarTopic = "/test_anchor_lifecycle/points";
constexpr const char * kWheelTopic = "/test_anchor_lifecycle/twist";
constexpr const char * kTypedTopic = "/lio_slam/frontend_observation";
constexpr const char * kRelocalizationTopic = "/relocalization/result";
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
constexpr double kReceiptWaitSeconds = 5.0;  // bound on waiting for the node to show it took what a slot sent
constexpr std::int64_t kReceiptWaitNs = 5000000000;
constexpr std::int64_t kScanPathPeriodNs = 10000000;  // the front end runs its scan path off a 100 Hz timer
constexpr unsigned kChildAlarmSeconds = 300;

constexpr std::array<double, 3> kFixtureVelocity = {0.4, -0.3, 0.05};  // m/s, below divergence_guard_max_speed
constexpr std::array<double, 23> kFixtureCovDiag = {1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6,
                                                    1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2, 1e-6,
                                                    1e-6, 1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4};

constexpr unsigned kFlagDegraded = 1u << 0;
constexpr unsigned kFlagStaticHold = 1u << 1;
constexpr unsigned kFlagLost = 1u << 2;

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

// What a scan slot puts on the point-cloud topic.
enum class Cloud : std::uint8_t
{
  kDense,   // the fixture's 2000-point scan
  kSparse,  // three points: the scan path leaves before the anchor step
  kEmpty,   // no point: dropped when taken, for having no usable timing
  kNone,    // nothing is sent: the slot is a hole in the scan stream
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
  kAnchorMove,        // healthy, moving: the scan whose pose becomes the anchor
  kAnchorStatic,      // healthy, parked at the anchor
  kMove,              // degraded and moving: the event that ends the stationary stretch
  kMoveDegraded,      // degraded and moving on
  kStatic,            // parked again (degraded or healthy per scenario), at the pre-event pose
  kGap,               // IMU coverage gap: the scan path returns before the anchor step
  kHold,              // degraded and parked at the anchor: a legitimate pin
  kHoldReset,         // the same, with a runaway map reset after the anchor step
  kAfterReset,        // degraded and parked after the reset
  kNatural,           // no hooks beyond the record
  kParkedReset,       // healthy and parked at the anchor, with a runaway map reset after the anchor step
  kParkedAfterReset,  // healthy and parked after that reset
  kParkedAfterLatch,  // healthy and parked after the unsupported recovery ran out
  kBefore,            // input-driven scenarios: the stretch before the event
  kEvent,             // the scan slot(s) the event is on
  kAfter,             // the stretch after it
  kLater,             // a second stretch after that
};

struct Step
{
  Phase phase = Phase::kNatural;
  Imu imu = Imu::kStatic;
  Guard guard = Guard::kNatural;
  bool inject = false;
  std::array<double, 3> pos = {0.0, 0.0, 0.0};
  bool trip = false;
  Cloud cloud = Cloud::kDense;
  // IMU samples strictly inside (scan end + hole_from, scan end + hole_to) are not sent; equal bounds, no hole.
  // The samples up to 6 ms past the previous scan's end went out with that scan.
  double hole_from = 0.0;
  double hole_to = 0.0;
  bool bad_imu = false;      // the last IMU sample before the scan end carries a non-finite rate
  bool late_imu = false;     // an IMU sample stamped 50 ms before the newest one follows the scan's samples
  bool clock_back = false;   // an IMU sample stamped 1.5 s before the newest one precedes the scan: an epoch fault
  bool expect_scan = true;   // false: the node is not expected to take this slot's scan at all
  bool wheel = false;        // a wheel sample is published before the scan
  double wheel_speed = 0.0;  // m/s; NaN is the bridge saying it cannot measure
  double wheel_age = 0.0;    // s between the wheel sample's stamp and the scan end
  int relocalization = -1;   // index into Scenario::relocalizations, published before the scan
  // Input-driven slots: the cloud is queued before any IMU sample of the slot; the slot sends no IMU sample (a
  // later slot sends them, in stamp order); one more IMU sample is stamped exactly at the slot's scan end.
  bool cloud_first = false;
  bool hold_imu = false;
  bool imu_at_end = false;
  // Input-driven slots: > 0 sends, in place of a new scan, the cloud of the scan this many slots back once more, stamp
  // and points alike. The slot's IMU samples still go out.
  int repeat = 0;
  bool no_build_point = false;  // a normal initial map build on this scan is handed only points the backend skips
};

struct Relocalization
{
  std::array<double, 3> translation = {0.0, 0.0, 0.0};
};

struct Scenario
{
  std::string name;
  std::vector<Step> steps;
  std::map<std::string, rclcpp::Parameter> params;
  bool trace = false;
  bool flexible = false;     // slots may hold no scan, and a scan may be taken slots after it was sent
  bool lidar_first = false;  // the cloud is queued before the IMU sample that passes its end
  bool typed = false;        // publish and collect the typed observation
  bool reference = false;    // real R1 publisher, subscriber and CDR round trip
  bool prior_map = false;    // a prior map waiting for /relocalization/result
  std::vector<Relocalization> relocalizations;
  double scan_length = kScanLength;  // s, last point time within each scan; kScanPeriod: scans that abut
  bool warm_up = true;               // false: the steps are the process's first input, its start included
  bool capture_raw_state = false;    // startup-only: snapshot at scope exit or an unlocked empty-IMU drop
};

struct Observation
{
  int step = -1;  // index into Scenario::steps; -1 = warm-up
  int phase = 0;
  fast_lio::AnchorScanRecord record;
  int odom_seen = 0;
  std::int64_t odom_stamp_ns = 0;
  double odom_p[3] = {0.0, 0.0, 0.0};
  double odom_q[4] = {0.0, 0.0, 0.0, 0.0};  // w, x, y, z
  double odom_cov[36] = {};
  double odom_twist_cov[36] = {};
  int typed_seen = 0;  // the typed observation of this scan's pose
  unsigned typed_state = 0;
  unsigned typed_validity = 0;
  unsigned typed_health = 0;
  unsigned typed_axes = 0;
  unsigned typed_substate = 0;
  std::array<std::uint8_t, 16> typed_instance{};
  std::array<std::uint8_t, 32> typed_id{};
  std::array<double, 7> typed_pose{};
  int imu_batch = 0;  // the scan settled an IMU batch through the filter's consumption outcome
  int imu_disposition = -1;
  int imu_reason = -1;
  std::uint64_t imu_samples = 0;
  double filter_end = 0.0;  // the end of the input the filter had processed once that batch was settled
  bool raw_state_seen = false;
  bool map_frozen = false;
  bool raw_degraded = false;
};
static_assert(std::is_trivially_copyable<Observation>::value, "observations cross the pipe as bytes");

// A typed observation without a pose: what the front end reports between scans.
struct IdleSample
{
  std::int64_t input_ns = -1;  // newest IMU stamp the fixture had sent when it arrived
  std::int64_t stamp_ns = -1;  // the stamp the front end gave its newest input; -1 when it reports none
  unsigned state = 0;
  unsigned health = 0;
  unsigned substate = 0;
};
static_assert(std::is_trivially_copyable<IdleSample>::value, "idle samples cross the pipe as bytes");

struct ChildResult
{
  int status = -1;
  std::string error;
  std::vector<Observation> observations;
  std::vector<std::string> lines;
  std::vector<std::int64_t> line_input_ns;  // per trace line: the newest IMU stamp sent when it was logged
  std::vector<IdleSample> idle;
  std::vector<std::int64_t> odom_ns;  // the stamp of every /Odometry message received, in order
  std::vector<fast_lio_interfaces::msg::ReferenceHealth> references;

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

// Where registration puts a platform that is still parked at the anchor: off it, but within the same stretch.
std::array<double, 3> nudgedAnchor(const FieldCase & c)
{
  return {c.old_anchor[0] + 0.3, c.old_anchor[1] + 0.2, c.old_anchor[2]};
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
  const std::array<double, 3> nudged = nudgedAnchor(c);
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kHold, Imu::kStatic, Guard::kDegraded, nudged));
  s.steps.push_back(step(Phase::kHoldReset, Imu::kStatic, Guard::kDegraded, nudged, true));
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kAfterReset, Imu::kStatic, Guard::kDegraded, nudged));
  return s;
}

// Parked at the anchor on healthy scans when a runaway map reset runs after the anchor step, then parked on.
// `total` is reanchor_total_timeout_sec: 0 leaves the platform policy off.
Scenario resetWhileParkedHealthy(double total)
{
  const FieldCase & c = kFieldCases[0];
  Scenario s = baseParams(total > 0.0 ? "parked_reset_policy_on" : "parked_reset_policy_off", false);
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  setParam(s, rclcpp::Parameter("reanchor_total_timeout_sec", total));
  setParam(s, rclcpp::Parameter("reanchor_recovery_stable_sec", 1.0));
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 4; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  s.steps.push_back(step(Phase::kParkedReset, Imu::kStatic, Guard::kHealthy, c.old_anchor, true));
  for (int i = 0; i < 5; ++i)
    s.steps.push_back(step(Phase::kParkedAfterReset, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  return s;
}

constexpr double kShortRecoveryBudget = 1.0;  // s, reanchor_total_timeout_sec of the run-out scenario
constexpr std::int64_t kShortRecoveryBudgetNs = 1000000000;

// The platform policy on with a short budget: a degraded moving stretch outlasts it, and the platform then parks
// on healthy scans. `scan_first` queues every cloud before the IMU sample that passes its end, so the first sample
// past the deadline arrives while the scan whose end is the deadline waits; without it that sample arrives before
// the scan.
Scenario recoveryRunsOut(bool trace = false, bool scan_first = true)
{
  const FieldCase & c = kFieldCases[1];
  Scenario s = baseParams(scan_first ? "recovery_runs_out" : "recovery_runs_out_imu_first", trace);
  s.lidar_first = scan_first;
  setParam(s, rclcpp::Parameter("reanchor_total_timeout_sec", kShortRecoveryBudget));
  setParam(s, rclcpp::Parameter("reanchor_recovery_stable_sec", 0.5));
  appendAnchorThenMove(s, c);
  for (int i = 0; i < 9; ++i)
    s.steps.push_back(step(Phase::kMoveDegraded, Imu::kMoving, Guard::kDegraded, c.pre_event));
  for (int i = 0; i < 6; ++i)
    s.steps.push_back(step(Phase::kParkedAfterLatch, Imu::kStatic, Guard::kHealthy, c.pre_event));
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
        odom_ns_.push_back(rclcpp::Time(msg->header.stamp).nanoseconds());
      });
    wheel_ = observer_->create_publisher<geometry_msgs::msg::TwistStamped>(
      kWheelTopic, rclcpp::QoS(rclcpp::KeepLast(100)).reliable());
    if (scenario_.typed) {
      typed_ = observer_->create_subscription<fast_lio_interfaces::msg::FrontendObservation>(
        kTypedTopic,
        rclcpp::QoS(rclcpp::KeepLast(200)).best_effort(),
        [this](fast_lio_interfaces::msg::FrontendObservation::ConstSharedPtr msg) { onTyped(*msg); });
    }
    if (scenario_.reference) {
      reference_ = observer_->create_subscription<fast_lio_interfaces::msg::ReferenceHealth>(
        "/lio_slam/reference_health",
        rclcpp::QoS(rclcpp::KeepLast(200)).best_effort().durability_volatile(),
        [this](fast_lio_interfaces::msg::ReferenceHealth::ConstSharedPtr message) {
          rclcpp::Serialization<fast_lio_interfaces::msg::ReferenceHealth> serialization;
          rclcpp::SerializedMessage wire;
          serialization.serialize_message(message.get(), &wire);
          const auto & data = wire.get_rcl_serialized_message();
          reference_cdr_.emplace_back(reinterpret_cast<const char *>(data.buffer), data.buffer_length);
          if (message->has_scan_sequence && message->kind != 4)
            reference_scans_.insert(message->scan_sequence);
        });
    }
    if (scenario_.prior_map) {
      // The front end takes the alignment over DDS only (latched, intra-process off).
      rclcpp::PublisherOptions options;
      options.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
      relocalization_ = observer_->create_publisher<geometry_msgs::msg::TransformStamped>(
        kRelocalizationTopic, rclcpp::QoS(1).transient_local(), options);
    }
    executor_->add_node(subject_->get_node_base_interface());
    executor_->add_node(observer_);
    if (!spinUntil([&] { return imu_->get_subscription_count() == 1 && lidar_->get_subscription_count() == 1; }))
      return "front end subscriptions never matched";
    if (relocalization_ && !spinUntil([&] { return relocalization_->get_subscription_count() == 1; }))
      return "relocalization subscription never matched";
    if (typed_ && !spinUntil([&] { return typed_->get_publisher_count() == 1; }))
      return "typed observation publisher never matched";
    if (reference_ && !spinUntil([&] { return reference_->get_publisher_count() == 1; }))
      return "reference health publisher never matched";

    std::string error = scenario_.warm_up ? warmUp() : std::string();
    for (std::size_t i = 0; error.empty() && i < scenario_.steps.size(); ++i)
      error = scenario_.flexible ? runSlot(static_cast<int>(i)) : runScan(static_cast<int>(i));
    if (error.empty() && scenario_.flexible)
      error = settle(true);
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
    if (lines > 0)
      writeAll(fd, line_input_ns_.data(), sizeof(std::int64_t) * line_input_ns_.size());
    const auto idle = static_cast<std::uint32_t>(idle_.size());
    writeAll(fd, &idle, sizeof(idle));
    if (idle > 0)
      writeAll(fd, idle_.data(), sizeof(IdleSample) * idle_.size());
    const auto odom = static_cast<std::uint32_t>(odom_ns_.size());
    writeAll(fd, &odom, sizeof(odom));
    if (odom > 0)
      writeAll(fd, odom_ns_.data(), sizeof(std::int64_t) * odom_ns_.size());
    const auto references = static_cast<std::uint32_t>(reference_cdr_.size());
    writeAll(fd, &references, sizeof(references));
    for (const auto & bytes : reference_cdr_)
      writeString(fd, bytes);
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
    fast_lio_test::initialBuildTakesNoPoint = [this](double end) {
      const Step * s = stepAt(end);
      return s != nullptr && s->no_build_point;
    };
    // A scan's IMU batch is settled before its record is finished, so the record takes the batch reported since the
    // last record; a scan dropped where it is synced settles none. The hook runs with the input buffer unlocked.
    fast_lio_test::afterImuConsumption =
      [this](std::uint64_t, int disposition, int reason, std::size_t samples, std::size_t, std::uint64_t) {
        batch_.disposition = disposition;
        batch_.reason = reason;
        batch_.samples = samples;
        batch_.filter_end = fast_lio_test::inspectImuInput().filterEnd;
        batch_pending_ = true;
      };
    fast_lio_test::afterAnchorScanRecord = [this](const fast_lio::AnchorScanRecord & record) {
      records_.push_back(record);
      if (scenario_.capture_raw_state)
        raw_by_seq_[record.scan_seq] = fast_lio_test::inspectImuInput();
      if (batch_pending_)
        batch_by_seq_[record.scan_seq] = batch_;
      batch_pending_ = false;
    };
    fast_lio_test::afterAnchorTraceLine = [this](const std::string & line) {
      lines_.push_back(line);
      line_input_ns_.push_back(newest_imu_ns_);
    };
  }

  struct ImuBatch
  {
    int disposition = -1;
    int reason = -1;
    std::uint64_t samples = 0;
    double filter_end = 0.0;
  };

  struct TypedSample
  {
    unsigned state = 0;
    unsigned validity = 0;
    unsigned health = 0;
    unsigned axes = 0;
    unsigned substate = 0;
    std::array<std::uint8_t, 16> instance{};
    std::array<std::uint8_t, 32> id{};
    std::array<double, 7> pose{};
  };

  void onTyped(const fast_lio_interfaces::msg::FrontendObservation & message)
  {
    if (message.has_pose && message.has_observation_stamp) {
      TypedSample sample;
      sample.state = message.estimator_state;
      sample.validity = message.output_validity;
      sample.health = message.health_flags;
      sample.axes = message.axis_validity_bits;
      sample.substate = message.fe_substate;
      sample.instance = message.source_instance;
      sample.id = message.observation_id;
      const auto & pose = message.pose.pose;
      sample.pose = {{pose.position.x,
                      pose.position.y,
                      pose.position.z,
                      pose.orientation.x,
                      pose.orientation.y,
                      pose.orientation.z,
                      pose.orientation.w}};
      typed_by_ms_[msKey(static_cast<double>(message.observation_stamp_ns) * 1e-9)] = sample;
      return;
    }
    IdleSample sample;
    sample.input_ns = newest_imu_ns_;
    sample.stamp_ns = message.has_input_stamp ? message.input_stamp_ns : -1;
    sample.state = message.estimator_state;
    sample.health = message.health_flags;
    sample.substate = message.fe_substate;
    idle_.push_back(sample);
  }

  static void clearHooks()
  {
    fast_lio_test::overrideGuardDegraded = {};
    fast_lio_test::beforeAnchorStep = {};
    fast_lio_test::forceRunawayTrip = {};
    fast_lio_test::initialBuildTakesNoPoint = {};
    fast_lio_test::afterAnchorScanRecord = {};
    fast_lio_test::afterAnchorTraceLine = {};
    fast_lio_test::afterImuConsumption = {};
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
  double scanEnd(int scan) const { return scanStart(scan) + scenario_.scan_length; }

  // The scan whose IMU window (previous end, this end] holds time t.
  int scanOfImuTime(double t) const
  {
    const double k = std::ceil((t - kFirstScan - scenario_.scan_length) / kScanPeriod - 1e-9);
    return static_cast<int>(std::max(0.0, k));
  }

  bool inImuHole(double t) const
  {
    for (const auto & hole : imu_holes_)
      if (t > hole.first && t < hole.second)
        return true;
    return false;
  }

  void publishImu(std::int64_t stamp_ns, double yaw_rate)
  {
    sensor_msgs::msg::Imu message;
    message.header.stamp = rclcpp::Time(stamp_ns);
    message.linear_acceleration.z = kGravity;
    message.angular_velocity.z = yaw_rate;
    newest_imu_ns_ = std::max(newest_imu_ns_, stamp_ns);
    imu_->publish(message);
    executor_->spin_some();
  }

  // Every IMU sample stamped up to `end` that has not gone out yet: the fixture's grid, and the extra samples
  // slots asked for, in stamp order.
  void sendImuThrough(double end)
  {
    const auto through = static_cast<std::int64_t>(std::llround(end * 1e9));
    for (;;) {
      const bool grid = next_imu_ns_ <= through;
      const bool extra =
        !extra_imu_ns_.empty() && *extra_imu_ns_.begin() <= through && (!grid || *extra_imu_ns_.begin() < next_imu_ns_);
      if (extra) {
        const std::int64_t stamp = *extra_imu_ns_.begin();
        extra_imu_ns_.erase(extra_imu_ns_.begin());
        sendImuSample(stamp);
        continue;
      }
      if (!grid)
        break;
      sendImuSample(next_imu_ns_);
      next_imu_ns_ += kImuPeriodNs;
    }
  }

  void sendImuSample(std::int64_t stamp_ns)
  {
    const double t = static_cast<double>(stamp_ns) * 1e-9;
    const int scan = scanOfImuTime(t);
    const Imu pattern = imuForScan(scan);
    if (pattern == Imu::kGap && t > scanEnd(scan - 1) + kGapMargin && t < scanEnd(scan) - kGapMargin)
      return;
    if (inImuHole(t))
      return;
    double yaw_rate = pattern == Imu::kMoving ? kMovingYawRate : 0.0;
    if (bad_imu_scan_ == scan && t > scanEnd(scan) - 0.004 && t <= scanEnd(scan))
      yaw_rate = std::numeric_limits<double>::quiet_NaN();
    publishImu(stamp_ns, yaw_rate);
  }

  void sendLidar(double start, Cloud kind = Cloud::kDense)
  {
    const std::size_t points = kind == Cloud::kDense ? kScanPoints : kind == Cloud::kSparse ? 3u : 0u;
    const auto length_ns = static_cast<std::uint64_t>(std::llround(scenario_.scan_length * 1e9));
    pcl::PointCloud<ouster_ros::Point> cloud;
    cloud.resize(points);
    for (std::size_t i = 0; i < points; ++i) {
      auto & point = cloud[i];
      point = ouster_ros::Point{};
      point.x = 10.0f + static_cast<float>(i % 317) * 0.01f;
      point.y = -1.6f + static_cast<float>(i / 317) * 0.01f;
      point.z = 1.0f;
      point.intensity = 1.0f;
      point.t = static_cast<std::uint32_t>(i * length_ns / (points - 1));
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

  // A cloud just sent is in the front end's queue: from here on an IMU sample finds a scan waiting.
  bool waitUntilQueued()
  {
    return spinUntil([] { return fast_lio_test::inspectImuInput().lidarQueued > 0; }, kReceiptWaitSeconds);
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
    if (scenario_.lidar_first) {
      // The cloud is queued before the IMU sample that passes its end: the scan, not that sample, is what the
      // front end processes next.
      sendImuThrough(scanEnd(scan) - 0.001);
      sendLidar(scanStart(scan));
      if (!waitUntilQueued())
        return "the front end never queued the scan";
    }
    sendImuThrough(scanEnd(scan) + 0.006);
    if (!scenario_.lidar_first)
      sendLidar(scanStart(scan));
    const double end = scanEnd(scan);
    const bool done = spinUntil([&] {
      const auto * r = recordAt(end);
      return r != nullptr && (!r->published || odom_by_ms_.count(msKey(end)) > 0) &&
             (!scenario_.reference ||
              (reference_scans_.count(r->scan_seq) != 0 && (!r->published || typed_by_ms_.count(msKey(end)) != 0)));
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
    const auto raw = raw_by_seq_.find(record.scan_seq);
    if (raw != raw_by_seq_.end()) {
      o.raw_state_seen = true;
      o.map_frozen = raw->second.mapFrozen;
      o.raw_degraded = raw->second.degradedOdom;
    }
    const auto odom = odom_by_ms_.find(msKey(static_cast<double>(record.end_ns) * 1e-9));
    if (record.published && odom != odom_by_ms_.end()) {
      const auto & m = odom->second;
      o.odom_seen = 1;
      o.odom_stamp_ns = rclcpp::Time(m.header.stamp).nanoseconds();
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
    const auto typed = typed_by_ms_.find(msKey(static_cast<double>(record.end_ns) * 1e-9));
    if (record.published && typed != typed_by_ms_.end()) {
      o.typed_seen = 1;
      o.typed_state = typed->second.state;
      o.typed_validity = typed->second.validity;
      o.typed_health = typed->second.health;
      o.typed_axes = typed->second.axes;
      o.typed_substate = typed->second.substate;
      o.typed_instance = typed->second.instance;
      o.typed_id = typed->second.id;
      o.typed_pose = typed->second.pose;
    }
    const auto batch = batch_by_seq_.find(record.scan_seq);
    if (batch != batch_by_seq_.end()) {
      o.imu_batch = 1;
      o.imu_disposition = batch->second.disposition;
      o.imu_reason = batch->second.reason;
      o.imu_samples = batch->second.samples;
      o.filter_end = batch->second.filter_end;
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

  // Input-driven scenarios. A slot sends what its step says and then waits only for what the front end can take
  // with the input it has: a scan whose end no IMU sample has passed yet stays queued until a later slot.
  struct PendingScan
  {
    int scan = 0;
    int index = 0;
    bool needs_imu = true;
    std::uint64_t after_seq = 0;  // a repeated cloud's record comes after the records there were when it was sent
  };

  const fast_lio::AnchorScanRecord * recordStartingAt(double start, std::uint64_t after_seq = 0) const
  {
    const std::int64_t key = msKey(start);
    for (auto it = records_.rbegin(); it != records_.rend(); ++it)
      if (it->scan_seq > after_seq && it->beg_ns >= 0 && msKey(static_cast<double>(it->beg_ns) * 1e-9) == key)
        return &*it;
    return nullptr;
  }

  // Moves every scan the front end has finished from the pending list to the observations, in order.
  void collect()
  {
    while (!pending_.empty()) {
      const PendingScan & p = pending_.front();
      const auto * r = recordStartingAt(scanStart(p.scan), p.after_seq);
      if (r == nullptr)
        return;
      if (scenario_.reference && reference_scans_.count(r->scan_seq) == 0)
        return;
      if (r->published) {
        const std::int64_t key = msKey(scanEnd(p.scan));
        if (odom_by_ms_.count(key) == 0 || (scenario_.typed && typed_by_ms_.count(key) == 0))
          return;
      }
      observe(p.index, *r);
      pending_.erase(pending_.begin());
    }
  }

  bool takeable(const PendingScan & p) const
  {
    return !p.needs_imu || newest_imu_ns_ >= static_cast<std::int64_t>(std::llround(scanEnd(p.scan) * 1e9));
  }

  std::string settle(bool final)
  {
    const bool done = spinUntil([&] {
      collect();
      return pending_.empty() || !takeable(pending_.front());
    });
    if (!done || (final && !pending_.empty())) {
      std::ostringstream out;
      out << "scan " << pending_.front().scan << " (step " << pending_.front().index << ") produced no record";
      return out.str();
    }
    return "";
  }

  // The front end's own subscription to the relocalization result.
  rclcpp::SubscriptionBase::SharedPtr relocalizationSubscription()
  {
    return subject_->get_node_base_interface()->get_default_callback_group()->find_subscription_ptrs_if(
      [](const rclcpp::SubscriptionBase::SharedPtr & subscription) {
        return std::strcmp(subscription->get_topic_name(), kRelocalizationTopic) == 0;
      });
  }

  // The timer the front end runs its scan path from: the one timer of the node with that period.
  rclcpp::TimerBase::SharedPtr scanPathTimer()
  {
    int found = 0;
    rclcpp::TimerBase::SharedPtr match;
    subject_->get_node_base_interface()->get_default_callback_group()->find_timer_ptrs_if(
      [&](const rclcpp::TimerBase::SharedPtr & timer) {
        std::int64_t period_ns = 0;
        if (rcl_timer_get_period(timer->get_timer_handle().get(), &period_ns) != RCL_RET_OK) {
          rcl_reset_error();
        } else if (period_ns == kScanPathPeriodNs) {
          ++found;
          match = timer;
        }
        return false;
      });
    return found == 1 ? match : nullptr;
  }

  // Waits up to `timeout_ns` for `subscription` to hold a message nobody has taken yet or, when no subscription is
  // given, for `timer` to come due, and says in `ready` whether it does. Nothing spins meanwhile, so nothing takes
  // the message or runs the timer. Returns what went wrong with the wait itself.
  std::string
  waitUnspun(const rcl_subscription_t * subscription, const rcl_timer_t * timer, std::int64_t timeout_ns, bool * ready)
  {
    *ready = false;
    const bool message = subscription != nullptr;
    rcl_wait_set_t wait_set = rcl_get_zero_initialized_wait_set();
    const auto context = subject_->get_node_base_interface()->get_context()->get_rcl_context();
    if (rcl_wait_set_init(
          &wait_set, message ? 1 : 0, 0, message ? 0 : 1, 0, 0, 0, context.get(), rcl_get_default_allocator()) !=
        RCL_RET_OK) {
      const std::string error = rcl_get_error_string().str;
      rcl_reset_error();
      return "wait set: " + error;
    }
    std::string error;
    rcl_ret_t ret = message ? rcl_wait_set_add_subscription(&wait_set, subscription, nullptr)
                            : rcl_wait_set_add_timer(&wait_set, timer, nullptr);
    if (ret == RCL_RET_OK)
      ret = rcl_wait(&wait_set, timeout_ns);
    if (ret == RCL_RET_OK)
      *ready = message ? wait_set.subscriptions[0] != nullptr : wait_set.timers[0] != nullptr;
    else if (ret != RCL_RET_TIMEOUT)
      error = std::string(message ? "wait on the subscription: " : "wait on the timer: ") + rcl_get_error_string().str;
    rcl_reset_error();
    if (rcl_wait_set_fini(&wait_set) != RCL_RET_OK)
      rcl_reset_error();
    return error;
  }

  // Says in `held` whether `subscription` holds, within `timeout_ns`, a message nobody has taken yet.
  std::string
  pollMessage(const rclcpp::SubscriptionBase::SharedPtr & subscription, std::int64_t timeout_ns, bool * held)
  {
    return waitUnspun(subscription->get_subscription_handle().get(), nullptr, timeout_ns, held);
  }

  // The scan path acts on what the input callbacks left it, and nothing it does without a scan is reported. Its
  // receipt is the timer it runs from: the timer comes due, with nothing spinning meanwhile, and then the executor
  // has run it. `what` names, in a failure, what the run was waited for.
  std::string runScanPath(const std::string & what)
  {
    std::ostringstream out;
    out << what << ": ";
    const auto timer = scanPathTimer();
    if (!timer) {
      out << "the front end does not have exactly one timer with the scan path's period";
      return out.str();
    }
    bool due = false;
    const std::string error = waitUnspun(nullptr, timer->get_timer_handle().get(), kReceiptWaitNs, &due);
    if (!error.empty()) {
      out << error;
      return out.str();
    }
    if (!due && !timer->is_ready()) {
      out << "the scan path's timer did not come due within " << kReceiptWaitSeconds << " s";
      return out.str();
    }
    if (!spinUntil([&] { return !timer->is_ready(); }, kReceiptWaitSeconds)) {
      out << "the scan path's timer came due, but the executor did not run it within " << kReceiptWaitSeconds << " s";
      return out.str();
    }
    return "";
  }

  // The front end takes the relocalization result over DDS, and its callback only stores the transform, so the
  // receipt is the subscription itself: it held the message, then the executor took it off the subscription, which
  // runs the callback.
  std::string deliverRelocalization(int number, const geometry_msgs::msg::TransformStamped & message)
  {
    std::ostringstream out;
    out << "relocalization result #" << number;
    const auto subscription = relocalizationSubscription();
    if (!subscription) {
      out << ": the front end has no subscription to " << kRelocalizationTopic;
      return out.str();
    }
    relocalization_->publish(message);
    bool held = false;
    std::string error = pollMessage(subscription, kReceiptWaitNs, &held);
    if (error.empty() && !held) {
      out << " never reached the front end's subscription within " << kReceiptWaitSeconds << " s";
      return out.str();
    }
    bool taken = false;
    if (error.empty()) {
      taken = spinUntil(
        [&] {
          bool still = false;
          error = pollMessage(subscription, 0, &still);
          return !error.empty() || !still;
        },
        kReceiptWaitSeconds);
    }
    if (!error.empty()) {
      out << ": " << error;
      return out.str();
    }
    if (!taken) {
      out << " reached the front end's subscription, but its callback did not take it within " << kReceiptWaitSeconds
          << " s";
      return out.str();
    }
    return "";
  }

  std::string runSlot(int index)
  {
    const Step & s = scenario_.steps[static_cast<std::size_t>(index)];
    const int scan = scan_++;
    step_by_scan_[scan] = index;
    step_by_end_ms_[msKey(scanEnd(scan))] = index;
    const bool has_next = static_cast<std::size_t>(index) + 1 < scenario_.steps.size();
    step_by_scan_[scan + 1] = has_next ? index + 1 : -1;
    if (s.hole_to > s.hole_from)
      imu_holes_.emplace_back(scanEnd(scan) + s.hole_from, scanEnd(scan) + s.hole_to);
    if (s.bad_imu)
      bad_imu_scan_ = scan;
    if (s.imu_at_end)
      extra_imu_ns_.insert(static_cast<std::int64_t>(std::llround(scanEnd(scan) * 1e9)));
    if (s.relocalization >= 0) {
      const auto & target = scenario_.relocalizations[static_cast<std::size_t>(s.relocalization)];
      geometry_msgs::msg::TransformStamped message;
      message.header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround(scanStart(scan) * 1e9)));
      message.transform.translation.x = target.translation[0];
      message.transform.translation.y = target.translation[1];
      message.transform.translation.z = target.translation[2];
      message.transform.rotation.w = 1.0;
      const std::string error = deliverRelocalization(++relocalizations_sent_, message);
      if (!error.empty())
        return error;
    }
    if (s.wheel) {
      geometry_msgs::msg::TwistStamped message;
      message.header.stamp = rclcpp::Time(static_cast<std::int64_t>(std::llround((scanEnd(scan) - s.wheel_age) * 1e9)));
      message.twist.linear.x = s.wheel_speed;
      wheel_->publish(message);
      executor_->spin_some();
    }
    if (s.clock_back)
      publishImu(std::max<std::int64_t>(1, newest_imu_ns_ - 1500000000LL), 0.0);
    const std::uint64_t refused = fast_lio_test::inspectImuInput().rejectedLidar;
    const bool send = s.cloud != Cloud::kNone;
    const bool cloud_first = send && s.cloud_first;
    const bool lidar_first = send && !cloud_first && scenario_.lidar_first;
    const int cloud_scan = scan - s.repeat;
    if (cloud_first || lidar_first) {
      if (lidar_first)
        sendImuThrough(scanEnd(scan) - 0.001);
      sendLidar(scanStart(cloud_scan), s.cloud);
      if (!waitUntilQueued())
        return "the front end never queued the scan";
    }
    if (!s.hold_imu)
      sendImuThrough(scanEnd(scan) + 0.006);
    if (s.late_imu)
      publishImu(newest_imu_ns_ - 50000000LL, 0.0);
    if (send && !cloud_first && !lidar_first)
      sendLidar(scanStart(cloud_scan), s.cloud);
    if (send && s.expect_scan) {
      const std::uint64_t after_seq = s.repeat > 0 && !records_.empty() ? records_.back().scan_seq : 0u;
      pending_.push_back({cloud_scan, index, s.cloud != Cloud::kEmpty, after_seq});
    } else if (send) {
      // A scan the node does not take leaves no record. Its cloud is refused where it arrives, and the count of
      // refused clouds is that receipt; the scan path then has its run, so what it does about the refusal is done
      // before the next slot, or the end of the scenario.
      std::ostringstream out;
      out << "scan " << scan << " (step " << index << ")";
      if (!spinUntil([&] { return fast_lio_test::inspectImuInput().rejectedLidar > refused; }, kReceiptWaitSeconds)) {
        const auto input = fast_lio_test::inspectImuInput();
        out << " was not refused within " << kReceiptWaitSeconds << " s: refused clouds stayed at "
            << input.rejectedLidar << ", " << input.lidarQueued << " queued";
        return out.str();
      }
      out << " was refused";
      const std::string error = runScanPath(out.str());
      if (!error.empty())
        return error;
    } else {
      executor_->spin_some();
    }
    return settle(false);
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
    typed_.reset();
    reference_.reset();
    relocalization_.reset();
    wheel_.reset();
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
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr wheel_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr relocalization_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_;
  rclcpp::Subscription<fast_lio_interfaces::msg::FrontendObservation>::SharedPtr typed_;
  rclcpp::Subscription<fast_lio_interfaces::msg::ReferenceHealth>::SharedPtr reference_;
  std::vector<std::string> reference_cdr_;
  std::set<std::uint64_t> reference_scans_;
  std::map<std::int64_t, nav_msgs::msg::Odometry> odom_by_ms_;
  std::map<std::int64_t, TypedSample> typed_by_ms_;
  std::vector<IdleSample> idle_;
  std::vector<std::pair<double, double>> imu_holes_;
  std::vector<PendingScan> pending_;
  std::vector<std::int64_t> line_input_ns_;
  std::vector<std::int64_t> odom_ns_;
  std::set<std::int64_t> extra_imu_ns_;  // IMU samples off the grid, sent with the grid in stamp order
  std::int64_t newest_imu_ns_ = -1;
  int bad_imu_scan_ = -1;
  std::map<std::int64_t, int> step_by_end_ms_;
  std::map<int, int> step_by_scan_;
  std::vector<fast_lio::AnchorScanRecord> records_;
  std::map<std::uint64_t, fast_lio_test::ImuInputSnapshot> raw_by_seq_;
  ImuBatch batch_;
  bool batch_pending_ = false;
  std::map<std::uint64_t, ImuBatch> batch_by_seq_;  // by scan_seq of the record that settled it
  std::vector<Observation> observations_;
  std::vector<std::string> lines_;
  std::int64_t next_imu_ns_ = kImuFirstNs;
  int scan_ = 0;
  int relocalizations_sent_ = 0;
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
  result.line_input_ns.resize(lines);
  if (lines > 0 && !reader.take(result.line_input_ns.data(), sizeof(std::int64_t) * lines)) {
    result.error = "child trace line stamps truncated";
    return result;
  }
  std::uint32_t idle = 0;
  if (!reader.take(&idle, sizeof(idle))) {
    result.error = "child idle samples truncated";
    return result;
  }
  result.idle.resize(idle);
  if (idle > 0 && !reader.take(result.idle.data(), sizeof(IdleSample) * idle)) {
    result.error = "child idle samples truncated";
    return result;
  }
  std::uint32_t odom = 0;
  if (!reader.take(&odom, sizeof(odom))) {
    result.error = "child odometry stamps truncated";
    return result;
  }
  result.odom_ns.resize(odom);
  if (odom > 0 && !reader.take(result.odom_ns.data(), sizeof(std::int64_t) * odom)) {
    result.error = "child odometry stamps truncated";
    return result;
  }
  std::uint32_t references = 0;
  if (!reader.take(&references, sizeof(references))) {
    result.error = "child reference count truncated";
    return result;
  }
  for (std::uint32_t i = 0; i < references; ++i) {
    std::string bytes;
    if (!reader.takeString(&bytes)) {
      result.error = "child reference CDR truncated";
      return result;
    }
    rclcpp::SerializedMessage wire(bytes.size());
    auto & data = wire.get_rcl_serialized_message();
    std::memcpy(data.buffer, bytes.data(), bytes.size());
    data.buffer_length = bytes.size();
    fast_lio_interfaces::msg::ReferenceHealth message;
    rclcpp::Serialization<fast_lio_interfaces::msg::ReferenceHealth> serialization;
    serialization.deserialize_message(&wire, &message);
    result.references.push_back(std::move(message));
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
  integer("anchor_state", static_cast<int>(x.anchor_state), static_cast<int>(y.anchor_state));
  integer("episode", static_cast<long long>(x.episode), static_cast<long long>(y.episode));
  integer("ref_gen", static_cast<long long>(x.ref_gen), static_cast<long long>(y.ref_gen));
  integer("evidence", static_cast<int>(x.evidence), static_cast<int>(y.evidence));
  integer("evidence_reason", static_cast<int>(x.evidence_reason), static_cast<int>(y.evidence_reason));
  integer("imu_only", x.imu_only, y.imu_only);
  integer("wheel_valid", x.wheel_valid, y.wheel_valid);
  integer("usable", x.usable, y.usable);
  integer("zero_only", x.zero_only, y.zero_only);
  integer("supported", x.supported, y.supported);
  integer("trust_pre", static_cast<int>(x.trust_pre), static_cast<int>(y.trust_pre));
  integer("trust_post", static_cast<int>(x.trust_post), static_cast<int>(y.trust_post));
  integer("recovery_episode", static_cast<long long>(x.recovery_episode), static_cast<long long>(y.recovery_episode));
  integer("recovery_start_ns", x.recovery_start_ns, y.recovery_start_ns);
  integer("recovery_deadline_ns", x.recovery_deadline_ns, y.recovery_deadline_ns);
  integer("startup_reference", static_cast<int>(x.startup_reference), static_cast<int>(y.startup_reference));
  integer("eff_degraded", x.eff_degraded, y.eff_degraded);
  integer("static_hold", x.static_hold, y.static_hold);
  integer("imu_batch", a.imu_batch, b.imu_batch);
  integer("imu_disposition", a.imu_disposition, b.imu_disposition);
  integer("imu_reason", a.imu_reason, b.imu_reason);
  integer("imu_samples", static_cast<long long>(a.imu_samples), static_cast<long long>(b.imu_samples));
  real("filter_end", a.filter_end, b.filter_end);
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

// Transition records of both tags, in the order they were emitted; the counter summary comes last.
std::vector<std::string> transitionLines(const ChildResult & r)
{
  std::vector<std::string> out;
  for (const auto & line : r.lines)
    if (line.rfind("[ZUPT-ANCHOR] ", 0) == 0 || line.rfind("[FE-RECOVERY] ", 0) == 0)
      out.push_back(line);
  return out;
}

// The records of one tag whose `key` reads `value`.
std::vector<std::string>
linesWhere(const ChildResult & r, const std::string & tag, const std::string & key, const std::string & value)
{
  std::vector<std::string> out;
  for (const auto & line : linesTagged(r, tag))
    if (valueOf(line, key) == value)
      out.push_back(line);
  return out;
}

// A key of the counter summary, the last anchor record of the run.
std::string counterValue(const ChildResult & r, const std::string & key)
{
  const auto anchor = linesTagged(r, "[ZUPT-ANCHOR]");
  if (anchor.empty() || valueOf(anchor.back(), "event") != "counters")
    return "<no counter summary>";
  return valueOf(anchor.back(), key);
}

// Exactly one record ended the anchor `id`, for `reason`, and left no anchor behind.
void expectInvalidatedOnce(const ChildResult & r,
                           std::uint64_t id,
                           const std::string & reason,
                           const std::string & name)
{
  std::vector<std::string> ended;
  for (const auto & line : linesWhere(r, "[ZUPT-ANCHOR]", "event", "invalidated"))
    if (valueOf(line, "anchor_id") == std::to_string(id))
      ended.push_back(line);
  ASSERT_EQ(ended.size(), 1u) << name << ": records ending anchor " << id;
  EXPECT_EQ(valueOf(ended.front(), "reason"), reason) << name << ": " << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "anchor_state"), "empty") << name << ": " << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "streak"), "0") << name << ": " << ended.front();
}

bool samePoseCovariance(const fast_lio::AnchorScanRecord & r)
{
  for (int i = 0; i < 36; ++i)
    if (!sameBits(r.p_pose_pre[i], r.p_pose_post[i]))
      return false;
  return true;
}

// Degraded and parked without a usable anchor: the velocity alone is zeroed. Position, attitude and their
// covariance block are bit for bit what the scan brought, /Odometry carries that position, and the scan reports
// Degraded without StaticHold.
bool zeroedVelocityOnly(const Observation & o, const std::array<double, 3> & position, std::string * why)
{
  const auto & r = o.record;
  std::ostringstream out;
  if (r.action != fast_lio::AnchorTraceAction::kZeroVelocity)
    out << "action ";
  if (!r.zero_only || r.usable)
    out << "zero_only=" << r.zero_only << " usable=" << r.usable << ' ';
  if (r.anchor_used != 0u)
    out << "anchor_used=" << r.anchor_used << ' ';
  if (!samePosition(r.pre_p, position) || !samePosition(r.post_p, position))
    out << "position pre " << formatPosition(r.pre_p) << " post " << formatPosition(r.post_p) << ' ';
  for (int i = 0; i < 4; ++i) {
    if (!sameBits(r.pre_q[i], r.post_q[i])) {
      out << "attitude ";
      break;
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (!sameBits(r.post_v[i], 0.0)) {
      out << "velocity ";
      break;
    }
  }
  if (!r.have_cov || !samePoseCovariance(r))
    out << "pose_covariance ";
  if (!o.odom_seen || !samePosition(o.odom_p, position))
    out << "odom " << formatPosition(o.odom_p) << ' ';
  if (odomFlags(o) != kFlagDegraded)
    out << "flags=" << odomFlags(o) << ' ';
  if (!r.eff_degraded || r.static_hold)
    out << "eff_degraded=" << r.eff_degraded << " static_hold=" << r.static_hold << ' ';
  *why = out.str();
  return why->empty();
}

bool sameArray(const double * a, const double * b, std::size_t n)
{
  return std::memcmp(a, b, sizeof(double) * n) == 0;
}

bool allZero(const double v[3])
{
  return sameBits(v[0], 0.0) && sameBits(v[1], 0.0) && sameBits(v[2], 0.0);
}

// What the transition records carry since trace format version 3: the parked streak after the event on every
// anchor record, the path that wrote it on every recovery record, and no new key on the counter summary.
void expectTransitionKeys(const std::vector<std::string> & transitions, const std::string & name)
{
  for (std::size_t i = 0; i + 1 < transitions.size(); ++i) {
    const std::string & line = transitions[i];
    if (line.rfind("[FE-RECOVERY] ", 0) == 0) {
      const std::string path = valueOf(line, "path");
      EXPECT_TRUE(path == "scan" || path == "idle") << name << ": " << line;
      continue;
    }
    const std::string event = valueOf(line, "event");
    const std::string streak = valueOf(line, "streak");
    EXPECT_NE(streak, "<missing>") << name << ": " << line;
    // Whatever breaks the evidence, or ends a candidate, an anchor or the prior, leaves no streak behind. A
    // candidate's first scan starts one, and its third makes the anchor.
    if (event == "evidence" || event == "candidate_dropped" || event == "invalidated" ||
        event == "parked_prior_released") {
      EXPECT_EQ(streak, "0") << name << ": " << line;
    }
    if (event == "candidate_started") {
      EXPECT_EQ(streak, "1") << name << ": " << line;
    }
    if (event == "created") {
      EXPECT_EQ(streak, "3") << name << ": " << line;
    }
  }
  if (!transitions.empty()) {
    EXPECT_EQ(valueOf(transitions.back(), "streak"), "<missing>") << name << ": " << transitions.back();
  }
}

// What only the per-scan trace observes (anchor_trace.hpp, "Since version 2"), on every scan of a run with the
// trace on: which of the fields are present; that a candidate's stored pose is the pose its first scan had, stays
// what it was, and is the target of every write that uses it; what a pin and a zero-only scan leave in the state;
// that a Pending scan adds nothing to the map.
void expectAnchorWriteFields(const ChildResult & r, const std::string & name)
{
  struct Pose
  {
    std::array<double, 3> p;
    std::array<double, 4> q;
  };
  std::map<std::uint64_t, Pose> stored;  // by candidate/anchor ID: the pose its first record carried
  for (const auto & o : r.observations) {
    const auto & x = o.record;
    const std::string at = name + " scan " + std::to_string(x.scan_seq);
    const bool wrote = x.action == fast_lio::AnchorTraceAction::kPin ||
                       x.action == fast_lio::AnchorTraceAction::kZeroVelocity || x.parked_applied;
    EXPECT_EQ(x.have_act, wrote) << at;
    EXPECT_EQ(x.have_target, x.anchor_used != 0u) << at;
    EXPECT_EQ(x.have_anchor_pose, x.anchor_state != fast_lio::AnchorState::kEmpty) << at;
    if (x.anchor_state == fast_lio::AnchorState::kEmpty) {
      EXPECT_EQ(x.streak, 0u) << at << ": a parked streak without a candidate or an anchor";
    }
    if (x.have_anchor_pose) {
      const auto known = stored.find(x.anchor_id);
      if (known == stored.end()) {
        Pose pose;
        std::copy(x.anchor_p, x.anchor_p + 3, pose.p.begin());
        std::copy(x.anchor_q, x.anchor_q + 4, pose.q.begin());
        stored.emplace(x.anchor_id, pose);
      } else {
        EXPECT_TRUE(sameArray(x.anchor_p, known->second.p.data(), 3) &&
                    sameArray(x.anchor_q, known->second.q.data(), 4))
          << at << ": the stored pose of anchor " << x.anchor_id << " changed";
      }
    }
    if (x.have_target) {
      const auto used = stored.find(x.anchor_used);
      if (used == stored.end()) {
        ADD_FAILURE() << at << ": a target from anchor " << x.anchor_used << ", whose pose no scan recorded";
      } else {
        EXPECT_TRUE(sameArray(x.target_p, used->second.p.data(), 3) && sameArray(x.target_q, used->second.q.data(), 4))
          << at << ": the target is not the stored pose of anchor " << x.anchor_used;
      }
    }
    if (x.action == fast_lio::AnchorTraceAction::kPin) {
      EXPECT_TRUE(x.have_target && sameArray(x.act_p, x.target_p, 3) && sameArray(x.act_q, x.target_q, 4))
        << at << ": the pin did not leave the state at its target";
      EXPECT_TRUE(allZero(x.act_v)) << at;
    }
    if (x.zero_only) {
      EXPECT_EQ(x.action, fast_lio::AnchorTraceAction::kZeroVelocity) << at;
      EXPECT_FALSE(x.parked_applied || x.static_hold || x.have_target) << at;
      EXPECT_TRUE(x.have_pre && sameArray(x.act_p, x.pre_p, 3) && sameArray(x.act_q, x.pre_q, 4))
        << at << ": a zero-only scan moved the pose";
      EXPECT_TRUE(allZero(x.act_v)) << at;
      EXPECT_TRUE(sameArray(x.p_pose_act, x.p_pose_pre, 36)) << at << ": a zero-only scan changed the pose covariance";
    }
    if (x.have_update) {
      EXPECT_TRUE(std::isfinite(x.upd_dp) && x.upd_dp >= 0.0) << at;
      EXPECT_TRUE(std::isfinite(x.upd_dq) && x.upd_dq >= 0.0) << at;
    }
    if (x.trust_post == fast_lio::RecoveryTrust::kPending) {
      EXPECT_FALSE(x.map_updated) << at << ": a Pending scan added its points to the map";
    }
  }
  for (const auto & line : linesWhere(r, "[ZUPT-ANCHOR]", "event", "candidate_started")) {
    const std::uint64_t seq = std::stoull(valueOf(line, "scan_seq"));
    ASSERT_TRUE(seq >= 1u && seq <= r.observations.size()) << name << ": " << line;
    const auto & x = r.observations[seq - 1].record;
    if (x.have_anchor_pose && std::to_string(x.anchor_id) == valueOf(line, "anchor_id")) {
      EXPECT_TRUE(x.have_pre && sameArray(x.anchor_p, x.pre_p, 3) && sameArray(x.anchor_q, x.pre_q, 4))
        << name << " scan " << seq << ": the candidate's pose is not the one its first scan had";
    }
  }
}

// K7 on the node: one scan record per scan taken, sequences without gaps or repeats, counters that reconcile.
void expectTraceAccountsForEveryScan(const ChildResult & r, bool trace_on, const std::string & name)
{
  const auto scans = linesTagged(r, "[ZUPT-ANCHOR-SCAN]");
  const auto transitions = transitionLines(r);
  ASSERT_FALSE(transitions.empty()) << name << ": no counter summary";
  const std::string counters = transitions.back();
  EXPECT_EQ(valueOf(counters, "event"), "counters") << name;
  EXPECT_EQ(valueOf(counters, "scans"), std::to_string(r.observations.size())) << name;
  EXPECT_EQ(valueOf(counters, "events"), std::to_string(transitions.size() - 1)) << name;
  for (std::size_t i = 0; i < transitions.size(); ++i) {
    EXPECT_EQ(valueOf(transitions[i], "v"), std::to_string(fast_lio::kAnchorTraceVersion)) << name;
    EXPECT_EQ(valueOf(transitions[i], "event_seq"), std::to_string(i + 1)) << name << ": " << transitions[i];
  }
  expectTransitionKeys(transitions, name);
  if (!trace_on) {
    EXPECT_TRUE(scans.empty()) << name << ": scan records without zupt_anchor_trace_en";
    // Nothing is observed for a record that is not printed.
    for (const auto & o : r.observations) {
      const auto & x = o.record;
      EXPECT_FALSE(x.have_act || x.have_target || x.have_anchor_pose || x.have_update || x.map_updated ||
                   x.streak != 0u)
        << name << " scan " << x.scan_seq << ": observed without zupt_anchor_trace_en";
    }
    return;
  }
  expectAnchorWriteFields(r, name);
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

// ---------------------------------------------------------------------------
// Input-driven scenarios: nothing injected into the filter. The guard's verdict is still fixed per scan, so the
// events come from what the fixture sends (or withholds) and from the node's own gates.
// ---------------------------------------------------------------------------

constexpr std::int64_t kNsPerSecond = 1000000000;
constexpr std::int64_t kScanPeriodNs = 100000000;
constexpr std::int64_t kScanLengthNs = 50000000;
constexpr std::int64_t kK5BoundNs = 100000000;  // a deadline is acted on no later than this after it

Step slot(Phase phase, Imu imu = Imu::kStatic, Guard guard = Guard::kHealthy, Cloud cloud = Cloud::kDense)
{
  Step s;
  s.phase = phase;
  s.imu = imu;
  s.guard = guard;
  s.cloud = cloud;
  return s;
}

void append(Scenario & s, int count, const Step & what)
{
  for (int i = 0; i < count; ++i)
    s.steps.push_back(what);
}

Scenario inputScenario(const std::string & name)
{
  Scenario s = baseParams(name, false);
  s.flexible = true;
  return s;
}

void setPolicy(Scenario & s, double total, double stable)
{
  setParam(s, rclcpp::Parameter("reanchor_total_timeout_sec", total));
  setParam(s, rclcpp::Parameter("reanchor_recovery_stable_sec", stable));
}

std::int64_t toI64(const std::string & text)
{
  return text == "<missing>" ? std::numeric_limits<std::int64_t>::min() : std::stoll(text);
}

// The newest IMU stamp the fixture had sent when the node logged `line`: the input progress the event came at.
std::int64_t inputNsOf(const ChildResult & r, const std::string & line)
{
  for (std::size_t i = 0; i < r.lines.size() && i < r.line_input_ns.size(); ++i)
    if (r.lines[i] == line)
      return r.line_input_ns[i];
  return std::numeric_limits<std::int64_t>::min();
}

std::vector<std::string> recoveryLines(const ChildResult & r, const std::string & event)
{
  return linesWhere(r, "[FE-RECOVERY]", "event", event);
}

// Anchor records of one event that name anchor `id`.
std::vector<std::string> anchorLines(const ChildResult & r, const std::string & event, std::uint64_t id)
{
  std::vector<std::string> out;
  for (const auto & line : linesWhere(r, "[ZUPT-ANCHOR]", "event", event))
    if (valueOf(line, "anchor_id") == std::to_string(id))
      out.push_back(line);
  return out;
}

std::vector<const Observation *> afterWarmup(const ChildResult & r)
{
  std::vector<const Observation *> out;
  for (const auto & o : r.observations)
    if (o.step >= 0)
      out.push_back(&o);
  return out;
}

// The last scan taken before the scenario's own steps: with the fixture's healthy static warm-up, the last scan
// the reference was trusted on.
const Observation * lastWarmup(const ChildResult & r)
{
  const Observation * last = nullptr;
  for (const auto & o : r.observations)
    if (o.step < 0)
      last = &o;
  return last;
}

bool hasEvent(const Observation & o, std::uint32_t bit)
{
  return (o.record.events & bit) != 0u;
}

// Everything the child reported, for the failure message of a scenario's assertions.
std::string describeRun(const ChildResult & r)
{
  std::ostringstream out;
  out << r.describe() << '\n';
  for (const auto & o : r.observations) {
    const auto & x = o.record;
    out << "  scan " << x.scan_seq << " step " << o.step << " phase " << o.phase << " outcome "
        << static_cast<int>(x.outcome) << " beg " << x.beg_ns << " end " << x.end_ns << " events 0x" << std::hex
        << x.events << std::dec << " anchor " << x.anchor_id << '/' << static_cast<int>(x.anchor_state) << " used "
        << x.anchor_used << " action " << static_cast<int>(x.action) << " usable " << x.usable << " zero_only "
        << x.zero_only << " parked " << x.parked << '/' << x.parked_applied << " hold " << x.zupt_active << " evidence "
        << static_cast<int>(x.evidence) << '/' << static_cast<int>(x.evidence_reason) << " guard " << x.guard_degraded
        << " degraded " << x.degraded << " eff " << x.eff_degraded << " gate " << static_cast<int>(x.gate_pre) << "->"
        << static_cast<int>(x.gate_post) << " trust " << static_cast<int>(x.trust_pre) << "->"
        << static_cast<int>(x.trust_post) << " supported " << x.supported << " ref " << x.ref_gen << " published "
        << x.published << " flags " << x.flags << " odom " << o.odom_seen << " typed " << o.typed_seen << ':'
        << o.typed_state << '/' << o.typed_validity << '/' << o.typed_health << " imu " << o.imu_batch << ':'
        << o.imu_disposition << '/' << o.imu_reason << '/' << o.imu_samples << " gap " << x.gap_ns << " startup "
        << static_cast<int>(x.startup_reference) << '\n';
  }
  for (std::size_t i = 0; i < r.lines.size(); ++i) {
    if (r.lines[i].rfind("[ZUPT-ANCHOR-SCAN] ", 0) == 0)
      continue;
    out << "  @" << (i < r.line_input_ns.size() ? r.line_input_ns[i] : -1) << ' ' << r.lines[i] << '\n';
  }
  return out.str();
}

// The anchor a stretch of healthy parked scans ended with; 0 unless it is valid.
std::uint64_t validAnchorAfter(const ChildResult & r, Phase phase)
{
  const auto scans = r.phase(phase);
  if (scans.empty() || scans.back()->record.anchor_state != fast_lio::AnchorState::kValid)
    return 0u;
  return scans.back()->record.anchor_id;
}

// The first sample of the fixture's IMU grid stamped at or after `t_ns`.
std::int64_t firstGridImuAtOrAfter(std::int64_t t_ns)
{
  if (t_ns <= kImuFirstNs)
    return kImuFirstNs;
  return kImuFirstNs + (t_ns - kImuFirstNs + kImuPeriodNs - 1) / kImuPeriodNs * kImuPeriodNs;
}

// Among the inputs the front end reported between scans, each at the stamp the front end gave it, the stamp nearest
// to `t_ns`; -1 when it reported none.
std::int64_t nearestReportedInputNs(const ChildResult & r, std::int64_t t_ns)
{
  std::int64_t nearest = -1;
  std::int64_t nearest_off = 0;
  for (const auto & idle : r.idle) {
    if (idle.stamp_ns < 0)
      continue;
    const std::int64_t off = idle.stamp_ns > t_ns ? idle.stamp_ns - t_ns : t_ns - idle.stamp_ns;
    if (nearest < 0 || off < nearest_off) {
      nearest = idle.stamp_ns;
      nearest_off = off;
    }
  }
  return nearest;
}

// One /Odometry message per published scan, each carrying that scan's end time: nothing went out between scans,
// such as a pose made up for an IMU sample.
void expectOdometryOnlyForPublishedScans(const ChildResult & r)
{
  std::multiset<std::int64_t> published;
  for (const auto & o : r.observations)
    if (o.record.published)
      published.insert(msKey(static_cast<double>(o.record.end_ns) * 1e-9));
  std::multiset<std::int64_t> received;
  for (const std::int64_t ns : r.odom_ns)
    received.insert(msKey(static_cast<double>(ns) * 1e-9));
  EXPECT_EQ(received, published) << r.odom_ns.size() << " /Odometry messages for " << published.size()
                                 << " published scans";
}

// Every scan of `scans` was processed under the latch: it took no pose from an anchor, held on none, claimed no
// StaticHold, left no candidate or anchor, and reported Degraded. No scan of the run confirmed a recovery.
void expectLatchedFrom(const ChildResult & r, const std::vector<const Observation *> & scans)
{
  for (const auto * o : scans) {
    const auto & x = o->record;
    EXPECT_EQ(x.trust_pre, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
    EXPECT_EQ(x.trust_post, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
    EXPECT_EQ(x.anchor_state, fast_lio::AnchorState::kEmpty) << "step " << o->step;
    EXPECT_EQ(x.anchor_used, 0u) << "step " << o->step;
    EXPECT_NE(x.action, fast_lio::AnchorTraceAction::kPin) << "step " << o->step;
    EXPECT_FALSE(x.parked_applied) << "step " << o->step;
    EXPECT_FALSE(x.static_hold) << "step " << o->step;
    EXPECT_TRUE(x.eff_degraded) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), kFlagDegraded) << "step " << o->step;
  }
  EXPECT_TRUE(recoveryLines(r, "qualified").empty());
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "0");
}

// No scan of `scans` took a pose from anchor `id`, held on it or claimed a static hold.
void expectNeverUsed(const std::vector<const Observation *> & scans, std::uint64_t id, const std::string & name)
{
  for (const auto * o : scans) {
    EXPECT_NE(o->record.anchor_used, id) << name << " step " << o->step << ": took a pose from the ended anchor";
    EXPECT_NE(o->record.action, fast_lio::AnchorTraceAction::kPin) << name << " step " << o->step;
    EXPECT_FALSE(o->record.parked_applied) << name << " step " << o->step;
    EXPECT_FALSE(o->record.static_hold) << name << " step " << o->step;
    if (o->odom_seen) {
      EXPECT_EQ(odomFlags(*o) & kFlagStaticHold, 0u) << name << " step " << o->step;
    }
  }
}

// ---- F02: a scan that left before the anchor step, between a candidate's scans --------------------------------

Scenario earlyExitBetweenCandidateScans(Cloud cloud)
{
  Scenario s = inputScenario(cloud == Cloud::kSparse ? "early_exit_sparse" : "early_exit_empty");
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_scans", 3));
  append(s, 1, slot(Phase::kBefore, Imu::kMoving));  // ends whatever the warm-up left
  append(s, 2, slot(Phase::kAnchorStatic));          // two qualifying scans
  append(s, 1, slot(Phase::kEvent, Imu::kStatic, Guard::kHealthy, cloud));
  append(s, 4, slot(Phase::kAfter));
  return s;
}

// ---- F04: IMU gaps the motion model bridges --------------------------------------------------------------------

constexpr double kBridgeHoleFrom = -0.065;  // s from the event scan's end: 35 ms into its IMU window

// `event_guard` is the guard's verdict on the scan the bridge starts on.
Scenario imuBridge(double hole_s, double total, Guard event_guard = Guard::kHealthy)
{
  std::ostringstream name;
  name << "imu_bridge_" << hole_s << (total > 0.0 ? "_policy_on" : "_policy_off")
       << (event_guard == Guard::kDegraded ? "_degraded" : "");
  Scenario s = inputScenario(name.str());
  setParam(s, rclcpp::Parameter("reanchor_en", true));
  setParam(s, rclcpp::Parameter("imu_gap_prior_en", true));
  // Below the fixture's scan period, so a scan without any IMU sample is bridged whichever way 0.1 s rounds.
  setParam(s, rclcpp::Parameter("imu_gap_min_s", 0.05));
  setParam(s, rclcpp::Parameter("imu_gap_empty_scans", true));
  setPolicy(s, total, 1.0);
  append(s, 5, slot(Phase::kBefore));
  Step event = slot(Phase::kEvent, Imu::kStatic, event_guard);
  event.hole_from = kBridgeHoleFrom;
  event.hole_to = kBridgeHoleFrom + hole_s;
  s.steps.push_back(event);
  append(s, 19, slot(Phase::kAfter));
  return s;
}

// ---- Scan holes -------------------------------------------------------------------------------------------------

Scenario scanHole(int missing, Imu inside)
{
  Scenario s = inputScenario(inside == Imu::kMoving ? "scan_hole_moving" : "scan_hole_static");
  setPolicy(s, 10.0, 1.0);
  append(s, 5, slot(Phase::kBefore));
  append(s, missing, slot(Phase::kEvent, inside, Guard::kHealthy, Cloud::kNone));
  append(s, 14, slot(Phase::kAfter));
  return s;
}

// ---- F01: the deadline on the IMU clock ------------------------------------------------------------------------

constexpr double kIdleBudget = 1.0;  // s, reanchor_total_timeout_sec of the idle scenarios
constexpr std::int64_t kIdleBudgetNs = 1000000000;

// One healthy moving scan (the last supported one), then `degraded` parked scans without an anchor: Pending.
Scenario pendingByDegradedScans(const std::string & name, int degraded, double total = kIdleBudget)
{
  Scenario s = inputScenario(name);
  setPolicy(s, total, 0.5);
  append(s, 1, slot(Phase::kBefore, Imu::kMoving));
  append(s, degraded, slot(Phase::kEvent, Imu::kStatic, Guard::kDegraded));
  return s;
}

// Scans stop while the IMU goes on.
Scenario idleWithImu()
{
  Scenario s = pendingByDegradedScans("idle_imu_only", 2);
  append(s, 14, slot(Phase::kAfter, Imu::kStatic, Guard::kHealthy, Cloud::kNone));
  return s;
}

// The matching run that would end Pending completes on the very scan whose end is the deadline. `total` just above
// the scenario's budget is the control: the same run then ends Pending. Every cloud is queued before the IMU sample
// that passes its end, so that scan waits while the first sample past the deadline arrives.
Scenario confirmationAtTheDeadline(double total)
{
  Scenario s =
    pendingByDegradedScans(total > kIdleBudget ? "confirmation_before_deadline" : "confirmation_at_deadline", 4, total);
  s.lidar_first = true;
  append(s, 6, slot(Phase::kAfter));
  append(s, 2, slot(Phase::kLater));
  return s;
}

constexpr double kResumeAfterDeadline = 0.2;  // s past the deadline: inside the old half-second settle margin

// Every input stops before the deadline and comes back after it.
Scenario allInputStops()
{
  Scenario s = pendingByDegradedScans("all_input_stops", 2);
  // The degraded scans end 0.1 s and 0.2 s after the last supported one; the deadline is 1.0 s after it.
  Step first = slot(Phase::kAfter, Imu::kStatic, Guard::kHealthy, Cloud::kNone);
  first.hole_from = -0.094;  // everything after the samples already sent with the last scan
  first.hole_to = (kIdleBudget + kResumeAfterDeadline) - 0.3;
  s.steps.push_back(first);
  append(s, 14, slot(Phase::kAfter, Imu::kStatic, Guard::kHealthy, Cloud::kNone));
  return s;
}

// ---- The deadline while a scan waits for its IMU --------------------------------------------------------------

constexpr double kQueuedStable = 0.5;            // s, reanchor_recovery_stable_sec where no run is to complete
constexpr double kConfirmStable = 1.0;           // s, the recovery's stable time where a run is to complete
constexpr int kConfirmRun = 11;                  // scans of a run that spans it: 10 x 0.1 s + 0.05 s
constexpr double kConfirmLate = 1.201;           // s: the deadline 1 ms after the end of the run's last scan
constexpr double kConfirmInTime = 1.2055;        // s: 5.5 ms after it, past the IMU sample 2.5 ms after that end
constexpr double kLidarBackAfterDeadline = 0.2;  // s: the IMU is back this long after the deadline
constexpr double kOnSampleBudget = 1.2;          // s: puts the deadline on 2.75 s, a stamp an IMU sample keeps
constexpr int kOnSampleDegraded = 11;            // degraded scans up to 0.1 s before that deadline
constexpr double kEverySampleReported = 1e-6;    // s, publish.frontend_idle_period_sec below the IMU period
constexpr std::int64_t kConfirmStableNs = 1000000000;
constexpr std::int64_t kOnSampleBudgetNs = 1200000000;

// One healthy moving scan, the last supported one, then `degraded` degraded parked scans: Pending, timed from that
// scan.
Scenario pendingFromOneScan(const std::string & name, double total, double stable, int degraded)
{
  Scenario s = inputScenario(name);
  setPolicy(s, total, stable);
  append(s, 1, slot(Phase::kBefore, Imu::kMoving));
  append(s, degraded, slot(Phase::kEvent, Imu::kStatic, Guard::kDegraded));
  return s;
}

// A slot that sends nothing: no cloud, and its IMU samples go out with a later slot.
Step heldSlot(Phase phase)
{
  Step s = slot(phase, Imu::kStatic, Guard::kHealthy, Cloud::kNone);
  s.hold_imu = true;
  return s;
}

// The scan S whose end is 0.2 s past the deadline D is queued first. The scans at D and D + 0.1 s never arrive,
// and the IMU samples from 0.1 s before D on go out only after S is queued, one by one: D - 2.5 ms, a sample on D
// itself, D + 2.5 ms, and so on up to S. Parked healthy scans follow. The front end takes an IMU stamp through
// seconds as a double and back, and drops what is left of a nanosecond: only a stamp whose fraction of a second is
// exact in binary reaches the deadline check as it was sent. The budget puts D on one, 2.75 s, so that the sample
// sent on D is the first one at or past it. The typed observation is on, with an idle period below the IMU period:
// between scans the front end then reports every IMU sample it takes, at the stamp it gave the sample.
Scenario scanQueuedPastTheDeadline()
{
  Scenario s = pendingFromOneScan("scan_queued_past_deadline", kOnSampleBudget, kQueuedStable, kOnSampleDegraded);
  s.typed = true;
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  setParam(s, rclcpp::Parameter("publish.frontend_idle_period_sec", kEverySampleReported));
  Step at_deadline = heldSlot(Phase::kEvent);
  at_deadline.imu_at_end = true;
  s.steps.push_back(at_deadline);
  s.steps.push_back(heldSlot(Phase::kEvent));
  Step queued = slot(Phase::kAfter);
  queued.cloud_first = true;
  s.steps.push_back(queued);
  append(s, 6, slot(Phase::kLater));
  return s;
}

// After one degraded scan, matching scans whose run reaches the 1 s stable time on its 11th scan. That scan is
// queued before its IMU samples: the one 2.5 ms past its end makes it processable. `total` puts the deadline just
// before that sample (kConfirmLate) or just after it (kConfirmInTime). Parked healthy scans follow.
Scenario confirmationQueuedAtTheDeadline(double total)
{
  Scenario s =
    pendingFromOneScan(total < kConfirmInTime ? "confirmation_queued_at_deadline" : "confirmation_queued_in_time",
                       total,
                       kConfirmStable,
                       1);
  append(s, kConfirmRun - 1, slot(Phase::kAfter));
  Step last = slot(Phase::kAfter);
  last.cloud_first = true;
  s.steps.push_back(last);
  append(s, 6, slot(Phase::kLater));
  return s;
}

// Every input stops after two degraded scans. The LiDAR is back first: the first scan after the hole, whose end is
// 0.3 s past the deadline, is queued before any IMU sample, and the IMU is back 0.2 s past the deadline. Parked
// healthy scans follow.
Scenario lidarBackBeforeTheImu()
{
  Scenario s = pendingFromOneScan("lidar_back_before_imu", kIdleBudget, kQueuedStable, 2);
  // The degraded scans end 0.1 s and 0.2 s after the last supported one; nothing is sent for the next ten slots.
  Step first = heldSlot(Phase::kEvent);
  first.hole_from = -0.094;  // everything after the samples already sent with the last scan
  first.hole_to = (kIdleBudget + kLidarBackAfterDeadline) - 0.3;
  s.steps.push_back(first);
  append(s, 9, heldSlot(Phase::kEvent));
  Step back = slot(Phase::kAfter);
  back.cloud_first = true;
  s.steps.push_back(back);
  append(s, 6, slot(Phase::kLater));
  return s;
}

// ---- IMU stretches that were not observed ----------------------------------------------------------------------

enum class LostImu : std::uint8_t
{
  kTail,     // no sample for the last part of the scan's window: coverage "end"
  kEmpty,    // a scan whose window holds no sample at all
  kInvalid,  // a sample with a non-finite rate: the batch is lost
  kBacklog,  // scans stop long enough for the IMU queue to shed valid samples
};

const char * lostImuName(LostImu what)
{
  switch (what) {
    case LostImu::kTail:
      return "coverage_end";
    case LostImu::kEmpty:
      return "empty_window";
    case LostImu::kInvalid:
      return "invalid_sample";
    case LostImu::kBacklog:
      return "backlog_loss";
  }
  return "unknown";
}

Scenario lostImu(LostImu what)
{
  Scenario s = inputScenario(lostImuName(what));
  setPolicy(s, 10.0, 1.0);
  append(s, 5, slot(Phase::kBefore));
  Step event = slot(Phase::kEvent);
  switch (what) {
    case LostImu::kTail:
      setParam(s, rclcpp::Parameter("reanchor_en", true));
      setParam(s, rclcpp::Parameter("imu_coverage.max_extrapolation_s", 0.02));
      event.hole_from = -0.035;
      event.hole_to = 0.0;
      s.steps.push_back(event);
      break;
    case LostImu::kEmpty:
      // The sample 2.5 ms past the previous end went out with that scan, so the first event scan still has one
      // sample; the second has none.
      event.hole_from = -0.094;
      event.hole_to = 0.101;
      s.steps.push_back(event);
      s.steps.push_back(slot(Phase::kEvent));
      break;
    case LostImu::kInvalid:
      event.bad_imu = true;
      s.steps.push_back(event);
      break;
    case LostImu::kBacklog:
      setParam(s, rclcpp::Parameter("reanchor_en", true));
      setParam(s, rclcpp::Parameter("imu_coverage.max_gap_s", 0.05));
      setParam(s, rclcpp::Parameter("max_imu_backlog_sec", 0.5));
      append(s, 12, slot(Phase::kEvent, Imu::kStatic, Guard::kHealthy, Cloud::kNone));
      s.steps.push_back(event);
      break;
  }
  append(s, 3, slot(Phase::kAfter, Imu::kStatic, Guard::kDegraded));
  return s;
}

Scenario lateImuSample()
{
  Scenario s = inputScenario("late_imu_sample");
  append(s, 5, slot(Phase::kBefore));
  Step event = slot(Phase::kEvent);
  event.late_imu = true;
  s.steps.push_back(event);
  append(s, 3, slot(Phase::kAfter, Imu::kStatic, Guard::kDegraded));
  return s;
}

Scenario inputEpochFault()
{
  Scenario s = inputScenario("input_epoch_fault");
  append(s, 5, slot(Phase::kBefore));
  Step event = slot(Phase::kEvent);
  event.clock_back = true;
  event.expect_scan = false;
  s.steps.push_back(event);
  Step after = slot(Phase::kAfter);
  after.expect_scan = false;
  append(s, 3, after);
  return s;
}

// ---- Wheel evidence ---------------------------------------------------------------------------------------------

constexpr int kWheelFreshScans = 4;

Step wheelSlot(Phase phase, double speed, double age)
{
  Step s = slot(phase);
  s.wheel = true;
  s.wheel_speed = speed;
  s.wheel_age = age;
  return s;
}

// A wheel sample is required. None at first; then four fresh zeros before each bad one: stale, non-finite, and
// stamped in another clock epoch.
Scenario wheelEvidence()
{
  Scenario s = inputScenario("wheel_evidence");
  setParam(s, rclcpp::Parameter("zupt_require_wheel", true));
  setParam(s, rclcpp::Parameter("wheel_topic", kWheelTopic));
  append(s, 4, slot(Phase::kBefore));
  append(s, kWheelFreshScans, wheelSlot(Phase::kAnchorStatic, 0.0, 0.01));
  s.steps.push_back(wheelSlot(Phase::kEvent, 0.0, 0.8));
  append(s, kWheelFreshScans, wheelSlot(Phase::kAnchorStatic, 0.0, 0.01));
  s.steps.push_back(wheelSlot(Phase::kEvent, std::numeric_limits<double>::quiet_NaN(), 0.01));
  append(s, kWheelFreshScans, wheelSlot(Phase::kAnchorStatic, 0.0, 0.01));
  s.steps.push_back(wheelSlot(Phase::kEvent, 0.0, -1000.0));
  return s;
}

// ---- F03: the real re-anchor gate ------------------------------------------------------------------------------

constexpr int kVerifyingScans = 54;  // past the 5 s verification timeout at 10 Hz

// One blind scan, then a verification no scan can pass, on the real gate with its default 5 s timeout.
Scenario verificationTimesOut(double total)
{
  Scenario s = inputScenario(total > 0.0 ? "gate_lost_policy_on" : "gate_lost_policy_off");
  s.typed = true;
  setParam(s, rclcpp::Parameter("reanchor_en", true));
  setParam(s, rclcpp::Parameter("reanchor_on_fail", "reset_map"));
  setParam(s, rclcpp::Parameter("reanchor_min_eff", 100000000));
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  setPolicy(s, total, 1.0);
  append(s, 5, slot(Phase::kBefore));
  append(s, 1, slot(Phase::kEvent, Imu::kStatic, Guard::kDegraded));
  append(s, kVerifyingScans, slot(Phase::kAfter));
  append(s, 5, slot(Phase::kLater));
  return s;
}

// ---- F05: the parked prior and a reset later in the same scan --------------------------------------------------

Scenario parkedPriorThenLateReset(bool zupt)
{
  const FieldCase & c = kFieldCases[0];
  Scenario s = baseParams(zupt ? "prior_late_reset_zupt_on" : "prior_late_reset_zupt_off", false);
  setParam(s, rclcpp::Parameter("zupt_en", zupt));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_scans", 3));
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 4; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  const std::array<double, 3> nudged = nudgedAnchor(c);
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kHold, Imu::kStatic, Guard::kHealthy, nudged));
  s.steps.push_back(step(Phase::kHoldReset, Imu::kStatic, Guard::kHealthy, nudged, true));
  for (int i = 0; i < 2; ++i)
    s.steps.push_back(step(Phase::kAfterReset, Imu::kStatic, Guard::kHealthy, nudged));
  return s;
}

// ---- The reference: prior map installs, a refused install, a reset that keeps the prior ------------------------

std::string priorMapPath()
{
  return std::string(FASTLIO_ANCHOR_NODE_SCRATCH) + "/test_anchor_lifecycle_node_prior.pcd";
}

// The plane the fixture's scans see, a little wider, as a prior map in its own frame.
bool writePriorMap(const std::string & path)
{
  pcl::PointCloud<pcl::PointXYZINormal> cloud;
  for (int ix = 0; ix < 70; ++ix) {
    for (int iy = 0; iy < 30; ++iy) {
      pcl::PointXYZINormal point;
      point.x = 8.0f + 0.1f * static_cast<float>(ix);
      point.y = -3.0f + 0.1f * static_cast<float>(iy);
      point.z = 1.0f;
      point.intensity = 1.0f;
      cloud.push_back(point);
    }
  }
  return pcl::io::savePCDFileBinary(path, cloud) == 0;
}

constexpr int kReferenceSettle = 4;  // parked scans after each event: enough to commit the next anchor

struct ReferenceEvent
{
  const char * name;
  int step;      // the scan the event is on
  bool changes;  // whether the reference generation moves on
  const char * reason;
};

// Parked on healthy scans throughout. In turn: the first alignment installs the prior map, the same alignment
// again, one that cannot be represented, a different one, and a runaway map reset with the prior installed.
Scenario referenceChanges(std::vector<ReferenceEvent> * events)
{
  Scenario s = inputScenario("reference_changes");
  s.prior_map = true;
  s.relocalizations = {{{0.0, 0.0, 0.0}}, {{1.0e12, 0.0, 0.0}}, {{0.5, 0.0, 0.0}}};
  setParam(s, rclcpp::Parameter("prior_map_pcd", priorMapPath()));
  setParam(s, rclcpp::Parameter("prior_map_align", "relocalization"));
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  append(s, 5, slot(Phase::kBefore));
  const auto event = [&](const char * name, int relocalization, bool trip, bool changes, const char * reason) {
    Step e = slot(Phase::kEvent);
    e.relocalization = relocalization;
    e.trip = trip;
    events->push_back({name, static_cast<int>(s.steps.size()), changes, reason});
    s.steps.push_back(e);
    append(s, kReferenceSettle, slot(Phase::kAfter));
  };
  event("first install", 0, false, true, "reference_changed");
  event("same alignment again", 0, false, false, "");
  event("alignment that cannot be installed", 1, false, false, "");
  event("new alignment", 2, false, true, "reference_changed");
  event("map reset with the prior installed", -1, true, true, "map_reset");
  return s;
}

// The same scenario with the per-scan trace switched.
Scenario traced(Scenario s, bool trace)
{
  s.trace = trace;
  setParam(s, rclcpp::Parameter("zupt_anchor_trace_en", trace));
  return s;
}

// The scans from the first one of `phase` on.
std::vector<const Observation *> fromPhase(const ChildResult & r, Phase phase)
{
  std::vector<const Observation *> out;
  bool reached = false;
  for (const auto & o : r.observations) {
    reached = reached || o.phase == static_cast<int>(phase);
    if (reached)
      out.push_back(&o);
  }
  return out;
}

// Degraded and parked without a usable anchor, wherever the scan's own pose is: the velocity alone was zeroed and
// the scan reports Degraded without StaticHold.
bool zeroOnlyScan(const Observation & o)
{
  const auto & r = o.record;
  return r.action == fast_lio::AnchorTraceAction::kZeroVelocity && r.zero_only && !r.usable && r.anchor_used == 0u &&
         !r.static_hold && o.odom_seen && odomFlags(o) == kFlagDegraded;
}

// The last scan before `scan` that was published and counted as supported: where a recovery's clock starts.
const Observation * lastSupportedBefore(const ChildResult & r, const Observation & scan)
{
  const Observation * last = nullptr;
  for (const auto & o : r.observations) {
    if (&o == &scan)
      break;
    if (o.record.published && o.record.supported)
      last = &o;
  }
  return last;
}

// The first scan of `scans`, from index `first` on, that ends at least `stable_ns` after scans[first] begins: the
// scan an unbroken matching run starting there is complete on.
const Observation *
completesRun(const std::vector<const Observation *> & scans, std::size_t first, std::int64_t stable_ns)
{
  for (std::size_t i = first; i < scans.size(); ++i)
    if (scans[i]->record.end_ns - scans[first]->record.beg_ns >= stable_ns)
      return scans[i];
  return nullptr;
}

unsigned typedValue(fast_lio::FrontendState state)
{
  return static_cast<unsigned>(state);
}

// Every trace record of one run, in the order the node logged them, as a file among this package's test results:
// a sample of the format for the tools that read it.
bool writeTraceSample(const ChildResult & r, const std::string & scenario)
{
  const std::string dir = FASTLIO_ANCHOR_NODE_RESULTS;
  for (std::size_t slash = dir.find('/', 1); slash != std::string::npos; slash = dir.find('/', slash + 1))
    ::mkdir(dir.substr(0, slash).c_str(), 0755);
  ::mkdir(dir.c_str(), 0755);
  std::ofstream out(dir + "/test_anchor_lifecycle_node." + scenario + ".trace-v" +
                    std::to_string(fast_lio::kAnchorTraceVersion) + ".log");
  for (const auto & line : r.lines)
    out << line << '\n';
  out.flush();
  return out.good();
}

// ---- A map reset whose rebuild takes no point ------------------------------------------------------------------

// A finite position whose map points the map backend cannot represent: at filter_size_map 0.05 their voxel indices
// pass the int range beyond about 1.07e8 m, so the backend adds none of them.
constexpr std::array<double, 3> kBeyondTheMap = {2.0e8, 0.0, 0.0};
constexpr int kRebuildBefore = 6;  // healthy parked scans: a valid anchor by the last one
constexpr int kRebuildAfter = 8;

// Parked on healthy scans with a valid anchor when the runaway watchdog's map reset runs on a scan that
// registration put beyond what the map backend can represent: the rebuild around that pose takes no point. With a
// prior map the prior is kept, and the first scan after the reset is registered back onto it (the fixture's
// stand-in for that registration); without one nothing is left to register against. `policy` switches the platform
// policy (a 10 s budget) on.
Scenario resetRebuildsNoMap(bool prior, bool policy)
{
  Scenario s = traced(inputScenario(std::string(prior ? "rebuild_kept_prior" : "rebuild_no_prior") +
                                    (policy ? "_policy_on" : "_policy_off")),
                      true);
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  setPolicy(s, policy ? 10.0 : 0.0, 1.0);
  append(s, kRebuildBefore, slot(Phase::kBefore));
  if (prior) {
    s.prior_map = true;
    s.relocalizations = {{{0.0, 0.0, 0.0}}};
    setParam(s, rclcpp::Parameter("prior_map_pcd", priorMapPath()));
    setParam(s, rclcpp::Parameter("prior_map_align", "relocalization"));
    s.steps.front().relocalization = 0;  // the prior map is installed on the first scan
  }
  Step reset = slot(Phase::kEvent);
  reset.inject = true;
  reset.pos = kBeyondTheMap;
  reset.trip = true;
  s.steps.push_back(reset);
  Step back = slot(Phase::kAfter);
  back.inject = true;  // at the origin, where the prior map is
  s.steps.push_back(back);
  append(s, kRebuildAfter - 1, slot(Phase::kAfter));
  return s;
}

// ---- The runaway watchdog's hold action ------------------------------------------------------------------------

// A late trip with the watchdog's hold action in place of the map reset, ZUPT on or off.
Scenario withHold(Scenario s, const std::string & name, bool zupt)
{
  s.name = name;
  setParam(s, rclcpp::Parameter("runaway_on_trip", "hold"));
  setParam(s, rclcpp::Parameter("zupt_en", zupt));
  return s;
}

// Degraded parked scans held on the anchor, then the trip on the next one (with ZUPT off nothing is pinned).
Scenario pinThenLateHold(bool zupt)
{
  return withHold(pinThenLateReset(), zupt ? "pin_late_hold_zupt_on" : "pin_late_hold_zupt_off", zupt);
}

// Healthy parked scans the parked prior pulls toward the anchor, then the trip on the next one.
Scenario parkedPriorThenLateHold(bool zupt)
{
  return withHold(parkedPriorThenLateReset(zupt), zupt ? "prior_late_hold_zupt_on" : "prior_late_hold_zupt_off", zupt);
}

// The pose covariance the fixture gives a scan before its anchor step, as /Odometry encodes it.
std::array<double, 36> fixtureOdomCovariance(bool degraded)
{
  Eigen::Matrix<double, 6, 6> pose = Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < 6; ++i)
    pose(i, i) = kFixtureCovDiag[static_cast<std::size_t>(i)];
  return fast_lio::makeOdomCovariance(pose, degraded);
}

// What a scan whose anchor write was taken back, or never made, publishes: the pose it brought, with its attitude
// and pose covariance, and `velocity`. `wrote` says whether the anchor step wrote the state: only then does the
// record carry a pose covariance, which must be the one from before the step. /Odometry carries the covariance
// the fixture gave the scan either way, as a scan reported `degraded` encodes it.
void expectTheScanOwnState(const Observation & o,
                           const std::array<double, 3> & position,
                           const std::array<double, 3> & velocity,
                           bool wrote,
                           bool degraded)
{
  const auto & x = o.record;
  EXPECT_TRUE(samePosition(x.pre_p, position) && samePosition(x.post_p, position))
    << "pre " << formatPosition(x.pre_p) << " post " << formatPosition(x.post_p);
  for (int i = 0; i < 4; ++i)
    EXPECT_TRUE(sameBits(x.pre_q[i], x.post_q[i])) << "attitude component " << i;
  for (int i = 0; i < 3; ++i)
    EXPECT_TRUE(sameBits(x.post_v[i], velocity[static_cast<std::size_t>(i)]))
      << "velocity component " << i << " is " << x.post_v[i];
  if (wrote)
    EXPECT_TRUE(x.have_cov && samePoseCovariance(x)) << "the anchor write's pose covariance outlived its take-back";
  else
    EXPECT_FALSE(x.have_cov) << "the record carries a pose covariance, as if the anchor step had written the state";
  EXPECT_TRUE(o.odom_seen && samePosition(o.odom_p, position)) << formatPosition(o.odom_p);
  const std::array<double, 36> own = fixtureOdomCovariance(degraded);
  EXPECT_TRUE(o.odom_seen && sameArray(o.odom_cov, own.data(), own.size()))
    << "the published pose covariance is not the scan's own";
}

// ---- K6-R9: what a scan reports while the recovery is pending --------------------------------------------------

constexpr double kExportBudget = 10.0;        // s, the platform's budget: the matching run ends Pending well within it
constexpr double kExportShortBudget = 1.2;    // s: above the stable 1.0 s, it runs out before the matching run ends
constexpr int kExportRun = 11;                // healthy parked scans of a run that spans the stable 1.0 s
constexpr double kDegradedInflation = 100.0;  // m2 the Degraded verdict adds to each wire translation variance

// One healthy moving scan, the last supported one; degraded parked scans with no anchor to hold to, which start
// Pending; healthy parked scans, the matching run; healthy parked scans that commit a new anchor; degraded parked
// scans that may hold on it. With kExportBudget the run ends Pending on its kExportRun-th scan; with
// kExportShortBudget the budget runs out first and latches Unanchored. The typed observation is on.
Scenario pendingExport(double total)
{
  Scenario s = inputScenario(total < kExportBudget ? "pending_export_runs_out" : "pending_export");
  s.typed = true;
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  setPolicy(s, total, 1.0);
  append(s, 1, slot(Phase::kBefore, Imu::kMoving));
  append(s, 3, slot(Phase::kEvent, Imu::kStatic, Guard::kDegraded));
  append(s, kExportRun + 1, slot(Phase::kAfter));
  append(s, 5, slot(Phase::kLater));
  append(s, 2, slot(Phase::kHold, Imu::kStatic, Guard::kDegraded));
  return s;
}

// The fixture's own translation variances are far below the inflation, so the two never overlap.
bool degradedInflation(const Observation & o)
{
  return o.odom_cov[21] >= kDegradedInflation && o.odom_cov[28] >= kDegradedInflation &&
         o.odom_cov[35] >= kDegradedInflation;
}

bool noDegradedInflation(const Observation & o)
{
  return o.odom_cov[21] < kDegradedInflation && o.odom_cov[28] < kDegradedInflation &&
         o.odom_cov[35] < kDegradedInflation;
}

// What a published scan reports (K6-R9): the Degraded flag, the pose covariance's inflation and the trace's export
// field carry the scan's own verdict, the re-anchor gate's `degraded`, or Unanchored; only a LocalQualified scan
// may carry StaticHold. These scenarios declare nothing lost.
void expectTheScanOwnVerdict(const Observation & o)
{
  const auto & x = o.record;
  ASSERT_TRUE(x.published && o.odom_seen) << "step " << o.step;
  const bool degraded = x.degraded || x.trust_post == fast_lio::RecoveryTrust::kUnanchored;
  EXPECT_EQ(x.eff_degraded, degraded) << "step " << o.step;
  EXPECT_EQ(odomFlags(o) & (kFlagDegraded | kFlagLost), degraded ? kFlagDegraded : 0u) << "step " << o.step;
  EXPECT_TRUE(degraded ? degradedInflation(o) : noDegradedInflation(o))
    << "step " << o.step << ": translation variances " << o.odom_cov[21] << ' ' << o.odom_cov[28] << ' '
    << o.odom_cov[35];
  if (x.trust_post != fast_lio::RecoveryTrust::kLocalQualified) {
    EXPECT_FALSE(x.static_hold) << "step " << o.step;
    EXPECT_EQ(odomFlags(o) & kFlagStaticHold, 0u) << "step " << o.step;
  }
}

// The typed observation of the same scan: the original verdict on the same facts. With the re-anchor gate off and
// nothing else forced, a scan reported Degraded is Unknown with no axes, any other one is Full.
void expectTheTypedOwnVerdict(const Observation & o)
{
  const auto & x = o.record;
  ASSERT_TRUE(o.typed_seen) << "step " << o.step;
  const bool degraded = x.degraded || x.trust_post == fast_lio::RecoveryTrust::kUnanchored;
  EXPECT_EQ(o.typed_health & fast_lio::kFrontendHealthDegraded, degraded ? fast_lio::kFrontendHealthDegraded : 0u)
    << "step " << o.step;
  EXPECT_EQ(o.typed_state,
            typedValue(degraded ? fast_lio::FrontendState::kDegraded : fast_lio::FrontendState::kTracking))
    << "step " << o.step;
  EXPECT_EQ(o.typed_validity,
            static_cast<unsigned>(degraded ? fast_lio::FrontendValidity::kUnknown : fast_lio::FrontendValidity::kFull))
    << "step " << o.step;
  EXPECT_EQ(o.typed_axes, degraded ? 0u : static_cast<unsigned>(fast_lio::kFrontendAllAxes)) << "step " << o.step;
  if (x.trust_post != fast_lio::RecoveryTrust::kLocalQualified) {
    EXPECT_EQ(o.typed_health & fast_lio::kFrontendHealthStaticHold, 0u) << "step " << o.step;
    EXPECT_NE(o.typed_substate, static_cast<unsigned>(fast_lio::FrontendSubstate::kStaticHold)) << "step " << o.step;
  }
}

// ---- A valid anchor across scans that took their IMU window and left before the anchor step --------------------

constexpr std::int64_t kScanGapBoundNs = 300000000;  // zupt_anchor_max_scan_gap_s: a longer gap is a scan hole
constexpr int kExitAfterScans = 3;                   // healthy parked scans after the exits: a new anchor's three

// Where registration puts the platform once it has moved on from the anchor of the first frozen case: 1 m away.
std::array<double, 3> movedOnFromTheAnchor(const FieldCase & c)
{
  return {c.old_anchor[0] + 0.8, c.old_anchor[1] - 0.6, c.old_anchor[2]};
}

// Healthy scans take an anchor at the first frozen case's old anchor: one moving, then three parked, the anchor valid
// on the last. Then `exits` scans whose three-point clouds the filter takes, IMU window and all, and which then leave
// before the anchor step for too few points: the first with `first`'s IMU, the others parked. Then one degraded
// parked scan, which registration puts at `parked`, and `after` healthy parked scans there. Every scan lasts its
// whole period, so the scan after the exits begins `exits` periods after the end of the last scan whose evidence an
// anchor step evaluated: within the scan-gap bound up to three exits. The trace, the typed observation and the
// parked prior are on; the prior's streak is engaged when the exits come.
Scenario consumedExits(const std::string & name, int exits, Imu first, const std::array<double, 3> & parked, int after)
{
  const FieldCase & c = kFieldCases[0];
  Scenario s = traced(inputScenario(name), true);
  s.scan_length = kScanPeriod;
  s.typed = true;
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_scans", 3));
  s.steps.push_back(step(Phase::kAnchorMove, Imu::kMoving, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < 3; ++i)
    s.steps.push_back(step(Phase::kAnchorStatic, Imu::kStatic, Guard::kHealthy, c.old_anchor));
  for (int i = 0; i < exits; ++i)
    s.steps.push_back(slot(Phase::kEvent, i == 0 ? first : Imu::kStatic, Guard::kHealthy, Cloud::kSparse));
  s.steps.push_back(step(Phase::kStatic, Imu::kStatic, Guard::kDegraded, parked));
  for (int i = 0; i < after; ++i)
    s.steps.push_back(step(Phase::kAfter, Imu::kStatic, Guard::kHealthy, parked));
  return s;
}

// One parked exit with the wheel required: a fresh wheel sample before every scan, zero speed except before the
// exit, where the bridge says it cannot measure (NaN).
Scenario consumedExitWithoutWheelEvidence()
{
  Scenario s = consumedExits(
    "consumed_exit_no_wheel_evidence", 1, Imu::kStatic, movedOnFromTheAnchor(kFieldCases[0]), kExitAfterScans);
  setParam(s, rclcpp::Parameter("zupt_require_wheel", true));
  setParam(s, rclcpp::Parameter("wheel_topic", kWheelTopic));
  for (auto & st : s.steps) {
    st.wheel = true;
    st.wheel_speed = st.phase == Phase::kEvent ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    st.wheel_age = 0.01;
  }
  return s;
}

// After the anchor, a slot that sends the cloud of the scan before it once more while the IMU of its period shows
// motion; then one parked exit, the degraded parked scan and healthy parked scans as above.
Scenario repeatBeforeAConsumedExit()
{
  Scenario s = consumedExits(
    "repeat_before_consumed_exit", 1, Imu::kStatic, movedOnFromTheAnchor(kFieldCases[0]), kExitAfterScans);
  Step repeat = slot(Phase::kEvent, Imu::kMoving);
  repeat.repeat = 1;
  s.steps.insert(s.steps.begin() + 4, repeat);  // after the moving scan and the three parked ones
  return s;
}

// Each of `exits` took its IMU window through the real filter and left before the anchor step: a committed,
// processed batch that advanced the filter's input to the scan's end; no anchor step, no output, no write. The i-th
// began `first_gap_ns` + i periods after the end of the last evidence an anchor step evaluated. Checking the window
// for the anchor's sake is no anchor-step observation: the scan carries no evidence of its own, nor its events.
void expectConsumedExits(const std::vector<const Observation *> & exits,
                         std::int64_t first_gap_ns,
                         const std::string & name)
{
  for (std::size_t i = 0; i < exits.size(); ++i) {
    const Observation & o = *exits[i];
    const auto & x = o.record;
    const std::string at = name + " exit " + std::to_string(i + 1);
    EXPECT_EQ(x.outcome, fast_lio::AnchorScanOutcome::kBeforeAnchorStep) << at;
    EXPECT_FALSE(x.anchor_step || x.published) << at;
    EXPECT_EQ(o.imu_batch, 1) << at << ": no IMU batch was settled";
    EXPECT_EQ(o.imu_disposition, static_cast<int>(fast_lio::ImuDisposition::kCommitted)) << at;
    EXPECT_EQ(o.imu_reason, static_cast<int>(fast_lio::ImuProcessReason::kProcessed)) << at;
    EXPECT_GT(o.imu_samples, 0u) << at;
    EXPECT_EQ(fast_lio::traceNs(o.filter_end), x.end_ns) << at << ": the filter's input did not advance to the end";
    EXPECT_EQ(x.gap_ns, first_gap_ns + static_cast<std::int64_t>(i) * kScanPeriodNs) << at;
    EXPECT_FALSE(hasEvent(o, fast_lio::kAnchorEventScanHole)) << at;
    EXPECT_FALSE(hasEvent(o, fast_lio::kAnchorEventMotion) || hasEvent(o, fast_lio::kAnchorEventEvidenceUnknown)) << at;
    EXPECT_EQ(x.body, fast_lio::AnchorTraceBody::kNa) << at;
    EXPECT_EQ(x.imu_n, 0) << at;
    EXPECT_EQ(x.action, fast_lio::AnchorTraceAction::kNa) << at;
    EXPECT_FALSE(x.usable || x.zero_only || x.parked_applied || x.have_act || x.have_target) << at;
  }
}

// The counter summary counts every anchor that ended, in all and by reason, as the records do; no anchor ended as
// scan_not_evaluated, a reason that only ever ends a candidate.
void expectInvalidationCounters(const ChildResult & r, const std::string & name)
{
  const auto ended = linesWhere(r, "[ZUPT-ANCHOR]", "event", "invalidated");
  EXPECT_EQ(counterValue(r, "invalidated"), std::to_string(ended.size())) << name;
  std::map<std::string, std::size_t> by_reason;
  for (const auto & line : ended) {
    ++by_reason[valueOf(line, "reason")];
    EXPECT_NE(valueOf(line, "reason"), "scan_not_evaluated") << name << ": " << line;
  }
  for (const char * reason : {"motion", "evidence_unknown", "scan_hole"}) {
    EXPECT_EQ(counterValue(r, std::string("invalidated_") + reason), std::to_string(by_reason[reason]))
      << name << ": invalidated_" << reason;
  }
}

// The degraded parked scan after the exits, within the scan-gap bound, has no anchor left to hold to: it zeroes its
// velocity alone at `parked`, the pose registration gave it, and reports Degraded without StaticHold in /Odometry
// and the typed observation. From the exit `ended_on` on, no scan holds an anchor until the healthy parked scans
// make a new one, with a new ID, on the usual three; nothing takes a pose from the anchor `old_id` again. It ended on
// that exit, once, as `reason`, and released the parked prior it had engaged.
void expectConsumedExitEndedTheAnchor(const ChildResult & r,
                                      std::uint64_t old_id,
                                      const Observation & ended_on,
                                      const std::string & reason,
                                      const std::array<double, 3> & parked,
                                      const std::string & name)
{
  const auto degraded = r.phase(Phase::kStatic);
  ASSERT_EQ(degraded.size(), 1u) << name;
  const Observation & d = *degraded.front();
  const auto & x = d.record;
  EXPECT_EQ(x.anchor_used, 0u) << name << ": the degraded parked scan took its pose from the ended anchor " << old_id;
  EXPECT_NE(x.action, fast_lio::AnchorTraceAction::kPin) << name << ": the degraded parked scan was pinned";
  EXPECT_TRUE(x.have_act && sameArray(x.act_p, x.pre_p, 3) && sameArray(x.act_q, x.pre_q, 4))
    << name << ": the degraded parked scan moved the pose from " << formatPosition(x.pre_p) << " to "
    << formatPosition(x.act_p);
  std::string why;
  EXPECT_TRUE(zeroedVelocityOnly(d, parked, &why)) << name << ": " << why;
  EXPECT_EQ(x.anchor_state, fast_lio::AnchorState::kEmpty) << name;
  EXPECT_FALSE(x.parked_applied || x.have_target) << name;
  EXPECT_TRUE(allZero(x.act_v)) << name;
  EXPECT_TRUE(sameArray(x.p_pose_act, x.p_pose_pre, 36)) << name << ": the pose covariance changed";
  EXPECT_LE(x.gap_ns, kScanGapBoundNs) << name;
  EXPECT_FALSE(hasEvent(d, fast_lio::kAnchorEventScanHole)) << name;
  EXPECT_EQ(x.flags, static_cast<int>(kFlagDegraded)) << name;
  EXPECT_TRUE(d.typed_seen) << name;
  EXPECT_EQ(d.typed_state, typedValue(fast_lio::FrontendState::kDegraded)) << name;
  EXPECT_EQ(d.typed_health & (fast_lio::kFrontendHealthDegraded | fast_lio::kFrontendHealthStaticHold),
            fast_lio::kFrontendHealthDegraded)
    << name;

  for (const auto * o : fromPhase(r, Phase::kEvent)) {
    if (o->phase != static_cast<int>(Phase::kEvent) || o->record.scan_seq < ended_on.record.scan_seq)
      continue;
    EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kEmpty) << name << " step " << o->step;
    EXPECT_EQ(o->record.anchor_id, old_id) << name << " step " << o->step << ": an exit started an anchor";
  }
  expectNeverUsed(fromPhase(r, Phase::kEvent), old_id, name);
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), static_cast<std::size_t>(kExitAfterScans)) << name;
  const std::uint64_t new_id = after.front()->record.anchor_id;
  EXPECT_GT(new_id, old_id) << name;
  for (std::size_t i = 0; i < after.size(); ++i) {
    EXPECT_EQ(after[i]->record.anchor_id, new_id) << name << " step " << after[i]->step;
    EXPECT_EQ(after[i]->record.anchor_state,
              i + 1 < after.size() ? fast_lio::AnchorState::kQualifying : fast_lio::AnchorState::kValid)
      << name << " step " << after[i]->step;
  }
  EXPECT_EQ(counterValue(r, "pin_applied"), "0") << name;

  const auto ended = anchorLines(r, "invalidated", old_id);
  EXPECT_EQ(ended.size(), 1u) << name << ": records ending anchor " << old_id;
  for (const auto & line : ended) {
    EXPECT_EQ(valueOf(line, "scan_seq"), std::to_string(ended_on.record.scan_seq)) << name << ": " << line;
    EXPECT_EQ(valueOf(line, "stamp_ns"), std::to_string(ended_on.record.end_ns)) << name << ": " << line;
  }
  expectInvalidatedOnce(r, old_id, reason, name);
  const auto released = anchorLines(r, "parked_prior_released", old_id);
  EXPECT_EQ(released.size(), 1u) << name << ": records releasing the parked prior on anchor " << old_id;
  for (const auto & line : released) {
    EXPECT_EQ(valueOf(line, "reason"), reason) << name << ": " << line;
    EXPECT_EQ(valueOf(line, "scan_seq"), std::to_string(ended_on.record.scan_seq)) << name << ": " << line;
  }
  expectInvalidationCounters(r, name);
  expectTraceAccountsForEveryScan(r, true, name);
}

// Parked windows show nothing that ends the anchor `old_id`: every exit leaves it valid, with its ID and its pose
// `anchor`; the degraded parked scan after them is pinned to it and reports StaticHold in /Odometry and the typed
// observation; the healthy parked scans after that still have it; it never ends.
void expectConsumedExitsKeptTheAnchor(const ChildResult & r,
                                      std::uint64_t old_id,
                                      const std::array<double, 3> & anchor,
                                      const std::string & name)
{
  for (const auto * o : r.phase(Phase::kEvent)) {
    const auto & x = o->record;
    EXPECT_EQ(x.anchor_state, fast_lio::AnchorState::kValid) << name << " step " << o->step;
    EXPECT_EQ(x.anchor_id, old_id) << name << " step " << o->step;
    EXPECT_TRUE(x.have_anchor_pose && samePosition(x.anchor_p, anchor)) << name << " step " << o->step;
  }
  EXPECT_TRUE(anchorLines(r, "invalidated", old_id).empty()) << name;
  const auto degraded = r.phase(Phase::kStatic);
  ASSERT_EQ(degraded.size(), 1u) << name;
  const Observation & d = *degraded.front();
  const auto & x = d.record;
  EXPECT_EQ(x.action, fast_lio::AnchorTraceAction::kPin) << name;
  EXPECT_EQ(x.anchor_used, old_id) << name;
  EXPECT_TRUE(x.usable && x.static_hold) << name;
  EXPECT_TRUE(samePosition(x.post_p, anchor)) << name << ": " << formatPosition(x.post_p);
  EXPECT_TRUE(d.odom_seen && samePosition(d.odom_p, anchor)) << name << ": " << formatPosition(d.odom_p);
  EXPECT_EQ(odomFlags(d), kFlagDegraded | kFlagStaticHold) << name;
  EXPECT_LE(x.gap_ns, kScanGapBoundNs) << name;
  EXPECT_FALSE(hasEvent(d, fast_lio::kAnchorEventScanHole)) << name;
  ASSERT_TRUE(d.typed_seen) << name;
  const std::uint32_t held = fast_lio::kFrontendHealthDegraded | fast_lio::kFrontendHealthStaticHold;
  EXPECT_EQ(d.typed_health & held, held) << name;
  for (const auto * o : r.phase(Phase::kAfter)) {
    EXPECT_EQ(o->record.anchor_id, old_id) << name << " step " << o->step;
    EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kValid) << name << " step " << o->step;
  }
  EXPECT_EQ(counterValue(r, "pin_applied"), "1") << name;
  expectInvalidationCounters(r, name);
  expectTraceAccountsForEveryScan(r, true, name);
}

bool consumedExitScenario(const Scenario & s)
{
  return s.name.rfind("consumed_exit_", 0) == 0;
}

// What a consumed-exit run published and wrote, read without the fields only the trace fills, so that it holds with
// the trace off as well: with motion in the first exit's window the anchor ends there and the degraded parked scan
// keeps the pose registration gave it; with parked windows the anchor stays and that scan is pinned to it.
void expectConsumedExitOutputs(const ChildResult & r, const Scenario & s, const std::string & name)
{
  const auto event =
    std::find_if(s.steps.begin(), s.steps.end(), [](const Step & st) { return st.phase == Phase::kEvent; });
  const auto parked =
    std::find_if(s.steps.begin(), s.steps.end(), [](const Step & st) { return st.phase == Phase::kStatic; });
  ASSERT_TRUE(event != s.steps.end() && parked != s.steps.end()) << name;
  const bool moved = event->imu == Imu::kMoving;
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
  ASSERT_NE(old_id, 0u) << name;
  for (const auto * o : r.phase(Phase::kEvent)) {
    EXPECT_EQ(o->record.anchor_state, moved ? fast_lio::AnchorState::kEmpty : fast_lio::AnchorState::kValid)
      << name << " step " << o->step;
    EXPECT_EQ(o->record.anchor_id, old_id) << name << " step " << o->step;
  }
  const auto degraded = r.phase(Phase::kStatic);
  ASSERT_EQ(degraded.size(), 1u) << name;
  const Observation & d = *degraded.front();
  EXPECT_EQ(d.record.action, moved ? fast_lio::AnchorTraceAction::kZeroVelocity : fast_lio::AnchorTraceAction::kPin)
    << name;
  EXPECT_EQ(d.record.anchor_used, moved ? 0u : old_id) << name;
  EXPECT_EQ(d.record.static_hold, !moved) << name;
  EXPECT_TRUE(d.odom_seen && samePosition(d.odom_p, moved ? parked->pos : kFieldCases[0].old_anchor))
    << name << ": " << formatPosition(d.odom_p);
  EXPECT_EQ(odomFlags(d), moved ? kFlagDegraded : kFlagDegraded | kFlagStaticHold) << name;
  EXPECT_EQ(anchorLines(r, "invalidated", old_id).size(), moved ? 1u : 0u) << name;
}

// ---- The first local reference ---------------------------------------------------------------------------------

// The process's first scans arrive before any IMU sample: each one is dropped for an empty window once the first
// sample is in. The node keeps the five queued meanwhile (max_scan_backlog_sec 0.5 s).
constexpr int kStartupGapScans = 5;
constexpr double kStartupGapEnd = 0.40;  // s after the first scan's end: no IMU sample before then
constexpr double kStartupBudget = 10.0;  // s, the platform's reanchor_total_timeout_sec
constexpr std::int64_t kStartupBudgetNs = 10000000000;
constexpr double kStillTimeout = 90.0;  // s, the platform's imu_init_still_timeout_s
constexpr std::int64_t kStillTimeoutNs = 90000000000;
constexpr int kStillInitScans = 11;       // parked scans the sustained still gate completes on: 1.0 s of samples
constexpr int kColdPublished = 6;         // parked scans published after the map build, before the empty window
constexpr int kColdAfter = 14;            // parked scans after an empty window: a 1 s matching run and some to spare
constexpr int kMovingScans = 164;         // 16.4 s of motion while the IMU initialization waits for the platform
constexpr int kLongMovingScans = 467;     // 46.7 s
constexpr int kTimeoutMovingScans = 910;  // past the still timeout, then the map build and scans published after it

// The first kStartupGapScans slots: no IMU sample until kStartupGapEnd after the first scan's end.
void appendStartupGap(Scenario & s)
{
  Step first = slot(Phase::kGap);
  first.hole_from = -kScanPeriod;
  first.hole_to = kStartupGapEnd;
  s.steps.push_back(first);
  append(s, kStartupGapScans - 1, slot(Phase::kGap));
}

// A process started fresh, with no prior map, on the platform's recovery budget and IMU initialization, while the
// platform moves: its first scans come before any IMU sample, then the sustained still gate restarts its window on
// each of `moving` scans. With `parks` the platform then parks: the gate completes on kStillInitScans scans, the
// next scan builds the local map and the parked scans after it are published; a scan whose window holds one sample
// and one whose window holds none follow, then parked scans again. Without it the platform moves on through the
// still timeout, and the initialization completes from a moving window.
Scenario coldStart(int moving, bool parks, bool trace = true)
{
  Scenario s = traced(
    inputScenario("cold_start_moving_" + std::to_string(moving) + (parks ? "_then_parked" : "_through_the_timeout")),
    trace);
  s.warm_up = false;
  setPolicy(s, kStartupBudget, 1.0);
  setParam(s, rclcpp::Parameter("imu_init_require_still", true));
  setParam(s, rclcpp::Parameter("imu_init_still_sustained_en", true));
  setParam(s, rclcpp::Parameter("imu_init_still_tol", 0.03));
  setParam(s, rclcpp::Parameter("imu_init_still_timeout_s", kStillTimeout));
  appendStartupGap(s);
  append(s, moving, slot(Phase::kMove, Imu::kMoving));
  if (!parks)
    return s;
  s.typed = true;
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  append(s, kStillInitScans + 1 + kColdPublished, slot(Phase::kStatic));
  Step thin = slot(Phase::kEvent);
  thin.hole_from = -0.094;  // as in lostImu(LostImu::kEmpty)
  thin.hole_to = 0.101;
  s.steps.push_back(thin);
  s.steps.push_back(slot(Phase::kEvent));
  append(s, kColdAfter, slot(Phase::kAfter));
  return s;
}

// A process started fresh whose first normal map builds take no point. The IMU initialization completes on the
// second scan. The next three scans' builds take no point, the second of them on a window holding one sample, and
// the scan between the second and the third is dropped for an empty window. The scan after the third build takes
// the map, the next one is dropped for an empty window, and parked scans are published after it.
Scenario emptyFirstBuilds()
{
  Scenario s = traced(inputScenario("empty_first_builds"), true);
  s.warm_up = false;
  setPolicy(s, kStartupBudget, 1.0);
  append(s, 2, slot(Phase::kBefore));
  Step attempt = slot(Phase::kEvent);
  attempt.no_build_point = true;
  s.steps.push_back(attempt);
  Step thin = attempt;
  thin.hole_from = -0.094;
  thin.hole_to = 0.101;
  s.steps.push_back(thin);
  s.steps.push_back(slot(Phase::kEvent));
  s.steps.push_back(attempt);
  Step build = slot(Phase::kAfter);
  build.hole_from = 0.001;  // the next scan's window holds no sample
  build.hole_to = 0.101;
  s.steps.push_back(build);
  s.steps.push_back(slot(Phase::kAfter));
  append(s, kColdAfter, slot(Phase::kLater));
  return s;
}

// A process started with a prior map that needs no alignment: the same first scans before any IMU sample, then
// parked scans, the IMU initialization, the map build from the prior map and the scan, and scans published after it.
Scenario priorMapStart()
{
  Scenario s = traced(inputScenario("prior_map_start"), true);
  s.warm_up = false;
  setPolicy(s, kStartupBudget, 1.0);
  setParam(s, rclcpp::Parameter("prior_map_pcd", priorMapPath()));
  setParam(s, rclcpp::Parameter("prior_map_align", "as_delivered"));
  appendStartupGap(s);
  append(s, 3 + kColdAfter, slot(Phase::kAfter));
  return s;
}

// The first scan whose record reports the first reference built: the scan of the build.
const Observation * firstEstablished(const ChildResult & r)
{
  for (const auto & o : r.observations)
    if (o.record.startup_reference == fast_lio::StartupReference::kEstablished)
      return &o;
  return nullptr;
}

std::size_t indexOf(const ChildResult & r, const Observation & o)
{
  return static_cast<std::size_t>(&o - r.observations.data());
}

// describeRun without the middle of a long IMU initialization: the first scans, and the scans from shortly before
// the first reference on.
std::string describeStart(const ChildResult & r)
{
  constexpr std::size_t kHead = kStartupGapScans + 3;
  constexpr std::size_t kTail = 3;
  const Observation * build = firstEstablished(r);
  const std::size_t from = build == nullptr ? r.observations.size() : indexOf(r, *build);
  ChildResult brief = r;
  brief.observations.clear();
  for (std::size_t i = 0; i < r.observations.size(); ++i)
    if (i < kHead || i + kTail >= from)
      brief.observations.push_back(r.observations[i]);
  return describeRun(brief);
}

// Cold on every scan before `build` and established from it on: the phase moved forward once, at the build.
void expectColdUntil(const ChildResult & r, const Observation & build)
{
  for (std::size_t i = 0; i < r.observations.size(); ++i) {
    const auto & x = r.observations[i].record;
    EXPECT_EQ(x.startup_reference,
              i < indexOf(r, build) ? fast_lio::StartupReference::kCold : fast_lio::StartupReference::kEstablished)
      << "scan " << x.scan_seq;
  }
}

// Up to and including the scan of `build`, nothing of the recovery moved: no episode, no clock, the reference still
// trusted as far as the local checks go, and no pose published. Every recovery record comes after that scan.
void expectNothingStartedBefore(const ChildResult & r, const Observation & build)
{
  for (std::size_t i = 0; i <= indexOf(r, build); ++i) {
    const auto & x = r.observations[i].record;
    EXPECT_FALSE(x.published) << "scan " << x.scan_seq;
    EXPECT_EQ(x.trust_pre, fast_lio::RecoveryTrust::kLocalQualified) << "scan " << x.scan_seq;
    EXPECT_EQ(x.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "scan " << x.scan_seq;
    EXPECT_EQ(x.recovery_episode, 0u) << "scan " << x.scan_seq;
    EXPECT_EQ(x.recovery_start_ns, -1) << "scan " << x.scan_seq;
    EXPECT_EQ(x.recovery_deadline_ns, -1) << "scan " << x.scan_seq;
  }
  for (const auto & line : linesTagged(r, "[FE-RECOVERY]"))
    EXPECT_GT(toI64(valueOf(line, "scan_seq")), static_cast<std::int64_t>(build.record.scan_seq)) << line;
}

// The scan of the first reference: the normal initial build returned with points in the map. It returned before the
// anchor step, published nothing and is no supported scan; no anchor or candidate exists.
void expectTheBuildScan(const Observation & build)
{
  const auto & b = build.record;
  EXPECT_EQ(b.outcome, fast_lio::AnchorScanOutcome::kBeforeAnchorStep);
  EXPECT_TRUE(b.map_updated);
  EXPECT_FALSE(b.anchor_step);
  EXPECT_FALSE(b.published);
  EXPECT_FALSE(b.supported);
  EXPECT_EQ(b.anchor_state, fast_lio::AnchorState::kEmpty);
}

}  // namespace

// Parked again after a moving degraded stretch: the motion ended the anchor taken before it, so nothing pins to
// that anchor again. The parked scans zero the velocity alone, keep position, attitude and their covariance bit for
// bit, and report Degraded without StaticHold, in all three frozen cases.
TEST(AnchorLifecycleNode, F1F2F3DegradedParkedAgainZeroesTheVelocityOnly)
{
  int fixed = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainDegraded(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    ASSERT_TRUE(anchorWasValidAtTheOldPose(r, c)) << c.name;
    const auto moved = r.phase(Phase::kMove);
    ASSERT_EQ(moved.size(), 1u) << c.name;
    EXPECT_NE(moved.front()->record.events & fast_lio::kAnchorEventMotion, 0u) << c.name << ": motion not traced";
    EXPECT_EQ(moved.front()->record.anchor_state, fast_lio::AnchorState::kEmpty) << c.name;
    expectInvalidatedOnce(r, old_id, "motion", c.name);
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 3u) << c.name;
    bool all = true;
    for (const auto * o : parked) {
      std::string why;
      const bool zeroed = zeroedVelocityOnly(*o, c.pre_event, &why) &&
                          o->record.anchor_state == fast_lio::AnchorState::kEmpty && o->record.anchor_id == old_id;
      EXPECT_TRUE(zeroed) << c.name << " step " << o->step << ": " << why << " anchor_id " << o->record.anchor_id
                          << " (old " << old_id << ")";
      all = all && zeroed;
    }
    EXPECT_GT(distance(c.pre_event, c.old_anchor), 30.0) << c.name;
    EXPECT_EQ(counterValue(r, "zero_only_unanchored"), "3") << c.name;
    EXPECT_EQ(counterValue(r, "blocked_stale_use"), "3") << c.name;
    EXPECT_EQ(counterValue(r, "pin_applied"), "0") << c.name;
    fixed += all ? 1 : 0;
  }
  EXPECT_EQ(fixed, 3) << "fixed in " << fixed << "/3 frozen cases";
}

// The late reset: rebuild_local_map runs after the anchor step of the first parked-again scan. The motion already
// ended the old anchor; the reset moves the reference generation on, and no parked scan names the old anchor or
// claims StaticHold.
TEST(AnchorLifecycleNode, F1F2F3LateResetLeavesNoOldAnchorAndNoStaticHold)
{
  int fixed = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainWithLateReset(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    expectInvalidatedOnce(r, old_id, "motion", c.name);
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 3u) << c.name;
    const auto & reset = *parked.front();
    const std::uint32_t reset_events = fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventMapReset;
    EXPECT_EQ(reset.record.events & reset_events, reset_events) << c.name;
    const auto before = r.phase(Phase::kMoveDegraded);
    ASSERT_FALSE(before.empty()) << c.name;
    EXPECT_EQ(reset.record.ref_gen, before.back()->record.ref_gen + 1) << c.name;
    bool all = true;
    for (const auto * o : parked) {
      std::string why;
      const bool zeroed = zeroedVelocityOnly(*o, c.pre_event, &why);
      EXPECT_TRUE(zeroed) << c.name << " step " << o->step << ": " << why;
      all = all && zeroed;
    }
    EXPECT_TRUE(linesTagged(r, "[FE-RECOVERY]").empty()) << c.name << ": the platform policy is off by default";
    fixed += all ? 1 : 0;
  }
  EXPECT_EQ(fixed, 3);
}

// Parked again on healthy scans with the parked prior on: the motion ended the old anchor, so the prior never pulls
// toward it. The parked stretch takes a new anchor, with a new ID, at its own pose; StaticHold appears only once
// the prior is engaged on that anchor and the anchor is usable.
TEST(AnchorLifecycleNode, F1F2F3HealthyParkedAgainTakesANewAnchorAndIsNotPulledBack)
{
  int fixed = 0;
  for (const auto & c : kFieldCases) {
    const ChildResult r = runScenario(parkedAgainHealthy(c));
    ASSERT_TRUE(r.ok()) << c.name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << c.name;
    expectInvalidatedOnce(r, old_id, "motion", c.name);
    const auto parked = r.phase(Phase::kStatic);
    ASSERT_EQ(parked.size(), 5u) << c.name;
    const std::uint64_t new_id = parked.front()->record.anchor_id;
    EXPECT_GT(new_id, old_id) << c.name;
    bool all = new_id > old_id;
    for (std::size_t i = 0; i < parked.size(); ++i) {
      const Observation & o = *parked[i];
      // The candidate commits on its third scan; nothing uses it before the next one.
      const bool usable = i >= 3;
      const bool held = (odomFlags(o) & kFlagStaticHold) != 0u;
      const bool fresh = o.record.anchor_id == new_id && o.record.usable == usable &&
                         (!o.record.parked_applied || o.record.anchor_used == new_id) && o.odom_seen &&
                         samePosition(o.odom_p, c.pre_event) && (odomFlags(o) & kFlagDegraded) == 0u &&
                         held == usable && o.record.static_hold == held;
      EXPECT_TRUE(fresh) << c.name << " step " << o.step << ": anchor_id " << o.record.anchor_id << " (old " << old_id
                         << ", new " << new_id << ") usable " << o.record.usable << " anchor_used "
                         << o.record.anchor_used << " published " << formatPosition(o.odom_p) << " flags "
                         << odomFlags(o);
      all = all && fresh;
    }
    EXPECT_EQ(parked[2]->record.anchor_state, fast_lio::AnchorState::kValid) << c.name;
    fixed += all ? 1 : 0;
  }
  EXPECT_EQ(fixed, 3);
}

// An IMU coverage gap returns from the scan path before the anchor step, and ends the anchor right there: the
// parked scans after it zero the velocity alone, although the IMU saw no motion and the pose came back elsewhere.
TEST(AnchorLifecycleNode, CoverageGapBeforeTheAnchorStepEndsTheAnchor)
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
  EXPECT_EQ(gap.front()->record.anchor_state, fast_lio::AnchorState::kEmpty);
  expectInvalidatedOnce(r, old_id, "imu_coverage_gap", "coverage");
  const auto parked = r.phase(Phase::kStatic);
  ASSERT_EQ(parked.size(), 3u);
  for (const auto * o : parked) {
    std::string why;
    EXPECT_TRUE(zeroedVelocityOnly(*o, c.pre_event, &why)) << "step " << o->step << ": " << why;
  }
}

// A legitimate pin (same stationary stretch), then a runaway map reset after the anchor step of a pinned scan. The
// reset ends the anchor within that scan, so the scan gives the anchor's pose back: the map is rebuilt around, and
// /Odometry carries, what the scan would have had without an anchor: its own pose with the velocity zeroed and no
// StaticHold. The parked scans after it zero the velocity alone as well.
TEST(AnchorLifecycleNode, LateResetAfterAPinEndsTheAnchorAndItsStaticHold)
{
  const FieldCase & c = kFieldCases[0];
  const ChildResult r = runScenario(pinThenLateReset());
  ASSERT_TRUE(r.ok()) << r.describe();
  const std::uint64_t old_id = oldAnchorId(r);
  ASSERT_NE(old_id, 0u);
  const auto held = r.phase(Phase::kHold);
  ASSERT_EQ(held.size(), 2u);
  for (const auto * o : held) {
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin);
    EXPECT_EQ(o->record.anchor_used, old_id);
    EXPECT_TRUE(o->odom_seen && samePosition(o->odom_p, c.old_anchor)) << formatPosition(o->odom_p);
    EXPECT_EQ(odomFlags(*o), kFlagDegraded | kFlagStaticHold);
  }
  const auto reset = r.phase(Phase::kHoldReset);
  ASSERT_EQ(reset.size(), 1u);
  const auto & rr = reset.front()->record;
  EXPECT_NE(rr.events & fast_lio::kAnchorEventMapReset, 0u);
  EXPECT_TRUE(rr.usable) << "the anchor step ran, and pinned, before the reset";
  EXPECT_EQ(rr.action, fast_lio::AnchorTraceAction::kZeroVelocity);
  EXPECT_EQ(rr.anchor_used, 0u);
  {
    // usable stays as the anchor step left it; everything else is the zero-only scan.
    Observation as_zero_only = *reset.front();
    as_zero_only.record.usable = false;
    std::string why;
    EXPECT_TRUE(zeroedVelocityOnly(as_zero_only, nudgedAnchor(c), &why)) << "reset scan: " << why;
  }
  EXPECT_EQ(rr.anchor_state, fast_lio::AnchorState::kEmpty);
  EXPECT_FALSE(rr.zupt_active);
  EXPECT_FALSE(rr.static_hold);
  EXPECT_EQ(odomFlags(*reset.front()), kFlagDegraded);
  EXPECT_EQ(counterValue(r, "pin_applied"), "2") << "only the two scans whose published pose is the anchor's";
  EXPECT_EQ(counterValue(r, "zero_only_unanchored"), "3") << "the reset scan and the two after it";
  EXPECT_EQ(counterValue(r, "blocked_stale_use"), "3");
  expectInvalidatedOnce(r, old_id, "map_reset", "pin_late_reset");
  const auto after = r.phase(Phase::kAfterReset);
  ASSERT_EQ(after.size(), 2u);
  for (const auto * o : after) {
    std::string why;
    EXPECT_TRUE(zeroedVelocityOnly(*o, nudgedAnchor(c), &why)) << "step " << o->step << ": " << why;
  }
}

// A runaway map reset on a healthy parked scan, with the platform policy off and on. Off: the reset ends the
// anchor and moves the reference generation on, the reset scan starts no candidate, and the parked scans after it
// take a new anchor in the new generation while reporting what the registration says. On: the reset also latches
// Unanchored, so every scan from it on reports Degraded and no anchor is taken again.
TEST(AnchorLifecycleNode, ResetOnAParkedScanLatchesOnlyWithThePlatformPolicy)
{
  for (const double total : {0.0, 10.0}) {
    const bool policy = total > 0.0;
    const std::string name = policy ? "policy on" : "policy off";
    const ChildResult r = runScenario(resetWhileParkedHealthy(total));
    ASSERT_TRUE(r.ok()) << name << ": " << r.describe();
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u) << name;
    const auto reset = r.phase(Phase::kParkedReset);
    ASSERT_EQ(reset.size(), 1u) << name;
    const auto & rr = reset.front()->record;
    const std::uint32_t reset_events = fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventMapReset;
    EXPECT_EQ(rr.events & reset_events, reset_events) << name;
    EXPECT_TRUE(rr.usable) << name << ": the anchor step ran before the reset";
    EXPECT_EQ(rr.anchor_state, fast_lio::AnchorState::kEmpty) << name;
    EXPECT_EQ(rr.anchor_id, old_id) << name << ": the reset scan started a candidate";
    EXPECT_EQ(rr.ref_gen, r.phase(Phase::kAnchorStatic).back()->record.ref_gen + 1) << name;
    EXPECT_FALSE(rr.static_hold) << name;
    expectInvalidatedOnce(r, old_id, "map_reset", name);
    const auto after = r.phase(Phase::kParkedAfterReset);
    ASSERT_EQ(after.size(), 5u) << name;
    const auto recovery = linesTagged(r, "[FE-RECOVERY]");
    if (!policy) {
      EXPECT_TRUE(recovery.empty()) << name;
      EXPECT_EQ(rr.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << name;
      EXPECT_EQ(odomFlags(*reset.front()), 0u) << name;
      for (std::size_t i = 0; i < after.size(); ++i) {
        const auto & o = after[i]->record;
        EXPECT_EQ(o.anchor_id, old_id + 1) << name << " step " << after[i]->step;
        EXPECT_EQ(o.ref_gen, rr.ref_gen) << name << " step " << after[i]->step;
        EXPECT_EQ(o.anchor_state, i < 2 ? fast_lio::AnchorState::kQualifying : fast_lio::AnchorState::kValid)
          << name << " step " << after[i]->step;
        EXPECT_EQ(o.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << name;
        EXPECT_FALSE(o.eff_degraded) << name << " step " << after[i]->step;
        EXPECT_EQ(odomFlags(*after[i]), 0u) << name << " step " << after[i]->step;
      }
      EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "0") << name;
    } else {
      ASSERT_EQ(recovery.size(), 1u) << name;
      EXPECT_EQ(valueOf(recovery.front(), "event"), "unanchored") << recovery.front();
      EXPECT_EQ(valueOf(recovery.front(), "reason"), "map_reset") << recovery.front();
      EXPECT_EQ(valueOf(recovery.front(), "reset_count"), "1") << recovery.front();
      EXPECT_EQ(valueOf(recovery.front(), "scan_seq"), std::to_string(rr.scan_seq)) << recovery.front();
      EXPECT_EQ(rr.trust_post, fast_lio::RecoveryTrust::kUnanchored) << name;
      EXPECT_EQ(odomFlags(*reset.front()), kFlagDegraded) << name;
      for (const auto * o : after) {
        EXPECT_FALSE(o->record.degraded) << name << " step " << o->step << ": the registration is healthy";
        EXPECT_TRUE(o->record.eff_degraded) << name << " step " << o->step;
        EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kUnanchored) << name << " step " << o->step;
        EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kEmpty) << name << " step " << o->step;
        EXPECT_EQ(o->record.anchor_id, old_id) << name << " step " << o->step << ": an anchor was taken";
        EXPECT_EQ(odomFlags(*o), kFlagDegraded) << name << " step " << o->step;
      }
      EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "1") << name;
      EXPECT_EQ(counterValue(r, "recovery_started"), "0") << name;
    }
  }
}

// The platform policy on with a 1 s budget. A degraded moving stretch without a hold starts the unsupported
// recovery, timed from the last supported scan; every scan before the deadline is Pending. The first IMU sample at
// or past the deadline latches Unanchored, before the scan whose end is the deadline is processed, and every scan
// from that one on is processed under the latch. The healthy parked scans after it do not clear the latch: they
// report Degraded, take no anchor and never claim StaticHold.
TEST(AnchorLifecycleNode, ARecoveryThatRunsOutStaysUnanchored)
{
  const ChildResult r = runScenario(recoveryRunsOut());
  ASSERT_TRUE(r.ok()) << r.describe();
  const auto recovery = linesTagged(r, "[FE-RECOVERY]");
  ASSERT_EQ(recovery.size(), 2u);
  const std::string & pending = recovery[0];
  const std::string & unanchored = recovery[1];
  EXPECT_EQ(valueOf(pending, "event"), "pending") << pending;
  EXPECT_EQ(valueOf(pending, "reason"), "degraded") << pending;
  EXPECT_EQ(valueOf(unanchored, "event"), "unanchored") << unanchored;
  EXPECT_EQ(valueOf(unanchored, "reason"), "deadline") << unanchored;
  EXPECT_EQ(valueOf(pending, "path"), "scan") << pending;
  EXPECT_EQ(valueOf(unanchored, "path"), "idle") << unanchored;
  const auto anchored = r.phase(Phase::kAnchorStatic);
  ASSERT_FALSE(anchored.empty());
  const std::int64_t start = anchored.back()->record.end_ns;
  const std::int64_t deadline = start + kShortRecoveryBudgetNs;
  EXPECT_EQ(valueOf(pending, "start_ns"), std::to_string(start)) << pending;
  EXPECT_EQ(valueOf(pending, "deadline_ns"), std::to_string(deadline)) << pending;
  EXPECT_EQ(valueOf(unanchored, "deadline_ns"), std::to_string(deadline)) << unanchored;

  std::vector<const Observation *> blind = r.phase(Phase::kMove);
  const auto more = r.phase(Phase::kMoveDegraded);
  blind.insert(blind.end(), more.begin(), more.end());
  ASSERT_EQ(blind.size(), 13u);
  const Observation * last_pending = nullptr;
  const Observation * first_past = nullptr;
  int pending_scans = 0;
  for (const auto * o : blind) {
    const bool past = o->record.end_ns >= deadline;
    EXPECT_EQ(o->record.trust_post, past ? fast_lio::RecoveryTrust::kUnanchored : fast_lio::RecoveryTrust::kPending)
      << "step " << o->step << " end_ns " << o->record.end_ns << " deadline " << deadline;
    EXPECT_FALSE(o->record.supported) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o) & (kFlagDegraded | kFlagStaticHold), kFlagDegraded) << "step " << o->step;
    pending_scans += past ? 0 : 1;
    if (!past)
      last_pending = o;
    else if (first_past == nullptr)
      first_past = o;
  }
  EXPECT_GT(pending_scans, 0);
  ASSERT_NE(last_pending, nullptr);
  ASSERT_NE(first_past, nullptr) << "the stretch never reached the deadline";
  const std::uint64_t latched_at = first_past->record.scan_seq;
  // Every cloud of this scenario is queued before the IMU sample that passes its end. The first sample at or past
  // the deadline therefore arrives while the scan whose end is the deadline waits: it latches at its own time,
  // naming the scan taken before, and the waiting scan is processed under the latch.
  // AnImuSamplePastTheDeadlineLatchesBeforeTheScanInTransit has the other order, and the same latch.
  const std::int64_t first_sample = firstGridImuAtOrAfter(deadline);
  ASSERT_EQ(first_sample, deadline + 2500000);
  EXPECT_EQ(valueOf(unanchored, "stamp_ns"), std::to_string(first_sample)) << unanchored;
  EXPECT_EQ(inputNsOf(r, unanchored), first_sample) << unanchored;
  EXPECT_EQ(valueOf(unanchored, "scan_seq"), std::to_string(last_pending->record.scan_seq)) << unanchored;
  EXPECT_EQ(first_past->record.trust_pre, fast_lio::RecoveryTrust::kUnanchored) << "step " << first_past->step;
  EXPECT_EQ(recoveryLines(r, "unanchored").size(), 1u);

  const auto parked = r.phase(Phase::kParkedAfterLatch);
  ASSERT_EQ(parked.size(), 6u);
  for (const auto * o : parked) {
    EXPECT_FALSE(o->record.degraded) << "step " << o->step << ": the registration is healthy";
    EXPECT_TRUE(o->record.eff_degraded) << "step " << o->step;
    EXPECT_EQ(o->record.trust_pre, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
    EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kEmpty) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), kFlagDegraded) << "step " << o->step;
  }
  for (const auto & line : linesWhere(r, "[ZUPT-ANCHOR]", "event", "candidate_started"))
    EXPECT_LT(std::stoull(valueOf(line, "scan_seq")), latched_at) << "a candidate after the latch: " << line;
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  EXPECT_EQ(counterValue(r, "recovery_expired"), "1");
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "0");
  EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "0");
}

// A valid anchor, then one, two or three scans whose three-point clouds the filter took, IMU window and all, before
// they left for too few points, the platform parked in every window. Those windows end nothing: the anchor stays
// valid with its ID and its pose, the exits move no evidence watermark, and the degraded parked scan after them,
// up to the scan-gap bound after the last evaluated evidence, is pinned to the anchor as it would be without them.
TEST(AnchorLifecycleNode, ParkedScansThatTookTheirImuBeforeLeavingKeepTheAnchor)
{
  const FieldCase & c = kFieldCases[0];
  for (int exits = 1; exits <= 3; ++exits) {
    const std::string name = "consumed_exit_parked_" + std::to_string(exits);
    const ChildResult r = runScenario(consumedExits(name, exits, Imu::kStatic, nudgedAnchor(c), 1));
    SCOPED_TRACE(name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
    ASSERT_NE(old_id, 0u);
    const auto exited = r.phase(Phase::kEvent);
    ASSERT_EQ(exited.size(), static_cast<std::size_t>(exits));
    expectConsumedExits(exited, 0, name);
    const auto degraded = r.phase(Phase::kStatic);
    ASSERT_EQ(degraded.size(), 1u);
    EXPECT_EQ(degraded.front()->record.gap_ns, exits * kScanPeriodNs);
    expectConsumedExitsKeptTheAnchor(r, old_id, c.old_anchor, name);
  }
}

// Four such parked exits: the scan after them begins 0.4 s after the last evaluated evidence. The exits kept the
// anchor, but as no evidence of their own, so they do not stretch the 0.3 s bound: that scan finds the hole, the
// anchor ends there, and the degraded parked scan zeroes its velocity alone, although the platform never moved.
TEST(AnchorLifecycleNode, ParkedScansThatTookTheirImuBeforeLeavingDoNotStretchTheScanGapBound)
{
  const FieldCase & c = kFieldCases[0];
  const std::array<double, 3> nudged = nudgedAnchor(c);
  const std::string name = "consumed_exit_parked_4";
  const ChildResult r = runScenario(consumedExits(name, 4, Imu::kStatic, nudged, kExitAfterScans));
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
  ASSERT_NE(old_id, 0u);
  const auto exited = r.phase(Phase::kEvent);
  ASSERT_EQ(exited.size(), 4u);
  expectConsumedExits(exited, 0, name);
  for (const auto * o : exited) {
    EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kValid) << "step " << o->step;
    EXPECT_EQ(o->record.anchor_id, old_id) << "step " << o->step;
  }
  const auto degraded = r.phase(Phase::kStatic);
  ASSERT_EQ(degraded.size(), 1u);
  const Observation & d = *degraded.front();
  EXPECT_TRUE(hasEvent(d, fast_lio::kAnchorEventScanHole));
  EXPECT_EQ(d.record.gap_ns, 4 * kScanPeriodNs);
  const auto ended = anchorLines(r, "invalidated", old_id);
  ASSERT_EQ(ended.size(), 1u);
  EXPECT_EQ(valueOf(ended.front(), "reason"), "scan_hole") << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "scan_seq"), std::to_string(d.record.scan_seq)) << ended.front();
  std::string why;
  EXPECT_TRUE(zeroedVelocityOnly(d, nudged, &why)) << why;
  EXPECT_EQ(d.record.anchor_used, 0u);
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), static_cast<std::size_t>(kExitAfterScans));
  EXPECT_GT(after.back()->record.anchor_id, old_id);
  EXPECT_EQ(after.back()->record.anchor_state, fast_lio::AnchorState::kValid);
  expectNeverUsed(fromPhase(r, Phase::kStatic), old_id, name);
  expectInvalidationCounters(r, name);
  expectTraceAccountsForEveryScan(r, true, name);
}

// K8: the observation switch changes no estimate, timestamp or flag. Each pair runs the same fixture with the
// per-scan trace on and off; every published and written value must match bit for bit.
TEST(AnchorLifecycleTraceSwitch, TraceOnAndOffPublishBitIdenticalStatesAndFlags)
{
  const std::vector<std::pair<Scenario, Scenario>> pairs = {
    {parkedAgainWithLateReset(kFieldCases[0], true), parkedAgainWithLateReset(kFieldCases[0], false)},
    {pinThenLateReset(true), pinThenLateReset(false)},
    {natural(true), natural(false)},
    {recoveryRunsOut(true), recoveryRunsOut(false)},
    // The parked prior, and a reset that takes it back; a scan that leaves before the measurement update; a
    // deadline that passes on the IMU clock.
    {traced(parkedPriorThenLateReset(true), true), traced(parkedPriorThenLateReset(true), false)},
    {traced(earlyExitBetweenCandidateScans(Cloud::kSparse), true),
     traced(earlyExitBetweenCandidateScans(Cloud::kSparse), false)},
    {traced(idleWithImu(), true), traced(idleWithImu(), false)},
    // A valid anchor and a scan that took its IMU window before it left: motion in the window ends the anchor,
    // a parked window keeps it.
    {consumedExits("consumed_exit_moving_1", 1, Imu::kMoving, movedOnFromTheAnchor(kFieldCases[0]), kExitAfterScans),
     traced(
       consumedExits("consumed_exit_moving_1", 1, Imu::kMoving, movedOnFromTheAnchor(kFieldCases[0]), kExitAfterScans),
       false)},
    {consumedExits("consumed_exit_parked_1", 1, Imu::kStatic, nudgedAnchor(kFieldCases[0]), 1),
     traced(consumedExits("consumed_exit_parked_1", 1, Imu::kStatic, nudgedAnchor(kFieldCases[0]), 1), false)},
  };
  // What the runs with the trace on walked through, all of them together.
  int pins = 0;
  int zero_only = 0;
  int priors = 0;
  int pins_taken_back = 0;
  int priors_taken_back = 0;
  int left_before_the_update = 0;
  int updates = 0;
  int map_updates = 0;
  int pending = 0;
  int idle_records = 0;
  for (const auto & pair : pairs) {
    const ChildResult on = runScenario(pair.first);
    const ChildResult off = runScenario(pair.second);
    ASSERT_TRUE(on.ok()) << pair.first.name << " (trace on): " << on.describe();
    ASSERT_TRUE(off.ok()) << pair.second.name << " (trace off): " << off.describe();
    expectSameRun(on, off, pair.first.name);
    expectTraceAccountsForEveryScan(on, true, pair.first.name + " (trace on)");
    expectTraceAccountsForEveryScan(off, false, pair.second.name + " (trace off)");
    EXPECT_EQ(transitionLines(on), transitionLines(off))
      << pair.first.name << ": transition records differ with the per-scan trace on";
    if (consumedExitScenario(pair.first)) {
      expectConsumedExitOutputs(on, pair.first, pair.first.name + " (trace on)");
      expectConsumedExitOutputs(off, pair.second, pair.second.name + " (trace off)");
    }
    EXPECT_TRUE(writeTraceSample(on, pair.first.name)) << pair.first.name;
    for (const auto & o : on.observations) {
      const auto & x = o.record;
      const bool reset = hasEvent(o, fast_lio::kAnchorEventMapReset);
      pins += x.action == fast_lio::AnchorTraceAction::kPin && x.have_act && x.have_target ? 1 : 0;
      zero_only += x.zero_only && !x.usable && x.have_act ? 1 : 0;
      priors += x.parked_applied && x.have_act && x.have_target ? 1 : 0;
      // A reset that came after the anchor step found the anchor usable: the pin it took back reads as a zero-only
      // scan, the prior it took back as a healthy scan without one. Neither leaves a target, and the state after
      // the anchor step has the pose the scan brought.
      if (reset && x.usable && x.zero_only) {
        ++pins_taken_back;
      }
      if (reset && x.usable && !x.degraded && o.phase == static_cast<int>(Phase::kHoldReset)) {
        ++priors_taken_back;
        EXPECT_FALSE(x.parked_applied || x.have_target) << pair.first.name << " scan " << x.scan_seq;
        EXPECT_TRUE(x.have_act && sameArray(x.act_p, x.pre_p, 3) && sameArray(x.act_q, x.pre_q, 4))
          << pair.first.name << " scan " << x.scan_seq << ": the prior's pull is still in the state";
      }
      // After the warm-up the map exists, so a scan that leaves this early has done nothing to it either.
      if (o.step >= 0 && x.outcome == fast_lio::AnchorScanOutcome::kBeforeAnchorStep && !x.have_update) {
        ++left_before_the_update;
        EXPECT_FALSE(x.map_updated || x.have_act || x.have_target) << pair.first.name << " scan " << x.scan_seq;
      }
      updates += x.have_update ? 1 : 0;
      map_updates += x.map_updated ? 1 : 0;
      pending += x.trust_post == fast_lio::RecoveryTrust::kPending ? 1 : 0;
    }
    for (const auto & line : linesTagged(on, "[FE-RECOVERY]"))
      idle_records += valueOf(line, "path") == "idle" ? 1 : 0;
  }
  EXPECT_GT(pins, 0);
  EXPECT_GT(zero_only, 0);
  EXPECT_GT(priors, 0);
  EXPECT_EQ(pins_taken_back, 1);
  EXPECT_EQ(priors_taken_back, 1);
  EXPECT_GT(left_before_the_update, 0);
  EXPECT_GT(updates, 0);
  EXPECT_GT(map_updates, 0);
  EXPECT_GT(pending, 0);
  EXPECT_GT(idle_records, 0);
}

// Every suite of this executable has a CTest entry of its own, with its own timeout.
TEST(AnchorNodeEntries, EverySuiteHasItsOwnCTestEntry)
{
  std::set<std::string> registered;
  std::istringstream in(FASTLIO_ANCHOR_NODE_SUITES);
  for (std::string name; std::getline(in, name, ':');)
    if (!name.empty())
      registered.insert(name);
  std::set<std::string> present;
  const auto * unit = ::testing::UnitTest::GetInstance();
#ifdef INSTANTIATE_TEST_SUITE_P  // googletest 1.10 and later
  const int suites = unit->total_test_suite_count();
  const auto suite = [unit](int i) {
    return unit->GetTestSuite(i);
  };
#else  // googletest 1.8 (Foxy) calls a suite a test case
  const int suites = unit->total_test_case_count();
  const auto suite = [unit](int i) {
    return unit->GetTestCase(i);
  };
#endif
  for (int i = 0; i < suites; ++i) {
    const std::string name = suite(i)->name();
    if (name != "AnchorNodeEntries")
      present.insert(name);
  }
  EXPECT_EQ(present, registered) << "CMakeLists.txt registers one CTest entry per suite (anchor_node_entries)";
}

// ---------------------------------------------------------------------------
// The deadline on the input clock
// ---------------------------------------------------------------------------

// While no scan arrives the IMU clock is the input progress: the first sample at or past the deadline latches
// Unanchored, stamped with its own time.
TEST(AnchorLifecycleIdle, TheDeadlinePassesOnTheImuClockWhileNoScanArrives)
{
  const Scenario scenario = idleWithImu();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto before = r.phase(Phase::kBefore);
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(before.size(), 1u);
  ASSERT_EQ(event.size(), 2u);
  const std::int64_t deadline = before.back()->record.end_ns + kIdleBudgetNs;
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "degraded") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "deadline_ns"), std::to_string(deadline)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "path"), "scan") << pending.front();
  for (const auto * o : event)
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << o->step;
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  EXPECT_EQ(valueOf(latched.front(), "reason"), "deadline") << latched.front();
  EXPECT_EQ(valueOf(latched.front(), "path"), "idle") << latched.front();
  const std::int64_t at = inputNsOf(r, latched.front());
  EXPECT_GE(at, deadline);
  EXPECT_LT(at - deadline, kK5BoundNs) << "acted on " << (at - deadline) << " ns after the deadline";
  EXPECT_LE(at - deadline, kImuPeriodNs) << "not by the first IMU sample at or past the deadline";
  EXPECT_EQ(toI64(valueOf(latched.front(), "stamp_ns")), at) << latched.front();
  EXPECT_EQ(valueOf(latched.front(), "scan_seq"), std::to_string(event.back()->record.scan_seq)) << latched.front();
  EXPECT_EQ(counterValue(r, "recovery_expired"), "1");
}

// The scan that would complete the stable time ends exactly at the deadline. It can only be processed once an IMU
// sample at or past its end has arrived, and that sample, the first one past the deadline, latches first: the
// recovery is too late. With a budget 0.1 s longer the same scan ends Pending.
TEST(AnchorLifecycleIdle, AConfirmationAtTheDeadlineIsTooLate)
{
  for (const double total : {kIdleBudget, kIdleBudget + 0.1}) {
    const bool in_time = total > kIdleBudget;
    const Scenario scenario = confirmationAtTheDeadline(total);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const auto before = r.phase(Phase::kBefore);
    const auto after = r.phase(Phase::kAfter);
    ASSERT_EQ(before.size(), 1u);
    ASSERT_EQ(after.size(), 6u);
    const std::int64_t start = before.back()->record.end_ns;
    const std::int64_t deadline = start + static_cast<std::int64_t>(std::llround(total * 1e9));
    const Observation & last = *after.back();
    // The fixture's premise: the matching run spans the stable 0.5 s on the scan that ends one second after the
    // last supported one, and not on the scan before it.
    ASSERT_EQ(last.record.end_ns, start + kIdleBudgetNs);
    ASSERT_EQ(completesRun(after, 0, 500000000), &last);
    for (std::size_t i = 0; i + 1 < after.size(); ++i)
      EXPECT_EQ(after[i]->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << after[i]->step;
    EXPECT_EQ(last.record.trust_pre,
              in_time ? fast_lio::RecoveryTrust::kPending : fast_lio::RecoveryTrust::kUnanchored);
    const auto qualified = recoveryLines(r, "qualified");
    const auto latched = recoveryLines(r, "unanchored");
    const auto later = r.phase(Phase::kLater);
    ASSERT_EQ(later.size(), 2u);
    if (in_time) {
      EXPECT_TRUE(latched.empty());
      ASSERT_EQ(qualified.size(), 1u);
      EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(last.record.scan_seq)) << qualified.front();
      EXPECT_EQ(last.record.trust_post, fast_lio::RecoveryTrust::kLocalQualified);
      for (const auto * o : later) {
        EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "step " << o->step;
        EXPECT_EQ(odomFlags(*o), 0u) << "step " << o->step;
      }
    } else {
      EXPECT_TRUE(qualified.empty());
      ASSERT_EQ(latched.size(), 1u);
      EXPECT_EQ(valueOf(latched.front(), "reason"), "deadline") << latched.front();
      // The first IMU sample past the deadline arrives while the last scan waits: it latches at its own time,
      // naming the scan taken before.
      const std::int64_t first_past = firstGridImuAtOrAfter(deadline);
      ASSERT_EQ(first_past, deadline + 2500000);
      EXPECT_EQ(valueOf(latched.front(), "stamp_ns"), std::to_string(first_past)) << latched.front();
      EXPECT_EQ(inputNsOf(r, latched.front()), first_past) << latched.front();
      EXPECT_EQ(valueOf(latched.front(), "scan_seq"), std::to_string(after[after.size() - 2]->record.scan_seq))
        << latched.front();
      EXPECT_EQ(valueOf(latched.front(), "path"), "idle") << latched.front();
      EXPECT_EQ(last.record.trust_post, fast_lio::RecoveryTrust::kUnanchored);
      EXPECT_EQ(odomFlags(last), kFlagDegraded);
      for (const auto * o : later) {
        EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
        EXPECT_EQ(o->record.anchor_state, fast_lio::AnchorState::kEmpty) << "step " << o->step;
        EXPECT_EQ(odomFlags(*o), kFlagDegraded) << "step " << o->step;
      }
    }
  }
}

// Every input stops before the deadline and comes back 0.2 s after it. Nothing is latched while nothing arrives;
// the first sample back acts on the deadline before anything else, stamped with its own time.
TEST(AnchorLifecycleIdle, AfterAllInputStopsTheFirstSampleBackActsOnTheDeadline)
{
  const Scenario scenario = allInputStops();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto before = r.phase(Phase::kBefore);
  ASSERT_EQ(before.size(), 1u);
  const std::int64_t deadline = before.back()->record.end_ns + kIdleBudgetNs;
  const std::int64_t resume = deadline + static_cast<std::int64_t>(std::llround(kResumeAfterDeadline * 1e9));
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  EXPECT_EQ(valueOf(latched.front(), "reason"), "deadline") << latched.front();
  const std::int64_t at = inputNsOf(r, latched.front());
  EXPECT_GT(at, resume) << "latched while no input arrived";
  EXPECT_LE(at - resume, kImuPeriodNs) << "not by the first sample back: " << (at - resume) << " ns after it";
  EXPECT_EQ(toI64(valueOf(latched.front(), "stamp_ns")), at) << latched.front();
  EXPECT_EQ(valueOf(latched.front(), "path"), "idle") << latched.front();
  EXPECT_EQ(counterValue(r, "recovery_expired"), "1");
}

// The run-out scenario in the fixture's usual order: each scan's cloud follows the IMU sample that passes its end.
// The sample past the deadline arrives while no scan is waiting, so it latches Unanchored itself, stamped with its
// own time, and the scan whose end is the deadline is still in transit: the latch names the scan before it.
TEST(AnchorLifecycleIdle, AnImuSamplePastTheDeadlineLatchesBeforeTheScanInTransit)
{
  const ChildResult r = runScenario(recoveryRunsOut(false, false));
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto anchored = r.phase(Phase::kAnchorStatic);
  ASSERT_FALSE(anchored.empty());
  const std::int64_t deadline = anchored.back()->record.end_ns + kShortRecoveryBudgetNs;
  std::vector<const Observation *> blind = r.phase(Phase::kMove);
  const auto more = r.phase(Phase::kMoveDegraded);
  blind.insert(blind.end(), more.begin(), more.end());
  const Observation * previous = nullptr;
  const Observation * at_deadline = nullptr;
  for (const auto * o : blind) {
    if (o->record.end_ns < deadline)
      previous = o;
    else if (o->record.end_ns == deadline)
      at_deadline = o;
  }
  ASSERT_NE(previous, nullptr);
  ASSERT_NE(at_deadline, nullptr);
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  EXPECT_EQ(valueOf(latched.front(), "reason"), "deadline") << latched.front();
  const std::int64_t at = inputNsOf(r, latched.front());
  EXPECT_GE(at, deadline);
  EXPECT_LE(at - deadline, kImuPeriodNs);
  EXPECT_EQ(toI64(valueOf(latched.front(), "stamp_ns")), at) << "the latch is the IMU sample's: " << latched.front();
  EXPECT_EQ(valueOf(latched.front(), "path"), "idle") << latched.front();
  EXPECT_EQ(valueOf(latched.front(), "scan_seq"), std::to_string(previous->record.scan_seq))
    << "the scan ending at the deadline had not been taken yet: " << latched.front();
  EXPECT_EQ(previous->record.trust_post, fast_lio::RecoveryTrust::kPending);
  EXPECT_EQ(at_deadline->record.trust_pre, fast_lio::RecoveryTrust::kUnanchored);
  EXPECT_EQ(at_deadline->record.trust_post, fast_lio::RecoveryTrust::kUnanchored);
}

// ---------------------------------------------------------------------------
// The deadline while a scan waits for its IMU
// ---------------------------------------------------------------------------

// The scan whose end is 0.2 s past the deadline is queued, then the IMU samples come one by one across the
// deadline. The sample on the deadline latches Unanchored at its own time, naming the last scan taken, and
// publishes nothing. The queued scan is processed under the latch: it confirms nothing and uses no anchor, and
// neither do the healthy parked scans after it.
TEST(AnchorLifecycleQueued, TheSampleOnTheDeadlineLatchesWhileAScanIsQueued)
{
  const ChildResult r = runScenario(scanQueuedPastTheDeadline());
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto before = r.phase(Phase::kBefore);
  const auto event = r.phase(Phase::kEvent);
  const auto queued = r.phase(Phase::kAfter);
  const auto later = r.phase(Phase::kLater);
  ASSERT_EQ(before.size(), 1u);
  ASSERT_EQ(event.size(), static_cast<std::size_t>(kOnSampleDegraded));
  ASSERT_EQ(queued.size(), 1u);
  ASSERT_EQ(later.size(), 6u);
  const std::int64_t deadline = before.back()->record.end_ns + kOnSampleBudgetNs;
  // The fixture's premise, read from what the front end reports of the IMU samples it takes: it took one at a stamp
  // exactly on the deadline. No cloud of this scenario is stamped there, so that input is the sample sent on the
  // deadline. The last scan taken ends 0.1 s before the deadline, the queued one 0.2 s past it.
  ASSERT_EQ(nearestReportedInputNs(r, deadline), deadline)
    << "the front end took no IMU sample at a stamp on the deadline, out of " << r.idle.size() << " reports";
  ASSERT_EQ(event.back()->record.end_ns, deadline - kScanPeriodNs);
  ASSERT_EQ(queued.front()->record.end_ns, deadline + 2 * kScanPeriodNs);
  for (const auto * o : event)
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << o->step;
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "deadline_ns"), std::to_string(deadline)) << pending.front();
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  const std::string & latch = latched.front();
  EXPECT_EQ(valueOf(latch, "reason"), "deadline") << latch;
  EXPECT_EQ(valueOf(latch, "path"), "idle") << latch;
  EXPECT_EQ(valueOf(latch, "stamp_ns"), std::to_string(deadline)) << "not the sample on the deadline: " << latch;
  EXPECT_EQ(inputNsOf(r, latch), deadline) << "logged while another sample was the newest: " << latch;
  EXPECT_EQ(valueOf(latch, "scan_seq"), std::to_string(event.back()->record.scan_seq)) << latch;
  std::vector<const Observation *> from_latch = queued;
  from_latch.insert(from_latch.end(), later.begin(), later.end());
  expectLatchedFrom(r, from_latch);
  expectOdometryOnlyForPublishedScans(r);
  EXPECT_EQ(counterValue(r, "recovery_expired"), "1");
}

// A matching run reaches the 1 s stable time on a scan that ends 1 ms before the deadline, and that scan is queued
// when the IMU sample 2.5 ms past its end, the first one past the deadline, arrives. The sample latches at its own
// time; the scan is processed under the latch and cannot confirm the recovery, and neither can the healthy parked
// scans after it. With the deadline 5.5 ms past the scan's end, the control, the same scan confirms it.
TEST(AnchorLifecycleQueued, AConfirmationStillQueuedAtTheDeadlineDoesNotEscapeTheLatch)
{
  for (const double total : {kConfirmLate, kConfirmInTime}) {
    const bool in_time = total > kConfirmLate;
    const Scenario scenario = confirmationQueuedAtTheDeadline(total);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const auto before = r.phase(Phase::kBefore);
    const auto event = r.phase(Phase::kEvent);
    const auto after = r.phase(Phase::kAfter);
    const auto later = r.phase(Phase::kLater);
    ASSERT_EQ(before.size(), 1u);
    ASSERT_EQ(event.size(), 1u);
    ASSERT_EQ(after.size(), static_cast<std::size_t>(kConfirmRun));
    ASSERT_EQ(later.size(), 6u);
    const std::int64_t deadline = before.back()->record.end_ns + static_cast<std::int64_t>(std::llround(total * 1e9));
    const Observation & last = *after.back();
    // The fixture's premise: the run spans the stable time on its last scan and not on the one before; that scan
    // ends before the deadline, and the first IMU sample at or past the deadline comes after its end.
    ASSERT_EQ(completesRun(after, 0, kConfirmStableNs), &last);
    ASSERT_LT(last.record.end_ns, deadline);
    const std::int64_t first_past = firstGridImuAtOrAfter(deadline);
    ASSERT_EQ(first_past, last.record.end_ns + (in_time ? 7500000 : 2500000));
    for (std::size_t i = 0; i + 1 < after.size(); ++i)
      EXPECT_EQ(after[i]->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << after[i]->step;
    const auto qualified = recoveryLines(r, "qualified");
    const auto latched = recoveryLines(r, "unanchored");
    if (in_time) {
      EXPECT_TRUE(latched.empty());
      ASSERT_EQ(qualified.size(), 1u);
      EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(last.record.scan_seq)) << qualified.front();
      EXPECT_EQ(last.record.trust_pre, fast_lio::RecoveryTrust::kPending);
      EXPECT_EQ(last.record.trust_post, fast_lio::RecoveryTrust::kLocalQualified);
      for (const auto * o : later)
        EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "step " << o->step;
    } else {
      ASSERT_EQ(latched.size(), 1u);
      const std::string & latch = latched.front();
      EXPECT_EQ(valueOf(latch, "reason"), "deadline") << latch;
      EXPECT_EQ(valueOf(latch, "path"), "idle") << latch;
      EXPECT_EQ(valueOf(latch, "stamp_ns"), std::to_string(first_past)) << latch;
      EXPECT_EQ(inputNsOf(r, latch), first_past) << latch;
      EXPECT_EQ(valueOf(latch, "scan_seq"), std::to_string(after[after.size() - 2]->record.scan_seq)) << latch;
      std::vector<const Observation *> from_latch = {&last};
      from_latch.insert(from_latch.end(), later.begin(), later.end());
      expectLatchedFrom(r, from_latch);
    }
    expectOdometryOnlyForPublishedScans(r);
  }
}

// Every input stops after two degraded scans, and the LiDAR is back first: its scan, 0.3 s past the deadline, is
// queued before any IMU sample. The first IMU sample back, 0.2 s past the deadline, latches at its own time,
// naming the last scan taken before the hole. The queued scan is processed under the latch, and so are the
// healthy parked scans after it.
TEST(AnchorLifecycleQueued, WhenTheLidarIsBackFirstTheFirstImuSampleBackLatches)
{
  const ChildResult r = runScenario(lidarBackBeforeTheImu());
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto before = r.phase(Phase::kBefore);
  const auto event = r.phase(Phase::kEvent);
  const auto back = r.phase(Phase::kAfter);
  const auto later = r.phase(Phase::kLater);
  ASSERT_EQ(before.size(), 1u);
  ASSERT_EQ(event.size(), 2u);
  ASSERT_EQ(back.size(), 1u);
  ASSERT_EQ(later.size(), 6u);
  const std::int64_t deadline = before.back()->record.end_ns + kIdleBudgetNs;
  const std::int64_t resume = deadline + static_cast<std::int64_t>(std::llround(kLidarBackAfterDeadline * 1e9));
  // The fixture's premise: the queued scan ends 0.3 s past the deadline, after the IMU is back.
  ASSERT_EQ(back.front()->record.end_ns, deadline + 3 * kScanPeriodNs);
  const std::int64_t first_back = firstGridImuAtOrAfter(resume);
  ASSERT_LT(first_back, back.front()->record.end_ns);
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  const std::string & latch = latched.front();
  EXPECT_EQ(valueOf(latch, "reason"), "deadline") << latch;
  EXPECT_EQ(valueOf(latch, "path"), "idle") << latch;
  EXPECT_EQ(valueOf(latch, "stamp_ns"), std::to_string(first_back)) << "not the first sample back: " << latch;
  EXPECT_EQ(inputNsOf(r, latch), first_back) << latch;
  EXPECT_EQ(valueOf(latch, "scan_seq"), std::to_string(event.back()->record.scan_seq)) << latch;
  std::vector<const Observation *> from_latch = back;
  from_latch.insert(from_latch.end(), later.begin(), later.end());
  expectLatchedFrom(r, from_latch);
  expectOdometryOnlyForPublishedScans(r);
  EXPECT_EQ(counterValue(r, "recovery_expired"), "1");
}

// ---------------------------------------------------------------------------
// Evidence that breaks between scans
// ---------------------------------------------------------------------------

// Two qualifying scans, then a scan the front end takes but never evaluates (too few points; no usable timing),
// then qualifying scans again. The unevaluated scan ends the candidate and the parked streak: the scans after it
// start a new candidate, and three of them make the anchor.
TEST(AnchorLifecycleEvidence, AScanThatLeavesBeforeTheAnchorStepEndsTheCandidate)
{
  for (const Cloud cloud : {Cloud::kSparse, Cloud::kEmpty}) {
    const Scenario scenario = earlyExitBetweenCandidateScans(cloud);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const auto candidate = r.phase(Phase::kAnchorStatic);
    ASSERT_EQ(candidate.size(), 2u);
    const std::uint64_t old_id = candidate.back()->record.anchor_id;
    ASSERT_NE(old_id, 0u);
    EXPECT_EQ(candidate.front()->record.anchor_id, old_id);
    EXPECT_EQ(candidate.back()->record.anchor_state, fast_lio::AnchorState::kQualifying);
    const auto event = r.phase(Phase::kEvent);
    ASSERT_EQ(event.size(), 1u);
    const auto & e = event.front()->record;
    EXPECT_FALSE(e.anchor_step);
    EXPECT_FALSE(e.published);
    EXPECT_EQ(e.outcome,
              cloud == Cloud::kSparse ? fast_lio::AnchorScanOutcome::kBeforeAnchorStep
                                      : fast_lio::AnchorScanOutcome::kDroppedInvalidTime);
    EXPECT_EQ(e.anchor_state, fast_lio::AnchorState::kEmpty) << "the candidate outlived a scan nothing evaluated";
    const auto dropped = anchorLines(r, "candidate_dropped", old_id);
    ASSERT_EQ(dropped.size(), 1u);
    EXPECT_EQ(valueOf(dropped.front(), "reason"), "scan_not_evaluated") << dropped.front();
    EXPECT_EQ(valueOf(dropped.front(), "scan_seq"), std::to_string(e.scan_seq)) << dropped.front();
    EXPECT_EQ(valueOf(dropped.front(), "streak"), "0") << dropped.front();
    EXPECT_TRUE(anchorLines(r, "created", old_id).empty()) << "the candidate became an anchor across that scan";
    const auto after = r.phase(Phase::kAfter);
    ASSERT_EQ(after.size(), 4u);
    const std::uint64_t new_id = after.front()->record.anchor_id;
    EXPECT_GT(new_id, old_id);
    for (std::size_t i = 0; i < after.size(); ++i) {
      const auto & o = after[i]->record;
      EXPECT_EQ(o.anchor_id, new_id) << "step " << after[i]->step;
      EXPECT_EQ(o.anchor_state, i < 2 ? fast_lio::AnchorState::kQualifying : fast_lio::AnchorState::kValid)
        << "step " << after[i]->step;
      // The parked streak restarted with the candidate: engaged on its third scan, not before.
      EXPECT_EQ(o.parked, i >= 2) << "step " << after[i]->step;
      EXPECT_NE(o.anchor_used, old_id) << "step " << after[i]->step;
      EXPECT_EQ(odomFlags(*after[i]) & kFlagDegraded, 0u) << "step " << after[i]->step;
    }
  }
}

// An IMU dropout of 0.2 s that the motion model bridges: shorter than imu_gap_verify_s, so the re-anchor gate is
// left alone and never sees a degraded scan. With the platform policy on, the bridge still starts an unsupported
// recovery, timed from the last supported scan, and the bridged scans do not count toward the stable time.
TEST(AnchorLifecycleEvidence, AShortImuBridgeStartsARecoveryAndLeavesTheGateAlone)
{
  const Scenario scenario = imuBridge(0.2, 10.0);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto before = r.phase(Phase::kBefore);
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  const auto scans = fromPhase(r, Phase::kEvent);
  std::size_t last_bridged = scans.size();
  for (std::size_t i = 0; i < scans.size(); ++i) {
    const auto & o = scans[i]->record;
    EXPECT_EQ(o.gate_post, fast_lio::AnchorTraceGate::kHealthy) << "step " << scans[i]->step;
    EXPECT_TRUE(o.anchor_step && !o.degraded) << "step " << scans[i]->step;
    if (hasEvent(*scans[i], fast_lio::kAnchorEventImuBridge))
      last_bridged = i;
  }
  EXPECT_TRUE(hasEvent(*event.front(), fast_lio::kAnchorEventImuBridge));
  ASSERT_LT(last_bridged + 1, scans.size());
  expectInvalidatedOnce(r, old_id, "imu_bridge", scenario.name);

  const std::int64_t start = before.back()->record.end_ns;
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_bridge") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(event.front()->record.scan_seq)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(start)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "deadline_ns"), std::to_string(start + 10 * kNsPerSecond)) << pending.front();
  EXPECT_FALSE(event.front()->record.supported);

  // The stable second is shown by the scans after the last bridged one, in one piece.
  const Observation * confirming = completesRun(scans, last_bridged + 1, kNsPerSecond);
  ASSERT_NE(confirming, nullptr);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  EXPECT_TRUE(recoveryLines(r, "unanchored").empty());
  // Each scan reports its own verdict, healthy on every one of them: the Pending ones and the bridged one included.
  for (const auto * o : scans) {
    const bool pending_scan = o->record.scan_seq < confirming->record.scan_seq;
    EXPECT_EQ(o->record.trust_post,
              pending_scan ? fast_lio::RecoveryTrust::kPending : fast_lio::RecoveryTrust::kLocalQualified)
      << "step " << o->step;
    EXPECT_FALSE(o->record.eff_degraded) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), 0u) << "step " << o->step;
  }
  expectNeverUsed(scans, old_id, scenario.name);
  EXPECT_EQ(scans.back()->record.anchor_state, fast_lio::AnchorState::kValid);
  EXPECT_GT(scans.back()->record.anchor_id, old_id);
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "1");
}

// The same bridge with the platform policy off: the anchor ends and nothing else changes.
TEST(AnchorLifecycleEvidence, AShortImuBridgeWithoutThePlatformPolicyOnlyEndsTheAnchor)
{
  const Scenario scenario = imuBridge(0.2, 0.0);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  EXPECT_TRUE(hasEvent(*event.front(), fast_lio::kAnchorEventImuBridge));
  expectInvalidatedOnce(r, old_id, "imu_bridge", scenario.name);
  EXPECT_TRUE(linesTagged(r, "[FE-RECOVERY]").empty());
  const auto scans = fromPhase(r, Phase::kEvent);
  for (const auto * o : scans) {
    EXPECT_EQ(o->record.gate_post, fast_lio::AnchorTraceGate::kHealthy) << "step " << o->step;
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "step " << o->step;
    EXPECT_FALSE(o->record.eff_degraded) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), 0u) << "step " << o->step;
  }
  expectNeverUsed(scans, old_id, scenario.name);
  EXPECT_EQ(scans.back()->record.anchor_state, fast_lio::AnchorState::kValid);
  EXPECT_GT(scans.back()->record.anchor_id, old_id);
}

// An IMU dropout of 0.35 s: longer than imu_gap_verify_s, so the scan that sees the whole interval also sends the
// re-anchor gate to blind. The unsupported recovery is the bridge's, started on the first bridged scan and timed
// from the last supported one; the blind scans after it start no second episode.
TEST(AnchorLifecycleEvidence, ALongImuBridgeStartsTheRecoveryAtTheBridgeAndBlindsTheGate)
{
  const Scenario scenario = imuBridge(0.35, 10.0);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto before = r.phase(Phase::kBefore);
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  const auto scans = fromPhase(r, Phase::kEvent);
  std::size_t last_bridged = scans.size();
  bool blinded = false;
  for (std::size_t i = 0; i < scans.size(); ++i) {
    if (hasEvent(*scans[i], fast_lio::kAnchorEventImuBridge))
      last_bridged = i;
    blinded = blinded || scans[i]->record.gate_post != fast_lio::AnchorTraceGate::kHealthy;
  }
  EXPECT_TRUE(blinded) << "an IMU interval above imu_gap_verify_s left the gate healthy";
  EXPECT_TRUE(hasEvent(*event.front(), fast_lio::kAnchorEventImuBridge));
  ASSERT_LT(last_bridged + 1, scans.size());
  expectInvalidatedOnce(r, old_id, "imu_bridge", scenario.name);

  const std::int64_t start = before.back()->record.end_ns;
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_bridge") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(event.front()->record.scan_seq)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(start)) << pending.front();
  const Observation * confirming = completesRun(scans, last_bridged + 1, kNsPerSecond);
  ASSERT_NE(confirming, nullptr);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  expectNeverUsed(scans, old_id, scenario.name);
  EXPECT_EQ(scans.back()->record.anchor_state, fast_lio::AnchorState::kValid);
  EXPECT_GT(scans.back()->record.anchor_id, old_id);
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
}

// Half a second without scans while the platform stays parked: the next scan finds the hole, ends the anchor and
// starts the recovery from the last scan before it. The scans after the hole then show the stable second.
TEST(AnchorLifecycleEvidence, AScanHoleIsFoundByTheNextScan)
{
  const Scenario scenario = scanHole(5, Imu::kStatic);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto before = r.phase(Phase::kBefore);
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), 14u);
  const Observation & first = *after.front();
  const std::int64_t last = before.back()->record.end_ns;
  EXPECT_TRUE(hasEvent(first, fast_lio::kAnchorEventScanHole));
  EXPECT_EQ(first.record.gap_ns, 5 * kScanPeriodNs + (kScanPeriodNs - kScanLengthNs));
  const auto ended = anchorLines(r, "invalidated", old_id);
  ASSERT_EQ(ended.size(), 1u);
  EXPECT_EQ(valueOf(ended.front(), "reason"), "scan_hole") << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "scan_seq"), std::to_string(first.record.scan_seq)) << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "stamp_ns"), std::to_string(first.record.end_ns)) << ended.front();
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "scan_hole") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(last)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(first.record.scan_seq)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "path"), "scan") << pending.front();
  const Observation * confirming = completesRun(after, 0, kNsPerSecond);
  ASSERT_NE(confirming, nullptr);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  expectNeverUsed(after, old_id, scenario.name);
  EXPECT_GT(after.back()->record.anchor_id, old_id) << "no candidate after the recovery qualified";
}

// A second without scans while the platform moves: the IMU clock finds the hole once it is certain (the gap bound
// plus the half second a scan in transit is given), before any scan is back. The anchor ends and the recovery
// starts there, stamped with the IMU sample that found it; the first scan back finds no anchor to use.
TEST(AnchorLifecycleEvidence, AScanHoleIsFoundByTheImuClockWhileNoScanArrives)
{
  const Scenario scenario = scanHole(10, Imu::kMoving);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto before = r.phase(Phase::kBefore);
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), 14u);
  const std::int64_t last = before.back()->record.end_ns;
  const std::int64_t certain = 300000000 + 500000000;
  const auto ended = anchorLines(r, "invalidated", old_id);
  ASSERT_EQ(ended.size(), 1u);
  EXPECT_EQ(valueOf(ended.front(), "reason"), "scan_hole") << ended.front();
  EXPECT_EQ(valueOf(ended.front(), "scan_seq"), std::to_string(before.back()->record.scan_seq)) << ended.front();
  const std::int64_t at = inputNsOf(r, ended.front());
  EXPECT_GT(at - last, certain);
  EXPECT_LE(at - last, certain + kImuPeriodNs);
  EXPECT_EQ(toI64(valueOf(ended.front(), "stamp_ns")), at) << ended.front();
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "scan_hole") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(last)) << pending.front();
  EXPECT_EQ(toI64(valueOf(pending.front(), "stamp_ns")), at)
    << "stamped with the input that found it: " << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "path"), "idle") << pending.front();
  EXPECT_TRUE(hasEvent(*after.front(), fast_lio::kAnchorEventScanHole));
  EXPECT_EQ(after.front()->record.trust_pre, fast_lio::RecoveryTrust::kPending);
  expectNeverUsed(after, old_id, scenario.name);
}

// A valid anchor, then one, two or three scans whose three-point clouds the filter took, IMU window and all, before
// they left for too few points; the platform moved in the first window only. No anchor step saw that motion, but it
// ends the anchor on that very exit: the parked exits after it neither keep nor start an anchor, the degraded
// parked scan within the scan-gap bound zeroes its velocity alone where registration put it, and the healthy parked
// scans after that make a new anchor with a new ID.
TEST(AnchorLifecycleEvidence, MotionInAScanThatTookItsImuBeforeLeavingEndsTheAnchor)
{
  const FieldCase & c = kFieldCases[0];
  const std::array<double, 3> moved = movedOnFromTheAnchor(c);
  for (int exits = 1; exits <= 3; ++exits) {
    const std::string name = "consumed_exit_moving_" + std::to_string(exits);
    const ChildResult r = runScenario(consumedExits(name, exits, Imu::kMoving, moved, kExitAfterScans));
    SCOPED_TRACE(name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
    ASSERT_NE(old_id, 0u);
    const auto & anchored = r.phase(Phase::kAnchorStatic).back()->record;
    EXPECT_TRUE(anchored.have_anchor_pose && samePosition(anchored.anchor_p, c.old_anchor));
    const auto exited = r.phase(Phase::kEvent);
    ASSERT_EQ(exited.size(), static_cast<std::size_t>(exits));
    expectConsumedExits(exited, 0, name);
    const auto degraded = r.phase(Phase::kStatic);
    ASSERT_EQ(degraded.size(), 1u);
    EXPECT_EQ(degraded.front()->record.gap_ns, exits * kScanPeriodNs);
    expectConsumedExitEndedTheAnchor(r, old_id, *exited.front(), "motion", moved, name);
  }
}

// ---------------------------------------------------------------------------
// Inputs that are lost, late or refused
// ---------------------------------------------------------------------------

// A stretch of a scan's IMU window that no sample observed, four ways: the samples stop before the scan ends, the
// window holds none, a sample is invalid and its batch is dropped, the queue shed samples while scans were
// missing. Each ends the anchor and, with the platform policy on, starts an unsupported recovery timed from the
// last supported scan; the degraded parked scans after it have no anchor to hold to.
TEST(AnchorLifecycleInputs, AnImuStretchNothingObservedEndsTheAnchorAndStartsTheRecovery)
{
  for (const LostImu what : {LostImu::kTail, LostImu::kEmpty, LostImu::kInvalid, LostImu::kBacklog}) {
    const Scenario scenario = lostImu(what);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
    ASSERT_NE(old_id, 0u);
    // The scan the stretch is on: the event scan that stopped before publishing.
    const Observation * lost = nullptr;
    int stopped = 0;
    for (const auto * o : r.phase(Phase::kEvent)) {
      if (!o->record.published) {
        lost = o;
        ++stopped;
      }
    }
    ASSERT_EQ(stopped, 1);
    fast_lio::AnchorScanOutcome outcome = fast_lio::AnchorScanOutcome::kImuCoverageGap;
    if (what == LostImu::kEmpty)
      outcome = fast_lio::AnchorScanOutcome::kDroppedEmptyImu;
    if (what == LostImu::kInvalid)
      outcome = fast_lio::AnchorScanOutcome::kImuNotProcessed;
    EXPECT_EQ(lost->record.outcome, outcome);
    EXPECT_FALSE(lost->record.anchor_step);
    EXPECT_EQ(lost->record.anchor_state, fast_lio::AnchorState::kEmpty);
    // Scans stop for the backlog case, so the IMU clock finds a scan hole first; the loss is then on the first
    // scan back.
    const bool by_hole = what == LostImu::kBacklog;
    expectInvalidatedOnce(r, old_id, by_hole ? "scan_hole" : "imu_coverage_gap", scenario.name);
    const Observation * supported = lastSupportedBefore(r, *lost);
    ASSERT_NE(supported, nullptr);
    const auto pending = recoveryLines(r, "pending");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(valueOf(pending.front(), "reason"), by_hole ? "scan_hole" : "imu_coverage_gap") << pending.front();
    EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(supported->record.end_ns)) << pending.front();
    if (by_hole) {
      EXPECT_TRUE(hasEvent(*lost, fast_lio::kAnchorEventImuCoverageGap));
      EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(supported->record.scan_seq)) << pending.front();
    } else {
      EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(lost->record.scan_seq)) << pending.front();
    }
    EXPECT_EQ(lost->record.trust_post, fast_lio::RecoveryTrust::kPending);
    const auto after = r.phase(Phase::kAfter);
    ASSERT_EQ(after.size(), 3u);
    for (const auto * o : after) {
      EXPECT_TRUE(zeroOnlyScan(*o)) << "step " << o->step;
      EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << o->step;
    }
    expectNeverUsed(after, old_id, scenario.name);
    EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  }
}

// An IMU sample older than the newest one is dropped where it arrives: no evidence is broken, and the degraded
// parked scans after it hold on the same anchor.
TEST(AnchorLifecycleInputs, ALateImuSampleIsDroppedAndBreaksNothing)
{
  const Scenario scenario = lateImuSample();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  EXPECT_EQ(event.front()->record.events, 0u);
  EXPECT_EQ(event.front()->record.anchor_state, fast_lio::AnchorState::kValid);
  EXPECT_EQ(event.front()->record.anchor_id, old_id);
  EXPECT_TRUE(anchorLines(r, "invalidated", old_id).empty());
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), 3u);
  for (const auto * o : after) {
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin) << "step " << o->step;
    EXPECT_EQ(o->record.anchor_used, old_id) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), kFlagDegraded | kFlagStaticHold) << "step " << o->step;
  }
}

// An IMU sample stamped more than a second before the newest one is a clock fault: the anchor ends as soon as the
// scan path sees it, and no scan is taken afterwards.
TEST(AnchorLifecycleInputs, AnInputClockFaultEndsTheAnchorAndStopsTheScanPath)
{
  const Scenario scenario = inputEpochFault();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
  ASSERT_NE(old_id, 0u);
  expectInvalidatedOnce(r, old_id, "time_epoch", scenario.name);
  EXPECT_TRUE(r.phase(Phase::kEvent).empty()) << "a scan was taken after the fault";
  EXPECT_TRUE(r.phase(Phase::kAfter).empty()) << "a scan was taken after the fault";
  EXPECT_EQ(counterValue(r, "time_epoch"), "1");
  EXPECT_EQ(counterValue(r, "invalidated_time_epoch"), "1");
  EXPECT_EQ(counterValue(r, "scans"), std::to_string(r.observations.size()));
}

// With a wheel sample required, the static verdict is "unknown" without one, and whenever the newest one is stale,
// non-finite or stamped in another clock epoch: no candidate starts, and an anchor built on fresh samples ends.
TEST(AnchorLifecycleInputs, WheelEvidenceThatIsMissingStaleOrNotANumberProvesNothing)
{
  const Scenario scenario = wheelEvidence();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  for (const auto * o : r.phase(Phase::kBefore)) {
    EXPECT_EQ(o->record.evidence, fast_lio::StaticEvidence::kUnknown) << "step " << o->step;
    EXPECT_EQ(o->record.evidence_reason, fast_lio::StaticEvidenceReason::kWheelMissing) << "step " << o->step;
    EXPECT_FALSE(o->record.wheel_valid) << "step " << o->step;
    EXPECT_EQ(o->record.anchor_id, 0u) << "step " << o->step << ": a candidate without wheel evidence";
  }
  const auto fresh = r.phase(Phase::kAnchorStatic);
  const auto bad = r.phase(Phase::kEvent);
  ASSERT_EQ(fresh.size(), 3u * kWheelFreshScans);
  ASSERT_EQ(bad.size(), 3u);
  std::uint64_t previous_id = 0;
  for (std::size_t round = 0; round < bad.size(); ++round) {
    const std::string name = round == 0 ? "stale sample" : round == 1 ? "non-finite speed" : "other clock epoch";
    std::uint64_t id = 0;
    for (int i = 0; i < kWheelFreshScans; ++i) {
      const auto & o = fresh[round * kWheelFreshScans + static_cast<std::size_t>(i)]->record;
      EXPECT_EQ(o.evidence, fast_lio::StaticEvidence::kStationary) << name << " round, fresh scan " << i;
      EXPECT_EQ(o.evidence_reason, fast_lio::StaticEvidenceReason::kImuAndWheel) << name << " round, fresh scan " << i;
      EXPECT_TRUE(o.wheel_valid) << name << " round, fresh scan " << i;
      EXPECT_EQ(o.anchor_state, i < 2 ? fast_lio::AnchorState::kQualifying : fast_lio::AnchorState::kValid)
        << name << " round, fresh scan " << i;
      id = o.anchor_id;
    }
    EXPECT_GT(id, previous_id) << name;
    previous_id = id;
    const auto & b = bad[round]->record;
    EXPECT_EQ(b.evidence, fast_lio::StaticEvidence::kUnknown) << name;
    EXPECT_EQ(b.evidence_reason, fast_lio::StaticEvidenceReason::kWheelMissing) << name;
    EXPECT_FALSE(b.wheel_valid) << name;
    EXPECT_EQ(b.anchor_state, fast_lio::AnchorState::kEmpty) << name;
    expectInvalidatedOnce(r, id, "evidence_unknown", name);
  }
}

// The wheel is required. A valid anchor on fresh zero wheel samples, then a scan that took a parked IMU window while
// the bridge said it could not measure (a fresh NaN), and left before the anchor step. Nothing proves the platform
// stayed put: the anchor ends as evidence_unknown on that exit, which writes nothing to the state itself. The
// degraded parked scan after it, with a fresh zero again, zeroes its velocity alone.
TEST(AnchorLifecycleInputs, AScanThatTookItsImuWithoutWheelEvidenceBeforeLeavingEndsTheAnchor)
{
  const Scenario scenario = consumedExitWithoutWheelEvidence();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
  ASSERT_NE(old_id, 0u);
  for (const auto * o : r.phase(Phase::kAnchorStatic)) {
    EXPECT_EQ(o->record.evidence, fast_lio::StaticEvidence::kStationary) << "step " << o->step;
    EXPECT_TRUE(o->record.wheel_valid) << "step " << o->step;
  }
  const auto exited = r.phase(Phase::kEvent);
  ASSERT_EQ(exited.size(), 1u);
  expectConsumedExits(exited, 0, scenario.name);
  const auto degraded = r.phase(Phase::kStatic);
  ASSERT_EQ(degraded.size(), 1u);
  EXPECT_EQ(degraded.front()->record.evidence, fast_lio::StaticEvidence::kStationary);
  expectConsumedExitEndedTheAnchor(
    r, old_id, *exited.front(), "evidence_unknown", movedOnFromTheAnchor(kFieldCases[0]), scenario.name);
}

// A repeated scan, stamp and points alike, is no new input: it is dropped where it is synced, settles no IMU batch
// and leaves the anchor valid. The IMU samples of its period, which show motion, stay queued; the next scan takes
// them with its own and, although it then leaves before the anchor step, the motion in them ends the anchor.
TEST(AnchorLifecycleInputs, ARepeatedScanTakesNoImuAndTheScanThatTakesItSeesItsMotion)
{
  const Scenario scenario = repeatBeforeAConsumedExit();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const std::uint64_t old_id = validAnchorAfter(r, Phase::kAnchorStatic);
  ASSERT_NE(old_id, 0u);
  const auto & last = *r.phase(Phase::kAnchorStatic).back();
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 2u);
  const Observation & repeated = *event.front();
  EXPECT_EQ(repeated.record.outcome, fast_lio::AnchorScanOutcome::kDroppedNonadvancing);
  EXPECT_EQ(repeated.record.beg_ns, last.record.beg_ns);
  EXPECT_EQ(repeated.record.end_ns, last.record.end_ns);
  EXPECT_EQ(repeated.imu_batch, 0) << "a repeated scan settled an IMU batch";
  EXPECT_EQ(repeated.record.anchor_state, fast_lio::AnchorState::kValid);
  EXPECT_EQ(repeated.record.anchor_id, old_id);
  expectConsumedExits({event.back()}, kScanPeriodNs, scenario.name);
  EXPECT_EQ(event.back()->imu_samples, 2 * last.imu_samples) << "the exit did not take the repeated period's IMU";
  expectConsumedExitEndedTheAnchor(
    r, old_id, *event.back(), "motion", movedOnFromTheAnchor(kFieldCases[0]), scenario.name);
}

// ---------------------------------------------------------------------------
// The re-anchor gate, the map and the reference
// ---------------------------------------------------------------------------

// The real re-anchor gate with its 5 s verification timeout and reanchor_on_fail=reset_map. The scan that declares
// Lost resets the map, which re-arms the gate within that same scan; the scan still reports Lost in the legacy
// flags and in the typed observation, and it is the only one that does. With the platform policy on, the scans
// after it report Degraded (Unanchored); with it off they report what the registration says.
TEST(AnchorLifecycleGate, TheScanThatDeclaresLostReportsItAlthoughItsResetReArmsTheGate)
{
  for (const double total : {0.0, 10.0}) {
    const bool policy = total > 0.0;
    const Scenario scenario = verificationTimesOut(total);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
    ASSERT_NE(old_id, 0u);
    const auto verifying = r.phase(Phase::kAfter);
    ASSERT_EQ(verifying.size(), static_cast<std::size_t>(kVerifyingScans));
    const auto scans = fromPhase(r, Phase::kEvent);
    std::size_t lost_at = scans.size();
    int losses = 0;
    for (std::size_t i = 0; i < scans.size(); ++i) {
      if (hasEvent(*scans[i], fast_lio::kAnchorEventLost)) {
        lost_at = i;
        ++losses;
      }
    }
    ASSERT_EQ(losses, 1) << "the verification timed out " << losses << " times";
    const Observation & lost = *scans[lost_at];
    const std::int64_t waited = lost.record.end_ns - verifying.front()->record.end_ns;
    EXPECT_GT(waited, 5 * kNsPerSecond);
    EXPECT_LE(waited, 5 * kNsPerSecond + 2 * kScanPeriodNs);
    EXPECT_TRUE(hasEvent(lost, fast_lio::kAnchorEventMapReset));
    EXPECT_EQ(lost.record.gate_pre, fast_lio::AnchorTraceGate::kVerifying);
    EXPECT_EQ(lost.record.gate_post, fast_lio::AnchorTraceGate::kHealthy) << "the reset re-arms the gate";
    expectInvalidatedOnce(r, old_id, "lost", scenario.name);

    // Up to the loss the platform is held on its anchor: degraded, static hold, not lost.
    for (std::size_t i = 0; i < lost_at; ++i) {
      const Observation & o = *scans[i];
      EXPECT_EQ(o.record.anchor_used, old_id) << "step " << o.step;
      EXPECT_EQ(odomFlags(o), kFlagDegraded | kFlagStaticHold) << "step " << o.step;
      EXPECT_TRUE(o.typed_seen) << "step " << o.step;
      EXPECT_EQ(o.typed_health & fast_lio::kFrontendHealthLost, 0u) << "step " << o.step;
    }
    // The scan that lost it says so in every encoding.
    ASSERT_TRUE(lost.odom_seen);
    EXPECT_EQ(odomFlags(lost), kFlagDegraded | kFlagLost);
    EXPECT_EQ(lost.record.flags, static_cast<int>(kFlagDegraded | kFlagLost));
    ASSERT_TRUE(lost.typed_seen);
    EXPECT_EQ(lost.typed_state, typedValue(fast_lio::FrontendState::kLost));
    EXPECT_EQ(lost.typed_validity, static_cast<unsigned>(fast_lio::FrontendValidity::kInvalid));
    EXPECT_NE(lost.typed_health & fast_lio::kFrontendHealthLost, 0u);
    EXPECT_FALSE(lost.record.static_hold);
    // Only that scan: the gate is healthy again from the next one on.
    ASSERT_LT(lost_at + 1, scans.size());
    for (std::size_t i = lost_at + 1; i < scans.size(); ++i) {
      const Observation & o = *scans[i];
      EXPECT_EQ(odomFlags(o), policy ? kFlagDegraded : 0u) << "step " << o.step;
      EXPECT_TRUE(o.typed_seen) << "step " << o.step;
      EXPECT_EQ(o.typed_health & fast_lio::kFrontendHealthLost, 0u) << "step " << o.step;
      EXPECT_EQ(o.typed_state,
                typedValue(policy ? fast_lio::FrontendState::kDegraded : fast_lio::FrontendState::kTracking))
        << "step " << o.step;
      EXPECT_EQ(o.record.trust_post,
                policy ? fast_lio::RecoveryTrust::kUnanchored : fast_lio::RecoveryTrust::kLocalQualified)
        << "step " << o.step;
      EXPECT_NE(o.record.anchor_used, old_id) << "step " << o.step;
    }
    for (const auto & idle : r.idle) {
      if (idle.input_ns > lost.record.end_ns) {
        EXPECT_EQ(idle.health & fast_lio::kFrontendHealthLost, 0u) << "an idle observation at " << idle.input_ns;
      }
    }
    const auto recovery = linesTagged(r, "[FE-RECOVERY]");
    if (policy) {
      ASSERT_EQ(recovery.size(), 1u);
      EXPECT_EQ(valueOf(recovery.front(), "event"), "unanchored") << recovery.front();
      EXPECT_EQ(valueOf(recovery.front(), "reason"), "map_reset") << recovery.front();
      EXPECT_EQ(valueOf(recovery.front(), "scan_seq"), std::to_string(lost.record.scan_seq)) << recovery.front();
    } else {
      EXPECT_TRUE(recovery.empty());
    }
    EXPECT_EQ(counterValue(r, "lost"), "1");
    EXPECT_EQ(counterValue(r, "map_reset"), "1");
  }
}

// Parked on healthy scans with the parked prior pulling toward the anchor, when a runaway map reset runs after the
// anchor step. The reset ends the anchor within that scan, so the scan gives the prior's pull back: the map is
// rebuilt around, and /Odometry carries, the scan's own pose. What remains of the anchor step is what it would
// have written without an anchor: the zeroed velocity where the static verdict zeroes it, nothing otherwise.
TEST(AnchorLifecycleGate, ALateResetTakesBackTheParkedPriorOfItsScan)
{
  const FieldCase & c = kFieldCases[0];
  const std::array<double, 3> nudged = nudgedAnchor(c);
  for (const bool zupt : {true, false}) {
    const Scenario scenario = parkedPriorThenLateReset(zupt);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u);
    ASSERT_TRUE(anchorWasValidAtTheOldPose(r, c));
    const auto written = zupt ? fast_lio::AnchorTraceAction::kZeroVelocity : fast_lio::AnchorTraceAction::kNone;
    const auto held = r.phase(Phase::kHold);
    ASSERT_EQ(held.size(), 2u);
    for (const auto * o : held) {
      EXPECT_TRUE(o->record.parked_applied) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_used, old_id) << "step " << o->step;
      EXPECT_EQ(o->record.action, written) << "step " << o->step;
      ASSERT_TRUE(o->odom_seen);
      EXPECT_LT(distance(o->odom_p, c.old_anchor), distance(nudged, c.old_anchor)) << formatPosition(o->odom_p);
      EXPECT_GT(distance(o->odom_p, c.old_anchor), 0.0) << formatPosition(o->odom_p);
      EXPECT_EQ(odomFlags(*o), kFlagStaticHold) << "step " << o->step;
    }
    const auto reset = r.phase(Phase::kHoldReset);
    ASSERT_EQ(reset.size(), 1u);
    const auto & rr = reset.front()->record;
    const std::uint32_t reset_events = fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventMapReset;
    EXPECT_EQ(rr.events & reset_events, reset_events);
    EXPECT_TRUE(rr.usable) << "the anchor step ran, and applied the prior, before the reset";
    EXPECT_FALSE(rr.parked_applied);
    EXPECT_EQ(rr.anchor_used, 0u);
    EXPECT_EQ(rr.action, written);
    EXPECT_FALSE(rr.zero_only);
    EXPECT_TRUE(samePosition(rr.pre_p, nudged) && samePosition(rr.post_p, nudged))
      << "pre " << formatPosition(rr.pre_p) << " post " << formatPosition(rr.post_p);
    EXPECT_TRUE(reset.front()->odom_seen && samePosition(reset.front()->odom_p, nudged))
      << formatPosition(reset.front()->odom_p);
    for (int i = 0; i < 4; ++i)
      EXPECT_TRUE(sameBits(rr.pre_q[i], rr.post_q[i])) << "attitude component " << i;
    for (int i = 0; i < 3; ++i)
      EXPECT_TRUE(sameBits(rr.post_v[i], zupt ? 0.0 : kFixtureVelocity[static_cast<std::size_t>(i)]))
        << "velocity component " << i << " is " << rr.post_v[i];
    EXPECT_TRUE(rr.have_cov && samePoseCovariance(rr)) << "the prior's covariance update outlived its anchor";
    EXPECT_EQ(rr.anchor_state, fast_lio::AnchorState::kEmpty);
    EXPECT_FALSE(rr.static_hold);
    EXPECT_EQ(odomFlags(*reset.front()), 0u);
    EXPECT_EQ(counterValue(r, "parked_prior_applied"), "2") << "only the two scans whose published pose carries it";
    EXPECT_EQ(counterValue(r, "zero_only_unanchored"), "0") << "a healthy scan is no zero-only scan";
    expectInvalidatedOnce(r, old_id, "map_reset", scenario.name);
    const auto after = r.phase(Phase::kAfterReset);
    ASSERT_EQ(after.size(), 2u);
    for (const auto * o : after) {
      EXPECT_FALSE(o->record.parked_applied) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_used, 0u) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_id, old_id + 1) << "step " << o->step;
      EXPECT_TRUE(o->odom_seen && samePosition(o->odom_p, nudged)) << formatPosition(o->odom_p);
    }
  }
}

// What moves the reference generation on, with a prior map waiting for its alignment: the first install, a new
// alignment, and a map reset (which keeps the installed prior). The producer confirming its alignment, an
// alignment the map backend cannot represent, and every ordinary scan with its map insertion and field-of-view
// crop leave the generation, and the anchor, alone.
TEST(AnchorLifecycleGate, OnlyAReplacedReferenceMovesTheGenerationOn)
{
  ASSERT_TRUE(writePriorMap(priorMapPath()));
  std::vector<ReferenceEvent> events;
  const Scenario scenario = referenceChanges(&events);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto scans = afterWarmup(r);
  ASSERT_EQ(scans.size(), scenario.steps.size());
  const Observation * warm = lastWarmup(r);
  ASSERT_NE(warm, nullptr);
  std::uint64_t generation = warm->record.ref_gen;
  std::size_t next = 0;
  for (std::size_t i = 0; i < scans.size(); ++i) {
    const Observation & o = *scans[i];
    const ReferenceEvent * event = next < events.size() && events[next].step == o.step ? &events[next] : nullptr;
    const std::string name = event != nullptr ? event->name : "an ordinary scan";
    const bool changes = event != nullptr && event->changes;
    EXPECT_EQ(o.record.ref_gen, generation + (changes ? 1u : 0u)) << name << ", step " << o.step;
    generation = o.record.ref_gen;
    if (event == nullptr)
      continue;
    ++next;
    ASSERT_GT(i, 0u);
    const Observation & previous = *scans[i - 1];
    ASSERT_EQ(previous.record.anchor_state, fast_lio::AnchorState::kValid) << name;
    if (changes) {
      EXPECT_EQ(o.record.anchor_state, fast_lio::AnchorState::kEmpty) << name;
      expectInvalidatedOnce(r, previous.record.anchor_id, event->reason, name);
    } else {
      EXPECT_EQ(o.record.anchor_state, fast_lio::AnchorState::kValid) << name;
      EXPECT_EQ(o.record.anchor_id, previous.record.anchor_id) << name;
      EXPECT_EQ(o.record.events, 0u) << name;
    }
  }
  EXPECT_EQ(next, events.size());
  EXPECT_EQ(counterValue(r, "reference_changed"), "2");
  EXPECT_EQ(counterValue(r, "invalidated_reference_changed"), "2");
  EXPECT_EQ(counterValue(r, "map_reset"), "1");
  EXPECT_EQ(counterValue(r, "invalidated_map_reset"), "1");
}

// ---------------------------------------------------------------------------
// A map reset whose rebuild takes no point
// ---------------------------------------------------------------------------

// The runaway watchdog's map reset on a healthy parked scan with a valid anchor, at a pose whose map points the
// backend cannot represent: the rebuild takes none of them. The reset is the one change of reference: the anchor
// ends there, for that reason, and the scan seeds no candidate, uses no anchor and claims no StaticHold. With the
// platform policy on the reset latches Unanchored and nothing after it clears the latch. What follows is the
// backend's. Without a prior map nothing is left to register against: every scan after the reset finds the map
// empty, builds it again at the same pose, and leaves before the anchor step unpublished, so no scan after the
// reset is evaluated at all. With the prior kept, the scans after it register against the prior and are published.
TEST(AnchorLifecycleRebuild, AResetWhoseRebuildTakesNoPointEndsTheAnchorOnce)
{
  ASSERT_TRUE(writePriorMap(priorMapPath()));
  for (const bool prior : {false, true}) {
    for (const bool policy : {false, true}) {
      const Scenario scenario = resetRebuildsNoMap(prior, policy);
      const ChildResult r = runScenario(scenario);
      SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
      ASSERT_TRUE(r.ok());
      const std::uint64_t old_id = validAnchorAfter(r, Phase::kBefore);
      ASSERT_NE(old_id, 0u) << "no valid anchor before the reset";
      const auto before = r.phase(Phase::kBefore);
      const auto reset = r.phase(Phase::kEvent);
      const auto after = r.phase(Phase::kAfter);
      ASSERT_EQ(reset.size(), 1u);
      ASSERT_EQ(after.size(), static_cast<std::size_t>(kRebuildAfter));
      const Observation & rs = *reset.front();
      const auto & rr = rs.record;

      // The reset ran after the anchor step, at the pose the backend cannot represent.
      const std::uint32_t reset_events = fast_lio::kAnchorEventRunawayTrip | fast_lio::kAnchorEventMapReset;
      EXPECT_EQ(rr.events & reset_events, reset_events);
      EXPECT_TRUE(rr.anchor_step && rr.usable) << "the anchor step ran on the valid anchor before the reset";
      EXPECT_TRUE(rr.map_updated) << "the rebuild was handed the scan's points";
      EXPECT_TRUE(samePosition(rr.post_p, kBeyondTheMap)) << formatPosition(rr.post_p);
      ASSERT_TRUE(rs.odom_seen);
      EXPECT_TRUE(samePosition(rs.odom_p, kBeyondTheMap)) << formatPosition(rs.odom_p);

      // The reference moved on once, at the reset.
      EXPECT_EQ(rr.ref_gen, before.back()->record.ref_gen + 1);
      for (const auto * o : after)
        EXPECT_EQ(o->record.ref_gen, rr.ref_gen) << "step " << o->step;
      EXPECT_EQ(counterValue(r, "map_reset"), "1");
      EXPECT_EQ(counterValue(r, "invalidated_map_reset"), "1");

      // The anchor ended there, for that reason, and the scan seeded, used and held on nothing.
      expectInvalidatedOnce(r, old_id, "map_reset", scenario.name);
      EXPECT_EQ(rr.anchor_state, fast_lio::AnchorState::kEmpty);
      EXPECT_FALSE(rr.anchor_valid);
      EXPECT_EQ(rr.anchor_id, old_id) << "the reset scan started a candidate";
      for (const auto & line : linesWhere(r, "[ZUPT-ANCHOR]", "event", "candidate_started"))
        EXPECT_NE(valueOf(line, "scan_seq"), std::to_string(rr.scan_seq)) << line;
      EXPECT_EQ(rr.anchor_used, 0u);
      EXPECT_NE(rr.action, fast_lio::AnchorTraceAction::kPin);
      EXPECT_FALSE(rr.parked_applied);
      EXPECT_FALSE(rr.static_hold);
      EXPECT_EQ(odomFlags(rs) & kFlagStaticHold, 0u);

      const auto recovery = linesTagged(r, "[FE-RECOVERY]");
      if (policy) {
        ASSERT_EQ(recovery.size(), 1u);
        EXPECT_EQ(valueOf(recovery.front(), "event"), "unanchored") << recovery.front();
        EXPECT_EQ(valueOf(recovery.front(), "reason"), "map_reset") << recovery.front();
        EXPECT_EQ(valueOf(recovery.front(), "scan_seq"), std::to_string(rr.scan_seq)) << recovery.front();
        EXPECT_EQ(rr.trust_post, fast_lio::RecoveryTrust::kUnanchored);
        EXPECT_EQ(odomFlags(rs), kFlagDegraded);
        for (const auto * o : after) {
          EXPECT_EQ(o->record.trust_pre, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
          EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
        }
        EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "1");
        EXPECT_EQ(counterValue(r, "recovery_qualified"), "0");
      } else {
        EXPECT_TRUE(recovery.empty());
        EXPECT_EQ(rr.trust_post, fast_lio::RecoveryTrust::kLocalQualified);
        EXPECT_EQ(odomFlags(rs), 0u);
        for (const auto * o : after)
          EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "step " << o->step;
        EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "0");
      }

      if (!prior) {
        // Nothing to register against, ever again.
        for (const auto * o : after) {
          const auto & x = o->record;
          EXPECT_EQ(x.outcome, fast_lio::AnchorScanOutcome::kBeforeAnchorStep) << "step " << o->step;
          EXPECT_FALSE(x.anchor_step) << "step " << o->step;
          EXPECT_TRUE(x.map_updated) << "step " << o->step << ": the empty map was not built again";
          EXPECT_FALSE(x.have_update) << "step " << o->step;
          EXPECT_FALSE(x.published) << "step " << o->step;
          EXPECT_EQ(x.anchor_state, fast_lio::AnchorState::kEmpty) << "step " << o->step;
          EXPECT_EQ(x.anchor_id, old_id) << "step " << o->step;
        }
      } else {
        // The prior was kept: the scans after the reset register against it again and are published.
        const std::array<double, 3> origin = {0.0, 0.0, 0.0};
        for (std::size_t i = 0; i < after.size(); ++i) {
          const Observation & o = *after[i];
          EXPECT_TRUE(o.record.anchor_step && o.record.published && o.odom_seen) << "step " << o.step;
          if (i > 0) {
            EXPECT_TRUE(o.record.have_update) << "step " << o.step;
            EXPECT_LT(distance(o.odom_p, origin), 1.0) << "step " << o.step << " " << formatPosition(o.odom_p);
          }
          if (policy) {
            EXPECT_TRUE(o.record.eff_degraded) << "step " << o.step;
            EXPECT_EQ(o.record.anchor_state, fast_lio::AnchorState::kEmpty) << "step " << o.step;
            EXPECT_EQ(o.record.anchor_id, old_id) << "step " << o.step << ": an anchor was taken under the latch";
            EXPECT_EQ(odomFlags(o), kFlagDegraded) << "step " << o.step;
          }
        }
      }
      expectOdometryOnlyForPublishedScans(r);
      expectTraceAccountsForEveryScan(r, true, scenario.name);
    }
  }
}

// ---------------------------------------------------------------------------
// The runaway watchdog's hold action
// ---------------------------------------------------------------------------

// A trip with the hold action on a degraded parked scan held on its anchor. The pin of that scan is taken back
// before the front end declares itself lost: the scan publishes its own pose, attitude and pose covariance with the
// velocity alone zeroed, as a zero-only scan, and its outputs carry Lost without StaticHold. The anchor ends for
// that reason. With ZUPT off nothing was pinned and nothing is taken back: the scan publishes the state it brought.
// The loss is that scan's: the degraded parked scans after it report Degraded alone.
TEST(AnchorLifecycleHold, AHoldTakesBackThePinOfItsScanAndReportsLost)
{
  const FieldCase & c = kFieldCases[0];
  const std::array<double, 3> nudged = nudgedAnchor(c);
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  for (const bool zupt : {true, false}) {
    const Scenario scenario = pinThenLateHold(zupt);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u);
    ASSERT_TRUE(anchorWasValidAtTheOldPose(r, c));
    const auto held = r.phase(Phase::kHold);
    ASSERT_EQ(held.size(), 2u);
    for (const auto * o : held) {
      EXPECT_EQ(o->record.action, zupt ? fast_lio::AnchorTraceAction::kPin : fast_lio::AnchorTraceAction::kNone)
        << "step " << o->step;
      EXPECT_EQ(o->record.anchor_used, zupt ? old_id : 0u) << "step " << o->step;
      EXPECT_EQ(odomFlags(*o), zupt ? kFlagDegraded | kFlagStaticHold : kFlagDegraded) << "step " << o->step;
    }
    const auto trip = r.phase(Phase::kHoldReset);
    ASSERT_EQ(trip.size(), 1u);
    const Observation & t = *trip.front();
    const auto & tr = t.record;
    EXPECT_TRUE(hasEvent(t, fast_lio::kAnchorEventRunawayTrip));
    EXPECT_TRUE(hasEvent(t, fast_lio::kAnchorEventLost));
    EXPECT_FALSE(hasEvent(t, fast_lio::kAnchorEventMapReset));
    EXPECT_EQ(tr.anchor_used, 0u);
    EXPECT_FALSE(tr.parked_applied);
    if (zupt) {
      EXPECT_TRUE(tr.usable) << "the anchor step ran, and pinned, before the trip";
      EXPECT_EQ(tr.action, fast_lio::AnchorTraceAction::kZeroVelocity);
      EXPECT_TRUE(tr.zero_only);
    } else {
      EXPECT_EQ(tr.action, fast_lio::AnchorTraceAction::kNone);
      EXPECT_FALSE(tr.zero_only);
    }
    expectTheScanOwnState(t, nudged, zupt ? zero : kFixtureVelocity, zupt, true);
    EXPECT_EQ(tr.anchor_state, fast_lio::AnchorState::kEmpty);
    EXPECT_FALSE(tr.static_hold);
    EXPECT_EQ(odomFlags(t), kFlagDegraded | kFlagLost);
    EXPECT_EQ(tr.flags, static_cast<int>(kFlagDegraded | kFlagLost));
    expectInvalidatedOnce(r, old_id, "lost", scenario.name);
    EXPECT_EQ(counterValue(r, "lost"), "1");
    EXPECT_EQ(counterValue(r, "map_reset"), "0");
    EXPECT_EQ(counterValue(r, "pin_applied"), zupt ? "2" : "0")
      << "only the scans whose published pose is the anchor's";
    EXPECT_EQ(counterValue(r, "zero_only_unanchored"), zupt ? "3" : "0") << "the trip scan and the two after it";
    const auto after = r.phase(Phase::kAfterReset);
    ASSERT_EQ(after.size(), 2u);
    for (const auto * o : after) {
      if (zupt) {
        std::string why;
        EXPECT_TRUE(zeroedVelocityOnly(*o, nudged, &why)) << "step " << o->step << ": " << why;
      } else {
        EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kNone) << "step " << o->step;
        EXPECT_EQ(o->record.anchor_used, 0u) << "step " << o->step;
        EXPECT_TRUE(o->odom_seen && samePosition(o->odom_p, nudged)) << formatPosition(o->odom_p);
        EXPECT_EQ(odomFlags(*o), kFlagDegraded) << "step " << o->step;
      }
    }
  }
}

// A trip with the hold action on a healthy parked scan the parked prior pulled toward its anchor. The pull is taken
// back before the front end declares itself lost: the scan publishes its own pose, attitude and pose covariance,
// with the velocity zeroed where ZUPT zeroes it, and its outputs carry Lost without StaticHold. The anchor ends for
// that reason; the parked scans after it start a new candidate and report nothing.
TEST(AnchorLifecycleHold, AHoldTakesBackTheParkedPriorOfItsScanAndReportsLost)
{
  const FieldCase & c = kFieldCases[0];
  const std::array<double, 3> nudged = nudgedAnchor(c);
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  for (const bool zupt : {true, false}) {
    const Scenario scenario = parkedPriorThenLateHold(zupt);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
    ASSERT_TRUE(r.ok());
    const std::uint64_t old_id = oldAnchorId(r);
    ASSERT_NE(old_id, 0u);
    ASSERT_TRUE(anchorWasValidAtTheOldPose(r, c));
    const auto written = zupt ? fast_lio::AnchorTraceAction::kZeroVelocity : fast_lio::AnchorTraceAction::kNone;
    const auto held = r.phase(Phase::kHold);
    ASSERT_EQ(held.size(), 2u);
    for (const auto * o : held) {
      EXPECT_TRUE(o->record.parked_applied) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_used, old_id) << "step " << o->step;
      EXPECT_EQ(odomFlags(*o), kFlagStaticHold) << "step " << o->step;
    }
    const auto trip = r.phase(Phase::kHoldReset);
    ASSERT_EQ(trip.size(), 1u);
    const Observation & t = *trip.front();
    const auto & tr = t.record;
    EXPECT_TRUE(hasEvent(t, fast_lio::kAnchorEventRunawayTrip));
    EXPECT_TRUE(hasEvent(t, fast_lio::kAnchorEventLost));
    EXPECT_FALSE(hasEvent(t, fast_lio::kAnchorEventMapReset));
    EXPECT_TRUE(tr.usable) << "the anchor step ran, and applied the prior, before the trip";
    EXPECT_FALSE(tr.parked_applied);
    EXPECT_EQ(tr.anchor_used, 0u);
    EXPECT_EQ(tr.action, written);
    EXPECT_FALSE(tr.zero_only);
    expectTheScanOwnState(t, nudged, zupt ? zero : kFixtureVelocity, true, false);
    EXPECT_EQ(tr.anchor_state, fast_lio::AnchorState::kEmpty);
    EXPECT_FALSE(tr.static_hold);
    EXPECT_EQ(odomFlags(t), kFlagLost);
    EXPECT_EQ(tr.flags, static_cast<int>(kFlagLost));
    expectInvalidatedOnce(r, old_id, "lost", scenario.name);
    EXPECT_EQ(counterValue(r, "lost"), "1");
    EXPECT_EQ(counterValue(r, "map_reset"), "0");
    EXPECT_EQ(counterValue(r, "parked_prior_applied"), "2") << "only the two scans whose published pose carries it";
    EXPECT_EQ(counterValue(r, "zero_only_unanchored"), "0") << "a healthy scan is no zero-only scan";
    const auto after = r.phase(Phase::kAfterReset);
    ASSERT_EQ(after.size(), 2u);
    for (const auto * o : after) {
      EXPECT_FALSE(o->record.parked_applied) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_used, 0u) << "step " << o->step;
      EXPECT_EQ(o->record.anchor_id, old_id + 1) << "step " << o->step;
      EXPECT_TRUE(o->odom_seen && samePosition(o->odom_p, nudged)) << formatPosition(o->odom_p);
      EXPECT_EQ(odomFlags(*o), 0u) << "step " << o->step;
    }
  }
}

// ---------------------------------------------------------------------------
// What a scan reports while the recovery is pending (K6-R9)
// ---------------------------------------------------------------------------

// Pending stays inside the front end: each Pending scan reports its own verdict in every encoding. The degraded
// parked scans that start it report Degraded; the healthy parked scans of the matching run report neither Degraded
// nor StaticHold, carry no degraded covariance inflation and are Full. The platform is parked throughout and no
// Pending scan claims a hold; the first one is the pin on the anchor committed after the recovery qualified.
TEST(AnchorLifecycleExport, APendingScanReportsItsOwnVerdictAndNoHold)
{
  const Scenario scenario = pendingExport(kExportBudget);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto event = r.phase(Phase::kEvent);
  const auto after = r.phase(Phase::kAfter);
  const auto held = r.phase(Phase::kHold);
  ASSERT_EQ(event.size(), 3u);
  ASSERT_EQ(after.size(), static_cast<std::size_t>(kExportRun + 1));
  ASSERT_EQ(held.size(), 2u);
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "degraded") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(event.front()->record.scan_seq)) << pending.front();
  const Observation * confirming = completesRun(after, 0, kNsPerSecond);
  ASSERT_EQ(confirming, after[static_cast<std::size_t>(kExportRun - 1)]);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  EXPECT_TRUE(recoveryLines(r, "unanchored").empty());

  for (const auto * o : fromPhase(r, Phase::kEvent)) {
    expectTheScanOwnVerdict(*o);
    expectTheTypedOwnVerdict(*o);
    if (o->record.trust_post == fast_lio::RecoveryTrust::kPending) {
      EXPECT_EQ(o->record.anchor_used, 0u) << "step " << o->step;
      EXPECT_NE(o->record.action, fast_lio::AnchorTraceAction::kPin) << "step " << o->step;
    }
  }
  for (const auto * o : event) {
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << o->step;
    EXPECT_TRUE(o->record.degraded) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), kFlagDegraded) << "step " << o->step;
  }
  for (std::size_t i = 0; i + 1 < static_cast<std::size_t>(kExportRun); ++i) {
    const Observation & o = *after[i];
    EXPECT_EQ(o.record.trust_post, fast_lio::RecoveryTrust::kPending) << "step " << o.step;
    EXPECT_FALSE(o.record.degraded) << "step " << o.step;
    EXPECT_FALSE(o.record.eff_degraded) << "step " << o.step;
    EXPECT_EQ(odomFlags(o), 0u) << "step " << o.step;
    EXPECT_TRUE(noDegradedInflation(o)) << "step " << o.step;
    EXPECT_EQ(o.typed_validity, static_cast<unsigned>(fast_lio::FrontendValidity::kFull)) << "step " << o.step;
  }
  // The same degraded parked scans claim the hold once the reference is LocalQualified and an anchor is committed.
  for (const auto * o : held) {
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "step " << o->step;
    EXPECT_EQ(o->record.action, fast_lio::AnchorTraceAction::kPin) << "step " << o->step;
    EXPECT_TRUE(o->record.static_hold) << "step " << o->step;
    EXPECT_EQ(odomFlags(*o), kFlagDegraded | kFlagStaticHold) << "step " << o->step;
    EXPECT_NE(o->typed_health & fast_lio::kFrontendHealthStaticHold, 0u) << "step " << o->step;
  }
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "1");
}

// The same scans with a 1.2 s budget: it runs out on the matching run and latches Unanchored. The Pending scans before
// the latch report their own verdict; every scan from it on reports Degraded in every encoding, its registration
// healthy or not, is Unknown with no axes, and neither takes an anchor nor claims a hold.
TEST(AnchorLifecycleExport, AnUnanchoredScanStaysDegradedWhateverItsRegistration)
{
  const Scenario scenario = pendingExport(kExportShortBudget);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  EXPECT_EQ(valueOf(latched.front(), "reason"), "deadline") << latched.front();
  std::vector<const Observation *> under_latch;
  int pending_clear = 0;
  for (const auto * o : fromPhase(r, Phase::kEvent)) {
    expectTheScanOwnVerdict(*o);
    expectTheTypedOwnVerdict(*o);
    if (o->record.trust_pre == fast_lio::RecoveryTrust::kUnanchored)
      under_latch.push_back(o);
    else if (o->record.trust_post == fast_lio::RecoveryTrust::kPending && !o->record.degraded)
      ++pending_clear;
  }
  EXPECT_GT(pending_clear, 0) << "no healthy Pending scan before the latch";
  ASSERT_FALSE(under_latch.empty());
  expectLatchedFrom(r, under_latch);
  int healthy_under_latch = 0;
  for (const auto * o : under_latch) {
    healthy_under_latch += o->record.degraded ? 0 : 1;
    EXPECT_TRUE(degradedInflation(*o)) << "step " << o->step;
  }
  const auto later = r.phase(Phase::kLater);
  ASSERT_EQ(later.size(), 5u);
  for (const auto * o : later) {
    EXPECT_EQ(o->record.trust_pre, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
    EXPECT_FALSE(o->record.degraded) << "step " << o->step << ": the registration is healthy";
  }
  EXPECT_GE(healthy_under_latch, 5);
}

// A bridged scan reports its own verdict like any other: the bridge adds no veto of its own. This is the short
// bridge of AShortImuBridgeStartsARecoveryAndLeavesTheGateAlone, where every scan is healthy and reports nothing,
// with the guard calling the bridged scan degraded: the re-anchor gate goes blind and verifies on the scans after
// it. The scans the gate calls degraded report Degraded; the Pending scans after its verification report nothing.
TEST(AnchorLifecycleExport, ABridgedScanReportsItsOwnVerdict)
{
  const Scenario scenario = imuBridge(0.2, 10.0, Guard::kDegraded);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto event = r.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  const Observation & bridged = *event.front();
  EXPECT_TRUE(hasEvent(bridged, fast_lio::kAnchorEventImuBridge));
  EXPECT_TRUE(bridged.record.degraded);
  EXPECT_EQ(bridged.record.trust_post, fast_lio::RecoveryTrust::kPending);
  EXPECT_EQ(odomFlags(bridged), kFlagDegraded);
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_bridge") << pending.front();
  int pending_degraded = 0;
  int pending_clear = 0;
  for (const auto * o : fromPhase(r, Phase::kEvent)) {
    expectTheScanOwnVerdict(*o);
    if (o->record.trust_post != fast_lio::RecoveryTrust::kPending)
      continue;
    if (o->record.degraded)
      ++pending_degraded;
    else
      ++pending_clear;
  }
  EXPECT_GT(pending_degraded, 0);
  EXPECT_GT(pending_clear, 0) << "no healthy Pending scan after the verification";
}

// ---------------------------------------------------------------------------
// The first local reference: what starts an unsupported recovery before it
// ---------------------------------------------------------------------------

// A process started fresh while the platform moves. Its first scans come before any IMU sample: each one is dropped
// for an empty window, breaks the anchor evidence and blinds the raw re-anchor gate as before, but starts no
// recovery. The sustained still gate restarts its window on every moving scan, for 16.4 s or 46.7 s, far past the
// 10 s budget, and nothing runs out: no episode was started. The scan after the initialization builds the local map,
// the first reference; nothing is published before it, and it is no supported scan. From there on the rules apply as
// before: a scan whose window holds no sample starts the recovery from the last supported scan, and the matching run
// after it qualifies it.
TEST(AnchorLifecycleStartup, AColdStartStartsNoRecoveryBeforeItsFirstReference)
{
  for (const int moving : {kMovingScans, kLongMovingScans}) {
    const Scenario scenario = coldStart(moving, true);
    const ChildResult r = runScenario(scenario);
    SCOPED_TRACE(scenario.name + "\n" + describeStart(r));
    ASSERT_TRUE(r.ok());
    const Observation * build = firstEstablished(r);
    ASSERT_NE(build, nullptr);
    expectColdUntil(r, *build);
    expectNothingStartedBefore(r, *build);
    expectTheBuildScan(*build);

    const auto gap = r.phase(Phase::kGap);
    ASSERT_EQ(gap.size(), static_cast<std::size_t>(kStartupGapScans));
    for (const auto * o : gap) {
      EXPECT_EQ(o->record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu) << "step " << o->step;
      EXPECT_TRUE(hasEvent(*o, fast_lio::kAnchorEventImuCoverageGap)) << "step " << o->step;
      EXPECT_EQ(o->record.gate_post, fast_lio::AnchorTraceGate::kBlind) << "step " << o->step;
    }

    // The first scan with IMU, a restart of the still window on every moving scan after it, and the completion on
    // the scan before the build.
    const std::size_t at = indexOf(r, *build);
    ASSERT_GT(at, static_cast<std::size_t>(kStartupGapScans + moving));
    EXPECT_EQ(r.observations[kStartupGapScans].record.outcome, fast_lio::AnchorScanOutcome::kBootstrap);
    int restarts = 0;
    int completions = 0;
    for (const auto & o : r.observations) {
      restarts += o.imu_reason == static_cast<int>(fast_lio::ImuProcessReason::kInitializationMotion) ? 1 : 0;
      completions += o.imu_reason == static_cast<int>(fast_lio::ImuProcessReason::kInitializationComplete) ? 1 : 0;
    }
    EXPECT_EQ(restarts, moving - 1);
    EXPECT_EQ(completions, 1);
    EXPECT_EQ(r.observations[at - 1].imu_reason, static_cast<int>(fast_lio::ImuProcessReason::kInitializationComplete));
    // A recovery started on the first dropped scan would have run out long before the build.
    EXPECT_GT(build->record.end_ns - gap.front()->record.end_ns, kStartupBudgetNs + kK5BoundNs);

    // Published from the scan after the build on: healthy, supported, the initialization diagnostic clear.
    std::vector<const Observation *> published;
    for (const auto * o : r.phase(Phase::kStatic))
      if (o->record.published)
        published.push_back(o);
    ASSERT_GE(published.size(), 3u);
    EXPECT_EQ(published.front()->record.scan_seq, build->record.scan_seq + 1);
    for (const auto * o : published) {
      EXPECT_TRUE(o->record.supported) << "step " << o->step;
      EXPECT_EQ(odomFlags(*o) & (kFlagDegraded | kFlagLost), 0u) << "step " << o->step;
      EXPECT_EQ(o->odom_twist_cov[5], 0.0) << "step " << o->step;
    }

    // After the reference the first scan with an empty window starts the recovery, from the last supported scan.
    const auto event = r.phase(Phase::kEvent);
    ASSERT_EQ(event.size(), 2u);
    EXPECT_TRUE(event.front()->record.published);
    const Observation & dropped = *event.back();
    EXPECT_EQ(dropped.record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu);
    const Observation * supported = lastSupportedBefore(r, dropped);
    ASSERT_NE(supported, nullptr);
    EXPECT_GT(supported->record.scan_seq, build->record.scan_seq);
    EXPECT_EQ(dropped.record.trust_post, fast_lio::RecoveryTrust::kPending);
    EXPECT_EQ(dropped.record.recovery_start_ns, supported->record.end_ns);
    EXPECT_EQ(dropped.record.recovery_deadline_ns, supported->record.end_ns + kStartupBudgetNs);
    const auto pending = recoveryLines(r, "pending");
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_coverage_gap") << pending.front();
    EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(dropped.record.scan_seq)) << pending.front();
    EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(supported->record.end_ns)) << pending.front();
    EXPECT_EQ(valueOf(pending.front(), "path"), "scan") << pending.front();
    const auto after = r.phase(Phase::kAfter);
    ASSERT_EQ(after.size(), static_cast<std::size_t>(kColdAfter));
    const Observation * confirming = completesRun(after, 0, kNsPerSecond);
    ASSERT_NE(confirming, nullptr);
    const auto qualified = recoveryLines(r, "qualified");
    ASSERT_EQ(qualified.size(), 1u);
    EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
    EXPECT_TRUE(recoveryLines(r, "unanchored").empty());
    EXPECT_EQ(counterValue(r, "recovery_started"), "1");
    EXPECT_EQ(counterValue(r, "recovery_qualified"), "1");
    EXPECT_EQ(counterValue(r, "recovery_expired"), "0");
    EXPECT_EQ(counterValue(r, "imu_coverage_gap"), std::to_string(kStartupGapScans + 1));

    // Every published scan reports its own verdict, on /Odometry and in the typed observation alike.
    for (const auto & o : r.observations) {
      if (o.record.published) {
        expectTheScanOwnVerdict(o);
        expectTheTypedOwnVerdict(o);
      }
    }
    expectOdometryOnlyForPublishedScans(r);
    expectTraceAccountsForEveryScan(r, true, scenario.name);
  }
}

// An enabled raw re-anchor gate still blinds and freezes the map on the first empty-IMU drop. Initialization and
// the initial Build leave that state alone; the first registered healthy scan starts verification, still frozen.
// An unreachable match count keeps verification open within its original 5 s budget. Turning the unsupported
// recovery policy off gives the same raw gate and map-freeze timeline, including the gap after the first reference.
TEST(AnchorLifecycleStartup, AColdStartStillBlindsTheRawGateAndFreezesTheMap)
{
  Scenario on = coldStart(kMovingScans, true);
  on.name = "cold_start_reanchor_policy_on";
  on.capture_raw_state = true;
  setParam(on, rclcpp::Parameter("reanchor_en", true));
  setParam(on, rclcpp::Parameter("reanchor_min_eff", 100000000));
  Scenario off = on;
  off.name = "cold_start_reanchor_policy_off";
  setPolicy(off, 0.0, 1.0);
  const ChildResult a = runScenario(on);
  const ChildResult b = runScenario(off);
  SCOPED_TRACE(on.name + "\n" + describeStart(a) + "\n" + off.name + "\n" + describeStart(b));
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  const Observation * build = firstEstablished(a);
  const Observation * control_build = firstEstablished(b);
  ASSERT_NE(build, nullptr);
  ASSERT_NE(control_build, nullptr);
  EXPECT_EQ(build->record.scan_seq, control_build->record.scan_seq);
  expectColdUntil(a, *build);
  expectNothingStartedBefore(a, *build);
  expectTheBuildScan(*build);

  ASSERT_EQ(a.observations.size(), b.observations.size());
  const std::size_t at = indexOf(a, *build);
  ASSERT_LT(at + 1, a.observations.size());
  for (std::size_t i = 0; i < a.observations.size(); ++i) {
    const Observation & o = a.observations[i];
    const Observation & control = b.observations[i];
    SCOPED_TRACE("scan " + std::to_string(o.record.scan_seq));
    ASSERT_TRUE(o.raw_state_seen && control.raw_state_seen);
    EXPECT_EQ(o.record.scan_seq, control.record.scan_seq);
    EXPECT_EQ(o.record.outcome, control.record.outcome);
    EXPECT_EQ(o.record.gate_pre, control.record.gate_pre);
    EXPECT_EQ(o.record.gate_post, control.record.gate_post);
    EXPECT_EQ(o.map_frozen, control.map_frozen);
    EXPECT_EQ(o.raw_degraded, control.raw_degraded);
    EXPECT_EQ(o.record.map_updated, control.record.map_updated);
    EXPECT_TRUE(o.map_frozen);
    EXPECT_TRUE(o.raw_degraded);
    if (i <= at) {
      EXPECT_EQ(o.record.gate_pre, i == 0 ? fast_lio::AnchorTraceGate::kHealthy : fast_lio::AnchorTraceGate::kBlind);
      EXPECT_EQ(o.record.gate_post, fast_lio::AnchorTraceGate::kBlind);
    }
  }

  const auto gap = a.phase(Phase::kGap);
  ASSERT_EQ(gap.size(), static_cast<std::size_t>(kStartupGapScans));
  for (const auto * o : gap) {
    EXPECT_EQ(o->record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu);
    EXPECT_TRUE(hasEvent(*o, fast_lio::kAnchorEventImuCoverageGap));
  }
  const Observation & registered = a.observations[at + 1];
  EXPECT_TRUE(registered.record.published);
  EXPECT_FALSE(registered.record.guard_degraded);
  EXPECT_EQ(registered.record.gate_pre, fast_lio::AnchorTraceGate::kBlind);
  EXPECT_EQ(registered.record.gate_post, fast_lio::AnchorTraceGate::kVerifying);
  EXPECT_EQ(registered.record.trust_post, fast_lio::RecoveryTrust::kPending);
  EXPECT_FALSE(registered.record.map_updated);
  EXPECT_EQ(odomFlags(registered) & kFlagDegraded, kFlagDegraded);

  const auto event = a.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 2u);
  const Observation & dropped = *event.back();
  EXPECT_EQ(dropped.record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu);
  EXPECT_TRUE(hasEvent(dropped, fast_lio::kAnchorEventImuCoverageGap));
  EXPECT_EQ(dropped.record.gate_post, fast_lio::AnchorTraceGate::kBlind);
  const auto after = a.phase(Phase::kAfter);
  ASSERT_FALSE(after.empty());
  EXPECT_EQ(after.front()->record.gate_pre, fast_lio::AnchorTraceGate::kBlind);
  EXPECT_EQ(after.front()->record.gate_post, fast_lio::AnchorTraceGate::kVerifying);
  EXPECT_TRUE(after.front()->map_frozen);
  EXPECT_EQ(counterValue(a, "imu_coverage_gap"), std::to_string(kStartupGapScans + 1));
  expectOdometryOnlyForPublishedScans(a);
  expectTraceAccountsForEveryScan(a, true, on.name);
}

// The per-scan trace changes nothing of a cold start: the same states, flags, phases and transition records with it
// on as with it off. With it off the build is not observed, and it is the first reference all the same.
TEST(AnchorLifecycleStartup, TheTraceSwitchChangesNothingOfAColdStart)
{
  const Scenario on = coldStart(kMovingScans, true, true);
  const Scenario off = coldStart(kMovingScans, true, false);
  const ChildResult a = runScenario(on);
  const ChildResult b = runScenario(off);
  ASSERT_TRUE(a.ok()) << on.name << " (trace on): " << a.describe();
  ASSERT_TRUE(b.ok()) << off.name << " (trace off): " << b.describe();
  expectSameRun(a, b, on.name);
  EXPECT_EQ(transitionLines(a), transitionLines(b)) << on.name << ": transition records differ with the trace on";
  expectTraceAccountsForEveryScan(a, true, on.name + " (trace on)");
  expectTraceAccountsForEveryScan(b, false, off.name + " (trace off)");
  const Observation * build = firstEstablished(b);
  ASSERT_NE(build, nullptr);
  expectColdUntil(b, *build);
  EXPECT_FALSE(build->record.map_updated);
  EXPECT_TRUE(writeTraceSample(a, on.name)) << on.name;
}

// The platform never stops: the still gate times out after 90 s and the initialization completes from a moving
// window, which /Odometry keeps reporting in its own diagnostic cell. The build after it is the first reference all
// the same, and the scans after it are published without Degraded. Nothing started on the scans dropped before any
// IMU sample, 90 s earlier.
TEST(AnchorLifecycleStartup, AStillGateThatTimesOutStillGivesTheFirstReference)
{
  const Scenario scenario = coldStart(kTimeoutMovingScans, false);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(scenario.name + "\n" + describeStart(r));
  ASSERT_TRUE(r.ok());
  const Observation * build = firstEstablished(r);
  ASSERT_NE(build, nullptr);
  expectColdUntil(r, *build);
  expectNothingStartedBefore(r, *build);
  expectTheBuildScan(*build);
  const std::size_t at = indexOf(r, *build);
  ASSERT_GT(at, static_cast<std::size_t>(kStartupGapScans + 2));
  EXPECT_EQ(r.observations[kStartupGapScans].record.outcome, fast_lio::AnchorScanOutcome::kBootstrap);
  const Observation & first_try = r.observations[kStartupGapScans + 1];
  const Observation & completed = r.observations[at - 1];
  EXPECT_EQ(completed.imu_reason, static_cast<int>(fast_lio::ImuProcessReason::kInitializationComplete));
  EXPECT_GE(completed.record.beg_ns - first_try.record.beg_ns, kStillTimeoutNs);
  int restarts = 0;
  for (std::size_t i = kStartupGapScans + 1; i + 1 < at; ++i)
    restarts += r.observations[i].imu_reason == static_cast<int>(fast_lio::ImuProcessReason::kInitializationMotion);
  EXPECT_EQ(restarts, static_cast<int>(at - kStartupGapScans - 2));
  int published = 0;
  for (std::size_t i = at + 1; i < r.observations.size(); ++i) {
    const Observation & o = r.observations[i];
    EXPECT_TRUE(o.record.published) << "scan " << o.record.scan_seq;
    EXPECT_EQ(o.record.trust_post, fast_lio::RecoveryTrust::kLocalQualified) << "scan " << o.record.scan_seq;
    EXPECT_EQ(odomFlags(o) & (kFlagDegraded | kFlagLost), 0u) << "scan " << o.record.scan_seq;
    EXPECT_EQ(o.odom_twist_cov[5], 1.0) << "scan " << o.record.scan_seq << ": initialized from a moving window";
    published += o.record.published ? 1 : 0;
  }
  EXPECT_GE(published, 3);
  EXPECT_TRUE(linesTagged(r, "[FE-RECOVERY]").empty());
  EXPECT_EQ(counterValue(r, "recovery_started"), "0");
  expectOdometryOnlyForPublishedScans(r);
  expectTraceAccountsForEveryScan(r, true, scenario.name);
}

// Normal initial builds that take no point are no reference: the phase stays cold through them, and the scan with
// an empty window between them starts nothing. The build that takes the scan is the first reference. The scan with
// an empty window right after it starts the recovery at once, with its clock at that scan, there being no supported
// scan before it; the scans published after it qualify the recovery.
TEST(AnchorLifecycleStartup, AnInitialBuildThatTakesNoPointIsNoReference)
{
  const Scenario scenario = emptyFirstBuilds();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  const auto before = r.phase(Phase::kBefore);
  const auto attempts = r.phase(Phase::kEvent);
  const auto built = r.phase(Phase::kAfter);
  const auto later = r.phase(Phase::kLater);
  ASSERT_EQ(before.size(), 2u);
  ASSERT_EQ(attempts.size(), 4u);
  ASSERT_EQ(built.size(), 2u);
  ASSERT_EQ(later.size(), static_cast<std::size_t>(kColdAfter));
  EXPECT_EQ(before.front()->record.outcome, fast_lio::AnchorScanOutcome::kBootstrap);
  EXPECT_EQ(before.back()->imu_reason, static_cast<int>(fast_lio::ImuProcessReason::kInitializationComplete));
  for (std::size_t i = 0; i < attempts.size(); ++i) {
    const auto & x = attempts[i]->record;
    const bool dropped = i == 2;
    EXPECT_EQ(x.outcome,
              dropped ? fast_lio::AnchorScanOutcome::kDroppedEmptyImu : fast_lio::AnchorScanOutcome::kBeforeAnchorStep)
      << "step " << attempts[i]->step;
    // Each build ran on the scan's points, and the map backend kept none of them: the next scan builds again.
    EXPECT_EQ(x.map_updated, !dropped) << "step " << attempts[i]->step;
  }
  EXPECT_TRUE(hasEvent(*attempts[2], fast_lio::kAnchorEventImuCoverageGap));

  const Observation * build = firstEstablished(r);
  ASSERT_EQ(build, built.front());
  expectColdUntil(r, *build);
  expectNothingStartedBefore(r, *build);
  expectTheBuildScan(*build);

  const Observation & dropped = *built.back();
  EXPECT_EQ(dropped.record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu);
  EXPECT_EQ(lastSupportedBefore(r, dropped), nullptr);
  EXPECT_EQ(dropped.record.trust_post, fast_lio::RecoveryTrust::kPending);
  EXPECT_EQ(dropped.record.recovery_start_ns, dropped.record.end_ns);
  EXPECT_EQ(dropped.record.recovery_deadline_ns, dropped.record.end_ns + kStartupBudgetNs);
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_coverage_gap") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(dropped.record.scan_seq)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(dropped.record.end_ns)) << pending.front();
  for (const auto * o : later)
    EXPECT_TRUE(o->record.published) << "step " << o->step;
  const Observation * confirming = completesRun(later, 0, kNsPerSecond);
  ASSERT_NE(confirming, nullptr);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  EXPECT_TRUE(recoveryLines(r, "unanchored").empty());
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "1");
  EXPECT_EQ(counterValue(r, "recovery_expired"), "0");
  expectOdometryOnlyForPublishedScans(r);
  expectTraceAccountsForEveryScan(r, true, scenario.name);
}

// A map reset after the first reference: the reference stays established, the reset latches as before, and nothing
// after it gives the process a cold start back. Here the rebuild takes no point, and every scan after it builds the
// empty map again without taking a point either.
TEST(AnchorLifecycleStartup, AMapResetAfterTheFirstReferenceLatchesAndIsNoColdStart)
{
  const Scenario scenario = resetRebuildsNoMap(false, true);
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(scenario.name + "\n" + describeRun(r));
  ASSERT_TRUE(r.ok());
  const Observation * build = firstEstablished(r);
  ASSERT_NE(build, nullptr);
  EXPECT_EQ(build->step, -1) << "the first reference is the warm-up's map build";
  expectColdUntil(r, *build);
  expectNothingStartedBefore(r, *build);
  expectTheBuildScan(*build);
  const auto reset = r.phase(Phase::kEvent);
  ASSERT_EQ(reset.size(), 1u);
  EXPECT_TRUE(hasEvent(*reset.front(), fast_lio::kAnchorEventMapReset));
  EXPECT_EQ(reset.front()->record.trust_post, fast_lio::RecoveryTrust::kUnanchored);
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), static_cast<std::size_t>(kRebuildAfter));
  for (const auto * o : after) {
    EXPECT_EQ(o->record.outcome, fast_lio::AnchorScanOutcome::kBeforeAnchorStep) << "step " << o->step;
    EXPECT_TRUE(o->record.map_updated) << "step " << o->step;
    EXPECT_FALSE(o->record.published) << "step " << o->step;
    EXPECT_EQ(o->record.trust_post, fast_lio::RecoveryTrust::kUnanchored) << "step " << o->step;
  }
  const auto latched = recoveryLines(r, "unanchored");
  ASSERT_EQ(latched.size(), 1u);
  EXPECT_EQ(valueOf(latched.front(), "reason"), "map_reset") << latched.front();
  EXPECT_TRUE(recoveryLines(r, "pending").empty());
  EXPECT_EQ(counterValue(r, "map_reset_unanchored"), "1");
}

// A process started with a prior map is no cold start: every scan reports it excluded, and the first scan with an
// empty window starts the recovery at once, with its clock at that scan. The map is built from the prior map and the
// scan while the recovery runs, as before, and the scans published after it qualify the recovery. (The build of a
// map that did not exist is the one map update a Pending scan makes, so the trace's own account of the run is not
// asked of this one: it holds Pending scans to adding nothing to the map.)
TEST(AnchorLifecycleStartup, AStartWithAPriorMapFollowsTheRulesFromTheBeginning)
{
  ASSERT_TRUE(writePriorMap(priorMapPath()));
  const Scenario scenario = priorMapStart();
  const ChildResult r = runScenario(scenario);
  SCOPED_TRACE(describeRun(r));
  ASSERT_TRUE(r.ok());
  for (const auto & o : r.observations)
    EXPECT_EQ(o.record.startup_reference, fast_lio::StartupReference::kExcluded) << "scan " << o.record.scan_seq;
  const auto gap = r.phase(Phase::kGap);
  ASSERT_EQ(gap.size(), static_cast<std::size_t>(kStartupGapScans));
  const Observation & first = *gap.front();
  EXPECT_EQ(first.record.outcome, fast_lio::AnchorScanOutcome::kDroppedEmptyImu);
  EXPECT_EQ(first.record.trust_post, fast_lio::RecoveryTrust::kPending);
  EXPECT_EQ(first.record.recovery_start_ns, first.record.end_ns);
  EXPECT_EQ(first.record.recovery_deadline_ns, first.record.end_ns + kStartupBudgetNs);
  const auto pending = recoveryLines(r, "pending");
  ASSERT_EQ(pending.size(), 1u);
  EXPECT_EQ(valueOf(pending.front(), "reason"), "imu_coverage_gap") << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "scan_seq"), std::to_string(first.record.scan_seq)) << pending.front();
  EXPECT_EQ(valueOf(pending.front(), "start_ns"), std::to_string(first.record.end_ns)) << pending.front();

  // The first scan with IMU, the one that completes the initialization, the map build, then the published scans.
  const auto after = r.phase(Phase::kAfter);
  ASSERT_EQ(after.size(), static_cast<std::size_t>(3 + kColdAfter));
  EXPECT_EQ(after[0]->record.outcome, fast_lio::AnchorScanOutcome::kBootstrap);
  EXPECT_EQ(after[1]->imu_reason, static_cast<int>(fast_lio::ImuProcessReason::kInitializationComplete));
  const auto & b = after[2]->record;
  EXPECT_EQ(b.outcome, fast_lio::AnchorScanOutcome::kBeforeAnchorStep);
  EXPECT_TRUE(b.map_updated);
  EXPECT_FALSE(b.published);
  EXPECT_EQ(b.trust_post, fast_lio::RecoveryTrust::kPending);
  const std::vector<const Observation *> published(after.begin() + 3, after.end());
  for (const auto * o : published)
    EXPECT_TRUE(o->record.published) << "step " << o->step;
  const Observation * confirming = completesRun(published, 0, kNsPerSecond);
  ASSERT_NE(confirming, nullptr);
  const auto qualified = recoveryLines(r, "qualified");
  ASSERT_EQ(qualified.size(), 1u);
  EXPECT_EQ(valueOf(qualified.front(), "scan_seq"), std::to_string(confirming->record.scan_seq)) << qualified.front();
  EXPECT_TRUE(recoveryLines(r, "unanchored").empty());
  EXPECT_EQ(counterValue(r, "recovery_started"), "1");
  EXPECT_EQ(counterValue(r, "recovery_qualified"), "1");
  EXPECT_EQ(counterValue(r, "scans"), std::to_string(r.observations.size()));
  EXPECT_EQ(linesTagged(r, "[ZUPT-ANCHOR-SCAN]").size(), r.observations.size());
  expectOdometryOnlyForPublishedScans(r);
}

namespace
{
Scenario withReference(Scenario scenario)
{
  scenario.reference = scenario.typed = true;
  setParam(scenario, rclcpp::Parameter("publish.reference_health", true));
  setParam(scenario, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(scenario, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  return scenario;
}

const fast_lio_interfaces::msg::ReferenceHealth * referenceFor(const ChildResult & result, std::uint64_t scan)
{
  for (const auto & value : result.references)
    if (value.has_scan_sequence && value.scan_sequence == scan && value.kind != 4)
      return &value;
  return nullptr;
}

void expectReferenceProjection(const Observation & observation, const fast_lio_interfaces::msg::ReferenceHealth & value)
{
  const auto & trace = observation.record.reference_health;
  EXPECT_EQ(value.schema_version, 1);
  EXPECT_EQ(value.identity_version, 1);
  EXPECT_TRUE(value.has_source_instance && value.has_coordinate_epoch && value.has_snapshot_sequence &&
              value.has_state_revision && value.has_qualified_event_sequence);
  EXPECT_EQ(value.clock_domain, "fixture");
  EXPECT_EQ(value.implemented_mask, 5u);
  EXPECT_EQ(value.enabled_policy_mask, 5u);
  EXPECT_TRUE(trace.present);
  EXPECT_EQ(value.snapshot_sequence, trace.snapshotSequence);
  EXPECT_EQ(value.state_revision, trace.stateRevision);
  EXPECT_EQ(value.scan_sequence, trace.scanSequence);
  EXPECT_EQ(value.qualified_event_sequence, trace.qualifiedSequence);
  EXPECT_EQ(value.kind, trace.kind);
  EXPECT_EQ(value.lifecycle, trace.lifecycle);
  EXPECT_EQ(value.startup_reference, trace.startup);
  EXPECT_EQ(value.trust, trace.trust);
  EXPECT_EQ(value.local, trace.local);
  EXPECT_EQ(value.legacy_flags, trace.legacyFlags);
  EXPECT_EQ(value.true_mask, trace.trueMask);
  EXPECT_EQ(value.present_mask, 143u);
  EXPECT_EQ(value.cause_mask, trace.cause);
  EXPECT_EQ(value.taint_mask, trace.taint);
  EXPECT_EQ(value.hard_mask, trace.hard);
  EXPECT_EQ(value.has_pose_binding, observation.record.published);
  EXPECT_FALSE(value.has_rw || value.has_raw_episode || value.has_raw_start || value.has_rw_origin ||
               value.has_frozen_residual || value.has_withdrawal_rearm || value.has_rw_valid_until ||
               value.has_wheel_witness_epochs || value.has_support || value.has_support_seed ||
               value.has_supported_through || value.has_first_loss || value.has_support_deadline ||
               value.has_proof_id || value.has_trajectory_id);
  EXPECT_EQ(value.present_mask & 112u, 0u);
  if (!value.has_pose_binding) {
    EXPECT_EQ(value.local, 0);
    EXPECT_EQ(value.hr_witness_kind, 0);
    EXPECT_FALSE(value.has_observation_id || value.has_hr_valid_until);
    return;
  }
  ASSERT_TRUE(observation.odom_seen && observation.typed_seen);
  EXPECT_EQ(value.kind, 1);
  EXPECT_EQ(value.scan_end_ns, observation.odom_stamp_ns);
  EXPECT_EQ(value.legacy_flags, odomFlags(observation));
  EXPECT_EQ(value.legacy_flags, observation.typed_health);
  EXPECT_EQ(value.source_instance, observation.typed_instance);
  EXPECT_EQ(value.observation_id, observation.typed_id);
  EXPECT_EQ(value.pose_binding, observation.typed_pose);
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_TRUE(sameBits(value.pose_binding[i], observation.odom_p[i]));
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_TRUE(sameBits(value.pose_binding[i + 3], observation.odom_q[i + 1]));
  EXPECT_TRUE(sameBits(value.pose_binding[6], observation.odom_q[0]));
}
}  // namespace

TEST(AnchorLifecycleReference, BuildPublishesEstablishedWithoutPoseAndKeepsEmptyBuildsCold)
{
  const auto result = runScenario(withReference(emptyFirstBuilds()));
  ASSERT_TRUE(result.ok()) << result.describe();
  const auto * build = firstEstablished(result);
  ASSERT_NE(build, nullptr);
  const auto * value = referenceFor(result, build->record.scan_seq);
  ASSERT_NE(value, nullptr);
  expectReferenceProjection(*build, *value);
  EXPECT_EQ(value->kind, 2);
  EXPECT_EQ(value->startup_reference, 2);
  EXPECT_EQ(value->lifecycle, 2);
  EXPECT_EQ(value->true_mask, 3u);
  EXPECT_EQ(value->qualified_event_sequence, 0u);
  EXPECT_FALSE(value->has_last_supported_end);
  int cold = 0, tainted = 0;
  for (const auto & observation : result.observations) {
    const auto * sample = referenceFor(result, observation.record.scan_seq);
    ASSERT_NE(sample, nullptr) << observation.record.scan_seq;
    expectReferenceProjection(observation, *sample);
    if (observation.record.scan_seq < build->record.scan_seq) {
      EXPECT_EQ(sample->startup_reference, 1);
      EXPECT_EQ(sample->lifecycle, 1);
      EXPECT_EQ(sample->qualified_event_sequence, 0u);
      ++cold;
    }
    if (!sample->has_pose_binding && (sample->taint_mask & 64u) != 0)
      ++tainted;
  }
  EXPECT_GT(cold, 0);
  EXPECT_GT(tainted, 0) << "committed initialization exit was not settled before the no-pose snapshot";
}

TEST(AnchorLifecycleReference, FinalSnapshotBindsPoseFlagsTypedAndTrace)
{
  const auto result = runScenario(withReference(pendingExport(kExportBudget)));
  ASSERT_TRUE(result.ok()) << result.describe();
  int pendingClear = 0, normal = 0, held = 0;
  std::uint64_t previous = 0;
  for (const auto & observation : result.observations) {
    const auto * value = referenceFor(result, observation.record.scan_seq);
    ASSERT_NE(value, nullptr);
    expectReferenceProjection(observation, *value);
    EXPECT_GT(value->snapshot_sequence, previous);
    previous = value->snapshot_sequence;
    EXPECT_LE(value->qualified_event_sequence, 1u);
    if (value->trust == 2 && value->local == 1) {
      EXPECT_EQ(value->true_mask & 12u, 0u);
      EXPECT_EQ(value->qualified_event_sequence, 0u);
      ++pendingClear;
    }
    if (value->hr_witness_kind == 1) {
      EXPECT_EQ(value->true_mask & 139u, 139u);
      EXPECT_EQ(value->hr_valid_until_ns, value->scan_end_ns + 200000000);
      ++normal;
    }
    if (value->hr_witness_kind == 2) {
      EXPECT_EQ(value->legacy_flags & 2u, 2u);
      EXPECT_EQ(value->trust, 1);
      ++held;
    }
  }
  EXPECT_GT(pendingClear, 0);
  EXPECT_GT(normal, 0);
  EXPECT_GT(held, 0);
  EXPECT_EQ(result.references.back().qualified_event_sequence, 1u);
}

TEST(AnchorLifecycleReference, LateResetOverridesHealthyAndPreservesTheEstablishedOnceFact)
{
  const auto result = runScenario(withReference(resetRebuildsNoMap(false, true)));
  ASSERT_TRUE(result.ok()) << result.describe();
  const auto event = result.phase(Phase::kEvent);
  ASSERT_EQ(event.size(), 1u);
  const auto * value = referenceFor(result, event.front()->record.scan_seq);
  ASSERT_NE(value, nullptr);
  expectReferenceProjection(*event.front(), *value);
  EXPECT_EQ(value->startup_reference, 2);
  EXPECT_EQ(value->trust, 3);
  EXPECT_EQ(value->local, 3);
  EXPECT_EQ(value->true_mask & 12u, 0u);
  EXPECT_NE(value->hard_mask & 4u, 0u);
  for (const auto * observation : result.phase(Phase::kAfter)) {
    const auto * after = referenceFor(result, observation->record.scan_seq);
    ASSERT_NE(after, nullptr);
    EXPECT_EQ(after->startup_reference, 2);
    EXPECT_EQ(after->trust, 3);
    EXPECT_FALSE(after->has_pose_binding);
  }
}

TEST(AnchorLifecycleReference, QueuedScanCannotCoverANewerNoPoseImuLatch)
{
  const auto result = runScenario(withReference(scanQueuedPastTheDeadline()));
  ASSERT_TRUE(result.ok()) << result.describe();
  const auto latches = recoveryLines(result, "unanchored");
  ASSERT_EQ(latches.size(), 1u);
  const auto deadline = std::stoll(valueOf(latches.front(), "stamp_ns"));
  int noPose = 0, repeat = 0;
  std::uint64_t negativeRevision = 0;
  for (const auto & value : result.references) {
    if (value.trust == 3 && !value.has_pose_binding) {
      EXPECT_EQ(value.local, 0);
      EXPECT_EQ(value.hr_witness_kind, 0);
      EXPECT_GE(value.evaluated_input_ns, deadline);
      if (negativeRevision == 0)
        negativeRevision = value.state_revision;
      ++noPose;
    }
    if (value.kind == 4)
      ++repeat;
    if (value.has_pose_binding && value.scan_end_ns >= deadline)
      EXPECT_GE(value.state_revision, negativeRevision);
    EXPECT_EQ(value.qualified_event_sequence, 0u);
  }
  EXPECT_GT(noPose, 0);
  // Repeats are timing dependent; every one actually observed must retain its original lease.
  for (const auto & value : result.references) {
    if (value.kind != 4)
      continue;
    const auto original = std::find_if(result.references.begin(), result.references.end(), [&](const auto & candidate) {
      return candidate.kind != 4 && candidate.snapshot_sequence == value.snapshot_sequence;
    });
    ASSERT_NE(original, result.references.end());
    EXPECT_EQ(value.state_valid_until_ns, original->state_valid_until_ns);
    EXPECT_EQ(value.effective_ns, original->effective_ns);
  }
  (void)repeat;
}

TEST(AnchorLifecycleReference, PolicyOffAndPriorRemainReadOnlyProjections)
{
  ASSERT_TRUE(writePriorMap(priorMapPath()));
  for (auto scenario : {resetWhileParkedHealthy(0.0), priorMapStart()}) {
    const bool prior = scenario.name == "prior_map_start";
    const auto result = runScenario(withReference(std::move(scenario)));
    ASSERT_TRUE(result.ok()) << result.describe();
    for (const auto & observation : result.observations) {
      const auto * value = referenceFor(result, observation.record.scan_seq);
      ASSERT_NE(value, nullptr);
      expectReferenceProjection(observation, *value);
      if (prior) {
        EXPECT_EQ(value->startup_reference, 3);
        EXPECT_EQ(value->lifecycle, 0);
        EXPECT_EQ(value->hr_witness_kind, 0);
      } else {
        EXPECT_EQ(value->trust, 1);
      }
    }
  }
}

TEST(AnchorLifecycleReference, ClosedArmAndTraceSwitchPreserveLegacyFields)
{
  auto baseline = pendingExport(kExportBudget);
  baseline.trace = false;
  setParam(baseline, rclcpp::Parameter("zupt_anchor_trace_en", false));
  const auto off = runScenario(baseline);
  auto enabled = withReference(baseline);
  const auto on = runScenario(enabled);
  enabled.trace = true;
  setParam(enabled, rclcpp::Parameter("zupt_anchor_trace_en", true));
  const auto traced = runScenario(enabled);
  ASSERT_TRUE(off.ok() && on.ok() && traced.ok()) << off.describe() << on.describe() << traced.describe();
  EXPECT_TRUE(off.references.empty());
  ASSERT_EQ(off.observations.size(), on.observations.size());
  ASSERT_EQ(off.observations.size(), traced.observations.size());
  for (std::size_t i = 0; i < off.observations.size(); ++i) {
    const auto & expected = off.observations[i];
    for (const auto * candidate : {&on.observations[i], &traced.observations[i]}) {
      EXPECT_EQ(candidate->record.published, expected.record.published);
      EXPECT_EQ(candidate->odom_seen, expected.odom_seen);
      EXPECT_EQ(candidate->odom_stamp_ns, expected.odom_stamp_ns);
      EXPECT_TRUE(sameArray(candidate->odom_p, expected.odom_p, 3));
      EXPECT_TRUE(sameArray(candidate->odom_q, expected.odom_q, 4));
      EXPECT_TRUE(sameArray(candidate->odom_cov, expected.odom_cov, 36));
      EXPECT_TRUE(sameArray(candidate->odom_twist_cov, expected.odom_twist_cov, 36));
      EXPECT_EQ(candidate->typed_state, expected.typed_state);
      EXPECT_EQ(candidate->typed_validity, expected.typed_validity);
      EXPECT_EQ(candidate->typed_health, expected.typed_health);
      EXPECT_EQ(candidate->typed_axes, expected.typed_axes);
      EXPECT_EQ(candidate->typed_substate, expected.typed_substate);
    }
  }
}

TEST(AnchorLifecycleReference, EpochFaultPublishesNegativeWithoutPoseBeforeAndAfterReference)
{
  for (bool before : {false, true}) {
    auto scenario = inputEpochFault();
    if (before) {
      scenario.warm_up = false;
      scenario.steps.clear();
      append(scenario, 1, slot(Phase::kBefore));
      Step fault = slot(Phase::kEvent);
      fault.clock_back = true;
      fault.expect_scan = false;
      scenario.steps.push_back(fault);
      Step after = slot(Phase::kAfter);
      after.expect_scan = false;
      append(scenario, 2, after);
    }
    const auto result = runScenario(withReference(std::move(scenario)));
    ASSERT_TRUE(result.ok()) << result.describe();
    int negatives = 0;
    for (const auto & value : result.references) {
      if ((value.hard_mask & 64u) == 0)
        continue;
      EXPECT_FALSE(value.has_pose_binding);
      EXPECT_EQ(value.local, 0);
      EXPECT_FALSE(value.has_coordinate_epoch);
      EXPECT_EQ(value.startup_reference, before ? 3 : 2);
      EXPECT_EQ(value.hr_witness_kind, 0);
      ++negatives;
    }
    EXPECT_GT(negatives, 0);
  }
}

TEST(AnchorLifecycleReference, EnablingRequiresTheExistingTypedIdentity)
{
  auto scenario = baseParams("reference_without_typed", false);
  setParam(scenario, rclcpp::Parameter("publish.reference_health", true));
  auto result = runScenario(scenario);
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.error.find("requires publish.frontend_observation=true"), std::string::npos);
  scenario.typed = true;
  setParam(scenario, rclcpp::Parameter("publish.frontend_observation", true));
  result = runScenario(scenario);
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.error.find("explicit clock domain"), std::string::npos);
}

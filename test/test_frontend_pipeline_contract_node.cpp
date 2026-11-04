// The pipeline contract of the legacy front-end node (laserMapping S0, design checklist step 4).
//
// Each scenario runs the real LaserMappingNode (fastlio_mapping_test_component, through class_loader and its
// NodeFactory) in a forked child of its own, fed through the node's real IMU and point-cloud subscriptions: every scan
// takes the production scan path, off the node's own 100 Hz timer, to the /Odometry publisher. The existing
// FASTLIO_TEST_HOOKS fix only what a synthetic stream cannot produce on demand (the divergence guard's verdict, the
// registration state test_anchor_lifecycle_node injects before the anchor step, a runaway trip, a failing IMU
// processing step) and report what the scan path did. The child writes one record, in the order things happened:
//   slot      what the fixture sent for one scan period: the scan, its IMU window and what was done to them;
//   scan_begin, imu_process, imu_batch, scan_prepared, initial_build, h_rows, guard, anchor_step_fixture,
//   runaway_check, map_reset_join, map_reset_pending_cleared, map_reset, pose_publish
//             every hook the scan path called, in call order: the stage order, with the IMU batch's settlement, the
//             prepared scan, the H-model's row counts of every iteration, the guard and watchdog calls, a map reset
//             and the publish;
//   record    the scan's complete anchor record: outcome and times, the state before, at and after the anchor step
//             (position, attitude, velocity), the 6x6 pose covariance blocks, the anchor, the recovery trust, the
//             measurement update's size and the reference-health projection;
//   odom      the /Odometry message of every published scan: pose, pose covariance, twist and the side-channel
//             covariance cells;
//   trace     every anchor and recovery trace line (zupt_anchor_trace_en), the destructor's counter summary
//             included; the per-scan [ZUPT-ANCHOR-SCAN] line is left out, the record holds all of its fields;
//   inputs    the input transaction after the slot: queued and refused messages, the fault and freeze latches and
//             the filter's input end.
// Doubles are written with 17 significant digits (round trip) and NaN with its bit pattern; a record is compared with
// its golden line for line, character for character, with no tolerance anywhere.
//
// What makes a record reproducible: the map worker is off (FLIO_ASYNC_MAP=0) except in the worker suite, OpenMP runs
// one thread, and the scan path runs (the fixture waits for its timer to come due and lets the executor run it) after
// every IMU sample and at the end of every slot, so nothing it evaluates depends on when its timer happened to fire.
// In the worker suite the worker's own hook only counts map additions; the count is read where the scan path prepares
// the next scan, after it has joined the worker. A generation run records every scenario twice and writes a golden
// only when both records agree.
//
// Masked: nothing. Not recorded: log output (its throttling runs on the wall clock), timing diagnostics, and the
// typed FrontendObservation/ReferenceHealth outputs (left off: their identity is random per activation).
//
// Every scenario counts the branches it went through; the receipt lists each branch, triggered or not, with its
// count. A branch the scenario claims and did not reach fails it. Branches the existing hooks cannot reach are listed
// in the receipt as suspended, with the hook S1 has to add to laserMapping.cpp and where. Where an existing suite
// already asserts a branch's semantics, the receipt names it; this fixture records the branch's outputs bit for bit
// and does not repeat those assertions.
//
// The goldens apply to x86_64 only: on other architectures CMake registers the entries as skipped, and the
// executable, run there in comparison mode, fails with that statement rather than pass.

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include <pcl/point_types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <class_loader/class_loader.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <openssl/evp.h>
#include <optional>
#include <rcl/error_handling.h>
#include <rcl/timer.h>
#include <rcl/wait.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "anchor_trace.hpp"
#include "imu_process_outcome.hpp"
#include "laser_mapping_test_hooks.hpp"
#include "preprocess.h"

namespace
{

constexpr const char * kLaserMappingFactory = "rclcpp_components::NodeFactoryTemplate<LaserMappingNode>";
constexpr const char * kFixture = "pipeline";
constexpr const char * kImuTopic = "/test_frontend_pipeline/imu";
constexpr const char * kLidarTopic = "/test_frontend_pipeline/points";
constexpr double kFirstScan = 1.0;    // s, start of the first scan
constexpr double kScanPeriod = 0.1;   // s
constexpr double kScanLength = 0.05;  // s, last point time within a scan
constexpr std::size_t kScanPoints = 2000;
constexpr std::int64_t kImuPeriodNs = 5000000;
constexpr std::int64_t kImuFirstNs = 992500000;  // 2.5 ms off the scan grid: no sample sits on a scan end
constexpr double kMovingYawRate = 0.3;           // rad/s, well above zupt_gyro_thresh
constexpr double kGravity = 9.81;
constexpr float kFarOffset = 30.0f;  // m along x: no point of such a scan has a map neighbour
constexpr int kWarmupPublished = 2;
constexpr int kWarmupMaxScans = 12;
constexpr double kScanWaitSeconds = 20.0;
constexpr double kReceiptWaitSeconds = 5.0;
constexpr std::int64_t kReceiptWaitNs = 5000000000;
constexpr std::int64_t kScanPathPeriodNs = 10000000;  // the front end runs its scan path off a 100 Hz timer
constexpr unsigned kChildAlarmSeconds = 600;

// test_anchor_lifecycle_node's F1 anchor, the parked pose 0.36 m from it, and its fixture velocity and covariance:
// the stand-in for what registration produced, written before the anchor step.
constexpr std::array<double, 3> kAnchorPosition = {28.41, 8.25, -5.21};
constexpr std::array<double, 3> kNudgedPosition = {28.71, 8.45, -5.21};
constexpr std::array<double, 3> kFixtureVelocity = {0.4, -0.3, 0.05};
constexpr std::array<double, 23> kFixtureCovDiag = {1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3, 1e-6, 1e-6,
                                                    1e-6, 1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2, 1e-6,
                                                    1e-6, 1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4};

// What overrideGuardDegraded answers for a scan.
constexpr int kGuardNatural = -1;  // the divergence guard's own verdict
constexpr int kGuardHealthy = 0;
constexpr int kGuardDegraded = 1;

#if defined(__x86_64__)
constexpr const char * kArch = "x86_64";
#elif defined(__aarch64__)
constexpr const char * kArch = "aarch64";
#else
constexpr const char * kArch = "other";
#endif

// ---------------------------------------------------------------------------
// Text of a record
// ---------------------------------------------------------------------------

std::string formatDouble(double value)
{
  char text[64];
  if (std::isnan(value)) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::snprintf(text, sizeof(text), "nan:0x%016llx", static_cast<unsigned long long>(bits));
  } else if (std::isinf(value)) {
    std::snprintf(text, sizeof(text), "%s", value > 0.0 ? "inf" : "-inf");
  } else {
    std::snprintf(text, sizeof(text), "%.17g", value);
  }
  return text;
}

std::string vec(const double * values, std::size_t size)
{
  std::string out = "[";
  for (std::size_t i = 0; i < size; ++i) {
    if (i != 0)
      out += ',';
    out += formatDouble(values[i]);
  }
  return out + "]";
}

template<std::size_t N>
std::string vec(const std::array<double, N> & values)
{
  return vec(values.data(), N);
}

std::string quote(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (byte < 0x20 || byte >= 0x7f) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\x%02x", static_cast<unsigned>(byte));
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string hex(const unsigned char * bytes, std::size_t size)
{
  std::string out;
  char digits[3];
  for (std::size_t i = 0; i < size; ++i) {
    std::snprintf(digits, sizeof(digits), "%02x", static_cast<unsigned>(bytes[i]));
    out += digits;
  }
  return out;
}

std::string hexWord(std::uint32_t value)
{
  char text[16];
  std::snprintf(text, sizeof(text), "0x%x", static_cast<unsigned>(value));
  return text;
}

std::string sha256Hex(const std::string & bytes)
{
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest, &length, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("sha256 unavailable");
  return hex(digest, length);
}

bool readFile(const std::string & path, std::string * out)
{
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return !in.bad();
}

std::vector<std::string> splitLines(const std::string & text)
{
  std::vector<std::string> lines;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

std::int64_t msKey(double seconds)
{
  return static_cast<std::int64_t>(std::llround(seconds * 1000.0));
}

const char * dispositionName(int disposition)
{
  switch (static_cast<fast_lio::ImuDisposition>(disposition)) {
    case fast_lio::ImuDisposition::kUncommitted:
      return "uncommitted";
    case fast_lio::ImuDisposition::kCommitted:
      return "committed";
    case fast_lio::ImuDisposition::kInvalid:
      return "invalid";
    case fast_lio::ImuDisposition::kFatal:
      return "fatal";
  }
  return "unknown";
}

const char * guardName(int guard)
{
  return guard == kGuardNatural ? "natural" : guard == kGuardHealthy ? "healthy" : "degraded";
}

void appendRecord(std::vector<std::string> & out, const fast_lio::AnchorScanRecord & r)
{
  std::ostringstream line;
  const auto flush = [&] {
    out.push_back(line.str());
    line.str("");
  };
  line << "record seq=" << r.scan_seq << " outcome=" << fast_lio::anchorScanOutcomeName(r.outcome)
       << " beg_ns=" << r.beg_ns << " end_ns=" << r.end_ns << " gap_ns=" << r.gap_ns << " published=" << r.published
       << " flags=" << r.flags << " events=" << hexWord(r.events);
  flush();
  line << "record.step anchor_step=" << r.anchor_step << " body=" << fast_lio::anchorTraceBodyName(r.body)
       << " imu_n=" << r.imu_n << " gyr_max=" << formatDouble(r.gyr_max) << " acc_span=" << formatDouble(r.acc_span)
       << " wheel_age=" << formatDouble(r.wheel_age) << " wheel_speed=" << formatDouble(r.wheel_speed)
       << " guard_degraded=" << r.guard_degraded << " degraded=" << r.degraded
       << " gate_pre=" << fast_lio::anchorTraceGateName(r.gate_pre)
       << " gate_post=" << fast_lio::anchorTraceGateName(r.gate_post)
       << " action=" << fast_lio::anchorTraceActionName(r.action) << " parked=" << r.parked
       << " parked_applied=" << r.parked_applied << " zupt_active=" << r.zupt_active
       << " eff_degraded=" << r.eff_degraded << " static_hold=" << r.static_hold;
  flush();
  line << "record.anchor id=" << r.anchor_id << " valid=" << r.anchor_valid << " created_ns=" << r.anchor_created_ns
       << " state=" << fast_lio::anchorStateName(r.anchor_state) << " episode=" << r.episode << " ref_gen=" << r.ref_gen
       << " used=" << r.anchor_used << " streak=" << r.streak << " have_pose=" << r.have_anchor_pose
       << " p=" << vec(r.anchor_p, 3) << " q=" << vec(r.anchor_q, 4);
  flush();
  line << "record.evidence evidence=" << fast_lio::staticEvidenceName(r.evidence)
       << " reason=" << fast_lio::staticEvidenceReasonName(r.evidence_reason) << " imu_only=" << r.imu_only
       << " wheel_valid=" << r.wheel_valid << " usable=" << r.usable << " zero_only=" << r.zero_only
       << " supported=" << r.supported;
  flush();
  line << "record.recovery trust_pre=" << fast_lio::recoveryTrustName(r.trust_pre)
       << " trust_post=" << fast_lio::recoveryTrustName(r.trust_post) << " episode=" << r.recovery_episode
       << " start_ns=" << r.recovery_start_ns << " deadline_ns=" << r.recovery_deadline_ns
       << " startup_reference=" << fast_lio::startupReferenceName(r.startup_reference);
  flush();
  line << "record.pre have=" << r.have_pre << " p=" << vec(r.pre_p, 3) << " q=" << vec(r.pre_q, 4)
       << " v=" << vec(r.pre_v, 3);
  flush();
  line << "record.act have=" << r.have_act << " p=" << vec(r.act_p, 3) << " q=" << vec(r.act_q, 4)
       << " v=" << vec(r.act_v, 3);
  if (r.have_act)
    line << " pose_cov=" << vec(r.p_pose_act, 36);
  flush();
  line << "record.post have=" << r.have_post << " p=" << vec(r.post_p, 3) << " q=" << vec(r.post_q, 4)
       << " v=" << vec(r.post_v, 3);
  flush();
  line << "record.target have=" << r.have_target << " p=" << vec(r.target_p, 3) << " q=" << vec(r.target_q, 4);
  flush();
  line << "record.cov have=" << r.have_cov << " finite=" << r.p_finite << " sym=" << formatDouble(r.p_sym)
       << " min_eig=" << formatDouble(r.p_min_eig);
  if (r.have_cov)
    line << " pose_pre=" << vec(r.p_pose_pre, 36) << " pose_post=" << vec(r.p_pose_post, 36);
  flush();
  line << "record.update map_updated=" << r.map_updated << " have_update=" << r.have_update
       << " dp=" << formatDouble(r.upd_dp) << " dq=" << formatDouble(r.upd_dq);
  flush();
  const auto & h = r.reference_health;
  line << "record.reference present=" << h.present << " snapshot=" << h.snapshotSequence
       << " revision=" << h.stateRevision << " scan=" << h.scanSequence << " qualified=" << h.qualifiedSequence
       << " kind=" << static_cast<unsigned>(h.kind) << " lifecycle=" << static_cast<unsigned>(h.lifecycle)
       << " startup=" << static_cast<unsigned>(h.startup) << " trust=" << static_cast<unsigned>(h.trust)
       << " local=" << static_cast<unsigned>(h.local) << " hr_kind=" << static_cast<unsigned>(h.hrKind)
       << " legacy_flags=" << h.legacyFlags << " present_mask=" << h.presentMask << " true_mask=" << h.trueMask
       << " cause=" << h.cause << " taint=" << h.taint << " hard=" << h.hard << " has_pose=" << h.hasPose
       << " evaluated_ns=" << h.evaluatedNs << " effective_ns=" << h.effectiveNs << " state_until_ns=" << h.stateUntilNs
       << " hr_until_ns=" << h.hrUntilNs;
  flush();
}

void appendOdometry(std::vector<std::string> & out, const nav_msgs::msg::Odometry & m)
{
  const auto & pose = m.pose.pose;
  const auto & twist = m.twist.twist;
  out.push_back("odom stamp_ns=" + std::to_string(rclcpp::Time(m.header.stamp).nanoseconds()) +
                " frame=" + quote(m.header.frame_id) + " child=" + quote(m.child_frame_id));
  const double p[3] = {pose.position.x, pose.position.y, pose.position.z};
  const double q[4] = {pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
  out.push_back("odom.pose p=" + vec(p, 3) + " q_xyzw=" + vec(q, 4));
  out.push_back("odom.pose_cov " + vec(m.pose.covariance));
  const double linear[3] = {twist.linear.x, twist.linear.y, twist.linear.z};
  const double angular[3] = {twist.angular.x, twist.angular.y, twist.angular.z};
  out.push_back("odom.twist linear=" + vec(linear, 3) + " angular=" + vec(angular, 3));
  out.push_back("odom.twist_cov " + vec(m.twist.covariance));
}

// The value of ` key=` in a trace line, or "" when the line has none.
std::string valueOf(const std::string & line, const std::string & key)
{
  const std::string tag = " " + key + "=";
  const std::size_t at = line.find(tag);
  if (at == std::string::npos)
    return "";
  const std::size_t start = at + tag.size();
  return line.substr(start, line.find(' ', start) - start);
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

enum class Cloud : std::uint8_t
{
  kDense,   // the fixture's plane, 2000 points over the scan
  kFar,     // the same plane 30 m further along x: no map neighbour for any point
  kSparse,  // three points: too few to register, the scan leaves before the anchor step
  kEmpty,   // no point: no scan timing
  kNone,    // no scan this period
};

const char * cloudName(Cloud cloud)
{
  switch (cloud) {
    case Cloud::kDense:
      return "dense";
    case Cloud::kFar:
      return "far";
    case Cloud::kSparse:
      return "sparse";
    case Cloud::kEmpty:
      return "empty";
    case Cloud::kNone:
      return "none";
  }
  return "unknown";
}

// One scan period.
struct Slot
{
  std::string label;
  bool moving = false;  // the IMU samples of this scan's window turn at kMovingYawRate
  Cloud cloud = Cloud::kDense;
  int guard = kGuardHealthy;
  bool inject = false;  // beforeAnchorStep writes `pos`, kFixtureVelocity and kFixtureCovDiag
  std::array<double, 3> pos = {0.0, 0.0, 0.0};
  bool trip = false;       // forceRunawayTrip trips the watchdog on this scan
  bool imu_throw = false;  // beforeImuProcess throws for this scan
  // IMU samples strictly inside (scan end + hole_from, scan end + hole_to) are not sent; equal bounds, no hole. The
  // samples up to 6 ms past the previous scan's end went out with that scan.
  double hole_from = 0.0;
  double hole_to = 0.0;
  bool bad_imu = false;     // the IMU samples in the last 4 ms before the scan end carry a non-finite yaw rate
  bool clock_back = false;  // an IMU sample stamped 1.5 s before the newest one precedes the scan: an epoch fault
  int repeat = 0;           // > 0: the cloud of the scan this many slots back is sent once more instead of a new one
  bool refused = false;     // the node refuses this slot's cloud where it arrives: it leaves no record
};

struct Scenario
{
  std::string name;
  bool async_map = false;
  std::map<std::string, rclcpp::Parameter> params;
  std::vector<Slot> slots;          // after the warm-up
  std::vector<std::string> claims;  // the branches the scenario exists to reach
};

Slot slot(const std::string & label, bool moving = false, int guard = kGuardHealthy, Cloud cloud = Cloud::kDense)
{
  Slot s;
  s.label = label;
  s.moving = moving;
  s.guard = guard;
  s.cloud = cloud;
  return s;
}

Slot injected(const std::string & label, bool moving, int guard, const std::array<double, 3> & pos, bool trip = false)
{
  Slot s = slot(label, moving, guard);
  s.inject = true;
  s.pos = pos;
  s.trip = trip;
  return s;
}

void append(Scenario & scenario, int count, const Slot & what)
{
  for (int i = 0; i < count; ++i)
    scenario.slots.push_back(what);
}

void setParam(Scenario & scenario, const rclcpp::Parameter & parameter)
{
  scenario.params[parameter.get_name()] = parameter;
}

void setPolicy(Scenario & scenario, double total, double stable)
{
  setParam(scenario, rclcpp::Parameter("reanchor_total_timeout_sec", total));
  setParam(scenario, rclcpp::Parameter("reanchor_recovery_stable_sec", stable));
}

// test_anchor_lifecycle_node's base configuration, with the anchor trace on.
Scenario baseScenario(const std::string & name)
{
  Scenario s;
  s.name = name;
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
        rclcpp::Parameter("reanchor_en", false),
        rclcpp::Parameter("runaway_watchdog_en", false),
        rclcpp::Parameter("zupt_en", true),
        rclcpp::Parameter("zupt_parked_hold_en", false),
        rclcpp::Parameter("zupt_anchor_trace_en", true)}) {
    setParam(s, p);
  }
  return s;
}

// Bootstrap and the first scans (the warm-up), ordinary matching while the IMU turns, a scan far from every map
// point (no match), a scan with too few points to register, and parked scans: nothing forced but the warm-up's guard.
Scenario matchScenario()
{
  Scenario s = baseScenario("match");
  append(s, 4, slot("moving", true, kGuardNatural));
  s.slots.push_back(slot("far_cloud", true, kGuardNatural, Cloud::kFar));
  append(s, 3, slot("moving", true, kGuardNatural));
  s.slots.push_back(slot("sparse_cloud", false, kGuardNatural, Cloud::kSparse));
  append(s, 4, slot("static", false, kGuardNatural));
  s.claims = {"bootstrap", "imu_commit", "normal_match", "no_match", "early_exit_scan", "published", "odometry"};
  return s;
}

// test_anchor_lifecycle_node's prior_late_reset_zupt_on: an anchor from injected registration states, two parked
// scans 0.36 m from it that take the parked prior, and a runaway map reset after the anchor step of a parked scan.
Scenario parkedPriorScenario()
{
  Scenario s = baseScenario("parked_prior_reset");
  setParam(s, rclcpp::Parameter("zupt_parked_hold_en", true));
  setParam(s, rclcpp::Parameter("zupt_parked_hold_scans", 3));
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  s.slots.push_back(injected("anchor_move", true, kGuardHealthy, kAnchorPosition));
  append(s, 4, injected("anchor_static", false, kGuardHealthy, kAnchorPosition));
  append(s, 2, injected("parked", false, kGuardHealthy, kNudgedPosition));
  s.slots.push_back(injected("parked_reset", false, kGuardHealthy, kNudgedPosition, true));
  append(s, 2, injected("after_reset", false, kGuardHealthy, kNudgedPosition));
  s.claims = {"anchor_created", "parked_prior", "runaway_trip", "map_reset", "published", "odometry"};
  return s;
}

// test_anchor_lifecycle_node's pin_late_reset: the anchor, degraded parked scans it pins, and a runaway map reset
// after the anchor step of a pinned scan.
Scenario pinScenario()
{
  Scenario s = baseScenario("pin_reset");
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  s.slots.push_back(injected("anchor_move", true, kGuardHealthy, kAnchorPosition));
  append(s, 4, injected("anchor_static", false, kGuardHealthy, kAnchorPosition));
  append(s, 2, injected("hold", false, kGuardDegraded, kNudgedPosition));
  s.slots.push_back(injected("hold_reset", false, kGuardDegraded, kNudgedPosition, true));
  append(s, 2, injected("after_reset", false, kGuardDegraded, kNudgedPosition));
  s.claims = {"anchor_created", "anchor_pin", "runaway_trip", "map_reset"};
  return s;
}

// Scans dropped where they are synced: a cloud without a point (no scan timing), the previous scan's cloud once more
// (it does not advance), and a scan whose IMU window holds no sample (test_anchor_lifecycle_node's empty_window: the
// first scan of the hole keeps the sample 2.5 ms past the previous end, the second has none, and its borrowed IMU
// prefix is left in the buffer).
Scenario scanDropsScenario()
{
  Scenario s = baseScenario("scan_drops");
  append(s, 2, slot("static"));
  s.slots.push_back(slot("empty_cloud", false, kGuardHealthy, Cloud::kEmpty));
  append(s, 2, slot("static"));
  Slot repeat = slot("repeated_cloud");
  repeat.repeat = 1;
  s.slots.push_back(repeat);
  append(s, 2, slot("static"));
  Slot hole = slot("imu_hole");
  hole.hole_from = -0.094;
  hole.hole_to = 0.101;
  s.slots.push_back(hole);
  s.slots.push_back(slot("empty_imu_window"));
  append(s, 3, slot("static"));
  s.claims = {"invalid_scan", "nonadvancing_scan", "imu_retain_empty_window"};
  return s;
}

// test_anchor_lifecycle_node's tail_gap: the IMU stops 35 ms before a scan's end, beyond the 20 ms the coverage
// policy extrapolates.
Scenario imuCoverageScenario()
{
  Scenario s = baseScenario("imu_coverage_gap");
  setPolicy(s, 10.0, 1.0);
  setParam(s, rclcpp::Parameter("reanchor_en", true));
  setParam(s, rclcpp::Parameter("imu_coverage.max_extrapolation_s", 0.02));
  append(s, 5, slot("static"));
  Slot gap = slot("imu_tail_gap");
  gap.hole_from = -0.035;
  gap.hole_to = 0.0;
  s.slots.push_back(gap);
  append(s, 3, slot("static", false, kGuardDegraded));
  s.claims = {"imu_coverage_gap"};
  return s;
}

// test_anchor_lifecycle_node's invalid_sample and scan_hole_static one after the other: a non-finite IMU rate in the
// last 4 ms of a scan's window (an invalid batch, the scan not processed), fourteen scans, then five scan periods
// without a scan and the recovery after them.
Scenario imuInvalidScanHoleScenario()
{
  Scenario s = baseScenario("imu_invalid_scan_hole");
  setPolicy(s, 10.0, 1.0);
  append(s, 5, slot("static"));
  Slot bad = slot("invalid_imu_sample");
  bad.bad_imu = true;
  s.slots.push_back(bad);
  append(s, 14, slot("static"));
  append(s, 5, slot("scan_hole", false, kGuardHealthy, Cloud::kNone));
  append(s, 14, slot("static"));
  s.claims = {"imu_invalid", "imu_not_processed", "scan_hole", "recovery_started", "recovery_qualified"};
  return s;
}

// test_ros_scan_bounds' fatal transaction, on a scenario's stream: the IMU processing step throws for one scan (the
// batch settles fatal, the scan is an IMU fault), and the front end refuses the next slot's input.
Scenario imuFatalScenario()
{
  Scenario s = baseScenario("imu_fatal");
  append(s, 2, slot("static"));
  Slot fatal = slot("imu_processing_throws");
  fatal.imu_throw = true;
  s.slots.push_back(fatal);
  Slot after = slot("refused_after_fault");
  after.refused = true;
  s.slots.push_back(after);
  s.claims = {"imu_fatal", "imu_fault_scan", "refused_input"};
  return s;
}

// test_anchor_lifecycle_node's input_epoch_fault: an IMU sample 1.5 s older than the newest one ends the input epoch;
// the front end refuses that slot's scan and every later one.
Scenario inputEpochScenario()
{
  Scenario s = baseScenario("input_epoch");
  append(s, 5, slot("static"));
  Slot event = slot("clock_back");
  event.clock_back = true;
  event.refused = true;
  s.slots.push_back(event);
  Slot after = slot("refused_after_epoch");
  after.refused = true;
  append(s, 3, after);
  s.claims = {"input_epoch", "refused_input"};
  return s;
}

// The map worker on (FLIO_ASYNC_MAP=1) with a two-scan insertion delay, and a runaway map reset in the middle: the
// worker's additions, the batches the delay holds back, and the reset's join and discard, on the production path.
Scenario workerScenario()
{
  Scenario s = baseScenario("worker_delay_reset");
  s.async_map = true;
  setParam(s, rclcpp::Parameter("mapping.map_insert_delay_scans", 2));
  setParam(s, rclcpp::Parameter("runaway_watchdog_en", true));
  setParam(s, rclcpp::Parameter("runaway_on_trip", "reset_map"));
  append(s, 4, slot("moving", true, kGuardNatural));
  append(s, 4, slot("static", false, kGuardNatural));
  Slot reset = slot("runaway_reset", false, kGuardNatural);
  reset.trip = true;
  s.slots.push_back(reset);
  append(s, 6, slot("static", false, kGuardNatural));
  s.claims = {"async_map_add", "delayed_insertion", "runaway_trip", "map_reset", "published", "odometry"};
  return s;
}

constexpr double kEntryViewStep = 0.04;  // m along x per scan: the fixture's 0.4 m/s

// reanchor_recovery_entry_view on (#369), test_anchor_lifecycle_node's entry_view_moving: the platform moves on along
// x through a recovery, its registration states injected. One healthy scan, the last supported one; three degraded
// scans, which start Pending and stay out of the map; healthy scans, which go into the map while Pending and are
// matched against the entry view until it qualifies the recovery after the stable time, and go on after it.
Scenario entryViewScenario()
{
  Scenario s = baseScenario("entry_view_moving");
  setPolicy(s, 10.0, 1.0);
  setParam(s, rclcpp::Parameter("reanchor_recovery_entry_view", true));
  int k = 0;
  const auto along = [&k]() {
    return std::array<double, 3>{kEntryViewStep * k++, 0.0, 0.0};
  };
  s.slots.push_back(injected("last_supported", true, kGuardHealthy, along()));
  for (int i = 0; i < 3; ++i)
    s.slots.push_back(injected("degraded_moving", true, kGuardDegraded, along()));
  for (int i = 0; i < 15; ++i)
    s.slots.push_back(injected("healthy_moving", true, kGuardHealthy, along()));
  s.claims = {"recovery_started", "recovery_qualified", "pending_map_update", "published", "odometry"};
  return s;
}

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

struct Branch
{
  const char * name;
  const char * counted;    // what the count is
  const char * reference;  // the existing suite that asserts the branch's semantics, or ""
};

constexpr Branch kBranches[] = {
  {"bootstrap", "records with outcome bootstrap (the first scan)", ""},
  {"imu_commit", "IMU batches settled committed (afterImuConsumption)", ""},
  {"imu_invalid",
   "IMU batches settled invalid",
   "test_anchor_lifecycle_node AnchorLifecycleInputs.AnImuStretchNothingObservedEndsTheAnchorAndStartsTheRecovery"},
  {"imu_fatal",
   "IMU batches settled fatal",
   "test_ros_scan_bounds RosScanBounds.FatalTransactionRejectsRealInputsAndBoundsBothQueues"},
  {"imu_uncommitted_hook",
   "IMU batches settled uncommitted through afterImuConsumption (see the suspended entries)",
   ""},
  {"imu_retain_empty_window",
   "records with outcome dropped_empty_imu: the borrowed prefix is settled uncommitted and stays in the buffer",
   "test_anchor_lifecycle_node AnchorLifecycleInputs.AnImuStretchNothingObservedEndsTheAnchorAndStartsTheRecovery"},
  {"imu_not_processed", "records with outcome imu_not_processed", ""},
  {"imu_coverage_gap",
   "records with outcome imu_coverage_gap",
   "test_anchor_lifecycle_node AnchorLifecycleNode.CoverageGapBeforeTheAnchorStepEndsTheAnchor"},
  {"imu_fault_scan", "records with outcome imu_fault", ""},
  {"normal_match",
   "scans whose H-model iterations found at least one effective point",
   "test_h_model_scan_bounds HModelScanBounds.*"},
  {"no_match", "scans whose H-model iterations found no effective point", ""},
  {"early_exit_scan",
   "records with outcome before_anchor_step",
   "test_anchor_lifecycle_node AnchorLifecycleEvidence.AScanThatLeavesBeforeTheAnchorStepEndsTheCandidate"},
  {"published", "records of published scans", ""},
  {"odometry", "/Odometry messages matched to a published record", ""},
  {"invalid_scan", "records with outcome dropped_invalid_time", ""},
  {"nonadvancing_scan",
   "records with outcome dropped_nonadvancing",
   "test_anchor_lifecycle_node AnchorLifecycleInputs.ARepeatedScanTakesNoImuAndTheScanThatTakesItSeesItsMotion"},
  {"anchor_created", "counter summary: created", "test_anchor_lifecycle_node AnchorLifecycleNode.*"},
  {"parked_prior",
   "counter summary: parked_prior_applied",
   "test_anchor_lifecycle_node AnchorLifecycleNode.F1F2F3LateResetLeavesNoOldAnchorAndNoStaticHold"},
  {"anchor_pin",
   "counter summary: pin_applied",
   "test_anchor_lifecycle_node AnchorLifecycleNode.LateResetAfterAPinEndsTheAnchorAndItsStaticHold"},
  {"scan_hole",
   "counter summary: scan_hole",
   "test_anchor_lifecycle_node AnchorLifecycleEvidence.AScanHoleIsFoundByTheNextScan"},
  {"recovery_started", "counter summary: recovery_started", "test_anchor_lifecycle_node AnchorLifecycleNode.*"},
  {"recovery_qualified", "counter summary: recovery_qualified", "test_anchor_lifecycle_node AnchorLifecycleNode.*"},
  {"input_epoch",
   "counter summary: time_epoch",
   "test_anchor_lifecycle_node AnchorLifecycleInputs.AnInputClockFaultEndsTheAnchorAndStopsTheScanPath"},
  {"refused_input", "slots whose cloud the node refused (rejected clouds rose)", ""},
  {"runaway_trip", "scans forceRunawayTrip tripped", ""},
  {"map_reset",
   "map resets (afterPriorMapReset)",
   "test_async_map_reset AsyncMapReset.*; test_anchor_lifecycle_node AnchorLifecycleRebuild.*"},
  {"async_map_add", "batches the map worker added (beforeMapAdd, worker suite only)", "test_async_map_reset"},
  {"delayed_insertion",
   "published map updates whose batch the worker had not added when the next scan was prepared (worker suite only)",
   "test_async_map_reset AsyncMapReset.EveryBatchOfAMultiScanDelayIsDiscarded"},
  {"pending_map_update",
   "records of scans still Pending after the recovery step whose points went into the map "
   "(reanchor_recovery_entry_view)",
   "test_anchor_lifecycle_node AnchorLifecycleEntryView.*"},
};

// What the existing hooks cannot reach. Each entry names the hook S1 adds under FASTLIO_TEST_HOOKS and where; the
// line numbers are those of laserMapping.cpp at fork base 0afda1d.
struct Suspended
{
  const char * branch;
  const char * missing;
  const char * hook;
};

constexpr Suspended kSuspended[] = {
  {"h_matrix_values",
   "afterHModelRows reports row counts only: the measurement Jacobian rows (ekfom_data.h_x) and residuals "
   "(ekfom_data.h) of each iteration are not reachable",
   "afterHModelMatrices(const Eigen::MatrixXd & h_x, const Eigen::VectorXd & h, double lidar_end_time) in "
   "h_share_model, right before 'solve_time += omp_get_wtime() - solve_start_;' (L4162), and with empty matrices at "
   "the no-effective-point exit (L3947-3949)"},
  {"full_state_and_covariance",
   "the record has position, attitude, velocity and the 6x6 pose covariance around the anchor step, /Odometry the "
   "6x6 pose covariance; the 23-dimensional state (extrinsics, biases, gravity) and the 23x23 P after the iterated "
   "update are not reachable",
   "afterLidarUpdate(const state_ikfom & x, const Eigen::Matrix<double, 23, 23> & P, double lidar_end_time) right "
   "after kf.update_iterated_dyn_share_modified(...) (L6124-6125), before state_point = kf.get_x() (L6128)"},
  {"typed_outputs_with_fixed_identity",
   "FrontendObservation and ReferenceHealth carry the activation's random source instance (getrandom, L4347), and "
   "the reference repeat runs on a steady-clock period (publishReferenceRepeat, L1782): no hook fixes either",
   "overrideFrontendInstance(std::array<std::uint8_t, 16> & instance) after the getrandom loop (L4354), and a fixture "
   "clock for publishReferenceRepeat (L1782); test_anchor_lifecycle_node asserts the typed outputs' semantics"},
  {"imu_retain_hook_visible",
   "an empty IMU window settles the borrowed prefix uncommitted in sync_packages without afterImuConsumption, so the "
   "retained batch shows only as the record outcome dropped_empty_imu and in the input snapshot",
   "afterImuConsumption(sequence, kUncommitted, reason, 0, pending, first_id) right after "
   "imu_fifo_transaction.settle(..., ImuDisposition::kUncommitted) in sync_packages (L2702)"},
};

// ---------------------------------------------------------------------------
// The child: one component activation, one scenario
// ---------------------------------------------------------------------------

class PipelineRunner
{
public:
  explicit PipelineRunner(const Scenario & scenario) : scenario_(scenario) {}

  std::string run()
  {
    // The map worker as the scenario says; one OpenMP thread; nothing else from the environment.
    ::setenv("FLIO_ASYNC_MAP", scenario_.async_map ? "1" : "0", 1);
    ::setenv("OMP_NUM_THREADS", "1", 1);
    for (const char * name : {"FLIO_OFFLINE_STEP", "FLIO_LIDAR_QDEPTH", "FLIO_IKD_PROFILE"})
      ::unsetenv(name);
    main_thread_ = std::this_thread::get_id();
    rclcpp::init(0, nullptr);
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    loader_ = std::make_unique<class_loader::ClassLoader>(FASTLIO_TEST_COMPONENT_PATH);
    const auto names = loader_->getAvailableClasses<rclcpp_components::NodeFactory>();
    std::string error;
    if (std::find(names.begin(), names.end(), kLaserMappingFactory) == names.end())
      error = "LaserMappingNode factory not found";
    if (error.empty()) {
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
      observer_ = std::make_shared<rclcpp::Node>("frontend_pipeline_contract_observer", observer_options);
      imu_ =
        observer_->create_publisher<sensor_msgs::msg::Imu>(kImuTopic, rclcpp::QoS(rclcpp::KeepLast(1000)).reliable());
      lidar_ = observer_->create_publisher<sensor_msgs::msg::PointCloud2>(kLidarTopic,
                                                                          rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
      odom_ = observer_->create_subscription<nav_msgs::msg::Odometry>(
        "/Odometry",
        rclcpp::QoS(rclcpp::KeepLast(50)).reliable(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr message) {
          odom_by_ms_[msKey(rclcpp::Time(message->header.stamp).seconds())] = *message;
        });
      executor_->add_node(subject_->get_node_base_interface());
      executor_->add_node(observer_);
      if (!spinUntil([&] { return imu_->get_subscription_count() == 1 && lidar_->get_subscription_count() == 1; }))
        error = "front end subscriptions never matched";
    }
    if (error.empty())
      error = warmUp();
    for (std::size_t i = 0; error.empty() && i < scenario_.slots.size(); ++i) {
      const bool next_moving = i + 1 < scenario_.slots.size() && scenario_.slots[i + 1].moving;
      error = runSlot(scenario_.slots[i], next_moving);
    }
    if (error.empty())
      error = settle(true);
    teardown();
    finish();
    return error;
  }

  std::string record() const
  {
    std::string out;
    for (const auto & line : stream_)
      out += line + '\n';
    return out;
  }

  std::string branches() const
  {
    std::ostringstream out;
    for (const auto & branch : kBranches) {
      const auto it = counts_.find(branch.name);
      out << "branch " << branch.name << ' ' << (it == counts_.end() ? 0u : it->second) << '\n';
    }
    std::lock_guard<std::mutex> lock(off_thread_mutex_);
    for (const auto & entry : off_thread_)
      out << "off_thread " << entry.first << ' ' << entry.second << '\n';
    return out.str();
  }

private:
  struct PendingScan
  {
    int scan = 0;
    int slot = 0;
    bool needs_imu = true;
    std::uint64_t after_seq = 0;  // a repeated cloud's record comes after the records there were when it was sent
  };

  bool onMainThread() const { return std::this_thread::get_id() == main_thread_; }

  // Appends an event of the executor thread to the record. A hook that runs on another thread is only counted,
  // outside the record: its place in the stream would depend on scheduling.
  bool note(const char * hook, const std::string & text)
  {
    if (!onMainThread()) {
      std::lock_guard<std::mutex> lock(off_thread_mutex_);
      ++off_thread_[hook];
      return false;
    }
    stream_.push_back(text);
    return true;
  }

  const Slot * slotAt(double end) const
  {
    const auto it = slot_by_end_ms_.find(msKey(end));
    return it == slot_by_end_ms_.end() ? nullptr : &slots_[static_cast<std::size_t>(it->second)];
  }

  void installHooks()
  {
    // The first call of every scan that has its input; its answer schedules no reset.
    fast_lio_test::resetBeforeReference = [this](double end) {
      note("resetBeforeReference", "scan_begin end=" + formatDouble(end));
      return false;
    };
    fast_lio_test::beforeImuProcess = [this] {
      if (note("beforeImuProcess", "imu_process") && throw_next_imu_process_) {
        throw_next_imu_process_ = false;
        throw std::runtime_error("frontend pipeline contract: injected IMU processing failure");
      }
    };
    fast_lio_test::afterImuConsumption = [this](std::uint64_t sequence,
                                                int disposition,
                                                int reason,
                                                std::size_t samples,
                                                std::size_t pending,
                                                std::uint64_t first_id) {
      std::ostringstream out;
      out << "imu_batch seq=" << sequence << " disposition=" << dispositionName(disposition)
          << " reason=" << fast_lio::imuProcessReasonName(static_cast<fast_lio::ImuProcessReason>(reason))
          << " samples=" << samples << " pending=" << pending << " first_id=" << first_id;
      if (!note("afterImuConsumption", out.str()))
        return;
      switch (static_cast<fast_lio::ImuDisposition>(disposition)) {
        case fast_lio::ImuDisposition::kUncommitted:
          ++counts_["imu_uncommitted_hook"];
          break;
        case fast_lio::ImuDisposition::kCommitted:
          ++counts_["imu_commit"];
          break;
        case fast_lio::ImuDisposition::kInvalid:
          ++counts_["imu_invalid"];
          break;
        case fast_lio::ImuDisposition::kFatal:
          ++counts_["imu_fatal"];
          break;
      }
    };
    fast_lio_test::afterScanPrepared = [this](std::size_t count, double end) {
      std::ostringstream out;
      out << "scan_prepared count=" << count << " end=" << formatDouble(end);
      if (scenario_.async_map && onMainThread()) {
        // The scan path has joined the worker before it prepares a scan: every batch handed over before this scan
        // has been added, and this scan has handed over none yet.
        const std::uint64_t adds = map_adds_.load(std::memory_order_acquire);
        out << " map_adds=" << adds;
        if (update_awaiting_add_ && adds == adds_before_update_)
          ++counts_["delayed_insertion"];
        update_awaiting_add_ = false;
        adds_at_prepare_ = adds;
      }
      note("afterScanPrepared", out.str());
    };
    fast_lio_test::afterHModelRows = [this](std::size_t feats, double end, int effective) {
      std::ostringstream out;
      out << "h_rows feats=" << feats << " end=" << formatDouble(end) << " effective=" << effective;
      if (!note("afterHModelRows", out.str()))
        return;
      const auto key = msKey(end);
      const auto it = h_max_effective_.find(key);
      if (it == h_max_effective_.end())
        h_max_effective_[key] = effective;
      else
        it->second = std::max(it->second, effective);
    };
    fast_lio_test::overrideGuardDegraded = [this](double end) {
      const Slot * s = slotAt(end);
      const int forced = s == nullptr ? kGuardHealthy : s->guard;
      note("overrideGuardDegraded", "guard end=" + formatDouble(end) + " forced=" + guardName(forced));
      return forced;
    };
    fast_lio_test::forceRunawayTrip = [this](double end) {
      const Slot * s = slotAt(end);
      const bool trip = s != nullptr && s->trip;
      if (note("forceRunawayTrip", "runaway_check end=" + formatDouble(end) + " trip=" + (trip ? "1" : "0")) && trip)
        ++counts_["runaway_trip"];
      return trip;
    };
    // Installed only where a slot injects: an installed hook makes the scan path re-read its state after it.
    if (std::any_of(scenario_.slots.begin(), scenario_.slots.end(), [](const Slot & s) { return s.inject; })) {
      fast_lio_test::beforeAnchorStep = [this](double end, fast_lio_test::AnchorFixtureState & fixture) {
        const Slot * s = slotAt(end);
        const bool inject = s != nullptr && s->inject;
        note("beforeAnchorStep", "anchor_step_fixture end=" + formatDouble(end) + " inject=" + (inject ? "1" : "0"));
        if (!inject)
          return;
        fixture.overwrite = true;
        for (int i = 0; i < 3; ++i) {
          fixture.pos[i] = s->pos[static_cast<std::size_t>(i)];
          fixture.vel[i] = kFixtureVelocity[static_cast<std::size_t>(i)];
        }
        fixture.rot_wxyz[0] = 1.0;
        fixture.rot_wxyz[1] = fixture.rot_wxyz[2] = fixture.rot_wxyz[3] = 0.0;
        fixture.overwrite_cov = true;
        for (std::size_t i = 0; i < kFixtureCovDiag.size(); ++i)
          fixture.cov_diag[i] = kFixtureCovDiag[i];
      };
    }
    fast_lio_test::initialBuildTakesNoPoint = [this](double end) {
      note("initialBuildTakesNoPoint", "initial_build end=" + formatDouble(end));
      return false;
    };
    fast_lio_test::beforeResetJoin = [this] {
      note("beforeResetJoin", "map_reset_join");
    };
    fast_lio_test::afterPendingReset = [this] {
      note("afterPendingReset", "map_reset_pending_cleared");
    };
    fast_lio_test::afterPriorMapReset = [this](bool before, bool after, bool lost, std::size_t pinned) {
      std::ostringstream out;
      out << "map_reset prior_installed_before=" << before << " prior_installed_after=" << after
          << " required_prior_lost=" << lost << " pinned_voxels=" << pinned;
      if (note("afterPriorMapReset", out.str()))
        ++counts_["map_reset"];
    };
    fast_lio_test::afterAnchorScanRecord = [this](const fast_lio::AnchorScanRecord & record) {
      if (!onMainThread()) {
        std::lock_guard<std::mutex> lock(off_thread_mutex_);
        ++off_thread_["afterAnchorScanRecord"];
        return;
      }
      onRecord(record);
    };
    fast_lio_test::afterAnchorTraceLine = [this](const std::string & line) {
      if (line.rfind("[ZUPT-ANCHOR-SCAN]", 0) == 0)
        return;  // every field of it is in the record
      if (note("afterAnchorTraceLine", "trace " + quote(line)) && valueOf(line, "event") == "counters")
        counters_line_ = line;
    };
    fast_lio_test::afterPosePublish =
      [this](std::uint8_t encoding, std::int64_t stamp, const std::array<std::uint8_t, 32> & id) {
        note("afterPosePublish",
             "pose_publish encoding=" + std::to_string(encoding) + " stamp_ns=" + std::to_string(stamp) +
               " observation_id=" + hex(id.data(), id.size()));
      };
    // The worker's hook: a count, read where the scan path has joined the worker.
    fast_lio_test::beforeMapAdd = [this] {
      map_adds_.fetch_add(1, std::memory_order_acq_rel);
    };
  }

  static void clearHooks()
  {
    fast_lio_test::resetBeforeReference = {};
    fast_lio_test::beforeImuProcess = {};
    fast_lio_test::afterImuConsumption = {};
    fast_lio_test::afterScanPrepared = {};
    fast_lio_test::afterHModelRows = {};
    fast_lio_test::overrideGuardDegraded = {};
    fast_lio_test::forceRunawayTrip = {};
    fast_lio_test::beforeAnchorStep = {};
    fast_lio_test::initialBuildTakesNoPoint = {};
    fast_lio_test::beforeResetJoin = {};
    fast_lio_test::afterPendingReset = {};
    fast_lio_test::afterPriorMapReset = {};
    fast_lio_test::afterAnchorScanRecord = {};
    fast_lio_test::afterAnchorTraceLine = {};
    fast_lio_test::afterPosePublish = {};
    fast_lio_test::beforeMapAdd = {};
  }

  void onRecord(const fast_lio::AnchorScanRecord & r)
  {
    records_.push_back(r);
    appendRecord(stream_, r);
    switch (r.outcome) {
      case fast_lio::AnchorScanOutcome::kBootstrap:
        ++counts_["bootstrap"];
        break;
      case fast_lio::AnchorScanOutcome::kImuFault:
        ++counts_["imu_fault_scan"];
        break;
      case fast_lio::AnchorScanOutcome::kImuNotProcessed:
        ++counts_["imu_not_processed"];
        break;
      case fast_lio::AnchorScanOutcome::kImuCoverageGap:
        ++counts_["imu_coverage_gap"];
        break;
      case fast_lio::AnchorScanOutcome::kBeforeAnchorStep:
        ++counts_["early_exit_scan"];
        break;
      case fast_lio::AnchorScanOutcome::kDroppedInvalidTime:
        ++counts_["invalid_scan"];
        break;
      case fast_lio::AnchorScanOutcome::kDroppedNonadvancing:
        ++counts_["nonadvancing_scan"];
        break;
      case fast_lio::AnchorScanOutcome::kDroppedEmptyImu:
        ++counts_["imu_retain_empty_window"];
        break;
      default:
        break;
    }
    const std::int64_t key = msKey(static_cast<double>(r.end_ns) * 1e-9);
    if (r.published) {
      ++counts_["published"];
      published_ends_.push_back(key);
    }
    if (r.trust_post == fast_lio::RecoveryTrust::kPending && r.map_updated)
      ++counts_["pending_map_update"];
    const auto rows = h_max_effective_.find(key);
    if (r.end_ns >= 0 && rows != h_max_effective_.end()) {
      ++counts_[rows->second > 0 ? "normal_match" : "no_match"];
      h_max_effective_.erase(rows);
    }
    if (scenario_.async_map && r.outcome == fast_lio::AnchorScanOutcome::kPublished && r.map_updated && r.have_update) {
      update_awaiting_add_ = true;
      adds_before_update_ = adds_at_prepare_;
    }
  }

  template<class Predicate>
  bool spinUntil(Predicate ready, double seconds = kScanWaitSeconds)
  {
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!ready() && std::chrono::steady_clock::now() < end)
      executor_->spin_once(std::chrono::milliseconds(2));
    return ready();
  }

  double scanStart(int scan) const { return kFirstScan + kScanPeriod * scan; }
  double scanEnd(int scan) const { return scanStart(scan) + kScanLength; }

  // The scan whose IMU window (previous end, this end] holds time t.
  int scanOfImuTime(double t) const
  {
    const double k = std::ceil((t - kFirstScan - kScanLength) / kScanPeriod - 1e-9);
    return static_cast<int>(std::max(0.0, k));
  }

  bool movingAt(int scan) const
  {
    const auto it = moving_by_scan_.find(scan);
    return it != moving_by_scan_.end() && it->second;
  }

  bool inImuHole(double t) const
  {
    for (const auto & hole : imu_holes_)
      if (t > hole.first && t < hole.second)
        return true;
    return false;
  }

  // The timer the front end runs its scan path from: the one timer of the node with that period.
  rclcpp::TimerBase::SharedPtr scanPathTimer()
  {
    if (scan_path_timer_)
      return scan_path_timer_;
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
    if (found == 1)
      scan_path_timer_ = match;
    return scan_path_timer_;
  }

  // Waits up to `timeout_ns` for `timer` to come due, with nothing spinning, and says in `due` whether it did.
  std::string waitForTimer(const rcl_timer_t * timer, std::int64_t timeout_ns, bool * due)
  {
    *due = false;
    rcl_wait_set_t wait_set = rcl_get_zero_initialized_wait_set();
    const auto context = subject_->get_node_base_interface()->get_context()->get_rcl_context();
    if (rcl_wait_set_init(&wait_set, 0, 0, 1, 0, 0, 0, context.get(), rcl_get_default_allocator()) != RCL_RET_OK) {
      const std::string error = rcl_get_error_string().str;
      rcl_reset_error();
      return "wait set: " + error;
    }
    std::string error;
    rcl_ret_t ret = rcl_wait_set_add_timer(&wait_set, timer, nullptr);
    if (ret == RCL_RET_OK)
      ret = rcl_wait(&wait_set, timeout_ns);
    if (ret == RCL_RET_OK)
      *due = wait_set.timers[0] != nullptr;
    else if (ret != RCL_RET_TIMEOUT)
      error = std::string("wait on the timer: ") + rcl_get_error_string().str;
    rcl_reset_error();
    if (rcl_wait_set_fini(&wait_set) != RCL_RET_OK)
      rcl_reset_error();
    return error;
  }

  // One run of the scan path on everything the input callbacks have left it: its timer comes due, with nothing
  // spinning meanwhile, and then the executor runs it.
  std::string runScanPath(const std::string & what)
  {
    const auto timer = scanPathTimer();
    if (!timer)
      return what + ": the front end does not have exactly one timer with the scan path's period";
    bool due = false;
    const std::string error = waitForTimer(timer->get_timer_handle().get(), kReceiptWaitNs, &due);
    if (!error.empty())
      return what + ": " + error;
    if (!due && !timer->is_ready())
      return what + ": the scan path's timer did not come due within 5 s";
    if (!spinUntil([&] { return !timer->is_ready(); }, kReceiptWaitSeconds))
      return what + ": the scan path's timer came due, but the executor did not run it within 5 s";
    return "";
  }

  // Publishes one IMU sample, lets the subscription take it and the scan path run on it.
  std::string publishImu(std::int64_t stamp_ns, double yaw_rate)
  {
    sensor_msgs::msg::Imu message;
    message.header.stamp = rclcpp::Time(stamp_ns);
    message.linear_acceleration.z = kGravity;
    message.angular_velocity.z = yaw_rate;
    newest_imu_ns_ = std::max(newest_imu_ns_, stamp_ns);
    imu_->publish(message);
    executor_->spin_some();
    return runScanPath("after the IMU sample at " + std::to_string(stamp_ns) + " ns");
  }

  // Every grid IMU sample stamped up to `end` that has not gone out yet, in stamp order.
  std::string sendImuThrough(double end)
  {
    const auto through = static_cast<std::int64_t>(std::llround(end * 1e9));
    while (next_imu_ns_ <= through) {
      const std::int64_t stamp = next_imu_ns_;
      next_imu_ns_ += kImuPeriodNs;
      const double t = static_cast<double>(stamp) * 1e-9;
      if (inImuHole(t))
        continue;
      const int scan = scanOfImuTime(t);
      double yaw_rate = movingAt(scan) ? kMovingYawRate : 0.0;
      if (bad_imu_scan_ == scan && t > scanEnd(scan) - 0.004 && t <= scanEnd(scan))
        yaw_rate = std::numeric_limits<double>::quiet_NaN();
      const std::string error = publishImu(stamp, yaw_rate);
      if (!error.empty())
        return error;
    }
    return "";
  }

  void sendLidar(double start, Cloud kind)
  {
    const std::size_t points = kind == Cloud::kDense || kind == Cloud::kFar ? kScanPoints
                               : kind == Cloud::kSparse                     ? 3u
                                                                            : 0u;
    const float offset = kind == Cloud::kFar ? kFarOffset : 0.0f;
    const auto length_ns = static_cast<std::uint64_t>(std::llround(kScanLength * 1e9));
    pcl::PointCloud<ouster_ros::Point> cloud;
    cloud.resize(points);
    for (std::size_t i = 0; i < points; ++i) {
      auto & point = cloud[i];
      point = ouster_ros::Point{};
      point.x = 10.0f + offset + static_cast<float>(i % 317) * 0.01f;
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

  const fast_lio::AnchorScanRecord * recordStartingAt(double start, std::uint64_t after_seq) const
  {
    const std::int64_t key = msKey(start);
    for (auto it = records_.rbegin(); it != records_.rend(); ++it)
      if (it->scan_seq > after_seq && it->beg_ns >= 0 && msKey(static_cast<double>(it->beg_ns) * 1e-9) == key)
        return &*it;
    return nullptr;
  }

  // Drops every pending scan the front end has finished, in order: its record is in, and so is its /Odometry when it
  // was published.
  void collect()
  {
    while (!pending_.empty()) {
      const PendingScan & p = pending_.front();
      const auto * r = recordStartingAt(scanStart(p.scan), p.after_seq);
      if (r == nullptr)
        return;
      if (r->published && odom_by_ms_.count(msKey(static_cast<double>(r->end_ns) * 1e-9)) == 0)
        return;
      pending_.erase(pending_.begin());
    }
  }

  bool takeable(const PendingScan & p) const
  {
    return !p.needs_imu || newest_imu_ns_ >= static_cast<std::int64_t>(std::llround(scanEnd(p.scan) * 1e9));
  }

  // Waits for every pending scan the front end can take with the input it has; `final`: all of them.
  std::string settle(bool final)
  {
    const bool done = spinUntil([&] {
      collect();
      return pending_.empty() || !takeable(pending_.front());
    });
    if (!done || (final && !pending_.empty())) {
      std::ostringstream out;
      out << "scan " << pending_.front().scan << " (slot " << pending_.front().slot << ") produced no record";
      return out.str();
    }
    return "";
  }

  std::string warmUp()
  {
    const Slot warm = slot("warm_up");
    int published = 0;
    for (int i = 0; i < kWarmupMaxScans && published < kWarmupPublished; ++i) {
      const std::size_t before = records_.size();
      const std::string error = runSlot(warm, false);
      if (!error.empty())
        return "warm-up: " + error;
      for (std::size_t r = before; r < records_.size(); ++r)
        published += records_[r].published ? 1 : 0;
    }
    return published >= kWarmupPublished ? "" : "warm-up never published odometry";
  }

  std::string describeSlot(int index, int scan, const Slot & s) const
  {
    std::ostringstream out;
    out << "slot " << index << ' ' << s.label << " scan=" << scan << " start=" << formatDouble(scanStart(scan))
        << " end=" << formatDouble(scanEnd(scan)) << " imu=" << (s.moving ? "moving" : "static")
        << " cloud=" << cloudName(s.cloud) << " guard=" << guardName(s.guard);
    if (s.inject)
      out << " inject=" << vec(s.pos);
    if (s.trip)
      out << " trip";
    if (s.imu_throw)
      out << " imu_throw";
    if (s.hole_to > s.hole_from)
      out << " imu_hole=(" << formatDouble(s.hole_from) << ',' << formatDouble(s.hole_to) << ')';
    if (s.bad_imu)
      out << " bad_imu";
    if (s.clock_back)
      out << " clock_back";
    if (s.repeat > 0)
      out << " repeat=" << s.repeat;
    if (s.refused)
      out << " refused";
    return out.str();
  }

  std::string describeInputs() const
  {
    const auto in = fast_lio_test::inspectImuInput();
    std::ostringstream out;
    out << "inputs lidar_queued=" << in.lidarQueued << " imu_queued=" << in.imuQueued
        << " rejected_lidar=" << in.rejectedLidar << " rejected_imu=" << in.rejectedImu
        << " discarded_lidar=" << in.discardedLidar << " discarded_imu=" << in.discardedImu << " faulted=" << in.faulted
        << " map_frozen=" << in.mapFrozen << " degraded_odom=" << in.degradedOdom
        << " filter_end=" << formatDouble(in.filterEnd);
    return out.str();
  }

  // The /Odometry of the published records that has arrived, after the slot's other lines.
  void appendArrivedOdometry()
  {
    for (auto it = published_ends_.begin(); it != published_ends_.end();) {
      const auto odom = odom_by_ms_.find(*it);
      if (odom == odom_by_ms_.end()) {
        ++it;
        continue;
      }
      appendOdometry(stream_, odom->second);
      ++counts_["odometry"];
      it = published_ends_.erase(it);
    }
  }

  std::string runSlot(const Slot & s, bool next_moving)
  {
    const int index = static_cast<int>(slots_.size());
    slots_.push_back(s);
    const int scan = scan_++;
    // The IMU samples of this scan's window that go out with the previous slot already took this slot's pattern; the
    // ones past this scan's end that go out now take the next slot's.
    moving_by_scan_[scan] = s.moving;
    moving_by_scan_[scan + 1] = next_moving;
    slot_by_end_ms_[msKey(scanEnd(scan))] = index;
    if (s.hole_to > s.hole_from)
      imu_holes_.emplace_back(scanEnd(scan) + s.hole_from, scanEnd(scan) + s.hole_to);
    if (s.bad_imu)
      bad_imu_scan_ = scan;
    stream_.push_back(describeSlot(index, scan, s));
    std::string error;
    if (s.clock_back)
      error = publishImu(std::max<std::int64_t>(1, newest_imu_ns_ - 1500000000LL), 0.0);
    const std::uint64_t refused_before = fast_lio_test::inspectImuInput().rejectedLidar;
    if (error.empty())
      error = sendImuThrough(scanEnd(scan) + 0.006);
    if (!error.empty())
      return error;
    if (s.cloud != Cloud::kNone) {
      const int cloud_scan = scan - s.repeat;
      const std::uint64_t after_seq = s.repeat > 0 && !records_.empty() ? records_.back().scan_seq : 0u;
      throw_next_imu_process_ = s.imu_throw;
      sendLidar(scanStart(cloud_scan), s.cloud);
      if (!s.refused) {
        pending_.push_back({cloud_scan, index, s.cloud != Cloud::kEmpty, after_seq});
      } else if (!spinUntil([&] { return fast_lio_test::inspectImuInput().rejectedLidar > refused_before; },
                            kReceiptWaitSeconds)) {
        return "slot " + std::to_string(index) + ": the cloud was not refused within 5 s";
      } else {
        ++counts_["refused_input"];
      }
    }
    error = settle(false);
    if (error.empty())
      error = runScanPath("at the end of slot " + std::to_string(index));
    if (error.empty() && throw_next_imu_process_)
      error = "slot " + std::to_string(index) + ": the IMU processing step it was to fail never ran";
    if (!error.empty())
      return error;
    appendArrivedOdometry();
    stream_.push_back(describeInputs());
    return "";
  }

  void teardown()
  {
    if (executor_ && subject_)
      executor_->remove_node(subject_->get_node_base_interface());
    if (executor_ && observer_)
      executor_->remove_node(observer_);
    scan_path_timer_.reset();
    subject_.reset();  // the destructor joins the node's workers and writes the counter summary through the hook
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

  // What the record ends with, and the counts the counter summary holds.
  void finish()
  {
    appendArrivedOdometry();
    for (const auto key : published_ends_)
      stream_.push_back("odom_missing end_ms=" + std::to_string(key));
    std::ostringstream out;
    out << "teardown odometry_received=" << odom_by_ms_.size();
    if (scenario_.async_map) {
      const std::uint64_t adds = map_adds_.load(std::memory_order_acquire);
      out << " map_adds=" << adds;
      counts_["async_map_add"] = adds;
    }
    stream_.push_back(out.str());
    const std::pair<const char *, const char *> summary[] = {{"created", "anchor_created"},
                                                             {"pin_applied", "anchor_pin"},
                                                             {"parked_prior_applied", "parked_prior"},
                                                             {"scan_hole", "scan_hole"},
                                                             {"recovery_started", "recovery_started"},
                                                             {"recovery_qualified", "recovery_qualified"},
                                                             {"time_epoch", "input_epoch"}};
    for (const auto & entry : summary) {
      const std::string value = valueOf(counters_line_, entry.first);
      std::uint64_t count = 0;
      if (!value.empty() && value.find_first_not_of("0123456789") == std::string::npos)
        count = std::stoull(value);
      counts_[entry.second] = count;
    }
  }

  const Scenario & scenario_;
  std::thread::id main_thread_;
  std::unique_ptr<class_loader::ClassLoader> loader_;
  std::shared_ptr<rclcpp_components::NodeFactory> factory_;
  std::optional<rclcpp_components::NodeInstanceWrapper> subject_;
  rclcpp::Node::SharedPtr observer_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::TimerBase::SharedPtr scan_path_timer_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_;
  std::map<std::int64_t, nav_msgs::msg::Odometry> odom_by_ms_;
  std::vector<std::int64_t> published_ends_;  // ms keys of published records whose /Odometry is not in the record yet
  std::vector<Slot> slots_;                   // every slot run, the warm-up's included
  std::map<std::int64_t, int> slot_by_end_ms_;
  std::map<int, bool> moving_by_scan_;
  std::vector<std::pair<double, double>> imu_holes_;
  int bad_imu_scan_ = -1;
  std::int64_t newest_imu_ns_ = -1;
  std::int64_t next_imu_ns_ = kImuFirstNs;
  int scan_ = 0;
  std::vector<PendingScan> pending_;
  std::vector<fast_lio::AnchorScanRecord> records_;
  std::vector<std::string> stream_;
  std::map<std::string, std::uint64_t> counts_;
  std::map<std::int64_t, int> h_max_effective_;  // by scan end: the most effective points of an H-model iteration
  std::string counters_line_;
  bool throw_next_imu_process_ = false;
  std::atomic<std::uint64_t> map_adds_{0};
  std::uint64_t adds_at_prepare_ = 0;
  std::uint64_t adds_before_update_ = 0;
  bool update_awaiting_add_ = false;
  mutable std::mutex off_thread_mutex_;
  std::map<std::string, std::uint64_t> off_thread_;
};

// ---------------------------------------------------------------------------
// The parent: fork, collect, compare
// ---------------------------------------------------------------------------

void writeAll(int fd, const std::string & text)
{
  const char * bytes = text.data();
  std::size_t size = text.size();
  while (size > 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return;
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
}

bool readAll(int fd, std::string * out)
{
  char buffer[65536];
  for (;;) {
    const ssize_t got = ::read(fd, buffer, sizeof(buffer));
    if (got == 0)
      return true;
    if (got < 0 && errno == EINTR)
      continue;
    if (got < 0)
      return false;
    out->append(buffer, static_cast<std::size_t>(got));
  }
}

struct ChildRun
{
  int status = -1;
  std::string error;  // the child's own error, or what went wrong with the child
  std::string record;
  std::map<std::string, std::uint64_t> branches;
  std::map<std::string, std::uint64_t> off_thread;

  bool ok() const { return error.empty() && WIFEXITED(status) && WEXITSTATUS(status) == 0; }
  std::uint64_t count(const std::string & branch) const
  {
    const auto it = branches.find(branch);
    return it == branches.end() ? 0 : it->second;
  }
  std::string describe() const
  {
    std::ostringstream out;
    out << "child status=" << status;
    if (WIFSIGNALED(status))
      out << " signal=" << WTERMSIG(status);
    if (WIFEXITED(status))
      out << " exit=" << WEXITSTATUS(status);
    out << " error='" << error << "'";
    return out.str();
  }
};

std::string oneLine(std::string text)
{
  std::replace(text.begin(), text.end(), '\n', ' ');
  return text;
}

ChildRun runScenario(const Scenario & scenario)
{
  ChildRun run;
  int fds[2];
  if (::pipe(fds) != 0) {
    run.error = "pipe failed";
    return run;
  }
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    run.error = "fork failed";
    return run;
  }
  if (pid == 0) {
    ::close(fds[0]);
    ::alarm(kChildAlarmSeconds);
    PipelineRunner runner(scenario);
    std::string error;
    try {
      error = runner.run();
    } catch (const std::exception & e) {
      error = std::string("exception: ") + e.what();
    } catch (...) {
      error = "unknown exception";
    }
    const std::string status = error.empty() ? "ok" : "error " + oneLine(error);
    writeAll(fds[1], "@@status " + status + "\n@@record\n" + runner.record() + "@@branches\n" + runner.branches());
    ::close(fds[1]);
    ::_exit(error.empty() ? 0 : 3);
  }
  ::close(fds[1]);
  std::string output;
  const bool read_ok = readAll(fds[0], &output);
  ::close(fds[0]);
  ::waitpid(pid, &run.status, 0);
  if (!read_ok) {
    run.error = "pipe read failed";
    return run;
  }
  const std::string status_tag = "@@status ";
  const std::string record_tag = "\n@@record\n";
  const std::string branches_tag = "@@branches\n";
  const std::size_t record_at = output.find(record_tag);
  const std::size_t branches_at = output.rfind(branches_tag);
  if (output.compare(0, status_tag.size(), status_tag) != 0 || record_at == std::string::npos ||
      branches_at == std::string::npos || branches_at < record_at) {
    run.error = "the child wrote no complete report";
    return run;
  }
  const std::string status = output.substr(status_tag.size(), record_at - status_tag.size());
  if (status != "ok")
    run.error = status.rfind("error ", 0) == 0 ? status.substr(6) : status;
  run.record = output.substr(record_at + record_tag.size(), branches_at - record_at - record_tag.size());
  for (const auto & line : splitLines(output.substr(branches_at + branches_tag.size()))) {
    std::istringstream in(line);
    std::string kind, name;
    std::uint64_t value = 0;
    if (!(in >> kind >> name >> value))
      continue;
    if (kind == "branch")
      run.branches[name] = value;
    else if (kind == "off_thread")
      run.off_thread[name] = value;
  }
  return run;
}

bool makeDirectories(const std::string & path)
{
  std::string partial;
  std::istringstream parts(path);
  if (!path.empty() && path[0] == '/')
    partial = "/";
  for (std::string part; std::getline(parts, part, '/');) {
    if (part.empty())
      continue;
    partial += part + "/";
    if (::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST)
      return false;
  }
  return true;
}

bool writeFile(const std::string & directory, const std::string & name, const std::string & text)
{
  if (!makeDirectories(directory))
    return false;
  std::ofstream out(directory + "/" + name, std::ios::binary | std::ios::trunc);
  out << text;
  return static_cast<bool>(out);
}

std::string goldenOut()
{
  const char * value = std::getenv("FASTLIO_CONTRACT_GOLDEN_OUT");
  return value != nullptr ? std::string(value) : std::string();
}

std::string goldenDir()
{
  return std::string(ROOT_DIR) + "test/frontend_contract/golden/" + kFixture;
}

std::string maskRoot(std::string text)
{
  const std::string root = ROOT_DIR;
  for (std::size_t at = text.find(root); !root.empty() && at != std::string::npos; at = text.find(root, at))
    text.replace(at, root.size(), "<ROOT_DIR>/");
  return text;
}

std::string goldenHeader(const std::string & case_name)
{
  const char * sha = std::getenv("FASTLIO_CONTRACT_SOURCE_SHA");
  std::ostringstream out;
  out << "# frontend contract golden: " << kFixture << '/' << case_name << '\n'
      << "# fork_sha: " << (sha != nullptr ? sha : "") << '\n'
      << "# arch: " << kArch << '\n'
      << "# ros_distro: " << FASTLIO_CONTRACT_ROS_DISTRO << '\n'
      << "# build: " << FASTLIO_CONTRACT_BUILD << '\n';
  return out.str();
}

std::string headerValue(const std::string & golden, const std::string & key)
{
  const std::string prefix = "# " + key + ": ";
  for (const auto & line : splitLines(golden))
    if (line.rfind(prefix, 0) == 0)
      return line.substr(prefix.size());
  return "";
}

std::string bodyOf(const std::string & golden)
{
  std::string out;
  for (const auto & line : splitLines(golden))
    if (line.rfind("# ", 0) != 0)
      out += line + '\n';
  return out;
}

std::string jsonString(const std::string & text)
{
  std::string out = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (byte < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(byte));
      out += escaped;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

struct Comparison
{
  std::string result;  // equal, different, generated, no_golden, not_comparable, nondeterministic, claims_missed, error
  std::string detail;
  std::size_t line = 0;
  std::string expected, actual;
};

Comparison compareLines(const std::string & expected, const std::string & actual)
{
  Comparison comparison;
  const auto want = splitLines(expected);
  const auto got = splitLines(actual);
  for (std::size_t i = 0; i < std::max(want.size(), got.size()); ++i) {
    const std::string a = i < want.size() ? want[i] : "<end of record>";
    const std::string b = i < got.size() ? got[i] : "<end of record>";
    if (a != b) {
      comparison.result = "different";
      comparison.line = i + 1;
      comparison.expected = a;
      comparison.actual = b;
      return comparison;
    }
  }
  comparison.result = "equal";
  return comparison;
}

void writeReceipt(const Scenario & scenario,
                  const std::string & mode,
                  const ChildRun & run,
                  const Comparison & comparison,
                  const std::string & golden_path,
                  const std::string & golden_text)
{
  std::ostringstream out;
  out << "{\"fixture\":" << jsonString(kFixture) << ",\"case\":" << jsonString(scenario.name)
      << ",\"mode\":" << jsonString(mode) << ",\"arch\":" << jsonString(kArch)
      << ",\"ros_distro\":" << jsonString(FASTLIO_CONTRACT_ROS_DISTRO)
      << ",\"build\":" << jsonString(FASTLIO_CONTRACT_BUILD)
      << ",\"async_map\":" << (scenario.async_map ? "true" : "false") << ",\"child\":" << jsonString(run.describe())
      << ",\"record_lines\":" << splitLines(run.record).size()
      << ",\"record_sha256\":" << jsonString(sha256Hex(run.record))
      << ",\"golden\":" << jsonString(maskRoot(golden_path))
      << ",\"golden_sha256\":" << jsonString(golden_text.empty() ? std::string() : sha256Hex(golden_text))
      << ",\"golden_fork_sha\":" << jsonString(headerValue(golden_text, "fork_sha"))
      << ",\"golden_build\":" << jsonString(headerValue(golden_text, "build"))
      << ",\"result\":" << jsonString(comparison.result) << ",\"detail\":" << jsonString(comparison.detail)
      << ",\"first_difference_line\":" << comparison.line << ",\"expected\":" << jsonString(comparison.expected)
      << ",\"actual\":" << jsonString(comparison.actual) << ",\"branches\":[";
  bool first = true;
  for (const auto & branch : kBranches) {
    const bool claimed =
      std::find(scenario.claims.begin(), scenario.claims.end(), branch.name) != scenario.claims.end();
    const std::uint64_t count = run.count(branch.name);
    out << (first ? "" : ",") << "{\"branch\":" << jsonString(branch.name) << ",\"count\":" << count
        << ",\"triggered\":" << (count > 0 ? "true" : "false") << ",\"claimed\":" << (claimed ? "true" : "false")
        << ",\"counted\":" << jsonString(branch.counted) << ",\"reference\":" << jsonString(branch.reference) << "}";
    first = false;
  }
  out << "],\"suspended\":[";
  first = true;
  for (const auto & entry : kSuspended) {
    out << (first ? "" : ",") << "{\"branch\":" << jsonString(entry.branch) << ",\"triggered\":false"
        << ",\"missing\":" << jsonString(entry.missing) << ",\"hook\":" << jsonString(entry.hook) << "}";
    first = false;
  }
  out << "],\"off_thread_hooks\":{";
  first = true;
  for (const auto & entry : run.off_thread) {
    out << (first ? "" : ",") << jsonString(entry.first) << ':' << entry.second;
    first = false;
  }
  out << "}}\n";
  const std::string name = scenario.name + ".receipt.json";
  writeFile(std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture, name, out.str());
  const std::string generate = goldenOut();
  if (!generate.empty())
    writeFile(generate + "/receipts/" + kFixture, name, out.str());
}

void checkScenario(const Scenario & scenario)
{
  const std::string generate = goldenOut();
  if (generate.empty() && std::string(kArch) != "x86_64") {
    FAIL() << "golden 只适用于 x86_64: the front-end contract goldens were generated on x86_64 and this build is "
           << kArch << "; the comparison is not run here (CMake registers this entry as skipped on other "
           << "architectures)";
  }
  const ChildRun run = runScenario(scenario);
  std::vector<std::string> missed;
  for (const auto & claim : scenario.claims)
    if (run.count(claim) == 0)
      missed.push_back(claim);

  Comparison comparison;
  std::string golden_text;
  std::string golden_path = goldenDir() + "/" + scenario.name + ".golden";
  const std::string results_dir = std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture;
  if (!run.ok()) {
    comparison.result = "error";
    comparison.detail = run.describe();
  } else if (!generate.empty()) {
    // A golden is written only from a record two runs of this build agree on, and that reaches every claimed branch.
    const ChildRun again = runScenario(scenario);
    if (!again.ok()) {
      comparison.result = "error";
      comparison.detail = "second run: " + again.describe();
    } else if (again.record != run.record) {
      comparison = compareLines(run.record, again.record);
      comparison.result = "nondeterministic";
      comparison.detail = "two runs of this build recorded different streams";
      writeFile(results_dir, scenario.name + ".second.actual", again.record);
    } else if (!missed.empty()) {
      comparison.result = "claims_missed";
      comparison.detail = "no golden written: a claimed branch was not reached";
    } else {
      golden_path = generate + "/golden/" + kFixture + "/" + scenario.name + ".golden";
      golden_text = goldenHeader(scenario.name) + run.record;
      if (writeFile(generate + "/golden/" + kFixture, scenario.name + ".golden", golden_text)) {
        comparison.result = "generated";
      } else {
        comparison.result = "error";
        comparison.detail = "cannot write " + golden_path;
      }
    }
  } else if (!readFile(golden_path, &golden_text)) {
    comparison.result = "no_golden";
    comparison.detail = "no golden at " + maskRoot(golden_path);
  } else if (headerValue(golden_text, "arch") != "x86_64" ||
             headerValue(golden_text, "ros_distro") != FASTLIO_CONTRACT_ROS_DISTRO) {
    comparison.result = "not_comparable";
    comparison.detail = "the golden was generated on " + headerValue(golden_text, "arch") + "/" +
                        headerValue(golden_text, "ros_distro") + ", this build is " + kArch + "/" +
                        FASTLIO_CONTRACT_ROS_DISTRO;
  } else {
    comparison = compareLines(bodyOf(golden_text), run.record);
    if (comparison.result != "equal")
      comparison.detail =
        "golden build: " + headerValue(golden_text, "build") + "; this build: " + FASTLIO_CONTRACT_BUILD;
  }
  if (comparison.result != "equal" && comparison.result != "generated")
    writeFile(results_dir, scenario.name + ".actual", run.record);
  writeReceipt(scenario, generate.empty() ? "compare" : "generate", run, comparison, golden_path, golden_text);

  EXPECT_TRUE(run.ok()) << scenario.name << ": " << run.describe();
  EXPECT_TRUE(missed.empty()) << scenario.name << " did not reach the claimed branch " << missed.front();
  EXPECT_EQ(comparison.result, generate.empty() ? "equal" : "generated")
    << scenario.name << ": " << comparison.detail << "\n  line " << comparison.line
    << "\n  golden: " << comparison.expected << "\n  actual: " << comparison.actual;
}

TEST(FrontendPipelineContract, Match)
{
  checkScenario(matchScenario());
}

TEST(FrontendPipelineContract, ParkedPriorAndReset)
{
  checkScenario(parkedPriorScenario());
}

TEST(FrontendPipelineContract, PinAndReset)
{
  checkScenario(pinScenario());
}

TEST(FrontendPipelineContract, ScanDrops)
{
  checkScenario(scanDropsScenario());
}

TEST(FrontendPipelineContract, ImuCoverageGap)
{
  checkScenario(imuCoverageScenario());
}

TEST(FrontendPipelineContract, ImuInvalidAndScanHole)
{
  checkScenario(imuInvalidScanHoleScenario());
}

TEST(FrontendPipelineContract, ImuFatal)
{
  checkScenario(imuFatalScenario());
}

TEST(FrontendPipelineContract, InputEpoch)
{
  checkScenario(inputEpochScenario());
}

TEST(FrontendPipelineContract, EntryViewMovingRecovery)
{
  checkScenario(entryViewScenario());
}

// The scenario the front-end ThreadSanitizer phase runs: the map worker and the reset that joins it.
TEST(FrontendPipelineWorker, AsyncMapDelayAndReset)
{
  checkScenario(workerScenario());
}

}  // namespace

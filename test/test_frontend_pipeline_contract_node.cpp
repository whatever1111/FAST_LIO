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
//             the filter's input end;
// and, on lines whose first word is a `.v1` name (added to records that existed before them: removing these lines
// gives each earlier record unchanged):
//   h_matrix.v1       every H-model exit: the scan end, the validity, and the dimensions and SHA-256 of the Jacobian
//                     rows and of the residuals handed to the filter, or none at the exit without a correspondence;
//   filter_update.v1  the filter right after every iterated lidar update: each state component, the SHA-256 of the
//                     23x23 error-state covariance and its diagonal;
//   imu_retain.v1     an empty IMU window settled uncommitted: its batch, the samples it leaves queued and the FIFO
//                     fault latch;
//   h_select.v1       every H-model call right after compaction: the scan end and the call, the search the filter
//                     asked for and whether the call searched (the refresh gate skips a search the state has not
//                     moved enough for), the gate's refusals and the scan's robust residual scale, the rows the ground
//                     recovery added, and the SHA-256 of the selection in point order, of the kept points' residuals
//                     and of the rows' points, normals, residuals and kernel weights;
//   h_exit.v1         every H-model exit: validity and rows, the mean residual, the rows' position information, the
//                     attitude hold's verdicts and the roll/pitch information, the subspace weighting's verdict,
//                     spectrum and dose, and whether the call appended to the correspondence dump;
//   h_dump.v1         the correspondence dump scenario only, at its end: each dump file's size, lines and SHA-256 (the
//                     test build writes the files to the scenario's result directory);
//   identity.v1, frontend_observation.v1, reference_health.v1
//                     the typed scenario only: the fixed activation identity with the contract and configuration IDs
//                     the constructor computed, then every FrontendObservation and ReferenceHealth message received,
//                     field by field, per topic in arrival order, after each slot's other lines.
// Doubles are written with 17 significant digits (round trip) and NaN with its bit pattern; on the `.v1` lines every
// double is its IEEE-754 bit pattern, and a SHA-256 covers a matrix's patterns row-major, eight little-endian bytes
// each. A record is compared with its golden line for line, character for character, with no tolerance anywhere.
//
// The measurement scenarios (h_*, laserMapping S2) turn on the H-model branches the default configuration leaves off,
// each on a synthetic scene built to reach it: three tilted walls 4-6 m away, near ground with clutter below walls, a
// corridor with a patch across its end. A scene's points sit on a fixed grid and move along their surface normals by
// noise drawn from the scan's start time, new in every scan and the same in every run.
//
// What makes a record reproducible: the map worker is off (FLIO_ASYNC_MAP=0) except in the worker suite, OpenMP runs
// one thread, and the scan path runs (the fixture waits for its timer to come due and lets the executor run it) after
// every IMU sample and at the end of every slot, so nothing it evaluates depends on when its timer happened to fire.
// In the worker suite the worker's own hook only counts map additions; the count is read where the scan path prepares
// the next scan, after it has joined the worker. A generation run records every scenario twice and writes a golden
// only when both records agree.
//
// The typed outputs carry an activation identity the constructor draws at random and repeat the reference health on a
// steady-clock throttle, so only the typed scenario turns them on: its hooks fix the identity, and the throttle reads a
// fixture clock that a scan path run before any input starts and that advances 200 ms right before each slot's last
// scan path run and nowhere else.
//
// Masked: nothing. Not recorded: log output (its throttling runs on the wall clock) and timing diagnostics.
//
// Every scenario counts the branches it went through; the receipt lists each branch, triggered or not, with its
// count. A branch the scenario claims and did not reach fails it. Where an existing suite already asserts a branch's
// semantics, the receipt names it; this fixture records the branch's outputs bit for bit and does not repeat those
// assertions.
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
#include <fast_lio_interfaces/msg/frontend_observation.hpp>
#include <fast_lio_interfaces/msg/reference_health.hpp>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
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

// The typed scenario's activation identity, and the clock its reference repeat throttle (100 ms) reads.
constexpr std::array<std::uint8_t, 16> kFixedInstance = {
  0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
constexpr std::chrono::seconds kRepeatClockStart{1};
constexpr std::chrono::milliseconds kRepeatClockStep{200};
constexpr const char * kFrontendObservationTopic = "/lio_slam/frontend_observation";
constexpr const char * kReferenceHealthTopic = "/lio_slam/reference_health";

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

// A double's IEEE-754 bit pattern: tells +0 from -0 and every NaN payload apart.
std::string bits(double value)
{
  std::uint64_t pattern = 0;
  std::memcpy(&pattern, &value, sizeof(pattern));
  char text[24];
  std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(pattern));
  return text;
}

std::string bitsList(const double * values, std::size_t size)
{
  std::string out = "[";
  for (std::size_t i = 0; i < size; ++i) {
    if (i != 0)
      out += ',';
    out += bits(values[i]);
  }
  return out + "]";
}

std::string bitsList(std::initializer_list<double> values)
{
  return bitsList(values.begin(), values.size());
}

// SHA-256 over the values' bit patterns, eight little-endian bytes each, in the order given.
std::string bitsDigest(const double * values, std::size_t size)
{
  std::string bytes;
  bytes.reserve(size * 8);
  for (std::size_t i = 0; i < size; ++i) {
    std::uint64_t pattern = 0;
    std::memcpy(&pattern, &values[i], sizeof(pattern));
    for (int b = 0; b < 8; ++b)
      bytes += static_cast<char>((pattern >> (8 * b)) & 0xffu);
  }
  return sha256Hex(bytes);
}

// The same over a matrix's coefficients, row-major; "none" for a matrix without a coefficient.
template<class Derived>
std::string matrixDigest(const Eigen::MatrixBase<Derived> & m)
{
  if (m.size() == 0)
    return "none";
  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(m.size()));
  for (Eigen::Index r = 0; r < m.rows(); ++r) {
    for (Eigen::Index c = 0; c < m.cols(); ++c)
      rows.push_back(m(r, c));
  }
  return bitsDigest(rows.data(), rows.size());
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

// Every field of FrontendObservation, in declaration order (generated from its .msg file).
std::string formatFrontendObservation(const fast_lio_interfaces::msg::FrontendObservation & m)
{
  std::string out = "frontend_observation.v1";
  out += " schema_version=" + std::to_string(static_cast<unsigned long long>(m.schema_version));
  out += " identity_version=" + std::to_string(static_cast<unsigned long long>(m.identity_version));
  out += " has_source_instance=" + std::string(m.has_source_instance ? "1" : "0");
  out += " source_instance=" + hex(m.source_instance.data(), m.source_instance.size());
  out += " has_sample_sequence=" + std::string(m.has_sample_sequence ? "1" : "0");
  out += " sample_sequence=" + std::to_string(static_cast<unsigned long long>(m.sample_sequence));
  out += " has_observation_id=" + std::string(m.has_observation_id ? "1" : "0");
  out += " observation_id=" + hex(m.observation_id.data(), m.observation_id.size());
  out += " clock_domain=" + quote(m.clock_domain);
  out += " has_input_stamp=" + std::string(m.has_input_stamp ? "1" : "0");
  out += " input_stamp_ns=" + std::to_string(static_cast<long long>(m.input_stamp_ns));
  out += " has_observation_stamp=" + std::string(m.has_observation_stamp ? "1" : "0");
  out += " observation_stamp_ns=" + std::to_string(static_cast<long long>(m.observation_stamp_ns));
  out += " has_effective_observation_stamp=" + std::string(m.has_effective_observation_stamp ? "1" : "0");
  out += " effective_observation_stamp_ns=" + std::to_string(static_cast<long long>(m.effective_observation_stamp_ns));
  out += " frame_id=" + quote(m.frame_id);
  out += " child_frame_id=" + quote(m.child_frame_id);
  out += " has_coordinate_epoch=" + std::string(m.has_coordinate_epoch ? "1" : "0");
  out += " coordinate_epoch=" + std::to_string(static_cast<unsigned long long>(m.coordinate_epoch));
  out += " has_pose=" + std::string(m.has_pose ? "1" : "0");
  out += " pose.position=" + bitsList({m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z});
  out +=
    " pose.orientation_xyzw=" +
    bitsList(
      {m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w});
  out += " pose.covariance=" + bitsList(m.pose.covariance.data(), m.pose.covariance.size());
  out += " has_pose_covariance=" + std::string(m.has_pose_covariance ? "1" : "0");
  out += " has_twist=" + std::string(m.has_twist ? "1" : "0");
  out += " twist.linear=" + bitsList({m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z});
  out += " twist.angular=" + bitsList({m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z});
  out += " twist.covariance=" + bitsList(m.twist.covariance.data(), m.twist.covariance.size());
  out += " has_twist_covariance=" + std::string(m.has_twist_covariance ? "1" : "0");
  out += " output_validity=" + std::to_string(static_cast<unsigned long long>(m.output_validity));
  out += " axis_validity_bits=" + std::to_string(static_cast<unsigned long long>(m.axis_validity_bits));
  out += " estimator_state=" + std::to_string(static_cast<unsigned long long>(m.estimator_state));
  out += " activity=" + std::to_string(static_cast<unsigned long long>(m.activity));
  out += " fe_substate=" + std::to_string(static_cast<unsigned long long>(m.fe_substate));
  out += " has_guard_reason_bits=" + std::string(m.has_guard_reason_bits ? "1" : "0");
  out += " guard_reason_bits=" + std::to_string(static_cast<unsigned long long>(m.guard_reason_bits));
  out += " has_health_flags=" + std::string(m.has_health_flags ? "1" : "0");
  out += " health_flags=" + std::to_string(static_cast<unsigned long long>(m.health_flags));
  out += " has_observability_along=" + std::string(m.has_observability_along ? "1" : "0");
  out += " observability_along=" + bits(m.observability_along);
  out += " has_effective_feature_count=" + std::string(m.has_effective_feature_count ? "1" : "0");
  out += " effective_feature_count=" + std::to_string(static_cast<unsigned long long>(m.effective_feature_count));
  out += " has_far_point_fraction=" + std::string(m.has_far_point_fraction ? "1" : "0");
  out += " far_point_fraction=" + bits(m.far_point_fraction);
  out += " has_mean_residual=" + std::string(m.has_mean_residual ? "1" : "0");
  out += " mean_residual_m=" + bits(m.mean_residual_m);
  out += " has_body_speed=" + std::string(m.has_body_speed ? "1" : "0");
  out += " body_speed_mps=" + bits(m.body_speed_mps);
  out += " has_gravity=" + std::string(m.has_gravity ? "1" : "0");
  out += " gravity_world_mps2=" + bitsList(m.gravity_world_mps2.data(), m.gravity_world_mps2.size());
  out += " has_input_sequence=" + std::string(m.has_input_sequence ? "1" : "0");
  out += " input_sequence=" + std::to_string(static_cast<unsigned long long>(m.input_sequence));
  out += " has_completion_sequence=" + std::string(m.has_completion_sequence ? "1" : "0");
  out += " completion_sequence=" + std::to_string(static_cast<unsigned long long>(m.completion_sequence));
  out += " has_output_sequence=" + std::string(m.has_output_sequence ? "1" : "0");
  out += " output_sequence=" + std::to_string(static_cast<unsigned long long>(m.output_sequence));
  out += " has_queue_count=" + std::string(m.has_queue_count ? "1" : "0");
  out += " queue_count=" + std::to_string(static_cast<unsigned long long>(m.queue_count));
  out += " has_dropped_count=" + std::string(m.has_dropped_count ? "1" : "0");
  out += " dropped_count=" + std::to_string(static_cast<unsigned long long>(m.dropped_count));
  return out;
}

// Every field of ReferenceHealth, in declaration order (generated from its .msg file).
std::string formatReferenceHealth(const fast_lio_interfaces::msg::ReferenceHealth & m)
{
  std::string out = "reference_health.v1";
  out += " schema_version=" + std::to_string(static_cast<unsigned long long>(m.schema_version));
  out += " identity_version=" + std::to_string(static_cast<unsigned long long>(m.identity_version));
  out += " has_source_instance=" + std::string(m.has_source_instance ? "1" : "0");
  out += " source_instance=" + hex(m.source_instance.data(), m.source_instance.size());
  out += " clock_domain=" + quote(m.clock_domain);
  out += " frame_id=" + quote(m.frame_id);
  out += " child_frame_id=" + quote(m.child_frame_id);
  out += " has_coordinate_epoch=" + std::string(m.has_coordinate_epoch ? "1" : "0");
  out += " coordinate_epoch=" + std::to_string(static_cast<unsigned long long>(m.coordinate_epoch));
  out += " producer_contract_id=" + hex(m.producer_contract_id.data(), m.producer_contract_id.size());
  out += " config_id=" + hex(m.config_id.data(), m.config_id.size());
  out += " implemented_mask=" + std::to_string(static_cast<unsigned long long>(m.implemented_mask));
  out += " enabled_policy_mask=" + std::to_string(static_cast<unsigned long long>(m.enabled_policy_mask));
  out += " has_snapshot_sequence=" + std::string(m.has_snapshot_sequence ? "1" : "0");
  out += " snapshot_sequence=" + std::to_string(static_cast<unsigned long long>(m.snapshot_sequence));
  out += " has_state_revision=" + std::string(m.has_state_revision ? "1" : "0");
  out += " state_revision=" + std::to_string(static_cast<unsigned long long>(m.state_revision));
  out += " has_scan_sequence=" + std::string(m.has_scan_sequence ? "1" : "0");
  out += " scan_sequence=" + std::to_string(static_cast<unsigned long long>(m.scan_sequence));
  out += " has_observation_id=" + std::string(m.has_observation_id ? "1" : "0");
  out += " observation_id=" + hex(m.observation_id.data(), m.observation_id.size());
  out += " kind=" + std::to_string(static_cast<unsigned long long>(m.kind));
  out += " has_evaluated_input=" + std::string(m.has_evaluated_input ? "1" : "0");
  out += " evaluated_input_ns=" + std::to_string(static_cast<long long>(m.evaluated_input_ns));
  out += " has_effective=" + std::string(m.has_effective ? "1" : "0");
  out += " effective_ns=" + std::to_string(static_cast<long long>(m.effective_ns));
  out += " has_state_valid_until=" + std::string(m.has_state_valid_until ? "1" : "0");
  out += " state_valid_until_ns=" + std::to_string(static_cast<long long>(m.state_valid_until_ns));
  out += " has_scan_begin=" + std::string(m.has_scan_begin ? "1" : "0");
  out += " scan_begin_ns=" + std::to_string(static_cast<long long>(m.scan_begin_ns));
  out += " has_scan_end=" + std::string(m.has_scan_end ? "1" : "0");
  out += " scan_end_ns=" + std::to_string(static_cast<long long>(m.scan_end_ns));
  out += " has_pose_binding=" + std::string(m.has_pose_binding ? "1" : "0");
  out += " pose_binding=" + bitsList(m.pose_binding.data(), m.pose_binding.size());
  out += " lifecycle=" + std::to_string(static_cast<unsigned long long>(m.lifecycle));
  out += " startup_reference=" + std::to_string(static_cast<unsigned long long>(m.startup_reference));
  out += " trust=" + std::to_string(static_cast<unsigned long long>(m.trust));
  out += " local=" + std::to_string(static_cast<unsigned long long>(m.local));
  out += " legacy_flags=" + std::to_string(static_cast<unsigned long long>(m.legacy_flags));
  out += " has_raw_registration_degraded=" + std::string(m.has_raw_registration_degraded ? "1" : "0");
  out += " raw_registration_degraded=" + std::string(m.raw_registration_degraded ? "1" : "0");
  out += " reference_generation=" + std::to_string(static_cast<unsigned long long>(m.reference_generation));
  out += " recovery_episode=" + std::to_string(static_cast<unsigned long long>(m.recovery_episode));
  out += " has_qualified_event_sequence=" + std::string(m.has_qualified_event_sequence ? "1" : "0");
  out += " qualified_event_sequence=" + std::to_string(static_cast<unsigned long long>(m.qualified_event_sequence));
  out += " has_original_start=" + std::string(m.has_original_start ? "1" : "0");
  out += " original_start_ns=" + std::to_string(static_cast<long long>(m.original_start_ns));
  out += " has_original_deadline=" + std::string(m.has_original_deadline ? "1" : "0");
  out += " original_deadline_ns=" + std::to_string(static_cast<long long>(m.original_deadline_ns));
  out += " has_last_supported_end=" + std::string(m.has_last_supported_end ? "1" : "0");
  out += " last_supported_end_ns=" + std::to_string(static_cast<long long>(m.last_supported_end_ns));
  out += " present_mask=" + std::to_string(static_cast<unsigned long long>(m.present_mask));
  out += " true_mask=" + std::to_string(static_cast<unsigned long long>(m.true_mask));
  out += " has_negative_masks=" + std::string(m.has_negative_masks ? "1" : "0");
  out += " cause_mask=" + std::to_string(static_cast<unsigned long long>(m.cause_mask));
  out += " taint_mask=" + std::to_string(static_cast<unsigned long long>(m.taint_mask));
  out += " hard_mask=" + std::to_string(static_cast<unsigned long long>(m.hard_mask));
  out += " hr_witness_kind=" + std::to_string(static_cast<unsigned long long>(m.hr_witness_kind));
  out += " has_hr_valid_until=" + std::string(m.has_hr_valid_until ? "1" : "0");
  out += " hr_valid_until_ns=" + std::to_string(static_cast<long long>(m.hr_valid_until_ns));
  out += " has_rw=" + std::string(m.has_rw ? "1" : "0");
  out += " rw_contract=" + std::to_string(static_cast<unsigned long long>(m.rw_contract));
  out += " has_raw_episode=" + std::string(m.has_raw_episode ? "1" : "0");
  out += " raw_episode_id=" + std::to_string(static_cast<unsigned long long>(m.raw_episode_id));
  out += " has_raw_start=" + std::string(m.has_raw_start ? "1" : "0");
  out += " raw_start_ns=" + std::to_string(static_cast<long long>(m.raw_start_ns));
  out += " has_rw_origin=" + std::string(m.has_rw_origin ? "1" : "0");
  out += " origin_scan=" + std::to_string(static_cast<unsigned long long>(m.origin_scan));
  out +=
    " origin_reference_generation=" + std::to_string(static_cast<unsigned long long>(m.origin_reference_generation));
  out += " has_frozen_residual=" + std::string(m.has_frozen_residual ? "1" : "0");
  out += " frozen_residual_median=" + bits(m.frozen_residual_median);
  out += " has_withdrawal_rearm=" + std::string(m.has_withdrawal_rearm ? "1" : "0");
  out += " withdrawal_rearm_event=" + std::to_string(static_cast<unsigned long long>(m.withdrawal_rearm_event));
  out += " has_rw_valid_until=" + std::string(m.has_rw_valid_until ? "1" : "0");
  out += " rw_valid_until_ns=" + std::to_string(static_cast<long long>(m.rw_valid_until_ns));
  out += " has_wheel_witness_epochs=" + std::string(m.has_wheel_witness_epochs ? "1" : "0");
  out += " lower_wheel_witness_epoch=" + std::to_string(static_cast<unsigned long long>(m.lower_wheel_witness_epoch));
  out += " upper_wheel_witness_epoch=" + std::to_string(static_cast<unsigned long long>(m.upper_wheel_witness_epoch));
  out += " wheel_witness_kind=" + std::to_string(static_cast<unsigned long long>(m.wheel_witness_kind));
  out += " has_support=" + std::string(m.has_support ? "1" : "0");
  out += " support_contract=" + std::to_string(static_cast<unsigned long long>(m.support_contract));
  out += " support_kind=" + std::to_string(static_cast<unsigned long long>(m.support_kind));
  out += " has_support_seed=" + std::string(m.has_support_seed ? "1" : "0");
  out += " support_seed_reference=" + std::to_string(static_cast<unsigned long long>(m.support_seed_reference));
  out += " support_seed_scan=" + std::to_string(static_cast<unsigned long long>(m.support_seed_scan));
  out += " support_seed_source=" + hex(m.support_seed_source.data(), m.support_seed_source.size());
  out += " has_supported_through=" + std::string(m.has_supported_through ? "1" : "0");
  out += " supported_through_ns=" + std::to_string(static_cast<long long>(m.supported_through_ns));
  out += " has_first_loss=" + std::string(m.has_first_loss ? "1" : "0");
  out += " first_loss_ns=" + std::to_string(static_cast<long long>(m.first_loss_ns));
  out += " has_support_deadline=" + std::string(m.has_support_deadline ? "1" : "0");
  out += " support_deadline_ns=" + std::to_string(static_cast<long long>(m.support_deadline_ns));
  out += " has_proof_id=" + std::string(m.has_proof_id ? "1" : "0");
  out += " proof_id=" + hex(m.proof_id.data(), m.proof_id.size());
  out += " has_trajectory_id=" + std::string(m.has_trajectory_id ? "1" : "0");
  out += " trajectory_id=" + hex(m.trajectory_id.data(), m.trajectory_id.size());
  return out;
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

// bitsDigest over the values, or "none" without one.
std::string valuesDigest(const std::vector<double> & values)
{
  return values.empty() ? std::string("none") : bitsDigest(values.data(), values.size());
}

// The h_select.v1 line of one H-model call. The selection is digested as one byte per point (1: the point has a row),
// the residuals as the kept points' |residual| in point order; a refused or unmatched point's entries are stale and
// left out. `weighted` returns the rows whose kernel weight is below 1.
std::string formatSelection(const fast_lio_test::HModelSelection & s, std::size_t * weighted)
{
  const std::size_t points = s.points > 0 ? static_cast<std::size_t>(s.points) : 0u;
  std::string flags;
  std::vector<double> residuals;
  std::size_t selected = 0;
  if (s.selected != nullptr) {
    for (std::size_t i = 0; i < points; ++i) {
      const bool kept = s.selected[i] != 0;
      flags += kept ? '\1' : '\0';
      if (!kept)
        continue;
      ++selected;
      if (s.residuals != nullptr)
        residuals.push_back(static_cast<double>(s.residuals[i]));
    }
  }
  std::vector<double> row_points, row_normals, row_residuals, row_weights;
  *weighted = 0;
  for (const auto & row : s.rows) {
    for (int d = 0; d < 3; ++d) {
      row_points.push_back(static_cast<double>(row.point[d]));
      row_normals.push_back(static_cast<double>(row.normal[d]));
    }
    row_residuals.push_back(static_cast<double>(row.residual));
    row_weights.push_back(row.weight);
    if (row.weight < 1.0)
      ++*weighted;
  }
  std::ostringstream out;
  out << "h_select.v1 end=" << bits(s.scan_end) << " call=" << s.call << " requested=" << (s.search_requested ? 1 : 0)
      << " searched=" << (s.searched ? 1 : 0) << " refresh_skips=" << s.refresh_skips
      << " refresh_max_shift=" << bits(s.refresh_max_shift) << " points=" << s.points << " selected=" << selected
      << " selected_sha256=" << (flags.empty() ? std::string("none") : sha256Hex(flags))
      << " residuals_sha256=" << valuesDigest(residuals) << " gate_rejected=" << s.gate_rejected
      << " gate_scan_sigma=" << bits(s.gate_scan_sigma) << " ground_rows=" << s.ground_rows << " rows=" << s.rows.size()
      << " rows_weighted=" << *weighted << " row_points_sha256=" << valuesDigest(row_points)
      << " row_normals_sha256=" << valuesDigest(row_normals) << " row_residuals_sha256=" << valuesDigest(row_residuals)
      << " row_weights_sha256=" << valuesDigest(row_weights);
  return out.str();
}

// The h_exit.v1 line of one H-model exit.
std::string formatExit(const fast_lio_test::HModelExit & e)
{
  std::ostringstream out;
  out << "h_exit.v1 end=" << bits(e.scan_end) << " call=" << e.call << " valid=" << (e.valid ? 1 : 0)
      << " effective=" << e.effective << " mean_residual=" << bits(e.mean_residual)
      << " pos_eig=" << bitsList({e.pos_eig_min, e.pos_eig_max}) << " z_weak=" << bits(e.pos_z_weak)
      << " hold_scene=" << (e.hold_scene ? 1 : 0) << " hold_allowed=" << (e.hold_allowed ? 1 : 0)
      << " hold_rows=" << (e.hold_rows ? 1 : 0) << " hold_info=" << (e.hold_info ? 1 : 0)
      << " rp_info_min=" << bits(e.rp_info_min) << " rp_consider=" << (e.rp_consider ? 1 : 0)
      << " degeneracy_engaged=" << (e.degeneracy_engaged ? 1 : 0)
      << " degeneracy_shares=" << bitsList(e.degeneracy_shares, 3)
      << " degeneracy_lambdas=" << bitsList(e.degeneracy_lambdas, 3)
      << " degeneracy_weights=" << bitsList(e.degeneracy_weights, 3)
      << " degeneracy_weighted_points=" << e.degeneracy_weighted_points
      << " degeneracy_weight_sum=" << bits(e.degeneracy_weight_sum) << " dump_written=" << (e.dump_written ? 1 : 0);
  return out.str();
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

enum class Cloud : std::uint8_t
{
  kDense,     // the fixture's plane, 2000 points over the scan
  kFar,       // the same plane 30 m further along x: no map neighbour for any point
  kSparse,    // three points: too few to register, the scan leaves before the anchor step
  kEmpty,     // no point: no scan timing
  kNone,      // no scan this period
  kRoom,      // three steep, tilted, non-parallel walls 4-6 m away: every position and attitude direction observed
  kGround,    // the ground 0.8 m below, a third of it clutter 0.2-0.5 m high, and the room's walls above it
  kCorridor,  // two parallel walls 3.2 m apart and a small patch across its end: nothing vertical, little along it
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
    case Cloud::kRoom:
      return "room";
    case Cloud::kGround:
      return "ground";
    case Cloud::kCorridor:
      return "corridor";
  }
  return "unknown";
}

using Point3 = std::array<float, 3>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kSurfaceNoise = 0.004;      // m: the largest offset along a surface normal, new in every scan
constexpr double kOutlierOffset = 0.17;      // m along the normal: past the gate, within the plane test at 4-7 m
constexpr double kOutlierShare = 0.05;       // of an outlier scan's points, a different set in every scan
constexpr double kGroundZ = -0.8;            // m: the ground under the sensor
constexpr double kClutterShare = 1.0 / 3.0;  // of the ground points, the same ones in every scan
constexpr int kSparseRingStride = 3;         // in the quadrant x < 0 <= y only every third ground ring

// A number in [0, 1) fixed by (a, b, salt): splitmix64 of their combination.
double hashUnit(std::uint64_t a, std::uint64_t b, std::uint64_t salt)
{
  std::uint64_t z = a * 0x9e3779b97f4a7c15ULL + b * 0xbf58476d1ce4e5b9ULL + salt * 0x94d049bb133111ebULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

// A rectangle on a plane: its normal points along `azimuth` and `tilt` above the horizontal, its centre lies
// `distance` from the sensor along `azimuth` at sensor height; `columns` points across `half_width` either way along
// the horizontal tangent, `rows` from `v_min` to `v_max` along the up-slope tangent.
struct Patch
{
  double azimuth_deg;
  double tilt_deg;
  double distance;
  double half_width;
  double v_min;
  double v_max;
  int columns;
  int rows;
};

constexpr Patch kRoomWalls[] = {{0.0, 20.0, 4.0, 2.4, -1.2, 1.8, 29, 23},
                                {125.0, -15.0, 5.0, 2.4, -1.2, 1.8, 29, 23},
                                {235.0, 25.0, 6.0, 2.4, -1.2, 1.8, 29, 23}};
// The same walls from 0.2 m above the sensor up: none of their points is a ground candidate.
constexpr Patch kGroundWalls[] = {{0.0, 20.0, 4.0, 2.4, 0.25, 1.8, 20, 12},
                                  {125.0, -15.0, 5.0, 2.4, 0.25, 1.8, 20, 12},
                                  {235.0, 25.0, 6.0, 2.4, 0.25, 1.8, 20, 12}};

double surfaceNoise(std::int64_t seed, std::size_t index)
{
  return kSurfaceNoise * (2.0 * hashUnit(static_cast<std::uint64_t>(seed), index, 1) - 1.0);
}

void appendPatch(std::vector<Point3> & out, const Patch & patch, std::int64_t seed, bool outliers)
{
  const double phi = patch.azimuth_deg * kPi / 180.0;
  const double theta = patch.tilt_deg * kPi / 180.0;
  const double cp = std::cos(phi), sp = std::sin(phi), ct = std::cos(theta), st = std::sin(theta);
  const double normal[3] = {ct * cp, ct * sp, st};
  const double across[3] = {-sp, cp, 0.0};
  const double up[3] = {-st * cp, -st * sp, ct};
  const double centre[3] = {patch.distance * cp, patch.distance * sp, 0.0};
  for (int r = 0; r < patch.rows; ++r) {
    for (int c = 0; c < patch.columns; ++c) {
      const std::size_t index = out.size();
      const double u = -patch.half_width + 2.0 * patch.half_width * c / (patch.columns - 1);
      const double v = patch.v_min + (patch.v_max - patch.v_min) * r / (patch.rows - 1);
      double along = surfaceNoise(seed, index);
      if (outliers && hashUnit(static_cast<std::uint64_t>(seed), index, 4) < kOutlierShare)
        along += kOutlierOffset;
      Point3 point;
      for (int d = 0; d < 3; ++d)
        point[d] = static_cast<float>(centre[d] + u * across[d] + v * up[d] + along * normal[d]);
      out.push_back(point);
    }
  }
}

// The points of a measurement scene for the scan that starts at `seed` ms; `outliers` moves about one point in 20 of
// the room off its wall.
std::vector<Point3> scenePoints(Cloud kind, std::int64_t seed, bool outliers)
{
  std::vector<Point3> out;
  if (kind == Cloud::kRoom) {
    for (const auto & wall : kRoomWalls)
      appendPatch(out, wall, seed, outliers);
  } else if (kind == Cloud::kGround) {
    // Rings 1-5 m out, a point every 0.25 m along each. The two angular sectors of the quadrant x < 0 <= y have too
    // few ground candidates for a fit of their own and take the whole scan's ground plane; the other six fit their
    // own.
    for (int ring = 0; ring <= 16; ++ring) {
      const double radius = 1.0 + 0.25 * ring;
      const int count = static_cast<int>(std::lround(2.0 * kPi * radius / 0.25));
      for (int k = 0; k < count; ++k) {
        const double angle = 2.0 * kPi * (k + 0.5 * (ring % 2)) / count;
        const auto x = static_cast<float>(radius * std::cos(angle));
        const auto y = static_cast<float>(radius * std::sin(angle));
        if (x < 0.0f && y >= 0.0f && ring % kSparseRingStride != 0)
          continue;
        const std::size_t index = out.size();
        double z = kGroundZ + surfaceNoise(seed, index);
        const auto r = static_cast<std::uint64_t>(ring), j = static_cast<std::uint64_t>(k);
        if (hashUnit(r, j, 2) < kClutterShare)
          z = kGroundZ + 0.2 + 0.3 * hashUnit(r, j, 3);
        out.push_back({x, y, static_cast<float>(z)});
      }
    }
    for (const auto & wall : kGroundWalls)
      appendPatch(out, wall, seed, false);
  } else if (kind == Cloud::kCorridor) {
    // The walls 12 m long and 2.4 m high; the patch, 4 x 3 points 0.15 m apart, 0.5 m past the walls' end.
    for (const double side : {1.6, -1.6}) {
      for (int r = 0; r < 20; ++r) {
        for (int c = 0; c < 50; ++c) {
          const std::size_t index = out.size();
          out.push_back({static_cast<float>(-6.0 + 12.0 * c / 49),
                         static_cast<float>(side + surfaceNoise(seed, index)),
                         static_cast<float>(-0.8 + 2.4 * r / 19)});
        }
      }
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        const std::size_t index = out.size();
        out.push_back({static_cast<float>(6.5 + surfaceNoise(seed, index)),
                       static_cast<float>(-0.225 + 0.15 * c),
                       static_cast<float>(0.25 + 0.15 * r)});
      }
    }
  }
  return out;
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
  bool outliers = false;    // a room cloud with about one point in 20 moved off its wall
};

struct Scenario
{
  std::string name;
  bool async_map = false;
  bool typed = false;  // the typed outputs on, with the fixed activation identity and repeat clock
  Cloud warm_up = Cloud::kDense;  // the cloud the warm-up scans carry: the scene the map starts from
  bool dump = false;  // the correspondence dump goes to the scenario's result directory; h_dump.v1 records its files
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

// test_anchor_lifecycle_node's base configuration, with the anchor trace on, and online extrinsic estimation on: the
// node defaults to off, and h_matrix.v1 and filter_update.v1 are to cover the extrinsic columns of H.
Scenario baseScenario(const std::string & name)
{
  Scenario s;
  s.name = name;
  for (const auto & p :
       {rclcpp::Parameter("mapping.extrinsic_T", std::vector<double>{0.0, 0.0, 0.0}),
        rclcpp::Parameter("mapping.extrinsic_R", std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
        rclcpp::Parameter("mapping.extrinsic_est_en", true),
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
  s.claims = {"bootstrap",
              "imu_commit",
              "normal_match",
              "no_match",
              "early_exit_scan",
              "published",
              "odometry",
              "h_matrix_values",
              "full_state_and_covariance"};
  return s;
}

// The match scenario with both typed outputs on: the record adds the fixed activation identity and every
// FrontendObservation and ReferenceHealth message, the reference repeats included.
Scenario typedMatchScenario()
{
  Scenario s = matchScenario();
  s.name = "typed_match";
  s.typed = true;
  setParam(s, rclcpp::Parameter("publish.frontend_observation", true));
  setParam(s, rclcpp::Parameter("publish.reference_health", true));
  setParam(s, rclcpp::Parameter("publish.frontend_clock_domain", "fixture"));
  s.claims.push_back("typed_outputs_with_fixed_identity");
  s.claims.push_back("reference_repeat");
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
  s.claims = {"invalid_scan", "nonadvancing_scan", "imu_retain_empty_window", "imu_retain_hook_visible"};
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

// The measurement scenarios (laserMapping S2): every branch of the H-model the default configuration leaves off, on
// a scene that reaches it. Each starts from the base configuration with its scene as the warm-up's cloud; their
// slots leave the guard its own verdict.
Scenario measurementScenario(const std::string & name, Cloud scene)
{
  Scenario s = baseScenario(name);
  s.warm_up = scene;
  return s;
}

Slot sceneSlot(Cloud cloud, bool moving = false)
{
  const std::string name = cloudName(cloud);
  return slot(moving ? name + "_moving" : name, moving, kGuardNatural, cloud);
}

Slot outlierSlot()
{
  Slot s = slot("room_outliers", false, kGuardNatural, Cloud::kRoom);
  s.outliers = true;
  return s;
}

// The correspondence gate's innovation test (corr_innov_gate_k, scaled by the larger of the pose's projected sigma,
// its rotation floored, and the scan's robust residual scale): two scans with about one room point in 20 moved 0.17 m
// off its wall, which the plane test keeps and the gate refuses.
Scenario gateInnovationScenario()
{
  Scenario s = measurementScenario("h_gate_innov", Cloud::kRoom);
  setParam(s, rclcpp::Parameter("corr_innov_gate_k", 1.0));
  setParam(s, rclcpp::Parameter("corr_gate_mad_scale", 1.0));
  setParam(s, rclcpp::Parameter("corr_gate_rot_sigma_floor_deg", 0.3));
  append(s, 3, sceneSlot(Cloud::kRoom));
  append(s, 2, outlierSlot());
  append(s, 2, sceneSlot(Cloud::kRoom));
  s.claims = {"h_gate_rejected"};
  return s;
}

// The gate's nearest-neighbour bound (corr_nn0_max) alone: the map holds the scene's first scan, so a point's nearest
// map neighbour is its own earlier copy, which the per-scan noise puts up to 8 mm away; the bound refuses the points
// farther than 5 mm from it.
Scenario gateNearestScenario()
{
  Scenario s = measurementScenario("h_gate_nn0", Cloud::kRoom);
  setParam(s, rclcpp::Parameter("corr_nn0_max", 0.005));
  append(s, 5, sceneSlot(Cloud::kRoom));
  s.claims = {"h_gate_rejected"};
  return s;
}

// A robust kernel on the rows (corr_kernel, its scale the larger of corr_kernel_mad_scale times the scan's residual
// scale and corr_kernel_min_scale): every row with a residual is weighted below 1, the outliers far below.
Scenario kernelScenario(const std::string & name, const char * kernel)
{
  Scenario s = measurementScenario(name, Cloud::kRoom);
  setParam(s, rclcpp::Parameter("corr_kernel", kernel));
  append(s, 3, sceneSlot(Cloud::kRoom));
  append(s, 2, outlierSlot());
  append(s, 2, sceneSlot(Cloud::kRoom));
  s.claims = {"h_kernel_weighted"};
  return s;
}

// The neighbour refresh gate (nn_refresh_gate_en): on parked scans the iteration that asks for a new search after
// converging has hardly moved, and the gate skips the search; the turning scans need iterations that do not converge,
// which reuse the cached planes without asking.
Scenario refreshScenario()
{
  Scenario s = measurementScenario("h_nn_refresh", Cloud::kRoom);
  setParam(s, rclcpp::Parameter("nn_refresh_gate_en", true));
  setParam(s, rclcpp::Parameter("nn_refresh_min_shift", 0.05));
  append(s, 4, sceneSlot(Cloud::kRoom));
  append(s, 3, sceneSlot(Cloud::kRoom, true));
  append(s, 3, sceneSlot(Cloud::kRoom));
  s.claims = {"h_nn_skip", "h_nn_reuse"};
  return s;
}

// The ground recovery (ground_recover_en): the plane test refuses ground points whose neighbourhood holds clutter, and
// the recovery adds them back as rows against the ground fitted in their sector (the whole scan's ground in the two
// sparse sectors).
Scenario groundScenario()
{
  Scenario s = measurementScenario("h_ground_recovery", Cloud::kGround);
  setParam(s, rclcpp::Parameter("ground_recover_en", true));
  setParam(s, rclcpp::Parameter("canopy_ground_z", kGroundZ));
  setParam(s, rclcpp::Parameter("canopy_ground_halfband", 0.6));
  setParam(s, rclcpp::Parameter("ground_recover_min_inliers", 75));
  append(s, 8, sceneSlot(Cloud::kGround));
  s.claims = {"h_ground_rows"};
  return s;
}

// The subspace weighting (degeneracy_weighting_en) on the corridor, judged on the Schur complement over the rotation
// columns, with the residual histogram and the spectrum trace on: the vertical gets no weight, the direction along the
// corridor, seen only by the end patch, a partial one, and every row a weight from its normal.
Scenario degeneracyScenario()
{
  Scenario s = measurementScenario("h_degeneracy", Cloud::kCorridor);
  setParam(s, rclcpp::Parameter("degeneracy_weighting_en", true));
  setParam(s, rclcpp::Parameter("degeneracy_schur_en", true));
  setParam(s, rclcpp::Parameter("degeneracy_debug", true));
  setParam(s, rclcpp::Parameter("degeneracy_spectrum_trace", true));
  append(s, 8, sceneSlot(Cloud::kCorridor));
  s.claims = {"h_degeneracy_weighted"};
  return s;
}

// The attitude hold (lidar_attitude_hold_en; it rules out online extrinsic estimation). Its scene rule holds a room
// scan after a room scan, whose weakest position direction is near vertical (z_weak about 0.9 > 0.5); the information
// rule holds the fixture's plane, whose roll/pitch information (about 30) is below 300, where the room's (about 3000)
// is not. The first plane scan matches nothing and goes into the map; the far scan matches nothing and releases the
// hold on the next scan.
Scenario attitudeHoldScenario()
{
  Scenario s = measurementScenario("h_attitude_hold", Cloud::kRoom);
  setParam(s, rclcpp::Parameter("mapping.extrinsic_est_en", false));
  setParam(s, rclcpp::Parameter("lidar_attitude_hold_en", true));
  setParam(s, rclcpp::Parameter("lidar_attitude_hold_far_frac", 2.0));
  setParam(s, rclcpp::Parameter("lidar_attitude_hold_z_weak", 0.5));
  setParam(s, rclcpp::Parameter("lidar_attitude_hold_min_rp_info", 300.0));
  append(s, 3, sceneSlot(Cloud::kRoom));
  append(s, 4, sceneSlot(Cloud::kDense));
  append(s, 3, sceneSlot(Cloud::kRoom));
  s.slots.push_back(sceneSlot(Cloud::kFar));
  append(s, 2, sceneSlot(Cloud::kRoom));
  s.claims = {"h_hold_rows", "h_hold_info"};
  return s;
}

// Online extrinsic estimation on and off (mapping.extrinsic_est_en): the rows' extrinsic columns filled or zero.
Scenario extrinsicScenario(const std::string & name, bool estimate)
{
  Scenario s = measurementScenario(name, Cloud::kRoom);
  setParam(s, rclcpp::Parameter("mapping.extrinsic_est_en", estimate));
  append(s, 6, sceneSlot(Cloud::kRoom));
  s.claims = {estimate ? "h_extrinsic_rows" : "h_extrinsic_fixed"};
  return s;
}

// The correspondence dump (corr_dump_en) over the window that holds the scan ending at 1.55 s, written to the
// scenario's result directory; h_dump.v1 records the three files.
Scenario dumpScenario()
{
  Scenario s = measurementScenario("h_dump", Cloud::kRoom);
  s.dump = true;
  setParam(s, rclcpp::Parameter("corr_dump_en", true));
  setParam(s, rclcpp::Parameter("corr_dump_windows", "1.52:1.58"));
  append(s, 8, sceneSlot(Cloud::kRoom));
  s.claims = {"h_dump_written"};
  return s;
}

// The H-model keys of config/profiles/m20 (three iterations, the refresh gate, the correspondence gate, the
// degeneracy thresholds with the weighting off, roll/pitch considered after a scan with little roll/pitch information,
// extrinsic estimation off) on parked and turning room scans; the fixture's plane, matched once it is in the map,
// leaves too little roll/pitch information for the next scan.
Scenario m20Scenario()
{
  Scenario s = measurementScenario("h_m20", Cloud::kRoom);
  for (const auto & p : {rclcpp::Parameter("max_iteration", 3),
                         rclcpp::Parameter("nn_refresh_gate_en", true),
                         rclcpp::Parameter("nn_refresh_min_shift", 0.05),
                         rclcpp::Parameter("degeneracy_lambda_bad", 0.05),
                         rclcpp::Parameter("degeneracy_lambda_good", 0.30),
                         rclcpp::Parameter("degeneracy_lambda_abs_bad", 6.0),
                         rclcpp::Parameter("degeneracy_lambda_abs_good", 30.0),
                         rclcpp::Parameter("degeneracy_schur_en", true),
                         rclcpp::Parameter("degeneracy_schur_ridge", 1.0e-6),
                         rclcpp::Parameter("degeneracy_hard_projection", true),
                         rclcpp::Parameter("degeneracy_min_weight", 0.0),
                         rclcpp::Parameter("degeneracy_weighting_en", false),
                         rclcpp::Parameter("corr_innov_gate_k", 2.5),
                         rclcpp::Parameter("corr_gate_mad_scale", 1.0),
                         rclcpp::Parameter("corr_gate_rot_sigma_floor_deg", 0.3),
                         rclcpp::Parameter("lidar_consider_rp_en", true),
                         rclcpp::Parameter("lidar_consider_rp_post_release_s", 2.5),
                         rclcpp::Parameter("lidar_consider_rp_min_info", 1500.0),
                         rclcpp::Parameter("mapping.extrinsic_est_en", false)}) {
    setParam(s, p);
  }
  append(s, 3, sceneSlot(Cloud::kRoom));
  append(s, 3, sceneSlot(Cloud::kRoom, true));
  append(s, 2, sceneSlot(Cloud::kRoom));
  append(s, 3, sceneSlot(Cloud::kDense));
  append(s, 2, sceneSlot(Cloud::kRoom));
  s.claims = {"normal_match", "h_nn_skip", "h_rp_consider"};
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
   "IMU batches settled uncommitted through afterImuConsumption (an empty window settles in sync_packages and reports "
   "through afterEmptyImuWindow: imu_retain_hook_visible)",
   ""},
  {"imu_retain_empty_window",
   "records with outcome dropped_empty_imu: the borrowed prefix is settled uncommitted and stays in the buffer",
   "test_anchor_lifecycle_node AnchorLifecycleInputs.AnImuStretchNothingObservedEndsTheAnchorAndStartsTheRecovery"},
  {"imu_retain_hook_visible",
   "empty IMU windows afterEmptyImuWindow reported (imu_retain.v1): the batch settled uncommitted and the samples it "
   "leaves queued",
   ""},
  {"imu_not_processed", "records with outcome imu_not_processed", ""},
  {"imu_coverage_gap",
   "records with outcome imu_coverage_gap",
   "test_anchor_lifecycle_node AnchorLifecycleNode.CoverageGapBeforeTheAnchorStepEndsTheAnchor"},
  {"imu_fault_scan", "records with outcome imu_fault", ""},
  {"normal_match",
   "scans whose H-model iterations found at least one effective point",
   "test_h_model_scan_bounds HModelScanBounds.*"},
  {"no_match", "scans whose H-model iterations found no effective point", ""},
  {"h_matrix_values",
   "H-model exits whose Jacobian rows and residuals were recorded (h_matrix.v1 with at least one row)",
   "test_h_model_scan_bounds HModelScanBounds.*"},
  {"full_state_and_covariance",
   "iterated lidar updates whose full state and covariance were recorded (filter_update.v1)",
   ""},
  {"typed_outputs_with_fixed_identity",
   "FrontendObservation and ReferenceHealth messages received with the fixed activation identity (typed scenario)",
   "test_anchor_lifecycle_node AnchorLifecycleReference.*"},
  {"reference_repeat", "ReferenceHealth messages of KIND_STATE_REPEAT received (typed scenario)", ""},
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
  {"h_gate_rejected",
   "H-model calls whose correspondence gate refused at least one match (h_select.v1 gate_rejected > 0)",
   "test_correspondence_gate CorrespondenceGate.*"},
  {"h_kernel_weighted",
   "H-model calls with a row whose robust kernel weight is below 1 (h_select.v1 rows_weighted > 0)",
   "test_correspondence_gate CorrespondenceGate.KernelWeightsAreOneAtZeroAndDecayWithTheResidual; "
   "test_correspondence_compaction CorrespondenceCompaction.*"},
  {"h_nn_skip", "H-model calls whose requested neighbour search the refresh gate skipped (nn_refresh_gate_en)", ""},
  {"h_nn_reuse",
   "H-model calls that kept the cached planes because the filter asked for no search (an unconverged iteration)",
   "test_cached_plane CachedPlane.*"},
  {"h_ground_rows", "H-model calls to which the ground recovery added rows (ground_recover_en)", ""},
  {"h_degeneracy_weighted",
   "H-model calls with rows whose subspace weighting scaled at least one of them (degeneracy_weighting_en)",
   "test_degeneracy_policy DegeneracyPolicy.*; test_degeneracy_policy DegeneracyWeight.*; "
   "test_degeneracy_policy SchurComplement.*"},
  {"h_hold_rows",
   "H-model calls with rows whose rotation rows the attitude hold's scene rule restricted to yaw",
   "test_attitude_hold AttitudeHold.*"},
  {"h_hold_info",
   "H-model calls with rows whose rotation rows the information rule alone restricted to yaw "
   "(lidar_attitude_hold_min_rp_info)",
   "test_attitude_hold AttitudeHold.RollPitchInformationSeparatesFloorFromNearWall"},
  {"h_rp_consider",
   "H-model calls with rows on scans whose update leaves roll/pitch to the gyro (lidar_consider_rp_en)",
   "test_rp_consider ConsiderPolicy.*; test_rp_consider_filter ConsiderFilter.*"},
  {"h_extrinsic_rows",
   "H-model exits with rows whose extrinsic columns are filled (mapping.extrinsic_est_en true)",
   ""},
  {"h_extrinsic_fixed",
   "H-model exits with rows whose extrinsic columns are all zero (mapping.extrinsic_est_en false)",
   ""},
  {"h_dump_written", "H-model calls that appended to the correspondence dump (corr_dump_en, in a dump window)", ""},
};

// ---------------------------------------------------------------------------
// The child: one component activation, one scenario
// ---------------------------------------------------------------------------

bool makeDirectories(const std::string & path);

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
    // The constructor opens the dump files: their directory exists before it runs.
    if (error.empty() && scenario_.dump) {
      dump_dir_ =
        std::string(FASTLIO_CONTRACT_RESULTS) + "/frontend_contract/" + kFixture + "/" + scenario_.name + ".dump";
      if (!makeDirectories(dump_dir_))
        error = "cannot create the correspondence dump directory " + dump_dir_;
    }
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
      if (scenario_.typed)
        error = appendIdentity();
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
      if (scenario_.typed) {
        const auto typed_qos = rclcpp::QoS(rclcpp::KeepLast(1000)).best_effort();
        frontend_sub_ = observer_->create_subscription<fast_lio_interfaces::msg::FrontendObservation>(
          kFrontendObservationTopic,
          typed_qos,
          [this](fast_lio_interfaces::msg::FrontendObservation::ConstSharedPtr message) {
            onFrontendObservation(*message);
          });
        reference_sub_ = observer_->create_subscription<fast_lio_interfaces::msg::ReferenceHealth>(
          kReferenceHealthTopic, typed_qos, [this](fast_lio_interfaces::msg::ReferenceHealth::ConstSharedPtr message) {
            onReferenceHealth(*message);
          });
      }
      executor_->add_node(subject_->get_node_base_interface());
      executor_->add_node(observer_);
      if (!spinUntil([&] {
            return imu_->get_subscription_count() == 1 && lidar_->get_subscription_count() == 1 &&
                   (!scenario_.typed ||
                    (frontend_sub_->get_publisher_count() == 1 && reference_sub_->get_publisher_count() == 1));
          }))
        error = "front end subscriptions never matched";
    }
    // The repeat throttle keeps the time of its first call: make that call before any input, at the clock's start.
    if (error.empty() && scenario_.typed)
      error = runScanPath("before the first input");
    if (error.empty())
      error = warmUp();
    for (std::size_t i = 0; error.empty() && i < scenario_.slots.size(); ++i) {
      const bool next_moving = i + 1 < scenario_.slots.size() && scenario_.slots[i + 1].moving;
      error = runSlot(scenario_.slots[i], next_moving);
    }
    if (error.empty())
      error = settle(true);
    if (error.empty() && scenario_.typed)
      error = appendTyped();
    if (error.empty() && scenario_.typed && (typed_frontend_total_ == 0 || typed_reference_total_ == 0))
      error = "the typed scenario received no FrontendObservation or no ReferenceHealth";
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
    fast_lio_test::afterHModelMatrices =
      [this](double end, bool valid, const Eigen::MatrixXd & h_x, const Eigen::VectorXd & h) {
        std::ostringstream out;
        out << "h_matrix.v1 end=" << bits(end) << " valid=" << (valid ? 1 : 0) << " h_x=" << h_x.rows() << 'x'
            << h_x.cols() << " h_x_sha256=" << matrixDigest(h_x) << " h=" << h.rows()
            << " h_sha256=" << matrixDigest(h);
        if (!note("afterHModelMatrices", out.str()) || h_x.rows() == 0)
          return;
        ++counts_["h_matrix_values"];
        // Columns 6-11: the extrinsic rotation and translation.
        const bool extrinsic = h_x.cols() >= 12 && (h_x.middleCols(6, 6).array() != 0.0).any();
        ++counts_[extrinsic ? "h_extrinsic_rows" : "h_extrinsic_fixed"];
      };
    fast_lio_test::afterHModelSelection = [this](const fast_lio_test::HModelSelection & s) {
      std::size_t weighted = 0;
      if (!note("afterHModelSelection", formatSelection(s, &weighted)))
        return;
      if (s.gate_rejected > 0)
        ++counts_["h_gate_rejected"];
      if (weighted > 0)
        ++counts_["h_kernel_weighted"];
      if (s.search_requested && !s.searched)
        ++counts_["h_nn_skip"];
      if (!s.search_requested)
        ++counts_["h_nn_reuse"];
      if (s.ground_rows > 0)
        ++counts_["h_ground_rows"];
    };
    fast_lio_test::afterHModelExit = [this](const fast_lio_test::HModelExit & e) {
      if (!note("afterHModelExit", formatExit(e)))
        return;
      // The weighted-row count runs since construction: a call weighted rows when it rose.
      if (e.valid && e.degeneracy_engaged && e.degeneracy_weighted_points > degeneracy_weighted_seen_)
        ++counts_["h_degeneracy_weighted"];
      degeneracy_weighted_seen_ = e.degeneracy_weighted_points;
      if (e.valid && e.hold_rows && e.hold_scene)
        ++counts_["h_hold_rows"];
      if (e.valid && e.hold_rows && e.hold_info && !e.hold_scene)
        ++counts_["h_hold_info"];
      if (e.valid && e.rp_consider)
        ++counts_["h_rp_consider"];
      if (e.dump_written)
        ++counts_["h_dump_written"];
    };
    fast_lio_test::correspondenceDumpDir = dump_dir_;
    fast_lio_test::afterLidarUpdate = [this](double end, const fast_lio_test::FilterSnapshot & s) {
      constexpr std::size_t kDim = 23;
      double diagonal[kDim];
      for (std::size_t i = 0; i < kDim; ++i)
        diagonal[i] = s.P[i * kDim + i];
      std::ostringstream out;
      out << "filter_update.v1 end=" << bits(end) << " pos=" << bitsList(s.pos, 3)
          << " rot_xyzw=" << bitsList(s.rot_xyzw, 4) << " offset_R_L_I_xyzw=" << bitsList(s.offset_R_L_I_xyzw, 4)
          << " offset_T_L_I=" << bitsList(s.offset_T_L_I, 3) << " vel=" << bitsList(s.vel, 3)
          << " bg=" << bitsList(s.bg, 3) << " ba=" << bitsList(s.ba, 3) << " grav=" << bitsList(s.grav, 3)
          << " P=23x23 P_sha256=" << bitsDigest(s.P, kDim * kDim) << " P_diag=" << bitsList(diagonal, kDim);
      if (note("afterLidarUpdate", out.str()))
        ++counts_["full_state_and_covariance"];
    };
    fast_lio_test::afterEmptyImuWindow =
      [this](std::uint64_t sequence, std::size_t borrowed, std::size_t pending, std::uint64_t first, bool faulted) {
        std::ostringstream out;
        out << "imu_retain.v1 seq=" << sequence << " disposition=uncommitted reason=empty_window borrowed=" << borrowed
            << " pending=" << pending << " first_queued_id=" << first << " faulted=" << (faulted ? 1 : 0);
        if (note("afterEmptyImuWindow", out.str()))
          ++counts_["imu_retain_hook_visible"];
      };
    // Installed only for the typed scenario: the activation identity, the repeat clock and the constructor's IDs.
    if (scenario_.typed) {
      fast_lio_test::overrideFrontendInstance = [](std::array<std::uint8_t, 16> & instance) {
        instance = kFixedInstance;
        return true;
      };
      fast_lio_test::referenceRepeatClock = [this] {
        return repeat_clock_;
      };
      fast_lio_test::afterConstruction = [this](const std::vector<fast_lio_test::EffectiveValue> & values) {
        for (const auto & value : values) {
          if (value.name == "derived:config_id:reference_contract_id")
            contract_id_ = value.value;
          else if (value.name == "derived:config_id:reference_config_id")
            config_id_ = value.value;
        }
      };
    }
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
    fast_lio_test::afterHModelMatrices = {};
    fast_lio_test::afterHModelSelection = {};
    fast_lio_test::afterHModelExit = {};
    fast_lio_test::correspondenceDumpDir.clear();
    fast_lio_test::afterLidarUpdate = {};
    fast_lio_test::afterEmptyImuWindow = {};
    fast_lio_test::overrideFrontendInstance = {};
    fast_lio_test::referenceRepeatClock = {};
    fast_lio_test::afterConstruction = {};
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

  // One scan's cloud, its points spread evenly over the scan time. A measurement scene's noise follows the scan's
  // start, so a repeated scan carries the same points.
  void sendLidar(double start, Cloud kind, bool outliers)
  {
    const bool scene = kind == Cloud::kRoom || kind == Cloud::kGround || kind == Cloud::kCorridor;
    const std::vector<Point3> scene_points = scene ? scenePoints(kind, msKey(start), outliers) : std::vector<Point3>();
    const std::size_t points = scene                                          ? scene_points.size()
                               : kind == Cloud::kDense || kind == Cloud::kFar ? kScanPoints
                               : kind == Cloud::kSparse                       ? 3u
                                                                              : 0u;
    const float offset = kind == Cloud::kFar ? kFarOffset : 0.0f;
    const auto length_ns = static_cast<std::uint64_t>(std::llround(kScanLength * 1e9));
    pcl::PointCloud<ouster_ros::Point> cloud;
    cloud.resize(points);
    for (std::size_t i = 0; i < points; ++i) {
      auto & point = cloud[i];
      point = ouster_ros::Point{};
      if (scene) {
        point.x = scene_points[i][0];
        point.y = scene_points[i][1];
        point.z = scene_points[i][2];
      } else {
        point.x = 10.0f + offset + static_cast<float>(i % 317) * 0.01f;
        point.y = -1.6f + static_cast<float>(i / 317) * 0.01f;
        point.z = 1.0f;
      }
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
    Slot warm = slot("warm_up");
    warm.cloud = scenario_.warm_up;
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
    if (s.outliers)
      out << " outliers";
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

  // The typed scenario's activation identity: the fixed instance and the IDs the constructor computed.
  std::string appendIdentity()
  {
    if (contract_id_.empty() || config_id_.empty())
      return "the constructor reported no contract or configuration ID";
    stream_.push_back("identity.v1 source_instance=" + hex(kFixedInstance.data(), kFixedInstance.size()) +
                      " producer_contract_id=" + contract_id_ + " config_id=" + config_id_);
    return "";
  }

  void typedMismatch(const std::string & what)
  {
    if (typed_error_.empty())
      typed_error_ = what;
  }

  void onFrontendObservation(const fast_lio_interfaces::msg::FrontendObservation & m)
  {
    if (!m.has_source_instance || m.source_instance != kFixedInstance)
      typedMismatch("a FrontendObservation without the fixed source instance");
    typed_frontend_.push_back(formatFrontendObservation(m));
    ++typed_received_;
  }

  void onReferenceHealth(const fast_lio_interfaces::msg::ReferenceHealth & m)
  {
    if (!m.has_source_instance || m.source_instance != kFixedInstance)
      typedMismatch("a ReferenceHealth without the fixed source instance");
    if (hex(m.producer_contract_id.data(), m.producer_contract_id.size()) != contract_id_ ||
        hex(m.config_id.data(), m.config_id.size()) != config_id_)
      typedMismatch("a ReferenceHealth whose contract or configuration ID is not the constructor's");
    if (m.kind == fast_lio_interfaces::msg::ReferenceHealth::KIND_STATE_REPEAT)
      ++counts_["reference_repeat"];
    typed_reference_.push_back(formatReferenceHealth(m));
    ++typed_received_;
  }

  // Every typed message published so far, per topic in arrival order. Intra-process delivery waits for the executor:
  // spin until a pass delivers nothing more.
  std::string appendTyped()
  {
    for (std::uint64_t before = typed_received_ + 1; before != typed_received_;) {
      before = typed_received_;
      executor_->spin_some();
    }
    counts_["typed_outputs_with_fixed_identity"] += typed_frontend_.size() + typed_reference_.size();
    typed_frontend_total_ += typed_frontend_.size();
    typed_reference_total_ += typed_reference_.size();
    for (auto & line : typed_frontend_)
      stream_.push_back(std::move(line));
    for (auto & line : typed_reference_)
      stream_.push_back(std::move(line));
    typed_frontend_.clear();
    typed_reference_.clear();
    return typed_error_;
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
      sendLidar(scanStart(cloud_scan), s.cloud, s.outliers);
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
    // The one reference repeat a slot can make falls on its last scan path run: nothing is left to take there.
    if (error.empty() && scenario_.typed)
      repeat_clock_ += kRepeatClockStep;
    if (error.empty())
      error = runScanPath("at the end of slot " + std::to_string(index));
    if (error.empty() && throw_next_imu_process_)
      error = "slot " + std::to_string(index) + ": the IMU processing step it was to fail never ran";
    if (!error.empty())
      return error;
    appendArrivedOdometry();
    if (scenario_.typed) {
      error = appendTyped();
      if (!error.empty())
        return error;
    }
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
    reference_sub_.reset();
    frontend_sub_.reset();
    odom_.reset();
    imu_.reset();
    lidar_.reset();
    observer_.reset();
    factory_.reset();
    loader_.reset();
    executor_.reset();
    rclcpp::shutdown();
  }

  // The correspondence dump's files as the scan path left them: it flushes them after every scan in a window.
  void appendDumpFiles()
  {
    for (const char * name : {"corr_dump.csv", "corr_iter.csv", "corr_scan.csv"}) {
      std::string text;
      std::ostringstream out;
      out << "h_dump.v1 file=" << name;
      if (readFile(dump_dir_ + "/" + name, &text))
        out << " bytes=" << text.size() << " lines=" << std::count(text.begin(), text.end(), '\n')
            << " sha256=" << sha256Hex(text);
      else
        out << " missing";
      stream_.push_back(out.str());
    }
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
    if (scenario_.dump)
      appendDumpFiles();
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
  rclcpp::Subscription<fast_lio_interfaces::msg::FrontendObservation>::SharedPtr frontend_sub_;
  rclcpp::Subscription<fast_lio_interfaces::msg::ReferenceHealth>::SharedPtr reference_sub_;
  std::chrono::steady_clock::time_point repeat_clock_{kRepeatClockStart};
  std::string contract_id_, config_id_;                        // hex, from the constructor's effective configuration
  std::vector<std::string> typed_frontend_, typed_reference_;  // received since the last appendTyped
  std::uint64_t typed_received_ = 0;
  std::uint64_t typed_frontend_total_ = 0;
  std::uint64_t typed_reference_total_ = 0;
  std::string typed_error_;
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
  long degeneracy_weighted_seen_ = 0;            // the subspace weighting's weighted-row count at the last H exit
  std::string dump_dir_;                         // the correspondence dump's directory (dump scenario only)
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
  // Every branch the hooks could not reach before is a counted branch above: none is suspended.
  out << "],\"suspended\":[],\"off_thread_hooks\":{";
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

TEST(FrontendPipelineContract, TypedMatch)
{
  checkScenario(typedMatchScenario());
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

TEST(FrontendPipelineContract, MeasurementGateInnovation)
{
  checkScenario(gateInnovationScenario());
}

TEST(FrontendPipelineContract, MeasurementGateNearestNeighbour)
{
  checkScenario(gateNearestScenario());
}

TEST(FrontendPipelineContract, MeasurementKernelGemanMcClure)
{
  checkScenario(kernelScenario("h_kernel_gm", "gm"));
}

TEST(FrontendPipelineContract, MeasurementKernelCauchy)
{
  checkScenario(kernelScenario("h_kernel_cauchy", "cauchy"));
}

TEST(FrontendPipelineContract, MeasurementNeighbourRefresh)
{
  checkScenario(refreshScenario());
}

TEST(FrontendPipelineContract, MeasurementGroundRecovery)
{
  checkScenario(groundScenario());
}

TEST(FrontendPipelineContract, MeasurementDegeneracyWeighting)
{
  checkScenario(degeneracyScenario());
}

TEST(FrontendPipelineContract, MeasurementAttitudeHold)
{
  checkScenario(attitudeHoldScenario());
}

TEST(FrontendPipelineContract, MeasurementExtrinsicEstimationOn)
{
  checkScenario(extrinsicScenario("h_extrinsic_on", true));
}

TEST(FrontendPipelineContract, MeasurementExtrinsicEstimationOff)
{
  checkScenario(extrinsicScenario("h_extrinsic_off", false));
}

TEST(FrontendPipelineContract, MeasurementCorrespondenceDump)
{
  checkScenario(dumpScenario());
}

TEST(FrontendPipelineContract, MeasurementM20Profile)
{
  checkScenario(m20Scenario());
}

// The worker suite is what the front-end ThreadSanitizer phase runs. Here: the map worker and the reset that joins it.
TEST(FrontendPipelineWorker, AsyncMapDelayAndReset)
{
  checkScenario(workerScenario());
}

// A record without what only a run with the map worker writes: the count of the batches it added.
std::string withoutMapAdds(std::string record)
{
  const std::string tag = " map_adds=";
  for (auto at = record.find(tag); at != std::string::npos; at = record.find(tag, at))
    record.erase(at, record.find_first_not_of("0123456789", at + tag.size()) - at);
  return record;
}

// The entry view under the map worker, as production runs it: the view opens, is searched and closes between the
// worker's additions. The scan path joins the worker before any of that, so each scan finds the map the synchronous
// path gives it, and the record, the worker's counts aside, is the synchronous run's. That run is compared with its
// golden, so this one needs none.
TEST(FrontendPipelineWorker, EntryViewRecoveryRecordsWhatTheSynchronousPathRecords)
{
  const Scenario synchronous = entryViewScenario();
  Scenario worker = synchronous;
  worker.async_map = true;
  worker.claims.push_back("async_map_add");
  const ChildRun expected = runScenario(synchronous);
  const ChildRun actual = runScenario(worker);
  ASSERT_TRUE(expected.ok()) << expected.describe();
  ASSERT_TRUE(actual.ok()) << actual.describe();
  for (const auto & claim : worker.claims)
    EXPECT_GT(actual.count(claim), 0u) << "the worker run did not reach the claimed branch " << claim;
  const Comparison comparison = compareLines(expected.record, withoutMapAdds(actual.record));
  EXPECT_EQ(comparison.result, "equal") << "line " << comparison.line << "\n  synchronous: " << comparison.expected
                                        << "\n  worker: " << comparison.actual;
}

}  // namespace

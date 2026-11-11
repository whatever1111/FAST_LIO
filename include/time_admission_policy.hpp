#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "imu_process_outcome.hpp"
#include "scan_time_policy.hpp"

// Admission-M1 only. No runtime node includes this header. Storage and inputs
// belong to the single caller; all spans stay alive and immutable (except via
// applyPlan) until that State/plan is retired. No plan/apply allocates memory.
namespace fast_lio::time_admission
{
template<class T>
struct Span
{
  T * data = nullptr;
  std::size_t size = 0;
  T & operator[](std::size_t i) const noexcept { return data[i]; }
};
using Bytes = Span<const std::uint8_t>;
using MutableBytes = Span<std::uint8_t>;

inline bool checkedAdd(std::size_t a, std::size_t b, std::size_t & out) noexcept
{
  if (b > std::numeric_limits<std::size_t>::max() - a)
    return false;
  out = a + b;
  return true;
}
inline bool checkedMultiply(std::size_t a, std::size_t b, std::size_t & out) noexcept
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  out = a * b;
  return true;
}
inline bool checkedDeadline(std::int64_t now, std::int64_t wait, std::int64_t & out) noexcept
{
  if (now < 0 || wait < 0 || wait > std::numeric_limits<std::int64_t>::max() - now)
    return false;
  out = now + wait;
  return true;
}
inline std::uint64_t timeBits(double time) noexcept
{
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(time));
  std::memcpy(&bits, &time, sizeof(bits));
  return bits;
}
inline bool validTime(double time) noexcept
{
  return std::isfinite(time) && time >= 0.0;
}
inline double boundarySeconds(std::int64_t ns) noexcept
{
  return static_cast<double>(ns / 1000000000) + static_cast<double>(ns % 1000000000) / 1e9;
}
struct WindowBounds
{
  std::int64_t index = 0;
  double lower = 0.0;
  double upper = 0.0;
};
inline bool boundsAt(std::int64_t index, std::int64_t width, WindowBounds & out) noexcept
{
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  if (index < 0 || width <= 0 || index >= maximum / width)
    return false;
  out = {index, boundarySeconds(index * width), boundarySeconds((index + 1) * width)};
  return out.lower < out.upper;
}
inline bool locateWindow(double time, std::int64_t width, WindowBounds & out) noexcept
{
  if (!validTime(time) || width <= 0)
    return false;
  const long double guess = std::floor(static_cast<long double>(time) * 1000000000.0L / width);
  if (guess < 0 || guess >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
    return false;
  auto index = static_cast<std::int64_t>(guess);
  if (!boundsAt(index, width, out))
    return false;
  // One bounded correction handles rounding in the initial quotient. Boundary
  // equality always uses the same sec+nsec conversion, never an epsilon.
  if (time < out.lower) {
    if (index == 0 || !boundsAt(index - 1, width, out))
      return false;
  } else if (time >= out.upper) {
    if (!boundsAt(index + 1, width, out))
      return false;
  }
  return out.lower <= time && time < out.upper;
}

struct EventContext
{
  std::uint64_t sequence = 0;
  std::int64_t receive_ns = 0;
};
struct Frontier
{
  std::optional<double> H, C, Fc, max_canonical_begin;
};
struct Limits
{
  std::int64_t width_ns = 0, lateness_ns = 0, wait_ns = 0, imu_wait_ns = 0;
  double max_original_span_s = 0.0;
  std::size_t max_points = 0, max_messages = 0, max_windows = 0;
  std::size_t max_identity_records = 0, max_identity_bytes = 0;
  std::size_t max_imu_samples = 0, max_imu_bytes = 0;
  std::size_t max_ingress_bytes = 0, max_owned_bytes = 0;
};
struct StorageCaps
{
  std::size_t ingress_points = 0, identity_records = 0, action_slots = 0, scratch_bytes = 0;
};
struct Dependency
{
  // Caller supplies Fc/predecessor-knot or initialization dependency, not just
  // the first point. Unknown is conservative on cap and never lossless proof.
  bool known = false;
  double from_s = 0.0;
};
struct PointRef
{
  std::uint64_t source_epoch = 0, message_sequence = 0, raw_index = 0;
  double absolute_s = 0.0;
  std::size_t payload_index = 0;
};
inline bool samePoint(const PointRef & a, const PointRef & b) noexcept
{
  return a.source_epoch == b.source_epoch && a.message_sequence == b.message_sequence && a.raw_index == b.raw_index &&
         timeBits(a.absolute_s) == timeBits(b.absolute_s) && a.payload_index == b.payload_index;
}
struct ImmutablePointContent
{
  Bytes bytes;
};
struct MessageView
{
  std::uint64_t source_epoch = 0, source_sequence = 0;
  double canonical_begin_s = 0.0;
  ScanTimeStatus original_status = ScanTimeStatus::kEmpty;
  double original_duration_s = 0.0;
  std::uint64_t original_effective_count = 0, declared_count = 0, decoded_count = 0, effective_count = 0;
  bool declared_known = false, decoded_known = false, effective_known = false;
  Span<const PointRef> points;
  Span<const ImmutablePointContent> payload;
  Bytes message_content;
  Dependency dependency;
};
struct ImuView
{
  std::uint64_t source_epoch = 0, sequence = 0;
  double absolute_s = 0.0;
  Bytes content;
};
struct ImuRecord
{
  std::uint64_t sequence = 0;
  double absolute_s = 0.0;
  std::size_t offset = 0, bytes = 0;
};
inline bool sameImu(const ImuRecord & a, const ImuRecord & b) noexcept
{
  return a.sequence == b.sequence && timeBits(a.absolute_s) == timeBits(b.absolute_s) && a.offset == b.offset &&
         a.bytes == b.bytes;
}
struct Token
{
  std::uint64_t epoch = 0, sequence = 0, revision = 0;
  std::int64_t window_index = 0;
  double begin = 0.0, end = 0.0;
  std::size_t point_count = 0, imu_count = 0;
  bool dependency_known = false;
};
inline bool sameToken(const Token & a, const Token & b) noexcept
{
  return a.epoch == b.epoch && a.sequence == b.sequence && a.revision == b.revision &&
         a.window_index == b.window_index && timeBits(a.begin) == timeBits(b.begin) &&
         timeBits(a.end) == timeBits(b.end) && a.point_count == b.point_count && a.imu_count == b.imu_count &&
         a.dependency_known == b.dependency_known;
}
struct ConsumptionAck
{
  Token token;
  Span<const PointRef> points;
  Span<const ImuRecord> imu;
  std::optional<double> fc_after;
  ImuDisposition disposition = ImuDisposition::kUncommitted;
  ImuProcessReason reason = ImuProcessReason::kNone;
  bool produced_cloud = false;
};
enum class EventKind
{
  kLidar,
  kImu,
  kTick,
  kAck,
  kLidarEof,
  kAllEof,
  kAbort,
  kFault
};
struct Event
{
  EventKind kind = EventKind::kTick;
  MessageView message;
  ImuView imu;
  ConsumptionAck ack;
};
enum class Reason : std::size_t
{
  kInvalidMessage,
  kInvalidSpan,
  kHistory,
  kClosed,
  kDuplicate,
  kUidExpired,
  kCapacityNew,
  kNonadvancing,
  kImuTimeout,
  kEofImuUncovered,
  kImuCapacityLoss,
  kSettled,
  kAbort,
  kFault,
  kCount
};
enum class Fault
{
  kNone,
  kEpoch,
  kIdentityConflict,
  kIdentityExhausted,
  kEventOrder,
  kSequenceExhausted,
  kArithmetic,
  kWindowGeometry,
  kAck,
  kPartialPropagation,
  kAfterEof,
  kImuOrder,
  kImuInFlightCapacity,
  kExplicit,
  kPlanResource
};
enum class Code
{
  kOk,
  kInvalidConfiguration,
  kPlanResourceFailure,
  kStalePlan,
  kAlreadyStopped
};
struct Diagnostics
{
  std::array<std::uint64_t, static_cast<std::size_t>(Reason::kCount)> points{}, messages{};
  std::uint64_t accepted_points = 0, reserved_batches = 0, settled_batches = 0, imu_capacity_samples = 0;
  bool dependency_unknown = false, arithmetic_overflow = false;
  bool watermark_certified = false;  // H-L is a waiting policy, never a lateness certificate.
  ImuProcessOutcome last_ack;
  std::optional<double> last_ack_fc;
};
struct PointRecord
{
  bool used = false;
  PointRef point;
  std::int64_t window_index = 0;
};
struct WindowRecord
{
  bool used = false, ready = false;
  WindowBounds bounds;
  std::int64_t wait_deadline = 0, imu_deadline = 0;
  Dependency dependency;
};
struct IdentityRecord
{
  bool used = false;
  std::uint64_t sequence = 0;
  std::size_t offset = 0, bytes = 0;
};
struct Storage
{
  Span<PointRecord> points;
  Span<WindowRecord> windows;
  Span<IdentityRecord> identities;
  MutableBytes identity_bytes;
  Span<ImuRecord> imu;
  MutableBytes imu_bytes;
  Span<PointRef> token_points;
  Span<ImuRecord> token_imu;
};
enum class ActionKind
{
  kPoint,
  kWindow
};
struct Action
{
  ActionKind kind = ActionKind::kPoint;
  std::size_t index = 0;
  PointRecord point;
  WindowRecord window;
};
struct ScalarState
{
  Frontier frontier;
  Diagnostics diagnostics;
  std::uint64_t event_sequence = 0, expiry_floor = 0, batch_sequence = 0;
  std::int64_t receive_ns = 0;
  Fault fault = Fault::kNone;
  bool lidar_eof = false, all_eof = false, aborted = false, in_flight = false;
  Token token;
  std::optional<double> last_imu_time;
  std::optional<std::uint64_t> last_imu_sequence;
};
namespace detail
{
class Engine;
}
class State;
class PlanWorkspace;
class EventPlan
{
public:
  EventPlan() = default;

private:
  const State * owner_ = nullptr;
  PlanWorkspace * workspace_ = nullptr;
  std::uint64_t revision_ = 0, generation_ = 0;
  friend class detail::Engine;
  friend Code applyPlan(State &, const EventPlan &) noexcept;
};
struct PlanResult
{
  Code code = Code::kPlanResourceFailure;
  EventPlan plan;
};
class State
{
public:
  State() = default;
  State(const State &) = delete;
  State & operator=(const State &) = delete;
  State(State &&) = delete;
  State & operator=(State &&) = delete;
  const ScalarState & scalar() const noexcept { return scalar_; }
  const Limits & limits() const noexcept { return limits_; }
  std::uint64_t revision() const noexcept { return revision_; }
  std::uint64_t epoch() const noexcept { return epoch_; }
  std::size_t ownedBytes() const noexcept { return owned_bytes_; }
  Span<const PointRecord> points() const noexcept { return {storage_.points.data, storage_.points.size}; }
  Span<const WindowRecord> windows() const noexcept { return {storage_.windows.data, storage_.windows.size}; }
  Span<const IdentityRecord> identities() const noexcept
  {
    return {storage_.identities.data, storage_.identities.size};
  }
  Bytes identityBytes() const noexcept { return {storage_.identity_bytes.data, identity_used_}; }
  Span<const ImuRecord> imu() const noexcept { return {storage_.imu.data, imu_count_}; }
  Span<const PointRef> tokenPoints() const noexcept
  {
    return {storage_.token_points.data, scalar_.in_flight ? scalar_.token.point_count : 0};
  }
  Span<const ImuRecord> tokenImu() const noexcept
  {
    return {storage_.token_imu.data, scalar_.in_flight ? scalar_.token.imu_count : 0};
  }
  bool eofComplete() const noexcept
  {
    if (!scalar_.all_eof || scalar_.aborted || scalar_.fault != Fault::kNone || scalar_.in_flight)
      return false;
    for (std::size_t i = 0; i < limits_.max_points; ++i)
      if (storage_.points[i].used)
        return false;
    for (std::size_t i = 0; i < limits_.max_windows; ++i)
      if (storage_.windows[i].used)
        return false;
    return true;
  }

private:
  Storage storage_;
  Limits limits_;
  ScalarState scalar_;
  std::uint64_t epoch_ = 0, revision_ = 0;
  std::size_t identity_used_ = 0, imu_count_ = 0, imu_used_ = 0, owned_bytes_ = 0;
  bool initialized_ = false;
  friend class detail::Engine;
  friend Code initialize(State &, std::uint64_t, const Limits &, Storage, PlanWorkspace &) noexcept;
  friend Code applyPlan(State &, const EventPlan &) noexcept;
  friend void latchPlanResourceFailure(State &) noexcept;
};
struct StateView
{
  const State & state;
};
class PlanWorkspace
{
public:
  PlanWorkspace(Span<Action> actions, Span<PointRef> ingress, Span<std::size_t> indices) noexcept
      : actions_(actions), ingress_(ingress), indices_(indices)
  {}
  PlanWorkspace(const PlanWorkspace &) = delete;
  PlanWorkspace & operator=(const PlanWorkspace &) = delete;
  PlanWorkspace(PlanWorkspace &&) = delete;
  PlanWorkspace & operator=(PlanWorkspace &&) = delete;
  StorageCaps caps(std::size_t identities) const noexcept
  {
    std::size_t bytes = 0;
    checkedMultiply(indices_.size, sizeof(std::size_t), bytes);
    return {ingress_.size, identities, actions_.size, bytes};
  }

private:
  Span<Action> actions_;
  Span<PointRef> ingress_;
  Span<std::size_t> indices_;
  std::uint64_t generation_ = 0;
  const State * owner_ = nullptr;
  std::size_t action_count_ = 0;
  ScalarState next_;
  MessageView message_;
  ImuView imu_;
  std::size_t identity_size_ = 0, imu_drop_ = 0;
  bool add_identity_ = false, append_imu_ = false, clear_ = false, reserve_ = false, valid_ = false;
  friend class detail::Engine;
  friend Code initialize(State &, std::uint64_t, const Limits &, Storage, PlanWorkspace &) noexcept;
  friend Code applyPlan(State &, const EventPlan &) noexcept;
};

inline bool workspaceRequirements(std::size_t points,
                                  std::size_t ingress,
                                  std::size_t windows,
                                  std::size_t messages,
                                  std::size_t & actions,
                                  std::size_t & scratch) noexcept
{
  std::size_t total, twice, window_actions;
  return checkedAdd(points, ingress, total) && checkedMultiply(total, sizeof(std::size_t), scratch) &&
         checkedMultiply(total, 2, twice) && checkedMultiply(windows, 2, window_actions) &&
         checkedAdd(twice, window_actions, actions) && checkedAdd(actions, messages, actions) &&
         checkedAdd(actions, 32, actions);
}

namespace detail
{
struct Region
{
  std::uintptr_t first = 0, last = 0;
};
template<class T>
inline bool region(Span<T> span, Region & out, std::size_t & total) noexcept
{
  std::size_t bytes;
  if (!span.data || span.size == 0 || !checkedMultiply(span.size, sizeof(T), bytes) || !checkedAdd(total, bytes, total))
    return false;
  const auto first = reinterpret_cast<std::uintptr_t>(span.data);
  if (first % alignof(T) != 0 || bytes > std::numeric_limits<std::uintptr_t>::max() - first)
    return false;
  out = {first, first + bytes};
  return true;
}
inline bool validBytes(Bytes bytes) noexcept
{
  return bytes.size == 0 || bytes.data != nullptr;
}
inline bool equalBytes(Bytes a, Bytes b) noexcept
{
  return a.size == b.size && (a.size == 0 || std::memcmp(a.data, b.data, a.size) == 0);
}
// Explicit field encoding avoids padding, borrowed addresses and digest-only
// identity. Both complete message authority and every payload span are retained.
class IdentityWriter
{
public:
  IdentityWriter(std::uint8_t * destination, const std::uint8_t * compare) noexcept
      : destination_(destination), compare_(compare)
  {}
  template<class T>
  void field(const T & value) noexcept
  {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes({reinterpret_cast<const std::uint8_t *>(&value), sizeof(value)});
  }
  void bytes(Bytes value) noexcept
  {
    if (!good_ || !checkedAdd(size_, value.size, size_)) {
      good_ = false;
      return;
    }
    const auto offset = size_ - value.size;
    if (value.size && destination_)
      std::memcpy(destination_ + offset, value.data, value.size);
    if (value.size && compare_ && std::memcmp(compare_ + offset, value.data, value.size) != 0)
      equal_ = false;
  }
  bool good() const noexcept { return good_; }
  bool equal() const noexcept { return equal_; }
  std::size_t size() const noexcept { return size_; }

private:
  std::uint8_t * destination_;
  const std::uint8_t * compare_;
  std::size_t size_ = 0;
  bool good_ = true, equal_ = true;
};
inline void encodeIdentity(IdentityWriter & out, const MessageView & m) noexcept
{
  out.field(m.source_epoch);
  out.field(m.source_sequence);
  out.field(timeBits(m.canonical_begin_s));
  out.field(m.original_status);
  out.field(timeBits(m.original_duration_s));
  out.field(m.original_effective_count);
  out.field(m.declared_count);
  out.field(m.decoded_count);
  out.field(m.effective_count);
  out.field(m.declared_known);
  out.field(m.decoded_known);
  out.field(m.effective_known);
  out.field(m.dependency.known);
  out.field(timeBits(m.dependency.from_s));
  out.field(m.message_content.size);
  out.bytes(m.message_content);
  out.field(m.points.size);
  out.field(m.payload.size);
  for (std::size_t i = 0; i < m.points.size; ++i) {
    const auto & p = m.points[i];
    out.field(p.source_epoch);
    out.field(p.message_sequence);
    out.field(p.raw_index);
    out.field(timeBits(p.absolute_s));
    out.field(p.payload_index);
  }
  for (std::size_t i = 0; i < m.payload.size; ++i) {
    out.field(m.payload[i].bytes.size);
    out.bytes(m.payload[i].bytes);
  }
}
}  // namespace detail

inline Code initialize(
  State & state, std::uint64_t epoch, const Limits & limits, Storage storage, PlanWorkspace & workspace) noexcept
{
  if (state.initialized_ || limits.width_ns <= 0 || limits.lateness_ns < 0 || limits.lateness_ns >= 1000000000 ||
      limits.wait_ns < 0 || limits.imu_wait_ns < 0 || !validTime(limits.max_original_span_s) ||
      limits.max_original_span_s == 0 || limits.max_points == 0 || limits.max_messages == 0 ||
      limits.max_windows == 0 || limits.max_identity_records == 0 || limits.max_identity_bytes == 0 ||
      limits.max_imu_samples == 0 || limits.max_imu_bytes == 0 || limits.max_ingress_bytes == 0 ||
      limits.max_owned_bytes == 0)
    return Code::kInvalidConfiguration;
  std::array<detail::Region, 13> regions;
  std::size_t total = 0, required_actions = 0, required_scratch = 0;
  const bool layout =
    detail::region(Span<State>{&state, 1}, regions[0], total) &&
    detail::region(Span<PlanWorkspace>{&workspace, 1}, regions[1], total) &&
    detail::region(storage.points, regions[2], total) && detail::region(storage.windows, regions[3], total) &&
    detail::region(storage.identities, regions[4], total) &&
    detail::region(storage.identity_bytes, regions[5], total) && detail::region(storage.imu, regions[6], total) &&
    detail::region(storage.imu_bytes, regions[7], total) && detail::region(storage.token_points, regions[8], total) &&
    detail::region(storage.token_imu, regions[9], total) && detail::region(workspace.actions_, regions[10], total) &&
    detail::region(workspace.ingress_, regions[11], total) && detail::region(workspace.indices_, regions[12], total);
  if (!layout || total > limits.max_owned_bytes || limits.max_points > storage.points.size ||
      limits.max_windows > storage.windows.size || limits.max_identity_records > storage.identities.size ||
      limits.max_identity_bytes > storage.identity_bytes.size || limits.max_imu_samples > storage.imu.size ||
      limits.max_imu_bytes > storage.imu_bytes.size || limits.max_points > storage.token_points.size ||
      limits.max_imu_samples > storage.token_imu.size ||
      !workspaceRequirements(limits.max_points,
                             workspace.ingress_.size,
                             limits.max_windows,
                             limits.max_messages,
                             required_actions,
                             required_scratch) ||
      workspace.actions_.size < required_actions || workspace.indices_.size < required_scratch / sizeof(std::size_t))
    return Code::kInvalidConfiguration;
  for (std::size_t i = 0; i < regions.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (regions[i].first < regions[j].last && regions[j].first < regions[i].last)
        return Code::kInvalidConfiguration;
  state.storage_ = storage;
  state.limits_ = limits;
  state.epoch_ = epoch;
  state.owned_bytes_ = total;
  state.initialized_ = true;
  for (std::size_t i = 0; i < storage.points.size; ++i)
    storage.points[i] = {};
  for (std::size_t i = 0; i < storage.windows.size; ++i)
    storage.windows[i] = {};
  for (std::size_t i = 0; i < storage.identities.size; ++i)
    storage.identities[i] = {};
  workspace.valid_ = false;
  workspace.owner_ = &state;
  return Code::kOk;
}

namespace detail
{
class Engine
{
public:
  Engine(const State & state, PlanWorkspace & workspace) noexcept : s_(state), w_(workspace), n_(workspace.next_) {}
  PlanResult plan(const Event & event, EventContext context) noexcept
  {
    w_.valid_ = false;
    if (!s_.initialized_)
      return {Code::kInvalidConfiguration, {}};
    if (w_.owner_ != &s_)
      return {Code::kPlanResourceFailure, {}};
    if (s_.scalar_.fault != Fault::kNone || s_.scalar_.aborted)
      return {Code::kAlreadyStopped, {}};
    if (w_.generation_ == std::numeric_limits<std::uint64_t>::max())
      return {Code::kPlanResourceFailure, {}};
    ++w_.generation_;
    w_.action_count_ = 0;
    w_.add_identity_ = false;
    w_.append_imu_ = false;
    w_.clear_ = false;
    w_.reserve_ = false;
    w_.imu_drop_ = 0;
    n_ = s_.scalar_;
    if (context.sequence <= n_.event_sequence || context.receive_ns < n_.receive_ns || context.receive_ns < 0)
      fault(Fault::kEventOrder);
    else if (context.sequence == std::numeric_limits<std::uint64_t>::max() ||
             s_.revision_ >= std::numeric_limits<std::uint64_t>::max() - 1)
      fault(Fault::kSequenceExhausted);
    else {
      n_.event_sequence = context.sequence;
      n_.receive_ns = context.receive_ns;
      deadlines();
      if (healthy())
        dispatch(event);
      if (healthy())
        reserve();
    }
    if (resource_failure_)
      return {Code::kPlanResourceFailure, {}};
    w_.valid_ = true;
    EventPlan handle;
    handle.owner_ = &s_;
    handle.workspace_ = &w_;
    handle.revision_ = s_.revision_;
    handle.generation_ = w_.generation_;
    return {Code::kOk, handle};
  }
  static Code apply(State & s, const EventPlan & plan) noexcept;

private:
  const State & s_;
  PlanWorkspace & w_;
  ScalarState & n_;
  bool resource_failure_ = false;
  bool healthy() const noexcept { return n_.fault == Fault::kNone && !n_.aborted && !resource_failure_; }
  bool increment(std::uint64_t & value, std::size_t count) noexcept
  {
    if (count > std::numeric_limits<std::uint64_t>::max() - value) {
      n_.diagnostics.arithmetic_overflow = true;
      fault(Fault::kArithmetic);
      return false;
    }
    value += count;
    return true;
  }
  void count(Reason reason, std::size_t points, std::size_t messages = 0) noexcept
  {
    const auto i = static_cast<std::size_t>(reason);
    increment(n_.diagnostics.points[i], points);
    increment(n_.diagnostics.messages[i], messages);
  }
  PointRecord pointAt(std::size_t i) const noexcept
  {
    for (std::size_t a = 0; a < w_.action_count_; ++a)
      if (w_.actions_[a].kind == ActionKind::kPoint && w_.actions_[a].index == i)
        return w_.actions_[a].point;
    return w_.clear_ ? PointRecord{} : s_.storage_.points[i];
  }
  WindowRecord windowAt(std::size_t i) const noexcept
  {
    for (std::size_t a = 0; a < w_.action_count_; ++a)
      if (w_.actions_[a].kind == ActionKind::kWindow && w_.actions_[a].index == i)
        return w_.actions_[a].window;
    return w_.clear_ ? WindowRecord{} : s_.storage_.windows[i];
  }
  void write(Action action) noexcept
  {
    for (std::size_t i = 0; i < w_.action_count_; ++i)
      if (w_.actions_[i].kind == action.kind && w_.actions_[i].index == action.index) {
        w_.actions_[i] = action;
        return;
      }
    if (w_.action_count_ == w_.actions_.size) {
      resource_failure_ = true;
      return;
    }
    w_.actions_[w_.action_count_++] = action;
  }
  void writePoint(std::size_t i, PointRecord p) noexcept { write({ActionKind::kPoint, i, p, {}}); }
  void writeWindow(std::size_t i, WindowRecord r) noexcept { write({ActionKind::kWindow, i, {}, r}); }
  void fault(Fault reason) noexcept
  {
    if (n_.fault != Fault::kNone)
      return;
    n_.fault = reason;
    clear(Reason::kFault);
  }
  void clear(Reason reason) noexcept
  {
    std::size_t points = 0;
    for (std::size_t i = 0; i < s_.limits_.max_points; ++i)
      points += pointAt(i).used;
    // Do not recursively count if the diagnostic counter itself overflowed.
    const auto r = static_cast<std::size_t>(reason);
    if (points <= std::numeric_limits<std::uint64_t>::max() - n_.diagnostics.points[r])
      n_.diagnostics.points[r] += points;
    else
      n_.diagnostics.arithmetic_overflow = true;
    w_.action_count_ = 0;
    w_.clear_ = true;
    w_.reserve_ = false;
    n_.in_flight = false;
  }
  bool active(std::uint64_t sequence) const noexcept
  {
    for (std::size_t i = 0; i < s_.limits_.max_points; ++i) {
      const auto p = pointAt(i);
      if (p.used && p.point.message_sequence == sequence)
        return true;
    }
    return false;
  }
  void advanceClosed(double upper) noexcept
  {
    if (!n_.frontier.C || upper > *n_.frontier.C)
      n_.frontier.C = upper;
  }
  bool flightWindow(const WindowRecord & r) const noexcept
  {
    return n_.in_flight && r.bounds.index == n_.token.window_index;
  }
  void terminalWindow(std::size_t index, Reason reason) noexcept
  {
    const auto r = windowAt(index);
    for (std::size_t p = 0; p < s_.limits_.max_points; ++p) {
      const auto point = pointAt(p);
      if (point.used && point.window_index == r.bounds.index) {
        count(reason, 1);
        writePoint(p, {});
      }
    }
    writeWindow(index, {});
  }
  void closePrefix(double upper) noexcept
  {
    advanceClosed(upper);
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      auto r = windowAt(i);
      if (!r.used || r.ready || r.bounds.upper > upper)
        continue;
      if (!checkedDeadline(n_.receive_ns, s_.limits_.imu_wait_ns, r.imu_deadline)) {
        fault(Fault::kArithmetic);
        return;
      }
      r.ready = true;
      writeWindow(i, r);
    }
  }
  void terminalPrefix(double upper, Reason reason) noexcept
  {
    advanceClosed(upper);
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (r.used && !flightWindow(r) && r.bounds.upper <= upper)
        terminalWindow(i, reason);
    }
  }
  void deadlines() noexcept
  {
    std::optional<double> close;
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (r.used && !r.ready && r.wait_deadline <= n_.receive_ns && (!close || r.bounds.upper > *close))
        close = r.bounds.upper;
    }
    if (close)
      closePrefix(*close);
    // Includes zero imu_wait on a just-closed window. Never borrow here.
    std::optional<double> terminal;
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (r.used && r.ready && !flightWindow(r) && !tailCovered(r) && r.imu_deadline <= n_.receive_ns &&
          (!terminal || r.bounds.upper > *terminal))
        terminal = r.bounds.upper;
    }
    if (terminal && healthy())
      terminalPrefix(*terminal, Reason::kImuTimeout);
  }
  bool tailCovered(const WindowRecord & window) const noexcept
  {
    if (imuCount() == 0)
      return false;
    const auto tail = imuAt(imuCount() - 1).absolute_s;
    for (std::size_t i = 0; i < s_.limits_.max_points; ++i) {
      const auto p = pointAt(i);
      if (p.used && p.window_index == window.bounds.index && p.point.absolute_s > tail)
        return false;
    }
    return true;
  }
  void dispatch(const Event & event) noexcept
  {
    switch (event.kind) {
      case EventKind::kLidar:
        if (n_.lidar_eof)
          fault(Fault::kAfterEof);
        else
          ingest(event.message);
        break;
      case EventKind::kImu:
        if (n_.all_eof)
          fault(Fault::kAfterEof);
        else
          ingestImu(event.imu);
        break;
      case EventKind::kAck:
        acknowledge(event.ack);
        break;
      case EventKind::kLidarEof:
        if (n_.lidar_eof)
          fault(Fault::kAfterEof);
        else {
          n_.lidar_eof = true;
          closeAll();
        }
        break;
      case EventKind::kAllEof:
        if (n_.all_eof)
          fault(Fault::kAfterEof);
        else {
          n_.lidar_eof = true;
          n_.all_eof = true;
          closeAll();
        }
        break;
      case EventKind::kAbort:
        n_.aborted = true;
        clear(Reason::kAbort);
        break;
      case EventKind::kFault:
        fault(Fault::kExplicit);
        break;
      case EventKind::kTick:
        if (n_.all_eof)
          fault(Fault::kAfterEof);
        break;
      default:
        fault(Fault::kExplicit);
        break;
    }
  }
  void closeAll() noexcept
  {
    std::optional<double> upper;
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (r.used && (!upper || r.bounds.upper > *upper))
        upper = r.bounds.upper;
    }
    if (upper)
      closePrefix(*upper);
  }
  void expire(std::uint64_t sequence) noexcept
  {
    if (sequence == std::numeric_limits<std::uint64_t>::max()) {
      fault(Fault::kIdentityExhausted);
      return;
    }
    n_.expiry_floor = std::max(n_.expiry_floor, sequence + 1);
  }
  bool messageShape(const MessageView & m, std::size_t & identity_size) noexcept
  {
    if (!m.declared_known || !m.decoded_known || !m.effective_known || m.declared_count < m.decoded_count ||
        m.decoded_count < m.effective_count || m.effective_count != m.points.size ||
        m.original_effective_count != m.points.size || m.payload.size != m.points.size ||
        (m.points.size && (!m.points.data || !m.payload.data)) || !validBytes(m.message_content) ||
        m.message_content.size == 0 || !validTime(m.original_duration_s) ||
        (m.dependency.known && !validTime(m.dependency.from_s)))
      return false;
    for (std::size_t i = 0; i < m.payload.size; ++i)
      if (!validBytes(m.payload[i].bytes) || m.payload[i].bytes.size == 0)
        return false;
    for (std::size_t i = 0; i < m.points.size; ++i) {
      const auto & p = m.points[i];
      if (p.source_epoch != m.source_epoch || p.message_sequence != m.source_sequence ||
          p.raw_index >= m.declared_count || p.payload_index >= m.payload.size || !validTime(p.absolute_s) ||
          (m.dependency.known && m.dependency.from_s > p.absolute_s))
        return false;
    }
    IdentityWriter measure(nullptr, nullptr);
    encodeIdentity(measure, m);
    identity_size = measure.size();
    return measure.good();
  }
  bool retainIdentity(const MessageView & m, std::size_t bytes) noexcept
  {
    // Retire only non-active records. Advancing a sequence floor is a bounded,
    // conservative alternative to retaining rejected content, never forgetting
    // an active message. No per-record eviction actions are needed.
    for (std::size_t attempt = 0; attempt <= s_.limits_.max_identity_records; ++attempt) {
      std::size_t records = 0, used = 0;
      std::optional<std::uint64_t> oldest;
      for (std::size_t i = 0; i < s_.limits_.max_identity_records; ++i) {
        const auto & r = s_.storage_.identities[i];
        if (!r.used)
          continue;
        const bool is_active = active(r.sequence);
        if (!is_active && r.sequence < n_.expiry_floor)
          continue;
        ++records;
        if (!checkedAdd(used, r.bytes, used)) {
          fault(Fault::kArithmetic);
          return false;
        }
        if (!is_active && (!oldest || r.sequence < *oldest))
          oldest = r.sequence;
      }
      if (records < s_.limits_.max_identity_records && bytes <= s_.limits_.max_identity_bytes - used) {
        w_.add_identity_ = true;
        w_.message_ = m;
        w_.identity_size_ = bytes;
        return true;
      }
      if (!oldest)
        break;
      expire(*oldest);
      if (!healthy())
        return false;
    }
    expire(m.source_sequence);
    return false;
  }
  bool knownIdentity(const MessageView & m, std::size_t bytes) noexcept
  {
    for (std::size_t i = 0; i < s_.limits_.max_identity_records; ++i) {
      const auto & r = s_.storage_.identities[i];
      if (!r.used || r.sequence != m.source_sequence || (!active(r.sequence) && r.sequence < n_.expiry_floor))
        continue;
      if (r.bytes != bytes) {
        fault(Fault::kIdentityConflict);
        return true;
      }
      IdentityWriter compare(nullptr, s_.storage_.identity_bytes.data + r.offset);
      encodeIdentity(compare, m);
      if (!compare.equal())
        fault(Fault::kIdentityConflict);
      else
        count(Reason::kDuplicate, m.points.size, 1);
      return true;
    }
    if (m.source_sequence < n_.expiry_floor) {
      count(Reason::kUidExpired, m.points.size, 1);
      return true;
    }
    return false;
  }
  Reason classify(const PointRef & p) const noexcept
  {
    if (n_.frontier.Fc && p.absolute_s < *n_.frontier.Fc)
      return Reason::kHistory;
    if (n_.frontier.C && p.absolute_s < *n_.frontier.C)
      return Reason::kClosed;
    return Reason::kCount;
  }
  bool duplicateInMessage(const MessageView & m, std::size_t index) noexcept
  {
    const auto & p = m.points[index];
    for (std::size_t i = 0; i < index; ++i) {
      const auto & previous = m.points[i];
      if (p.raw_index != previous.raw_index)
        continue;
      if (timeBits(p.absolute_s) != timeBits(previous.absolute_s) ||
          !equalBytes(m.payload[p.payload_index].bytes, m.payload[previous.payload_index].bytes))
        fault(Fault::kIdentityConflict);
      else
        count(Reason::kDuplicate, 1);
      return true;
    }
    return false;
  }
  void rejectUnstaged(const MessageView & m, Reason reason) noexcept
  {
    count(reason, m.points.size, 1);
    expire(m.source_sequence);
  }
  void ingest(const MessageView & m) noexcept
  {
    if (m.source_epoch != s_.epoch_) {
      fault(Fault::kEpoch);
      return;
    }
    InputEpochGuard guard;
    if (!guard.observe(n_.frontier.max_canonical_begin.value_or(-1.0), m.canonical_begin_s, 1.0)) {
      fault(Fault::kEpoch);
      return;
    }
    if (!n_.frontier.max_canonical_begin || m.canonical_begin_s > *n_.frontier.max_canonical_begin)
      n_.frontier.max_canonical_begin = m.canonical_begin_s;
    bool known = false;
    for (std::size_t i = 0; i < s_.limits_.max_identity_records; ++i) {
      const auto & r = s_.storage_.identities[i];
      if (r.used && r.sequence == m.source_sequence && (active(r.sequence) || r.sequence >= n_.expiry_floor))
        known = true;
    }
    if (!known && m.source_sequence < n_.expiry_floor) {
      count(Reason::kUidExpired, m.points.size, 1);
      return;
    }
    // Check ingress dimensions before touching pointed-to elements or staging.
    if (m.points.size > w_.ingress_.size || m.payload.size > w_.ingress_.size ||
        m.message_content.size > s_.limits_.max_ingress_bytes) {
      if (known)
        fault(Fault::kIdentityConflict);
      else
        rejectUnstaged(m, Reason::kCapacityNew);
      return;
    }
    std::size_t bytes = 0;
    if (!messageShape(m, bytes)) {
      if (known)
        fault(Fault::kIdentityConflict);
      else
        rejectUnstaged(m, Reason::kInvalidMessage);
      return;
    }
    if (bytes > s_.limits_.max_ingress_bytes) {
      if (known)
        fault(Fault::kIdentityConflict);
      else
        rejectUnstaged(m, Reason::kCapacityNew);
      return;
    }
    if (knownIdentity(m, bytes) || !healthy())
      return;
    if (m.original_status != ScanTimeStatus::kValid || m.original_duration_s > s_.limits_.max_original_span_s) {
      retainIdentity(m, bytes);
      count(Reason::kInvalidSpan, m.points.size, 1);
      return;
    }
    std::size_t candidates = 0;
    for (std::size_t i = 0; i < m.points.size; ++i) {
      if (duplicateInMessage(m, i))
        continue;
      if (!healthy())
        return;
      const auto reason = classify(m.points[i]);
      if (reason != Reason::kCount)
        count(reason, 1);
      else
        w_.ingress_[candidates++] = m.points[i];
    }
    if (!healthy())
      return;
    const bool identity_retained = retainIdentity(m, bytes);
    if (!healthy())
      return;
    if (candidates == 0)
      return;
    const bool capacity_fits = identity_retained && fits(candidates);
    if (!healthy())
      return;
    if (!capacity_fits) {
      count(Reason::kCapacityNew, candidates, 1);
      return;
    }
    insert(candidates, m.dependency);
    if (!healthy())
      return;
    increment(n_.diagnostics.accepted_points, candidates);
    watermark();
  }
  bool fits(std::size_t candidates) noexcept
  {
    std::size_t points = 0, messages = 0, windows = 0;
    for (std::size_t i = 0; i < s_.limits_.max_points; ++i)
      points += pointAt(i).used;
    for (std::size_t i = 0; i < s_.limits_.max_identity_records; ++i) {
      const auto & r = s_.storage_.identities[i];
      if (r.used && active(r.sequence))
        ++messages;
    }
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i)
      windows += windowAt(i).used;
    if (candidates > s_.limits_.max_points - points || messages >= s_.limits_.max_messages)
      return false;
    for (std::size_t i = 0; i < candidates; ++i) {
      WindowBounds b;
      if (!locateWindow(w_.ingress_[i].absolute_s, s_.limits_.width_ns, b)) {
        fault(Fault::kWindowGeometry);
        return false;
      }
      bool existing = false;
      for (std::size_t j = 0; j < s_.limits_.max_windows; ++j) {
        const auto r = windowAt(j);
        existing = existing || (r.used && r.bounds.index == b.index);
      }
      for (std::size_t j = 0; j < i && !existing; ++j) {
        WindowBounds previous;
        if (!locateWindow(w_.ingress_[j].absolute_s, s_.limits_.width_ns, previous)) {
          fault(Fault::kWindowGeometry);
          return false;
        }
        existing = previous.index == b.index;
      }
      if (!existing && ++windows > s_.limits_.max_windows)
        return false;
    }
    return true;
  }
  void insert(std::size_t candidates, Dependency dependency) noexcept
  {
    for (std::size_t p = 0; p < candidates && healthy(); ++p) {
      const auto point = w_.ingress_[p];
      WindowBounds b;
      if (!locateWindow(point.absolute_s, s_.limits_.width_ns, b)) {
        fault(Fault::kWindowGeometry);
        return;
      }
      std::size_t window = s_.limits_.max_windows, empty = window;
      for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
        const auto r = windowAt(i);
        if (r.used && r.bounds.index == b.index) {
          window = i;
          break;
        }
        if (!r.used)
          empty = i;
      }
      if (window == s_.limits_.max_windows) {
        window = empty;
        WindowRecord r;
        r.used = true;
        r.bounds = b;
        r.dependency = dependency;
        if (!checkedDeadline(n_.receive_ns, s_.limits_.wait_ns, r.wait_deadline)) {
          fault(Fault::kArithmetic);
          return;
        }
        writeWindow(window, r);
      } else {
        auto r = windowAt(window);
        if (!dependency.known)
          r.dependency.known = false;
        else if (r.dependency.known)
          r.dependency.from_s = std::min(r.dependency.from_s, dependency.from_s);
        writeWindow(window, r);
      }
      for (std::size_t i = 0; i < s_.limits_.max_points; ++i)
        if (!pointAt(i).used) {
          writePoint(i, {true, point, b.index});
          break;
        }
      if (!n_.frontier.H || point.absolute_s > *n_.frontier.H)
        n_.frontier.H = point.absolute_s;
    }
  }
  void watermark() noexcept
  {
    if (!n_.frontier.H)
      return;
    const double watermark = *n_.frontier.H - static_cast<double>(s_.limits_.lateness_ns) / 1e9;
    if (watermark >= 0.0) {
      WindowBounds b;
      if (!locateWindow(watermark, s_.limits_.width_ns, b)) {
        fault(Fault::kWindowGeometry);
        return;
      }
      closePrefix(b.lower);
    }
    if (s_.limits_.wait_ns == 0 && healthy())
      closeAll();
  }
  void ingestImu(const ImuView & imu) noexcept
  {
    if (imu.source_epoch != s_.epoch_) {
      fault(Fault::kEpoch);
      return;
    }
    if (!validTime(imu.absolute_s) || !validBytes(imu.content) || imu.content.size == 0) {
      fault(Fault::kImuOrder);
      return;
    }
    if ((n_.last_imu_sequence && imu.sequence <= *n_.last_imu_sequence) ||
        (n_.last_imu_time && imu.absolute_s < *n_.last_imu_time)) {
      fault(Fault::kImuOrder);
      return;
    }
    n_.last_imu_sequence = imu.sequence;
    n_.last_imu_time = imu.absolute_s;
    std::size_t count = s_.imu_count_ - w_.imu_drop_, bytes = s_.imu_used_;
    for (std::size_t i = 0; i < w_.imu_drop_; ++i)
      bytes -= s_.storage_.imu[i].bytes;
    if (imu.content.size > s_.limits_.max_imu_bytes) {
      // A sample cannot fit even alone; affected dependencies are all unissued.
      if (n_.in_flight) {
        fault(Fault::kImuInFlightCapacity);
        return;
      }
      closeAll();
      for (std::size_t i = 0; i < s_.limits_.max_windows; ++i)
        if (windowAt(i).used)
          terminalWindow(i, Reason::kImuCapacityLoss);
      n_.diagnostics.dependency_unknown = true;
      increment(n_.diagnostics.imu_capacity_samples, 1);
      return;
    }
    std::size_t sum;
    if (!checkedAdd(bytes, imu.content.size, sum)) {
      fault(Fault::kArithmetic);
      return;
    }
    const auto initial_drop = w_.imu_drop_;
    while (count >= s_.limits_.max_imu_samples || sum > s_.limits_.max_imu_bytes) {
      if (n_.in_flight) {
        fault(Fault::kImuInFlightCapacity);
        return;
      }
      sum -= s_.storage_.imu[w_.imu_drop_++].bytes;
      --count;
    }
    if (w_.imu_drop_ > initial_drop) {
      capacityLoss(s_.storage_.imu[initial_drop].absolute_s, s_.storage_.imu[w_.imu_drop_ - 1].absolute_s);
      increment(n_.diagnostics.imu_capacity_samples, w_.imu_drop_ - initial_drop);
    }
    w_.append_imu_ = true;
    w_.imu_ = imu;
  }
  void capacityLoss(double first, double last) noexcept
  {
    std::optional<double> upper;
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (!r.used || flightWindow(r))
        continue;
      if (!r.dependency.known)
        n_.diagnostics.dependency_unknown = true;
      std::optional<double> end;
      for (std::size_t p = 0; p < s_.limits_.max_points; ++p) {
        const auto point = pointAt(p);
        if (point.used && point.window_index == r.bounds.index && (!end || point.point.absolute_s > *end))
          end = point.point.absolute_s;
      }
      if ((!r.dependency.known || (end && r.dependency.from_s <= last && *end >= first)) &&
          (!upper || r.bounds.upper > *upper))
        upper = r.bounds.upper;
    }
    if (upper)
      terminalPrefix(*upper, Reason::kImuCapacityLoss);
  }
  void acknowledge(const ConsumptionAck & ack) noexcept
  {
    if (!n_.in_flight || !sameToken(ack.token, n_.token) || ack.points.size != n_.token.point_count ||
        ack.imu.size != n_.token.imu_count || !ack.points.data || !ack.imu.data ||
        (ack.fc_after && !validTime(*ack.fc_after)) ||
        (n_.frontier.Fc && (!ack.fc_after || *ack.fc_after < *n_.frontier.Fc)) ||
        static_cast<std::size_t>(ack.reason) >= static_cast<std::size_t>(ImuProcessReason::kCount) ||
        static_cast<unsigned>(ack.disposition) > static_cast<unsigned>(ImuDisposition::kFatal)) {
      fault(Fault::kAck);
      return;
    }
    for (std::size_t i = 0; i < ack.points.size; ++i)
      if (!samePoint(ack.points[i], s_.storage_.token_points[i])) {
        fault(Fault::kAck);
        return;
      }
    for (std::size_t i = 0; i < ack.imu.size; ++i)
      if (!sameImu(ack.imu[i], s_.storage_.token_imu[i])) {
        fault(Fault::kAck);
        return;
      }
    n_.frontier.Fc = ack.fc_after;
    n_.diagnostics.last_ack = {ack.disposition, ack.reason, ack.produced_cloud};
    n_.diagnostics.last_ack_fc = ack.fc_after;
    if (ack.disposition == ImuDisposition::kFatal || ack.reason == ImuProcessReason::kPartialPropagation) {
      fault(Fault::kPartialPropagation);
      return;
    }
    for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
      const auto r = windowAt(i);
      if (r.used && flightWindow(r))
        terminalWindow(i, Reason::kSettled);
    }
    n_.in_flight = false;
    increment(n_.diagnostics.settled_batches, 1);
    if (ack.disposition != ImuDisposition::kUncommitted)
      while (w_.imu_drop_ < s_.imu_count_ && s_.storage_.imu[w_.imu_drop_].absolute_s <= ack.token.end)
        ++w_.imu_drop_;
  }
  std::size_t imuCount() const noexcept { return s_.imu_count_ - w_.imu_drop_ + w_.append_imu_; }
  ImuRecord imuAt(std::size_t index) const noexcept
  {
    const auto old = s_.imu_count_ - w_.imu_drop_;
    if (index < old)
      return s_.storage_.imu[index + w_.imu_drop_];
    return {w_.imu_.sequence, w_.imu_.absolute_s, 0, w_.imu_.content.size};
  }
  void reserve() noexcept
  {
    if (n_.in_flight)
      return;
    // At most W actual windows are examined. Empty gaps allocate no windows.
    for (std::size_t attempt = 0; attempt < s_.limits_.max_windows && healthy(); ++attempt) {
      std::size_t oldest = s_.limits_.max_windows;
      for (std::size_t i = 0; i < s_.limits_.max_windows; ++i) {
        const auto r = windowAt(i);
        if (r.used && (oldest == s_.limits_.max_windows || r.bounds.index < windowAt(oldest).bounds.index))
          oldest = i;
      }
      if (oldest == s_.limits_.max_windows || !windowAt(oldest).ready)
        return;
      const auto r = windowAt(oldest);
      std::optional<double> begin, end;
      std::size_t points = 0;
      for (std::size_t i = 0; i < s_.limits_.max_points; ++i) {
        const auto p = pointAt(i);
        if (!p.used || p.window_index != r.bounds.index)
          continue;
        if (n_.frontier.Fc && p.point.absolute_s < *n_.frontier.Fc) {
          count(Reason::kHistory, 1);
          writePoint(i, {});
          continue;
        }
        ++points;
        if (!begin || p.point.absolute_s < *begin)
          begin = p.point.absolute_s;
        if (!end || p.point.absolute_s > *end)
          end = p.point.absolute_s;
      }
      if (!end || (n_.frontier.Fc && *end <= *n_.frontier.Fc)) {
        terminalWindow(oldest, Reason::kNonadvancing);
        continue;
      }
      const auto imus = imuCount();
      if (imus == 0 || imuAt(imus - 1).absolute_s < *end) {
        if (n_.all_eof) {
          terminalWindow(oldest, Reason::kEofImuUncovered);
          continue;
        }
        return;
      }
      if (n_.batch_sequence == std::numeric_limits<std::uint64_t>::max()) {
        fault(Fault::kSequenceExhausted);
        return;
      }
      std::size_t prefix = 0;
      while (prefix < imus && imuAt(prefix).absolute_s < *end)
        ++prefix;
      if (prefix < imus)
        ++prefix;
      n_.token = {s_.epoch_,
                  ++n_.batch_sequence,
                  s_.revision_ + 1,
                  r.bounds.index,
                  *begin,
                  *end,
                  points,
                  prefix,
                  r.dependency.known};
      n_.diagnostics.dependency_unknown = n_.diagnostics.dependency_unknown || !r.dependency.known;
      n_.in_flight = true;
      w_.reserve_ = true;
      increment(n_.diagnostics.reserved_batches, 1);
      return;
    }
  }
  static bool finalActive(const State & s, std::uint64_t sequence) noexcept
  {
    for (std::size_t i = 0; i < s.limits_.max_points; ++i)
      if (s.storage_.points[i].used && s.storage_.points[i].point.message_sequence == sequence)
        return true;
    return false;
  }
  static void compactIdentities(State & s, PlanWorkspace & w) noexcept
  {
    // Existing offsets are in insertion order, independent of slot reuse.
    std::size_t count = 0;
    for (std::size_t i = 0; i < s.limits_.max_identity_records; ++i) {
      auto & r = s.storage_.identities[i];
      if (r.used && !finalActive(s, r.sequence) && r.sequence < w.next_.expiry_floor)
        r.used = false;
      if (r.used)
        ++count;
    }
    std::size_t offset = 0, previous = 0;
    bool first = true;
    for (std::size_t remaining = count; remaining > 0; --remaining) {
      std::size_t slot = s.limits_.max_identity_records;
      for (std::size_t i = 0; i < s.limits_.max_identity_records; ++i) {
        const auto & r = s.storage_.identities[i];
        if (r.used && (first || r.offset > previous) &&
            (slot == s.limits_.max_identity_records || r.offset < s.storage_.identities[slot].offset))
          slot = i;
      }
      auto & r = s.storage_.identities[slot];
      previous = r.offset;
      first = false;
      std::memmove(s.storage_.identity_bytes.data + offset, s.storage_.identity_bytes.data + r.offset, r.bytes);
      r.offset = offset;
      offset += r.bytes;
    }
    if (w.add_identity_) {
      std::size_t slot = 0;
      while (s.storage_.identities[slot].used)
        ++slot;
      IdentityWriter writer(s.storage_.identity_bytes.data + offset, nullptr);
      encodeIdentity(writer, w.message_);
      s.storage_.identities[slot] = {true, w.message_.source_sequence, offset, w.identity_size_};
      offset += w.identity_size_;
    }
    s.identity_used_ = offset;
  }
};

inline Code Engine::apply(State & s, const EventPlan & plan) noexcept
{
  auto * w = plan.workspace_;
  if (!w || plan.owner_ != &s || !w->valid_ || plan.generation_ != w->generation_ || plan.revision_ != s.revision_ ||
      s.revision_ == std::numeric_limits<std::uint64_t>::max())
    return Code::kStalePlan;
  // All capacity, arithmetic and protocol decisions have been validated before
  // this point. Only noexcept POD writes/memmove follow; no partial-error path.
  if (w->clear_) {
    for (std::size_t i = 0; i < s.limits_.max_points; ++i)
      s.storage_.points[i] = {};
    for (std::size_t i = 0; i < s.limits_.max_windows; ++i)
      s.storage_.windows[i] = {};
  }
  for (std::size_t i = 0; i < w->action_count_; ++i) {
    const auto & a = w->actions_[i];
    if (a.kind == ActionKind::kPoint)
      s.storage_.points[a.index] = a.point;
    else
      s.storage_.windows[a.index] = a.window;
  }
  compactIdentities(s, *w);
  std::size_t imu_bytes = 0, count = 0;
  for (std::size_t i = w->imu_drop_; i < s.imu_count_; ++i) {
    auto r = s.storage_.imu[i];
    std::memmove(s.storage_.imu_bytes.data + imu_bytes, s.storage_.imu_bytes.data + r.offset, r.bytes);
    r.offset = imu_bytes;
    imu_bytes += r.bytes;
    s.storage_.imu[count++] = r;
  }
  if (w->append_imu_) {
    const auto & imu = w->imu_;
    std::memcpy(s.storage_.imu_bytes.data + imu_bytes, imu.content.data, imu.content.size);
    s.storage_.imu[count++] = {imu.sequence, imu.absolute_s, imu_bytes, imu.content.size};
    imu_bytes += imu.content.size;
  }
  s.imu_count_ = count;
  s.imu_used_ = imu_bytes;
  if (w->reserve_) {
    std::size_t points = 0;
    for (std::size_t i = 0; i < s.limits_.max_points; ++i) {
      const auto & p = s.storage_.points[i];
      if (p.used && p.window_index == w->next_.token.window_index)
        w->indices_[points++] = i;
    }
    std::sort(w->indices_.data, w->indices_.data + points, [&](std::size_t a, std::size_t b) {
      const auto & x = s.storage_.points[a].point;
      const auto & y = s.storage_.points[b].point;
      if (x.absolute_s != y.absolute_s)
        return x.absolute_s < y.absolute_s;
      if (x.message_sequence != y.message_sequence)
        return x.message_sequence < y.message_sequence;
      return x.raw_index < y.raw_index;
    });
    for (std::size_t i = 0; i < points; ++i)
      s.storage_.token_points[i] = s.storage_.points[w->indices_[i]].point;
    for (std::size_t i = 0; i < w->next_.token.imu_count; ++i)
      s.storage_.token_imu[i] = s.storage_.imu[i];
  }
  s.scalar_ = w->next_;
  ++s.revision_;
  w->valid_ = false;
  return Code::kOk;
}
}  // namespace detail

inline PlanResult
planEvent(StateView state, const Event & event, EventContext context, PlanWorkspace & workspace) noexcept
{
  return detail::Engine(state.state, workspace).plan(event, context);
}
inline Code applyPlan(State & state, const EventPlan & plan) noexcept
{
  return detail::Engine::apply(state, plan);
}

// Dedicated fixed-storage failure path: future callers must not retry the same
// occurrence after PlanResourceFailure. No action workspace is needed to latch.
inline void latchPlanResourceFailure(State & state) noexcept
{
  if (!state.initialized_ || state.scalar_.fault != Fault::kNone || state.scalar_.aborted)
    return;
  auto & scalar = state.scalar_;
  scalar.fault = Fault::kPlanResource;
  scalar.in_flight = false;
  auto & count = scalar.diagnostics.points[static_cast<std::size_t>(Reason::kFault)];
  for (std::size_t i = 0; i < state.limits_.max_points; ++i) {
    if (state.storage_.points[i].used) {
      if (count == std::numeric_limits<std::uint64_t>::max())
        scalar.diagnostics.arithmetic_overflow = true;
      else
        ++count;
    }
    state.storage_.points[i] = {};
  }
  for (std::size_t i = 0; i < state.limits_.max_windows; ++i)
    state.storage_.windows[i] = {};
  if (state.revision_ != std::numeric_limits<std::uint64_t>::max())
    ++state.revision_;
}
}  // namespace fast_lio::time_admission

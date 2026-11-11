#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <vector>

#include "time_admission_policy.hpp"

// Observe allocations only around policy calls, never around gtest reporting.
namespace
{
thread_local bool observe_allocations = false;
thread_local std::size_t allocations = 0;
}  // namespace
void * operator new(std::size_t size)
{
  if (observe_allocations)
    ++allocations;
  if (void * result = std::malloc(size ? size : 1))
    return result;
  throw std::bad_alloc();
}
void * operator new[](std::size_t size)
{
  return ::operator new(size);
}
void operator delete(void * ptr) noexcept
{
  std::free(ptr);
}
void operator delete[](void * ptr) noexcept
{
  std::free(ptr);
}
void operator delete(void * ptr, std::size_t) noexcept
{
  std::free(ptr);
}
void operator delete[](void * ptr, std::size_t) noexcept
{
  std::free(ptr);
}
void * operator new(std::size_t size, std::align_val_t alignment)
{
  if (observe_allocations)
    ++allocations;
  void * result = nullptr;
  if (posix_memalign(&result, static_cast<std::size_t>(alignment), size ? size : 1) != 0)
    throw std::bad_alloc();
  return result;
}
void * operator new[](std::size_t size, std::align_val_t alignment)
{
  return ::operator new(size, alignment);
}
void operator delete(void * ptr, std::align_val_t) noexcept
{
  std::free(ptr);
}
void operator delete[](void * ptr, std::align_val_t) noexcept
{
  std::free(ptr);
}
void operator delete(void * ptr, std::size_t, std::align_val_t) noexcept
{
  std::free(ptr);
}
void operator delete[](void * ptr, std::size_t, std::align_val_t) noexcept
{
  std::free(ptr);
}

namespace
{
namespace ta = fast_lio::time_admission;
using fast_lio::ImuDisposition;
using fast_lio::ImuProcessReason;

template<class T, std::size_t N>
ta::Span<T> span(std::array<T, N> & values)
{
  return {values.data(), values.size()};
}
template<class T>
ta::Span<const T> span(const std::vector<T> & values)
{
  return {values.data(), values.size()};
}

ta::Limits fixtureLimits()
{
  ta::Limits limits;
  limits.width_ns = 100000000;
  limits.lateness_ns = 150000000;
  limits.wait_ns = 100000000;
  limits.imu_wait_ns = 200000000;
  limits.max_original_span_s = 0.2;
  limits.max_points = 8;
  limits.max_messages = 4;
  limits.max_windows = 4;
  limits.max_identity_records = 8;
  limits.max_identity_bytes = 8192;
  limits.max_imu_samples = 8;
  limits.max_imu_bytes = 128;
  limits.max_ingress_bytes = 4096;
  limits.max_owned_bytes = 1024 * 1024;
  return limits;
}
struct Buffers
{
  std::array<ta::PointRecord, 8> points;
  std::array<ta::WindowRecord, 4> windows;
  std::array<ta::IdentityRecord, 8> identities;
  std::array<std::uint8_t, 8192> identity_bytes;
  std::array<ta::ImuRecord, 8> imu;
  std::array<std::uint8_t, 128> imu_bytes;
  std::array<ta::PointRef, 8> token_points;
  std::array<ta::ImuRecord, 8> token_imu;
  std::array<ta::Action, 128> actions;
  std::array<ta::PointRef, 8> ingress;
  std::array<std::size_t, 16> indices;
  ta::Storage storage()
  {
    return {span(points),
            span(windows),
            span(identities),
            span(identity_bytes),
            span(imu),
            span(imu_bytes),
            span(token_points),
            span(token_imu)};
  }
};
struct Message
{
  std::vector<ta::PointRef> points;
  std::vector<std::array<std::uint8_t, 4>> contents;
  std::vector<ta::ImmutablePointContent> payload;
  std::array<std::uint8_t, 3> authority{1, 2, 3};
  ta::MessageView facts;
  Message(std::uint64_t sequence, std::initializer_list<double> times, double canonical = 10.0)
  {
    std::size_t index = 0;
    for (const auto time : times)
      points.push_back({7, sequence, index, time, index++});
    contents.resize(points.size());
    payload.resize(points.size());
    for (std::size_t i = 0; i < contents.size(); ++i)
      contents[i] = {static_cast<std::uint8_t>(i), 7, 9, 11};
    facts.source_epoch = 7;
    facts.source_sequence = sequence;
    facts.canonical_begin_s = canonical;
    facts.original_status = fast_lio::ScanTimeStatus::kValid;
    facts.original_duration_s = 0.2;
    facts.original_effective_count = facts.declared_count = facts.decoded_count = facts.effective_count = points.size();
    facts.declared_known = facts.decoded_known = facts.effective_known = true;
    facts.dependency = {true, 0.0};
  }
  ta::MessageView view()
  {
    for (std::size_t i = 0; i < payload.size(); ++i)
      payload[i].bytes = {contents[i].data(), contents[i].size()};
    auto result = facts;
    result.points = span(points);
    result.payload = span(payload);
    result.message_content = {authority.data(), authority.size()};
    return result;
  }
};
struct Harness
{
  Buffers buffers;
  ta::State state;
  ta::PlanWorkspace workspace{span(buffers.actions), span(buffers.ingress), span(buffers.indices)};
  std::uint64_t sequence = 0;
  std::array<std::uint8_t, 4> imu_content{4, 5, 6, 7};
  explicit Harness(ta::Limits limits = fixtureLimits())
  {
    EXPECT_EQ(ta::initialize(state, 7, limits, buffers.storage(), workspace), ta::Code::kOk);
  }
  ta::Code step(ta::Event event, std::int64_t now = 0)
  {
    const auto plan = ta::planEvent({state}, event, {++sequence, now}, workspace);
    return plan.code == ta::Code::kOk ? ta::applyPlan(state, plan.plan) : plan.code;
  }
  void lidar(Message & m, std::int64_t now = 0)
  {
    ta::Event event;
    event.kind = ta::EventKind::kLidar;
    event.message = m.view();
    ASSERT_EQ(step(event, now), ta::Code::kOk);
  }
  void imu(double time, std::uint64_t id = 1, std::int64_t now = 0)
  {
    ta::Event event;
    event.kind = ta::EventKind::kImu;
    event.imu = {7, id, time, {imu_content.data(), imu_content.size()}};
    ASSERT_EQ(step(event, now), ta::Code::kOk);
  }
  void control(ta::EventKind kind, std::int64_t now = 0)
  {
    ta::Event event;
    event.kind = kind;
    ASSERT_EQ(step(event, now), ta::Code::kOk);
  }
  ta::ConsumptionAck ack(std::optional<double> fc,
                         ImuDisposition disposition = ImuDisposition::kCommitted,
                         ImuProcessReason reason = ImuProcessReason::kProcessed,
                         bool cloud = true)
  {
    return {state.scalar().token, state.tokenPoints(), state.tokenImu(), fc, disposition, reason, cloud};
  }
  void settle(ta::ConsumptionAck ack, std::int64_t now = 100000000)
  {
    ta::Event event;
    event.kind = ta::EventKind::kAck;
    event.ack = ack;
    ASSERT_EQ(step(event, now), ta::Code::kOk);
  }
  std::uint64_t points(ta::Reason reason) const
  {
    return state.scalar().diagnostics.points[static_cast<std::size_t>(reason)];
  }
  std::size_t pending() const
  {
    std::size_t result = 0;
    for (const auto & p : buffers.points)
      result += p.used;
    return result;
  }
  std::size_t identities() const
  {
    std::size_t result = 0;
    for (const auto & r : buffers.identities)
      result += r.used;
    return result;
  }
};

TEST(TimeAdmission, CanonicalEpochCounterexampleDoesNotUseAbsoluteTime)
{
  Harness h;
  Message a(1, {2.1000000004}, 2.1), b(2, {1.1000000003}, 1.1);
  h.lidar(a);
  h.lidar(b);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kNone);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.max_canonical_begin, 2.1);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 2.1000000004);
  EXPECT_GT(2.1000000004 - 1.1000000003, 1.0);
  Message c(3, {2.2}, std::nextafter(1.1, 0.0));
  h.lidar(c);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kEpoch);
  EXPECT_EQ(h.pending(), 0U);
}

TEST(TimeAdmission, RealEpochCanonicalEqualityAndRollbackNeighbor)
{
  const double epoch = 1788617881.0;
  Harness equal, rollback;
  Message a(1, {epoch + 0.05}, epoch), b(2, {epoch + 0.05}, epoch - 1.0);
  equal.lidar(a);
  equal.lidar(b);
  EXPECT_EQ(equal.state.scalar().fault, ta::Fault::kNone);
  Message c(2, {epoch + 0.06}, std::nextafter(epoch - 1.0, 0.0));
  rollback.lidar(a);
  rollback.lidar(c);
  EXPECT_EQ(rollback.state.scalar().fault, ta::Fault::kEpoch);
}

TEST(TimeAdmission, OriginalSpanCheckedStrictlyBeforeHistoryAndRejectedSpanAdvancesCanonicalOnly)
{
  Harness equal, below, above, invalid;
  Message a(1, {0.01});
  a.facts.original_duration_s = 0.2;
  equal.lidar(a);
  EXPECT_EQ(equal.pending(), 1U);
  a.facts.original_duration_s = std::nextafter(0.2, 0.0);
  below.lidar(a);
  EXPECT_EQ(below.pending(), 1U);
  a.facts.original_duration_s = std::nextafter(0.2, 1.0);
  above.lidar(a);
  EXPECT_EQ(above.points(ta::Reason::kInvalidSpan), 1U);
  EXPECT_FALSE(above.state.scalar().frontier.H);
  EXPECT_DOUBLE_EQ(*above.state.scalar().frontier.max_canonical_begin, 10.0);
  Message newer(2, {0.01}, 11.1);
  newer.facts.original_status = fast_lio::ScanTimeStatus::kInvalidOffset;
  above.lidar(newer);
  EXPECT_DOUBLE_EQ(*above.state.scalar().frontier.max_canonical_begin, 11.1);
  EXPECT_FALSE(above.state.scalar().frontier.H);
  Message old(3, {0.01}, 10.0);
  above.lidar(old);
  EXPECT_EQ(above.state.scalar().fault, ta::Fault::kEpoch);
  invalid.imu(1.0);
  Message processed(4, {0.05, 0.06});
  invalid.lidar(processed);
  invalid.control(ta::EventKind::kTick, 100000000);
  invalid.settle(invalid.ack(0.06));
  Message bad(5, {0.001});
  bad.facts.original_status = fast_lio::ScanTimeStatus::kInvalidOffset;
  invalid.lidar(bad, 100000000);
  EXPECT_EQ(invalid.points(ta::Reason::kInvalidSpan), 1U);
  EXPECT_EQ(invalid.points(ta::Reason::kHistory), 0U);
}

TEST(TimeAdmission, DeadlineCapacityOneClosesButCannotConsumeBeforeAtomicIngress)
{
  auto limits = fixtureLimits();
  limits.max_points = 1;
  limits.wait_ns = 10000000;
  Harness h(limits);
  h.imu(0.3);
  Message a(1, {0.01}), b(2, {0.11});
  h.lidar(a);
  h.lidar(b, 10000000);
  EXPECT_EQ(h.points(ta::Reason::kCapacityNew), 1U);
  ASSERT_TRUE(h.state.scalar().in_flight);
  EXPECT_EQ(h.state.tokenPoints()[0].message_sequence, 1U);
  EXPECT_DOUBLE_EQ(h.state.scalar().token.begin, 0.01);
  EXPECT_DOUBLE_EQ(h.state.scalar().token.end, 0.01);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.C, 0.1);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 0.01);
}

TEST(TimeAdmission, RejectedOccurrenceCannotBeProcessedWhenCapacityLaterFrees)
{
  auto limits = fixtureLimits();
  limits.max_points = 1;
  limits.wait_ns = 10000000;
  Harness h(limits);
  h.imu(0.3);
  Message a(1, {0.01}), b(2, {0.11});
  h.lidar(a);
  h.lidar(b, 10000000);
  h.settle(h.ack(0.01), 10000000);
  h.lidar(b, 10000000);
  EXPECT_EQ(h.points(ta::Reason::kCapacityNew), 1U);
  EXPECT_EQ(h.points(ta::Reason::kDuplicate), 1U);
  EXPECT_EQ(h.state.scalar().diagnostics.reserved_batches, 1U);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 0.01);
}

TEST(TimeAdmission, WholeMessageAdmittedBeforeItsOwnWatermarkClosesItsFrontHalf)
{
  Harness h;
  Message a(1, {0.01, 0.21, 0.11});
  h.lidar(a);
  EXPECT_EQ(h.pending(), 3U);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 0U);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 0.21);
  EXPECT_FALSE(h.state.scalar().diagnostics.watermark_certified);
  h.imu(1.0);
  h.control(ta::EventKind::kTick, 100000000);
  EXPECT_EQ(h.state.scalar().token.point_count, 1U);
  EXPECT_EQ(h.state.tokenPoints()[0].raw_index, 0U);
}

TEST(TimeAdmission, AtomicCapacityRejectPreservesIndependentHistoryAndClosedReasons)
{
  auto limits = fixtureLimits();
  limits.max_points = 2;
  Harness h(limits);
  h.imu(1.0);
  Message a(1, {0.05});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  h.settle(h.ack(0.05));
  Message held(2, {0.11});
  h.lidar(held, 100000000);
  Message mixed(3, {0.04, 0.05, 0.12, 0.13});
  h.lidar(mixed, 100000000);
  EXPECT_EQ(h.points(ta::Reason::kHistory), 1U);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 1U);
  EXPECT_EQ(h.points(ta::Reason::kCapacityNew), 2U);
  EXPECT_EQ(h.pending(), 1U);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 0.11);
}

TEST(TimeAdmission, DistinctUidsAtSameTimeAndDeterministicSortingPreserveBitsAndPayloadIndex)
{
  Harness h;
  Message a(20, {0.03, 0.01, 0.02}), b(10, {0.02});
  h.lidar(a);
  h.lidar(b);
  h.imu(0.3);
  h.control(ta::EventKind::kTick, 100000000);
  const auto p = h.state.tokenPoints();
  ASSERT_EQ(p.size, 4U);
  EXPECT_EQ(p[0].message_sequence, 20U);
  EXPECT_EQ(p[0].raw_index, 1U);
  EXPECT_EQ(p[0].payload_index, 1U);
  EXPECT_EQ(p[1].message_sequence, 10U);
  EXPECT_EQ(p[1].raw_index, 0U);
  EXPECT_EQ(p[2].message_sequence, 20U);
  EXPECT_EQ(p[2].raw_index, 2U);
  EXPECT_EQ(p[2].payload_index, 2U);
  EXPECT_EQ(p[3].raw_index, 0U);
  EXPECT_EQ(ta::timeBits(p[0].absolute_s), ta::timeBits(a.points[1].absolute_s));
  EXPECT_EQ(ta::timeBits(p[3].absolute_s), ta::timeBits(a.points[0].absolute_s));
  EXPECT_DOUBLE_EQ(h.state.scalar().token.begin, 0.01);
  EXPECT_DOUBLE_EQ(h.state.scalar().token.end, 0.03);
}

TEST(TimeAdmission, AuthorityBytesAreOwnedAndConflictIncludesPayloadWithIdenticalTimestamp)
{
  Harness h;
  Message a(1, {0.01});
  h.lidar(a);
  a.contents[0][3] = 99;
  h.lidar(a);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kIdentityConflict);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_EQ(h.identities(), 1U);
  EXPECT_FALSE(h.state.identityBytes().size == 0);
  Harness authority;
  Message b(1, {0.01});
  authority.lidar(b);
  b.authority[2] = 90;
  authority.lidar(b);
  EXPECT_EQ(authority.state.scalar().fault, ta::Fault::kIdentityConflict);
}

TEST(TimeAdmission, DuplicatePointUidWithinMessageChecksActualContent)
{
  Harness same, conflict;
  Message a(1, {0.01, 0.01});
  a.points[1].raw_index = 0;
  a.contents[1] = a.contents[0];
  same.lidar(a);
  EXPECT_EQ(same.pending(), 1U);
  EXPECT_EQ(same.points(ta::Reason::kDuplicate), 1U);
  a.contents[1][1] = 0;
  conflict.lidar(a);
  EXPECT_EQ(conflict.state.scalar().fault, ta::Fault::kIdentityConflict);
  EXPECT_EQ(conflict.pending(), 0U);
}

TEST(TimeAdmission, ActiveIdentitySurvivesConservativeExpiryIncludingPartialConsumption)
{
  auto limits = fixtureLimits();
  limits.max_identity_records = 1;
  Harness h(limits);
  h.imu(1.0);
  Message a(1, {0.01, 0.11}), rejected(50, {0.21}), unseen_old(40, {0.25});
  h.lidar(a);
  h.lidar(rejected);
  EXPECT_EQ(h.state.scalar().expiry_floor, 51U);
  EXPECT_EQ(h.pending(), 2U);
  h.lidar(a);
  EXPECT_EQ(h.points(ta::Reason::kDuplicate), 2U);
  h.lidar(unseen_old);
  EXPECT_EQ(h.points(ta::Reason::kUidExpired), 1U);
  h.control(ta::EventKind::kTick, 100000000);
  h.settle(h.ack(0.01));
  ASSERT_TRUE(h.state.scalar().in_flight);
  EXPECT_EQ(h.state.scalar().token.point_count, 1U);
  h.lidar(a, 100000000);
  EXPECT_EQ(h.points(ta::Reason::kDuplicate), 4U);
  h.settle(h.ack(0.11));
  h.lidar(a, 100000000);
  EXPECT_EQ(h.points(ta::Reason::kUidExpired), 3U);
  EXPECT_EQ(h.identities(), 0U);
}

TEST(TimeAdmission, CapacityTombstoneFallbackExpiresRejectedUidAndIdentityExhaustionFaults)
{
  auto limits = fixtureLimits();
  limits.max_identity_bytes = 1;
  Harness h(limits);
  Message a(100, {0.01});
  h.lidar(a);
  h.lidar(a);
  EXPECT_EQ(h.points(ta::Reason::kCapacityNew), 1U);
  EXPECT_EQ(h.points(ta::Reason::kUidExpired), 1U);
  EXPECT_EQ(h.state.scalar().expiry_floor, 101U);
  EXPECT_FALSE(h.state.scalar().frontier.H);
  Message last(std::numeric_limits<std::uint64_t>::max(), {0.02});
  h.lidar(last);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kIdentityExhausted);
}

TEST(TimeAdmission, MessagePointWindowAndIdentityCapsAreIndependentAcrossAllOwnershipPhases)
{
  auto message_limit = fixtureLimits();
  message_limit.max_messages = 1;
  Harness messages(message_limit);
  Message a(1, {0.01}), b(2, {0.02});
  messages.lidar(a);
  messages.lidar(b);
  EXPECT_EQ(messages.points(ta::Reason::kCapacityNew), 1U);
  auto point_limit = fixtureLimits();
  point_limit.max_points = 1;
  Harness ready(point_limit);
  ready.lidar(a);
  ready.control(ta::EventKind::kTick, 100000000);
  Message later(2, {0.11});
  ready.lidar(later, 100000000);
  EXPECT_EQ(ready.points(ta::Reason::kCapacityNew), 1U);
  ready.imu(1.0, 1, 100000000);
  ready.lidar(later, 100000000);
  EXPECT_TRUE(ready.state.scalar().in_flight);
  EXPECT_EQ(ready.points(ta::Reason::kDuplicate), 1U);
  Message c(3, {0.12});
  ready.lidar(c, 100000000);
  EXPECT_EQ(ready.points(ta::Reason::kCapacityNew), 2U);
  auto window_limit = fixtureLimits();
  window_limit.max_windows = 1;
  Harness windows(window_limit);
  Message split(1, {0.01, 0.11});
  windows.lidar(split);
  EXPECT_EQ(windows.pending(), 0U);
  EXPECT_EQ(windows.points(ta::Reason::kCapacityNew), 2U);
  EXPECT_FALSE(windows.state.scalar().frontier.H);
  auto identity_limit = fixtureLimits();
  identity_limit.max_identity_records = 1;
  Harness ids(identity_limit);
  ids.lidar(a);
  ids.lidar(b);
  EXPECT_EQ(ids.points(ta::Reason::kCapacityNew), 1U);
}

TEST(TimeAdmission, LowEpochOnePointFourBoundaryAndWatermarkUseExactDivision)
{
  const double edge = 1.4;
  const double below = std::nextafter(edge, 0.0);
  const double above = std::nextafter(edge, std::numeric_limits<double>::infinity());
  EXPECT_EQ(ta::boundarySeconds(1400000000), edge);
  ta::WindowBounds bounds;
  ASSERT_TRUE(ta::locateWindow(below, 100000000, bounds));
  EXPECT_EQ(bounds.index, 13);
  ASSERT_TRUE(ta::locateWindow(edge, 100000000, bounds));
  EXPECT_EQ(bounds.index, 14);
  ASSERT_TRUE(ta::locateWindow(above, 100000000, bounds));
  EXPECT_EQ(bounds.index, 14);
  auto limits = fixtureLimits();
  limits.lateness_ns = 100000000;
  Harness h(limits);
  Message message(1, {below, edge, above, 1.5});
  h.lidar(message);
  ASSERT_TRUE(h.state.scalar().frontier.C);
  EXPECT_EQ(*h.state.scalar().frontier.C, edge);
  h.imu(2.0);
  ASSERT_TRUE(h.state.scalar().in_flight);
  ASSERT_EQ(h.state.tokenPoints().size, 1u);
  EXPECT_EQ(h.state.tokenPoints()[0].absolute_s, below);
}

TEST(TimeAdmission, RegressedImuSequenceLatchesTheImuOrderFault)
{
  Harness h;
  h.imu(0.15, 2);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kNone);
  h.imu(0.16, 1);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kImuOrder);
  EXPECT_FALSE(h.state.scalar().in_flight);
}

TEST(TimeAdmission, RealEpochHalfOpenClosedAndHistoryNeighborsHaveLiteralMembership)
{
  const double edge = 1788617881.1;
  ta::WindowBounds b;
  ASSERT_TRUE(ta::locateWindow(edge, 100000000, b));
  EXPECT_EQ(b.index, 17886178811LL);
  ASSERT_TRUE(ta::locateWindow(std::nextafter(edge, 0.0), 100000000, b));
  EXPECT_EQ(b.index, 17886178810LL);
  ASSERT_TRUE(ta::locateWindow(std::nextafter(edge, std::numeric_limits<double>::infinity()), 100000000, b));
  EXPECT_EQ(b.index, 17886178811LL);
  Harness h;
  Message a(1, {edge - 0.05}, 1788617881.0);
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  Message near(2, {std::nextafter(edge, 0.0), edge, std::nextafter(edge, 1e20)}, 1788617881.0);
  h.lidar(near, 100000000);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 1U);
  EXPECT_EQ(h.pending(), 3U);
  h.imu(edge + 1.0, 1, 100000000);
  h.settle(h.ack(edge), 100000000);
  Message history(3, {std::nextafter(edge, 0.0), edge}, 1788617881.0);
  h.lidar(history, 100000000);
  EXPECT_EQ(h.points(ta::Reason::kHistory), 1U);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 1U);
  EXPECT_EQ(h.pending(), 3U);
}

TEST(TimeAdmission, ExactFcOnlyBatchIsNonadvancingButFcEqualityWithLaterPointIsUsable)
{
  Harness h;
  h.imu(2.0);
  Message a(1, {0.01});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  h.settle(h.ack(0.15));
  Message equal(2, {0.15});
  h.lidar(equal, 100000000);
  h.control(ta::EventKind::kTick, 200000000);
  EXPECT_EQ(h.points(ta::Reason::kNonadvancing), 1U);
  EXPECT_FALSE(h.state.scalar().in_flight);
  Harness mixed;
  mixed.imu(2.0);
  mixed.lidar(a);
  mixed.control(ta::EventKind::kTick, 100000000);
  mixed.settle(mixed.ack(0.15));
  Message both(2, {0.15, 0.16});
  mixed.lidar(both, 100000000);
  mixed.control(ta::EventKind::kTick, 200000000);
  EXPECT_EQ(mixed.state.scalar().token.point_count, 2U);
  EXPECT_DOUBLE_EQ(mixed.state.scalar().token.begin, 0.15);
  EXPECT_DOUBLE_EQ(mixed.state.scalar().token.end, 0.16);
}

TEST(TimeAdmission, BootstrapInvalidInitializationAndCoveragePreserveActualAckCombinations)
{
  Harness h;
  h.imu(2.0);
  Message a(1, {0.01}), b(2, {0.11}), c(3, {0.21});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  h.settle(h.ack(std::nullopt, ImuDisposition::kCommitted, ImuProcessReason::kBootstrap, false));
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
  EXPECT_FALSE(h.state.scalar().diagnostics.last_ack.producedOutput);
  h.lidar(b, 100000000);
  h.control(ta::EventKind::kTick, 200000000);
  h.settle(h.ack(0.11, ImuDisposition::kInvalid, ImuProcessReason::kInitializationInvalidMean, false), 200000000);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.Fc, 0.11);
  EXPECT_EQ(h.state.scalar().diagnostics.last_ack.disposition, ImuDisposition::kInvalid);
  h.lidar(c, 200000000);
  h.control(ta::EventKind::kTick, 300000000);
  h.settle(h.ack(0.21, ImuDisposition::kCommitted, ImuProcessReason::kCoverageGap, false), 300000000);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.Fc, 0.21);
  EXPECT_EQ(h.state.scalar().diagnostics.last_ack.reason, ImuProcessReason::kCoverageGap);
  EXPECT_EQ(h.state.scalar().diagnostics.settled_batches, 3U);
  EXPECT_FALSE(h.state.scalar().diagnostics.last_ack.producedOutput);
}

TEST(TimeAdmission, DuplicateOrWrongTokenAckAndRegressedFcLatchFault)
{
  Harness duplicate;
  duplicate.imu(1.0);
  Message a(1, {0.01});
  duplicate.lidar(a);
  duplicate.control(ta::EventKind::kTick, 100000000);
  auto ack = duplicate.ack(0.01);
  duplicate.settle(ack);
  duplicate.settle(ack);
  EXPECT_EQ(duplicate.state.scalar().fault, ta::Fault::kAck);
  EXPECT_EQ(duplicate.state.scalar().diagnostics.settled_batches, 1U);
  EXPECT_DOUBLE_EQ(*duplicate.state.scalar().frontier.Fc, 0.01);
  Harness wrong;
  wrong.imu(1.0);
  wrong.lidar(a);
  wrong.control(ta::EventKind::kTick, 100000000);
  auto bad = wrong.ack(0.01);
  ++bad.token.epoch;
  wrong.settle(bad);
  EXPECT_EQ(wrong.state.scalar().fault, ta::Fault::kAck);
  EXPECT_EQ(wrong.points(ta::Reason::kFault), 1U);
  Harness regression;
  regression.imu(1.0);
  regression.lidar(a);
  regression.control(ta::EventKind::kTick, 100000000);
  regression.settle(regression.ack(0.01));
  Message b(2, {0.11});
  regression.lidar(b, 100000000);
  regression.control(ta::EventKind::kTick, 200000000);
  regression.settle(regression.ack(0.0), 200000000);
  EXPECT_EQ(regression.state.scalar().fault, ta::Fault::kAck);
  EXPECT_DOUBLE_EQ(*regression.state.scalar().frontier.Fc, 0.01);
}

TEST(TimeAdmission, AckChecksExactPointSetAndImuPrefixRatherThanDigest)
{
  Harness points;
  points.imu(1.0);
  Message a(1, {0.01, 0.02});
  points.lidar(a);
  points.control(ta::EventKind::kTick, 100000000);
  auto ack = points.ack(0.02);
  std::array<ta::PointRef, 2> changed{ack.points[0], ack.points[1]};
  changed[1].raw_index = 0;
  ack.points = {changed.data(), changed.size()};
  points.settle(ack);
  EXPECT_EQ(points.state.scalar().fault, ta::Fault::kAck);
  Harness imu;
  imu.imu(1.0);
  imu.lidar(a);
  imu.control(ta::EventKind::kTick, 100000000);
  auto other = imu.ack(0.02);
  auto altered = other.imu[0];
  ++altered.sequence;
  other.imu = {&altered, 1};
  imu.settle(other);
  EXPECT_EQ(imu.state.scalar().fault, ta::Fault::kAck);
}

TEST(TimeAdmission, PartialPropagationRetainsActualFcAndFaultPreventsAnotherReservation)
{
  Harness h;
  h.imu(1.0);
  Message a(1, {0.01, 0.11});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  h.settle(h.ack(0.01, ImuDisposition::kFatal, ImuProcessReason::kPartialPropagation, false));
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kPartialPropagation);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_FALSE(h.state.scalar().in_flight);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.Fc, 0.01);
  EXPECT_EQ(h.points(ta::Reason::kFault), 2U);
  const auto revision = h.state.revision();
  ta::Event tick;
  EXPECT_EQ(h.step(tick, 200000000), ta::Code::kAlreadyStopped);
  EXPECT_EQ(h.state.revision(), revision);
  EXPECT_EQ(h.state.scalar().diagnostics.reserved_batches, 1U);
  EXPECT_EQ(h.identities(), 1U);
}

TEST(TimeAdmission, OldestMissingTailPreventsPassingItAndNoTailDoesNotCommitFc)
{
  Harness h;
  Message a(1, {0.01, 0.11});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  EXPECT_FALSE(h.state.scalar().in_flight);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
  h.imu(0.005, 1, 100000000);
  EXPECT_FALSE(h.state.scalar().in_flight);
  h.imu(0.01, 2, 100000000);
  ASSERT_TRUE(h.state.scalar().in_flight);
  EXPECT_DOUBLE_EQ(h.state.scalar().token.end, 0.01);
  h.settle(h.ack(0.01));
  EXPECT_FALSE(h.state.scalar().in_flight);
  EXPECT_EQ(h.pending(), 1U);
  h.imu(0.2, 3, 100000000);
  EXPECT_DOUBLE_EQ(h.state.scalar().token.end, 0.11);
}

TEST(TimeAdmission, SameTimeImuCannotRescueTailTimeoutAndEofDoesNotRewriteReason)
{
  Harness timeout;
  Message a(1, {0.01});
  timeout.lidar(a);
  timeout.control(ta::EventKind::kTick, 100000000);
  timeout.imu(0.5, 1, 300000000);
  EXPECT_EQ(timeout.points(ta::Reason::kImuTimeout), 1U);
  EXPECT_EQ(timeout.pending(), 0U);
  EXPECT_FALSE(timeout.state.scalar().in_flight);
  timeout.control(ta::EventKind::kAllEof, 300000000);
  EXPECT_TRUE(timeout.state.eofComplete());
  EXPECT_EQ(timeout.points(ta::Reason::kEofImuUncovered), 0U);
  Harness early;
  early.lidar(a);
  early.control(ta::EventKind::kLidarEof);
  early.imu(0.5, 1, 199999999);
  EXPECT_TRUE(early.state.scalar().in_flight);
  EXPECT_EQ(early.points(ta::Reason::kImuTimeout), 0U);
  Harness exact;
  exact.lidar(a);
  exact.control(ta::EventKind::kLidarEof);
  exact.control(ta::EventKind::kAllEof, 200000000);
  EXPECT_EQ(exact.points(ta::Reason::kImuTimeout), 1U);
  EXPECT_EQ(exact.points(ta::Reason::kEofImuUncovered), 0U);
}

TEST(TimeAdmission, LidarEofAllowsImuAllEofTerminatesUncoveredAndAbortIsDistinct)
{
  Harness h;
  Message a(1, {0.01});
  h.lidar(a);
  h.control(ta::EventKind::kLidarEof);
  EXPECT_FALSE(h.state.eofComplete());
  h.imu(0.1);
  ASSERT_TRUE(h.state.scalar().in_flight);
  h.control(ta::EventKind::kAllEof);
  EXPECT_FALSE(h.state.eofComplete());
  h.settle(h.ack(0.01), 0);
  EXPECT_TRUE(h.state.eofComplete());
  Harness missing;
  missing.lidar(a);
  missing.control(ta::EventKind::kAllEof);
  EXPECT_EQ(missing.points(ta::Reason::kEofImuUncovered), 1U);
  EXPECT_TRUE(missing.state.eofComplete());
  EXPECT_FALSE(missing.state.scalar().frontier.Fc);
  Harness empty;
  empty.control(ta::EventKind::kAllEof);
  EXPECT_TRUE(empty.state.eofComplete());
  Harness abort;
  abort.lidar(a);
  abort.control(ta::EventKind::kAbort);
  EXPECT_TRUE(abort.state.scalar().aborted);
  EXPECT_FALSE(abort.state.eofComplete());
  EXPECT_FALSE(abort.state.scalar().all_eof);
  EXPECT_EQ(abort.points(ta::Reason::kAbort), 1U);
}

TEST(TimeAdmission, InputAndRepeatedControlsAfterEofAreProtocolFaults)
{
  Message a(1, {0.01});
  Harness lidar;
  lidar.control(ta::EventKind::kLidarEof);
  lidar.lidar(a);
  EXPECT_EQ(lidar.state.scalar().fault, ta::Fault::kAfterEof);
  Harness imu;
  imu.control(ta::EventKind::kAllEof);
  imu.imu(1.0);
  EXPECT_EQ(imu.state.scalar().fault, ta::Fault::kAfterEof);
  Harness duplicate;
  duplicate.control(ta::EventKind::kLidarEof);
  duplicate.control(ta::EventKind::kLidarEof);
  EXPECT_EQ(duplicate.state.scalar().fault, ta::Fault::kAfterEof);
  Harness all;
  all.control(ta::EventKind::kAllEof);
  all.control(ta::EventKind::kAllEof);
  EXPECT_EQ(all.state.scalar().fault, ta::Fault::kAfterEof);
}

TEST(TimeAdmission, ImuCapTerminatesEntireEarlierPrefixThroughLastAffectedWindowWithoutProcess)
{
  auto limits = fixtureLimits();
  limits.max_imu_samples = 2;
  Harness h(limits);
  h.imu(0.0, 1);
  h.imu(0.05, 2);
  Message early(1, {0.15}), late(2, {0.25});
  early.facts.dependency = {true, 0.1};
  late.facts.dependency = {true, 0.0};
  h.lidar(early);
  h.lidar(late);
  h.imu(0.1, 3);
  EXPECT_EQ(h.points(ta::Reason::kImuCapacityLoss), 2U);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.C, 0.3);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
  EXPECT_EQ(h.state.scalar().diagnostics.reserved_batches, 0U);
  EXPECT_EQ(h.state.scalar().diagnostics.imu_capacity_samples, 1U);
  ASSERT_EQ(h.state.imu().size, 2U);
  EXPECT_EQ(h.state.imu()[0].sequence, 2U);
  EXPECT_EQ(h.state.imu()[1].sequence, 3U);
  Message closed(3, {0.2});
  h.lidar(closed);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 1U);
}

TEST(TimeAdmission, ImuBytesCapExactBoundaryAndUnknownDependenciesAreConservative)
{
  auto limits = fixtureLimits();
  limits.max_imu_bytes = 8;
  Harness h(limits);
  h.imu(0.0, 1);
  h.imu(0.01, 2);
  EXPECT_EQ(h.state.scalar().diagnostics.imu_capacity_samples, 0U);
  Message a(1, {0.11});
  a.facts.dependency = {false, 0.0};
  h.lidar(a);
  h.imu(0.02, 3);
  EXPECT_EQ(h.state.scalar().diagnostics.imu_capacity_samples, 1U);
  EXPECT_TRUE(h.state.scalar().diagnostics.dependency_unknown);
  EXPECT_EQ(h.points(ta::Reason::kImuCapacityLoss), 1U);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
}

TEST(TimeAdmission, InFlightImuCannotBeTrimmedAndOverflowFaultRetainsOldPrefix)
{
  auto limits = fixtureLimits();
  limits.max_imu_samples = 1;
  Harness h(limits);
  h.imu(1.0);
  Message a(1, {0.01});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  h.imu(1.1, 2, 100000000);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kImuInFlightCapacity);
  EXPECT_EQ(h.state.imu().size, 1U);
  EXPECT_EQ(h.state.imu()[0].sequence, 1U);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_EQ(h.state.scalar().diagnostics.imu_capacity_samples, 0U);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
}

TEST(TimeAdmission, TailCoverageIsNotInternalGapOrInitializationDependencyCertification)
{
  Harness h;
  h.imu(0.0, 1);
  h.imu(10.0, 2);
  Message a(1, {0.01});
  a.facts.dependency = {false, 0.0};
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  EXPECT_TRUE(h.state.scalar().in_flight);
  EXPECT_FALSE(h.state.scalar().token.dependency_known);
  EXPECT_TRUE(h.state.scalar().diagnostics.dependency_unknown);
  h.settle(h.ack(0.01, ImuDisposition::kCommitted, ImuProcessReason::kCoverageGap, false));
  EXPECT_FALSE(h.state.scalar().diagnostics.last_ack.producedOutput);
  EXPECT_EQ(h.state.scalar().diagnostics.last_ack.reason, ImuProcessReason::kCoverageGap);
}

TEST(TimeAdmission, InvalidCountsIndicesAndAuthoritySpansRejectBeforeAccess)
{
  Harness count, index, missing, point_epoch, null_points, nan;
  Message a(1, {0.01});
  auto event = ta::Event{};
  event.kind = ta::EventKind::kLidar;
  event.message = a.view();
  event.message.effective_count = 2;
  ASSERT_EQ(count.step(event), ta::Code::kOk);
  EXPECT_EQ(count.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].payload_index = 1;
  index.lidar(a);
  EXPECT_EQ(index.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].payload_index = 0;
  event.message = a.view();
  event.message.message_content = {nullptr, 3};
  ASSERT_EQ(missing.step(event), ta::Code::kOk);
  EXPECT_EQ(missing.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].source_epoch = 8;
  point_epoch.lidar(a);
  EXPECT_EQ(point_epoch.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].source_epoch = 7;
  event.message = a.view();
  event.message.points.data = nullptr;
  ASSERT_EQ(null_points.step(event), ta::Code::kOk);
  EXPECT_EQ(null_points.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].absolute_s = std::numeric_limits<double>::quiet_NaN();
  nan.lidar(a);
  EXPECT_EQ(nan.points(ta::Reason::kInvalidMessage), 1U);
  EXPECT_FALSE(nan.state.scalar().frontier.H);
}

TEST(TimeAdmission, MalformedPayloadCountUnknownFactsAndRawIndexAreNotCompletenessProof)
{
  Harness unknown, payload, raw, null_bytes;
  Message a(1, {0.01});
  a.facts.declared_known = false;
  unknown.lidar(a);
  a.facts.declared_known = true;
  EXPECT_EQ(unknown.points(ta::Reason::kInvalidMessage), 1U);
  ta::Event event;
  event.kind = ta::EventKind::kLidar;
  event.message = a.view();
  event.message.payload.size = 0;
  ASSERT_EQ(payload.step(event), ta::Code::kOk);
  EXPECT_EQ(payload.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].raw_index = 1;
  raw.lidar(a);
  EXPECT_EQ(raw.points(ta::Reason::kInvalidMessage), 1U);
  a.points[0].raw_index = 0;
  event.message = a.view();
  a.payload[0].bytes = {nullptr, 4};
  ASSERT_EQ(null_bytes.step(event), ta::Code::kOk);
  EXPECT_EQ(null_bytes.points(ta::Reason::kInvalidMessage), 1U);
}

TEST(TimeAdmission, OversizedIngressUsesConstantSizeRejectionAndCannotReplay)
{
  Harness h;
  Message a(100, {0.01});
  ta::Event event;
  event.kind = ta::EventKind::kLidar;
  event.message = a.view();
  event.message.points = {nullptr, std::numeric_limits<std::size_t>::max()};
  const auto before = h.state.revision();
  EXPECT_EQ(h.step(event), ta::Code::kOk);
  EXPECT_EQ(h.state.revision(), before + 1);
  EXPECT_EQ(h.state.scalar().expiry_floor, 101U);
  EXPECT_EQ(h.pending(), 0U);
  h.lidar(a);
  EXPECT_EQ(h.points(ta::Reason::kUidExpired), 1U);
  EXPECT_FALSE(h.state.scalar().frontier.H);
  auto limits = fixtureLimits();
  limits.max_ingress_bytes = 1;
  Harness bytes(limits);
  bytes.lidar(a);
  EXPECT_EQ(bytes.points(ta::Reason::kCapacityNew), 1U);
  EXPECT_EQ(bytes.state.scalar().expiry_floor, 101U);
}

TEST(TimeAdmission, EventSequenceClockAndDeadlineOverflowHaveFixedFaults)
{
  Harness h;
  ta::Event tick;
  auto plan = ta::planEvent({h.state}, tick, {8, 100}, h.workspace);
  ASSERT_EQ(plan.code, ta::Code::kOk);
  ASSERT_EQ(ta::applyPlan(h.state, plan.plan), ta::Code::kOk);
  plan = ta::planEvent({h.state}, tick, {7, 100}, h.workspace);
  ASSERT_EQ(plan.code, ta::Code::kOk);
  ASSERT_EQ(ta::applyPlan(h.state, plan.plan), ta::Code::kOk);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kEventOrder);
  Harness clock;
  EXPECT_EQ(clock.step(tick, 100), ta::Code::kOk);
  EXPECT_EQ(clock.step(tick, 99), ta::Code::kOk);
  EXPECT_EQ(clock.state.scalar().fault, ta::Fault::kEventOrder);
  Harness exhausted;
  plan = ta::planEvent({exhausted.state}, tick, {std::numeric_limits<std::uint64_t>::max(), 0}, exhausted.workspace);
  ASSERT_EQ(plan.code, ta::Code::kOk);
  ASSERT_EQ(ta::applyPlan(exhausted.state, plan.plan), ta::Code::kOk);
  EXPECT_EQ(exhausted.state.scalar().fault, ta::Fault::kSequenceExhausted);
  Harness deadline;
  Message a(1, {0.01});
  deadline.lidar(a, std::numeric_limits<std::int64_t>::max() - 1);
  EXPECT_EQ(deadline.state.scalar().fault, ta::Fault::kArithmetic);
  EXPECT_EQ(deadline.pending(), 0U);
}

TEST(TimeAdmission, GeometryRejectsCollapsedAdjacentBoundariesAndCheckedIntegerOverflow)
{
  ta::WindowBounds b;
  EXPECT_FALSE(ta::locateWindow(1788617881.1, 1, b));
  EXPECT_FALSE(ta::boundsAt(std::numeric_limits<std::int64_t>::max(), 100000000, b));
  EXPECT_FALSE(ta::locateWindow(std::numeric_limits<double>::max(), 100000000, b));
  EXPECT_FALSE(ta::locateWindow(-1.0, 100000000, b));
  auto limits = fixtureLimits();
  limits.width_ns = 1;
  Harness h(limits);
  Message a(1, {1788617881.1});
  h.lidar(a);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kWindowGeometry);
  EXPECT_EQ(h.pending(), 0U);
  std::size_t value = 123, actions = 0, scratch = 0;
  EXPECT_FALSE(ta::checkedAdd(std::numeric_limits<std::size_t>::max(), 1, value));
  EXPECT_EQ(value, 123U);
  EXPECT_FALSE(ta::checkedMultiply(std::numeric_limits<std::size_t>::max(), 2, value));
  EXPECT_FALSE(ta::workspaceRequirements(std::numeric_limits<std::size_t>::max(), 1, 1, 1, actions, scratch));
}

TEST(TimeAdmission, InvalidAndInsufficientStorageConfigurationNeverMutatesStateOrBuffers)
{
  Buffers buffers;
  buffers.points[0].used = true;
  ta::PlanWorkspace workspace(span(buffers.actions), span(buffers.ingress), span(buffers.indices));
  ta::State state;
  auto limits = fixtureLimits();
  limits.max_owned_bytes = 1;
  EXPECT_EQ(ta::initialize(state, 7, limits, buffers.storage(), workspace), ta::Code::kInvalidConfiguration);
  EXPECT_TRUE(buffers.points[0].used);
  EXPECT_EQ(state.revision(), 0U);
  limits = fixtureLimits();
  limits.width_ns = 0;
  EXPECT_EQ(ta::initialize(state, 7, limits, buffers.storage(), workspace), ta::Code::kInvalidConfiguration);
  limits = fixtureLimits();
  limits.max_points = 0;
  EXPECT_EQ(ta::initialize(state, 7, limits, buffers.storage(), workspace), ta::Code::kInvalidConfiguration);
  limits = fixtureLimits();
  limits.lateness_ns = 1000000000;
  EXPECT_EQ(ta::initialize(state, 7, limits, buffers.storage(), workspace), ta::Code::kInvalidConfiguration);
  ta::PlanWorkspace small({buffers.actions.data(), 75}, span(buffers.ingress), span(buffers.indices));
  EXPECT_EQ(ta::initialize(state, 7, fixtureLimits(), buffers.storage(), small), ta::Code::kInvalidConfiguration);
  ta::PlanWorkspace scratch(span(buffers.actions), span(buffers.ingress), {buffers.indices.data(), 15});
  EXPECT_EQ(ta::initialize(state, 7, fixtureLimits(), buffers.storage(), scratch), ta::Code::kInvalidConfiguration);
  ta::PlanWorkspace zero({}, {}, {});
  EXPECT_EQ(ta::initialize(state, 7, fixtureLimits(), buffers.storage(), zero), ta::Code::kInvalidConfiguration);
  auto alias = buffers.storage();
  alias.token_points.data = reinterpret_cast<ta::PointRef *>(buffers.points.data());
  EXPECT_EQ(ta::initialize(state, 7, fixtureLimits(), alias, workspace), ta::Code::kInvalidConfiguration);
}

TEST(TimeAdmission, ActualPeakIncludesAllPhysicalCapacitiesNotOnlyPendingSizes)
{
  Harness h;
  const auto & b = h.buffers;
  const auto expected = sizeof(ta::State) + sizeof(ta::PlanWorkspace) + sizeof(b.points) + sizeof(b.windows) +
                        sizeof(b.identities) + sizeof(b.identity_bytes) + sizeof(b.imu) + sizeof(b.imu_bytes) +
                        sizeof(b.token_points) + sizeof(b.token_imu) + sizeof(b.actions) + sizeof(b.ingress) +
                        sizeof(b.indices);
  EXPECT_EQ(h.state.ownedBytes(), expected);
  EXPECT_EQ(h.pending(), 0U);
  auto limits = fixtureLimits();
  limits.max_owned_bytes = expected;
  Harness exact(limits);
  EXPECT_EQ(exact.state.ownedBytes(), expected);
  Buffers smaller;
  ta::PlanWorkspace workspace(span(smaller.actions), span(smaller.ingress), span(smaller.indices));
  ta::State state;
  --limits.max_owned_bytes;
  EXPECT_EQ(ta::initialize(state, 7, limits, smaller.storage(), workspace), ta::Code::kInvalidConfiguration);
}

TEST(TimeAdmission, StaleRevisionCrossStateAndReusedWorkspaceCannotPartiallyApply)
{
  Harness h, other;
  Message a(1, {0.01});
  ta::Event event;
  event.kind = ta::EventKind::kLidar;
  event.message = a.view();
  const auto first = ta::planEvent({h.state}, event, {1, 0}, h.workspace);
  ASSERT_EQ(first.code, ta::Code::kOk);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_EQ(h.state.revision(), 0U);
  EXPECT_EQ(ta::applyPlan(other.state, first.plan), ta::Code::kStalePlan);
  ASSERT_EQ(ta::applyPlan(h.state, first.plan), ta::Code::kOk);
  EXPECT_EQ(ta::applyPlan(h.state, first.plan), ta::Code::kStalePlan);
  EXPECT_EQ(h.pending(), 1U);
  EXPECT_EQ(h.state.revision(), 1U);
  ta::Event tick;
  const auto abandoned = ta::planEvent({h.state}, tick, {2, 0}, h.workspace);
  const auto replacement = ta::planEvent({h.state}, tick, {3, 0}, h.workspace);
  EXPECT_EQ(ta::applyPlan(h.state, abandoned.plan), ta::Code::kStalePlan);
  EXPECT_EQ(h.state.scalar().event_sequence, 1U);
  ASSERT_EQ(ta::applyPlan(h.state, replacement.plan), ta::Code::kOk);
  EXPECT_EQ(h.state.scalar().event_sequence, 3U);
}

TEST(TimeAdmission, WorkspaceFailureLeavesStateUntouchedThenFixedFaultPathTerminatesWithoutRetry)
{
  Harness h;
  Message a(1, {0.01});
  h.lidar(a);
  ta::PlanWorkspace zero({}, {}, {});
  ta::Event tick;
  const auto revision = h.state.revision();
  const auto failed = ta::planEvent({h.state}, tick, {2, 0}, zero);
  EXPECT_EQ(failed.code, ta::Code::kPlanResourceFailure);
  EXPECT_EQ(h.pending(), 1U);
  EXPECT_EQ(h.state.revision(), revision);
  allocations = 0;
  observe_allocations = true;
  ta::latchPlanResourceFailure(h.state);
  observe_allocations = false;
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kPlanResource);
  EXPECT_EQ(h.points(ta::Reason::kFault), 1U);
  EXPECT_EQ(h.identities(), 1U);
  EXPECT_EQ(h.step(tick), ta::Code::kAlreadyStopped);
}

TEST(TimeAdmission, FullActionWorkspaceAndSortingPerformNoAllocationAndLongHistoryStaysBounded)
{
  Buffers buffers;
  ta::PlanWorkspace workspace({buffers.actions.data(), 76}, span(buffers.ingress), span(buffers.indices));
  ta::State state;
  ASSERT_EQ(ta::initialize(state, 7, fixtureLimits(), buffers.storage(), workspace), ta::Code::kOk);
  Message full(1, {0.03, 0.02, 0.01, 0.04, 0.13, 0.12, 0.11, 0.14});
  ta::Event event;
  event.kind = ta::EventKind::kLidar;
  event.message = full.view();
  allocations = 0;
  observe_allocations = true;
  const auto planned = ta::planEvent({state}, event, {1, 0}, workspace);
  const auto applied = ta::applyPlan(state, planned.plan);
  observe_allocations = false;
  ASSERT_EQ(planned.code, ta::Code::kOk);
  ASSERT_EQ(applied, ta::Code::kOk);
  EXPECT_EQ(allocations, 0U);
  std::array<std::uint8_t, 4> imu_bytes{1, 2, 3, 4};
  event = {};
  event.kind = ta::EventKind::kImu;
  event.imu = {7, 1, 100.0, {imu_bytes.data(), 4}};
  observe_allocations = true;
  const auto imu_plan = ta::planEvent({state}, event, {2, 0}, workspace);
  const auto imu_apply = ta::applyPlan(state, imu_plan.plan);
  event = {};
  const auto close = ta::planEvent({state}, event, {3, 100000000}, workspace);
  const auto close_apply = ta::applyPlan(state, close.plan);
  observe_allocations = false;
  ASSERT_EQ(imu_apply, ta::Code::kOk);
  ASSERT_EQ(close_apply, ta::Code::kOk);
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(state.tokenPoints().size, 4U);
  Harness bounded;
  for (std::uint64_t sequence = 1; sequence <= 100; ++sequence) {
    Message rejected(sequence, {0.01});
    rejected.facts.original_status = fast_lio::ScanTimeStatus::kInvalidOffset;
    bounded.lidar(rejected);
  }
  EXPECT_EQ(bounded.identities(), 8U);
  EXPECT_EQ(bounded.state.scalar().expiry_floor, 93U);
  EXPECT_EQ(bounded.pending(), 0U);
  EXPECT_EQ(bounded.points(ta::Reason::kInvalidSpan), 100U);
}

TEST(TimeAdmission, LargeEmptyGapDoesNotAllocateIntermediateWindows)
{
  Harness h;
  Message a(1, {0.01}), b(2, {1000000000.01});
  h.lidar(a);
  h.lidar(b);
  std::size_t windows = 0;
  for (const auto & r : h.buffers.windows)
    windows += r.used;
  EXPECT_EQ(windows, 2U);
  EXPECT_EQ(h.pending(), 2U);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 1000000000.01);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.C, 999999999.8);
  EXPECT_FALSE(h.state.scalar().in_flight);
}

TEST(TimeAdmission, CanonicalGuardKeepsHeaderWhenSamplingRemovedEarliestPoint)
{
  Harness h;
  Message a(1, {10.2}, 10.0);
  a.facts.declared_count = a.facts.decoded_count = 2;
  a.points[0].raw_index = 1;
  Message b(2, {9.1}, 9.0);
  h.lidar(a);
  h.lidar(b);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kNone);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.max_canonical_begin, 10.0);
  EXPECT_DOUBLE_EQ(*h.state.scalar().frontier.H, 10.2);
  EXPECT_EQ(h.points(ta::Reason::kClosed), 1U);
}

TEST(TimeAdmission, EmptyAndNonfiniteMessagesNeverEstablishPointFrontier)
{
  Harness empty, negative, infinite;
  Message a(1, {});
  a.facts.original_status = fast_lio::ScanTimeStatus::kEmpty;
  empty.lidar(a);
  EXPECT_FALSE(empty.state.scalar().frontier.H);
  EXPECT_FALSE(empty.state.scalar().frontier.C);
  EXPECT_FALSE(empty.state.scalar().frontier.Fc);
  EXPECT_EQ(empty.state.scalar().diagnostics.messages[static_cast<std::size_t>(ta::Reason::kInvalidSpan)], 1U);
  Message b(1, {-0.01});
  negative.lidar(b);
  EXPECT_EQ(negative.points(ta::Reason::kInvalidMessage), 1U);
  Message c(1, {std::numeric_limits<double>::infinity()});
  infinite.lidar(c);
  EXPECT_EQ(infinite.points(ta::Reason::kInvalidMessage), 1U);
}

TEST(TimeAdmission, RecentAndRejectedIdentityConflictsRemainDetectableUntilExpiry)
{
  Harness recent;
  recent.imu(1.0);
  Message a(1, {0.01});
  recent.lidar(a);
  recent.control(ta::EventKind::kTick, 100000000);
  recent.settle(recent.ack(0.01));
  a.contents[0][0] = 101;
  recent.lidar(a, 100000000);
  EXPECT_EQ(recent.state.scalar().fault, ta::Fault::kIdentityConflict);
  auto limits = fixtureLimits();
  limits.max_points = 1;
  Harness rejected(limits);
  Message held(1, {0.01}), no_room(2, {0.11});
  rejected.lidar(held);
  rejected.lidar(no_room);
  no_room.contents[0][0] = 33;
  rejected.lidar(no_room);
  EXPECT_EQ(rejected.state.scalar().fault, ta::Fault::kIdentityConflict);
  EXPECT_EQ(rejected.points(ta::Reason::kCapacityNew), 1U);
}

TEST(TimeAdmission, CompactionPreservesRetainedAuthorityAfterRepeatedSlotReuse)
{
  auto limits = fixtureLimits();
  limits.max_identity_records = 2;
  Harness h(limits);
  Message a(1, {0.01}), b(2, {0.02}), c(3, {0.03}), d(4, {0.04});
  a.facts.original_status = b.facts.original_status = c.facts.original_status = d.facts.original_status =
    fast_lio::ScanTimeStatus::kInvalidOffset;
  h.lidar(a);
  h.lidar(b);
  h.lidar(c);
  h.lidar(d);
  h.lidar(c);
  h.lidar(d);
  EXPECT_EQ(h.points(ta::Reason::kDuplicate), 2U);
  EXPECT_EQ(h.identities(), 2U);
  EXPECT_EQ(h.state.scalar().expiry_floor, 3U);
  c.authority[0] = 0;
  h.lidar(c);
  EXPECT_EQ(h.state.scalar().fault, ta::Fault::kIdentityConflict);
}

TEST(TimeAdmission, UncommittedSettleTerminatesOnceRetainsImuAndDoesNotInferFc)
{
  Harness h;
  h.imu(0.0, 1);
  h.imu(0.1, 2);
  Message a(1, {0.01});
  h.lidar(a);
  h.control(ta::EventKind::kTick, 100000000);
  ta::Event event;
  event.kind = ta::EventKind::kAck;
  event.ack = h.ack(std::nullopt, ImuDisposition::kUncommitted, ImuProcessReason::kNoImu, false);
  allocations = 0;
  observe_allocations = true;
  const auto result = h.step(event, 100000000);
  observe_allocations = false;
  ASSERT_EQ(result, ta::Code::kOk);
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(h.pending(), 0U);
  EXPECT_EQ(h.state.imu().size, 2U);
  EXPECT_FALSE(h.state.scalar().frontier.Fc);
  EXPECT_EQ(h.points(ta::Reason::kSettled), 1U);
  EXPECT_EQ(h.state.scalar().diagnostics.settled_batches, 1U);
  EXPECT_FALSE(h.state.scalar().diagnostics.last_ack.producedOutput);
}

TEST(TimeAdmission, ImuDeadlineArithmeticOverflowAndOversizedSampleHaveExplicitTerminals)
{
  auto limits = fixtureLimits();
  limits.wait_ns = 0;
  Harness deadline(limits);
  Message a(1, {0.01});
  deadline.lidar(a, std::numeric_limits<std::int64_t>::max() - 1);
  EXPECT_EQ(deadline.state.scalar().fault, ta::Fault::kArithmetic);
  EXPECT_EQ(deadline.pending(), 0U);
  limits = fixtureLimits();
  limits.max_imu_bytes = 3;
  Harness oversized(limits);
  oversized.lidar(a);
  oversized.imu(0.02);
  EXPECT_EQ(oversized.points(ta::Reason::kImuCapacityLoss), 1U);
  EXPECT_EQ(oversized.state.imu().size, 0U);
  EXPECT_EQ(oversized.state.scalar().diagnostics.imu_capacity_samples, 1U);
  EXPECT_FALSE(oversized.state.scalar().frontier.Fc);
}
}  // namespace

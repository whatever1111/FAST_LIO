#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <type_traits>

#include "imu_process_outcome.hpp"
#include "scan_history_policy.hpp"
#include "scan_time_policy.hpp"

namespace fast_lio
{
// These are local populations, never physical point or cross-node identities.
enum class AdmissionUnit
{
  kDeclaredInput,
  kEffective,
  kDeskew,
  kDownsampled,
  kMapPresented
};
enum class AdmissionBucket
{
  kReceived,
  kEpochRejected,
  kLateRejected,
  kEnqueuedDeclared,
  kEnqueued,
  kTrimCount,
  kTrimAge,
  kClearEpoch,
  kClearFatal,
  kSyncInvalid,
  kSyncNonadvancing,
  kBorrowed,
  kEmptyImu,
  kBootstrap,
  kProcessEntered,
  kDeskew,
  kDownsampled,
  kLeadRejected,
  kTransactionRejected,
  kCount
};
enum class AdmissionInputResult
{
  kAccepted,
  kEpochRejected,
  kLeadRejected,
  kTransactionRejected
};
inline AdmissionInputResult classifyAdmissionInput(InputEpochGuard & epoch,
                                                   InputLeadFilter & lead,
                                                   double previous,
                                                   double current,
                                                   bool input_allowed,
                                                   double max_rollback,
                                                   double max_lead)
{
  if (!input_allowed)
    return epoch.faulted() ? AdmissionInputResult::kEpochRejected : AdmissionInputResult::kTransactionRejected;
  if (!lead.admit(previous, current, max_lead))
    return AdmissionInputResult::kLeadRejected;
  return epoch.observe(previous, current, max_rollback) ? AdmissionInputResult::kAccepted
                                                        : AdmissionInputResult::kEpochRejected;
}
enum class AdmissionExit
{
  kBootstrap,
  kProcessException,
  kSettleFailed,
  kRequiredPriorLost,
  kNotProcessed,
  kNoCloud,
  kPrepareFailed,
  kInitialMap,
  kPriorCapacity,
  kDownsampleSmall,
  kRecoveryFailed,
  kFaultExit,
  kPublished,
  kUnknown,
  kCount
};
enum class AdmissionMapKind
{
  kInitialBuild,
  kPriorBuild,
  kPriorAdd,
  kRecoveryBuild,
  kRecoveryAdd,
  kIncrementalDownsampled,
  kIncrementalDirect,
  kPriorInstall,
  kCount
};
struct AdmissionCount
{
  std::uint64_t messages = 0, points = 0;
};
struct AdmissionMapOperation
{
  std::uint64_t submitted = 0, completed = 0;
  std::uint64_t presented = 0, current = 0, prior = 0, reused = 0;
};
struct ProcessAudit
{
  bool entered = false, completed = false;
  std::uint64_t input_points = 0;
  bool history_evaluated = false;
  ScanHistoryAction history_action = ScanHistoryAction::kKeepAll;
  bool history_erase_completed = false;
  std::uint64_t history_erased_points = 0, history_retained_points = 0;
  bool deskew_completed = false;
  std::uint64_t deskew_points = 0;
  bool fc_before_known = false, fc_after_known = false;
  std::uint64_t fc_before_bits = 0, fc_after_bits = 0;
  ImuProcessReason reason = ImuProcessReason::kNone;
  ImuDisposition disposition = ImuDisposition::kUncommitted;
  bool produced_output = false;
};
inline std::uint64_t admissionDoubleBits(double value) noexcept
{
  std::uint64_t bits = 0;
  static_assert(sizeof bits == sizeof value, "double must have 64 bits");
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}
inline void beginProcessAudit(ProcessAudit & audit, std::uint64_t input_points, double fc) noexcept
{
  audit = {};
  audit.entered = true;
  audit.input_points = input_points;
  audit.fc_before_known = true;
  audit.fc_before_bits = admissionDoubleBits(fc);
}
inline void finishProcessAudit(ProcessAudit & audit, const ImuProcessOutcome & outcome, double fc) noexcept
{
  audit.completed = true;
  audit.reason = outcome.reason;
  audit.disposition = outcome.disposition;
  audit.produced_output = outcome.producedOutput;
  audit.fc_after_known = true;
  audit.fc_after_bits = admissionDoubleBits(fc);
}
struct TimeAdmissionAccounting
{
  bool enabled = false, supported = true, overflow = false, exact = true, accounting_complete = true;
  std::uint64_t local_event_seq = 0;
  std::array<AdmissionCount, static_cast<std::size_t>(AdmissionBucket::kCount)> counts{};
  std::array<AdmissionCount, static_cast<std::size_t>(ImuProcessReason::kCount)> process_outcomes{};
  std::array<std::uint64_t, 4> process_dispositions{};
  AdmissionCount process_exceptions{}, history_keep{}, history_reject{}, history_mixed{}, history_unknown{};
  AdmissionCount history_not_reached{};
  std::uint64_t history_erased = 0, history_retained = 0;
  std::uint64_t history_keep_erased = 0, history_keep_retained = 0;
  std::uint64_t history_reject_erased = 0, history_reject_retained = 0;
  std::array<std::uint64_t, static_cast<std::size_t>(AdmissionExit::kCount)> exits{};
  std::array<std::uint64_t, 5> scan_time_status{};
  std::array<std::uint64_t, 7> confirmations{};
  std::array<std::uint64_t, static_cast<std::size_t>(ImuProcessReason::kCount)> settle_reasons{};
  std::array<std::uint64_t, 4> settle_dispositions{};
  std::uint64_t wait_no_imu = 0, wait_tail = 0, trim_also_age = 0;
  std::uint64_t callback_exceptions = 0, settle_samples = 0, last_token = 0, last_epoch = 0;
  std::uint64_t settle_reason_not_recorded = 0;
  bool last_settle_known = false;
  std::uint64_t settle_fc_bits = 0;
  bool canonical_header_known = false, bootstrap_fc_known = false;
  std::uint64_t canonical_header_bits = 0, bootstrap_fc_before_bits = 0, bootstrap_fc_after_bits = 0;
  ProcessAudit last_process{};
  std::array<AdmissionMapOperation, static_cast<std::size_t>(AdmissionMapKind::kCount)> map{};
  AdmissionCount staged{}, released{}, reset_discarded{};
  std::array<std::uint64_t, 4> map_gates{};  // operator, recovery, prior, map not open; labels may overlap
  std::uint64_t async_dispatches = 0, async_joins = 0;
  bool async_inflight = false;

  void add(std::uint64_t & value, std::uint64_t increment = 1) noexcept
  {
    if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
      value = std::numeric_limits<std::uint64_t>::max();
      overflow = true;
      exact = accounting_complete = false;
    } else {
      value += increment;
    }
  }
  void count(AdmissionCount & value, std::uint64_t points, std::uint64_t messages = 1) noexcept
  {
    add(value.messages, messages);
    add(value.points, points);
  }
  void record(AdmissionBucket bucket, std::uint64_t points) noexcept
  {
    add(local_event_seq);
    count(counts[static_cast<std::size_t>(bucket)], points);
  }
  void recordInputRejection(AdmissionInputResult result, std::uint64_t points) noexcept
  {
    if (result == AdmissionInputResult::kEpochRejected)
      record(AdmissionBucket::kEpochRejected, points);
    else if (result == AdmissionInputResult::kLeadRejected)
      record(AdmissionBucket::kLeadRejected, points);
    else if (result == AdmissionInputResult::kTransactionRejected)
      record(AdmissionBucket::kTransactionRejected, points);
  }
  void process(const ProcessAudit & audit) noexcept
  {
    last_process = audit;
    add(local_event_seq);
    if (!audit.entered) {
      accounting_complete = false;
      return;
    }
    record(AdmissionBucket::kProcessEntered, audit.input_points);
    if (audit.completed) {
      count(process_outcomes[static_cast<std::size_t>(audit.reason)], audit.input_points);
      add(process_dispositions[static_cast<std::size_t>(audit.disposition)]);
    } else {
      count(process_exceptions, audit.input_points);
      accounting_complete = false;
    }
    if (audit.completed && !audit.history_evaluated) {
      count(history_not_reached, audit.input_points);
    } else if (!audit.history_evaluated || !audit.history_erase_completed) {
      count(history_unknown, audit.input_points);
    } else if (audit.history_action == ScanHistoryAction::kKeepAll) {
      count(history_keep, audit.input_points);
      add(history_keep_erased, audit.history_erased_points);
      add(history_keep_retained, audit.history_retained_points);
    } else if (audit.history_action == ScanHistoryAction::kReject) {
      count(history_reject, audit.input_points);
      add(history_reject_erased, audit.history_erased_points);
      add(history_reject_retained, audit.history_retained_points);
    } else {
      count(history_mixed, audit.input_points);
      add(history_erased, audit.history_erased_points);
      add(history_retained, audit.history_retained_points);
    }
    if (audit.deskew_completed)
      record(AdmissionBucket::kDeskew, audit.deskew_points);
  }
  void submit(AdmissionMapKind kind, std::uint64_t current, std::uint64_t prior = 0, std::uint64_t reused = 0) noexcept
  {
    auto & op = map[static_cast<std::size_t>(kind)];
    add(local_event_seq);
    add(op.submitted);
    add(op.current, current);
    add(op.prior, prior);
    add(op.reused, reused);
    add(op.presented, current);
    add(op.presented, prior);
    add(op.presented, reused);
  }
  void complete(AdmissionMapKind kind) noexcept { add(map[static_cast<std::size_t>(kind)].completed); }
  void joined() noexcept
  {
    if (async_inflight) {
      complete(AdmissionMapKind::kIncrementalDownsampled);
      complete(AdmissionMapKind::kIncrementalDirect);
      add(async_joins);
      async_inflight = false;
    }
  }
};
struct TimeAdmissionSnapshot
{
  std::uint32_t schema = 1;
  TimeAdmissionAccounting accounting{};
  AdmissionCount queue_pending{}, delayed_pending{};
  AdmissionCount callback_unresolved_declared{}, borrowed_unresolved{};
  bool callback_partition = false, queue_partition = false, borrowed_partition = false;
  bool process_partition = false, history_partition = false, map_partition = false;
  bool pending_known = false, map_joined = false, shutdown = false, reader_eof_verified = false;
  bool map_pending_known = false;
  std::uint64_t map_pending_operations = 0;
  bool source_identity_known = false, config_identity_known = false, fc_known = false;
  std::uint64_t fc_bits = 0;
  std::uint64_t imu_duplicate_confirmations = 0, imu_duplicate_integrations = 0, imu_watermark_regressions = 0;
};
// Scopes only finalize existing control-flow exits. They own no resource and never throw.
struct AdmissionCallbackScope
{
  TimeAdmissionAccounting * observer;
  bool resolved = false;
  int uncaught = observer ? std::uncaught_exceptions() : 0;
  void resolve() noexcept
  {
    if (observer)
      resolved = true;
  }
  ~AdmissionCallbackScope()
  {
    if (observer && (!resolved || std::uncaught_exceptions() > uncaught)) {
      observer->add(observer->callback_exceptions);
      observer->accounting_complete = false;
    }
  }
};
struct AdmissionCallerScope
{
  TimeAdmissionAccounting * observer;
  AdmissionExit exit = AdmissionExit::kUnknown;
  int uncaught = observer ? std::uncaught_exceptions() : 0;
  void finish(AdmissionExit reason) noexcept
  {
    if (observer)
      exit = reason;
  }
  ~AdmissionCallerScope()
  {
    if (observer) {
      if (std::uncaught_exceptions() > uncaught)
        exit = AdmissionExit::kUnknown;
      observer->add(observer->exits[static_cast<std::size_t>(exit)]);
      if (exit == AdmissionExit::kUnknown)
        observer->accounting_complete = false;
    }
  }
};
inline AdmissionUnit admissionPointUnit(AdmissionBucket bucket) noexcept
{
  if (bucket <= AdmissionBucket::kEnqueuedDeclared || bucket == AdmissionBucket::kLeadRejected ||
      bucket == AdmissionBucket::kTransactionRejected)
    return AdmissionUnit::kDeclaredInput;
  if (bucket == AdmissionBucket::kDeskew)
    return AdmissionUnit::kDeskew;
  if (bucket == AdmissionBucket::kDownsampled)
    return AdmissionUnit::kDownsampled;
  return AdmissionUnit::kEffective;
}
inline bool admissionPartition(TimeAdmissionAccounting & a,
                               const AdmissionCount & whole,
                               const AdmissionCount * parts,
                               std::size_t size) noexcept
{
  AdmissionCount sum{};
  for (std::size_t i = 0; i < size; ++i)
    a.count(sum, parts[i].points, parts[i].messages);
  return a.exact && whole.messages == sum.messages && whole.points == sum.points;
}
inline bool admissionRemainder(TimeAdmissionAccounting & a,
                               const AdmissionCount & whole,
                               const AdmissionCount * parts,
                               std::size_t size,
                               AdmissionCount & remainder) noexcept
{
  AdmissionCount sum{};
  for (std::size_t i = 0; i < size; ++i)
    a.count(sum, parts[i].points, parts[i].messages);
  if (!a.exact || sum.messages > whole.messages || sum.points > whole.points)
    return false;
  remainder = {whole.messages - sum.messages, whole.points - sum.points};
  return true;
}
inline void checkAdmissionSnapshot(TimeAdmissionSnapshot & snapshot) noexcept
{
  auto & a = snapshot.accounting;
  const auto get = [&](AdmissionBucket b) {
    return a.counts[static_cast<std::size_t>(b)];
  };
  const AdmissionCount callback[] = {get(AdmissionBucket::kEpochRejected),
                                     get(AdmissionBucket::kLeadRejected),
                                     get(AdmissionBucket::kTransactionRejected),
                                     get(AdmissionBucket::kLateRejected),
                                     get(AdmissionBucket::kEnqueuedDeclared)};
  snapshot.callback_partition =
    admissionRemainder(a, get(AdmissionBucket::kReceived), callback, 5, snapshot.callback_unresolved_declared);
  const AdmissionCount queue[] = {snapshot.queue_pending,
                                  get(AdmissionBucket::kTrimCount),
                                  get(AdmissionBucket::kTrimAge),
                                  get(AdmissionBucket::kClearEpoch),
                                  get(AdmissionBucket::kClearFatal),
                                  get(AdmissionBucket::kSyncInvalid),
                                  get(AdmissionBucket::kSyncNonadvancing),
                                  get(AdmissionBucket::kBorrowed)};
  snapshot.queue_partition = snapshot.pending_known && admissionPartition(a, get(AdmissionBucket::kEnqueued), queue, 8);
  const AdmissionCount borrowed[] = {
    get(AdmissionBucket::kEmptyImu), get(AdmissionBucket::kBootstrap), get(AdmissionBucket::kProcessEntered)};
  snapshot.borrowed_partition =
    admissionRemainder(a, get(AdmissionBucket::kBorrowed), borrowed, 3, snapshot.borrowed_unresolved);
  AdmissionCount process = a.process_exceptions;
  for (const auto & outcome : a.process_outcomes)
    a.count(process, outcome.points, outcome.messages);
  snapshot.process_partition = admissionPartition(a, get(AdmissionBucket::kProcessEntered), &process, 1);
  const AdmissionCount history[] = {
    a.history_keep, a.history_reject, a.history_mixed, a.history_unknown, a.history_not_reached};
  std::uint64_t mixed = a.history_erased;
  a.add(mixed, a.history_retained);
  snapshot.history_partition = admissionPartition(a, get(AdmissionBucket::kProcessEntered), history, 5) && a.exact &&
                               mixed == a.history_mixed.points;
  const AdmissionCount delayed[] = {a.released, a.reset_discarded, snapshot.delayed_pending};
  snapshot.map_partition = snapshot.pending_known && admissionPartition(a, a.staged, delayed, 3);
  for (std::size_t i = 0; i < a.map.size(); ++i) {
    const auto & op = a.map[i];
    const bool async_kind = i == static_cast<std::size_t>(AdmissionMapKind::kIncrementalDownsampled) ||
                            i == static_cast<std::size_t>(AdmissionMapKind::kIncrementalDirect);
    std::uint64_t accounted = op.completed;
    if (a.async_inflight && async_kind)
      a.add(accounted);
    snapshot.map_partition = snapshot.map_partition && a.exact && op.submitted == accounted;
  }
  if (!a.supported || !snapshot.callback_partition || !snapshot.queue_partition || !snapshot.borrowed_partition ||
      !snapshot.process_partition || !snapshot.history_partition || !snapshot.map_partition ||
      snapshot.callback_unresolved_declared.messages || snapshot.borrowed_unresolved.messages)
    a.accounting_complete = false;
}
static_assert(std::is_trivially_copyable<TimeAdmissionSnapshot>::value &&
                std::is_standard_layout<TimeAdmissionSnapshot>::value,
              "accounting snapshots must remain fixed PODs");
}  // namespace fast_lio

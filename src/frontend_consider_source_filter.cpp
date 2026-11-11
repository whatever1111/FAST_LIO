#include "frontend_consider_source_filter.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <set>
#include <tuple>

namespace fastlio::joint_source
{
namespace
{

using detail::appendCanonicalCoordinates;
using detail::appendCanonicalMatrix;
using detail::appendCanonicalText;
using detail::coordinatesEqual;
using Watermarks = std::map<std::uint32_t, std::uint64_t>;

struct RawLess
{
  bool operator()(const RawId & a, const RawId & b) const
  {
    return std::tie(a.stream, a.ordinal, a.sub) < std::tie(b.stream, b.ordinal, b.sub);
  }
};

void appendRaw(std::string & bytes, const RawId & raw)
{
  appendCanonicalText(bytes, std::to_string(raw.stream));
  appendCanonicalText(bytes, std::to_string(raw.ordinal));
  appendCanonicalText(bytes, std::to_string(raw.sub));
}

void appendSequence(std::string & bytes, const std::optional<RawId> & raw)
{
  appendCanonicalText(bytes, raw ? "typed" : "opaque");
  if (raw)
    appendRaw(bytes, *raw);
}

std::size_t sumBytes(std::size_t a, std::size_t b)
{
  return b > std::numeric_limits<std::size_t>::max() - a ? std::numeric_limits<std::size_t>::max() : a + b;
}

std::size_t productBytes(std::size_t a, std::size_t b)
{
  return a != 0 && b > std::numeric_limits<std::size_t>::max() / a ? std::numeric_limits<std::size_t>::max() : a * b;
}

std::size_t matrixBytes(const Eigen::MatrixXd & value)
{
  return sumBytes(sizeof(value), productBytes(static_cast<std::size_t>(value.size()), sizeof(double)));
}

std::size_t textBytes(const std::string & value)
{
  return sumBytes(sizeof(value), value.capacity() + 1);
}

std::size_t coordinatesBytes(const Coordinates & value)
{
  auto result = sumBytes(sizeof(value), sumBytes(textBytes(value.frame), textBytes(value.tangent)));
  result = sumBytes(result, textBytes(value.epoch));
  for (const auto & text : value.components)
    result = sumBytes(result, textBytes(text));
  for (const auto & text : value.units)
    result = sumBytes(result, textBytes(text));
  return result;
}

std::size_t sourceBytes(const Source & value)
{
  auto result = sumBytes(sizeof(value), coordinatesBytes(value.coordinates));
  for (const auto * text : {&value.key, &value.kind, &value.model, &value.lifetime})
    result = sumBytes(result, textBytes(*text));
  for (const auto & text : value.rawAncestry)
    result = sumBytes(result, textBytes(text));
  for (const auto & text : value.aliases)
    result = sumBytes(result, textBytes(text));
  for (const auto & raw : value.typedRaw)
    result = sumBytes(result, sizeof(raw) + textBytes(raw.raw));
  for (const auto & role : value.rawRoles)
    result = sumBytes(result, sizeof(role) + textBytes(role.raw));
  return result;
}

void mirrorLower(Eigen::MatrixXd & matrix)
{
  for (Eigen::Index col = 0; col < matrix.cols(); ++col)
    for (Eigen::Index row = 0; row < col; ++row)
      matrix(row, col) = matrix(col, row);
}

Eigen::MatrixXd selectRows(const Eigen::MatrixXd & matrix, const std::vector<Eigen::Index> & rows)
{
  Eigen::MatrixXd result(static_cast<Eigen::Index>(rows.size()), matrix.cols());
  for (std::size_t i = 0; i < rows.size(); ++i)
    result.row(static_cast<Eigen::Index>(i)) = matrix.row(rows[i]);
  return result;
}

struct Injection
{
  RawId raw;
  RawId frame;
  Eigen::MatrixXd loading;  // -G L, or an explicit process/copy cross input.
};

enum class OperationKind
{
  kUpdate,
  kProcess,
  kClone,
  kRetire
};

// Immutable accepted INPUTS, not a covariance cache. Replaying these recovers
// Cov([e;zeta],eta) and labelled derivatives after cross-call reuse and cloning.
struct Operation
{
  OperationKind kind = OperationKind::kUpdate;
  Eigen::Index oldSourceDof = 0;
  Eigen::MatrixXd active;
  Eigen::MatrixXd source;
  std::vector<Eigen::Index> sourceRows;
  std::vector<Injection> copies;
  std::vector<std::pair<std::string, Eigen::MatrixXd>> sensitivities;
};

struct FrameCopy
{
  RawId frame;
  std::size_t birth = 0;
};

struct NoiseModel
{
  ReuseUse use;
  Coordinates coordinates;
  std::string evidence;
  Eigen::MatrixXd covariance;  // Unscaled immutable INPUT.
  std::vector<FrameCopy> frames;
};

struct SensitivityModel
{
  Source source;
  Eigen::MatrixXd initial;  // Complete derivative INPUT at registration.
  std::size_t birth = 0;
};

struct Metadata
{
  Coordinates active;
  std::vector<Source> sources;
  std::vector<Source> sensitivityLabels;
  std::map<std::string, std::string> names;  // Canonical aliases, including retired ones.
  std::set<std::string> retiredNames;
  std::set<std::string> retiredDesignatedRaw;
  std::set<std::string> consumedRaw;
  std::set<std::string> remainders;
  std::vector<TypedRaw> bindings;
  Watermarks rawHigh;
  Watermarks sourceHigh;
  Watermarks compacted;
  Watermarks factorHigh;
  std::map<std::string, RawId> factorBindings;
  std::vector<FactorIdentity> recent;
  std::vector<NoiseModel> noises;
  std::vector<SensitivityModel> sensitivities;
  std::vector<std::shared_ptr<const Operation>> history;
  MomentKnowledge knowledge = MomentKnowledge::kUnknown;
  std::size_t opaqueRetired = 0;
};

std::size_t operationBytes(const Operation & value)
{
  auto result = sumBytes(sizeof(value), sumBytes(matrixBytes(value.active), matrixBytes(value.source)));
  result = sumBytes(result, productBytes(value.sourceRows.size(), sizeof(Eigen::Index)));
  for (const auto & copy : value.copies)
    result = sumBytes(result, sizeof(copy) + matrixBytes(copy.loading));
  for (const auto & sensitivity : value.sensitivities)
    result = sumBytes(result, textBytes(sensitivity.first) + matrixBytes(sensitivity.second));
  return result;
}

std::string identityBytes(const FactorIdentity & value)
{
  std::string result;
  for (const auto * text : {&value.key, &value.payload, &value.remainderKey, &value.independenceEvidence})
    appendCanonicalText(result, *text);
  appendSequence(result, value.sequence);
  appendCanonicalText(result, std::to_string(value.rawAncestry.size()));
  for (const auto & raw : value.rawAncestry)
    appendCanonicalText(result, raw);
  appendCanonicalText(result, std::to_string(value.shared.size()));
  for (const auto & shared : value.shared) {
    appendCanonicalText(result, shared.raw);
    appendCanonicalText(result, shared.source);
  }
  appendCanonicalText(result, std::to_string(value.typedRaw.size()));
  for (const auto & raw : value.typedRaw) {
    appendCanonicalText(result, raw.raw);
    appendRaw(result, raw.identity);
  }
  appendCanonicalText(result, std::to_string(value.reuse.size()));
  for (const auto & use : value.reuse) {
    appendRaw(result, use.raw);
    appendRaw(result, use.frame);
    appendCanonicalText(result, std::to_string(use.totalUses));
    appendCanonicalText(result, use.owner.key);
    appendSequence(result, use.owner.sequence);
  }
  return result;
}

std::size_t identityStorageBytes(const FactorIdentity & value)
{
  auto result = sizeof(value);
  for (const auto * text : {&value.key, &value.payload, &value.remainderKey, &value.independenceEvidence})
    result = sumBytes(result, textBytes(*text));
  for (const auto & raw : value.rawAncestry)
    result = sumBytes(result, textBytes(raw));
  for (const auto & shared : value.shared)
    result = sumBytes(result, sizeof(shared) + textBytes(shared.raw) + textBytes(shared.source));
  for (const auto & raw : value.typedRaw)
    result = sumBytes(result, sizeof(raw) + textBytes(raw.raw));
  for (const auto & use : value.reuse)
    result = sumBytes(result, sizeof(use) + textBytes(use.owner.key));
  return result;
}

std::size_t metadataBytes(const Metadata & value)
{
  auto result = sumBytes(sizeof(value), coordinatesBytes(value.active));
  for (const auto & source : value.sources)
    result = sumBytes(result, sourceBytes(source));
  for (const auto & source : value.sensitivityLabels)
    result = sumBytes(result, sourceBytes(source));
  for (const auto & name : value.names)
    result = sumBytes(result, sizeof(name) + textBytes(name.first) + textBytes(name.second));
  for (const auto * names : {&value.retiredNames, &value.retiredDesignatedRaw, &value.consumedRaw, &value.remainders})
    for (const auto & name : *names)
      result = sumBytes(result, sizeof(std::string) + textBytes(name));
  for (const auto & binding : value.bindings)
    result = sumBytes(result, sizeof(binding) + textBytes(binding.raw));
  result = sumBytes(
    result,
    productBytes(value.rawHigh.size() + value.sourceHigh.size() + value.compacted.size() + value.factorHigh.size(),
                 sizeof(Watermarks::value_type) + 4 * sizeof(void *)));
  for (const auto & binding : value.factorBindings)
    result = sumBytes(result, sizeof(binding) + textBytes(binding.first));
  for (const auto & identity : value.recent)
    result = sumBytes(result, identityStorageBytes(identity));
  for (const auto & noise : value.noises) {
    result = sumBytes(result, sizeof(noise) + textBytes(noise.use.owner.key) + textBytes(noise.evidence));
    result = sumBytes(result, coordinatesBytes(noise.coordinates) + matrixBytes(noise.covariance));
    result = sumBytes(result, productBytes(noise.frames.size(), sizeof(FrameCopy)));
  }
  for (const auto & sensitivity : value.sensitivities)
    result = sumBytes(result, sizeof(sensitivity) + sourceBytes(sensitivity.source) + matrixBytes(sensitivity.initial));
  for (const auto & operation : value.history)
    result = sumBytes(result, operationBytes(*operation));
  // Conservative allowance for vector growth, tree nodes and allocator metadata.
  // Shared immutable history is deliberately charged again for each candidate.
  return productBytes(2, result);
}

struct Account
{
  std::size_t resident = 0;
  std::size_t pending = 0;
  std::size_t limit = 0;
};

class Reservation
{
public:
  Reservation() = default;
  ~Reservation() { release(); }
  Reservation(const Reservation &) = delete;
  Reservation & operator=(const Reservation &) = delete;

  bool acquire(const std::shared_ptr<Account> & account, std::size_t bytes)
  {
    if (bytes > account->limit || account->pending > account->limit - bytes ||
        account->resident > account->limit - bytes - account->pending)
      return false;
    account_ = account;
    bytes_ = bytes;
    account_->pending += bytes_;
    return true;
  }

  void release()
  {
    if (account_) {
      account_->pending -= bytes_;
      account_.reset();
    }
  }

private:
  std::shared_ptr<Account> account_;
  std::size_t bytes_ = 0;
};

Eigen::Index resolve(const Metadata & data, const std::string & key)
{
  for (std::size_t i = 0; i < data.sources.size(); ++i)
    if (data.sources[i].key == key || contains(data.sources[i].aliases, key))
      return static_cast<Eigen::Index>(i);
  return -1;
}

Eigen::Index offset(const Metadata & data, Eigen::Index index)
{
  Eigen::Index result = 0;
  for (Eigen::Index i = 0; i < index; ++i)
    result += static_cast<Eigen::Index>(data.sources[static_cast<std::size_t>(i)].coordinates.components.size());
  return result;
}

std::string canonicalName(const Metadata & data, const std::string & name)
{
  const auto found = data.names.find(name);
  return found == data.names.end() ? name : found->second;
}

bool atOrBelow(const Watermarks & marks, const RawId & id)
{
  const auto found = marks.find(id.stream);
  return found != marks.end() && id.ordinal <= found->second;
}

void raise(Watermarks & marks, const RawId & id)
{
  const auto found = marks.find(id.stream);
  if (found == marks.end())
    marks.emplace(id.stream, id.ordinal);
  else
    found->second = std::max(found->second, id.ordinal);
}

Status bindRaw(Metadata & data, const std::vector<std::string> & names, const std::vector<TypedRaw> & bindings)
{
  for (const auto & name : names) {
    const auto incoming =
      std::find_if(bindings.begin(), bindings.end(), [&](const TypedRaw & item) { return item.raw == name; });
    const auto old =
      std::find_if(data.bindings.begin(), data.bindings.end(), [&](const TypedRaw & item) { return item.raw == name; });
    if (old != data.bindings.end() && (incoming == bindings.end() || incoming->identity != old->identity))
      return Status::kIdentityConflict;
    if (incoming == bindings.end())
      continue;
    for (const auto & binding : data.bindings)
      if (binding.identity == incoming->identity && binding.raw != name)
        return Status::kIdentityConflict;
    if (old == data.bindings.end()) {
      if (data.consumedRaw.count(name) != 0)
        return Status::kIdentityConflict;  // An opaque past cannot be recast as typed.
      data.bindings.push_back(*incoming);
    }
  }
  return Status::kOk;
}

Eigen::MatrixXd transport(const Operation & operation, const Eigen::MatrixXd & old)
{
  const auto n = operation.active.cols();
  if (operation.kind == OperationKind::kRetire)
    return selectRows(old, operation.sourceRows);
  if (operation.kind == OperationKind::kClone) {
    Eigen::MatrixXd result(old.rows() + operation.active.rows(), old.cols());
    result.topRows(old.rows()) = old;
    result.bottomRows(operation.active.rows()).noalias() = operation.active * old.topRows(n);
    return result;
  }
  Eigen::MatrixXd result = old;
  result.topRows(n).noalias() = operation.active * old.topRows(n);
  if (operation.source.cols() != 0)
    result.topRows(n).noalias() += operation.source * selectRows(old.bottomRows(old.rows() - n), operation.sourceRows);
  return result;
}

Eigen::MatrixXd copyCross(const Metadata & data, const NoiseModel & noise, const FrameCopy & copy, Eigen::Index n)
{
  const auto d = noise.covariance.rows();
  Eigen::Index initialSources = sourceDof(data.sources);
  // A copy is born immediately before its first update. That update's source
  // row indices need not cover all sources; use the history's stored joint size.
  if (copy.birth < data.history.size())
    initialSources = data.history[copy.birth]->oldSourceDof;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n + initialSources, d);
  const Eigen::MatrixXd covariance = static_cast<double>(noise.use.totalUses) * noise.covariance;
  for (std::size_t i = copy.birth; i < data.history.size(); ++i) {
    const auto & operation = *data.history[i];
    result = transport(operation, result);
    for (const auto & injection : operation.copies) {
      if (injection.raw != noise.use.raw || injection.frame != copy.frame)
        continue;
      if (operation.kind == OperationKind::kProcess)
        result.topRows(n) += injection.loading;
      else
        result.topRows(n).noalias() += injection.loading * covariance;
    }
  }
  return result;
}

Eigen::MatrixXd sensitivityJoint(const Metadata & data, const SensitivityModel & model)
{
  Eigen::MatrixXd result = model.initial;
  for (std::size_t i = model.birth; i < data.history.size(); ++i) {
    const auto & operation = *data.history[i];
    result = transport(operation, result);
    for (const auto & input : operation.sensitivities)
      if (input.first == model.source.key)
        result.topRows(input.second.rows()) += input.second;
  }
  return result;
}

}  // namespace

struct ConsiderSourceFilter::Prepared
{
  Reservation reservation;
  std::shared_ptr<const std::uint64_t> prior;
  std::shared_ptr<const std::uint64_t> next;
  Eigen::MatrixXd p;
  Eigen::MatrixXd k;
  Metadata metadata;
  std::size_t resident = 0;
};

struct ConsiderSourceFilter::Impl
{
  Budget budget;
  Metadata metadata;
  Eigen::MatrixXd p;
  Eigen::MatrixXd k;
  Eigen::MatrixXd c;
  std::shared_ptr<const std::uint64_t> version;
  std::shared_ptr<Account> account = std::make_shared<Account>();

  std::size_t residentBytes() const
  {
    return sumBytes(
      sizeof(Impl),
      sumBytes(metadataBytes(metadata), sumBytes(matrixBytes(p), sumBytes(matrixBytes(k), matrixBytes(c)))));
  }

  std::size_t workspace(std::size_t extra = 0) const
  {
    const auto n = static_cast<std::size_t>(p.rows());
    const auto s = static_cast<std::size_t>(k.cols());
    // P/K candidates, sampled audit, operation replay, and simultaneous GEMM
    // temporaries. No C-sized term is reserved by an actual-gain proposal.
    const auto sampled = static_cast<std::size_t>(kMaxSourceDof);
    const auto terms = productBytes(8, productBytes(n + sampled, n + std::max(s, sampled)));
    return sumBytes(metadataBytes(metadata), sumBytes(extra, productBytes(sizeof(double), terms)));
  }

  Status audit(const Eigen::MatrixXd & active, const Eigen::MatrixXd & cross, const Eigen::MatrixXd & sources) const
  {
    if (const auto status = detail::auditCovariance(active); status != Status::kOk)
      return status;
    const auto n = active.rows();
    const auto count = std::min(sources.rows(), kMaxSourceDof);
    if (count == 0)
      return Status::kOk;
    // First and last source principal windows, at most 23+32 each. This is an
    // arithmetic audit of a constructive recursion, not a global PSD proof.
    for (const auto start : {Eigen::Index{0}, sources.rows() - count}) {
      Eigen::MatrixXd sample(n + count, n + count);
      sample.topLeftCorner(n, n) = active;
      sample.topRightCorner(n, count) = cross.middleCols(start, count);
      sample.bottomLeftCorner(count, n) = sample.topRightCorner(n, count).transpose();
      sample.bottomRightCorner(count, count) = sources.block(start, start, count, count);
      if (const auto status = detail::auditCovariance(sample); status != Status::kOk)
        return status;
    }
    return Status::kOk;
  }

  Status publish(Eigen::MatrixXd & active, Eigen::MatrixXd & cross, Eigen::MatrixXd & sources, Metadata & data)
  {
    const auto size = sumBytes(
      sizeof(Impl),
      sumBytes(metadataBytes(data), sumBytes(matrixBytes(active), sumBytes(matrixBytes(cross), matrixBytes(sources)))));
    if (size > budget.memoryBytes)
      return Status::kModelCapacityExceeded;
    if (const auto status = audit(active, cross, sources); status != Status::kOk)
      return status;
    auto next = std::make_shared<const std::uint64_t>(*version + 1);
    p.swap(active);
    k.swap(cross);
    if (&c != &sources)
      c.swap(sources);
    std::swap(metadata, data);
    version = std::move(next);
    account->resident = size;
    return Status::kOk;
  }
};

ConsiderSourceFilter::ConsiderSourceFilter() : impl_(std::make_unique<Impl>()) {}
ConsiderSourceFilter::~ConsiderSourceFilter() = default;
ConsiderSourceFilter::Candidate::Candidate() = default;
ConsiderSourceFilter::Candidate::~Candidate() = default;
ConsiderSourceFilter::Candidate::Candidate(Candidate &&) noexcept = default;
ConsiderSourceFilter::Candidate & ConsiderSourceFilter::Candidate::operator=(Candidate &&) noexcept = default;

const Eigen::MatrixXd * ConsiderSourceFilter::Candidate::activeCovariance() const
{
  return data_ && status_ == Status::kOk ? &data_->p : nullptr;
}

Status ConsiderSourceFilter::initialize(const JointMoments & active, const Budget & budget)
{
  try {
    if (impl_->version)
      return Status::kIdentityConflict;
    if (!active.sources.empty() || !validCoordinates(active.active))
      return Status::kDimensionMismatch;
    const auto n = static_cast<Eigen::Index>(active.active.components.size());
    if (n > kMaxActiveDof || budget.rowsPerBatch == 0 || budget.recentLedger == 0 || budget.memoryBytes == 0 ||
        budget.sourceDof > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        budget.rowsPerBatch > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()))
      return Status::kModelCapacityExceeded;
    if ((active.knowledge != MomentKnowledge::kKnown && active.knowledge != MomentKnowledge::kUpperBound) ||
        !validKey(active.evidence))
      return Status::kUnknownMoment;
    if (active.mean.size() != n || active.covariance.rows() != n || active.covariance.cols() != n)
      return Status::kDimensionMismatch;
    if (!active.mean.allFinite())
      return Status::kNonFinite;
    if (const auto status = detail::auditCovariance(active.covariance); status != Status::kOk)
      return status;
    auto prepared = std::make_unique<Impl>();
    prepared->budget = budget;
    prepared->p = active.covariance;
    prepared->k.resize(n, 0);
    prepared->c.resize(0, 0);
    prepared->metadata.active = active.active;
    prepared->metadata.knowledge = active.knowledge;
    prepared->version = std::make_shared<const std::uint64_t>(1);
    prepared->account->limit = budget.memoryBytes;
    prepared->account->resident = prepared->residentBytes();
    if (prepared->account->resident > budget.memoryBytes)
      return Status::kModelCapacityExceeded;
    impl_.swap(prepared);
    return Status::kOk;
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

namespace
{

void remember(Metadata & data, const std::shared_ptr<const Operation> & operation)
{
  if (!data.noises.empty() || !data.sensitivities.empty())
    data.history.push_back(operation);
  else
    data.history.clear();
}

Status validSource(const Source & source)
{
  // A single block still uses the existing coordinate/identity contract. Only
  // the new class's cumulative live block and DOF budgets are different.
  return validateSources({source});
}

Status considerIdentity(const FactorIdentity & identity)
{
  // Keep the v1 opaque-entry and byte limits, without borrowing its 128-item
  // per-factor lists as an undocumented cap on the new batch budget.
  if (identity.payload.size() > kMaxFactorPayloadBytes)
    return Status::kModelCapacityExceeded;
  if (!validKey(identity.key) || identity.payload.empty() || !validKey(identity.remainderKey) ||
      !validKey(identity.independenceEvidence) || identity.rawAncestry.empty() || !uniqueKeys(identity.rawAncestry) ||
      !validTypedRaw(identity.rawAncestry, identity.typedRaw))
    return Status::kInvalidIdentity;
  for (std::size_t i = 0; i < identity.shared.size(); ++i) {
    const auto & shared = identity.shared[i];
    if (!contains(identity.rawAncestry, shared.raw) || !validKey(shared.source) ||
        std::find(identity.shared.begin(), identity.shared.begin() + static_cast<std::ptrdiff_t>(i), shared) !=
          identity.shared.begin() + static_cast<std::ptrdiff_t>(i))
      return Status::kInvalidIdentity;
  }
  for (std::size_t i = 0; i < identity.reuse.size(); ++i) {
    const auto & use = identity.reuse[i];
    if (use.totalUses == 0 || !validKey(use.owner.key) ||
        std::none_of(identity.typedRaw.begin(), identity.typedRaw.end(), [&](const TypedRaw & raw) {
          return raw.identity == use.raw;
        }))
      return Status::kInvalidIdentity;
    for (std::size_t j = 0; j < i; ++j)
      if (use.raw == identity.reuse[j].raw && use.frame == identity.reuse[j].frame)
        return Status::kInvalidIdentity;
  }
  return Status::kOk;
}

Status sourceHistory(Metadata & data, const Source & source)
{
  if (std::any_of(data.sensitivityLabels.begin(), data.sensitivityLabels.end(), [&](const Source & label) {
        return label.key == source.key;
      }))
    return Status::kIdentityConflict;
  if (source.sequence && atOrBelow(data.sourceHigh, *source.sequence))
    return Status::kSourceHistoryUnavailable;
  for (const auto & raw : source.rawAncestry)
    if (data.retiredDesignatedRaw.count(raw) != 0)
      return Status::kSourceHistoryUnavailable;
  std::vector<std::string> names = source.aliases;
  names.push_back(source.key);
  for (const auto & name : names) {
    if (data.names.count(name) != 0)
      return data.retiredNames.count(canonicalName(data, name)) != 0 ? Status::kSourceHistoryUnavailable
                                                                     : Status::kIdentityConflict;
  }
  if (const auto status = bindRaw(data, source.rawAncestry, source.typedRaw); status != Status::kOk)
    return status;
  for (const auto & noise : data.noises) {
    if (noise.use.owner.key != source.key)
      continue;
    if (noise.use.owner.sequence != source.sequence)
      return Status::kIdentityConflict;
    const auto binding = std::find_if(source.typedRaw.begin(), source.typedRaw.end(), [&](const TypedRaw & raw) {
      return raw.identity == noise.use.raw;
    });
    if (binding == source.typedRaw.end() || !designatesRaw(source, binding->raw))
      return Status::kUnsupportedNoiseStructure;
  }
  for (const auto & name : names)
    data.names.emplace(name, source.key);
  if (source.sequence)
    raise(data.sourceHigh, *source.sequence);
  return Status::kOk;
}

Status canonicalFactor(const Metadata & data, const LinearFactor & factor, FactorIdentity & identity)
{
  identity = factor.identity;
  if (const auto status = considerIdentity(identity); status != Status::kOk)
    return status;
  for (auto & shared : identity.shared)
    shared.source = canonicalName(data, shared.source);
  for (auto & use : identity.reuse)
    use.owner.key = canonicalName(data, use.owner.key);
  std::string bytes;
  appendCanonicalText(bytes, identity.payload);
  appendCanonicalText(bytes, factor.model);
  appendCanonicalText(bytes, factor.linearization);
  appendCanonicalCoordinates(bytes, factor.active);
  appendCanonicalText(bytes, std::to_string(factor.sourceKeys.size()));
  for (std::size_t i = 0; i < factor.sourceKeys.size(); ++i) {
    appendCanonicalText(bytes, canonicalName(data, factor.sourceKeys[i]));
    appendCanonicalCoordinates(bytes, factor.sourceCoordinates[i]);
  }
  for (const auto & unit : factor.rowUnits)
    appendCanonicalText(bytes, unit);
  appendCanonicalMatrix(bytes, factor.h);
  appendCanonicalMatrix(bytes, factor.a);
  appendCanonicalMatrix(bytes, factor.observation);
  appendCanonicalMatrix(bytes, factor.remainderMean);
  appendCanonicalMatrix(bytes, factor.remainderVariance);
  identity.payload = std::move(bytes);
  return considerIdentity(identity);
}

Status factorFormat(const LinearFactor & input, const Budget & budget)
{
  const auto rows = input.h.rows();
  if (static_cast<std::size_t>(rows) > budget.rowsPerBatch || input.sourceKeys.size() > budget.sourcesPerBatch)
    return Status::kModelCapacityExceeded;
  if (input.domain != ModelDomain::kLinearFixedModel)
    return Status::kUnsupportedModel;
  if (input.remainderKnowledge != MomentKnowledge::kKnown)
    return Status::kUnknownMoment;
  if (!input.independentRemainder || (input.remainderVariance.array() <= 0.0).any())
    return Status::kUnsupportedNoiseStructure;
  if (!validCoordinates(input.active) || input.active.components.size() > static_cast<std::size_t>(kMaxActiveDof) ||
      !validKey(input.model) || !validKey(input.linearization) || !uniqueKeys(input.sourceKeys))
    return Status::kInvalidIdentity;
  if (input.sourceCoordinates.size() != input.sourceKeys.size())
    return Status::kDimensionMismatch;
  Eigen::Index columns = 0;
  for (const auto & coordinates : input.sourceCoordinates) {
    if (!validCoordinates(coordinates))
      return Status::kInvalidIdentity;
    columns += static_cast<Eigen::Index>(coordinates.components.size());
  }
  if (rows <= 0 || input.h.cols() != static_cast<Eigen::Index>(input.active.components.size()) ||
      input.a.rows() != rows || input.a.cols() != columns || input.observation.size() != rows ||
      input.remainderMean.size() != rows || input.remainderVariance.size() != rows ||
      input.rowUnits.size() != static_cast<std::size_t>(rows))
    return Status::kDimensionMismatch;
  if (!input.h.allFinite() || !input.a.allFinite() || !input.observation.allFinite() ||
      !input.remainderMean.allFinite() || !input.remainderVariance.allFinite())
    return Status::kNonFinite;
  if (!std::all_of(input.rowUnits.begin(), input.rowUnits.end(), validKey))
    return Status::kInvalidIdentity;
  return Status::kOk;
}

Status appendSideInputs(const Metadata & data,
                        const LinearFactor & input,
                        const std::vector<ReuseNoise> & reuse,
                        const std::vector<SensitivityLoading> & sensitivities,
                        FactorIdentity & identity)
{
  for (const auto & noise : reuse) {
    for (const auto & loading : noise.factors) {
      if (loading.factor != input.identity.key)
        continue;
      auto use = noise.use;
      use.owner.key = canonicalName(data, use.owner.key);
      if (std::find(identity.reuse.begin(), identity.reuse.end(), use) == identity.reuse.end())
        return Status::kIdentityConflict;
      appendRaw(identity.payload, use.raw);
      appendRaw(identity.payload, use.frame);
      appendCanonicalCoordinates(identity.payload, noise.coordinates);
      appendCanonicalText(identity.payload, noise.evidence);
      appendCanonicalText(identity.payload, noise.independentAtBirth ? "independent" : "unknown");
      appendCanonicalText(identity.payload, std::to_string(static_cast<int>(noise.knowledge)));
      appendCanonicalMatrix(identity.payload, noise.covariance);
      appendCanonicalMatrix(identity.payload, loading.loading);
    }
  }
  for (const auto & sensitivity : sensitivities) {
    if (sensitivity.factor != input.identity.key)
      continue;
    appendCanonicalText(identity.payload, sensitivity.sensitivity);
    appendCanonicalMatrix(identity.payload, sensitivity.loading);
  }
  return considerIdentity(identity);
}

Status validateBinding(const Metadata & data, const LinearFactor & input, const FactorIdentity & identity)
{
  if (!coordinatesEqual(data.active, input.active))
    return Status::kIdentityConflict;
  std::set<Eigen::Index> blocks;
  for (std::size_t i = 0; i < input.sourceKeys.size(); ++i) {
    const auto index = resolve(data, input.sourceKeys[i]);
    if (index < 0)
      return Status::kSourceHistoryUnavailable;
    if (!blocks.insert(index).second ||
        !coordinatesEqual(input.sourceCoordinates[i], data.sources[static_cast<std::size_t>(index)].coordinates))
      return Status::kIdentityConflict;
  }
  for (const auto & shared : identity.shared) {
    const auto index = resolve(data, shared.source);
    if (index < 0 || !contains(data.sources[static_cast<std::size_t>(index)].rawAncestry, shared.raw))
      return Status::kSourceHistoryUnavailable;
    Eigen::Index col = 0;
    bool covered = false;
    for (std::size_t i = 0; i < input.sourceKeys.size(); ++i) {
      const auto count = static_cast<Eigen::Index>(input.sourceCoordinates[i].components.size());
      if (resolve(data, input.sourceKeys[i]) == index && !input.a.middleCols(col, count).isZero(0.0))
        covered = true;
      col += count;
    }
    if (!covered)
      return Status::kUnsupportedNoiseStructure;
  }
  return Status::kOk;
}

Status ancestry(const Metadata & data, const FactorIdentity & identity, const Watermarks & priorHigh)
{
  for (const auto & raw : identity.rawAncestry) {
    if (data.retiredDesignatedRaw.count(raw) != 0)
      return Status::kSourceHistoryUnavailable;
    bool routed = false;
    for (const auto & source : data.sources) {
      if (!designatesRaw(source, raw))
        continue;
      routed = true;
      if (std::find(identity.shared.begin(), identity.shared.end(), SharedRawSource{raw, source.key}) ==
          identity.shared.end())
        return Status::kUnsupportedNoiseStructure;
    }
    const auto typed = std::find_if(
      identity.typedRaw.begin(), identity.typedRaw.end(), [&](const TypedRaw & binding) { return binding.raw == raw; });
    const bool reused = typed != identity.typedRaw.end() &&
                        std::any_of(identity.reuse.begin(),
                                    identity.reuse.end(),
                                    [&](const ReuseUse & use) { return use.raw == typed->identity; }) &&
                        std::any_of(data.noises.begin(), data.noises.end(), [&](const NoiseModel & model) {
                          return model.use.raw == typed->identity;
                        });
    if (!routed && !reused) {
      if (typed != identity.typedRaw.end() && atOrBelow(priorHigh, typed->identity))
        return Status::kLedgerCompacted;
      if (data.consumedRaw.count(raw) != 0)
        return Status::kUnsupportedNoiseStructure;
    }
  }
  return Status::kOk;
}

Status compactRecent(Metadata & data, const Budget & budget)
{
  while (data.recent.size() > budget.recentLedger) {
    const auto old = std::find_if(
      data.recent.begin(), data.recent.end(), [](const FactorIdentity & item) { return item.sequence.has_value(); });
    if (old == data.recent.end() || old == std::prev(data.recent.end()))
      return Status::kModelCapacityExceeded;
    raise(data.compacted, *old->sequence);
    data.recent.erase(old);
  }
  return Status::kOk;
}

Status appendLedger(Metadata & data, const FactorIdentity & identity)
{
  if (data.remainders.count(identity.remainderKey) != 0)
    return Status::kUnsupportedNoiseStructure;
  if (!identity.sequence &&
      static_cast<std::size_t>(std::count_if(data.recent.begin(), data.recent.end(), [](const FactorIdentity & item) {
        return !item.sequence;
      })) >= kMaxLedger)
    return Status::kModelCapacityExceeded;
  data.recent.push_back(identity);
  data.remainders.insert(identity.remainderKey);
  for (const auto & raw : identity.rawAncestry)
    data.consumedRaw.insert(raw);
  for (const auto & raw : identity.typedRaw)
    raise(data.rawHigh, raw.identity);
  if (identity.sequence) {
    data.factorBindings.emplace(identity.key, *identity.sequence);
    raise(data.factorHigh, *identity.sequence);
  }
  return Status::kOk;
}

const ReuseNoise * findNoise(const std::vector<ReuseNoise> & noises, const ReuseUse & use, const Metadata & data)
{
  for (const auto & noise : noises) {
    auto canonical = noise.use;
    canonical.owner.key = canonicalName(data, canonical.owner.key);
    if (canonical == use)
      return &noise;
  }
  return nullptr;
}

Status sideFormats(const std::vector<LinearFactor> & factors,
                   const std::vector<ReuseNoise> & noises,
                   const std::vector<SensitivityLoading> & sensitivities)
{
  std::set<std::pair<RawId, RawId>, bool (*)(const std::pair<RawId, RawId> &, const std::pair<RawId, RawId> &)> copies(
    [](const auto & a, const auto & b) {
      return RawLess{}(a.first, b.first) || (a.first == b.first && RawLess{}(a.second, b.second));
    });
  for (const auto & noise : noises) {
    if (!validKey(noise.use.owner.key) || noise.use.totalUses == 0 ||
        !copies.insert({noise.use.raw, noise.use.frame}).second || !validCoordinates(noise.coordinates) ||
        !validKey(noise.evidence) || noise.factors.empty())
      return Status::kInvalidIdentity;
    if (noise.knowledge != MomentKnowledge::kKnown)
      return Status::kUnknownMoment;
    if (!noise.independentAtBirth)
      return Status::kUnsupportedNoiseStructure;
    if (noise.covariance.rows() != static_cast<Eigen::Index>(noise.coordinates.components.size()) ||
        noise.covariance.cols() != noise.covariance.rows())
      return Status::kDimensionMismatch;
    if (const auto status = detail::auditCovariance(noise.covariance); status != Status::kOk)
      return status;
    std::set<std::string> keys;
    for (const auto & loading : noise.factors) {
      const auto factor = std::find_if(factors.begin(), factors.end(), [&](const LinearFactor & value) {
        return value.identity.key == loading.factor;
      });
      if (factor == factors.end() || !keys.insert(loading.factor).second)
        return Status::kInvalidIdentity;
      if (loading.loading.rows() != factor->h.rows() || loading.loading.cols() != noise.covariance.rows())
        return Status::kDimensionMismatch;
      if (!loading.loading.allFinite())
        return Status::kNonFinite;
    }
  }
  std::set<std::pair<std::string, std::string>> bindings;
  for (const auto & sensitivity : sensitivities) {
    const auto factor = std::find_if(factors.begin(), factors.end(), [&](const LinearFactor & value) {
      return value.identity.key == sensitivity.factor;
    });
    if (factor == factors.end() || !bindings.insert({sensitivity.factor, sensitivity.sensitivity}).second ||
        !validKey(sensitivity.sensitivity))
      return Status::kInvalidIdentity;
    if (sensitivity.loading.rows() != factor->h.rows())
      return Status::kDimensionMismatch;
    if (!sensitivity.loading.allFinite())
      return Status::kNonFinite;
  }
  return Status::kOk;
}

Status useNoise(Metadata & data, const FactorIdentity & identity, const std::vector<ReuseNoise> & noises)
{
  constexpr std::uint64_t kLargestExactInteger = std::uint64_t{1} << 53U;
  for (const auto & use : identity.reuse) {
    const auto * input = findNoise(noises, use, data);
    if (!input || std::none_of(input->factors.begin(), input->factors.end(), [&](const FactorLoading & loading) {
          return loading.factor == identity.key;
        }))
      return Status::kUnknownMoment;
    if (use.totalUses > kLargestExactInteger)
      return Status::kModelCapacityExceeded;
    auto model = std::find_if(
      data.noises.begin(), data.noises.end(), [&](const NoiseModel & old) { return old.use.raw == use.raw; });
    if (model == data.noises.end()) {
      const auto raw = std::find_if(identity.typedRaw.begin(), identity.typedRaw.end(), [&](const TypedRaw & value) {
        return value.identity == use.raw;
      });
      if (raw == identity.typedRaw.end() || data.consumedRaw.count(raw->raw) != 0)
        return Status::kSourceHistoryUnavailable;  // No late registration.
      const auto owner = resolve(data, use.owner.key);
      if (owner < 0 && (data.retiredNames.count(use.owner.key) != 0 ||
                        (use.owner.sequence && atOrBelow(data.sourceHigh, *use.owner.sequence))))
        return Status::kSourceHistoryUnavailable;
      if (owner >= 0) {
        const auto & source = data.sources[static_cast<std::size_t>(owner)];
        if (source.sequence != use.owner.sequence || !designatesRaw(source, raw->raw))
          return Status::kIdentityConflict;
      }
      data.noises.push_back({use, input->coordinates, input->evidence, input->covariance, {}});
      model = std::prev(data.noises.end());
    } else if (model->use.totalUses != use.totalUses || !(model->use.owner == use.owner) ||
               !coordinatesEqual(model->coordinates, input->coordinates) || model->evidence != input->evidence ||
               model->covariance.rows() != input->covariance.rows() ||
               !(model->covariance.array() == input->covariance.array()).all()) {
      return Status::kIdentityConflict;
    }
    if (std::none_of(model->frames.begin(), model->frames.end(), [&](const FrameCopy & copy) {
          return copy.frame == use.frame;
        })) {
      if (model->frames.size() >= use.totalUses)
        return Status::kReuseBudgetExceeded;
      model->frames.push_back({use.frame, data.history.size()});
    }
    data.knowledge = MomentKnowledge::kUpperBound;
  }
  return Status::kOk;
}

Status sensitivityBindings(const Metadata & data,
                           const FactorIdentity & identity,
                           const std::vector<SensitivityLoading> & sensitivities)
{
  for (const auto & model : data.sensitivities) {
    const auto binding =
      std::find_if(sensitivities.begin(), sensitivities.end(), [&](const SensitivityLoading & value) {
        return value.factor == identity.key && value.sensitivity == model.source.key;
      });
    if (binding == sensitivities.end())
      return Status::kUnknownMoment;
    if (binding->loading.cols() != static_cast<Eigen::Index>(model.source.coordinates.components.size()))
      return Status::kDimensionMismatch;
  }
  for (const auto & binding : sensitivities) {
    if (binding.factor == identity.key &&
        std::none_of(data.sensitivities.begin(), data.sensitivities.end(), [&](const SensitivityModel & value) {
          return value.source.key == binding.sensitivity;
        }))
      return Status::kSourceHistoryUnavailable;
  }
  return Status::kOk;
}

struct Batch
{
  std::vector<std::size_t> fresh;
  std::vector<Eigen::Index> inputRows;
  std::vector<Eigen::Index> selectedRows;
  std::vector<Eigen::Index> sourceRows;
  Eigen::MatrixXd h;
  Eigen::MatrixXd a;
  Eigen::MatrixXd gain;
  Eigen::VectorXd variance;
};

Status consume(Metadata & data,
               const Budget & budget,
               const std::vector<LinearFactor> & factors,
               const std::vector<ReuseNoise> & noises,
               const std::vector<SensitivityLoading> & sensitivities,
               Batch & batch)
{
  const auto priorHigh = data.rawHigh;
  Eigen::Index row = 0;
  Eigen::Index selected = 0;
  for (std::size_t i = 0; i < factors.size(); ++i) {
    const auto & input = factors[i];
    batch.inputRows.push_back(row);
    batch.selectedRows.push_back(selected);
    row += input.h.rows();
    FactorIdentity identity;
    if (const auto status = canonicalFactor(data, input, identity); status != Status::kOk)
      return status;
    if (const auto status = appendSideInputs(data, input, noises, sensitivities, identity); status != Status::kOk)
      return status;
    const auto old = std::find_if(
      data.recent.begin(), data.recent.end(), [&](const FactorIdentity & value) { return value.key == identity.key; });
    if (old != data.recent.end()) {
      if (!(*old == identity))
        return Status::kIdentityConflict;
      continue;
    }
    if (identity.sequence && atOrBelow(data.compacted, *identity.sequence))
      return Status::kLedgerCompacted;
    const auto binding = data.factorBindings.find(identity.key);
    if (binding != data.factorBindings.end())
      return identity.sequence == std::optional<RawId>{binding->second} ? Status::kLedgerCompacted
                                                                        : Status::kIdentityConflict;
    if (identity.sequence) {
      const auto high = data.factorHigh.find(identity.sequence->stream);
      if (high != data.factorHigh.end() && identity.sequence->ordinal < high->second)
        return Status::kLedgerCompacted;
    }
    for (const auto & entry : data.recent)
      if (identity.sequence && entry.sequence == identity.sequence)
        return Status::kIdentityConflict;
    if (const auto status = validateBinding(data, input, identity); status != Status::kOk)
      return status;
    if (const auto status = bindRaw(data, identity.rawAncestry, identity.typedRaw); status != Status::kOk)
      return status;
    if (const auto status = ancestry(data, identity, priorHigh); status != Status::kOk)
      return status;
    if (const auto status = useNoise(data, identity, noises); status != Status::kOk)
      return status;
    if (const auto status = sensitivityBindings(data, identity, sensitivities); status != Status::kOk)
      return status;
    if (const auto status = appendLedger(data, identity); status != Status::kOk)
      return status;
    selected += input.h.rows();
    batch.fresh.push_back(i);
  }
  return batch.fresh.empty() ? Status::kNoOp : compactRecent(data, budget);
}

Status assemble(const Metadata & data,
                const Budget & budget,
                const std::vector<LinearFactor> & factors,
                const Eigen::MatrixXd & gain,
                Batch & batch)
{
  std::set<Eigen::Index> sourceBlocks;
  std::set<Eigen::Index> sourceRows;
  Eigen::Index rows = 0;
  for (const auto index : batch.fresh) {
    const auto & input = factors[index];
    rows += input.h.rows();
    Eigen::Index col = 0;
    for (std::size_t j = 0; j < input.sourceKeys.size(); ++j) {
      const auto block = resolve(data, input.sourceKeys[j]);
      sourceBlocks.insert(block);
      const auto count = static_cast<Eigen::Index>(input.sourceCoordinates[j].components.size());
      if (!input.a.middleCols(col, count).isZero(0.0))
        for (Eigen::Index k = 0; k < count; ++k)
          sourceRows.insert(offset(data, block) + k);
      col += count;
    }
  }
  if (sourceBlocks.size() > budget.sourcesPerBatch)
    return Status::kModelCapacityExceeded;
  batch.sourceRows.assign(sourceRows.begin(), sourceRows.end());
  batch.h.resize(rows, gain.rows());
  batch.a = Eigen::MatrixXd::Zero(rows, static_cast<Eigen::Index>(sourceRows.size()));
  batch.gain.resize(gain.rows(), rows);
  batch.variance.resize(rows);
  Eigen::Index row = 0;
  for (const auto index : batch.fresh) {
    const auto & input = factors[index];
    batch.h.middleRows(row, input.h.rows()) = input.h;
    batch.gain.middleCols(row, input.h.rows()) = gain.middleCols(batch.inputRows[index], input.h.rows());
    batch.variance.segment(row, input.h.rows()) = input.remainderVariance;
    Eigen::Index col = 0;
    for (std::size_t j = 0; j < input.sourceKeys.size(); ++j) {
      const auto first = offset(data, resolve(data, input.sourceKeys[j]));
      const auto count = static_cast<Eigen::Index>(input.sourceCoordinates[j].components.size());
      for (Eigen::Index k = 0; k < count; ++k) {
        const auto found = std::lower_bound(batch.sourceRows.begin(), batch.sourceRows.end(), first + k);
        if (found != batch.sourceRows.end() && *found == first + k)
          batch.a.block(row, static_cast<Eigen::Index>(found - batch.sourceRows.begin()), input.h.rows(), 1) =
            input.a.col(col + k);
      }
      col += count;
    }
    row += input.h.rows();
  }
  return Status::kOk;
}

std::shared_ptr<Operation> updateOperation(const Metadata & data,
                                           const std::vector<LinearFactor> & factors,
                                           const Batch & batch,
                                           const std::vector<ReuseNoise> & noises,
                                           const std::vector<SensitivityLoading> & sensitivities)
{
  auto result = std::make_shared<Operation>();
  const auto n = batch.gain.rows();
  result->oldSourceDof = sourceDof(data.sources);
  result->active = Eigen::MatrixXd::Identity(n, n) - batch.gain * batch.h;
  result->source = -batch.gain * batch.a;
  result->sourceRows = batch.sourceRows;
  for (const auto & noise : noises) {
    Eigen::MatrixXd loading = Eigen::MatrixXd::Zero(batch.h.rows(), noise.covariance.rows());
    bool used = false;
    for (const auto index : batch.fresh)
      for (const auto & input : noise.factors)
        if (input.factor == factors[index].identity.key) {
          loading.middleRows(batch.selectedRows[index], input.loading.rows()) = input.loading;
          used = true;
        }
    if (used)
      result->copies.push_back({noise.use.raw, noise.use.frame, -batch.gain * loading});
  }
  for (const auto & model : data.sensitivities) {
    Eigen::MatrixXd loading =
      Eigen::MatrixXd::Zero(batch.h.rows(), static_cast<Eigen::Index>(model.source.coordinates.components.size()));
    for (const auto index : batch.fresh)
      for (const auto & input : sensitivities)
        if (input.factor == factors[index].identity.key && input.sensitivity == model.source.key)
          loading.middleRows(batch.selectedRows[index], input.loading.rows()) = input.loading;
    result->sensitivities.emplace_back(model.source.key, -batch.gain * loading);
  }
  return result;
}

void advanceCopies(const Metadata & data, const Operation & operation, Eigen::MatrixXd & p, Eigen::MatrixXd & k)
{
  const auto n = p.rows();
  for (const auto & input : operation.copies) {
    const auto model = std::find_if(
      data.noises.begin(), data.noises.end(), [&](const NoiseModel & noise) { return noise.use.raw == input.raw; });
    const auto copy = std::find_if(
      model->frames.begin(), model->frames.end(), [&](const FrameCopy & value) { return value.frame == input.frame; });
    const Eigen::MatrixXd x = copyCross(data, *model, *copy, n);
    Eigen::MatrixXd transported = operation.active * x.topRows(n);
    if (!operation.sourceRows.empty())
      transported.noalias() += operation.source * selectRows(x.bottomRows(k.cols()), operation.sourceRows);
    const Eigen::MatrixXd term = transported * input.loading.transpose();
    p += term + term.transpose();
    p.noalias() +=
      static_cast<double>(model->use.totalUses) * input.loading * model->covariance * input.loading.transpose();
    if (k.cols() != 0)
      k.noalias() += input.loading * x.bottomRows(k.cols()).transpose();
  }
}

std::size_t proposalBytes(const std::vector<LinearFactor> & factors,
                          const std::vector<ReuseNoise> & noises,
                          const std::vector<SensitivityLoading> & sensitivities)
{
  std::size_t result = 0;
  for (const auto & factor : factors) {
    result = sumBytes(result, identityStorageBytes(factor.identity));
    result =
      sumBytes(result, coordinatesBytes(factor.active) + textBytes(factor.model) + textBytes(factor.linearization));
    for (const auto & coordinates : factor.sourceCoordinates)
      result = sumBytes(result, coordinatesBytes(coordinates));
    for (const auto & key : factor.sourceKeys)
      result = sumBytes(result, textBytes(key));
    for (const auto & unit : factor.rowUnits)
      result = sumBytes(result, textBytes(unit));
    result = sumBytes(result, sumBytes(matrixBytes(factor.h), matrixBytes(factor.a)));
    result = sumBytes(result,
                      sumBytes(matrixBytes(factor.observation),
                               sumBytes(matrixBytes(factor.remainderMean), matrixBytes(factor.remainderVariance))));
  }
  for (const auto & noise : noises) {
    result = sumBytes(result,
                      matrixBytes(noise.covariance) + coordinatesBytes(noise.coordinates) + textBytes(noise.evidence) +
                        textBytes(noise.use.owner.key));
    for (const auto & loading : noise.factors)
      result = sumBytes(result, matrixBytes(loading.loading) + textBytes(loading.factor));
  }
  for (const auto & sensitivity : sensitivities)
    result = sumBytes(
      result, matrixBytes(sensitivity.loading) + textBytes(sensitivity.factor) + textBytes(sensitivity.sensitivity));
  return productBytes(8, result);
}

}  // namespace

ConsiderSourceFilter::Candidate
ConsiderSourceFilter::proposeAppliedGain(const std::vector<LinearFactor> & factors,
                                         const Eigen::MatrixXd & activeGain,
                                         const std::vector<ReuseNoise> & reuse,
                                         const std::vector<SensitivityLoading> & sensitivities) const
{
  Candidate result;
  if (!impl_->version)
    return result;
  try {
    auto prepared = std::make_unique<Prepared>();
    const auto prepare = [&]() -> Status {
      if (!prepared->reservation.acquire(impl_->account,
                                         impl_->workspace(proposalBytes(factors, reuse, sensitivities))))
        return Status::kModelCapacityExceeded;
      if (factors.empty() || factors.size() > impl_->budget.rowsPerBatch)
        return factors.empty() ? Status::kInvalidIdentity : Status::kModelCapacityExceeded;
      std::size_t rows = 0;
      for (const auto & factor : factors) {
        if (const auto status = factorFormat(factor, impl_->budget); status != Status::kOk)
          return status;
        rows = sumBytes(rows, static_cast<std::size_t>(factor.h.rows()));
      }
      if (rows > impl_->budget.rowsPerBatch)
        return Status::kModelCapacityExceeded;
      if (activeGain.rows() != impl_->p.rows() || activeGain.cols() != static_cast<Eigen::Index>(rows))
        return Status::kDimensionMismatch;
      if (!activeGain.allFinite())
        return Status::kNonFinite;
      if (const auto status = sideFormats(factors, reuse, sensitivities); status != Status::kOk)
        return status;
      prepared->prior = impl_->version;
      prepared->metadata = impl_->metadata;
      Batch batch;
      if (const auto status = consume(prepared->metadata, impl_->budget, factors, reuse, sensitivities, batch);
          status != Status::kOk)
        return status;
      if (const auto status = assemble(prepared->metadata, impl_->budget, factors, activeGain, batch);
          status != Status::kOk)
        return status;
      const auto operation = updateOperation(prepared->metadata, factors, batch, reuse, sensitivities);
      const auto & t = operation->active;
      const auto & v = operation->source;
      prepared->k.noalias() = t * impl_->k;
      prepared->p.noalias() = t * impl_->p * t.transpose();
      if (!batch.sourceRows.empty()) {
        // Accumulate V*C by retained rows, never materializing C_J,* or C_J,J.
        // In particular J may contain every source: update workspace still
        // scales as n*s, rather than silently copying the entire retained C.
        Eigen::MatrixXd vC = Eigen::MatrixXd::Zero(impl_->p.rows(), impl_->c.cols());
        for (std::size_t j = 0; j < batch.sourceRows.size(); ++j)
          vC.noalias() += v.col(static_cast<Eigen::Index>(j)) * impl_->c.row(batch.sourceRows[j]);
        const Eigen::MatrixXd selectedK = selectRows(impl_->k.transpose(), batch.sourceRows).transpose();
        const Eigen::MatrixXd selectedVC = selectRows(vC.transpose(), batch.sourceRows).transpose();
        prepared->k += vC;
        prepared->p.noalias() += selectedVC * v.transpose();
        const Eigen::MatrixXd term = t * selectedK * v.transpose();
        prepared->p += term + term.transpose();
      }
      prepared->p.noalias() += batch.gain * batch.variance.asDiagonal() * batch.gain.transpose();
      advanceCopies(prepared->metadata, *operation, prepared->p, prepared->k);
      mirrorLower(prepared->p);
      if (const auto status = impl_->audit(prepared->p, prepared->k, impl_->c); status != Status::kOk)
        return status;
      remember(prepared->metadata, operation);
      prepared->resident = sumBytes(
        sizeof(Impl),
        sumBytes(metadataBytes(prepared->metadata),
                 sumBytes(matrixBytes(prepared->p), sumBytes(matrixBytes(prepared->k), matrixBytes(impl_->c)))));
      if (prepared->resident > impl_->budget.memoryBytes ||
          impl_->account->pending > impl_->budget.memoryBytes - prepared->resident)
        return Status::kModelCapacityExceeded;
      prepared->next = std::make_shared<const std::uint64_t>(*impl_->version + 1);
      return Status::kOk;
    };
    result.status_ = prepare();
    if (result.status_ == Status::kNoOp) {
      prepared->metadata = {};
      prepared->reservation.release();
      if (!prepared->reservation.acquire(impl_->account, productBytes(2, sizeof(Prepared))))
        result.status_ = Status::kModelCapacityExceeded;
    }
    if (result.status_ == Status::kOk || result.status_ == Status::kNoOp)
      result.data_ = std::move(prepared);
  } catch (const std::bad_alloc &) {
    result.status_ = Status::kModelCapacityExceeded;
  }
  return result;
}

Status ConsiderSourceFilter::commit(Candidate && candidate)
{
  if (candidate.status_ != Status::kOk && candidate.status_ != Status::kNoOp)
    return candidate.status_;
  if (!candidate.data_ || candidate.data_->prior != impl_->version)
    return Status::kStaleProposal;
  if (candidate.status_ == Status::kNoOp)
    return Status::kNoOp;
  auto & prepared = *candidate.data_;
  impl_->p.swap(prepared.p);
  impl_->k.swap(prepared.k);
  std::swap(impl_->metadata, prepared.metadata);
  impl_->version = std::move(prepared.next);
  prepared.reservation.release();
  impl_->account->resident = prepared.resident;
  candidate.data_.reset();
  candidate.status_ = Status::kStaleProposal;
  return Status::kOk;
}

Status ConsiderSourceFilter::augment(const Source & source, const Eigen::MatrixXd & activeGenerator)
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    if (const auto status = validSource(source); status != Status::kOk)
      return status;
    const auto n = impl_->p.rows();
    const auto s = impl_->c.rows();
    const auto d = static_cast<Eigen::Index>(source.coordinates.components.size());
    if (impl_->metadata.sources.size() >= impl_->budget.liveSources ||
        sumBytes(static_cast<std::size_t>(s), static_cast<std::size_t>(d)) > impl_->budget.sourceDof)
      return Status::kModelCapacityExceeded;
    if (activeGenerator.rows() != d || activeGenerator.cols() != n)
      return Status::kDimensionMismatch;
    if (!activeGenerator.allFinite())
      return Status::kNonFinite;
    Reservation reservation;
    const auto size =
      productBytes(sizeof(double), productBytes(static_cast<std::size_t>(s + d), static_cast<std::size_t>(s + d)));
    if (!reservation.acquire(impl_->account, impl_->workspace(sumBytes(size, productBytes(2, sourceBytes(source))))))
      return Status::kModelCapacityExceeded;
    auto data = impl_->metadata;
    if (const auto status = sourceHistory(data, source); status != Status::kOk)
      return status;
    Eigen::MatrixXd c(s + d, s + d);
    c.topLeftCorner(s, s) = impl_->c;
    c.bottomLeftCorner(d, s).noalias() = activeGenerator * impl_->k;
    c.topRightCorner(s, d) = c.bottomLeftCorner(d, s).transpose();
    c.bottomRightCorner(d, d).noalias() = activeGenerator * impl_->p * activeGenerator.transpose();
    mirrorLower(c);
    Eigen::MatrixXd k(n, s + d);
    k.leftCols(s) = impl_->k;
    k.rightCols(d).noalias() = impl_->p * activeGenerator.transpose();
    Eigen::MatrixXd p = impl_->p;
    auto operation = std::make_shared<Operation>();
    operation->kind = OperationKind::kClone;
    operation->oldSourceDof = s;
    operation->active = activeGenerator;
    remember(data, operation);
    data.sources.push_back(source);
    return impl_->publish(p, k, c, data);
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

Status ConsiderSourceFilter::reset(const Eigen::MatrixXd & transform)
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    const auto n = impl_->p.rows();
    if (transform.rows() != n || transform.cols() != n)
      return Status::kDimensionMismatch;
    if (!transform.allFinite())
      return Status::kNonFinite;
    Reservation reservation;
    if (!reservation.acquire(impl_->account, impl_->workspace()))
      return Status::kModelCapacityExceeded;
    auto data = impl_->metadata;
    Eigen::MatrixXd p = transform * impl_->p * transform.transpose();
    Eigen::MatrixXd k = transform * impl_->k;
    mirrorLower(p);
    auto operation = std::make_shared<Operation>();
    operation->kind = OperationKind::kProcess;
    operation->oldSourceDof = k.cols();
    operation->active = transform;
    remember(data, operation);
    return impl_->publish(p, k, impl_->c, data);
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

namespace
{

Status processFormat(const ProcessNoise & noise, Eigen::Index n, Eigen::Index s)
{
  if (noise.knowledge != MomentKnowledge::kKnown || !validKey(noise.evidence))
    return Status::kUnknownMoment;
  if (noise.mean.size() != n || noise.covariance.rows() != n || noise.covariance.cols() != n ||
      noise.priorCross.rows() != n + s || noise.priorCross.cols() != n)
    return Status::kDimensionMismatch;
  if (!noise.mean.allFinite() || !noise.priorCross.allFinite())
    return Status::kNonFinite;
  if (const auto status = detail::auditCovariance(noise.covariance); status != Status::kOk)
    return status;
  return noise.priorCross.isZero(0.0) ? Status::kOk : Status::kUnsupportedNoiseStructure;
}

Status
processCopies(const Metadata & data, Eigen::Index n, const std::vector<CopyNoiseCross> & crosses, Operation & operation)
{
  std::size_t count = 0;
  for (const auto & model : data.noises) {
    count += model.frames.size();
    for (const auto & copy : model.frames) {
      const auto cross = std::find_if(crosses.begin(), crosses.end(), [&](const CopyNoiseCross & input) {
        return input.raw == model.use.raw && input.frame == copy.frame;
      });
      if (cross == crosses.end())
        return Status::kUnknownMoment;
      if (cross->covariance.rows() != n || cross->covariance.cols() != model.covariance.rows())
        return Status::kDimensionMismatch;
      if (!cross->covariance.allFinite())
        return Status::kNonFinite;
      if (std::count_if(crosses.begin(), crosses.end(), [&](const CopyNoiseCross & input) {
            return input.raw == model.use.raw && input.frame == copy.frame;
          }) != 1)
        return Status::kInvalidIdentity;
      operation.copies.push_back({model.use.raw, copy.frame, cross->covariance});
    }
  }
  return count == crosses.size() ? Status::kOk : Status::kInvalidIdentity;
}

Status processSensitivities(const Metadata & data,
                            Eigen::Index n,
                            const std::vector<ProcessSensitivity> & inputs,
                            Operation & operation)
{
  std::set<std::string> keys;
  for (const auto & input : inputs) {
    if (!validKey(input.sensitivity) || !keys.insert(input.sensitivity).second)
      return Status::kInvalidIdentity;
    const auto model =
      std::find_if(data.sensitivities.begin(), data.sensitivities.end(), [&](const SensitivityModel & value) {
        return value.source.key == input.sensitivity;
      });
    if (model == data.sensitivities.end())
      return Status::kSourceHistoryUnavailable;
    if (input.loading.rows() != n ||
        input.loading.cols() != static_cast<Eigen::Index>(model->source.coordinates.components.size()))
      return Status::kDimensionMismatch;
    if (!input.loading.allFinite())
      return Status::kNonFinite;
    operation.sensitivities.emplace_back(input.sensitivity, input.loading);
  }
  return keys.size() == data.sensitivities.size() ? Status::kOk : Status::kUnknownMoment;
}

Status processCopyAudit(const Metadata & data,
                        const Eigen::MatrixXd & p,
                        const Eigen::MatrixXd & k,
                        const Eigen::MatrixXd & c,
                        const Eigen::MatrixXd & q,
                        const Operation & operation)
{
  if (std::all_of(operation.copies.begin(), operation.copies.end(), [](const Injection & copy) {
        return copy.loading.isZero(0.0);
      }))
    return Status::kOk;
  const auto n = p.rows();
  const auto sampled = std::min(c.rows(), kMaxSourceDof);
  Eigen::Index copiesDof = 0;
  for (const auto & model : data.noises)
    copiesDof += model.covariance.rows() * static_cast<Eigen::Index>(model.frames.size());
  Eigen::MatrixXd sample = Eigen::MatrixXd::Zero(n + sampled + copiesDof + n, n + sampled + copiesDof + n);
  sample.topLeftCorner(n, n) = p;
  sample.block(0, n, n, sampled) = k.leftCols(sampled);
  sample.block(n, 0, sampled, n) = k.leftCols(sampled).transpose();
  sample.block(n, n, sampled, sampled) = c.topLeftCorner(sampled, sampled);
  sample.bottomRightCorner(n, n) = q;
  Eigen::Index start = n + sampled;
  for (const auto & model : data.noises) {
    const auto d = model.covariance.rows();
    for (const auto & copy : model.frames) {
      const auto x = copyCross(data, model, copy, n);
      sample.block(0, start, n + sampled, d) = x.topRows(n + sampled);
      sample.block(start, 0, d, n + sampled) = x.topRows(n + sampled).transpose();
      sample.block(start, start, d, d) = static_cast<double>(model.use.totalUses) * model.covariance;
      const auto input = std::find_if(operation.copies.begin(), operation.copies.end(), [&](const Injection & value) {
        return value.raw == model.use.raw && value.frame == copy.frame;
      });
      sample.block(sample.rows() - n, start, n, d) = input->loading;
      sample.block(start, sample.rows() - n, d, n) = input->loading.transpose();
      start += d;
    }
  }
  return detail::auditCovariance(sample);
}

}  // namespace

Status ConsiderSourceFilter::propagate(const Eigen::MatrixXd & phi,
                                       const ProcessNoise & noise,
                                       const std::vector<CopyNoiseCross> & crosses,
                                       const std::vector<ProcessSensitivity> & sensitivities)
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    const auto n = impl_->p.rows();
    const auto s = impl_->c.rows();
    if (phi.rows() != n || phi.cols() != n)
      return Status::kDimensionMismatch;
    if (!phi.allFinite())
      return Status::kNonFinite;
    if (const auto status = processFormat(noise, n, s); status != Status::kOk)
      return status;
    std::size_t copiesDof = 0;
    for (const auto & model : impl_->metadata.noises)
      copiesDof =
        sumBytes(copiesDof, productBytes(static_cast<std::size_t>(model.covariance.rows()), model.frames.size()));
    const auto auditDof = sumBytes(copiesDof, static_cast<std::size_t>(n + n + std::min(s, kMaxSourceDof)));
    std::size_t sensitivityBytes = 0;
    for (const auto & input : sensitivities)
      sensitivityBytes = sumBytes(sensitivityBytes, matrixBytes(input.loading) + textBytes(input.sensitivity));
    Reservation reservation;
    if (!reservation.acquire(
          impl_->account,
          impl_->workspace(sumBytes(productBytes(8 * sizeof(double), productBytes(auditDof, auditDof)),
                                    productBytes(4, sensitivityBytes)))))
      return Status::kModelCapacityExceeded;
    auto operation = std::make_shared<Operation>();
    operation->kind = OperationKind::kProcess;
    operation->oldSourceDof = s;
    operation->active = phi;
    if (const auto status = processCopies(impl_->metadata, n, crosses, *operation); status != Status::kOk)
      return status;
    if (const auto status = processSensitivities(impl_->metadata, n, sensitivities, *operation); status != Status::kOk)
      return status;
    if (const auto status =
          processCopyAudit(impl_->metadata, impl_->p, impl_->k, impl_->c, noise.covariance, *operation);
        status != Status::kOk)
      return status;
    auto data = impl_->metadata;
    Eigen::MatrixXd p = phi * impl_->p * phi.transpose() + noise.covariance;
    Eigen::MatrixXd k = phi * impl_->k;
    mirrorLower(p);
    remember(data, operation);
    return impl_->publish(p, k, impl_->c, data);
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

Status
ConsiderSourceFilter::retire(const std::vector<std::string> & keys, bool willNeverReuse, const std::string & evidence)
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    if (keys.empty())
      return Status::kNoOp;
    if (!willNeverReuse || !validKey(evidence))
      return Status::kSourceHistoryUnavailable;
    std::set<Eigen::Index> removed;
    for (const auto & key : keys) {
      const auto index = resolve(impl_->metadata, key);
      if (index < 0)
        return Status::kSourceHistoryUnavailable;
      if (!removed.insert(index).second)
        return Status::kIdentityConflict;
    }
    Reservation reservation;
    if (!reservation.acquire(impl_->account, impl_->workspace(matrixBytes(impl_->c))))
      return Status::kModelCapacityExceeded;
    auto data = impl_->metadata;
    auto operation = std::make_shared<Operation>();
    operation->kind = OperationKind::kRetire;
    operation->oldSourceDof = impl_->c.rows();
    const auto n = impl_->p.rows();
    operation->sourceRows.resize(static_cast<std::size_t>(n));
    std::iota(operation->sourceRows.begin(), operation->sourceRows.end(), Eigen::Index{0});
    std::vector<Eigen::Index> kept;
    std::vector<Source> sources;
    for (std::size_t i = 0; i < data.sources.size(); ++i) {
      const auto & source = data.sources[i];
      if (removed.count(static_cast<Eigen::Index>(i)) != 0) {
        data.retiredNames.insert(source.key);
        for (const auto & raw : source.rawAncestry)
          if (designatesRaw(source, raw))
            data.retiredDesignatedRaw.insert(raw);
        continue;
      }
      sources.push_back(source);
      const auto start = offset(data, static_cast<Eigen::Index>(i));
      for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(source.coordinates.components.size()); ++j) {
        kept.push_back(start + j);
        operation->sourceRows.push_back(n + start + j);
      }
    }
    // Opaque retirement is bounded cumulatively; typed retirement uses watermarks
    // and exact raw-role history, never a per-typed-source 128-entry tombstone cap.
    for (const auto index : removed) {
      const auto & source = data.sources[static_cast<std::size_t>(index)];
      if (!source.sequence)
        ++data.opaqueRetired;
    }
    if (data.opaqueRetired > kMaxLedger)
      return Status::kModelCapacityExceeded;
    data.noises.erase(
      std::remove_if(data.noises.begin(),
                     data.noises.end(),
                     [&](const NoiseModel & noise) { return data.retiredNames.count(noise.use.owner.key) != 0; }),
      data.noises.end());
    remember(data, operation);
    data.sources = std::move(sources);
    Eigen::MatrixXd p = impl_->p;
    Eigen::MatrixXd k = selectRows(impl_->k.transpose(), kept).transpose();
    Eigen::MatrixXd c(static_cast<Eigen::Index>(kept.size()), static_cast<Eigen::Index>(kept.size()));
    for (std::size_t row = 0; row < kept.size(); ++row)
      for (std::size_t col = 0; col < kept.size(); ++col)
        c(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = impl_->c(kept[row], kept[col]);
    return impl_->publish(p, k, c, data);
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

Status ConsiderSourceFilter::registerSensitivity(const Source & source, const Eigen::MatrixXd & completeJointLoading)
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    if (const auto status = validSource(source); status != Status::kOk)
      return status;
    if (std::any_of(impl_->metadata.sensitivityLabels.begin(),
                    impl_->metadata.sensitivityLabels.end(),
                    [&](const Source & value) { return value.key == source.key; }) ||
        impl_->metadata.names.count(source.key) != 0)
      return Status::kIdentityConflict;
    if (completeJointLoading.rows() != impl_->p.rows() + impl_->c.rows() ||
        completeJointLoading.cols() != static_cast<Eigen::Index>(source.coordinates.components.size()))
      return Status::kDimensionMismatch;
    if (!completeJointLoading.allFinite())
      return Status::kNonFinite;
    Reservation reservation;
    if (!reservation.acquire(impl_->account,
                             impl_->workspace(sumBytes(productBytes(4, matrixBytes(completeJointLoading)),
                                                       productBytes(4, sourceBytes(source))))))
      return Status::kModelCapacityExceeded;
    auto data = impl_->metadata;
    data.sensitivityLabels.push_back(source);
    data.sensitivities.push_back({source, completeJointLoading, data.history.size()});
    Eigen::MatrixXd p = impl_->p, k = impl_->k;
    return impl_->publish(p, k, impl_->c, data);
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

const Eigen::MatrixXd * ConsiderSourceFilter::activeCovariance() const
{
  return impl_->version ? &impl_->p : nullptr;
}

Status ConsiderSourceFilter::cross(const std::string & key, Eigen::MatrixXd & out) const
{
  if (!impl_->version)
    return Status::kUnknownMoment;
  const auto block = resolve(impl_->metadata, key);
  if (block < 0)
    return Status::kSourceHistoryUnavailable;
  const auto d =
    static_cast<Eigen::Index>(impl_->metadata.sources[static_cast<std::size_t>(block)].coordinates.components.size());
  out = impl_->k.middleCols(offset(impl_->metadata, block), d);
  return Status::kOk;
}

Status ConsiderSourceFilter::sourceCovariance(const std::string & first,
                                              const std::string & second,
                                              Eigen::MatrixXd & out) const
{
  if (!impl_->version)
    return Status::kUnknownMoment;
  const auto a = resolve(impl_->metadata, first), b = resolve(impl_->metadata, second);
  if (a < 0 || b < 0)
    return Status::kSourceHistoryUnavailable;
  const auto da =
    static_cast<Eigen::Index>(impl_->metadata.sources[static_cast<std::size_t>(a)].coordinates.components.size());
  const auto db =
    static_cast<Eigen::Index>(impl_->metadata.sources[static_cast<std::size_t>(b)].coordinates.components.size());
  out = impl_->c.block(offset(impl_->metadata, a), offset(impl_->metadata, b), da, db);
  return Status::kOk;
}

Status ConsiderSourceFilter::sensitivity(const std::string & key, Eigen::MatrixXd & out) const
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    const auto model = std::find_if(impl_->metadata.sensitivities.begin(),
                                    impl_->metadata.sensitivities.end(),
                                    [&](const SensitivityModel & value) { return value.source.key == key; });
    if (model == impl_->metadata.sensitivities.end())
      return Status::kSourceHistoryUnavailable;
    Reservation reservation;
    if (!reservation.acquire(impl_->account, impl_->workspace(productBytes(8, matrixBytes(model->initial)))))
      return Status::kModelCapacityExceeded;
    const Eigen::MatrixXd complete = sensitivityJoint(impl_->metadata, *model);
    out = complete.topRows(impl_->p.rows());
    return Status::kOk;
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

Status ConsiderSourceFilter::sourceSensitivity(const std::string & source,
                                               const std::string & key,
                                               Eigen::MatrixXd & out) const
{
  try {
    if (!impl_->version)
      return Status::kUnknownMoment;
    const auto block = resolve(impl_->metadata, source);
    const auto model = std::find_if(impl_->metadata.sensitivities.begin(),
                                    impl_->metadata.sensitivities.end(),
                                    [&](const SensitivityModel & value) { return value.source.key == key; });
    if (block < 0 || model == impl_->metadata.sensitivities.end())
      return Status::kSourceHistoryUnavailable;
    Reservation reservation;
    if (!reservation.acquire(impl_->account, impl_->workspace(productBytes(8, matrixBytes(model->initial)))))
      return Status::kModelCapacityExceeded;
    const Eigen::MatrixXd complete = sensitivityJoint(impl_->metadata, *model);
    const auto d =
      static_cast<Eigen::Index>(impl_->metadata.sources[static_cast<std::size_t>(block)].coordinates.components.size());
    out = complete.middleRows(impl_->p.rows() + offset(impl_->metadata, block), d);
    return Status::kOk;
  } catch (const std::bad_alloc &) {
    return Status::kModelCapacityExceeded;
  }
}

const std::vector<Source> & ConsiderSourceFilter::sensitivitySources() const
{
  return impl_->metadata.sensitivityLabels;
}
MomentKnowledge ConsiderSourceFilter::knowledge() const
{
  return impl_->metadata.knowledge;
}
std::uint64_t ConsiderSourceFilter::revision() const
{
  return impl_->version ? *impl_->version : 0;
}
std::size_t ConsiderSourceFilter::memoryBytes() const
{
  return sumBytes(impl_->account->resident, impl_->account->pending);
}

std::uint64_t ConsiderSourceFilter::reuseCount(const RawId & raw) const
{
  const auto model = std::find_if(impl_->metadata.noises.begin(),
                                  impl_->metadata.noises.end(),
                                  [&](const NoiseModel & value) { return value.use.raw == raw; });
  return model == impl_->metadata.noises.end() ? 0 : static_cast<std::uint64_t>(model->frames.size());
}

std::string ConsiderSourceFilter::ledgerIdentity() const
{
  const auto & data = impl_->metadata;
  std::string result;
  appendCanonicalText(result, std::to_string(revision()));
  appendCanonicalText(result, std::to_string(static_cast<int>(data.knowledge)));
  appendCanonicalCoordinates(result, data.active);
  for (const auto & identity : data.recent)
    appendCanonicalText(result, identityBytes(identity));
  for (const auto * marks : {&data.rawHigh, &data.sourceHigh, &data.compacted, &data.factorHigh}) {
    appendCanonicalText(result, "watermarks");
    for (const auto & mark : *marks) {
      appendCanonicalText(result, std::to_string(mark.first));
      appendCanonicalText(result, std::to_string(mark.second));
    }
  }
  for (const auto & binding : data.factorBindings) {
    appendCanonicalText(result, binding.first);
    appendRaw(result, binding.second);
  }
  for (const auto * names : {&data.retiredNames, &data.retiredDesignatedRaw, &data.consumedRaw, &data.remainders}) {
    appendCanonicalText(result, "names");
    for (const auto & name : *names)
      appendCanonicalText(result, name);
  }
  for (const auto & binding : data.bindings) {
    appendCanonicalText(result, binding.raw);
    appendRaw(result, binding.identity);
  }
  for (const auto & name : data.names) {
    appendCanonicalText(result, name.first);
    appendCanonicalText(result, name.second);
  }
  for (const auto & model : data.noises) {
    appendRaw(result, model.use.raw);
    appendCanonicalText(result, std::to_string(model.use.totalUses));
    appendCanonicalText(result, model.use.owner.key);
    appendSequence(result, model.use.owner.sequence);
    for (const auto & copy : model.frames) {
      appendRaw(result, copy.frame);
      appendCanonicalText(result, std::to_string(copy.birth));
    }
  }
  appendCanonicalText(result, std::to_string(data.history.size()));
  appendCanonicalText(result, std::to_string(data.sensitivities.size()));
  appendCanonicalText(result, std::to_string(data.opaqueRetired));
  for (const auto & source : data.sources) {
    appendCanonicalText(result, source.key);
    appendSequence(result, source.sequence);
  }
  return result;
}

}  // namespace fastlio::joint_source

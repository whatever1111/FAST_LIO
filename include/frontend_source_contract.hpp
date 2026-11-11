#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fastlio::joint_source
{

inline constexpr Eigen::Index kMaxActiveDof = 23;
inline constexpr Eigen::Index kMaxSourceDof = 32;
inline constexpr Eigen::Index kMaxRows = 64;
inline constexpr std::size_t kMaxLedger = 128;
inline constexpr std::size_t kMaxIdentityBytes = 256;
inline constexpr std::size_t kMaxFactorPayloadBytes = static_cast<std::size_t>(256) * 1024;

enum class Status
{
  kOk,
  kNoOp,
  kUnknownMoment,
  kInvalidIdentity,
  kIdentityConflict,
  kDimensionMismatch,
  kNonFinite,
  kInvalidCovariance,
  kUnsupportedNoiseStructure,
  kUnsupportedModel,
  kModelCapacityExceeded,
  kSourceHistoryUnavailable,
  kStaleProposal,
  kNumericalFailure,
  kLedgerCompacted,
  kReuseBudgetExceeded
};

// Known includes deterministic zero covariance. Unknown never denotes zero.
enum class MomentKnowledge
{
  kUnknown,
  kKnown,
  kUpperBound
};
enum class ModelDomain
{
  kUnknown,
  kLinearFixedModel
};

struct Coordinates
{
  std::vector<std::string> components;
  std::vector<std::string> units;
  std::string frame;
  std::string tangent;
  std::string epoch;
};

struct RawId
{
  std::uint32_t stream = 0;
  std::uint64_t ordinal = 0;
  std::uint32_t sub = 0;

  bool operator==(const RawId & other) const
  {
    return stream == other.stream && ordinal == other.ordinal && sub == other.sub;
  }
  bool operator!=(const RawId & other) const { return !(*this == other); }
};

struct TypedRaw
{
  std::string raw;  // Exact binding to the existing canonical raw name.
  RawId identity;

  bool operator==(const TypedRaw & other) const { return raw == other.raw && identity == other.identity; }
};

enum class RawRole
{
  kDesignated,
  kDerived
};

struct RawRoleBinding
{
  // Unspecified remains designated. Derived names inherited statistical
  // dependence without asserting a factor route through this source.
  std::string raw;
  RawRole role = RawRole::kDesignated;
};

struct SourceOwner
{
  std::string key;
  std::optional<RawId> sequence = std::nullopt;

  bool operator==(const SourceOwner & other) const { return key == other.key && sequence == other.sequence; }
};

struct ReuseUse
{
  RawId raw;
  RawId frame;
  std::uint64_t totalUses = 0;
  SourceOwner owner;

  bool operator==(const ReuseUse & other) const
  {
    return raw == other.raw && frame == other.frame && totalUses == other.totalUses && owner == other.owner;
  }
};

// These keys denote physical observations, never XYZ proximity or delivery order.
// An alias names this same variable. Derived variables need an explicit generator.
struct Source
{
  std::string key;
  std::string kind;
  std::vector<std::string> rawAncestry;
  std::vector<std::string> aliases;
  Coordinates coordinates;
  std::string model;
  std::string lifetime;
  std::optional<RawId> sequence = std::nullopt;
  std::vector<TypedRaw> typedRaw = {};
  std::vector<RawRoleBinding> rawRoles = {};
};

struct JointMoments
{
  Coordinates active;
  std::vector<Source> sources;
  MomentKnowledge knowledge = MomentKnowledge::kUnknown;
  std::string evidence;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
};

// One raw primitive may feed several geometric functions. Such reuse must name
// the SAME retained source and have an independently justified new remainder.
struct SharedRawSource
{
  std::string raw;
  std::string source;

  bool operator==(const SharedRawSource & other) const { return raw == other.raw && source == other.source; }
};

struct FactorIdentity
{
  // Fresh raw ancestry carried by a live source requires a shared entry naming
  // that source with nonzero loading. Retired ancestry is unavailable. Exact
  // retransmissions are checked by the ledger before fresh ancestry validation.
  // Without role declarations, live and retired carriers of the same raw keep
  // the conservative kSourceHistoryUnavailable behavior. Only an explicitly
  // derived retired dependence is excluded from that routing requirement.
  std::string key;      // complete canonical likelihood identity, excluding delivery aliases
  std::string payload;  // exact canonical bytes, including all numeric inputs
  std::vector<std::string> rawAncestry;
  std::vector<SharedRawSource> shared;
  std::string remainderKey;  // independent realization, unique across new factors
  std::string independenceEvidence;
  std::optional<RawId> sequence = std::nullopt;
  std::vector<TypedRaw> typedRaw = {};
  std::vector<ReuseUse> reuse = {};

  bool operator==(const FactorIdentity & other) const
  {
    return key == other.key && payload == other.payload && rawAncestry == other.rawAncestry && shared == other.shared &&
           remainderKey == other.remainderKey && independenceEvidence == other.independenceEvidence &&
           sequence == other.sequence && typedRaw == other.typedRaw && reuse == other.reuse;
  }
};

inline bool validKey(const std::string & key)
{
  return !key.empty() && key.size() <= kMaxIdentityBytes;
}

inline bool contains(const std::vector<std::string> & keys, const std::string & key)
{
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

inline bool uniqueKeys(const std::vector<std::string> & keys)
{
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!validKey(keys[i]) || std::find(keys.begin(), keys.begin() + i, keys[i]) != keys.begin() + i)
      return false;
  }
  return true;
}

inline bool validTypedRaw(const std::vector<std::string> & ancestry, const std::vector<TypedRaw> & typed)
{
  if (typed.size() > ancestry.size())
    return false;
  for (std::size_t i = 0; i < typed.size(); ++i) {
    if (!contains(ancestry, typed[i].raw))
      return false;
    for (std::size_t j = 0; j < i; ++j)
      if (typed[i].raw == typed[j].raw || typed[i].identity == typed[j].identity)
        return false;
  }
  return true;
}

inline bool validSourceExtensions(const Source & source)
{
  if (source.rawRoles.size() > source.rawAncestry.size() || !validTypedRaw(source.rawAncestry, source.typedRaw))
    return false;
  std::vector<std::string> declared;
  for (const auto & binding : source.rawRoles) {
    if (!contains(source.rawAncestry, binding.raw) ||
        (binding.role != RawRole::kDesignated && binding.role != RawRole::kDerived))
      return false;
    declared.push_back(binding.raw);
  }
  return uniqueKeys(declared);
}

inline bool designatesRaw(const Source & source, const std::string & raw)
{
  return contains(source.rawAncestry, raw) &&
         std::none_of(source.rawRoles.begin(), source.rawRoles.end(), [&](const RawRoleBinding & binding) {
           return binding.raw == raw && binding.role == RawRole::kDerived;
         });
}

inline bool validCoordinates(const Coordinates & coordinates)
{
  return !coordinates.components.empty() && coordinates.components.size() == coordinates.units.size() &&
         coordinates.components.size() <= static_cast<std::size_t>(kMaxSourceDof) &&
         uniqueKeys(coordinates.components) &&
         std::all_of(coordinates.units.begin(), coordinates.units.end(), validKey) && validKey(coordinates.frame) &&
         validKey(coordinates.tangent) && validKey(coordinates.epoch);
}

inline Eigen::Index sourceDof(const std::vector<Source> & sources)
{
  Eigen::Index count = 0;
  for (const auto & source : sources)
    count += static_cast<Eigen::Index>(source.coordinates.components.size());
  return count;
}

inline Status validateSources(const std::vector<Source> & sources)
{
  if (sources.size() > static_cast<std::size_t>(kMaxSourceDof) || sourceDof(sources) > kMaxSourceDof)
    return Status::kModelCapacityExceeded;
  std::vector<std::string> names;
  for (const auto & source : sources) {
    if (!validKey(source.key) || !validKey(source.kind) || !validKey(source.model) || !validKey(source.lifetime) ||
        !validCoordinates(source.coordinates) || source.rawAncestry.empty() || source.rawAncestry.size() > kMaxLedger ||
        source.aliases.size() > kMaxLedger || !uniqueKeys(source.rawAncestry) || !uniqueKeys(source.aliases) ||
        !validSourceExtensions(source))
      return Status::kInvalidIdentity;
    names.push_back(source.key);
    names.insert(names.end(), source.aliases.begin(), source.aliases.end());
  }
  if (names.size() > kMaxLedger)
    return Status::kModelCapacityExceeded;
  return uniqueKeys(names) ? Status::kOk : Status::kIdentityConflict;
}

inline Status validateIdentity(const FactorIdentity & factor)
{
  if (factor.payload.size() > kMaxFactorPayloadBytes)
    return Status::kModelCapacityExceeded;
  if (!validKey(factor.key) || factor.payload.empty() || !validKey(factor.remainderKey) ||
      !validKey(factor.independenceEvidence) || factor.rawAncestry.empty() || !uniqueKeys(factor.rawAncestry))
    return Status::kInvalidIdentity;
  if (factor.rawAncestry.size() > kMaxLedger || factor.shared.size() > kMaxLedger ||
      factor.typedRaw.size() > kMaxLedger || factor.reuse.size() > kMaxLedger)
    return Status::kModelCapacityExceeded;
  if (!validTypedRaw(factor.rawAncestry, factor.typedRaw))
    return Status::kInvalidIdentity;
  for (std::size_t i = 0; i < factor.reuse.size(); ++i) {
    const auto & use = factor.reuse[i];
    if (use.totalUses == 0 || !validKey(use.owner.key) ||
        std::none_of(factor.typedRaw.begin(), factor.typedRaw.end(), [&](const TypedRaw & raw) {
          return raw.identity == use.raw;
        }))
      return Status::kInvalidIdentity;
    for (std::size_t j = 0; j < i; ++j)
      if (use.raw == factor.reuse[j].raw && use.frame == factor.reuse[j].frame)
        return Status::kInvalidIdentity;
  }
  for (std::size_t i = 0; i < factor.shared.size(); ++i) {
    const auto & shared = factor.shared[i];
    if (!contains(factor.rawAncestry, shared.raw) || !validKey(shared.source) ||
        std::find(factor.shared.begin(), factor.shared.begin() + i, shared) != factor.shared.begin() + i)
      return Status::kInvalidIdentity;
  }
  return Status::kOk;
}

inline bool sharedRawCovered(const FactorIdentity & first, const FactorIdentity & second, const std::string & raw)
{
  for (const auto & shared : first.shared) {
    if (shared.raw == raw && std::find(second.shared.begin(), second.shared.end(), shared) != second.shared.end())
      return true;
  }
  return false;
}

struct Consumption
{
  Status status = Status::kNoOp;
  std::vector<std::size_t> fresh;
};

// Value-owned ledger. consume validates on a copy, then swaps once. Numerical
// proposals consume a PRIVATE copy and publish it together with joint moments.
class FactorLedger
{
public:
  const std::vector<FactorIdentity> & entries() const { return entries_; }

  Consumption consume(const std::vector<FactorIdentity> & factors)
  {
    if (factors.size() > kMaxLedger)
      return {Status::kModelCapacityExceeded, {}};
    auto candidate = entries_;
    Consumption result;
    for (std::size_t i = 0; i < factors.size(); ++i) {
      const auto status = validateIdentity(factors[i]);
      if (status != Status::kOk)
        return {status, {}};
      const auto appended = append(candidate, factors[i]);
      if (appended == Status::kOk)
        result.fresh.push_back(i);
      else if (appended != Status::kNoOp)
        return {appended, {}};
    }
    if (result.fresh.empty())
      return result;
    entries_.swap(candidate);
    result.status = Status::kOk;
    return result;
  }

private:
  static Status append(std::vector<FactorIdentity> & entries, const FactorIdentity & factor)
  {
    for (const auto & entry : entries) {
      if (entry.key == factor.key)
        return entry == factor ? Status::kNoOp : Status::kIdentityConflict;
      if (entry.remainderKey == factor.remainderKey)
        return Status::kUnsupportedNoiseStructure;
      for (const auto & raw : factor.rawAncestry) {
        if (contains(entry.rawAncestry, raw) && !sharedRawCovered(entry, factor, raw))
          return Status::kUnsupportedNoiseStructure;
      }
    }
    if (entries.size() == kMaxLedger)
      return Status::kModelCapacityExceeded;
    entries.push_back(factor);
    return Status::kOk;
  }

  std::vector<FactorIdentity> entries_;
};

}  // namespace fastlio::joint_source

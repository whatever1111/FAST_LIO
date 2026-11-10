#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
  kNumericalFailure
};

// Known includes deterministic zero covariance. Unknown never denotes zero.
enum class MomentKnowledge
{
  kUnknown,
  kKnown
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
  std::string key;      // complete canonical likelihood identity, excluding delivery aliases
  std::string payload;  // exact canonical bytes, including all numeric inputs
  std::vector<std::string> rawAncestry;
  std::vector<SharedRawSource> shared;
  std::string remainderKey;  // independent realization, unique across new factors
  std::string independenceEvidence;

  bool operator==(const FactorIdentity & other) const
  {
    return key == other.key && payload == other.payload && rawAncestry == other.rawAncestry && shared == other.shared &&
           remainderKey == other.remainderKey && independenceEvidence == other.independenceEvidence;
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
        source.aliases.size() > kMaxLedger || !uniqueKeys(source.rawAncestry) || !uniqueKeys(source.aliases))
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
  if (factor.rawAncestry.size() > kMaxLedger || factor.shared.size() > kMaxLedger)
    return Status::kModelCapacityExceeded;
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

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <openssl/evp.h>
#include <string>
#include <vector>

namespace fast_lio
{

// Runtime telemetry is observational: payload/report failures must not escape
// into the estimator callback. Owned by its serialized callback group.
class FrontendPublicationGuard
{
public:
  bool faulted() const noexcept { return faulted_; }

  template<class Publish, class Report>
  bool run(Publish && publish, Report && report) noexcept
  {
    if (faulted_)
      return false;
    try {
      publish();
      return true;
    } catch (const std::exception & error) {
      faulted_ = true;
      reportOnce(report, error.what());
    } catch (...) {
      faulted_ = true;
      reportOnce(report, "unknown exception");
    }
    return false;
  }

private:
  template<class Report>
  void reportOnce(Report & report, const char * reason) noexcept
  {
    try {
      report(reason);
    } catch (...) {
      // Even a failed logger must leave the activation latched off.
      faulted_ = true;
    }
  }
  bool faulted_ = false;
};

// ROS strings must be serializable UTF-8, with no embedded NUL identifiers.
inline bool frontendIdentifier(const std::string & value, std::size_t maximum)
{
  if (value.empty() || value.size() > maximum)
    return false;
  std::size_t i = 0;
  while (i < value.size()) {
    const auto lead = static_cast<std::uint8_t>(value[i++]);
    if (lead == 0)
      return false;
    if (lead < 0x80)
      continue;
    unsigned continuation = 0;
    std::uint32_t code = 0;
    std::uint32_t minimum = 0;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation = 1;
      code = lead & 0x1f;
      minimum = 0x80;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation = 2;
      code = lead & 0x0f;
      minimum = 0x800;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation = 3;
      code = lead & 0x07;
      minimum = 0x10000;
    } else
      return false;
    if (value.size() - i < continuation)
      return false;
    for (unsigned j = 0; j < continuation; ++j) {
      const auto next = static_cast<std::uint8_t>(value[i++]);
      if ((next & 0xc0) != 0x80)
        return false;
      code = (code << 6) | (next & 0x3f);
    }
    if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
      return false;
  }
  return true;
}

enum class FrontendState : std::uint8_t
{
  kUnknown = 0,
  kInitializing = 1,
  kTracking = 2,
  kDegraded = 3,
  kLost = 4
};
enum class FrontendSubstate : std::uint8_t
{
  kUnknown = 0,
  kNormal = 1,
  kBlind = 2,
  kVerifying = 3,
  kStaticHold = 4,
  kRunaway = 5
};
enum class FrontendValidity : std::uint8_t
{
  kUnknown = 0,
  kFull = 1,
  kPartial = 2,
  kInvalid = 3
};
enum class FrontendActivity : std::uint8_t
{
  kUnknown = 0,
  kIdle = 1,
  kBusy = 2,
  kBlocked = 3
};
constexpr std::uint32_t kFrontendGuardDegraded = 1;
constexpr std::uint32_t kFrontendGuardStaticHold = 2;
constexpr std::uint32_t kFrontendGuardReanchor = 4;
constexpr std::uint32_t kFrontendGuardRunaway = 8;
constexpr std::uint32_t kFrontendGuardLevelling = 16;
constexpr std::uint32_t kFrontendHealthDegraded = 1;
constexpr std::uint32_t kFrontendHealthStaticHold = 2;
constexpr std::uint32_t kFrontendHealthLost = 4;
constexpr std::uint32_t kFrontendHealthLevelling = 8;
constexpr std::uint8_t kFrontendAllAxes = 63;
template<class Enum>
inline std::uint8_t frontendWire(Enum value)
{
  return static_cast<std::uint8_t>(value);
}

inline bool frontendHasEffectiveFeatureCount(int measurementIterations, int effectiveFeatures)
{
  return measurementIterations > 0 && effectiveFeatures >= 0;
}

inline bool frontendHasObservabilityAlong(int measurementIterations, double observability)
{
  return measurementIterations > 0 && std::isfinite(observability) && observability >= 0.0 && observability <= 1.0;
}

// Legacy residual storage survives a scan with no admitted measurement rows.
// Require evidence from this scan before exposing it as an observation metric.
inline bool frontendHasMeanResidual(int measurementIterations, int effectiveFeatures, double residual)
{
  return measurementIterations > 0 && effectiveFeatures > 0 && std::isfinite(residual) && residual >= 0.0;
}

struct FrontendFacts
{
  bool completed = false;
  bool initialized = false;
  bool lost = false;
  bool degraded = false;
  bool blind = false;
  bool verifying = false;
  bool staticHold = false;
  bool runaway = false;
  bool levelling = false;
  bool blocked = false;
  bool contextKnown = false;
  bool stampKnown = false;
  std::array<double, 7> pose{};  // position xyz, quaternion xyzw
};

struct FrontendVerdict
{
  std::uint8_t state = 0;
  std::uint8_t substate = 0;
  std::uint8_t validity = 0;
  std::uint8_t axes = 0;
  std::uint8_t activity = 1;
  std::uint32_t guards = 0;
  std::uint32_t health = 0;
};

inline FrontendVerdict frontendVerdict(const FrontendFacts & f)
{
  FrontendVerdict v;
  v.activity = frontendWire(f.blocked ? FrontendActivity::kBlocked : FrontendActivity::kIdle);
  v.health = (f.degraded ? kFrontendHealthDegraded : 0u) | (f.staticHold ? kFrontendHealthStaticHold : 0u) |
             (f.lost ? kFrontendHealthLost : 0u) | (f.levelling ? kFrontendHealthLevelling : 0u);
  v.guards = (f.degraded ? kFrontendGuardDegraded : 0u) | (f.staticHold ? kFrontendGuardStaticHold : 0u) |
             ((f.blind || f.verifying || f.lost) ? kFrontendGuardReanchor : 0u) |
             (f.runaway ? kFrontendGuardRunaway : 0u) | (f.levelling ? kFrontendGuardLevelling : 0u);
  const bool degraded = f.degraded || f.blind || f.verifying || f.runaway || f.levelling;
  v.state = frontendWire(f.lost           ? FrontendState::kLost
                         : !f.initialized ? FrontendState::kInitializing
                         : degraded       ? FrontendState::kDegraded
                                          : FrontendState::kTracking);
  v.substate = frontendWire(f.runaway      ? FrontendSubstate::kRunaway
                            : f.staticHold ? FrontendSubstate::kStaticHold
                            : f.verifying  ? FrontendSubstate::kVerifying
                            : f.blind      ? FrontendSubstate::kBlind
                                           : FrontendSubstate::kNormal);
  if (!f.completed)
    return v;
  if (f.lost) {
    v.validity = frontendWire(FrontendValidity::kInvalid);
    return v;
  }
  // A guard does not prove any individual pose axis. Static hold alone is not Lost.
  if (!f.initialized || degraded || f.blocked || !f.contextKnown || !f.stampKnown)
    return v;
  for (const auto value : f.pose) {
    if (!std::isfinite(value))
      return v;
  }
  double normSquared = 0.0;
  for (std::size_t i = 3; i < 7; ++i)
    normSquared += f.pose[i] * f.pose[i];
  constexpr double kQuaternionNormTolerance = 1e-6;
  if (std::abs(normSquared - 1.0) > kQuaternionNormTolerance)
    return v;
  v.validity = frontendWire(FrontendValidity::kFull);
  v.axes = kFrontendAllAxes;
  return v;
}

inline bool frontendStamp(std::int32_t sec, std::uint32_t nanosec, std::int64_t & result)
{
  constexpr std::int64_t kNanosPerSecond = 1000000000;
  if (nanosec >= kNanosPerSecond)
    return false;
  result = static_cast<std::int64_t>(sec) * kNanosPerSecond + nanosec;
  return true;
}

// Match legacy floor-seconds/truncate-nanoseconds without an unchecked cast.
inline bool frontendSecondsStamp(double seconds, std::int64_t & result)
{
  result = 0;
  if (!std::isfinite(seconds))
    return false;
  const double whole = std::floor(seconds);
  if (whole < std::numeric_limits<std::int32_t>::min() || whole > std::numeric_limits<std::int32_t>::max())
    return false;
  constexpr double kNanosPerSecond = 1000000000.0;
  const double fraction = (seconds - whole) * kNanosPerSecond;
  if (fraction < 0.0 || fraction >= kNanosPerSecond)
    return false;
  return frontendStamp(static_cast<std::int32_t>(whole), static_cast<std::uint32_t>(fraction), result);
}

inline bool frontendObservationId(const std::array<std::uint8_t, 16> & instance,
                                  bool hasInstance,
                                  std::uint64_t completion,
                                  std::int64_t stamp,
                                  bool hasStamp,
                                  const std::string & domain,
                                  std::array<std::uint8_t, 32> & digest)
{
  digest.fill(0);
  if (!hasInstance || !hasStamp || completion == 0 || !frontendIdentifier(domain, 48))
    return false;
  const char prefix[] = "LIO-FE-OBS-V1";
  std::vector<std::uint8_t> bytes(prefix, prefix + sizeof(prefix));  // includes exactly one NUL
  bytes.insert(bytes.end(), instance.begin(), instance.end());
  const auto append = [&bytes](std::uint64_t value, int width) {
    for (int i = width - 1; i >= 0; --i)
      bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
  };
  append(completion, 8);
  append(static_cast<std::uint64_t>(stamp), 8);
  append(domain.size(), 4);
  bytes.insert(bytes.end(), domain.begin(), domain.end());
  unsigned int size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(), nullptr) != 1 ||
      size != digest.size()) {
    digest.fill(0);
    return false;
  }
  return true;
}

struct FrontendProgress
{
  std::uint64_t input = 0;
  std::uint64_t completion = 0;
  std::uint64_t output = 0;
  std::uint64_t sample = 0;
  std::uint64_t dropped = 0;
  bool valid = true;
  bool hasInputStamp = false;
  std::int64_t inputStamp = 0;
  bool hasIdleStamp = false;
  std::int64_t idleStamp = 0;
  std::uint32_t previousHealth = 0;
  std::uint8_t previousState = 0;
  std::uint8_t previousActivity = 0;
  std::uint8_t previousSubstate = 0;

  void increment(std::uint64_t & counter, std::uint64_t count = 1)
  {
    if (count > std::numeric_limits<std::uint64_t>::max() - counter) {
      valid = false;
      return;
    }
    counter += count;
  }
  void admit(std::int64_t stamp, bool known)
  {
    increment(input);
    hasInputStamp = known;
    inputStamp = known ? stamp : 0;
  }
  void finish(const FrontendVerdict & verdict)
  {
    increment(completion);
    if (verdict.validity == frontendWire(FrontendValidity::kFull) ||
        (verdict.validity == frontendWire(FrontendValidity::kPartial) && verdict.axes != 0))
      increment(output);
  }
  bool idleDue(std::int64_t stamp, std::int64_t period, const FrontendVerdict & verdict)
  {
    const bool changed = verdict.health != previousHealth || verdict.state != previousState ||
                         verdict.activity != previousActivity || verdict.substate != previousSubstate;
    if (hasIdleStamp && !changed &&
        (stamp < idleStamp || static_cast<std::uint64_t>(stamp) - static_cast<std::uint64_t>(idleStamp) <
                                static_cast<std::uint64_t>(period)))
      return false;
    hasIdleStamp = true;
    idleStamp = stamp;
    previousHealth = verdict.health;
    previousState = verdict.state;
    previousActivity = verdict.activity;
    previousSubstate = verdict.substate;
    return true;
  }
};

}  // namespace fast_lio

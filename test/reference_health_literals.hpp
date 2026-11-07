// Independent wire literals, copied byte-for-byte to the outer decoder test.
// No producer function or ROS serializer computes these expected values.
#pragma once
#include <cstdint>
namespace reference_health_literals
{
struct Vector
{
  const char * name;
  std::uint8_t kind, startup, lifecycle, trust, local, hr;
  std::uint32_t flags, present, truth;
  bool pose, valid;
};
inline constexpr Vector kVectors[] = {
  {"cold-no-pose", 2, 1, 1, 1, 0, 0, 0, 143, 2, false, true},
  {"first-established-no-pose", 2, 2, 2, 1, 0, 0, 0, 143, 3, false, true},
  {"prior-excluded", 3, 3, 0, 1, 0, 0, 0, 143, 2, false, true},
  {"normal", 1, 2, 2, 1, 1, 1, 0, 143, 139, true, true},
  {"pending-raw-clear", 1, 2, 2, 2, 1, 0, 0, 143, 129, true, true},
  {"unanchored-local-healthy", 1, 2, 2, 3, 1, 0, 1, 143, 129, true, true},
  {"trusted-static-hold", 1, 2, 2, 1, 2, 2, 3, 143, 7, true, true},
  {"normal-and-held", 1, 2, 2, 1, 1, 2, 2, 143, 143, true, true},
  {"normal-local-unknown", 1, 2, 2, 1, 0, 1, 0, 143, 11, true, false},
  {"normal-local-degraded", 1, 2, 2, 1, 2, 1, 0, 143, 11, true, false},
  {"held-without-static-hold", 1, 2, 2, 1, 2, 2, 1, 143, 7, true, false},
  {"both-bits-local-degraded", 1, 2, 2, 1, 2, 2, 2, 143, 15, true, false},
  {"normal-with-lost", 1, 2, 2, 1, 1, 1, 4, 143, 139, true, false},
  {"normal-with-levelling", 1, 2, 2, 1, 1, 1, 8, 143, 139, true, false},
  {"held-with-lost", 1, 2, 2, 1, 2, 2, 7, 143, 7, true, false},
  {"lost", 1, 2, 2, 2, 3, 0, 5, 143, 1, true, true},
  {"unknown-startup", 2, 4, 0, 1, 0, 0, 0, 143, 2, false, false},
  {"contradictory-lifecycle", 2, 1, 2, 1, 0, 0, 0, 143, 3, false, false},
  {"unknown-kind", 5, 2, 2, 1, 0, 0, 0, 143, 3, false, false},
  {"unknown-trust", 2, 2, 2, 4, 0, 0, 0, 143, 1, false, false},
  {"unknown-positive", 2, 2, 2, 1, 0, 0, 0, 399, 3, false, false},
  {"true-without-present", 2, 2, 2, 1, 0, 0, 0, 135, 11, false, false},
  {"idle-local-is-not-a-measurement", 3, 2, 2, 1, 1, 0, 0, 143, 131, false, false},
  {"scan-without-pose", 1, 2, 2, 1, 1, 1, 0, 143, 139, false, false},
  {"pending-cannot-hold", 1, 2, 2, 2, 2, 0, 3, 143, 1, true, false},
  {"pending-cannot-normal", 1, 2, 2, 2, 1, 1, 0, 143, 137, true, false},
};

// Time fields of an otherwise valid snapshot. Every state carries all three, and its lease runs from the evaluated
// input for at most 200 ms. A vector that drops a field keeps the complete vector's values.
struct TimeVector
{
  const char * name;
  bool evaluated, effective, validUntil;
  std::int64_t evaluatedNs, effectiveNs, validUntilNs;
  bool valid;
};
inline constexpr TimeVector kTimeVectors[] = {
  {"complete", true, true, true, 1100000000, 1100000000, 1250000000, true},
  {"no-effective", true, false, true, 1100000000, 1100000000, 1250000000, false},
  {"no-evaluated-input", false, true, true, 1100000000, 1100000000, 1250000000, false},
  {"no-valid-until", true, true, false, 1100000000, 1100000000, 1250000000, false},
  {"lease-exactly-200ms", true, true, true, 1100000000, 1100000000, 1300000000, true},
  {"lease-200ms-and-1ns", true, true, true, 1100000000, 1100000000, 1300000001, false},
  {"valid-until-before-evaluated", true, true, true, 1100000000, 1100000000, 1099999999, false},
};
}  // namespace reference_health_literals

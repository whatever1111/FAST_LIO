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

// RW carry contract 3 on an RW-enabled Established scan [20.0 s, 20.1 s]: enabled 7, present 191. A zero time
// means the optional field is absent; a nonzero `lower` means the fallback epochs are present.
inline constexpr std::int64_t kRwScanBeginNs = 20000000000;
inline constexpr std::int64_t kRwScanEndNs = 20100000000;
struct RwVector
{
  const char * name;
  std::uint8_t trust, local, hr, witness;
  std::uint32_t flags, truth, taint, hard;
  std::int64_t rawStart, until, deadline;
  std::uint64_t lower, upper;
  bool valid;
};
inline constexpr RwVector kRwVectors[] = {
  {"rw-quiet-origin", 1, 1, 1, 0, 0, 155, 0, 0, 0, 0, 0, 0, 0, true},
  {"rw-latched-no-origin", 1, 1, 0, 0, 0, 131, 8, 0, 0, 0, 0, 0, 0, true},
  {"rw-origin-on-taint", 1, 1, 0, 0, 0, 147, 8, 0, 0, 0, 0, 0, 0, false},
  {"rw-origin-on-hard", 1, 3, 0, 0, 0, 19, 0, 1, 0, 0, 0, 0, 0, false},
  {"rw-current-window", 1, 2, 0, 1, 1, 51, 0, 0, 13000000000, 20300000000, 0, 0, 0, true},
  {"rw-pending-raw-clear-tail", 2, 1, 0, 1, 0, 177, 0, 0, 13000000000, 20249999999, 20250000000, 0, 0, true},
  {"rw-pending-at-deadline", 2, 1, 0, 1, 0, 177, 0, 0, 13000000000, 20250000000, 20250000000, 0, 0, false},
  {"rw-pending-without-deadline", 2, 1, 0, 1, 0, 177, 0, 0, 13000000000, 20249999999, 0, 0, 0, false},
  {"rw-fallback", 1, 2, 0, 2, 1, 51, 0, 0, 13000000000, 20250000000, 0, 19250000000, 19900000000, true},
  {"rw-fallback-past-witness", 1, 2, 0, 2, 1, 51, 0, 0, 13000000000, 20250000001, 0, 19250000000, 19900000000, false},
  {"rw-past-lease", 1, 2, 0, 1, 1, 51, 0, 0, 13000000000, 20300000001, 0, 0, 0, false},
  {"rw-at-raw-budget", 1, 2, 0, 1, 1, 51, 0, 0, 12200000000, 20200000000, 0, 0, 0, true},
  {"rw-past-raw-budget", 1, 2, 0, 1, 1, 51, 0, 0, 12200000000, 20200000001, 0, 0, 0, false},
  {"rw-before-scan-end", 1, 2, 0, 1, 1, 51, 0, 0, 13000000000, 20099999999, 0, 0, 0, false},
  {"rw-carry-on-clear-local-qualified", 1, 1, 1, 1, 0, 187, 0, 0, 13000000000, 20300000000, 0, 0, 0, false},
  {"rw-carry-with-lost", 1, 2, 0, 1, 5, 51, 0, 0, 13000000000, 20300000000, 0, 0, 0, false},
  {"rw-carry-with-hold", 1, 2, 0, 1, 3, 51, 0, 0, 13000000000, 20300000000, 0, 0, 0, false},
  {"rw-carry-without-origin", 1, 2, 0, 1, 1, 35, 0, 0, 13000000000, 20300000000, 0, 0, 0, false},
  {"rw-witness-without-carry", 1, 2, 0, 1, 1, 19, 0, 0, 13000000000, 0, 0, 0, 0, false},
  {"rw-current-with-epochs", 1, 2, 0, 1, 1, 51, 0, 0, 13000000000, 20300000000, 0, 19250000000, 19900000000, false},
  {"rw-carry-without-raw-start", 1, 2, 0, 1, 1, 51, 0, 0, 0, 20300000000, 0, 0, 0, false},
};
}  // namespace reference_health_literals

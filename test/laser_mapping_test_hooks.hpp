#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

// Only the non-installed test component defines or references these callbacks.
// Install before starting the worker/reset threads, clear only after joining both.
namespace fast_lio_test
{
extern std::function<void()> beforeMapAdd;
extern std::function<void()> beforeResetJoin;
extern std::function<void()> afterPendingReset;
extern std::function<void(std::size_t, double)> afterScanPrepared;
extern std::function<void(std::size_t, double, int)> afterHModelRows;
extern std::function<void(std::uint64_t, int, int, std::size_t, std::size_t, std::uint64_t)> afterImuConsumption;
extern std::function<void()> beforeImuProcess;

struct ImuInputSnapshot
{
  std::size_t lidarQueued;
  std::size_t imuQueued;
  std::uint64_t rejectedLidar;
  std::uint64_t rejectedImu;
  std::uint64_t discardedLidar;
  std::uint64_t discardedImu;
  bool faulted;
  bool mapFrozen;
  bool degradedOdom;
  double filterEnd;
};
ImuInputSnapshot inspectImuInput();
int reconfirmLastImuBatch();
}  // namespace fast_lio_test

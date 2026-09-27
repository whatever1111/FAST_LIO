#pragma once

#include <cstddef>
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
}  // namespace fast_lio_test

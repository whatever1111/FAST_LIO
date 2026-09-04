// adaptive_downsample.hpp — 让质量降级,而不是让延迟降级。
//
// 前端的单帧计算量随场景和地图规模变化:狗 2 在室内 20 ms,走到开阔室外(有效点翻倍、
// 地图持续增长)冲到几百毫秒。一旦越过扫描周期,队列就再也排不空 —— 延迟无上限增长,
// 而估计本身其实还是好的。正确的取舍是降采样变粗、点数变少、精度略降,把成本压回预算内。
//
// Adaptive scan downsampling: when the per-scan compute crosses its budget the voxel grows,
// so the cost of a hard scene is fewer points (slightly worse accuracy) instead of unbounded
// latency. It steps back down once the scene relaxes, with hysteresis so it cannot oscillate.
//
// 纯 C++,无 ROS/Eigen/PCL。| Pure C++.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace fast_lio
{

struct AdaptiveDownsampleParams
{
  bool enabled = false;        ///< 默认关:只有实测会超预算的平台才开
  double budget_sec = 0.060;   ///< 单帧计算超过此值即算超预算(扫描周期的 60%)
  double release_sec = 0.035;  ///< 低于此值足够久才回退一档
  double step = 0.05;          ///< 每档体素增量 (m)
  double max_voxel = 0.50;     ///< 体素上限;base_voxel 是下限,来自 filter_size_surf
  int raise_after = 5;         ///< 连续超预算多少帧才升档(抗突发)
  int lower_after = 100;       ///< 连续低于释放线多少帧才降档(慢退,抗抖动)
};

struct AdaptiveDownsampleState
{
  double voxel = 0.0;  ///< 当前体素;0 = 尚未初始化
  int over_streak = 0;
  int under_streak = 0;
  std::uint32_t raises = 0;
  std::uint32_t lowers = 0;
};

/// 一帧一次。`scan_cost_sec` 是这一帧的**计算**耗时(不含排队),`base_voxel` 是 profile
/// 配置的 filter_size_surf(下限)。返回 true 表示体素变了,调用方需要重设 VoxelGrid。
/// One scan. Returns true when the voxel changed and the filter must be reconfigured.
inline bool updateAdaptiveDownsample(AdaptiveDownsampleState * state,
                                     double scan_cost_sec,
                                     double base_voxel,
                                     const AdaptiveDownsampleParams & params)
{
  if (state == nullptr || base_voxel <= 0.0)
    return false;
  if (state->voxel <= 0.0)
    state->voxel = base_voxel;
  if (!params.enabled || !std::isfinite(scan_cost_sec) || scan_cost_sec < 0.0)
    return false;

  const double ceiling = std::max(base_voxel, params.max_voxel);
  if (scan_cost_sec > params.budget_sec) {
    state->under_streak = 0;
    if (++state->over_streak >= params.raise_after && state->voxel < ceiling) {
      state->over_streak = 0;
      state->voxel = std::min(ceiling, state->voxel + params.step);
      ++state->raises;
      return true;
    }
    return false;
  }
  // 与基线的比较留 1 µm 容差:0.20-0.05 的浮点残差否则会让它在已经回到基线后再降一次,
  // 白白重设一次滤波器并打一行日志。| A micron of slack keeps FP residue from re-stepping.
  constexpr double kVoxelEps = 1e-6;
  if (scan_cost_sec < params.release_sec) {
    state->over_streak = 0;
    if (++state->under_streak >= params.lower_after && state->voxel > base_voxel + kVoxelEps) {
      state->under_streak = 0;
      const double lowered = state->voxel - params.step;
      state->voxel = lowered <= base_voxel + kVoxelEps ? base_voxel : lowered;
      ++state->lowers;
      return true;
    }
    return false;
  }
  // 预算与释放线之间是死区:既不升也不降,连胜计数一起清零。
  // The band between budget and release is a dead zone; neither streak survives it.
  state->over_streak = 0;
  state->under_streak = 0;
  return false;
}

}  // namespace fast_lio

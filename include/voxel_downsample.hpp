// voxel_downsample.hpp — 扫描降采样:一趟哈希占格,而不是全量排序。
//
// PCL 的 VoxelGrid 每帧要给每个点造一条 (体素索引, 点索引) 记录、对整张表 std::sort、
// 再分组求质心 —— O(n log n) 加两轮分配。扫描降采样每帧都跑,而每格真正要的只是一个
// 代表点,于是这里用开放寻址的哈希占格一趟扫完:每格保留**最靠近格心的真实测点**,
// 而不是被平均出来的、并不存在的质心。代表点是真实测量,强度/法向与坐标自洽。
//
// 哈希表跨帧复用(靠 epoch 计数免清表),输出直接写进调用方的缓冲,稳态零分配。
//
// One-pass hash-bucketed scan downsampling. PCL's VoxelGrid sorts every point's voxel index
// (O(n log n) plus two allocations) to average each cell; this keeps, per cell, the real
// measured point nearest its centre — a genuine measurement rather than a synthesised
// centroid. The table is reused across scans (epoch stamping instead of clearing) and the
// representatives are written straight into the caller's buffer, so a steady state allocates
// nothing.
//
// 纯 C++,无 ROS/Eigen/PCL。| Pure C++.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fast_lio
{

class VoxelGridHash
{
public:
  /// 体素边长 (m)。<=0 表示不降采样(原样透传)。| Leaf size; <=0 passes the scan through.
  void setLeaf(double leaf)
  {
    leaf_ = static_cast<float>(leaf);
    inv_leaf_ = (leaf > kMinLeaf) ? static_cast<float>(1.0 / leaf) : 0.0f;
  }

  double leaf() const { return static_cast<double>(leaf_); }

  /// in 的每个占用体素输出一个代表点(最靠近格心者),按体素首次出现顺序。
  /// 非有限坐标、以及远到装不进格索引的点被丢弃(与 PCL 的非有限点处理一致)。
  /// Representatives in first-seen voxel order; non-finite and out-of-range points are dropped.
  template <class PointT, class Alloc>
  void filter(const std::vector<PointT, Alloc> & in, std::vector<PointT, Alloc> & out)
  {
    out.clear();
    if (in.empty()) {
      return;
    }
    if (inv_leaf_ <= 0.0f) {
      out = in;
      return;
    }
    prepare(in.size());

    for (;;) {
      const size_t mask = table_.size() - 1;
      const size_t room = table_.size() / 2;
      size_t occupied = 0;
      bool overflow = false;
      out.clear();
      for (size_t n = 0; n < in.size(); ++n) {
        const PointT & p = in[n];
        const float x = static_cast<float>(p.x);
        const float y = static_cast<float>(p.y);
        const float z = static_cast<float>(p.z);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }
        const float gx = std::floor(x * inv_leaf_);
        const float gy = std::floor(y * inv_leaf_);
        const float gz = std::floor(z * inv_leaf_);
        if (std::fabs(gx) >= kMaxCoord || std::fabs(gy) >= kMaxCoord || std::fabs(gz) >= kMaxCoord) {
          continue;
        }
        const auto i = static_cast<int64_t>(gx);
        const auto j = static_cast<int64_t>(gy);
        const auto k = static_cast<int64_t>(gz);
        const int64_t key = ((i + kBias) << 42) | ((j + kBias) << 21) | (k + kBias);

        const float dx = x - (gx + 0.5f) * leaf_;
        const float dy = y - (gy + 0.5f) * leaf_;
        const float dz = z - (gz + 0.5f) * leaf_;
        const float d2 = dx * dx + dy * dy + dz * dz;

        size_t slot = static_cast<size_t>(mix(static_cast<uint64_t>(key))) & mask;
        while (table_[slot].key != kEmptyKey && table_[slot].key != key) {
          slot = (slot + 1) & mask;
        }
        Cell & cell = table_[slot];
        if (cell.key == kEmptyKey) {
          if (occupied == room) {  // load factor would pass 1/2: regrow and redo the scan
            overflow = true;
            break;
          }
          ++occupied;
          cell.key = key;
          cell.d2 = d2;
          cell.out_index = static_cast<uint32_t>(out.size());
          out.push_back(p);
        } else if (d2 < cell.d2) {
          cell.d2 = d2;
          out[cell.out_index] = p;
        }
      }
      if (!overflow) {
        last_occupied_ = occupied;
        return;
      }
      grow();
    }
  }

private:
  static constexpr int64_t kEmptyKey = INT64_MIN;

  /// 16 B/格,让整张表尽量待在 L2:随机探测的代价远大于哈希本身。
  /// 16 B per cell keeps the table in L2 — the random probe, not the hashing, is the cost.
  struct Cell
  {
    int64_t key = kEmptyKey;
    float d2 = 0.0f;
    uint32_t out_index = 0;
  };

  static constexpr double kMinLeaf = 1e-6;
  /// 格索引打包进 3×21 bit,留足量程:0.05 m 体素下 ±52 km。| ±52 km at a 0.05 m leaf.
  static constexpr float kMaxCoord = 1048576.0f;  // 2^20
  static constexpr int64_t kBias = 1048576;

  /// splitmix64 finalizer: the packed key's low bits alone would collide badly.
  static uint64_t mix(uint64_t v)
  {
    constexpr int kShift = 33;
    constexpr uint64_t kMul1 = 0xff51afd7ed558ccdULL;
    constexpr uint64_t kMul2 = 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> kShift;
    v *= kMul1;
    v ^= v >> kShift;
    v *= kMul2;
    v ^= v >> kShift;
    return v;
  }

  /// 表按**上一帧的占格数**定容(装载因子 <=1/2),而不是按点数 —— 一帧 2.4 万点只占
  /// 约 4 千格,按点数开表会白白撑到 L2 之外。装不下就翻倍重跑该帧(只可能发生在场景
  /// 突变的第一帧)。清表是一次 memset,比每格 epoch 分支便宜。
  /// Capacity follows the PREVIOUS scan's occupancy, not the point count: 24k points fill
  /// ~4k cells, and sizing by points pushes the table out of L2 for nothing. An overflow
  /// doubles and redoes that one scan; clearing is a single memset.
  void prepare(size_t n)
  {
    const size_t expect = (last_occupied_ > 0) ? last_occupied_ + last_occupied_ / 4 : n;
    size_t want = kMinTable;
    const size_t cap = 2 * n;
    while (want < 2 * expect && want < cap) {
      want <<= 1;
    }
    if (table_.size() != want) {
      table_.resize(want);
    }
    std::fill(table_.begin(), table_.end(), Cell{});
  }

  /// 占格数不会超过点数,所以翻倍必然收敛到装载因子 <=1/2。| Doubling always converges.
  void grow()
  {
    table_.assign(table_.size() * 2, Cell{});
    last_occupied_ = 0;
  }

  static constexpr size_t kMinTable = 1024;

  std::vector<Cell> table_;
  size_t last_occupied_ = 0;
  float leaf_ = 0.5f;
  float inv_leaf_ = 2.0f;
};

}  // namespace fast_lio

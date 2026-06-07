// Incremental Voxel map (iVox) — a drop-in replacement for the subset of the ikd-Tree
// API that FAST-LIO's laserMapping uses (option B). Hashed sparse voxel grid: O(1) insert,
// kNN = gather points from the query voxel + its NEARBY{6,18,26} neighbors and brute-force
// over that small candidate set. No tree, no rebuild, no background thread → the ikd-Tree
// rebuild stalls and add-spikes disappear, and per-query search is cache-friendly.
//
// Phase-0 NN-parity probe (vs PCL KdTreeFLANN exact kNN, FAST-LIO esti_plane, on the DD
// Livox grass map): NEARBY26 @res 0.5 gives 100% NN recall, 99.9% plane-validity agreement,
// plane-residual |exact-iVox| mean ~2e-5 m / max 0.043 m, and ~1.8-2.4x faster per query.
// REQUIREMENT confirmed there: use nearby_type >= 18 (NEARBY6 is inadequate).
//
// Concurrency: single-operator, like ikd-Tree. Nearest_Search is read-only and SAFE to call
// concurrently from many threads (FAST-LIO's h_share runs it under `omp parallel for`) — it
// only does const unordered_map lookups. AddPoints mutates the grid and MUST NOT overlap a
// search; FAST-LIO's async map worker already guarantees that via its join-before-search.
#pragma once

#include <pcl/point_types.h>
#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <robin_hood/robin_hood.h>  // cache-friendly open-addressing map (vs node-based std::unordered_map)
#include <utility>
#include <vector>

namespace lio_ivox {

template <typename PointType>
class IVox {
 public:
  using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

  IVox() { setNearby(26); }  // safe defaults; overridden by Init()

  // resolution: voxel edge (m); nearby_type: 6/18/26 (use >=18); per_voxel_cap: backstop
  // on points per voxel; max_voxels: LRU capacity (replaces ikd-Tree FOV box-deletion).
  void Init(float resolution, int nearby_type, int per_voxel_cap, std::size_t max_voxels) {
    res_ = resolution > 1e-4f ? resolution : 0.5f;
    inv_res_ = 1.0f / res_;
    per_voxel_cap_ = per_voxel_cap > 0 ? per_voxel_cap : 50;
    max_voxels_ = max_voxels > 0 ? max_voxels : 5000000u;
    setNearby(nearby_type);
  }

  // --- ikd-Tree-compatible surface used by laserMapping ---------------------------------
  // Match ikd-Tree's downsample-on-add: keep <=1 point per filter_size_map box. Without this
  // the map inflates ~3x vs ikd-Tree (map_incremental's need_add gate is not a strict dedup),
  // which slows every kNN. downsample_size_ is the fine (sub-voxel) dedup grid.
  void set_downsample_param(float leaf) {
    if (leaf > 1e-4f) { downsample_size_ = leaf; inv_ds_ = 1.0f / leaf; }
  }
  void Build(PointVector& points) { Add_Points(points, true); }

  int Add_Points(PointVector& points, bool downsample_on) {
    int added = 0;
    for (const auto& p : points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      // GLOBAL fine-grid (downsample_size_) dedup: <=1 point per fine box across the WHOLE
      // map (not per-voxel), so fine boxes straddling NN-voxel borders aren't double-kept.
      // This keeps res=0.5 NN voxels (best probe agreement) at ikd-Tree's 542k map size.
      Key fk{};
      if (downsample_on) {
        fk = fineKey(p);
        if (occupied_fine_.find(fk) != occupied_fine_.end()) continue;
      }
      const Key k = key(p);
      auto it = map_.find(k);
      bool was_added = false;
      if (it == map_.end()) {
        lru_.push_front(k);
        Voxel v;
        v.pts.reserve(8);
        v.pts.push_back(p);
        v.lru = lru_.begin();
        map_.emplace(k, std::move(v));
        was_added = true;
        if (map_.size() > max_voxels_) evictOldest();
      } else {
        Voxel& v = it->second;
        lru_.splice(lru_.begin(), lru_, v.lru);  // touch (most-recently-used)
        if (static_cast<int>(v.pts.size()) < per_voxel_cap_) {
          v.pts.push_back(p);
          was_added = true;
        }
      }
      if (was_added) {
        ++num_points_;
        ++added;
        if (downsample_on) occupied_fine_.insert(fk);
      }
    }
    return added;
  }

  // Fills `nearest` with up to k closest map points (ascending) and `sqdist` with their
  // squared distances — exactly what ikd-Tree::Nearest_Search returns. h_share rejects when
  // nearest.size() < k or sqdist[k-1] > 5, so returning <k on a sparse query is fine.
  void Nearest_Search(const PointType& point, int k, PointVector& nearest,
                      std::vector<float>& sqdist,
                      float max_dist = std::numeric_limits<float>::infinity()) const {
    const Key qk = key(point);
    const float max_sq =
        std::isinf(max_dist) ? std::numeric_limits<float>::infinity() : max_dist * max_dist;
    thread_local std::vector<std::pair<float, const PointType*>> cand;
    cand.clear();
    for (const auto& o : nbr_) {
      auto it = map_.find(Key{qk.x + o[0], qk.y + o[1], qk.z + o[2]});
      if (it == map_.end()) continue;
      for (const auto& p : it->second.pts) {
        const float dx = p.x - point.x, dy = p.y - point.y, dz = p.z - point.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= max_sq) cand.emplace_back(d2, &p);
      }
    }
    const int m = std::min(static_cast<int>(cand.size()), k);
    std::partial_sort(cand.begin(), cand.begin() + m, cand.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
    nearest.resize(m);
    sqdist.resize(m);
    for (int i = 0; i < m; ++i) {
      nearest[i] = *cand[i].second;
      sqdist[i] = cand[i].first;
    }
  }

  template <typename BoxT>
  int Delete_Point_Boxes(std::vector<BoxT>& /*boxes*/) { return 0; }  // FOV via LRU instead
  void acquire_removed_points(PointVector& /*out*/) {}                // caller discards it

  int size() const { return static_cast<int>(num_points_); }
  int validnum() const { return static_cast<int>(num_points_); }
  bool empty() const { return map_.empty(); }
  std::size_t numVoxels() const { return map_.size(); }

 private:
  struct Key {
    int x, y, z;
    bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return static_cast<std::size_t>((k.x * 73856093) ^ (k.y * 471943) ^ (k.z * 83492791));
    }
  };
  struct Voxel {
    PointVector pts;
    typename std::list<Key>::iterator lru;  // dependent type in a class template → typename
  };

  Key key(const PointType& p) const {
    return Key{static_cast<int>(std::floor(p.x * inv_res_)),
               static_cast<int>(std::floor(p.y * inv_res_)),
               static_cast<int>(std::floor(p.z * inv_res_))};
  }
  Key fineKey(const PointType& p) const {  // downsample_size_ grid for global dedup
    return Key{static_cast<int>(std::floor(p.x * inv_ds_)),
               static_cast<int>(std::floor(p.y * inv_ds_)),
               static_cast<int>(std::floor(p.z * inv_ds_))};
  }

  void setNearby(int n) {
    nbr_.clear();
    nbr_.push_back({0, 0, 0});
    const int f[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (auto& o : f) nbr_.push_back({o[0], o[1], o[2]});
    if (n >= 18) {
      const int e[12][3] = {{1, 1, 0},  {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
                            {1, 0, 1},  {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
                            {0, 1, 1},  {0, 1, -1}, {0, -1, 1}, {0, -1, -1}};
      for (auto& o : e) nbr_.push_back({o[0], o[1], o[2]});
    }
    if (n >= 26) {
      const int c[8][3] = {{1, 1, 1},   {1, 1, -1},  {1, -1, 1},  {1, -1, -1},
                           {-1, 1, 1},  {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
      for (auto& o : c) nbr_.push_back({o[0], o[1], o[2]});
    }
  }

  void evictOldest() {
    const Key old = lru_.back();
    lru_.pop_back();
    auto it = map_.find(old);
    if (it != map_.end()) {
      for (const auto& q : it->second.pts) occupied_fine_.erase(fineKey(q));  // free its fine boxes
      num_points_ -= it->second.pts.size();
      map_.erase(it);
    }
  }

  float res_ = 0.5f, inv_res_ = 2.0f;
  float downsample_size_ = 0.3f, inv_ds_ = 1.0f / 0.3f;  // fine dedup grid (filter_size_map)
  int per_voxel_cap_ = 50;
  std::size_t max_voxels_ = 5000000u;
  std::vector<std::array<int, 3>> nbr_;
  // Open-addressing flat map: ~3-5x faster lookups than std::unordered_map (the measured
  // bottleneck — 27 lookups/query). Safe here: search candidates point into each voxel's
  // pts HEAP buffer (stable across rehash), and no insertion runs during a search.
  robin_hood::unordered_flat_map<Key, Voxel, KeyHash> map_;
  robin_hood::unordered_flat_set<Key, KeyHash> occupied_fine_;  // global 1-pt-per-fine-box dedup
  std::list<Key> lru_;
  std::size_t num_points_ = 0;
};

}  // namespace lio_ivox

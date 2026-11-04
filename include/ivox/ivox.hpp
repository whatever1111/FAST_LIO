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
// The entry view follows the same rule: opening and closing it must not overlap an addition,
// and Nearest_Search_View is a search like any other.
#pragma once

#include <Eigen/Core>
#include <pcl/point_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <robin_hood/robin_hood.h>  // cache-friendly open-addressing map (vs node-based std::unordered_map)
#include <stdexcept>
#include <utility>
#include <vector>

namespace lio_ivox
{

template<typename PointType>
class IVox
{
public:
  using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

  IVox()
  {
    setNearby(26);  // safe defaults; overridden by Init()
    updateReserve();
  }

  // resolution: voxel edge (m); nearby_type: 6/18/26 (use >=18); per_voxel_cap: backstop
  // on points per voxel; max_voxels: LRU capacity (replaces ikd-Tree FOV box-deletion).
  void Init(float resolution, int nearby_type, int per_voxel_cap, std::size_t max_voxels)
  {
    res_ = resolution > 1e-4f ? resolution : 0.5f;
    inv_res_ = 1.0f / res_;
    per_voxel_cap_ = per_voxel_cap > 0 ? per_voxel_cap : 50;
    max_voxels_ = max_voxels > 0 ? max_voxels : 5000000u;
    setNearby(nearby_type);
    updateReserve();
  }

  // --- ikd-Tree-compatible surface used by laserMapping ---------------------------------
  // Match ikd-Tree's downsample-on-add: keep <=1 point per filter_size_map box. Without this
  // the map inflates ~3x vs ikd-Tree (map_incremental's need_add gate is not a strict dedup),
  // which slows every kNN. downsample_size_ is the fine (sub-voxel) dedup grid.
  void set_downsample_param(float leaf)
  {
    if (leaf > 1e-4f) {
      downsample_size_ = leaf;
      inv_ds_ = 1.0f / leaf;
      updateReserve();
    }
  }
  // Rebuild the map from `points` alone. ikd-Tree's Build deletes the existing tree first,
  // and the caller that matters (the guard's "rebuild the local map here" recovery) depends
  // on that: without the clear, a map the estimate can no longer register against survives
  // the recovery it was supposed to be dropped by, and the log says "rebuilt" either way.
  void Build(PointVector & points)
  {
    clear();
    Add_Points(points, true);
    view_n_.clear();  // an open entry view reopens on the rebuilt map, the built points included
  }

  // Drop everything the live scans put here, keep the pinned tier (a preloaded prior map).
  // This is what the guard's recovery wants once a prior map is loaded: the local map that
  // no longer fits goes away, the map the operator delivered stays.
  void clearLive()
  {
    for (const Key & k : lru_) {
      auto it = map_.find(k);
      if (it == map_.end()) {
        continue;
      }
      for (const auto & q : it->second.pts)
        eraseOwnedFine(q, it->first);
      num_points_ -= it->second.pts.size();
      map_.erase(it);
    }
    lru_.clear();
  }

  // Drop the pinned tier, keep what the live scans put here. The counterpart of
  // clearLive(): a prior map installed at one alignment has to go away whole before one
  // installed at a better alignment takes its place, or the two sit in the map together
  // and every plane fit gets to choose between them.
  void clearPinned()
  {
    for (auto it = map_.begin(); it != map_.end();) {
      if (!it->second.pinned) {
        ++it;
        continue;
      }
      for (const auto & q : it->second.pts)
        eraseOwnedFine(q, it->first);
      num_points_ -= it->second.pts.size();
      it = map_.erase(it);
    }
    pinned_voxels_ = 0;
  }

  void clear()
  {
    map_.clear();
    occupied_fine_.clear();
    lru_.clear();
    num_points_ = 0;
    pinned_voxels_ = 0;
    view_n_.clear();  // an open entry view reopens on the empty map
  }

  // Add points that must survive both eviction and clearLive(): a prior map is not something
  // this run learned, so nothing this run does may drop it.
  // Conservative coarse demand, independent of fine-grid first-point-wins.
  // Both checks are read-only and stop as soon as capacity is exceeded.
  bool canAddPinnedPoints(const PointVector & points) const { return pinnedDemandFits(points, false); }
  bool canReplacePinnedPoints(const PointVector & points) const { return pinnedDemandFits(points, true); }
  int AddPinnedPoints(PointVector & points)
  {
    if (!canAddPinnedPoints(points))
      throw std::length_error("iVox pinned coarse-voxel demand exceeds capacity");
    return addPoints(points, true, true);
  }

  int Add_Points(PointVector & points, bool downsample_on) { return addPoints(points, downsample_on, false); }

  // --- Entry view --------------------------------------------------------------------------
  // The map as it stood when the view was opened, kept while the map itself goes on growing:
  // what laserMapping qualifies a recovery against, so that the scans being qualified cannot
  // vouch for themselves with points they added. A voxel only ever appends points and is only
  // ever removed whole, so what it held at the opening is a prefix of its points. An open view
  // records that prefix's length the first time a voxel grows, 0 for a voxel created after the
  // opening, and nothing for a voxel that has not grown. A removed voxel leaves the view with
  // the map; clear() and Build() reopen an open view on the new map (Build's points are in it).
  // A pinned tier is another run's map: the view does not open while one is loaded, and a
  // pinned addition while it is open is kept out of it like any later point and reported by
  // entryViewPinned().
  bool BeginEntryView()
  {
    EndEntryView();
    if (pinned_voxels_ > 0)
      return false;
    view_active_ = true;
    return true;
  }
  void EndEntryView()
  {
    view_active_ = false;
    view_pinned_ = false;
    view_frames_ = 0;
    robin_hood::unordered_flat_map<Key, std::size_t, KeyHash>().swap(view_n_);  // release the table
  }
  bool entryViewActive() const { return view_active_; }
  bool entryViewPinned() const { return view_pinned_; }
  // The caller's count of the scans it inserted while the view was open; the map itself counts
  // points, not scans. An addition never touches it, so it belongs to the caller's thread alone.
  void countEntryViewFrame()
  {
    if (view_active_)
      ++view_frames_;
  }
  std::size_t entryViewFrames() const { return view_frames_; }

private:
  int addPoints(PointVector & points, bool downsample_on, bool pinned)
  {
    int added = 0;
    for (const auto & p : points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        continue;
      if (!representable(p, inv_res_) || !representable(p, inv_ds_))
        continue;
      // GLOBAL fine-grid (downsample_size_) dedup: <=1 point per fine box across the WHOLE
      // map (not per-voxel), so fine boxes straddling NN-voxel borders aren't double-kept.
      // This keeps res=0.5 NN voxels (best probe agreement) at ikd-Tree's 542k map size.
      //
      // First point wins, and that includes a live point holding a box a later PINNED add
      // wants: a prior map fills what this run has not already seen, and does not displace
      // what it has. That is the right way round — the live points there came from the
      // scans being matched right now, so they are exactly consistent with them, while the
      // prior map's are from another run and another trajectory — but it has a consequence
      // worth knowing: the pinned tier is not a complete copy of the delivered map, and
      // pinnedVoxels() counts what went in, not what was offered. AddPinnedPoints returns
      // the same count, so a caller can report the shortfall.
      Key fk{};
      if (downsample_on) {
        fk = fineKey(p);
        if (occupied_fine_.find(fk) != occupied_fine_.end())
          continue;
      }
      const Key k = key(p);
      auto it = map_.find(k);
      bool was_added = false;
      if (it == map_.end()) {
        if (map_.size() >= max_voxels_) {
          if (lru_.empty())
            continue;  // no evictable live voxel: do not count or claim this point
          evictOldest();
        }
        Voxel v;
        v.pts.reserve(pts_reserve_);
        v.pts.push_back(p);
        v.pinned = pinned;
        if (pinned) {
          ++pinned_voxels_;
        } else {
          lru_.push_front(k);
          v.lru = lru_.begin();
        }
        map_.emplace(k, std::move(v));
        was_added = true;
        if (view_active_) {
          view_n_[k] = 0;  // created after the opening: none of it is in the view
          view_pinned_ = view_pinned_ || pinned;
        }
      } else {
        Voxel & v = it->second;
        if (pinned && !v.pinned) {  // a live voxel the prior map also covers becomes pinned
          lru_.erase(v.lru);
          v.pinned = true;
          ++pinned_voxels_;
          view_pinned_ = view_pinned_ || view_active_;
        }
        if (!v.pinned)
          lru_.splice(lru_.begin(), lru_, v.lru);  // touch (most-recently-used)
        if (static_cast<int>(v.pts.size()) < per_voxel_cap_) {
          if (view_active_)
            view_n_.emplace(k, v.pts.size());  // the first growth since the opening keeps the old length
          v.pts.push_back(p);
          was_added = true;
        }
      }
      if (was_added) {
        ++num_points_;
        ++added;
        if (downsample_on)
          occupied_fine_.emplace(fk, k);
      }
    }
    return added;
  }

public:
  // Fills `nearest` with up to k closest map points (ascending) and `sqdist` with their
  // squared distances — exactly what ikd-Tree::Nearest_Search returns. h_share rejects when
  // nearest.size() < k or sqdist[k-1] > 5, so returning <k on a sparse query is fine.
  void Nearest_Search(const PointType & point,
                      int k,
                      PointVector & nearest,
                      std::vector<float> & sqdist,
                      float max_dist = std::numeric_limits<float>::infinity()) const
  {
    search(point, k, nearest, sqdist, max_dist, false);
  }
  // Nearest_Search over the entry view: every voxel up to the length it had when the view was
  // opened. Without an open view it searches the whole map, as Nearest_Search does.
  void Nearest_Search_View(const PointType & point,
                           int k,
                           PointVector & nearest,
                           std::vector<float> & sqdist,
                           float max_dist = std::numeric_limits<float>::infinity()) const
  {
    search(point, k, nearest, sqdist, max_dist, view_active_);
  }

private:
  void search(const PointType & point,
              int k,
              PointVector & nearest,
              std::vector<float> & sqdist,
              float max_dist,
              bool in_view) const
  {
    nearest.clear();
    sqdist.clear();
    if (k <= 0 || !representable(point, inv_res_))
      return;
    const Key qk = key(point);
    const float max_sq = std::isinf(max_dist) ? std::numeric_limits<float>::infinity() : max_dist * max_dist;
    thread_local std::vector<std::pair<float, const PointType *>> cand;
    cand.clear();
    for (const auto & o : nbr_) {
      const std::int64_t x = static_cast<std::int64_t>(qk.x) + o[0];
      const std::int64_t y = static_cast<std::int64_t>(qk.y) + o[1];
      const std::int64_t z = static_cast<std::int64_t>(qk.z) + o[2];
      if (x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max() ||
          y < std::numeric_limits<int>::min() || y > std::numeric_limits<int>::max() ||
          z < std::numeric_limits<int>::min() || z > std::numeric_limits<int>::max())
        continue;
      auto it = map_.find(Key{static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)});
      if (it == map_.end())
        continue;
      const PointVector & pts = it->second.pts;
      std::size_t visible = pts.size();
      if (in_view) {
        const auto recorded = view_n_.find(it->first);
        if (recorded != view_n_.end())
          visible = recorded->second;
      }
      for (std::size_t i = 0; i < visible; ++i) {
        const PointType & p = pts[i];
        const float dx = p.x - point.x, dy = p.y - point.y, dz = p.z - point.z;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 <= max_sq)
          cand.emplace_back(d2, &p);
      }
    }
    const int m = std::min(static_cast<int>(cand.size()), k);
    std::partial_sort(
      cand.begin(), cand.begin() + m, cand.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    nearest.resize(m);
    sqdist.resize(m);
    for (int i = 0; i < m; ++i) {
      nearest[i] = *cand[i].second;
      sqdist[i] = cand[i].first;
    }
  }

public:
  template<typename BoxT>
  int Delete_Point_Boxes(std::vector<BoxT> & /*boxes*/)
  {
    return 0;
  }                                                      // FOV via LRU instead
  void acquire_removed_points(PointVector & /*out*/) {}  // caller discards it

  int size() const { return static_cast<int>(num_points_); }
  int validnum() const { return static_cast<int>(num_points_); }
  bool empty() const { return map_.empty(); }
  std::size_t numVoxels() const { return map_.size(); }

  // --- Capacity accounting ---------------------------------------------------------------
  // Reaching max_voxels_ is not an error by itself (LRU is how this backend bounds memory),
  // but it is silent: the oldest-WRITTEN voxel is dropped with no trace, and "oldest written"
  // is not "furthest away" — Nearest_Search never touches the LRU, so a region the robot is
  // standing in front of is evicted just the same if nothing has been added to it lately.
  // Pinned additions are preflighted atomically against coarse capacity; live
  // insertions may evict only older live voxels. Counters describe retained state.
  std::size_t maxVoxels() const { return max_voxels_; }
  std::size_t evictedVoxels() const { return evicted_voxels_; }
  std::size_t pinnedVoxels() const { return pinned_voxels_; }
  bool atCapacity() const { return map_.size() >= max_voxels_; }

private:
  struct Key
  {
    int x, y, z;
    bool operator==(const Key & o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct KeyHash
  {
    std::size_t operator()(const Key & k) const
    {
      // Convert before multiplying: ordinary voxel coordinates already overflow
      // a signed int (x = 30 at the deployed 0.3 m resolution is only 9 m).
      // Unsigned 64-bit arithmetic also defines the wraparound for negative keys.
      constexpr std::uint64_t kXPrime = 73856093;
      constexpr std::uint64_t kYPrime = 471943;
      constexpr std::uint64_t kZPrime = 83492791;
      return static_cast<std::size_t>((static_cast<std::uint64_t>(k.x) * kXPrime) ^
                                      (static_cast<std::uint64_t>(k.y) * kYPrime) ^
                                      (static_cast<std::uint64_t>(k.z) * kZPrime));
    }
  };
  struct Voxel
  {
    PointVector pts;
    typename std::list<Key>::iterator lru;  // dependent type in a class template → typename
    bool pinned = false;                    // prior-map voxel: never evicted, survives clearLive()
  };

  bool representable(const PointType & point, float inverse) const
  {
    // Preserve the existing float grid multiplication. Compare its floored
    // result in double: converting INT_MAX to float would round it out of range.
    for (float coordinate : {point.x, point.y, point.z}) {
      const double cell = std::floor(coordinate * inverse);
      if (!std::isfinite(cell) || cell < static_cast<double>(std::numeric_limits<int>::min()) ||
          cell > static_cast<double>(std::numeric_limits<int>::max()))
        return false;
    }
    return true;
  }

  void eraseOwnedFine(const PointType & point, const Key & owner)
  {
    const auto found = occupied_fine_.find(fineKey(point));
    if (found != occupied_fine_.end() && found->second == owner)
      occupied_fine_.erase(found);
  }

  bool pinnedDemandFits(const PointVector & points, bool replacement) const
  {
    robin_hood::unordered_flat_set<Key, KeyHash> needed;
    if (!replacement) {
      for (const auto & entry : map_) {
        if (entry.second.pinned)
          needed.insert(entry.first);
      }
    }
    for (const auto & point : points) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        continue;
      if (!representable(point, inv_res_) || !representable(point, inv_ds_))
        return false;
      needed.insert(key(point));
      if (needed.size() > max_voxels_)
        return false;
    }
    return true;
  }

  Key key(const PointType & p) const
  {
    return Key{static_cast<int>(std::floor(p.x * inv_res_)),
               static_cast<int>(std::floor(p.y * inv_res_)),
               static_cast<int>(std::floor(p.z * inv_res_))};
  }
  Key fineKey(const PointType & p) const
  {  // downsample_size_ grid for global dedup
    return Key{static_cast<int>(std::floor(p.x * inv_ds_)),
               static_cast<int>(std::floor(p.y * inv_ds_)),
               static_cast<int>(std::floor(p.z * inv_ds_))};
  }

  // Points a voxel can actually come to hold: the global fine-grid dedup keeps at most one
  // point per downsample_size_ box, so a voxel spans (res_/downsample_size_)^3 of them. A
  // profile that aligns the NN voxel with the dedup grid therefore stores ONE point per
  // voxel, and a fixed reserve of 8 would hand each of them an eight-slot buffer — at the
  // map's voxel ceiling that dwarfs every other per-voxel cost. Points added with
  // downsample_on = false bypass the dedup, so this is a starting size and never a cap:
  // the vector grows for the few voxels that exceed it, and per_voxel_cap_ is what bounds them.
  void updateReserve()
  {
    // The epsilon keeps an exact integer ratio (0.3/0.3, 0.5/0.25) from rounding up on float noise.
    const float boxes_per_edge = std::max(1.0f, std::ceil(res_ / downsample_size_ - 1e-4f));
    const double slots = static_cast<double>(boxes_per_edge) * boxes_per_edge * boxes_per_edge;
    pts_reserve_ = static_cast<std::size_t>(std::min(slots, static_cast<double>(per_voxel_cap_)));
  }

  void setNearby(int n)
  {
    nbr_.clear();
    nbr_.push_back({0, 0, 0});
    const int f[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (auto & o : f)
      nbr_.push_back({o[0], o[1], o[2]});
    if (n >= 18) {
      const int e[12][3] = {{1, 1, 0},
                            {1, -1, 0},
                            {-1, 1, 0},
                            {-1, -1, 0},
                            {1, 0, 1},
                            {1, 0, -1},
                            {-1, 0, 1},
                            {-1, 0, -1},
                            {0, 1, 1},
                            {0, 1, -1},
                            {0, -1, 1},
                            {0, -1, -1}};
      for (auto & o : e)
        nbr_.push_back({o[0], o[1], o[2]});
    }
    if (n >= 26) {
      const int c[8][3] = {
        {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}, {-1, 1, 1}, {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};
      for (auto & o : c)
        nbr_.push_back({o[0], o[1], o[2]});
    }
  }

  void evictOldest()
  {
    const Key old = lru_.back();
    lru_.pop_back();
    ++evicted_voxels_;
    auto it = map_.find(old);
    if (it != map_.end()) {
      for (const auto & q : it->second.pts)
        eraseOwnedFine(q, it->first);  // free its fine boxes
      num_points_ -= it->second.pts.size();
      map_.erase(it);
    }
  }

  float res_ = 0.5f, inv_res_ = 2.0f;
  float downsample_size_ = 0.3f, inv_ds_ = 1.0f / 0.3f;  // fine dedup grid (filter_size_map)
  int per_voxel_cap_ = 50;
  std::size_t pts_reserve_ = 1;  // derived from res_ / downsample_size_ by updateReserve()
  std::size_t max_voxels_ = 5000000u;
  std::vector<std::array<int, 3>> nbr_;
  // Open-addressing flat map: ~3-5x faster lookups than std::unordered_map (the measured
  // bottleneck — 27 lookups/query). Safe here: search candidates point into each voxel's
  // pts HEAP buffer (stable across rehash), and no insertion runs during a search.
  robin_hood::unordered_flat_map<Key, Voxel, KeyHash> map_;
  robin_hood::unordered_flat_map<Key, Key, KeyHash> occupied_fine_;  // global 1-pt-per-fine-box dedup
  std::list<Key> lru_;
  std::size_t num_points_ = 0;
  std::size_t evicted_voxels_ = 0;
  std::size_t pinned_voxels_ = 0;
  // Entry view: the points each voxel that grew since the opening had then; see BeginEntryView().
  robin_hood::unordered_flat_map<Key, std::size_t, KeyHash> view_n_;
  bool view_active_ = false;
  bool view_pinned_ = false;
  std::size_t view_frames_ = 0;
};

}  // namespace lio_ivox

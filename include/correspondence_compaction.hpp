#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fast_lio
{

// Check before narrowing to the estimator's int-indexed loops. A rejected size
// leaves the current scan's storage and count unchanged.
inline bool prepareCorrespondenceScratch(std::size_t count,
                                         std::vector<std::uint8_t> & selected,
                                         std::vector<float> & residuals,
                                         int & scanCount)
{
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return false;
  selected.resize(count, std::uint8_t{0});
  residuals.resize(count, 0.0f);
  scanCount = static_cast<int>(count);
  return true;
}

// Compact admitted rows without changing selection, row order or weights. The observer
// sees each selected source index in that same order (residual sums / diagnostics).
// Input and output clouds must be distinct; check before clearing reusable outputs.
// The caller owns the observer's input bounds and must not mutate these containers
// from the observer. Cloud::push_back maintains the cloud's width/height as well as size.
template<class Cloud, class Selection, class Observer>
std::size_t compactSelectedCorrespondences(std::size_t count,
                                           const Cloud & sourcePoints,
                                           const Cloud & sourceNormals,
                                           const Selection & selected,
                                           const std::vector<double> & sourceWeights,
                                           bool weighted,
                                           Cloud & points,
                                           Cloud & normals,
                                           std::vector<double> & weights,
                                           Observer onSelected)
{
  const auto selectionSize = static_cast<std::size_t>(std::distance(std::begin(selected), std::end(selected)));
  if (count > sourcePoints.size() || count > sourceNormals.size() || count > selectionSize ||
      (weighted && count > sourceWeights.size())) {
    throw std::invalid_argument("correspondence compaction inputs shorter than count");
  }
  if (&points == &sourcePoints || &points == &sourceNormals || &normals == &sourcePoints ||
      &normals == &sourceNormals || &points == &normals || &weights == &sourceWeights) {
    throw std::invalid_argument("correspondence compaction inputs and outputs must be distinct");
  }
  points.clear();
  normals.clear();
  weights.clear();
  points.points.reserve(count);
  normals.points.reserve(count);
  weights.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (!selected[i]) {
      continue;
    }
    points.push_back(sourcePoints[i]);
    normals.push_back(sourceNormals[i]);
    weights.push_back(weighted ? sourceWeights[i] : 1.0);
    onSelected(i);
  }
  return points.size();
}

}  // namespace fast_lio

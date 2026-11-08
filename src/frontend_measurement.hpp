// frontend_measurement.hpp — the front end's lidar measurement model, as the object the filter calls through its
// context entry.
#pragma once

#include "frontend_params.hpp"
#include "ikfom_types.hpp"

namespace fast_lio
{
/// The lidar measurement model of the iterated update. The node holds one by value and binds it with
/// init_dyn_share_context; the filter keeps its address until the node clears that binding, so it is neither copied
/// nor moved.
class FrontendMeasurement
{
public:
  /// params is the node's configuration, read from the first evaluation on: the node passes it before reading its
  /// parameters, and it must outlive this object.
  explicit FrontendMeasurement(const FrontendParams & params) : params_(params) {}
  ~FrontendMeasurement() = default;
  FrontendMeasurement(const FrontendMeasurement &) = delete;
  FrontendMeasurement & operator=(const FrontendMeasurement &) = delete;
  FrontendMeasurement(FrontendMeasurement &&) = delete;
  FrontendMeasurement & operator=(FrontendMeasurement &&) = delete;

  /// One evaluation at the filter's current iterate: fills data's rows, or sets data.valid to false when no point
  /// is usable.
  void evaluate(state_ikfom & state, esekfom::dyn_share_datastruct<double> & data);

private:
  const FrontendParams & params_;
};
}  // namespace fast_lio

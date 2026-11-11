#pragma once

#include <memory>

#include "frontend_source_contract.hpp"

namespace fastlio::joint_source
{

// Every row is a fixed linear likelihood with explicitly independent positive
// scalar remainder. Latent-plane/EIV/selection/random-H adapters are external.
struct LinearFactor
{
  ModelDomain domain = ModelDomain::kUnknown;
  FactorIdentity identity;
  std::string model;
  std::string linearization;
  Coordinates active;
  std::vector<std::string> sourceKeys;  // ordered blocks, aliases canonicalized
  std::vector<Coordinates> sourceCoordinates;
  std::vector<std::string> rowUnits;
  MomentKnowledge remainderKnowledge = MomentKnowledge::kUnknown;
  bool independentRemainder = false;
  Eigen::MatrixXd h;
  Eigen::MatrixXd a;
  Eigen::VectorXd observation;
  Eigen::VectorXd remainderMean;
  Eigen::VectorXd remainderVariance;
};

// Cov([old joint; noise]) must be complete and PSD. A known structural zero
// noise/cross is legal here. Measurement remainders have a stricter D > 0 domain.
// Zero priorCross reuses the retained prior root. Nonzero priorCross requires
// covariance refactorization, which can reject a rounded singular prior.
struct ProcessNoise
{
  MomentKnowledge knowledge = MomentKnowledge::kUnknown;
  // Required bounded, nonempty validation input. The caller retains the
  // supporting evidence outside the numerical state.
  std::string evidence;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  Eigen::MatrixXd priorCross;
};

struct GainResult
{
  Status status = Status::kUnknownMoment;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
};

// Shared process-local identity encoding and the consider-class covariance
// audit: exact symmetry, no negative variance, an exactly zero row for an
// exactly zero variance, and positive semidefiniteness within the v1 arithmetic
// budget 64 * dimension * eps * maxAbs. These helpers do not select an
// estimator, change capacities, or relax the v1 covariance domain.
namespace detail
{
Status auditCovariance(const Eigen::MatrixXd & covariance);
bool coordinatesEqual(const Coordinates & first, const Coordinates & second);
void appendCanonicalText(std::string & payload, const std::string & value);
void appendCanonicalCoordinates(std::string & payload, const Coordinates & coordinates);
void appendCanonicalMatrix(std::string & payload, const Eigen::MatrixXd & matrix);
}  // namespace detail

// Thread-confined owner. Callers serialize access. moments()/ledger() are borrowed
// views valid until the next mutation. Proposals own their immutable snapshots.
class JointSourceFilter
{
private:
  struct Snapshot;

public:
  JointSourceFilter() = default;
  ~JointSourceFilter() = default;
  JointSourceFilter(const JointSourceFilter &) = delete;
  JointSourceFilter & operator=(const JointSourceFilter &) = delete;

  class Proposal
  {
  public:
    Status status() const { return status_; }
    const JointMoments * posterior() const;

  private:
    friend class JointSourceFilter;
    Status status_ = Status::kUnknownMoment;
    std::shared_ptr<const Snapshot> prior_;
    std::shared_ptr<const Snapshot> candidate_;
  };

  Status initialize(const JointMoments & prior);
  const JointMoments * moments() const;
  const FactorLedger * ledger() const;
  std::size_t retiredSources() const;

  // Multiple proposals share one immutable prior. Only one successful commit
  // can replace it. Stale/cross-instance proposals cannot publish anything.
  Proposal propose(const std::vector<LinearFactor> & factors) const;
  Status commit(const Proposal & proposal);

  // T acts on the ENTIRE joint vector. Means transform as T*m + offset + m_w.
  Status propagate(const Eigen::MatrixXd & transform,
                   const Eigen::VectorXd & offset,
                   const ProcessNoise & noise,
                   const Coordinates & active,
                   const std::vector<Coordinates> & sourceCoordinates);
  Status reset(const Eigen::MatrixXd & transform,
               const Eigen::VectorXd & offset,
               const Coordinates & active,
               const std::vector<Coordinates> & sourceCoordinates);
  // New source = generator * oldJoint + offset + noise, including noise cross.
  Status augment(const Source & source,
                 const Eigen::MatrixXd & generator,
                 const Eigen::VectorXd & offset,
                 const ProcessNoise & noise);
  // Evidence is checked as bounded and nonempty. The caller retains it.
  Status marginalize(const std::vector<std::string> & sourceKeys, bool willNeverReuse, const std::string & evidence);

  GainResult appliedGain(const std::vector<LinearFactor> & factors, const Eigen::MatrixXd & activeGain) const;

private:
  std::shared_ptr<const Snapshot> current_;
};

}  // namespace fastlio::joint_source

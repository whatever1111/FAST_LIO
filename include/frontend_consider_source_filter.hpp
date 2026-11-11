#pragma once

#include <memory>

#include "frontend_correlated_update.hpp"

namespace fastlio::joint_source
{

struct Budget
{
  std::size_t sourceDof = 0;
  std::size_t liveSources = 0;
  std::size_t sourcesPerBatch = 0;
  std::size_t rowsPerBatch = 0;
  std::size_t recentLedger = 0;
  std::size_t memoryBytes = 0;
};

struct FactorLoading
{
  std::string factor;
  Eigen::MatrixXd loading;
};

// A centred echo error, separate from the strictly positive fresh remainder D.
// covariance is the UNINFLATED covariance. The kernel forms r*covariance once.
// A new (raw,frame) copy is explicitly independent of the old modelled joint;
// repeated uses of that same copy retain both active and source cross moments.
// The evidence must support that independent-copy upper-bound construction from
// the first query. The owner can name a source that will be cloned after query.
struct ReuseNoise
{
  ReuseUse use;
  Coordinates coordinates;
  MomentKnowledge knowledge = MomentKnowledge::kUnknown;
  std::string evidence;
  bool independentAtBirth = false;
  Eigen::MatrixXd covariance;
  std::vector<FactorLoading> factors;
};

// Cov(process noise, retained inflated copy). All retained copies must be named
// explicitly, including structural zeros; priorCross=0 alone does not imply it.
struct CopyNoiseCross
{
  RawId raw;
  RawId frame;
  Eigen::MatrixXd covariance;
};

struct SensitivityLoading
{
  std::string factor;
  std::string sensitivity;
  Eigen::MatrixXd loading;
};

struct ProcessSensitivity
{
  std::string sensitivity;
  Eigen::MatrixXd loading;
};

// Thread-confined covariance companion for given production states and gains.
// It never estimates a new state or chooses an optimal gain. Means, observations
// and linearizations are validated/identified inputs, not a second trajectory.
// P and K advance; C changes only on exact active cloning or retirement.
class ConsiderSourceFilter
{
private:
  struct Impl;
  struct Prepared;

public:
  ConsiderSourceFilter();
  ~ConsiderSourceFilter();
  ConsiderSourceFilter(const ConsiderSourceFilter &) = delete;
  ConsiderSourceFilter & operator=(const ConsiderSourceFilter &) = delete;

  class Candidate
  {
  public:
    Candidate();
    ~Candidate();
    Candidate(Candidate &&) noexcept;
    Candidate & operator=(Candidate &&) noexcept;
    Candidate(const Candidate &) = delete;
    Candidate & operator=(const Candidate &) = delete;
    Status status() const { return status_; }
    const Eigen::MatrixXd * activeCovariance() const;

  private:
    friend class ConsiderSourceFilter;
    Status status_ = Status::kUnknownMoment;
    std::unique_ptr<Prepared> data_;
  };

  Status initialize(const JointMoments & active, const Budget & budget);
  Candidate proposeAppliedGain(const std::vector<LinearFactor> & factors,
                               const Eigen::MatrixXd & activeGain,
                               const std::vector<ReuseNoise> & reuse = {},
                               const std::vector<SensitivityLoading> & sensitivities = {}) const;
  Status commit(Candidate && candidate);

  // Every registered unknown-amplitude source needs an explicit direct process
  // loading, including zero; known random Q does not establish that derivative.
  Status propagate(const Eigen::MatrixXd & phi,
                   const ProcessNoise & noise,
                   const std::vector<CopyNoiseCross> & copyCross = {},
                   const std::vector<ProcessSensitivity> & sensitivities = {});
  Status reset(const Eigen::MatrixXd & transport);
  Status augment(const Source & source, const Eigen::MatrixXd & activeGenerator);
  Status retire(const std::vector<std::string> & keys, bool willNeverReuse, const std::string & evidence);

  // Sensitivity to an unknown-amplitude, labelled constant. The complete initial
  // derivative of [e;sources] must be supplied, including explicit zeros. No
  // covariance or amplitude is inferred. Later factors must explicitly supply a
  // direct loading (possibly zero) for every registered sensitivity.
  Status registerSensitivity(const Source & source, const Eigen::MatrixXd & completeJointLoading);
  Status sensitivity(const std::string & key, Eigen::MatrixXd & out) const;
  Status sourceSensitivity(const std::string & source, const std::string & key, Eigen::MatrixXd & out) const;
  const std::vector<Source> & sensitivitySources() const;

  const Eigen::MatrixXd * activeCovariance() const;
  Status cross(const std::string & key, Eigen::MatrixXd & out) const;
  Status sourceCovariance(const std::string & first, const std::string & second, Eigen::MatrixXd & out) const;
  MomentKnowledge knowledge() const;
  std::uint64_t revision() const;
  std::uint64_t reuseCount(const RawId & raw) const;
  std::size_t memoryBytes() const;
  // Exact metadata bytes for audit/atomicity, excluding covariance matrices. Not
  // a wire format and not a substitute for the labelled unknown sensitivities.
  std::string ledgerIdentity() const;

private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace fastlio::joint_source

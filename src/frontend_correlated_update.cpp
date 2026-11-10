#include "frontend_correlated_update.hpp"

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include <cmath>
#include <limits>

namespace fastlio::joint_source
{
namespace
{

// This is an arithmetic backward-error budget, never covariance inflation.
// Negative LDLT pivots are rejected, including small negative pivots. No clipping.
double arithmeticTolerance(Eigen::Index dimension)
{
  return 64.0 * static_cast<double>(dimension) * std::numeric_limits<double>::epsilon();
}

Status covarianceRoot(const Eigen::MatrixXd & covariance, Eigen::MatrixXd & root)
{
  if (covariance.rows() != covariance.cols() || covariance.rows() == 0)
    return Status::kDimensionMismatch;
  if (!covariance.allFinite())
    return Status::kNonFinite;
  if (!(covariance.array() == covariance.transpose().array()).all())
    return Status::kInvalidCovariance;
  Eigen::LDLT<Eigen::MatrixXd> factor(covariance);
  if (factor.info() != Eigen::Success || (factor.vectorD().array() < 0.0).any())
    return Status::kInvalidCovariance;
  const Eigen::MatrixXd lower = factor.matrixL();
  root = factor.transpositionsP().transpose() * lower * factor.vectorD().cwiseSqrt().asDiagonal();
  if (!root.allFinite())
    return Status::kNumericalFailure;
  const double scale = covariance.cwiseAbs().maxCoeff();
  const double error = (root * root.transpose() - covariance).cwiseAbs().maxCoeff();
  if (!std::isfinite(error) || error > arithmeticTolerance(covariance.rows()) * scale)
    return Status::kInvalidCovariance;
  return Status::kOk;
}

Eigen::MatrixXd compactRoot(const Eigen::MatrixXd & root)
{
  const Eigen::Index dimension = root.rows();
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(root.transpose());
  const Eigen::MatrixXd upper = qr.matrixQR().topLeftCorner(dimension, dimension).triangularView<Eigen::Upper>();
  return upper.transpose();
}

Status validateMoments(const JointMoments & moments, Eigen::MatrixXd & root)
{
  const auto active = static_cast<Eigen::Index>(moments.active.components.size());
  if (active > kMaxActiveDof)
    return Status::kModelCapacityExceeded;
  if (!validCoordinates(moments.active))
    return Status::kInvalidIdentity;
  const auto status = validateSources(moments.sources);
  if (status != Status::kOk)
    return status;
  const Eigen::Index dimension = active + sourceDof(moments.sources);
  if (moments.knowledge != MomentKnowledge::kKnown || !validKey(moments.evidence))
    return Status::kUnknownMoment;
  if (moments.mean.size() != dimension || moments.covariance.rows() != dimension ||
      moments.covariance.cols() != dimension)
    return Status::kDimensionMismatch;
  if (!moments.mean.allFinite())
    return Status::kNonFinite;
  return covarianceRoot(moments.covariance, root);
}

const Source * resolveSource(const std::vector<Source> & sources, const std::string & key)
{
  for (const auto & source : sources) {
    if (source.key == key || contains(source.aliases, key))
      return &source;
  }
  return nullptr;
}

void appendText(std::string & payload, const std::string & value)
{
  payload += std::to_string(value.size()) + "/";
  payload.append(value);
}

void appendCoordinates(std::string & payload, const Coordinates & coordinates)
{
  appendText(payload, std::to_string(coordinates.components.size()));
  for (const auto & component : coordinates.components)
    appendText(payload, component);
  for (const auto & unit : coordinates.units)
    appendText(payload, unit);
  appendText(payload, coordinates.frame);
  appendText(payload, coordinates.tangent);
  appendText(payload, coordinates.epoch);
}

// Exact bytes are local transaction identity, not a portable wire format or hash.
void appendMatrix(std::string & payload, const Eigen::MatrixXd & matrix)
{
  appendText(payload, std::to_string(matrix.rows()));
  appendText(payload, std::to_string(matrix.cols()));
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double value = matrix(row, column);
      payload.append(reinterpret_cast<const char *>(&value), sizeof(value));
    }
  }
}

bool sameCoordinates(const Coordinates & first, const Coordinates & second)
{
  return first.components == second.components && first.units == second.units && first.frame == second.frame &&
         first.tangent == second.tangent && first.epoch == second.epoch;
}

Status validateBinding(const JointMoments & moments, const LinearFactor & factor)
{
  for (const auto & key : factor.sourceKeys) {
    if (!resolveSource(moments.sources, key))
      return Status::kSourceHistoryUnavailable;
  }
  if (!sameCoordinates(factor.active, moments.active) || factor.sourceKeys.size() != moments.sources.size())
    return Status::kDimensionMismatch;
  for (std::size_t i = 0; i < moments.sources.size(); ++i) {
    const auto * resolved = resolveSource(moments.sources, factor.sourceKeys[i]);
    if (resolved != &moments.sources[i] || !sameCoordinates(factor.sourceCoordinates[i], resolved->coordinates))
      return Status::kIdentityConflict;
  }
  return Status::kOk;
}

Status
canonicalizeSources(const std::vector<Source> & dictionary, const LinearFactor & factor, FactorIdentity & identity)
{
  for (auto & shared : identity.shared) {
    const auto * source = resolveSource(dictionary, shared.source);
    if (!source || !contains(source->rawAncestry, shared.raw))
      return Status::kSourceHistoryUnavailable;
    shared.source = source->key;
    Eigen::Index column = 0;
    bool covered = false;
    for (std::size_t i = 0; i < factor.sourceKeys.size(); ++i) {
      const auto count = static_cast<Eigen::Index>(factor.sourceCoordinates[i].components.size());
      if (resolveSource(dictionary, factor.sourceKeys[i]) == source && !factor.a.middleCols(column, count).isZero(0.0))
        covered = true;
      column += count;
    }
    if (!covered)
      return Status::kUnsupportedNoiseStructure;
  }
  return Status::kOk;
}

Status validateFactor(const std::vector<Source> & dictionary, const LinearFactor & factor, FactorIdentity & identity)
{
  const Eigen::Index rows = factor.observation.size();
  if (rows > kMaxRows || factor.active.components.size() > static_cast<std::size_t>(kMaxActiveDof) ||
      factor.sourceKeys.size() > static_cast<std::size_t>(kMaxSourceDof))
    return Status::kModelCapacityExceeded;
  if (factor.domain != ModelDomain::kLinearFixedModel)
    return Status::kUnsupportedModel;
  if (factor.remainderKnowledge != MomentKnowledge::kKnown)
    return Status::kUnknownMoment;
  if (!factor.independentRemainder)
    return Status::kUnsupportedNoiseStructure;
  if (!validCoordinates(factor.active) || factor.sourceCoordinates.size() != factor.sourceKeys.size())
    return Status::kDimensionMismatch;
  Eigen::Index sources = 0;
  for (const auto & coordinates : factor.sourceCoordinates) {
    if (!validCoordinates(coordinates))
      return Status::kInvalidIdentity;
    sources += static_cast<Eigen::Index>(coordinates.components.size());
  }
  if (sources > kMaxSourceDof)
    return Status::kModelCapacityExceeded;
  if (rows == 0 || factor.h.rows() != rows || factor.a.rows() != rows ||
      factor.h.cols() != static_cast<Eigen::Index>(factor.active.components.size()) || factor.a.cols() != sources ||
      factor.remainderMean.size() != rows || factor.remainderVariance.size() != rows ||
      factor.rowUnits.size() != static_cast<std::size_t>(rows))
    return Status::kDimensionMismatch;
  if (!factor.h.allFinite() || !factor.a.allFinite() || !factor.observation.allFinite() ||
      !factor.remainderMean.allFinite() || !factor.remainderVariance.allFinite())
    return Status::kNonFinite;
  if ((factor.remainderVariance.array() <= 0.0).any())
    return Status::kUnsupportedNoiseStructure;
  if (!validKey(factor.model) || !validKey(factor.linearization) ||
      !std::all_of(factor.rowUnits.begin(), factor.rowUnits.end(), validKey))
    return Status::kInvalidIdentity;
  identity = factor.identity;
  if (const auto status = validateIdentity(identity); status != Status::kOk)
    return status;
  if (const auto status = canonicalizeSources(dictionary, factor, identity); status != Status::kOk)
    return status;
  if (const auto status = validateIdentity(identity); status != Status::kOk)
    return status;
  std::string payload;
  appendText(payload, identity.payload);
  identity.payload = std::move(payload);
  appendText(identity.payload, factor.model);
  appendText(identity.payload, factor.linearization);
  appendCoordinates(identity.payload, factor.active);
  appendText(identity.payload, std::to_string(factor.sourceKeys.size()));
  for (std::size_t i = 0; i < factor.sourceKeys.size(); ++i) {
    const auto * source = resolveSource(dictionary, factor.sourceKeys[i]);
    if (!source)
      return Status::kSourceHistoryUnavailable;
    appendText(identity.payload, source->key);
    appendCoordinates(identity.payload, factor.sourceCoordinates[i]);
  }
  for (const auto & unit : factor.rowUnits)
    appendText(identity.payload, unit);
  appendMatrix(identity.payload, factor.h);
  appendMatrix(identity.payload, factor.a);
  appendMatrix(identity.payload, factor.observation);
  appendMatrix(identity.payload, factor.remainderMean);
  appendMatrix(identity.payload, factor.remainderVariance);
  return validateIdentity(identity);
}

Status validateBatch(const std::vector<Source> & dictionary,
                     const std::vector<LinearFactor> & factors,
                     std::vector<FactorIdentity> & identities)
{
  if (factors.size() > static_cast<std::size_t>(kMaxRows))
    return Status::kModelCapacityExceeded;
  Eigen::Index rows = 0;
  for (const auto & factor : factors) {
    rows += factor.observation.size();
    if (rows > kMaxRows)
      return Status::kModelCapacityExceeded;
    FactorIdentity identity;
    const auto status = validateFactor(dictionary, factor, identity);
    if (status != Status::kOk)
      return status;
    identities.push_back(std::move(identity));
  }
  return Status::kOk;
}

Eigen::MatrixXd jointLoading(const LinearFactor & factor)
{
  Eigen::MatrixXd loading(factor.h.rows(), factor.h.cols() + factor.a.cols());
  loading << factor.h, factor.a;
  return loading;
}

Status updateRows(const LinearFactor & factor, Eigen::VectorXd & mean, Eigen::MatrixXd & root)
{
  const Eigen::MatrixXd loading = jointLoading(factor);
  const Eigen::Index dimension = mean.size();
  for (Eigen::Index row = 0; row < loading.rows(); ++row) {
    const Eigen::VectorXd response = root.transpose() * loading.row(row).transpose();
    const double innovation = factor.observation(row) - loading.row(row).dot(mean) - factor.remainderMean(row);
    Eigen::MatrixXd array = Eigen::MatrixXd::Zero(dimension + 1, dimension + 1);
    array(0, 0) = std::sqrt(factor.remainderVariance(row));
    array.topRightCorner(1, dimension) = response.transpose();
    array.bottomRightCorner(dimension, dimension) = root;
    if (!array.allFinite() || !std::isfinite(innovation))
      return Status::kNumericalFailure;
    const Eigen::MatrixXd lower = compactRoot(array);
    if (!lower.allFinite() || lower(0, 0) == 0.0)
      return Status::kNumericalFailure;
    mean += (lower.bottomLeftCorner(dimension, 1) / lower(0, 0)) * innovation;
    root = lower.bottomRightCorner(dimension, dimension);
  }
  return mean.allFinite() && root.allFinite() ? Status::kOk : Status::kNumericalFailure;
}

Status jointNoiseRoot(const JointMoments & moments,
                      const Eigen::MatrixXd & priorRoot,
                      const ProcessNoise & noise,
                      Eigen::Index noiseDof,
                      Eigen::MatrixXd & root)
{
  if (noise.knowledge != MomentKnowledge::kKnown || !validKey(noise.evidence))
    return Status::kUnknownMoment;
  const Eigen::Index dimension = moments.mean.size();
  if (noise.mean.size() != noiseDof || noise.covariance.rows() != noiseDof || noise.covariance.cols() != noiseDof ||
      noise.priorCross.rows() != dimension || noise.priorCross.cols() != noiseDof)
    return Status::kDimensionMismatch;
  if (!noise.mean.allFinite())
    return Status::kNonFinite;
  // Preserve deterministic support in the retained root for independent noise.
  // Reconstructing and refactorizing the prior would introduce rounding pivots.
  if (noise.priorCross.isZero(0.0)) {
    Eigen::MatrixXd noiseRoot;
    if (const auto status = covarianceRoot(noise.covariance, noiseRoot); status != Status::kOk)
      return status;
    root = Eigen::MatrixXd::Zero(dimension + noiseDof, priorRoot.cols() + noiseDof);
    root.topLeftCorner(dimension, priorRoot.cols()) = priorRoot;
    root.bottomRightCorner(noiseDof, noiseDof) = noiseRoot;
    return Status::kOk;
  }
  Eigen::MatrixXd joint(dimension + noiseDof, dimension + noiseDof);
  joint << moments.covariance, noise.priorCross, noise.priorCross.transpose(), noise.covariance;
  return covarianceRoot(joint, root);
}

bool ancestryOverlaps(const Source & source, const Source & other)
{
  for (const auto & raw : source.rawAncestry) {
    if (contains(other.rawAncestry, raw))
      return true;
  }
  return false;
}

Status validateAncestry(const std::vector<Source> & live,
                        const std::vector<Source> & retired,
                        const FactorIdentity & canonical)
{
  for (const auto & raw : canonical.rawAncestry) {
    const auto carries = [&raw](const Source & source) {
      return contains(source.rawAncestry, raw);
    };
    if (std::any_of(retired.begin(), retired.end(), carries))
      return Status::kSourceHistoryUnavailable;
    if (!std::any_of(live.begin(), live.end(), carries))
      continue;
    const bool loaded = std::any_of(live.begin(), live.end(), [&](const Source & source) {
      return carries(source) &&
             std::find(canonical.shared.begin(), canonical.shared.end(), SharedRawSource{raw, source.key}) !=
               canonical.shared.end();
    });
    if (!loaded)
      return Status::kUnsupportedNoiseStructure;
  }
  return Status::kOk;
}

}  // namespace

struct JointSourceFilter::Snapshot
{
  JointMoments moments;
  Eigen::MatrixXd root;
  FactorLedger ledger;
  std::vector<Source> retired;

  std::vector<Source> dictionary() const
  {
    auto sources = moments.sources;
    sources.insert(sources.end(), retired.begin(), retired.end());
    return sources;
  }

  Status validateFreshFactors(const std::vector<LinearFactor> & factors,
                              const std::vector<FactorIdentity> & identities) const
  {
    for (std::size_t index = 0; index < factors.size(); ++index) {
      const auto & identity = identities[index];
      const auto previous = std::find_if(ledger.entries().begin(),
                                         ledger.entries().end(),
                                         [&](const FactorIdentity & entry) { return entry.key == identity.key; });
      if (previous != ledger.entries().end()) {
        if (!(*previous == identity))
          return Status::kIdentityConflict;
        continue;
      }
      if (const auto status = validateBinding(moments, factors[index]); status != Status::kOk)
        return status;
      if (const auto status = validateAncestry(moments.sources, retired, identity); status != Status::kOk)
        return status;
    }
    return Status::kOk;
  }

  Status finish()
  {
    if (!moments.mean.allFinite() || !root.allFinite())
      return Status::kNumericalFailure;
    // Copy one rounded triangle to both halves. GEMM may accumulate mirrored
    // entries differently, while subsequent inputs require exact symmetry.
    const Eigen::MatrixXd product = root * root.transpose();
    moments.covariance = product.selfadjointView<Eigen::Lower>();
    return moments.covariance.allFinite() ? Status::kOk : Status::kNumericalFailure;
  }
};

const JointMoments * JointSourceFilter::Proposal::posterior() const
{
  return candidate_ ? &candidate_->moments : nullptr;
}

Status JointSourceFilter::initialize(const JointMoments & prior)
{
  if (current_)
    return Status::kIdentityConflict;
  auto candidate = std::make_shared<Snapshot>();
  const auto status = validateMoments(prior, candidate->root);
  if (status != Status::kOk)
    return status;
  candidate->moments = prior;
  current_ = std::move(candidate);
  return Status::kOk;
}

const JointMoments * JointSourceFilter::moments() const
{
  return current_ ? &current_->moments : nullptr;
}
const FactorLedger * JointSourceFilter::ledger() const
{
  return current_ ? &current_->ledger : nullptr;
}
std::size_t JointSourceFilter::retiredSources() const
{
  return current_ ? current_->retired.size() : 0;
}

JointSourceFilter::Proposal JointSourceFilter::propose(const std::vector<LinearFactor> & factors) const
{
  Proposal proposal;
  if (!current_)
    return proposal;
  std::vector<FactorIdentity> identities;
  proposal.status_ = validateBatch(current_->dictionary(), factors, identities);
  if (proposal.status_ != Status::kOk)
    return proposal;
  // Exact identities bypass fresh-source checks. Fresh ancestry is validated
  // before cross-factor raw checks so retirement retains its specific reason.
  proposal.status_ = current_->validateFreshFactors(factors, identities);
  if (proposal.status_ != Status::kOk)
    return proposal;
  auto candidate = std::make_shared<Snapshot>(*current_);
  const auto consumed = candidate->ledger.consume(identities);
  proposal.status_ = consumed.status;
  if (consumed.status != Status::kOk && consumed.status != Status::kNoOp)
    return proposal;
  proposal.prior_ = current_;
  if (consumed.status == Status::kNoOp) {
    proposal.candidate_ = current_;
    return proposal;
  }
  for (const auto index : consumed.fresh) {
    proposal.status_ = updateRows(factors[index], candidate->moments.mean, candidate->root);
    if (proposal.status_ != Status::kOk)
      return proposal;
  }
  proposal.status_ = candidate->finish();
  if (proposal.status_ == Status::kOk)
    proposal.candidate_ = std::move(candidate);
  return proposal;
}

Status JointSourceFilter::commit(const Proposal & proposal)
{
  if (!proposal.candidate_)
    return proposal.status_;
  if (proposal.prior_ != current_)
    return Status::kStaleProposal;
  if (proposal.status_ == Status::kOk)
    current_ = proposal.candidate_;
  return proposal.status_;
}

Status JointSourceFilter::propagate(const Eigen::MatrixXd & transform,
                                    const Eigen::VectorXd & offset,
                                    const ProcessNoise & noise,
                                    const Coordinates & active,
                                    const std::vector<Coordinates> & sourceCoordinates)
{
  if (!current_)
    return Status::kUnknownMoment;
  const Eigen::Index dimension = current_->moments.mean.size();
  if (transform.rows() != dimension || transform.cols() != dimension || offset.size() != dimension ||
      active.components.size() != current_->moments.active.components.size() ||
      sourceCoordinates.size() != current_->moments.sources.size())
    return Status::kDimensionMismatch;
  if (!transform.allFinite() || !offset.allFinite())
    return Status::kNonFinite;
  auto candidate = std::make_shared<Snapshot>(*current_);
  candidate->moments.active = active;
  for (std::size_t i = 0; i < sourceCoordinates.size(); ++i) {
    if (sourceCoordinates[i].components.size() != candidate->moments.sources[i].coordinates.components.size())
      return Status::kDimensionMismatch;
    candidate->moments.sources[i].coordinates = sourceCoordinates[i];
  }
  if (!validCoordinates(active))
    return Status::kInvalidIdentity;
  if (const auto status = validateSources(candidate->moments.sources); status != Status::kOk)
    return status;
  Eigen::MatrixXd jointRoot;
  const auto status = jointNoiseRoot(current_->moments, current_->root, noise, dimension, jointRoot);
  if (status != Status::kOk)
    return status;
  Eigen::MatrixXd mapping(dimension, 2 * dimension);
  mapping << transform, Eigen::MatrixXd::Identity(dimension, dimension);
  candidate->root = compactRoot((mapping * jointRoot).eval());
  candidate->moments.mean = transform * current_->moments.mean + offset + noise.mean;
  if (const auto finished = candidate->finish(); finished != Status::kOk)
    return finished;
  current_ = std::move(candidate);
  return Status::kOk;
}

Status JointSourceFilter::reset(const Eigen::MatrixXd & transform,
                                const Eigen::VectorXd & offset,
                                const Coordinates & active,
                                const std::vector<Coordinates> & sourceCoordinates)
{
  if (!current_)
    return Status::kUnknownMoment;
  const Eigen::Index dimension = current_->moments.mean.size();
  ProcessNoise zero{MomentKnowledge::kKnown,
                    "deterministic reset",
                    Eigen::VectorXd::Zero(dimension),
                    Eigen::MatrixXd::Zero(dimension, dimension),
                    Eigen::MatrixXd::Zero(dimension, dimension)};
  return propagate(transform, offset, zero, active, sourceCoordinates);
}

Status JointSourceFilter::augment(const Source & source,
                                  const Eigen::MatrixXd & generator,
                                  const Eigen::VectorXd & offset,
                                  const ProcessNoise & noise)
{
  if (!current_)
    return Status::kUnknownMoment;
  const auto count = static_cast<Eigen::Index>(source.coordinates.components.size());
  if (sourceDof(current_->moments.sources) + count > kMaxSourceDof)
    return Status::kModelCapacityExceeded;
  for (const auto & retired : current_->retired) {
    if (retired.key == source.key || contains(retired.aliases, source.key) || ancestryOverlaps(retired, source))
      return Status::kSourceHistoryUnavailable;
    for (const auto & alias : source.aliases) {
      if (alias == retired.key || contains(retired.aliases, alias))
        return Status::kSourceHistoryUnavailable;
    }
  }
  for (const auto & entry : current_->ledger.entries()) {
    for (const auto & raw : source.rawAncestry) {
      const bool retained = std::any_of(current_->moments.sources.begin(),
                                        current_->moments.sources.end(),
                                        [&](const Source & live) { return contains(live.rawAncestry, raw); });
      if (contains(entry.rawAncestry, raw) && !retained)
        return Status::kSourceHistoryUnavailable;
    }
  }
  auto candidate = std::make_shared<Snapshot>(*current_);
  candidate->moments.sources.push_back(source);
  if (const auto status = validateSources(candidate->moments.sources); status != Status::kOk)
    return status;
  const Eigen::Index dimension = current_->moments.mean.size();
  if (generator.rows() != count || generator.cols() != dimension || offset.size() != count)
    return Status::kDimensionMismatch;
  if (!generator.allFinite() || !offset.allFinite())
    return Status::kNonFinite;
  Eigen::MatrixXd jointRoot;
  if (const auto status = jointNoiseRoot(current_->moments, current_->root, noise, count, jointRoot);
      status != Status::kOk)
    return status;
  Eigen::MatrixXd mapping = Eigen::MatrixXd::Identity(dimension + count, dimension + count);
  mapping.bottomLeftCorner(count, dimension) = generator;
  candidate->root = compactRoot((mapping * jointRoot).eval());
  candidate->moments.mean.resize(dimension + count);
  candidate->moments.mean.head(dimension) = current_->moments.mean;
  candidate->moments.mean.tail(count) = generator * current_->moments.mean + offset + noise.mean;
  if (const auto status = candidate->finish(); status != Status::kOk)
    return status;
  current_ = std::move(candidate);
  return Status::kOk;
}

Status JointSourceFilter::marginalize(const std::vector<std::string> & sourceKeys,
                                      bool willNeverReuse,
                                      const std::string & evidence)
{
  if (!current_)
    return Status::kUnknownMoment;
  if (sourceKeys.empty())
    return Status::kNoOp;
  if (!willNeverReuse || !validKey(evidence))
    return Status::kSourceHistoryUnavailable;
  if (!uniqueKeys(sourceKeys))
    return Status::kIdentityConflict;
  if (current_->retired.size() + sourceKeys.size() > kMaxLedger)
    return Status::kModelCapacityExceeded;
  auto candidate = std::make_shared<Snapshot>(*current_);
  std::vector<const Source *> removed;
  for (const auto & key : sourceKeys) {
    const auto * source = resolveSource(current_->moments.sources, key);
    if (!source)
      return Status::kSourceHistoryUnavailable;
    if (std::find(removed.begin(), removed.end(), source) != removed.end())
      return Status::kIdentityConflict;
    removed.push_back(source);
  }
  std::vector<Eigen::Index> kept;
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(current_->moments.active.components.size()); ++i)
    kept.push_back(i);
  candidate->moments.sources.clear();
  Eigen::Index column = static_cast<Eigen::Index>(kept.size());
  for (const auto & source : current_->moments.sources) {
    const auto count = static_cast<Eigen::Index>(source.coordinates.components.size());
    if (std::find(removed.begin(), removed.end(), &source) != removed.end())
      candidate->retired.push_back(source);
    else {
      candidate->moments.sources.push_back(source);
      for (Eigen::Index i = 0; i < count; ++i)
        kept.push_back(column + i);
    }
    column += count;
  }
  const auto dimension = static_cast<Eigen::Index>(kept.size());
  Eigen::MatrixXd selected(dimension, current_->root.cols());
  candidate->moments.mean.resize(dimension);
  for (Eigen::Index i = 0; i < dimension; ++i) {
    selected.row(i) = current_->root.row(kept[i]);
    candidate->moments.mean(i) = current_->moments.mean(kept[i]);
  }
  candidate->root = compactRoot(selected);
  if (const auto status = candidate->finish(); status != Status::kOk)
    return status;
  current_ = std::move(candidate);
  return Status::kOk;
}

GainResult JointSourceFilter::appliedGain(const std::vector<LinearFactor> & factors,
                                          const Eigen::MatrixXd & activeGain) const
{
  GainResult result;
  if (!current_)
    return result;
  std::vector<FactorIdentity> identities;
  result.status = validateBatch(current_->dictionary(), factors, identities);
  if (result.status != Status::kOk)
    return result;
  const auto active = static_cast<Eigen::Index>(current_->moments.active.components.size());
  Eigen::Index rows = 0;
  for (const auto & factor : factors)
    rows += factor.observation.size();
  if (activeGain.rows() != active || activeGain.cols() != rows) {
    result.status = Status::kDimensionMismatch;
    return result;
  }
  if (!activeGain.allFinite()) {
    result.status = Status::kNonFinite;
    return result;
  }
  // Validate independence/duplicates before constructing D. This helper does
  // not consume anything and accepts only a fresh independent factor batch.
  result.status = current_->validateFreshFactors(factors, identities);
  if (result.status != Status::kOk)
    return result;
  auto ledger = current_->ledger;
  const auto consumed = ledger.consume(identities);
  if (consumed.status != Status::kOk || consumed.fresh.size() != factors.size()) {
    result.status = Status::kUnsupportedNoiseStructure;
    return result;
  }
  Eigen::MatrixXd mapping = Eigen::MatrixXd::Identity(active, current_->moments.mean.size());
  Eigen::MatrixXd remainder(active, rows);
  result.mean = current_->moments.mean.head(active);
  Eigen::Index column = 0;
  for (std::size_t index = 0; index < factors.size(); ++index) {
    const auto & factor = factors[index];
    const auto count = factor.observation.size();
    const auto gain = activeGain.middleCols(column, count);
    const auto loading = jointLoading(factor);
    mapping -= gain * loading;
    remainder.middleCols(column, count) = gain * factor.remainderVariance.cwiseSqrt().asDiagonal();
    result.mean += gain * (factor.observation - loading * current_->moments.mean - factor.remainderMean);
    column += count;
  }
  const Eigen::MatrixXd errorRoot = mapping * current_->root;
  result.covariance = errorRoot * errorRoot.transpose() + remainder * remainder.transpose();
  result.status = result.mean.allFinite() && result.covariance.allFinite() ? Status::kOk : Status::kNumericalFailure;
  return result;
}

}  // namespace fastlio::joint_source

#include "frontend_lidar_joint_adapter.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/OrderingMethods>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <set>
#include <utility>

#include "IKFoM_toolkit/mtk/types/S2.hpp"

namespace fastlio::lidar_joint
{
namespace
{
constexpr Eigen::Index kRotation = 3, kExtrinsicRotation = 6, kExtrinsicTranslation = 9;
constexpr Eigen::Index kVelocity = 12, kGyroBias = 15, kAccelBias = 18, kGravity = 21;
constexpr double kArithmeticFactor = 32768.0;  // Four rounding terms per maximum-budget row, plus 23-D solves.
constexpr double kRankFactor = 64.0;
constexpr double kPoolMinimumDof = 60.0;
constexpr double kTypicalRangeSigma = 0.015, kWebsiteRangeSigma = 0.01;
constexpr double kQuarter = 0.25;
constexpr std::size_t kSparseEntryBytes = sizeof(double) + sizeof(int);
constexpr int kGravityNumerator = 98090, kGravityDenominator = 10000;
using Gravity = MTK::S2<double, kGravityNumerator, kGravityDenominator, 1>;
using Clock = std::chrono::steady_clock;
using PoseMatrix = Eigen::Matrix<double, kPoseDof, kPoseDof>;

std::uint64_t elapsed(Clock::time_point start)
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

void flag(std::uint32_t & flags, FrameFlag value)
{
  flags |= static_cast<std::uint32_t>(value);
}

double rankTolerance(Eigen::Index dimension, double scale)
{
  return kRankFactor * std::numeric_limits<double>::epsilon() *
         static_cast<double>(std::max<Eigen::Index>(1, dimension)) * scale;
}

double arithmeticTolerance(double condition = 1.0)
{
  return kArithmeticFactor * std::numeric_limits<double>::epsilon() * std::max(1.0, condition);
}

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d result;
  result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return result;
}

Gravity gravity(const State & state)
{
  Gravity result;
  result.vec = state.gravity;
  return result;
}

Vector23 difference(const State & left, const State & right)
{
  Vector23 result;
  result.head<3>() = left.position - right.position;
  MTK::vect<3, double> rotation, extrinsic;
  MTK::SO3<double>(left.rotation).boxminus(rotation, MTK::SO3<double>(right.rotation));
  MTK::SO3<double>(left.extrinsicRotation).boxminus(extrinsic, MTK::SO3<double>(right.extrinsicRotation));
  result.segment<3>(kRotation) = rotation;
  result.segment<3>(kExtrinsicRotation) = extrinsic;
  result.segment<3>(kExtrinsicTranslation) = left.extrinsicTranslation - right.extrinsicTranslation;
  result.segment<3>(kVelocity) = left.velocity - right.velocity;
  result.segment<3>(kGyroBias) = left.gyroBias - right.gyroBias;
  result.segment<3>(kAccelBias) = left.accelBias - right.accelBias;
  MTK::vect<2, double> direction;
  gravity(left).boxminus(direction, gravity(right));
  result.tail<2>() = direction;
  return result;
}

State plus(const State & state, const Vector23 & increment)
{
  State result = state;
  result.position += increment.head<3>();
  MTK::SO3<double> rotation(state.rotation), extrinsic(state.extrinsicRotation);
  MTK::vect<3, double> delta = increment.segment<3>(kRotation);
  rotation.boxplus(delta);
  delta = increment.segment<3>(kExtrinsicRotation);
  extrinsic.boxplus(delta);
  result.rotation = rotation;
  result.extrinsicRotation = extrinsic;
  result.extrinsicTranslation += increment.segment<3>(kExtrinsicTranslation);
  result.velocity += increment.segment<3>(kVelocity);
  result.gyroBias += increment.segment<3>(kGyroBias);
  result.accelBias += increment.segment<3>(kAccelBias);
  auto direction = gravity(state);
  MTK::vect<2, double> tangent = increment.tail<2>();
  direction.boxplus(tangent);
  result.gravity = direction.get_vect();
  return result;
}

Matrix23 transport(const State & from, const State & to, const Vector23 & delta)
{
  Matrix23 result = Matrix23::Identity();
  for (const Eigen::Index index : {kRotation, kExtrinsicRotation}) {
    MTK::vect<3, double> angle = delta.segment<3>(index);
    result.block<3, 3>(index, index) = MTK::A_matrix(angle).transpose();
  }
  Eigen::Matrix<double, 2, 3> nx;
  Eigen::Matrix<double, 3, 2> mx;
  auto newGravity = gravity(to), oldGravity = gravity(from);
  newGravity.S2_Nx_yy(nx);
  MTK::vect<2, double> direction = delta.tail<2>();
  oldGravity.S2_Mx(mx, direction);
  result.bottomRightCorner<2, 2>() = nx * mx;
  return result;
}

bool finiteState(const State & state)
{
  return state.position.allFinite() && state.rotation.coeffs().allFinite() &&
         state.extrinsicRotation.coeffs().allFinite() && state.extrinsicTranslation.allFinite() &&
         state.velocity.allFinite() && state.gyroBias.allFinite() && state.accelBias.allFinite() &&
         state.gravity.allFinite();
}

bool validState(const State & state)
{
  return std::abs(state.rotation.squaredNorm() - 1.0) <= arithmeticTolerance() &&
         std::abs(state.extrinsicRotation.squaredNorm() - 1.0) <= arithmeticTolerance() &&
         std::abs(state.gravity.norm() - kGravityMagnitude) <= arithmeticTolerance() * kGravityMagnitude;
}

Eigen::Matrix<double, kStateDof, kPoseDof> poseDirections(const State & state)
{
  Eigen::Matrix<double, kStateDof, kPoseDof> result = Eigen::Matrix<double, kStateDof, kPoseDof>::Zero();
  result.topLeftCorner<3, 3>() = state.rotation.toRotationMatrix();
  result.block<3, 3>(kRotation, kRotation).setIdentity();
  return result;
}

template<typename Derived>
Eigen::MatrixXd symmetric(const Eigen::MatrixBase<Derived> & matrix)
{
  const Eigen::MatrixXd value = matrix.eval();
  return value.selfadjointView<Eigen::Lower>();
}

// SHA-256 is only an exact-byte identity codec, not a statistical approximation.
std::string sha256(const std::string & input)
{
  constexpr std::size_t kBlockBytes = 64, kWords = 16, kRounds = 64, kDigestWords = 8;
  constexpr unsigned kBitsPerByte = 8, kWordBits = 32, kWordBytes = 4, kByteMask = 255, kPadding = 128;
  constexpr std::array<unsigned, 3> kSmall0{7, 18, 3}, kSmall1{17, 19, 10};
  constexpr std::array<unsigned, 3> kBig0{2, 13, 22}, kBig1{6, 11, 25};
  constexpr std::array<std::uint32_t, kRounds> kRound{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  constexpr std::array<std::uint32_t, kDigestWords> kInitialDigest{
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto digest = kInitialDigest;
  constexpr std::size_t kScheduleLag = 7, kWorkingWordF = 5;
  std::vector<unsigned char> data(input.begin(), input.end());
  const auto bits = static_cast<std::uint64_t>(data.size()) * kBitsPerByte;
  data.push_back(kPadding);
  while ((data.size() + sizeof(bits)) % kBlockBytes != 0)
    data.push_back(0);
  for (std::size_t byte = sizeof(bits); byte > 0; --byte)
    data.push_back(static_cast<unsigned char>((bits >> ((byte - 1) * kBitsPerByte)) & kByteMask));
  const auto rotate = [](std::uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (kWordBits - shift));
  };
  for (std::size_t block = 0; block < data.size(); block += kBlockBytes) {
    std::array<std::uint32_t, kRounds> words{};
    for (std::size_t i = 0; i < kWords; ++i)
      for (unsigned byte = 0; byte < kWordBytes; ++byte)
        words[i] = (words[i] << kBitsPerByte) | data[block + i * kWordBytes + byte];
    for (std::size_t i = kWords; i < kRounds; ++i) {
      const auto x = words[i - 15], y = words[i - 2];
      words[i] = words[i - kWords] + (rotate(x, kSmall0[0]) ^ rotate(x, kSmall0[1]) ^ (x >> kSmall0[2])) +
                 words[i - kScheduleLag] + (rotate(y, kSmall1[0]) ^ rotate(y, kSmall1[1]) ^ (y >> kSmall1[2]));
    }
    auto v = digest;
    for (std::size_t i = 0; i < kRounds; ++i) {
      const auto t1 = v[7] + (rotate(v[4], kBig1[0]) ^ rotate(v[4], kBig1[1]) ^ rotate(v[4], kBig1[2])) +
                      ((v[4] & v[5]) ^ (~v[4] & v[6])) + kRound[i] + words[i];
      const auto t2 = (rotate(v[0], kBig0[0]) ^ rotate(v[0], kBig0[1]) ^ rotate(v[0], kBig0[2])) +
                      ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
      v = {t1 + t2, v[0], v[1], v[2], v[3] + t1, v[4], v[kWorkingWordF], v[6]};
    }
    for (std::size_t i = 0; i < digest.size(); ++i)
      digest[i] += v[i];
  }
  constexpr std::size_t kHexDigits = 16;
  constexpr std::array<char, kHexDigits> kHex{
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  constexpr unsigned kNibbleBits = 4, kNibbleMask = 15;
  std::string result;
  for (const auto word : digest)
    for (unsigned nibble = kWordBits / kNibbleBits; nibble > 0; --nibble)
      result.push_back(kHex[(word >> ((nibble - 1) * kNibbleBits)) & kNibbleMask]);
  return result;
}

void appendInteger(std::string & bytes, std::uint64_t value)
{
  constexpr unsigned kBitsPerByte = 8, kByteMask = 255;
  for (std::size_t i = 0; i < sizeof(value); ++i)
    bytes.push_back(static_cast<char>((value >> (i * kBitsPerByte)) & kByteMask));
}

bool finiteSparse(const SparseMatrix & matrix)
{
  for (Eigen::Index column = 0; column < matrix.outerSize(); ++column)
    for (SparseMatrix::InnerIterator entry(matrix, column); entry; ++entry)
      if (!std::isfinite(entry.value()))
        return false;
  return true;
}

// Davis' elimination-tree column count, before Eigen allocates the numerical L.
// Each marked ancestor is one structural off-diagonal entry of that L column.
bool symbolicFill(const SparseMatrix & matrix, std::size_t limit, std::size_t & count)
{
  const auto dimension = matrix.cols();
  std::vector<Eigen::Index> parent(static_cast<std::size_t>(dimension), -1);
  std::vector<Eigen::Index> marked(static_cast<std::size_t>(dimension), -1);
  count = 0;
  for (Eigen::Index column = 0; column < dimension; ++column) {
    marked[static_cast<std::size_t>(column)] = column;
    for (SparseMatrix::InnerIterator entry(matrix, column); entry; ++entry) {
      Eigen::Index row = entry.row();
      if (row >= column)
        continue;
      while (marked[static_cast<std::size_t>(row)] != column) {
        auto & ancestor = parent[static_cast<std::size_t>(row)];
        if (ancestor < 0)
          ancestor = column;
        if (++count > limit)
          return false;
        marked[static_cast<std::size_t>(row)] = column;
        row = ancestor;
      }
    }
  }
  return true;
}

class FrameNoise
{
public:
  FrameStatus initialize(const SparseMatrix & b,
                         const Eigen::VectorXd & diagonal,
                         double mapVariance,
                         const Budgets & budgets,
                         Counts & counts)
  {
    const auto rows = b.rows(), columns = b.cols();
    const std::size_t byteLimit = std::min(budgets.bytes, Budgets{}.bytes);
    if (diagonal.size() != rows || rows <= 0 || columns > static_cast<Eigen::Index>(3 * kSupportCount) * rows)
      return FrameStatus::kIdentityMismatch;
    if (static_cast<std::size_t>(rows) > budgets.rows || rows > static_cast<Eigen::Index>(Budgets{}.rows))
      return FrameStatus::kFrameRowsOverBudget;
    const auto minimumBytes = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(6 * kStateDof + 2 * joint_source::kMaxRows) * sizeof(double);
    counts.accountedPeakBytes = std::max(counts.accountedPeakBytes.value_or(0), minimumBytes);
    if (minimumBytes > byteLimit || b.nonZeros() > static_cast<Eigen::Index>(3 * kSupportCount) * rows)
      return FrameStatus::kMemoryBudget;
    if (!diagonal.allFinite() || !finiteSparse(b) || !std::isfinite(mapVariance))
      return FrameStatus::kNonfiniteInput;
    if ((diagonal.array() <= 0.0).any() || mapVariance < 0.0)
      return FrameStatus::kNumericalFailure;
    if (static_cast<std::size_t>(rows) > budgets.rows)
      return FrameStatus::kFrameRowsOverBudget;
    b_ = &b;
    mapVariance_ = mapVariance;
    std::vector<std::size_t> rowCounts(static_cast<std::size_t>(rows), 0);
    for (Eigen::Index column = 0; column < b.outerSize(); ++column)
      for (SparseMatrix::InnerIterator entry(b, column); entry; ++entry)
        ++rowCounts[static_cast<std::size_t>(entry.row())];
    auto productBound = static_cast<std::size_t>(columns);
    for (const auto count : rowCounts)
      productBound += count * count;
    // Dense RHS/SVD/compression buffers, three sparse-product/permutation
    // buffers, ordering workspace and point-key nodes. This is a conservative
    // required-buffer bound, not a claim about malloc RSS.
    const std::size_t base = static_cast<std::size_t>(rows) *
                               static_cast<std::size_t>(6 * kStateDof + 2 * joint_source::kMaxRows) * sizeof(double) +
                             static_cast<std::size_t>(columns) *
                               (static_cast<std::size_t>(2 * kStateDof) * sizeof(double) + 32 * sizeof(int)) +
                             3 * productBound * kSparseEntryBytes +
                             static_cast<std::size_t>(b.nonZeros()) * kSparseEntryBytes;
    counts.accountedPeakBytes = std::max(counts.accountedPeakBytes.value_or(0), base);
    if (base > byteLimit)
      return FrameStatus::kMemoryBudget;
    inverseDiagonal_ = diagonal.cwiseInverse();
    if (!inverseDiagonal_.allFinite())
      return FrameStatus::kNumericalFailure;
    if (columns == 0 || mapVariance == 0.0) {
      counts.nnzFactor = std::max(counts.nnzFactor.value_or(0), std::size_t{0});
      return FrameStatus::kOkFrameConditional;
    }
    SparseMatrix scaled = b;
    for (Eigen::Index column = 0; column < scaled.outerSize(); ++column)
      for (SparseMatrix::InnerIterator entry(scaled, column); entry; ++entry)
        entry.valueRef() *= std::sqrt(mapVariance * inverseDiagonal_[entry.row()]);
    if (!finiteSparse(scaled))
      return FrameStatus::kNumericalFailure;
    SparseMatrix inner = scaled.transpose() * scaled;
    for (Eigen::Index i = 0; i < columns; ++i)
      inner.coeffRef(i, i) += 1.0;
    inner.makeCompressed();
    if (!finiteSparse(inner))
      return FrameStatus::kNumericalFailure;
    // Eigen ordering methods return the inverse permutation: entry k names the
    // k-th pivot. The elimination order is its inverse, as SimplicialCholesky
    // applies it; twisting by the returned matrix itself is not fill-reducing.
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pivots;
    Eigen::AMDOrdering<int> ordering;
    ordering(inner, pivots);
    permutation_ = pivots.inverse();
    SparseMatrix ordered = permutation_ * inner * permutation_.transpose();
    ordered.makeCompressed();
    std::size_t fill = 0;
    if (!symbolicFill(ordered, (byteLimit - base) / kSparseEntryBytes, fill))
      return FrameStatus::kMemoryBudget;
    counts.accountedPeakBytes = std::max(counts.accountedPeakBytes.value_or(0), base + fill * kSparseEntryBytes);
    factor_.analyzePattern(ordered);
    if (factor_.info() != Eigen::Success)
      return FrameStatus::kNumericalFailure;
    factor_.factorize(ordered);
    if (factor_.info() != Eigen::Success || !factor_.vectorD().allFinite() || (factor_.vectorD().array() <= 0.0).any())
      return FrameStatus::kNumericalFailure;
    const auto actualFill = static_cast<std::size_t>(factor_.matrixL().nestedExpression().nonZeros());
    if (actualFill != fill)
      return FrameStatus::kNumericalFailure;
    counts.nnzFactor = std::max(counts.nnzFactor.value_or(0), actualFill);
    factored_ = true;
    return FrameStatus::kOkFrameConditional;
  }

  Eigen::MatrixXd solve(const Eigen::MatrixXd & rhs) const
  {
    Eigen::MatrixXd scaled = inverseDiagonal_.asDiagonal() * rhs;
    if (!factored_)
      return scaled;
    const Eigen::MatrixXd local = b_->transpose() * scaled;
    const Eigen::MatrixXd permuted = permutation_ * local;
    const Eigen::MatrixXd solution = permutation_.transpose() * factor_.solve(permuted);
    return scaled - mapVariance_ * inverseDiagonal_.asDiagonal() * ((*b_) * solution);
  }

  Eigen::MatrixXd multiply(const Eigen::MatrixXd & rhs) const
  {
    return inverseDiagonal_.cwiseInverse().asDiagonal() * rhs + mapVariance_ * (*b_) * (b_->transpose() * rhs);
  }

private:
  const SparseMatrix * b_ = nullptr;  // Borrowed for this one frame operation.
  Eigen::VectorXd inverseDiagonal_;
  double mapVariance_ = 0.0;
  bool factored_ = false;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation_;
  Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower, Eigen::NaturalOrdering<int>> factor_;
};

double relativeError(const Eigen::MatrixXd & value, const Eigen::MatrixXd & expected)
{
  if (!value.allFinite() || !expected.allFinite())
    return std::numeric_limits<double>::infinity();
  if (value.size() == 0 && expected.size() == 0)
    return 0.0;
  return (value - expected).cwiseAbs().maxCoeff() / std::max(1.0, expected.cwiseAbs().maxCoeff());
}

FrameStatus compressWithNoise(const Eigen::MatrixXd & h,
                              const Eigen::VectorXd & y,
                              const Eigen::MatrixXd & gain,
                              const FrameNoise & noise,
                              const Budgets & budgets,
                              CompressedFrame & result,
                              Counts & counts)
{
  const auto rows = h.rows(), active = h.cols();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(gain, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success)
    return FrameStatus::kNumericalFailure;
  const double gainScale = svd.singularValues().size() == 0 ? 0.0 : svd.singularValues()[0];
  const Eigen::Index rankGain =
    (svd.singularValues().array() > rankTolerance(std::max(rows, active), gainScale)).count();
  const Eigen::MatrixXd basis = svd.matrixV().leftCols(rankGain).transpose();
  const Eigen::MatrixXd inverseH = noise.solve(h);
  Eigen::MatrixXd statistic(active + rankGain, rows), response(rows, active + rankGain);
  statistic.topRows(active) = inverseH.transpose();
  statistic.bottomRows(rankGain) = basis;
  response.leftCols(active) = h;
  response.rightCols(rankGain) = noise.multiply(basis.transpose());
  // Equilibrate the statistics before deciding their rank. Their physical
  // units differ (position, velocity, S2 tangent and gain rows); a raw C(t)
  // eigenvalue cutoff would otherwise discard independently informative rows
  // solely because a different coordinate has a large scale.
  const Eigen::VectorXd variances = (statistic * response).diagonal();
  if (!variances.allFinite() || (variances.array() < 0.0).any())
    return FrameStatus::kNumericalFailure;
  Eigen::VectorXd scales(variances.size());
  for (Eigen::Index row = 0; row < scales.size(); ++row) {
    scales[row] = variances[row] > 0.0 ? 1.0 / std::sqrt(variances[row]) : 1.0;
    statistic.row(row) *= scales[row];
    response.col(row) *= scales[row];
  }
  const Eigen::MatrixXd covariance = symmetric(statistic * response);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(covariance);
  if (spectrum.info() != Eigen::Success || !spectrum.eigenvalues().allFinite())
    return FrameStatus::kNumericalFailure;
  const double scale = spectrum.eigenvalues().cwiseAbs().maxCoeff();
  const double threshold = rankTolerance(std::max(rows, active + rankGain), scale);
  if (spectrum.eigenvalues()[0] < -threshold)
    return FrameStatus::kNumericalFailure;
  const Eigen::Index rank = (spectrum.eigenvalues().array() > threshold).count();
  if (rank > rows)
    return FrameStatus::kNumericalFailure;
  counts.rankT = counts.compressedRows = rank;
  counts.rankGain = rankGain;
  if (rank > budgets.compressedRows || rank > joint_source::kMaxRows)
    return FrameStatus::kRowCapacity;
  const Eigen::MatrixXd vectors = spectrum.eigenvectors().rightCols(rank);
  const Eigen::VectorXd values = spectrum.eigenvalues().tail(rank);
  const Eigen::MatrixXd whitening = values.cwiseSqrt().cwiseInverse().asDiagonal() * vectors.transpose();
  CompressedFrame candidate;
  candidate.h = whitening * statistic * h;
  candidate.observation = whitening * statistic * y;
  candidate.gain = gain * basis.transpose() * scales.tail(rankGain).cwiseInverse().asDiagonal() *
                   vectors.bottomRows(rankGain) * values.cwiseSqrt().asDiagonal();
  const Eigen::VectorXd residual = y - response * whitening.transpose() * candidate.observation;
  const Eigen::VectorXd inverseResidual = noise.solve(residual);
  candidate.chi2Perpendicular = residual.dot(inverseResidual);
  candidate.dofPerpendicular = rows - rank;
  const double condition = rank == 0 ? 1.0 : scale / values[0];
  const double tolerance = arithmeticTolerance(condition);
  if (!candidate.h.allFinite() || !candidate.gain.allFinite() || !candidate.observation.allFinite() ||
      !std::isfinite(candidate.chi2Perpendicular) || candidate.chi2Perpendicular < 0.0 ||
      relativeError(candidate.gain * candidate.h, gain * h) > tolerance ||
      relativeError(candidate.gain * candidate.observation, gain * y) > tolerance ||
      relativeError(candidate.gain * candidate.gain.transpose(), gain * noise.multiply(gain.transpose())) > tolerance ||
      relativeError(candidate.h.transpose() * candidate.h, h.transpose() * inverseH) > tolerance)
    return FrameStatus::kNumericalFailure;
  result = std::move(candidate);
  return FrameStatus::kOkFrameConditional;
}

bool finiteRow(const Row & row)
{
  if (!row.point.allFinite() || !std::isfinite(row.weight) || (row.relativeTime && !std::isfinite(*row.relativeTime)) ||
      (row.beamDirection && !row.beamDirection->allFinite()))
    return false;
  for (const auto & point : row.support)
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
        !std::isfinite(point.intensity))
      return false;
  return !row.plane ||
         (row.plane->normal.allFinite() && std::isfinite(row.plane->offset) && row.plane->eigenvalues.allFinite() &&
          row.plane->axis1.allFinite() && row.plane->axis2.allFinite());
}

FrameStatus validateSnapshot(const FrameSnapshot & snapshot, const Budgets & budgets)
{
  const auto rows = static_cast<Eigen::Index>(snapshot.rows.size());
  if (!snapshot.selected)
    return FrameStatus::kSkippedDecimated;
  if (!snapshot.identity || !snapshot.posterior || !snapshot.posteriorCovariance || !snapshot.diagnostics.valid ||
      rows == 0)
    return FrameStatus::kNoUpdate;
  if (snapshot.rows.size() > budgets.rows || snapshot.rows.size() > Budgets{}.rows)
    return FrameStatus::kFrameRowsOverBudget;
  if (!snapshot.prior || !snapshot.linearization || !snapshot.priorCovariance || snapshot.productionH.rows() != rows ||
      snapshot.productionH.cols() != kProductionColumns || snapshot.preFloorH.rows() != rows ||
      snapshot.preFloorH.cols() != kProductionColumns || snapshot.productionObservation.size() != rows ||
      snapshot.preFloorObservation.size() != rows)
    return FrameStatus::kIdentityMismatch;
  if (!finiteState(*snapshot.prior) || !finiteState(*snapshot.linearization) || !finiteState(*snapshot.posterior) ||
      !snapshot.priorCovariance->allFinite() || !snapshot.posteriorCovariance->allFinite() ||
      !snapshot.productionH.allFinite() || !snapshot.preFloorH.allFinite() ||
      !snapshot.productionObservation.allFinite() || !snapshot.preFloorObservation.allFinite() ||
      !snapshot.diagnostics.lastDx.allFinite() || (snapshot.considerKeep && !snapshot.considerKeep->allFinite()) ||
      (snapshot.angularVelocity && !snapshot.angularVelocity->allFinite()) ||
      (snapshot.scanDuration && !std::isfinite(*snapshot.scanDuration)))
    return FrameStatus::kNonfiniteInput;
  for (const auto & row : snapshot.rows)
    if (!finiteRow(row))
      return FrameStatus::kNonfiniteInput;
  if (snapshot.kernel != Kernel::kNone)
    return FrameStatus::kUnsupportedKernel;
  if (snapshot.floorMode != FloorMode::kOff && snapshot.floorMode != FloorMode::kObserve)
    return FrameStatus::kUnsupportedFloorApply;
  if (!validState(*snapshot.prior) || !validState(*snapshot.linearization) || !validState(*snapshot.posterior) ||
      !joint_source::validKey(snapshot.identity->worldFrame) || snapshot.identity->scanEndNs < 0 ||
      snapshot.identity->evaluationOrdinal != snapshot.diagnostics.iterations ||
      snapshot.diagnostics.measurementDimension != snapshot.rows.size() ||
      (snapshot.scanDuration && *snapshot.scanDuration <= 0.0))
    return FrameStatus::kIdentityMismatch;
  if (snapshot.queryAmplitude.source != QueryAmplitudeSource::kPooled &&
      snapshot.queryAmplitude.source != QueryAmplitudeSource::kVendorRange)
    return FrameStatus::kIdentityMismatch;
  if (snapshot.queryAmplitude.specification != VendorRangeSpecification::kTypical15mm &&
      snapshot.queryAmplitude.specification != VendorRangeSpecification::kWebsite10mm)
    return FrameStatus::kIdentityMismatch;
  std::set<std::uint64_t> queries;
  for (const auto & row : snapshot.rows) {
    if (row.kind != RowKind::kPointPlane || !row.plane || row.plane->count != kSupportCount)
      return FrameStatus::kUnsupportedRowKind;
    if (row.weight <= 0.0 || !queries.insert(row.queryOrdinal).second ||
        (snapshot.queryAmplitude.source == QueryAmplitudeSource::kVendorRange && !row.beamDirection) ||
        (row.relativeTime &&
         (*row.relativeTime > 0.0 || (snapshot.scanDuration && *row.relativeTime < -*snapshot.scanDuration))))
      return FrameStatus::kIdentityMismatch;
  }
  return FrameStatus::kOkFrameConditional;
}

struct Closure
{
  Matrix23 prior, rawPrior, reset, productionInformation;
  Vector23 priorMean, transportedDifference;
  Eigen::MatrixXd gain;
  State predictedState;
  double condition = 1.0;
};

double conditionNumber(const Eigen::MatrixXd & matrix)
{
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  if (svd.info() != Eigen::Success || !svd.singularValues().allFinite() || svd.singularValues().tail(1)[0] <= 0.0)
    return std::numeric_limits<double>::infinity();
  return svd.singularValues()[0] / svd.singularValues().tail(1)[0];
}

FrameStatus closeProduction(const FrameSnapshot & snapshot,
                            Closure & result,
                            ClosureDiagnostics & diagnostics,
                            std::uint32_t & flags)
{
  if (!snapshot.prior || !snapshot.linearization || !snapshot.priorCovariance || !snapshot.posterior ||
      !snapshot.posteriorCovariance)
    return FrameStatus::kIdentityMismatch;
  const auto & priorState = *snapshot.prior;
  const auto & linearization = *snapshot.linearization;
  const auto & posteriorState = *snapshot.posterior;
  const auto & capturedPosterior = *snapshot.posteriorCovariance;
  const auto rows = snapshot.productionH.rows();
  const Vector23 delta = difference(linearization, priorState);
  const Matrix23 priorTransport = transport(priorState, linearization, delta);
  result.transportedDifference = priorTransport * delta;
  result.priorMean = -result.transportedDifference;
  result.rawPrior = priorTransport * *snapshot.priorCovariance * priorTransport.transpose();
  result.prior = 0.5 * (result.rawPrior + result.rawPrior.transpose());
  if (!result.rawPrior.allFinite() || !result.prior.allFinite() || !result.transportedDifference.allFinite())
    return FrameStatus::kNumericalFailure;
  const double priorAsymmetry = (result.rawPrior - result.rawPrior.transpose()).cwiseAbs().maxCoeff();
  const double posteriorAsymmetry = (capturedPosterior - capturedPosterior.transpose()).cwiseAbs().maxCoeff();
  diagnostics.priorAsymmetry = priorAsymmetry;
  diagnostics.posteriorAsymmetry = posteriorAsymmetry;
  if (priorAsymmetry != 0.0 || posteriorAsymmetry != 0.0)
    flag(flags, FrameFlag::kPriorSymmetrized);
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(rows, kStateDof);
  h.leftCols(kProductionColumns) = snapshot.productionH;
  result.productionInformation = h.transpose() * h / kProductionVariance;
  Eigen::MatrixXd gain;
  if (rows < kStateDof) {
    const Eigen::MatrixXd innovation =
      h * result.rawPrior * h.transpose() + kProductionVariance * Eigen::MatrixXd::Identity(rows, rows);
    result.condition = conditionNumber(innovation);
    gain = result.rawPrior * h.transpose() * innovation.fullPivLu().inverse();
  } else {
    Eigen::FullPivLU<Matrix23> priorFactor(result.rawPrior);
    if (!priorFactor.isInvertible())
      return FrameStatus::kNumericalFailure;
    const Eigen::MatrixXd information = h.transpose() * h + kProductionVariance * priorFactor.inverse();
    result.condition = conditionNumber(information);
    gain = information.fullPivLu().solve(h.transpose());
  }
  const double tolerance = arithmeticTolerance(result.condition);
  if (!std::isfinite(tolerance) || tolerance >= 1.0 || !gain.allFinite())
    return FrameStatus::kNumericalFailure;
  diagnostics.tolerance = tolerance;
  Matrix23 projection = Matrix23::Identity();
  if (snapshot.considerKeep) {
    if (relativeError(*snapshot.considerKeep, snapshot.considerKeep->transpose()) > tolerance)
      return FrameStatus::kIdentityMismatch;
    projection.block<3, 3>(kRotation, kRotation) = *snapshot.considerKeep;
  }
  const Eigen::MatrixXd effectiveGain = projection * gain;
  const Vector23 increment = effectiveGain * snapshot.productionObservation +
                             (effectiveGain * h - Matrix23::Identity()) * result.transportedDifference;
  if (!increment.allFinite())
    return FrameStatus::kNumericalFailure;
  result.predictedState = plus(linearization, increment);
  // Mirror the existing producer, including its use of x_propagated as the
  // S2 reset origin. O8's actual-filter acceptance belongs to K3-2.
  result.reset = transport(priorState, result.predictedState, increment);
  Matrix23 posterior = result.rawPrior - gain * h * result.rawPrior;
  if (snapshot.considerKeep) {
    const Matrix23 error = Matrix23::Identity() - effectiveGain * h;
    posterior =
      error * result.rawPrior * error.transpose() + kProductionVariance * effectiveGain * effectiveGain.transpose();
    posterior = (0.5 * (posterior + posterior.transpose())).eval();
  }
  posterior = (result.reset * posterior * result.reset.transpose()).eval();
  if (!posterior.allFinite() || !finiteState(result.predictedState))
    return FrameStatus::kNumericalFailure;
  const double dxError = relativeError(increment, snapshot.diagnostics.lastDx);
  const double covarianceError = relativeError(posterior, capturedPosterior);
  const double stateError = difference(result.predictedState, posteriorState).cwiseAbs().maxCoeff();
  diagnostics.dxError = dxError;
  diagnostics.covarianceError = covarianceError;
  diagnostics.stateError = stateError;
  if (dxError > tolerance || covarianceError > tolerance || stateError > tolerance ||
      relativeError(snapshot.productionH, snapshot.preFloorH) > tolerance ||
      relativeError(snapshot.productionObservation, snapshot.preFloorObservation) > tolerance)
    return FrameStatus::kIdentityMismatch;
  result.gain = effectiveGain;
  for (Eigen::Index row = 0; row < rows; ++row)
    result.gain.col(row) *= snapshot.rows[static_cast<std::size_t>(row)].weight;
  return FrameStatus::kOkFrameConditional;
}

Eigen::Vector3d pointVector(const SupportPoint & point)
{
  return {point.x, point.y, point.z};
}

Eigen::Vector3d center(const Row & row)
{
  const Eigen::Vector3d reference = pointVector(row.support[0]);
  Eigen::Vector3d local = Eigen::Vector3d::Zero();
  for (const auto & point : row.support)
    local += pointVector(point) - reference;
  return reference + local / static_cast<double>(kSupportCount);
}

using PointKey = std::array<std::uint32_t, 4>;
PointKey pointKey(const SupportPoint & point)
{
  static_assert(sizeof(float) == sizeof(std::uint32_t), "Point identity requires binary32 float");
  const std::array<float, 4> values{point.x, point.y, point.z, point.intensity};
  PointKey result{};
  std::memcpy(result.data(), values.data(), sizeof(result));
  return result;
}

struct Model
{
  Eigen::MatrixXd h, rigidH, calibration;
  Eigen::VectorXd observation, deskewOffset;
  SparseMatrix b;
  std::optional<Eigen::VectorXd> queryVariance;
  std::optional<double> pooledVariance;
  std::size_t poolDof = 0;
};

FrameStatus buildModel(const FrameSnapshot & snapshot,
                       const Closure & closure,
                       const Budgets & budgets,
                       Model & model,
                       FrameReadout & readout)
{
  if (!snapshot.linearization || !readout.closure)
    return FrameStatus::kIdentityMismatch;
  const auto & closureDiagnostics = *readout.closure;
  if (!closureDiagnostics.tolerance)
    return FrameStatus::kIdentityMismatch;
  const double closureTolerance = *closureDiagnostics.tolerance;
  const auto rows = static_cast<Eigen::Index>(snapshot.rows.size());
  if (rows <= 0)
    return FrameStatus::kNoUpdate;
  const std::size_t baseBytes = static_cast<std::size_t>(rows) *
                                static_cast<std::size_t>(6 * kStateDof + 2 * joint_source::kMaxRows) * sizeof(double);
  readout.counts.accountedPeakBytes = baseBytes;
  if (baseBytes > std::min(budgets.bytes, Budgets{}.bytes))
    return FrameStatus::kMemoryBudget;
  model.h.resize(rows, kStateDof);
  model.rigidH.resize(rows, kStateDof);
  model.calibration.resize(rows, kPoseDof);
  model.observation.resize(rows);
  model.deskewOffset.resize(rows);
  const auto & state = *snapshot.linearization;
  const Eigen::Matrix3d rotation = state.rotation.toRotationMatrix(),
                        extrinsic = state.extrinsicRotation.toRotationMatrix();
  const bool timesAvailable = std::all_of(
    snapshot.rows.begin(), snapshot.rows.end(), [](const Row & row) { return row.relativeTime.has_value(); });
  flag(readout.flags, timesAvailable ? FrameFlag::kDeskewColumnsRigidApprox : FrameFlag::kDeskewColumnsUnavailable);
  if (snapshot.queryAmplitude.source == QueryAmplitudeSource::kVendorRange)
    model.queryVariance.emplace(rows);
  Eigen::VectorXd * queryVariance = model.queryVariance ? &*model.queryVariance : nullptr;
  std::map<PointKey, Eigen::Index> points;
  std::set<Eigen::Index> pooledPoints;
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(snapshot.rows.size() * kSupportCount * 3);
  double scatter = 0.0;
  for (Eigen::Index i = 0; i < rows; ++i) {
    const auto & row = snapshot.rows[static_cast<std::size_t>(i)];
    if (!row.plane)
      return FrameStatus::kUnsupportedRowKind;
    const auto & plane = *row.plane;
    StateRowJacobian h;
    if (const auto status = geometricRow(row, state, snapshot.extrinsicEstimated, snapshot.angularVelocity, h);
        status != FrameStatus::kOkFrameConditional)
      return status;
    if (!timesAvailable)
      h.rightCols(kStateDof - kProductionColumns).setZero();
    model.h.row(i) = h;
    model.rigidH.row(i) = h;
    model.rigidH.rightCols(kStateDof - kProductionColumns).row(i).setZero();
    if (!snapshot.holdActive &&
        relativeError(row.weight * h.leftCols(kProductionColumns), snapshot.preFloorH.row(i)) > closureTolerance)
      return FrameStatus::kIdentityMismatch;
    model.deskewOffset[i] = (model.h.row(i) - model.rigidH.row(i)).dot(closure.transportedDifference);
    model.observation[i] = snapshot.preFloorObservation[i] / row.weight - model.deskewOffset[i];
    const Eigen::Vector3d normalBody = rotation.transpose() * plane.normal;
    model.calibration.block<1, 3>(i, 0) = row.point.cross(extrinsic.transpose() * normalBody).transpose();
    model.calibration.block<1, 3>(i, kRotation) = normalBody.transpose();
    const Eigen::Vector3d worldPoint = rotation * (extrinsic * row.point + state.extrinsicTranslation) + state.position;
    std::array<double, kSupportCount> weights{};
    if (const auto status = hatWeights(row, worldPoint, weights); status != FrameStatus::kOkFrameConditional)
      return status;
    std::set<Eigen::Index> planePoints;
    for (std::size_t a = 0; a < kSupportCount; ++a) {
      const auto next = static_cast<Eigen::Index>(points.size());
      const auto index = points.emplace(pointKey(row.support[a]), next).first->second;
      planePoints.insert(index);
      for (Eigen::Index axis = 0; axis < 3; ++axis)
        if (weights[a] * plane.normal[axis] != 0.0)
          triplets.emplace_back(i, 3 * index + axis, weights[a] * plane.normal[axis]);
    }
    if (planePoints.size() == kSupportCount &&
        std::none_of(
          planePoints.begin(), planePoints.end(), [&](Eigen::Index point) { return pooledPoints.count(point) != 0; })) {
      if (plane.eigenvalues[0] < 0.0)
        return FrameStatus::kNumericalFailure;
      scatter += plane.eigenvalues[0];
      model.poolDof += kSupportCount - 3;
      pooledPoints.insert(planePoints.begin(), planePoints.end());
    }
    if (queryVariance != nullptr) {
      if (!row.beamDirection)
        return FrameStatus::kIdentityMismatch;
      const double length = row.beamDirection->norm();
      if (!(length > 0.0) || !std::isfinite(length))
        return FrameStatus::kNumericalFailure;
      const double sigma = snapshot.queryAmplitude.specification == VendorRangeSpecification::kTypical15mm
                             ? kTypicalRangeSigma
                             : kWebsiteRangeSigma;
      const double projection = plane.normal.dot(rotation * extrinsic * (*row.beamDirection / length));
      (*queryVariance)[i] = sigma * sigma * projection * projection;
    }
  }
  const Eigen::Index columns = 3 * static_cast<Eigen::Index>(points.size());
  if (columns <= 0)
    return FrameStatus::kNumericalFailure;
  model.b.resize(rows, columns);
  if (triplets.empty())
    model.b.setZero();
  else
    model.b.setFromTriplets(triplets.begin(), triplets.end());
  model.b.prune(0.0);
  readout.counts.supportPoints = points.size();
  readout.counts.duplicateSupportReferences = snapshot.rows.size() * kSupportCount - points.size();
  readout.counts.nnzB = static_cast<std::size_t>(model.b.nonZeros());
  if (model.poolDof >= static_cast<std::size_t>(kPoolMinimumDof)) {
    const double pooledVariance = scatter / static_cast<double>(model.poolDof);
    model.pooledVariance = pooledVariance;
    if (!model.queryVariance)
      model.queryVariance = Eigen::VectorXd::Constant(rows, pooledVariance);
  } else {
    flag(readout.flags, FrameFlag::kSigmaPoolDofLow);
  }
  return model.h.allFinite() && model.observation.allFinite() ? FrameStatus::kOkFrameConditional
                                                              : FrameStatus::kNumericalFailure;
}

struct PoseInformation
{
  PoseMatrix fixed, schur;
};

FrameStatus poseInformation(const Matrix23 & information, const State & state, PoseInformation & result)
{
  Matrix23 axes = Matrix23::Identity();
  axes.topLeftCorner<3, 3>() = state.rotation.toRotationMatrix();
  const Matrix23 body = symmetric(axes.transpose() * information * axes);
  result.fixed = body.topLeftCorner<kPoseDof, kPoseDof>();
  constexpr Eigen::Index kNuisance = kStateDof - kPoseDof;
  const Eigen::MatrixXd nuisance = body.bottomRightCorner<kNuisance, kNuisance>();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(nuisance);
  if (spectrum.info() != Eigen::Success || !spectrum.eigenvalues().allFinite())
    return FrameStatus::kNumericalFailure;
  const double threshold = rankTolerance(kStateDof, spectrum.eigenvalues().cwiseAbs().maxCoeff());
  if (spectrum.eigenvalues()[0] < -threshold)
    return FrameStatus::kNumericalFailure;
  const Eigen::Index rank = (spectrum.eigenvalues().array() > threshold).count();
  const Eigen::MatrixXd cross = body.topRightCorner<kPoseDof, kNuisance>() * spectrum.eigenvectors().rightCols(rank);
  result.schur =
    symmetric(result.fixed - cross * spectrum.eigenvalues().tail(rank).cwiseInverse().asDiagonal() * cross.transpose());
  return result.schur.allFinite() ? FrameStatus::kOkFrameConditional : FrameStatus::kNumericalFailure;
}

FrameStatus ratios(const PoseMatrix & numerator,
                   const PoseMatrix & denominator,
                   double multiplier,
                   DirectionValues & output,
                   bool bounded)
{
  const double threshold = rankTolerance(kStateDof, denominator.cwiseAbs().maxCoeff());
  for (Eigen::Index axis = 0; axis < kPoseDof; ++axis) {
    if (denominator(axis, axis) <= threshold)
      continue;  // An unobservable reference direction has no defined ratio.
    const double ratio = multiplier * numerator(axis, axis) / denominator(axis, axis);
    if (!std::isfinite(ratio) || ratio < -arithmeticTolerance() ||
        (bounded && (ratio <= 0.0 || ratio > 1.0 + arithmeticTolerance())))
      return FrameStatus::kNumericalFailure;
    output[static_cast<std::size_t>(axis)] = ratio;
  }
  return FrameStatus::kOkFrameConditional;
}

std::optional<EigenRange> generalized(const PoseMatrix & numerator, const PoseMatrix & denominator)
{
  Eigen::SelfAdjointEigenSolver<PoseMatrix> reference(denominator);
  if (reference.info() != Eigen::Success ||
      reference.eigenvalues()[0] <= rankTolerance(kPoseDof, reference.eigenvalues().cwiseAbs().maxCoeff()))
    return std::nullopt;
  Eigen::GeneralizedSelfAdjointEigenSolver<PoseMatrix> spectrum(numerator, denominator);
  if (spectrum.info() != Eigen::Success || !spectrum.eigenvalues().allFinite())
    return std::nullopt;
  EigenRange result;
  result.minimum = spectrum.eigenvalues()[0];
  result.maximum = spectrum.eigenvalues()[kPoseDof - 1];
  result.minimumDirection = spectrum.eigenvectors().col(0).normalized();
  return result;
}

FrameStatus correlationReadings(const FrameSnapshot & snapshot,
                                const Closure & closure,
                                const Model & model,
                                const Budgets & budgets,
                                FrameReadout & readout,
                                CompressedFrame & central)
{
  if (!snapshot.linearization || !readout.readings)
    return FrameStatus::kIdentityMismatch;
  const auto & linearization = *snapshot.linearization;
  auto & readings = *readout.readings;
  PoseInformation independent;
  const Matrix23 independentInformation = model.h.transpose() * model.h;
  if (poseInformation(independentInformation, linearization, independent) != FrameStatus::kOkFrameConditional)
    return FrameStatus::kNumericalFailure;
  constexpr std::array<double, 3> kBrackets{kQuarter, 1.0, 4.0};
  for (std::size_t index = 0; index < kBrackets.size(); ++index) {
    const double rho = kBrackets[index];
    FrameNoise noise;
    auto status =
      noise.initialize(model.b, Eigen::VectorXd::Constant(model.h.rows(), rho), 1.0, budgets, readout.counts);
    if (status != FrameStatus::kOkFrameConditional)
      return status;
    const Matrix23 information = symmetric(model.h.transpose() * noise.solve(model.h));
    PoseInformation correlated;
    if (poseInformation(information, linearization, correlated) != FrameStatus::kOkFrameConditional ||
        ratios(correlated.fixed, independent.fixed, rho, readings.gammaFixed[index], true) !=
          FrameStatus::kOkFrameConditional ||
        ratios(correlated.schur, independent.schur, rho, readings.gammaSchur[index], true) !=
          FrameStatus::kOkFrameConditional)
      return FrameStatus::kNumericalFailure;
    if (index == 1) {
      status = compressWithNoise(model.h, model.observation, closure.gain, noise, budgets, central, readout.counts);
      if (status != FrameStatus::kOkFrameConditional)
        return status;
      PoseInformation rigid;
      const Matrix23 rigidInformation = symmetric(model.rigidH.transpose() * noise.solve(model.rigidH));
      if (poseInformation(rigidInformation, linearization, rigid) != FrameStatus::kOkFrameConditional ||
          ratios(correlated.schur, rigid.schur, 1.0, readings.deskewInformationGain, false) !=
            FrameStatus::kOkFrameConditional)
        return FrameStatus::kNumericalFailure;
    }
  }
  return FrameStatus::kOkFrameConditional;
}

// Query-side unit continuous-white-PSD responses under the declared rigid
// endpoint approximation. Integrate the shared Brownian kernels, rather than
// treating the deskew error of each return as an independent IMU sample.
FrameStatus whiteSensitivities(const FrameSnapshot & snapshot,
                               const State & state,
                               const Eigen::MatrixXd & response,
                               Readings & readings)
{
  if (std::any_of(snapshot.rows.begin(), snapshot.rows.end(), [](const Row & row) { return !row.relativeTime; }))
    return FrameStatus::kOkFrameConditional;
  std::vector<double> times;
  times.reserve(snapshot.rows.size());
  for (const auto & row : snapshot.rows) {
    if (!row.relativeTime)
      return FrameStatus::kOkFrameConditional;
    times.push_back(*row.relativeTime);
  }
  std::vector<std::size_t> order(snapshot.rows.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return times[a] < times[b]; });
  Eigen::Matrix<double, kPoseDof, 3> gyro = Eigen::Matrix<double, kPoseDof, 3>::Zero();
  Eigen::Matrix<double, kPoseDof, 3> accel = gyro, accelOffset = gyro;
  double gyroEnergy = 0.0, accelEnergy = 0.0;
  const Eigen::Matrix3d rotation = state.rotation.toRotationMatrix();
  const Eigen::Matrix3d extrinsic = state.extrinsicRotation.toRotationMatrix();
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto index = order[i];
    const auto & row = snapshot.rows[index];
    if (!row.plane)
      return FrameStatus::kUnsupportedRowKind;
    const double begin = times[index];
    const double end = i + 1 < order.size() ? times[order[i + 1]] : 0.0;
    const Eigen::Vector3d normalBody = rotation.transpose() * row.plane->normal;
    const Eigen::Vector3d point = extrinsic * row.point + state.extrinsicTranslation;
    const auto column = response.col(static_cast<Eigen::Index>(index));
    gyro += column * point.cross(normalBody).transpose();
    const Eigen::Matrix<double, kPoseDof, 3> load = column * normalBody.transpose();
    accel += load;
    accelOffset -= begin * load;
    const double duration = end - begin, midpoint = 0.5 * (begin + end);
    gyroEnergy += duration * gyro.squaredNorm();
    constexpr double kLinearRampVarianceDenominator = 12.0;
    accelEnergy += duration * (midpoint * accel + accelOffset).squaredNorm() +
                   duration * duration * duration / kLinearRampVarianceDenominator * accel.squaredNorm();
  }
  readings.gyroWhiteSensitivity = std::sqrt(gyroEnergy);
  readings.accelWhiteSensitivity = std::sqrt(accelEnergy);
  return FrameStatus::kOkFrameConditional;
}

FrameStatus
sensitivities(const FrameSnapshot & snapshot, const Closure & closure, const Model & model, Readings & readings)
{
  if (!snapshot.posterior || !snapshot.linearization)
    return FrameStatus::kIdentityMismatch;
  const auto & state = *snapshot.linearization;
  const Eigen::MatrixXd response = poseDirections(*snapshot.posterior).transpose() * closure.reset * closure.gain;
  const Eigen::MatrixXd calibration = response * model.calibration;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    readings.extrinsicRotationSensitivity[axis] = calibration.col(static_cast<Eigen::Index>(axis)).norm();
    readings.extrinsicTranslationSensitivity[axis] = calibration.col(static_cast<Eigen::Index>(axis) + 3).norm();
  }
  readings.deskewOffsetNorm = model.deskewOffset.norm();
  if (snapshot.angularVelocity) {
    Eigen::VectorXd clock(static_cast<Eigen::Index>(snapshot.rows.size()));
    const Eigen::Matrix3d rotation = state.rotation.toRotationMatrix(),
                          extrinsic = state.extrinsicRotation.toRotationMatrix();
    for (Eigen::Index i = 0; i < clock.size(); ++i) {
      const auto & row = snapshot.rows[static_cast<std::size_t>(i)];
      if (!row.plane)
        return FrameStatus::kUnsupportedRowKind;
      const Eigen::Vector3d point = extrinsic * row.point + state.extrinsicTranslation;
      clock[i] = row.plane->normal.dot(state.velocity + rotation * snapshot.angularVelocity->cross(point));
    }
    readings.clockSensitivity = (response * clock).norm();
  }
  return whiteSensitivities(snapshot, state, response, readings);
}

std::string stateDigest(const State & state)
{
  std::string bytes;
  const std::array<Eigen::Vector3d, 6> vectors{
    state.position, state.extrinsicTranslation, state.velocity, state.gyroBias, state.accelBias, state.gravity};
  const auto appendDouble = [&](double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "State identity requires binary64");
    std::memcpy(&bits, &value, sizeof(bits));
    appendInteger(bytes, bits);
  };
  for (const auto & vector : vectors)
    for (Eigen::Index axis = 0; axis < 3; ++axis)
      appendDouble(vector[axis]);
  for (const auto & rotation : {state.rotation.toRotationMatrix(), state.extrinsicRotation.toRotationMatrix()})
    for (Eigen::Index column = 0; column < 3; ++column)
      for (Eigen::Index row = 0; row < 3; ++row)
        appendDouble(rotation(row, column));
  return sha256(bytes);
}

joint_source::Coordinates activeCoordinates(const FrameIdentity & identity)
{
  joint_source::Coordinates result{{},
                                   {},
                                   identity.worldFrame,
                                   "MTK tangent at x_lin",
                                   std::to_string(identity.scanEndNs) + "/" +
                                     std::to_string(identity.evaluationOrdinal)};
  const std::array<std::string, 7> blocks{
    "position", "rotation", "extrinsic_rotation", "extrinsic_translation", "velocity", "gyro_bias", "accel_bias"};
  const std::array<std::string, 7> units{"m", "rad", "rad", "m", "m/s", "rad/s", "m/s2"};
  for (std::size_t block = 0; block < blocks.size(); ++block)
    for (const auto * axis : {"x", "y", "z"}) {
      result.components.push_back(blocks[block] + "/" + axis);
      result.units.push_back(units[block]);
    }
  result.components.insert(result.components.end(), {"gravity/tangent0", "gravity/tangent1"});
  result.units.insert(result.units.end(), {"rad", "rad"});
  return result;
}

joint_source::LinearFactor frameFactor(const FrameSnapshot & snapshot,
                                       const FrameIdentity & identity,
                                       const State & linearization,
                                       const joint_source::Coordinates & active,
                                       const CompressedFrame & compressed)
{
  joint_source::LinearFactor result;
  const std::string scan = std::to_string(identity.scanEndNs), digest = rowSetDigest(snapshot.rows);
  result.domain = joint_source::ModelDomain::kLinearFixedModel;
  result.identity = {"lidar/" + scan + "/" + std::to_string(identity.evaluationOrdinal) + "/v2",
                     "frame-conditional TLS5 compressed linear likelihood",
                     {"scan/" + scan + "/rows/" + digest},
                     {},
                     "lidar-frame-noise/" + scan + "/" + digest,
                     "unit whitened noise; fixed support; pooled map amplitude; explicit query-amplitude assumption"};
  result.model = "pointplane-tls5-fixed-support-v2";
  result.linearization = stateDigest(linearization);
  result.active = active;
  result.rowUnits = std::vector<std::string>(static_cast<std::size_t>(compressed.h.rows()), "dimensionless");
  result.remainderKnowledge = joint_source::MomentKnowledge::kKnown;
  result.independentRemainder = true;
  result.h = compressed.h;
  result.a.resize(compressed.h.rows(), 0);
  result.observation = compressed.observation;
  result.remainderMean = Eigen::VectorXd::Zero(compressed.h.rows());
  result.remainderVariance = Eigen::VectorXd::Ones(compressed.h.rows());
  return result;
}

FrameStatus amplitudeComparisons(const FrameSnapshot & snapshot,
                                 const Closure & closure,
                                 const CompressedFrame & compressed,
                                 AmplitudeReadings & result)
{
  if (!snapshot.linearization || !snapshot.posterior || !snapshot.posteriorCovariance)
    return FrameStatus::kIdentityMismatch;
  const auto & linearization = *snapshot.linearization;
  const auto & posterior = *snapshot.posterior;
  const auto & capturedCovariance = *snapshot.posteriorCovariance;
  PoseInformation model, production;
  const Matrix23 information = symmetric(compressed.h.transpose() * compressed.h);
  if (poseInformation(information, linearization, model) != FrameStatus::kOkFrameConditional ||
      poseInformation(closure.productionInformation, linearization, production) != FrameStatus::kOkFrameConditional ||
      ratios(model.fixed, production.fixed, 1.0, result.informationFixed, false) != FrameStatus::kOkFrameConditional ||
      ratios(model.schur, production.schur, 1.0, result.informationSchur, false) != FrameStatus::kOkFrameConditional)
    return FrameStatus::kNumericalFailure;
  result.informationEigenvalues = generalized(model.fixed, production.fixed);
  const auto directions = poseDirections(posterior);
  const PoseMatrix actual = symmetric(directions.transpose() * result.gainCovariance * directions);
  const PoseMatrix optimal = symmetric(directions.transpose() * result.optimalCovariance * directions);
  const PoseMatrix produced = symmetric(directions.transpose() * capturedCovariance * directions);
  if (ratios(actual, produced, 1.0, result.gainOverProduction, false) != FrameStatus::kOkFrameConditional ||
      ratios(optimal, produced, 1.0, result.optimalOverProduction, false) != FrameStatus::kOkFrameConditional)
    return FrameStatus::kNumericalFailure;
  result.covarianceEigenvalues = generalized(actual, produced);
  Eigen::SelfAdjointEigenSolver<Matrix23> ordering(symmetric(result.gainCovariance - result.optimalCovariance));
  if (ordering.info() != Eigen::Success || !ordering.eigenvalues().allFinite())
    return FrameStatus::kNumericalFailure;
  result.gainMinusOptimalMinimum = ordering.eigenvalues()[0];
  const double scale = std::max({1.0, result.gainCovariance.norm(), result.optimalCovariance.norm()});
  return result.gainMinusOptimalMinimum >= -arithmeticTolerance(closure.condition) * scale
           ? FrameStatus::kOkFrameConditional
           : FrameStatus::kNumericalFailure;
}

FrameStatus evaluateK2(const FrameSnapshot & snapshot,
                       const Closure & closure,
                       const CompressedFrame & compressed,
                       FrameReadout & readout)
{
  if (!snapshot.identity || !snapshot.linearization || !readout.readings)
    return FrameStatus::kIdentityMismatch;
  const auto & identity = *snapshot.identity;
  const auto & linearization = *snapshot.linearization;
  auto & readings = *readout.readings;
  joint_source::JointMoments prior;
  prior.active = activeCoordinates(identity);
  prior.knowledge = joint_source::MomentKnowledge::kKnown;
  prior.evidence = "transported production prior; FRAME_CONDITIONAL; not accepted physical covariance";
  prior.mean = closure.priorMean;
  prior.covariance = symmetric(closure.prior);
  joint_source::JointSourceFilter filter;
  readout.k2Status = filter.initialize(prior);
  if (readout.k2Status != joint_source::Status::kOk)
    return FrameStatus::kK2Rejected;
  const auto factor = frameFactor(snapshot, identity, linearization, prior.active, compressed);
  const auto proposal = filter.propose({factor});
  readout.k2Status = proposal.status();
  if (readout.k2Status != joint_source::Status::kOk || !proposal.posterior())
    return FrameStatus::kK2Rejected;
  const auto gain = filter.appliedGain({factor}, compressed.gain);
  readout.k2Status = gain.status;
  if (gain.status != joint_source::Status::kOk)
    return FrameStatus::kK2Rejected;
  AmplitudeReadings result;
  result.optimalCovariance = symmetric(closure.reset * proposal.posterior()->covariance * closure.reset.transpose());
  result.gainCovariance = symmetric(closure.reset * gain.covariance * closure.reset.transpose());
  result.chi2Perpendicular = compressed.chi2Perpendicular;
  result.dofPerpendicular = compressed.dofPerpendicular;
  const auto status = amplitudeComparisons(snapshot, closure, compressed, result);
  if (status == FrameStatus::kOkFrameConditional)
    readings.amplitude = std::move(result);
  return status;
}

const char * k2StatusName(joint_source::Status status)
{
  using Status = joint_source::Status;
  switch (status) {
    case Status::kOk:
      return "kOk";
    case Status::kNoOp:
      return "kNoOp";
    case Status::kUnknownMoment:
      return "kUnknownMoment";
    case Status::kInvalidIdentity:
      return "kInvalidIdentity";
    case Status::kIdentityConflict:
      return "kIdentityConflict";
    case Status::kDimensionMismatch:
      return "kDimensionMismatch";
    case Status::kNonFinite:
      return "kNonFinite";
    case Status::kInvalidCovariance:
      return "kInvalidCovariance";
    case Status::kUnsupportedNoiseStructure:
      return "kUnsupportedNoiseStructure";
    case Status::kUnsupportedModel:
      return "kUnsupportedModel";
    case Status::kModelCapacityExceeded:
      return "kModelCapacityExceeded";
    case Status::kSourceHistoryUnavailable:
      return "kSourceHistoryUnavailable";
    case Status::kStaleProposal:
      return "kStaleProposal";
    case Status::kNumericalFailure:
      return "kNumericalFailure";
  }
  return "kUnknownMoment";
}

FrameStatus calculateFrame(const FrameSnapshot & snapshot, const Budgets & budgets, FrameReadout & readout)
{
  auto status = validateSnapshot(snapshot, budgets);
  if (status != FrameStatus::kOkFrameConditional)
    return status;
  auto start = Clock::now();
  Closure closure;
  readout.closure.emplace();
  status = closeProduction(snapshot, closure, *readout.closure, readout.flags);
  readout.timings.closureNs = elapsed(start);
  if (status != FrameStatus::kOkFrameConditional)
    return status;
  start = Clock::now();
  Model model;
  status = buildModel(snapshot, closure, budgets, model, readout);
  readout.timings.modelNs = elapsed(start);
  if (status != FrameStatus::kOkFrameConditional)
    return status;
  readout.readings.emplace();
  auto & readings = *readout.readings;
  readings.queryAmplitude = snapshot.queryAmplitude;
  readings.pooledMapVariance = model.pooledVariance;
  readings.poolDof = model.poolDof;
  if (snapshot.queryAmplitude.source == QueryAmplitudeSource::kVendorRange)
    readings.vendorRangeSigma = snapshot.queryAmplitude.specification == VendorRangeSpecification::kTypical15mm
                                  ? kTypicalRangeSigma
                                  : kWebsiteRangeSigma;
  status = sensitivities(snapshot, closure, model, readings);
  if (status != FrameStatus::kOkFrameConditional)
    return status;
  const auto finiteOptional = [](const std::optional<double> & value) {
    return !value || std::isfinite(*value);
  };
  const auto finite = [](double value) {
    return std::isfinite(value);
  };
  if (!std::all_of(
        readings.extrinsicRotationSensitivity.begin(), readings.extrinsicRotationSensitivity.end(), finite) ||
      !std::all_of(
        readings.extrinsicTranslationSensitivity.begin(), readings.extrinsicTranslationSensitivity.end(), finite) ||
      !std::isfinite(readings.deskewOffsetNorm) || !finiteOptional(readings.clockSensitivity) ||
      !finiteOptional(readings.gyroWhiteSensitivity) || !finiteOptional(readings.accelWhiteSensitivity))
    return FrameStatus::kNumericalFailure;
  start = Clock::now();
  CompressedFrame compressed;
  status = correlationReadings(snapshot, closure, model, budgets, readout, compressed);
  if (status == FrameStatus::kOkFrameConditional && model.pooledVariance && model.queryVariance) {
    if (snapshot.queryAmplitude.source == QueryAmplitudeSource::kPooled) {
      const double sigma = std::sqrt(*model.pooledVariance);
      if (!(sigma > 0.0) || !std::isfinite(sigma))
        status = FrameStatus::kNumericalFailure;
      else {
        compressed.h /= sigma;
        compressed.observation /= sigma;
        compressed.gain *= sigma;
        compressed.chi2Perpendicular /= *model.pooledVariance;
      }
    } else {
      status = compressRows(model.h,
                            model.observation,
                            closure.gain,
                            model.b,
                            *model.queryVariance,
                            *model.pooledVariance,
                            budgets,
                            compressed,
                            readout.counts);
    }
  }
  readout.timings.factorNs = elapsed(start);
  if (status != FrameStatus::kOkFrameConditional || !model.pooledVariance || !model.queryVariance)
    return status;
  start = Clock::now();
  status = evaluateK2(snapshot, closure, compressed, readout);
  readout.timings.k2Ns = elapsed(start);
  return status;
}

}  // namespace

std::string rowSetDigest(const std::vector<Row> & rows)
{
  std::string bytes;
  appendInteger(bytes, rows.size());
  for (const auto & row : rows)
    appendInteger(bytes, row.queryOrdinal);
  return sha256(bytes);
}

FrameStatus hatWeights(const Row & row, const Eigen::Vector3d & worldPoint, std::array<double, kSupportCount> & weights)
{
  if (!row.plane || row.plane->count != kSupportCount)
    return FrameStatus::kUnsupportedRowKind;
  if (!finiteRow(row) || !worldPoint.allFinite())
    return FrameStatus::kNonfiniteInput;
  const auto & plane = *row.plane;
  const double rankThreshold =
    std::numeric_limits<double>::epsilon() * static_cast<double>(kSupportCount) * plane.eigenvalues[2];
  if (!(plane.eigenvalues[2] > 0.0) || !(plane.eigenvalues[1] > rankThreshold))
    return FrameStatus::kRowLeverageUnbounded;
  const Eigen::Vector3d origin = center(row), displacement = worldPoint - origin;
  constexpr double kLeverageLimit = 1.0 / (static_cast<double>(kSupportCount) * std::numeric_limits<float>::epsilon());
  std::array<double, kSupportCount> candidate{};
  for (std::size_t a = 0; a < kSupportCount; ++a) {
    const Eigen::Vector3d local = pointVector(row.support[a]) - origin;
    candidate[a] = 1.0 / static_cast<double>(kSupportCount) +
                   displacement.dot(plane.axis1) * local.dot(plane.axis1) / plane.eigenvalues[1] +
                   displacement.dot(plane.axis2) * local.dot(plane.axis2) / plane.eigenvalues[2];
    if (!std::isfinite(candidate[a]) || std::abs(candidate[a]) > kLeverageLimit)
      return FrameStatus::kRowLeverageUnbounded;
  }
  weights = candidate;
  return FrameStatus::kOkFrameConditional;
}

FrameStatus geometricRow(const Row & row,
                         const State & state,
                         bool extrinsicEstimated,
                         const std::optional<Eigen::Vector3d> & angularVelocity,
                         StateRowJacobian & result)
{
  if (!row.plane || row.kind != RowKind::kPointPlane)
    return FrameStatus::kUnsupportedRowKind;
  if (!finiteRow(row) || !finiteState(state) || (angularVelocity && !angularVelocity->allFinite()))
    return FrameStatus::kNonfiniteInput;
  StateRowJacobian candidate = StateRowJacobian::Zero();
  const Eigen::Matrix3d rotation = state.rotation.toRotationMatrix(),
                        extrinsic = state.extrinsicRotation.toRotationMatrix();
  const Eigen::Vector3d point = extrinsic * row.point + state.extrinsicTranslation;
  const Eigen::Vector3d normal = row.plane->normal, normalBody = rotation.transpose() * normal;
  const Eigen::Vector3d rotationRow = point.cross(normalBody);
  candidate.head<3>() = normal.transpose();
  candidate.segment<3>(kRotation) = rotationRow.transpose();
  if (extrinsicEstimated) {
    candidate.segment<3>(kExtrinsicRotation) = row.point.cross(extrinsic.transpose() * normalBody).transpose();
    candidate.segment<3>(kExtrinsicTranslation) = normalBody.transpose();
  }
  if (row.relativeTime) {
    const double time = *row.relativeTime, halfTimeSquared = 0.5 * time * time;
    if (time > 0.0)
      return FrameStatus::kIdentityMismatch;
    candidate.segment<3>(kVelocity) = time * normal.transpose();
    MTK::vect<3, double> angle = angularVelocity ? Eigen::Vector3d(time * *angularVelocity) : Eigen::Vector3d::Zero();
    candidate.segment<3>(kGyroBias) = -time * rotationRow.transpose() * MTK::A_matrix(angle);
    candidate.segment<3>(kAccelBias) = -halfTimeSquared * normalBody.transpose();
    auto direction = gravity(state);
    Eigen::Matrix<double, 3, 2> gravityDerivative;
    MTK::vect<2, double> zero = Eigen::Vector2d::Zero();
    direction.S2_Mx(gravityDerivative, zero);
    candidate.tail<2>() = halfTimeSquared * normal.transpose() * gravityDerivative;
  }
  if (!candidate.allFinite())
    return FrameStatus::kNumericalFailure;
  result = candidate;
  return FrameStatus::kOkFrameConditional;
}

FrameStatus compressRows(const Eigen::MatrixXd & h,
                         const Eigen::VectorXd & observation,
                         const Eigen::MatrixXd & gain,
                         const SparseMatrix & b,
                         const Eigen::VectorXd & queryVariance,
                         double mapVariance,
                         const Budgets & budgets,
                         CompressedFrame & result,
                         Counts & counts)
{
  if (h.rows() == 0 || h.cols() == 0 || h.cols() > kStateDof || observation.size() != h.rows() ||
      gain.cols() != h.rows() || gain.rows() != h.cols() || b.rows() != h.rows())
    return FrameStatus::kIdentityMismatch;
  if (!h.allFinite() || !gain.allFinite() || !observation.allFinite())
    return FrameStatus::kNonfiniteInput;
  counts.rows = static_cast<std::size_t>(h.rows());
  counts.nnzB = static_cast<std::size_t>(b.nonZeros());
  try {
    FrameNoise noise;
    const auto status = noise.initialize(b, queryVariance, mapVariance, budgets, counts);
    if (status != FrameStatus::kOkFrameConditional)
      return status;
    return compressWithNoise(h, observation, gain, noise, budgets, result, counts);
  } catch (const std::bad_alloc &) {
    return FrameStatus::kMemoryBudget;
  }
}

FrameReadout observeFrame(const FrameSnapshot & snapshot, const Budgets & budgets)
{
  const auto start = Clock::now();
  FrameReadout result;
  result.identity = snapshot.identity;
  result.counts.rows = snapshot.rows.size();
  result.timings.captureNs = snapshot.captureDurationNs;
  if (!snapshot.diagnostics.converged)
    flag(result.flags, FrameFlag::kLinearizationNotConverged);
  if (!snapshot.nearestNeighborsRefreshed)
    flag(result.flags, FrameFlag::kNnNotRefreshed);
  if (snapshot.holdActive)
    flag(result.flags, FrameFlag::kHoldActive);
  if (snapshot.considerKeep)
    flag(result.flags, FrameFlag::kConsiderActive);
  if (snapshot.floorMode == FloorMode::kObserve)
    flag(result.flags, FrameFlag::kFloorObserve);
  flag(result.flags, FrameFlag::kAmplitudeAssumed);
  flag(result.flags, FrameFlag::kQuerySideOnly);
  if (snapshot.queryAmplitude.source == QueryAmplitudeSource::kVendorRange)
    flag(result.flags, FrameFlag::kAmplitudeRangeOnly);
  try {
    result.status = calculateFrame(snapshot, budgets, result);
  } catch (const std::bad_alloc &) {
    result.status = FrameStatus::kMemoryBudget;
  }
  if (result.status != FrameStatus::kOkFrameConditional) {
    result.readings.reset();
    result.rejectReason = statusName(result.status);
    if (result.status == FrameStatus::kK2Rejected)
      result.rejectReason += ":" + std::string(k2StatusName(result.k2Status));
  }
  result.timings.totalNs = elapsed(start);
  return result;
}

const char * statusName(FrameStatus status)
{
  switch (status) {
    case FrameStatus::kOkFrameConditional:
      return "OK_FRAME_CONDITIONAL";
    case FrameStatus::kSkippedDecimated:
      return "SKIPPED_DECIMATED";
    case FrameStatus::kNoUpdate:
      return "NO_UPDATE";
    case FrameStatus::kNonfiniteInput:
      return "NONFINITE_INPUT";
    case FrameStatus::kIdentityMismatch:
      return "IDENTITY_MISMATCH";
    case FrameStatus::kUnsupportedRowKind:
      return "UNSUPPORTED_ROW_KIND";
    case FrameStatus::kUnsupportedKernel:
      return "UNSUPPORTED_KERNEL";
    case FrameStatus::kUnsupportedFloorApply:
      return "UNSUPPORTED_FLOOR_APPLY";
    case FrameStatus::kRowLeverageUnbounded:
      return "ROW_LEVERAGE_UNBOUNDED";
    case FrameStatus::kFrameRowsOverBudget:
      return "FRAME_ROWS_OVER_BUDGET";
    case FrameStatus::kMemoryBudget:
      return "MEMORY_BUDGET";
    case FrameStatus::kRowCapacity:
      return "ROW_CAPACITY";
    case FrameStatus::kSourceCapacity:
      return "SOURCE_CAPACITY";
    case FrameStatus::kK2Rejected:
      return "K2_REJECTED";
    case FrameStatus::kNumericalFailure:
      return "NUMERICAL_FAILURE";
  }
  return "NUMERICAL_FAILURE";
}

}  // namespace fastlio::lidar_joint

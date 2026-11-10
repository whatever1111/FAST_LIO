#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "frontend_correlated_update.hpp"

namespace fastlio::lidar_joint
{

inline constexpr Eigen::Index kStateDof = 23;
inline constexpr Eigen::Index kProductionColumns = 12;
inline constexpr Eigen::Index kPoseDof = 6;
inline constexpr std::size_t kSupportCount = 5;
inline constexpr std::uint32_t kContractVersion = 2;  // Adds explicit range-amplitude provenance to design v1.
inline constexpr double kProductionVariance = 0.001;
inline constexpr double kGravityMagnitude = 98090.0 / 10000.0;
using Vector23 = Eigen::Matrix<double, kStateDof, 1>;
using Matrix23 = Eigen::Matrix<double, kStateDof, kStateDof>;
using StateRowJacobian = Eigen::Matrix<double, 1, kStateDof>;
using SparseMatrix = Eigen::SparseMatrix<double>;

enum class FrameStatus
{
  kOkFrameConditional,
  kSkippedDecimated,
  kNoUpdate,
  kNonfiniteInput,
  kIdentityMismatch,
  kUnsupportedRowKind,
  kUnsupportedKernel,
  kUnsupportedFloorApply,
  kRowLeverageUnbounded,
  kFrameRowsOverBudget,
  kMemoryBudget,
  kRowCapacity,
  kSourceCapacity,  // Reserved for B; A has no persistent source input.
  kK2Rejected,
  kNumericalFailure
};

enum class FrameFlag : std::uint32_t
{
  kLinearizationNotConverged = 1U << 0U,
  kAmplitudeAssumed = 1U << 1U,
  kSigmaPoolDofLow = 1U << 2U,
  kDeskewColumnsRigidApprox = 1U << 3U,
  kDeskewColumnsUnavailable = 1U << 4U,
  kNnNotRefreshed = 1U << 5U,
  kHoldActive = 1U << 6U,
  kConsiderActive = 1U << 7U,
  kFloorObserve = 1U << 8U,
  kPriorSymmetrized = 1U << 9U,
  kQuerySideOnly = 1U << 10U,
  kAmplitudeRangeOnly = 1U << 11U
};

enum class QueryAmplitudeSource
{
  kPooled,
  kVendorRange
};
enum class VendorRangeSpecification
{
  kTypical15mm,
  kWebsite10mm
};
enum class RowKind
{
  kPointPlane,
  kGroundRecovery
};
enum class FloorMode
{
  kOff,
  kObserve,
  kApply
};
enum class Kernel
{
  kNone,
  kRobust
};

// Manufacturer typical range precision only (50% target, channel variation).
// Angular precision is unknown. VENDOR_RANGE is a lower-bound model, never a
// complete point-noise covariance; map/angle assumptions retain AMPLITUDE_ASSUMED.
struct QueryAmplitude
{
  QueryAmplitudeSource source = QueryAmplitudeSource::kPooled;
  VendorRangeSpecification specification = VendorRangeSpecification::kTypical15mm;
};

struct State
{
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond extrinsicRotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d extrinsicTranslation = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accelBias = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -kGravityMagnitude);
};

struct SupportPoint
{
  float x = 0.0F, y = 0.0F, z = 0.0F, intensity = 0.0F;
};

struct Plane
{
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  double offset = 0.0;
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  Eigen::Vector3d axis1 = Eigen::Vector3d::UnitX();
  Eigen::Vector3d axis2 = Eigen::Vector3d::UnitY();
  std::size_t count = kSupportCount;
};

struct Row
{
  std::uint64_t queryOrdinal = 0;  // Original return index, unique within this scan.
  RowKind kind = RowKind::kPointPlane;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();  // Deskewed lidar-frame point, metres.
  // Original ray transported by deskew rotation into scan-end lidar axes,
  // without translation. Required for VENDOR_RANGE: the translated point does
  // not determine its physical beam. Magnitude is ignored; zero is rejected.
  std::optional<Eigen::Vector3d> beamDirection;
  std::optional<double> relativeTime;  // Seconds from scan end; absence is not zero.
  std::array<SupportPoint, kSupportCount> support;
  std::optional<Plane> plane;
  double weight = 1.0;
};

struct FrameIdentity
{
  std::uint64_t scanOrdinal = 0;
  std::int64_t scanEndNs = 0;
  std::uint32_t evaluationOrdinal = 0;
  std::string worldFrame;
};

struct Diagnostics
{
  bool valid = false;
  bool converged = false;
  std::uint32_t iterations = 0;
  std::size_t measurementDimension = 0;
  Vector23 lastDx = Vector23::Zero();
};

struct FrameSnapshot
{
  bool selected = true;  // The K3-2 caller owns decimation; no counter lives here.
  std::optional<FrameIdentity> identity;
  std::optional<State> prior, linearization, posterior;
  std::optional<Matrix23> priorCovariance, posteriorCovariance;
  Eigen::MatrixXd productionH, preFloorH;
  Eigen::VectorXd productionObservation, preFloorObservation;
  std::vector<Row> rows;
  Diagnostics diagnostics;
  bool extrinsicEstimated = false;
  bool holdActive = false;
  bool nearestNeighborsRefreshed = true;
  FloorMode floorMode = FloorMode::kOff;
  Kernel kernel = Kernel::kNone;
  std::optional<Eigen::Matrix3d> considerKeep;
  std::optional<Eigen::Vector3d> angularVelocity;
  std::optional<double> scanDuration;
  std::optional<std::uint64_t> captureDurationNs;
  QueryAmplitude queryAmplitude;
};

struct Budgets
{
  // Callers may lower these hard caps; larger values do not raise them.
  std::size_t rows = 8192;
  std::size_t bytes = static_cast<std::size_t>(32U) * 1024U * 1024U;
  Eigen::Index compressedRows = joint_source::kMaxRows;
};

struct Counts
{
  std::size_t rows = 0;
  std::optional<std::size_t> supportPoints, duplicateSupportReferences, nnzB, nnzFactor, accountedPeakBytes;
  std::optional<Eigen::Index> rankT, compressedRows, rankGain;
};

struct Timings
{
  std::optional<std::uint64_t> captureNs, closureNs, modelNs, factorNs, k2Ns, totalNs;
};

struct ClosureDiagnostics
{
  std::optional<double> dxError, covarianceError, stateError, tolerance, priorAsymmetry, posteriorAsymmetry;
};

using DirectionValues = std::array<std::optional<double>, kPoseDof>;
using BracketValues = std::array<DirectionValues, 3>;

struct EigenRange
{
  double minimum = 0.0, maximum = 0.0;
  Eigen::Matrix<double, kPoseDof, 1> minimumDirection = Eigen::Matrix<double, kPoseDof, 1>::Zero();
};

struct AmplitudeReadings
{
  Matrix23 optimalCovariance = Matrix23::Zero(), gainCovariance = Matrix23::Zero();
  double chi2Perpendicular = 0.0;
  Eigen::Index dofPerpendicular = 0;
  DirectionValues informationFixed, informationSchur, gainOverProduction, optimalOverProduction;
  std::optional<EigenRange> informationEigenvalues, covarianceEigenvalues;
  double gainMinusOptimalMinimum = 0.0;
};

struct Readings
{
  std::optional<double> pooledMapVariance;
  std::size_t poolDof = 0;
  QueryAmplitude queryAmplitude;
  std::optional<double> vendorRangeSigma;
  BracketValues gammaFixed, gammaSchur;
  DirectionValues deskewInformationGain;
  double deskewOffsetNorm = 0.0;
  std::array<double, 3> extrinsicRotationSensitivity{}, extrinsicTranslationSensitivity{};
  std::optional<double> clockSensitivity, gyroWhiteSensitivity, accelWhiteSensitivity;
  std::optional<AmplitudeReadings> amplitude;
};

struct FrameReadout
{
  std::uint32_t contractVersion = kContractVersion;
  std::optional<FrameIdentity> identity;
  FrameStatus status = FrameStatus::kNoUpdate;
  std::string rejectReason;
  std::uint32_t flags = 0;
  Counts counts;
  Timings timings;
  std::optional<ClosureDiagnostics> closure;
  std::optional<Readings> readings;  // Rejection clears the entire numerical model result.
  joint_source::Status k2Status = joint_source::Status::kUnknownMoment;
};

struct CompressedFrame
{
  Eigen::MatrixXd h, gain;
  Eigen::VectorXd observation;
  double chi2Perpendicular = 0.0;
  Eigen::Index dofPerpendicular = 0;
};

// Fixed-support first-order TLS weights. Failure leaves the output unchanged.
FrameStatus
hatWeights(const Row & row, const Eigen::Vector3d & worldPoint, std::array<double, kSupportCount> & weights);
FrameStatus geometricRow(const Row & row,
                         const State & state,
                         bool extrinsicEstimated,
                         const std::optional<Eigen::Vector3d> & angularVelocity,
                         StateRowJacobian & result);
std::string rowSetDigest(const std::vector<Row> & rows);

// Exact local-source marginalization and sufficient-statistic/gain-row
// compression. D must be strictly positive. No n-by-n covariance is formed.
// Resource accounting includes conservative temporary-buffer/fill bounds.
FrameStatus compressRows(const Eigen::MatrixXd & h,
                         const Eigen::VectorXd & observation,
                         const Eigen::MatrixXd & gain,
                         const SparseMatrix & b,
                         const Eigen::VectorXd & queryVariance,
                         double mapVariance,
                         const Budgets & budgets,
                         CompressedFrame & result,
                         Counts & counts);

// Each call creates an independent K2 instance, proposes and evaluates the
// actual gain, and never commits. All successful readings are FRAME_CONDITIONAL.
// No ROS types, file I/O, producer hooks, retained covariance or cross-frame state.
FrameReadout observeFrame(const FrameSnapshot & snapshot, const Budgets & budgets = {});
const char * statusName(FrameStatus status);

}  // namespace fastlio::lidar_joint

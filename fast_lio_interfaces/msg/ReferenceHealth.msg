# R1-OBS-1, schema 1. Observation only; see REFERENCE_HEALTH_CONTRACT.md.
uint16 schema_version
uint8 identity_version
bool has_source_instance
uint8[16] source_instance
string<=48 clock_domain
string<=64 frame_id
string<=64 child_frame_id
bool has_coordinate_epoch
uint64 coordinate_epoch
uint8[32] producer_contract_id
uint8[32] config_id
uint32 implemented_mask
uint32 enabled_policy_mask
bool has_snapshot_sequence
uint64 snapshot_sequence
bool has_state_revision
uint64 state_revision
bool has_scan_sequence
uint64 scan_sequence
bool has_observation_id
uint8[32] observation_id
uint8 KIND_UNKNOWN=0
uint8 KIND_SCAN=1
uint8 KIND_STATE_CHANGE=2
uint8 KIND_INPUT_PROGRESS=3
uint8 KIND_STATE_REPEAT=4
uint8 kind
bool has_evaluated_input
int64 evaluated_input_ns
bool has_effective
int64 effective_ns
bool has_state_valid_until
int64 state_valid_until_ns
bool has_scan_begin
int64 scan_begin_ns
bool has_scan_end
int64 scan_end_ns
bool has_pose_binding
float64[7] pose_binding
uint8 lifecycle                         # Unknown=0, PreReference=1, Established=2
uint8 startup_reference                 # Unknown=0, Cold=1, Established=2, Excluded=3
uint8 trust                             # Unknown=0, LocalQualified=1, Pending=2, Unanchored=3
uint8 local                             # Unknown=0, Healthy=1, Degraded=2, Invalid=3
uint32 legacy_flags                     # the unchanged low four bits
bool has_raw_registration_degraded
bool raw_registration_degraded
uint64 reference_generation             # zero is a valid generation
uint64 recovery_episode
bool has_qualified_event_sequence
uint64 qualified_event_sequence
bool has_original_start
int64 original_start_ns
bool has_original_deadline
int64 original_deadline_ns
bool has_last_supported_end
int64 last_supported_end_ns
uint32 present_mask
uint32 true_mask
bool has_negative_masks
uint32 cause_mask
uint32 taint_mask
uint32 hard_mask
uint8 hr_witness_kind                   # None=0, Normal=1, TrustedStaticHold=2, Constrained=3
bool has_hr_valid_until
int64 hr_valid_until_ns
# Reserved RW contract 3. All presence fields remain false in PR1.
bool has_rw
uint16 rw_contract
bool has_raw_episode
uint64 raw_episode_id
bool has_raw_start
int64 raw_start_ns
bool has_rw_origin
uint64 origin_scan
uint64 origin_reference_generation
bool has_frozen_residual
float64 frozen_residual_median
bool has_withdrawal_rearm
uint64 withdrawal_rearm_event
bool has_rw_valid_until
int64 rw_valid_until_ns
bool has_wheel_witness_epochs
uint64 lower_wheel_witness_epoch
uint64 upper_wheel_witness_epoch
uint8 wheel_witness_kind                # none/current-window/fallback
# Reserved g-prime support contract 1. Never populated by PR1.
bool has_support
uint16 support_contract
uint8 support_kind
bool has_support_seed
uint64 support_seed_reference
uint64 support_seed_scan
uint8[16] support_seed_source
bool has_supported_through
int64 supported_through_ns
bool has_first_loss
int64 first_loss_ns
bool has_support_deadline
int64 support_deadline_ns
bool has_proof_id
uint8[32] proof_id
bool has_trajectory_id
uint8[32] trajectory_id

# Schema 1. FAST_LIO-owned interface; independent of the estimator package.
uint16 schema_version
uint8 identity_version                 # 1: FE observation ID algorithm; 0 unknown
uint8 STATE_UNKNOWN=0
uint8 STATE_INITIALIZING=1
uint8 STATE_TRACKING=2
uint8 STATE_DEGRADED=3
uint8 STATE_LOST=4
uint8 SUBSTATE_UNKNOWN=0
uint8 SUBSTATE_NORMAL=1
uint8 SUBSTATE_BLIND=2
uint8 SUBSTATE_VERIFYING=3
uint8 SUBSTATE_STATIC_HOLD=4
uint8 SUBSTATE_RUNAWAY_GUARD=5
uint8 VALIDITY_UNKNOWN=0
uint8 VALIDITY_FULL=1
uint8 VALIDITY_PARTIAL=2
uint8 VALIDITY_INVALID=3
uint8 ACTIVITY_UNKNOWN=0
uint8 ACTIVITY_IDLE=1
uint8 ACTIVITY_BUSY=2
uint8 ACTIVITY_BLOCKED=3
uint32 GUARD_DEGRADED=1
uint32 GUARD_STATIC_HOLD=2
uint32 GUARD_REANCHOR=4
uint32 GUARD_RUNAWAY=8
uint32 GUARD_LEVELLING=16

bool has_source_instance
uint8[16] source_instance              # opaque UUID, valid only with has_source_instance
bool has_sample_sequence
uint64 sample_sequence                 # publisher order within known instance, not content ID
bool has_observation_id
uint8[32] observation_id               # FE-specific scan ID; false on idle, never GNSS raw ID
string<=48 clock_domain               # e.g. sensor clock identifier; never inferred from receipt
bool has_input_stamp
int64 input_stamp_ns                   # latest IMU or scan acquisition time in clock_domain
bool has_observation_stamp
int64 observation_stamp_ns             # scan end / pose estimate time in clock_domain
bool has_effective_observation_stamp
int64 effective_observation_stamp_ns   # guard/state change applies from this sensor instant
string<=64 frame_id                    # parent frame of pose, e.g. lidar_odom
string<=64 child_frame_id              # child frame of pose, e.g. body
bool has_coordinate_epoch
uint64 coordinate_epoch                # FE coordinate-frame context epoch, never inferred from stamp

bool has_pose                           # false on IMU-driven idle sample; no synthetic pose
geometry_msgs/PoseWithCovariance pose   # xyz/rotation in frame_id; ROS position-first 6x6
bool has_pose_covariance                # full 6x6, same estimate/frame/error convention verified; else false
bool has_twist
geometry_msgs/TwistWithCovariance twist # linear/angular in child_frame_id; no health side channel
bool has_twist_covariance               # independent of has_twist; old health cells do not count
uint8 output_validity                   # Full/Partial/Invalid for this observation, not heartbeat
uint8 axis_validity_bits                # bit0 x,1 y,2 z,3 roll,4 pitch,5 yaw
uint8 estimator_state
uint8 activity
uint8 fe_substate
bool has_guard_reason_bits
uint32 guard_reason_bits                # named bit assignments fixed before generated interface
bool has_health_flags
uint32 health_flags                     # schema-1 Degraded=1, StaticHold=2, Lost=4, Levelling=8
bool has_observability_along
float64 observability_along             # [0,1], old -1 means invalid here
bool has_effective_feature_count
uint32 effective_feature_count          # features/scan
bool has_far_point_fraction
float64 far_point_fraction              # [0,1]
bool has_mean_residual
float64 mean_residual_m                 # metres
bool has_body_speed
float64 body_speed_mps                  # metres/second
bool has_gravity
float64[3] gravity_world_mps2           # world-frame gravity xyz, metres/second squared
bool has_input_sequence
uint64 input_sequence                   # admitted IMU/scan input count in this instance
bool has_completion_sequence
uint64 completion_sequence              # completed FE scan estimate count; idle does not advance
bool has_output_sequence
uint64 output_sequence                  # valid FE pose outputs, not idle/heartbeat count
bool has_queue_count
uint32 queue_count                      # pending FE inputs at sample time
bool has_dropped_count
uint64 dropped_count                    # cumulative FE input drops within instance

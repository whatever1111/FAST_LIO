# Front-end contract fixtures

Inputs and goldens of the two contract fixtures of the legacy front-end node (laserMapping S0):

- `test/test_frontend_parameter_contract_node.cpp`: what the node does with its parameters: the declared
  defaults, the platform profiles (this package's own files and the parent repository's merged profiles), the
  deprecated alias, type errors, vector lengths, NaN and Inf, prior-map settings, the read-only descriptor, typed
  outputs with their configuration ID, the attitude hold alone and with online extrinsic estimation, two invalid
  values at once, the entry checks of the normal-noise floor and of the wheel measurement model, the floor in
  observe mode, and the wheel update with its variance topic. Per case it records the configuration
  (profile path and sha256, every override), whether construction returned or threw (type and text), what
  construction logged at INFO and above with the constructor's stage boundaries between those lines (`event.v1`),
  every declared parameter with type, value and descriptor as the node's parameter services report them, the
  outcome of fixed runtime set requests, and the configuration in effect when the constructor returned
  (`effective.v1`: every parameter's storage and the state derived from it, floating values as bit patterns).
- `test/test_frontend_pipeline_contract_node.cpp`: what the node does with a fixed input stream, through its real
  subscriptions, scan path and `/Odometry` publisher: the stage order the existing `FASTLIO_TEST_HOOKS` report,
  every scan's anchor record (outcome, state around the anchor step, pose covariance, anchor, recovery), the
  published odometry, the anchor and recovery trace and the input transaction, for bootstrap, matching, no match,
  invalid, repeated and empty scans, IMU commit, invalid, fatal and coverage gap, the input epoch, the anchor, the
  parked prior, the pin, scan holes and recovery, map resets, and the map worker with a delayed insertion. Every
  scenario also records the measurement Jacobian and residuals of each iteration (`h_matrix.v1`), the filter state
  and covariance after each lidar update (`filter_update.v1`) and the IMU samples an empty scan leaves queued
  (`imu_retain.v1`); the typed scenario records both typed outputs, field by field, with a fixed instance and a
  fixed repeat clock. The scenarios set `mapping.extrinsic_est_en` to true, which is off by default, so the recorded
  Jacobian keeps its extrinsic columns; `h_attitude_hold`, `h_extrinsic_off` and `h_m20` set it false.
  The measurement scenarios (`h_*`, laserMapping S2) turn on the H-model branches the default configuration leaves
  off, each on a synthetic scene that reaches it (three tilted walls, near ground with clutter, a corridor with a
  patch across its end): the correspondence gate's innovation test (`h_gate_innov`) and nearest-neighbour bound
  (`h_gate_nn0`), the Geman-McClure and Cauchy kernels (`h_kernel_gm`, `h_kernel_cauchy`), the neighbour refresh
  gate's skip and the reuse of the cached planes (`h_nn_refresh`), the ground recovery with sector and whole-scan
  planes (`h_ground_recovery`), the normal-noise floor applied to the Schur complement with its diagnostics
  (`h_degeneracy`), the attitude hold's scene and information rules (`h_attitude_hold`), online extrinsic
  estimation on and off (`h_extrinsic_on`, `h_extrinsic_off`), the correspondence dump (`h_dump`) and the H-model
  keys of the M20 profile with roll/pitch considered (`h_m20`). Every scenario records each H-model call's
  correspondence selection (`h_select.v1`: the search decision, the gate, kernel and ground recovery counts, and
  digests of the selection, the residuals and the rows) and its exit (`h_exit.v1`: position information, the
  hold's verdicts, roll/pitch information and consideration, and in its `degeneracy_*` fields the normal-noise
  floor's decision with the schema-2 meanings below, and since construction the transformed rows and their actual
  position-row norms); the dump scenario records
  its three files (`h_dump.v1`), which the test build writes to the scenario's result directory.

Lines whose first word is a `.v1` name (`event.v1`, `effective.v1`, `h_matrix.v1`, ...) were added to records
that existed before them; removing them gives each earlier record unchanged.

## Joint-row floor telemetry

The floor uses `schema=2 rule=joint_row_v1`. Its row contraction acts on all 12 Jacobian columns and the residual
with the same L. D1 uses the exact rotation Schur M0. `M_ridge` remains a diagnostic of the former definition.
R36 withdraws the former pure-position deletion guarantee and replaces it with deletion of the joint
position-rotation mode. R37 accepts jumps at M0 and rotation-support rank boundaries. A nonzero floor with
rank_m below three is invalid and leaves the rows unchanged. A zero floor is an exact byte-preserving no-op.
R39 measures position rank against epsilon times the row count times lambda_max(Ht-transpose Ht), before rotation
elimination. The complement is orthogonalized twice and its rank is bounded by the rows minus the rotation rank.

`h_exit.v1` keeps its existing field names and order. With the mechanism off its values keep their original meanings
and bytes. With observe or apply enabled, each serialized exit appends `schema=2 rule=joint_row_v1`,
`gain_order=tau_ascending` and `lambda_gain_pairing=independent`. The floor fields then have these definitions.

| Field | Schema-2 meaning |
| --- | --- |
| `degeneracy_engaged` | True only when apply actually commits the row contraction |
| `degeneracy_lambdas` | Ascending eigenvalues of M0 |
| `degeneracy_weights` | Generalized gains g_gen, ordered by ascending tau |
| `degeneracy_shares` | Each M0 eigenvalue divided by the effective row count, or the existing invalid sentinel |
| `degeneracy_weighted_points` | Cumulative count of rows in committed contractions |
| `degeneracy_weight_sum` | Cumulative sum of actual transformed position-row norms |

The lambda and gain arrays have independent ordering and their same-index entries are not paired. Each exit record
carries the identity needed to interpret its compatibility fields. Invalid attempted evaluations
carry nan gains and empty generalized spectra in the log. The floor-off neutral record retains zero lambdas and
unit gains. The correspondence dump keeps its columns and ordering. On a projected evaluation its existing `w`
column contains `floor_row_pos_norm`, the norm of the actual transformed position row. Each projected iteration is
preceded by `# schema=2 rule=joint_row_v1 column_w=floor_row_pos_norm t=... iter=...`. This versioned mapping applies
only to rows with the same `(t,iter)` CSV key. Readers map their `w` column to `floor_row_pos_norm`; rows without a
mapping retain the legacy `w` meaning. An unmatched point has `w=-1`. Row mixing allows an individual row norm to
grow. Observe and off retain the existing dump values and CSV bytes.

Each enabled frame emits one `[FLOOR]` line after the final snapshot is fixed. All real values use `%.17g` and
`stamp_ns` uses the same ROS conversion as `/Odometry`. Symmetric matrices use xx,xy,xz,yy,yz,zz and B is row-major
with position rows and rotation columns. The row units are `normal_equivalent`. S0 excludes kappa and the margin.

| Fields | Meaning |
| --- | --- |
| `schema rule mode t stamp_ns evaluation_iter` | Identity, scan end and last H-model iteration that evaluated the floor |
| `valid reason projected` | Decision validity, failure reason and actual application |
| `points points_without_covariance iterations invalid_iterations` | Row and per-scan evaluation counts |
| `rank_m rank_r kappa sigma_rel z a m_definition units` | Supports, model constants and exact Schur identity |
| `M M_ridge S0 lambda tau g_gen K F_sym transform_kind` | Full decision, with F_sym in the symmetric M0-root basis |
| `u u_source u_valid` | R_world_imu times UnitX from the published frame-end quaternion |
| `rD_num rD_den rD rD_valid rD_zero_denominator rD_reason` | u-transpose M0 u, u-transpose S0 u and their ratio |
| `A B C A_keep B_keep C_keep` | Original and candidate-contracted six-state Gram blocks |
| `shift_kind shift_coords_valid d_t d_r dh_norm` | Apply-only joint-shift diagnostics and actual removed residual norm |

Observe computes the same candidate decision and keep blocks while preserving H and h byte for byte. It omits
all shift fields. Off emits no FLOOR line. Invalid numerical fields are nan and invalid tau/g_gen arrays are
empty. Successfully computed M0 and S0 remain available when apply is invalid, so rD_valid is independent of valid.
The zero-floor fast path has K=M0, zero tau and unit gains, with spectrum length rank_m. Coordinate overflow marks
both d arrays nan without rejecting a finite row update. Other invalid applies keep dh_norm zero and commit no rows.

A coordinate-epoch mismatch emits `rD_reason=frame_mismatch`, invalid u and invalid rD. A zero denominator emits a
per-frame `rD_zero_denominator=1`, which the reader sums separately. All other unavailable ratios retain their
reason. Calibration readers include every computable ratio, including rank_m-invalid evaluations, and keep M0
and historical ridge-Schur datasets separate.

Each case is compared with its golden line for line, character for character. Doubles are written with 17
significant digits and NaN with its bit pattern; nothing is compared with a tolerance. Masked: the package's
source directory in recorded text (`<ROOT_DIR>/`), which differs between build machines; an effective text value
it masks is typed `string-masked` and its raw bytes stay in the case's receipt. The fixtures' headers list what is
not recorded and why. Each case's receipt (`test_results/fast_lio/frontend_contract/`) records the comparison's
result and the identity of the golden it used; a pipeline receipt also lists the branches the scenario reached
with their counts.

## Files

- `golden/parameter/<case>.golden`, `golden/pipeline/<scenario>.golden`: the goldens. The `# ` header lines name
  the fork commit, the architecture, the ROS distribution and the compile options a golden was generated with; the
  comparison reads them and compares the rest. The parameter `defaults` golden holds the whole parameter table
  and effective configuration; every other parameter golden holds only the lines of either that differ from it.
- `m20_laser_mapping.yaml`: the M20 profile's front-end parameters as the parent repository merges them, with LOST
  recovery on; the fixture cannot read the parent repository. It is a test input of the parameter fixture, not a
  contract on the parent's configuration. Provenance: LIO-SLAM commit `849299d59cddebb94048e1d01bb308557d38d0f5`,
  `config/fast_lio.base.yaml` (sha256 `06304727793dc7e85e32c421a8b9b7fdf3cec9fc6ca0867da3c24af16a7aa644`) merged
  with `config/profiles/m20/fast_lio.yaml` (sha256
  `acc340c2346b0c979b47e31656310e05288a1e3c0a33b9e233d08f4334f6f10b`) by
  `python3 tools/offline_params.py --config-root config --profile m20 --out <dir>` (its `laser_mapping.yaml`, sha256
  `28c12ac8cd52373eb9fba8f1d9a8925c4d9469313e3f23b035586884d539a6f7`). One line is added to that file:
  `lost_recovery_en: true` after `reanchor_recovery_entry_view`, the value the M20 profile takes for LOST recovery;
  sha256 `00445c3f27e3c69179e0199c616cd0b9902bfc4bc51590641169dd0b5663e5bf`.
- `giant_allpoint_laser_mapping.yaml`, `giant_laser_mapping.yaml`, `giant_hesai_fp_laser_mapping.yaml`,
  `m20_bodyimu_laser_mapping.yaml`: the parent repository's other merged profiles, made by the same command with
  `--profile <name>` at LIO-SLAM commit `849299d59cddebb94048e1d01bb308557d38d0f5` from `config/fast_lio.base.yaml`
  (sha256 above) and `config/profiles/<name>/fast_lio.yaml`; each file is that run's `laser_mapping.yaml`, unchanged.

  | profile | `fast_lio.yaml` sha256 | merged file sha256 |
  | --- | --- | --- |
  | `giant_allpoint` | `2868fa9a1907e198d2d1a8c6c6a145e2b3c2cf3887f5aa383dd6788002448257` | `f461a66ef3fb4c9b1f5d25f290af7cc84c4538171b1d4c462967dc215521d5ad` |
  | `giant` | `c4975a4893f08018a1576c4366cf68f4436616ab3a1cb0b7934093ab131dc0e1` | `3d988bbbf0b91116ccab783c14ea49fe1a0110c5ba8da88f0ecafc8dedd43311` |
  | `giant_hesai_fp` | `082ef82c5677d4269a07535229633db6f328a74199a0340da4dc979e419dcd1a` | `404e8101fc7481faa5e7964688d4b75329693dcd73896e0dd6c25c914592e600` |
  | `m20_bodyimu` | `854f72c8e04a4c51de6be08e35fb7f893c46b1dd5cf143699f57d7b375146e7c` | `a1dbe30368193e6ebf2d95a5ef528c2527a7ce98ed5e6083f7ca6b13dcff3f2d` |

## The goldens apply to x86_64 only

They are generated and compared on x86_64. On other architectures CMake registers the contract entries as skipped
and says so at configuration ("golden 只适用于 x86_64"); the executables, run there in comparison mode, fail with the
same statement instead of passing.

## Generating the goldens

The same executables generate them: `FASTLIO_CONTRACT_GOLDEN_OUT=<dir>` switches a run from comparison to
generation, and `FASTLIO_CONTRACT_SOURCE_SHA` names the fork commit it records. A generation run records every
case twice and writes a golden only when both records agree and every branch the scenario claims was reached.
Generate and compare with the same image and compile options. The goldens in this directory are generated with
those of the parent repository's cloud fast_lio job (`tools/cloud_build/inner_fastlio.sh`: Humble, Debug,
`-g1 -O1`, the options CI's FAST_LIO job compares with); from a workspace whose `src/` holds LIO-SLAM:

```sh
colcon build --packages-select fast_lio_interfaces fast_lio --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O1" "-DCMAKE_C_FLAGS_DEBUG=-g1 -O1" \
  -DBUILD_TESTING=ON
source install/setup.bash
out="$PWD/frontend_contract_out"
(cd build/fast_lio && \
  FASTLIO_CONTRACT_GOLDEN_OUT="$out" \
  FASTLIO_CONTRACT_SOURCE_SHA="$(git -C ../../src/LIO-SLAM/third_party/FAST_LIO rev-parse HEAD)" \
  ctest -R '^test_frontend_(parameter|pipeline)_contract_node' --output-on-failure)
cp "$out"/golden/parameter/*.golden src/LIO-SLAM/third_party/FAST_LIO/test/frontend_contract/golden/parameter/
cp "$out"/golden/pipeline/*.golden src/LIO-SLAM/third_party/FAST_LIO/test/frontend_contract/golden/pipeline/
```

`$out/receipts/` holds every case's receipt. Regenerate whenever the legacy node's behaviour changes on purpose or
a change adds records: generate from a commit that holds the code without the new goldens, the commit their headers
then name, and commit the goldens on their own after it. A commit that only moves code compares against them and
must reproduce every golden unchanged. Only goldens that gain records are replaced: the measurement records replaced
the pipeline goldens, and the parameter goldens keep the commit they were generated from.

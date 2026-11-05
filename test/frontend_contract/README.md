# Front-end contract fixtures

Inputs and goldens of the two contract fixtures of the legacy front-end node (laserMapping S0):

- `test/test_frontend_parameter_contract_node.cpp`: what the node does with its parameters: the declared
  defaults, the platform profiles (this package's own files and the parent repository's merged profiles), the
  deprecated alias, type errors, vector lengths, NaN and Inf, prior-map settings, the read-only descriptor, typed
  outputs with their configuration ID, and two invalid values at once. Per case it records the configuration
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
  fixed repeat clock.

Lines whose first word is a `.v1` name (`event.v1`, `effective.v1`, `h_matrix.v1`, ...) were added to records
that existed before them; removing them gives each earlier record unchanged.

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

`$out/receipts/` holds every case's receipt. Regenerate whenever the legacy node's behaviour changes on purpose;
S1 regenerates once on its own base before it starts, and every later slice compares against those goldens.

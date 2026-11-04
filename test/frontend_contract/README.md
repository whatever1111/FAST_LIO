# Front-end contract fixtures

Inputs and goldens of the two contract fixtures of the legacy front-end node (laserMapping S0):

- `test/test_frontend_parameter_contract_node.cpp`: what the node does with its parameters: the declared
  defaults, the three platform profiles, the deprecated alias, type errors, vector lengths, NaN and Inf, prior-map
  settings and the read-only descriptor. Per case it records the configuration (profile path and sha256, every
  override), whether construction returned or threw (type and text), what construction logged at INFO and above,
  every declared parameter with type, value and descriptor as the node's parameter services report them, and the
  outcome of fixed runtime set requests.
- `test/test_frontend_pipeline_contract_node.cpp`: what the node does with a fixed input stream, through its real
  subscriptions, scan path and `/Odometry` publisher: the stage order the existing `FASTLIO_TEST_HOOKS` report,
  every scan's anchor record (outcome, state around the anchor step, pose covariance, anchor, recovery), the
  published odometry, the anchor and recovery trace and the input transaction, for bootstrap, matching, no match,
  invalid, repeated and empty scans, IMU commit, invalid, fatal and coverage gap, the input epoch, the anchor, the
  parked prior, the pin, scan holes and recovery, map resets, and the map worker with a delayed insertion.

Each case is compared with its golden line for line, character for character. Doubles are written with 17
significant digits and NaN with its bit pattern; nothing is compared with a tolerance. Masked: the package's
source directory in recorded text (`<ROOT_DIR>/`), which differs between build machines. The fixtures' headers
list what is not recorded and why. Each case's receipt (`test_results/fast_lio/frontend_contract/`) records the
comparison's result and the identity of the golden it used; a pipeline receipt also lists the branches the
scenario reached with their counts, the branches the existing hooks cannot reach, and the hooks S1 adds for them.

## Files

- `golden/parameter/<case>.golden`, `golden/pipeline/<scenario>.golden`: the goldens. The `# ` header lines name
  the fork commit, the architecture, the ROS distribution and the compile options a golden was generated with; the
  comparison reads them and compares the rest. The parameter `defaults` golden holds the whole parameter table;
  every other parameter golden holds only the table lines that differ from it.
- `m20_laser_mapping.yaml`: the M20 profile's front-end parameters as the parent repository merges them; the
  fixture cannot read the parent repository. It is a test input of the parameter fixture, not a contract on the
  parent's configuration. Provenance: LIO-SLAM commit `78d278d7cb65bc0e8c144db4c2136388a17dc015`,
  `config/fast_lio.base.yaml` (sha256 `06304727793dc7e85e32c421a8b9b7fdf3cec9fc6ca0867da3c24af16a7aa644`) merged
  with `config/profiles/m20/fast_lio.yaml` (sha256
  `244e57bf7dc493fc249dbf94e09d0a36d8a5ec5a29357dbcbd84516b48b1dfaf`) by
  `python3 tools/offline_params.py --config-root config --profile m20 --out <dir>` (its `laser_mapping.yaml`,
  unchanged); sha256 `8e6575d416bcad226c25571f2efc3891b5f2d80b96a85ce28ee18fe5c499932f`.

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
`-g1 -O0`); from a workspace whose `src/` holds LIO-SLAM:

```sh
colcon build --packages-select fast_lio_interfaces fast_lio --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_CXX_FLAGS_DEBUG=-g1 -O0" "-DCMAKE_C_FLAGS_DEBUG=-g1 -O0" \
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

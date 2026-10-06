# Frozen 0906 timing fixture

`m20_0906_scan_timing.txt` contains all 8968 input scans, no point coordinates,
intensity, rings or recorded IMU values. It is 346980 bytes, below 512 KiB.

The first line is the initial raw header and adapter begin epoch in nanoseconds.
Each following line is:

```text
raw_sqlite_message_id header_delta_ns begin_delta_ns first_curvature_bits last_curvature_bits expected
```

The deltas apply to the previous line; the first scan has zero deltas. Curvature
bits are hexadecimal IEEE binary32 values in milliseconds. They preserve the
earliest and latest point times entering `Process` after the frozen adapter,
stride and blind filtering, including the float32 conversions. Reconstruct the
scan begin as `double(begin_ns) / 1e9`, then the first/last time as
`begin + double(curvature) / 1000`. Raw header stamps are retained separately.

The expected history set comes from the golden `/Odometry` recording: every gap
greater than 150 ms has exactly one raw scan strictly inside it. These 161 raw
message identities get `H`. The extraction does not call the history predicate
or the candidate's temporal model. `O` means a unique golden output epoch, `N`
means one of five duplicate/nonadvancing input scans, and `B/I/M` are the three
startup scans (bootstrap, initialization, map seed). All 8799 golden epochs are
claimed exactly once; startup and duplicates are outside the 161-hole set.

The test drives sequential `Process` calls with two synthetic points at the
retained time bounds and continuous, finite rest IMU. Bootstrap bypasses Process
as in the node. Nonadvancing scans exercise the direct Process guard without
borrowing the FIFO, matching production sync's lack of consumption. It checks
the exact rejected identities and each successful/retained watermark transition.
This timing fixture does not replace the preregistered P5 end-to-end replay.

Pinned inputs:

- Raw `m20b_0906_merged_0.db3`: `26d918a3767ec06f1ae222309bf09fd1f988b79f4c537a5f9c880df5628e4877`.
- Golden `recorded_bag_0.db3`: `7d5da5277448c83328efb770b8350607bca0176c85c270ce1a5e592730638b6e`.
- Golden manifest: `c1840b90bc36d2f4a8c7b6be2a5ef62dd35e9664f4d71d68e06d4112c95dc7d5`.
- Sealed E03 metadata reader: `7e3965e6fe0ae83ab30424199cf27725beb980057cba595a02e980a64c927b71`.

Rebuild with the task-owned script (kept outside this fork):

```bash
W=/var/tmp/lio-workspaces/p6-f10a-20261006-b59e
env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$W/tmp" \
  "$W/source/tools/ci/preflight.sh" --lane lint --memory-gb 2 --wait=3600 --run -- \
  python3 -B "$W/tools/make_0906_fixture.py"
```

`W/output-fix/checks/fixture-0906.json` records the full input hashes, script hash,
fixture hash, lease and all 161 independent pairs of golden output stamps.

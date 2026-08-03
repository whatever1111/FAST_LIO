#!/usr/bin/env python3
"""Audit FAST-LIO LiDAR/wheel Kalman-channel diagnostics.

Reads Log/kalman_channels.csv (schema v1) and reports which measurement
channel moved gravity and accelerometer bias, whether direct state deltas close,
and whether gravity gain accompanies covariance cross-coupling.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = "1"
REQUIRED_COLUMNS = (
    "schema_version",
    "scan_index",
    "time",
    "channel",
    "valid",
    "converged",
    "iterations",
    "measurement_dim",
    "residual_norm",
    "weighted_residual_squared",
    "dx_ba_x",
    "dx_ba_y",
    "dx_ba_z",
    "dx_grav_t0",
    "dx_grav_t1",
    "dba_x",
    "dba_y",
    "dba_z",
    "dgrav_x",
    "dgrav_y",
    "dgrav_z",
    "gain_grav",
    "projection_grav",
    "p_grav_pos",
    "p_grav_rot",
    "p_grav_vel",
    "p_grav_ba",
    "z0",
    "h0",
)
NUMERIC_COLUMNS = tuple(name for name in REQUIRED_COLUMNS if name not in {"schema_version", "channel"})


def load_rows(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        duplicates = sorted(name for name, count in Counter(reader.fieldnames).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate CSV columns: {', '.join(duplicates)}")
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        rows = []
        for line_number, raw in enumerate(reader, 2):
            if raw["schema_version"] != SCHEMA_VERSION:
                raise ValueError(
                    f"line {line_number}: unsupported schema_version={raw['schema_version']!r}, expected {SCHEMA_VERSION}"
                )
            if raw["channel"] not in {"lidar", "wheel"}:
                raise ValueError(f"line {line_number}: invalid channel={raw['channel']!r}")
            row = dict(raw)
            try:
                for name in NUMERIC_COLUMNS:
                    row[name] = float(raw[name])
            except ValueError as exc:
                raise ValueError(f"line {line_number}: non-numeric diagnostic field") from exc
            rows.append(row)
    if not rows:
        raise ValueError("no diagnostic rows")
    return rows


def vector_norm(row, prefix):
    return math.sqrt(sum(row[f"{prefix}_{axis}"] ** 2 for axis in ("x", "y", "z")))


def summarize(rows, windows=()):
    result = {"schema_version": int(SCHEMA_VERSION), "rows": len(rows), "channels": {}, "windows": []}
    by_channel = defaultdict(list)
    for row in rows:
        by_channel[row["channel"]].append(row)

    for channel in ("lidar", "wheel"):
        group = by_channel.get(channel, [])
        if not group:
            continue
        gravity_norms = [vector_norm(row, "dgrav") for row in group]
        ba_norms = [vector_norm(row, "dba") for row in group]
        tangent_gravity_norms = [math.hypot(row["dx_grav_t0"], row["dx_grav_t1"]) for row in group]
        tangent_ba_norms = [math.sqrt(row["dx_ba_x"] ** 2 + row["dx_ba_y"] ** 2 + row["dx_ba_z"] ** 2) for row in group]
        gravity_cross = [
            math.sqrt(
                row["p_grav_pos"] ** 2
                + row["p_grav_rot"] ** 2
                + row["p_grav_vel"] ** 2
                + row["p_grav_ba"] ** 2
            )
            for row in group
        ]
        innovations = [row["z0"] - row["h0"] for row in group]
        result["channels"][channel] = {
            "rows": len(group),
            "valid_rows": sum(bool(row["valid"]) for row in group),
            "net_dgrav": [sum(row[f"dgrav_{axis}"] for row in group) for axis in ("x", "y", "z")],
            "absolute_dgrav_norm_sum": sum(gravity_norms),
            "max_dgrav_norm": max(gravity_norms),
            "net_dba": [sum(row[f"dba_{axis}"] for row in group) for axis in ("x", "y", "z")],
            "absolute_dba_norm_sum": sum(ba_norms),
            "max_dba_norm": max(ba_norms),
            "tangent_gravity_norm_sum": sum(tangent_gravity_norms),
            "tangent_ba_norm_sum": sum(tangent_ba_norms),
            "gravity_gain_norm_mean": sum(row["gain_grav"] for row in group) / len(group),
            "gravity_gain_norm_max": max(row["gain_grav"] for row in group),
            "gravity_projection_norm_mean": sum(row["projection_grav"] for row in group) / len(group),
            "gravity_cross_covariance_mean": sum(gravity_cross) / len(group),
            "gravity_cross_covariance_max": max(gravity_cross),
            "innovation_x_mean": sum(innovations) / len(group),
            "innovation_x_rms": math.sqrt(sum(value * value for value in innovations) / len(group)),
        }

    largest = sorted(rows, key=lambda row: vector_norm(row, "dgrav"), reverse=True)[:10]
    result["largest_gravity_updates"] = [
        {
            "time": row["time"],
            "scan_index": int(row["scan_index"]),
            "channel": row["channel"],
            "dgrav_norm": vector_norm(row, "dgrav"),
            "dgrav": [row[f"dgrav_{axis}"] for axis in ("x", "y", "z")],
            "gravity_gain_norm": row["gain_grav"],
            "gravity_cross_covariance": math.sqrt(
                row["p_grav_pos"] ** 2
                + row["p_grav_rot"] ** 2
                + row["p_grav_vel"] ** 2
                + row["p_grav_ba"] ** 2
            ),
            "innovation_x": row["z0"] - row["h0"],
        }
        for row in largest
    ]

    for start, end in windows:
        selected = [row for row in rows if start <= row["time"] < end]
        result["windows"].append(
            {
                "start": start,
                "end": end,
                "rows": len(selected),
                "channel_dgrav": {
                    channel: [
                        sum(row[f"dgrav_{axis}"] for row in selected if row["channel"] == channel)
                        for axis in ("x", "y", "z")
                    ]
                    for channel in ("lidar", "wheel")
                },
            }
        )
    return result


def parse_window(value):
    try:
        start_text, end_text = value.split(":", 1)
        start, end = float(start_text), float(end_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("window must be START:END") from exc
    if not start < end:
        raise argparse.ArgumentTypeError("window START must be less than END")
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--window", action="append", default=[], type=parse_window, help="report START:END seconds")
    parser.add_argument("--json", type=Path, dest="json_path", help="write the full report as JSON")
    args = parser.parse_args()

    report = summarize(load_rows(args.csv_path), args.window)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_path is not None:
        args.json_path.write_text(text + "\n")


if __name__ == "__main__":
    main()

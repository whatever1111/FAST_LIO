import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_kalman_channels.py"
SPEC = importlib.util.spec_from_file_location("analyze_kalman_channels", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def diagnostic_row(channel="lidar", time=1.0, dgrav=(0.1, 0.0, 0.0)):
    row = {name: "0" for name in MODULE.REQUIRED_COLUMNS}
    row.update(
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "scan_index": "1",
            "time": str(time),
            "channel": channel,
            "valid": "1",
            "converged": "1",
            "iterations": "2",
            "measurement_dim": "100",
            "dgrav_x": str(dgrav[0]),
            "dgrav_y": str(dgrav[1]),
            "dgrav_z": str(dgrav[2]),
            "gain_grav": "0.25",
            "projection_grav": "0.5",
            "p_grav_pos": "0.1",
            "p_grav_rot": "0.2",
            "z0": "2.0",
            "h0": "1.5",
        }
    )
    return row


def write_rows(path, rows, fieldnames=None):
    names = fieldnames or list(MODULE.REQUIRED_COLUMNS)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_separates_channels_and_windows(tmp_path):
    path = tmp_path / "channels.csv"
    write_rows(
        path,
        [
            diagnostic_row("lidar", 1.0, (0.1, 0.0, 0.0)),
            diagnostic_row("wheel", 1.0, (0.0, 0.2, 0.0)),
            diagnostic_row("lidar", 5.0, (-0.05, 0.0, 0.0)),
        ],
    )

    report = MODULE.summarize(MODULE.load_rows(path), [(0.0, 2.0)])

    assert report["channels"]["lidar"]["rows"] == 2
    assert report["channels"]["lidar"]["net_dgrav"] == pytest.approx([0.05, 0.0, 0.0])
    assert report["channels"]["wheel"]["net_dgrav"] == pytest.approx([0.0, 0.2, 0.0])
    assert report["channels"]["wheel"]["innovation_x_mean"] == pytest.approx(0.5)
    assert report["windows"][0]["channel_dgrav"]["lidar"] == pytest.approx([0.1, 0.0, 0.0])
    assert report["windows"][0]["channel_dgrav"]["wheel"] == pytest.approx([0.0, 0.2, 0.0])


def test_missing_required_column_is_rejected(tmp_path):
    path = tmp_path / "missing.csv"
    names = [name for name in MODULE.REQUIRED_COLUMNS if name != "gain_grav"]
    write_rows(path, [diagnostic_row()], names)

    with pytest.raises(ValueError, match="missing required columns: gain_grav"):
        MODULE.load_rows(path)


def test_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "schema.csv"
    row = diagnostic_row()
    row["schema_version"] = "99"
    write_rows(path, [row])

    with pytest.raises(ValueError, match="unsupported schema_version"):
        MODULE.load_rows(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    write_rows(path, [])

    with pytest.raises(ValueError, match="no diagnostic rows"):
        MODULE.load_rows(path)

"""Real normal-process scan/idle/clock-fault contract, without estimator hooks.

Small synthetic Ouster planes exercise subscriptions, preprocessing and the EKF.
This is bounded runtime contract evidence, not accuracy, load or target DDS proof.
"""

import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix
from fast_lio_interfaces.msg import FrontendObservation
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def qos(depth, reliable=True):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def cloud(start, offset=0.0):
    message = PointCloud2()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(start, 1_000_000_000)
    message.header.frame_id = "lidar"
    message.height, message.width = 1, 2000
    message.is_dense = True
    message.point_step = 32
    message.row_step = message.point_step * message.width
    for name, position, kind in (
        ("x", 0, PointField.FLOAT32),
        ("y", 4, PointField.FLOAT32),
        ("z", 8, PointField.FLOAT32),
        ("intensity", 12, PointField.FLOAT32),
        ("t", 16, PointField.UINT32),
        ("reflectivity", 20, PointField.UINT16),
        ("ring", 22, PointField.UINT8),
        ("ambient", 24, PointField.UINT16),
        ("range", 28, PointField.UINT32),
    ):
        message.fields.append(PointField(name=name, offset=position, datatype=kind, count=1))
    data = bytearray(message.row_step)
    for i in range(message.width):
        struct.pack_into(
            "<ffffIHBBH2xI",
            data,
            i * message.point_step,
            10.0 + offset + (i % 100) * 0.02,
            -0.2 + (i // 100) * 0.02,
            1.0,
            1.0,
            i * 50_000_000 // (message.width - 1),
            0,
            0,
            0,
            0,
            0,
        )
    message.data = bytes(data)
    return message


def evidence(sample):
    return {
        "sample": sample.sample_sequence,
        "completion": sample.completion_sequence,
        "output": sample.output_sequence,
        "input": sample.input_sequence,
        "has_pose": sample.has_pose,
        "stamp_ns": sample.observation_stamp_ns,
        "instance": bytes(sample.source_instance).hex(),
        "id": bytes(sample.observation_id).hex() if sample.has_observation_id else None,
        "validity": sample.output_validity,
        "activity": sample.activity,
        "has_count": sample.has_effective_feature_count,
        "count": sample.effective_feature_count,
        "has_residual": sample.has_mean_residual,
        "residual": sample.mean_residual_m,
    }


def test_real_scans_idle_duplicate_rollback_restart():
    output = Path(os.environ["FASTLIO_SCAN_CONTRACT_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    context = rclpy.context.Context()
    rclpy.init(context=context)
    executor = SingleThreadedExecutor(context=context)
    observer = rclpy.create_node("scan_contract_" + uuid.uuid4().hex, context=context)
    prefix = "/scan_contract_" + uuid.uuid4().hex
    samples, odometry = deque(maxlen=256), deque(maxlen=128)
    subscription = observer.create_subscription(FrontendObservation, prefix + "/typed", samples.append, qos(64, False))
    odom_subscription = observer.create_subscription(Odometry, prefix + "/odom", odometry.append, qos(32))
    imu = observer.create_publisher(Imu, prefix + "/imu", qos(1000))
    lidar = observer.create_publisher(PointCloud2, prefix + "/points", qos(10))
    binary = Path(get_package_prefix("fast_lio")) / "lib/fast_lio/fastlio_mapping"
    domain = "fixture_scan_clock"
    records = []

    def settle(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            rclpy.spin_once(observer, executor=executor, timeout_sec=0.01)

    def wait(predicate, timeout=8.0):
        until = time.monotonic() + timeout
        while not predicate() and time.monotonic() < until:
            rclpy.spin_once(observer, executor=executor, timeout_sec=0.01)
        assert predicate(), "timed out waiting for real scan contract; inspect activation log and receipt"

    def activation(index, exercise_faults):
        samples.clear()
        odometry.clear()
        parameters = {
            "common.imu_topic": prefix + "/imu",
            "common.lid_topic": prefix + "/points",
            "preprocess.lidar_type": "3",
            "preprocess.timestamp_unit": "3",
            "point_filter_num": "1",
            "feature_extract_enable": "false",
            "filter_size_surf": "0.001",
            "filter_size_map": "0.05",
            "imu_init_require_still": "false",
            "publish.scan_publish_en": "false",
            "publish.path_en": "false",
            "runtime_pos_log_enable": "false",
            "pcd_save.pcd_save_en": "false",
            "publish.frontend_observation": "true",
            "publish.frontend_clock_domain": domain,
            "mapping.extrinsic_T": "[0.0, 0.0, 0.0]",
            "mapping.extrinsic_R": "[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
        }
        args = [str(binary), "--ros-args", "-r", f"__node:=scan_subject_{index}"]
        for key, value in parameters.items():
            args += ["-p", f"{key}:={value}"]
        args += ["-r", "/Odometry:=" + prefix + "/odom", "-r", "/lio_slam/frontend_observation:=" + prefix + "/typed"]
        environment = dict(os.environ)
        environment.pop("FLIO_OFFLINE_STEP", None)  # exercise the normal production timer
        log_path = output / f"activation_{index}.log"
        record = {"activation": index, "binary": str(binary), "checks": {}}
        records.append(record)
        next_imu = 9_990_000_000

        def send_imu(through):
            nonlocal next_imu
            while next_imu <= through:
                msg = Imu()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(next_imu, 1_000_000_000)
                msg.linear_acceleration.z = 9.81
                imu.publish(msg)
                next_imu += 5_000_000
                settle(0.001)
            settle(0.05)

        def send_scan(start, offset=0.0):
            send_imu(start + 60_000_000)
            lidar.publish(cloud(start, offset))
            settle(0.15)

        def completed_near(start):
            return next(
                (s for s in reversed(samples) if s.has_pose and abs(s.observation_stamp_ns - start - 50_000_000) <= 2),
                None,
            )

        def verify_completed(sample):
            assert sample.has_source_instance and sample.has_observation_id
            assert sample.has_observation_stamp and sample.has_coordinate_epoch
            assert sample.has_completion_sequence and sample.has_output_sequence
            wait(lambda: any(stamp_ns(o.header.stamp) == sample.observation_stamp_ns for o in odometry))
            legacy = next(o for o in odometry if stamp_ns(o.header.stamp) == sample.observation_stamp_ns)
            assert sample.frame_id == legacy.header.frame_id
            assert sample.child_frame_id == legacy.child_frame_id
            assert sample.pose.pose == legacy.pose.pose
            payload = (
                b"LIO-FE-OBS-V1\0"
                + bytes(sample.source_instance)
                + struct.pack(">QqI", sample.completion_sequence, sample.observation_stamp_ns, len(domain))
                + domain.encode()
            )
            assert bytes(sample.observation_id) == hashlib.sha256(payload).digest()
            assert not sample.has_pose_covariance and not sample.has_twist and not sample.has_twist_covariance
            if sample.output_validity == FrontendObservation.VALIDITY_FULL:
                assert sample.axis_validity_bits == 63
            if sample.output_validity in (FrontendObservation.VALIDITY_UNKNOWN, FrontendObservation.VALIDITY_INVALID):
                assert sample.axis_validity_bits == 0

        with log_path.open("w") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=environment)
            try:
                wait(lambda: imu.get_subscription_count() == lidar.get_subscription_count() == 1)
                wait(lambda: observer.count_publishers(prefix + "/typed") == 1)
                if hasattr(subscription, "get_publisher_count"):
                    wait(lambda: subscription.get_publisher_count() == 1)
                if hasattr(odom_subscription, "get_publisher_count"):
                    wait(lambda: odom_subscription.get_publisher_count() == 1)
                start = 10_000_000_000
                record["warmup_issued_stamps_ns"] = []  # bounded by the fixed 15-scan input below
                for warmup_index in range(15):
                    record["warmup_issued_stamps_ns"].append(start)
                    send_scan(start)
                    if warmup_index < 14:
                        start += 100_000_000
                # Drain through the last issued scan before sending the zero-row scan.
                # Do not select an older positive sample while warmup scans remain queued.
                wait_started = time.monotonic()
                try:
                    wait(lambda: completed_near(start) is not None)
                finally:
                    matching = completed_near(start)
                    latest = next((sample for sample in reversed(samples) if sample.has_pose), None)
                    record["warmup_wait"] = {
                        "last_issued_start_ns": start,
                        "expected_end_ns": start + 50_000_000,
                        "timeout_seconds": 8.0,
                        "steady_elapsed_seconds": time.monotonic() - wait_started,
                        "matching_stamp_ns": matching.observation_stamp_ns if matching else None,
                        "latest_completed": evidence(latest) if latest else None,
                    }
                positive = completed_near(start)
                assert positive is not None, "warmup did not complete its last issued scan"
                assert positive.has_effective_feature_count and positive.effective_feature_count > 0
                verify_completed(positive)
                assert positive.has_mean_residual and math.isfinite(positive.mean_residual_m)
                assert positive.mean_residual_m >= 0
                record["checks"]["positive_rows"] = evidence(positive)
                identity = bytes(positive.source_instance)
                if not exercise_faults:
                    return identity

                # A separated plane has no map neighbours; it still completes a scan.
                start += 100_000_000
                send_scan(start, offset=100.0)
                wait(lambda: completed_near(start) is not None)
                zero = completed_near(start)
                verify_completed(zero)
                assert zero.completion_sequence == positive.completion_sequence + 1
                assert zero.has_effective_feature_count and zero.effective_feature_count == 0
                assert not zero.has_mean_residual  # never borrow the positive scan's residual
                record["checks"]["zero_rows"] = evidence(zero)

                # Re-send the same scan end: consumed-scan guard must prevent a second completion.
                odom_count = len(odometry)
                lidar.publish(cloud(start, offset=100.0))
                settle(0.2)
                send_imu(start + 700_000_000)
                wait(
                    lambda: samples and samples[-1].sample_sequence > zero.sample_sequence and not samples[-1].has_pose
                )
                idle = samples[-1]
                assert idle.completion_sequence == zero.completion_sequence
                assert idle.output_sequence == zero.output_sequence
                assert len(odometry) == odom_count
                assert not idle.has_observation_id and not idle.has_effective_feature_count
                assert not idle.has_mean_residual and not idle.has_observability_along
                assert idle.input_sequence > zero.input_sequence
                record["checks"]["duplicate_and_imu_only"] = evidence(idle)

                # >1s rollback latches the production input-epoch guard; no synthetic pose/reset.
                rollback = Imu()
                rollback.header.stamp.sec = 1
                rollback.linear_acceleration.z = 9.81
                imu.publish(rollback)
                wait(lambda: samples[-1].activity == FrontendObservation.ACTIVITY_BLOCKED)
                blocked = samples[-1]
                assert not blocked.has_pose and not blocked.has_observation_id and not blocked.has_coordinate_epoch
                assert blocked.completion_sequence == zero.completion_sequence
                assert blocked.output_sequence == zero.output_sequence
                assert bytes(blocked.source_instance) == identity
                settle(0.15)
                assert len(odometry) == odom_count
                record["checks"]["rollback"] = evidence(blocked)
                return identity
            finally:
                original_error = sys.exc_info()[1]
                killed = False
                cleanup_error = None
                shutdown_signal = signal.SIGINT if index == 0 else signal.SIGTERM
                shutdown_started = time.monotonic()
                try:
                    if process.poll() is None:
                        process.send_signal(shutdown_signal)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        killed = True
                        process.kill()
                        process.wait(timeout=5)
                except Exception as error:
                    cleanup_error = repr(error)
                record.update(
                    returncode=process.returncode,
                    shutdown_signal=shutdown_signal.name,
                    shutdown_elapsed_seconds=time.monotonic() - shutdown_started,
                    sigkill=killed,
                    cleanup_error=cleanup_error,
                    original_error=repr(original_error) if original_error else None,
                    samples=[evidence(s) for s in samples],
                    odometry_stamps=[stamp_ns(o.header.stamp) for o in odometry],
                )
                (output / "receipt.json").write_text(json.dumps(records, indent=2) + "\n")
                if original_error is not None:
                    print(
                        f"scan subject cleanup: rc={process.returncode}, kill={killed}, error={cleanup_error}",
                        file=sys.stderr,
                    )
                else:
                    assert cleanup_error is None and not killed and process.returncode == 0, record
                    wait(lambda: imu.get_subscription_count() == lidar.get_subscription_count() == 0)
                    settle(0.1)

    try:
        first = activation(0, True)
        second = activation(1, False)
        assert first != second, "independent process activation reused the producer UUID"
    finally:
        observer.destroy_node()
        executor.shutdown()
        rclpy.shutdown(context=context)

"""Public-topic actual-node contract: idle, stopped inputs, default-off, restart.

Scan/guard numerical equivalence is a separate frozen-input replay gate. This
fixture never substitutes helper verdict tests for a real completed scan.
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix
from fast_lio_interfaces.msg import FrontendObservation
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


def test_frontend_idle_and_restart(tmp_path):
    context = rclpy.context.Context()
    rclpy.init(context=context)
    executor = SingleThreadedExecutor(context=context)
    observer = rclpy.create_node("frontend_contract_" + uuid.uuid4().hex, context=context)
    prefix = "/frontend_contract_" + uuid.uuid4().hex
    imu_topic = prefix + "/imu"
    observation_topic = prefix + "/observation"
    samples = []
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=32,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    subscription = observer.create_subscription(FrontendObservation, observation_topic, samples.append, qos)
    # The subject's envelope writer is best effort, volatile and keeps one sample, so an
    # envelope it publishes before it has matched the reader above reaches no one; the
    # graph shows only this side of that match. The subject learns this node's readers from
    # its subscription announcements, which it receives reliably and in order, and this
    # /rosout reader is announced after the envelope reader: a /rosout message from the
    # subject shows it has matched the envelope reader too.
    rosout_senders = set()
    rosout_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1000,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    rosout = observer.create_subscription(Log, "/rosout", lambda log: rosout_senders.add(log.name), rosout_qos)
    imu_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    publisher = observer.create_publisher(Imu, imu_topic, imu_qos)
    binary = Path(get_package_prefix("fast_lio")) / "lib/fast_lio/fastlio_mapping"

    def spin_until(predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(observer, executor=executor, timeout_sec=0.02)
        assert predicate(), "DDS contract condition timed out"

    def settle(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(observer, executor=executor, timeout_sec=0.02)

    def activation(index, enabled):
        samples.clear()
        log_path = tmp_path / f"activation_{index}.log"
        subject = f"frontend_subject_{index}"
        args = [str(binary), "--ros-args", "-r", f"__node:={subject}"]
        parameters = {
            "common.imu_topic": imu_topic,
            "common.lid_topic": prefix + "/lidar",
            "mapping.extrinsic_T": "[0.0, 0.0, 0.0]",
            "mapping.extrinsic_R": "[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]",
            "publish.frontend_clock_domain": "fixture_sensor_epoch_1",
            "pcd_save.pcd_save_en": "false",
        }
        if enabled:
            parameters["publish.frontend_observation"] = "true"
        for key, value in parameters.items():
            args += ["-p", f"{key}:={value}"]
        args += ["-r", "/lio_slam/frontend_observation:=" + observation_topic]
        with log_path.open("w") as log:
            process = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "FLIO_OFFLINE_STEP": "1"}
            )
            try:
                spin_until(lambda: publisher.get_subscription_count() == 1)
                if enabled:
                    spin_until(
                        lambda: (
                            [info.node_name for info in observer.get_publishers_info_by_topic(observation_topic)]
                            == [subject]
                        )
                    )
                    if hasattr(subscription, "get_publisher_count"):
                        spin_until(lambda: subscription.get_publisher_count() == 1)
                    spin_until(lambda: subject in rosout_senders)
                for stamp_ns in (100_000_000_000, 100_100_000_000, 100_500_000_000):
                    msg = Imu()
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
                    msg.linear_acceleration.z = 9.81
                    publisher.publish(msg)
                    settle(0.15)
                assert process.poll() is None, log_path.read_text()
                if not enabled:
                    assert not samples
                    assert observer.count_publishers(observation_topic) == 0
                    return None
                spin_until(lambda: len(samples) >= 2)
                assert len(samples) == 2  # 0.1s input is throttled, 0.5s input publishes
                for sample in samples:
                    assert sample.schema_version == 1
                    assert sample.has_source_instance
                    assert not sample.has_pose and not sample.has_observation_id
                    assert not sample.has_observation_stamp
                    assert sample.output_validity == FrontendObservation.VALIDITY_UNKNOWN
                    assert sample.axis_validity_bits == 0
                    assert sample.completion_sequence == sample.output_sequence == 0
                    assert not sample.has_twist and not sample.has_pose_covariance
                assert samples[0].input_sequence == 1
                assert samples[1].input_sequence == 3
                assert samples[1].queue_count == 3
                assert samples[1].sample_sequence == samples[0].sample_sequence + 1
                assert bytes(samples[0].source_instance) == bytes(samples[1].source_instance)
                before = len(samples)
                settle(0.6)  # no input: no wall timer heartbeat, even across an idle period
                assert len(samples) == before
                return bytes(samples[0].source_instance)
            finally:
                original_error = sys.exc_info()[1]
                killed = False
                cleanup_error = None
                try:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        killed = True
                        process.kill()
                        process.wait(timeout=5)
                except Exception as error:
                    cleanup_error = repr(error)
                receipt = {
                    "returncode": process.returncode,
                    "sigkill": killed,
                    "cleanup_error": cleanup_error,
                    "original_error": repr(original_error) if original_error is not None else None,
                }
                (tmp_path / f"activation_{index}.exit.json").write_text(json.dumps(receipt) + "\n")
                if original_error is not None:
                    print(f"subject cleanup: {receipt}", file=sys.stderr)
                else:
                    assert cleanup_error is None, receipt
                    assert not killed, "subject required SIGKILL; shutdown contract failed"
                    assert process.returncode == 0, log_path.read_text()
                    spin_until(lambda: publisher.get_subscription_count() == 0)
                    settle(0.1)

    try:
        activation(0, False)
        first = activation(1, True)
        second = activation(2, True)
        assert first != second
    finally:
        observer.destroy_subscription(rosout)
        observer.destroy_subscription(subscription)
        observer.destroy_node()
        executor.shutdown()
        rclpy.shutdown(context=context)

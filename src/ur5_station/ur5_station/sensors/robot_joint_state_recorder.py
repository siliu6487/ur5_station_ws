#!/usr/bin/env python3
import csv
from pathlib import Path
from threading import Event, Lock
from typing import List, Optional, Tuple

from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINT_ORDER: List[str] = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

_FOLDER = "joint_states"

_MIN_SAMPLES = 5
_MIN_DURATION_S = 0.1
_MAX_GAP_S = 1.0


class RobotJointStateRecorder:
    """
    Subscribes to /joint_states continuously.
    Call start_recording()/stop_recording() to capture a window, then
    save() and validate() to write and check the output folder.

    Output folder: <trial_dir>/joint_states/
        joint_states.csv   — timestamp + 6 joint positions
        joint_states.jpg   — position vs time plot
    """

    def __init__(self, node: Node):
        self._node = node
        self._lock = Lock()
        self._recording = False
        self._rows: List[Tuple] = []
        self._has_data = Event()
        self._sub = node.create_subscription(
            JointState, "/joint_states", self._cb, 50
        )

    def _cb(self, msg: JointState) -> None:
        self._has_data.set()
        if not self._recording:
            return
        by_name = dict(zip(msg.name, msg.position))
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        row = (ts,) + tuple(by_name.get(j, float("nan")) for j in ARM_JOINT_ORDER)
        with self._lock:
            self._rows.append(row)

    def health_check(self, timeout: float = 5.0) -> bool:
        ok = self._has_data.wait(timeout=timeout)
        if not ok:
            self._node.get_logger().error(
                f"[JointStateRecorder] /joint_states not received within {timeout:.0f}s"
            )
        else:
            self._node.get_logger().info("[JointStateRecorder] /joint_states OK")
        return ok

    def start_recording(self) -> None:
        with self._lock:
            self._rows = []
            self._recording = True
        self._node.get_logger().info("[JointStateRecorder] Recording started")

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            count = len(self._rows)
        self._node.get_logger().info(
            f"[JointStateRecorder] Recording stopped ({count} samples)"
        )

    def save(self, output_dir: Path) -> Path:
        folder = Path(output_dir) / _FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / "joint_states.csv"

        with self._lock:
            rows = list(self._rows)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + ARM_JOINT_ORDER)
            writer.writerows(rows)

        self._node.get_logger().info(
            f"[JointStateRecorder] Saved {len(rows)} rows → {out_path}"
        )
        self._save_plot(folder, rows)
        return out_path

    def _save_plot(self, folder: Path, rows: list) -> None:
        if not rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            timestamps = [float(r[0]) for r in rows]
            t0 = timestamps[0]
            t_rel = [t - t0 for t in timestamps]

            fig, ax = plt.subplots(figsize=(10, 4))
            for i, name in enumerate(ARM_JOINT_ORDER):
                vals = [float(r[i + 1]) for r in rows]
                label = name.replace("left_", "").replace("_joint", "")
                ax.plot(t_rel, vals, label=label)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (rad)")
            ax.set_title("Joint Positions")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(folder / "joint_states.jpg"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            self._node.get_logger().warn(
                f"[JointStateRecorder] Plot skipped: {exc}"
            )

    def validate(self, output_dir: Path) -> dict:
        csv_path = Path(output_dir) / _FOLDER / "joint_states.csv"
        report: dict = {"status": "ok", "issues": []}

        if not csv_path.exists():
            return {"status": "fail", "issues": ["joint_states/joint_states.csv missing"]}

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            rows = list(reader)

        sample_count = len(rows)
        report["sample_count"] = sample_count

        if sample_count < _MIN_SAMPLES:
            report["issues"].append(
                f"only {sample_count} samples (min {_MIN_SAMPLES})"
            )

        if sample_count >= 2:
            timestamps = [float(r[0]) for r in rows]
            duration = timestamps[-1] - timestamps[0]
            report["duration_sec"] = round(duration, 3)

            if duration < _MIN_DURATION_S:
                report["issues"].append(f"duration {duration:.3f}s is too short")

            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i - 1]:
                    report["issues"].append(
                        f"non-monotonic timestamp at row {i}"
                    )
                    break

            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                if gap > _MAX_GAP_S:
                    report["issues"].append(
                        f"gap of {gap:.2f}s between rows {i - 1} and {i}"
                    )
                    break

            if duration > 0:
                report["estimated_hz"] = round(sample_count / duration, 1)

            # Range over the whole recording, not first-vs-last: a trial that
            # returns to its start (poke and retract) would otherwise read as 0.
            try:
                columns = list(zip(*([float(v) for v in r[1:]] for r in rows)))
                max_delta = max(max(c) - min(c) for c in columns)
                report["max_joint_delta_rad"] = round(max_delta, 4)
                if max_delta < 0.001:
                    report["issues"].append("robot did not appear to move")
            except (ValueError, IndexError):
                pass

        if report["issues"]:
            report["status"] = "warn"
        return report

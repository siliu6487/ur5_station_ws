from __future__ import annotations

import csv
import math
from pathlib import Path
from threading import Event, Lock
from typing import List, Tuple, TYPE_CHECKING

from geometry_msgs.msg import WrenchStamped

if TYPE_CHECKING:
    from rclpy.node import Node

_FOLDER = "torque_force"

_MIN_SAMPLES = 5
_MIN_DURATION_S = 0.5
_MAX_GAP_S = 0.5


class WristFTRecorder:
    """
    Records wrist force/torque from geometry_msgs/WrenchStamped.

    Subscribes at __init__ so data is ready before any trial starts.
    Buffers rows in memory during recording; writes on save().

    Discover the correct topic at the robot station:
        ros2 topic list | grep -i 'force\\|wrench\\|ft\\|torque'
    Then set sensors.wrist_force_torque.topic in workspace.yaml.

    Output folder: <trial_dir>/torque_force/
        wrist_force_torque.csv   — timestamp, force_x/y/z, torque_x/y/z
        wrist_force_torque.jpg   — force and torque vs time plot
    """

    def __init__(self, node: "Node", topic: str) -> None:
        self._node = node
        self._topic = topic
        self._lock = Lock()
        self._recording = False
        self._rows: List[Tuple] = []
        self._has_data = Event()
        self._sub = node.create_subscription(
            WrenchStamped, topic, self._cb, 100
        )

    def _cb(self, msg: WrenchStamped) -> None:
        self._has_data.set()
        if not self._recording:
            return
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        f = msg.wrench.force
        t = msg.wrench.torque
        with self._lock:
            self._rows.append((ts, f.x, f.y, f.z, t.x, t.y, t.z))

    def health_check(self, timeout: float = 3.0) -> bool:
        ok = self._has_data.wait(timeout=timeout)
        if ok:
            self._node.get_logger().info(f"[WristFTRecorder] {self._topic} OK")
        else:
            self._node.get_logger().warn(
                f"[WristFTRecorder] No message on {self._topic} within {timeout:.0f}s"
            )
        return ok

    def start_recording(self) -> None:
        with self._lock:
            self._rows = []
            self._recording = True
        self._node.get_logger().info("[WristFTRecorder] Recording started")

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            count = len(self._rows)
        self._node.get_logger().info(
            f"[WristFTRecorder] Recording stopped ({count} samples)"
        )

    def save(self, output_dir: Path) -> Path:
        folder = Path(output_dir) / _FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "wrist_force_torque.csv"

        with self._lock:
            rows = list(self._rows)

        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "force_x", "force_y", "force_z",
                 "torque_x", "torque_y", "torque_z"]
            )
            writer.writerows(rows)

        self._node.get_logger().info(
            f"[WristFTRecorder] Saved {len(rows)} rows → {out}"
        )
        self._save_plot(folder, rows)
        return out

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

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            for i, label in enumerate(["fx", "fy", "fz"]):
                ax1.plot(t_rel, [float(r[i + 1]) for r in rows], label=label)
            ax1.set_ylabel("Force (N)")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3)

            for i, label in enumerate(["tx", "ty", "tz"]):
                ax2.plot(t_rel, [float(r[i + 4]) for r in rows], label=label)
            ax2.set_ylabel("Torque (Nm)")
            ax2.set_xlabel("Time (s)")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, alpha=0.3)

            fig.suptitle("Wrist Force / Torque")
            fig.tight_layout()
            fig.savefig(str(folder / "wrist_force_torque.jpg"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            self._node.get_logger().warn(
                f"[WristFTRecorder] Plot skipped: {exc}"
            )

    def validate(self, output_dir: Path) -> dict:
        out = Path(output_dir) / _FOLDER / "wrist_force_torque.csv"
        report: dict = {"status": "ok", "issues": []}

        if not out.exists():
            return {"status": "fail", "issues": ["torque_force/wrist_force_torque.csv missing"]}

        with open(out) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            rows = list(reader)

        n = len(rows)
        report["sample_count"] = n

        if n < _MIN_SAMPLES:
            report["issues"].append(f"only {n} samples (min {_MIN_SAMPLES})")

        if n >= 2:
            timestamps = [float(r[0]) for r in rows]
            duration = timestamps[-1] - timestamps[0]
            report["duration_sec"] = round(duration, 3)

            if duration < _MIN_DURATION_S:
                report["issues"].append(
                    f"duration {duration:.3f}s too short (min {_MIN_DURATION_S}s)"
                )

            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i - 1]:
                    report["issues"].append(f"non-monotonic timestamp at row {i}")
                    break

            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                if gap > _MAX_GAP_S:
                    report["issues"].append(
                        f"gap of {gap:.2f}s between rows {i - 1} and {i}"
                    )
                    break

            if duration > 0:
                report["estimated_hz"] = round(n / duration, 1)

            try:
                force_norms = [
                    math.sqrt(float(r[1])**2 + float(r[2])**2 + float(r[3])**2)
                    for r in rows
                ]
                torque_norms = [
                    math.sqrt(float(r[4])**2 + float(r[5])**2 + float(r[6])**2)
                    for r in rows
                ]
                report["max_force_norm_N"] = round(max(force_norms), 3)
                report["max_torque_norm_Nm"] = round(max(torque_norms), 3)
            except (ValueError, IndexError):
                pass

        if report["issues"]:
            report["status"] = "warn" if n >= _MIN_SAMPLES else "fail"

        return report

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

if TYPE_CHECKING:
    from rclpy.node import Node

_FILENAME = "video.mp4"
_METADATA = "video_metadata.csv"

# Tried in order; the first one OpenCV can actually open wins. avc1 (H.264)
# gives ~10x smaller files than mp4v but needs an ffmpeg build that supports it,
# which is not guaranteed across machines — hence the fallback.
_CODECS = ("avc1", "mp4v")

_DEFAULT_FPS = 30.0
_MIN_FRAMES = 5
_MIN_DURATION_S = 0.5


class CameraRecorder:
    """
    Records an RGB camera stream from a sensor_msgs/msg/CompressedImage topic
    into a single <trial_dir>/video.mp4.

    Subscribes at __init__ so the subscription is established before any trial
    starts — avoids DDS setup latency dropping the first frames.

    The callback buffers the *compressed* JPEG bytes rather than decoded frames:
    at 30 Hz a 640x480 BGR frame is ~0.9 MB (~27 MB/s) versus ~50 KB compressed,
    and decoding on the executor thread would compete with the guard watchdog
    during a poke. Decode + encode happen once, in save().

    Playback fps is measured from the frame timestamps rather than assumed, so
    the mp4 plays back at real speed even if the publisher drifts off its
    nominal rate.

    Alignment
    ---------
    The mp4 alone is NOT enough to align video frames against the other streams:
    it is encoded at a single constant fps, so its playback clock is uniform,
    while the real arrival times are not (DDS drops, scheduler jitter). Endpoints
    plus a nominal rate would therefore drift after any dropped frame.

    save() also writes video_metadata.csv, one row per frame actually written:

        video_idx, t_wall, t_header, video_time_s

      video_idx    — frame index inside video.mp4
      t_wall       — time.time() when the message was received. THIS IS THE
                     ALIGNMENT KEY: it is the same clock as tactile_metadata.csv
                     and events.csv.
      t_header     — msg.header.stamp, the publisher's own clock (0.0 when the
                     publisher does not set it). Kept separate rather than mixed
                     into t_wall, so a driver stamping from camera hardware time
                     cannot silently shift the trial clock.
      video_time_s — video_idx / fps, i.e. where the frame sits in the mp4.

    So: seek time in the player -> video_time_s -> t_wall -> any other stream.

    Output: <trial_dir>/video.mp4
            <trial_dir>/video_metadata.csv

    Usage:
        camera  = CameraRecorder(node, topic='/camera/.../compressed')
        manager = RecordingManager(node, output_root, camera_recorder=camera)
    """

    def __init__(
        self,
        node: "Node",
        topic: str = "/camera/realsense2_camera/color/image_raw/compressed",
        queue_size: int = 60,
    ) -> None:
        self._node = node
        self._topic = topic
        self._lock = threading.Lock()
        self._recording = False
        # (t_wall, t_header, jpeg_bytes). Both clocks are kept: t_wall is the
        # trial-wide reference (same clock as tactile/events), t_header is the
        # publisher's own stamp, which may come from a different clock domain.
        self._frames: List[Tuple[float, float, bytes]] = []
        # Depth 60 ≈ 2 s of buffer at 30 Hz, so a slow scheduler tick during a
        # poke drops nothing.
        self._sub = self._node.create_subscription(
            CompressedImage, self._topic, self._callback, queue_size
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        log = self._node.get_logger()
        log.info(f"[CameraRecorder] Checking {self._topic} ...")
        received: List[CompressedImage] = []
        got = threading.Event()

        def _cb(msg):
            if not received:
                received.append(msg)
            got.set()

        tmp_sub = self._node.create_subscription(
            CompressedImage, self._topic, _cb, 10
        )
        ok = got.wait(timeout=3.0)
        self._node.destroy_subscription(tmp_sub)

        if not ok:
            log.warn(f"[CameraRecorder] No message on {self._topic} within 3 s")
            return False

        frame = self._decode(received[0])
        if frame is None:
            log.error("[CameraRecorder] Received a message but could not decode it")
            return False

        h, w = frame.shape[:2]
        log.info(
            f"[CameraRecorder] Health check OK — {w}x{h}, "
            f"format='{received[0].format}'"
        )
        return True

    def start_recording(self) -> None:
        with self._lock:
            self._frames = []
            self._recording = True
        self._node.get_logger().info("[CameraRecorder] Recording started")

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            n = len(self._frames)
        self._node.get_logger().info(f"[CameraRecorder] Recording stopped ({n} frames)")

    def _callback(self, msg: CompressedImage) -> None:
        t_wall = time.time()
        stamp = msg.header.stamp
        t_header = stamp.sec + stamp.nanosec * 1e-9
        with self._lock:
            if not self._recording:
                return
            self._frames.append((t_wall, t_header, bytes(msg.data)))

    # ── encoding ─────────────────────────────────────────────────────────────

    @staticmethod
    def _decode(msg: CompressedImage) -> Optional[np.ndarray]:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _measured_fps(self, timestamps: List[float]) -> float:
        """Real capture rate, so playback speed matches wall-clock reality."""
        if len(timestamps) < 2:
            return _DEFAULT_FPS
        span = timestamps[-1] - timestamps[0]
        if span <= 0.0:
            return _DEFAULT_FPS
        return (len(timestamps) - 1) / span

    def _open_writer(self, out: Path, fps: float, size: Tuple[int, int]):
        for codec in _CODECS:
            writer = cv2.VideoWriter(
                str(out), cv2.VideoWriter_fourcc(*codec), fps, size
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
        return None, None

    def save(self, output_dir: Path) -> Optional[Path]:
        log = self._node.get_logger()
        out = Path(output_dir) / _FILENAME
        out.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            frames = list(self._frames)

        if not frames:
            log.warn("[CameraRecorder] No frames captured — video.mp4 not written")
            return None

        timestamps = [t for t, _, _ in frames]
        fps = self._measured_fps(timestamps)

        first = cv2.imdecode(np.frombuffer(frames[0][2], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            log.error("[CameraRecorder] First frame failed to decode — nothing saved")
            return None
        h, w = first.shape[:2]

        writer, codec = self._open_writer(out, fps, (w, h))
        if writer is None:
            log.error(
                f"[CameraRecorder] No usable codec (tried {', '.join(_CODECS)}) — "
                f"nothing saved"
            )
            return None

        # Rows are built from the frames ACTUALLY written: an undecodable frame
        # is absent from the mp4, so buffer index and video index diverge and
        # only the latter indexes the file the user will scrub through.
        rows = []
        n_written, n_bad = 0, 0
        for t_wall, t_header, data in frames:
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                n_bad += 1
                continue
            # A mid-stream resolution change would silently produce a corrupt
            # file, since VideoWriter is fixed at the size given above.
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            rows.append((n_written, t_wall, t_header, n_written / fps))
            n_written += 1
        writer.release()

        meta = Path(output_dir) / _METADATA
        with open(meta, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["video_idx", "t_wall", "t_header", "video_time_s"])
            for idx, t_wall, t_header, vt in rows:
                wr.writerow([idx, f"{t_wall:.6f}", f"{t_header:.6f}", f"{vt:.6f}"])

        duration = timestamps[-1] - timestamps[0]
        note = f", {n_bad} undecodable frame(s) skipped" if n_bad else ""
        gaps = self._gap_summary([r[1] for r in rows], fps)
        log.info(
            f"[CameraRecorder] Saved {n_written} frames "
            f"({duration:.2f}s @ {fps:.1f} fps, {w}x{h}, {codec}) → {out}{note}"
        )
        log.info(f"[CameraRecorder] Timing index → {meta}{gaps}")
        return out

    @staticmethod
    def _gap_summary(timestamps: List[float], fps: float) -> str:
        """Flag inter-frame gaps that look like dropped frames.

        Uniform playback hides drops entirely, so this is the only warning the
        user gets that a stretch of the video advances faster than wall clock.
        """
        if len(timestamps) < 3 or fps <= 0:
            return ""
        nominal = 1.0 / fps
        big = [
            (i, timestamps[i] - timestamps[i - 1])
            for i in range(1, len(timestamps))
            if timestamps[i] - timestamps[i - 1] > 2.0 * nominal
        ]
        if not big:
            return "  (no frame gaps)"
        worst = max(g for _, g in big)
        return f"  ({len(big)} gap(s) > 2x nominal, worst {worst * 1000:.0f} ms)"

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self, output_dir: Path) -> dict:
        out = Path(output_dir) / _FILENAME
        if not out.exists():
            return {"status": "fail", "issues": [f"{_FILENAME} missing"]}

        report: dict = {"status": "ok", "issues": []}
        report["size_bytes"] = out.stat().st_size

        cap = cv2.VideoCapture(str(out))
        if not cap.isOpened():
            cap.release()
            return {"status": "fail", "issues": [f"could not open {_FILENAME}"]}

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = n_frames / fps if fps > 0 else 0.0
        report.update(
            {
                "n_frames": n_frames,
                "fps": round(fps, 2),
                "resolution": f"{width}x{height}",
                "duration_sec": round(duration, 3),
            }
        )

        if n_frames < _MIN_FRAMES:
            report["issues"].append(f"only {n_frames} frames (min {_MIN_FRAMES})")
        if duration < _MIN_DURATION_S:
            report["issues"].append(
                f"duration {duration:.3f}s too short (min {_MIN_DURATION_S}s)"
            )
        if report["size_bytes"] == 0:
            report["issues"].append("file is empty")

        # The timing index is what makes the video alignable; a video without it
        # is only watchable, so treat it as an issue rather than an absence.
        meta = Path(output_dir) / _METADATA
        if not meta.exists():
            report["issues"].append(f"{_METADATA} missing — video cannot be aligned")
        else:
            with open(meta) as f:
                rows = list(csv.DictReader(f))
            report["index_rows"] = len(rows)
            if len(rows) != n_frames:
                report["issues"].append(
                    f"{_METADATA} has {len(rows)} rows but video has {n_frames} frames"
                )
            if rows:
                t = [float(r["t_wall"]) for r in rows]
                report["t_wall_start"] = round(t[0], 6)
                report["t_wall_end"] = round(t[-1], 6)
                report["wall_duration_sec"] = round(t[-1] - t[0], 3)
                if any(t[i] <= t[i - 1] for i in range(1, len(t))):
                    report["issues"].append("non-monotonic t_wall in index")
                if all(float(r["t_header"]) == 0.0 for r in rows):
                    report["header_stamps"] = "absent (publisher left them unset)"

        if report["issues"]:
            report["status"] = "warn"
        return report

#!/usr/bin/env python3
"""
TactileRecorder — captures full-resolution JPEG frame sequences from the
GelSight Mini over USB (V4L2).

Output layout inside the trial directory:
    tactile/
        tactile_metadata.csv   # frame_idx,timestamp,filename
        frame_000000.jpg
        frame_000001.jpg
        ...

No crop or resize is applied. Raw sensor frames are saved at JPEG quality 92.
"""
import csv
import glob
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rclpy.node import Node

from ur5_station.sensors.marker_tracker import (
    detect_markers_with_fallback,
    inner_region_mask,
    reference_marker_set,
    locate_reference_markers,
    fit_marker_grid,
)

_JPEG_QUALITY = 92
_MIN_FRAMES = 5
_MIN_DURATION_S = 0.5
_MAX_GAP_S = 0.5        # flag if any inter-frame gap exceeds this
_MIN_FPS = 18.0         # overall average fps floor
_FPS_WINDOW_S = 1.0     # sliding window size for per-second fps check
_MIN_FPS_PER_WINDOW = 18
_RECORDING_WARMUP_S = 0.5  # flush before each trial to let AWB/AGC settle


def find_gelsight_device() -> str:
    """Return the /dev/v4l/by-id/ path of the GelSight Mini. Raise if there isn't one.

    Deliberately does NOT fall back to another camera. This used to return paths[0]
    when no GelSight name matched, which sorts to the RealSense on this machine — so
    with the GelSight unplugged, a trial would record depth-camera frames into
    tactile/ and look superficially fine, with only health_check in the way.
    See dev_notes/logs/09_24_log.md §6.

    Raises:
        RuntimeError: no GelSight-named device present, listing what was found so the
            failure is diagnosable without going to look at /dev by hand.
    """
    paths = sorted(glob.glob("/dev/v4l/by-id/*"))
    for p in paths:
        if "gelsight" in p.lower():
            return p
    found = "\n  ".join(paths) if paths else "(none)"
    raise RuntimeError(
        "No GelSight Mini found under /dev/v4l/by-id/.\n"
        f"Devices present:\n  {found}\n"
        "Refusing to substitute another camera. If you really mean to use one of the "
        "above, pass device=<path> explicitly."
    )


class TactileRecorder:
    """
    Records JPEG frame sequences from the GelSight Mini tactile sensor.

    Frames are resized to target_size before saving (full border retained, no crop).
    Raw sensor resolution is 3280×1464; default target is 320×240.

    Usage (mirrors RobotJointStateRecorder interface):
        recorder.health_check()
        recorder.start_recording()
        ...robot motion...
        recorder.stop_recording()
        recorder.save(trial_dir)
        recorder.validate(trial_dir)
    """

    def __init__(
        self,
        node: Node,
        device: str = None,
        target_size: Tuple[int, int] = (320, 240),
    ):
        """
        Args:
            node:        ROS2 node used for logging.
            device:      V4L2 device path (e.g. '/dev/v4l/by-id/usb-...')
                         or integer index (e.g. 0). Auto-detected if None.
            target_size: (width, height) to resize each frame before saving.
                         Set to None to save at full sensor resolution.
        """
        self._node = node
        self._device = device
        self._target_size = target_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._recording = False
        # List of (unix_timestamp, bgr_frame_ndarray)
        self._frames: List[Tuple[float, object]] = []
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def _resolved_device(self):
        """Return device path/index to pass to VideoCapture."""
        if self._device is not None:
            return self._device
        dev = find_gelsight_device()
        if dev is None:
            raise RuntimeError(
                "No V4L2 device found under /dev/v4l/by-id/ — "
                "is the GelSight Mini plugged in?"
            )
        self._node.get_logger().info(f"[TactileRecorder] Auto-detected device: {dev}")
        self._device = dev
        return dev

    def _open_cap(self) -> cv2.VideoCapture:
        device = self._resolved_device()
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"[TactileRecorder] Could not open device: {device}")
        return cap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def health_check(
        self,
        warmup_s: float = 2.0,
        confirm_s: float = 1.0,
        check_markers: bool = True,
        expected_markers: int = 63,
        border_fraction: float = 0.15,
        snapshot_path: Optional[Path] = None,
        marker_binary_threshold: Optional[float] = None,
        raw_snapshot_path: Optional[Path] = None,
        reference_frame_bgr: Optional[np.ndarray] = None,
        reference_centers: Optional[np.ndarray] = None,
        grid_shape=None,
    ) -> bool:
        """
        Warm up the sensor, confirm fps, and optionally check marker detection.

        warmup_s     — seconds to flush before measuring fps (GStreamer + auto-exposure settle)
        confirm_s    — seconds to count frames for the fps measurement
        check_markers — run MarkerTracker on one warm frame; warn if count != expected_markers
        snapshot_path — if given, save an annotated marker-check image here so the user can
                        inspect detected blobs (orange overlay), inner markers (green dots),
                        border markers (yellow dots), and fallback-tracked markers (blue dots)
                        before committing to a trial
        marker_binary_threshold — override MarkerTracker's blob-detection threshold (default
                        -0.05). Loosen (move toward 0, e.g. -0.03) if markers with reduced
                        contrast — e.g. from a mounted tool locally compressing the gel — are
                        being missed. None = use MarkerTracker's default.
        raw_snapshot_path — if given, also save the plain (unannotated) frame that produced
                        snapshot_path — guaranteed to be the exact same frame, unlike calling
                        snapshot() separately afterward.
        reference_frame_bgr / reference_centers — a known-good prior capture (frame + (N,2)
                        (row, col) marker positions). Any reference marker fresh blob detection
                        misses on this frame is recovered via LK optical flow from the
                        reference instead — more robust than loosening binary_threshold, which
                        can pick up false positives elsewhere without actually fixing genuinely
                        merged/occluded markers. None = fresh detection only, no fallback.
        """
        try:
            cap = self._open_cap()
        except RuntimeError as e:
            self._node.get_logger().error(f"[TactileRecorder] Health check FAILED: {e}")
            return False

        log = self._node.get_logger()

        # ── First frame sanity check ──────────────────────────────────────────
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            log.error("[TactileRecorder] Device opened but could not read a frame")
            return False

        raw_h, raw_w = frame.shape[:2]
        log.info(f"[TactileRecorder] Warming up ({warmup_s:.0f}s flush + {confirm_s:.0f}s confirm) ...")

        # ── Warm-up flush: discard frames for warmup_s seconds ───────────────
        t_start = time.time()
        flush_count = 0
        while time.time() - t_start < warmup_s:
            cap.read()
            flush_count += 1

        # ── Confirmation window: count frames to measure actual fps ──────────
        t_confirm = time.time()
        confirm_count = 0
        while time.time() - t_confirm < confirm_s:
            ret, _ = cap.read()
            if ret:
                confirm_count += 1

        # ── Grab one stable frame for marker check before releasing ──────────
        marker_frame = None
        if check_markers:
            ret, marker_frame = cap.read()
            if not ret:
                marker_frame = None

        cap.release()

        actual_fps = confirm_count / confirm_s if confirm_s > 0 else 0.0

        if self._target_size is not None:
            save_w, save_h = self._target_size
            size_str = f"raw {raw_w}×{raw_h} → save {save_w}×{save_h} px"
        else:
            size_str = f"{raw_w}×{raw_h} px"

        if actual_fps < _MIN_FPS:
            log.error(
                f"[TactileRecorder] Health check FAILED — fps after warm-up: "
                f"{actual_fps:.1f} (need ≥{_MIN_FPS})"
            )
            return False

        log.info(
            f"[TactileRecorder] Health check OK — {size_str}  "
            f"fps after warm-up: {actual_fps:.1f}  (flushed {flush_count} frames)"
        )

        # ── Marker check (non-blocking: warns but does not fail health_check) ─
        if check_markers and marker_frame is not None:
            self._check_markers_on_frame(
                marker_frame, expected_markers, border_fraction, snapshot_path,
                binary_threshold=marker_binary_threshold,
                raw_snapshot_path=raw_snapshot_path,
                reference_frame_bgr=reference_frame_bgr,
                reference_centers=reference_centers,
                grid_shape=grid_shape,
            )

        return True

    def _check_markers_on_frame(
        self,
        frame_bgr: np.ndarray,
        expected_markers: int,
        border_fraction: float,
        snapshot_path: Optional[Path],
        binary_threshold: Optional[float] = None,
        raw_snapshot_path: Optional[Path] = None,
        reference_frame_bgr: Optional[np.ndarray] = None,
        reference_centers: Optional[np.ndarray] = None,
        grid_shape=None,
    ) -> int:
        """
        Locate markers on a single BGR frame, log count vs expected, and optionally
        save an annotated snapshot.

        When a reference is supplied the marker set comes from the reference lattice
        (see reference_marker_set), giving exactly the markers the flow analysis
        tracks, in the same order — the mask is then a faithful preview of what will
        be measured, not an independent second opinion.

        Visualisation layers in the snapshot:
          orange overlay  — raw detected blobs (shows merged blobs too)
          green rectangle — inner-region boundary (border_fraction)
          green dots      — freshly detected inner markers
          yellow dots     — freshly detected border markers (excluded zone)
          blue dots       — inner markers recovered via reference fallback tracking
                            (see reference_frame_bgr / reference_centers)

        raw_snapshot_path — if given, also save the plain (unannotated) frame that was
                        actually analyzed. Use this instead of a separate snapshot() call
                        when you need the raw image and marker_mask to be the exact same
                        frame for debugging — snapshot() opens the device again later and
                        grabs a different frame, so it won't match snapshot_path pixel-for-
                        pixel.
        reference_frame_bgr / reference_centers — see health_check().

        Returns the inner-region marker count.
        """
        log = self._node.get_logger()

        # MarkerTracker's default min/max_blob_size assume the saved (smaller)
        # resolution. Running it on the full-res frame throws those size
        # estimates off, so resize first for consistent blob detection.
        if self._target_size is not None:
            frame_bgr = cv2.resize(frame_bgr, self._target_size)
            if reference_frame_bgr is not None:
                reference_frame_bgr = cv2.resize(reference_frame_bgr, self._target_size)

        if raw_snapshot_path is not None:
            raw_snapshot_path = Path(raw_snapshot_path)
            raw_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_snapshot_path), frame_bgr)

        tracker_kwargs = {} if binary_threshold is None else {"binary_threshold": binary_threshold}
        H, W = frame_bgr.shape[:2]

        if reference_frame_bgr is not None and reference_centers is not None \
                and len(reference_centers):
            # Anchor to the reference lattice, exactly as MarkerFlowTracker does,
            # so this mask and the trial's flow analysis describe the SAME markers.
            # Detecting independently on each frame made them disagree: the two run
            # on different captures, so fresh detection found different blobs and
            # each appended its own strays.
            ref_set, _grid_info = reference_marker_set(
                reference_centers, frame_bgr, border_fraction, grid_shape,
            )
            all_centers, source, _found, marker_mask = locate_reference_markers(
                frame_bgr, reference_frame_bgr, ref_set,
                **tracker_kwargs,
            )
        else:
            all_centers, source, marker_mask = detect_markers_with_fallback(
                frame_bgr,
                reference_frame_bgr=reference_frame_bgr,
                reference_centers=reference_centers,
                **tracker_kwargs,
            )  # all_centers: (N, 2) as (row, col); source: "detected"|"tracked"
            # No reference to anchor to, so filter the lattice on this frame
            # instead. Without it the mask shows every vignette blob and split
            # detection, which is what made it disagree with the flow analysis.
            if grid_shape is not None and len(all_centers):
                on_grid, _, _, _ = fit_marker_grid(all_centers, grid_shape)
                all_centers, source = all_centers[on_grid], source[on_grid]
        r_min = H * border_fraction
        r_max = H * (1 - border_fraction)
        c_min = W * border_fraction
        c_max = W * (1 - border_fraction)

        inner_mask = inner_region_mask(all_centers, H, W, border_fraction)
        inner_centers = all_centers[inner_mask]
        inner_source = source[inner_mask]
        border_centers = all_centers[~inner_mask]
        n_inner = len(inner_centers)
        n_border = len(border_centers)
        n_total = len(all_centers)
        n_tracked = int((inner_source == "tracked").sum())

        tracked_note = f"  ({n_tracked} via reference fallback)" if n_tracked else ""
        if n_inner > expected_markers:
            log.warn(
                f"[TactileRecorder] Marker check: {n_inner} inner markers, "
                f"{n_inner - expected_markers} MORE than expected ({expected_markers}) — "
                f"likely false positives (vignette/corners/noise), not missed markers; "
                f"{n_border} in border zone; {n_total} total blobs{tracked_note}"
            )
        elif n_inner < expected_markers:
            log.warn(
                f"[TactileRecorder] Marker check: {n_inner} inner markers, "
                f"{expected_markers - n_inner} FEWER than expected ({expected_markers}) — "
                f"likely missed/merged markers; {n_border} in border zone; "
                f"{n_total} total blobs{tracked_note}"
            )
        else:
            log.info(
                f"[TactileRecorder] Marker check OK — {n_inner}/{expected_markers} "
                f"inner markers, {n_border} in border zone{tracked_note}"
            )

        if snapshot_path is not None:
            vis = frame_bgr.copy()

            # Orange overlay: raw detected blobs (merged blobs appear as one large blob)
            blob_pixels = marker_mask > 0
            overlay = vis.copy()
            overlay[blob_pixels] = (0, 140, 255)  # BGR orange
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

            # Inner-region boundary
            cv2.rectangle(
                vis,
                (int(c_min), int(r_min)),
                (int(c_max), int(r_max)),
                (0, 255, 0), 2,
            )

            # Inner markers: green dots for fresh detections, blue for reference fallback
            for (r, c), src in zip(inner_centers, inner_source):
                color = (255, 128, 0) if src == "tracked" else (0, 255, 0)  # BGR blue / green
                cv2.circle(vis, (int(c), int(r)), 5, color, -1)

            # Border markers: yellow dots
            for r, c in border_centers:
                cv2.circle(vis, (int(c), int(r)), 5, (0, 255, 255), -1)

            label_color = (0, 255, 0) if n_inner == expected_markers else (0, 0, 255)
            cv2.putText(
                vis,
                f"inner: {n_inner}/{expected_markers}  border: {n_border}{tracked_note}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2,
            )

            snap = Path(snapshot_path)
            snap.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap), vis)
            log.info(f"[TactileRecorder] Marker snapshot → {snap}")

        return n_inner

    def snapshot(
        self, output_dir: Path, name: str = "snapshot", warmup_s: float = 0.0
    ) -> Optional[Path]:
        """Capture one frame and save it as <name>.jpg. Returns the saved path.

        warmup_s — seconds to flush frames before capturing, letting AWB/AGC settle
                   (same idea as health_check()'s warmup). 0 = capture immediately,
                   which on this camera comes out under-exposed/darker than a
                   settled frame — use a couple seconds for anything you intend to
                   compare against a warmed-up capture (e.g. a marker reference).
        """
        try:
            cap = self._open_cap()
            t_start = time.time()
            while time.time() - t_start < warmup_s:
                cap.read()
            ret, frame = cap.read()
            cap.release()
        except RuntimeError as e:
            self._node.get_logger().error(f"[TactileRecorder] Snapshot failed: {e}")
            return None

        if not ret or frame is None:
            self._node.get_logger().error("[TactileRecorder] Snapshot: empty frame")
            return None

        if self._target_size is not None:
            frame = cv2.resize(frame, self._target_size)

        out = Path(output_dir) / f"{name}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        self._node.get_logger().info(f"[TactileRecorder] Snapshot → {out}")
        return out

    def start_recording(self) -> None:
        if self._recording:
            return
        self._cap = self._open_cap()
        # Flush frames so AWB/AGC have settled before trial data is saved.
        t_flush = time.time()
        flush_count = 0
        while time.time() - t_flush < _RECORDING_WARMUP_S:
            self._cap.read()
            flush_count += 1
        self._node.get_logger().info(
            f"[TactileRecorder] Pre-recording flush done ({flush_count} frames, "
            f"{_RECORDING_WARMUP_S:.1f}s)"
        )
        with self._lock:
            self._frames = []
            self._recording = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="tactile_capture"
        )
        self._thread.start()
        self._node.get_logger().info("[TactileRecorder] Recording started")

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            count = len(self._frames)
        self._node.get_logger().info(
            f"[TactileRecorder] Recording stopped ({count} frames)"
        )

    def save(self, output_dir: Path) -> Path:
        """Write frames to <output_dir>/tactile/ and the CSV index."""
        with self._lock:
            frames = list(self._frames)

        img_dir = Path(output_dir) / "tactile"
        img_dir.mkdir(parents=True, exist_ok=True)

        csv_path = img_dir / "tactile_metadata.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "timestamp", "filename"])
            for idx, (ts, frame) in enumerate(frames):
                fname = f"frame_{idx:06d}.jpg"
                cv2.imwrite(
                    str(img_dir / fname),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                )
                writer.writerow([idx, f"{ts:.6f}", fname])

        self._node.get_logger().info(
            f"[TactileRecorder] Saved {len(frames)} frames → {img_dir}"
        )
        return csv_path

    def validate(self, output_dir: Path) -> dict:
        """Validate the tactile/ directory produced by save()."""
        img_dir = Path(output_dir) / "tactile"
        csv_path = img_dir / "tactile_metadata.csv"
        report: dict = {"status": "ok", "issues": []}

        if not csv_path.exists():
            return {
                "status": "fail",
                "issues": ["tactile/tactile_metadata.csv missing"],
            }

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        frame_count = len(rows)
        report["frames_saved"] = frame_count

        if frame_count < _MIN_FRAMES:
            report["issues"].append(
                f"only {frame_count} frames (min {_MIN_FRAMES})"
            )

        missing = [r["filename"] for r in rows if not (img_dir / r["filename"]).exists()]
        report["missing_files"] = len(missing)
        if missing:
            report["issues"].append(f"{len(missing)} frame file(s) missing on disk")

        if frame_count >= 2:
            timestamps = [float(r["timestamp"]) for r in rows]
            duration = timestamps[-1] - timestamps[0]
            report["duration_sec"] = round(duration, 3)
            report["first_frame"] = f"tactile/{rows[0]['filename']}"
            report["last_frame"] = f"tactile/{rows[-1]['filename']}"

            if duration < _MIN_DURATION_S:
                report["issues"].append(f"duration {duration:.3f}s is too short")

            for i in range(1, len(timestamps)):
                if timestamps[i] <= timestamps[i - 1]:
                    report["issues"].append(
                        f"non-monotonic timestamp at frame {i}"
                    )
                    break

            gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            max_gap = max(gaps)
            if max_gap > _MAX_GAP_S:
                report["issues"].append(f"max inter-frame gap {max_gap:.2f}s exceeds {_MAX_GAP_S}s")

            if duration > 0:
                fps = frame_count / duration
                report["estimated_fps"] = round(fps, 1)
                if fps < _MIN_FPS:
                    report["issues"].append(
                        f"estimated fps {fps:.1f} below minimum {_MIN_FPS}"
                    )

            # Per-second window check: every 1-second slice must have >= _MIN_FPS_PER_WINDOW frames.
            # Uses non-overlapping windows aligned to t=0 of the recording.
            t0 = timestamps[0]
            t_end = timestamps[-1]
            low_windows = []
            t = t0
            while t + _FPS_WINDOW_S <= t_end + 1e-6:
                count = sum(1 for ts in timestamps if t <= ts < t + _FPS_WINDOW_S)
                if count < _MIN_FPS_PER_WINDOW:
                    low_windows.append((round(t - t0, 2), count))
                t += _FPS_WINDOW_S
            report["low_fps_windows"] = low_windows
            if low_windows:
                worst = min(c for _, c in low_windows)
                report["issues"].append(
                    f"{len(low_windows)} 1-second window(s) below {_MIN_FPS_PER_WINDOW} fps "
                    f"(worst: {worst} frames/s) at t={[w[0] for w in low_windows]}"
                )

        if report["issues"]:
            report["status"] = "warn" if frame_count >= _MIN_FRAMES else "fail"

        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> tuple:
        """Return (timestamp, bgr_frame) of the most recently captured frame.

        Returns (None, None) if recording has not started or no frame has arrived yet.
        Safe to call from any thread while recording is active.
        """
        with self._lock:
            if not self._frames:
                return None, None
            ts, frame = self._frames[-1]
            return ts, frame.copy()

    def _capture_loop(self) -> None:
        """Background thread: reads frames from cap, resizes, and appends to _frames."""
        while True:
            with self._lock:
                if not self._recording:
                    break
            ret, frame = self._cap.read()
            if not ret:
                self._node.get_logger().warn(
                    "[TactileRecorder] Frame read failed — stopping capture"
                )
                break
            ts = time.time()
            if self._target_size is not None:
                frame = cv2.resize(frame, self._target_size)
            with self._lock:
                self._frames.append((ts, frame))

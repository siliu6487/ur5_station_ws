#!/usr/bin/env python3
import csv
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from rclpy.node import Node

from ur5_station.sensors.robot_joint_state_recorder import RobotJointStateRecorder
from ur5_station.sensors.tactile_recorder import TactileRecorder

# start_trial_recording(overwrite=...) modes
_OVERWRITE_MODES = ("ask", "always", "never")


# Entries written BEFORE a trial starts (sensor reference captures, health-check
# output). They describe the setup the trial runs under, not the trial's own data,
# so they never count as "this trial dir is occupied" and survive an overwrite.
# Add any other pre-trial artefact name here.
_PRE_TRIAL_ENTRIES = frozenset({"tactile_check", "sensor_check", "health_check"})


def _dir_summary(paths):
    """(file count, total bytes) across the given files/directories."""
    files = []
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*") if f.is_file())
    return len(files), sum(f.stat().st_size for f in files)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class RecordingManager:
    """
    Manages sensor recorders over a trial lifecycle.

    Usage:
        manager.health_check_all()
        manager.start_trial_recording(trial_id, metadata)   # prompts if data exists
        manager.mark_event("motion_begin")
        ...
        manager.stop_trial_recording()
        manager.validate_last_trial()

    A recording directory holds exactly one recording: start_trial_recording()
    clears any existing contents first (see _prepare_trial_dir). Headless callers
    must pass overwrite='always' or 'never', since the default 'ask' needs stdin.

    Sections
    --------
    A behavior that records several windows per trial (a slide records its
    descend, stroke and retract separately) passes a `section` name, and that
    window is written to <trial_dir>/<section>/ instead. The two directories are
    kept apart on purpose:

        trial_dir      always the trial itself, ending in the trial number —
                       the stable handle for the trial as a whole, and what to
                       delete or archive
        recording_dir  where the window that closed last actually wrote —
                       trial_dir/<section> when sectioned, trial_dir otherwise

    Everything written per-recording (events.csv, metadata.yaml, the recorders'
    own output, validation_report.yaml) lands in recording_dir. Anything that
    describes the trial as a whole — the pre-trial sensor check, a combined
    behavior_results.yaml — belongs in trial_dir, alongside the section folders.
    """

    def __init__(
        self,
        node: Node,
        output_root: Path,
        tactile_recorder: Optional[TactileRecorder] = None,
        wrist_ft_recorder=None,   # Optional[WristFTRecorder]
        audio_recorder=None,      # Optional[AudioRecorder]
        camera_recorder=None,     # Optional[CameraRecorder]
    ):
        self._node = node
        self._output_root = Path(output_root)
        self._joint_recorder = RobotJointStateRecorder(node)
        self._tactile_recorder = tactile_recorder
        self._wrist_ft_recorder = wrist_ft_recorder
        self._audio_recorder = audio_recorder
        self._camera_recorder = camera_recorder
        self._trial_dir: Optional[Path] = None
        self._recording_dir: Optional[Path] = None
        self._trial_id: Optional[str] = None
        self._section: Optional[str] = None
        self._metadata: dict = {}
        self._events: List[Dict] = []

    def health_check_all(
        self,
        marker_snapshot_path: Optional[Path] = None,
        marker_binary_threshold: Optional[float] = None,
        marker_raw_snapshot_path: Optional[Path] = None,
        marker_reference_frame_bgr: Optional[np.ndarray] = None,
        marker_reference_centers: Optional[np.ndarray] = None,
        marker_grid_shape=None,
    ) -> bool:
        self._node.get_logger().info("[RecordingManager] Running health checks ...")
        ok = self._joint_recorder.health_check()
        if self._tactile_recorder is not None:
            ok = self._tactile_recorder.health_check(
                snapshot_path=marker_snapshot_path,
                marker_binary_threshold=marker_binary_threshold,
                raw_snapshot_path=marker_raw_snapshot_path,
                reference_frame_bgr=marker_reference_frame_bgr,
                reference_centers=marker_reference_centers,
                grid_shape=marker_grid_shape,
            ) and ok
        if self._wrist_ft_recorder is not None:
            ok = self._wrist_ft_recorder.health_check() and ok
        if self._audio_recorder is not None:
            ok = self._audio_recorder.health_check() and ok
        if self._camera_recorder is not None:
            ok = self._camera_recorder.health_check() and ok
        if ok:
            self._node.get_logger().info("[RecordingManager] All health checks passed")
        else:
            self._node.get_logger().error("[RecordingManager] Health check FAILED")
        return ok

    def _prepare_trial_dir(self, trial_dir: Path, overwrite: str) -> None:
        """
        Ensure trial_dir holds no data from a previous recording.

        Recorders write per-stream files (frame_000000.jpg, *.csv, metadata.yaml)
        and none of their save() methods clear the directory first. Writing a
        shorter trial over a longer one therefore leaves orphaned files behind —
        stale frames that the CSV index does not list but a glob still finds.
        Clearing up front makes a trial directory mean exactly one recording.

        Anything named in _PRE_TRIAL_ENTRIES is exempt: it is setup captured
        before the trial ran, so it neither blocks recording nor gets deleted.

        Destructive, so it is fenced:
          * refuses any path that is not strictly inside output_root
          * refuses symlinks, and never follows one when deleting
          * does nothing at all if there is no prior trial data to remove
        """
        if overwrite not in _OVERWRITE_MODES:
            raise ValueError(
                f"overwrite must be one of {_OVERWRITE_MODES}, got {overwrite!r}"
            )

        root = self._output_root.resolve()
        target = trial_dir.resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError(
                f"Refusing to prepare {target} — it is not inside output_root {root}. "
                f"Check trial_id for '..' or an absolute path."
            )

        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            return

        # Validate the path before listing it, so iterdir() can never follow a
        # symlink out of output_root.
        if target.is_symlink() or not target.is_dir():
            raise NotADirectoryError(
                f"Refusing to delete {target} — it is a symlink or not a directory."
            )

        # Pre-trial artefacts (see _PRE_TRIAL_ENTRIES) are setup, not trial data:
        # they are written before recording starts and the trial that follows is
        # the one they describe. Excluding them here means a directory holding
        # only a sensor check is treated as fresh — no prompt, nothing deleted.
        stale = [p for p in target.iterdir() if p.name not in _PRE_TRIAL_ENTRIES]
        if not stale:
            return

        n_files, n_bytes = _dir_summary(stale)
        if n_files == 0:
            return   # only empty directories left behind — nothing worth confirming

        if overwrite == "never":
            raise FileExistsError(
                f"{target} already holds {n_files} files. Pass overwrite='ask' or "
                f"'always' to replace it, or use a different trial_id."
            )

        kept = sorted(p.name for p in target.iterdir() if p.name in _PRE_TRIAL_ENTRIES)
        if overwrite == "ask":
            print("=" * 68)
            print("OVERWRITE TRIAL DATA")
            print(f"  {target}")
            print(f"  {n_files} files, {_human_bytes(n_bytes)} "
                  f"— will be PERMANENTLY DELETED")
            if kept:
                print(f"  keeping pre-trial setup: {', '.join(kept)}")
            print("=" * 68)
            try:
                reply = input("Type 'y' to delete and re-record, anything else aborts: ")
            except (EOFError, OSError) as exc:
                # No usable stdin (headless script, non-interactive kernel). Do not
                # guess at consent for a destructive operation.
                raise RuntimeError(
                    f"Cannot prompt for confirmation and {target} is not empty. "
                    f"Pass overwrite='always' or 'never' explicitly."
                ) from exc
            if reply.strip().lower() not in ("y", "yes"):
                raise RuntimeError(
                    f"Aborted — {target} left untouched. Nothing was recorded."
                )

        # Delete the stale entries individually rather than rmtree(target), so the
        # pre-trial artefacts above are preserved.
        for p in stale:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        self._node.get_logger().info(
            f"[RecordingManager] Cleared {n_files} files "
            f"({_human_bytes(n_bytes)}) from {target}"
            + (f", kept {', '.join(kept)}" if kept else "")
        )

    def start_trial_recording(
        self,
        trial_id: str,
        metadata: dict,
        overwrite: str = "ask",
        section: Optional[str] = None,
    ) -> None:
        """
        Begin a recording, replacing any existing data at its destination.

        Args:
            trial_id: names the trial directory under output_root. Always the
                trial itself (".../trial_0000"), never a section of it.
            metadata: written to metadata.yaml at stop_trial_recording().
            overwrite: what to do when the destination already has files.
                'ask'    — prompt for confirmation, abort on anything but y/yes
                           (default; raises if no stdin is available rather than
                           assuming consent)
                'always' — delete without prompting. For scripted/headless runs.
                'never'  — raise instead of deleting.
            section: optional name of a section of the trial. When given, this
                recording is written to <trial_dir>/<section>/ and only that
                subdirectory is cleared — sibling sections and anything at the
                trial root (the sensor check) are left alone.
        """
        trial_dir = self._output_root / trial_id
        recording_dir = trial_dir / section if section else trial_dir
        # Clear BEFORE any recorder starts, so aborting costs nothing.
        self._prepare_trial_dir(recording_dir, overwrite)

        self._trial_id = trial_id
        self._section = section
        self._metadata = metadata
        self._events = []
        self._trial_dir = trial_dir
        self._recording_dir = recording_dir
        self._joint_recorder.start_recording()
        if self._tactile_recorder is not None:
            self._tactile_recorder.start_recording()
        if self._wrist_ft_recorder is not None:
            self._wrist_ft_recorder.start_recording()
        if self._audio_recorder is not None:
            self._audio_recorder.start_recording()
        if self._camera_recorder is not None:
            self._camera_recorder.start_recording()
        self._node.get_logger().info(
            f"[RecordingManager] Trial started: {trial_id}"
            + (f" [{section}]" if section else "")
            + f" → {self._recording_dir}"
        )

    def mark_event(self, event_name: str) -> None:
        ts = time.time()
        self._events.append({"timestamp": ts, "event": event_name})
        self._node.get_logger().info(f"[RecordingManager] Event: {event_name}")

    def stop_trial_recording(self) -> None:
        if self._recording_dir is None:
            self._node.get_logger().error(
                "[RecordingManager] stop called before start — nothing to save"
            )
            return

        # Everything here is per-recording, so it goes to recording_dir — the
        # section subdirectory when sectioned, the trial directory otherwise.
        self._joint_recorder.stop_recording()
        self._joint_recorder.save(self._recording_dir)
        if self._tactile_recorder is not None:
            self._tactile_recorder.stop_recording()
            self._tactile_recorder.save(self._recording_dir)
        if self._wrist_ft_recorder is not None:
            self._wrist_ft_recorder.stop_recording()
            self._wrist_ft_recorder.save(self._recording_dir)
        if self._audio_recorder is not None:
            self._audio_recorder.stop_recording()
            self._audio_recorder.save(self._recording_dir)
        if self._camera_recorder is not None:
            self._camera_recorder.stop_recording()
            self._camera_recorder.save(self._recording_dir)

        events_path = self._recording_dir / "events.csv"
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "event"])
            writer.writeheader()
            writer.writerows(self._events)

        meta_path = self._recording_dir / "metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(self._metadata, f, default_flow_style=False, allow_unicode=True)

        self._node.get_logger().info(
            f"[RecordingManager] Trial saved → {self._recording_dir}"
        )

    @property
    def trial_dir(self) -> Optional[Path]:
        """The trial directory — always ends with the trial number.

        Never a section of the trial, even when the last recording was one:
        this is the handle for the trial as a whole. Use recording_dir for the
        window that actually closed last.
        """
        return self._trial_dir

    @property
    def recording_dir(self) -> Optional[Path]:
        """Where the window that closed last wrote — trial_dir/<section>, or trial_dir."""
        return self._recording_dir

    @property
    def section(self) -> Optional[str]:
        """Name of the section last recorded, or None if the trial is unsectioned."""
        return self._section

    @property
    def output_root(self) -> Path:
        return self._output_root

    def validate_last_trial(
        self,
        run_flow_analysis: bool = True,
        marker_binary_threshold: Optional[float] = None,
        marker_reference_frame_bgr: Optional[np.ndarray] = None,
        marker_reference_centers: Optional[np.ndarray] = None,
        marker_grid_shape=None,
        recording_dir: Optional[Path] = None,
        section: Optional[str] = None,
    ) -> dict:
        """Validate a finished recording and write its validation_report.yaml.

        Defaults to the window that closed most recently. Pass recording_dir to
        validate a specific one instead — a behavior that records several sections
        per trial leaves only the last of them in self._recording_dir, so the
        earlier ones have to be named explicitly. The recorders' validate()
        methods all read from disk, so any finished directory works, including
        one from an earlier trial.

        Args:
            recording_dir: the finished recording to validate. Defaults to the
                window that closed last.
            section: which section of its trial that recording is, used to label
                the report. Stated rather than inferred from the path — a
                directory name alone cannot tell a section apart from a trial.
                Defaults to the manager's own section when validating the window
                that closed last, None (unsectioned) otherwise.
        """
        if recording_dir is None:
            recording_dir = self._recording_dir
            if section is None:
                section = self._section
        else:
            recording_dir = Path(recording_dir)

        if recording_dir is None:
            return {"status": "fail", "issues": ["no trial recorded"]}
        if not recording_dir.is_dir():
            return {"status": "fail", "issues": [f"{recording_dir} does not exist"]}

        joint_report = self._joint_recorder.validate(recording_dir)
        tactile_report = (
            self._tactile_recorder.validate(recording_dir)
            if self._tactile_recorder is not None
            else None
        )
        flow_report = (
            self._run_flow_analysis(
                recording_dir, marker_binary_threshold,
                marker_reference_frame_bgr, marker_reference_centers,
                marker_grid_shape,
            )
            # Marker flow needs a marked gel; a DIGIT (has_markers=False) has none.
            if run_flow_analysis and self._tactile_recorder is not None
            and getattr(self._tactile_recorder, "has_markers", True)
            else None
        )
        wrist_ft_report = (
            self._wrist_ft_recorder.validate(recording_dir)
            if self._wrist_ft_recorder is not None
            else None
        )
        audio_report = (
            self._audio_recorder.validate(recording_dir)
            if self._audio_recorder is not None
            else None
        )
        camera_report = (
            self._camera_recorder.validate(recording_dir)
            if self._camera_recorder is not None
            else None
        )

        expected_files = ["metadata.yaml", "events.csv"]
        missing = [f for f in expected_files if not (recording_dir / f).exists()]

        sub_statuses = [joint_report.get("status", "fail")]
        if tactile_report is not None:
            sub_statuses.append(tactile_report.get("status", "fail"))
        if flow_report is not None:
            sub_statuses.append(flow_report.get("status", "fail"))
        if wrist_ft_report is not None:
            sub_statuses.append(wrist_ft_report.get("status", "fail"))
        if audio_report is not None:
            sub_statuses.append(audio_report.get("status", "fail"))
        if camera_report is not None:
            sub_statuses.append(camera_report.get("status", "fail"))

        if missing:
            overall = "fail"
        elif "fail" in sub_statuses:
            overall = "fail"
        elif "warn" in sub_statuses:
            overall = "warn"
        else:
            overall = "ok"

        # A section's report names both the section it covers and the trial it
        # belongs to, so a report read on its own is never ambiguous about where
        # it came from.
        trial_dir = recording_dir.parent if section else recording_dir
        try:
            trial_id = str(trial_dir.resolve().relative_to(self._output_root.resolve()))
        except ValueError:
            # Outside output_root (an archived trial passed in by path) — the
            # manager's own trial_id would be a lie, so report the path only.
            trial_id = None

        report = {
            "trial_id": trial_id,
            "section": section,
            "trial_dir": str(trial_dir),
            "recording_dir": str(recording_dir),
            "status": overall,
            "joint_states": joint_report,
        }
        if tactile_report is not None:
            report["tactile_images"] = tactile_report
        if flow_report is not None:
            report["tactile_flow"] = flow_report
        if wrist_ft_report is not None:
            report["wrist_force_torque"] = wrist_ft_report
        if audio_report is not None:
            report["audio"] = audio_report
        if camera_report is not None:
            report["camera"] = camera_report
        if missing:
            report["missing_files"] = missing

        report_path = recording_dir / "validation_report.yaml"
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)

        self._node.get_logger().info(
            f"[RecordingManager] Validation: {overall} — {recording_dir}"
        )
        return report

    def _run_flow_analysis(
        self,
        trial_dir: Path,
        marker_binary_threshold: Optional[float] = None,
        reference_frame_bgr: Optional[np.ndarray] = None,
        reference_centers: Optional[np.ndarray] = None,
        grid_shape=None,
    ) -> dict:
        """Run marker-flow anomaly detection and return a concise report dict.

        Pass the reference capture whenever one exists: by frame 0 the tool is
        already pressed into the gel, and seeding without it silently drops the
        markers under the tool — the very region a poke deforms most.
        """
        image_dir = trial_dir / "tactile"
        if not image_dir.exists():
            return {"status": "fail", "issues": ["tactile/ not found"]}

        try:
            from ur5_station.sensors.tactile_flow import (
                compute_marker_flow,
                check_flow_anomalies,
            )
            flow_dir = trial_dir / "tactile" / "flow"
            result = compute_marker_flow(
                image_dir, output_dir=flow_dir,
                marker_binary_threshold=marker_binary_threshold,
                reference_frame_bgr=reference_frame_bgr,
                reference_centers=reference_centers,
                grid_shape=grid_shape,
            )
            anomaly = check_flow_anomalies(
                result,
                output_dir=flow_dir / "anomalies",
                verbose=False,
            )
        except Exception as exc:
            return {"status": "fail", "issues": [f"flow analysis error: {exc}"]}

        alarm_frames = anomaly["alarm_frames"]
        n_alarm = len(alarm_frames)
        max_bad = int(anomaly["n_anomalous"].max()) if n_alarm else 0
        max_dev = float(anomaly["deviation"].max()) if n_alarm else 0.0

        issues = []
        if n_alarm:
            issues.append(
                f"{n_alarm} frame(s) with anomalous markers "
                f"(worst: {max_bad} markers, {max_dev:.1f} px deviation)"
            )

        return {
            "status": "warn" if n_alarm else "ok",
            "n_markers": result["n_markers"],
            "n_frames": len(result["displacements"]),
            "n_alarm_frames": n_alarm,
            "alarm_frames": alarm_frames,
            "max_anomalous_markers": max_bad,
            "max_deviation_px": round(max_dev, 1),
            "issues": issues,
        }

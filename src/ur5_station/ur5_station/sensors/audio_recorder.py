from __future__ import annotations

import csv
import threading
import time
import wave
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

from audio_common_msgs.msg import AudioData

if TYPE_CHECKING:
    from rclpy.node import Node

# WAV params — must match the audio publisher configuration.
_CHANNELS = 1
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # S16_LE = 16-bit = 2 bytes per sample

_FOLDER = "audio"

_MIN_DURATION_S = 0.5
_MIN_FRAMES = int(_SAMPLE_RATE * _MIN_DURATION_S)


class AudioRecorder:
    """
    Records audio from audio_common_msgs/msg/AudioData ROS2 topic.

    Subscribes at __init__ so the subscription is established before any trial
    starts — avoids DDS setup latency dropping the first audio chunk.
    Buffers raw PCM chunks (msg.data) during recording; writes WAV on save().

    WAV format: mono, 16000 Hz, S16_LE — matches the ROS1 audio_recorder_node.

    Alignment
    ---------
    Sample index -> time is exact WITHIN the wav (contiguous 16 kHz), but that
    says nothing about where the wav sits on the trial clock, and a chunk dropped
    by DDS silently shortens the stream — shifting every later sample earlier
    with no trace in the wav itself. Start time plus sample rate cannot detect
    that; only per-chunk arrival times can.

    save() therefore also writes audio_metadata.csv, one row per received chunk:

        chunk_idx, t_wall, sample_offset, n_samples, wav_time_s

      t_wall        — time.time() on arrival. Same clock as tactile_metadata.csv,
                      video_metadata.csv and events.csv; this is the join key.
      sample_offset — cumulative samples before this chunk, so sample s maps back
                      to the chunk that carried it.
      wav_time_s    — sample_offset / rate, the chunk's position inside the wav.

    Comparing consecutive t_wall against n_samples/rate is what exposes a drop:
    save() logs any gap, and validate() reports the accumulated discrepancy.

    Output folder: <trial_dir>/audio/
        audio.wav            — full recording
        audio_metadata.csv   — per-chunk timing index
        audio_waveform.jpg   — amplitude vs time plot

    Usage:
        recorder = AudioRecorder(node, topic="/audio")
        manager = RecordingManager(node, output_root, audio_recorder=recorder)
    """

    def __init__(self, node: "Node", topic: str = "/audio") -> None:
        self._node = node
        self._topic = topic
        self._lock = threading.Lock()
        self._recording = False
        # (t_wall, pcm_bytes) — the arrival time is what makes the audio
        # alignable against the other streams; see the class docstring.
        self._chunks: List[Tuple[float, bytes]] = []
        # QoS depth 200: ~800 ms of buffer at 250 Hz (128-byte / 4 ms chunks).
        # Without this, slow scheduler ticks can silently drop messages.
        self._sub = self._node.create_subscription(
            AudioData, self._topic, self._callback, 200
        )

    def health_check(self) -> bool:
        log = self._node.get_logger()
        log.info(f"[AudioRecorder] Checking {self._topic} ...")
        received = threading.Event()

        def _cb(_msg):
            received.set()

        tmp_sub = self._node.create_subscription(AudioData, self._topic, _cb, 10)
        ok = received.wait(timeout=3.0)
        self._node.destroy_subscription(tmp_sub)

        if ok:
            log.info("[AudioRecorder] Health check OK")
        else:
            log.warn(f"[AudioRecorder] No message on {self._topic} within 3 s")
        return ok

    def start_recording(self) -> None:
        with self._lock:
            self._chunks = []
            self._recording = True
        self._node.get_logger().info("[AudioRecorder] Recording started")

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
        with self._lock:
            n_bytes = sum(len(c) for _, c in self._chunks)
        duration = n_bytes / (_SAMPLE_RATE * _SAMPLE_WIDTH * _CHANNELS)
        self._node.get_logger().info(
            f"[AudioRecorder] Stopped ({n_bytes} bytes, ~{duration:.2f}s)"
        )

    def _callback(self, msg: AudioData) -> None:
        t_wall = time.time()
        with self._lock:
            if self._recording:
                self._chunks.append((t_wall, bytes(msg.data)))

    def save(self, output_dir: Path) -> Path:
        folder = Path(output_dir) / _FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / "audio.wav"

        with self._lock:
            chunks = list(self._chunks)

        pcm = [c for _, c in chunks]
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(b"".join(pcm))

        n_bytes = sum(len(c) for c in pcm)
        self._node.get_logger().info(
            f"[AudioRecorder] Saved {n_bytes} bytes → {out}"
        )
        self._save_index(folder, chunks)
        self._save_plot(folder, pcm)
        return out

    def _save_index(self, folder: Path, chunks: list) -> None:
        """Write the per-chunk timing index and warn about apparent drops."""
        meta = folder / "audio_metadata.csv"
        samples_per_byte = 1.0 / (_SAMPLE_WIDTH * _CHANNELS)

        sample_offset = 0
        rows = []
        for idx, (t_wall, data) in enumerate(chunks):
            n_samples = int(len(data) * samples_per_byte)
            rows.append((idx, t_wall, sample_offset, n_samples,
                         sample_offset / _SAMPLE_RATE))
            sample_offset += n_samples

        with open(meta, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                ["chunk_idx", "t_wall", "sample_offset", "n_samples", "wav_time_s"]
            )
            for idx, t_wall, off, n, wt in rows:
                wr.writerow([idx, f"{t_wall:.6f}", off, n, f"{wt:.6f}"])

        # Wall-clock elapsed vs audio actually delivered. A positive drift means
        # more time passed than there is sound to fill it — i.e. dropped chunks.
        note = ""
        if len(rows) >= 2:
            wall_span = rows[-1][1] - rows[0][1]
            audio_span = rows[-1][2] / _SAMPLE_RATE
            drift = wall_span - audio_span
            note = f"  (wall {wall_span:.2f}s vs audio {audio_span:.2f}s, drift {drift:+.3f}s)"
        self._node.get_logger().info(f"[AudioRecorder] Timing index → {meta}{note}")

    def _save_plot(self, folder: Path, chunks: list) -> None:
        if not chunks:
            return
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            raw = b"".join(chunks)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            t = np.arange(len(samples)) / _SAMPLE_RATE

            # Downsample for plotting — 16000 pts/s is too dense to render usefully.
            max_pts = 10_000
            if len(samples) > max_pts:
                step = max(1, len(samples) // max_pts)
                samples = samples[::step]
                t = t[::step]

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(t, samples, linewidth=0.4, color="steelblue")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Audio Waveform")
            # ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(folder / "audio_waveform.jpg"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            self._node.get_logger().warn(
                f"[AudioRecorder] Waveform plot skipped: {exc}"
            )

    def validate(self, output_dir: Path) -> dict:
        out = Path(output_dir) / _FOLDER / "audio.wav"
        if not out.exists():
            return {"status": "fail", "issues": ["audio/audio.wav missing"]}

        report: dict = {"status": "ok", "issues": []}

        try:
            with wave.open(str(out)) as wf:
                n_frames = wf.getnframes()
                rate = wf.getframerate()
                duration = n_frames / rate
                report["duration_sec"] = round(duration, 3)
                report["sample_rate"] = rate
                report["channels"] = wf.getnchannels()
                report["n_frames"] = n_frames

                if duration < _MIN_DURATION_S:
                    report["issues"].append(
                        f"duration {duration:.3f}s too short (min {_MIN_DURATION_S}s)"
                    )
                if n_frames < _MIN_FRAMES:
                    report["issues"].append(
                        f"only {n_frames} frames (min {_MIN_FRAMES})"
                    )
        except Exception as exc:
            return {"status": "fail", "issues": [f"could not open audio.wav: {exc}"]}

        meta = Path(output_dir) / _FOLDER / "audio_metadata.csv"
        if not meta.exists():
            report["issues"].append(
                "audio/audio_metadata.csv missing — audio cannot be aligned"
            )
        else:
            with open(meta) as f:
                rows = list(csv.DictReader(f))
            report["index_rows"] = len(rows)
            if rows:
                t = [float(r["t_wall"]) for r in rows]
                report["t_wall_start"] = round(t[0], 6)
                report["t_wall_end"] = round(t[-1], 6)
                delivered = (
                    int(rows[-1]["sample_offset"]) + int(rows[-1]["n_samples"])
                )
                if delivered != report.get("n_frames"):
                    report["issues"].append(
                        f"index accounts for {delivered} samples but wav holds "
                        f"{report.get('n_frames')}"
                    )
                # Wall time elapsed minus audio delivered: the size of the hole
                # left by dropped chunks, and the bound on alignment error.
                drift = (t[-1] - t[0]) - int(rows[-1]["sample_offset"]) / _SAMPLE_RATE
                report["wall_vs_audio_drift_sec"] = round(drift, 4)
                if abs(drift) > 0.05:
                    report["issues"].append(
                        f"wall/audio drift {drift:+.3f}s — likely dropped chunks; "
                        f"sample-index alignment is unreliable past the first gap"
                    )

        if report["issues"]:
            report["status"] = "warn"

        return report

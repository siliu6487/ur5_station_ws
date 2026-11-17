#!/usr/bin/env python3
"""
Live OpenCV preview of the GelSight Mini feed over USB (V4L2).

Frames are resized to target_size before display so the preview matches what
TactileRecorder actually saves (raw sensor resolution is 3280x1464; default
target is 320x240) — no ROS node or topic involved.

Usage:
    ros2 run ur5_station tactile_live_view [--device PATH] [--width 320] [--height 240]

Press 'q' or Esc to quit.
"""
import argparse
import time
from typing import Optional, Tuple

import cv2

from ur5_station.sensors.tactile_recorder import find_gelsight_device


def live_view(
    device: Optional[str] = None,
    target_size: Tuple[int, int] = (320, 240),
    window_name: str = "Tactile Live View",
) -> None:
    """Open the GelSight Mini and display resized frames until 'q'/Esc is pressed."""
    if device is None:
        device = find_gelsight_device()
        if device is None:
            raise RuntimeError(
                "No V4L2 device found under /dev/v4l/by-id/ — "
                "is the GelSight Mini plugged in?"
            )
        print(f"[tactile_live_view] Auto-detected device: {device}")

    # Explicit V4L2 backend: cv2.VideoCapture(path) on this system otherwise
    # picks GStreamer, which opens "successfully" but then fails every read().
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"[tactile_live_view] Could not open device: {device}")

    print(f"[tactile_live_view] Streaming at {target_size[0]}x{target_size[1]} "
          "— press 'q' or Esc to quit")

    t_last = time.time()
    fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[tactile_live_view] Frame read failed — stopping")
                break

            frame = cv2.resize(frame, target_size)

            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(
                frame, f"{fps:4.1f} fps", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[tactile_live_view] Stopped")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default=None,
        help="V4L2 device path (e.g. /dev/v4l/by-id/usb-...). Auto-detected if omitted.",
    )
    parser.add_argument("--width", type=int, default=320, help="Preview width (default 320)")
    parser.add_argument("--height", type=int, default=240, help="Preview height (default 240)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    live_view(device=args.device, target_size=(args.width, args.height))


if __name__ == "__main__":
    main()

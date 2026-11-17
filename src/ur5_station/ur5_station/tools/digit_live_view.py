#!/usr/bin/env python3
"""
Live OpenCV preview of a DIGIT tactile sensor, using the same device lookup and
settings as DigitRecorder — no ROS node or topic involved.

Usage:
    ros2 run ur5_station digit_live_view --list     # print connected DIGIT serials
    ros2 run ur5_station digit_live_view [--serial D20503] [--led 15] [--vga]

Press 'q' or Esc to quit, 'r' to capture a reference frame and show the
difference from it (useful to see contact on an unmarked gel).
"""
import argparse
import time

import cv2

from ur5_station.sensors.digit_recorder import (
    DIGIT_QVGA, DIGIT_VGA, LED_MAX, find_digit_device, list_digit_devices, set_digit_led,
)


def live_view(serial=None, resolution=DIGIT_QVGA, fps=60, led=LED_MAX) -> None:
    device = find_digit_device(serial)
    print(f"[digit_live_view] Device: {device}")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"[digit_live_view] Could not open device: {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    set_digit_led(cap, led, led, led)

    print("[digit_live_view] 'q'/Esc quit, 'r' toggle reference-difference view")
    window = f"DIGIT {serial or ''}".strip()
    reference = None
    t_last = time.time()
    rate = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[digit_live_view] Frame read failed — stopping")
                break

            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                rate = 0.9 * rate + 0.1 * (1.0 / dt)

            shown = frame if reference is None else cv2.absdiff(frame, reference) * 4
            cv2.putText(
                shown, f"{rate:4.1f} fps", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(window, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                reference = None if reference is not None else frame.copy()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[digit_live_view] Stopped")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List connected DIGITs and exit")
    parser.add_argument("--serial", default=None,
                        help="DIGIT serial, e.g. D20503. Optional if only one is connected.")
    parser.add_argument("--led", type=int, default=LED_MAX, help="LED intensity 0-15 (default 15)")
    parser.add_argument("--vga", action="store_true", help="640x480 @ 30 fps instead of 320x240 @ 60")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list:
        digits = list_digit_devices()
        for s, p in digits.items():
            print(f"{s}  {p}")
        if not digits:
            print("No DIGIT connected.")
        return
    if args.vga:
        live_view(args.serial or None, DIGIT_VGA, 30, args.led)
    else:
        live_view(args.serial or None, DIGIT_QVGA, 60, args.led)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DigitRecorder — captures JPEG frame sequences from a DIGIT tactile sensor over
USB (V4L2), identified by its serial number (e.g. "D20503").

Same output layout and interface as TactileRecorder, which it subclasses — only
device discovery and opening differ:

    tactile/
        tactile_metadata.csv   # frame_idx,timestamp,filename
        frame_000000.jpg
        ...

Frames are saved as the sensor delivers them: 320x240 landscape. The official
digit-interface library rotates every frame to 240x320 portrait
(cv2.transpose + cv2.flip(.., 0)); apply that offline if a downstream tool
expects the library's orientation.

Does not depend on digit-interface / pyudev — the serial is part of the
/dev/v4l/by-id/ name, so lookup works the same way as find_gelsight_device().
"""
import glob
import time
from typing import Optional

import cv2
from rclpy.node import Node

from ur5_station.sensors.tactile_recorder import TactileRecorder

# Supported streams (from v4l2-ctl --list-formats-ext on D20503):
#   320x240 @ 60 / 30 fps,  640x480 @ 30 / 15 fps
DIGIT_QVGA = (320, 240)
DIGIT_VGA = (640, 480)

LED_MAX = 15

# Auto-exposure ramps from bright to steady for ~0.85 s after the stream starts
# (measured on D20503: mean grey 216 -> 111). Frames are read and discarded for
# this long on every open, or each recording starts with a fake "contact" ramp.
_SETTLE_S = 1.5


def list_digit_devices() -> dict:
    """Return {serial: /dev/v4l/by-id/ path} for every DIGIT plugged in.

    Each DIGIT exposes two nodes; video-index0 is the image stream, index1 is
    metadata and cannot be read as frames.
    """
    found = {}
    for p in sorted(glob.glob("/dev/v4l/by-id/*DIGIT*-video-index0")):
        # usb-Facebook_DIGIT_D20503-video-index0 -> D20503
        serial = p.rsplit("DIGIT_", 1)[-1].split("-video-index")[0]
        found[serial] = p
    return found


def find_digit_device(serial: Optional[str] = None) -> str:
    """Return the /dev/v4l/by-id/ path of the DIGIT with this serial. Raise if absent.

    serial=None is accepted only when exactly one DIGIT is connected, so a
    two-finger setup can never record the wrong finger silently. Like
    find_gelsight_device(), never falls back to an unrelated camera.
    """
    digits = list_digit_devices()
    if serial is not None and serial in digits:
        return digits[serial]
    if serial is None and len(digits) == 1:
        return next(iter(digits.values()))

    present = "\n  ".join(sorted(glob.glob("/dev/v4l/by-id/*"))) or "(none)"
    if serial is None and len(digits) > 1:
        problem = f"Several DIGITs connected ({', '.join(digits)}) — pass serial=."
    elif serial is None:
        problem = "No DIGIT found under /dev/v4l/by-id/."
    else:
        problem = f"No DIGIT with serial {serial!r} found under /dev/v4l/by-id/."
    raise RuntimeError(
        f"{problem}\nDevices present:\n  {present}\n"
        "Refusing to substitute another camera."
    )


def set_digit_led(cap: cv2.VideoCapture, r: int = LED_MAX, g: int = LED_MAX,
                  b: int = LED_MAX) -> None:
    """Set the DIGIT's RGB LED intensities, each 0 (off) to 15 (full).

    The firmware packs them into the UVC zoom control as 0xRGB. It applies the
    value but answers with a protocol error, so cap.set() returns False and the
    control reads back a stale 255 — neither can be used to confirm the write.
    Verified on D20503 (firmware rev 0200) by watching the image change instead.
    The LEDs keep their last value across opens, so set them every time.
    """
    if not all(0 <= v <= LED_MAX for v in (r, g, b)):
        raise ValueError(f"LED intensities must be 0-{LED_MAX}, got {(r, g, b)}")
    cap.set(cv2.CAP_PROP_ZOOM, (r << 8) | (g << 4) | b)


class DigitRecorder(TactileRecorder):
    """
    Records JPEG frame sequences from a DIGIT, found by serial number.

    Usage (drop-in for TactileRecorder, including in RecordingManager):
        tactile = DigitRecorder(node, serial="D20503")
        recorder = RecordingManager(node, session_dir, tactile_recorder=tactile)
        ...
        recorder.validate_last_trial()   # skips marker flow: has_markers is False
    """

    has_markers = False

    def __init__(
        self,
        node: Node,
        serial: Optional[str] = None,
        resolution=DIGIT_QVGA,
        fps: int = 60,
        led: int = LED_MAX,
        target_size=None,
    ):
        """
        Args:
            node:        ROS2 node used for logging.
            serial:      DIGIT serial printed on the sensor, e.g. "D20503".
                         None = the only connected DIGIT.
            resolution:  DIGIT_QVGA (60/30 fps) or DIGIT_VGA (30/15 fps).
            fps:         stream rate requested from the sensor.
            led:         LED intensity 0-15, applied to all three colours.
            target_size: (width, height) to resize to before saving. None keeps
                         the stream resolution.
        """
        super().__init__(node, device=find_digit_device(serial), target_size=target_size)
        self._serial = serial
        self._resolution = tuple(resolution)
        self._fps = fps
        self._led = led

    def _open_cap(self) -> cv2.VideoCapture:
        # Explicit V4L2: the default backend can pick GStreamer on this system,
        # which ignores the size/fps/LED controls below.
        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"[DigitRecorder] Could not open device: {self._device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        set_digit_led(cap, self._led, self._led, self._led)
        t_start = time.time()
        while time.time() - t_start < _SETTLE_S:
            cap.read()
        return cap

    def health_check(self, *args, check_markers: bool = False, **kwargs) -> bool:
        """TactileRecorder.health_check with the marker check off — DIGIT gels have none."""
        return super().health_check(*args, check_markers=check_markers, **kwargs)

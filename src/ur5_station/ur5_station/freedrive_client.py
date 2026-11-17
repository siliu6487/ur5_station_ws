#!/usr/bin/env python3
"""Hand-guide the UR5 arm from Python, for teaching poses.

Freedrive on this rig is a ros2_control controller (`freedrive_mode_controller`,
spawned inactive by ur5_bringup — real hardware only). Two things are needed to
use it:

  1. It and the arm's trajectory controller claim the same command interfaces, so
     exactly one of them can be active. Entering freedrive means deactivating
     `scaled_joint_trajectory_controller`; leaving it means switching back.
     MoveIt motions FAIL while freedrive is active — this class always switches
     back in disable() / on context-manager exit.
  2. The controller is a dead-man switch: it releases the brakes only while
     `True` keeps arriving on ~/enable_freedrive_mode, and drops out of freedrive
     after `inactive_timeout` (1 s by default) of silence. A keepalive thread
     publishes at publish_hz for as long as freedrive is enabled.

Usage (notebook):
    fd = FreedriveClient(node, cbg)
    fd.enable()
    input('Move the arm by hand to point A, then press Enter...')
    start_xyz, _ = robot.get_current_eef_pose()
    fd.disable()

or, so a raised exception can't leave the arm limp and MoveIt locked out:

    with FreedriveClient(node, cbg) as fd:
        input('Move the arm by hand to point A, then press Enter...')
        start_xyz, _ = robot.get_current_eef_pose()

The robot must be in remote control mode with the external control program
running — the same precondition as any other motion through the driver.
"""
from __future__ import annotations

import threading
import time as _time
from typing import Dict, List, Optional, TYPE_CHECKING

from builtin_interfaces.msg import Duration as RosDuration
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool

if TYPE_CHECKING:
    from rclpy.node import Node

ARM_CONTROLLER = "scaled_joint_trajectory_controller"
FREEDRIVE_CONTROLLER = "freedrive_mode_controller"
ENABLE_TOPIC = f"/{FREEDRIVE_CONTROLLER}/enable_freedrive_mode"
SWITCH_SERVICE = "/controller_manager/switch_controller"
LIST_SERVICE = "/controller_manager/list_controllers"


class FreedriveClient:
    """Enable/disable UR freedrive mode and switch the arm controller around it."""

    def __init__(
        self,
        node: "Node",
        callback_group: Optional[ReentrantCallbackGroup] = None,
        arm_controller: str = ARM_CONTROLLER,
        freedrive_controller: str = FREEDRIVE_CONTROLLER,
        publish_hz: float = 10.0,
    ) -> None:
        self._node = node
        self._cbg = callback_group or ReentrantCallbackGroup()
        self._arm_controller = arm_controller
        self._freedrive_controller = freedrive_controller
        self._publish_hz = publish_hz

        self._enable_pub = node.create_publisher(Bool, ENABLE_TOPIC, 10)
        self._switch_cli = node.create_client(
            SwitchController, SWITCH_SERVICE, callback_group=self._cbg
        )
        self._list_cli = node.create_client(
            ListControllers, LIST_SERVICE, callback_group=self._cbg
        )

        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._active = False

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while this client holds freedrive enabled."""
        return self._active

    def list_controllers(self, timeout: float = 5.0) -> Dict[str, str]:
        """Return {controller_name: state} from the controller manager."""
        res = self._call(self._list_cli, ListControllers.Request(), timeout)
        if res is None:
            return {}
        return {c.name: c.state for c in res.controller}

    # ── enable / disable ──────────────────────────────────────────────────────

    def enable(self, timeout: float = 5.0) -> bool:
        """Deactivate the arm controller, activate freedrive, start the keepalive.

        Returns True once the arm is limp and hand-guidable. The arm cannot be
        commanded through MoveIt until disable() is called.
        """
        if self._active:
            self._node.get_logger().info("[FreedriveClient] already enabled")
            return True

        states = self.list_controllers(timeout=timeout)
        if self._freedrive_controller not in states:
            self._node.get_logger().error(
                f"[FreedriveClient] '{self._freedrive_controller}' is not loaded. "
                f"It is only spawned on real hardware "
                f"(use_fake_hardware:=false) — jog the arm in RViz instead. "
                f"Loaded: {sorted(states)}"
            )
            return False

        if not self._switch(
            activate=[self._freedrive_controller],
            deactivate=[self._arm_controller],
            timeout=timeout,
        ):
            return False

        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive, daemon=True, name="freedrive_keepalive"
        )
        self._keepalive_thread.start()
        self._active = True

        self._node.get_logger().warn(
            "[FreedriveClient] FREEDRIVE ON — support the tool, the arm is limp. "
            "MoveIt motions will fail until disable()."
        )
        return True

    def disable(self, timeout: float = 5.0) -> bool:
        """Stop the keepalive, leave freedrive, reactivate the arm controller.

        Safe to call when freedrive is already off — it still restores the arm
        controller, so it doubles as a recovery call if a cell died mid-teach.
        """
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None

        # Explicit False first: the controller would time out on its own after
        # inactive_timeout, but that leaves a window where the brakes are still
        # released while the switch below is already in flight.
        self._enable_pub.publish(Bool(data=False))
        _time.sleep(0.2)

        ok = self._switch(
            activate=[self._arm_controller],
            deactivate=[self._freedrive_controller],
            timeout=timeout,
        )
        self._active = False
        if ok:
            self._node.get_logger().info(
                f"[FreedriveClient] freedrive OFF — '{self._arm_controller}' active again"
            )
        else:
            self._node.get_logger().error(
                f"[FreedriveClient] failed to reactivate '{self._arm_controller}' — "
                f"the arm will not accept MoveIt goals. Check "
                f"`ros2 control list_controllers`."
            )
        return ok

    def __enter__(self) -> "FreedriveClient":
        self.enable()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.disable()
        return False

    # ── private ───────────────────────────────────────────────────────────────

    def _keepalive(self) -> None:
        msg = Bool(data=True)
        period = 1.0 / self._publish_hz
        while not self._keepalive_stop.is_set():
            self._enable_pub.publish(msg)
            self._keepalive_stop.wait(period)

    def _switch(self, activate: List[str], deactivate: List[str], timeout: float) -> bool:
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = RosDuration(sec=int(timeout), nanosec=0)

        res = self._call(self._switch_cli, req, timeout)
        if res is None:
            return False
        if not res.ok:
            self._node.get_logger().error(
                f"[FreedriveClient] switch failed: activate={activate} deactivate={deactivate}"
            )
        return bool(res.ok)

    def _call(self, client, request, timeout: float):
        """Call a service and poll its future — never calls spin_once().

        The notebook's MultiThreadedExecutor is already spinning this node in a
        daemon thread; spinning again here would race it (same reason
        RobotMotionClient polls its plan futures).
        """
        if not client.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().error(
                f"[FreedriveClient] service {client.srv_name} unavailable after {timeout:.0f}s"
            )
            return None
        future = client.call_async(request)
        deadline = _time.time() + timeout
        while not future.done():
            if _time.time() > deadline:
                self._node.get_logger().error(
                    f"[FreedriveClient] {client.srv_name} timed out after {timeout:.0f}s"
                )
                return None
            _time.sleep(0.02)
        return future.result()

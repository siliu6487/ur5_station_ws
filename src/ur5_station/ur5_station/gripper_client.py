from __future__ import annotations

import tempfile
import threading
import time as _time
from threading import Event, Lock
from typing import Optional, TYPE_CHECKING

from builtin_interfaces.msg import Duration as RosDuration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import (
    ConfigureController, ListControllers, LoadController, SwitchController,
)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint

if TYPE_CHECKING:
    from rclpy.node import Node

_GRIPPER_JOINT = "left_robotiq_85_left_knuckle_joint"
_GRIPPER_ACTION = "/gripper_controller/follow_joint_trajectory"

# Default joint position range (radians) for the Robotiq 85 without obstruction.
# 0.0 = fully open, ~0.8 = fully closed.
# calibrate() updates these from actual /joint_states feedback so the values
# reflect the real limits with whatever tool or sensor is between the fingers.
_DEFAULT_OPEN_RAD  = 0.0
_DEFAULT_CLOSE_RAD = 0.8

# Force / speed setpoints.  robotiq_driver exports two extra command interfaces on
# the knuckle joint that it writes to the gripper's rFR / rSP registers on every
# comms cycle, scaled as value / max * 255.  Nothing in bringup claims them, so
# enable_force_control() loads a ForwardCommandController on each at runtime.
_CONTROLLER_MANAGER = "/controller_manager"
_FORCE_CONTROLLER   = "gripper_force_controller"
_SPEED_CONTROLLER   = "gripper_speed_controller"
_FORCE_INTERFACE    = "set_gripper_max_effort"
_SPEED_INTERFACE    = "set_gripper_max_velocity"
_FORWARD_CMD_TYPE   = "forward_command_controller/ForwardCommandController"

# Must match gripper_max_force / gripper_max_speed in the driver's hardware params
# (2f_85.ros2_control.xacro defaults).  Bringup leaves the multipliers at
# force 0.5 / speed 1.0, i.e. 117.5 N and 0.150 m/s until set_force/set_speed.
GRIPPER_MAX_FORCE_N    = 235.0
GRIPPER_MAX_SPEED_M_S  = 0.150
DEFAULT_FORCE_N        = 0.5 * GRIPPER_MAX_FORCE_N
DEFAULT_SPEED_M_S      = 1.0 * GRIPPER_MAX_SPEED_M_S


class GripperClient:
    """
    Controls the Robotiq 85 gripper on the left arm.

    Sends FollowJointTrajectory goals to /gripper_controller/follow_joint_trajectory
    and reads position feedback from /joint_states.

    Position convention (ratio API):
        ratio=1.0  →  fully open  (open_pos_rad)
        ratio=0.0  →  fully closed (close_pos_rad)
        ratio=0.3  →  30% open — typical firm grasp on a thin tool

    Calibrate after startup (or after changing the tool) to discover the actual
    min/max joint positions given what is currently between the fingers:
        gripper.calibrate()

    Usage:
        gripper = GripperClient(node, callback_group)
        gripper.wait_until_ready()
        gripper.calibrate()              # optional but recommended at session start

        gripper.open()                   # fully open
        gripper.open_by_ratio(0.3)       # firm grasp
        gripper.close()                  # fully closed
        gripper.open_by_joint_pos(0.4)   # exact radian command
        print(gripper.get_joint_pos())   # current position in rad

    Force-limited grasp (real gripper only):
        gripper.enable_force_control()   # once per bringup
        gripper.set_speed(0.03)          # m/s, slow approach
        gripper.grip_with_force(40.0)    # close until contact, squeeze at ~40 N
    """

    def __init__(self, node: "Node", callback_group: ReentrantCallbackGroup) -> None:
        self._node = node
        self._lock = Lock()
        self._joint_pos: Optional[float] = None
        self._joint_received = Event()

        self._js_sub = node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10,
            callback_group=callback_group,
        )
        self._action_client = ActionClient(
            node, FollowJointTrajectory, _GRIPPER_ACTION,
            callback_group=callback_group,
        )

        # Updated by calibrate(); used by open(), close(), open_by_ratio().
        self.open_pos_rad  = _DEFAULT_OPEN_RAD
        self.close_pos_rad = _DEFAULT_CLOSE_RAD

        # Force / speed control — see enable_force_control().
        self._cm_list      = node.create_client(ListControllers,     f"{_CONTROLLER_MANAGER}/list_controllers",     callback_group=callback_group)
        self._cm_load      = node.create_client(LoadController,      f"{_CONTROLLER_MANAGER}/load_controller",      callback_group=callback_group)
        self._cm_configure = node.create_client(ConfigureController, f"{_CONTROLLER_MANAGER}/configure_controller", callback_group=callback_group)
        self._cm_switch    = node.create_client(SwitchController,    f"{_CONTROLLER_MANAGER}/switch_controller",    callback_group=callback_group)
        self._cm_set_params = node.create_client(SetParameters,      f"{_CONTROLLER_MANAGER}/set_parameters",       callback_group=callback_group)
        self._force_pub = node.create_publisher(Float64MultiArray, f"/{_FORCE_CONTROLLER}/commands", 10)
        self._speed_pub = node.create_publisher(Float64MultiArray, f"/{_SPEED_CONTROLLER}/commands", 10)
        # Last values this client sent; None = never set, hardware is at the
        # bringup defaults (DEFAULT_FORCE_N / DEFAULT_SPEED_M_S).
        self.commanded_force_n:   Optional[float] = None
        self.commanded_speed_m_s: Optional[float] = None

    # ── state ─────────────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        if _GRIPPER_JOINT in msg.name:
            idx = list(msg.name).index(_GRIPPER_JOINT)
            with self._lock:
                self._joint_pos = msg.position[idx]
            self._joint_received.set()

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the first gripper joint state has been received."""
        ok = self._joint_received.wait(timeout=timeout)
        if ok:
            self._node.get_logger().info("[GripperClient] Ready")
        else:
            self._node.get_logger().error(
                f"[GripperClient] No joint state for {_GRIPPER_JOINT} within {timeout:.0f}s"
            )
        return ok

    def get_joint_pos(self) -> float:
        """Return current gripper joint position in radians (blocks at startup)."""
        if not self._joint_received.wait(timeout=5.0):
            raise RuntimeError(f"[GripperClient] No joint state received for {_GRIPPER_JOINT}")
        with self._lock:
            return self._joint_pos

    # ── calibration ───────────────────────────────────────────────────────────

    def calibrate(self) -> None:
        """
        Move to fully open then fully closed to discover the actual joint
        position range.  Necessary because the close position changes depending
        on what is between the fingers (tactile sensor, tool, nothing).

        Updates self.open_pos_rad and self.close_pos_rad.
        """
        log = self._node.get_logger()
        log.info("[GripperClient] Calibrating — moving to open ...")
        self._send_goal(_DEFAULT_OPEN_RAD)
        _time.sleep(0.3)  # let joint states settle after action completes
        self.open_pos_rad = self.get_joint_pos()

        log.info("[GripperClient] Calibrating — moving to close ...")
        self._send_goal(_DEFAULT_CLOSE_RAD)
        _time.sleep(0.3)
        self.close_pos_rad = self.get_joint_pos()

        log.info(
            f"[GripperClient] Calibrated: "
            f"open={self.open_pos_rad:.4f} rad, "
            f"close={self.close_pos_rad:.4f} rad"
        )

    # ── motion commands ───────────────────────────────────────────────────────

    def open(self, speed_rad_s: float = 0.5) -> bool:
        """Move to fully open position (open_pos_rad)."""
        self._node.get_logger().info("[GripperClient] Opening gripper")
        return self._send_goal(self.open_pos_rad, speed_rad_s=speed_rad_s)

    def close(self, speed_rad_s: float = 0.5) -> bool:
        """Move to fully closed position (close_pos_rad, calibrated)."""
        self._node.get_logger().info("[GripperClient] Closing gripper")
        return self._send_goal(self.close_pos_rad, speed_rad_s=speed_rad_s)

    def open_by_ratio(self, ratio: float, speed_rad_s: float = 0.5) -> bool:
        """
        Command a fractional position between fully closed and fully open.

        ratio=1.0  →  open_pos_rad  (fully open)
        ratio=0.0  →  close_pos_rad (fully closed)
        ratio=0.3  →  30% open — typical firm grasp on a thin tool

        The force applied to the tool is determined by how closed the gripper
        is relative to the tool thickness.  Use ratio rather than the force
        field in GripperCmd, which is unreliable.
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0.0, 1.0], got {ratio!r}")
        target = self.close_pos_rad + (self.open_pos_rad - self.close_pos_rad) * ratio
        self._node.get_logger().info(
            f"[GripperClient] open_by_ratio({ratio:.2f}) → {target:.4f} rad"
        )
        return self._send_goal(target, speed_rad_s=speed_rad_s)

    def open_by_joint_pos(self, pos_rad: float, speed_rad_s: float = 0.5) -> bool:
        """Command a specific joint position in radians (0.0 = open, ~0.8 = closed)."""
        return self._send_goal(pos_rad, speed_rad_s=speed_rad_s)

    # ── force / speed ─────────────────────────────────────────────────────────

    def enable_force_control(self, timeout: float = 5.0) -> None:
        """
        Load and activate the force and speed forward-command controllers.

        Idempotent: controllers that are already active are left alone, so this
        is safe to call at the start of every notebook.  The controllers only
        exist until bringup is restarted.

        Raises RuntimeError on failure — typically because the gripper is mock
        hardware, which does not export the force/speed interfaces.
        """
        log = self._node.get_logger()
        states = self._list_controllers(timeout)
        for name, interface in (
            (_FORCE_CONTROLLER, _FORCE_INTERFACE),
            (_SPEED_CONTROLLER, _SPEED_INTERFACE),
        ):
            state = states.get(name)
            if state == "active":
                continue
            if state is None:
                self._load_forward_controller(name, interface, timeout)
                state = "unconfigured"
            if state == "unconfigured":
                req = ConfigureController.Request(name=name)
                if not self._call(self._cm_configure, req, timeout).ok:
                    raise RuntimeError(f"[GripperClient] Could not configure {name}")
            req = SwitchController.Request()
            req.activate_controllers = [name]
            req.strictness = SwitchController.Request.STRICT
            req.activate_asap = True
            req.timeout = RosDuration(sec=int(timeout))
            if not self._call(self._cm_switch, req, timeout).ok:
                raise RuntimeError(
                    f"[GripperClient] Could not activate {name} — is the gripper "
                    f"real hardware? Mock hardware has no '{interface}' interface."
                )
            log.info(f"[GripperClient] {name} active")

    def set_force(self, force_n: float, timeout: float = 2.0) -> None:
        """
        Set the gripper's grip-force limit in newtons (0 .. GRIPPER_MAX_FORCE_N).

        This is the Robotiq rFR register: once the fingers stall on an object
        they squeeze with this force, however far past the object the position
        target lies.  It is a motor-current limit mapped linearly onto the spec
        range, not a closed-loop force — measure it if absolute newtons matter.
        The gripper ignores a bare force change; it applies the force on the
        next NEW position target.  On a held grasp that only works upward: a
        higher force plus a changed close target tightens, but the 2F-85 holds
        its squeeze mechanically, so a lower force does nothing.  To lighten,
        back off a few counts and re-close.  rFR 0 is the ~20 N minimum, not
        zero.  (Bench-tested 10-21: scripts/robotiq_held_force_test.py.)
        """
        if not 0.0 <= force_n <= GRIPPER_MAX_FORCE_N:
            raise ValueError(f"force_n must be in [0, {GRIPPER_MAX_FORCE_N}], got {force_n!r}")
        self._publish(self._force_pub, force_n, timeout)
        self.commanded_force_n = float(force_n)
        self._node.get_logger().info(
            f"[GripperClient] set_force({force_n:.1f} N) → rFR={int(force_n / GRIPPER_MAX_FORCE_N * 0xFF)}"
        )

    def set_speed(self, speed_m_s: float, timeout: float = 2.0) -> None:
        """
        Set the gripper's finger speed limit in m/s (0 .. GRIPPER_MAX_SPEED_M_S).

        This is the Robotiq rSP register.  The trajectory controller still
        interpolates the position target over speed_rad_s, so the fingers move
        at whichever of the two is slower.
        """
        if not 0.0 <= speed_m_s <= GRIPPER_MAX_SPEED_M_S:
            raise ValueError(f"speed_m_s must be in [0, {GRIPPER_MAX_SPEED_M_S}], got {speed_m_s!r}")
        self._publish(self._speed_pub, speed_m_s, timeout)
        self.commanded_speed_m_s = float(speed_m_s)
        self._node.get_logger().info(
            f"[GripperClient] set_speed({speed_m_s:.3f} m/s) → rSP={int(speed_m_s / GRIPPER_MAX_SPEED_M_S * 0xFF)}"
        )

    def grip_with_force(self, force_n: float, speed_rad_s: float = 0.5) -> bool:
        """
        Force-limited grasp: set the force, then command full physical closure.

        The position target (_DEFAULT_CLOSE_RAD) is deliberately past any tool,
        so the fingers stop on contact and hold at force_n.  The grip force is
        then the same for every tool; read get_joint_pos() afterwards for where
        the fingers actually stopped.
        """
        self.set_force(force_n)
        self._node.get_logger().info(f"[GripperClient] grip_with_force({force_n:.1f} N)")
        return self._send_goal(_DEFAULT_CLOSE_RAD, speed_rad_s=speed_rad_s)

    def _list_controllers(self, timeout: float) -> dict:
        resp = self._call(self._cm_list, ListControllers.Request(), timeout)
        return {c.name: c.state for c in resp.controller}

    def _load_forward_controller(self, name: str, interface: str, timeout: float) -> None:
        # Same mechanism as `spawner --controller-type --param-file`: set the
        # type and a params file on the controller manager, then load.
        with tempfile.NamedTemporaryFile(
            "w", prefix=f"{name}_", suffix=".yaml", delete=False
        ) as f:
            f.write(
                f"{name}:\n"
                f"  ros__parameters:\n"
                f"    joints: [{_GRIPPER_JOINT}]\n"
                f"    interface_name: {interface}\n"
            )
        req = SetParameters.Request()
        for key, value in (("type", _FORWARD_CMD_TYPE), ("params_file", f.name)):
            req.parameters.append(Parameter(
                name=f"{name}.{key}",
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value),
            ))
        results = self._call(self._cm_set_params, req, timeout).results
        if not all(r.successful for r in results):
            reasons = "; ".join(r.reason for r in results if not r.successful)
            raise RuntimeError(f"[GripperClient] Could not set params for {name}: {reasons}")
        if not self._call(self._cm_load, LoadController.Request(name=name), timeout).ok:
            raise RuntimeError(f"[GripperClient] Could not load {name}")

    def _publish(self, pub, value: float, timeout: float) -> None:
        # A message published before discovery completes is silently dropped.
        deadline = _time.monotonic() + timeout
        while pub.get_subscription_count() == 0:
            if _time.monotonic() > deadline:
                raise RuntimeError(
                    f"[GripperClient] Nobody subscribed to {pub.topic_name} — "
                    f"call enable_force_control() first"
                )
            _time.sleep(0.05)
        pub.publish(Float64MultiArray(data=[float(value)]))

    def _call(self, client, request, timeout: float):
        """Call a service and block until it responds (executor spins elsewhere)."""
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"[GripperClient] Service {client.srv_name} unavailable")
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=timeout):
            raise RuntimeError(f"[GripperClient] {client.srv_name} timed out after {timeout:.0f}s")
        return future.result()

    # ── internal ──────────────────────────────────────────────────────────────

    def _send_goal(
        self,
        target_rad: float,
        speed_rad_s: float = 0.5,
        timeout: float = 10.0,
    ) -> bool:
        """Send a FollowJointTrajectory goal and block until it completes."""
        current = self.get_joint_pos()
        stroke = abs(target_rad - current)
        duration_s = max(1.0, stroke / speed_rad_s)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [_GRIPPER_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [target_rad]
        pt.velocities = [0.0]
        secs = int(duration_s)
        pt.time_from_start = RosDuration(sec=secs, nanosec=int((duration_s - secs) * 1e9))
        goal_msg.trajectory.points = [pt]

        done = threading.Event()
        success = [True]

        def _on_goal(future):
            handle = future.result()
            if not handle.accepted:
                self._node.get_logger().error("[GripperClient] Goal rejected")
                success[0] = False
                done.set()
                return
            handle.get_result_async().add_done_callback(lambda _: done.set())

        self._action_client.send_goal_async(goal_msg).add_done_callback(_on_goal)
        if not done.wait(timeout=timeout):
            self._node.get_logger().error(
                f"[GripperClient] Goal timed out after {timeout:.0f}s"
            )
            return False
        return success[0]

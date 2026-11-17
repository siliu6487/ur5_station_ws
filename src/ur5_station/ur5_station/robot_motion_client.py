#!/usr/bin/env python3
import math
import time as _time
from threading import Event, Lock
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, LookupException, ExtrapolationException, ConnectivityException
from tf2_ros.transform_listener import TransformListener

from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from pymoveit2 import MoveIt2

if TYPE_CHECKING:
    from ur5_station.gripper_client import GripperClient

ARM_JOINT_ORDER: List[str] = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

BASE_LINK = "left_base_link"
END_EFFECTOR = "left_tool0"
MOVE_GROUP = "left_manipulator"
WORLD_FRAME = "world"  # root TF frame; parent of left_base_link via the mount transform

# Joint values in ARM_JOINT_ORDER (pan, lift, elbow, wrist1, wrist2, wrist3)
NAMED_POSES: Dict[str, List[float]] = {
    # Measured 2026-07-11 at robot station. EEF: x=0.547 y=0.239 z=1.235 m
    # pan=16.02°  lift=-145.77°  elbow=-129.34°  wrist1=-6.67°  wrist2=-135.48°  wrist3=-27.76°
    "home": [0.34226909279823303, -2.5050624052630823, -2.122422520314352, -0.34111148515810186, -2.382864538823263, -0.5200718084918421],
    # ready == home (same safe configuration)
    "ready": [0.34226909279823303, -2.5050624052630823, -2.122422520314352, -0.34111148515810186, -2.382864538823263, -0.5200718084918421],
    # Measured 2026-07-11 at robot station. EEF: x=0.542 y=0.245 z=1.166 m
    # pan=23.35°  lift=-135.33°  elbow=-131.52°  wrist1=-21.90°  wrist2=-132.99°  wrist3=-37.58°
    "poke_ready": [0.413859486579895, -2.3915308157550257, -2.1292784849749964, -0.5193236509906214, -2.3556557337390345, -0.617225472127096]
    }


def quat_to_rpy(quat_xyzw):
    """Convert quaternion [x, y, z, w] to roll, pitch, yaw in radians."""
    x, y, z, w = quat_xyzw

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rpy_to_quat(roll, pitch, yaw):
    """Convert roll, pitch, yaw in radians to quaternion [x, y, z, w]."""
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return [qx, qy, qz, qw]

class RobotMotionClient:
    """
    Thin wrapper around pymoveit2 for the left UR5 arm.

    Works identically on fake and real hardware — both use the same MoveIt
    interface. Set dry_run=True to log intended motions without executing.
    """

    def __init__(
        self,
        node: Node,
        callback_group: ReentrantCallbackGroup,
        velocity: float = 0.1,
        acceleration: float = 0.1,
        dry_run: bool = False,
        gripper: Optional["GripperClient"] = None,
    ):
        self._node = node
        self._dry_run = dry_run
        self._moveit2 = MoveIt2(
            node=node,
            joint_names=ARM_JOINT_ORDER,
            base_link_name=BASE_LINK,
            end_effector_name=END_EFFECTOR,
            group_name=MOVE_GROUP,
            callback_group=callback_group,
            use_move_group_action=True,
        )
        self._moveit2.max_velocity = velocity
        self._moveit2.max_acceleration = acceleration
        # Kept in sync with the MoveIt2 scaling factors. compute_cartesian_path
        # (moveit_msgs in Humble) has no scaling fields in its request, so the
        # service always returns a full-speed trajectory — _retime() applies the
        # scaling client-side instead. See set_speed().
        self._velocity_scale = velocity
        self._acceleration_scale = acceleration
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        # Optional gripper — pass a GripperClient instance to enable open/close.
        self._gripper: Optional["GripperClient"] = gripper

        # Publisher for previewing planned trajectories in RViz before execution.
        self._display_pub = node.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )

        # Persistent joint-state cache so get_current_joint_positions() never
        # races with executor threads that are busy with MoveIt2 callbacks.
        self._joint_positions: Dict[str, float] = {}
        self._joint_lock = Lock()
        self._joint_received = Event()
        self._joint_sub = node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10,
            callback_group=callback_group,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        with self._joint_lock:
            self._joint_positions = dict(zip(msg.name, msg.position))
        self._joint_received.set()

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Block until the first /joint_states message has been cached."""
        self._node.get_logger().info("Waiting for /joint_states ...")
        ok = self._joint_received.wait(timeout=timeout)
        if not ok:
            self._node.get_logger().error(
                f"/joint_states did not arrive within {timeout:.0f}s"
            )
        return ok

    def get_current_joint_positions(self) -> Dict[str, float]:
        """Return the most recently cached joint positions as {name: rad}.
        Blocks only at startup before the first message arrives."""
        if not self._joint_received.wait(timeout=5.0):
            raise RuntimeError("/joint_states did not arrive within 5s")
        with self._joint_lock:
            return dict(self._joint_positions)

    def _plan_and_execute(
        self,
        plan_kwargs: dict,
        plan_timeout: float = 15.0,
        exec_timeout: float = 30.0,
    ) -> bool:
        """Plan via plan_async() then execute — never calls spin_once().

        plan_kwargs are forwarded directly to plan_async() (e.g.
        joint_positions=..., or position=..., quat_xyzw=..., cartesian=True).
        Handles both joint/pose planning (MotionPlanResponse) and Cartesian path
        planning (GetCartesianPath_Response) which return different result types.
        """
        cartesian = plan_kwargs.get("cartesian", False)
        future = self._moveit2.plan_async(**plan_kwargs)
        if future is None:
            self._node.get_logger().error("plan_async returned None")
            return False
        deadline = _time.time() + plan_timeout
        while not future.done():
            if _time.time() > deadline:
                self._node.get_logger().error(f"Planning timed out after {plan_timeout:.0f}s")
                return False
            _time.sleep(0.05)

        if cartesian:
            # GetCartesianPath_Response: trajectory is in res.solution.joint_trajectory
            res = future.result()
            if not hasattr(res, "solution") or not res.solution.joint_trajectory.points:
                self._node.get_logger().error("Cartesian planning failed — empty solution")
                return False
            trajectory = self._retime(res.solution.joint_trajectory)
        else:
            trajectory = self._moveit2.get_trajectory(future)
            if trajectory is None:
                self._node.get_logger().error("Planning failed — no valid trajectory")
                return False

        self._moveit2.execute(trajectory)
        return self._wait_for_motion(timeout=exec_timeout)

    def move_to_named_pose(self, name: str) -> bool:
        if name not in NAMED_POSES:
            raise ValueError(f"Unknown named pose '{name}'. Known: {list(NAMED_POSES)}")
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_to_named_pose('{name}')")
            return True
        self._node.get_logger().info(f"Moving to '{name}' ...")
        ok = self._plan_and_execute(
            {"joint_positions": NAMED_POSES[name], "joint_names": ARM_JOINT_ORDER}
        )
        if ok:
            self._node.get_logger().info(f"Reached '{name}'")
        return ok

    def rotate_joint(self, joint_name: str, degrees: float) -> bool:
        """Rotate a single joint by `degrees` relative to its current position."""
        if joint_name not in ARM_JOINT_ORDER:
            raise ValueError(f"Unknown joint '{joint_name}'")
        if self._dry_run:
            self._node.get_logger().info(
                f"[dry_run] rotate_joint('{joint_name}', {degrees:+.1f}°)"
            )
            return True

        delta_rad = math.radians(degrees)
        current = self.get_current_joint_positions()
        target_by_name = {j: current.get(j, 0.0) for j in ARM_JOINT_ORDER}
        before = target_by_name[joint_name]
        target_by_name[joint_name] += delta_rad
        after = target_by_name[joint_name]

        self._node.get_logger().info(
            f"{joint_name}: {math.degrees(before):.2f}° → {math.degrees(after):.2f}°"
            f"  (Δ {degrees:+.1f}°)"
        )
        return self._plan_and_execute(
            {"joint_positions": [target_by_name[j] for j in ARM_JOINT_ORDER],
             "joint_names": ARM_JOINT_ORDER}
        )

    def move_to_joint_degrees(self, joints_deg: Dict[str, float]) -> bool:
        """
        Move to absolute joint angles specified in degrees.
        Only the joints listed are set; omitted joints stay at their current position.

        Example:
            robot.move_to_joint_degrees({
                "left_shoulder_pan_joint": 180.0,
                "left_wrist_3_joint": -90.0,
            })
        """
        unknown = [j for j in joints_deg if j not in ARM_JOINT_ORDER]
        if unknown:
            raise ValueError(f"Unknown joints: {unknown}")
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_to_joint_degrees({joints_deg})")
            return True

        current = self.get_current_joint_positions()
        target_by_name = {j: current.get(j, 0.0) for j in ARM_JOINT_ORDER}
        for joint, deg in joints_deg.items():
            self._node.get_logger().info(
                f"  {joint}: {math.degrees(target_by_name[joint]):.1f}° → {deg:.1f}°"
            )
            target_by_name[joint] = math.radians(deg)

        return self._plan_and_execute(
            {"joint_positions": [target_by_name[j] for j in ARM_JOINT_ORDER],
             "joint_names": ARM_JOINT_ORDER}
        )

    def move_to_joint_positions(self, joints: Dict[str, float]) -> bool:
        """Move to absolute joint positions given as {name: radians}."""
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_to_joint_positions({joints})")
            return True
        self._node.get_logger().info("Moving to specified joint positions ...")
        return self._plan_and_execute(
            {"joint_positions": [joints[j] for j in ARM_JOINT_ORDER],
             "joint_names": ARM_JOINT_ORDER}
        )

    def get_current_eef_pose(
        self, timeout: float = 5.0, reference_frame: str = WORLD_FRAME
    ) -> Tuple[List[float], List[float]]:
        """
        Returns the current EEF pose expressed in `reference_frame`.
        Returns ([x, y, z], [qx, qy, qz, qw]).
        Use reference_frame=WORLD_FRAME for gravity-aligned coordinates.
        """
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            try:
                t = self._tf_buffer.lookup_transform(reference_frame, END_EFFECTOR, Time())
                tr = t.transform.translation
                ro = t.transform.rotation
                return [tr.x, tr.y, tr.z], [ro.x, ro.y, ro.z, ro.w]
            except (LookupException, ConnectivityException, ExtrapolationException):
                _time.sleep(0.05)
        raise RuntimeError(
            f"TF {reference_frame} → {END_EFFECTOR} not available after {timeout:.0f}s"
        )

    def move_to_eef_pose(
        self,
        pose,
        quat_xyzw: List[float] = None,
        frame_id: str = WORLD_FRAME,
        cartesian: bool = True,
    ) -> bool:
        """Move the EEF to a pose in the format get_current_eef_pose() returns.

        The inverse of get_current_eef_pose(): capture a pose now, go back to it
        later.

        Three call forms:

            pose = robot.get_current_eef_pose()      # ([x,y,z], [qx,qy,qz,qw])
            robot.move_to_eef_pose(pose)             # position + orientation

            robot.move_to_eef_pose([0.43, 0.19, 1.20], [-0.698, 0.713, 0.072, 0.005])

            robot.move_to_eef_pose([0.43, 0.19, 1.20])
                -> position only; the CURRENT orientation is kept, the same way
                   move_to_pose_rpy() preserves whatever it isn't told to change

        Orientation goes through as a quaternion rather than via quat_to_rpy(),
        so there is no Euler round trip and no gimbal-lock edge case.

        frame_id:  must match the reference_frame the pose was read in
                   (both default to WORLD_FRAME). Also the frame the current
                   orientation is read in when the pose carries none.
        cartesian: True (the default) forces a straight-line tool path. Pass
                   False to plan in joint space when only the destination
                   matters and the route does not.
        """
        if quat_xyzw is None:
            # ([x,y,z], [qx,qy,qz,qw]) — a pair whose first element is itself a
            # sequence. A flat [x, y, z] is position-only and keeps the current
            # orientation, so the two cases have to be told apart by shape.
            if len(pose) == 2 and hasattr(pose[0], "__len__"):
                position, quat_xyzw = pose
            else:
                position = pose
        else:
            position = pose

        position = [float(v) for v in position]
        if len(position) != 3:
            raise ValueError(
                f"move_to_eef_pose: expected [x, y, z], the (position, quat_xyzw) tuple "
                f"from get_current_eef_pose(), or both as two arguments — got {pose!r}"
            )

        if quat_xyzw is None:
            # Read it now rather than leaving it to the planner: the goal that gets
            # logged and sent is then the one that was actually evaluated.
            _, quat_xyzw = self.get_current_eef_pose(reference_frame=frame_id)
            self._node.get_logger().info(
                "[RobotMotionClient] move_to_eef_pose: no orientation given — "
                "keeping the current one"
            )

        quat_xyzw = [float(v) for v in quat_xyzw]
        if len(quat_xyzw) != 4:
            raise ValueError(
                f"move_to_eef_pose: quat_xyzw must be [qx, qy, qz, qw], got {quat_xyzw}"
            )

        # MoveIt's orientation constraint assumes a unit quaternion. A pose that
        # was printed, rounded and pasted back rarely still is one, so renormalise
        # rather than letting the goal be silently misinterpreted.
        norm = math.sqrt(sum(v * v for v in quat_xyzw))
        if norm < 1e-6:
            raise ValueError(f"move_to_eef_pose: quat_xyzw {quat_xyzw} has zero norm")
        if abs(norm - 1.0) > 1e-3:
            self._node.get_logger().warn(
                f"[RobotMotionClient] quat_xyzw norm {norm:.6f} != 1 — renormalising"
            )
        quat_xyzw = [v / norm for v in quat_xyzw]

        if self._dry_run:
            self._node.get_logger().info(
                f"[dry_run] move_to_eef_pose(position={position}, quat_xyzw={quat_xyzw}, "
                f"frame_id={frame_id}, cartesian={cartesian})"
            )
            return True

        roll, pitch, yaw = quat_to_rpy(quat_xyzw)
        self._node.get_logger().info(
            f"Moving EEF in {frame_id}: "
            f"position=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
            f"rpy_deg=({math.degrees(roll):.1f}, {math.degrees(pitch):.1f}, "
            f"{math.degrees(yaw):.1f})"
        )
        return self._plan_and_execute({
            "position": position,
            "quat_xyzw": quat_xyzw,
            "cartesian": cartesian,
            "frame_id": frame_id,
        })

    # ── DISABLED 2026-08-26 — kept for reference, do not re-enable as-is ───────
    # Speed cannot be controlled through this function, which makes it unsafe for
    # any motion that touches the medium. It defaults to cartesian=False, and a
    # joint-space plan ignores acceleration entirely and velocity below ~0.23
    # (dev_notes/logs/07_27_log.md, Errors 1-2) — so a stroke meant to be slow
    # runs at full speed with no warning.
    #
    # Use move_to_eef_pose() instead: it defaults to cartesian=True, and Cartesian
    # plans do honour set_speed(), which _retime() applies client-side. The
    # trade-off is that it takes a FULL position — it cannot change one axis while
    # inheriting the rest, so pass the whole [x, y, z] and read
    # get_current_eef_pose() for the components you mean to keep.
    #
    # Still calling this, and therefore broken until converted:
    #   behaviors/poke_base.py:146,172   (approach and retract)
    #   notebooks/four_region_poke_trial.ipynb
    #   notebooks/setup_and_poke.ipynb
    # behaviors/slide_base.py was converted on 2026-08-26.
    # ──────────────────────────────────────────────────────────────────────────
    # def move_to_pose_rpy(
    #     self,
    #     x: float = None,
    #     y: float = None,
    #     z: float = None,
    #     roll_deg: float = None,
    #     pitch_deg: float = None,
    #     yaw_deg: float = None,
    #     frame_id: str = WORLD_FRAME,
    #     cartesian: bool = False,
    # ) -> bool:
    #     """
    #     # FIXME: varify later, the pose might not be correct
    #     Move EEF to a pose, overriding only provided position/orientation values.

    #     Any None value preserves the current EEF value in the requested reference frame.

    #     Examples:
    #         move_to_pose_rpy(z=0.35)
    #             -> keep current x/y and orientation, change only z

    #         move_to_pose_rpy(x=0.4, y=0.1, yaw_deg=30)
    #             -> change x/y/yaw, keep current z/roll/pitch

    #         move_to_pose_rpy(x=0.4, y=0.1, z=0.3)
    #             -> change position, keep current orientation
    #     """
    #     if self._dry_run:
    #         self._node.get_logger().info(
    #             f"[dry_run] move_to_pose_rpy("
    #             f"x={x}, y={y}, z={z}, "
    #             f"r={roll_deg}, p={pitch_deg}, yaw={yaw_deg}, "
    #             f"frame_id={frame_id}, cartesian={cartesian})"
    #         )
    #         return True

    #     current_pos, current_quat = self.get_current_eef_pose(reference_frame=frame_id)

    #     # Preserve current position components unless explicitly overridden.
    #     target_x = current_pos[0] if x is None else x
    #     target_y = current_pos[1] if y is None else y
    #     target_z = current_pos[2] if z is None else z

    #     # Convert current orientation to RPY.
    #     current_roll, current_pitch, current_yaw = quat_to_rpy(current_quat)

    #     # Preserve current orientation components unless explicitly overridden.
    #     target_roll = current_roll if roll_deg is None else math.radians(roll_deg)
    #     target_pitch = current_pitch if pitch_deg is None else math.radians(pitch_deg)
    #     target_yaw = current_yaw if yaw_deg is None else math.radians(yaw_deg)

    #     target_quat = rpy_to_quat(target_roll, target_pitch, target_yaw)

    #     self._node.get_logger().info(
    #         f"Moving EEF in {frame_id}: "
    #         f"position=({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), "
    #         f"rpy_deg=("
    #         f"{math.degrees(target_roll):.1f}, "
    #         f"{math.degrees(target_pitch):.1f}, "
    #         f"{math.degrees(target_yaw):.1f})"
    #     )

    #     return self._plan_and_execute({
    #         "position": [target_x, target_y, target_z],
    #         "quat_xyzw": target_quat,
    #         "cartesian": cartesian,
    #         "frame_id": frame_id,
    #     })

    def move_eef_delta_z(self, delta_z_m: float, z_floor_m: float = None) -> bool:
        """Move the end effector by delta_z_m along the world Z axis (negative = down toward floor).

        z_floor_m: if set, the target Z is clamped to this minimum (safety floor).
        """
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_eef_delta_z({delta_z_m:+.3f} m)")
            return True
        pos, quat = self.get_current_eef_pose(reference_frame=WORLD_FRAME)
        target_z = pos[2] + delta_z_m
        if z_floor_m is not None and target_z < z_floor_m:
            self._node.get_logger().warn(
                f"[safety] target Z {target_z:.4f} m would breach floor {z_floor_m:.4f} m — clamping"
            )
            target_z = z_floor_m
        target_pos = [pos[0], pos[1], target_z]
        self._node.get_logger().info(
            f"EEF world Z: {pos[2]:.4f} m → {target_pos[2]:.4f} m  (Δ {delta_z_m:+.3f} m)"
        )
        return self._plan_and_execute({
            "position": target_pos,
            "quat_xyzw": quat,
            "cartesian": True,
            "frame_id": WORLD_FRAME,
        })

    def move_eef_delta_x(self, delta_x_m: float) -> bool:
        """Move the end effector by delta_x_m along the world X axis.

        No safety floor: X is a horizontal axis, so there is no equivalent of
        z_floor_m (the table is what makes descending the dangerous direction).
        Keep the arm clear of obstacles yourself, or add a collision object.
        """
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_eef_delta_x({delta_x_m:+.3f} m)")
            return True
        pos, quat = self.get_current_eef_pose(reference_frame=WORLD_FRAME)
        target_pos = [pos[0] + delta_x_m, pos[1], pos[2]]
        self._node.get_logger().info(
            f"EEF world X: {pos[0]:.4f} m → {target_pos[0]:.4f} m  (Δ {delta_x_m:+.3f} m)"
        )
        return self._plan_and_execute({
            "position": target_pos,
            "quat_xyzw": quat,
            "cartesian": True,
            "frame_id": WORLD_FRAME,
        })

    def move_eef_delta_y(self, delta_y_m: float) -> bool:
        """Move the end effector by delta_y_m along the world Y axis.

        No safety floor — see move_eef_delta_x().
        """
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] move_eef_delta_y({delta_y_m:+.3f} m)")
            return True
        pos, quat = self.get_current_eef_pose(reference_frame=WORLD_FRAME)
        target_pos = [pos[0], pos[1] + delta_y_m, pos[2]]
        self._node.get_logger().info(
            f"EEF world Y: {pos[1]:.4f} m → {target_pos[1]:.4f} m  (Δ {delta_y_m:+.3f} m)"
        )
        return self._plan_and_execute({
            "position": target_pos,
            "quat_xyzw": quat,
            "cartesian": True,
            "frame_id": WORLD_FRAME,
        })

    def move_eef_delta(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        z_floor_m: float = None,
    ) -> bool:
        """Move the end effector by (x, y, z) metres in the world frame.

        Composed of the three single-axis moves, so the path is a SEQUENCE OF
        AXIS-ALIGNED LEGS, not a straight diagonal. For a straight line to an
        offset point, use move_to_eef_pose() instead:

            pos, _ = robot.get_current_eef_pose()
            robot.move_to_eef_pose([pos[0] + dx, pos[1] + dy, pos[2] + dz])

        The Z leg is ordered around the horizontal ones so the tool never drags
        through the medium: going UP, Z moves first (lift, then translate); going
        DOWN, Z moves last (translate in the clear, then descend). Axes with a
        zero delta are skipped.

        Returns True only if every leg succeeded; stops at the first failure
        rather than pressing on from a position it did not reach.

        z_floor_m: passed through to move_eef_delta_z() as the safety floor.
        """
        legs = []
        if z > 0.0:
            legs.append(("z", lambda: self.move_eef_delta_z(z, z_floor_m=z_floor_m)))
        if x != 0.0:
            legs.append(("x", lambda: self.move_eef_delta_x(x)))
        if y != 0.0:
            legs.append(("y", lambda: self.move_eef_delta_y(y)))
        if z < 0.0:
            legs.append(("z", lambda: self.move_eef_delta_z(z, z_floor_m=z_floor_m)))

        if not legs:
            self._node.get_logger().info("[RobotMotionClient] move_eef_delta: all deltas zero — nothing to do")
            return True

        self._node.get_logger().info(
            f"[RobotMotionClient] move_eef_delta Δ=({x:+.3f}, {y:+.3f}, {z:+.3f}) m "
            f"as legs {[axis for axis, _ in legs]}"
        )
        for axis, leg in legs:
            if not leg():
                self._node.get_logger().error(
                    f"[RobotMotionClient] move_eef_delta: {axis} leg failed — stopping "
                    f"(remaining legs not executed)"
                )
                return False
        return True

    def _retime(self, trajectory):
        """Uniformly slow a full-speed trajectory down to the current scaling factors.

        compute_cartesian_path always returns a trajectory parameterised at full
        joint speed — its request carries no scaling fields in this moveit_msgs
        version, so pymoveit2 silently drops max_velocity/max_acceleration for
        Cartesian plans. Stretching time by s divides velocity by s and
        acceleration by s², so s = max(1/v_scale, sqrt(1/a_scale)) honours both.
        """
        v = self._velocity_scale
        a = self._acceleration_scale
        if not trajectory.points or (v >= 1.0 and a >= 1.0):
            return trajectory
        s = max(1.0 / v if v > 0.0 else 1.0, math.sqrt(1.0 / a) if a > 0.0 else 1.0)
        if s <= 1.0:
            return trajectory

        for p in trajectory.points:
            t = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            t *= s
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            p.velocities = [x / s for x in p.velocities]
            p.accelerations = [x / (s * s) for x in p.accelerations]

        total = trajectory.points[-1].time_from_start
        self._node.get_logger().info(
            f"[RobotMotionClient] Cartesian trajectory retimed ×{s:.1f} "
            f"(v={v}, a={a}) → {total.sec + total.nanosec * 1e-9:.2f} s"
        )
        return trajectory

    def set_speed(
        self, velocity: Optional[float] = None, acceleration: Optional[float] = None
    ) -> None:
        """Change planning speed for subsequent motions. None leaves that value unchanged.

        velocity/acceleration are MoveIt *scaling factors* in (0, 1], not m/s —
        0.03 means 3% of the robot's maximum joint speed.

        Useful to slow down just one motion (e.g. a poke) without reconstructing
        RobotMotionClient — set it before the motion, then set it back afterward.
        """
        if velocity is not None:
            self._moveit2.max_velocity = velocity
            self._velocity_scale = velocity
        if acceleration is not None:
            self._moveit2.max_acceleration = acceleration
            self._acceleration_scale = acceleration
        self._node.get_logger().info(
            f"[RobotMotionClient] Speed set: velocity={self._moveit2.max_velocity}, "
            f"acceleration={self._moveit2.max_acceleration}"
        )

    def setup_table_collision(
        self,
        table_z_m: float,
        table_size_m: float = 3.0,
        table_thickness_m: float = 0.05,
        settle_s: float = 0.5,
    ) -> None:
        """Add a large flat collision box at table_z_m in world frame.

        Call once at session start after the bringup is ready.
        settle_s: seconds to wait for MoveIt to process the update.
        """
        center_z = table_z_m - table_thickness_m / 2.0
        self._moveit2.add_collision_box(
            id="table",
            size=(table_size_m, table_size_m, table_thickness_m),
            position=(0.0, 0.0, center_z),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id=WORLD_FRAME,
        )
        _time.sleep(settle_s)
        self._node.get_logger().info(
            f"[RobotMotionClient] Table collision box added: "
            f"surface Z={table_z_m:.4f} m, size={table_size_m}×{table_size_m}×{table_thickness_m} m"
        )

    # ── gripper ───────────────────────────────────────────────────────────────

    @property
    def has_gripper(self) -> bool:
        return self._gripper is not None

    @property
    def gripper(self) -> Optional["GripperClient"]:
        """Direct access to the GripperClient for ratio/position commands."""
        return self._gripper

    def open_gripper(self, speed_rad_s: float = 0.5) -> bool:
        """Open the gripper fully. Raises if no gripper was configured."""
        self._require_gripper()
        return self._gripper.open(speed_rad_s=speed_rad_s)

    def close_gripper(self, speed_rad_s: float = 0.5) -> bool:
        """Close the gripper fully (calibrated close position)."""
        self._require_gripper()
        return self._gripper.close(speed_rad_s=speed_rad_s)

    def _require_gripper(self) -> None:
        if self._gripper is None:
            raise RuntimeError(
                "No gripper configured — pass gripper=GripperClient(...) to RobotMotionClient"
            )

    def plan_to_named_pose(self, name: str, timeout: float = 15.0):
        """Plan to a named pose and display the trajectory in RViz without executing.

        Returns the planned JointTrajectory, or None if planning failed.
        Pass the result to execute_planned() to actually move.

        Uses plan_async() + future polling to avoid rclpy.spin_once() conflicts
        with our MultiThreadedExecutor daemon thread.

        Notebook pattern:
            traj = robot.plan_to_named_pose('poke_ready')  # ghost appears in RViz
            # inspect, then in next cell:
            robot.execute_planned(traj)
        """
        if name not in NAMED_POSES:
            raise ValueError(f"Unknown pose '{name}'. Known: {list(NAMED_POSES)}")
        if self._dry_run:
            self._node.get_logger().info(f"[dry_run] plan_to_named_pose('{name}')")
            return None

        self._node.get_logger().info(f"Planning to '{name}' ...")
        future = self._moveit2.plan_async(
            joint_positions=NAMED_POSES[name],
            joint_names=ARM_JOINT_ORDER,
        )
        if future is None:
            self._node.get_logger().error(f"Planning to '{name}' failed — plan_async returned None")
            return None

        deadline = _time.time() + timeout
        while not future.done():
            if _time.time() > deadline:
                self._node.get_logger().error(
                    f"Planning to '{name}' timed out after {timeout:.0f}s"
                )
                return None
            _time.sleep(0.05)

        trajectory = self._moveit2.get_trajectory(future)
        if trajectory is None:
            self._node.get_logger().error(f"Planning to '{name}' failed — no valid trajectory")
            return None

        msg = DisplayTrajectory()
        msg.model_id = "ur5_robotiq"
        rt = RobotTrajectory()
        rt.joint_trajectory = trajectory
        msg.trajectory = [rt]
        self._display_pub.publish(msg)

        n_pts = len(trajectory.points)
        self._node.get_logger().info(
            f"Plan ready ({n_pts} waypoints) — inspect ghost in RViz, "
            f"then call execute_planned() to move"
        )
        return trajectory

    # How long to wait for execute() to show up as requested/executing before
    # concluding the goal was never sent. pymoveit2 sets __is_motion_requested
    # inside execute() itself, so this only has to cover scheduling jitter.
    _MOTION_START_GRACE_S = 0.5

    def _motion_active(self) -> bool:
        """True while MoveIt2 has a trajectory requested or executing."""
        return bool(
            getattr(self._moveit2, '_MoveIt2__is_motion_requested', False)
            or getattr(self._moveit2, '_MoveIt2__is_executing', False)
        )

    def _wait_for_motion(self, timeout: float = 30.0) -> bool:
        """Poll MoveIt2 execution state without calling rclpy.spin_once().

        pymoveit2's wait_until_executed() calls spin_once() internally, which
        conflicts with our MultiThreadedExecutor daemon thread and causes a
        deadlock. Instead we poll the private state flags directly.

        Two failure modes this has to tell apart, because "no motion is active"
        looks identical in both:

        never started — execute() returns without setting any flag when the
            action server is not ready (pymoveit2 moveit2.py, execute()). Polling
            immediately then sees an idle client and would report success for a
            goal that was never sent. So an idle client within the first
            _MOTION_START_GRACE_S is treated as "did not start", not "finished".

        timed out — the trajectory is still running. Returning False alone would
            leave it running on the robot while the caller moves on to the next
            leg, so the goal is CANCELLED before returning. Callers must still
            check the return value; a False here means the arm is not where it
            was asked to go.
        """
        start = _time.time()
        deadline = start + timeout

        # Wait for the goal to show up rather than assuming a fixed sleep is
        # enough — under a loaded executor 0.1 s is not a guarantee.
        while not self._motion_active():
            if _time.time() - start > self._MOTION_START_GRACE_S:
                self._node.get_logger().error(
                    "Motion never started — execute() did not register a goal "
                    "(is the execute_trajectory action server up?)"
                )
                return False
            _time.sleep(0.02)

        while _time.time() < deadline:
            if not self._motion_active():
                self._node.get_logger().info("Motion complete.")
                return True
            _time.sleep(0.05)

        self._node.get_logger().error(
            f"Motion timed out after {timeout:.0f}s — cancelling the goal so it "
            f"does not keep running while the caller continues"
        )
        self.cancel_current_motion()
        # Give the cancel a moment to land, so the next leg does not plan from a
        # pose that is still changing underneath it.
        cancel_deadline = _time.time() + 2.0
        while self._motion_active() and _time.time() < cancel_deadline:
            _time.sleep(0.05)
        if self._motion_active():
            self._node.get_logger().error(
                "Goal still active after cancel — STOP THE ROBOT MANUALLY"
            )
        return False

    def execute_planned(self, trajectory) -> bool:
        """Execute a trajectory returned by plan_to_named_pose()."""
        if trajectory is None:
            self._node.get_logger().error("[RobotMotionClient] No trajectory to execute")
            return False
        self._node.get_logger().info("Executing planned trajectory ...")
        self._moveit2.execute(trajectory)
        return self._wait_for_motion()

    def cancel_current_motion(self) -> None:
        """Signal MoveIt2 to stop the current trajectory execution.

        Safe to call from any thread. If no motion is active the call is a no-op
        (pymoveit2 logs a warning internally).
        """
        self._moveit2.cancel_execution()

    def remove_collision_object(self, name: str) -> None:
        """Remove a named collision object from the MoveIt planning scene."""
        self._moveit2.remove_collision_object(name)
        self._node.get_logger().info(
            f"[RobotMotionClient] Removed collision object '{name}'"
        )

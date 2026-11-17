#!/usr/bin/env python3
"""
Rotate wrist_3 of the left UR5 by a given angle.

Reads the live joint state first, then adds the offset — safe on both
real and simulated hardware because it never hardcodes a start position.

Clockwise (when looking at the tool tip toward the base) = negative angle.

Usage:
  python3 src/ur5/ur5_bringup/scripts/test_wrist_rotate.py            # -20 deg default
  python3 src/ur5/ur5_bringup/scripts/test_wrist_rotate.py --degrees 20
  python3 src/ur5/ur5_bringup/scripts/test_wrist_rotate.py --degrees -20
"""

import argparse
import math
import sys
from threading import Event, Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2

# ── Robot constants ──────────────────────────────────────────────────────────

JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

BASE_LINK    = "left_base_link"
END_EFFECTOR = "left_tool0"
MOVE_GROUP   = "left_manipulator"

VELOCITY     = 0.05   # 5 % — conservative for first real-robot test
ACCELERATION = 0.05


# ── Joint state reader ───────────────────────────────────────────────────────

def read_current_joints(node: Node) -> dict:
    """Block until one /joint_states message arrives. Returns {name: position}."""
    received = Event()
    positions = {}

    def cb(msg: JointState):
        if not received.is_set():
            positions.update(dict(zip(msg.name, msg.position)))
            received.set()

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    node.get_logger().info("Waiting for /joint_states ...")
    received.wait(timeout=5.0)
    node.destroy_subscription(sub)

    if not positions:
        raise RuntimeError("/joint_states did not arrive within 5 s")
    return positions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--degrees",
        type=float,
        default=40.0,
        help="Wrist_3 rotation in degrees. Negative = clockwise from tool tip. (default: -20)",
    )
    args = parser.parse_args()

    delta_rad = math.radians(args.degrees)

    rclpy.init()
    node = Node("test_wrist_rotate")

    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=END_EFFECTOR,
        group_name=MOVE_GROUP,
        callback_group=callback_group,
    )
    moveit2.max_velocity     = VELOCITY
    moveit2.max_acceleration = ACCELERATION

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()

    # Read live joint state
    joint_positions = read_current_joints(node)

    # Build target: all joints stay, wrist_3 gets the offset
    try:
        target = [joint_positions[name] for name in JOINT_NAMES]
    except KeyError as e:
        node.get_logger().error(f"Joint not found in /joint_states: {e}")
        rclpy.shutdown()
        sys.exit(1)

    wrist3_before = target[JOINT_NAMES.index("left_wrist_3_joint")]
    target[JOINT_NAMES.index("left_wrist_3_joint")] += delta_rad
    wrist3_after = target[JOINT_NAMES.index("left_wrist_3_joint")]

    node.get_logger().info(
        f"wrist_3: {math.degrees(wrist3_before):.2f}° → {math.degrees(wrist3_after):.2f}°  "
        f"(Δ {args.degrees:+.1f}°)"
    )
    node.get_logger().info("Planning ...")
    moveit2.move_to_configuration(target)
    moveit2.wait_until_executed()
    node.get_logger().info("Done.")

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()

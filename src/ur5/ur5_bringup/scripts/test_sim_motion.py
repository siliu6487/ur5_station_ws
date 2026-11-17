#!/usr/bin/env python3
"""
Simulation motion smoke test for the left UR5 arm.

Moves the arm through two named poses to verify the full pipeline:
  fake bringup → move_group → controller → joint state update

Run (with bringup + moveit_rviz already running):
  python3 src/ur5/ur5_bringup/scripts/test_sim_motion.py
"""

from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import MoveIt2

# ── Robot constants ─────────────────────────────────────────────────────────

JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
]

BASE_LINK      = "left_base_link"
END_EFFECTOR   = "left_tool0"
MOVE_GROUP     = "left_manipulator"

# ── Named poses (joint order matches JOINT_NAMES above) ─────────────────────
#    pan, lift, elbow, wrist1, wrist2, wrist3

POSES = {
    # SRDF "home" — all joints near zero, safe upright
    "home": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],

    # Measured physical pose captured 2026-06-13
    "ready": [0.0135, -3.2743, 1.0228, -2.0360, -4.7906, -1.6848],
}

VELOCITY     = 0.15   # fraction of joint max (conservative for first tests)
ACCELERATION = 0.15


def move_to(moveit2: MoveIt2, node: Node, name: str) -> bool:
    positions = POSES[name]
    node.get_logger().info(f"Moving to '{name}' ...")
    moveit2.move_to_configuration(positions)
    moveit2.wait_until_executed()
    node.get_logger().info(f"Reached '{name}'")
    return True


def main():
    rclpy.init()
    node = Node("test_sim_motion")

    callback_group = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=END_EFFECTOR,
        group_name=MOVE_GROUP,
        callback_group=callback_group,
        use_move_group_action=True,
    )
    moveit2.max_velocity     = VELOCITY
    moveit2.max_acceleration = ACCELERATION

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()

    # Give move_group time to come up and publish current state
    node.create_rate(1.0).sleep()

    try:
        move_to(moveit2, node, "home")
        node.create_rate(0.5).sleep()   # pause 2 s at home
        move_to(moveit2, node, "ready")
        node.get_logger().info("Smoke test passed.")
    except Exception as exc:
        node.get_logger().error(f"Motion failed: {exc}")

    rclpy.shutdown()
    thread.join()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


CANONICAL_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class JointStatePrinter(Node):
    def __init__(self, topic, prefix):
        super().__init__("print_arm_joint_state")
        self.topic = topic
        self.prefix = prefix
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscription = self.create_subscription(
            JointState, topic, self.joint_state_callback, qos_profile
        )

    def joint_state_callback(self, msg):
        positions_by_name = dict(zip(msg.name, msg.position))
        full_names = [self.prefix + joint for joint in CANONICAL_JOINTS]
        missing = [name for name in full_names if name not in positions_by_name]

        if missing:
            self.get_logger().error(
                "Missing expected arm joints on %s: %s"
                % (self.topic, ", ".join(missing))
            )
            rclpy.shutdown()
            return

        incoming_arm_order = [name for name in msg.name if name in full_names]

        print("# Incoming arm joint order from %s:" % self.topic)
        for name in incoming_arm_order:
            print("#   %s: %.10f" % (name, positions_by_name[name]))

        if incoming_arm_order != full_names:
            print("#")
            print("# NOTE: incoming order differs from canonical UR5 order.")
            print("# Use names, not raw position[] indices.")

        print("")
        print("# Canonical arm order:")
        for name in full_names:
            print("%s: %.10f" % (name, positions_by_name[name]))

        print("")
        print("# Copy into ur5_description/config/initial_positions.yaml")
        print("# without the left_ prefix:")
        for joint, full_name in zip(CANONICAL_JOINTS, full_names):
            print("%s: %.10f" % (joint, positions_by_name[full_name]))

        rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print UR5 arm joint positions from /joint_states in canonical order."
    )
    parser.add_argument("--topic", default="/joint_states")
    parser.add_argument("--prefix", default="left_")
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = JointStatePrinter(args.topic, args.prefix)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""
Publish static collision objects to the MoveIt planning scene at startup.

Measured at the robot station (2026-07-11):
  back wall   — 13" (0.330 m) behind the table back edge (world Y = 0.75)
  right wall  — 20" (0.508 m) right of the table right edge (world X = 0.91)
  table pad   — thin layer on top of the RViz table (z=0.72) to match
                the physical table top at 30.2" (0.767 m) above ground.
                Table top corners: (0.91, 0, 0.72) and (0, 0.72, 0.72).

Run once after MoveIt is up; node exits after publishing.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, Header

WORLD_FRAME = "world"

# Wall geometry
WALL_THICKNESS_M = 0.01   # thin — barely visible
WALL_WIDTH_M = 3.0
WALL_HEIGHT_M = 2.2

# Transparent purple — walls from this planning scene script
_WALL_COLOR = ColorRGBA(r=0.6, g=0.0, b=0.8, a=0.25)
# Solid red — table padding (corrects RViz table to physical height)
_PAD_COLOR  = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

# Table edges (world frame): back y=0.75, right x=0.91
# Walls are offset from those edges: 13" (0.330 m) behind, 20" (0.508 m) to the right
BACK_WALL_Y_M  = 0.75 + 0.330   # 1.080 m
RIGHT_WALL_X_M = 0.91 + 0.508   # 1.418 m

# Table pad: corrects the RViz table surface (z=0.72) to the physical top (30.2" = 0.767 m)
# Table top corners: (0.91, 0, 0.72) and (0, 0.72, 0.72)
_TABLE_RVIZ_Z  = 0.72
_TABLE_REAL_Z  = 0.767   # 30.2 inches
_PAD_THICKNESS = _TABLE_REAL_Z - _TABLE_RVIZ_Z        # 0.047 m
_TABLE_WIDTH_X = 0.91
_TABLE_DEPTH_Y = 0.72


def _make_wall(wall_id: str, size_xyz, pos_xyz) -> CollisionObject:
    obj = CollisionObject()
    obj.header = Header()
    obj.header.frame_id = WORLD_FRAME
    obj.id = wall_id
    obj.operation = CollisionObject.ADD

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [size_xyz[0], size_xyz[1], size_xyz[2]]
    obj.primitives = [box]

    pose = Pose()
    pose.position.x = pos_xyz[0]
    pose.position.y = pos_xyz[1]
    pose.position.z = pos_xyz[2]
    pose.orientation.w = 1.0
    obj.primitive_poses = [pose]

    return obj


class PlanningScenePublisher(Node):
    def __init__(self):
        super().__init__("planning_scene_publisher")
        self._pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        # Small delay to ensure MoveIt is ready to receive the scene
        self.create_timer(1.0, self._publish_once)

    def _publish_once(self) -> None:
        back_wall = _make_wall(
            "wall_back",
            size_xyz=(WALL_WIDTH_M, WALL_THICKNESS_M, WALL_HEIGHT_M),
            pos_xyz=(0.0,
                     BACK_WALL_Y_M + WALL_THICKNESS_M / 2.0,
                     WALL_HEIGHT_M / 2.0),
        )
        right_wall = _make_wall(
            "wall_right",
            size_xyz=(WALL_THICKNESS_M, WALL_WIDTH_M, WALL_HEIGHT_M),
            pos_xyz=(RIGHT_WALL_X_M + WALL_THICKNESS_M / 2.0,
                     0.0,
                     WALL_HEIGHT_M / 2.0),
        )
        table_pad = _make_wall(
            "table_pad",
            size_xyz=(_TABLE_WIDTH_X, _TABLE_DEPTH_Y, _PAD_THICKNESS),
            pos_xyz=(_TABLE_WIDTH_X / 2.0,
                     _TABLE_DEPTH_Y / 2.0,
                     _TABLE_RVIZ_Z + _PAD_THICKNESS / 2.0),
        )

        def _color(obj_id: str) -> ObjectColor:
            c = ObjectColor()
            c.id = obj_id
            c.color = _WALL_COLOR
            return c

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [back_wall, right_wall, table_pad]
        pad_color = ObjectColor()
        pad_color.id = "table_pad"
        pad_color.color = _PAD_COLOR

        scene.object_colors = [_color("wall_back"), _color("wall_right"), pad_color]
        self._pub.publish(scene)

        self.get_logger().info(
            f"Planning scene published: wall_back (y={BACK_WALL_Y_M:.3f} m), "
            f"wall_right (x={RIGHT_WALL_X_M:.3f} m), "
            f"table_pad (surface z={_TABLE_REAL_Z:.3f} m)"
        )
        raise SystemExit  # one-shot: exit after publishing


def main():
    rclpy.init()
    node = PlanningScenePublisher()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

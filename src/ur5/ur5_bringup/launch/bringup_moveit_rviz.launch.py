"""
bringup_moveit_rviz.launch.py

Convenience launcher: robot bringup + MoveIt move_group + RViz in one command.

Equivalent to running both of these in sequence:
  ros2 launch ur5_bringup ur5.launch.py ...
  ros2 launch ur5_moveit_config moveit_rviz.launch.py  (or move_group only)

Usage — fake hardware (no real robot):
  ros2 launch ur5_bringup bringup_moveit_rviz.launch.py use_fake_hardware:=true

Usage — real robot:
  ros2 launch ur5_bringup bringup_moveit_rviz.launch.py \\
    robot_ip:=182.22.22.2 reverse_ip:=182.22.22.1 gripper_com_port:=/dev/ttyUSB0

Arguments:
  use_fake_hardware  — true → mock hardware (no real robot), false → real robot (default: false)
  launch_rviz        — true → start RViz with MoveIt plugin (default: true)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Seconds to wait for controllers to be active before starting move_group / RViz.
# Fake hardware is fast (~3 s); real hardware may need a bit more.
_MOVEIT_DELAY_S = 2.0



def declare_arguments():
    return [
        # Hardware selection
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="true → mock hardware (no real robot needed), false → connect to real robot",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="true → launch RViz with MoveIt plugin",
        ),
        # Real-robot connection args (ignored when use_fake_hardware=true)
        DeclareLaunchArgument("robot_ip", default_value="182.22.22.2"),
        DeclareLaunchArgument("reverse_ip", default_value="182.22.22.1"),
        DeclareLaunchArgument("reverse_port", default_value="50001"),
        DeclareLaunchArgument("script_sender_port", default_value="50002"),
        DeclareLaunchArgument("trajectory_port", default_value="50003"),
        DeclareLaunchArgument("script_command_port", default_value="50004"),
        DeclareLaunchArgument("gripper_com_port", default_value="/dev/robotiq_gripper"),
        # Optional overrides
        DeclareLaunchArgument("mock_sensor_commands", default_value="false"),
        DeclareLaunchArgument("headless_mode", default_value="false"),
        DeclareLaunchArgument("use_fake_gripper_hardware", default_value="false"),
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
        ),
        DeclareLaunchArgument("left_mount_xyz", default_value="0.645 0.635 1.275"),
        DeclareLaunchArgument("left_mount_rpy", default_value="-2.356 0 -1.57"),
    ]


def _hardware_args():
    """Common hardware args forwarded to both included launch files."""
    names = [
        "robot_ip",
        "reverse_ip",
        "reverse_port",
        "script_sender_port",
        "trajectory_port",
        "script_command_port",
        "use_fake_hardware",
        "mock_sensor_commands",
        "headless_mode",
        "gripper_com_port",
        "use_fake_gripper_hardware",
        "left_mount_xyz",
        "left_mount_rpy",
    ]
    return {n: LaunchConfiguration(n) for n in names}


def generate_launch_description():
    # --- Robot bringup (controllers, hardware interfaces) ---
    bringup_args = _hardware_args()

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur5_bringup"), "launch", "ur5.launch.py"]
            )
        ),
        launch_arguments=bringup_args.items(),
    )

    # --- MoveIt move_group + RViz (when launch_rviz=true) ---
    moveit_rviz = TimerAction(
        period=_MOVEIT_DELAY_S,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("ur5_moveit_config"),
                            "launch",
                            "moveit_rviz.launch.py",
                        ]
                    )
                ),
                launch_arguments=_hardware_args().items(),
            )
        ],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    # --- MoveIt move_group only (when launch_rviz=false) ---
    move_group_only = TimerAction(
        period=_MOVEIT_DELAY_S,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("ur5_moveit_config"),
                            "launch",
                            "move_group.launch.py",
                        ]
                    )
                ),
                launch_arguments=_hardware_args().items(),
            )
        ],
        condition=UnlessCondition(LaunchConfiguration("launch_rviz")),
    )

    # --- Static planning-scene walls (real robot only) ---
    # Delayed so move_group is fully up before publishing.
    # respawn=False + on_exit not wired to anything: if it fails, bringup continues.
    planning_scene = TimerAction(
        period=_MOVEIT_DELAY_S + 6.0,
        actions=[
            Node(
                package="ur5_bringup",
                executable="publish_planning_scene.py",
                name="planning_scene_publisher",
                output="screen",
                respawn=False,
            )
        ],
        condition=UnlessCondition(LaunchConfiguration("use_fake_hardware")),
    )

    return LaunchDescription(
        declare_arguments() + [bringup, moveit_rviz, move_group_only, planning_scene]
    )

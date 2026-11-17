import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def load_yaml(package_name, relative_path):
    package_path = get_package_share_directory(package_name)
    absolute_path = os.path.join(package_path, relative_path)
    with open(absolute_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_text(package_name, relative_path):
    package_path = get_package_share_directory(package_name)
    absolute_path = os.path.join(package_path, relative_path)
    with open(absolute_path, "r", encoding="utf-8") as file:
        return file.read()


def declared_arguments():
    return [
        DeclareLaunchArgument("robot_ip", default_value="0.0.0.0"),
        DeclareLaunchArgument("reverse_ip", default_value="182.22.22.1"),
        DeclareLaunchArgument("reverse_port", default_value="50001"),
        DeclareLaunchArgument("script_sender_port", default_value="50002"),
        DeclareLaunchArgument("trajectory_port", default_value="50003"),
        DeclareLaunchArgument("script_command_port", default_value="50004"),
        DeclareLaunchArgument("use_fake_hardware", default_value="true"),
        DeclareLaunchArgument("mock_sensor_commands", default_value="false"),
        DeclareLaunchArgument("headless_mode", default_value="false"),
        DeclareLaunchArgument("gripper_com_port", default_value="/dev/robotiq_gripper"),
        DeclareLaunchArgument("use_fake_gripper_hardware", default_value="false"),
        DeclareLaunchArgument("left_mount_xyz", default_value="0.645 0.635 1.275"),
        DeclareLaunchArgument("left_mount_rpy", default_value="-2.356 0 -1.57"),
    ]


def robot_description_parameters():
    effective_fake_gripper_hardware = PythonExpression(
        [
            "'true' if '",
            LaunchConfiguration("use_fake_hardware"),
            "' == 'true' or '",
            LaunchConfiguration("use_fake_gripper_hardware"),
            "' == 'true' else 'false'",
        ]
    )
    script_filename = PathJoinSubstitution(
        [FindPackageShare("ur_client_library"), "resources", "external_control.urscript"]
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur5_description"), "urdf", "ur5.urdf.xacro"]),
            " robot_ip:=", LaunchConfiguration("robot_ip"),
            " reverse_ip:=", LaunchConfiguration("reverse_ip"),
            " reverse_port:=", LaunchConfiguration("reverse_port"),
            " script_sender_port:=", LaunchConfiguration("script_sender_port"),
            " trajectory_port:=", LaunchConfiguration("trajectory_port"),
            " script_command_port:=", LaunchConfiguration("script_command_port"),
            " use_fake_hardware:=", LaunchConfiguration("use_fake_hardware"),
            " mock_sensor_commands:=", LaunchConfiguration("mock_sensor_commands"),
            " headless_mode:=", LaunchConfiguration("headless_mode"),
            " gripper_com_port:=", LaunchConfiguration("gripper_com_port"),
            " use_fake_gripper_hardware:=", effective_fake_gripper_hardware,
            " left_mount_xyz:='", LaunchConfiguration("left_mount_xyz"), "'",
            " left_mount_rpy:='", LaunchConfiguration("left_mount_rpy"), "'",
            " script_filename:=", script_filename,
            " input_recipe_filename:=", input_recipe_filename,
            " output_recipe_filename:=", output_recipe_filename,
        ]
    )

    return {
        "robot_description": ParameterValue(robot_description_content, value_type=str),
        "robot_description_semantic": load_text("ur5_moveit_config", "config/ur5.srdf"),
    }


def moveit_parameters():
    ompl_config = load_yaml("ur5_moveit_config", "config/ompl_planning.yaml")
    return [
        robot_description_parameters(),
        load_yaml("ur5_moveit_config", "config/kinematics.yaml"),
        load_yaml("ur5_moveit_config", "config/joint_limits.yaml"),
        load_yaml("ur5_moveit_config", "config/moveit_controllers.yaml"),
        load_yaml("ur5_moveit_config", "config/planning_scene_monitor.yaml"),
        {
            "planning_pipelines": ["ompl"],
            "default_planning_pipeline": "ompl",
            "ompl": ompl_config,
            "allow_trajectory_execution": True,
            "moveit_manage_controllers": False,
            "trajectory_execution.allowed_execution_duration_scaling": 1.5,
            "trajectory_execution.allowed_goal_duration_margin": 1.0,
            "publish_robot_description": True,
            "publish_robot_description_semantic": True,
        },
    ]


def generate_launch_description():
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur5_moveit_config"), "launch", "move_group.launch.py"]
            )
        ),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            "reverse_ip": LaunchConfiguration("reverse_ip"),
            "reverse_port": LaunchConfiguration("reverse_port"),
            "script_sender_port": LaunchConfiguration("script_sender_port"),
            "trajectory_port": LaunchConfiguration("trajectory_port"),
            "script_command_port": LaunchConfiguration("script_command_port"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "mock_sensor_commands": LaunchConfiguration("mock_sensor_commands"),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "gripper_com_port": LaunchConfiguration("gripper_com_port"),
            "use_fake_gripper_hardware": LaunchConfiguration("use_fake_gripper_hardware"),
            "left_mount_xyz": LaunchConfiguration("left_mount_xyz"),
            "left_mount_rpy": LaunchConfiguration("left_mount_rpy"),
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=[
            "-d",
            PathJoinSubstitution([FindPackageShare("ur5_bringup"), "config", "moveit.rviz"]),
        ],
        parameters=moveit_parameters(),
    )

    return LaunchDescription(declared_arguments() + [move_group_launch, rviz_node])

"""
ur5.launch.py  —  Full bringup for UR5 + Robotiq 2F-85 + MoveIt 2

Usage (mock hardware, no physical robot required):
  ros2 launch ur5_bringup ur5.launch.py use_fake_hardware:=true

Usage (real hardware):
  ros2 launch ur5_bringup ur5.launch.py robot_ip:=...
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
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


# ---------------------------------------------------------------------------
# Launch arguments
# ---------------------------------------------------------------------------

def declare_arguments():
    return [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="0.0.0.0",
            description="IP address of the UR robot (ignored when use_fake_hardware=true)",
        ),

        DeclareLaunchArgument(
            "reverse_ip",
            default_value="182.22.22.1",
            description="IP address of this PC for UR reverse connection",
        ),
        DeclareLaunchArgument(
            "reverse_port",
            default_value="50001",
        ),
        DeclareLaunchArgument(
            "script_sender_port",
            default_value="50002",
        ),
        DeclareLaunchArgument(
            "trajectory_port",
            default_value="50003",
        ),
        DeclareLaunchArgument(
            "script_command_port",
            default_value="50004",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Use mock hardware instead of talking to a real robot/gripper",
        ),
        DeclareLaunchArgument(
            "mock_sensor_commands",
            default_value="false",
            description="Enable mock sensor command interfaces (only with use_fake_hardware)",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Run UR driver in headless mode (no teach-pendant hand-off)",
        ),
        DeclareLaunchArgument(
            "gripper_com_port",
            default_value="/dev/robotiq_gripper",
            description="Serial port for Robotiq gripper (udev rule: /dev/robotiq_gripper)",
        ),
        DeclareLaunchArgument(
            "use_fake_gripper_hardware",
            default_value="false",
            description="Use mock hardware for the gripper instead of robotiq_driver.",
        ),
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Controller to activate initially: scaled_joint_trajectory_controller | joint_trajectory_controller",
        ),
        DeclareLaunchArgument(
            "update_rate_config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("ur5_bringup"), "config", "update_rate.yaml"]
            ),
            description="YAML file setting controller_manager update_rate (replaces the old "
                        "rtde_frequency arg). Default is 125 Hz (config/update_rate.yaml).",
        ),
        DeclareLaunchArgument(
            "left_mount_xyz",
            default_value="0.645 0.635 1.275",
            description="Fixed transform translation from frame to left UR5 base.",
        ),
        DeclareLaunchArgument(
            "left_mount_rpy",
            default_value="-2.356 0 -1.57",
            description="Fixed transform rotation from frame to left UR5 base.",
        ),
    ]


# ---------------------------------------------------------------------------
# Main setup (runs after launch arguments are resolved)
# ---------------------------------------------------------------------------

def launch_setup(context, *args, **kwargs):
    robot_ip               = LaunchConfiguration("robot_ip")
    reverse_ip = LaunchConfiguration("reverse_ip")
    reverse_port = LaunchConfiguration("reverse_port")
    script_sender_port = LaunchConfiguration("script_sender_port")
    trajectory_port = LaunchConfiguration("trajectory_port")
    script_command_port = LaunchConfiguration("script_command_port")

    use_fake_hardware      = LaunchConfiguration("use_fake_hardware")
    mock_sensor_commands   = LaunchConfiguration("mock_sensor_commands")
    headless_mode          = LaunchConfiguration("headless_mode")
    gripper_com_port       = LaunchConfiguration("gripper_com_port")

    initial_joint_controller      = LaunchConfiguration("initial_joint_controller")
    update_rate_config_file       = LaunchConfiguration("update_rate_config_file")
    use_fake_gripper_hardware     = LaunchConfiguration("use_fake_gripper_hardware")
    left_mount_xyz                = LaunchConfiguration("left_mount_xyz")
    left_mount_rpy                = LaunchConfiguration("left_mount_rpy")
    effective_fake_gripper_hardware = PythonExpression(
        [
            "'true' if '",
            use_fake_hardware,
            "' == 'true' or '",
            use_fake_gripper_hardware,
            "' == 'true' else 'false'",
        ]
    )

    # --- Script / recipe paths (only used for real hardware) ---
    script_filename = PathJoinSubstitution(
        [FindPackageShare("ur_client_library"), "resources", "external_control.urscript"]
    )
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    # --- URDF / robot description ---
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur5_description"), "urdf", "ur5.urdf.xacro"]),
            " robot_ip:=",             robot_ip,
            " reverse_ip:=", reverse_ip,
            " reverse_port:=", reverse_port,
            " script_sender_port:=", script_sender_port,
            " trajectory_port:=", trajectory_port,
            " script_command_port:=", script_command_port,
            " use_fake_hardware:=",    use_fake_hardware,
            " mock_sensor_commands:=", mock_sensor_commands,
            " headless_mode:=",        headless_mode,
            " gripper_com_port:=",              gripper_com_port,
            " use_fake_gripper_hardware:=",     effective_fake_gripper_hardware,
            " left_mount_xyz:='",               left_mount_xyz, "'",
            " left_mount_rpy:='",               left_mount_rpy, "'",
            " script_filename:=",               script_filename,
            " input_recipe_filename:=",     input_recipe_filename,
            " output_recipe_filename:=",    output_recipe_filename,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    # --- Controller manager config (ros2_control) ---
    controllers_file = PathJoinSubstitution(
        [FindPackageShare("ur5_bringup"), "config", "ros2_controllers.yaml"]
    )

    # =========================================================================
    # Nodes
    # =========================================================================

    # 1. Publish TF from URDF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # 2. ros2_control manager — loads hardware interfaces from URDF ros2_control tags
    # update_rate_config_file sets controller_manager update_rate, which also
    # drives the RTDE read/write rate (replaces the old rtde_frequency arg).
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            update_rate_config_file,
            controllers_file,
        ],
        output="screen",
        condition=IfCondition(use_fake_hardware),
    )

    ur_control_node = Node(
        package="ur_robot_driver",
        executable="ur_ros2_control_node",
        parameters=[
            robot_description,
            update_rate_config_file,
            controllers_file,
        ],
        output="screen",
        condition=UnlessCondition(use_fake_hardware),
    )

    # 3. UR robot state helper (manages power-on / e-stop state transitions)
    robot_state_helper = Node(
        package="ur_robot_driver",
        executable="robot_state_helper",
        name="ur_robot_state_helper",
        output="screen",
        parameters=[{"robot_ip": robot_ip}],
        condition=UnlessCondition(use_fake_hardware),
    )

    # 4. UR dashboard client (connects to the teach-pendant dashboard server)
    dashboard_client = Node(
        package="ur_robot_driver",
        executable="dashboard_client",
        name="dashboard_client",
        output="screen",
        parameters=[{"robot_ip": robot_ip}],
        condition=UnlessCondition(use_fake_hardware),
    )

    # -------------------------------------------------------------------------
    # Controller spawners — each waits for the previous to be active
    # -------------------------------------------------------------------------

    def spawner(name, active=True, **kwargs):
        args = [name]
        if not active:
            args += ["--inactive"]
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=args,
            output="screen",
            **kwargs,
        )

    joint_state_broadcaster_spawner = spawner("joint_state_broadcaster")

    # Activate the chosen arm controller; keep the other loaded but inactive
    active_arm_controller = initial_joint_controller.perform(context)

    scaled_jtc_spawner = spawner(
        "scaled_joint_trajectory_controller",
        active=(active_arm_controller == "scaled_joint_trajectory_controller"),
    )
    jtc_spawner = spawner(
        "joint_trajectory_controller",
        active=(active_arm_controller == "joint_trajectory_controller"),
    )

    gripper_controller_spawner = spawner("gripper_controller")

    # UR extras (skip for mock hardware — these interfaces don't exist on GenericSystem)
    io_and_status_spawner = spawner(
        "io_and_status_controller", condition=UnlessCondition(use_fake_hardware)
    )
    speed_scaling_spawner = spawner(
        "speed_scaling_state_broadcaster", condition=UnlessCondition(use_fake_hardware)
    )
    force_torque_spawner = spawner(
        "force_torque_sensor_broadcaster", condition=UnlessCondition(use_fake_hardware)
    )
    real_gripper_condition = IfCondition(
        PythonExpression(
            [
                "'",
                use_fake_hardware,
                "' == 'false' and '",
                use_fake_gripper_hardware,
                "' == 'false'",
            ]
        )
    )
    robotiq_activation_spawner = spawner(
        "robotiq_activation_controller", condition=real_gripper_condition
    )
    freedrive_spawner = spawner(
        "freedrive_mode_controller", active=False, condition=UnlessCondition(use_fake_hardware)
    )

    # Chain spawners so each waits for joint_state_broadcaster to finish first
    delay_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[
                scaled_jtc_spawner,
                jtc_spawner,
                gripper_controller_spawner,
                io_and_status_spawner,
                speed_scaling_spawner,
                force_torque_spawner,
                robotiq_activation_spawner,
                freedrive_spawner,
            ],
        )
    )

    return [
        robot_state_publisher,
        control_node,
        ur_control_node,
        robot_state_helper,
        dashboard_client,
        joint_state_broadcaster_spawner,
        delay_after_jsb,
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_launch_description():
    return LaunchDescription(declare_arguments() + [OpaqueFunction(function=launch_setup)])

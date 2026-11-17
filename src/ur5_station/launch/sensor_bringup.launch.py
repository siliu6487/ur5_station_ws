from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    view_image_arg = DeclareLaunchArgument(
        'view_image',
        default_value='false',
        description='Launch rqt_image_view to preview the RealSense stream',
    )

    audio_node = Node(
        package='audio_capture',
        executable='audio_capture_node',
        name='audio_capture',
        parameters=[{
            # Ubuntu 22 uses PipeWire; alsasrc (default) ignores the device name
            # and silently falls back to the system default mic.
            # pulsesrc + the PulseAudio source name targets the ReSpeaker directly.
            'src': 'pulsesrc',
            'device': 'alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input',
            'channels': 1,
            'sample_rate': 16000,
            # Must be 'wave' — default is 'mp3'. AudioRecorder writes msg.data
            # directly as raw PCM; if the node sends mp3 frames those bytes look
            # like noise when interpreted as S16LE samples.
            'format': 'wave',
        }],
        output='screen',
    )

    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        output='screen',
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/realsense2_camera/color/image_raw',
        description='Image topic to display in rqt_image_view',
    )

    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=[LaunchConfiguration('image_topic')],
        condition=IfCondition(LaunchConfiguration('view_image')),
        output='screen',
    )

    view_tactile_arg = DeclareLaunchArgument(
        'view_tactile',
        default_value='false',
        description='Launch a live OpenCV preview of the GelSight Mini feed',
    )

    tactile_live_view = Node(
        package='ur5_station',
        executable='tactile_live_view',
        condition=IfCondition(LaunchConfiguration('view_tactile')),
        output='screen',
    )

    view_digit_arg = DeclareLaunchArgument(
        'view_digit',
        default_value='false',
        description='Launch a live OpenCV preview of a DIGIT (see digit_serial)',
    )

    digit_serial_arg = DeclareLaunchArgument(
        'digit_serial',
        default_value='',
        description='DIGIT serial, e.g. D20503. Empty = the only DIGIT connected.',
    )

    # The preview holds the camera open: close it before recording, or the
    # recorder cannot open the device.
    digit_live_view = Node(
        package='ur5_station',
        executable='digit_live_view',
        arguments=['--serial', LaunchConfiguration('digit_serial')],
        condition=IfCondition(LaunchConfiguration('view_digit')),
        output='screen',
    )

    return LaunchDescription([
        view_image_arg,
        image_topic_arg,
        view_tactile_arg,
        view_digit_arg,
        digit_serial_arg,
        audio_node,
        realsense_node,
        image_view,
        tactile_live_view,
        digit_live_view,
    ])

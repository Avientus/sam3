from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("server_url",    default_value="http://localhost:8000"),
        DeclareLaunchArgument("text_prompt",   default_value="person"),
        DeclareLaunchArgument("include_masks", default_value="true"),
        DeclareLaunchArgument("trigger_mode",  default_value="continuous"),
        DeclareLaunchArgument("throttle_hz",   default_value="2.0"),

        Node(
            package="sam3_ros2",
            executable="sam3_node",
            name="sam3",
            output="screen",
            parameters=[{
                "server_url":    LaunchConfiguration("server_url"),
                "text_prompt":   LaunchConfiguration("text_prompt"),
                "include_masks": LaunchConfiguration("include_masks"),
                "trigger_mode":  LaunchConfiguration("trigger_mode"),
                "throttle_hz":   LaunchConfiguration("throttle_hz"),
            }],
        ),
    ])

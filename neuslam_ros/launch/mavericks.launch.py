"""NeuSLAM in MAVericks (namespace uav1, Unity simulation topics by default).

  ros2 launch neuslam_ros mavericks.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

ARGS = {
    "namespace": "uav1",
    "mode": "stereo",
    "left_image": "stereo_left/image_raw",
    "left_info": "stereo_left/camera_info",
    "right_image": "stereo_right/image_raw",
    "right_info": "stereo_right/camera_info",
    "imu": "IMU/fmu",
    "use_sim_time": "true",
    "voxels": "true",
}


def generate_launch_description():
    share = FindPackageShare("neuslam_ros")
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "neuslam.launch.py"])),
        launch_arguments={**{k: LaunchConfiguration(k) for k in ARGS},
                          "params_file": PathJoinSubstitution([share, "config", "mavericks.yaml"]),
                          "tf_topic": "tf"}.items())
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()] + [include])

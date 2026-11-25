"""NeuSLAM node and the voxel map.

  ros2 launch neuslam_ros neuslam.launch.py mode:=stereo left_image:=/cam0/image_raw right_image:=/cam1/image_raw
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ARGS = {
    "params_file": (PathJoinSubstitution([FindPackageShare("neuslam_ros"), "config", "neuslam.yaml"]),
                    "node parameters"),
    "mode": ("stereo", "stereo, stereo_inertial, rgbd, rgbd_inertial, mono, mono_inertial"),
    "namespace": ("", "namespace of both nodes"),
    "left_image": ("left/image_raw", ""),
    "left_info": ("left/camera_info", ""),
    "right_image": ("right/image_raw", ""),
    "right_info": ("right/camera_info", ""),
    "depth_image": ("depth/image", "RGB-D: depth registered to the colour camera"),
    "imu": ("imu", ""),
    "tf_topic": ("/tf", "dynamic TF topic"),
    "voxels": ("true", "run the voxel map node"),
    "use_sim_time": ("false", ""),
}


def generate_launch_description():
    c = {k: LaunchConfiguration(k) for k in ARGS}
    common = dict(namespace=c["namespace"], output="screen", emulate_tty=True)
    tf = [("/tf", c["tf_topic"])]
    slam = Node(package="neuslam_ros", executable="neuslam_node", name="neuslam",
                parameters=[c["params_file"], {"mode": c["mode"], "use_sim_time": c["use_sim_time"]}],
                remappings=tf + [("left/image", c["left_image"]), ("left/camera_info", c["left_info"]),
                                 ("right/image", c["right_image"]), ("right/camera_info", c["right_info"]),
                                 ("depth/image", c["depth_image"]), ("imu", c["imu"])],
                sigterm_timeout="300", **common)
    voxels = Node(package="neuslam_ros", executable="voxel_map", name="neuslam_voxels",
                  parameters=[c["params_file"], {"use_sim_time": c["use_sim_time"]}],
                  remappings=tf + [("depth/image", "neuslam/depth/image"),
                                   ("depth/camera_info", "neuslam/depth/camera_info"),
                                   ("~/voxels", "neuslam/map/voxels"), ("~/voxels/points", "neuslam/map/voxels/points")],
                  condition=IfCondition(c["voxels"]), **common)
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
                             + [slam, voxels])

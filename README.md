# neuslam_ros

ROS 2 (Jazzy) node for [NeuSLAM](https://github.com/aniket-gupta1/Dynamic_SLAM): stereo, RGB-D and monocular
visual SLAM, each with or without an IMU. Poses follow REP-105 (`map → odom → base_link`); depth and a local
voxel grid around the vehicle are published for obstacle avoidance.

| Package | |
|---|---|
| `neuslam_ros` | the node (`neuslam_node`), the voxel map (`voxel_map`), the offline bag runner (`bag_runner`) |
| `neuslam_msgs` | `Status`, `LoopClosure`, `SaveMap` |

NeuSLAM itself is a Python package, installed separately at the version in [`neuslam.repos`](neuslam.repos)
(its `ros2` branch).

## Interfaces

Inputs (remap to your topics; `config/neuslam.yaml` has every parameter):

| Topic | Type | Modes |
|---|---|---|
| `left/image`, `left/camera_info` | `sensor_msgs/Image`, `CameraInfo` | all |
| `right/image`, `right/camera_info` | same | `stereo`, `stereo_inertial` |
| `depth/image` | `Image` (16UC1 mm or 32FC1 m), registered to the colour camera | `rgbd`, `rgbd_inertial` |
| `imu` | `sensor_msgs/Imu` | `*_inertial` |
| TF | `base_link → <left optical>`, `<left optical> → <imu frame>` | |

Images may be raw (`input.rectified: false`: undistorted and rectified from CameraInfo `K, D, R, P`; stereo
needs the stereo calibration's `R` and `P`) or rectified. Encodings: `rgb8`, `bgr8`, `mono8`, `rgb16`, `mono16`,
`bayer_*8/16`. Without a `base_link → camera` transform, `base_link` is placed at the left camera (x forward,
z up), with a warning and `base_link_assumed` in `~/status`.

Outputs (`~` = `/neuslam`):

| Topic | Type | |
|---|---|---|
| TF `odom → base_link`, `map → odom` | | REP-105; each can be turned off (`tf.publish_*`) |
| `~/odom` | `nav_msgs/Odometry` | odometry chain (no jumps), pose and twist covariance; stamp = image stamp |
| `~/pose` | `geometry_msgs/PoseWithCovarianceStamped` | loop-closed pose in `map` |
| `~/path` | `nav_msgs/Path` | republished whole after each correction |
| `~/depth/image`, `~/depth/camera_info` | `Image` 32FC1 m (or 16UC1 mm), `CameraInfo` | each keyframe; every n frames with `depth.every_n_frames` (stereo: one extra network pass each) |
| `~/depth/points` | `sensor_msgs/PointCloud2` | optional |
| `~/map/voxels` | `nav2_msgs/VoxelGrid` | `voxel_map` node: 16 levels around the vehicle, in `odom` |
| `~/map/points` | `PointCloud2` | NeuSLAM's sparse map (static points), when subscribed |
| `~/status` | `neuslam_msgs/Status` | tracking state, keyframes, latency, drops |
| `~/loop_closure` | `neuslam_msgs/LoopClosure` | per loop closure |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1 Hz |
| `~/save_map` | `neuslam_msgs/srv/SaveMap` | NeuSLAM's outputs so far, without stopping |
| `~/reset` | `std_srvs/srv/Trigger` | restart SLAM |

With `output_dir` set, NeuSLAM's outputs (poses, map) and a per-frame `timing.csv` are written there on shutdown.

## Models

Put `neuslamnet.pth` and `ORBvoc.txt` in [`neuslam_ros/models/`](neuslam_ros/models) (not in git). The TensorRT
engines are built there on first start if missing (several minutes; they are specific to the GPU and TensorRT
version), or beforehand with `neuslam build-engine --models neuslam_ros/models --workspace-mb 16384`.
`neuslam.models_dir` or `$NEUSLAM_MODELS` point elsewhere.

## Docker

```bash
docker/build.sh                    # x86_64: ros:jazzy-ros-base + PyTorch + TensorRT 10.15
docker/run.sh                      # needs nvidia-container-toolkit
docker/run.sh ros2 launch neuslam_ros neuslam.launch.py mode:=stereo left_image:=/cam0/image_raw right_image:=/cam1/image_raw
```

Jetson Orin (JetPack 6): build a base with ROS 2 Jazzy, PyTorch and TensorRT (e.g. jetson-containers
`ros:jazzy-ros-base` + `pytorch`), then
`BASE=<that image> TORCH_FROM_PIP=0 NAV2_MSGS_FROM_SOURCE=1 docker/build.sh`, and build the engines on the device.

## Native

```bash
# NeuSLAM (see its README for PyTorch / TensorRT), with numpy < 2 for ROS Jazzy
pip install "numpy<2" "opencv-python<4.12" ./third_party/pydbow3 -e <Dynamic_SLAM>
sudo apt install ros-jazzy-nav2-msgs
colcon build --symlink-install && source install/setup.bash
ros2 launch neuslam_ros neuslam.launch.py mode:=stereo left_image:=... right_image:=...
```

## MAVericks

```bash
ros2 launch neuslam_ros mavericks.launch.py           # namespace uav1, Unity stereo_left/right topics
```

PX4 (`px4_ros2_interface`) owns `uav1/map → uav1/odom → uav1/base_link`, so `config/mavericks.yaml` publishes
no TF and uses `uav1/neuslam_*` frame names. To fuse NeuSLAM in EKF2, remap `px4_ros2_interface`'s `vio/in` to
`/uav1/neuslam/odom` (it takes a world-frame twist, which this config sets). The voxel grid is built in PX4's
`uav1/odom`.

## Bags

`ros2 bag play` into the node exercises the live path. For evaluation, `bag_runner` reads the bag directly, uses
every frame in order and writes NeuSLAM's outputs plus `base_link` trajectories in `odom` and `map` (TUM format):

```bash
ros2 run neuslam_ros bag_runner --bag <bag> --mode stereo --left /cam0/image_raw --right /cam1/image_raw \
    --out runs/<name> [--rectified] [--imu /imu] [--write-bag]
```

## Tests

```bash
cd neuslam_ros && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test
```

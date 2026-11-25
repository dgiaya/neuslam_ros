"""ROS messages from pipeline outputs."""
from __future__ import annotations

import array
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point32, PoseStamped, PoseWithCovarianceStamped, Transform, TransformStamped, Vector3
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header

from neuslam_msgs.msg import LoopClosure, Status
from neuslam_ros.core import geometry as geo
from neuslam_ros.core.images import depth_array, fill

STATES = {"initializing": 0, "tracking": 1, "lost": 2}


def to_time(ns: int) -> Time:
    s, n = divmod(int(ns), 1_000_000_000)
    return Time(sec=int(s), nanosec=int(n))


def header(ns: int, frame: str) -> Header:
    return Header(stamp=to_time(ns), frame_id=frame)


def _pose(msg_pose, T: np.ndarray) -> None:
    msg_pose.position.x, msg_pose.position.y, msg_pose.position.z = (float(v) for v in T[:3, 3])
    q = geo.quat_xyzw(T[:3, :3])
    msg_pose.orientation.x, msg_pose.orientation.y, msg_pose.orientation.z, msg_pose.orientation.w = (float(v) for v in q)


def transform(T: np.ndarray) -> Transform:
    m = Transform()
    m.translation.x, m.translation.y, m.translation.z = (float(v) for v in T[:3, 3])
    q = geo.quat_xyzw(T[:3, :3])
    m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w = (float(v) for v in q)
    return m


def tf(T: np.ndarray, ns: int, parent: str, child: str) -> TransformStamped:
    return TransformStamped(header=header(ns, parent), child_frame_id=child, transform=transform(T))


def odometry(o, odom_frame: str, base_frame: str) -> Odometry:
    m = Odometry(header=header(o.stamp_ns, odom_frame), child_frame_id=base_frame)
    _pose(m.pose.pose, o.T_odom_base)
    m.pose.covariance = o.pose_cov.ravel().tolist()
    m.twist.twist.linear = Vector3(x=float(o.linear[0]), y=float(o.linear[1]), z=float(o.linear[2]))
    m.twist.twist.angular = Vector3(x=float(o.angular[0]), y=float(o.angular[1]), z=float(o.angular[2]))
    m.twist.covariance = o.twist_cov.ravel().tolist()
    return m


def pose_with_cov(o, map_frame: str) -> PoseWithCovarianceStamped:
    m = PoseWithCovarianceStamped(header=header(o.stamp_ns, map_frame))
    _pose(m.pose.pose, o.T_map_base)
    m.pose.covariance = o.pose_cov.ravel().tolist()
    return m


def path(poses: Iterable[Tuple[int, np.ndarray]], map_frame: str, stamp_ns: int) -> Path:
    m = Path(header=header(stamp_ns, map_frame))
    for ts, T in poses:
        p = PoseStamped(header=header(ts, map_frame))
        _pose(p.pose, T)
        m.poses.append(p)
    return m


def status(o, *, mode: str, base_frame: str, rate_hz: float, latency_ms: float, dropped: int,
           base_link_assumed: bool) -> Status:
    return Status(header=header(o.stamp_ns, base_frame), state=STATES[o.state], mode=mode,
                  frame_id=int(o.frame_id), keyframe_id=int(o.keyframe_id), is_keyframe=bool(o.is_keyframe),
                  num_keyframes=int(o.num_keyframes), num_map_points=int(o.num_map_points),
                  num_loop_closures=int(o.num_loop_closures), rate_hz=float(rate_hz), latency_ms=float(latency_ms),
                  pipeline_ms=float(o.pipeline_s * 1e3), frames_dropped=int(dropped),
                  base_link_assumed=bool(base_link_assumed))


def loop_closures(o, map_frame: str, num_revised: int) -> list:
    return [LoopClosure(header=header(o.stamp_ns, map_frame), query_keyframe_id=int(max(i, j)),
                        match_keyframe_id=int(min(i, j)), map_to_odom=transform(o.T_map_odom),
                        num_poses_revised=int(num_revised)) for i, j in o.loops]


def camera_info(K: np.ndarray, ns: int, frame: str, width: int, height: int) -> CameraInfo:
    m = CameraInfo(header=header(ns, frame), width=int(width), height=int(height), distortion_model="plumb_bob")
    m.d = [0.0] * 5
    m.k = np.asarray(K, np.float64).ravel().tolist()
    m.r = np.eye(3).ravel().tolist()
    P = np.zeros((3, 4))
    P[:, :3] = K
    m.p = P.ravel().tolist()
    return m


def image(array: np.ndarray, encoding: str, ns: int, frame: str) -> Image:
    m = Image(header=header(ns, frame))
    fill(m, array, encoding)
    return m


def depth_image(depth_m: np.ndarray, encoding: str, ns: int, frame: str) -> Image:
    return image(depth_array(depth_m, encoding), encoding, ns, frame)


def cloud(points: np.ndarray, ns: int, frame: str, colors: Optional[np.ndarray] = None) -> PointCloud2:
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    if colors is not None:
        c = np.asarray(colors, np.uint8).reshape(-1, 3).astype(np.uint32)
        rgb = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
        data = np.column_stack([pts, rgb[:, None]]).astype(np.float32)
        fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
    else:
        data = pts
    return PointCloud2(header=header(ns, frame), height=1, width=int(len(data)), fields=fields, is_bigendian=False,
                       point_step=4 * data.shape[1], row_step=4 * data.shape[1] * len(data), is_dense=True,
                       data=data.tobytes())


def depth_points(depth_m: np.ndarray, K: np.ndarray, stride: int = 4) -> np.ndarray:
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    ok = np.isfinite(d)
    z = d[ok]
    return np.stack([(u[ok] - K[0, 2]) / K[0, 0] * z, (v[ok] - K[1, 2]) / K[1, 1] * z, z], 1)


def voxel_grid(packed: np.ndarray, origin: Sequence[float], res: Sequence[float], size: Sequence[int],
               ns: int, frame: str):
    from nav2_msgs.msg import VoxelGrid
    m = VoxelGrid(header=header(ns, frame))
    m.data = array.array("I", np.ascontiguousarray(packed, np.uint32).tobytes())
    m.origin = Point32(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
    m.resolutions = Vector3(x=float(res[0]), y=float(res[1]), z=float(res[2]))
    m.size_x, m.size_y, m.size_z = (int(v) for v in size)
    return m

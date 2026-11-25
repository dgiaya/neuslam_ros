"""Local voxel grid around the vehicle from depth images and TF: nav2_msgs/VoxelGrid."""
from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from message_filters import Subscriber, TimeSynchronizer
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener

from neuslam_ros.core import messages as M
from neuslam_ros.core.geometry import from_tf
from neuslam_ros.core.images import depth_to_metres
from neuslam_ros.core.voxels import LocalVoxelGrid


class VoxelMapNode(Node):
    def __init__(self) -> None:
        super().__init__("neuslam_voxels")
        P = self.declare_parameter
        self.frame = P("frame", "odom").value
        self.base_frame = P("base_frame", "base_link").value
        self.grid = LocalVoxelGrid(resolution=P("resolution", 0.2).value, z_resolution=P("z_resolution", 0.25).value,
                                   size_x=P("size_x", 200).value, size_y=P("size_y", 200).value,
                                   size_z=P("size_z", 16).value, max_range=P("max_range", 15.0).value)
        self.stride = int(P("ray_stride", 8).value)
        self.max_age = float(P("tf_wait_s", 1.0).value)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        q = QoSProfile(depth=10)
        self.pub = self.create_publisher(_voxel_type(), "~/voxels", 1)
        self.pub_pts = self.create_publisher(PointCloud2, "~/voxels/points", 1) if P("points", False).value else None
        subs = [Subscriber(self, Image, "depth/image", qos_profile=q),
                Subscriber(self, CameraInfo, "depth/camera_info", qos_profile=q)]
        self.sync = TimeSynchronizer(subs, 20)
        self.sync.registerCallback(self._on_depth)
        self.create_timer(1.0 / float(P("publish_rate_hz", 2.0).value), self._publish)
        self.create_timer(0.02, self._integrate_ready)
        self._pending: deque = deque(maxlen=4)
        self._stamp = None

    def _on_depth(self, img: Image, info: CameraInfo) -> None:
        self._pending.append((img, info))

    def _integrate_ready(self) -> None:
        """Integrate the newest depth whose camera pose is in TF; older ones are dropped."""
        now = self.get_clock().now()
        while self._pending:
            img, info = self._pending[-1]
            stamp = Time.from_msg(img.header.stamp)
            if self.tf_buffer.can_transform(self.frame, img.header.frame_id, stamp):
                self._pending.clear()
                self._integrate(img, info, stamp)
                return
            if (now - stamp).nanoseconds * 1e-9 > self.max_age:
                self.get_logger().warning(f"no TF {self.frame} <- {img.header.frame_id} at the depth stamp",
                                          throttle_duration_sec=5.0)
                self._pending.clear()
            return

    def _integrate(self, img: Image, info: CameraInfo, stamp: Time) -> None:
        t = self.tf_buffer.lookup_transform(self.frame, img.header.frame_id, stamp)
        centre = None
        try:
            centre = from_tf(self.tf_buffer.lookup_transform(self.frame, self.base_frame, Time()))[:3, 3]
        except Exception:
            pass
        K = np.asarray(info.k, np.float64).reshape(3, 3)
        self.grid.integrate(depth_to_metres(img), K, from_tf(t), stride=self.stride, centre=centre)
        self._stamp = img.header.stamp

    def _publish(self) -> None:
        if self._stamp is None or self.grid.origin_idx is None:
            return
        ns = Time.from_msg(self._stamp).nanoseconds
        g = self.grid
        self.pub.publish(M.voxel_grid(g.packed(), g.origin, g.res, g.size, ns, self.frame))
        if self.pub_pts is not None:
            self.pub_pts.publish(M.cloud(g.marked_points(), ns, self.frame))


def _voxel_type():
    from nav2_msgs.msg import VoxelGrid
    return VoxelGrid


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoxelMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

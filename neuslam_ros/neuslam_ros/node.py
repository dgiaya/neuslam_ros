"""NeuSLAM ROS 2 node."""
from __future__ import annotations

import csv
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener

from neuslam_msgs.msg import LoopClosure, Status
from neuslam_msgs.srv import SaveMap
from neuslam_ros.core import config as nscfg
from neuslam_ros.core import geometry as geo
from neuslam_ros.core import messages as M
from neuslam_ros.core.pipeline import Output, Params, Pipeline
from neuslam_ros.core.runner import SlamRunner


def default_models_dir() -> str:
    if os.environ.get("NEUSLAM_MODELS"):
        return os.environ["NEUSLAM_MODELS"]
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory("neuslam_ros")) / "models")


class NeuSlamNode(Node):
    def __init__(self) -> None:
        super().__init__("neuslam")
        P = self.declare_parameter
        self.mode = P("mode", "stereo").value
        self.sensors = nscfg.sensors(self.mode)
        P("input.rectified", False)
        P("input.reliable", False)
        P("input.sync_slop", 0.0)
        P("input.sync_queue", 10)
        P("input.frame_stride", 1)
        P("input.max_queue", 2)
        P("input.imu_wait_s", 0.05)
        self.map_frame = P("frames.map", "map").value
        self.odom_frame = P("frames.odom", "odom").value
        self.base_frame = P("frames.base", "base_link").value
        P("tf.publish_map_odom", True)
        P("tf.publish_odom_base", True)
        P("tf.extrinsics_timeout_s", 2.0)
        P("imu.T_optical_imu", [0.0])
        P("odometry.twist_frame", "child")
        P("odometry.gravity_align", True)
        P("covariance.pose", [0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        P("covariance.twist", [0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        P("covariance.lost_scale", 1000.0)
        P("depth.keyframes", True)
        P("depth.every_n_frames", 0)
        P("depth.encoding", "32FC1")
        P("depth.points", False)
        P("depth.points_stride", 4)
        P("map.points_period_s", 0.0)
        P("map.max_points", 100000)
        P("path.publish_every", 5)
        P("path.max_poses", 20000)
        P("debug.image_rect", False)
        P("preprocess.device", "cpu")
        P("preprocess.scale", 0.0)
        P("neuslam.config", "")
        P("neuslam.overrides", [""])
        P("neuslam.models_dir", "")
        P("neuslam.backend", "tensorrt")
        P("neuslam.multiprocess", True)
        P("neuslam.queue_size", 2)
        P("neuslam.build_engines", True)
        P("output_dir", "")
        g = self.get_parameter

        self.p = Params(mode=self.mode, rectified=g("input.rectified").value,
                        frame_stride=g("input.frame_stride").value, device=g("preprocess.device").value,
                        scale=float(g("preprocess.scale").value), gravity_align=g("odometry.gravity_align").value,
                        depth_every_n_frames=int(g("depth.every_n_frames").value),
                        pose_covariance=tuple(g("covariance.pose").value),
                        twist_covariance=tuple(g("covariance.twist").value),
                        lost_covariance_scale=float(g("covariance.lost_scale").value),
                        twist_in_parent=g("odometry.twist_frame").value == "parent",
                        imu_wait_s=float(g("input.imu_wait_s").value), path_max_poses=g("path.max_poses").value,
                        keep_images=g("debug.image_rect").value)
        self.pipeline = Pipeline(self.p)
        self.output_dir = self._output_dir(g("output_dir").value)
        cfg = self._neuslam_config()

        def make_slam():
            from neuslam.stream import StreamingSLAM
            return StreamingSLAM(cfg, out_dir=self.output_dir, multiprocess=g("neuslam.multiprocess").value,
                                 depth=bool(g("depth.keyframes").value),
                                 depth_every_n_frames=0 if self.sensors["rgbd"] else self.p.depth_every_n_frames)

        self.runner = SlamRunner(make_slam, self.pipeline, on_output=self._publish, on_map=self._publish_map,
                                 on_error=self._on_error, max_queue=g("input.max_queue").value,
                                 log=lambda s: self.get_logger().info(s))

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_pub = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.base_link_assumed = False
        self._first_frame_t: Optional[float] = None
        self._infos = {}
        self._calib_error = ""
        self._optical = ""
        self._imu_frame: Optional[str] = None

        reliable = QoSProfile(depth=10)
        self.pub_odom = self.create_publisher(Odometry, "~/odom", reliable)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "~/pose", reliable)
        self.pub_path = self.create_publisher(PathMsg, "~/path", 1)
        self.pub_status = self.create_publisher(Status, "~/status", reliable)
        self.pub_loop = self.create_publisher(LoopClosure, "~/loop_closure", reliable)
        self.pub_depth = self.create_publisher(Image, "~/depth/image", reliable)
        self.pub_depth_info = self.create_publisher(CameraInfo, "~/depth/camera_info", reliable)
        self.pub_depth_points = (self.create_publisher(PointCloud2, "~/depth/points", 1)
                                 if g("depth.points").value else None)
        self.pub_map = self.create_publisher(PointCloud2, "~/map/points", 1)
        self.pub_image = self.create_publisher(Image, "~/image_rect", 1) if self.p.keep_images else None
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 1)

        sensor = QoSProfile(depth=5) if g("input.reliable").value else qos_profile_sensor_data
        subs = [Subscriber(self, Image, "left/image", qos_profile=sensor)]
        self.create_subscription(CameraInfo, "left/camera_info", lambda m: self._infos.__setitem__("left", m),
                                 sensor)
        if self.sensors["stereo"]:
            subs.append(Subscriber(self, Image, "right/image", qos_profile=sensor))
            self.create_subscription(CameraInfo, "right/camera_info", lambda m: self._infos.__setitem__("right", m),
                                     sensor)
        if self.sensors["rgbd"]:
            subs.append(Subscriber(self, Image, "depth/image", qos_profile=sensor))
        if self.sensors["imu"]:
            self.create_subscription(Imu, "imu", self._on_imu, QoSProfile(depth=400, reliability=sensor.reliability))
        slop, q = float(g("input.sync_slop").value), int(g("input.sync_queue").value)
        if len(subs) == 1:
            subs[0].registerCallback(self._on_frame)
        else:
            sync = ApproximateTimeSynchronizer(subs, q, slop) if slop > 0 else TimeSynchronizer(subs, q)
            sync.registerCallback(self._on_frame)
        self._subs = subs
        # Callbacks only queue the images; preprocessing runs here, off the executor.
        self._raw: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self._stopping = threading.Event()
        self._prep_thread = threading.Thread(target=self._prepare_loop, name="preprocess", daemon=True)

        self.create_service(SaveMap, "~/save_map", self._srv_save)
        self.create_service(Trigger, "~/reset", self._srv_reset)
        period = float(g("map.points_period_s").value)
        if period > 0:
            self.create_timer(period, self._request_map)
        self.create_timer(1.0, self._diagnostics)

        self._out_times: deque = deque(maxlen=30)
        self._latency_ms = 0.0
        self._last: Optional[Output] = None
        self._path_count = 0
        self._timing = None
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            f = open(Path(self.output_dir) / "timing.csv", "w", newline="")
            self._timing = (f, csv.writer(f))
            self._timing[1].writerow(["stamp_ns", "recv", "pre", "submit", "result", "publish"])
        self.runner.start()
        self._prep_thread.start()
        self.get_logger().info(f"mode {self.mode}, {'rectified' if self.p.rectified else 'raw'} input; "
                               f"loading NeuSLAM (engines, vocabulary)")

    # -- setup -----------------------------------------------------------------------------------------
    def _output_dir(self, d: str) -> Optional[str]:
        if not d:
            return None
        return str(Path(os.path.expanduser(d)) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    def _neuslam_config(self):
        g = self.get_parameter
        models = g("neuslam.models_dir").value or default_models_dir()
        overrides = [o for o in g("neuslam.overrides").value if o]
        cfg = nscfg.build(self.mode, config=g("neuslam.config").value, overrides=overrides, models=models,
                          backend=g("neuslam.backend").value, queue_size=int(g("neuslam.queue_size").value))
        need = nscfg.engines_needed(cfg, current_pair_depth=self.p.depth_every_n_frames > 0
                                    and self.sensors["stereo"])
        missing = {k: v for k, v in need.items() if not v.is_file()}
        if missing:
            if not g("neuslam.build_engines").value:
                raise FileNotFoundError(f"TensorRT engines missing: {', '.join(map(str, missing.values()))}")
            nscfg.build_engines(missing, Path(cfg["model"]["weights"]), log=self.get_logger().info)
        return cfg

    def _extrinsics(self, optical: str, imu_frame: Optional[str]) -> bool:
        """Look up base_link <- left optical (and the IMU); fall back to base_link at the left camera."""
        if self._first_frame_t is None:
            self._first_frame_t = time.monotonic()
        waited = time.monotonic() - self._first_frame_t > float(self.get_parameter("tf.extrinsics_timeout_s").value)
        T_base_opt = None
        try:
            T_base_opt = geo.from_tf(self.tf_buffer.lookup_transform(self.base_frame, optical, Time()))
        except Exception:
            if not waited:
                return False
        if T_base_opt is None:
            T_base_opt = geo.make(geo.R_BODY_OPTICAL)
            self.base_link_assumed = True
            self.tf_static.sendTransform(M.tf(T_base_opt, 0, self.base_frame, optical))
            self.get_logger().warning(
                f"no TF {self.base_frame} -> {optical}: assuming {self.base_frame} is at the left camera "
                f"(x forward, z up) and publishing that transform; publish the camera's TF or set frames.base "
                f"to override")
        T_opt_imu = None
        if self.sensors["imu"]:
            ext = list(self.get_parameter("imu.T_optical_imu").value)
            if len(ext) == 16:
                T_opt_imu = np.asarray(ext, np.float64).reshape(4, 4)
            elif imu_frame:
                try:
                    T_opt_imu = geo.from_tf(self.tf_buffer.lookup_transform(optical, imu_frame, Time()))
                except Exception:
                    if imu_frame == self.base_frame:
                        T_opt_imu = geo.inv(T_base_opt)
                    elif not waited:
                        return False
            if T_opt_imu is None:
                self.get_logger().error(f"no TF {optical} -> {imu_frame or '<imu frame>'} and no imu.T_optical_imu: "
                                        "inertial modes need the camera-IMU extrinsic", throttle_duration_sec=5.0)
                return False
        self.pipeline.set_extrinsics(T_base_opt, T_opt_imu)
        if not self.p.rectified:
            self.tf_static.sendTransform(M.tf(geo.make(self.pipeline.R_raw_rect), 0, optical, self._rect_frame))
        return True

    @property
    def _rect_frame(self) -> str:
        f = self.pipeline.left.calib.frame_id if self.pipeline.left else ""
        f = f or self._optical
        return f if self.p.rectified else f"{f}_rect"

    # -- input -----------------------------------------------------------------------------------------
    def _on_imu(self, msg: Imu) -> None:
        self._imu_frame = msg.header.frame_id
        self.pipeline.add_imu(msg)

    def _on_frame(self, *msgs) -> None:
        if not self.runner.ready.is_set():
            self.runner.dropped += 1
            return
        try:
            self._raw.put_nowait(msgs)
        except queue.Full:
            self.runner.dropped += 1
            self.get_logger().warning("preprocessing is behind: frame dropped", throttle_duration_sec=5.0)

    def _prepare_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                msgs = self._raw.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handle_frame(*msgs)
            except Exception as e:
                self.get_logger().error(f"frame failed: {e!r}", throttle_duration_sec=5.0)

    def _handle_frame(self, *msgs) -> None:
        left = msgs[0]
        right = msgs[1] if self.sensors["stereo"] else None
        depth = msgs[1] if self.sensors["rgbd"] else None
        log = self.get_logger()
        try:
            if "left" not in self._infos or (self.sensors["stereo"] and "right" not in self._infos):
                log.warning("waiting for camera_info", throttle_duration_sec=5.0)
                return
            changed = self.pipeline.set_calibration(self._infos["left"], self._infos.get("right"))
        except ValueError as e:
            self._calib_error = str(e)
            log.error(f"unusable calibration: {e}", throttle_duration_sec=5.0)
            return
        self._calib_error = ""
        self._optical = left.header.frame_id
        if changed or self.pipeline.T_base_cam is None:
            if self.pipeline.anchor is not None and changed:
                log.warning("calibration changed while running")
            if not self._extrinsics(left.header.frame_id, self._imu_frame):
                return
        if not self.pipeline.take():
            return
        prep = self.pipeline.prepare(left, right, depth)
        if not self.runner.offer(prep):
            log.warning("NeuSLAM is behind: frame dropped", throttle_duration_sec=5.0)

    # -- output ----------------------------------------------------------------------------------------
    def _publish(self, o: Output) -> None:
        if not self.context.ok():   # results drained after Ctrl-C: NeuSLAM's files have them
            return
        now_ns = self.get_clock().now().nanoseconds
        self._latency_ms = (now_ns - o.stamp_ns) * 1e-6
        self._out_times.append(time.monotonic())
        g = self.get_parameter
        self.pub_odom.publish(M.odometry(o, self.odom_frame, self.base_frame))
        self.pub_pose.publish(M.pose_with_cov(o, self.map_frame))
        tfs = []
        if g("tf.publish_odom_base").value:
            tfs.append(M.tf(o.T_odom_base, o.stamp_ns, self.odom_frame, self.base_frame))
        if g("tf.publish_map_odom").value:
            tfs.append(M.tf(o.T_map_odom, o.stamp_ns, self.map_frame, self.odom_frame))
        if tfs:
            self.tf_pub.sendTransform(tfs)
        self._path_count += 1
        if o.path is not None or self._path_count % max(1, int(g("path.publish_every").value)) == 0:
            self.pub_path.publish(M.path(o.path if o.path is not None else self.pipeline.path, self.map_frame,
                                         o.stamp_ns))
        for m in M.loop_closures(o, self.map_frame, len(o.path or [])):
            self.pub_loop.publish(m)
        if o.depth is not None:
            ts, d = o.depth
            frame = self._rect_frame
            self.pub_depth.publish(M.depth_image(d, g("depth.encoding").value, ts, frame))
            self.pub_depth_info.publish(M.camera_info(self.pipeline.K, ts, frame, d.shape[1], d.shape[0]))
            if self.pub_depth_points is not None:
                pts = M.depth_points(d, self.pipeline.K, int(g("depth.points_stride").value))
                self.pub_depth_points.publish(M.cloud(pts, ts, frame))
        if self.pub_image is not None and o.image_u8 is not None:
            self.pub_image.publish(M.image(o.image_u8, "rgb8", o.stamp_ns, self._rect_frame))
        self.pub_status.publish(M.status(o, mode=self.mode, base_frame=self.base_frame, rate_hz=self._rate(),
                                         latency_ms=self._latency_ms, dropped=self.runner.dropped,
                                         base_link_assumed=self.base_link_assumed))
        self._last = o
        if self._timing is not None and o.timing:
            t = o.timing
            self._timing[1].writerow([o.stamp_ns, t["recv"], t["pre"], t["submit"], t["result"], time.time()])

    def _publish_map(self, snap) -> None:
        if not self.context.ok() or self.pipeline.anchor is None or len(snap.points) == 0:
            return
        pts, cols = self.pipeline.map_points(snap)
        self.pub_map.publish(M.cloud(pts, self.get_clock().now().nanoseconds, self.map_frame, cols))

    def _request_map(self) -> None:
        if self.runner.ready.is_set() and self.pub_map.get_subscription_count() > 0:
            n = int(self.get_parameter("map.max_points").value)
            self.runner.call(lambda s: s.request_map(max_points=n))

    def _rate(self) -> float:
        t = self._out_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def _diagnostics(self) -> None:
        st = DiagnosticStatus(name="neuslam", hardware_id=self.mode)
        o = self._last
        if not self.runner.ready.is_set():
            st.level, st.message = DiagnosticStatus.WARN, "starting" if not self.runner.done.is_set() else "stopped"
        elif self._calib_error:
            st.level, st.message = DiagnosticStatus.ERROR, self._calib_error
        elif o is None:
            st.level, st.message = DiagnosticStatus.WARN, "no frames yet"
        elif o.state != "tracking":
            st.level, st.message = DiagnosticStatus.WARN, o.state
        else:
            st.level, st.message = DiagnosticStatus.OK, "tracking"
        if self.base_link_assumed:
            st.message += f"; {self.base_frame} assumed at the left camera"
        vals = {"rate_hz": f"{self._rate():.2f}", "latency_ms": f"{self._latency_ms:.0f}",
                "frames_dropped": str(self.runner.dropped), "base_link_assumed": str(self.base_link_assumed)}
        if o is not None:
            vals.update(pipeline_ms=f"{o.pipeline_s * 1e3:.0f}", keyframes=str(o.num_keyframes),
                        map_points=str(o.num_map_points), loop_closures=str(o.num_loop_closures))
        st.values = [KeyValue(key=k, value=v) for k, v in vals.items()]
        self.pub_diag.publish(DiagnosticArray(header=M.header(self.get_clock().now().nanoseconds, ""), status=[st]))

    # -- services --------------------------------------------------------------------------------------
    def _srv_save(self, req, res):
        d = req.directory or str(Path(self.output_dir or os.path.expanduser("~/neuslam_runs"))
                                 / datetime.now(timezone.utc).strftime("save_%Y%m%dT%H%M%SZ"))
        try:
            files = self.runner.call(lambda s: s.save(d)).result(timeout=180)
            res.success, res.message, res.files = True, f"saved to {d}", sorted(str(v) for v in files.values())
        except Exception as e:
            res.success, res.message = False, repr(e)
        return res

    def _srv_reset(self, req, res):
        try:
            self.runner.reset().result(timeout=180)
            res.success, res.message = True, "NeuSLAM restarted"
        except Exception as e:
            res.success, res.message = False, repr(e)
        return res

    def _on_error(self, e: Exception) -> None:
        self.get_logger().fatal(f"NeuSLAM failed: {e!r}")

    def shutdown(self) -> None:
        self.get_logger().info("stopping NeuSLAM" + (f", saving to {self.output_dir}" if self.output_dir else ""))
        self._stopping.set()
        self._prep_thread.join(timeout=5)
        self.runner.stop(save_to=self.output_dir, timeout=300)
        if self._timing is not None:
            self._timing[0].close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NeuSlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

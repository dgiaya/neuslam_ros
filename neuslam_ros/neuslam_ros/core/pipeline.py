"""ROS-message frames -> NeuSLAM -> REP-105 poses, depth and events (no rclpy)."""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from neuslam.stream import Frame, MapSnapshot, Result
from neuslam_ros.core import geometry as geo
from neuslam_ros.core.camera import Calibration, Resampler, check, stereo_baseline
from neuslam_ros.core.config import sensors
from neuslam_ros.core.images import depth_to_metres, to_numpy
from neuslam_ros.core.imu import ImuBuffer

def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


@dataclass
class Params:
    mode: str = "stereo"
    rectified: bool = True
    frame_stride: int = 1
    device: str = "cpu"
    scale: float = 0.0
    gravity_align: bool = True
    depth_every_n_frames: int = 0
    pose_covariance: Tuple[float, ...] = (0.01, 0.01, 0.01, 0.001, 0.001, 0.001)
    twist_covariance: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.01, 0.01, 0.01)
    lost_covariance_scale: float = 1000.0
    twist_in_parent: bool = False
    imu_wait_s: float = 0.05
    path_max_poses: int = 20000
    keep_images: bool = False


@dataclass
class Prepared:
    stamp_ns: int
    frame: Frame
    t_recv: float
    t_pre: float
    image_u8: Optional[np.ndarray] = None
    depth_m: Optional[np.ndarray] = None
    t_submit: float = 0.0


@dataclass
class Output:
    stamp_ns: int
    frame_id: int
    state: str
    is_keyframe: bool
    keyframe_id: int
    num_keyframes: int
    num_map_points: int
    num_loop_closures: int
    T_odom_base: np.ndarray
    T_map_base: np.ndarray
    T_map_odom: np.ndarray
    linear: np.ndarray
    angular: np.ndarray
    pose_cov: np.ndarray
    twist_cov: np.ndarray
    loops: List[Tuple[int, int]] = field(default_factory=list)
    path: Optional[List[Tuple[int, np.ndarray]]] = None     # whole revised path (map, base_link)
    depth: Optional[Tuple[int, np.ndarray]] = None          # (stamp_ns, metres) in the rectified camera
    image_u8: Optional[np.ndarray] = None
    pipeline_s: float = 0.0
    timing: Dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, params: Params) -> None:
        self.p = params
        self.s = sensors(params.mode)
        self.imu = ImuBuffer()
        self.left: Optional[Resampler] = None
        self.right: Optional[Resampler] = None
        self.baseline: Optional[float] = None
        self._calib_key: Optional[tuple] = None
        self.T_base_cam: Optional[np.ndarray] = None   # rectified left camera in base_link
        self.T_cam_imu: Optional[np.ndarray] = None    # rectified left camera <- IMU
        self.anchor: Optional[geo.Anchor] = None
        self._seen = 0
        self.dropped = 0
        self._t_prev: Optional[float] = None
        self._pending: "OrderedDict[int, Prepared]" = OrderedDict()   # key: microsecond stamp
        self._frames: "OrderedDict[int, Tuple[int, int, np.ndarray, np.ndarray]]" = OrderedDict()
        self._path: deque = deque(maxlen=max(1, int(params.path_max_poses)))
        self._last: Optional[Tuple[int, np.ndarray]] = None
        self.num_keyframes = 0
        self.num_loops = 0

    # -- calibration and extrinsics ------------------------------------------------------------------
    def set_calibration(self, left_info, right_info=None) -> bool:
        """(Re)build the resampling; True when it changed. Raises ValueError on unusable calibration."""
        lc = Calibration.from_msg(left_info)
        rc = Calibration.from_msg(right_info) if right_info is not None else None
        key = (lc.key(), rc.key() if rc else None)
        if key == self._calib_key:
            return False
        check(lc, rectified=self.p.rectified, name="left camera")
        if self.s["stereo"]:
            if rc is None:
                raise ValueError("stereo mode needs the right CameraInfo")
            check(rc, rectified=self.p.rectified, name="right camera")
            self.baseline = stereo_baseline(lc, rc)
        self.left = Resampler(lc, rectified=self.p.rectified, device=self.p.device, scale=self.p.scale)
        self.right = (Resampler(rc, rectified=self.p.rectified, device=self.p.device, scale=self.p.scale)
                      if self.s["stereo"] else None)
        self._calib_key = key
        return True

    @property
    def calibrated(self) -> bool:
        return self.left is not None

    @property
    def R_raw_rect(self) -> np.ndarray:
        """Rotation of the rectified left camera in the raw left optical frame (identity for rectified input)."""
        return np.eye(3) if self.p.rectified or self.left is None else self.left.calib.R.T

    @property
    def path(self) -> List[Tuple[int, np.ndarray]]:
        return list(self._path)

    @property
    def K(self) -> np.ndarray:
        return self.left.K

    def set_extrinsics(self, T_base_optical: np.ndarray, T_optical_imu: Optional[np.ndarray] = None) -> None:
        """Left camera's optical frame in base_link; IMU in the optical frame (inertial modes)."""
        T_opt_rect = geo.make(self.R_raw_rect)
        self.T_base_cam = np.asarray(T_base_optical, np.float64) @ T_opt_rect
        if T_optical_imu is not None:
            self.T_cam_imu = geo.inv(T_opt_rect) @ np.asarray(T_optical_imu, np.float64)

    # -- input -----------------------------------------------------------------------------------------
    def add_imu(self, msg) -> None:
        w, a = msg.angular_velocity, msg.linear_acceleration
        self.imu.add(stamp_ns(msg.header) * 1e-9, (w.x, w.y, w.z), (a.x, a.y, a.z))

    def take(self) -> bool:
        """Frame stride: whether the next synchronised set is used."""
        keep = self._seen % max(1, int(self.p.frame_stride)) == 0
        self._seen += 1
        return keep

    def prepare(self, left, right=None, depth=None) -> Prepared:
        t_recv = time.time()
        if self.left is None:
            raise RuntimeError("no calibration yet")
        ts = stamp_ns(left.header)
        lt = self.left.image(to_numpy(left), left.encoding)
        frame = Frame(timestamp=ts * 1e-9, left=lt, K=self.left.K, pad_box=self.left.pad_box)
        if self.s["stereo"]:
            frame.right = self.right.image(to_numpy(right), right.encoding)
            frame.baseline = self.baseline
        depth_m = None
        if self.s["rgbd"]:
            d = self.left.depth(depth_to_metres(depth))
            frame.depth = d.cpu()
            if self.p.depth_every_n_frames > 0:
                depth_m = frame.depth.numpy()
        image_u8 = (lt.permute(1, 2, 0) * 255).byte().cpu().numpy() if self.p.keep_images else None
        frame.left = lt.cpu()
        if frame.right is not None:
            frame.right = frame.right.cpu()
        return Prepared(ts, frame, t_recv, time.time(), image_u8, depth_m)

    def submit(self, slam, prep: Prepared, *, timeout: Optional[float] = None) -> bool:
        f = prep.frame
        t = f.timestamp
        if self.s["imu"]:
            self.imu.wait_until(t, self.p.imu_wait_s)
            t0 = t if self._t_prev is None else self._t_prev
            f.imu, f.imu_t0, f.T_cam_imu = self.imu.chunk(t0, t), t0, self.T_cam_imu
            if f.imu is None:
                f.imu = f.imu_t0 = None
        prep.t_submit = time.time()
        if not slam.submit(f, timeout=timeout):
            return False
        self._t_prev = t
        self._pending[round(t * 1e6)] = prep
        while len(self._pending) > 512:
            self._pending.popitem(last=False)
        return True

    # -- output ----------------------------------------------------------------------------------------
    def _ensure_anchor(self, t_s: float) -> None:
        if self.anchor is not None:
            return
        if self.T_base_cam is None:
            raise RuntimeError("extrinsics not set")
        R0 = None
        if self.p.gravity_align and self.s["imu"] and self.T_cam_imu is not None:
            a = self.imu.mean_accel(t_s - 0.5, t_s)
            if a is not None and np.linalg.norm(a) > 1.0:
                R_base_imu = (self.T_base_cam @ self.T_cam_imu)[:3, :3]
                R0 = geo.gravity_aligned(R_base_imu @ a)
        self.anchor = geo.Anchor(self.T_base_cam, R0)

    def _cov(self, diag, state: str) -> np.ndarray:
        k = 1.0 if state == "tracking" else float(self.p.lost_covariance_scale)
        return np.diag(np.asarray(diag, np.float64) * k)

    def _revise(self, ids: np.ndarray, poses: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        kf = {int(i): P for i, P in zip(ids, poses)}
        for fid, (ts, ref, T_ref_cur, T_c0) in list(self._frames.items()):
            if fid in kf:
                self._frames[fid] = (ts, ref, T_ref_cur, kf[fid])
            elif ref in kf:
                self._frames[fid] = (ts, ref, T_ref_cur, kf[ref] @ T_ref_cur)
        self._path = deque(((ts, self.anchor.base(T)) for ts, _, _, T in self._frames.values()),
                           maxlen=self._path.maxlen)
        return list(self._path)

    def output(self, r: Result) -> Optional[Output]:
        prep = self._pending.pop(round(r.timestamp * 1e6), None)
        ts = prep.stamp_ns if prep is not None else int(round(r.timestamp * 1e9))
        self._ensure_anchor(ts * 1e-9)
        T_c0_cam = np.asarray(r.T_world_cam, np.float64)
        ref = self._frames.get(int(r.ref_frame_id))
        T_ref_cur = geo.inv(ref[3]) @ T_c0_cam if ref is not None else np.eye(4)
        self._frames[int(r.frame_id)] = (ts, int(r.ref_frame_id), T_ref_cur, T_c0_cam)
        while len(self._frames) > self._path.maxlen:
            self._frames.popitem(last=False)
        T_odom_base = self.anchor.base(r.T_world_cam_odom)
        T_map_base = self.anchor.base(T_c0_cam)
        path = None
        if (r.poses_revised or r.loops) and r.keyframe_ids is not None and len(r.keyframe_ids):
            path = self._revise(r.keyframe_ids, r.keyframe_poses)
        else:
            self._path.append((ts, T_map_base))
        lin, ang = np.zeros(3), np.zeros(3)
        if self._last is not None:
            lin, ang = geo.twist(self._last[1], T_odom_base, (ts - self._last[0]) * 1e-9,
                                 body=not self.p.twist_in_parent)
        self._last = (ts, T_odom_base)
        self.num_keyframes += int(r.is_keyframe)
        self.num_loops += len(r.loops)
        depth = None
        if r.depth is not None and r.depth_frame_id is not None:
            dts = self._frames.get(int(r.depth_frame_id), (ts,))[0]
            depth = (dts, np.asarray(r.depth, np.float32))
        elif prep is not None and prep.depth_m is not None and self.p.depth_every_n_frames > 0 \
                and int(r.frame_id) % self.p.depth_every_n_frames == 0:
            depth = (ts, prep.depth_m)
        timing = {}
        if prep is not None:
            timing = {"recv": prep.t_recv, "pre": prep.t_pre, "submit": prep.t_submit,
                      "result": time.time()}
        return Output(
            stamp_ns=ts, frame_id=int(r.frame_id), state=r.state, is_keyframe=bool(r.is_keyframe),
            keyframe_id=int(r.ref_frame_id), num_keyframes=self.num_keyframes, num_map_points=int(r.num_map_points),
            num_loop_closures=self.num_loops, T_odom_base=T_odom_base, T_map_base=T_map_base,
            T_map_odom=T_map_base @ geo.inv(T_odom_base), linear=lin, angular=ang,
            pose_cov=self._cov(self.p.pose_covariance, r.state), twist_cov=self._cov(self.p.twist_covariance, r.state),
            loops=list(r.loops), path=path, depth=depth, image_u8=prep.image_u8 if prep is not None else None,
            pipeline_s=(timing["result"] - timing["submit"]) if timing else 0.0, timing=timing)

    def map_points(self, snap: MapSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        """Map snapshot -> (N, 3) points in map, (N, 3) uint8 colours."""
        return self.anchor.world_points(np.asarray(snap.points, np.float64)), np.asarray(snap.colors, np.uint8)

    def reset(self) -> None:
        self.anchor = None
        self._t_prev = None
        self._pending.clear()
        self._frames.clear()
        self._path.clear()
        self._last = None
        self.num_keyframes = self.num_loops = 0

"""Rigid transforms, REP-103/105 frames, and the anchoring of NeuSLAM's world in odom/map."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Columns: the optical axes (x right, y down, z forward) in body axes (x forward, y left, z up).
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def make(R: np.ndarray, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make(R.T, -R.T @ t)


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat()


def from_xyz_quat(t, q_xyzw) -> np.ndarray:
    return make(Rotation.from_quat(q_xyzw).as_matrix(), t)


def from_tf(t) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4."""
    tr, q = t.transform.translation, t.transform.rotation
    return from_xyz_quat((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))


def gravity_aligned(up_in_base: np.ndarray) -> np.ndarray:
    """R_odom_base at the start: z along `up_in_base` (e.g. mean accelerometer), x along the
    base's x projected on the horizontal plane (yaw kept)."""
    z = np.asarray(up_in_base, np.float64)
    z = z / np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).T


class Anchor:
    """NeuSLAM's world is the first (rectified) camera frame C0. odom and map start at base_link
    of the first frame (gravity-aligned when `R_odom_base0` is given), so map == odom until the
    first correction."""

    def __init__(self, T_base_cam: np.ndarray, R_odom_base0: Optional[np.ndarray] = None) -> None:
        self.T_base_cam = np.asarray(T_base_cam, np.float64)
        self.T_cam_base = inv(self.T_base_cam)
        R0 = np.eye(3) if R_odom_base0 is None else np.asarray(R_odom_base0, np.float64)
        self.T_odom_c0 = make(R0) @ self.T_base_cam

    def base(self, T_c0_cam: np.ndarray) -> np.ndarray:
        """T_odom_base (or T_map_base) of a NeuSLAM camera pose."""
        return self.T_odom_c0 @ np.asarray(T_c0_cam, np.float64) @ self.T_cam_base

    def world_points(self, pts_c0: np.ndarray) -> np.ndarray:
        """Points in C0 -> map/odom."""
        T = self.T_odom_c0
        return pts_c0 @ T[:3, :3].T + T[:3, 3]


def twist(T0: np.ndarray, T1: np.ndarray, dt: float, *, body: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(linear, angular) velocity from two poses of the same body, in the body frame (REP-105)
    or in the parent frame."""
    if dt <= 0:
        return np.zeros(3), np.zeros(3)
    v = (T1[:3, 3] - T0[:3, 3]) / dt
    w_body = Rotation.from_matrix(T0[:3, :3].T @ T1[:3, :3]).as_rotvec() / dt
    if body:
        return T1[:3, :3].T @ v, w_body
    return v, T1[:3, :3] @ w_body

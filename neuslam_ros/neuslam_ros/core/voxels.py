"""Rolling occupancy grid around the vehicle, packed as nav2_msgs/VoxelGrid (16 levels per column)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

MAX_Z = 16


class LocalVoxelGrid:
    """Cells are unknown, free (seen through) or marked (a depth return), in a fixed frame.
    The grid is re-centred on the vehicle; cells that leave it are forgotten."""

    def __init__(self, *, resolution: float = 0.2, z_resolution: float = 0.25, size_x: int = 200,
                 size_y: int = 200, size_z: int = 16, max_range: float = 15.0) -> None:
        if not 1 <= size_z <= MAX_Z:
            raise ValueError(f"size_z must be 1..{MAX_Z} (nav2 VoxelGrid packs a column in 32 bits)")
        self.res = np.array([resolution, resolution, z_resolution], np.float64)
        self.size = np.array([size_x, size_y, size_z], np.int64)
        self.max_range = float(max_range)
        self.known = np.zeros(self.size[::-1], bool)     # [z, y, x]
        self.marked = np.zeros(self.size[::-1], bool)
        self.origin_idx: Optional[np.ndarray] = None    # world cell index of grid cell (0, 0, 0)

    @property
    def origin(self) -> np.ndarray:
        return self.origin_idx * self.res

    def recentre(self, centre: np.ndarray) -> None:
        idx = np.floor(np.asarray(centre, np.float64) / self.res).astype(np.int64) - self.size // 2
        if self.origin_idx is None:
            self.origin_idx = idx
            return
        shift = idx - self.origin_idx
        if not shift.any():
            return
        self.origin_idx = idx
        for a in (self.known, self.marked):
            moved = np.zeros_like(a)
            src = [slice(max(0, s), min(n, n + s)) for s, n in zip(shift[::-1], a.shape)]
            dst = [slice(max(0, -s), min(n, n - s)) for s, n in zip(shift[::-1], a.shape)]
            if all(sl.stop > sl.start for sl in src):
                moved[tuple(dst)] = a[tuple(src)]
            a[...] = moved

    def _cells(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = np.floor(pts / self.res).astype(np.int64) - self.origin_idx
        ok = np.all((c >= 0) & (c < self.size), axis=1)
        return c, ok

    def integrate(self, depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, *, stride: int = 8,
                  centre: Optional[np.ndarray] = None) -> None:
        """Add one depth image (metres, NaN = none) seen by a camera at T_world_cam."""
        self.recentre(T_world_cam[:3, 3] if centre is None else centre)
        h, w = depth.shape
        v, u = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        z = depth[v, u]
        ok = np.isfinite(z) & (z > 0)
        u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok].astype(np.float64)
        if z.size == 0:
            return
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], 1)
        dist = z * np.linalg.norm(rays, axis=1)
        dirs = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
        dirs_w = dirs @ R.T
        hit = dist <= self.max_range
        length = np.minimum(dist, self.max_range)
        step = float(self.res.min())
        n = int(np.ceil(length.max() / step))
        ts = (np.arange(n) * step)[None, :]
        along = ts < (length[:, None] - step)
        pts = o + dirs_w[:, None, :] * ts[..., None]
        c, inside = self._cells(pts[along])
        c = c[inside]
        self.known[c[:, 2], c[:, 1], c[:, 0]] = True
        self.marked[c[:, 2], c[:, 1], c[:, 0]] = False
        ends = o + dirs_w[hit] * dist[hit, None]
        c, inside = self._cells(ends)
        c = c[inside]
        self.known[c[:, 2], c[:, 1], c[:, 0]] = True
        self.marked[c[:, 2], c[:, 1], c[:, 0]] = True

    def packed(self) -> np.ndarray:
        """(size_y * size_x,) uint32 columns, index y * size_x + x: bit z = unknown or marked,
        bit z + 16 = marked (nav2_voxel_grid)."""
        z = np.arange(self.size[2], dtype=np.uint32)[:, None, None]
        low = (~self.known | self.marked).astype(np.uint32) << z
        high = self.marked.astype(np.uint32) << (z + np.uint32(16))
        return (low | high).sum(axis=0, dtype=np.uint32).reshape(-1)

    def marked_points(self) -> np.ndarray:
        """(N, 3) centres of marked cells."""
        zi, yi, xi = np.nonzero(self.marked)
        return (np.stack([xi, yi, zi], 1) + self.origin_idx + 0.5) * self.res

"""IMU samples -> per-frame chunks in NeuSLAM's row layout."""
from __future__ import annotations

import bisect
import threading
from typing import List, Optional

import numpy as np

from neuslam.stream import imu_rows


class ImuBuffer:
    """Thread-safe buffer of (t, gyro, accel); `chunk(t0, t1)` returns the rows covering
    [t0, t1] including one sample at or before t0 and one at or after t1."""

    def __init__(self, keep_s: float = 30.0) -> None:
        self.keep_s = float(keep_s)
        self._t: List[float] = []
        self._v: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def add(self, t: float, gyro, accel) -> None:
        with self._cv:
            if self._t and t <= self._t[-1]:
                return
            self._t.append(float(t))
            self._v.append(np.concatenate([np.asarray(gyro, np.float64), np.asarray(accel, np.float64)]))
            if self._t[-1] - self._t[0] > self.keep_s:
                k = bisect.bisect_left(self._t, self._t[-1] - self.keep_s)
                del self._t[:k], self._v[:k]
            self._cv.notify_all()

    @property
    def latest(self) -> Optional[float]:
        with self._lock:
            return self._t[-1] if self._t else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._t)

    def wait_until(self, t: float, timeout: float) -> bool:
        """Block until a sample at or after `t` exists (or `timeout` s pass)."""
        with self._cv:
            return self._cv.wait_for(lambda: bool(self._t) and self._t[-1] >= t, timeout=timeout)

    def chunk(self, t0: float, t1: float) -> Optional[np.ndarray]:
        """(N, 10) rows; the last sample is held up to t1 when the stream has not reached it."""
        with self._lock:
            if len(self._t) < 2 or self._t[0] > t0:
                return None
            a = max(0, bisect.bisect_right(self._t, t0) - 1)
            b = bisect.bisect_left(self._t, t1)
            b = min(b, len(self._t) - 1)
            t = np.asarray(self._t[max(0, a - 1):b + 2])
            v = np.stack(self._v[max(0, a - 1):b + 2])
        if t[-1] < t1:
            t = np.append(t, t1)
            v = np.vstack([v, v[-1]])
        rows = imu_rows(t, v[:, :3], v[:, 3:])
        lo = max(0, np.searchsorted(rows[:, 0], t0, side="right") - 1)
        hi = min(len(rows), np.searchsorted(rows[:, 0], t1, side="left") + 1)
        return rows[lo:hi]

    def mean_accel(self, t0: float, t1: float) -> Optional[np.ndarray]:
        with self._lock:
            sel = [v[3:] for t, v in zip(self._t, self._v) if t0 <= t <= t1]
        return np.mean(sel, axis=0) if sel else None

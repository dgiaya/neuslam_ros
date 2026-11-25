"""sensor_msgs/Image <-> numpy, without cv_bridge."""
from __future__ import annotations

from typing import Tuple

import numpy as np

_DTYPES = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8UC1": (np.uint8, 1), "8UC3": (np.uint8, 3),
    "rgb16": (np.uint16, 3), "bgr16": (np.uint16, 3), "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}
BAYER = ("bayer_rggb", "bayer_bggr", "bayer_gbrg", "bayer_grbg")


def is_bayer(encoding: str) -> bool:
    return encoding.startswith(BAYER)


def _layout(encoding: str) -> Tuple[np.dtype, int]:
    if is_bayer(encoding):
        return (np.uint16 if encoding.endswith("16") else np.uint8), 1
    if encoding not in _DTYPES:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    return _DTYPES[encoding]


def to_numpy(msg) -> np.ndarray:
    """(H, W) or (H, W, C) view of the message's pixels (copied only when rows are padded)."""
    dtype, ch = _layout(msg.encoding)
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    row = msg.width * ch * dt.itemsize
    a = buf.reshape(msg.height, msg.step)[:, :row] if msg.step != row else buf[: msg.height * row]
    a = np.ascontiguousarray(a).view(dt).reshape(msg.height, msg.width, ch)
    if dt.byteorder == ">":
        a = a.astype(dt.newbyteorder("="))
    return a[..., 0] if ch == 1 else a


def depth_to_metres(msg) -> np.ndarray:
    """Depth image (16UC1 mm or 32FC1 m) -> (H, W) float32 metres, invalid = NaN."""
    a = to_numpy(msg)
    if msg.encoding in ("16UC1", "mono16"):
        d = a.astype(np.float32) * 1e-3
        d[a == 0] = np.nan
    elif msg.encoding == "32FC1":
        d = a.astype(np.float32, copy=True)
        d[~np.isfinite(d) | (d <= 0)] = np.nan
    else:
        raise ValueError(f"depth encoding {msg.encoding!r}: use 16UC1 (mm) or 32FC1 (m)")
    return d


def fill(msg, array: np.ndarray, encoding: str) -> None:
    """Write `array` into an Image message (header untouched)."""
    a = np.ascontiguousarray(array)
    msg.height, msg.width = int(a.shape[0]), int(a.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(a.strides[0])
    msg.data = a.tobytes()


def depth_array(depth_m: np.ndarray, encoding: str) -> np.ndarray:
    """Metres (NaN = invalid) -> 32FC1 metres or 16UC1 millimetres (0 = invalid)."""
    if encoding == "32FC1":
        return depth_m.astype(np.float32)
    if encoding == "16UC1":
        mm = np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(np.round(mm), 0, 65535).astype(np.uint16)
    raise ValueError(f"depth output encoding {encoding!r}: use 32FC1 or 16UC1")

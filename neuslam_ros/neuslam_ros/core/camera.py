"""Camera calibration from CameraInfo, and the GPU resampling to NeuSLAMNet's input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from neuslam_ros.core.images import is_bayer

OUT_H, OUT_W = 368, 768


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    P: np.ndarray
    model: str
    frame_id: str

    @classmethod
    def from_msg(cls, msg) -> "Calibration":
        R = np.asarray(msg.r, dtype=np.float64).reshape(3, 3)
        if not np.any(R):
            R = np.eye(3)
        return cls(int(msg.width), int(msg.height), np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
                   np.asarray(msg.d, dtype=np.float64).reshape(-1), R,
                   np.asarray(msg.p, dtype=np.float64).reshape(3, 4), str(msg.distortion_model or "plumb_bob"),
                   str(msg.header.frame_id))

    def key(self) -> tuple:
        return (self.width, self.height, self.model, *np.round(self.K, 9).ravel(), *np.round(self.D, 9),
                *np.round(self.R, 9).ravel(), *np.round(self.P, 9).ravel())

    @property
    def has_projection(self) -> bool:
        return bool(self.P[0, 0] > 0 and self.P[1, 1] > 0)

    @property
    def has_intrinsics(self) -> bool:
        return bool(self.K[0, 0] > 0 and self.K[1, 1] > 0)


def check(calib: Calibration, *, rectified: bool, name: str) -> None:
    if calib.width <= 0 or calib.height <= 0:
        raise ValueError(f"{name}: CameraInfo has no image size (uncalibrated camera?)")
    if not calib.has_projection:
        raise ValueError(f"{name}: CameraInfo has no projection matrix P (uncalibrated camera?)")
    if not rectified and not calib.has_intrinsics:
        raise ValueError(f"{name}: raw input needs K and D in CameraInfo")
    if not rectified and calib.model not in ("plumb_bob", "rational_polynomial", "equidistant"):
        raise ValueError(f"{name}: distortion model {calib.model!r} not supported "
                         "(plumb_bob, rational_polynomial, equidistant)")


def stereo_baseline(left: Calibration, right: Calibration) -> float:
    """Baseline (m) from the right camera's P, checked against the left's."""
    if not np.allclose(left.P[:, :3], right.P[:, :3], rtol=1e-3, atol=1e-3):
        raise ValueError("left and right CameraInfo P differ in fx, fy, cx or cy: not a rectified stereo pair")
    b = -right.P[0, 3] / right.P[0, 0]
    if not b > 0:
        raise ValueError(f"right CameraInfo P[0,3] gives baseline {b:.4f} m: stereo needs the stereo "
                         "calibration's P (P[0,3] = -fx * baseline)")
    return float(b)


def output_geometry(calib: Calibration, scale: float = 0.0) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float, float, float]:
    """(K of the network image, pad_box or None, scale, x offset, y offset) for the rectified image
    scaled by `scale` (0: to the network width) and centre-cropped or padded to OUT_W x OUT_H."""
    s = float(scale) if scale > 0 else OUT_W / calib.width
    ox = (calib.width * s - OUT_W) / 2.0
    oy = (calib.height * s - OUT_H) / 2.0
    Kr = calib.P[:, :3]
    K = np.array([[Kr[0, 0] * s, 0.0, (Kr[0, 2] + 0.5) * s - 0.5 - ox],
                  [0.0, Kr[1, 1] * s, (Kr[1, 2] + 0.5) * s - 0.5 - oy],
                  [0.0, 0.0, 1.0]])
    pad_l, pad_t = max(0, int(np.ceil(-ox))), max(0, int(np.ceil(-oy)))
    pad = None if pad_l == 0 and pad_t == 0 else (pad_l, pad_t, OUT_W - 2 * pad_l, OUT_H - 2 * pad_t)
    return K, pad, s, ox, oy


_CV_BAYER = {"rggb": cv2.COLOR_BayerBG2RGB, "bggr": cv2.COLOR_BayerRG2RGB,
             "gbrg": cv2.COLOR_BayerGR2RGB, "grbg": cv2.COLOR_BayerGB2RGB}


class Resampler:
    """Source image -> (3, OUT_H, OUT_W) float in [0, 1]: undistortion and rectification (raw input),
    scaling, and the crop or pad, in one remap. `device`: cpu (OpenCV) or cuda (grid_sample)."""

    def __init__(self, calib: Calibration, *, rectified: bool, device: str = "cpu", scale: float = 0.0) -> None:
        self.calib = calib
        self.device = torch.device(device)
        self.K, self.pad_box, s, ox, oy = output_geometry(calib, scale)
        self.scale = s
        if rectified:
            u, v = np.meshgrid(np.arange(OUT_W, dtype=np.float64), np.arange(OUT_H, dtype=np.float64))
            mx = (u + ox + 0.5) / s - 0.5
            my = (v + oy + 0.5) / s - 0.5
        elif calib.model == "equidistant":
            mx, my = cv2.fisheye.initUndistortRectifyMap(calib.K, calib.D[:4], calib.R, self.K, (OUT_W, OUT_H),
                                                         cv2.CV_32FC1)
        else:
            mx, my = cv2.initUndistortRectifyMap(calib.K, calib.D, calib.R, self.K, (OUT_W, OUT_H), cv2.CV_32FC1)
        mx, my = np.asarray(mx, np.float64), np.asarray(my, np.float64)
        self.pool = max(1, int(np.floor(1.0 / s))) if s < 0.5 else 1
        if self.device.type == "cpu":
            f = self.pool
            self._maps = cv2.convertMaps(((mx + 0.5) / f - 0.5).astype(np.float32),
                                         ((my + 0.5) / f - 0.5).astype(np.float32), cv2.CV_16SC2)
            self._maps_full = cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)
        else:
            # Normalised source coordinates (align_corners=False): the same for any resolution of the image.
            gx = (mx + 0.5) / calib.width * 2.0 - 1.0
            gy = (my + 0.5) / calib.height * 2.0 - 1.0
            self.grid = torch.from_numpy(np.stack([gx, gy], -1)[None].astype(np.float32)).to(self.device)

    def _rgb(self, img: np.ndarray, encoding: str) -> np.ndarray:
        if is_bayer(encoding):
            return cv2.cvtColor(img, _CV_BAYER[encoding[len("bayer_"):len("bayer_") + 4]])
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if encoding.startswith("bgr"):
            return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        return img[..., :3]

    def _image_cpu(self, img: np.ndarray, encoding: str) -> torch.Tensor:
        x = self._rgb(img, encoding)
        if self.pool > 1:
            x = cv2.resize(x, (x.shape[1] // self.pool, x.shape[0] // self.pool), interpolation=cv2.INTER_AREA)
        out = cv2.remap(x, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if out.dtype == np.uint16:
            out = np.round(out.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1))).float().mul_(1 / 255)

    def _to_tensor(self, img: np.ndarray, encoding: str) -> torch.Tensor:
        """(3, h, w) float [0, 1] on the device; Bayer at half resolution."""
        x = torch.from_numpy(img).to(self.device, non_blocking=True).float()
        x = x * (1.0 / (65535.0 if img.dtype == np.uint16 else 255.0))
        if is_bayer(encoding):
            pat = encoding[len("bayer_"):len("bayer_") + 4]
            q = {pat[0]: x[0::2, 0::2], pat[3]: x[1::2, 1::2]}
            g = 0.5 * (x[0::2, 1::2] + x[1::2, 0::2])
            return torch.stack([q["r"], g, q["b"]])
        if x.ndim == 2:
            return x[None].expand(3, -1, -1)
        if encoding.startswith("bgr"):
            x = x[..., [2, 1, 0]]
        return x[..., :3].permute(2, 0, 1)

    def image(self, img: np.ndarray, encoding: str) -> torch.Tensor:
        if self.device.type == "cpu":
            return self._image_cpu(img, encoding)
        x = self._to_tensor(img, encoding)
        if self.pool > 1 and not is_bayer(encoding):
            x = F.avg_pool2d(x[None], self.pool)[0]
        out = F.grid_sample(x[None].contiguous(), self.grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)[0]
        return (out.clamp_(0, 1) * 255).round_().mul_(1 / 255)

    def depth(self, depth_m: np.ndarray) -> torch.Tensor:
        """(OUT_H, OUT_W) metres, NaN = invalid (nearest neighbour: no mixing across edges).
        The depth image is registered to this camera at its full resolution."""
        d = np.ascontiguousarray(depth_m, dtype=np.float32)
        if self.device.type == "cpu":
            out = cv2.remap(d, *self._maps_full, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=float("nan"))
            return torch.from_numpy(out)
        t = torch.from_numpy(d).to(self.device)
        out = F.grid_sample(t[None, None], self.grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
        out[out <= 0] = float("nan")
        return out

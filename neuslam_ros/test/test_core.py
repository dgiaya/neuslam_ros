import types

import cv2
import numpy as np
import pytest
import torch

from neuslam_ros.core import geometry as geo
from neuslam_ros.core.camera import OUT_H, OUT_W, Calibration, Resampler, output_geometry, stereo_baseline
from neuslam_ros.core.images import depth_array, depth_to_metres, fill, to_numpy
from neuslam_ros.core.imu import ImuBuffer
from neuslam_ros.core.voxels import LocalVoxelGrid


def calib(w=1280, h=720, f=700.0, D=(0.0,) * 5, R=np.eye(3), tx=0.0, model="plumb_bob"):
    K = np.array([[f, 0, w / 2 - 3.5], [0, f * 1.01, h / 2 + 2.0], [0, 0, 1]])
    P = np.hstack([K if not any(D) else np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), [[tx], [0], [0]]])
    return Calibration(w, h, K, np.asarray(D, float), np.asarray(R, float), P, model, "cam")


def msg(**kw):
    return types.SimpleNamespace(**kw)


def test_rectified_geometry_maps_projection():
    c = calib()
    r = Resampler(c, rectified=True, device="cpu")
    X = np.array([[0.3, -0.2, 4.0], [-1.0, 0.4, 6.0]])
    src = (c.P[:, :3] @ X.T).T
    src = src[:, :2] / src[:, 2:]
    out = (r.K @ X.T).T
    out = out[:, :2] / out[:, 2:]
    mx, my = cv2.convertMaps(*r._maps_full, cv2.CV_32FC1)
    for (u, v), (su, sv) in zip(out, src):
        ui, vi = int(round(u)), int(round(v))
        assert abs(mx[vi, ui] - (su + (ui - u) / r.scale)) < 0.05
        assert abs(my[vi, ui] - (sv + (vi - v) / r.scale)) < 0.05


def test_raw_undistortion_inverts_projection():
    D = (-0.25, 0.08, 0.001, -0.0005, 0.0)
    Rr = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    c = calib(D=D, R=Rr)
    r = Resampler(c, rectified=False, device="cpu")
    mx, my = cv2.convertMaps(*r._maps_full, cv2.CV_32FC1)
    rng = np.random.default_rng(0)
    X_rect = np.column_stack([rng.uniform(-1, 1, 20), rng.uniform(-0.4, 0.4, 20), rng.uniform(3, 8, 20)])
    uv = (r.K @ X_rect.T).T
    uv = uv[:, :2] / uv[:, 2:]
    X_raw = X_rect @ Rr      # x_rect = R x_raw
    raw, _ = cv2.projectPoints(X_raw, np.zeros(3), np.zeros(3), c.K, np.asarray(D))
    for (u, v), (su, sv) in zip(uv, raw[:, 0]):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < OUT_W and 0 <= vi < OUT_H:
            du, dv = mx[vi, ui] - su, my[vi, ui] - sv
            assert np.hypot(du, dv) < 2.0


def test_pad_box_for_wide_images():
    K, pad, s, ox, oy = output_geometry(calib(w=1241, h=376), scale=0.0)
    assert pad is not None and pad[0] == 0 and pad[1] > 0 and pad[3] < OUT_H
    K, pad, s, ox, oy = output_geometry(calib(w=1241, h=376), scale=1.0)
    assert pad is None and s == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_and_gpu_paths_agree():
    c = calib(D=(-0.2, 0.05, 0.0, 0.0, 0.0))
    img = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8), (9, 9), 3)
    a = Resampler(c, rectified=False, device="cpu").image(img, "rgb8")
    b = Resampler(c, rectified=False, device="cuda").image(img, "rgb8").cpu()
    inner = (slice(None), slice(20, -20), slice(20, -20))
    assert (a[inner] - b[inner]).abs().mean() < 2.0 / 255


def test_mono_and_bayer_give_three_channels():
    c = calib(w=640, h=480)
    r = Resampler(c, rectified=True, device="cpu")
    assert r.image(np.full((480, 640), 100, np.uint8), "mono8").shape == (3, OUT_H, OUT_W)
    bayer = np.zeros((480, 640), np.uint8)
    bayer[0::2, 0::2] = 200           # red sites of RGGB
    out = r.image(bayer, "bayer_rggb8")
    centre = out[:, OUT_H // 2 - 50:OUT_H // 2 + 50, 300:400].mean(dim=(1, 2))
    assert centre[0] > 0.5 and centre[2] < 0.1


def test_stereo_baseline_checks():
    left, right = calib(), calib(tx=-700.0 * 0.12)
    assert abs(stereo_baseline(left, right) - 0.12) < 1e-9
    with pytest.raises(ValueError):
        stereo_baseline(left, calib())


def test_image_roundtrip_with_padded_rows():
    a = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    padded = np.zeros((5, 32), np.uint8)
    padded[:, :21] = a.reshape(5, 21)
    m = msg(height=5, width=7, encoding="rgb8", is_bigendian=0, step=32, data=padded.tobytes())
    assert np.array_equal(to_numpy(m), a)
    m2 = msg()
    fill(m2, a, "rgb8")
    assert np.array_equal(to_numpy(m2), a)


def test_depth_encodings():
    d = np.array([[0.0, 1.234], [np.nan, 5.0]], np.float32)
    mm = depth_array(d, "16UC1")
    assert mm.dtype == np.uint16 and mm[0, 1] == 1234 and mm[1, 0] == 0
    m = msg()
    fill(m, mm, "16UC1")
    back = depth_to_metres(m)
    assert np.isnan(back[0, 0]) and abs(back[0, 1] - 1.234) < 1e-6


def test_imu_chunk_brackets_interval():
    b = ImuBuffer()
    for i in range(100):
        t = i * 0.005
        b.add(t, (0.1 * t, 0, 0), (0, 0, 9.81))
    rows = b.chunk(0.1012, 0.2013)
    assert rows.shape[1] == 10
    assert rows[0, 0] <= 0.1012 and rows[-1, 0] >= 0.2013
    assert np.allclose(rows[1:-1, 7], 0.1, atol=1e-6)       # dw/dt
    assert b.chunk(0.4, 0.6)[-1, 0] == pytest.approx(0.6)   # held past the last sample


def test_voxel_packing_matches_nav2():
    g = LocalVoxelGrid(resolution=0.5, z_resolution=0.5, size_x=4, size_y=4, size_z=4, max_range=10)
    g.origin_idx = np.zeros(3, np.int64)
    g.known[1, 2, 3] = True                    # free
    g.known[2, 2, 3] = g.marked[2, 2, 3] = True
    col = int(g.packed()[2 * 4 + 3])
    unknown = col & 0xFFFF & ~(col >> 16)
    marked = col >> 16
    assert marked == 1 << 2
    assert unknown == (0b1111 & ~0b0110)


def test_voxel_integrate_marks_wall_and_clears_before_it():
    g = LocalVoxelGrid(resolution=0.25, z_resolution=0.25, size_x=80, size_y=80, size_z=16, max_range=15)
    K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    depth = np.full((96, 128), 4.0, np.float32)
    T = geo.make(geo.R_BODY_OPTICAL)           # camera looking along +x of the grid frame
    g.integrate(depth, K, T, stride=4)
    pts = g.marked_points()
    assert len(pts) and np.all(np.abs(pts[:, 0] - 4.0) < 0.5)
    c = (np.floor(np.array([2.0, 0.0, 0.0]) / g.res).astype(int) - g.origin_idx)
    assert g.known[c[2], c[1], c[0]] and not g.marked[c[2], c[1], c[0]]


def test_frames_and_twist():
    assert np.allclose(geo.R_BODY_OPTICAL @ [0, 0, 1], [1, 0, 0])      # optical z = forward
    assert np.allclose(geo.R_BODY_OPTICAL @ [0, 1, 0], [0, 0, -1])     # optical y = down
    R = geo.gravity_aligned(np.array([0.0, -np.sin(0.2), np.cos(0.2)]))
    assert np.allclose(R @ np.array([0.0, -np.sin(0.2), np.cos(0.2)]), [0, 0, 1])
    T0 = geo.make(np.eye(3))
    T1 = geo.make(cv2.Rodrigues(np.array([0, 0, 0.1]))[0], (1.0, 0, 0))
    lin, ang = geo.twist(T0, T1, 0.5)
    assert np.allclose(ang, [0, 0, 0.2]) and np.allclose(np.linalg.norm(lin), 2.0)


def test_anchor_starts_odom_at_base():
    T_base_cam = geo.make(geo.R_BODY_OPTICAL, (0.1, 0.0, 0.05))
    a = geo.Anchor(T_base_cam)
    assert np.allclose(a.base(np.eye(4)), np.eye(4))
    fwd = geo.make(np.eye(3), (0, 0, 1.0))      # camera moved 1 m along optical z
    assert np.allclose(a.base(fwd)[:3, 3], [1.0, 0, 0])

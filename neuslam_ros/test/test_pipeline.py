import types

import numpy as np

from neuslam.stream import Result
from neuslam_ros.core import geometry as geo
from neuslam_ros.core.pipeline import Params, Pipeline


def info(w=640, h=480, tx=0.0, frame="cam"):
    K = [500.0, 0, 320, 0, 500.0, 240, 0, 0, 1]
    return types.SimpleNamespace(width=w, height=h, k=K, d=[0.0] * 5, r=[1.0, 0, 0, 0, 1, 0, 0, 0, 1],
                                 p=[500.0, 0, 320, tx, 0, 500.0, 240, 0, 0, 0, 1, 0], distortion_model="plumb_bob",
                                 header=types.SimpleNamespace(frame_id=frame))


def image(t_ns, frame="cam"):
    s, n = divmod(t_ns, 1_000_000_000)
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=s, nanosec=n),
                                                              frame_id=frame),
                                 height=480, width=640, encoding="rgb8", is_bigendian=0, step=640 * 3,
                                 data=np.full((480, 640, 3), 128, np.uint8).tobytes())


class StubSlam:
    """Camera moves 0.1 m along optical z per frame; frame 2 revises keyframe 0 by +1 m in x."""

    def __init__(self):
        self.frames, self.out = [], []

    def submit(self, f, timeout=None):
        i = len(self.frames)
        self.frames.append(f)
        T = geo.make(np.eye(3), (0, 0, 0.1 * i))
        r = Result(frame_id=i, timestamp=f.timestamp, state="tracking", is_keyframe=i == 0, ref_frame_id=0,
                   T_world_cam=T, T_world_cam_odom=T, num_map_points=10)
        if i == 2:
            r.T_world_cam = geo.make(np.eye(3), (1.0, 0, 0.2))
            r.poses_revised, r.loops = True, [(2, 0)]
            r.keyframe_ids, r.keyframe_poses = np.array([0]), np.stack([geo.make(np.eye(3), (1.0, 0, 0))])
        if i == 1:
            r.depth, r.depth_frame_id = np.ones((368, 768), np.float32), 0
        self.out.append(r)
        return True


def test_stereo_pipeline_outputs():
    p = Pipeline(Params(mode="stereo", rectified=True))
    assert p.set_calibration(info(), info(tx=-500.0 * 0.1))
    assert not p.set_calibration(info(), info(tx=-500.0 * 0.1))
    assert abs(p.baseline - 0.1) < 1e-9
    p.set_extrinsics(geo.make(geo.R_BODY_OPTICAL))
    slam = StubSlam()
    outs = []
    for k in range(3):
        t = 1_700_000_000_000_000_000 + k * 100_000_000
        prep = p.prepare(image(t), image(t))
        assert prep.frame.left.shape == (3, 368, 768) and prep.frame.baseline == p.baseline
        assert p.submit(slam, prep)
        outs.append(p.output(slam.out[-1]))
    o0, o1, o2 = outs
    assert np.allclose(o0.T_map_odom, np.eye(4))
    assert np.allclose(o1.T_odom_base[:3, 3], [0.1, 0, 0])          # forward is base x
    assert o1.depth is not None and o1.depth[0] == o0.stamp_ns       # keyframe 0's depth, at its stamp
    assert np.allclose(o1.linear, [1.0, 0, 0])                       # 0.1 m in 0.1 s
    assert o2.loops == [(2, 0)] and o2.path is not None and len(o2.path) == 3
    assert np.allclose(o2.path[0][1][:3, 3], [0.0, -1.0, 0.0])       # keyframe 0 moved +1 m optical x = -y base
    assert np.allclose(o2.path[1][1][:3, 3], [0.1, -1.0, 0.0])       # frame 1 follows its keyframe
    assert np.allclose(o2.T_odom_base[:3, 3], [0.2, 0, 0])           # odom unchanged
    assert not np.allclose(o2.T_map_odom, np.eye(4))

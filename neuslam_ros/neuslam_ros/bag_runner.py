"""Run NeuSLAM over a rosbag offline: every frame, in order, as fast as NeuSLAM takes them.

  ros2 run neuslam_ros bag_runner --bag <bag> --mode stereo --left /cam0/image_raw --right /cam1/image_raw \\
      --out runs/<name> [--write-bag]
"""
from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from neuslam_ros.core import config as nscfg
from neuslam_ros.core import geometry as geo
from neuslam_ros.core import messages as M
from neuslam_ros.core.pipeline import Output, Params, Pipeline, stamp_ns


def _storage(uri: str) -> str:
    p = Path(uri)
    files = [p] if p.is_file() else list(p.glob("*"))
    return "mcap" if any(f.suffix == ".mcap" for f in files) else "sqlite3"


def _tum(path: Path, rows: List[tuple]) -> None:
    with open(path, "w") as f:
        for ts, T in rows:
            q = geo.quat_xyzw(T[:3, :3])
            f.write(f"{ts * 1e-9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


class Writer:
    def __init__(self, uri: Path) -> None:
        self.w = rosbag2_py.SequentialWriter()
        self.w.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
        self.topics: Dict[str, bool] = {}

    def write(self, topic: str, msg, ns: int) -> None:
        if topic not in self.topics:
            t = type(msg)
            name = f"{t.__module__.split('.')[0]}/msg/{t.__name__}"
            self.w.create_topic(rosbag2_py.TopicMetadata(id=len(self.topics), name=topic, type=name,
                                                         serialization_format="cdr"))
            self.topics[topic] = True
        self.w.write(topic, serialize_message(msg), int(ns))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--mode", default="stereo", choices=nscfg.MODES)
    ap.add_argument("--left", required=True, help="left (or only) image topic")
    ap.add_argument("--left-info", help="default: <left image namespace>/camera_info")
    ap.add_argument("--right")
    ap.add_argument("--right-info")
    ap.add_argument("--depth")
    ap.add_argument("--imu")
    ap.add_argument("--rectified", action="store_true", help="images are already rectified")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--odom-frame", default="odom")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.0, help="image scale before the crop/pad (0: to the network width)")
    ap.add_argument("--start", type=float, default=0.0, help="skip this many seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = all)")
    ap.add_argument("--config", default="", help="NeuSLAM config name, YAML, or overlay YAML")
    ap.add_argument("--override", action="append", default=[], help="NeuSLAM KEY=VALUE (repeatable)")
    ap.add_argument("--models", default="", help="default: the package's models/ (or $NEUSLAM_MODELS)")
    ap.add_argument("--backend", default="tensorrt", choices=("tensorrt", "pytorch"))
    ap.add_argument("--sequential", action="store_true", help="NeuSLAM's single-process (deterministic) pipeline")
    ap.add_argument("--depth-every", type=int, default=0, help="current-pair depth every n frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--write-bag", action="store_true", help="also write the node's output topics to <out>/output")
    a = ap.parse_args(argv)

    from neuslam.stream import StreamingSLAM
    from neuslam_ros.node import default_models_dir

    s = nscfg.sensors(a.mode)
    info_of = lambda img: img.rsplit("/", 1)[0] + "/camera_info"
    topics = {"left": a.left, "left_info": a.left_info or info_of(a.left)}
    if s["stereo"]:
        if not a.right:
            ap.error("stereo modes need --right")
        topics.update(right=a.right, right_info=a.right_info or info_of(a.right))
    if s["rgbd"]:
        if not a.depth:
            ap.error("RGB-D modes need --depth")
        topics["depth"] = a.depth
    if s["imu"]:
        if not a.imu:
            ap.error("inertial modes need --imu")
        topics["imu"] = a.imu
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    p = Params(mode=a.mode, rectified=a.rectified, frame_stride=a.stride, depth_every_n_frames=a.depth_every, scale=a.scale,
               imu_wait_s=0.0)
    pipe = Pipeline(p)
    cfg = nscfg.build(a.mode, config=a.config, overrides=a.override, models=a.models or default_models_dir(),
                      backend=a.backend)
    need = nscfg.engines_needed(cfg, current_pair_depth=a.depth_every > 0 and s["stereo"])
    missing = {k: v for k, v in need.items() if not v.is_file()}
    if missing:
        nscfg.build_engines(missing, Path(cfg["model"]["weights"]))

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id=_storage(a.bag)), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for k, t in topics.items():
        if t not in types:
            ap.error(f"topic {t} ({k}) not in the bag")
    wanted = set(topics.values()) | {"/tf_static"} & set(types)
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
    cls = {t: get_message(types[t]) for t in wanted}

    tf = Buffer()
    writer = Writer(out / "output") if a.write_bag else None
    rows_odom: List[tuple] = []
    last: Optional[Output] = None
    image_keys = [k for k in ("left", "right", "depth") if k in topics]
    sets: "OrderedDict[int, dict]" = OrderedDict()
    held: List = []
    infos: Dict[str, object] = {}
    t_first = None
    n_in = 0
    t0 = time.time()

    def on_output(o: Output) -> None:
        nonlocal last
        last = o
        rows_odom.append((o.stamp_ns, o.T_odom_base))
        if writer is None:
            return
        writer.write("/neuslam/odom", M.odometry(o, a.odom_frame, a.base_frame), o.stamp_ns)
        writer.write("/neuslam/pose", M.pose_with_cov(o, a.map_frame), o.stamp_ns)
        writer.write("/tf", TFMessage(transforms=[M.tf(o.T_odom_base, o.stamp_ns, a.odom_frame, a.base_frame),
                                                  M.tf(o.T_map_odom, o.stamp_ns, a.map_frame, a.odom_frame)]),
                     o.stamp_ns)
        writer.write("/neuslam/status", M.status(o, mode=a.mode, base_frame=a.base_frame, rate_hz=0.0, latency_ms=0.0,
                                                 dropped=0, base_link_assumed=base_assumed), o.stamp_ns)
        if o.depth is not None:
            ts, d = o.depth
            writer.write("/neuslam/depth/image", M.depth_image(d, "32FC1", ts, rect_frame), ts)
            writer.write("/neuslam/depth/camera_info", M.camera_info(pipe.K, ts, rect_frame, d.shape[1], d.shape[0]), ts)

    def drain(slam) -> None:
        for r in slam.poll():
            o = pipe.output(r)
            if o is not None:
                on_output(o)

    base_assumed, rect_frame = False, ""
    with StreamingSLAM(cfg, out_dir=str(out), multiprocess=not a.sequential, depth=True,
                       depth_every_n_frames=0 if s["rgbd"] else a.depth_every) as slam:
        def submit(msgs: dict) -> None:
            nonlocal base_assumed, rect_frame
            left = msgs["left"]
            if not pipe.calibrated or pipe.T_base_cam is None:
                pipe.set_calibration(infos["left_info"], infos.get("right_info"))
                opt = left.header.frame_id
                try:
                    T = geo.from_tf(tf.lookup_transform(a.base_frame, opt, Time()))
                except Exception:
                    T, base_assumed = geo.make(geo.R_BODY_OPTICAL), True
                    print(f"no TF {a.base_frame} -> {opt} in the bag: {a.base_frame} assumed at the left camera")
                T_opt_imu = None
                if s["imu"]:
                    imu_frame = msgs.get("imu_frame") or a.base_frame
                    try:
                        T_opt_imu = geo.from_tf(tf.lookup_transform(opt, imu_frame, Time()))
                    except Exception:
                        T_opt_imu = geo.inv(T) if imu_frame == a.base_frame else None
                    if T_opt_imu is None:
                        raise SystemExit(f"no TF {opt} -> {imu_frame}: inertial modes need the camera-IMU extrinsic")
                pipe.set_extrinsics(T, T_opt_imu)
                rect_frame = opt if a.rectified else f"{opt}_rect"
            if not pipe.take():
                return
            prep = pipe.prepare(left, msgs.get("right"), msgs.get("depth"))
            while not pipe.submit(slam, prep, timeout=0.05):
                drain(slam)
            drain(slam)

        imu_frame = None
        for topic, data, t_rec in iter(lambda: reader.read_next() if reader.has_next() else None, None):
            msg = deserialize_message(data, cls[topic])
            if topic == "/tf_static":
                for t in msg.transforms:
                    tf.set_transform_static(t, "bag")
                continue
            if topic == topics.get("imu"):
                imu_frame = msg.header.frame_id
                pipe.add_imu(msg)
                while held and pipe.imu.latest is not None and pipe.imu.latest >= stamp_ns(held[0]["left"].header) * 1e-9:
                    submit(held.pop(0))
                continue
            key = next(k for k, v in topics.items() if v == topic)
            ts = stamp_ns(msg.header)
            if key.endswith("_info"):
                infos[key] = msg
            else:
                if t_first is None:
                    t_first = ts
                if (ts - t_first) * 1e-9 < a.start:
                    continue
                if a.duration > 0 and (ts - t_first) * 1e-9 > a.start + a.duration:
                    break
                sets.setdefault(ts, {})[key] = msg
                while len(sets) > 50:
                    sets.popitem(last=False)
            if ts in sets and all(k in sets[ts] for k in image_keys) and "left_info" in infos and (
                    not s["stereo"] or "right_info" in infos):
                msgs = sets.pop(ts)
                for old in [k for k in sets if k < ts]:
                    del sets[old]
                msgs["imu_frame"] = imu_frame
                n_in += 1
                if s["imu"]:
                    held.append(msgs)
                    continue
                submit(msgs)
        for msgs in held:
            submit(msgs)
        slam.finish(save_to=str(out))
        drain(slam)

    _tum(out / "base_link_odom.tum", rows_odom)
    _tum(out / "base_link_map.tum", pipe.path)
    if writer is not None and last is not None:
        writer.write("/neuslam/path", M.path(pipe.path, a.map_frame, last.stamp_ns), last.stamp_ns)
    dt = time.time() - t0
    n = len(rows_odom)
    print(f"{n_in} frame sets, {n} poses in {dt:.1f} s ({n / max(dt, 1e-9):.1f} poses/s); "
          f"keyframes {last.num_keyframes if last else 0}, loop closures {last.num_loop_closures if last else 0}; "
          f"outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

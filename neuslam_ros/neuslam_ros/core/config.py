"""NeuSLAM configuration for a sensor mode, and the TensorRT engines it needs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from neuslam.stream import build_stream_config

MODES = ("stereo", "stereo_inertial", "rgbd", "rgbd_inertial", "mono", "mono_inertial")
ENGINES = {"full": "neuslamnet_fp16.plan", "flow": "neuslamnet_flow_fp16.plan"}


def sensors(mode: str) -> Dict[str, bool]:
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {', '.join(MODES)}")
    return {"stereo": mode.startswith("stereo"), "rgbd": mode.startswith("rgbd"),
            "mono": mode.startswith("mono"), "imu": mode.endswith("_inertial")}


def default_base(mode: str) -> str:
    return "tum_rgbd" if sensors(mode)["rgbd"] else "euroc"


def _overlay(path: Path) -> tuple[str, List[str]]:
    """A YAML with `base:` and dotted KEY: VALUE entries -> (base, overrides)."""
    ov = yaml.safe_load(path.read_text()) or {}
    base = ov.pop("base")
    return base, [f"{k}={yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}"
                  for k, v in ov.items()]


def build(mode: str, *, config: str = "", overrides: Iterable[str] = (), models: str,
          backend: str = "tensorrt", queue_size: int = 0) -> Dict[str, Any]:
    """`config`: a bundled NeuSLAM config name, a NeuSLAM YAML, or an overlay YAML (`base:` + KEY: VALUE)."""
    s = sensors(mode)
    base, extra = config or default_base(mode), []
    p = Path(os.path.expanduser(base))
    if p.suffix in (".yaml", ".yml") and p.is_file() and "base" in (yaml.safe_load(p.read_text()) or {}):
        base, extra = _overlay(p)
    cfg = build_stream_config(base, [f"model.backend={backend}"] + extra + list(overrides), models=models)
    slam = cfg["slam"]
    for section in ("imu", "mono"):
        if s[section]:
            if len(slam.get(section, {}) or {}) <= 1:   # absent or `enabled: false` only: EuRoC's settings
                slam[section] = build_stream_config("euroc", models=models)["slam"][section]
            slam[section]["enabled"] = True
        elif section in slam:
            slam[section]["enabled"] = False
    if queue_size > 0:
        slam["mp"]["queue_size"] = int(queue_size)
    slam["vis"]["rerun"]["enabled"] = False
    return cfg


def engines_needed(cfg: Dict[str, Any], *, current_pair_depth: bool) -> Dict[str, Path]:
    """Engine variant -> path that this config loads."""
    out: Dict[str, Path] = {}
    if str(cfg["model"]["backend"]).lower() != "tensorrt":
        return out
    paths = [Path(cfg["model"]["trt_engine_path"])]
    mono = cfg["slam"].get("mono", {}) or {}
    if mono.get("enabled", False):
        paths = [Path(mono["trt_engine_path"])]
    for p in paths:
        out["flow" if p.name == ENGINES["flow"] else "full"] = p
    if current_pair_depth and "full" not in out:
        out["full"] = paths[0].with_name(ENGINES["full"])
    return out


def build_engines(missing: Dict[str, Path], weights: Path, *, workspace_mb: int = 16384, log=print) -> None:
    import tempfile

    from neuslam.models.export import build_engine, export_onnx
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    with tempfile.TemporaryDirectory() as tmp:
        for variant, plan in missing.items():
            log(f"building TensorRT engine {plan.name} (one-off, several minutes)")
            onnx = export_onnx(weights, Path(tmp) / f"{variant}.onnx", flow_only=(variant == "flow"))
            build_engine(onnx, plan, fp16=True, workspace_mb=workspace_mb)


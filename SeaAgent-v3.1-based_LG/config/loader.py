"""读取 SeaAgent 分层配置并生成视频流水线运行参数。"""
from __future__ import annotations
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

_CONFIG_FILES = ("app.yaml", "yolo.yaml", "pipeline.yaml", "prompts.yaml", "runtime.yaml")
_ROOT = Path(__file__).resolve().parent.parent

def project_root() -> Path:
    return _ROOT

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是对象：{path}")
    return data

def _resolve_paths(config: dict[str, Any]) -> None:
    for key, value in list(config.setdefault("paths", {}).items()):
        path = Path(str(value)).expanduser()
        config["paths"][key] = str(path if path.is_absolute() else (_ROOT / path).resolve())

def _build_runtime_config(config: dict[str, Any]) -> None:
    pipeline = config.setdefault("pipeline", {})
    yolo = config.setdefault("yolo", {})
    # 兼容旧 runtime.yaml，但不再保留已移除的独立跟踪候选阈值。
    yolo.pop("tracking_candidate_confidence", None)
    pipeline.pop("tracking_candidate_confidence", None)
    mapping = {
        "yolo_model": yolo.get("model"), "device": yolo.get("device"),
        "conf_threshold": yolo.get("confidence"),
        "iou_threshold": yolo.get("iou"),
        "detect_classes": yolo.get("classes"), "detect_every_n_frames": yolo.get("detect_every_n_frames"),
        "tracker": yolo.get("tracker"), "tracker_params": yolo.get("tracker_params"),
        "appearance_tracking": yolo.get("appearance_tracking"),
    }
    for key, value in mapping.items():
        if value is not None:
            pipeline.setdefault(key, value)
    paths = config["paths"]
    config.setdefault("demo_video", {})
    config["demo_video"].setdefault("dir", paths["video_dir"])
    config["demo_video"].setdefault("output_dir", str((_ROOT / "output").resolve()))
    config["demo_video"].setdefault("allowed_extensions", [".mp4", ".avi", ".mkv", ".mov", ".webm"])
    config["demo_video"].setdefault("max_file_size_mb", 1024)

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """合并 config 目录配置，可额外传入覆盖文件。"""
    config_dir = Path(os.getenv("SEAAGENT_CONFIG_DIR", Path(__file__).resolve().parent))
    merged: dict[str, Any] = {}
    for name in _CONFIG_FILES:
        merged = _merge(merged, _read_yaml(config_dir / name))
    if config_path:
        merged = _merge(merged, _read_yaml(Path(config_path)))
    _resolve_paths(merged)
    _build_runtime_config(merged)
    return merged

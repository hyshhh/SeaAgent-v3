"""读取 SeaAgent 分层配置并生成视频流水线运行参数。"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_FILES = ("app.yaml", "yolo.yaml", "pipeline.yaml", "prompts.yaml", "harness.yaml", "tools.yaml")
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
        raise TypeError(f"配置文件顶层必须是对象：{path}")
    return data

def _resolve_paths(config: dict[str, Any]) -> None:
    """把相对路径按项目根展开；已经是绝对路径的原样保留。

    注意 ``/media/...`` 这类 POSIX 绝对路径在 Windows 上并不是绝对路径（没有盘符），
    pathlib 会把它挂到当前盘符下，于是数据盘上的先验库会被解析成 ``<当前盘>:\\media\\...``
    并被当成空库新建。这里出声提醒，避免换机器调试时对着空先验库找原因。
    """
    for key, value in list(config.setdefault("paths", {}).items()):
        raw = str(value)
        path = Path(raw).expanduser()
        if os.name == "nt" and raw.startswith("/") and not path.is_absolute():
            logger.warning("配置项 %s=%s 是 POSIX 绝对路径，Windows 下会被解析到当前盘符：%s", key, raw, (_ROOT / path))
        config["paths"][key] = str(path if path.is_absolute() else (_ROOT / path).resolve())

def _build_runtime_config(config: dict[str, Any]) -> None:
    pipeline = config.setdefault("pipeline", {})
    yolo = config.setdefault("yolo", {})
    # 保持运行参数由分层配置生成，不在业务代码中复制配置规则。
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
    merged = _merge(merged, _read_yaml(config_dir / "runtime.yaml"))
    if config_path:
        merged = _merge(merged, _read_yaml(Path(config_path)))
    _resolve_paths(merged)
    _build_runtime_config(merged)
    return merged

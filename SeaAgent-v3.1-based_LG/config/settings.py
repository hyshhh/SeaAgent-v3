"""运行时可调参数的读取、校验和持久化。"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml

from .loader import load_config, project_root


DETECTION_MODEL_OPTIONS: tuple[str, ...] = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
)

PROMPT_SETTING_KEYS: tuple[str, ...] = (
    "single_frame_recognition",
    "verify_description",
    "verify_registry",
    "text_retrieval_instruction",
)

PROMPT_SETTING_SPECS: dict[str, dict[str, Any]] = {
    f"prompts.{key}": {
        "type": "text",
        "multiline": key != "text_retrieval_instruction",
        "min_chars": 1,
        "max_chars": 8000,
        "label": {
            "single_frame_recognition": "单帧识别提示词",
            "verify_description": "描述核验提示词",
            "verify_registry": "库项核验提示词",
            "text_retrieval_instruction": "文本检索指令",
        }.get(key, key),
        "help": {
            "single_frame_recognition": "单帧舷号可读性与外观描述识别。",
            "verify_description": "灰区时核验轨迹图像是否符合文字描述。",
            "verify_registry": "灰区时比较库参考图与轨迹图像，判断是否为同一艘船。",
            "text_retrieval_instruction": "文本向量检索时使用的英文指令。",
        }.get(key, ""),
    }
    for key in PROMPT_SETTING_KEYS
}

SETTING_SPECS: dict[str, dict[str, Any]] = {

    "yolo.model": {"type": "enum", "choices": list(DETECTION_MODEL_OPTIONS)},
    "yolo.device": {"type": "string", "allow_empty": True, "max_chars": 128},
    "yolo.confidence": {"type": "float", "min": 0.01, "max": 0.99, "step": 0.01},
    "yolo.iou": {"type": "float", "min": 0.01, "max": 0.99, "step": 0.01},
    "yolo.detect_every_n_frames": {"type": "int", "min": 1, "max": 100, "step": 1},
    "pipeline.target_fps": {"type": "float", "min": 0.0, "max": 240.0, "step": 1.0},
    "pipeline.pipe_scale": {"type": "float", "min": 0.05, "max": 1.0, "step": 0.05},
    "pipeline.max_frames": {"type": "int", "min": 0, "max": 100000000, "step": 1},
    "pipeline.save_output_video": {"type": "bool"},
    "yolo.tracker_params.track_high_thresh": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
    "yolo.tracker_params.track_low_thresh": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
    "yolo.tracker_params.new_track_thresh": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
    "yolo.tracker_params.track_buffer": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "yolo.tracker_params.match_thresh": {"type": "float", "min": 0.01, "max": 1.0, "step": 0.01},
    "yolo.appearance_tracking.enabled": {"type": "bool"},
    "yolo.appearance_tracking.appearance_thresh": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
    "yolo.appearance_tracking.proximity_thresh": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
    "pipeline.max_stale_frames": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "pipeline.candidate_every_n_frames": {"type": "int", "min": 1, "max": 1000, "step": 1},
    "pipeline.candidate_pool_size": {"type": "int", "min": 1, "max": 100, "step": 1},
    "pipeline.keyframe_pool_size": {"type": "int", "min": 1, "max": 6, "step": 1},
    "pipeline.quality.min_score": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.recognition_timeout_seconds": {"type": "int", "min": 1, "max": 600, "step": 1},
    "pipeline.retention.replace_margin": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.aggregation.null_weight": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.aggregation.conflict_margin": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.aggregation.confirmed_support": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.aggregation.confirmed_count": {"type": "int", "min": 1, "max": 6, "step": 1},
    "pipeline.aggregation.null_margin": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.01},
    "pipeline.retrieval.top_k": {"type": "int", "min": 1, "max": 20, "step": 1},
    "pipeline.retrieval.broad_match_top_k": {"type": "int", "min": 0, "max": 100000, "step": 1},
    "pipeline.retrieval.text_match": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.text_exclude": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.image_match": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.image_exclude": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.dedup_high": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.retrieval.dedup_low": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "pipeline.evidence.clip_width": {"type": "int", "min": 320, "max": 1280, "step": 16},
    "pipeline.evidence.clip_height": {"type": "int", "min": 180, "max": 720, "step": 2},
    "pipeline.evidence.clip_fps": {"type": "int", "min": 1, "max": 25, "step": 1},
    "pipeline.evidence.clip_crf": {"type": "int", "min": 18, "max": 40, "step": 1},
    "pipeline.evidence.poster_quality": {"type": "int", "min": 40, "max": 95, "step": 1},
    "pipeline.agent.max_rounds": {"type": "int", "min": 1, "max": 10, "step": 1},
}


RUNTIME_FILE = Path(os.getenv("SEAAGENT_CONFIG_DIR", project_root() / "config")) / "runtime.yaml"


def _get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten(value, path))
        else:
            values[path] = value
    return values


def _read_runtime() -> dict[str, Any]:
    if not RUNTIME_FILE.exists():
        return {}
    content = yaml.safe_load(RUNTIME_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        return {}
    # 清理旧版本遗留的独立跟踪候选阈值，避免再次写回运行时配置。
    for section in ("yolo", "pipeline"):
        if isinstance(content.get(section), dict):
            content[section].pop("tracking_candidate_confidence", None)
    return content


def _write_runtime(data: dict[str, Any]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUNTIME_FILE.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(RUNTIME_FILE)


def _coerce(path: str, value: Any) -> int | float | str | bool:
    # numeric specs first; prompt specs second
    if path in SETTING_SPECS:
        spec = SETTING_SPECS[path]
        if spec["type"] == "bool":
            if isinstance(value, bool):
                return value
            text_value = str(value if value is not None else "").strip().lower()
            if text_value in {"1", "true", "yes", "on"}:
                return True
            if text_value in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"参数 {path} 必须是布尔值")
        if spec["type"] == "string":
            text_value = str(value if value is not None else "").strip()
            if not text_value and not spec.get("allow_empty", False):
                raise ValueError(f"参数 {path} 不能为空")
            if len(text_value) > int(spec.get("max_chars", 1024)):
                raise ValueError(f"参数 {path} 过长")
            return text_value
        if spec["type"] == "enum":
            text_value = str(value if value is not None else "").strip().lower()
            choices = [str(item).lower() for item in spec.get("choices", [])]
            if text_value not in choices:
                raise ValueError(f"参数 {path} 必须是 {', '.join(spec.get('choices', []))}")
            return text_value
        try:
            number = int(value) if spec["type"] == "int" else float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"参数 {path} 必须是数字") from error
        if not spec["min"] <= number <= spec["max"]:
            raise ValueError(f"参数 {path} 必须在 {spec['min']} 到 {spec['max']} 之间")
        return number
    if path in PROMPT_SETTING_SPECS:
        spec = PROMPT_SETTING_SPECS[path]
        text = str(value if value is not None else "").strip()
        if len(text) < int(spec.get("min_chars", 1)):
            raise ValueError(f"提示词 {path} 不能为空")
        if len(text) > int(spec.get("max_chars", 8000)):
            raise ValueError(f"提示词 {path} 过长，最多 {spec['max_chars']} 字符")
        return text
    raise ValueError(f"未知参数：{path}")


def _validate_relations(config: dict[str, Any]) -> None:
    yolo = config["yolo"]
    tracker = yolo["tracker_params"]
    if tracker["track_low_thresh"] > tracker["track_high_thresh"]:
        raise ValueError("跟踪低分阈值不能高于跟踪高分阈值")
    if tracker["new_track_thresh"] < tracker["track_high_thresh"]:
        raise ValueError("新建轨迹阈值不能低于跟踪高分阈值")
    retrieval = config["pipeline"]["retrieval"]
    if retrieval["text_exclude"] >= retrieval["text_match"]:
        raise ValueError("文本排除阈值必须小于文本匹配阈值")
    if retrieval["image_exclude"] >= retrieval["image_match"]:
        raise ValueError("图像排除阈值必须小于图像匹配阈值")
    if retrieval["dedup_low"] >= retrieval["dedup_high"]:
        raise ValueError("去重低阈值必须小于去重高阈值")


def public_settings(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in SETTING_SPECS:
        value = _get(config, path)
        _set(values, path, value)
    for path in PROMPT_SETTING_SPECS:
        _set(values, path, _get(config, path) or "")
    specs = deepcopy(SETTING_SPECS)
    specs.update(deepcopy(PROMPT_SETTING_SPECS))
    return {
        "settings": values,
        "specs": specs,
        "promptKeys": list(PROMPT_SETTING_KEYS),
    }


def update_settings(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    submitted = payload.get("settings", payload)
    if not isinstance(submitted, dict):
        raise ValueError("设置内容格式不正确")
    allowed = set(SETTING_SPECS) | set(PROMPT_SETTING_SPECS)
    changes = {path: _coerce(path, value) for path, value in _flatten(submitted).items() if path in allowed}
    if not changes:
        raise ValueError("没有收到可修改的参数")
    updated = deepcopy(config)
    for path, value in changes.items():
        _set(updated, path, value)
    _validate_relations(updated)
    runtime = _read_runtime()
    for path, value in changes.items():
        _set(runtime, path, value)
    _write_runtime(runtime)
    for path, value in changes.items():
        _set(config, path, value)
    # 同步到已创建的 LLM 服务提示词缓存
    return public_settings(config)


def reset_settings(config: dict[str, Any]) -> dict[str, Any]:
    if RUNTIME_FILE.exists():
        RUNTIME_FILE.unlink()
    defaults = load_config()
    for path in SETTING_SPECS:
        _set(config, path, _get(defaults, path))
    for path in PROMPT_SETTING_SPECS:
        _set(config, path, _get(defaults, path) or "")
    return public_settings(config)

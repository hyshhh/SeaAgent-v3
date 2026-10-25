"""候选帧质量评分与准入判定。"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np

# 清晰度归一化参考值。检测器已把候选裁剪缩放到 256~512px，这个尺度下
# 拉普拉斯方差实测 p10≈930、p50≈1290、p90≈1460；旧值 180 会让 clarity
# 恒为 1.0，质量分失去区分度（实测 450 个候选的 qualityScore 全落在
# 0.83~0.89）。取 1400 后 clarity 覆盖约 0.66~1.0，评分才有意义。
DEFAULT_SHARPNESS_REFERENCE = 1400.0

# 各分项权重，合计 1.0
_SCORE_WEIGHTS = {"clarity": 0.40, "completeness": 0.25, "exposure": 0.20, "scale": 0.15}


def score_frame(crop: np.ndarray, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...], config: dict[str, Any]) -> dict[str, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    reference = float(config.get("sharpness_reference", DEFAULT_SHARPNESS_REFERENCE))
    clarity = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / max(1e-6, reference))
    touches = sum((x1 <= 1, y1 <= 1, x2 >= width - 1, y2 >= height - 1))
    completeness = max(0.0, 1.0 - 0.2 * touches)
    dark = float(np.mean(gray < 25))
    bright = float(np.mean(gray > 230))
    exposure = max(0.0, 1.0 - dark - bright)
    relative_size = float(np.sqrt(max(0, (x2 - x1) * (y2 - y1)) / max(1, width * height)))
    size_min, size_max = float(config.get("size_min", 0.04)), float(config.get("size_max", 0.35))
    scale = float(np.clip((relative_size - size_min) / max(1e-6, size_max - size_min), 0, 1))
    parts = {"clarity": clarity, "completeness": completeness, "exposure": exposure, "scale": scale}
    score = sum(_SCORE_WEIGHTS[key] * value for key, value in parts.items())
    return {"qualityScore": round(score, 6), "clarity": clarity, "completeness": completeness, "exposure": exposure, "scale": scale}


def quality_gate(metrics: dict[str, float], config: dict[str, Any]) -> str | None:
    """质量准入判定：通过返回 None，拒绝返回原因。

    旧逻辑只在正式池已满时才检查 min_score；池子空着的时候任何模糊、过曝、
    贴边的裁剪都能进候选并直接成为关键帧——质量约束形同虚设。这里改成无条件
    判定，并且除了加权总分，还给每个分项设硬下限，避免"总分够但某一项崩了"
    （例如很清晰但严重过曝）的帧混进来。
    """
    min_score = float(config.get("min_score", 0.2))
    if metrics["qualityScore"] < min_score:
        return f"quality_score<{min_score:g}"
    floors = (
        ("clarity", "min_clarity", "blurry"),
        ("completeness", "min_completeness", "clipped"),
        ("exposure", "min_exposure", "exposure"),
        ("scale", "min_scale", "too_small"),
    )
    for metric_key, config_key, reason in floors:
        floor = float(config.get(config_key, 0.0))
        if floor > 0 and metrics[metric_key] < floor:
            return f"{reason}<{floor:g}"
    return None

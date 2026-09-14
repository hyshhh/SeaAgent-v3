"""候选帧质量评分。"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np

def score_frame(crop: np.ndarray, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...], config: dict[str, Any]) -> dict[str, float]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clarity = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / float(config.get("sharpness_reference", 180.0)))
    touches = sum((x1 <= 1, y1 <= 1, x2 >= width - 1, y2 >= height - 1))
    completeness = max(0.0, 1.0 - 0.2 * touches)
    dark = float(np.mean(gray < 25))
    bright = float(np.mean(gray > 230))
    exposure = max(0.0, 1.0 - dark - bright)
    relative_size = float(np.sqrt(max(0, (x2 - x1) * (y2 - y1)) / max(1, width * height)))
    size_min, size_max = float(config.get("size_min", 0.04)), float(config.get("size_max", 0.35))
    scale = float(np.clip((relative_size - size_min) / max(1e-6, size_max - size_min), 0, 1))
    score = 0.40 * clarity + 0.25 * completeness + 0.20 * exposure + 0.15 * scale
    return {"qualityScore": round(score, 6), "clarity": clarity, "completeness": completeness, "exposure": exposure, "scale": scale}

"""视频监控演示渲染：检测框、轨迹编号、轨迹级舷号状态和运行信息。"""
from __future__ import annotations
import logging
from typing import Any
import cv2
import numpy as np
logger = logging.getLogger(__name__)

class DemoRenderer:
    """在视频帧上叠加轨迹记忆状态。"""
    STATUS_COLORS = {
        "confirmed": (0, 190, 0),
        "candidate": (0, 215, 255),
        "conflict": (180, 0, 220),
        "unknown": (80, 80, 230),
    }

    def __init__(self, show_fps: bool = True, show_track_id: bool = True, font_scale: float = 0.5):
        self._show_fps = show_fps
        self._show_track_id = show_track_id
        self._font_scale = font_scale

    def render(self, frame: np.ndarray, detections: list[Any], tracks: dict[int, Any], fps_info: dict[str, float] | None = None, frame_id: int = 0, queue_depth: int = 0, max_queue: int = 0) -> np.ndarray:
        canvas = frame.copy()
        for detection in detections:
            self._render_detection(canvas, detection, tracks.get(detection.track_id))
        if self._show_fps and fps_info:
            self._render_hud(canvas, fps_info, frame_id, queue_depth, max_queue)
        return canvas

    def _render_detection(self, canvas: np.ndarray, detection: Any, track_info: Any) -> None:
        x1, y1, x2, y2 = detection.bbox
        status = getattr(track_info, "final_match_type", "unknown") if track_info else "unknown"
        if track_info and getattr(track_info, "pending", False) and not getattr(track_info, "recognized", False):
            color = (255, 255, 0)
        elif track_info and getattr(track_info, "recognized", False):
            color = self.STATUS_COLORS.get(status, self.STATUS_COLORS["unknown"])
        else:
            color = (180, 180, 180)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if self._show_track_id:
            cv2.putText(canvas, f"ID:{detection.track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, color, 1)
        text = self._get_display_text(track_info)
        if text:
            self._render_label(canvas, text, x1, y2, color)

    @staticmethod
    def _get_display_text(track_info: Any) -> str:
        if not track_info:
            return ""
        if not getattr(track_info, "recognized", False):
            return "(recognizing...)" if getattr(track_info, "pending", False) else ""
        status = getattr(track_info, "final_match_type", "unknown") or "unknown"
        hull_number = getattr(track_info, "hull_number", "") or "none"
        return f"({status}: {hull_number})"

    def _render_label(self, canvas: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        width = max(100, len(text) * 9)
        cv2.rectangle(canvas, (x, y + 2), (x + width + 6, y + 22), color, -1)
        cv2.putText(canvas, text, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    @staticmethod
    def _render_hud(canvas: np.ndarray, fps_info: dict[str, float], frame_id: int, queue_depth: int, max_queue: int) -> None:
        lines = [f"Frame: {frame_id}", *[f"{name}: {fps:.1f} FPS" for name, fps in fps_info.items()]]
        if max_queue > 0:
            lines.append(f"Queue: {queue_depth}/{max_queue}")
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 18 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
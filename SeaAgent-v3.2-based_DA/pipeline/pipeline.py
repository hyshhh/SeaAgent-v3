"""SeaAgent 视频检测、轨迹记忆构建与实时推流主流水线。"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from config import load_config
from pipeline.demo import DemoRenderer
from pipeline.detector import Detection, ShipDetector
from pipeline.fps import FPSMeter, LatencyMeter
from pipeline.track_memory_builder import TrackMemoryBuilder
from pipeline.video_input import InputSource

logger = logging.getLogger(__name__)


class ShipPipeline:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        pool_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._config = config or load_config()
        settings = self._config["pipeline"]
        self._target_fps = float(settings.get("target_fps", 0))
        self._monitor_start_time = float(settings.get("monitor_start_time", 0))
        self._detect_every_n = max(1, int(settings.get("detect_every_n_frames", 1)))
        self._demo_enabled = bool(settings.get("demo", True))
        self._save_output_video = bool(settings.get("save_output_video", True))
        pipe_output_size = settings.get("pipe_output_size") or settings.get("output_size")
        self._pipe_output_size = tuple(pipe_output_size) if pipe_output_size else None
        self._stop_file = Path(settings["stop_file"]) if settings.get("stop_file") else None
        self._raw_stdout = bool(settings.get("raw_stdout", False))
        self._no_output = bool(settings.get("no_output", False))
        self._detector = ShipDetector(
            model_path=settings.get("yolo_model", "yolov8n.pt"),
            device=settings.get("device", ""),
            conf_threshold=float(settings.get("conf_threshold", 0.5)),
            iou_threshold=float(settings.get("iou_threshold", 0.5)),
            tracker_type=settings.get("tracker", "bytetrack"),
            tracker_params=settings.get("tracker_params"),
            appearance_tracking=settings.get("appearance_tracking"),
            classes=settings.get("detect_classes", [8]),
        )
        self._memory = TrackMemoryBuilder(self._config, pool_event_callback=pool_event_callback)
        self._renderer = DemoRenderer(show_fps=True, show_track_id=True)
        self._fps = FPSMeter(window_seconds=10.0)
        self._latency = LatencyMeter(window_seconds=10.0)

    def process(
        self,
        source: str | int | object,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
        stream_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """处理单个输入源；内部复用播放列表实现，保持原有调用兼容。"""
        return self.process_playlist(
            sources=[source],
            output_path=output_path,
            display=display,
            max_frames=max_frames,
            frame_callback=frame_callback,
            stream_dir=stream_dir,
            segment_gap_seconds=0,
            failure_policy="stop",
        )

    def process_playlist(
        self,
        sources: Sequence[str | int | object],
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
        stream_dir: str | Path | None = None,
        segment_gap_seconds: float = 0,
        failure_policy: str = "skip",
        segment_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """在一个流水线实例中顺序处理多个视频，模型与识别线程池只初始化一次。"""
        source_list = list(sources)
        if not source_list:
            raise ValueError("播放列表不能为空")
        failure_policy = "stop" if str(failure_policy).lower() == "stop" else "skip"
        gap_seconds = max(0.0, float(segment_gap_seconds or 0))
        stop_file = Path(stream_dir) / "__STOP__" if stream_dir else self._stop_file
        stream_path = Path(stream_dir) if stream_dir else None
        if stream_path:
            stream_path.mkdir(parents=True, exist_ok=True)

        if self._save_output_video and not output_path and not self._no_output:
            output_dir = Path(self._config["demo_video"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"seaagent_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

        writer: cv2.VideoWriter | None = None
        writer_size: tuple[int, int] | None = None
        total_frames = processed = detections_total = 0
        seen_track_ids: set[str] = set()
        segment_results: list[dict[str, Any]] = []
        timeline_offset = 0.0
        started_at = time.time()
        stream_every = max(1, int(self._config["pipeline"].get("stream_write_every_n_frames", 2)))
        stream_jpeg_quality = int(self._config["pipeline"].get("stream_jpeg_quality", 65))
        memory_closed = False
        stopped = False
        frame_limit_reached = False

        def emit_segment(index: int, source: object, status: str, **extra: Any) -> None:
            if segment_callback:
                segment_callback({
                    "index": index,
                    "total": len(source_list),
                    "source": str(source),
                    "status": status,
                    **extra,
                })

        try:
            for segment_index, source in enumerate(source_list):
                if stopped or frame_limit_reached:
                    reason = "用户停止" if stopped else "已达到最大处理帧数"
                    result = {"index": segment_index, "source": str(source), "status": "skipped", "reason": reason}
                    segment_results.append(result)
                    emit_segment(segment_index, source, "skipped", reason=reason)
                    continue

                emit_segment(segment_index, source, "running")
                input_source: InputSource | None = None
                local_frames = local_processed = local_detections = 0
                local_track_ids: set[str] = set()
                segment_started_at = time.time()
                source_path = str(source) if isinstance(source, (str, Path)) else ""

                try:
                    input_source = InputSource(source)
                    source_fps = float(input_source.source_fps or 25)
                    target_fps = self._target_fps or source_fps
                    skip_interval = max(1, round(source_fps / target_fps)) if target_fps and source_fps > target_fps else 1

                    if output_path and not self._no_output and writer is None:
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        writer_size = (input_source.width, input_source.height)
                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source_fps,
                            writer_size,
                        )
                        if not writer.isOpened():
                            writer = None
                            writer_size = None
                            logger.warning("无法创建输出视频：%s", output_path)

                    last_detections: list[Detection] = []
                    while True:
                        if stop_file and stop_file.exists():
                            stopped = True
                            break
                        ok, frame = input_source.read()
                        if not ok:
                            break
                        local_frames += 1
                        total_frames += 1
                        if max_frames and total_frames > max_frames:
                            total_frames -= 1
                            local_frames -= 1
                            frame_limit_reached = True
                            break
                        if skip_interval > 1 and local_frames % skip_interval != 1:
                            continue

                        processed += 1
                        local_processed += 1
                        self._fps.tick("stream")
                        if local_processed % self._detect_every_n == 0:
                            with self._latency.measure("yolo"):
                                last_detections = self._detector.detect(frame, total_frames)
                            timeline_timestamp = timeline_offset + local_frames / source_fps
                            self._memory.observe(
                                frame,
                                last_detections,
                                total_frames,
                                timeline_timestamp,
                                source_path,
                                source_fps,
                            )
                            detected_ids = {str(item.track_id) for item in last_detections}
                            seen_track_ids.update(f"{segment_index}:{track_id}" for track_id in detected_ids)
                            local_track_ids.update(detected_ids)
                            detections_total += len(last_detections)
                            local_detections += len(last_detections)

                        needs_render = self._demo_enabled or writer or display or stream_path or self._raw_stdout
                        display_frame = self._renderer.render(
                            frame,
                            last_detections,
                            self._memory.display_tracks(),
                            self._fps.get_all_fps(),
                            total_frames,
                        ) if needs_render else frame

                        if writer and writer_size:
                            writer_frame = display_frame
                            if (writer_frame.shape[1], writer_frame.shape[0]) != writer_size:
                                writer_frame = cv2.resize(writer_frame, writer_size, interpolation=cv2.INTER_AREA)
                            writer.write(writer_frame)
                        if stream_path and local_processed % stream_every == 0:
                            temp = stream_path / "latest.tmp.jpg"
                            cv2.imwrite(str(temp), display_frame, [cv2.IMWRITE_JPEG_QUALITY, stream_jpeg_quality])
                            temp.replace(stream_path / "latest.jpg")
                        if self._raw_stdout:
                            output_frame = display_frame
                            if self._pipe_output_size and (output_frame.shape[1], output_frame.shape[0]) != self._pipe_output_size:
                                output_frame = cv2.resize(output_frame, self._pipe_output_size, interpolation=cv2.INTER_AREA)
                            sys.stdout.buffer.write(np.ascontiguousarray(output_frame).tobytes())
                            sys.stdout.buffer.flush()
                        if frame_callback:
                            frame_callback(display_frame, total_frames)
                        if display:
                            cv2.imshow("SeaAgent", display_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stopped = True
                                break

                    # 视频边界完成当前活跃轨迹，但保留模型和识别线程池供下一段继续使用。
                    self._memory.finalize_active()
                    self._detector.reset_tracking()
                    duration = local_frames / source_fps
                    segment_result = {
                        "index": segment_index,
                        "source": str(source),
                        "status": "completed",
                        "total_frames": local_frames,
                        "processed_frames": local_processed,
                        "total_detections": local_detections,
                        "total_tracks": len(local_track_ids),
                        "video_duration_seconds": round(duration, 6),
                        "timeline_start_seconds": round(timeline_offset, 6),
                        "timeline_end_seconds": round(timeline_offset + duration, 6),
                        "elapsed_seconds": round(max(0.0, time.time() - segment_started_at), 2),
                    }
                    segment_results.append(segment_result)
                    emit_segment(segment_index, source, "completed", summary=segment_result)
                    timeline_offset += duration
                    if segment_index < len(source_list) - 1:
                        timeline_offset += gap_seconds
                except Exception as error:
                    logger.exception("视频片段处理失败：%s", source)
                    try:
                        self._memory.finalize_active()
                    except Exception as finalize_error:
                        logger.warning("失败片段轨迹收尾失败：%s", finalize_error)
                    self._detector.reset_tracking()
                    result = {
                        "index": segment_index,
                        "source": str(source),
                        "status": "failed",
                        "reason": str(error),
                    }
                    segment_results.append(result)
                    emit_segment(segment_index, source, "failed", reason=str(error))
                    if failure_policy == "stop":
                        raise
                    if segment_index < len(source_list) - 1:
                        timeline_offset += gap_seconds
                finally:
                    if input_source is not None:
                        input_source.release()

            self._memory.finalize_all()
            memory_closed = True
            elapsed = max(1e-6, time.time() - started_at)
            monitor_start_time = self._monitor_start_time or started_at
            monitor_end_time = monitor_start_time + timeline_offset if self._monitor_start_time else time.time()
            completed_segments = sum(1 for item in segment_results if item.get("status") == "completed")
            failed_segments = sum(1 for item in segment_results if item.get("status") == "failed")
            return {
                "total_frames": total_frames,
                "processed_frames": processed,
                "total_detections": detections_total,
                "total_tracks": len(seen_track_ids),
                "video_duration_seconds": round(timeline_offset, 6),
                "monitor_start_time": monitor_start_time,
                "monitor_end_time": monitor_end_time,
                "elapsed_seconds": round(elapsed, 2),
                "avg_fps": round(processed / elapsed, 2),
                "output_path": output_path or "",
                "playlist_total": len(source_list),
                "playlist_completed": completed_segments,
                "playlist_failed": failed_segments,
                "segments": segment_results,
                "stopped": stopped,
            }
        finally:
            if not memory_closed:
                try:
                    self._memory.finalize_all()
                except Exception as error:
                    logger.warning("轨迹记忆收尾失败：%s", error)
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            self._detector.cleanup()

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        return list(self._memory.trace)

    def set_demo(self, enabled: bool) -> None:
        self._demo_enabled = enabled

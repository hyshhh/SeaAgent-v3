"""感知工作记忆到轨迹记忆的双池构建器。"""
from __future__ import annotations
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import cv2
import numpy as np
from config import load_config
from memory import MemoryRepository, TrackMemoryManager
from pipeline.aggregation import aggregate_keyframes
from pipeline.quality import score_frame
from services import AgentLLMService, QwenMultimodalEmbedder
from vector_store import VectorCatalog, stable_vector_id

logger = logging.getLogger(__name__)
_POOL_EVENT_PREFIX = "__POOL_EVENT__:"

@dataclass
class CandidateFrame:
    candidate_id: str
    track_id: str
    timestamp: float
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    quality_score: float
    future: Future | None = None

@dataclass
class ActiveTrack:
    track_id: str
    start_time: float
    last_time: float
    video_start_time: float
    video_last_time: float
    latest_bbox: tuple[int, int, int, int]
    missed_frames: int = 0
    candidate_frames: dict[str, CandidateFrame] = field(default_factory=dict)
    keyframe_pool: list[dict[str, Any]] = field(default_factory=list)
    pool_revision: int = 0
    aggregated_revision: int = -1
    final_hull_number: str | None = None
    final_description: str = ""
    final_match_type: str = "unknown"
    bbox_history: list[dict[str, Any]] = field(default_factory=list)

class TrackMemoryBuilder:
    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None, pool_event_callback: Callable[[dict[str, Any]], None] | None = None):
        self.config = config or load_config()
        self.settings = self.config["pipeline"]
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.memory_manager = TrackMemoryManager(self.config, self.repository, self.vectors)
        self.active: dict[str, ActiveTrack] = {}
        self._executor = ThreadPoolExecutor(max_workers=int(self.settings.get("recognition_workers", 2)), thread_name_prefix="frame-recognition")
        self._lock = threading.RLock()
        self._source_path = ""
        self._source_fps = 25.0
        self._frame_size = [0, 0]
        self._monitor_start_time = float(self.settings.get("monitor_start_time", 0))
        self._last_maintenance_time = -1.0
        self.trace: list[dict[str, Any]] = []
        self._pool_event_callback = pool_event_callback
        self._next_track_id_value = self._load_next_track_id()

    def _load_next_track_id(self) -> int:
        values = [int(row["track_id"]) for row in self.repository.tracks.rows() if str(row.get("track_id") or "").isdigit()]
        return max(values, default=0) + 1

    def _allocate_track_id(self) -> str:
        track_id = str(self._next_track_id_value)
        self._next_track_id_value += 1
        return track_id

    def _emit_pool_event(self, pool: str, action: str, **data: Any) -> None:
        event = {"pool": pool, "action": action, **data}
        try:
            if self._pool_event_callback:
                self._pool_event_callback(event)
                return
            print(f"{_POOL_EVENT_PREFIX}{json.dumps(event, ensure_ascii=False)}", file=sys.stderr, flush=True)
        except Exception as error:
            logger.warning("池状态事件发送失败：%s", error)

    def _emit_track_status(self, state: ActiveTrack, timestamp: float, force: bool = False, remove: bool = False) -> None:
        """发送活跃轨迹快照，供监控页独立展示当前追踪状态。"""
        record_id = f"track-{state.track_id}"
        if remove:
            self._emit_pool_event("track", "remove", recordId=record_id)
            return
        last_emitted = getattr(state, "status_emitted_at", -1.0)
        if not force and timestamp - last_emitted < 0.5:
            return
        state.status_emitted_at = timestamp
        limit = int(self.settings.get("keyframe_pool_size", 6))
        latest = state.keyframe_pool[0] if state.keyframe_pool else {}
        status = f"暂时丢失（{state.missed_frames} 帧）" if state.missed_frames else "正在追踪"
        self._emit_pool_event(
            "track", "upsert",
            recordId=record_id,
            trackId=state.track_id,
            hullNumber=state.final_hull_number or latest.get("vlmHullNumber"),
            description=state.final_description or latest.get("description") or "等待识别结果",
            status=status,
            memoryInfo=f"临时帧 {len(state.candidate_frames)} · 正式帧 {len(state.keyframe_pool)}/{limit}",
        )

    def reset_persistent_memory(self) -> None:
        self.memory_manager.clear_all()
        self.trace.clear()
        self._next_track_id_value = 1

    def observe(self, frame: np.ndarray, detections: list[Any], frame_index: int, timestamp: float, source_path: str = "", source_fps: float = 25.0) -> None:
        self._source_path, self._source_fps = source_path, float(source_fps or 25)
        self._frame_size = [int(frame.shape[1]), int(frame.shape[0])]
        observed_at = self._monitor_start_time + timestamp if self._monitor_start_time > 0 else time.time()
        seen = set()
        interval = max(1, int(self.settings.get("candidate_every_n_frames", 10)))
        with self._lock:
            self._drain_completed()
            for detection in detections:
                source_track_id = str(detection.track_id)
                seen.add(source_track_id)
                state = self.active.get(source_track_id)
                if state is None:
                    state = ActiveTrack(self._allocate_track_id(), observed_at, observed_at, timestamp, timestamp, tuple(detection.bbox))
                    self.active[source_track_id] = state
                    self._write_track(state)
                state.last_time, state.video_last_time, state.latest_bbox, state.missed_frames = observed_at, timestamp, tuple(detection.bbox), 0
                state.bbox_history.append({"frameIndex": frame_index, "timestamp": round(timestamp, 6), "observedAt": round(observed_at, 6), "bbox": list(detection.bbox)})
                if frame_index % interval == 0 and detection.crop is not None:
                    self._submit_candidate(state, detection.crop.copy(), tuple(detection.bbox), frame.shape, timestamp)
            for source_track_id, state in list(self.active.items()):
                if source_track_id not in seen:
                    state.missed_frames += 1
            self._drain_completed()
            self._prune_expired(observed_at)
            self._finalize_stale()
            for state in self.active.values():
                self._emit_track_status(state, observed_at)

    def _submit_candidate(self, state: ActiveTrack, crop: np.ndarray, bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...], timestamp: float) -> None:
        min_width, min_height = int(self.settings.get("min_crop_width", 80)), int(self.settings.get("min_crop_height", 80))
        if crop.shape[1] < min_width or crop.shape[0] < min_height:
            return
        quality = score_frame(crop, bbox, frame_shape, self.settings.get("quality", {}))["qualityScore"]
        keyframe_limit = int(self.settings.get("keyframe_pool_size", 6))
        if len(state.keyframe_pool) >= keyframe_limit and quality < float(self.settings.get("quality", {}).get("min_score", 0.2)):
            return
        if not self._reserve_candidate_slot(state, quality):
            return
        candidate_id = f"candidate-{uuid.uuid4().hex[:12]}"
        candidate = CandidateFrame(candidate_id, state.track_id, timestamp, bbox, crop, quality)
        candidate.future = self._executor.submit(self.llm.recognize, crop)
        state.candidate_frames[candidate_id] = candidate
        self.trace.append({"event": "candidate_submitted", "trackId": state.track_id, "timestamp": timestamp, "qualityScore": quality})
        self._emit_track_status(state, state.last_time, force=True)

    def _reserve_candidate_slot(self, state: ActiveTrack, quality_score: float) -> bool:
        limit = int(self.settings.get("candidate_pool_size", 12))
        if len(state.candidate_frames) < limit:
            return True
        queued = sorted(
            (
                candidate
                for candidate in state.candidate_frames.values()
                if candidate.future and not candidate.future.running() and not candidate.future.done()
            ),
            key=lambda candidate: candidate.quality_score,
        )
        for candidate in queued:
            if quality_score <= candidate.quality_score:
                return False
            if candidate.future and candidate.future.cancel():
                state.candidate_frames.pop(candidate.candidate_id, None)
                self.trace.append({"event": "candidate_replaced", "trackId": state.track_id, "candidateId": candidate.candidate_id, "qualityScore": candidate.quality_score})
                self._emit_pool_event("candidate", "upsert", recordId=candidate.candidate_id, trackId=state.track_id, hullNumber=None, description="被更高质量候选替换", status="已替换")
                return True
        return False

    def _drain_completed(self, wait_for_all: bool = False) -> None:
        for state in list(self.active.values()):
            self._drain_track(state, wait_for_all)

    def _drain_track(self, state: ActiveTrack, wait_for_all: bool = False) -> None:
        if wait_for_all:
            futures = [candidate.future for candidate in state.candidate_frames.values() if candidate.future]
            if futures:
                timeout = float(self.settings.get("recognition_timeout_seconds", 60))
                wait(futures, timeout=timeout)
        for candidate_id, candidate in list(state.candidate_frames.items()):
            future = candidate.future
            if not future or not future.done():
                if wait_for_all:
                    future.cancel() if future else None
                    state.candidate_frames.pop(candidate_id, None)
                    self.trace.append({"event": "recognition_timeout", "trackId": state.track_id, "candidateId": candidate_id})
                    self._emit_pool_event("candidate", "upsert", recordId=candidate_id, trackId=state.track_id, hullNumber=None, description="单帧识别超时", status="识别超时")
                continue
            try:
                self._promote_candidate(state, candidate, future.result())
            except Exception as error:
                logger.warning("单帧识别失败 track=%s: %s", state.track_id, error)
                self.trace.append({"event": "recognition_failed", "trackId": state.track_id, "error": str(error)})
                self._emit_pool_event("candidate", "upsert", recordId=candidate_id, trackId=state.track_id, hullNumber=None, description=str(error), status="识别失败")
            finally:
                state.candidate_frames.pop(candidate_id, None)
                self._emit_track_status(state, state.last_time, force=True)

    def _promote_candidate(self, state: ActiveTrack, candidate: CandidateFrame, result: dict[str, Any]) -> None:
        readable = result.get("has_readable_hull_number") == "yes"
        confidence = float(result.get("readability_confidence") or 0)
        retention_config = self.settings.get("retention", {})
        retention = (
            float(retention_config.get("readable_bonus", 1.0))
            + float(retention_config.get("confidence_weight", 0.70)) * confidence
            + float(retention_config.get("quality_weight", 0.30)) * candidate.quality_score
            if readable
            else candidate.quality_score
        )
        record = {"keyframeId": f"keyframe-{uuid.uuid4().hex[:12]}", "trackId": state.track_id, "timestamp": candidate.timestamp, "bbox": list(candidate.bbox), "qualityScore": candidate.quality_score, "retentionScore": round(retention, 6), "hasReadableHullNumber": "yes" if readable else "no", "vlmHullNumber": result.get("vlm_hull_number") if readable else None, "readabilityConfidence": confidence, "description": result.get("description", "")}
        candidate_event = {"recordId": candidate.candidate_id, "trackId": state.track_id, "hullNumber": record["vlmHullNumber"], "description": record["description"]}
        self._emit_pool_event("candidate", "upsert", **candidate_event, status="识别完成")
        limit = int(self.settings.get("keyframe_pool_size", 6))
        replaced = self._select_replacement(state, record, limit)
        if len(state.keyframe_pool) >= limit and replaced is None:
            self.trace.append({"event": "keyframe_rejected", "trackId": state.track_id, "candidateId": candidate.candidate_id, "reason": "retention_or_time_diversity"})
            self._emit_pool_event("candidate", "upsert", **candidate_event, status="未进入正式池")
            return
        committed = self._commit_keyframe(record, candidate.crop, replaced)
        if committed is None:
            self._emit_pool_event("candidate", "upsert", **candidate_event, status="向量生成失败")
            return
        if replaced:
            state.keyframe_pool.remove(replaced)
            self._emit_pool_event("keyframe", "remove", recordId=replaced["keyframeId"])
        state.keyframe_pool.append(committed)
        state.keyframe_pool.sort(key=lambda frame: (-frame["retentionScore"], frame["timestamp"]))
        state.pool_revision += 1
        if len(state.keyframe_pool) == limit:
            self._aggregate_if_changed(state)
        self.trace.append({"event": "keyframe_committed", "trackId": state.track_id, "keyframeId": committed["keyframeId"], "isEmbedded": committed["isEmbedded"]})
        self._emit_pool_event("candidate", "upsert", **candidate_event, status="已进入正式池")
        self._emit_pool_event("keyframe", "upsert", recordId=committed["keyframeId"], trackId=committed["trackId"], hullNumber=committed["vlmHullNumber"], description=committed["description"], status="正式帧")
        self._emit_track_status(state, state.last_time, force=True)

    def _select_replacement(self, state: ActiveTrack, record: dict[str, Any], limit: int) -> dict[str, Any] | None:
        if len(state.keyframe_pool) < limit:
            return None
        minimum = min(state.keyframe_pool, key=lambda frame: frame["retentionScore"])
        if record["retentionScore"] < minimum["retentionScore"]:
            return None
        nearest = min(state.keyframe_pool, key=lambda frame: abs(float(frame["timestamp"]) - float(record["timestamp"])))
        retention_config = self.settings.get("retention", {})
        time_gap = float(retention_config.get("min_time_gap_seconds", 30.0))
        replace_margin = float(retention_config.get("replace_margin", 0.05))
        if abs(float(record["timestamp"]) - float(nearest["timestamp"])) < time_gap:
            return nearest if record["retentionScore"] >= nearest["retentionScore"] + replace_margin else None
        return minimum

    def _commit_keyframe(self, record: dict[str, Any], crop: np.ndarray, replaced: dict[str, Any] | None) -> dict[str, Any] | None:
        track_dir = Path(self.config["paths"]["keyframe_dir"]) / record["trackId"]
        track_dir.mkdir(parents=True, exist_ok=True)
        image_path = track_dir / f"{record['keyframeId']}.jpg"
        pending_path = track_dir / f".{record['keyframeId']}.pending.jpg"
        # 嵌入前先缩小，降低同步 encode 的卡顿（主线程在 drain 时会阻塞）
        embed_crop = crop
        max_side = int(self.settings.get("embed_max_side", 384))
        h, w = crop.shape[:2]
        if max(h, w) > max_side > 0:
            scale = max_side / float(max(h, w))
            embed_crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(pending_path), embed_crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise OSError(f"关键帧保存失败：{pending_path}")
        vector_id = stable_vector_id(record["keyframeId"])
        vector = None
        try:
            vector = self.embedder.encode_images([pending_path])[0]
        except Exception as error:
            logger.warning("关键帧向量生成失败 %s: %s", record["keyframeId"], error)
        if vector is None:
            pending_path.unlink(missing_ok=True)
            self.trace.append({"event": "keyframe_commit_rejected", "trackId": record["trackId"], "reason": "embedding_failed"})
            return None
        # 落盘保留更高清原图，向量已用缩小图算过
        if not cv2.imwrite(str(pending_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"关键帧保存失败：{pending_path}")
        record.update(keyframePath=str(image_path), keyframeVectorId=vector_id, isEmbedded=True)
        table_rows = self.repository.keyframes.rows()
        index_path = self.vectors.keyframes.path
        manifest_path = self.vectors.keyframes.manifest_path
        index_backup = self._read_optional(index_path)
        manifest_backup = self._read_optional(manifest_path)
        try:
            if replaced and replaced.get("keyframeVectorId") is not None:
                self.vectors.keyframes.remove([int(replaced["keyframeVectorId"])])
            if replaced:
                self.repository.delete_keyframe(replaced["keyframeId"])
            os.replace(pending_path, image_path)
            self.vectors.keyframes.add(vector_id, vector)
            self.repository.upsert_keyframe(self._keyframe_row(record))
        except Exception:
            rollback_errors = []
            try:
                self.repository.keyframes.replace_all(table_rows)
            except Exception as error:
                rollback_errors.append(f"关键帧表恢复失败：{error}")
            for path, backup in ((index_path, index_backup), (manifest_path, manifest_backup)):
                try:
                    self._restore_optional(path, backup)
                except Exception as error:
                    rollback_errors.append(f"特征文件恢复失败 {path}：{error}")
            try:
                self.vectors.keyframes.reset_cache()
            except Exception as error:
                rollback_errors.append(f"特征缓存重置失败：{error}")
            pending_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            if rollback_errors:
                logger.error("关键帧事务回滚不完整：%s", "；".join(rollback_errors))
            raise
        if replaced:
            replaced_path = replaced.get("keyframePath")
            if replaced_path:
                Path(replaced_path).unlink(missing_ok=True)
        return record

    @staticmethod
    def _keyframe_row(record: dict[str, Any]) -> dict[str, Any]:
        return {"keyframe_id": record["keyframeId"], "track_id": record["trackId"], "timestamp": record["timestamp"], "keyframe_path": record["keyframePath"], "bbox": record["bbox"], "quality_score": record["qualityScore"], "retention_score": record["retentionScore"], "has_readable_hull_number": record["hasReadableHullNumber"], "vlm_hull_number": record["vlmHullNumber"], "readability_confidence": record["readabilityConfidence"], "description": record["description"], "keyframe_vector_id": record["keyframeVectorId"], "is_embedded": record["isEmbedded"]}

    @staticmethod
    def _read_optional(path: Path) -> bytes | None:
        return path.read_bytes() if path.exists() else None

    @staticmethod
    def _restore_optional(path: Path, content: bytes | None) -> None:
        if content is None:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".rollback")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def _aggregate(self, state: ActiveTrack) -> None:
        result = aggregate_keyframes(state.keyframe_pool, self.settings.get("aggregation", {}))
        state.final_hull_number = result["finalHullNumber"]
        state.final_description = result["finalDescription"]
        state.final_match_type = result["finalMatchType"]
        self._write_track(state)
        state.aggregated_revision = state.pool_revision

    def _aggregate_if_changed(self, state: ActiveTrack) -> None:
        if state.aggregated_revision != state.pool_revision:
            self._aggregate(state)

    def _write_track(self, state: ActiveTrack, trajectory_path: str = "") -> None:
        current = self.repository.get_track(state.track_id)
        saved_path = current.get("trajectoryPath", "") if current else ""
        self.repository.upsert_track({"track_id": state.track_id, "start_time": state.start_time, "end_time": state.last_time, "video_start_time": state.video_start_time, "video_end_time": state.video_last_time, "final_hull_number": state.final_hull_number, "final_description": state.final_description, "final_match_type": state.final_match_type, "trajectory_path": trajectory_path or saved_path})

    def _prune_expired(self, timestamp: float) -> None:
        if self._last_maintenance_time >= 0 and timestamp - self._last_maintenance_time < 1.0:
            return
        self._last_maintenance_time = timestamp
        for state in self.active.values():
            self._write_track(state)
        retention = self.memory_manager.settings.read()["retentionSeconds"]
        protected = [state.track_id for state in self.active.values() if retention <= 0 or timestamp - state.last_time <= retention]
        expired = self.memory_manager.prune_expired(timestamp, protected)
        if expired:
            expired_ids = {str(track_id) for track_id in expired}
            for source_track_id, state in list(self.active.items()):
                if state.track_id in expired_ids:
                    self.active.pop(source_track_id, None)
                    for candidate in state.candidate_frames.values():
                        if candidate.future:
                            candidate.future.cancel()
                    self._emit_track_status(state, timestamp, remove=True)
            self.trace.append({"event": "memory_expired", "trackIds": expired, "timestamp": timestamp})
            logger.info("轨迹记忆自动清理：%s", ", ".join(expired))

    def _finalize_stale(self) -> None:
        threshold = int(self.settings.get("max_stale_frames", 120))
        for track_id, state in list(self.active.items()):
            if state.missed_frames > threshold:
                self._drain_track(state, wait_for_all=True)
                self._finalize_track(state)
                self.active.pop(track_id, None)

    def _finalize_track(self, state: ActiveTrack) -> None:
        path = Path(self.config["paths"]["trajectory_dir"]) / f"{state.track_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"trackId": state.track_id, "sourceVideoPath": self._source_path, "sourceFps": self._source_fps, "monitorStartTime": state.start_time, "monitorEndTime": state.last_time, "frameSize": self._frame_size, "boxes": sorted(state.bbox_history, key=lambda item: item["timestamp"])}
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        self._aggregate_if_changed(state)
        self._write_track(state, str(path))
        self._emit_track_status(state, state.last_time, remove=True)
        self.trace.append({"event": "track_finalized", "trackId": state.track_id, "trajectoryPath": str(path)})

    def finalize_active(self) -> None:
        """完成当前视频中的活跃轨迹，但保留识别线程池供后续视频复用。"""
        with self._lock:
            self._drain_completed(wait_for_all=True)
            for state in list(self.active.values()):
                self._finalize_track(state)
            self.active.clear()

    def finalize_all(self) -> None:
        self.finalize_active()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def display_tracks(self) -> dict[int, Any]:
        result = {}
        for source_track_id, state in self.active.items():
            result[int(source_track_id)] = SimpleNamespace(track_id=state.track_id, hull_number=state.final_hull_number or "", description=state.final_description, recognized=bool(state.keyframe_pool), pending=bool(state.candidate_frames), final_match_type=state.final_match_type)
        return result

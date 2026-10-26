"""SeaAgent 原子工具服务。"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from config import load_config
from memory import MemoryRepository, normalize_hull_number
from memory.time_format import MONITOR_TIMEZONE, format_monitor_time
from services import AgentLLMService, QwenMultimodalEmbedder
from vector_store import VectorCatalog


class ToolService:
    MAX_TRACK_PAGE_SIZE = 100
    DEFAULT_TIMEZONE = MONITOR_TIMEZONE
    IDENTIFICATION_TIMEOUT_SECONDS = 60
    STATISTICS_TIMEOUT_SECONDS = 120
    EVIDENCE_TIMEOUT_SECONDS = 120
    TOOL_TIMEOUTS: ClassVar[dict[str, int]] = {
        "recognize_tracks": IDENTIFICATION_TIMEOUT_SECONDS,
        "match_registry_tracks": IDENTIFICATION_TIMEOUT_SECONDS,
        "verify_vessel_images": IDENTIFICATION_TIMEOUT_SECONDS,
        "count_total_entities": STATISTICS_TIMEOUT_SECONDS,
        "count_registry_entities": STATISTICS_TIMEOUT_SECONDS,
        "count_unregistered_entities": STATISTICS_TIMEOUT_SECONDS,
        "show_evidence": EVIDENCE_TIMEOUT_SECONDS,
    }
    MAX_RETURNED_MATCHES = 10
    MAX_FRAMES_PER_TRACK = 3

    # ------------------------------------------------------------------ #
    # 一、服务初始化与执行守卫                                                       #
    # ------------------------------------------------------------------ #
    # 1.1 装配仓库、嵌入器、LLM 与向量库
    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.settings = self.config["pipeline"]["retrieval"]

    # 1.2 线程内执行并施加超时
    # 时间预算由服务层自己执行，运行时只负责调用和记录结果。
    def _run_with_timeout(self, operation: Any, timeout: float, tool_name: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        finished = threading.Event()

        def invoke() -> None:
            try:
                result["value"] = operation()
            except Exception as error:  # noqa: BLE001 - 工具边界统一返回模型可读错误。
                result["error"] = error
            finally:
                finished.set()

        threading.Thread(target=invoke, name=f"sea-{tool_name}", daemon=True).start()
        if not finished.wait(max(0.001, float(timeout))):
            return {  # 超过工具时限：不再等待，直接返回超时结果
                "ok": False,  # 调用失败
                "error": "tool_timeout",  # 错误码：超时
                "errorType": "TimeoutError",  # 异常类型
                "tool": tool_name,  # 超时的工具名
                "timeoutSeconds": float(timeout),  # 本轮实际超时阈值
                "hint": f"{tool_name} 超过 {float(timeout):.0f} 秒，已返回超时结果。",  # 给模型的说明
            }
        if "error" in result:
            error = result["error"]
            return {  # 工具内部抛异常：包成统一错误结构
                "ok": False,  # 调用失败
                "error": "tool_failed",  # 错误码：执行异常
                "errorType": type(error).__name__,  # 原始异常类型
                "tool": tool_name,  # 出错的工具名
                "hint": str(error),  # 原始异常信息
            }
        value = result.get("value")
        return value if isinstance(value, dict) else {"ok": True, "value": value}  # 字典原样返回，其他值包一层

    # 1.3 业务工具的统一起点
    def _run_tool(self, tool_name: str, operation: Any) -> dict[str, Any]:
        return self._run_with_timeout(operation, self.TOOL_TIMEOUTS[tool_name], tool_name)

    # ------------------------------------------------------------------ #
    # 二、轨迹识别                                                             #
    # ------------------------------------------------------------------ #
    # 2.1 读取完整的匹配轨迹范围
    def _scan_tracks(
        self,
        scope: tuple[float, float] | None,
        hull_number: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Read the complete matching track scope without exposing raw pages to the model."""
        tracks: list[dict[str, Any]] = []
        cursor: tuple[float, str] | None = None
        complete = False
        for _ in range(1000):
            page, has_more = self.repository.find_tracks_page(
                scope, hull_number, None, cursor, self.MAX_TRACK_PAGE_SIZE
            )
            tracks.extend(page)
            if not has_more or not page:
                complete = True
                break
            cursor = (float(page[-1]["startTime"]), str(page[-1]["trackId"]))
        return tracks, complete

    # 2.2 每条轨迹只保留少量可检索帧
    def _load_search_frames(
        self,
        tracks: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        """Keep only a few searchable frames per track; full frame records stay inside the service."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        track_ids = [str(track["trackId"]) for track in tracks]
        for offset in range(0, len(track_ids), 40):
            batch = track_ids[offset : offset + 40]
            raw = self.repository.get_keyframes(batch, embedded_only=False)
            for track_id in batch:
                frames = sorted(
                    raw.get(track_id, []),
                    key=lambda item: (
                        str(item.get("hasReadableHullNumber") or "no").lower() == "yes",
                        float(item.get("readabilityConfidence") or 0),
                        float(item.get("retentionScore") or 0),
                    ),
                    reverse=True,
                )[: self.MAX_FRAMES_PER_TRACK]
                if not frames:
                    missing.append(track_id)
                    grouped[track_id] = []
                    continue
                vectors = self._load_indexed_vectors(
                    frames, vector_key="keyframeVectorId", index=self.vectors.keyframes
                )
                searchable = [
                    frame for frame in frames
                    if frame.get("keyframeVectorId") is not None
                    and int(frame["keyframeVectorId"]) in vectors
                ]
                grouped[track_id] = searchable
                if not searchable:
                    missing.append(track_id)
        return grouped, missing

    # 2.3 把按轨迹分组的帧拍平
    @staticmethod
    def _flatten_frames(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [frame for frames in groups.values() for frame in frames]

    # 2.5 组装单条轨迹的匹配摘要
    def _track_match_summary(
        self,
        track: dict[str, Any],
        frames: list[dict[str, Any]],
        *,
        score: float | None = None,
        band: str = "mismatch",
        reference_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "trackId": str(track["trackId"]),  # 轨迹编号
            "scoreBand": band,  # 原始档位 match / uncertain / mismatch
        }
        for field in ("startTimeText", "endTimeText"):
            if track.get(field):
                summary[field] = track[field]  # 轨迹时间
        hull = track.get("finalHullNumber")
        description = str(track.get("finalDescription") or "").strip()
        keyframe_ids = [str(frame["keyframeId"]) for frame in frames[: self.MAX_FRAMES_PER_TRACK]]
        if hull:
            summary["hullNumber"] = hull  # 轨迹最终的舷号结论
        if description:
            summary["description"] = description[:160]  # 轨迹最终的外观描述
        if score is not None:
            summary["score"] = round(float(score), 6)  # 相似度分数
        if keyframe_ids:
            summary["keyframeIds"] = keyframe_ids  # 代表关键帧，每条轨迹限量
        if reference_ids:
            summary["registryReferenceIds"] = list(dict.fromkeys(reference_ids))[: self.MAX_FRAMES_PER_TRACK]  # 关联的库参考图编号
        return summary

    # 2.6 按外观文本给轨迹打分
    def _track_text_scores(
        self,
        description: str,
        frame_groups: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        gallery = self._flatten_frames(frame_groups)
        if not gallery:
            return {}
        try:
            query = self.embedder.encode_text(
                description,
                self.config["prompts"]["text_retrieval_instruction"],
            )
            vectors = self._load_indexed_vectors(
                gallery, vector_key="keyframeVectorId", index=self.vectors.keyframes
            )
        except Exception:  # noqa: BLE001 - failed embedding must remain an unconfirmed search.
            return {}
        scored: list[dict[str, Any]] = []
        for track_id, track_frames in frame_groups.items():
            values = [
                (float(np.dot(query, vectors[int(frame["keyframeVectorId"])])), frame)
                for frame in track_frames
                if frame.get("keyframeVectorId") is not None
                and int(frame["keyframeVectorId"]) in vectors
            ]
            if not values:
                continue
            values.sort(key=lambda item: item[0], reverse=True)
            score = values[0][0]
            scored.append({
                "matchedTrackId": track_id,
                "embeddingScore": round(score, 6),
                "scoreBand": self._band(score, "text"),
                "matchedKeyframeIds": [item[1]["keyframeId"] for item in values[:2]],
            })
        scored.sort(key=lambda item: float(item["embeddingScore"]), reverse=True)
        return {str(item["matchedTrackId"]): item for item in scored}

    # 2.7 按图像相似度给轨迹打分
    def _track_image_scores(
        self,
        image_path: str,
        frame_groups: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        frames = self._flatten_frames(frame_groups)
        if not frames or not Path(image_path).is_file():
            return {}
        try:
            query = self.embedder.encode_images([image_path])[0]
            vectors = self._load_indexed_vectors(
                frames, vector_key="keyframeVectorId", index=self.vectors.keyframes
            )
        except Exception:  # noqa: BLE001 - failed embedding must remain an unconfirmed search.
            return {}
        scored: list[dict[str, Any]] = []
        for track_id, track_frames in frame_groups.items():
            values = [
                (float(np.dot(query, vectors[int(frame["keyframeVectorId"])])), frame)
                for frame in track_frames
                if frame.get("keyframeVectorId") is not None
                and int(frame["keyframeVectorId"]) in vectors
            ]
            if not values:
                continue
            values.sort(key=lambda item: item[0], reverse=True)
            score = values[0][0]
            scored.append(
                {
                    "matchedTrackId": track_id,
                    "embeddingScore": round(score, 6),
                    "scoreBand": self._band(score, "image"),
                    "matchedKeyframeIds": [item[1]["keyframeId"] for item in values[:2]],
                }
            )
        scored.sort(key=lambda item: float(item["embeddingScore"]), reverse=True)
        return {str(item["matchedTrackId"]): item for item in scored}

    # 2.8 轨迹识别的内部实现
    def _recognize_tracks(
        self,
        hull_number: str | None = None,
        description: str | None = None,
        image_path: str | None = None,
        time_range: tuple[str | float, str | float] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        scope, problem = self._validate_time_range(time_range)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem}
        hull = normalize_hull_number(hull_number)
        description = str(description or "").strip()
        image_path = str(image_path or "").strip()
        if not hull and not description and not image_path:
            return {"ok": False, "error": "recognition_query_required", "hint": "至少提供 hull_number、description 或 image_path。"}

        tracks, search_complete = self._scan_tracks(scope)
        frames, missing = self._load_search_frames(tracks)
        keyframe_hull_matches: dict[str, list[dict[str, Any]]] = {}
        if hull:
            track_ids = [str(track["trackId"]) for track in tracks]
            for offset in range(0, len(track_ids), 40):
                batch = track_ids[offset : offset + 40]
                for track_id, stored_frames in self.repository.get_keyframes(batch, embedded_only=False).items():
                    matching = [
                        frame for frame in stored_frames
                        if normalize_hull_number(frame.get("vlmHullNumber")) == hull
                        and float(frame.get("readabilityConfidence") or 0) >= 0.75
                    ]
                    if matching:
                        keyframe_hull_matches[track_id] = matching
        by_id = {str(track["trackId"]): track for track in tracks}
        text_scores = self._track_text_scores(description, frames) if description else {}
        image_scores = self._track_image_scores(image_path, frames) if image_path else {}
        unscored_tracks = [
            track_id for track_id in by_id
            if description or image_path
            if track_id not in text_scores and track_id not in image_scores
            and not (hull and normalize_hull_number(by_id[track_id].get("finalHullNumber")) == hull)
            and not keyframe_hull_matches.get(track_id)
        ]
        missing = list(dict.fromkeys(missing + unscored_tracks))
        candidates: list[dict[str, Any]] = []
        for track_id, track in by_id.items():
            text_item = text_scores.get(track_id, {})
            image_item = image_scores.get(track_id, {})
            exact = bool(
                hull
                and (
                    normalize_hull_number(track.get("finalHullNumber")) == hull
                    or keyframe_hull_matches.get(track_id)
                )
            )
            score_values = [
                float(item["embeddingScore"])
                for item in (text_item, image_item)
                if item.get("embeddingScore") is not None
            ]
            score = max(score_values) if score_values else (1.0 if exact else None)
            band = "match" if exact else (
                self._band(score, "image" if image_item else "text") if score is not None else "uncertain"
            )
            frame_ids = list(dict.fromkeys(
                text_item.get("matchedKeyframeIds", []) + image_item.get("matchedKeyframeIds", [])
            ))
            if exact and not frame_ids:
                frame_ids = [frame["keyframeId"] for frame in keyframe_hull_matches.get(track_id, [])[:2]]
            selected_frames = [frame for frame in frames.get(track_id, []) if frame["keyframeId"] in frame_ids]
            if exact:
                selected_frames.extend(keyframe_hull_matches.get(track_id, [])[:2])
            # 只把有明确舷号证据或实际匹配分数的轨迹返回；未评分轨迹只影响覆盖状态，
            # 不应把整段轨迹列表膨胀进模型上下文。
            if exact or score is not None:
                candidates.append(self._track_match_summary(track, selected_frames, score=score, band=band))
        candidates = [item for item in candidates if item["scoreBand"] != "mismatch"]
        candidates.sort(key=lambda item: (item["scoreBand"] != "match", -(item["score"] or -1)))
        selected = candidates[: max(1, min(self.MAX_RETURNED_MATCHES, int(top_k or self.MAX_RETURNED_MATCHES)))]
        search_complete = search_complete and not missing
        status = "confirmed" if any(item["scoreBand"] == "match" for item in selected) else "unconfirmed" if selected or not search_complete else "mismatch"
        result = {
            "ok": True,  # 调用成功
            "status": status,  # 本轮结论：confirmed / unconfirmed / mismatch
            "matches": selected,  # 入选候选轨迹摘要（确认优先、分数降序、按 topK 截断）
            "coverageComplete": search_complete,  # 时间扫描完整且不存在无法检索的轨迹
            "statistics": {  # 本轮检索统计
                "searchedTrackCount": len(tracks),  # 时间范围内扫描到的轨迹数
                "candidateCount": len(selected),  # 入选候选数
                "unsearchableTrackCount": len(missing),  # 无法检索的轨迹数
            },
        }
        if hull and status != "confirmed":
            result["nextStep"] = "match_registry_tracks"  # 有舷号但未确认时，引导去登记表比对
        return result

    # 2.9 轨迹识别工具的入口
    def recognizeTracks(self, hullNumber: str | None = None, description: str | None = None, imagePath: str | None = None, timeRange: tuple[str | float, str | float] | None = None, topK: int | None = None) -> dict[str, Any]:
        return self._run_tool(
            "recognize_tracks",
            lambda: self._recognize_tracks(hullNumber, description, imagePath, timeRange, topK),
        )

    # ------------------------------------------------------------------ #
    # 三、先验库匹配                                                            #
    # ------------------------------------------------------------------ #
    # 3.1 轨迹与先验库匹配的内部实现
    def _match_registry_tracks(
        self,
        hull_number: str,
        time_range: tuple[str | float, str | float] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        scope, problem = self._validate_time_range(time_range)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem}
        hull = normalize_hull_number(hull_number)
        if not hull:
            return {"ok": False, "error": "hull_number_required", "hint": "先验库辅助识别必须提供舷号。"}
        registry_items = self.repository.registry_by_hull(hull)
        references = [
            reference
            for item in registry_items
            for reference in item.get("references", [])
            if isinstance(reference, dict)
        ]
        tracks, complete = self._scan_tracks(scope)
        frames, missing = self._load_search_frames(tracks)
        if not registry_items or not references:
            coverage_complete = complete and not missing and bool(registry_items and references)
            return {  # 库里查不到该舷号或没有参考图：只能返回未确认
                "ok": True,  # 调用成功
                "status": "unconfirmed",  # 没有可比对证据，无法给结论
                "registryFound": bool(registry_items),  # 库里是否登记过该舷号
                "matches": [],  # 没有可匹配的参考图
                "coverageComplete": coverage_complete,  # 扫描是否完整
                "statistics": {  # 本轮统计
                    "searchedTrackCount": len(tracks),  # 扫描轨迹数
                    "candidateCount": 0,  # 候选数
                    "unsearchableTrackCount": len(missing),  # 无法检索的轨迹数
                },
            }
        result = self._match_images(
            queryImages=references,
            galleryImages=self._flatten_frames(frames),
            topK=top_k or self.MAX_RETURNED_MATCHES,
        )
        by_id = {str(track["trackId"]): track for track in tracks}
        matches: list[dict[str, Any]] = []
        for item in result.get("matches", []):
            track = by_id.get(str(item.get("matchedTrackId")))
            if track is None:
                continue
            frame_ids = item.get("matchedKeyframeIds") or []
            selected_frames = [frame for frame in frames.get(str(track["trackId"]), []) if frame["keyframeId"] in frame_ids]
            matches.append(self._track_match_summary(
                track,
                selected_frames,
                score=item.get("embeddingScore"),
                band=str(item.get("scoreBand") or "uncertain"),
                reference_ids=item.get("matchedRegistryReferenceIds"),
            ))
        matches.sort(key=lambda item: (item["scoreBand"] != "match", -(item.get("score") or -1)))
        matches = matches[: max(1, min(self.MAX_RETURNED_MATCHES, int(top_k or self.MAX_RETURNED_MATCHES)))]
        missing = list(dict.fromkeys(missing + [str(value) for value in result.get("unscoredTrackIds", [])]))
        visual_complete = bool(result.get("registryCoverageComplete")) and complete and not missing
        status = "confirmed" if any(item["scoreBand"] == "match" for item in matches) else "unconfirmed" if matches or not visual_complete else "mismatch"
        return {
            "ok": True,  # 调用成功
            "status": status,  # confirmed / unconfirmed / mismatch
            "registryFound": True,  # 已找到该舷号的库项
            "matches": matches,  # 每条轨迹取最高分配对后的候选
            "statistics": {  # 本轮统计
                "searchedTrackCount": len(tracks),  # 扫描轨迹数
                "candidateCount": len(matches),  # 候选数
                "unsearchableTrackCount": len(missing),  # 无法检索的轨迹数
            },
            "coverageComplete": visual_complete,  # 时间与视觉覆盖都完整、无遗漏轨迹
        }

    # 3.2 先验库匹配工具的入口
    def matchRegistryTracks(self, hullNumber: str, timeRange: tuple[str | float, str | float] | None = None, topK: int | None = None) -> dict[str, Any]:
        return self._run_tool(
            "match_registry_tracks",
            lambda: self._match_registry_tracks(hullNumber, timeRange, topK),
        )

    # ------------------------------------------------------------------ #
    # 四、视觉核验                                                             #
    # ------------------------------------------------------------------ #
    # 4.1 用视觉模型核验船舶图像
    def verifyVesselImages(
        self,
        referenceImage: str | None = None,
        referenceRegistryReferenceId: str | None = None,
        candidateImage: str | None = None,
        candidateKeyframeId: str | None = None,
    ) -> dict[str, Any]:
        def verify() -> dict[str, Any]:
            reference = str(referenceImage or "").strip()
            reference_id = str(referenceRegistryReferenceId or "").strip()
            candidate = str(candidateImage or "").strip()
            keyframe_id = str(candidateKeyframeId or "").strip()
            if bool(reference) == bool(reference_id) or bool(candidate) == bool(keyframe_id):
                return {"ok": False, "error": "two_images_required", "decision": "uncertain"}  # 参考图与候选图必须各给且只给一张
            if reference_id:
                references = self.repository.references_by_ids([reference_id])
                reference = str(references[0].get("imagePath") or "") if references else ""
                if not reference:
                    return {  # 库参考图编号查不到对应的图片
                        "ok": False,  # 调用失败
                        "error": "registry_reference_not_found",  # 错误码：参考图不存在
                        "decision": "uncertain",  # 无法核验，只能给不确定
                        "registryReferenceId": reference_id,  # 出错的参考图编号
                    }
            if keyframe_id:
                frame = self.repository.get_keyframe(keyframe_id)
                candidate = str(frame.get("keyframePath") or "") if frame else ""
                if not candidate:
                    return {"ok": False, "error": "keyframe_not_found", "decision": "uncertain"}  # 关键帧编号查不到图片
            if not Path(reference).is_file() or not Path(candidate).is_file():
                return {"ok": False, "error": "image_not_found", "decision": "uncertain"}  # 文件路径不存在
            result = self.llm.verify([reference], [candidate])
            facts = result.get("facts") or []
            facts = facts if isinstance(facts, list) else [facts]
            response = {
                "ok": True,  # 调用成功
                "decision": result.get("decision", "uncertain"),  # 视觉模型的核验结论
                "facts": [str(fact).strip()[:200] for fact in facts[:3] if str(fact).strip()],  # 限长的可见依据
            }
            if reference_id:
                response["registryReferenceId"] = reference_id  # 使用的库参考图编号
            if keyframe_id:
                response["keyframeId"] = keyframe_id  # 使用的轨迹关键帧编号
            return response

        return self._run_tool("verify_vessel_images", verify)

    # ------------------------------------------------------------------ #
    # 五、实体统计                                                             #
    # ------------------------------------------------------------------ #
    # 5.1 去重与库比对后的实体分析
    def _entity_analysis(self, time_range: tuple[str | float, str | float] | None = None, compare_registry: bool = True) -> dict[str, Any]:
        scope, problem = self._validate_time_range(time_range)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem}
        tracks, complete = self._scan_tracks(scope)
        frames, missing = self._load_search_frames(tracks)
        dedup = self._deduplicate_tracks(tracks, {track_id: {"keyframes": values} for track_id, values in frames.items()})
        missing = list(dict.fromkeys(missing + dedup.get("unsearchableTrackIds", [])))
        groups = dedup.get("highGroups") or [[str(track["trackId"])] for track in tracks]
        low_groups = dedup.get("lowGroups") or groups
        pending_merge_count = int(dedup.get("pendingMergeCount") or 0)
        dedup_stable = str(dedup.get("countStability") or "stable") == "stable"
        track_by_id = {str(track["trackId"]): track for track in tracks}
        visual_matches: dict[str, dict[str, Any]] = {}
        visual: dict[str, Any] = {}
        registry_items = self.repository.list_registry() if compare_registry else []
        references = [reference for item in registry_items for reference in item.get("references", []) if isinstance(reference, dict)]
        if compare_registry and references and frames:
            visual = self._match_images(
                queryImages=references,
                galleryImages=self._flatten_frames(frames),
                registryItems=registry_items,
            )
            visual_matches = {str(item.get("matchedTrackId")): item for item in visual.get("matches", [])}
        visual_coverage_complete = not compare_registry or not registry_items or bool(
            references and visual.get("registryCoverageComplete")
        )
        visual_unscored = {
            str(value) for value in (visual.get("unscoredTrackIds", []) if compare_registry else [])
        }
        registry_hulls = {
            normalize_hull_number(value)
            for item in registry_items
            for value in [item.get("hullNumber"), *(item.get("aliases") or [])]
            if normalize_hull_number(value)
        } if compare_registry else set()
        in_registry = unregistered = uncertain = 0
        track_ids: list[str] = []
        keyframe_ids: list[str] = []
        reference_ids: list[str] = []
        for group in groups:
            statuses = []
            for track_id in group:
                track_id = str(track_id)
                track_frames = frames.get(track_id, [])
                track_ids.append(track_id)
                keyframe_ids.extend(
                    str(frame["keyframeId"])
                    for frame in track_frames[:2]
                    if frame.get("keyframeId")
                )
                if not compare_registry:
                    continue
                track = track_by_id.get(track_id, {})
                hull = normalize_hull_number(track.get("finalHullNumber"))
                match = visual_matches.get(track_id, {})
                reference_ids.extend(
                    str(value) for value in match.get("matchedRegistryReferenceIds", []) if value
                )
                hull_frame_match = any(
                    normalize_hull_number(frame.get("vlmHullNumber")) in registry_hulls
                    and float(frame.get("readabilityConfidence") or 0) >= 0.75
                    for frame in track_frames
                )
                if hull in registry_hulls or hull_frame_match or match.get("scoreBand") == "match":
                    statuses.append("in_registry")
                elif hull or not registry_items or (
                    visual_coverage_complete
                    and track_id not in missing
                    and track_id not in visual_unscored
                ):
                    statuses.append("unregistered")
                else:
                    statuses.append("uncertain")
            if compare_registry:
                if "in_registry" in statuses:
                    in_registry += 1
                elif statuses and all(status == "unregistered" for status in statuses):
                    unregistered += 1
                else:
                    uncertain += 1
        registry_incomplete = compare_registry and bool(registry_items) and (
            not references or not visual_coverage_complete
        )
        count_incomplete = bool(
            missing or not complete or not dedup_stable or pending_merge_count or registry_incomplete
        )
        return {
            "ok": True,  # 调用成功
            "entityCount": len(groups),  # 去重后的实体（船）数
            "minimumEntityCount": len(low_groups),  # 低阈值合并出的最少实体数
            "trackCount": len(tracks),  # 原始轨迹数
            "inRegistryCount": in_registry,  # 库内实体数
            "unregisteredCount": unregistered,  # 库外实体数
            "uncertainCount": uncertain,  # 归属不确定的实体数
            "countComplete": not count_incomplete,  # 计数是否完整
            "countStability": dedup.get("countStability", "stable"),  # 去重稳定性 stable / sensitive / incomplete
            "pendingMergeCount": pending_merge_count,  # 待确认合并的组数
            "unsearchableTrackCount": len(missing),  # 缺帧或缺向量、无法检索的轨迹数
            "trackIds": list(dict.fromkeys(track_ids))[: self.MAX_RETURNED_MATCHES],  # 实体覆盖的轨迹编号（截断）
            "keyframeIds": list(dict.fromkeys(keyframe_ids))[: self.MAX_RETURNED_MATCHES * self.MAX_FRAMES_PER_TRACK],  # 代表关键帧编号（截断）
            "registryReferenceIds": list(dict.fromkeys(reference_ids))[: self.MAX_RETURNED_MATCHES],  # 命中的库参考图编号（截断）
        }

    # 5.2 统计全部实体
    def countTotalEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_total_entities",
            lambda: self._entity_count(timeRange, False, "entityCount"),
        )

    # 5.3 统计库内实体
    def countRegistryEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_registry_entities",
            lambda: self._entity_count(timeRange, True, "inRegistryCount"),
        )

    # 5.4 统计库外实体
    def countUnregisteredEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_unregistered_entities",
            lambda: self._entity_count(timeRange, True, "unregisteredCount"),
        )

    # 5.5 计数工具的公共封装
    def _entity_count(self, time_range: Any, compare_registry: bool, field: str) -> dict[str, Any]:
        result = self._entity_analysis(time_range, compare_registry)
        if not result.get("ok"):
            return result
        compact = {
            "ok": True,
            "count": result[field],
            "countComplete": result["countComplete"],
            "countStability": result["countStability"],
            "uncertainCount": result["uncertainCount"],
            "pendingMergeCount": result["pendingMergeCount"],
            "trackCount": result["trackCount"],
            "unsearchableTrackCount": result["unsearchableTrackCount"],
            "trackIds": result["trackIds"],
            "keyframeIds": result["keyframeIds"],
            "registryReferenceIds": result["registryReferenceIds"],
        }
        if field == "entityCount" and result["countStability"] != "stable":
            compact["minimumCount"] = result["minimumEntityCount"]
        return compact

    # epoch 秒的合理下限（2017 年）：早于此说明模型在编数字，而不是在描述监控录像
    MIN_PLAUSIBLE_EPOCH = datetime(2017, 1, 1, tzinfo=DEFAULT_TIMEZONE).timestamp()

    # ------------------------------------------------------------------ #
    # 六、时间处理                                                             #
    # ------------------------------------------------------------------ #
    # 6.1 把时间转成 epoch 秒
    @classmethod
    def _epoch_seconds(cls, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("boolean is not a timestamp")
        try:
            return float(value)
        except (TypeError, ValueError):
            text = str(value).strip()
            try:
                timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("expected an ISO-8601 timestamp") from error
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=cls.DEFAULT_TIMEZONE)
            return timestamp.timestamp()

    # 6.2 校验并规范化时间范围
    @classmethod
    def _validate_time_range(cls, time_range: Any) -> tuple[tuple[float, float] | None, str]:
        """将 ISO-8601 时间或 epoch 秒转成统一的 epoch 范围。"""
        if time_range is None:
            return None, ""
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            return None, "time_range 必须包含起点和终点"
        if isinstance(time_range[0], str) != isinstance(time_range[1], str):
            return None, "time_range 两端必须使用相同格式：均为 ISO-8601 时间，或均为 epoch 秒"
        try:
            start, end = (cls._epoch_seconds(value) for value in time_range)
        except (TypeError, ValueError, OverflowError):
            return None, "time_range 必须是两个 ISO-8601 时间或 epoch 秒"
        if not (math.isfinite(start) and math.isfinite(end)):
            return None, "time_range 不能包含 nan 或 inf"
        try:
            datetime.fromtimestamp(start, cls.DEFAULT_TIMEZONE)
            datetime.fromtimestamp(end, cls.DEFAULT_TIMEZONE)
        except (OverflowError, OSError, ValueError):
            return None, "time_range 超出支持的日期范围"
        if end < start:
            return None, f"time_range 的起点晚于终点：{[start, end]}"
        if start < cls.MIN_PLAUSIBLE_EPOCH:
            return None, (
                "time_range 早于监控数据支持的 2017 年。请核对时间；没有时间条件时省略此参数。"
            )
        return (start, end), ""

    # ------------------------------------------------------------------ #
    # 七、帧与片段                                                             #
    # ------------------------------------------------------------------ #
    # 7.1 组装关键帧结果
    def _build_frames(self, trackIds: Iterable[str | int]) -> dict[str, Any]:
        ids = [str(value) for value in dict.fromkeys(trackIds)]
        all_frames = self.repository.get_keyframes(ids, embedded_only=False)
        vector_ids = [
            int(frame["keyframeVectorId"])
            for frames in all_frames.values()
            for frame in frames
            if frame.get("keyframeVectorId") is not None
        ]
        available_vectors = self.vectors.keyframes.get_many(vector_ids) if vector_ids else {}
        grouped, discarded, missing = {}, [], []
        for track_id in ids:
            valid = []
            invalid = []
            for frame in all_frames.get(track_id, []):
                vid = frame.get("keyframeVectorId")
                patched = dict(frame)
                patched["isEmbedded"] = vid is not None and int(vid) in available_vectors
                valid.append(patched)
                if not patched["isEmbedded"]:
                    invalid.append(frame.get("keyframeId"))
            grouped[track_id] = {"keyframeIds": [frame["keyframeId"] for frame in valid], "keyframes": valid}
            discarded.extend(invalid)
            if not valid:
                missing.append(track_id)
        frames = [frame for group in grouped.values() for frame in group["keyframes"]]
        return {
            "ok": True,  # 调用成功
            "keyframeIds": [frame["keyframeId"] for frame in frames],  # 全部关键帧编号
            "keyframes": frames,  # 关键帧明细（含 isEmbedded 标记）
            "keyframesByTrack": grouped,  # 按轨迹分组的编号与明细
            "discardedKeyframeIds": discarded,  # 没有可用向量的关键帧编号
            "unsearchableTrackIds": missing,  # 一个有效关键帧都没有的轨迹
        }

    # 7.2 生成视频证据片段与海报
    def _build_clip(self, trackId: str | int, timeRange: tuple[str | float, str | float] | None = None, scale: float | None = None) -> dict[str, Any]:
        scope, problem = self._validate_time_range(timeRange)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem, "trackId": str(trackId)}  # 时间范围非法
        track = self.repository.get_track(trackId)
        if not track:
            return {"ok": False, "error": "track_not_found", "trackId": str(trackId)}  # 轨迹不存在
        start, end = track["startTime"], track["endTime"]
        monitor_scope = (max(start, scope[0]), min(end, scope[1])) if scope else (start, end)
        if monitor_scope[0] > monitor_scope[1]:
            return {"ok": True, "found": False, "trackId": str(trackId)}  # 时间窗与轨迹无交集
        trajectory_value = track.get("trajectoryPath")
        if not trajectory_value:
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}  # 轨迹缺轨迹文件字段
        trajectory_path = Path(trajectory_value)
        if not trajectory_path.is_file():
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}  # 轨迹文件不存在
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        source_value = trajectory.get("sourceVideoPath")
        source_path = Path(source_value) if source_value else None
        all_boxes = trajectory.get("boxes", [])
        if scope and any("observedAt" in item for item in all_boxes):
            boxes = [item for item in all_boxes if monitor_scope[0] <= float(item.get("observedAt") or 0) <= monitor_scope[1]]
        elif scope:
            # Older trajectory files only have source-video timestamps. Map the
            # requested monitoring window into that video-time domain instead of
            # silently rendering the full track.
            video_start = float(track.get("videoStartTime", start))
            video_end = float(track.get("videoEndTime", end))
            monitor_span = end - start
            if monitor_span > 0:
                scale_start = (monitor_scope[0] - start) / monitor_span
                scale_end = (monitor_scope[1] - start) / monitor_span
                selected_start = video_start + (video_end - video_start) * scale_start
                selected_end = video_start + (video_end - video_start) * scale_end
            else:
                selected_start, selected_end = video_start, video_end
            boxes = [item for item in all_boxes if selected_start <= float(item["timestamp"]) <= selected_end]
        else:
            boxes = all_boxes
        if source_path is None or not source_path.is_file() or not boxes:
            return {"ok": False, "error": "source_evidence_unavailable", "trackId": str(trackId)}  # 找不到源视频或没有框
        box_map = {int(item["frameIndex"]): item["bbox"] for item in boxes}
        evidence = self.config["pipeline"].get("evidence", {})
        media_scale = self._evidence_scale(scale)
        canvas_size = (
            self._even_size(max(160, round(evidence.get("clip_width", 640) * media_scale))),
            self._even_size(max(90, round(evidence.get("clip_height", 360) * media_scale))),
        )
        clip_fps = max(1.0, float(evidence.get("clip_fps", 15)))
        clip_crf = max(0, min(51, int(evidence.get("clip_crf", 31))))
        poster_quality = max(1, min(100, int(evidence.get("poster_quality", 75))))
        output_dir = Path(self.config["paths"]["clip_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        video_start_time = min(float(item["timestamp"]) for item in boxes)
        video_end_time = max(float(item["timestamp"]) for item in boxes)
        cache_key = (
            f"evidence-v2|{trackId}|{video_start_time:.3f}|{video_end_time:.3f}|"
            f"{trajectory_path.stat().st_mtime_ns}|{canvas_size[0]}x{canvas_size[1]}|"
            f"{clip_fps:.3f}|{clip_crf}|{poster_quality}"
        )
        segment_id = f"segment-{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()[:12]}"
        output_path = output_dir / f"{segment_id}.mp4"
        poster_path = output_dir / f"{segment_id}.jpg"
        time_fields = {
            "startTime": monitor_scope[0],  # 监控时间起点（epoch 秒）
            "endTime": monitor_scope[1],  # 监控时间终点（epoch 秒）
            "startTimeText": format_monitor_time(monitor_scope[0]),  # 监控起点文本
            "endTimeText": format_monitor_time(monitor_scope[1]),  # 监控终点文本
            "videoStartTime": video_start_time,  # 片段的源视频时间起点
            "videoEndTime": video_end_time,  # 片段的源视频时间终点
            "videoStartTimeText": format_monitor_time(video_start_time),  # 源视频起点文本
            "videoEndTimeText": format_monitor_time(video_end_time),  # 源视频终点文本
        }
        if output_path.is_file() and output_path.stat().st_size > 0:
            self._ensure_clip_poster(output_path, poster_path, poster_quality)
            return {  # 命中缓存：直接复用已有片段
                "ok": True,  # 调用成功
                "found": True,  # 找到证据片段
                "trackId": str(trackId),  # 轨迹编号
                "shipSegmentId": segment_id,  # 片段编号（由内容哈希生成）
                "segmentPath": str(output_path),  # 片段文件路径
                "posterPath": str(poster_path) if poster_path.is_file() else None,  # 海报文件路径
                "codec": "cached",  # 未重新编码
                **time_fields,  # 摊入监控时间与源视频时间
            }
        source_fps = max(1.0, float(trajectory.get("sourceFps") or 25))
        output_fps = min(source_fps, clip_fps)
        temporary_path = output_dir / f"{segment_id}.raw.mp4"
        temporary_path.unlink(missing_ok=True)
        writer, codec = self._open_video_writer(temporary_path, output_fps, canvas_size)
        capture = cv2.VideoCapture(str(source_path))
        if writer is None or not capture.isOpened():
            capture.release()
            if writer is not None:
                writer.release()
            temporary_path.unlink(missing_ok=True)
            return {"ok": False, "error": "segment_codec_unavailable", "trackId": str(trackId)}  # 打不开源视频或编码器
        written = 0
        last_written_frame: int | None = None
        frame_gap = max(1, round(source_fps / output_fps))
        poster_frame: np.ndarray | None = None
        poster_area = -1
        try:
            first, last = min(box_map), max(box_map)
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, first))
            for frame_index in range(first, last + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index not in box_map:
                    continue
                if last_written_frame is not None and frame_index - last_written_frame < frame_gap:
                    continue
                x1, y1, x2, y2 = self._clamp_box(box_map[frame_index], frame.shape)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                canvas = self._fit_crop_to_canvas(crop, canvas_size)
                writer.write(canvas)
                last_written_frame = frame_index
                crop_area = crop.shape[0] * crop.shape[1]
                if crop_area > poster_area:
                    poster_frame = canvas.copy()
                    poster_area = crop_area
                written += 1
        finally:
            capture.release()
            writer.release()
        if written == 0:
            temporary_path.unlink(missing_ok=True)
            return {"ok": False, "error": "empty_target_segment", "trackId": str(trackId)}  # 源视频里没渲染出任何帧
        codec = self._compress_evidence_clip(temporary_path, output_path, codec, clip_crf)
        self._write_poster(poster_path, poster_frame, poster_quality)
        return {  # 新生成的证据片段
            "ok": True,  # 调用成功
            "found": True,  # 找到证据片段
            "trackId": str(trackId),  # 轨迹编号
            "shipSegmentId": segment_id,  # 片段编号
            "segmentPath": str(output_path),  # 片段文件路径
            "posterPath": str(poster_path) if poster_path.is_file() else None,  # 海报文件路径
            "codec": codec,  # 实际使用的编码器
            **time_fields,  # 摊入监控时间与源视频时间
        }

    # 7.3 按比例生成图片预览
    def getImagePreview(self, sourcePath: str | Path, scale: float | None = None) -> Path:
        source_path = Path(sourcePath)
        media_scale = self._evidence_scale(scale)
        if media_scale >= 0.999:
            return source_path
        source_stat = source_path.stat()
        cache_key = f"image-preview-v1|{source_path.resolve()}|{source_stat.st_mtime_ns}|{source_stat.st_size}|{media_scale:.3f}"
        cache_dir = Path(self.config["paths"]["clip_dir"]) / "image-previews"
        preview_path = cache_dir / f"image-{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()[:16]}.jpg"
        if preview_path.is_file() and preview_path.stat().st_size > 0:
            return preview_path
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return source_path
        height, width = image.shape[:2]
        target_size = (max(1, round(width * media_scale)), max(1, round(height * media_scale)))
        preview = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        cache_dir.mkdir(parents=True, exist_ok=True)
        quality = max(1, min(100, int(self.config["pipeline"].get("evidence", {}).get("image_preview_quality", 82))))
        temporary_path = preview_path.with_name(f"{preview_path.stem}-{uuid.uuid4().hex[:8]}.tmp.jpg")
        try:
            if cv2.imwrite(str(temporary_path), preview, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                temporary_path.replace(preview_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return preview_path if preview_path.is_file() else source_path

    # ------------------------------------------------------------------ #
    # 八、图像匹配                                                             #
    # ------------------------------------------------------------------ #
    # 8.1 图像匹配与覆盖率统计的核心
    def _match_images(
        self,
        queryImages: list[dict[str, Any]] | dict[str, Any] | None = None,
        galleryImages: list[dict[str, Any]] | dict[str, Any] | None = None,
        topK: int | None = None,
        registryItems: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        queryImages = self._flatten_image_records(queryImages)
        galleryImages = self._flatten_image_records(galleryImages)

        def _registry_ids(value: Any) -> list[str]:
            found: list[str] = []
            def visit(node: Any) -> None:
                if isinstance(node, (list, tuple)):
                    for child in node:
                        visit(child)
                elif isinstance(node, dict):
                    if node.get("registryId") is not None:
                        found.append(str(node["registryId"]))
                    if isinstance(node.get("registryItems"), list):
                        visit(node["registryItems"])
            visit(value)
            return list(dict.fromkeys(found))

        declared_registry_ids = _registry_ids(registryItems)
        if not queryImages or not galleryImages:
            return {
                "ok": True,
                "matches": [],
                "unscoredTrackIds": [],
                "registryCoverageComplete": False,
            }

        def _is_keyframe(item: dict[str, Any]) -> bool:
            return item.get("trackId") is not None and item.get("keyframeId") is not None

        def _is_reference(item: dict[str, Any]) -> bool:
            return item.get("registryId") is not None and item.get("referenceId") is not None

        query_keyframes = [item for item in queryImages if _is_keyframe(item)]
        query_references = [item for item in queryImages if _is_reference(item)]
        gallery_keyframes = [item for item in galleryImages if _is_keyframe(item)]
        gallery_references = [item for item in galleryImages if _is_reference(item)]
        query_is_registry = bool(query_references) and bool(gallery_keyframes)
        query_is_track = (not query_is_registry) and bool(query_keyframes) and bool(gallery_references)
        if not query_is_track and not query_is_registry:
            return {
                "ok": False,
                "error": "one_keyframe_side_and_one_registry_side_required",
                "matches": [],
                "unscoredTrackIds": [],
                "registryCoverageComplete": False,
            }
        keyframes = query_keyframes if query_is_track else gallery_keyframes
        references = gallery_references if query_is_track else query_references

        frame_vectors = self._load_indexed_vectors(
            keyframes, vector_key="keyframeVectorId", index=self.vectors.keyframes
        )
        reference_vectors = self._load_indexed_vectors(
            references, vector_key="registryVectorId", index=self.vectors.registry
        )
        frames_by_track = self._group(keyframes, "trackId")
        refs_by_registry = self._group(references, "registryId")
        visual_registry_ids = {str(key) for key in refs_by_registry}
        all_registry_ids = declared_registry_ids or list(visual_registry_ids)
        total_registry_count = len(all_registry_ids)

        best_by_track: dict[str, dict[str, Any]] = {}
        compared_by_track: dict[str, set[str]] = {str(track_id): set() for track_id in frames_by_track}
        for track_id, track_frames in frames_by_track.items():
            for registry_id, registry_refs in refs_by_registry.items():
                frame_scores = []
                for frame in track_frames:
                    vid = frame.get("keyframeVectorId")
                    if vid is None or frame_vectors.get(vid) is None:
                        continue
                    frame_vector = frame_vectors[vid]
                    candidates = []
                    for reference in registry_refs:
                        rid = reference.get("registryVectorId")
                        if rid is None or reference_vectors.get(rid) is None:
                            continue
                        candidates.append((float(np.dot(frame_vector, reference_vectors[rid])), reference))
                    if candidates:
                        score, reference = max(candidates, key=lambda item: item[0])
                        frame_scores.append((score, frame, reference))
                top = sorted(frame_scores, key=lambda item: item[0], reverse=True)[:2]
                if not top:
                    continue
                registry_text = str(registry_id)
                compared_by_track[str(track_id)].add(registry_text)
                score = float(np.mean([item[0] for item in top]))
                matched_keyframes = [item[1]["keyframeId"] for item in top]
                matched_references = list(dict.fromkeys(
                    item[2]["referenceId"] for item in top if item[2].get("referenceId")
                ))
                candidate = {
                    "matchedTrackId": str(track_id),
                    "embeddingScore": round(score, 6), "scoreBand": self._band(score, "image"),
                    "matchedKeyframeIds": matched_keyframes, "matchedRegistryReferenceIds": matched_references,
                }
                if (
                    str(track_id) not in best_by_track
                    or candidate["embeddingScore"] > best_by_track[str(track_id)]["embeddingScore"]
                ):
                    best_by_track[str(track_id)] = candidate

        # 核心语义：每条轨迹先对所有库项比较，再只保留该轨迹的最高匹配分。
        matches = list(best_by_track.values())
        for item in matches:
            track_id = item["matchedTrackId"]
            if (
                len(compared_by_track[track_id]) != total_registry_count
                and item["scoreBand"] == "mismatch"
            ):
                item["scoreBand"] = "uncertain"
        matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        if topK is not None:
            try:
                match_limit = max(0, int(topK))
            except (TypeError, ValueError):
                match_limit = 0
            if match_limit > 0:
                matches = matches[:match_limit]
        unscored_track_ids = [str(tid) for tid in frames_by_track if str(tid) not in best_by_track]
        registry_coverage_complete = bool(total_registry_count and frames_by_track) and all(
            len(ids) == total_registry_count for ids in compared_by_track.values()
        )
        return {
            "ok": True,
            "matches": matches,
            "unscoredTrackIds": unscored_track_ids,
            "registryCoverageComplete": registry_coverage_complete,
        }

    # ------------------------------------------------------------------ #
    # 九、证据汇总                                                             #
    # ------------------------------------------------------------------ #
    # 9.1 证据汇总的内部实现
    def _show_evidence(
        self,
        track_ids: list[str] | None = None,
        keyframe_ids: list[str] | None = None,
        registry_reference_ids: list[str] | None = None,
        time_range: tuple[str | float, str | float] | None = None,
        max_items: int | None = None,
    ) -> dict[str, Any]:
        scope, problem = self._validate_time_range(time_range)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem}
        limit = max(1, min(20, int(max_items or self.MAX_RETURNED_MATCHES)))
        track_ids = list(dict.fromkeys(str(value) for value in (track_ids or []) if value))[:limit]
        keyframe_ids = list(dict.fromkeys(str(value) for value in (keyframe_ids or []) if value))
        registry_reference_ids = list(dict.fromkeys(str(value) for value in (registry_reference_ids or []) if value))
        generated_segments: list[str] = []
        shown_keyframes = list(keyframe_ids)
        for track_id in track_ids:
            frames = self._build_frames([track_id]).get("keyframes", [])
            if frames:
                selected = frames[: self.MAX_FRAMES_PER_TRACK]
                # 该轨迹展示的关键帧
                shown_keyframes.extend(str(frame["keyframeId"]) for frame in selected)
            clip = self._build_clip(track_id, scope)
            if clip.get("ok") and clip.get("found") and clip.get("shipSegmentId"):
                # 该轨迹生成的视频片段
                generated_segments.append(str(clip["shipSegmentId"]))
        shown_keyframes = list(dict.fromkeys(shown_keyframes))[:limit * self.MAX_FRAMES_PER_TRACK]
        shown_segments = list(dict.fromkeys(generated_segments))[:limit]
        shown_references = self._representative_registry_reference_ids(registry_reference_ids)[:limit]
        return {
            "ok": True,  # 调用成功
            "shownKeyframeIds": shown_keyframes,  # 展示的关键帧编号
            "shownShipSegmentIds": shown_segments,  # 展示的视频片段编号
            "shownRegistryReferenceIds": shown_references,  # 展示的库参考图编号
            "trackIds": track_ids,  # 汇总的轨迹编号
            "statistics": {  # 展示数量统计
                "trackCount": len(track_ids),  # 轨迹数
                "keyframeCount": len(shown_keyframes),  # 关键帧数
                "clipCount": len(shown_segments),  # 片段数
                "registryReferenceCount": len(shown_references),  # 参考图数
            },
        }

    # 9.2 证据汇总工具的入口
    def showEvidence(
        self,
        trackIds: list[str] | None = None,
        keyframeIds: list[str] | None = None,
        registryReferenceIds: list[str] | None = None,
        timeRange: tuple[str | float, str | float] | None = None,
        maxItems: int | None = None,
    ) -> dict[str, Any]:
        return self._run_tool(
            "show_evidence",
            lambda: self._show_evidence(trackIds, keyframeIds, registryReferenceIds, timeRange, maxItems),
        )

    # ------------------------------------------------------------------ #
    # 十、轨迹去重                                                             #
    # ------------------------------------------------------------------ #
    # 10.1 按轨迹与关键帧合并同一艘船
    def _deduplicate_tracks(
        self,
        tracks: list[dict[str, Any]],
        keyframes_by_track: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        candidates: dict[str, list[dict[str, Any]]] = {}
        for track in tracks:
            track_id = str(track["trackId"])
            group = keyframes_by_track.get(track_id, {})
            frames = group.get("keyframes", [])
            valid_frames: list[dict[str, Any]] = []
            for frame in frames:
                if not isinstance(frame, dict) or frame.get("keyframeVectorId") is None:
                    continue
                normalized_frame = dict(frame)
                try:
                    normalized_frame["keyframeVectorId"] = int(normalized_frame["keyframeVectorId"])
                except (TypeError, ValueError):
                    continue
                valid_frames.append(normalized_frame)
            candidates[track_id] = valid_frames
        track_map = {str(item["trackId"]): item for item in tracks}
        frames = [frame for group in candidates.values() for frame in group]
        vectors = self._load_indexed_vectors(
            frames, vector_key="keyframeVectorId", index=self.vectors.keyframes
        )
        selected, missing = {}, []
        for track_id, frames in candidates.items():
            available = [frame for frame in frames if frame["keyframeVectorId"] in vectors]
            selected[track_id] = sorted(
                available,
                key=lambda item: float(item.get("retentionScore") or 0.0),
                reverse=True,
            )[:3]
            if not selected[track_id]:
                missing.append(track_id)
        pair_scores: dict[tuple[str, str], float] = {}
        track_ids = list(track_map)
        for index, left_id in enumerate(track_ids):
            for right_id in track_ids[index + 1:]:
                if self._overlap(track_map[left_id], track_map[right_id]) or not selected[left_id] or not selected[right_id]:
                    continue
                scores = []
                for left in selected[left_id]:
                    for right in selected[right_id]:
                        left_vector = vectors.get(int(left["keyframeVectorId"]))
                        right_vector = vectors.get(int(right["keyframeVectorId"]))
                        if left_vector is not None and right_vector is not None:
                            scores.append(float(np.dot(left_vector, right_vector)))
                if scores:
                    pair_scores[tuple(sorted((left_id, right_id)))] = float(np.mean(sorted(scores, reverse=True)[:3]))
        high = float(self.settings["dedup_high"])
        low = float(self.settings["dedup_low"])
        high_groups = self._complete_link(track_ids, track_map, pair_scores, high)
        low_groups = self._complete_link(track_ids, track_map, pair_scores, low)

        # 高阈值只接受确定合并，因此对应保守（较大）船数；低阈值纳入待确认合并，对应最小船数。
        pending_merge_count = 0
        for low_group in low_groups:
            low_group_ids = set(low_group)
            high_group_count = sum(
                set(high_group).issubset(low_group_ids) for high_group in high_groups
            )
            pending_merge_count += high_group_count > 1
        status = "incomplete" if missing else "stable" if len(high_groups) == len(low_groups) else "sensitive"
        return {
            "highGroups": high_groups,
            "lowGroups": low_groups,
            "pendingMergeCount": pending_merge_count,
            "countStability": status,
            "unsearchableTrackIds": missing,
        }

    # ------------------------------------------------------------------ #
    # 十一、通用工具                                                            #
    # ------------------------------------------------------------------ #
    # 11.1 拍平 references / keyframes 等图片记录
    @classmethod
    def _flatten_image_records(cls, value: Any) -> list[dict[str, Any]]:
        """兼容工具结果中的 references、keyframes 和按轨迹分组的关键帧。"""
        records: list[dict[str, Any]] = []

        def visit(item: Any) -> None:
            if isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            if isinstance(item.get("registryReferences"), list):
                visit(item["registryReferences"])
                return
            if isinstance(item.get("references"), list):
                visit(item["references"])
                return
            if isinstance(item.get("keyframes"), list):
                visit(item["keyframes"])
                return
            if isinstance(item.get("keyframesByTrack"), dict):
                visit(list(item["keyframesByTrack"].values()))
                return
            records.append(item)

        visit(value)
        return records

    # 11.2 读取已存在的图像向量
    def _load_indexed_vectors(
        self,
        records: list[dict[str, Any]],
        *,
        vector_key: str,
        index: Any,
    ) -> dict[int, np.ndarray]:
        """只读取已有向量；缺失项保持不可检索，由调用方标记覆盖不完整。"""
        ids = []
        for item in records:
            if not isinstance(item, dict) or item.get(vector_key) in (None, ""):
                continue
            try:
                vector_id = int(item[vector_key])
            except (TypeError, ValueError):
                item[vector_key] = None
                continue
            if vector_id < 0:
                item[vector_key] = None
                continue
            item[vector_key] = vector_id
            ids.append(vector_id)
        if not ids:
            return {}
        try:
            return dict(index.get_many(dict.fromkeys(ids)))
        except Exception:  # noqa: BLE001 - an unavailable index leaves images unsearchable.
            return {}

    # 11.3 返回本轮的确认/排除/灰区阈值
    def _match_thresholds(self, mode: str) -> dict[str, Any]:
        """返回本轮匹配实际使用的确认、排除与灰区边界。"""
        confirmation = float(self.settings[f"{mode}_match"])
        exclusion = float(self.settings[f"{mode}_exclude"])
        return {
            "mode": mode,  # 匹配模式 text / image
            "confirmation": confirmation,  # 确认阈值
            "exclusion": exclusion,  # 排除阈值
            "grayZone": {  # 灰区区间，两端都不含
                "lower": exclusion,  # 下界
                "upper": confirmation,  # 上界
                "lowerInclusive": False,  # 不含下界
                "upperInclusive": False,  # 不含上界
            },
        }

    # 11.4 把分数映射成档位
    def _band(self, score: float, mode: str) -> str:
        thresholds = self._match_thresholds(mode)
        match = float(thresholds["confirmation"])
        exclude = float(thresholds["exclusion"])
        return "match" if score >= match else "mismatch" if score <= exclude else "uncertain"

    # 11.5 按指定键分组
    @staticmethod
    def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(str(item[key]), []).append(item)
        return grouped

    # 11.6 挑选代表性参考图
    def _representative_registry_reference_ids(self, reference_ids: list[str]) -> list[str]:
        ordered_ids = list(dict.fromkeys(str(value) for value in reference_ids if value))
        reference_map = {item["referenceId"]: item for item in self.repository.references_by_ids(ordered_ids)}
        hull_by_registry = {
            str(item["registryId"]): normalize_hull_number(item.get("hullNumber")) or str(item["registryId"])
            for item in self.repository.list_registry()
        }
        selected, seen = [], set()
        for reference_id in ordered_ids:
            reference = reference_map.get(reference_id)
            registry_id = str(reference.get("registryId")) if reference else reference_id
            owner = hull_by_registry.get(registry_id, registry_id)
            if owner in seen:
                continue
            seen.add(owner)
            selected.append(reference_id)
        return selected

    # 11.7 把裁剪框限制在画面内
    @staticmethod
    def _clamp_box(box: list[int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = map(int, box)
        return max(0, min(x1, width - 1)), max(0, min(y1, height - 1)), max(1, min(x2, width)), max(1, min(y2, height))

    # 11.8 打开视频写入器
    @staticmethod
    def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any | None, str | None]:
        # Windows 常见 OpenCV 包通常没有 H264 编码器；优先 mp4v，避免片段生成失败和控制台刷错。
        for codec in ("mp4v", "avc1", "H264"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return writer, codec
            writer.release()
            path.unlink(missing_ok=True)
        return None, None

    # 11.9 证据画布的缩放比例
    @staticmethod
    def _evidence_scale(value: float | None) -> float:
        try:
            scale = float(value if value is not None else 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        return max(0.25, min(1.0, scale))

    # 11.10 把尺寸取成偶数
    @staticmethod
    def _even_size(value: int) -> int:
        value = max(2, int(value))
        return value if value % 2 == 0 else value + 1

    # 11.11 把裁剪图适配到画布
    @staticmethod
    def _fit_crop_to_canvas(crop: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        canvas_width, canvas_height = canvas_size
        content_width = max(2, int(canvas_width * 0.94))
        content_height = max(2, int(canvas_height * 0.94))
        crop_height, crop_width = crop.shape[:2]
        scale = min(content_width / max(1, crop_width), content_height / max(1, crop_height))
        resized_width = max(1, min(content_width, round(crop_width * scale)))
        resized_height = max(1, min(content_height, round(crop_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.full((canvas_height, canvas_width, 3), (5, 14, 20), dtype=np.uint8)
        offset_x = (canvas_width - resized_width) // 2
        offset_y = (canvas_height - resized_height) // 2
        canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        return canvas

    # 11.12 把一帧写成海报图
    @staticmethod
    def _write_poster(path: Path, frame: np.ndarray | None, quality: int) -> bool:
        if frame is None or frame.size == 0:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]))

    # 11.13 确保片段有封面
    @classmethod
    def _ensure_clip_poster(cls, clip_path: Path, poster_path: Path, quality: int) -> bool:
        if poster_path.is_file() and poster_path.stat().st_size > 0:
            return True
        capture = cv2.VideoCapture(str(clip_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        return cls._write_poster(poster_path, frame if ok else None, quality)

    # 11.14 压缩证据片段
    @staticmethod
    def _compress_evidence_clip(source_path: Path, output_path: Path, codec: str | None, crf: int) -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        converted = output_path.with_name(f"{output_path.stem}.compressing.mp4")
        if ffmpeg:
            try:
                result = subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(source_path), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted),
                ], capture_output=True, timeout=120, check=False)
                if result.returncode == 0 and converted.is_file() and converted.stat().st_size > 0:
                    converted.replace(output_path)
                    source_path.unlink(missing_ok=True)
                    return "h264"
            except (OSError, subprocess.SubprocessError):
                pass
        converted.unlink(missing_ok=True)
        source_path.replace(output_path)
        return codec

    # 11.15 判断两个时间区间是否重叠（去重辅助）
    @staticmethod
    def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return not (float(left["endTime"]) < float(right["startTime"]) or float(right["endTime"]) < float(left["startTime"]))

    # 11.16 补全可合并的轨迹链（去重辅助）
    def _complete_link(self, track_ids: list[str], tracks: dict[str, dict[str, Any]], scores: dict[tuple[str, str], float], threshold: float) -> list[list[str]]:
        groups = [{track_id} for track_id in track_ids]
        def score(left: str, right: str) -> float | None:
            return scores.get(tuple(sorted((left, right))))
        while True:
            best: tuple[float, int, int] | None = None
            for left_index in range(len(groups)):
                for right_index in range(left_index + 1, len(groups)):
                    cross = [(left, right) for left in groups[left_index] for right in groups[right_index]]
                    values = [score(left, right) for left, right in cross]
                    if any(self._overlap(tracks[left], tracks[right]) for left, right in cross) or any(value is None for value in values):
                        continue
                    minimum = min(float(value) for value in values)
                    if minimum >= threshold and (best is None or minimum > best[0]):
                        best = (minimum, left_index, right_index)
            if best is None:
                break
            _, left_index, right_index = best
            groups[left_index] |= groups[right_index]
            groups.pop(right_index)
        return [sorted(group) for group in groups]

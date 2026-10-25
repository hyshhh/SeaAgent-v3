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

    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.settings = self.config["pipeline"]["retrieval"]

    # 1.1 业务工具在服务层自行执行时间预算；运行时只负责调用和记录结果。
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
            return {
                "ok": False,
                "error": "tool_timeout",
                "tool": tool_name,
                "timeoutSeconds": float(timeout),
                "hint": f"{tool_name} 超过 {float(timeout):.0f} 秒，已返回超时结果。",
            }
        if "error" in result:
            error = result["error"]
            return {"ok": False, "error": "tool_failed", "tool": tool_name, "hint": str(error)}
        value = result.get("value")
        return value if isinstance(value, dict) else {"ok": True, "value": value}

    def _run_tool(self, tool_name: str, operation: Any) -> dict[str, Any]:
        return self._run_with_timeout(operation, self.TOOL_TIMEOUTS[tool_name], tool_name)

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
                searchable = [
                    frame for frame in frames
                    if frame.get("keyframeVectorId") is not None or frame.get("keyframePath")
                ]
                vectors, _ = self._ensure_image_vectors(
                    searchable,
                    vector_key="keyframeVectorId",
                    path_keys=("keyframePath", "imagePath", "path"),
                    index=self.vectors.keyframes,
                )
                searchable = [
                    frame for frame in searchable
                    if frame.get("keyframeVectorId") is not None
                    and int(frame["keyframeVectorId"]) in vectors
                ]
                grouped[track_id] = searchable
                if not searchable:
                    missing.append(track_id)
        return grouped, missing

    @staticmethod
    def _flatten_frames(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [frame for frames in groups.values() for frame in frames]

    def _result_scope(
        self,
        scope: tuple[float, float] | None,
        tracks: list[dict[str, Any]],
        complete: bool,
    ) -> dict[str, Any]:
        return {
            "queryScope": [
                datetime.fromtimestamp(value, self.DEFAULT_TIMEZONE).isoformat()
                for value in scope
            ]
            if scope is not None
            else None,
            "searchedTrackCount": len(tracks),
            "coverageComplete": complete,
        }

    def _track_match_summary(
        self,
        track: dict[str, Any],
        frames: list[dict[str, Any]],
        *,
        score: float | None = None,
        band: str = "mismatch",
        reference_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        confidence = {"match": "confirmed", "uncertain": "unconfirmed", "mismatch": "mismatch"}.get(band, "unconfirmed")
        return {
            "trackId": str(track["trackId"]),
            "startTime": track.get("startTime"),
            "endTime": track.get("endTime"),
            "startTimeText": track.get("startTimeText"),
            "endTimeText": track.get("endTimeText"),
            "hullNumber": track.get("finalHullNumber"),
            "description": track.get("finalDescription", ""),
            "score": round(float(score), 6) if score is not None else None,
            "confidence": confidence,
            "scoreBand": band,
            "keyframeIds": [str(frame["keyframeId"]) for frame in frames[: self.MAX_FRAMES_PER_TRACK]],
            "registryReferenceIds": list(dict.fromkeys(reference_ids or []))[: self.MAX_FRAMES_PER_TRACK],
        }

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
            vectors, _ = self._ensure_image_vectors(
                gallery,
                vector_key="keyframeVectorId",
                path_keys=("keyframePath", "imagePath", "path"),
                index=self.vectors.keyframes,
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
            vectors, _ = self._ensure_image_vectors(
                frames,
                vector_key="keyframeVectorId",
                path_keys=("keyframePath", "imagePath", "path"),
                index=self.vectors.keyframes,
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
        filter_missed = bool(hull) and not any(
            normalize_hull_number(track.get("finalHullNumber")) == hull for track in tracks
        )
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
            filter_missed = filter_missed and not keyframe_hull_matches
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
        candidates = [item for item in candidates if item["confidence"] != "mismatch"]
        candidates.sort(key=lambda item: (item["confidence"] != "confirmed", -(item["score"] or -1)))
        selected = candidates[: max(1, min(self.MAX_RETURNED_MATCHES, int(top_k or self.MAX_RETURNED_MATCHES)))]
        confirmed = [item for item in selected if item["confidence"] == "confirmed"]
        uncertain = [item for item in selected if item["confidence"] == "unconfirmed"]
        search_complete = search_complete and not missing
        status = "confirmed" if confirmed else "unconfirmed" if uncertain or not search_complete else "mismatch"
        return {
            "ok": True,
            "task": "direct_recognition",
            "status": status,
            "confidence": status,
            "matches": selected,
            "confirmedMatches": confirmed,
            "uncertainMatches": uncertain,
            "mismatch": status == "mismatch",
            "hullNumber": hull or None,
            "filterMissed": filter_missed,
            "searchCoverageComplete": search_complete,
            "nextStep": "match_registry_tracks" if hull and status != "confirmed" else None,
            "unsearchableTrackIds": missing[: self.MAX_RETURNED_MATCHES],
            "trackIds": [item["trackId"] for item in selected],
            "statistics": {
                "searchedTrackCount": len(tracks),
                "candidateCount": len(selected),
                "confirmedCount": len(confirmed),
                "unconfirmedCount": len(uncertain),
                "unsearchableTrackCount": len(missing),
            },
            **self._result_scope(scope, tracks, search_complete),
        }

    def recognizeTracks(self, hullNumber: str | None = None, description: str | None = None, imagePath: str | None = None, timeRange: tuple[str | float, str | float] | None = None, topK: int | None = None) -> dict[str, Any]:
        return self._run_tool(
            "recognize_tracks",
            lambda: self._recognize_tracks(hullNumber, description, imagePath, timeRange, topK),
        )

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
            return {
                "ok": True,
                "task": "registry_assisted_recognition",
                "status": "unconfirmed",
                "confidence": "unconfirmed",
                "registryFound": bool(registry_items),
                "registryReferenceIds": [str(item["referenceId"]) for item in references[: self.MAX_RETURNED_MATCHES]],
                "matches": [],
                "unsearchableTrackIds": missing[: self.MAX_RETURNED_MATCHES],
                "trackIds": [],
                "searchCoverageComplete": coverage_complete,
                "statistics": {
                    "searchedTrackCount": len(tracks),
                    "candidateCount": 0,
                    "confirmedCount": 0,
                    "unconfirmedCount": 0,
                    "registryReferenceCount": len(references),
                    "unsearchableTrackCount": len(missing),
                },
                **self._result_scope(scope, tracks, coverage_complete),
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
        matches.sort(key=lambda item: (item["confidence"] != "confirmed", -(item["score"] or -1)))
        matches = matches[: max(1, min(self.MAX_RETURNED_MATCHES, int(top_k or self.MAX_RETURNED_MATCHES)))]
        confirmed = [item for item in matches if item["confidence"] == "confirmed"]
        uncertain = [item for item in matches if item["confidence"] == "unconfirmed"]
        missing = list(dict.fromkeys(missing + [str(value) for value in result.get("unscoredTrackIds", [])]))
        visual_complete = bool(result.get("registryCoverageComplete")) and complete and not missing
        status = "confirmed" if confirmed else "unconfirmed" if uncertain or not visual_complete else "mismatch"
        return {
            "ok": True,
            "task": "registry_assisted_recognition",
            "status": status,
            "confidence": status,
            "registryFound": True,
            "registryReferenceIds": [str(item["referenceId"]) for item in references[: self.MAX_RETURNED_MATCHES]],
            "matches": matches,
            "confirmedMatches": confirmed,
            "uncertainMatches": uncertain,
            "unsearchableTrackIds": missing[: self.MAX_RETURNED_MATCHES],
            "trackIds": [item["trackId"] for item in matches],
            "statistics": {
                "searchedTrackCount": len(tracks),
                "candidateCount": len(matches),
                "confirmedCount": len(confirmed),
                "unconfirmedCount": len(uncertain),
                "registryReferenceCount": len(references),
                "unsearchableTrackCount": len(missing),
            },
            "searchCoverageComplete": visual_complete,
            **self._result_scope(scope, tracks, visual_complete),
        }

    def matchRegistryTracks(self, hullNumber: str, timeRange: tuple[str | float, str | float] | None = None, topK: int | None = None) -> dict[str, Any]:
        return self._run_tool(
            "match_registry_tracks",
            lambda: self._match_registry_tracks(hullNumber, timeRange, topK),
        )

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
                return {"ok": False, "error": "two_images_required", "decision": "uncertain"}
            if reference_id:
                references = self.repository.references_by_ids([reference_id])
                reference = str(references[0].get("imagePath") or "") if references else ""
                if not reference:
                    return {
                        "ok": False,
                        "error": "registry_reference_not_found",
                        "decision": "uncertain",
                        "registryReferenceId": reference_id,
                    }
            if keyframe_id:
                frame = self.repository.get_keyframe(keyframe_id)
                candidate = str(frame.get("keyframePath") or "") if frame else ""
                if not candidate:
                    return {"ok": False, "error": "keyframe_not_found", "decision": "uncertain"}
            if not Path(reference).is_file() or not Path(candidate).is_file():
                return {"ok": False, "error": "image_not_found", "decision": "uncertain"}
            result = self.llm.verify([reference], [candidate])
            return {
                "ok": True,
                "task": "vlm_image_verification",
                "registryReferenceId": reference_id or None,
                "keyframeId": keyframe_id or None,
                **result,
            }

        return self._run_tool("verify_vessel_images", verify)

    def _entity_analysis(self, time_range: tuple[str | float, str | float] | None = None, compare_registry: bool = True) -> dict[str, Any]:
        scope, problem = self._validate_time_range(time_range)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem}
        tracks, complete = self._scan_tracks(scope)
        frames, missing = self._load_search_frames(tracks)
        dedup = self._deduplicate_tracks(tracks, {track_id: {"keyframes": values} for track_id, values in frames.items()})
        missing = list(dict.fromkeys(missing + dedup.get("unsearchableTrackIds", [])))
        groups = dedup.get("highGroups") or [[str(track["trackId"])] for track in tracks]
        pending_merge_count = int(dedup.get("pendingMergeCount") or 0)
        dedup_stable = str(dedup.get("countStability") or "stable") == "stable"
        track_by_id = {str(track["trackId"]): track for track in tracks}
        group_items = [{"entityId": f"entity-{index + 1}", "trackIds": group, "status": "unknown"} for index, group in enumerate(groups)]
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
        }
        for entity in group_items:
            statuses = []
            entity["keyframeIds"] = []
            entity["registryReferenceIds"] = []
            for track_id in entity["trackIds"]:
                track = track_by_id.get(str(track_id), {})
                hull = normalize_hull_number(track.get("finalHullNumber"))
                track_frames = frames.get(str(track_id), [])
                entity["keyframeIds"].extend(
                    str(frame["keyframeId"])
                    for frame in track_frames[:2]
                    if frame.get("keyframeId")
                )
                visual = visual_matches.get(str(track_id), {})
                entity["registryReferenceIds"].extend(
                    str(value) for value in visual.get("matchedRegistryReferenceIds", []) if value
                )
                hull_frame_match = any(
                    normalize_hull_number(frame.get("vlmHullNumber")) in registry_hulls
                    and float(frame.get("readabilityConfidence") or 0) >= 0.75
                    for frame in track_frames
                )
                if hull in registry_hulls or hull_frame_match or visual.get("scoreBand") == "match":
                    statuses.append("in_registry")
                elif hull or hull_frame_match or not registry_items or (
                    visual_coverage_complete
                    and str(track_id) not in missing
                    and str(track_id) not in visual_unscored
                ):
                    statuses.append("unregistered")
                else:
                    statuses.append("uncertain")
            entity["status"] = (
                "in_registry" if "in_registry" in statuses
                else "unregistered" if statuses and all(item == "unregistered" for item in statuses)
                else "uncertain"
            ) if compare_registry else "unknown"
            entity["keyframeIds"] = list(dict.fromkeys(entity["keyframeIds"]))[: self.MAX_FRAMES_PER_TRACK * len(entity["trackIds"])]
            entity["registryReferenceIds"] = list(dict.fromkeys(entity["registryReferenceIds"]))
        in_registry = sum(item["status"] == "in_registry" for item in group_items)
        unregistered = sum(item["status"] == "unregistered" for item in group_items)
        uncertain = sum(item["status"] == "uncertain" for item in group_items)
        count_uncertain = uncertain if compare_registry else sum(
            1 for item in group_items if any(str(track_id) in missing for track_id in item["trackIds"])
        )
        count_uncertain += pending_merge_count
        registry_incomplete = compare_registry and bool(registry_items) and (
            not references or not visual_coverage_complete
        )
        count_incomplete = bool(
            missing or not complete or not dedup_stable or pending_merge_count or registry_incomplete
        )
        return {
            "ok": True,
            "entityCount": len(group_items),
            "minimumEntityCount": len(dedup.get("lowGroups") or groups),
            "maximumEntityCount": len(dedup.get("highGroups") or groups),
            "trackCount": len(tracks),
            "inRegistryCount": in_registry,
            "unregisteredCount": unregistered,
            "uncertainCount": uncertain,
            "countUncertain": count_uncertain,
            "countIsUpperBound": count_incomplete,
            "countConfidence": "uncertain" if count_incomplete else "confirmed",
            "countStability": dedup.get("countStability", "stable"),
            "pendingMergeCount": pending_merge_count,
            "unsearchableTrackIds": missing[: self.MAX_RETURNED_MATCHES],
            "registryCompared": compare_registry,
            "countType": "total" if not compare_registry else "membership",
            "trackIds": list(dict.fromkeys(
                str(track_id) for entity in group_items
                for track_id in entity["trackIds"]
            ))[: self.MAX_RETURNED_MATCHES],
            "keyframeIds": list(dict.fromkeys(
                keyframe_id for entity in group_items
                for keyframe_id in entity.get("keyframeIds", [])
            ))[: self.MAX_RETURNED_MATCHES * self.MAX_FRAMES_PER_TRACK],
            "registryReferenceIds": list(dict.fromkeys(
                reference_id for entity in group_items
                for reference_id in entity.get("registryReferenceIds", [])
            ))[: self.MAX_RETURNED_MATCHES],
            "statistics": {
                "trackCount": len(tracks),
                "entityCount": len(group_items),
                "minimumEntityCount": len(dedup.get("lowGroups") or groups),
                "maximumEntityCount": len(dedup.get("highGroups") or groups),
                "inRegistryCount": in_registry,
                "unregisteredCount": unregistered,
                "uncertainCount": uncertain,
                "countUncertain": count_uncertain,
                "pendingMergeCount": pending_merge_count,
                "countStability": dedup.get("countStability", "stable"),
                "countConfidence": "uncertain" if count_incomplete else "confirmed",
                "unsearchableTrackCount": len(missing),
            },
            **self._result_scope(scope, tracks, not count_incomplete),
        }

    def countTotalEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_total_entities",
            lambda: self._entity_count(timeRange, False, "entityCount"),
        )

    def countRegistryEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_registry_entities",
            lambda: self._entity_count(timeRange, True, "inRegistryCount"),
        )

    def countUnregisteredEntities(self, timeRange: tuple[str | float, str | float] | None = None) -> dict[str, Any]:
        return self._run_tool(
            "count_unregistered_entities",
            lambda: self._entity_count(timeRange, True, "unregisteredCount"),
        )

    def _entity_count(self, time_range: Any, compare_registry: bool, field: str) -> dict[str, Any]:
        result = self._entity_analysis(time_range, compare_registry)
        if result.get("ok"):
            result["count"] = result[field]
            result.setdefault("statistics", {})["count"] = result[field]
        return result

    # epoch 秒的合理下限（2017 年）：早于此说明模型在编数字，而不是在描述监控录像
    MIN_PLAUSIBLE_EPOCH = datetime(2017, 1, 1, tzinfo=DEFAULT_TIMEZONE).timestamp()

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
            "ok": True,
            "keyframeIds": [frame["keyframeId"] for frame in frames],
            "keyframes": frames,
            "keyframesByTrack": grouped,
            "discardedKeyframeIds": discarded,
            "unsearchableTrackIds": missing,
        }

    def _build_clip(self, trackId: str | int, timeRange: tuple[str | float, str | float] | None = None, scale: float | None = None) -> dict[str, Any]:
        scope, problem = self._validate_time_range(timeRange)
        if problem:
            return {"ok": False, "error": "invalid_time_range", "hint": problem, "trackId": str(trackId)}
        track = self.repository.get_track(trackId)
        if not track:
            return {"ok": False, "error": "track_not_found", "trackId": str(trackId)}
        start, end = track["startTime"], track["endTime"]
        monitor_scope = (max(start, scope[0]), min(end, scope[1])) if scope else (start, end)
        if monitor_scope[0] > monitor_scope[1]:
            return {"ok": True, "found": False, "trackId": str(trackId)}
        trajectory_value = track.get("trajectoryPath")
        if not trajectory_value:
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}
        trajectory_path = Path(trajectory_value)
        if not trajectory_path.is_file():
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}
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
            return {"ok": False, "error": "source_evidence_unavailable", "trackId": str(trackId)}
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
            "startTime": monitor_scope[0],
            "endTime": monitor_scope[1],
            "startTimeText": format_monitor_time(monitor_scope[0]),
            "endTimeText": format_monitor_time(monitor_scope[1]),
            "videoStartTime": video_start_time,
            "videoEndTime": video_end_time,
            "videoStartTimeText": format_monitor_time(video_start_time),
            "videoEndTimeText": format_monitor_time(video_end_time),
        }
        if output_path.is_file() and output_path.stat().st_size > 0:
            self._ensure_clip_poster(output_path, poster_path, poster_quality)
            return {
                "ok": True,
                "found": True,
                "trackId": str(trackId),
                "shipSegmentId": segment_id,
                "segmentPath": str(output_path),
                "posterPath": str(poster_path) if poster_path.is_file() else None,
                "codec": "cached",
                **time_fields,
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
            return {"ok": False, "error": "segment_codec_unavailable", "trackId": str(trackId)}
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
            return {"ok": False, "error": "empty_target_segment", "trackId": str(trackId)}
        codec = self._compress_evidence_clip(temporary_path, output_path, codec, clip_crf)
        self._write_poster(poster_path, poster_frame, poster_quality)
        return {
            "ok": True,
            "found": True,
            "trackId": str(trackId),
            "shipSegmentId": segment_id,
            "segmentPath": str(output_path),
            "posterPath": str(poster_path) if poster_path.is_file() else None,
            "codec": codec,
            **time_fields,
        }

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
                "ok": True, "matchMode": "image_to_image", "queryType": None, "matches": [],
                "bestMatchesAscending": [], "missingKeyframeIds": [], "missingRegistryReferenceIds": [],
                "visualAttempted": True, "totalTrackCount": 0, "scoredTrackCount": 0,
                "totalRegistryCount": len(declared_registry_ids), "visualRegistryCount": 0,
                "scoredRegistryCount": 0, "registryCoverageRatio": 0.0,
                "pairCoverageRatio": 0.0, "registryCoverageComplete": False,
                "matchThresholds": self._match_thresholds("image"),
                "hint": "queryImages 或 galleryImages 为空",
            }

        # 有本地路径但没有向量编号的图，也允许进入现场编码流程。
        def _has_path(item: dict[str, Any], keys: tuple[str, ...]) -> bool:
            return any(item.get(key) for key in keys)
        def _is_keyframe(item: dict[str, Any]) -> bool:
            return item.get("trackId") is not None and (
                item.get("keyframeVectorId") is not None or _has_path(item, ("keyframePath", "imagePath", "path"))
            )
        def _is_reference(item: dict[str, Any]) -> bool:
            return item.get("registryId") is not None and (
                item.get("registryVectorId") is not None or _has_path(item, ("imagePath", "keyframePath", "path"))
            )

        query_keyframes = [item for item in queryImages if _is_keyframe(item)]
        query_references = [item for item in queryImages if _is_reference(item)]
        gallery_keyframes = [item for item in galleryImages if _is_keyframe(item)]
        gallery_references = [item for item in galleryImages if _is_reference(item)]
        query_is_registry = bool(query_references) and bool(gallery_keyframes)
        query_is_track = (not query_is_registry) and bool(query_keyframes) and bool(gallery_references)
        if not query_is_track and not query_is_registry:
            return {
                "ok": False, "error": "one_keyframe_side_and_one_registry_side_required",
                "matchMode": "image_to_image", "matches": [], "bestMatchesAscending": [],
                "visualAttempted": True, "matchThresholds": self._match_thresholds("image"),
                "hint": f"query: kf={len(query_keyframes)} ref={len(query_references)}; gallery: kf={len(gallery_keyframes)} ref={len(gallery_references)}",
            }
        keyframes = query_keyframes if query_is_track else gallery_keyframes
        references = gallery_references if query_is_track else query_references

        frame_vectors, recovered_frames = self._ensure_image_vectors(
            keyframes, vector_key="keyframeVectorId", path_keys=("keyframePath", "imagePath", "path"),
            index=self.vectors.keyframes,
        )
        reference_vectors, recovered_refs = self._ensure_image_vectors(
            references, vector_key="registryVectorId", path_keys=("imagePath", "keyframePath", "path"),
            index=self.vectors.registry,
        )
        missing_keyframes = [item.get("keyframeId") for item in keyframes if item.get("keyframeVectorId") is None or int(item["keyframeVectorId"]) not in frame_vectors]
        missing_references = [item.get("referenceId") for item in references if item.get("registryVectorId") is None or int(item["registryVectorId"]) not in reference_vectors]
        frames_by_track = self._group(keyframes, "trackId")
        refs_by_registry = self._group(references, "registryId")
        visual_registry_ids = {str(key) for key in refs_by_registry}
        all_registry_ids = declared_registry_ids or list(visual_registry_ids)
        total_registry_count = len(all_registry_ids)

        all_matches: list[dict[str, Any]] = []
        scored_pairs = 0
        scored_registry_ids: set[str] = set()
        compared_by_track: dict[str, set[str]] = {str(track_id): set() for track_id in frames_by_track}
        for track_id, track_frames in frames_by_track.items():
            for registry_id, registry_refs in refs_by_registry.items():
                frame_scores = []
                for frame in track_frames:
                    vid = frame.get("keyframeVectorId")
                    if vid is None or frame_vectors.get(int(vid)) is None:
                        continue
                    frame_vector = frame_vectors[int(vid)]
                    candidates = []
                    for reference in registry_refs:
                        rid = reference.get("registryVectorId")
                        if rid is None or reference_vectors.get(int(rid)) is None:
                            continue
                        candidates.append((float(np.dot(frame_vector, reference_vectors[int(rid)])), reference))
                    if candidates:
                        score, reference = max(candidates, key=lambda item: item[0])
                        frame_scores.append((score, frame, reference))
                        scored_pairs += 1
                top = sorted(frame_scores, key=lambda item: item[0], reverse=True)[:2]
                if not top:
                    continue
                registry_text = str(registry_id)
                compared_by_track[str(track_id)].add(registry_text)
                scored_registry_ids.add(registry_text)
                score = float(np.mean([item[0] for item in top]))
                matched_keyframes = [item[1]["keyframeId"] for item in top]
                matched_references = list(dict.fromkeys(item[2]["referenceId"] for item in top))
                all_matches.append({
                    "matchedTrackId": track_id, "matchedRegistryId": registry_id,
                    "embeddingScore": round(score, 6), "scoreBand": self._band(score, "image"),
                    "queryKeyframeIds": matched_keyframes if query_is_track else [],
                    "queryRegistryReferenceIds": matched_references if query_is_registry else [],
                    "matchedKeyframeIds": matched_keyframes, "matchedRegistryReferenceIds": matched_references,
                })

        # 核心语义：每条轨迹先对所有库项比较，再只保留该轨迹的最高匹配分。
        best_by_track: dict[str, dict[str, Any]] = {}
        for item in all_matches:
            tid = str(item.get("matchedTrackId"))
            if tid not in best_by_track or item["embeddingScore"] > best_by_track[tid]["embeddingScore"]:
                best_by_track[tid] = dict(item)
        matches = []
        for tid, item in best_by_track.items():
            compared_count = len(compared_by_track.get(tid, set()))
            complete = total_registry_count > 0 and compared_count == total_registry_count
            raw_band = str(item.get("scoreBand") or "uncertain")
            item.update({
                "rawScoreBand": raw_band, "comparedRegistryCount": compared_count,
                "totalRegistryCount": total_registry_count, "registryCoverageComplete": complete,
                "coverageLimited": not complete,
            })
            if raw_band == "mismatch" and not complete:
                item["scoreBand"] = "uncertain"
            matches.append(item)
        matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        if topK is not None:
            try:
                match_limit = max(0, int(topK))
            except (TypeError, ValueError):
                match_limit = 0
            if match_limit > 0:
                matches = matches[:match_limit]
        best_ascending = sorted((dict(item) for item in matches), key=lambda item: item["embeddingScore"])

        confirmed = [m for m in matches if m.get("scoreBand") == "match"]
        uncertain = [m for m in matches if m.get("scoreBand") == "uncertain"]
        mismatch = [m for m in matches if m.get("scoreBand") == "mismatch"]
        total_track_count = len(frames_by_track)
        scored_track_count = len(best_by_track)
        expected_pairs = total_track_count * total_registry_count
        compared_pair_count = sum(len(value) for value in compared_by_track.values())
        coverage_ratio = (len(scored_registry_ids) / total_registry_count) if total_registry_count else 0.0
        pair_coverage_ratio = (compared_pair_count / expected_pairs) if expected_pairs else 0.0
        fully_compared = [tid for tid, ids in compared_by_track.items() if total_registry_count > 0 and len(ids) == total_registry_count]
        unscored_track_ids = [str(tid) for tid in frames_by_track if str(tid) not in best_by_track]
        unscored_registry_ids = [rid for rid in all_registry_ids if rid not in scored_registry_ids]
        unrepresented_registry_ids = [rid for rid in all_registry_ids if rid not in visual_registry_ids]
        registry_coverage_complete = bool(total_registry_count) and len(fully_compared) == total_track_count and total_track_count > 0
        hint = None
        if not matches:
            if missing_references and not reference_vectors:
                hint = "库参考图向量均未命中索引且无法从 imagePath 编码"
            elif missing_keyframes and not frame_vectors:
                hint = "关键帧向量均未命中索引且无法从 keyframePath 编码"
            else:
                hint = "完成比对但无有效配对"
        elif not registry_coverage_complete:
            hint = "先验库视觉覆盖不完整，低分结果仅能列为待确认，不能判定未在库"
        elif not confirmed and uncertain:
            hint = f"最高分未达 image_match 确认阈值，仅 {len(uncertain)} 条灰区；勿将 uncertain 直接当作确认出现"
        return {
            "ok": True, "matchMode": "image_to_image", "queryType": "track" if query_is_track else "registry",
            "matches": matches, "bestMatchesAscending": best_ascending,
            "confirmedMatches": confirmed, "uncertainMatches": uncertain, "mismatchCount": len(mismatch),
            "missingKeyframeIds": missing_keyframes, "missingRegistryReferenceIds": missing_references,
            "recoveredKeyframeCount": recovered_frames, "recoveredRegistryReferenceCount": recovered_refs,
            "queryReferenceCount": len(references), "galleryKeyframeCount": len(keyframes),
            "scoredPairCount": scored_pairs, "trackRegistryComparisonCount": compared_pair_count,
            "expectedTrackRegistryPairCount": expected_pairs, "totalTrackCount": total_track_count,
            "scoredTrackCount": scored_track_count, "totalRegistryCount": total_registry_count,
            "visualRegistryCount": len(visual_registry_ids), "scoredRegistryCount": len(scored_registry_ids),
            "registryCoverageRatio": round(coverage_ratio, 6), "pairCoverageRatio": round(pair_coverage_ratio, 6),
            "registryCoverageComplete": registry_coverage_complete, "unscoredTrackIds": unscored_track_ids,
            "unscoredRegistryIds": unscored_registry_ids, "unrepresentedRegistryIds": unrepresented_registry_ids,
            "fullyComparedTrackIds": fully_compared, "visualAttempted": True,
            "matchThresholds": self._match_thresholds("image"), "hint": hint,
        }

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
        track_summaries: list[dict[str, Any]] = []
        for track_id in track_ids:
            frames = self._build_frames([track_id]).get("keyframes", [])
            if frames:
                selected = frames[: self.MAX_FRAMES_PER_TRACK]
                shown_keyframes.extend(str(frame["keyframeId"]) for frame in selected)
            clip = self._build_clip(track_id, scope)
            if clip.get("ok") and clip.get("found") and clip.get("shipSegmentId"):
                generated_segments.append(str(clip["shipSegmentId"]))
            track_summaries.append({
                "trackId": track_id,
                "keyframeIds": [str(frame["keyframeId"]) for frame in frames[: self.MAX_FRAMES_PER_TRACK]],
                "shipSegmentIds": [str(clip["shipSegmentId"])] if clip.get("shipSegmentId") else [],
            })
        shown_keyframes = list(dict.fromkeys(shown_keyframes))[:limit * self.MAX_FRAMES_PER_TRACK]
        shown_segments = list(dict.fromkeys(generated_segments))[:limit]
        shown_references = self._representative_registry_reference_ids(registry_reference_ids)[:limit]
        return {
            "ok": True,
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "shownKeyframeIds": shown_keyframes,
            "shownShipSegmentIds": shown_segments,
            "shownRegistryReferenceIds": shown_references,
            "trackIds": track_ids,
            "trackSummaries": track_summaries,
            "statistics": {
                "trackCount": len(track_ids),
                "keyframeCount": len(shown_keyframes),
                "clipCount": len(shown_segments),
                "registryReferenceCount": len(shown_references),
                "totalEvidenceCount": len(shown_keyframes) + len(shown_segments) + len(shown_references),
            },
        }

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
                if not isinstance(frame, dict) or (
                    frame.get("keyframeVectorId") is None
                    and not any(frame.get(key) for key in ("keyframePath", "imagePath", "path"))
                ):
                    continue
                normalized_frame = dict(frame)
                if normalized_frame.get("keyframeVectorId") is not None:
                    try:
                        normalized_frame["keyframeVectorId"] = int(normalized_frame["keyframeVectorId"])
                    except (TypeError, ValueError):
                        normalized_frame["keyframeVectorId"] = None
                valid_frames.append(normalized_frame)
            candidates[track_id] = valid_frames
        track_map = {str(item["trackId"]): item for item in tracks}
        frames = [frame for group in candidates.values() for frame in group]
        vectors, _ = self._ensure_image_vectors(
            frames,
            vector_key="keyframeVectorId",
            path_keys=("keyframePath", "imagePath", "path"),
            index=self.vectors.keyframes,
        )
        selected, missing = {}, []
        for track_id, frames in candidates.items():
            available = [frame for frame in frames if int(frame["keyframeVectorId"]) in vectors]
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

        def _group_minimum_score(group: list[str]) -> float | None:
            values = [
                pair_scores.get(tuple(sorted((left_id, right_id))))
                for index, left_id in enumerate(group)
                for right_id in group[index + 1:]
            ]
            available = [float(value) for value in values if value is not None]
            return round(min(available), 6) if available else None

        confirmed_merge_groups = [
            {
                "groupId": f"confirmed-{index + 1}",
                "status": "confirmed",
                "trackIds": list(group),
                "minimumScore": _group_minimum_score(group),
                "threshold": high,
            }
            for index, group in enumerate(high_groups)
            if len(group) > 1
        ]
        pending_merge_groups = []
        for group in low_groups:
            group_set = set(group)
            current_groups = [list(item) for item in high_groups if set(item).issubset(group_set)]
            if len(current_groups) <= 1:
                continue
            pending_merge_groups.append({
                "groupId": f"pending-{len(pending_merge_groups) + 1}",
                "status": "pending",
                "trackIds": list(group),
                "currentGroups": current_groups,
                "minimumScore": _group_minimum_score(group),
                "thresholdRange": [low, high],
                "possibleReduction": len(current_groups) - 1,
            })

        gray = [{"trackIds": list(pair), "embeddingScore": round(score, 6)} for pair, score in pair_scores.items() if low < score < high]
        status = "incomplete" if missing else "stable" if len(high_groups) == len(low_groups) else "sensitive"
        scores = [{"trackIds": list(pair), "embeddingScore": round(score, 6)} for pair, score in sorted(pair_scores.items())]
        return {
            "ok": True,
            "trackCount": len(tracks),
            # 高阈值只接受确定合并，因此对应保守（较大）船数；低阈值纳入待确认合并，对应最小船数。
            "minimumShipCount": len(low_groups),
            "confirmedShipCount": len(high_groups),
            "maximumShipCount": len(high_groups),
            "highThresholdShipCount": len(high_groups),
            "lowThresholdShipCount": len(low_groups),
            "confirmedMergeCount": len(confirmed_merge_groups),
            "pendingMergeCount": len(pending_merge_groups),
            "confirmedReduction": len(tracks) - len(high_groups),
            "pendingReduction": len(high_groups) - len(low_groups),
            "countStability": status,
            "highThreshold": high,
            "lowThreshold": low,
            "highGroups": high_groups,
            "lowGroups": low_groups,
            "confirmedMergeGroups": confirmed_merge_groups,
            "pendingMergeGroups": pending_merge_groups,
            "pairScores": scores,
            "grayPairs": gray,
            "unsearchableTrackIds": missing,
            "unresolvedTrackIds": [],
        }

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

    def _ensure_image_vectors(
        self,
        records: list[dict[str, Any]],
        *,
        vector_key: str,
        path_keys: tuple[str, ...],
        index: Any,
    ) -> tuple[dict[int, np.ndarray], int]:
        """先读 FAISS；缺失时按本地图片路径现场编码并写回索引。"""
        # 对“有图片路径但无持久向量编号”的记录分配本轮临时负编号，使其也能现场编码。
        temporary_ids: list[int] = []
        for item in records:
            if not isinstance(item, dict) or item.get(vector_key) is None:
                continue
            try:
                value = int(item[vector_key])
            except (TypeError, ValueError):
                item[vector_key] = None
                continue
            item[vector_key] = value
            if value < 0:
                temporary_ids.append(value)
        next_temporary_id = min(temporary_ids, default=0) - 1
        for item in records:
            if not isinstance(item, dict) or item.get(vector_key) is not None:
                continue
            if any(item.get(key) for key in path_keys):
                item[vector_key] = next_temporary_id
                next_temporary_id -= 1
        ids = [
            int(item[vector_key])
            for item in records
            if isinstance(item, dict) and item.get(vector_key) is not None
        ]
        vectors: dict[int, np.ndarray] = {}
        if ids:
            try:
                vectors = dict(index.get_many(ids))
            except Exception:  # noqa: BLE001 - an unavailable vector index falls back to image encoding.
                vectors = {}
        missing_records = [
            item for item in records
            if isinstance(item, dict)
            and item.get(vector_key) is not None
            and int(item[vector_key]) not in vectors
        ]
        recovered = 0
        if not missing_records:
            return vectors, recovered

        path_list: list[str] = []
        id_list: list[int] = []
        for item in missing_records:
            path = None
            for key in path_keys:
                value = item.get(key)
                if value:
                    path = str(value)
                    break
            if not path:
                continue
            file_path = Path(path)
            if not file_path.is_file():
                # 相对路径尝试拼到项目根
                alt = Path.cwd() / path
                if alt.is_file():
                    file_path = alt
                else:
                    continue
            path_list.append(str(file_path))
            id_list.append(int(item[vector_key]))

        if not path_list:
            return vectors, recovered
        try:
            encoded = self.embedder.encode_images(path_list)
        except Exception:  # noqa: BLE001 - failed embedding leaves the item unsearchable.
            return vectors, recovered
        if encoded is None or len(encoded) == 0:
            return vectors, recovered
        for vector_id, vector in zip(id_list, encoded):
            vectors[int(vector_id)] = np.asarray(vector, dtype=np.float32)
            recovered += 1
        # 只写回真实（非负）向量编号；临时负编号仅在本轮内存中使用。
        persistent = [(vector_id, vector) for vector_id, vector in zip(id_list, encoded) if int(vector_id) >= 0]
        if persistent:
            try:
                index.add_many(
                    [item[0] for item in persistent],
                    np.asarray([item[1] for item in persistent], dtype=np.float32),
                )
            except Exception:  # noqa: BLE001, S110 - index persistence is best effort.
                pass
        return vectors, recovered

    def _match_thresholds(self, mode: str) -> dict[str, Any]:
        """返回本轮匹配实际使用的确认、排除与灰区边界。"""
        confirmation = float(self.settings[f"{mode}_match"])
        exclusion = float(self.settings[f"{mode}_exclude"])
        return {
            "mode": mode,
            "confirmation": confirmation,
            "exclusion": exclusion,
            "grayZone": {
                "lower": exclusion,
                "upper": confirmation,
                "lowerInclusive": False,
                "upperInclusive": False,
            },
        }

    def _band(self, score: float, mode: str) -> str:
        thresholds = self._match_thresholds(mode)
        match = float(thresholds["confirmation"])
        exclude = float(thresholds["exclusion"])
        return "match" if score >= match else "mismatch" if score <= exclude else "uncertain"

    @staticmethod
    def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(str(item[key]), []).append(item)
        return grouped

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

    @staticmethod
    def _clamp_box(box: list[int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = map(int, box)
        return max(0, min(x1, width - 1)), max(0, min(y1, height - 1)), max(1, min(x2, width)), max(1, min(y2, height))

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

    @staticmethod
    def _evidence_scale(value: float | None) -> float:
        try:
            scale = float(value if value is not None else 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        return max(0.25, min(1.0, scale))

    @staticmethod
    def _even_size(value: int) -> int:
        value = max(2, int(value))
        return value if value % 2 == 0 else value + 1

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

    @staticmethod
    def _write_poster(path: Path, frame: np.ndarray | None, quality: int) -> bool:
        if frame is None or frame.size == 0:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]))

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

    def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return not (float(left["endTime"]) < float(right["startTime"]) or float(right["endTime"]) < float(left["startTime"]))

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

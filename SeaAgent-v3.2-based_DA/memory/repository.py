"""以 tracks 为主表的三层记忆仓库。"""
from __future__ import annotations
import json
import re
import uuid
from typing import Any, Iterable
from config import load_config
from memory.csv_store import CsvTable
from memory.schema import KEYFRAME_FIELDS, QA_EVIDENCE_FIELDS, QA_ROUND_FIELDS, QA_SESSION_FIELDS, REGISTRY_FIELDS, REGISTRY_IMAGE_FIELDS, TRACK_FIELDS

def normalize_hull_number(value: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", (value or "").strip()).upper()

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def _loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (json.JSONDecodeError, TypeError):
        return default

def _bool(value: str | bool) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}

class MemoryRepository:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        paths = self.config["paths"]
        self.tracks = CsvTable(paths["tracks_csv"], TRACK_FIELDS)
        self.keyframes = CsvTable(paths["keyframes_csv"], KEYFRAME_FIELDS)
        self.registry = CsvTable(paths["registry_csv"], REGISTRY_FIELDS)
        self.registry_images = CsvTable(paths["registry_images_csv"], REGISTRY_IMAGE_FIELDS)
        self.qa_sessions = CsvTable(paths["qa_sessions_csv"], QA_SESSION_FIELDS)
        self.qa_rounds = CsvTable(paths["qa_rounds_csv"], QA_ROUND_FIELDS)
        self.qa_evidence = CsvTable(paths["qa_evidence_csv"], QA_EVIDENCE_FIELDS)

    def upsert_track(self, track: dict[str, Any]) -> None:
        row = dict(track)
        row["track_id"] = str(row.get("track_id") or row.get("trackId"))
        self.tracks.upsert(row, "track_id")

    def find_tracks(self, time_range: tuple[float, float] | None = None, hull_number: str | None = None, final_match_type: str | None = None) -> list[dict[str, Any]]:
        wanted_hull = normalize_hull_number(hull_number)
        results = []
        for row in self.tracks.rows():
            start = float(row.get("start_time") or 0)
            end = float(row.get("end_time") or start)
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue
            if wanted_hull and normalize_hull_number(row.get("final_hull_number")) != wanted_hull:
                continue
            if final_match_type and row.get("final_match_type") != final_match_type:
                continue
            item = self._track_record(row)
            if time_range:
                item.update(overlapStart=max(start, time_range[0]), overlapEnd=min(end, time_range[1]))
            results.append(item)
        return sorted(results, key=lambda item: (item["startTime"], item["trackId"]))

    def get_track(self, track_id: str | int) -> dict[str, Any] | None:
        rows = self.tracks.find(lambda row: row["track_id"] == str(track_id))
        return self._track_record(rows[0]) if rows else None

    def delete_track(self, track_id: str | int) -> None:
        value = str(track_id)
        self.tracks.delete(lambda row: row["track_id"] == value)
        self.keyframes.delete(lambda row: row["track_id"] == value)

    def clear_track_memory(self) -> None:
        self.tracks.replace_all([])
        self.keyframes.replace_all([])
        self.qa_sessions.replace_all([])
        self.qa_rounds.replace_all([])
        self.qa_evidence.replace_all([])

    def upsert_keyframe(self, keyframe: dict[str, Any]) -> None:
        row = dict(keyframe)
        row["keyframe_id"] = str(row.get("keyframe_id") or row.get("keyframeId"))
        row["track_id"] = str(row.get("track_id") or row.get("trackId"))
        bbox = row.get("bbox", [])
        row["bbox"] = bbox if isinstance(bbox, str) else _json(bbox)
        self.keyframes.upsert(row, "keyframe_id")

    def get_keyframes(self, track_ids: Iterable[str | int], embedded_only: bool = True) -> dict[str, list[dict[str, Any]]]:
        wanted = {str(track_id) for track_id in track_ids}
        grouped = {track_id: [] for track_id in wanted}
        for row in self.keyframes.rows():
            track_id = row["track_id"]
            if track_id not in wanted or embedded_only and not _bool(row.get("is_embedded", "")):
                continue
            grouped[track_id].append(self._keyframe_record(row))
        for frames in grouped.values():
            frames.sort(key=lambda item: (-item["retentionScore"], item["timestamp"]))
        return grouped

    def get_keyframe(self, keyframe_id: str) -> dict[str, Any] | None:
        rows = self.keyframes.find(lambda row: row["keyframe_id"] == keyframe_id)
        return self._keyframe_record(rows[0]) if rows else None

    def keyframes_by_ids(self, keyframe_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(keyframe_ids)
        return [self._keyframe_record(row) for row in self.keyframes.rows() if row["keyframe_id"] in wanted]

    def delete_keyframe(self, keyframe_id: str) -> dict[str, Any] | None:
        rows = self.keyframes.delete(lambda row: row["keyframe_id"] == keyframe_id)
        return self._keyframe_record(rows[0]) if rows else None

    def registry_by_hull(self, hull_number: str) -> list[dict[str, Any]]:
        target = normalize_hull_number(hull_number)
        items = []
        for row in self.registry.rows():
            aliases = [normalize_hull_number(alias) for alias in _loads(row.get("aliases", ""), [])]
            if normalize_hull_number(row["hull_number"]) == target or target in aliases:
                items.append(self._registry_record(row))
        return items

    def registry_items(self) -> list[dict[str, Any]]:
        return [self._registry_record(row) for row in self.registry.rows()]

    def list_registry(self) -> list[dict[str, Any]]:
        return self.registry_items()

    def registry_references(self, registry_ids: Iterable[str] | None = None, embedded_only: bool = False) -> list[dict[str, Any]]:
        wanted = set(registry_ids or [])
        return [self._reference_record(row) for row in self.registry_images.rows() if (not wanted or row["registry_id"] in wanted) and (not embedded_only or _bool(row.get("is_embedded", "")))]

    def references_by_ids(self, reference_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(reference_ids)
        return [self._reference_record(row) for row in self.registry_images.rows() if row["reference_id"] in wanted]

    def upsert_registry(self, item: dict[str, Any]) -> str:
        registry_id = item.get("registry_id") or item.get("registryId") or f"registry-{uuid.uuid4().hex[:12]}"
        row = {
            "registry_id": registry_id,
            "hull_number": normalize_hull_number(item.get("hull_number") or item.get("hullNumber")),
            "aliases": _json(item.get("aliases", [])),
            "description": item.get("description", ""),
            "structured_attributes": _json(item.get("structured_attributes") or item.get("structuredAttributes") or {}),
        }
        if not row["hull_number"]:
            raise ValueError("舷号不能为空")
        self.registry.upsert(row, "registry_id")
        return registry_id

    def delete_registry(self, registry_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        items = self.registry.delete(lambda row: row["registry_id"] == registry_id)
        refs = self.registry_images.delete(lambda row: row["registry_id"] == registry_id)
        item = self._registry_record(items[0], include_references=False) if items else None
        return item, [self._reference_record(row) for row in refs]

    def upsert_registry_reference(self, reference: dict[str, Any]) -> None:
        row = dict(reference)
        row["reference_id"] = str(row.get("reference_id") or row.get("referenceId"))
        row["registry_id"] = str(row.get("registry_id") or row.get("registryId"))
        self.registry_images.upsert(row, "reference_id")

    def delete_registry_reference(self, reference_id: str) -> dict[str, Any] | None:
        rows = self.registry_images.delete(lambda row: row["reference_id"] == reference_id)
        return self._reference_record(rows[0]) if rows else None

    def qa_memory_summary(self) -> dict[str, int]:
        return {
            "sessionCount": len(self.qa_sessions.rows()),
            "roundCount": len(self.qa_rounds.rows()),
            "evidenceCount": len(self.qa_evidence.rows()),
        }

    def clear_qa_memory(self) -> dict[str, int]:
        summary = self.qa_memory_summary()
        self.qa_sessions.replace_all([])
        self.qa_rounds.replace_all([])
        self.qa_evidence.replace_all([])
        return summary

    def add_session(self, session_id: str, query_info: dict[str, Any]) -> None:
        self.qa_sessions.upsert({"session_id": session_id, "query_info": _json(query_info), "final_result": ""}, "session_id")

    def finish_session(self, session_id: str, result: dict[str, Any]) -> None:
        rows = self.qa_sessions.find(lambda row: row["session_id"] == session_id)
        query_info = rows[0]["query_info"] if rows else "{}"
        self.qa_sessions.upsert({"session_id": session_id, "query_info": query_info, "final_result": _json(result)}, "session_id")

    def add_round(self, round_id: str, session_id: str, plan: dict[str, Any], reflection: dict[str, Any]) -> None:
        self.qa_rounds.upsert({"round_id": round_id, "session_id": session_id, "plan": _json(plan), "reflection": _json(reflection)}, "round_id")

    def add_evidence(self, evidence_id: str, round_id: str, tool_result: dict[str, Any], evidence_source: dict[str, Any]) -> None:
        self.qa_evidence.upsert({"evidence_id": evidence_id, "round_id": round_id, "tool_result": _json(tool_result), "evidence_source": _json(evidence_source)}, "evidence_id")

    @staticmethod
    def _track_record(row: dict[str, str]) -> dict[str, Any]:
        start_time = float(row.get("start_time") or 0)
        end_time = float(row.get("end_time") or start_time)
        video_start_time = float(row.get("video_start_time") or start_time)
        video_end_time = float(row.get("video_end_time") or video_start_time)
        return {"trackId": row["track_id"], "startTime": start_time, "endTime": end_time, "videoStartTime": video_start_time, "videoEndTime": video_end_time, "finalHullNumber": row.get("final_hull_number") or None, "finalDescription": row.get("final_description", ""), "finalMatchType": row.get("final_match_type") or "unknown", "trajectoryPath": row.get("trajectory_path", "")}

    @staticmethod
    def _keyframe_record(row: dict[str, str]) -> dict[str, Any]:
        return {"keyframeId": row["keyframe_id"], "trackId": row["track_id"], "timestamp": float(row.get("timestamp") or 0), "keyframePath": row.get("keyframe_path", ""), "bbox": _loads(row.get("bbox", ""), []), "qualityScore": float(row.get("quality_score") or 0), "retentionScore": float(row.get("retention_score") or 0), "hasReadableHullNumber": row.get("has_readable_hull_number") or "no", "vlmHullNumber": row.get("vlm_hull_number") or None, "readabilityConfidence": float(row.get("readability_confidence") or 0), "description": row.get("description", ""), "keyframeVectorId": int(row["keyframe_vector_id"]) if row.get("keyframe_vector_id") else None, "isEmbedded": _bool(row.get("is_embedded", ""))}

    def _registry_record(self, row: dict[str, str], include_references: bool = True) -> dict[str, Any]:
        registry_id = row["registry_id"]
        item = {"registryId": registry_id, "hullNumber": row["hull_number"], "aliases": _loads(row.get("aliases", ""), []), "description": row.get("description", ""), "structuredAttributes": _loads(row.get("structured_attributes", ""), {})}
        if include_references:
            item["references"] = self.registry_references([registry_id])
        return item

    @staticmethod
    def _reference_record(row: dict[str, str]) -> dict[str, Any]:
        return {"referenceId": row["reference_id"], "registryId": row["registry_id"], "imagePath": row.get("image_path", ""), "registryVectorId": int(row["registry_vector_id"]) if row.get("registry_vector_id") else None, "isEmbedded": _bool(row.get("is_embedded", ""))}

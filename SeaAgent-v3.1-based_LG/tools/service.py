"""SeaAgent 原子工具服务。"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np
from config import load_config
from memory import MemoryRepository, normalize_hull_number
from services import AgentLLMService, QwenMultimodalEmbedder
from vector_store import VectorCatalog

class ToolService:
    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.settings = self.config["pipeline"]["retrieval"]

    def getTrack(self, timeRange: tuple[float, float] | None = None, hullNumber: str | None = None, finalMatchType: str | None = None, offset: int = 0, limit: int = 0) -> dict[str, Any]:
        tracks = self.repository.find_tracks(timeRange, hullNumber, finalMatchType)
        start = max(0, int(offset or 0))
        page_size = max(0, min(200, int(limit or 0)))
        selected = tracks[start:start + page_size] if page_size else tracks[start:]
        next_offset = start + len(selected)
        return {"ok": True, "queryScope": list(timeRange) if timeRange else None, "queryHullNumber": normalize_hull_number(hullNumber), "queryFinalMatchType": finalMatchType, "trackIds": [item["trackId"] for item in selected], "tracks": selected, "totalTrackCount": len(tracks), "returnedTrackCount": len(selected), "offset": start, "limit": page_size, "hasMore": next_offset < len(tracks), "nextOffset": next_offset if next_offset < len(tracks) else None}

    def getFrames(self, trackIds: Iterable[str | int]) -> dict[str, Any]:
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
                if vid is not None and int(vid) in available_vectors:
                    patched = dict(frame)
                    patched["isEmbedded"] = True
                    valid.append(patched)
                else:
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

    def getClip(self, trackId: str | int, timeRange: tuple[float, float] | None = None, scale: float | None = None) -> dict[str, Any]:
        track = self.repository.get_track(trackId)
        if not track:
            return {"ok": False, "error": "track_not_found", "trackId": str(trackId)}
        start, end = track["startTime"], track["endTime"]
        monitor_scope = (max(start, timeRange[0]), min(end, timeRange[1])) if timeRange else (start, end)
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
        if timeRange and any("observedAt" in item for item in all_boxes):
            boxes = [item for item in all_boxes if monitor_scope[0] <= float(item.get("observedAt") or 0) <= monitor_scope[1]]
        elif timeRange:
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
            self._even_size(max(160, int(round(evidence.get("clip_width", 640) * media_scale)))),
            self._even_size(max(90, int(round(evidence.get("clip_height", 360) * media_scale)))),
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
        if output_path.is_file() and output_path.stat().st_size > 0:
            self._ensure_clip_poster(output_path, poster_path, poster_quality)
            return {"ok": True, "found": True, "trackId": str(trackId), "shipSegmentId": segment_id, "segmentPath": str(output_path), "posterPath": str(poster_path) if poster_path.is_file() else None, "codec": "cached", "startTime": monitor_scope[0], "endTime": monitor_scope[1], "videoStartTime": video_start_time, "videoEndTime": video_end_time}
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
        frame_gap = max(1, int(round(source_fps / output_fps)))
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
        return {"ok": True, "found": True, "trackId": str(trackId), "shipSegmentId": segment_id, "segmentPath": str(output_path), "posterPath": str(poster_path) if poster_path.is_file() else None, "codec": codec, "startTime": monitor_scope[0], "endTime": monitor_scope[1], "videoStartTime": video_start_time, "videoEndTime": video_end_time}

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
        target_size = (max(1, int(round(width * media_scale))), max(1, int(round(height * media_scale))))
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

    def getRegistry(self, hullNumber: str) -> dict[str, Any]:
        items = self.repository.registry_by_hull(hullNumber)
        # 库项自带 references；再按 registryId 扫一遍表，避免嵌套漏图
        references: list[dict[str, Any]] = []
        seen_ref: set[str] = set()
        for item in items:
            for reference in item.get("references") or []:
                if not isinstance(reference, dict):
                    continue
                rid = str(reference.get("referenceId") or "")
                if rid and rid in seen_ref:
                    continue
                if rid:
                    seen_ref.add(rid)
                references.append(reference)
        if items:
            extra = self.repository.registry_references(
                [str(item.get("registryId")) for item in items if item.get("registryId")],
                embedded_only=False,
            )
            for reference in extra:
                rid = str(reference.get("referenceId") or "")
                if rid and rid in seen_ref:
                    continue
                if rid:
                    seen_ref.add(rid)
                references.append(reference)
        # 以向量库是否命中为准，不唯 isEmbedded 标志（标志可能滞后）
        candidate_ids = [
            int(item["registryVectorId"])
            for item in references
            if item.get("registryVectorId") is not None
        ]
        available_vectors = self.vectors.registry.get_many(candidate_ids) if candidate_ids else {}
        valid: list[dict[str, Any]] = []
        discarded: list[str] = []
        for item in references:
            vid = item.get("registryVectorId")
            if vid is None:
                discarded.append(item.get("referenceId"))
                continue
            if int(vid) in available_vectors:
                patched = dict(item)
                patched["isEmbedded"] = True
                valid.append(patched)
            else:
                discarded.append(item.get("referenceId"))
        # 始终导出全部参考图：索引命中的直接使用，未命中的交给 matchImage 按路径现场编码。
        available_ids = {int(key) for key in available_vectors}
        export_refs = [
            dict(item, isEmbedded=(item.get("registryVectorId") is not None and int(item["registryVectorId"]) in available_ids))
            for item in references
            if isinstance(item, dict)
        ]
        registry_ids = [str(item["registryId"]) for item in items if item.get("registryId")]
        visual_ids = {str(item.get("registryId")) for item in export_refs if item.get("registryId")}
        searchable_ids = {str(item.get("registryId")) for item in valid if item.get("registryId")}
        return {
            "ok": True,
            "found": bool(items),
            "searchable": bool(valid),
            "hullNumber": normalize_hull_number(hullNumber),
            "registryIds": registry_ids,
            "registryItems": items,
            "registryReferenceIds": [item["referenceId"] for item in export_refs if item.get("referenceId")],
            "registryReferences": export_refs,
            "discardedReferenceIds": discarded,
            "referenceCount": len(references),
            "searchableReferenceCount": len(valid),
            "totalRegistryCount": len(registry_ids),
            "visualRegistryCount": len(visual_ids),
            "searchableRegistryCount": len(searchable_ids),
            "missingReferenceRegistryIds": [rid for rid in registry_ids if rid not in visual_ids],
        }

    def matchHull(self, hullNumberArray: Iterable[str | None]) -> dict[str, Any]:
        hull_numbers = list(dict.fromkeys(normalize_hull_number(value) for value in hullNumberArray if normalize_hull_number(value)))
        matches = {hull: self.repository.registry_by_hull(hull) for hull in hull_numbers}
        return {"ok": True, "exactMatches": {hull: items for hull, items in matches.items() if items}, "matchedHullNumbers": [hull for hull, items in matches.items() if items], "unmatchedHullNumbers": [hull for hull, items in matches.items() if not items]}

    def listRegistry(self) -> dict[str, Any]:
        items = self.repository.list_registry()
        references = [reference for item in items for reference in item.get("references", []) if isinstance(reference, dict)]
        vector_ids = [
            int(item["registryVectorId"])
            for item in references
            if item.get("registryVectorId") is not None
        ]
        available_vectors = self.vectors.registry.get_many(vector_ids) if vector_ids else {}
        valid: list[dict[str, Any]] = []
        discarded: list[str] = []
        for item in references:
            vid = item.get("registryVectorId")
            if vid is not None and int(vid) in available_vectors:
                patched = dict(item)
                patched["isEmbedded"] = True
                valid.append(patched)
            else:
                discarded.append(str(item.get("referenceId") or ""))
        # 始终导出全部参考图，不能因部分向量有效而丢掉其余库项。
        available_ids = {int(key) for key in available_vectors}
        export_refs = [
            dict(item, isEmbedded=(item.get("registryVectorId") is not None and int(item["registryVectorId"]) in available_ids))
            for item in references
        ]
        registry_ids = [str(item["registryId"]) for item in items if item.get("registryId")]
        visual_ids = {str(item.get("registryId")) for item in export_refs if item.get("registryId")}
        searchable_ids = {str(item.get("registryId")) for item in valid if item.get("registryId")}
        return {
            "ok": True,
            "found": bool(items),
            "searchable": bool(valid),
            "registryItems": items,
            "registryReferenceIds": [item["referenceId"] for item in export_refs if item.get("referenceId")],
            "registryReferences": export_refs,
            "discardedReferenceIds": discarded,
            "unsearchableRegistryIds": [rid for rid in registry_ids if rid not in searchable_ids],
            "missingReferenceRegistryIds": [rid for rid in registry_ids if rid not in visual_ids],
            "referenceCount": len(references),
            "searchableReferenceCount": len(valid),
            "totalRegistryCount": len(registry_ids),
            "visualRegistryCount": len(visual_ids),
            "searchableRegistryCount": len(searchable_ids),
        }

    def matchText(self, description: str, galleryImages: list[dict[str, Any]] | dict[str, Any] | None = None, topK: int | None = None) -> dict[str, Any]:
        if not description.strip():
            return {"ok": False, "error": "description_required", "matches": []}
        # 直接传入 registryItems 列表时保留，供无参考图时的关键字弱匹配
        raw_registry_items: list[dict[str, Any]] = []
        if isinstance(galleryImages, list):
            raw_registry_items = [
                item for item in galleryImages
                if isinstance(item, dict)
                and item.get("registryId") is not None
                and (item.get("description") is not None or item.get("hullNumber") is not None)
            ]
        galleryImages = self._flatten_image_records(galleryImages)
        if not galleryImages and not raw_registry_items:
            return {"ok": False, "error": "galleryImages_required", "matches": [], "hint": "请先 getFrames 或 listRegistry，再把 keyframes/registryReferences 作为 galleryImages"}
        registry_images = [item for item in galleryImages if item.get("registryId") is not None and item.get("registryVectorId") is not None and item.get("isEmbedded")]
        # 可嵌入库图为空时：对库项描述做关键字弱匹配
        if not registry_images:
            text_items = raw_registry_items or [
                item for item in galleryImages
                if (item.get("registryId") is not None or item.get("hullNumber") is not None)
                and (item.get("description") or item.get("hullNumber"))
            ]
            if text_items:
                return self._match_registry_by_text(description, text_items, topK)
        if registry_images:
            query = self.embedder.encode_text(description, self.config["prompts"]["text_retrieval_instruction"])
            vectors = self.vectors.registry.get_many(item["registryVectorId"] for item in registry_images)
            grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
            for image in registry_images:
                vector = vectors.get(int(image["registryVectorId"]))
                if vector is not None:
                    grouped.setdefault(str(image["registryId"]), []).append((float(np.dot(query, vector)), image))
            matches = []
            for registry_id, scored in grouped.items():
                ranked = sorted(scored, key=lambda item: item[0], reverse=True)
                if not ranked:
                    continue
                # 先验库多视角：以最高相似参考图为准，避免弱视角把强匹配平均稀释
                best_score = float(ranked[0][0])
                support = ranked[:3]
                mean_top3 = float(np.mean([item[0] for item in support]))
                matches.append({
                    "matchedRegistryId": registry_id,
                    "embeddingScore": round(best_score, 6),
                    "maxEmbeddingScore": round(best_score, 6),
                    "meanTop3Score": round(mean_top3, 6),
                    "scoreBand": self._band(best_score, "text"),
                    "matchedRegistryReferenceIds": [item[1]["referenceId"] for item in support],
                    "supportReferenceCount": len(support),
                })
            matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
            # 附加 rank gap，便于后续灰区判断
            if len(matches) >= 2:
                matches[0]["rankGap"] = round(float(matches[0]["embeddingScore"]) - float(matches[1]["embeddingScore"]), 6)
            elif matches:
                matches[0]["rankGap"] = round(float(matches[0]["embeddingScore"]), 6)
            selected_matches = matches[:topK] if topK else matches
            return {
                "ok": True,
                "matchMode": "text_to_registry",
                "matches": selected_matches,
                "confirmedMatches": [item for item in selected_matches if item.get("scoreBand") == "match"],
                "uncertainMatches": [item for item in selected_matches if item.get("scoreBand") == "uncertain"],
                "mismatchCount": sum(1 for item in selected_matches if item.get("scoreBand") == "mismatch"),
                "matchThresholds": self._match_thresholds("text"),
            }
        images = [item for item in galleryImages if item.get("isEmbedded") and item.get("keyframeVectorId") is not None]
        if not images:
            return {"ok": True, "matchMode": "text_to_image", "matches": [], "missingKeyframeIds": []}
        query = self.embedder.encode_text(description, self.config["prompts"]["text_retrieval_instruction"])
        vectors = self.vectors.keyframes.get_many(item["keyframeVectorId"] for item in images)
        missing = [item["keyframeId"] for item in images if int(item["keyframeVectorId"]) not in vectors]
        grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        for image in images:
            vector = vectors.get(int(image["keyframeVectorId"]))
            if vector is not None:
                grouped.setdefault(str(image["trackId"]), []).append((float(np.dot(query, vector)), image))
        matches = []
        for track_id, scored in grouped.items():
            ranked = sorted(scored, key=lambda item: item[0], reverse=True)
            if not ranked:
                continue
            top = ranked[:2]
            best_score = float(top[0][0])
            mean_top2 = float(np.mean([item[0] for item in top]))
            # 轨迹描述检索：主分用最高关键帧，避免次优帧把分数拉低
            matches.append({
                "matchedTrackId": track_id,
                "embeddingScore": round(best_score, 6),
                "maxEmbeddingScore": round(best_score, 6),
                "meanTop2Score": round(mean_top2, 6),
                "scoreBand": self._band(best_score, "text"),
                "matchedKeyframeIds": [item[1]["keyframeId"] for item in top],
            })
        matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        if topK:
            matches = matches[: max(1, int(topK))]
        confirmed = [m for m in matches if m.get("scoreBand") == "match"]
        uncertain = [m for m in matches if m.get("scoreBand") == "uncertain"]
        hint = None
        if matches and not confirmed and uncertain:
            hint = "最高分未达 text_match 确认阈值，仅灰区候选；勿将 uncertain 直接当作确认出现"
        # 附带分差，便于发现“分数塌缩导致总是同一批轨迹”
        if len(matches) >= 2:
            matches[0]["rankGap"] = round(
                float(matches[0]["embeddingScore"]) - float(matches[1]["embeddingScore"]), 6
            )
        elif matches:
            matches[0]["rankGap"] = round(float(matches[0]["embeddingScore"]), 6)
        return {
            "ok": True,
            "matchMode": "text_to_image",
            "matches": matches,
            "confirmedMatches": confirmed,
            "uncertainMatches": uncertain,
            "missingKeyframeIds": missing,
            "scoredTrackCount": len(grouped),
            "matchThresholds": self._match_thresholds("text"),
            "hint": hint,
        }

    def matchImage(
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
            index=self.vectors.keyframes, owner_key="keyframeId",
        )
        reference_vectors, recovered_refs = self._ensure_image_vectors(
            references, vector_key="registryVectorId", path_keys=("imagePath", "keyframePath", "path"),
            index=self.vectors.registry, owner_key="referenceId",
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

    def verifyTarget(self, description: str | None = None, registryReferenceIds: list[str] | None = None, keyframeIds: list[str] | None = None, shipSegmentIds: list[str] | None = None) -> dict[str, Any]:
        """支持三类核验：
        1) 文字描述 vs 轨迹关键帧/片段
        2) 库参考图 vs 轨迹关键帧/片段
        3) 文字描述 vs 库参考图（先验库描述灰区）
        """
        has_description = bool(description and str(description).strip())
        has_registry = bool(registryReferenceIds)
        has_keyframes = bool(keyframeIds)
        has_segments = bool(shipSegmentIds)
        text_to_registry = has_description and has_registry and not has_keyframes and not has_segments
        text_to_track = has_description and not has_registry and (has_keyframes != has_segments)
        registry_to_track = has_registry and not has_description and (has_keyframes != has_segments)
        if not (text_to_registry or text_to_track or registry_to_track):
            return {"ok": False, "error": "unsupported_verify_combination", "decision": "uncertain"}

        references = self.repository.references_by_ids(registryReferenceIds or [])
        reference_paths = [
            item["imagePath"]
            for item in references
            if Path(item["imagePath"]).exists() and Path(item["imagePath"]).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ][:3]
        if has_keyframes:
            frames = self.repository.keyframes_by_ids(keyframeIds or [])
            # 只传静态图路径；视频片段走 _sample_segments 抽帧，避免 vLLM 拉本地 video URL
            evidence = [
                item["keyframePath"]
                for item in frames
                if Path(item["keyframePath"]).exists() and Path(item["keyframePath"]).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            ][:6 if has_description else 3]
        elif has_segments:
            evidence = self._sample_segments(shipSegmentIds or [], 6 if has_description else 3)
        else:
            # 文字描述核验先验库：库参考图本身作为待审图像
            evidence = reference_paths[:3]

        if text_to_registry:
            if not evidence:
                return {"ok": True, "targetType": "description_registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": "uncertain", "facts": ["先验库参考图不可用"], "keyframeIds": [], "shipSegmentIds": []}
            result = self.llm.verify(description, [], evidence)
            return {"ok": True, "targetType": "description_registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": result["decision"], "facts": result["facts"], "keyframeIds": [], "shipSegmentIds": []}

        if not evidence or (registry_to_track and not reference_paths):
            return {"ok": True, "targetType": "description" if has_description else "registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": "uncertain", "facts": ["视觉证据不足"], "keyframeIds": keyframeIds or [], "shipSegmentIds": shipSegmentIds or []}
        result = self.llm.verify(description if has_description else None, reference_paths, evidence)
        return {"ok": True, "targetType": "description" if has_description else "registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": result["decision"], "facts": result["facts"], "keyframeIds": keyframeIds or [], "shipSegmentIds": shipSegmentIds or []}


    def showEvidence(self, keyframeIds: list[str] | None = None, shipSegmentIds: list[str] | None = None, registryReferenceIds: list[str] | None = None) -> dict[str, Any]:
        return {
            "ok": True,
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "shownKeyframeIds": list(dict.fromkeys(keyframeIds or [])),
            "shownShipSegmentIds": list(dict.fromkeys(shipSegmentIds or [])),
            "shownRegistryReferenceIds": self._representative_registry_reference_ids(registryReferenceIds or []),
        }

    def _normalize_dedup_track_records(self, tracks: Any) -> tuple[list[dict[str, Any]], list[str]]:
        """兼容完整轨迹、getTrack 结果外壳和纯 trackId 列表，并恢复完整时间字段。"""
        raw = tracks.get("tracks") if isinstance(tracks, dict) else tracks
        if not isinstance(raw, (list, tuple)):
            return [], ["<invalid-tracks>"] if raw not in (None, "") else []
        repository = getattr(self, "repository", None)
        records: list[dict[str, Any]] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        for item in raw:
            track_id = item.get("trackId") if isinstance(item, dict) else item
            if track_id is None:
                unresolved.append("<missing-trackId>")
                continue
            key = str(track_id)
            if key in seen:
                continue
            record = item if isinstance(item, dict) else None
            needs_full_record = not isinstance(record, dict) or any(
                record.get(field) is None for field in ("trackId", "startTime", "endTime")
            )
            if needs_full_record and repository is not None and callable(getattr(repository, "get_track", None)):
                recovered = repository.get_track(track_id)
                if isinstance(recovered, dict):
                    record = recovered
            if not isinstance(record, dict) or any(
                record.get(field) is None for field in ("trackId", "startTime", "endTime")
            ):
                unresolved.append(key)
                continue
            seen.add(key)
            records.append(record)
        return records, unresolved

    @staticmethod
    def _normalize_dedup_keyframe_groups(value: Any) -> dict[str, Any]:
        """兼容 getFrames 外壳、按轨迹分组对象和关键帧平铺列表。"""
        if isinstance(value, dict) and isinstance(value.get("keyframesByTrack"), dict):
            value = value["keyframesByTrack"]
        if isinstance(value, list):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for frame in value:
                if isinstance(frame, dict) and frame.get("trackId") is not None:
                    grouped.setdefault(str(frame["trackId"]), []).append(frame)
            return grouped
        if not isinstance(value, dict):
            return {}
        return {str(track_id): group for track_id, group in value.items()}

    def dedupTracks(self, tracks: Any, keyframesByTrack: Any) -> dict[str, Any]:
        tracks, unresolved_track_ids = self._normalize_dedup_track_records(tracks)
        if unresolved_track_ids:
            return {
                "ok": False,
                "error": "dedup_tracks_unresolved",
                "unresolvedTrackIds": unresolved_track_ids,
                "hint": "dedupTracks 需要完整轨迹记录；请传入 getTrack.tracks，而不是 trackIds",
            }
        keyframes_by_track = self._normalize_dedup_keyframe_groups(keyframesByTrack)
        candidates: dict[str, list[dict[str, Any]]] = {}
        for track in tracks:
            track_id = str(track["trackId"])
            group = keyframes_by_track.get(track_id, {})
            if isinstance(group, dict):
                frames = group.get("keyframes") if isinstance(group.get("keyframes"), list) else []
            elif isinstance(group, list):
                frames = group
            else:
                frames = []
            valid_frames: list[dict[str, Any]] = []
            for frame in frames:
                if not isinstance(frame, dict) or frame.get("keyframeVectorId") is None or frame.get("isEmbedded") is False:
                    continue
                try:
                    vector_id = int(frame["keyframeVectorId"])
                except (TypeError, ValueError):
                    continue
                normalized_frame = dict(frame)
                normalized_frame["keyframeVectorId"] = vector_id
                valid_frames.append(normalized_frame)
            candidates[track_id] = valid_frames
        track_map = {str(item["trackId"]): item for item in tracks}
        vector_ids = [frame["keyframeVectorId"] for frames in candidates.values() for frame in frames]
        vectors = self.vectors.keyframes.get_many(vector_ids) if vector_ids else {}
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

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = getattr(self, name, None)
        if not callable(tool) or name.startswith("_") or name == "execute":
            return {"ok": False, "error": "tool_not_allowed", "tool": name}
        try:
            return tool(**arguments)
        except Exception as error:
            return {"ok": False, "error": str(error), "tool": name}

    def _match_registry_by_text(
        self,
        description: str,
        items: list[dict[str, Any]],
        topK: int | None,
    ) -> dict[str, Any]:
        """无可用向量时：按描述关键字对库项做弱匹配（颜色/船型等）。"""
        query = str(description or "").strip().lower()
        tokens = [t for t in re.findall(r"[一-鿿]{1,4}|[a-z0-9]{2,}", query) if t not in {"哪些", "哪个", "什么", "有没有", "是否", "数据库", "先验库", "船", "船舶"}]
        if not tokens:
            tokens = [query] if query else []
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            text = f"{item.get('description') or ''} {item.get('hullNumber') or ''} {' '.join(item.get('aliases') or [])}".lower()
            if not text.strip():
                continue
            hits = sum(1 for token in tokens if token and token in text)
            if not hits:
                continue
            score = hits / max(1, len(tokens))
            # 强颜色词（如黄色）命中时抬高
            if any(token in text for token in tokens if token in {"黄", "黄色", "蓝", "蓝色", "白", "白色", "灰", "灰色", "红", "红色"}):
                score = min(1.0, score + 0.25)
            band = "match" if score >= 0.66 else "uncertain" if score >= 0.34 else "mismatch"
            if band == "mismatch":
                continue
            scored.append((score, {
                "matchedRegistryId": item.get("registryId"),
                "registryId": item.get("registryId"),
                "hullNumber": item.get("hullNumber"),
                "description": item.get("description"),
                "embeddingScore": round(score, 6),
                "scoreBand": band,
                "matchMode": "text_keyword_registry",
                "registryReferenceIds": [
                    ref.get("referenceId")
                    for ref in (item.get("references") or [])
                    if isinstance(ref, dict) and ref.get("referenceId")
                ],
            }))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        limit = max(1, min(20, int(topK or self.settings.get("top_k") or 3)))
        matches = [item for _, item in scored[:limit]]
        return {
            "ok": True,
            "matchMode": "text_keyword_registry",
            "matches": matches,
            "hint": "当前库参考图不可检索，已用描述关键字弱匹配库项",
        }

    def _ensure_image_vectors(
        self,
        records: list[dict[str, Any]],
        *,
        vector_key: str,
        path_keys: tuple[str, ...],
        index: Any,
        owner_key: str,
    ) -> tuple[dict[int, np.ndarray], int]:
        """先读 FAISS；缺失时按本地图片路径现场编码并写回索引。"""
        # 对“有图片路径但无持久向量编号”的记录分配本轮临时负编号，使其也能现场编码。
        next_temporary_id = -1
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
            except Exception:
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
        except Exception:
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
            except Exception:
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
        resized_width = max(1, min(content_width, int(round(crop_width * scale))))
        resized_height = max(1, min(content_height, int(round(crop_height * scale))))
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

    def _sample_segments(self, segment_ids: list[str], limit: int) -> list[np.ndarray]:
        samples = []
        for segment_id in segment_ids:
            path = Path(self.config["paths"]["clip_dir"]) / f"{segment_id}.mp4"
            capture = cv2.VideoCapture(str(path))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            positions = np.linspace(0, max(0, total - 1), min(limit - len(samples), max(1, total)), dtype=int)
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    samples.append(frame)
            capture.release()
            if len(samples) >= limit:
                break
        return samples[:limit]

    @staticmethod
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

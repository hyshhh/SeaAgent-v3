"""证据富化：把散落在各次工具结果里的关键帧 / 片段 / 库图 ID 汇总成前端可直接渲染的载荷。"""
from __future__ import annotations

import re
from typing import Any

KEYFRAME_ID_KEYS = (  # 工具结果里这些键下的值都是「可直接展示的证据 ID」
    "keyframeIds",
    "shownKeyframeIds",
    "matchedKeyframeIds",
    "queryKeyframeIds",
    "discardedKeyframeIds",
)
SEGMENT_ID_KEYS = ("shipSegmentIds", "shownShipSegmentIds", "segmentIds")
TRACK_ID_KEYS = ("trackIds", "shownTrackIds")
REFERENCE_ID_KEYS = (
    "registryReferenceIds",
    "shownRegistryReferenceIds",
    "matchedRegistryReferenceIds",
    "queryRegistryReferenceIds",
)
TOOL_STATISTIC_KEYS = (
    "status",
    "trackCount",
    "highThresholdShipCount",
    "lowThresholdShipCount",
    "entityCount",
    "inRegistryCount",
    "unregisteredCount",
    "coverageComplete",
    "searchedTrackCount",
    "candidateCount",
    "unsearchableTrackCount",
)
NESTED_KEYS = ("keyframes", "keyframesByTrack", "matches", "registryItems", "registryReferences", "tracks")  # 需要下钻的嵌套容器：工具结果常把明细放在这些键下


# ------------------------------------------------------------------ #
# 一、证据 ID 收集                                                         #
# ------------------------------------------------------------------ #
# 1.1 取值并归一成 ID 列表
def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        collected: list[str] = []
        for item in value.values():
            collected.extend(_as_id_list(item))
        return collected
    if isinstance(value, (list, tuple, set)):
        collected = []
        for item in value:
            collected.extend(_as_id_list(item))
        return collected
    return []


# 1.2 记录自身的 ID
def _record_id(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


# 1.3 切词，用于比对同类字段
def _tokens(key: str) -> list[str]:
    spaced = re.sub(r"[^0-9A-Za-z]+", " ", str(key))
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return [part.lower() for part in spaced.split() if part]


# 1.4 判断字段是否属于同一类证据
def _same_kind(field: str, stems: list[list[str]]) -> bool:
    parts = _tokens(field)
    if len(parts) < 2 or parts[-1] not in {"id", "ids"}:
        return False
    body = parts[:-1]
    for stem in stems:
        if not stem:
            continue
        cursor = 0
        for token in stem:
            if cursor < len(body) and token == body[cursor]:
                cursor += 1
        if cursor == len(body):
            return True
    return False


# 1.5 递归收集某一类证据 ID
def _collect_ids(payload: Any, keys: tuple[str, ...], depth: int = 0) -> list[str]:
    stems = [_tokens(key)[:-1] for key in keys if _tokens(key) and _tokens(key)[-1] in {"id", "ids"}]
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                found.extend(_as_id_list(value))
            elif isinstance(value, (str, int)) and _same_kind(key, stems):
                text = str(value).strip()
                if text:
                    found.append(text)
            elif depth < 4 and (key in NESTED_KEYS or isinstance(value, (dict, list))):
                found.extend(_collect_ids(value, keys, depth + 1))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_collect_ids(item, keys, depth + 1))
    return found


# 1.6 去重并限长
def _dedup(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


# ------------------------------------------------------------------ #
# 二、证据载荷                                                             #
# ------------------------------------------------------------------ #
# 2.1 按轨迹把关键帧与库图绑成一组
def _display_groups(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    def touch(track_id: str) -> dict[str, Any]:
        return groups.setdefault(track_id, {
            "trackId": track_id,
            "trackIds": [track_id],
            "keyframeIds": [],
            "shipSegmentIds": [],
            "registryReferenceIds": [],
            "score": None,
            "band": "",
            "matchedObject": None,
        })

    for record in records:
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        grouped = result.get("keyframesByTrack")  # 按轨迹分组的关键帧结果：键就是 trackId，值里是关键帧
        if isinstance(grouped, dict):
            for track_id, frames in grouped.items():
                group = touch(str(track_id))
                group["keyframeIds"].extend(_collect_ids(frames, KEYFRAME_ID_KEYS))
        nested = [item for item in (result.get("matches") or []) if isinstance(item, dict)]
        candidates = [result, *nested]
        if isinstance(result.get("tracks"), list):
            candidates.extend(item for item in result["tracks"] if isinstance(item, dict))
        for candidate in candidates:
            track_id = _record_id(candidate, "trackId", "matchedTrackId", "queryTrackId")
            if not track_id:
                continue
            group = touch(track_id)
            if str(record.get("tool") or "") == "match_entities":
                group["registryMatch"] = True
            group["trackIds"].extend(_collect_ids(candidate, TRACK_ID_KEYS))
            group["keyframeIds"].extend(_collect_ids(candidate, KEYFRAME_ID_KEYS))
            group["shipSegmentIds"].extend(_collect_ids(candidate, SEGMENT_ID_KEYS))
            group["registryReferenceIds"].extend(_collect_ids(candidate, REFERENCE_ID_KEYS))
            score = candidate.get("embeddingScore", candidate.get("score"))
            if isinstance(score, (int, float)) and (group["score"] is None or score > group["score"]):
                group["score"] = float(score)
                match_object = candidate.get("matchedObject")
                group["matchedObject"] = match_object if isinstance(match_object, dict) else None
            band = str(candidate.get("scoreBand") or candidate.get("band") or "").strip()
            if band:
                group["band"] = band
    ordered = sorted(groups.values(), key=lambda item: (item["score"] is None, -(item["score"] or 0.0)))
    for group in ordered:
        group["trackIds"] = _dedup(group["trackIds"], limit)
        group["keyframeIds"] = _dedup(group["keyframeIds"], limit)
        group["shipSegmentIds"] = _dedup(group["shipSegmentIds"], limit)
        group["registryReferenceIds"] = _dedup(group["registryReferenceIds"], limit)
    return ordered[:limit]


# 2.2 汇总成本轮的证据载荷
def build_evidence_payload(records: list[dict[str, Any]], *, limit: int = 60) -> dict[str, Any]:
    successful = [
        record for record in records
        if isinstance(record, dict)
        and str(record.get("status") or "") != "error"
        and isinstance(record.get("result"), (dict, list))
    ]
    payloads = [record["result"] for record in successful]

    keyframes = _dedup([item for payload in payloads for item in _collect_ids(payload, KEYFRAME_ID_KEYS)], limit)
    segments = _dedup([item for payload in payloads for item in _collect_ids(payload, SEGMENT_ID_KEYS)], limit)
    references = _dedup([item for payload in payloads for item in _collect_ids(payload, REFERENCE_ID_KEYS)], limit)
    tracks = _dedup([item for payload in payloads for item in _collect_ids(payload, TRACK_ID_KEYS)], limit)
    query_statistics: dict[str, Any] = {}
    query_statistics_by_tool: list[dict[str, Any]] = []
    display_statistics: dict[str, Any] = {}
    for record in successful:
        result = record["result"]
        if not isinstance(result, dict):
            continue
        statistics = result.get("statistics")
        statistics = dict(statistics) if isinstance(statistics, dict) else {}
        statistics.update({key: result[key] for key in TOOL_STATISTIC_KEYS if key in result})
        if str(record.get("tool") or "") == "show_evidence":
            display_statistics = statistics
        elif statistics:
            query_statistics.update(statistics)
            query_statistics_by_tool.append({
                "tool": str(record.get("tool") or ""),
                "statistics": statistics,
            })

    evidence = {
        "keyframeIds": keyframes,
        "shipSegmentIds": segments,
        "registryReferenceIds": references,
        "trackIds": tracks,
        "displayGroups": _display_groups(successful, limit),
        "statistics": {
            "query": query_statistics,
            "queryByTool": query_statistics_by_tool,
            "display": display_statistics,
        },
        "tools": [str(record.get("tool") or "") for record in successful],
    }
    evidence["isEmpty"] = not (keyframes or segments or references)
    return evidence

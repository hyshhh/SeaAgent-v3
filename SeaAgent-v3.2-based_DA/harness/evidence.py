"""证据富化：把本轮散落在各次工具结果里的证据 ID 汇总成一份完整载荷。

为什么这一层必须由代码做（其他地方一律不硬编码判定）：
    前端证据面板要一次拿到**全部**可渲染的证据 —— 关键帧、视频片段、先验库参考图。
    但 Deep Agents 的循环里，这些 ID 分散在不同步骤的工具返回值中，而主智能体的最终
    回答只是一段文本。让模型把这些 ID 一个不漏、一个不错地抄进回答，既不可靠也不必要：
    ID 本来就在本轮的工具结果里，代码直接收齐比"请模型复述"稳得多。

    所以这里做三件事，全部是机械劳动，没有任何判定：
      1. 按别名表遍历本轮每一次工具结果，收集关键帧 / 片段 / 参考图 ID；
      2. 去重、限长（前端一次展示不了几百条）；
      3. 产出 displayGroups：一条轨迹一组，把它的关键帧与匹配到的库图绑在一起，
         前端才能按"候选轨迹"而不是按"一堆散图"来渲染。

    证据"够不够、能不能退出"仍然由反思阶段按技能判定，这里不参与。
"""
from __future__ import annotations

import re
from typing import Any

# 工具结果里这些键下的值都是「可直接展示的证据 ID」
KEYFRAME_ID_KEYS = (
    "keyframeIds",
    "shownKeyframeIds",
    "matchedKeyframeIds",
    "queryKeyframeIds",
    "discardedKeyframeIds",
)
SEGMENT_ID_KEYS = ("shipSegmentIds", "shownShipSegmentIds", "segmentIds")
REFERENCE_ID_KEYS = (
    "registryReferenceIds",
    "shownRegistryReferenceIds",
    "matchedRegistryReferenceIds",
    "queryRegistryReferenceIds",
)
# 需要下钻的嵌套容器：工具结果常把明细放在这些键下
NESTED_KEYS = ("keyframes", "keyframesByTrack", "matches", "registryItems", "registryReferences", "tracks")


def _as_id_list(value: Any) -> list[str]:
    """把各种形态的值收敛成 ID 列表：字符串、字符串列表、或含 id 字段的字典列表。"""
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


def _record_id(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _tokens(key: str) -> list[str]:
    """把字段名切成小写词元：registryReferenceIds -> [registry, reference, ids]。

    先按驼峰边界与分隔符切分，再逐段小写——顺序反过来会把边界信息丢掉。
    """
    spaced = re.sub(r"[^0-9A-Za-z]+", " ", str(key))
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    return [part.lower() for part in spaced.split() if part]


def _same_kind(field: str, stems: list[list[str]]) -> bool:
    """字段名是否属于某一类 ID：去掉尾部 id/ids 后，按顺序出现该类的全部词元。

    referenceId 对 registryReferenceIds 成立（reference 命中词元子序列），
    trackId 对 keyframeIds 不成立（缺少 keyframe）——这正是要的分辨力。
    """
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


def _collect_ids(payload: Any, keys: tuple[str, ...], depth: int = 0) -> list[str]:
    """递归收集某一类证据 ID。

    两条路一起走：
      · keys 里列出的字段（keyframeIds / matchedKeyframeIds / ...）整列取值；
      · 嵌套明细里的**同类**身份字段（keyframes[].keyframeId、registryReferences[].referenceId）。
    类别词从 keys 推导（keyframe / shipsegment / registryreference），所以 trackId 不会被
    误收进关键帧那一路。深度限四层，避免在大结果集上白跑。
    """
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


def _dedup(values: list[str], limit: int) -> list[str]:
    """按出现顺序去重并限长；空串与空白先剔掉。"""
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


def _display_groups(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """一条轨迹一组：把该轨迹的关键帧、片段、库图绑在一起，并带上匹配分与结论档位。

    只在工具结果里出现过 trackId 时才建组；纯库图证据不建组（前端有单独的库证据列）。
    """
    groups: dict[str, dict[str, Any]] = {}

    def touch(track_id: str) -> dict[str, Any]:
        return groups.setdefault(track_id, {
            "trackId": track_id,
            "keyframeIds": [],
            "shipSegmentIds": [],
            "registryReferenceIds": [],
            "score": None,
            "band": "",
        })

    for record in records:
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        # 按轨迹分组的关键帧结果：键就是 trackId，值里是关键帧
        grouped = result.get("keyframesByTrack")
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
            group["keyframeIds"].extend(_collect_ids(candidate, KEYFRAME_ID_KEYS))
            group["shipSegmentIds"].extend(_collect_ids(candidate, SEGMENT_ID_KEYS))
            group["registryReferenceIds"].extend(_collect_ids(candidate, REFERENCE_ID_KEYS))
            score = candidate.get("embeddingScore", candidate.get("score"))
            if isinstance(score, (int, float)) and (group["score"] is None or score > group["score"]):
                group["score"] = float(score)
            band = str(candidate.get("scoreBand") or candidate.get("band") or "").strip()
            if band:
                group["band"] = band
    ordered = sorted(groups.values(), key=lambda item: (item["score"] is None, -(item["score"] or 0.0)))
    for group in ordered:
        group["keyframeIds"] = _dedup(group["keyframeIds"], limit)
        group["shipSegmentIds"] = _dedup(group["shipSegmentIds"], limit)
        group["registryReferenceIds"] = _dedup(group["registryReferenceIds"], limit)
    return ordered[:limit]


def build_evidence_payload(records: list[dict[str, Any]], *, limit: int = 60) -> dict[str, Any]:
    """把本轮的工具记录汇总成前端证据面板能直接渲染的载荷。

    ``records`` 是 ``_Trace.records``：每次工具调用一条，``result`` 是已解析的工具返回值。
    只读、不改动入参，重复调用结果一致。
    """
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

    evidence = {
        "keyframeIds": keyframes,
        "shipSegmentIds": segments,
        "registryReferenceIds": references,
        "displayGroups": _display_groups(successful, limit),
        "tools": [str(record.get("tool") or "") for record in successful],
    }
    evidence["isEmpty"] = not (keyframes or segments or references)
    return evidence

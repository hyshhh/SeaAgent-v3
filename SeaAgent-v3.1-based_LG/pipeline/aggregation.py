"""轨迹级舷号联合聚合。"""
from __future__ import annotations
from collections import Counter
from typing import Any
from memory import normalize_hull_number

def aggregate_keyframes(keyframes: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    support: dict[str | None, float] = {}
    counts: Counter[str | None] = Counter()
    for frame in keyframes:
        hull = normalize_hull_number(frame.get("vlmHullNumber")) if frame.get("hasReadableHullNumber") == "yes" else None
        hull = hull or None
        weight = 1.0 if hull else float(config.get("null_weight", 1 / 3))
        confidence = min(1.0, max(0.0, float(frame.get("readabilityConfidence", 0))))
        support[hull] = support.get(hull, 0.0) + weight * confidence
        counts[hull] += 1
    hulls = sorted((hull for hull in support if hull), key=lambda hull: support[hull], reverse=True)
    best = hulls[0] if hulls else None
    best_score = support.get(best, 0.0)
    second_score = support.get(hulls[1], 0.0) if len(hulls) > 1 else 0.0
    null_score = support.get(None, 0.0)
    if not best or null_score >= best_score:
        match_type, best = "unknown", None
    elif best_score - second_score < float(config.get("conflict_margin", 0.15)):
        match_type = "conflict"
    elif best_score >= float(config.get("confirmed_support", 1.5)) and counts[best] >= int(config.get("confirmed_count", 2)) and best_score - null_score >= float(config.get("null_margin", 0.3)):
        match_type = "confirmed"
    else:
        match_type = "candidate"
    descriptions = [str(frame.get("description") or "").strip() for frame in keyframes]
    description = "；".join(list(dict.fromkeys(item for item in descriptions if item))[:3])
    return {"finalHullNumber": best, "finalDescription": description, "finalMatchType": match_type, "classSupport": {str(key) if key else "null": round(value, 4) for key, value in support.items()}}

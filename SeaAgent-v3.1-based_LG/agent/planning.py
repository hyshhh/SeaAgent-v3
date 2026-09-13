"""Plan / Reflect 的确定性业务规则（原 graph.py 的 S5-S8 外移）。

这些函数原先内联在 agent/graph.py，与「谁在什么时候跑」的编排逻辑混在一个文件里。
外移后本模块只放**纯函数业务规则**：时间守卫、重规划指令族、计划生成与校验、
验收清单。graph.py 只保留状态契约、图装配与四个节点。

约束：
  - 本模块不得 import graph，以免循环依赖；graph 通过显式具名再导出使用这里的符号。
  - 所有函数都不写 state、不依赖节点闭包，只吃参数、吐结果。

文件分区（沿用原 S 编号）：
    S5  时间守卫        无显式时间时清空时间字段
    S6  重规划指令族    replan directive 与跨轮去重
    S7  计划生成与校验  _default_plan_calls 等确定性兜底
    S8  验收清单        _build_acceptance_progress，判定中枢
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tools import has_time_expression, normalize_time_range

from .plan_executor import PlanExecutor
from .task_profiles import registry_membership_list_mode, resolve_evidence_mode


def _normalize_broad_match_top_k(value: int | None) -> int:
    """规范化广泛库图匹配上限；0 表示不截断。"""
    try:
        return max(0, int(value if value is not None else 0))
    except (TypeError, ValueError):
        return 0


# ============================================================================
# S5 时间守卫
# 用户未显式给出时间时，强制清空时间字段并剥离时间过滤，避免模型凭空编造时间
# 范围。时间只影响检索参数，不进入验收项。
# ============================================================================
def _ground_intent_time(
    intent: dict[str, Any],
    question: str,
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """只允许用户原问题中明确出现的时间约束进入检索链。"""
    grounded = dict(intent or {})
    explicit = has_time_expression(question)
    grounded["hasExplicitTime"] = explicit
    if not explicit:
        # 模型可能把 referenceTime 误当成用户条件，或自行构造“最近一分钟”。
        # 无显式时间时必须查询全部监控记忆，不能保留任何模型/工具生成的范围。
        grounded["timeRange"] = None
        grounded["timeExpression"] = None
        grounded["queryScope"] = None
        grounded["timeSource"] = "all_monitoring_time"
        grounded.pop("timeParseError", None)
        return grounded

    normalized = normalize_time_range(question, now=reference_time)
    if normalized is not None:
        time_range = list(normalized)
        grounded["timeRange"] = time_range
        grounded["queryScope"] = time_range
        grounded["timeSource"] = "question"
    return grounded


def _enforce_plan_time_scope(
    calls: list[dict[str, Any]],
    intent: dict[str, Any],
) -> list[dict[str, Any]]:
    """落实意图查询范围：无时间不带范围，数量统计不做分页截断。"""
    strip_time = intent.get("hasExplicitTime") is False
    count_all = str(intent.get("operation") or "") == "count"
    if not strip_time and not count_all:
        return calls
    guarded: list[dict[str, Any]] = []
    for call in calls:
        item = dict(call)
        arguments = dict(item.get("arguments") or {})
        if strip_time:
            arguments.pop("timeRange", None)
        if count_all and str(item.get("tool") or "") == "getTrack":
            arguments["offset"] = 0
            arguments["limit"] = 0
        item["arguments"] = arguments
        guarded.append(item)
    return guarded


_CAPABILITY_TOOLS: dict[str, frozenset[str]] = {
    "registry_lookup": frozenset({"getRegistry"}),
    "registry_listing": frozenset({"listRegistry"}),
    "track_retrieval": frozenset({"getTrack"}),
    "keyframe_retrieval": frozenset({"getFrames"}),
    "image_matching": frozenset({"matchImage"}),
    "text_matching": frozenset({"matchText"}),
    "deduplication": frozenset({"dedupTracks"}),
}


# ============================================================================
# S6 重规划指令族
# reflect 判定需要重规划时，把「缺什么」编码成权威 directive 交给 plan；并做
# 跨轮去重，避免重复调用已完成的等价调用。
# ============================================================================
def _normalize_replan_directive(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    capabilities = []
    for raw in value.get("requiredCapabilities") or []:
        capability = str(raw or "").strip()
        if capability in _CAPABILITY_TOOLS and capability not in capabilities:
            capabilities.append(capability)
    target = value.get("target") if isinstance(value.get("target"), dict) else {}
    return {
        "objective": str(value.get("objective") or "complete_missing_evidence"),
        "requiredCapabilities": capabilities,
        "requiredEvidence": [str(item) for item in (value.get("requiredEvidence") or []) if str(item)],
        "target": dict(target),
        "reuseScopeKeys": [str(item) for item in (value.get("reuseScopeKeys") or []) if str(item)],
        "avoidRepeatCallIds": [str(item) for item in (value.get("avoidRepeatCallIds") or []) if str(item)],
    }


def _merge_replan_directives(authoritative: Any, proposed: Any) -> dict[str, Any]:
    """保留验收层要求的能力，同时允许 ReflectAgent 补充目标与复用信息。"""
    base = _normalize_replan_directive(authoritative)
    extra = _normalize_replan_directive(proposed)
    capabilities = list(base.get("requiredCapabilities") or [])
    for capability in extra.get("requiredCapabilities") or []:
        if capability not in capabilities:
            capabilities.append(capability)
    evidence = list(base.get("requiredEvidence") or [])
    for item in extra.get("requiredEvidence") or []:
        if item not in evidence:
            evidence.append(item)
    target = dict(extra.get("target") or {})
    target.update(base.get("target") or {})
    reuse = list(base.get("reuseScopeKeys") or [])
    for item in extra.get("reuseScopeKeys") or []:
        if item not in reuse:
            reuse.append(item)
    avoid = list(base.get("avoidRepeatCallIds") or [])
    for item in extra.get("avoidRepeatCallIds") or []:
        if item not in avoid:
            avoid.append(item)
    return {
        "objective": str(extra.get("objective") or base.get("objective") or "complete_missing_evidence"),
        "requiredCapabilities": capabilities,
        "requiredEvidence": evidence,
        "target": target,
        "reuseScopeKeys": reuse,
        "avoidRepeatCallIds": avoid,
    }


def _track_retrieval_has_full_coverage(
    result: dict[str, Any],
    arguments: dict[str, Any] | None = None,
) -> bool:
    """判断轨迹检索是否覆盖目标范围，避免把分页结果误当作全量候选。"""
    if not isinstance(result.get("trackIds"), list):
        return False
    returned = result.get("returnedTrackCount")
    total = result.get("totalTrackCount")
    try:
        if returned is not None and total is not None:
            return int(returned) >= int(total)
    except (TypeError, ValueError):
        pass
    try:
        return int((arguments or {}).get("limit")) == 0
    except (TypeError, ValueError):
        # 兼容未返回计数元数据的旧工具结果；已有明确列表时仍允许复用。
        return True


def _completed_agent_capabilities(
    tool_records: list[dict[str, Any]],
) -> tuple[set[str], list[str]]:
    completed: set[str] = set()
    reusable_ids: list[str] = []
    for record in tool_records or []:
        if not isinstance(record, dict) or record.get("ok") is False or record.get("skipped"):
            continue
        tool_name = str(record.get("tool") or "")
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
        call_id = str(record.get("id") or "")
        if call_id:
            reusable_ids.append(call_id)
        if tool_name == "getRegistry":
            completed.add("registry_lookup")
        elif tool_name == "listRegistry":
            completed.add("registry_listing")
        elif tool_name == "getTrack" and not str(arguments.get("hullNumber") or "").strip():
            if _track_retrieval_has_full_coverage(result, arguments):
                completed.add("track_retrieval")
        elif tool_name == "getFrames" and result.get("keyframes"):
            completed.add("keyframe_retrieval")
        elif tool_name == "matchImage" and not result.get("error"):
            completed.add("image_matching")
        elif tool_name == "matchText" and not result.get("error"):
            completed.add("text_matching")
        elif tool_name == "dedupTracks" and not result.get("error"):
            completed.add("deduplication")
    return completed, reusable_ids


def _build_replan_directive(
    intent: dict[str, Any],
    acceptance_progress: dict[str, Any],
    *,
    working_scope: dict[str, Any] | None = None,
    tool_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """把验收缺口转换为能力目标；不把具体问题文本或固定工具链作为规划条件。"""
    missing_keys = [
        str(item.get("key") or "")
        for item in (acceptance_progress.get("requirements") or [])
        if isinstance(item, dict) and not item.get("completed")
    ]
    target_kind = str(intent.get("targetKind") or "all")
    capability_map: dict[str, list[str]] = {
        "registry_lookup": ["registry_lookup"],
        "registry": ["registry_lookup" if target_kind == "hull" else "registry_listing"],
        "tracks": ["track_retrieval"],
        "frames": ["keyframe_retrieval"],
        "image_match": ["track_retrieval", "keyframe_retrieval", "image_matching"],
        "registry_text_match": ["registry_listing", "text_matching"],
        "text_match": ["track_retrieval", "keyframe_retrieval", "text_matching"],
        "dedup": ["track_retrieval", "keyframe_retrieval", "deduplication"],
    }
    requested: list[str] = []
    for key in missing_keys:
        for capability in capability_map.get(key, []):
            if capability not in requested:
                requested.append(capability)

    completed, reusable_call_ids = _completed_agent_capabilities(tool_records or [])
    required = [capability for capability in requested if capability not in completed]
    reusable_scope_keys = [
        str(key)
        for key, value in (working_scope or {}).items()
        if isinstance(value, dict) and value.get("ok") is not False
    ][-24:]
    target = {
        "kind": target_kind,
        "scope": str(intent.get("targetScope") or "track_memory"),
        "operation": str(intent.get("operation") or ""),
    }
    if str(intent.get("hullNumber") or "").strip():
        target["hullNumber"] = str(intent.get("hullNumber") or "").strip()
    if str(intent.get("description") or "").strip() and target_kind != "hull":
        target["description"] = str(intent.get("description") or "").strip()
    if intent.get("timeRange"):
        target["timeRange"] = intent.get("timeRange")
    return {
        "objective": "complete_missing_evidence",
        "requiredCapabilities": required,
        "requiredEvidence": missing_keys,
        "target": target,
        "reuseScopeKeys": reusable_scope_keys,
        "avoidRepeatCallIds": reusable_call_ids[-24:],
    }


def _plan_directive_issues(calls: list[dict[str, Any]], directive: dict[str, Any]) -> list[str]:
    normalized = _normalize_replan_directive(directive)
    required = normalized.get("requiredCapabilities") or []
    if not required:
        return []
    planned_tools = {str(call.get("tool") or "") for call in calls if isinstance(call, dict)}
    issues: list[str] = []
    for capability in required:
        if not planned_tools.intersection(_CAPABILITY_TOOLS.get(capability, frozenset())):
            issues.append(f"missing_capability:{capability}")
    return issues


def _remove_completed_call_repeats(
    calls: list[dict[str, Any]],
    tool_records: list[dict[str, Any]],
    working_scope: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """移除跨轮次等价调用，并把其下游引用改写到已有工作域结果。"""
    previous: dict[str, str] = {}
    for record in tool_records or []:
        if not isinstance(record, dict) or record.get("ok") is False or record.get("skipped"):
            continue
        tool_name = str(record.get("tool") or "")
        arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
        call_id = str(record.get("id") or "")
        if not tool_name or not call_id or call_id not in working_scope:
            continue
        previous[PlanExecutor.semantic_signature(tool_name, arguments)] = call_id

    aliases: dict[str, str] = {}
    retained: list[dict[str, Any]] = []
    removed: list[str] = []
    for call in calls:
        item = dict(call)
        item["arguments"] = PlanExecutor._rewrite_call_refs(item.get("arguments") or {}, aliases)
        if isinstance(item.get("condition"), dict):
            item["condition"] = PlanExecutor._rewrite_call_refs(item["condition"], aliases)
        resolved_arguments = PlanExecutor.resolve_references(
            item.get("arguments") or {},
            working_scope,
        )
        signature = PlanExecutor.semantic_signature(
            str(item.get("tool") or ""),
            resolved_arguments if isinstance(resolved_arguments, dict) else item.get("arguments") or {},
        )
        existing_id = previous.get(signature)
        if existing_id:
            aliases[str(item.get("id") or "")] = existing_id
            removed.append(str(item.get("id") or item.get("tool") or ""))
            continue
        retained.append(item)
    for item in retained:
        item["arguments"] = PlanExecutor._rewrite_call_refs(item.get("arguments") or {}, aliases)
        if isinstance(item.get("condition"), dict):
            item["condition"] = PlanExecutor._rewrite_call_refs(item["condition"], aliases)
    return retained, removed


# ============================================================================
# S7 计划生成与校验
# 模型不给计划、或计划不合法时的确定性兜底：
#   _default_plan_calls        按意图装配标准调用序列（含 $ref 引用）
#   _apply_retrieval_limits    按 evidenceMode 调整检索体量
#   _attach_dependency_conditions / _prepare_plan_calls  校验并修复计划
#   _find_tool_contract_failures  检出违反工具契约的调用
# ============================================================================
def _default_plan_calls(
    intent: dict[str, Any],
    top_k: int,
    broad_match_top_k: int = 0,
    *,
    replan_hint: str = "",
    replan_directive: dict[str, Any] | None = None,
    working_scope: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """模型未给出 calls 时的最小可执行链（结构化 $ref，不是业务硬编码分支表）。"""
    hull = str(intent.get("hullNumber") or "").strip()
    description = str(intent.get("description") or "").strip()
    time_range = None if intent.get("hasExplicitTime") is False else intent.get("timeRange")
    operation = str(intent.get("operation") or "list")
    target_scope = str(intent.get("targetScope") or "track_memory")
    top = max(1, min(20, int(top_k or 3)))
    broad_top = _normalize_broad_match_top_k(broad_match_top_k)
    # focused 证据模式：单目标判断只对少量候选轨迹取证，控制关键帧与匹配开销
    evidence_mode = resolve_evidence_mode(intent)
    focused = evidence_mode == "focused"
    frame_slice = top * 4 if focused else None
    directive = _normalize_replan_directive(replan_directive)
    required_capabilities = set(directive.get("requiredCapabilities") or [])
    hint_raw = str(replan_hint or intent.get("nextAgentFocus") or "")
    hint = hint_raw.lower()
    has_capability_directive = bool(required_capabilities)
    if has_capability_directive:
        # 再规划阶段只读取结构化能力契约，界面摘要不参与业务分支判断。
        wants_registry = bool(required_capabilities.intersection({"registry_lookup", "registry_listing"}))
        wants_visual_match = "image_matching" in required_capabilities
    else:
        wants_registry = (
            target_scope in {"registry", "both"}
            or str(intent.get("registryRelation") or "any") in {"in", "out"}
            or any(token in hint for token in ("先验库", "在库", "未在库", "getregistry", "listregistry", "matchhull", "registry"))
        )
        # 首轮安全兜底仍可结合结构化意图中的阶段摘要选择最小链。
        wants_visual_match = any(
            token in hint
            for token in (
                "matchimage", "match_image", "视觉匹配", "图像匹配", "图匹配",
                "registryreferences", "关键帧匹配", "库图", "对照视频",
            )
        ) or ("match" in hint and "image" in hint)
    registry_relation = str(intent.get("registryRelation") or "any")
    # 「有哪些在库船出现」：list + in + both/all，禁止当描述 matchText
    wants_registry_in_list = (
        registry_relation == "in"
        and operation == "list"
        and not hull
        and (
            target_scope in {"both", "registry"}
            or str(intent.get("questionType") or "") == "registry_in_list"
            or any(token in hint for token in ("listregistry", "在库", "哪些", "matchimage", "matchhull"))
        )
    )
    # 伪描述：整句问法残留，不当 matchText description
    bogus_description = bool(
        description
        and (
            any(token in description for token in ("哪些", "在库", "先验库", "有哪些", "未在库"))
            or description in {"船", "船舶", "船只", "目标"}
        )
    )
    if bogus_description:
        description = ""

    # 纯数据库范围优先级最高，必须在任何“视觉匹配”关键词判断之前截断视频工具链。
    if target_scope == "registry":
        if hull:
            return [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}]
        if description:
            return [
                {"id": "registry", "tool": "listRegistry", "arguments": {}},
                {
                    "id": "match",
                    "tool": "matchText",
                    "arguments": {
                        "description": description,
                        "galleryImages": {
                            "$ref": "registry.registryReferences",
                            "$default": {"$ref": "registry.registryItems"},
                        },
                        "topK": top,
                    },
                },
            ]
        return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    def _track_args(*, with_hull: bool, all_tracks: bool = False) -> dict[str, Any]:
        args: dict[str, Any] = {"offset": 0, "limit": 0 if all_tracks else 60}
        if time_range:
            args["timeRange"] = time_range
        if with_hull and hull:
            args["hullNumber"] = hull
        return args

    def _match_image_from_list_registry() -> list[dict[str, Any]]:
        """生成全库对照链；再规划时复用上一轮已取得的轨迹与关键帧。"""
        scope = working_scope or {}
        track_scope_id = "tracks"
        frame_scope_id = "frames"
        registry_scope_id = "registry"
        tracks_result: dict[str, Any] = {}
        frames_result: dict[str, Any] = {}
        registry_result: dict[str, Any] = {}
        # 模型可能使用自定义 call id；按结果字段识别并复用，避免第二轮重复检索。
        for scope_id, value in reversed(list(scope.items())):
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            if not tracks_result and isinstance(value.get("trackIds"), list):
                track_scope_id, tracks_result = str(scope_id), value
            if not frames_result and isinstance(value.get("keyframes"), list):
                frame_scope_id, frames_result = str(scope_id), value
            if not registry_result and isinstance(value.get("registryItems"), list):
                registry_scope_id, registry_result = str(scope_id), value

        tracks_ready = _track_retrieval_has_full_coverage(tracks_result)
        track_ids = list(tracks_result.get("trackIds") or []) if tracks_ready else []
        # 部分分页轨迹对应的关键帧也不能当作全量候选复用。
        frames_ready = tracks_ready and bool(frames_result.get("keyframes"))
        registry_ready = isinstance(registry_result.get("registryItems"), list)

        # 已明确无轨迹时，图像匹配没有视频侧候选，禁止查整库或制造 galleryImages=null 的伪调用。
        if tracks_ready and not track_ids:
            return []

        calls: list[dict[str, Any]] = []
        if not registry_ready:
            registry_scope_id = "registry"
            calls.append({"id": registry_scope_id, "tool": "listRegistry", "arguments": {}})
        if not tracks_ready:
            track_scope_id = "tracks"
            calls.append({"id": track_scope_id, "tool": "getTrack", "arguments": _track_args(with_hull=False, all_tracks=True)})
        if not frames_ready:
            frame_scope_id = "frames"
            calls.append({
                "id": frame_scope_id,
                "tool": "getFrames",
                "arguments": {"trackIds": {"$ref": f"{track_scope_id}.trackIds"}},
                "condition": {"ref": f"{track_scope_id}.trackIds"},
            })
        calls.append({
            "id": "match",
            "tool": "matchImage",
            "arguments": {
                "queryImages": {"$ref": f"{registry_scope_id}.registryReferences"},
                "galleryImages": {"$ref": f"{frame_scope_id}.keyframes"},
                "registryItems": {"$ref": f"{registry_scope_id}.registryItems"},
                "topK": broad_top,
            },
            "condition": {"ref": f"{frame_scope_id}.keyframes"},
        })
        return calls

    # 结构化能力目标只要求库查询时，生成最小库检索步骤；不解析 nextAction 文本。
    if has_capability_directive and not wants_visual_match:
        if hull and "registry_lookup" in required_capabilities:
            return [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}]
        if "registry_listing" in required_capabilities:
            return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    # 结构化图像匹配能力要求直接生成可复用的库图—关键帧对照链。
    if has_capability_directive and wants_visual_match and not hull:
        return _match_image_from_list_registry()

    # 在库船列表（视频中出现的库船）：listRegistry → getTrack → getFrames → matchImage
    if wants_registry_in_list or (wants_visual_match and not hull and wants_registry and not description):
        return _match_image_from_list_registry()

    # 指定库船存在性核验：复用已取得的库项，全量扫描视频轨迹并只做一次完整图像匹配。
    if wants_visual_match and hull:
        scope = working_scope or {}
        registry_scope_id = "registry"
        registry_ready = False
        for scope_id, value in reversed(list(scope.items())):
            if not isinstance(value, dict) or value.get("ok") is False:
                continue
            if isinstance(value.get("registryItems"), list) and value.get("registryItems"):
                registry_scope_id = str(scope_id)
                registry_ready = True
                break

        calls: list[dict[str, Any]] = []
        if not registry_ready:
            calls.append({"id": registry_scope_id, "tool": "getRegistry", "arguments": {"hullNumber": hull}})
        calls.extend([
            {"id": "tracks", "tool": "getTrack", "arguments": _track_args(with_hull=False, all_tracks=True)},
            {
                "id": "frames",
                "tool": "getFrames",
                # focused 模式只对少量候选轨迹取帧，避免全量关键帧开销
                "arguments": {"trackIds": {"$ref": "tracks.trackIds", "$slice": frame_slice} if frame_slice else {"$ref": "tracks.trackIds"}},
                "condition": {"ref": "tracks.trackIds"},
            },
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    # 只引用参考图列表；执行器会再从 registryItems.references 展开补齐。
                    "queryImages": {"$ref": f"{registry_scope_id}.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "registryItems": {"$ref": f"{registry_scope_id}.registryItems"},
                    # focused 单目标核验用普通 topK，broad 全库对照才不截断
                    "topK": top if focused else broad_top,
                },
                "condition": {"ref": "frames.keyframes"},
            },
        ])
        return calls
    if wants_visual_match and description:
        return [
            {"id": "tracks", "tool": "getTrack", "arguments": _track_args(with_hull=False)},
            {
                "id": "frames",
                "tool": "getFrames",
                "arguments": {"trackIds": {"$ref": "tracks.trackIds", "$slice": frame_slice} if frame_slice else {"$ref": "tracks.trackIds"}},
            },
            {
                "id": "match",
                "tool": "matchText",
                "arguments": {
                    "description": description,
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": top,
                },
            },
        ]

    if wants_registry and "gettrack" not in hint and operation != "count" and not wants_visual_match and not wants_registry_in_list:
        if hull:
            # 查库后仍可能需要视觉匹配；若 hint 只写查库则先 getRegistry
            return [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}}]
        if description:
            return [
                {"id": "registry", "tool": "listRegistry", "arguments": {}},
                {
                    "id": "match",
                    "tool": "matchText",
                    "arguments": {
                        "description": description,
                        "galleryImages": {
                            "$ref": "registry.registryReferences",
                            "$default": {"$ref": "registry.registryItems"},
                        },
                        "topK": top,
                    },
                },
            ]
        # 无描述的在库关系：给完整对照链，避免只 list 库
        if registry_relation in {"in", "out"}:
            return _match_image_from_list_registry()
        return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    # replan 明确要求查库（尚未要求视觉匹配）
    if wants_registry and any(token in hint for token in ("先验库", "getregistry", "listregistry", "matchhull", "在库")):
        if hull:
            return [
                {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": hull}},
                {"id": "matchHull", "tool": "matchHull", "arguments": {"hullNumberArray": [hull]}},
            ]
        if registry_relation in {"in", "out"} or wants_registry_in_list:
            return _match_image_from_list_registry()
        return [{"id": "registry", "tool": "listRegistry", "arguments": {}}]

    track_args = _track_args(with_hull=bool(hull), all_tracks=operation == "count")
    calls: list[dict[str, Any]] = [
        {"id": "tracks", "tool": "getTrack", "arguments": track_args},
    ]
    if operation == "count":
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        calls.append({
            "id": "dedup",
            "tool": "dedupTracks",
            "arguments": {
                "tracks": {"$ref": "tracks.tracks"},
                "keyframesByTrack": {"$ref": "frames.keyframesByTrack"},
            },
        })
        return calls
    if description:
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        calls.append({
            "id": "match",
            "tool": "matchText",
            "arguments": {
                "description": description,
                "galleryImages": {"$ref": "frames.keyframes"},
                "topK": top,
            },
        })
        return calls
    if hull:
        calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
        return calls
    # both + in 且无描述：默认在库对照视觉链
    if wants_registry and registry_relation == "in":
        return _match_image_from_list_registry()
    calls.append({"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}})
    return calls


def _apply_retrieval_limits(
    calls: list[dict[str, Any]],
    *,
    broad_match_top_k: int,
    broad_match_context: bool = False,
    intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """修正模型计划中的广泛库图匹配参数，避免退回普通检索上限。"""
    has_list_registry = False
    has_match_image = False
    has_unfiltered_track = False
    for call in calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool") or "")
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if tool_name == "listRegistry":
            has_list_registry = True
        elif tool_name == "matchImage":
            has_match_image = True
        elif tool_name == "getTrack" and not str(arguments.get("hullNumber") or "").strip():
            has_unfiltered_track = True

    # focused 单目标核验不做广泛匹配强制（matchImage 由计划链决定 topK）
    if intent is not None and resolve_evidence_mode(intent) == "focused":
        return calls

    # 全库对全轨迹、以及“指定库船是否在视频出现”的全轨迹核验，都属于广泛匹配。
    # 后者必须一次返回全部轨迹评分，不能先按普通 topK 截断后再重复匹配。
    is_broad_match = bool(
        has_match_image
        and (
            (has_list_registry and has_unfiltered_track)
            or broad_match_context
        )
    )
    if not is_broad_match:
        return calls

    broad_top = _normalize_broad_match_top_k(broad_match_top_k)
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = dict(call)
        arguments = dict(item.get("arguments") or {})
        tool_name = str(item.get("tool") or "")
        if tool_name == "getTrack" and not str(arguments.get("hullNumber") or "").strip():
            arguments["offset"] = 0
            arguments["limit"] = 0
        elif tool_name == "matchImage":
            arguments["topK"] = broad_top
        item["arguments"] = arguments
        normalized.append(item)
    return normalized


def _attach_dependency_conditions(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为引用上游列表的昂贵工具自动添加非空条件，阻止空参数伪调用。"""
    guarded: list[dict[str, Any]] = []
    for call in calls:
        item = dict(call)
        if isinstance(item.get("condition"), dict):
            guarded.append(item)
            continue
        tool_name = str(item.get("tool") or "")
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        dependency: Any = None
        if tool_name == "getFrames":
            dependency = arguments.get("trackIds")
        elif tool_name == "matchImage":
            dependency = arguments.get("galleryImages")
        if isinstance(dependency, dict) and isinstance(dependency.get("$ref"), str):
            item["condition"] = {"ref": dependency["$ref"]}
        guarded.append(item)
    return guarded


def _prepare_plan_calls(
    calls: Any,
    intent: dict[str, Any],
    top_k: int,
    *,
    broad_match_top_k: int,
    broad_match_context: bool = False,
    replan_hint: str = "",
    replan_directive: dict[str, Any] | None = None,
    working_scope: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """校验模型计划的工具参数契约；无效计划直接替换为确定性正确链。"""
    sanitized = PlanExecutor.sanitize_calls(calls)
    issues = PlanExecutor.call_contract_issues(sanitized)
    if str(intent.get("targetScope") or "") == "registry":
        forbidden = {
            str(call.get("tool") or "")
            for call in sanitized
            if str(call.get("tool") or "") in {
                "getTrack", "getFrames", "getClip", "matchImage", "dedupTracks"
            }
        }
        if forbidden:
            issues.append(f"scope_violation:registry_only:{','.join(sorted(forbidden))}")
    repair = ""
    if issues:
        sanitized = _default_plan_calls(
            intent,
            top_k,
            broad_match_top_k=broad_match_top_k,
            replan_hint=replan_hint,
            replan_directive=replan_directive,
            working_scope=working_scope or {},
        )
        repair = "；".join(issues)
    normalized = _apply_retrieval_limits(
        sanitized,
        broad_match_top_k=broad_match_top_k,
        broad_match_context=broad_match_context,
        intent=intent,
    )
    normalized = _enforce_plan_time_scope(normalized, intent)
    normalized = _attach_dependency_conditions(normalized)
    return normalized, repair


def _find_tool_contract_failures(records: list[dict[str, Any]], round_number: int) -> list[str]:
    failures: list[str] = []
    for record in records or []:
        if not isinstance(record, dict) or int(record.get("round") or 0) != int(round_number):
            continue
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        error = str(record.get("error") or result.get("error") or "")
        if any(token in error for token in (
            "argument_not_allowed:", "argument_invalid:", "tool_not_allowed:",
            "unexpected keyword argument", "dedup_tracks_",
        )):
            failures.append(f"{record.get('tool')}: {error}")
    return failures


# ============================================================================
# S8 验收清单
# 编排的判定中枢：按 targetScope / targetKind / operation / description 分派，
# 生成 requirements / pending / completed，并以「有工具证据 且 pending 为空」
# 定义 acceptanceSatisfied。reflect 的全部路由与 controller 的结论合成都以它
# 为依据。注意：本函数不读取任何时间字段。
# ============================================================================
def _build_acceptance_progress(
    intent: dict[str, Any],
    tool_names: set[str],
    *,
    track_count: int | None,
    registry_checked: bool,
    registry_listed: bool,
    registry_has_items: bool,
    can_try_visual: bool,
    visual_attempted: bool,
    match_image_attempted: bool,
    match_image_usable: bool,
    has_tool_evidence: bool,
    match_image_blocked: bool = False,
    registry_coverage_complete: bool | None = None,
    dedup_usable: bool = False,
) -> dict[str, Any]:
    """把验收标准转换为可执行清单，供 Reflect 决定结束或进入下一轮。"""
    mode = registry_membership_list_mode(intent)
    operation = str(intent.get("operation") or "")
    target_scope = str(intent.get("targetScope") or "track_memory")
    target_kind = str(intent.get("targetKind") or "all")
    hull = str(intent.get("hullNumber") or "").strip()
    description = str(intent.get("description") or "").strip()
    registry_only = target_scope == "registry"
    requirements: list[dict[str, Any]] = []

    def require(key: str, label: str, completed: bool) -> None:
        requirements.append({"key": key, "label": label, "completed": bool(completed)})

    if registry_only:
        # 纯数据库问题的证据边界止于先验库，禁止自动扩展到视频轨迹与关键帧。
        if target_kind == "hull" and hull:
            require("registry_lookup", f"已完成舷号 {hull} 的数据库精确查询", "getRegistry" in tool_names)
        else:
            require("registry", "已获取先验数据库记录", registry_listed)
            if description:
                require(
                    "registry_text_match",
                    "已完成描述与数据库参考图/库项匹配，或数据库已明确为空",
                    "matchText" in tool_names or (registry_listed and not registry_has_items),
                )
    elif mode:
        # 在库/未在库列表首先取决于视频侧是否存在候选目标。
        # 全量轨迹明确为 0 时，答案已是“没有船舶出现”，无需查询整库，更不能制造空 gallery 的匹配调用。
        require("tracks", "已完成全量视频轨迹检索", track_count is not None)
        if track_count is not None and track_count > 0:
            require("frames", "已获取全部候选轨迹关键帧", "getFrames" in tool_names)
            require("registry", "已获取完整先验库名录", registry_listed)
            if registry_listed and registry_has_items:
                require(
                    "image_match",
                    "已完成全部库图与全部轨迹关键帧匹配，或已确认匹配输入不可用",
                    match_image_usable or match_image_blocked,
                )
    elif operation == "count":
        require("tracks", "已获取视频轨迹", "getTrack" in tool_names)
        require("frames", "已获取轨迹关键帧", "getFrames" in tool_names)
        require("dedup", "已完成跨轨迹去重计数", dedup_usable)
    elif hull and operation == "existence":
        # 舷号是强结构化目标，优先级必须高于解析器残留的描述片段（如“大鱼01 在”）。
        # 否则会错误要求 matchText，并在已完成 matchImage 后继续重复全量匹配。
        require("tracks", f"已按舷号 {hull} 检索视频轨迹", "getTrack" in tool_names)
        if track_count == 0:
            require("registry", "视频未直接命中后已查询先验库", registry_checked)
            if registry_checked and can_try_visual:
                require("image_match", "已有库图时已完成库图与视频关键帧匹配", visual_attempted)
        elif registry_checked and can_try_visual:
            require("image_match", "已有库图时已完成库图与视频关键帧匹配", visual_attempted)
    elif description:
        require("tracks", "已获取视频轨迹", "getTrack" in tool_names)
        require("frames", "已获取轨迹关键帧", "getFrames" in tool_names)
        require("text_match", "已完成描述与关键帧匹配", "matchText" in tool_names)
    else:
        require("tracks", "已完成视频轨迹检索", "getTrack" in tool_names)

    pending = [item["label"] for item in requirements if not item["completed"]]
    completed = [item["label"] for item in requirements if item["completed"]]
    if registry_only and pending:
        missing_keys = {item["key"] for item in requirements if not item["completed"]}
        if "registry_lookup" in missing_keys:
            next_action = f"getRegistry(hullNumber={hull})"
        elif "registry" in missing_keys:
            next_action = "listRegistry；若存在描述条件则继续 matchText，并复用 registry.registryReferences"
        else:
            next_action = (
                f"matchText(description={description}, "
                "galleryImages=$ref registry.registryReferences)，禁止调用 getTrack/getFrames"
            )
    elif registry_only:
        next_action = "数据库查询验收清单已满足，可直接结束；禁止扩展到视频检索"
    elif mode and track_count == 0 and not pending:
        next_action = "全量视频轨迹为 0，可直接结束；无需 listRegistry 或 matchImage"
    elif mode and pending:
        missing_keys = {item["key"] for item in requirements if not item["completed"]}
        if "tracks" in missing_keys:
            next_action = "getTrack(全量，不带hullNumber, limit=0)；仅在 trackIds 非空时继续 getFrames"
        elif "frames" in missing_keys:
            next_action = "getFrames(复用已有全量 trackIds) → listRegistry → matchImage"
        else:
            next_action = (
                "listRegistry → matchImage(queryImages=$ref registry.registryReferences, "
                "galleryImages=$ref frames.keyframes)，复用上一轮全量轨迹与关键帧"
            )
    elif pending:
        next_action = str(intent.get("nextAgentFocus") or pending[0])
    else:
        next_action = "验收清单已满足，可结束循环"

    return {
        "mode": "registry_only" if registry_only else (mode or "general"),
        "goal": intent.get("successCriteria") or intent.get("expectedOutcome") or "工具证据足以回答用户问题",
        "currentFocus": intent.get("nextAgentFocus"),
        "requirements": requirements,
        "completedRequirements": completed,
        "pendingRequirements": pending,
        "acceptanceSatisfied": bool(has_tool_evidence and not pending),
        "nextAction": next_action,
        "matchImageAttempted": match_image_attempted,
        "matchImageBlocked": match_image_blocked,
        "registryCoverageComplete": registry_coverage_complete,
        "registryCoverageLimited": registry_coverage_complete is False,
        "dedupUsable": dedup_usable,
        "terminalState": "uncertain" if (match_image_blocked or registry_coverage_complete is False) else None,
        "videoEmptyShortCircuit": bool(mode and track_count == 0 and not pending),
    }

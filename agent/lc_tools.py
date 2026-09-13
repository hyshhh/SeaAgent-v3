"""将现有 ToolService / 解析函数封装为 LangChain tools。

定位：这是**模型直调**路径的工具封装，与 plan_executor.py 的**确定性执行**
路径并存。当前 graph 只用了本文件的 intent 工具与 loadSkill ——
observe 的业务工具走 PlanExecutor.execute，build_observe_tools 无调用方。

文件分区（以 L 编号为锚点检索）：
    L1  序列化辅助与参数 schema   _jsonable / _dump / 各 *Args
    L2  intent 工具                parseTime / parseTargets / extractHull
    L3  结果压缩                   完整工具结果 → 回传模型的摘要
    L4  observe 工具                业务工具封装（含会话缓存与自动补参）
    L5  技能加载工具                loadSkill
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.target_parser import extract_hull_number, extract_target_items
from tools.time_normalizer import normalize_time_range


# ===========================================================================
# L1 序列化辅助与参数 schema
# _jsonable / _dump 负责把工具结果转成可 JSON 化的形态；*Args 是各工具的参数
# schema —— 它们的 description 会进入模型看到的工具说明，是提示词的一部分。
# ===========================================================================
def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _dump(result: Any) -> str:
    return json.dumps(_jsonable(result), ensure_ascii=False)


class ParseTimeArgs(BaseModel):
    expression: str = Field(description="自然语言时间表达，如昨天下午")


class ParseTargetsArgs(BaseModel):
    question: str = Field(description="用户原问题，用于多目标切分")


class ExtractHullArgs(BaseModel):
    question: str = Field(description="用户原问题，用于抽取舷号")


class GetTrackArgs(BaseModel):
    timeRange: list[float] | None = Field(default=None, description="Unix 秒 [start, end]")
    hullNumber: str | None = Field(default=None, description="舷号")
    finalMatchType: str | None = Field(default=None)
    offset: int = Field(default=0)
    limit: int = Field(default=60)


class GetFramesArgs(BaseModel):
    trackIds: list[str | int] = Field(description="轨迹 ID 列表")


class GetClipArgs(BaseModel):
    trackId: str | int
    timeRange: list[float] | None = None
    scale: float | None = None


class GetRegistryArgs(BaseModel):
    hullNumber: str


class MatchHullArgs(BaseModel):
    hullNumberArray: list[str | None]


class MatchTextArgs(BaseModel):
    description: str = Field(description="外观/类别描述，如黄色无人艇")
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。关键帧列表或 getFrames 结果；省略时自动使用本轮最近 getFrames/listRegistry 结果",
    )
    topK: int | None = None


class MatchImageArgs(BaseModel):
    queryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。查询侧图像；省略时自动用最近 getFrames/listRegistry",
    )
    galleryImages: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="可选。图库侧图像；省略时自动用最近另一侧结果",
    )
    topK: int | None = None


class VerifyTargetArgs(BaseModel):
    description: str | None = None
    registryReferenceIds: list[str] | None = None
    keyframeIds: list[str] | None = None
    shipSegmentIds: list[str] | None = None


class ShowEvidenceArgs(BaseModel):
    keyframeIds: list[str] | None = None
    shipSegmentIds: list[str] | None = None
    registryReferenceIds: list[str] | None = None


class DedupTracksArgs(BaseModel):
    tracks: list[dict[str, Any]]
    keyframesByTrack: dict[str, Any]


class ListRegistryArgs(BaseModel):
    dummy: str | None = Field(default=None, description="无需参数，可忽略")


class LoadSkillArgs(BaseModel):
    skillId: str = Field(description="catalog 中的 skill id")


# ===========================================================================
# L2 intent 工具
# 三个纯解析工具（时间/多目标/舷号），由 IntentAgent 调用。返回的是「待确认」
# 的建议值，节点侧还会用 infer_intent_fields 做规则回填兜底。
# ===========================================================================
def build_intent_tools(reference_time: datetime | None = None) -> list[StructuredTool]:
    now = reference_time or datetime.now().astimezone()

    def parse_time(expression: str) -> str:
        rng = normalize_time_range(expression, now=now)
        return _dump({
            "ok": True,
            "timeRange": list(rng) if rng else None,
            "expression": expression,
            "hint": "确认后写入意图 timeRange",
        })

    def parse_targets(question: str) -> str:
        return _dump({
            "ok": True,
            "targetItems": extract_target_items(question),
            "hint": "确认后写入意图 targetItems",
        })

    def extract_hull(question: str) -> str:
        return _dump({"ok": True, "hullNumber": extract_hull_number(question)})

    return [
        StructuredTool.from_function(
            name="parseTime",
            description="仅当用户原问题明确包含时间表达时调用，将该表达归一化为 Unix 秒区间 [start, end]；禁止为无时间问题生成默认范围。",
            func=parse_time,
            args_schema=ParseTimeArgs,
        ),
        StructuredTool.from_function(
            name="parseTargets",
            description="从问题中切分多个船舶目标",
            func=parse_targets,
            args_schema=ParseTargetsArgs,
        ),
        StructuredTool.from_function(
            name="extractHull",
            description="从问题中抽取疑似舷号",
            func=extract_hull,
            args_schema=ExtractHullArgs,
        ),
    ]


# ===========================================================================
# L3 结果压缩
# 把各工具返回的大列表压成模型可用的小样本；完整数据仍留在会话缓存里，由
# L4 的 _auto_fill 自动注入，模型无需复述。这是「大 JSON 不进对话」的落点。
# ===========================================================================
def _compact_track(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trackId": item.get("trackId") or item.get("id"),
        "hullNumber": item.get("finalHullNumber") or item.get("hullNumber"),
        "startTime": item.get("startTime") or item.get("start_time"),
        "endTime": item.get("endTime") or item.get("end_time"),
        "finalMatchType": item.get("finalMatchType"),
    }


def _compact_keyframe(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "keyframeId": item.get("keyframeId") or item.get("id"),
        "trackId": item.get("trackId"),
        "timestamp": item.get("timestamp") or item.get("time"),
        "hullNumber": item.get("hullNumber") or item.get("finalHullNumber"),
    }


def _compact_registry(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "registryId": item.get("registryId") or item.get("id"),
        "hullNumber": item.get("hullNumber") or item.get("hull"),
        "description": item.get("description"),
        "referenceId": item.get("referenceId") or item.get("registryReferenceId"),
    }


def _compact_match(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "trackId": item.get("trackId"),
        "keyframeId": item.get("keyframeId") or item.get("id"),
        "hullNumber": item.get("hullNumber") or item.get("finalHullNumber"),
        "score": item.get("score") or item.get("embeddingScore") or item.get("similarity"),
        "registryId": item.get("registryId") or item.get("matchedRegistryId"),
    }


def compact_tool_result_for_model(name: str, result: dict[str, Any], *, sample: int = 8) -> dict[str, Any]:
    """压缩回传给模型的工具结果，避免关键帧/轨迹大 JSON 撑爆上下文。

    完整数据仍保留在会话缓存中供后续 matchText/matchImage 自动注入。
    """
    if not isinstance(result, dict):
        return {"ok": False, "error": "tool_result_invalid", "tool": name}
    compact: dict[str, Any] = {
        "ok": result.get("ok") is not False,
        "tool": name,
    }
    if result.get("error"):
        compact["error"] = result.get("error")
    if result.get("found") is not None:
        compact["found"] = result.get("found")
    if result.get("decision") is not None:
        compact["decision"] = result.get("decision")
    if result.get("hasMore") is not None:
        compact["hasMore"] = result.get("hasMore")
    if result.get("message"):
        compact["message"] = str(result.get("message"))[:200]

    tracks = result.get("tracks")
    if isinstance(tracks, list):
        compact["trackCount"] = len(tracks)
        # 轻量轨迹列表（仅 ID/舷号/时间），供后续合成与模型选 trackIds
        compact["tracks"] = [_compact_track(t) for t in tracks if isinstance(t, dict)][:80]
        compact["trackIds"] = [
            str(t.get("trackId"))
            for t in compact["tracks"]
            if t.get("trackId") is not None
        ]
        if len(tracks) > 80:
            compact["tracksOmitted"] = len(tracks) - 80
        compact["hint"] = "轨迹字段已压缩；后续 getFrames 只传需要的 trackIds（建议≤12）"

    keyframes = result.get("keyframes")
    if isinstance(keyframes, list):
        compact["keyframeCount"] = len(keyframes)
        # 只回传样本，完整列表在会话缓存；避免 80+ 关键帧 JSON 撑爆上下文
        compact["keyframes"] = [_compact_keyframe(k) for k in keyframes[:sample] if isinstance(k, dict)]
        compact["keyframeIds"] = [
            str(k.get("keyframeId"))
            for k in compact["keyframes"]
            if k.get("keyframeId") is not None
        ]
        if len(keyframes) > sample:
            compact["keyframesOmitted"] = len(keyframes) - sample
        compact["hint"] = (
            "关键帧已缓存；matchText/matchImage/showEvidence 可省略 galleryImages/keyframeIds。"
            "不要要求回传图像路径或完整关键帧列表。"
        )

    by_track = result.get("keyframesByTrack")
    if isinstance(by_track, dict):
        compact["tracksWithFrames"] = len(by_track)
        compact["frameTrackIds"] = [str(k) for k in list(by_track.keys())[:40]]
        # 不把 keyframesByTrack 原样塞回模型

    matches = result.get("matches") or result.get("results")
    if isinstance(matches, list):
        compact["matchCount"] = len(matches)
        compact["matches"] = [_compact_match(m) for m in matches if isinstance(m, dict)][:40]
        if len(matches) > 40:
            compact["matchesOmitted"] = len(matches) - 40

    for key in (
        "trackCount", "returnedTrackCount", "totalTrackCount", "keyframeCount", "matchCount",
        "registryCount", "exactMatchHullCount", "minimumShipCount", "confirmedShipCount",
        "highThresholdShipCount", "lowThresholdShipCount", "confirmedMergeCount", "pendingMergeCount",
        "shipSegmentId", "segmentId", "offset", "limit",
    ):
        if result.get(key) is not None and key not in compact:
            compact[key] = result.get(key)

    refs = result.get("registryReferences") or result.get("registryItems")
    if isinstance(refs, list):
        compact["registryCount"] = len(refs)
        compact["registrySample"] = [_compact_registry(r) for r in refs[:sample] if isinstance(r, dict)]
        if len(refs) > sample:
            compact["registryOmitted"] = len(refs) - sample
        compact["hint"] = "先验库参考图已缓存；matchImage/matchText 可省略 gallery/query"

    hulls = result.get("matchedHullNumbers")
    if isinstance(hulls, list):
        compact["exactMatchHullCount"] = len(hulls)
        compact["matchedHullNumbers"] = hulls[:20]

    # 计数/去重等轻量结果原样保留少量字段
    for key in ("uniqueShipCount", "dedupCount", "count", "answerHint"):
        if result.get(key) is not None:
            compact[key] = result.get(key)

    # 若几乎无字段，保留短错误/原文摘要
    if len(compact) <= 3 and result.get("ok") is not False:
        compact["note"] = "工具执行成功（结果已压缩）"
    return compact


# ===========================================================================
# L4 observe 工具：业务工具 → LangChain 工具（模型直调路径）
# 两个机制：
#   会话缓存  本轮 getFrames / listRegistry / getTrack 的结果存入 session，
#             后续 matchText / matchImage / dedupTracks 未传参时自动注入
#   自动补参  _auto_fill 按工具名决定从缓存取哪一侧图像
# 注意：本函数当前无调用方 —— graph 的 observe 节点走 PlanExecutor 确定性
# 执行路径，此处是保留的备选实现。
# ===========================================================================
def build_observe_tools(
    tools_service: Any,
    on_tool: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
) -> list[StructuredTool]:
    """业务检索工具：绑定 ToolService.execute。

    兼容旧版 $ref 行为：本轮 getFrames / listRegistry / getRegistry 的结果会缓存，
    matchText/matchImage 未传 galleryImages/queryImages 时自动注入。
    回传给模型的是压缩摘要，避免上下文超限。
    """
    # 本轮观察会话缓存（单次 ObserveAgent 内共享）
    session: dict[str, Any] = {
        "keyframes": None,
        "registryReferences": None,
        "tracks": None,
        "keyframesByTrack": None,
    }

    def _remember(name: str, result: dict[str, Any]) -> None:
        if not isinstance(result, dict) or result.get("ok") is False:
            return
        if name == "getFrames":
            if result.get("keyframes") is not None:
                session["keyframes"] = result.get("keyframes")
            if result.get("keyframesByTrack") is not None:
                session["keyframesByTrack"] = result.get("keyframesByTrack")
        elif name in {"listRegistry", "getRegistry"}:
            if result.get("registryReferences") is not None:
                session["registryReferences"] = result.get("registryReferences")
            elif result.get("registryItems") is not None:
                session["registryReferences"] = result.get("registryItems")
        elif name == "getTrack":
            if result.get("tracks") is not None:
                session["tracks"] = result.get("tracks")

    # 自动补参：模型省略图像参数时，从本轮缓存挑合适的来源填上。
    # matchImage 的四个分支覆盖两个方向（轨迹查库 / 库图查轨迹）。
    def _auto_fill(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        if name == "matchText" and args.get("galleryImages") is None:
            # 描述匹配轨迹关键帧优先；否则用先验库参考图
            if session.get("keyframes") is not None:
                args["galleryImages"] = session["keyframes"]
            elif session.get("registryReferences") is not None:
                args["galleryImages"] = session["registryReferences"]
        if name == "matchImage":
            if args.get("queryImages") is None and session.get("keyframes") is not None:
                args["queryImages"] = session["keyframes"]
            if args.get("galleryImages") is None and session.get("registryReferences") is not None:
                args["galleryImages"] = session["registryReferences"]
            # 反过来：查询库图、匹配轨迹
            if args.get("queryImages") is None and session.get("registryReferences") is not None:
                args["queryImages"] = session["registryReferences"]
            if args.get("galleryImages") is None and session.get("keyframes") is not None:
                args["galleryImages"] = session["keyframes"]
        if name == "dedupTracks":
            if args.get("tracks") is None and session.get("tracks") is not None:
                args["tracks"] = session["tracks"]
            if args.get("keyframesByTrack") is None and session.get("keyframesByTrack") is not None:
                args["keyframesByTrack"] = session["keyframesByTrack"]
        if name == "showEvidence":
            # 未指定证据 ID 时，从最近关键帧取一批展示
            if not args.get("keyframeIds") and isinstance(session.get("keyframes"), list):
                ids = [
                    str(item.get("keyframeId"))
                    for item in session["keyframes"][:12]
                    if isinstance(item, dict) and item.get("keyframeId") is not None
                ]
                if ids:
                    args["keyframeIds"] = ids
        return args

    # 通用包装器：参数整形（timeRange 转 tuple、getFrames 截断到 12 条）→ 自动
    # 补参 → 执行 → 写回会话缓存 → 派发事件（事件里不带大图像对象）→ 压缩返回。
    def _wrap(name: str, schema: type[BaseModel], description: str) -> StructuredTool:
        def _run(**kwargs: Any) -> str:
            arguments = {k: v for k, v in kwargs.items() if v is not None}
            # timeRange list -> tuple for ToolService
            if "timeRange" in arguments and isinstance(arguments["timeRange"], list):
                tr = arguments["timeRange"]
                if len(tr) == 2:
                    arguments["timeRange"] = (float(tr[0]), float(tr[1]))
            # getFrames 限制轨迹数量，避免一次取过多关键帧
            if name == "getFrames" and isinstance(arguments.get("trackIds"), list):
                track_ids = arguments["trackIds"]
                if len(track_ids) > 12:
                    arguments["trackIds"] = track_ids[:12]
                    arguments["_trackIdsTruncated"] = len(track_ids) - 12
            arguments = _auto_fill(name, arguments)
            result = tools_service.execute(name, arguments)
            if not isinstance(result, dict):
                result = {"ok": False, "error": "tool_result_invalid", "tool": name}
            if arguments.get("_trackIdsTruncated") and isinstance(result, dict):
                result = dict(result)
                result["trackIdsTruncated"] = arguments["_trackIdsTruncated"]
                result["message"] = (
                    f"已限制本轮最多 12 条轨迹取帧，另有 {arguments['_trackIdsTruncated']} 条未取；"
                    "如需更多请分页再调 getFrames"
                )
            _remember(name, result)
            if on_tool:
                try:
                    # 事件里不塞完整关键帧大对象，只报模型实际传入的关键参数
                    event_args = {
                        k: v for k, v in arguments.items()
                        if k not in {"galleryImages", "queryImages", "keyframesByTrack", "_trackIdsTruncated"}
                        or not isinstance(v, (list, dict))
                    }
                    if "galleryImages" in arguments and "galleryImages" not in event_args:
                        event_args["galleryImages"] = f"<auto:{len(arguments.get('galleryImages') or []) if isinstance(arguments.get('galleryImages'), list) else 'obj'}>"
                    if "queryImages" in arguments and "queryImages" not in event_args:
                        event_args["queryImages"] = f"<auto:{len(arguments.get('queryImages') or []) if isinstance(arguments.get('queryImages'), list) else 'obj'}>"
                    on_tool(name, event_args, result)
                except Exception:
                    pass
            return _dump(compact_tool_result_for_model(name, result))

        return StructuredTool.from_function(
            name=name,
            description=description,
            func=_run,
            args_schema=schema,
        )

    def _list_registry(dummy: str | None = None) -> str:
        result = tools_service.execute("listRegistry", {})
        if not isinstance(result, dict):
            result = {"ok": False, "error": "tool_result_invalid", "tool": "listRegistry"}
        _remember("listRegistry", result)
        if on_tool:
            try:
                on_tool("listRegistry", {}, result)
            except Exception:
                pass
        return _dump(compact_tool_result_for_model("listRegistry", result))

    return [
        _wrap("getTrack", GetTrackArgs, "按时间/舷号分页检索轨迹"),
        _wrap("getFrames", GetFramesArgs, "按轨迹取关键帧；结果会自动供给后续 matchText"),
        _wrap("getClip", GetClipArgs, "生成轨迹片段"),
        _wrap("getRegistry", GetRegistryArgs, "按舷号查先验库"),
        StructuredTool.from_function(
            name="listRegistry",
            description="列出先验库全部条目；结果可自动供给 matchText/matchImage",
            func=_list_registry,
            args_schema=ListRegistryArgs,
        ),
        _wrap("matchHull", MatchHullArgs, "舷号精确匹配先验库"),
        _wrap(
            "matchText",
            MatchTextArgs,
            "文本描述匹配关键帧或库图。galleryImages 可省略：默认用本轮 getFrames 的 keyframes",
        ),
        _wrap(
            "matchImage",
            MatchImageArgs,
            "图像相似度匹配。queryImages/galleryImages 可省略，自动用本轮关键帧与库参考图",
        ),
        _wrap("verifyTarget", VerifyTargetArgs, "VLM 核验目标"),
        _wrap("showEvidence", ShowEvidenceArgs, "汇总展示证据 ID"),
        _wrap("dedupTracks", DedupTracksArgs, "跨轨迹去重计数；tracks/keyframes 可省略用本轮缓存"),
    ]


# ===========================================================================
# L5 技能加载工具
# 模型在 ReAct 过程中按需拉取「只给了目录、未注入正文」的技能全文，
# 与 skill_loader 的两级披露配套。load_fn 由 graph 注入（_skill_loader）。
# ===========================================================================
def build_load_skill_tool(agent_key: str, load_fn: Callable[[str], dict[str, Any]]) -> StructuredTool:
    def _run(skillId: str) -> str:
        return _dump(load_fn(skillId))

    return StructuredTool.from_function(
        name="loadSkill",
        description="按需加载可选 skill 全文",
        func=_run,
        args_schema=LoadSkillArgs,
    )

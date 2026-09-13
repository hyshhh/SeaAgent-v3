"""LangGraph 四 Agent 控制器：对外保持 answer() 与事件契约。

本文件是 agent 子系统的公开边界（agent/__init__.py 只导出 AgentController）：
向内的编排在 graph.py 的 AgentState 状态机里，向外的 HTTP/NDJSON 协议在
web/routes/agent_api.py；这里只做「终态 state → 三份投影」的翻译与落库。

全部代码按职责分为七层：

    L0 装配层   依赖注入 + 会话状态槽                __init__ / _emit
    L1 生命周期 answer() 一次问答的七步主流程
    L2 审计层   落库 qa_rounds/qa_evidence + 决策计量
    L3 合成层   按事实分派，产出 conclusion/answerText（本文件最大的一块）
    L4 投影层   working_scope → 结构化事实（tracks/matches/registry/dedup/count）
    L5 组装层   拼装对外的 result dict
    L6 展示层   生成前端证据分组（缺帧时会现场调工具补齐）
    L7 聚合层   证据 ID 摊平 + 落库用瘦身摘要

阅读提示：物理顺序与调用顺序不一致——L3 的 _synthesize 调用了写在它之后的 L4。
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from config import load_config
from memory import MemoryRepository
from services import AgentLLMService, QwenMultimodalEmbedder
from tools import ToolService
from vector_store import VectorCatalog

from .graph import run_sea_agent
from .task_profiles import registry_membership_list_mode


def _nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


# ============================================================================
# L0 装配层：依赖注入与会话状态槽
# 5 个依赖全部可注入（便于测试替换）；下面的会话级状态槽是「图的终态在本类中
# 的投影缓存」，answer() 每次入口先清空它们，_finish 与 _display_* 再填充。
# ============================================================================
class AgentController:
    """前端入口：内部运行 LangGraph（Intent→Plan→Observe→Reflect）。"""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        repository: MemoryRepository | None = None,
        tools: ToolService | None = None,
        llm: AgentLLMService | None = None,
        embedder: QwenMultimodalEmbedder | None = None,
        vectors: VectorCatalog | None = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.tools = tools or ToolService(self.config, self.repository, embedder, self.llm, vectors)
        settings = self.config.get("pipeline", {}).get("agent", {})
        retrieval_settings = self.config.get("pipeline", {}).get("retrieval", {})
        self.max_rounds = int(settings.get("max_rounds", 3))
        self.display_limit = int(settings.get("display_limit", 3))
        self.broad_match_top_k = _nonnegative_int(retrieval_settings.get("broad_match_top_k", 0))
        self.event_handler = event_handler
        self.session_id = ""
        self.question = ""
        self.meta: dict[str, Any] = {}
        self.rounds: list[dict[str, Any]] = []
        self.tool_chain: list[str] = []
        self.tool_records: list[dict[str, Any]] = []
        self.display_record: dict[str, Any] | None = None
        self.display_groups: list[dict[str, Any]] = []
        self.working_scope: dict[str, Any] = {}
        self._pending_registry_items: list[dict[str, Any]] = []
        # 模型决策被确定性守卫纠偏/兜底的计量（③），落库到会话审计与最终结果
        self.decision_metrics: dict[str, Any] = {}
        self.memory_persist_error: str = ""

    def _emit(self, event_type: str, title: str, message: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        try:
            self.event_handler({"type": event_type, "title": title, "message": message, **payload})
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # L1 生命周期层：answer() —— 一次问答的七步主流程
    # ① 重置会话状态与参数   ② 调用 run_sea_agent（唯一一次图调用）
    # ③ 异常兜底降级         ④ 把终态回填到实例属性
    # ⑤ 落库 qa_*            ⑥ 计算决策计量    ⑦ 合成结论并结束会话
    # ------------------------------------------------------------------------
    def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self.question = str(question or "").strip()
        self.rounds, self.tool_chain, self.tool_records = [], [], []
        self.display_record, self.display_groups = None, []
        self.working_scope = {}
        self._pending_registry_items: list[dict[str, Any]] = []
        agent_settings = self.config.get("pipeline", {}).get("agent", {})
        self.max_rounds = int(agent_settings.get("max_rounds", self.max_rounds))
        self.display_limit = int(agent_settings.get("display_limit", self.display_limit))
        retrieval_settings = self.config.get("pipeline", {}).get("retrieval", {})
        default_top_k = int(retrieval_settings.get("top_k", 3))
        self.query_top_k = max(1, min(20, int(top_k if top_k is not None else default_top_k)))
        self.broad_match_top_k = _nonnegative_int(retrieval_settings.get("broad_match_top_k", 0))

        self._emit("status", "Controller", "LangGraph 四 Agent 协同启动", planMode="langgraph")
        try:
            state = run_sea_agent(
                self.question,
                self.llm,
                self.tools,
                max_rounds=self.max_rounds,
                query_top_k=self.query_top_k,
                broad_match_top_k=self.broad_match_top_k,
                event_handler=self.event_handler,
            )
        except Exception as error:
            result = self._finish(
                "执行失败",
                [],
                f"LangGraph 执行失败：{error}",
                "uncertain",
                extra={
                    "error": str(error),
                    "planMode": "langgraph",
                    "retrievalTopK": self.query_top_k,
                    "retrievalBroadMatchTopK": self.broad_match_top_k,
                },
            )
            result["decisionMetrics"] = self.decision_metrics
            try:
                self.repository.finish_session(self.session_id, self._session_audit_result(result))
            except Exception:
                pass
            return result

        self.meta = dict(state.get("intent") or {})
        self.meta["planMode"] = "langgraph"
        self.meta["maxRounds"] = self.max_rounds
        self.meta["retrievalTopK"] = self.query_top_k
        self.meta["retrievalBroadMatchTopK"] = self.broad_match_top_k
        self.working_scope = dict(state.get("working_scope") or {})
        self.rounds = list(state.get("rounds") or [])
        self.tool_chain = list(state.get("tool_chain") or [])
        self.tool_records = list(state.get("tool_records") or [])

        try:
            self.repository.add_session(self.session_id, {"question": self.question, **self.meta})
        except Exception:
            pass
        # ① 落库 LangGraph 轮次与工具证据（qa_rounds/qa_evidence），使问答记忆可审计可回放
        self._persist_qa_memory(state)
        # ③ 计量模型决策 vs 确定性守卫/兜底的占比，随审计与最终结果一起落库
        self.decision_metrics = self._build_decision_metrics(state)

        final_state = str(state.get("final_state") or "uncertain")
        final_reason = str(state.get("final_reason") or "协同结束")
        result = self._synthesize(final_state, final_reason)
        result["decisionMetrics"] = self.decision_metrics
        try:
            self.repository.finish_session(self.session_id, self._session_audit_result(result))
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------------
    # L2 审计层：落库与计量
    # _persist_qa_memory 把图内轮次写成 qa_rounds / qa_evidence（只写不读，
    # 不参与跨请求多轮）；_build_decision_metrics 统计「模型决策 vs 确定性
    # 守卫」的占比。两者失败都不打断主流程，只记录 self.memory_persist_error。
    # ------------------------------------------------------------------------
    def _persist_qa_memory(self, state: dict[str, Any]) -> None:
        """① 把 LangGraph 的轮次与工具证据落库到 qa_rounds / qa_evidence。

        设计：round 按 session 内的轮次编号唯一；evidence 以「会话-轮次-调用id」唯一，
        跨轮次重名 call id（如每轮都有 tracks/frames）不会互相覆盖。落库失败不打断
        answer 主流程，但把错误记录到 self.memory_persist_error 供审计观测。
        """
        self.memory_persist_error = ""
        try:
            session_id = self.session_id
            round_by_number: dict[int, str] = {}
            for item in state.get("rounds") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    round_number = int(item.get("round") or 0)
                except (TypeError, ValueError):
                    continue
                round_id = f"{session_id}-r{round_number}"
                plan = {
                    "planHint": str(item.get("planHint") or ""),
                    "observation": str(item.get("observation") or ""),
                    "toolChain": list(item.get("toolChain") or []),
                    "planRepair": str(item.get("planRepair") or ""),
                    "planUsedDefault": bool(item.get("planUsedDefault")),
                }
                reflection = item.get("reflection") if isinstance(item.get("reflection"), dict) else {}
                self.repository.add_round(round_id, session_id, plan, reflection)
                round_by_number[round_number] = round_id

            seen_evidence: set[str] = set()
            for record in state.get("tool_records") or []:
                if not isinstance(record, dict):
                    continue
                try:
                    round_number = int(record.get("round") or 0)
                except (TypeError, ValueError):
                    round_number = 0
                call_id = str(record.get("id") or record.get("tool") or "tool")
                evidence_id = f"{session_id}-r{round_number}-{call_id}"
                if evidence_id in seen_evidence:
                    evidence_id = f"{evidence_id}-{len(seen_evidence) + 1}"
                seen_evidence.add(evidence_id)
                round_id = round_by_number.get(round_number) or f"{session_id}-r{round_number}"
                self.repository.add_evidence(
                    evidence_id,
                    round_id,
                    self._evidence_tool_result(record),
                    {
                        "tool": record.get("tool"),
                        "round": round_number,
                        "source": "langgraph",
                        "planMode": "langgraph",
                    },
                )
        except Exception as error:  # 记忆落库失败不应让回答本身失败
            self.memory_persist_error = str(error)

    @staticmethod
    def _evidence_tool_result(record: dict[str, Any]) -> dict[str, Any]:
        """裁剪工具记录：丢弃关键帧/轨迹/匹配等大列表，只保留计数、结论与错误字段。"""
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        compact: dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
                continue
            compact[key] = value
        return {
            "tool": record.get("tool"),
            "arguments": record.get("arguments") or {},
            "ok": record.get("ok") is not False,
            "skipped": bool(record.get("skipped")),
            "error": record.get("error"),
            "summary": record.get("summary") or {},
            "resultSummary": compact,
        }

    @staticmethod
    def _build_decision_metrics(state: dict[str, Any]) -> dict[str, Any]:
        """③ 统计模型决策 vs 确定性守卫/兜底的占比，供审计与前端展示。"""
        rounds = state.get("rounds") or []
        sources: dict[str, int] = {}
        replan_count = 0
        finish_count = 0
        plan_fallback_count = 0
        plan_repair_count = 0
        for item in rounds:
            if not isinstance(item, dict):
                continue
            reflection = item.get("reflection") if isinstance(item.get("reflection"), dict) else {}
            source = str(reflection.get("decisionSource") or "unknown")
            sources[source] = sources.get(source, 0) + 1
            if str(reflection.get("handoff") or "") == "plan" or reflection.get("replan"):
                replan_count += 1
            else:
                finish_count += 1
            if item.get("planUsedDefault"):
                plan_fallback_count += 1
            if item.get("planRepair"):
                plan_repair_count += 1
        records = state.get("tool_records") or []
        failed_count = sum(
            1
            for record in records
            if isinstance(record, dict) and record.get("ok") is False and not record.get("skipped")
        )
        skipped_count = sum(1 for record in records if isinstance(record, dict) and record.get("skipped"))
        guard_count = sum(
            count for source, count in sources.items() if source not in {"model", "unknown"}
        )
        return {
            "roundCount": len(rounds),
            "decisionSourceCounts": sources,
            "modelDecisionCount": sources.get("model", 0),
            "guardDecisionCount": guard_count,
            "replanCount": replan_count,
            "finishCount": finish_count,
            "planFallbackCount": plan_fallback_count,
            "planRepairCount": plan_repair_count,
            "toolFailedCount": failed_count,
            "toolSkippedCount": skipped_count,
            "finalDecisionSource": str((state.get("reflection") or {}).get("decisionSource") or "unknown"),
            "finalState": str(state.get("final_state") or "unknown"),
        }

    # ------------------------------------------------------------------------
    # L3 合成层：按事实分派，产出 conclusion / answerText
    # 全文最大的一块，且不是文本生成器，而是一张按「已采集事实」落座的决策表。
    # 分支树：
    #   A 计数       count_value 非空 → 分「高低阈值一致 / 存在待确认合并组」
    #   B 零候选     in|out 列表且全量轨迹为 0 → 提前给否定结论
    #   C 伪描述     描述含「哪些/在库/先验库」等词 → 命中不可信
    #   D 有匹配     registry / in 列表 / out 列表 / 通用 / 全 mismatch 五种
    #   E out 无匹配 库为空 或 图像匹配不可用
    #   F 纯库结果   有库项、无轨迹（含描述查询的明确否定）
    #   G 有轨迹     存在判断 + 舷号防误判
    #   H 兜底       0 轨迹/0 匹配 → conflict / 明确否定 / 未找到可靠证据
    # 注意：本方法会重算终态（各分支传给 _finish 的 state 参数），并不完全沿用
    #       graph.reflect_node 给出的 final_state。
    # ------------------------------------------------------------------------
    def _synthesize(self, state: str, reason: str) -> dict[str, Any]:
        tracks = self._collect_tracks()
        matches = self._collect_matches()
        registry_items = self._collect_registry()
        dedup_summary = self._collect_dedup_summary()
        count_value = self._collect_count()
        answer_hint = reason
        operation = str(self.meta.get("operation") or "")
        target_kind = str(self.meta.get("targetKind") or "")
        target_scope = str(self.meta.get("targetScope") or "")
        description = str(self.meta.get("description") or "").strip()
        hull = str(self.meta.get("hullNumber") or "").strip()

        # ---- 分支 A：计数类问题 --------------------------------------------
        # 有 dedupSummary 时分两种措辞：高低阈值一致→「N 艘」；
        # 存在待确认合并组→「至少 N 艘」并附 _build_count_evidence 审计台账。
        if count_value is not None:
            if dedup_summary:
                minimum_count = int(dedup_summary.get("minimumShipCount", count_value))
                confirmed_count = int(dedup_summary.get("confirmedShipCount", minimum_count))
                track_count = int(dedup_summary.get("trackCount", len(tracks)))
                pending_groups = dedup_summary.get("pendingMergeGroups") or []
                confirmed_groups = dedup_summary.get("confirmedMergeGroups") or []
                pending_reduction = max(0, confirmed_count - minimum_count)
                if confirmed_count == minimum_count:
                    conclusion = f"统计结果：{minimum_count} 艘船"
                    answer_text = (
                        f"共获取 {track_count} 条轨迹；高、低阈值去重结果一致，"
                        f"确认对应 {minimum_count} 艘船。"
                    )
                else:
                    conclusion = f"统计结果：至少 {minimum_count} 艘船"
                    answer_text = (
                        f"共获取 {track_count} 条轨迹。按高阈值确认合并后为 {confirmed_count} 艘；"
                        f"另有 {len(pending_groups)} 组轨迹待确认合并，最多可再减少 {pending_reduction} 艘，"
                        f"若这些合并关系全部成立，最少为 {minimum_count} 艘。"
                    )
                missing = dedup_summary.get("unsearchableTrackIds") or []
                if missing:
                    answer_text += f"另有 {len(missing)} 条轨迹缺少可检索关键帧，需结合原视频复核。"
                public_dedup = {
                    key: dedup_summary.get(key)
                    for key in (
                        "trackCount", "minimumShipCount", "confirmedShipCount", "maximumShipCount",
                        "confirmedMergeCount", "pendingMergeCount", "confirmedReduction",
                        "pendingReduction", "countStability", "highThreshold", "lowThreshold",
                        "confirmedMergeGroups", "pendingMergeGroups", "unsearchableTrackIds",
                    )
                    if dedup_summary.get(key) is not None
                }
                count_evidence = self._build_count_evidence(dedup_summary, tracks)
                return self._finish(
                    conclusion,
                    tracks,
                    answer_text,
                    state,
                    extra={
                        "count": minimum_count,
                        "minimumCount": minimum_count,
                        "confirmedCount": confirmed_count,
                        "countRange": {"minimum": minimum_count, "confirmed": confirmed_count},
                        "confirmedMergeGroups": confirmed_groups,
                        "pendingMergeGroups": pending_groups,
                        "dedupSummary": public_dedup,
                        "countEvidence": count_evidence,
                        "planMode": "langgraph",
                    },
                    display={"dedupSummary": public_dedup, "tracks": tracks},
                )
            return self._finish(
                f"统计结果为 {count_value}",
                tracks,
                answer_hint,
                state,
                extra={"count": count_value, "planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        # 在库/未在库列表判定收敛到 task_profiles 单一事实源（与 graph 同源）
        membership_mode = registry_membership_list_mode(self.meta)
        is_registry_in_list = membership_mode == "in"
        is_registry_out_list = membership_mode == "out"
        registry_listed = any(
            isinstance(record, dict) and record.get("tool") == "listRegistry" and record.get("ok") is not False
            for record in self.tool_records
        )
        match_text_completed = any(
            isinstance(record, dict)
            and record.get("tool") == "matchText"
            and record.get("ok") is not False
            and not record.get("skipped")
            for record in self.tool_records
        )
        successful_track_records = [
            record for record in self.tool_records
            if isinstance(record, dict)
            and record.get("tool") == "getTrack"
            and record.get("ok") is not False
            and not record.get("skipped")
        ]
        successful_track_counts: list[int] = []
        for record in successful_track_records:
            result_tracks = (record.get("result") or {}).get("tracks")
            raw_count = (
                len(result_tracks)
                if isinstance(result_tracks, list)
                else (record.get("summary") or {}).get("trackCount")
                if (record.get("summary") or {}).get("trackCount") is not None
                else record.get("trackCount")
            )
            if raw_count is None:
                continue
            try:
                successful_track_counts.append(max(0, int(raw_count)))
            except (TypeError, ValueError):
                continue
        track_query_completed = bool(successful_track_counts)
        latest_track_count = successful_track_counts[-1] if successful_track_counts else None
        match_image_attempted = any(
            isinstance(record, dict) and record.get("tool") == "matchImage" and not record.get("skipped")
            for record in self.tool_records
        )
        match_image_blocked = any(
            isinstance(record, dict)
            and record.get("tool") == "matchImage"
            and not record.get("skipped")
            and bool((record.get("result") or {}).get("error") or record.get("error"))
            for record in self.tool_records
        )

        # 在库/未在库列表的首要事实是视频侧候选数。全量轨迹明确为 0 时直接给否定结论，
        # 禁止把整份先验库展示成“未在库船”，也禁止被空 gallery 的 matchImage 伪失败改成无法确认。
        # ---- 分支 B：视频侧零候选，提前否定 --------------------------------
        # 全量轨迹为 0 是首要事实，必须直接给否定结论；不得把整份先验库当成
        # 「未在库船」展示，也不得被空 gallery 的 matchImage 伪失败改成无法确认。
        if (
            (is_registry_in_list or is_registry_out_list)
            and track_query_completed
            and latest_track_count == 0
            and not tracks
        ):
            relation_label = "在库" if is_registry_in_list else "未在库"
            extra = {
                "planMode": "langgraph",
                "targetScope": "both",
                "found": False,
                "matchCount": 0,
            }
            if is_registry_out_list:
                extra.update({
                    "outOfRegistryTracks": [],
                    "outOfRegistryCount": 0,
                    "uncertainTracks": [],
                    "unscoredTracks": [],
                })
            else:
                extra["inRegistryMatchCount"] = 0
            return self._finish(
                f"当前时间范围内未检测到船舶轨迹，因此没有{relation_label}船舶出现",
                [],
                answer_hint or "全量视频轨迹为 0，已按视频侧零候选规则结束",
                "sufficient",
                extra=extra,
                display={"tracks": [], "includeClips": False, "includeRegistry": False},
            )

        # 伪描述：用户整句当 matchText 时，命中不可信
        # ---- 分支 C：伪描述检测 --------------------------------------------
        # 用户把整句问句当 matchText 传给工具时，库侧命中不可信，后续需降级。
        bogus_description = bool(
            description
            and any(token in description for token in ("哪些", "有哪些", "在库", "未在库", "先验库", "库船"))
        )

        # ---- 分支 D：存在匹配结果 ------------------------------------------
        # D1 out 列表(448) / D2 纯库匹配(558) / D3 in 列表(606) / D4 通用命中(639)
        # / D5 全 mismatch 仍展示 top 候选(696，禁止证据区空白)
        if matches:
            # mismatch 丢弃；uncertain 仅作灰区候选，match 才算确认命中
            ranked = sorted(
                [m for m in matches if isinstance(m, dict)],
                key=lambda m: float(m.get("embeddingScore") or m.get("score") or 0),
                reverse=True,
            )
            confirmed = [m for m in ranked if str(m.get("scoreBand") or "") == "match"]
            uncertain = [m for m in ranked if str(m.get("scoreBand") or "") == "uncertain"]
            supported = confirmed + uncertain  # 同时保留确认与灰区，分栏展示

            if is_registry_out_list:
                # 每条轨迹的“最高库匹配分”越低，越可能未在库；所有类别统一按低分优先。
                ascending = sorted(ranked, key=lambda m: float(m.get("embeddingScore") or m.get("score") or 0))
                mismatch_ascending = [m for m in ascending if str(m.get("scoreBand") or "") == "mismatch"]
                uncertain_ascending = [m for m in ascending if str(m.get("scoreBand") or "") == "uncertain"]
                out_tracks = self._tracks_ranked_by_matches(mismatch_ascending, tracks)
                uncertain_tracks = self._tracks_ranked_by_matches(uncertain_ascending, tracks)
                score_order = {
                    str(m.get("matchedTrackId") or m.get("trackId")): idx
                    for idx, m in enumerate(ascending)
                    if m.get("matchedTrackId") is not None or m.get("trackId") is not None
                }
                out_tracks.sort(key=lambda item: score_order.get(str(item.get("trackId")), 10**9))
                uncertain_tracks.sort(key=lambda item: score_order.get(str(item.get("trackId")), 10**9))
                scored_ids = {
                    str(item.get("matchedTrackId") or item.get("trackId"))
                    for item in ascending
                    if item.get("matchedTrackId") is not None or item.get("trackId") is not None
                }
                unscored_tracks = [item for item in tracks if str(item.get("trackId")) not in scored_ids]
                coverage = self._collect_image_match_summary()
                coverage_complete = bool(coverage.get("registryCoverageComplete"))
                if not coverage_complete:
                    # 旧工具结果没有覆盖字段时保持兼容；新结果明确 False 时，低分轨迹只能降级为灰区。
                    has_coverage_field = "registryCoverageComplete" in coverage
                    if has_coverage_field and out_tracks:
                        uncertain_tracks = [
                            {**item, "coverageLimited": True}
                            for item in out_tracks
                        ] + uncertain_tracks
                        out_tracks = []

                # scoreBand 描述图像分数区间；registryOutState 描述“未在库”查询下的业务结论。
                # mismatch 且完成全库比较才是确认未在库，uncertain 为灰区，未评分目标单独保留。
                out_tracks = [
                    {**item, "registryOutState": "confirmed_out"}
                    for item in out_tracks
                ]
                uncertain_tracks = [
                    {**item, "registryOutState": "gray_zone"}
                    for item in uncertain_tracks
                ]
                unscored_tracks = [
                    {**item, "registryOutState": "unscored"}
                    for item in unscored_tracks
                ]

                # 必须在覆盖降级和状态归一化之后重建展示轨迹，确保结果区与证据区完全一致。
                display_tracks = []
                seen_display: set[str] = set()
                for item in out_tracks + uncertain_tracks + unscored_tracks:
                    key = str(item.get("trackId"))
                    if key in seen_display:
                        continue
                    seen_display.add(key)
                    display_tracks.append(item)

                if out_tracks:
                    conclusion = f"确认发现 {len(out_tracks)} 条未在库轨迹"
                    if uncertain_tracks or unscored_tracks:
                        conclusion += f"，另有 {len(uncertain_tracks) + len(unscored_tracks)} 条待确认"
                    finish_state = "uncertain" if (uncertain_tracks or unscored_tracks) else "sufficient"
                elif uncertain_tracks or unscored_tracks:
                    conclusion = "尚未确认未在库轨迹，存在灰区、库覆盖不足或不可评分目标"
                    finish_state = "uncertain"
                else:
                    conclusion = "未发现未在库轨迹"
                    finish_state = "sufficient" if coverage_complete or not coverage else "uncertain"
                return self._finish(
                    conclusion,
                    out_tracks,
                    answer_hint or "已按每条轨迹的最高库匹配分从低到高完成对照",
                    finish_state,
                    extra={
                        "matches": ascending,
                        "outOfRegistryTracks": out_tracks,
                        "uncertainTracks": uncertain_tracks,
                        "unscoredTracks": unscored_tracks,
                        "outOfRegistryCount": len(out_tracks),
                        "inRegistryMatchCount": len(confirmed),
                        "uncertainMatchCount": len(uncertain_tracks),
                        "unscoredTrackCount": len(unscored_tracks),
                        "matchCount": len(display_tracks),
                        "classificationKind": "track",
                        "classificationMode": "registry_out",
                        "rankingBasis": "每条轨迹对全部先验库项的最高匹配分，按分数从低到高排序",
                        "registryCoverageComplete": coverage.get("registryCoverageComplete"),
                        "registryCoverageRatio": coverage.get("registryCoverageRatio"),
                        "scoredRegistryCount": coverage.get("scoredRegistryCount"),
                        "totalRegistryCount": coverage.get("totalRegistryCount"),
                        "unscoredRegistryIds": coverage.get("unscoredRegistryIds") or [],
                        "planMode": "langgraph",
                        "targetScope": "both",
                    },
                    display={"tracks": display_tracks, "includeClips": bool(display_tracks), "includeRegistry": False},
                )
            # 视频侧结果按轨迹分组：确认轨迹与灰区轨迹必须同时保留，且不受展示 top-k 截断。
            confirmed_tracks = self._tracks_ranked_by_matches(confirmed, tracks)
            confirmed_track_ids = {str(item.get("trackId")) for item in confirmed_tracks if item.get("trackId") is not None}
            uncertain_tracks = [
                item for item in self._tracks_ranked_by_matches(uncertain, tracks)
                if str(item.get("trackId")) not in confirmed_track_ids
            ]
            display_tracks = confirmed_tracks + uncertain_tracks
            hit_tracks = display_tracks
            match_thresholds = self._collect_match_thresholds()
            threshold_note = self._match_threshold_note(match_thresholds)

            if supported and not (is_registry_in_list and bogus_description):
                # 纯数据库描述匹配：确认库项与灰区库项同样分开返回。
                if target_scope == "registry" and not hit_tracks:
                    confirmed_items = self._registry_items_from_matches(confirmed)
                    uncertain_items = self._registry_items_from_matches(uncertain)
                    confirmed_registry_ids = {
                        str(item.get("registryId")) for item in confirmed_items if item.get("registryId") is not None
                    }
                    uncertain_items = [
                        item for item in uncertain_items
                        if str(item.get("registryId")) not in confirmed_registry_ids
                    ]
                    matched_items = confirmed_items + uncertain_items
                    matched_count = len(matched_items)
                    label = description or "目标"
                    if confirmed_items:
                        conclusion = (
                            f"数据库中确认存在「{label}」"
                            if operation == "existence"
                            else f"数据库中找到 {matched_count} 个匹配库项"
                        )
                        finish_state = "conflict" if state == "conflict" else "sufficient"
                        base_hint = answer_hint or f"数据库描述匹配得到 {len(confirmed_items)} 个确认库项"
                    else:
                        conclusion = f"数据库中发现 {matched_count} 个疑似「{label}」库项，尚未达到确认阈值"
                        finish_state = "uncertain"
                        base_hint = answer_hint or f"数据库描述匹配仅得到 {len(uncertain_items)} 个灰区库项"
                    hint = f"{base_hint}；{threshold_note}" if threshold_note else base_hint
                    return self._finish(
                        conclusion,
                        [],
                        hint,
                        finish_state,
                        extra={
                            "matches": ranked,
                            "registryItems": matched_items,
                            "confirmedRegistryItems": confirmed_items,
                            "uncertainRegistryItems": uncertain_items,
                            "classificationKind": "registry",
                            "matchThresholds": match_thresholds,
                            "planMode": "langgraph",
                            "targetScope": "registry",
                            "matchCount": matched_count,
                            "confirmedMatchCount": len(confirmed_items),
                            "uncertainMatchCount": len(uncertain_items),
                            "found": bool(confirmed_items),
                        },
                        display={"tracks": [], "includeClips": False, "includeRegistry": bool(matched_items)},
                    )
                # 在库列表 + matchImage 命中：左侧列确认轨迹，右侧列全部灰区轨迹。
                if is_registry_in_list:
                    hit_items = self._registry_items_from_matches(supported)
                    if confirmed_tracks:
                        label = f"视频中确认 {len(confirmed_tracks)} 条在库匹配轨迹"
                    else:
                        label = f"视频中发现 {len(uncertain_tracks)} 条灰区轨迹，尚未达到确认阈值"
                    base_hint = answer_hint or "已完成先验库与视频轨迹的图像对照"
                    hint = f"{base_hint}；{threshold_note}" if threshold_note else base_hint
                    return self._finish(
                        label,
                        confirmed_tracks,
                        hint,
                        state if state in {"sufficient", "uncertain", "conflict"} else "sufficient",
                        extra={
                            "matches": ranked,
                            "registryItems": hit_items or registry_items,
                            "confirmedTracks": confirmed_tracks,
                            "uncertainTracks": uncertain_tracks,
                            "classificationKind": "track",
                            "matchThresholds": match_thresholds,
                            "planMode": "langgraph",
                            "targetScope": "both",
                            "matchCount": len(display_tracks),
                            "confirmedMatchCount": len(confirmed_tracks),
                            "uncertainMatchCount": len(uncertain_tracks),
                            "found": bool(confirmed_tracks),
                        },
                        display={
                            "tracks": display_tracks,
                            "includeClips": True,
                            "includeRegistry": True,
                        },
                    )
                target_label = f"舷号 {hull}" if hull else (f"「{description}」" if description else "目标")
                if confirmed_tracks:
                    if operation == "existence":
                        conclusion = f"确认舷号 {hull} 在视频中出现" if hull else f"确认{target_label}在视频中出现"
                    else:
                        conclusion = "找到匹配目标"
                    finish_state = state if state in {"sufficient", "uncertain", "conflict"} else "sufficient"
                    base_hint = f"共确认 {len(confirmed_tracks)} 条匹配轨迹"
                    if uncertain_tracks:
                        base_hint += f"，另有 {len(uncertain_tracks)} 条灰区轨迹待复核"
                else:
                    if operation == "existence":
                        conclusion = (
                            f"尚未确认舷号 {hull} 在视频中出现，仅发现灰区轨迹"
                            if hull
                            else f"尚未确认{target_label}在视频中出现，仅发现灰区轨迹"
                        )
                    else:
                        conclusion = "仅有灰区匹配，未达确认阈值"
                    finish_state = "uncertain"
                    base_hint = f"共发现 {len(uncertain_tracks)} 条灰区轨迹，未达到确认阈值"
                if answer_hint:
                    base_hint = f"{base_hint}；{answer_hint}"
                hint = f"{base_hint}；{threshold_note}" if threshold_note else base_hint
                # matchImage 命中时必须展示库参考图；matchText 也可能带库侧 id
                has_registry_side = bool(registry_items) or any(
                    m.get("matchedRegistryId")
                    or m.get("matchedRegistryReferenceIds")
                    or m.get("queryRegistryReferenceIds")
                    for m in supported
                )
                extra_payload: dict[str, Any] = {
                    "matches": ranked,
                    "confirmedTracks": confirmed_tracks,
                    "uncertainTracks": uncertain_tracks,
                    "classificationKind": "track",
                    "matchThresholds": match_thresholds,
                    "planMode": "langgraph",
                    "matchCount": len(display_tracks),
                    "confirmedMatchCount": len(confirmed_tracks),
                    "uncertainMatchCount": len(uncertain_tracks),
                    "found": bool(confirmed_tracks),
                }
                if has_registry_side and registry_items:
                    extra_payload["registryItems"] = registry_items
                return self._finish(
                    conclusion,
                    confirmed_tracks,
                    hint,
                    finish_state,
                    extra=extra_payload,
                    display={
                        "tracks": display_tracks,
                        "includeClips": True,
                        "includeRegistry": has_registry_side,
                    },
                )
            # 有打分结果但无一达到 match/uncertain：仍按分数展示 top 候选作证据，禁止证据区空白
            candidate_tracks = self._tracks_ranked_by_matches(ranked, tracks)[: self.display_limit]
            top_score = float(ranked[0].get("embeddingScore") or ranked[0].get("score") or 0) if ranked else 0.0
            if is_registry_in_list:
                return self._finish(
                    "未在视频中确认在库船舶匹配",
                    candidate_tracks,
                    answer_hint or (
                        f"共 {len(ranked)} 条打分结果均为 mismatch（最高分 {top_score:.3f}）；"
                        "已附 top 候选供对照，勿视为确认命中"
                    ),
                    "sufficient" if state in {"sufficient", "uncertain"} else state,
                    extra={
                        "matches": ranked,
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": "both",
                        "matchCount": 0,
                        "candidateCount": len(ranked),
                        "found": False,
                    },
                    display={
                        "tracks": candidate_tracks,
                        "includeClips": bool(candidate_tracks),
                        "includeRegistry": bool(registry_items),
                    },
                )
            if target_scope == "registry":
                label = description or hull or "目标"
                candidate_items = self._registry_items_from_matches(ranked)[: self.display_limit]
                return self._finish(
                    f"数据库中未确认存在「{label}」",
                    [],
                    answer_hint or (
                        f"数据库匹配结果均未达到灰区或确认阈值（共 {len(ranked)} 条打分，最高分 {top_score:.3f}）；"
                        "已保留最高分库项作为对照证据，不能视为命中"
                    ),
                    "sufficient" if state != "conflict" else "conflict",
                    extra={
                        "matches": ranked,
                        "registryItems": candidate_items,
                        "planMode": "langgraph",
                        "targetScope": "registry",
                        "matchCount": 0,
                        "candidateCount": len(candidate_items),
                        "found": False,
                    },
                    display={"tracks": [], "includeClips": False, "includeRegistry": bool(candidate_items)},
                )
            no_match_conclusion = "未找到确认匹配目标"
            if operation == "existence":
                label = hull or description or "目标"
                if registry_items:
                    no_match_conclusion = f"未在视频中确认「{label}」（先验库有记录，视觉匹配未达阈值）"
                else:
                    no_match_conclusion = f"未确认符合条件的目标（{label}）"
            return self._finish(
                no_match_conclusion,
                candidate_tracks,
                answer_hint or (
                    f"匹配结果均为 mismatch 或空（共 {len(ranked)} 条打分，最高分 {top_score:.3f}）；"
                    "已展示 top 候选证据，非确认命中"
                ),
                "sufficient" if operation == "existence" else state,
                extra={
                    "matches": ranked,
                    "registryItems": registry_items,
                    "planMode": "langgraph",
                    "matchCount": 0,
                    "candidateCount": len(ranked),
                    "found": False,
                },
                display={
                    "tracks": candidate_tracks,
                    "includeClips": bool(candidate_tracks),
                    "includeRegistry": bool(registry_items),
                },
            )
        # ---- 分支 E：未在库查询但无匹配结果 --------------------------------
        # 先验库为空（全部轨迹皆为候选）或图像匹配输入不可用（降级为 uncertain）。
        if is_registry_out_list:
            if registry_listed and not registry_items:
                return self._finish(
                    f"先验库为空，视频中的 {len(tracks)} 条轨迹均属于未在库候选",
                    tracks,
                    answer_hint or "已列出完整先验库，当前库内无可对照项",
                    "sufficient",
                    extra={
                        "outOfRegistryTracks": tracks,
                        "outOfRegistryCount": len(tracks),
                        "registryItems": [],
                        "planMode": "langgraph",
                        "targetScope": "both",
                    },
                    display={"tracks": tracks, "includeClips": bool(tracks), "includeRegistry": False},
                )
            if match_image_attempted:
                return self._finish(
                    "候选轨迹存在，但全库图像匹配无法形成可评分结果",
                    tracks,
                    answer_hint or (
                        "图像匹配输入不可用，已停止重复调用；"
                        "这些轨迹只能列为待确认，不能直接判为未在库"
                    ),
                    "uncertain",
                    extra={
                        "planMode": "langgraph",
                        "targetScope": "both",
                        "outOfRegistryCount": 0,
                        "uncertainTracks": tracks,
                        "matchImageBlocked": match_image_blocked,
                        "registryItemCount": len(registry_items),
                    },
                    display={"tracks": tracks, "includeClips": bool(tracks), "includeRegistry": False},
                )

        # ---- 分支 F：纯库结果（有库项、无轨迹）-----------------------------
        # 含描述查询已执行 matchText 但返回空匹配的明确否定，以及舷号在库、
        # 但视频侧未命中的情形。
        if registry_items and not tracks:
            if not self.meta.get("targetScope"):
                self.meta["targetScope"] = "registry"
            # 纯数据库描述查询已执行 matchText 但返回空匹配，是明确的数据库否定证据。
            if description and target_kind == "description":
                if target_scope == "registry" and match_text_completed:
                    return self._finish(
                        f"数据库中未发现「{description}」",
                        [],
                        answer_hint or f"已读取 {len(registry_items)} 个数据库库项，描述匹配未返回候选",
                        "sufficient" if state != "conflict" else "conflict",
                        extra={
                            "registryItems": [],
                            "planMode": "langgraph",
                            "targetScope": "registry",
                            "matchCount": 0,
                            "found": False,
                            "registryItemCount": len(registry_items),
                        },
                        display={"tracks": [], "includeClips": False, "includeRegistry": False},
                    )
                return self._finish(
                    "先验库已列出但未完成描述筛选",
                    [],
                    answer_hint or f"已读取 {len(registry_items)} 个库项，缺少 matchText 筛选结果",
                    "uncertain" if state == "sufficient" else state,
                    extra={
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": self.meta.get("targetScope") or "registry",
                    },
                    display={"tracks": [], "includeClips": False, "includeRegistry": True},
                )
            # 舷号：库中有记录，但视频轨迹/视觉匹配未命中
            if operation == "existence" and hull:
                labels = "、".join(
                    str(item.get("hullNumber") or item.get("registryId") or "")
                    for item in registry_items[:3]
                    if item.get("hullNumber") or item.get("registryId")
                )
                return self._finish(
                    f"未在视频中发现「{hull}」" + (f"（先验库有：{labels}）" if labels else "（先验库有记录）"),
                    [],
                    answer_hint or f"getTrack 无舷号命中；先验库命中 {len(registry_items)} 项，视觉匹配未给出视频轨迹",
                    "sufficient" if state in {"sufficient", "uncertain"} else state,
                    extra={
                        "registryItems": registry_items,
                        "planMode": "langgraph",
                        "targetScope": "both",
                    },
                    display={"tracks": [], "includeClips": False, "includeRegistry": True},
                )
            return self._finish(
                f"先验库共 {len(registry_items)} 个库项" if not description else "已查询先验库",
                [],
                answer_hint,
                state,
                extra={
                    "registryItems": registry_items,
                    "planMode": "langgraph",
                    "targetScope": self.meta.get("targetScope") or "registry",
                },
                display={"tracks": [], "includeClips": False, "includeRegistry": True},
            )
        # ---- 分支 G：有轨迹 ------------------------------------------------
        if tracks:
            # 存在判断 + 舷号：全量扫轨得到的轨迹 ≠ 该舷号已确认出现
            # 仅当有非 mismatch 的视觉/文本匹配，或轨迹自身带该舷号时，才「确认出现」
            if operation == "existence" and hull:
                hull_on_track = any(
                    str(t.get("hullNumber") or "").upper() == hull.upper()
                    or str(t.get("finalHullNumber") or "").upper() == hull.upper()
                    for t in tracks
                    if isinstance(t, dict)
                )
                if not hull_on_track:
                    labels = "、".join(
                        str(item.get("hullNumber") or item.get("registryId") or "")
                        for item in registry_items[:3]
                        if item.get("hullNumber") or item.get("registryId")
                    ) if registry_items else ""
                    return self._finish(
                        f"未在视频中确认「{hull}」"
                        + (f"（先验库有：{labels}；全量检索 {len(tracks)} 条轨迹均未标此舷号/未视觉命中）" if labels
                           else f"（全量检索 {len(tracks)} 条轨迹，均未标此舷号）"),
                        tracks[: self.display_limit],
                        answer_hint or "放开舷号过滤后的轨迹不能直接当作目标命中",
                        "sufficient" if state in {"sufficient", "uncertain"} else state,
                        extra={
                            "registryItems": registry_items,
                            "planMode": "langgraph",
                            "found": False,
                            "targetScope": "both" if registry_items else target_scope,
                        },
                        display={"tracks": tracks[: self.display_limit], "includeClips": True, "includeRegistry": bool(registry_items)},
                    )
            if operation == "existence":
                conclusion = "确认出现" if state == "sufficient" else "疑似出现（证据不完整）"
            else:
                conclusion = "已定位相关轨迹" if state == "sufficient" else "仅获得部分轨迹证据"
            return self._finish(
                conclusion,
                tracks,
                answer_hint,
                state,
                extra={"planMode": "langgraph"},
                display={"tracks": tracks, "includeClips": True},
            )
        # ---- 分支 H：兜底（0 轨迹 / 0 匹配）--------------------------------
        # 0 轨迹/0 匹配：存在判断应给明确否定，而不是「未找到可靠证据」+ 误导标签
        if state == "conflict":
            return self._finish("证据存在冲突", [], answer_hint, state, extra={"planMode": "langgraph"})
        if operation == "existence":
            label = hull or description or "目标"
            scope_prefix = "数据库中" if target_scope == "registry" else ""
            reason_text = (
                "数据库已完成查询，未返回对应库项或匹配"
                if target_scope == "registry"
                else "检索范围内无对应轨迹或匹配"
            )
            return self._finish(
                f"{scope_prefix}未发现「{label}」",
                [],
                answer_hint or reason_text,
                "sufficient",
                extra={"planMode": "langgraph", "found": False, "targetScope": target_scope},
            )
        return self._finish("未找到可靠证据", [], answer_hint, state, extra={"planMode": "langgraph"})

    # ------------------------------------------------------------------------
    # L4 投影层：working_scope → 结构化事实
    # working_scope 的值是十几种业务工具各自的返回形态，这里没有 schema，全靠
    # isinstance/get 兜底，提取出 tracks / matches / registry / dedup / count，
    # 是 L3 合成与 L6 展示共同的输入源。注意物理位置在 _synthesize 之后。
    # ------------------------------------------------------------------------
    def _collect_tracks(self) -> list[dict[str, Any]]:
        collected: dict[str, dict[str, Any]] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    collected[str(track["trackId"])] = dict(track)
            for match in value.get("matches") or []:
                if not isinstance(match, dict):
                    continue
                track = match.get("track") if isinstance(match.get("track"), dict) else {}
                track_id = match.get("matchedTrackId") or match.get("trackId") or track.get("trackId")
                if track_id is None:
                    continue
                item = dict(track) if track else {"trackId": track_id}
                item.setdefault("trackId", track_id)
                for key in ("embeddingScore", "scoreBand", "hullNumber", "matchedRegistryId"):
                    if match.get(key) is not None:
                        item[key] = match.get(key)
                prev = collected.get(str(track_id), {})
                # 同轨迹多匹配时保留更高分
                if prev.get("embeddingScore") is not None and item.get("embeddingScore") is not None:
                    if float(item["embeddingScore"]) < float(prev["embeddingScore"]):
                        item = {**item, **{k: prev[k] for k in ("embeddingScore", "scoreBand") if k in prev}}
                collected[str(track_id)] = {**prev, **item}
        items = list(collected.values())
        # 有相似度的排前面，避免展示固定成轨迹编号顺序
        items.sort(
            key=lambda t: (
                0 if t.get("embeddingScore") is not None else 1,
                -float(t.get("embeddingScore") or 0),
                str(t.get("trackId") or ""),
            )
        )
        return items

    def _collect_matches(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for value in self.working_scope.values():
            if isinstance(value, dict):
                for match in value.get("matches") or []:
                    if isinstance(match, dict):
                        matches.append(match)
        matches.sort(key=lambda item: float(item.get("embeddingScore") or item.get("score") or 0), reverse=True)
        return matches

    def _tracks_ranked_by_matches(
        self,
        matches: list[dict[str, Any]],
        all_tracks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """只返回匹配命中的轨迹，按 embeddingScore 降序；用全量轨迹补全元数据。"""
        by_id = {
            str(t.get("trackId")): dict(t)
            for t in (all_tracks or [])
            if isinstance(t, dict) and t.get("trackId") is not None
        }
        # scope 里可能还有更完整的 track 记录
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    tid = str(track["trackId"])
                    by_id[tid] = {**by_id.get(tid, {}), **track}
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        ordered = sorted(
            [m for m in matches if isinstance(m, dict)],
            key=lambda m: float(m.get("embeddingScore") or m.get("score") or 0),
            reverse=True,
        )
        for match in ordered:
            track_id = match.get("matchedTrackId") or match.get("trackId")
            if track_id is None:
                continue
            key = str(track_id)
            if key in seen:
                continue
            seen.add(key)
            base = dict(by_id.get(key) or {"trackId": track_id})
            base["trackId"] = track_id
            if match.get("embeddingScore") is not None:
                base["embeddingScore"] = match.get("embeddingScore")
            if match.get("scoreBand") is not None:
                base["scoreBand"] = match.get("scoreBand")
            if match.get("matchedRegistryId") is not None:
                base["matchedRegistryId"] = match.get("matchedRegistryId")
            if match.get("hullNumber") is not None:
                base.setdefault("hullNumber", match.get("hullNumber"))
            # 把匹配关键帧挂到轨迹上，供 _display_tracks 直接用
            kids = match.get("matchedKeyframeIds") or match.get("queryKeyframeIds") or []
            if kids:
                base["matchedKeyframeIds"] = [str(x) for x in kids if x is not None]
                base["keyframeIds"] = list(base["matchedKeyframeIds"])
            refs = match.get("matchedRegistryReferenceIds") or match.get("queryRegistryReferenceIds") or []
            if refs:
                base["matchedRegistryReferenceIds"] = [str(x) for x in refs if x is not None]
            ranked.append(base)
        return ranked

    def _registry_items_from_matches(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从 matchText/matchImage 的库侧命中还原 registryItems，供前端只展示命中项。"""
        by_id: dict[str, dict[str, Any]] = {}
        all_items = {str(item.get("registryId")): item for item in self._collect_registry() if item.get("registryId")}
        for match in matches:
            if not isinstance(match, dict):
                continue
            rid = str(match.get("matchedRegistryId") or match.get("registryId") or "")
            if not rid:
                continue
            base = dict(all_items.get(rid) or {})
            base.setdefault("registryId", rid)
            if match.get("hullNumber"):
                base.setdefault("hullNumber", match.get("hullNumber"))
            if match.get("embeddingScore") is not None:
                base["embeddingScore"] = match.get("embeddingScore")
            if match.get("scoreBand"):
                base["scoreBand"] = match.get("scoreBand")
            by_id[rid] = base
        return list(by_id.values())

    def _collect_image_match_summary(self) -> dict[str, Any]:
        """取得最近一次图像匹配的全库覆盖指标。"""
        result: dict[str, Any] = {}
        for value in self.working_scope.values():
            if isinstance(value, dict) and value.get("matchMode") == "image_to_image":
                result = value
        return result

    def _collect_match_thresholds(self) -> dict[str, Any] | None:
        """优先返回工具本轮实际使用的匹配阈值，避免前端硬编码。"""
        for value in reversed(list(self.working_scope.values())):
            if not isinstance(value, dict):
                continue
            thresholds = value.get("matchThresholds")
            if isinstance(thresholds, dict):
                return dict(thresholds)
        return None

    @staticmethod
    def _match_threshold_note(thresholds: dict[str, Any] | None) -> str:
        if not isinstance(thresholds, dict):
            return ""
        try:
            confirmation = float(thresholds.get("confirmation"))
            exclusion = float(thresholds.get("exclusion"))
        except (TypeError, ValueError):
            return ""
        return (
            f"确认阈值为 {confirmation:.3f}（分数不低于该值）；"
            f"灰区为 {exclusion:.3f} 到 {confirmation:.3f} 之间（不含边界）"
        )

    def _collect_registry(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _add(item: dict[str, Any]) -> None:
            if not isinstance(item, dict):
                return
            key = str(item.get("registryId") or item.get("hullNumber") or len(items))
            if key in seen:
                return
            seen.add(key)
            items.append(item)

        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            for item in value.get("registryItems") or []:
                _add(item)
            # matchHull 返回 exactMatches: {hull: [registryItem, ...]}
            exact = value.get("exactMatches")
            if isinstance(exact, dict):
                for bucket in exact.values():
                    if isinstance(bucket, list):
                        for item in bucket:
                            _add(item)
                    elif isinstance(bucket, dict):
                        _add(bucket)
        return items

    def _collect_dedup_summary(self) -> dict[str, Any] | None:
        """提取最近一次跨轨迹去重结果，并兼容旧版字段。"""
        for value in reversed(list(self.working_scope.values())):
            if not isinstance(value, dict):
                continue
            if not any(
                value.get(key) is not None
                for key in (
                    "minimumShipCount", "confirmedShipCount", "highThresholdShipCount",
                    "lowThresholdShipCount", "highGroups", "lowGroups",
                )
            ):
                continue
            summary = dict(value)
            high_groups = [list(group) for group in (summary.get("highGroups") or []) if isinstance(group, list)]
            low_groups = [list(group) for group in (summary.get("lowGroups") or high_groups) if isinstance(group, list)]
            confirmed_count = summary.get("confirmedShipCount", summary.get("highThresholdShipCount"))
            minimum_count = summary.get("minimumShipCount", summary.get("lowThresholdShipCount"))
            if confirmed_count is None and high_groups:
                confirmed_count = len(high_groups)
            if minimum_count is None and low_groups:
                minimum_count = len(low_groups)
            if confirmed_count is None and minimum_count is None:
                continue
            if minimum_count is None:
                minimum_count = confirmed_count
            if confirmed_count is None:
                confirmed_count = minimum_count
            summary["minimumShipCount"] = int(minimum_count)
            summary["confirmedShipCount"] = int(confirmed_count)
            summary.setdefault("maximumShipCount", int(confirmed_count))
            summary.setdefault("highGroups", high_groups)
            summary.setdefault("lowGroups", low_groups)
            if not isinstance(summary.get("confirmedMergeGroups"), list):
                summary["confirmedMergeGroups"] = [
                    {"groupId": f"confirmed-{index + 1}", "status": "confirmed", "trackIds": group}
                    for index, group in enumerate(high_groups)
                    if len(group) > 1
                ]
            if not isinstance(summary.get("pendingMergeGroups"), list):
                pending_groups = []
                for group in low_groups:
                    group_set = set(str(item) for item in group)
                    current_groups = [
                        list(item) for item in high_groups
                        if set(str(track_id) for track_id in item).issubset(group_set)
                    ]
                    if len(current_groups) > 1:
                        pending_groups.append({
                            "groupId": f"pending-{len(pending_groups) + 1}",
                            "status": "pending",
                            "trackIds": group,
                            "currentGroups": current_groups,
                            "possibleReduction": len(current_groups) - 1,
                        })
                summary["pendingMergeGroups"] = pending_groups
            summary.setdefault("confirmedMergeCount", len(summary.get("confirmedMergeGroups") or []))
            summary.setdefault("pendingMergeCount", len(summary.get("pendingMergeGroups") or []))
            summary.setdefault("pendingReduction", max(0, int(confirmed_count) - int(minimum_count)))
            return summary
        return None

    def _collect_count(self) -> int | None:
        dedup = self._collect_dedup_summary()
        if dedup and dedup.get("minimumShipCount") is not None:
            return int(dedup["minimumShipCount"])
        for value in reversed(list(self.working_scope.values())):
            if not isinstance(value, dict):
                continue
            for key in ("minimumShipCount", "lowThresholdShipCount", "uniqueCount", "count", "dedupCount", "finalCount", "highThresholdShipCount"):
                if value.get(key) is not None:
                    try:
                        return int(value[key])
                    except (TypeError, ValueError):
                        pass
            groups = value.get("lowGroups") or value.get("highGroups") or value.get("groups")
            if isinstance(groups, list) and groups:
                return len(groups)
        return None

    @staticmethod
    def _count_evidence_track(track_id: str, track: dict[str, Any]) -> dict[str, Any]:
        """构造前端统计台账所需的最小轨迹时间证据。"""
        start_time = track.get("startTime") if track.get("startTime") is not None else track.get("start_time")
        end_time = track.get("endTime") if track.get("endTime") is not None else track.get("end_time")
        record: dict[str, Any] = {
            "trackId": str(track_id),
            "startTime": start_time,
            "endTime": end_time,
        }
        for key in ("hullNumber", "finalHullNumber", "shipSegmentIds", "keyframeIds"):
            if track.get(key) is not None:
                record[key] = track.get(key)
        return record

    def _build_count_evidence(
        self,
        dedup_summary: dict[str, Any],
        tracks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """将去重结果转换为“最终计数船舶单元 → 原始轨迹与时间”的可审计证据。

        这里不基于问题文本推断展示路径，只消费 dedupTracks 返回的 high/low groups。
        低阈值分组对应最小船数，便于回答“至少多少艘”时逐一列出每个计数单元；
        确认和灰区合并关系则作为独立的审计字段保留。
        """
        def _groups(value: Any) -> list[list[str]]:
            return [
                [str(track_id) for track_id in group if track_id is not None]
                for group in (value or [])
                if isinstance(group, list) and group
            ]

        by_id = {
            str(item.get("trackId")): dict(item)
            for item in tracks
            if isinstance(item, dict) and item.get("trackId") is not None
        }
        high_groups = _groups(dedup_summary.get("highGroups"))
        low_groups = _groups(dedup_summary.get("lowGroups")) or list(high_groups)
        confirmed_groups = [item for item in (dedup_summary.get("confirmedMergeGroups") or []) if isinstance(item, dict)]
        pending_groups = [item for item in (dedup_summary.get("pendingMergeGroups") or []) if isinstance(item, dict)]

        def _index_groups(groups: list[dict[str, Any]]) -> dict[frozenset[str], dict[str, Any]]:
            indexed: dict[frozenset[str], dict[str, Any]] = {}
            for item in groups:
                track_ids = frozenset(str(track_id) for track_id in (item.get("trackIds") or []) if track_id is not None)
                if track_ids:
                    indexed[track_ids] = item
            return indexed

        frame_by_track: dict[str, str] = {}
        for value in (getattr(self, "working_scope", {}) or {}).values():
            if not isinstance(value, dict):
                continue
            grouped = value.get("keyframesByTrack")
            if not isinstance(grouped, dict):
                continue
            for raw_track_id, bucket in grouped.items():
                frames = bucket.get("keyframes") if isinstance(bucket, dict) else bucket
                if not isinstance(frames, list):
                    continue
                best = max(
                    (item for item in frames if isinstance(item, dict) and item.get("keyframeId") is not None),
                    key=lambda item: float(item.get("retentionScore") or 0),
                    default=None,
                )
                if best:
                    frame_by_track[str(raw_track_id)] = str(best["keyframeId"])

        def _member(track_id: str) -> dict[str, Any]:
            record = self._count_evidence_track(track_id, by_id.get(track_id, {}))
            if track_id in frame_by_track:
                record["keyframeId"] = frame_by_track[track_id]
            return record

        confirmed_by_tracks = _index_groups(confirmed_groups)
        pending_by_tracks = _index_groups(pending_groups)
        high_sets = {frozenset(group) for group in high_groups}
        units: list[dict[str, Any]] = []
        covered_ids: set[str] = set()
        for index, track_ids in enumerate(low_groups, start=1):
            track_set = frozenset(track_ids)
            covered_ids.update(track_set)
            pending = pending_by_tracks.get(track_set)
            confirmed = confirmed_by_tracks.get(track_set)
            if pending:
                merge_state = "pending"
                merge_source = pending
            elif confirmed or (len(track_set) > 1 and track_set in high_sets):
                merge_state = "confirmed"
                merge_source = confirmed or {}
            else:
                merge_state = "independent"
                merge_source = {}
            unit = {
                "unitId": f"vessel-{index}",
                "trackIds": track_ids,
                "tracks": [_member(track_id) for track_id in track_ids],
                "mergeState": merge_state,
                "mergeGroupId": merge_source.get("groupId"),
                "minimumScore": merge_source.get("minimumScore"),
                "currentGroups": merge_source.get("currentGroups") or [],
            }
            units.append({key: value for key, value in unit.items() if value not in (None, [])})

        # 防止工具返回的分组遗漏已检出的轨迹；遗漏项必须单列，不能静默从计数证据中消失。
        for track_id in by_id:
            if track_id in covered_ids:
                continue
            units.append({
                "unitId": f"vessel-{len(units) + 1}",
                "trackIds": [track_id],
                "tracks": [_member(track_id)],
                "mergeState": "independent",
            })

        minimum_count = int(dedup_summary.get("minimumShipCount", len(units)))
        confirmed_count = int(dedup_summary.get("confirmedShipCount", minimum_count))
        return {
            "basis": "minimum" if minimum_count < confirmed_count else "confirmed",
            "minimumShipCount": minimum_count,
            "confirmedShipCount": confirmed_count,
            "evidenceUnitCount": len(units),
            "coverageComplete": len(units) == minimum_count,
            "vesselUnits": units,
            "rawTracks": [_member(track_id) for track_id in by_id],
        }


    # ------------------------------------------------------------------------
    # L5 组装层：_finish —— 把 conclusion 与证据拼成对外的 result dict
    # 顺序：按 evidenceMode 截断 → 委托 L6 生成展示 → 组装约 30 个字段返回。
    # ------------------------------------------------------------------------
    def _finish(
        self,
        conclusion: str,
        tracks: list[dict[str, Any]],
        reason: str,
        state: str,
        extra: dict[str, Any] | None = None,
        display: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        display = display if display is not None else {"tracks": tracks}
        # focused 证据模式（单目标判断）：只保留少量证据，避免 259 条级别的展示与片段开销
        evidence_mode = str((self.meta or {}).get("evidenceMode") or "")
        limited_tracks = tracks[: self.display_limit] if evidence_mode == "focused" else tracks
        if evidence_mode == "focused":
            display = dict(display or {})
            display["tracks"] = list(display.get("tracks") or [])[: self.display_limit]
        # 供 _display_tracks 优先使用的命中库项（避免展示整库）
        if extra and isinstance(extra.get("registryItems"), list):
            self._pending_registry_items = list(extra["registryItems"])
        else:
            self._pending_registry_items = []
        if isinstance(display.get("dedupSummary"), dict):
            self._display_dedup_groups(display.get("dedupSummary") or {}, display.get("tracks") or [])
        else:
            self._display_tracks(
                display.get("tracks") or [],
                display.get("includeClips", True) is not False,
                bool(display.get("includeRegistry")),
            )
        primary = [item["trackId"] for item in limited_tracks[: self.display_limit] if item.get("trackId") is not None]
        result = {
            "sessionId": self.session_id,
            "question": self.question,
            "questionType": self.meta.get("questionType"),
            "targetScope": self.meta.get("targetScope"),
            "targetKind": self.meta.get("targetKind"),
            "operation": self.meta.get("operation"),
            "registryRelation": self.meta.get("registryRelation"),
            "conclusion": conclusion,
            "answerText": f"{conclusion}。{reason}",
            "queryScope": list(self.meta["timeRange"]) if self.meta.get("timeRange") else None,
            "toolChain": self.tool_chain,
            "toolRecords": self.tool_records,
            "tracks": limited_tracks,
            "evidence": self._collect_evidence(),
            "displayGroups": self.display_groups,
            "display": self._public_display(),
            "uncertainty": state,
            "primaryTrackIds": primary,
            "remainingTrackIds": [
                item["trackId"] for item in limited_tracks[self.display_limit :] if item.get("trackId") is not None
            ],
            "rounds": [{k: v for k, v in item.items() if k != "scope"} for item in self.rounds],
            "intent": self.meta,
            "planMode": "langgraph",
            "evidenceMode": evidence_mode,
        }
        if extra:
            result.update(extra)
        self._emit("synthesis", "生成最终回答", reason, conclusion=conclusion, state=state, trackCount=len(limited_tracks))
        return result

    # ------------------------------------------------------------------------
    # L6 展示层：生成前端的证据分组（displayGroups）
    # 两种互斥模式：计数场景走 _display_dedup_groups（按合并组展示，避免罗列
    # 原始轨迹），其余走 _display_tracks（按轨迹展示画廊）。
    # 注意本层有副作用：缺关键帧/片段时会现场调用 tools.getFrames / getClip；
    # 且 display_record 一旦写入即不再重算（先到先得）。
    # ------------------------------------------------------------------------
    def _display_dedup_groups(self, summary: dict[str, Any], tracks: list[dict[str, Any]]) -> None:
        """按确认合并/待确认合并分组生成关键帧证据，避免继续罗列原始轨迹。"""
        if self.display_record is not None:
            return
        confirmed_groups = [item for item in (summary.get("confirmedMergeGroups") or []) if isinstance(item, dict)]
        pending_groups = [item for item in (summary.get("pendingMergeGroups") or []) if isinstance(item, dict)]
        merge_groups = [("confirmed", item) for item in confirmed_groups] + [("pending", item) for item in pending_groups]
        by_id = {
            str(item.get("trackId")): dict(item)
            for item in tracks
            if isinstance(item, dict) and item.get("trackId") is not None
        }
        frame_groups: dict[str, Any] = {}
        for value in self.working_scope.values():
            if not isinstance(value, dict):
                continue
            grouped = value.get("keyframesByTrack")
            if isinstance(grouped, dict):
                frame_groups.update({str(key): bucket for key, bucket in grouped.items()})
            for track in value.get("tracks") or []:
                if isinstance(track, dict) and track.get("trackId") is not None:
                    track_id = str(track["trackId"])
                    by_id[track_id] = {**by_id.get(track_id, {}), **track}

        all_track_ids = list(dict.fromkeys(
            str(track_id)
            for _, group in merge_groups
            for track_id in (group.get("trackIds") or [])
            if track_id is not None
        ))
        missing_frame_ids = [track_id for track_id in all_track_ids if track_id not in frame_groups]
        if missing_frame_ids:
            try:
                fetched = self.tools.getFrames(missing_frame_ids)
                if isinstance(fetched, dict) and isinstance(fetched.get("keyframesByTrack"), dict):
                    frame_groups.update({str(key): bucket for key, bucket in fetched["keyframesByTrack"].items()})
            except Exception:
                pass

        def _best_keyframe(track_id: str) -> str | None:
            bucket = frame_groups.get(track_id, {})
            if isinstance(bucket, dict):
                frames = bucket.get("keyframes") if isinstance(bucket.get("keyframes"), list) else []
            elif isinstance(bucket, list):
                frames = bucket
            else:
                frames = []
            best = max(
                (item for item in frames if isinstance(item, dict)),
                key=lambda item: float(item.get("retentionScore") or 0),
                default=None,
            )
            return str(best["keyframeId"]) if best and best.get("keyframeId") is not None else None

        for group_type, source in merge_groups:
            track_ids = [str(item) for item in (source.get("trackIds") or []) if item is not None]
            members = []
            keyframe_ids = []
            for track_id in track_ids:
                track = by_id.get(track_id, {})
                keyframe_id = _best_keyframe(track_id)
                if keyframe_id:
                    keyframe_ids.append(keyframe_id)
                members.append({
                    "trackId": track_id,
                    "keyframeId": keyframe_id,
                    "hullNumber": track.get("finalHullNumber") or track.get("hullNumber"),
                    "startTime": track.get("startTime") if track.get("startTime") is not None else track.get("start_time"),
                    "endTime": track.get("endTime") if track.get("endTime") is not None else track.get("end_time"),
                })
            display_group = {
                "trackId": source.get("groupId") or f"{group_type}-{len(self.display_groups) + 1}",
                "groupId": source.get("groupId") or f"{group_type}-{len(self.display_groups) + 1}",
                "groupType": group_type,
                "mergedTrackIds": track_ids,
                "currentGroups": source.get("currentGroups") or [],
                "minimumScore": source.get("minimumScore"),
                "possibleReduction": source.get("possibleReduction"),
                "memberEvidence": members,
                "keyframeIds": list(dict.fromkeys(keyframe_ids)),
                "shipSegmentIds": [],
                "registryReferenceIds": [],
            }
            self.display_groups.append(display_group)

        self.display_record = {
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "mode": "dedup-groups",
            "trackCount": len(all_track_ids),
            "groupCount": len(self.display_groups),
            "confirmedGroupCount": len(confirmed_groups),
            "pendingGroupCount": len(pending_groups),
            "groups": self.display_groups,
        }

    def _display_tracks(self, tracks: list[dict[str, Any]], include_clips: bool = True, include_registry: bool = False) -> None:
        if self.display_record is not None:
            return
        # 兼容测试、恢复任务等绕过 __init__ 构造的控制器实例。
        working_scope = getattr(self, "working_scope", {}) or {}
        pending_registry_items = getattr(self, "_pending_registry_items", []) or []
        unique_tracks = list({str(item["trackId"]): item for item in tracks if item.get("trackId") is not None}.values())
        # focused 证据模式兜底：单目标判断只展示少量证据（含片段生成开销）
        if str(getattr(self, "meta", {}).get("evidenceMode") or "") == "focused":
            limit = getattr(self, "display_limit", 3) or 3
            if len(unique_tracks) > limit:
                unique_tracks = unique_tracks[:limit]
        if not unique_tracks:
            # 仅先验库结果：优先用本轮合成得到的 registry 命中项，再回退 working_scope
            reference_ids: list[str] = []
            preferred_items: list[dict[str, Any]] = []
            if include_registry:
                preferred_items = list(pending_registry_items or [])
                sources = [preferred_items] if preferred_items else []
                if not sources:
                    for value in working_scope.values():
                        if isinstance(value, dict) and value.get("registryItems"):
                            sources.append(value.get("registryItems") or [])
                for value in working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for key in ("registryReferenceIds", "shownRegistryReferenceIds"):
                        for item in value.get(key) or []:
                            if item is not None:
                                reference_ids.append(str(item))
                    for item in (preferred_items or value.get("registryItems") or []):
                        if not isinstance(item, dict):
                            continue
                        for ref in item.get("references") or []:
                            if isinstance(ref, dict) and ref.get("referenceId"):
                                reference_ids.append(str(ref["referenceId"]))
                    exact = value.get("exactMatches")
                    if isinstance(exact, dict):
                        for bucket in exact.values():
                            rows = bucket if isinstance(bucket, list) else [bucket]
                            for item in rows:
                                if not isinstance(item, dict):
                                    continue
                                for ref in item.get("references") or []:
                                    if isinstance(ref, dict) and ref.get("referenceId"):
                                        reference_ids.append(str(ref["referenceId"]))
                # 有命中库项但无参考图 ID 时，仍按库项生成展示组
                if preferred_items and not reference_ids:
                    for item in preferred_items:
                        for ref in item.get("references") or []:
                            if isinstance(ref, dict) and ref.get("referenceId"):
                                reference_ids.append(str(ref["referenceId"]))
                reference_ids = list(dict.fromkeys(reference_ids))
                if hasattr(self.tools, "_representative_registry_reference_ids"):
                    try:
                        reference_ids = list(self.tools._representative_registry_reference_ids(reference_ids))
                    except Exception:
                        pass
            self.display_record = {
                "displayId": f"display-{uuid.uuid4().hex[:12]}",
                "mode": "lazy",
                "trackCount": 0,
                "registryReferenceCount": len(reference_ids),
                "registryItemCount": len(preferred_items) if preferred_items else 0,
            }
            if reference_ids:
                self.display_groups.append({
                    "trackId": None,
                    "keyframeIds": [],
                    "shipSegmentIds": [],
                    "registryReferenceIds": reference_ids[:12],
                })
            elif preferred_items:
                # 无向量参考图时仍把库项 ID 放进 evidence，前端可按 registryItems 渲染
                self.display_groups.append({
                    "trackId": None,
                    "keyframeIds": [],
                    "shipSegmentIds": [],
                    "registryReferenceIds": [],
                    "registryItems": preferred_items[:12],
                })
            return
        # 按相似度粗排（有分数的靠前）
        def _score(track: dict[str, Any]) -> float:
            for key in ("embeddingScore", "score", "matchScore"):
                if track.get(key) is not None:
                    try:
                        return float(track[key])
                    except (TypeError, ValueError):
                        pass
            return float("-inf")

        unique_tracks = sorted(unique_tracks, key=_score, reverse=True)
        missing = [
            str(t["trackId"])
            for t in unique_tracks
            if not self._ids(t, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
        ]
        frame_groups = {}
        if missing:
            try:
                frame_groups = self.tools.getFrames(missing).get("keyframesByTrack", {}) or {}
            except Exception:
                frame_groups = {}
        for track in unique_tracks:
            track_id = str(track["trackId"])
            keyframe_ids = self._ids(track, "matchedKeyframeIds", "queryKeyframeIds", "keyframeIds")
            # 匹配结果常只有 trackId+分数，从 working_scope.matches 补关键帧
            if not keyframe_ids:
                for value in working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for match in value.get("matches") or []:
                        if not isinstance(match, dict):
                            continue
                        mid = str(match.get("matchedTrackId") or match.get("trackId") or "")
                        if mid != track_id:
                            continue
                        for kid in match.get("matchedKeyframeIds") or match.get("queryKeyframeIds") or []:
                            if kid is not None:
                                keyframe_ids.append(str(kid))
                        if match.get("embeddingScore") is not None and track.get("embeddingScore") is None:
                            track["embeddingScore"] = match.get("embeddingScore")
                        if match.get("scoreBand") is not None and track.get("scoreBand") is None:
                            track["scoreBand"] = match.get("scoreBand")
                keyframe_ids = list(dict.fromkeys(keyframe_ids))
            if not keyframe_ids:
                frames = frame_groups.get(track_id, {}).get("keyframes", [])
                best = max(frames, key=lambda item: item.get("retentionScore", 0), default=None)
                keyframe_ids = [best["keyframeId"]] if best and best.get("keyframeId") else []
            segment_ids = self._ids(track, "shipSegmentIds")[:1]
            if include_clips and not segment_ids and hasattr(self.tools, "getClip"):
                try:
                    clip = self.tools.getClip(track_id)
                    if isinstance(clip, dict) and clip.get("shipSegmentId"):
                        segment_ids = [str(clip["shipSegmentId"])]
                except Exception:
                    pass
            # 库参考图：有 matchImage 结果时始终尝试挂上，不依赖 include_registry 开关
            raw_refs = self._ids(track, "matchedRegistryReferenceIds", "registryReferenceIds", "queryRegistryReferenceIds")
            if not raw_refs:
                for value in working_scope.values():
                    if not isinstance(value, dict):
                        continue
                    for match in value.get("matches") or []:
                        if not isinstance(match, dict):
                            continue
                        mid = str(match.get("matchedTrackId") or match.get("trackId") or "")
                        if mid != track_id:
                            continue
                        for rid in (
                            match.get("matchedRegistryReferenceIds")
                            or match.get("queryRegistryReferenceIds")
                            or []
                        ):
                            if rid is not None:
                                raw_refs.append(str(rid))
                        # 仅有 matchedRegistryId 时，从库项 references 展开
                        rid_item = match.get("matchedRegistryId")
                        if rid_item and not raw_refs:
                            for reg in self._collect_registry():
                                if str(reg.get("registryId") or "") != str(rid_item):
                                    continue
                                for ref in reg.get("references") or []:
                                    if isinstance(ref, dict) and ref.get("referenceId"):
                                        raw_refs.append(str(ref["referenceId"]))
            if include_registry and not raw_refs:
                # 无匹配参考图时，用本轮库项代表图兜底
                for item in pending_registry_items or self._collect_registry():
                    if not isinstance(item, dict):
                        continue
                    for ref in item.get("references") or []:
                        if isinstance(ref, dict) and ref.get("referenceId"):
                            raw_refs.append(str(ref["referenceId"]))
            if hasattr(self.tools, "_representative_registry_reference_ids") and raw_refs:
                try:
                    raw_refs = list(self.tools._representative_registry_reference_ids(raw_refs))
                except Exception:
                    pass
            reference_ids = list(dict.fromkeys(str(x) for x in raw_refs if x is not None))[:3]
            group: dict[str, Any] = {
                "trackId": track_id,
                "clipTrackId": track_id,
                "keyframeIds": keyframe_ids[:3],
                "shipSegmentIds": segment_ids,
                "registryReferenceIds": reference_ids,
            }
            for key in ("embeddingScore", "score", "matchScore", "scoreBand", "hullNumber", "registryOutState", "coverageLimited"):
                if track.get(key) is not None:
                    group[key] = track[key]
            self.display_groups.append(group)
        self.display_record = {
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "mode": "lazy",
            "trackCount": len(unique_tracks),
            "registryReferenceCount": sum(len(g.get("registryReferenceIds") or []) for g in self.display_groups),
            "groups": self.display_groups,
        }

    # ------------------------------------------------------------------------
    # L7 聚合层：证据摊平与审计摘要
    # _collect_evidence 汇出前端拉取证据所需的 ID 索引；_session_audit_result
    # 产出落库用的瘦身结果（不含展示明细）。
    # ------------------------------------------------------------------------
    def _collect_evidence(self) -> dict[str, list[str]]:
        if self.display_groups:
            return {
                key: list(dict.fromkeys(value for group in self.display_groups for value in group.get(key) or []))
                for key in ("keyframeIds", "shipSegmentIds", "registryReferenceIds")
            }
        return {"keyframeIds": [], "shipSegmentIds": [], "registryReferenceIds": []}

    def _public_display(self) -> dict[str, Any] | None:
        if not self.display_record:
            return None
        return {key: value for key, value in self.display_record.items() if key != "scope"}

    @staticmethod
    def _ids(item: dict[str, Any], *keys: str) -> list[str]:
        values = []
        for key in keys:
            value = item.get(key, [])
            values.extend(value if isinstance(value, list) else [value] if value else [])
        return list(dict.fromkeys(str(value) for value in values if value))

    def _session_audit_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "conclusion": result.get("conclusion"),
            "uncertainty": result.get("uncertainty"),
            "trackCount": len(result.get("tracks") or []),
            "toolChain": result.get("toolChain") or [],
            "planMode": result.get("planMode") or "langgraph",
            "decisionMetrics": self.decision_metrics,
            "memoryPersistError": self.memory_persist_error,
        }

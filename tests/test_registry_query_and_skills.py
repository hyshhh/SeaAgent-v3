"""纯数据库查询与三智能体技能循环回归。"""
from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.controller import AgentController
from agent.graph import (
    _build_acceptance_progress,
    _default_plan_calls,
    _prepare_plan_calls,
    _skill_read_records,
    run_sea_agent,
)
from tools.target_parser import clear_target_parser_cache, infer_intent_fields


def _registry_description_intent():
    clear_target_parser_cache()
    return infer_intent_fields("数据库中有没有黄色的无人艇？")


def test_database_description_question_is_registry_only():
    intent = _registry_description_intent()

    assert intent["targetScope"] == "registry"
    assert intent["targetKind"] == "description"
    assert intent["operation"] == "existence"
    assert intent["description"] == "黄色的无人艇"
    assert intent["questionType"] == "registry_description_existence"
    assert "getTrack" not in intent["nextAgentFocus"]
    assert "getFrames" not in intent["nextAgentFocus"]


def test_database_description_default_plan_never_uses_video_tools():
    calls = _default_plan_calls(_registry_description_intent(), top_k=3)

    assert [call["tool"] for call in calls] == ["listRegistry", "matchText"]
    assert calls[1]["arguments"]["description"] == "黄色的无人艇"
    assert calls[1]["arguments"]["galleryImages"]["$ref"] == "registry.registryReferences"
    assert not {call["tool"] for call in calls} & {"getTrack", "getFrames", "getClip", "matchImage"}


def test_database_scope_guard_repairs_semantically_wrong_video_plan():
    calls, repair = _prepare_plan_calls(
        [
            {"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 60}},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
        ],
        _registry_description_intent(),
        3,
        broad_match_top_k=0,
    )

    assert "scope_violation:registry_only" in repair
    assert [call["tool"] for call in calls] == ["listRegistry", "matchText"]


def test_database_description_acceptance_does_not_require_tracks_or_frames():
    intent = _registry_description_intent()
    progress = _build_acceptance_progress(
        intent,
        {"listRegistry", "matchText"},
        track_count=None,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=True,
        can_try_visual=True,
        visual_attempted=True,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert progress["mode"] == "registry_only"
    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []
    assert {item["key"] for item in progress["requirements"]} == {"registry", "registry_text_match"}
    assert "getTrack" not in progress["nextAction"]
    assert "getFrames" not in progress["nextAction"]


def test_empty_database_short_circuits_description_matching_acceptance():
    progress = _build_acceptance_progress(
        _registry_description_intent(),
        {"listRegistry"},
        track_count=None,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []


def test_skill_read_records_include_titles_and_sources():
    reads = _skill_read_records("plan_agent", ["core_planning", "registry_query"], source="auto")

    assert [item["skillId"] for item in reads] == ["core_planning", "registry_query"]
    assert all(item["title"] for item in reads)
    assert all(item["source"] == "auto" for item in reads)
    assert all(item["ok"] is True for item in reads)


def _tool_messages(tool_name: str, args: dict, payload: dict, call_id: str):
    return [
        AIMessage(
            content="",
            tool_calls=[{
                "name": tool_name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }],
        ),
        ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=call_id,
            name=tool_name,
        ),
    ]


def test_three_collaboration_nodes_emit_skill_reads_and_keep_database_scope():
    inferred = _registry_description_intent()
    events = []
    tool_sets = {}

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                wrong_intent = {
                    **inferred,
                    "targetScope": "track_memory",
                    "successCriteria": "getTrack → getFrames → matchText",
                    "nextAgentFocus": "getTrack → getFrames → matchText",
                }
                payload = {"ok": True, "handoff": "plan", "intent": wrong_intent, "note": ""}
                yield "values", {"messages": _tool_messages(
                    "handoff_to_plan", {"intent": wrong_intent, "note": ""}, payload, "intent-handoff"
                )}
                return
            if self.name == "plan":
                messages = []
                messages += _tool_messages(
                    "load_recovery", {},
                    {"ok": True, "skillId": "recovery", "content": "恢复规则"}, "plan-skill"
                )
                handoff = {
                    "ok": True,
                    "handoff": "observe",
                    "goal": "查询数据库描述",
                    # 故意给出语义错误但参数合法的视频计划，范围守卫必须纠正。
                    "calls": [{"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 60}}],
                    "planHint": "错误的视频计划",
                    "reason": "测试范围守卫",
                }
                messages += _tool_messages(
                    "handoff_to_observe",
                    {"goal": handoff["goal"], "calls": handoff["calls"], "planHint": handoff["planHint"], "reason": handoff["reason"]},
                    handoff,
                    "plan-handoff",
                )
                yield "values", {"messages": messages}
                return
            if self.name == "observe":
                messages = []
                messages += _tool_messages(
                    "load_argument_rules", {},
                    {"ok": True, "skillId": "argument_rules", "content": "参数规则"}, "observe-skill"
                )
                handoff = {
                    "ok": True,
                    "handoff": "reflect",
                    "summary": "数据库名录与描述匹配均已完成",
                    "evidenceGap": "",
                    "proposedState": "sufficient",
                }
                messages += _tool_messages(
                    "handoff_to_reflect",
                    {"summary": handoff["summary"], "evidenceGap": "", "proposedState": "sufficient"},
                    handoff,
                    "observe-handoff",
                )
                yield "values", {"messages": messages}
                return
            if self.name == "reflect":
                messages = []
                messages += _tool_messages(
                    "load_conflict_uncertain", {},
                    {"ok": True, "skillId": "conflict_uncertain", "content": "不确定规则"}, "reflect-skill"
                )
                handoff = {
                    "ok": True,
                    "handoff": "finish",
                    "state": "sufficient",
                    "reason": "纯数据库验收完成",
                    "answerHint": "数据库中有确认匹配",
                }
                messages += _tool_messages(
                    "handoff_finish",
                    {"state": "sufficient", "reason": handoff["reason"], "answerHint": handoff["answerHint"]},
                    handoff,
                    "reflect-handoff",
                )
                yield "values", {"messages": messages}
                return
            raise AssertionError(self.name)

        def invoke(self, *args, **kwargs):
            raise AssertionError("不应进入非流式兜底")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            if name == "listRegistry":
                return {
                    "ok": True,
                    "registryItems": [{
                        "registryId": "registry-yellow",
                        "description": "黄色无人艇",
                        "references": [{"referenceId": "registry-ref-yellow", "registryId": "registry-yellow"}],
                    }],
                    "registryReferences": [{
                        "referenceId": "registry-ref-yellow",
                        "registryId": "registry-yellow",
                        "isEmbedded": True,
                        "registryVectorId": 1,
                    }],
                }
            if name == "matchText":
                return {
                    "ok": True,
                    "matchMode": "text_to_registry",
                    "matches": [{
                        "matchedRegistryId": "registry-yellow",
                        "embeddingScore": 0.91,
                        "scoreBand": "match",
                    }],
                }
            raise AssertionError(f"纯数据库查询不应执行 {name}")

    def _fake_create_agent(model, tools, system_prompt, name):
        names = {str(getattr(tool, "name", "")) for tool in tools}
        tool_sets[name] = names
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        state = run_sea_agent(
            "数据库中有没有黄色的无人艇？",
            object(),
            _FakeTools(),
            max_rounds=3,
            query_top_k=3,
            event_handler=events.append,
        )

    assert state["intent"]["targetScope"] == "registry"
    assert state["intent"]["description"] == "黄色的无人艇"
    assert state["tool_chain"] == ["listRegistry", "matchText"]
    assert state["final_state"] == "sufficient"
    for name in ("plan", "observe", "reflect"):
        assert any(t.startswith("load_") for t in tool_sets[name])
    end_events = {
        event["role"]: event
        for event in events
        if event.get("type") == "agent_end" and event.get("role") in {"planner", "observer", "reflector"}
    }
    assert set(end_events) == {"planner", "observer", "reflector"}
    assert all(end_events[role].get("enabledSkills") for role in end_events)
    assert all(end_events[role].get("skillReads") for role in end_events)
    assert any(item.get("source") == "dynamic" for item in end_events["planner"]["skillReads"])
    assert any(item.get("source") == "dynamic" for item in end_events["observer"]["skillReads"])
    assert any(item.get("source") == "dynamic" for item in end_events["reflector"]["skillReads"])


def _controller_for_registry(matches):
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "existence",
        "targetScope": "registry",
        "targetKind": "description",
        "registryRelation": "in",
        "questionType": "registry_description_existence",
        "description": "黄色的无人艇",
    }
    registry_item = {
        "registryId": "registry-yellow",
        "description": "黄色无人艇",
        "references": [{"referenceId": "registry-ref-yellow"}],
    }
    controller.working_scope = {
        "registry": {"ok": True, "registryItems": [registry_item], "registryReferences": []},
        "match": {"ok": True, "matchMode": "text_to_registry", "matches": matches},
    }
    controller.tool_records = [
        {"tool": "listRegistry", "ok": True, "skipped": False, "result": controller.working_scope["registry"]},
        {"tool": "matchText", "ok": True, "skipped": False, "result": controller.working_scope["match"]},
    ]
    controller.tool_chain = ["listRegistry", "matchText"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-registry", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-registry"
    controller.question = "数据库中有没有黄色的无人艇？"
    controller.event_handler = None
    controller._pending_registry_items = []
    controller.tools = type("T", (), {})()
    return controller


def test_registry_synthesis_answers_database_not_video():
    controller = _controller_for_registry([{
        "matchedRegistryId": "registry-yellow",
        "embeddingScore": 0.91,
        "scoreBand": "match",
    }])

    result = controller._synthesize("sufficient", "数据库验收完成")

    assert "数据库中确认存在" in result["conclusion"]
    assert "视频" not in result["conclusion"]
    assert result["found"] is True
    assert result["targetScope"] == "registry"


def test_registry_synthesis_reports_mismatch_as_database_absence_with_candidates():
    controller = _controller_for_registry([{
        "matchedRegistryId": "registry-yellow",
        "embeddingScore": 0.22,
        "scoreBand": "mismatch",
    }])

    result = controller._synthesize("sufficient", "数据库验收完成")

    assert "数据库中未确认存在" in result["conclusion"]
    assert "视频" not in result["conclusion"]
    assert result["found"] is False
    assert result["candidateCount"] == 1

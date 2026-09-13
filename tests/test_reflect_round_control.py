import json
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent.controller import AgentController
from agent.graph import (
    _bounded_model,
    _build_acceptance_progress,
    _default_plan_calls,
    _stream_tool_chunk_chars,
    run_sea_agent,
)
from agent.task_profiles import registry_membership_list_mode
from tools.target_parser import clear_target_parser_cache, infer_intent_fields



def test_reflect_model_uses_independent_server_side_output_cap():
    class _Model:
        max_tokens = None

        def model_copy(self, *, update):
            clone = _Model()
            clone.max_tokens = update["max_tokens"]
            return clone

    base = _Model()
    reflect = _bounded_model(base, 256)

    assert base.max_tokens is None
    assert reflect is not base
    assert reflect.max_tokens == 256


def test_reflect_stream_guard_counts_tool_arguments_without_visible_text():
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{
            "name": "handoff_finish",
            "args": "{\"state\":\"sufficient\",\"reason\":\"证据充分\"}",
            "id": "reflect-call",
            "index": 0,
            "type": "tool_call_chunk",
        }],
    )

    assert _stream_tool_chunk_chars(chunk) >= len("handoff_finish")
    assert _stream_tool_chunk_chars(chunk) > len(chunk.content)

def _out_fields():
    clear_target_parser_cache()
    return infer_intent_fields("有哪些未在库船出现在视频中？")


def test_registry_out_intent_separates_acceptance_and_current_focus():
    fields = _out_fields()

    assert fields["operation"] == "list"
    assert fields["registryRelation"] == "out"
    assert fields["questionType"] == "registry_out_list"
    assert not fields.get("description")
    assert fields["successCriteria"] != fields["nextAgentFocus"]
    assert "listRegistry" in fields["successCriteria"]
    assert "matchImage" in fields["successCriteria"]
    assert "mismatch" in fields["successCriteria"]
    assert "第一轮" in fields["nextAgentFocus"]
    assert "getTrack" in fields["nextAgentFocus"]
    assert "getFrames" in fields["nextAgentFocus"]


def test_registry_out_first_round_enumerates_tracks_before_full_registry_audit():
    calls = _default_plan_calls(_out_fields(), top_k=1, broad_match_top_k=0)

    assert [call["tool"] for call in calls] == ["getTrack", "getFrames"]


def test_registry_out_replan_reuses_existing_tracks_and_frames_with_custom_call_ids():
    calls = _default_plan_calls(
        _out_fields(),
        top_k=1,
        broad_match_top_k=0,
        replan_hint="补全 listRegistry 与 matchImage",
        working_scope={
            "all_video_tracks": {
                "ok": True,
                "trackIds": ["1", "2"],
                "tracks": [{"trackId": "1"}, {"trackId": "2"}],
            },
            "all_video_frames": {
                "ok": True,
                "keyframes": [
                    {"trackId": "1", "keyframeId": "f1"},
                    {"trackId": "2", "keyframeId": "f2"},
                ],
            },
        },
    )

    assert [call["tool"] for call in calls] == ["listRegistry", "matchImage"]
    assert calls[1]["arguments"]["galleryImages"] == {"$ref": "all_video_frames.keyframes"}
    assert calls[1]["arguments"]["topK"] == 0


def test_registry_out_replan_generates_no_registry_or_match_calls_for_known_zero_tracks():
    calls = _default_plan_calls(
        _out_fields(),
        top_k=1,
        broad_match_top_k=0,
        replan_hint="补全 listRegistry 与 matchImage",
        working_scope={
            "all_video_tracks": {"ok": True, "trackIds": [], "tracks": []},
        },
    )

    assert calls == []


def test_reflect_acceptance_requests_next_round_when_registry_audit_is_missing():
    fields = _out_fields()
    progress = _build_acceptance_progress(
        fields,
        {"getTrack", "getFrames"},
        track_count=11,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert registry_membership_list_mode(fields) == "out"
    assert progress["acceptanceSatisfied"] is False
    assert progress["pendingRequirements"] == ["已获取完整先验库名录"]
    assert "listRegistry" in progress["nextAction"]
    assert "matchImage" in progress["nextAction"]


def test_reflect_acceptance_short_circuits_registry_audit_when_video_has_no_tracks():
    progress = _build_acceptance_progress(
        _out_fields(),
        {"getTrack"},
        track_count=0,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []
    assert progress["videoEmptyShortCircuit"] is True
    assert "无需 listRegistry" in progress["nextAction"]
    assert "matchImage" in progress["nextAction"]


def test_reflect_acceptance_stops_with_uncertainty_when_image_inputs_are_blocked():
    progress = _build_acceptance_progress(
        _out_fields(),
        {"getTrack", "getFrames", "listRegistry", "matchImage"},
        track_count=2,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=True,
        can_try_visual=True,
        visual_attempted=True,
        match_image_attempted=True,
        match_image_usable=False,
        has_tool_evidence=True,
        match_image_blocked=True,
    )

    assert progress["acceptanceSatisfied"] is True
    assert progress["pendingRequirements"] == []
    assert progress["terminalState"] == "uncertain"
    assert progress["matchImageBlocked"] is True


def test_reflect_acceptance_passes_after_full_registry_image_match():
    fields = _out_fields()
    progress = _build_acceptance_progress(
        fields,
        {"getTrack", "getFrames", "listRegistry", "matchImage"},
        track_count=11,
        registry_checked=True,
        registry_listed=True,
        registry_has_items=True,
        can_try_visual=True,
        visual_attempted=True,
        match_image_attempted=True,
        match_image_usable=True,
        has_tool_evidence=True,
    )

    assert progress["pendingRequirements"] == []
    assert progress["acceptanceSatisfied"] is True


def test_registry_out_synthesis_returns_only_mismatch_tracks():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {
            "ok": True,
            "tracks": [
                {"trackId": "1"},
                {"trackId": "2"},
                {"trackId": "3"},
            ],
        },
        "registry": {
            "ok": True,
            "registryItems": [{"registryId": "r1", "hullNumber": "0857"}],
        },
        "match": {
            "ok": True,
            "matches": [
                {"matchedTrackId": "1", "matchedRegistryId": "r1", "embeddingScore": 0.91, "scoreBand": "match"},
                {"matchedTrackId": "2", "matchedRegistryId": "r1", "embeddingScore": 0.31, "scoreBand": "mismatch"},
                {"matchedTrackId": "3", "matchedRegistryId": "r1", "embeddingScore": 0.61, "scoreBand": "uncertain"},
            ],
        },
    }
    controller.tool_records = [
        {"tool": "listRegistry", "ok": True},
        {"tool": "matchImage", "ok": True, "skipped": False},
    ]
    controller.tool_chain = ["getTrack", "getFrames", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-test", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []

    result = controller._synthesize("sufficient", "全库对照完成")

    assert result["outOfRegistryCount"] == 1
    assert [item["trackId"] for item in result["tracks"]] == ["2"]
    assert [item["trackId"] for item in result["uncertainTracks"]] == ["3"]
    assert [item["registryOutState"] for item in result["outOfRegistryTracks"]] == ["confirmed_out"]
    assert [item["registryOutState"] for item in result["uncertainTracks"]] == ["gray_zone"]
    assert result["matchCount"] == 2
    assert result["classificationKind"] == "track"
    assert result["classificationMode"] == "registry_out"
    assert result["inRegistryMatchCount"] == 1
    assert result["uncertainty"] == "uncertain"
    assert "确认发现 1 条未在库轨迹" in result["conclusion"]




def test_registry_out_synthesis_downgrades_low_score_to_gray_zone_when_registry_coverage_is_incomplete():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": [{"trackId": "7"}, {"trackId": "8"}]},
        "registry": {"ok": True, "registryItems": [{"registryId": "r1"}, {"registryId": "r2"}]},
        "match": {
            "ok": True,
            "matchMode": "image_to_image",
            "registryCoverageComplete": False,
            "registryCoverageRatio": 0.5,
            "scoredRegistryCount": 1,
            "totalRegistryCount": 2,
            "unscoredRegistryIds": ["r2"],
            "matches": [
                {"matchedTrackId": "7", "matchedRegistryId": "r1", "embeddingScore": 0.31, "scoreBand": "mismatch"},
                {"matchedTrackId": "8", "matchedRegistryId": "r1", "embeddingScore": 0.63, "scoreBand": "uncertain"},
            ],
        },
    }
    controller.tool_records = [{"tool": "matchImage", "ok": True, "skipped": False}]
    controller.tool_chain = ["getTrack", "getFrames", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-test", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []

    result = controller._synthesize("sufficient", "部分库对照完成")

    assert result["outOfRegistryTracks"] == []
    assert [item["trackId"] for item in result["uncertainTracks"]] == ["7", "8"]
    assert [item["registryOutState"] for item in result["uncertainTracks"]] == ["gray_zone", "gray_zone"]
    assert result["uncertainTracks"][0]["coverageLimited"] is True
    assert result["outOfRegistryCount"] == 0
    assert result["matchCount"] == 2
    assert result["uncertainty"] == "uncertain"


def test_registry_out_synthesis_orders_uncertain_by_lowest_best_score_and_hides_full_registry():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": [{"trackId": "1"}, {"trackId": "2"}, {"trackId": "3"}]},
        "registry": {"ok": True, "registryItems": [{"registryId": "r1"}]},
        "match": {
            "ok": True,
            "matchMode": "image_to_image",
            "registryCoverageComplete": False,
            "registryCoverageRatio": 0.5,
            "scoredRegistryCount": 1,
            "totalRegistryCount": 2,
            "unscoredRegistryIds": ["r2"],
            "matches": [
                {"matchedTrackId": "1", "embeddingScore": 0.709, "scoreBand": "uncertain"},
                {"matchedTrackId": "2", "embeddingScore": 0.596, "scoreBand": "uncertain"},
                {"matchedTrackId": "3", "embeddingScore": 0.655, "scoreBand": "uncertain"},
            ],
        },
    }
    controller.tool_records = [{"tool": "matchImage", "ok": True, "skipped": False}]
    controller.tool_chain = ["getTrack", "getFrames", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = {"displayId": "display-test", "mode": "lazy"}
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []

    result = controller._synthesize("sufficient", "全库对照完成")

    assert [item["trackId"] for item in result["uncertainTracks"]] == ["2", "3", "1"]
    assert [item["registryOutState"] for item in result["uncertainTracks"]] == ["gray_zone", "gray_zone", "gray_zone"]
    assert [item["embeddingScore"] for item in result["matches"]] == [0.596, 0.655, 0.709]
    assert result["registryCoverageComplete"] is False
    assert result["matchCount"] == 3
    assert result["classificationMode"] == "registry_out"
    assert result["uncertainty"] == "uncertain"
    assert "registryItems" not in result

def test_registry_out_synthesis_does_not_show_full_registry_when_video_has_no_tracks():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "questionType": "registry_out_list",
    }
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": [], "trackIds": []},
        "registry": {
            "ok": True,
            "registryItems": [{"registryId": "r1", "hullNumber": "0857"}],
        },
        "match": {
            "ok": True,
            "matches": [],
            "error": "argument_missing:galleryImages",
            "visualAttempted": True,
        },
    }
    controller.tool_records = [
        {
            "tool": "getTrack",
            "ok": True,
            "skipped": False,
            "result": {"ok": True, "tracks": [], "trackIds": []},
            "summary": {"trackCount": 0},
            "round": 1,
        },
        {"tool": "listRegistry", "ok": True, "skipped": False, "round": 2},
        {
            "tool": "matchImage",
            "ok": True,
            "skipped": False,
            "error": "argument_missing:galleryImages",
            "result": {"error": "argument_missing:galleryImages", "matches": []},
            "round": 2,
        },
    ]
    controller.tool_chain = ["getTrack", "listRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = None
    controller.display_groups = []
    controller.session_id = "session-zero"
    controller.question = "有哪些未在库船出现在视频中？"
    controller.event_handler = None
    controller._pending_registry_items = []
    controller.tools = type("T", (), {})()

    result = controller._synthesize("uncertain", "旧链路错误地进入了全库匹配")

    assert result["uncertainty"] == "sufficient"
    assert result["outOfRegistryCount"] == 0
    assert result["tracks"] == []
    assert "registryItems" not in result
    assert result["display"]["registryReferenceCount"] == 0
    assert "未检测到船舶轨迹" in result["conclusion"]


def test_reflect_stages_registry_out_query_without_repeating_completed_calls():
    fields = _out_fields()

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                payload = {"ok": True, "handoff": "plan", "intent": fields, "note": ""}
                messages = [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "handoff_to_plan",
                            "args": {"intent": fields, "note": ""},
                            "id": "intent-handoff",
                            "type": "tool_call",
                        }],
                    ),
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False),
                        tool_call_id="intent-handoff",
                        name="handoff_to_plan",
                    ),
                ]
                yield "values", {"messages": messages}
                return
            raise RuntimeError(f"{self.name} 模拟未移交")

        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} 模拟未移交")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            if name == "getTrack":
                tracks = [{"trackId": "1"}, {"trackId": "2"}, {"trackId": "3"}]
                return {
                    "ok": True,
                    "trackIds": ["1", "2", "3"],
                    "tracks": tracks,
                    "returnedTrackCount": 3,
                    "totalTrackCount": 3,
                }
            if name == "getFrames":
                frames = [
                    {"trackId": "1", "keyframeId": "frame-1"},
                    {"trackId": "2", "keyframeId": "frame-2"},
                    {"trackId": "3", "keyframeId": "frame-3"},
                ]
                return {
                    "ok": True,
                    "keyframes": frames,
                    "keyframeIds": ["frame-1", "frame-2", "frame-3"],
                    "keyframesByTrack": {},
                }
            if name == "listRegistry":
                return {
                    "ok": True,
                    "registryItems": [{
                        "registryId": "registry-1",
                        "hullNumber": "0857",
                        "references": [{"referenceId": "ref-1", "registryId": "registry-1"}],
                    }],
                    "registryReferences": [{"referenceId": "ref-1", "registryId": "registry-1"}],
                }
            if name == "matchImage":
                return {
                    "ok": True,
                    "visualAttempted": True,
                    "scoredPairCount": 3,
                    "matches": [
                        {"matchedTrackId": "1", "matchedRegistryId": "registry-1", "embeddingScore": 0.91, "scoreBand": "match"},
                        {"matchedTrackId": "2", "matchedRegistryId": "registry-1", "embeddingScore": 0.31, "scoreBand": "mismatch"},
                        {"matchedTrackId": "3", "matchedRegistryId": "registry-1", "embeddingScore": 0.29, "scoreBand": "mismatch"},
                    ],
                }
            raise AssertionError(name)

    def _fake_create_agent(model, tools, system_prompt, name):
        handoff_tools = [tool for tool in tools if str(getattr(tool, "name", "")).startswith("handoff")]
        assert handoff_tools
        assert all(getattr(tool, "return_direct", False) for tool in handoff_tools)
        if name in {"plan", "observe", "reflect"}:
            assert any(str(getattr(tool, "name", "")).startswith("load_") for tool in tools)
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        state = run_sea_agent(
            "有哪些未在库船出现在视频中？",
            object(),
            _FakeTools(),
            max_rounds=5,
            query_top_k=1,
            broad_match_top_k=0,
        )

    assert state["loop_count"] == 3
    assert len(state["rounds"]) == 3
    assert state["final_state"] == "sufficient"
    assert state["tool_chain"] == [
        "getTrack", "getFrames", "listRegistry", "matchImage"
    ]
    business_records = [
        record for record in state["tool_records"]
        if record.get("tool") in {"getTrack", "getFrames", "listRegistry", "matchImage"}
    ]
    assert [(record["tool"], record["round"]) for record in business_records] == [
        ("getTrack", 1), ("getFrames", 1), ("listRegistry", 2), ("matchImage", 3)
    ]
    assert state["reflection"]["acceptanceProgress"]["acceptanceSatisfied"] is True


def test_reflect_finishes_registry_out_query_in_first_round_when_video_has_no_tracks():
    fields = _out_fields()
    executed_tools = []

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                payload = {"ok": True, "handoff": "plan", "intent": fields, "note": ""}
                yield "values", {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "handoff_to_plan",
                                "args": {"intent": fields, "note": ""},
                                "id": "intent-zero",
                                "type": "tool_call",
                            }],
                        ),
                        ToolMessage(
                            content=json.dumps(payload, ensure_ascii=False),
                            tool_call_id="intent-zero",
                            name="handoff_to_plan",
                        ),
                    ]
                }
                return
            raise RuntimeError(f"{self.name} 模拟未移交")

        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} 模拟未移交")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            executed_tools.append(name)
            if name == "getTrack":
                return {
                    "ok": True,
                    "trackIds": [],
                    "tracks": [],
                    "returnedTrackCount": 0,
                    "totalTrackCount": 0,
                }
            raise AssertionError(f"零轨迹后不应执行 {name}")

    def _fake_create_agent(model, tools, system_prompt, name):
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        state = run_sea_agent(
            "有哪些未在库船出现在视频中？",
            object(),
            _FakeTools(),
            max_rounds=5,
            query_top_k=1,
            broad_match_top_k=0,
        )

    assert state["loop_count"] == 1
    assert state["final_state"] == "sufficient"
    assert executed_tools == ["getTrack"]
    assert state["tool_chain"] == ["getTrack"]
    assert state["reflection"]["acceptanceProgress"]["videoEmptyShortCircuit"] is True
    assert "无需查询整库" in state["final_reason"] or "没有未在库船舶" in state["final_reason"]



def test_hull_existence_acceptance_guard_uses_three_non_repeating_rounds():
    clear_target_parser_cache()
    fields = infer_intent_fields("大鱼01 有没有在视频中出现？")
    executed = []

    def _handoff_messages(tool_name, args, payload, call_id):
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

    class _FakeAgent:
        def __init__(self, name):
            self.name = name

        def stream(self, *args, **kwargs):
            if self.name == "intent":
                payload = {"ok": True, "handoff": "plan", "intent": fields, "note": ""}
                yield "values", {"messages": _handoff_messages(
                    "handoff_to_plan", {"intent": fields, "note": ""}, payload, "intent-hull"
                )}
                return
            if self.name == "plan":
                calls = [{
                    "id": "exact-track",
                    "tool": "getTrack",
                    "arguments": {"hullNumber": "大鱼01", "offset": 0, "limit": 60},
                }]
                payload = {
                    "ok": True,
                    "handoff": "observe",
                    "goal": "先做舷号精确查询",
                    "calls": calls,
                    "planHint": "先做低成本精确查询",
                    "reason": "分阶段核验",
                }
                yield "values", {"messages": _handoff_messages(
                    "handoff_to_observe",
                    {
                        "goal": payload["goal"],
                        "calls": calls,
                        "planHint": payload["planHint"],
                        "reason": payload["reason"],
                    },
                    payload,
                    "plan-hull",
                )}
                return
            raise RuntimeError(f"{self.name} 模拟未移交")

        def invoke(self, *args, **kwargs):
            raise RuntimeError(f"{self.name} 模拟未移交")

    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            executed.append((name, dict(arguments)))
            if name == "getTrack" and arguments.get("hullNumber"):
                return {
                    "ok": True, "trackIds": [], "tracks": [],
                    "returnedTrackCount": 0, "totalTrackCount": 0,
                }
            if name == "getRegistry":
                return {
                    "ok": True,
                    "registryItems": [{
                        "registryId": "registry-fish-01",
                        "hullNumber": "大鱼01",
                        "references": [{"referenceId": "reference-fish-01"}],
                    }],
                    "registryReferences": [{"referenceId": "reference-fish-01"}],
                }
            if name == "getTrack":
                return {
                    "ok": True,
                    "trackIds": [1, 2],
                    "tracks": [{"trackId": 1}, {"trackId": 2}],
                    "returnedTrackCount": 2,
                    "totalTrackCount": 2,
                }
            if name == "getFrames":
                return {
                    "ok": True,
                    "keyframes": [
                        {"trackId": 1, "keyframeId": "frame-1"},
                        {"trackId": 2, "keyframeId": "frame-2"},
                    ],
                    "keyframeIds": ["frame-1", "frame-2"],
                }
            if name == "matchImage":
                return {
                    "ok": True,
                    "visualAttempted": True,
                    "scoredPairCount": 2,
                    "matches": [{
                        "matchedTrackId": 1,
                        "matchedRegistryId": "registry-fish-01",
                        "embeddingScore": 0.91,
                        "scoreBand": "match",
                    }],
                }
            raise AssertionError(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=lambda model, tools, system_prompt, name: _FakeAgent(name)
    ):
        state = run_sea_agent(
            "大鱼01 有没有在视频中出现？",
            object(),
            _FakeTools(),
            max_rounds=5,
            query_top_k=5,
            broad_match_top_k=0,
        )

    assert state["loop_count"] == 3
    assert state["final_state"] == "sufficient"
    assert [name for name, _ in executed] == [
        "getTrack", "getRegistry", "getTrack", "getFrames", "matchImage"
    ]
    assert executed[0][1]["hullNumber"] == "大鱼01"
    assert "hullNumber" not in executed[2][1]
    assert executed[2][1]["limit"] == 0
    # focused 证据模式：matchImage 用普通 topK（query_top_k=5），不再全量评分
    assert executed[4][1]["topK"] == 5
    assert sum(1 for name, _ in executed if name == "matchImage") == 1

def test_hull_existence_synthesis_returns_all_confirmed_and_gray_zone_tracks_with_thresholds():
    controller = AgentController.__new__(AgentController)
    controller.meta = {
        "operation": "existence",
        "targetScope": "both",
        "targetKind": "hull",
        "hullNumber": "003",
        "questionType": "hull_existence",
    }
    tracks = [
        {
            "trackId": track_id,
            "startTime": index * 10.0,
            "endTime": index * 10.0 + 8.0,
            "keyframeIds": [f"frame-{track_id}"],
            "shipSegmentIds": [f"segment-{track_id}"],
        }
        for index, track_id in enumerate(("10", "13", "11", "15", "17"))
    ]
    matches = [
        {
            "matchedTrackId": track_id,
            "matchedRegistryId": "registry-003",
            "embeddingScore": score,
            "scoreBand": band,
            "matchedKeyframeIds": [f"frame-{track_id}"],
            "matchedRegistryReferenceIds": ["reference-003"],
        }
        for track_id, score, band in (
            ("10", 0.838, "match"),
            ("13", 0.836, "match"),
            ("11", 0.760, "match"),
            ("15", 0.731, "match"),
            ("17", 0.650, "uncertain"),
        )
    ]
    controller.working_scope = {
        "tracks": {"ok": True, "tracks": tracks},
        "registry": {
            "ok": True,
            "registryItems": [{
                "registryId": "registry-003",
                "hullNumber": "003",
                "references": [{"referenceId": "reference-003"}],
            }],
        },
        "match": {
            "ok": True,
            "matchMode": "image_to_image",
            "matches": matches,
            "matchThresholds": {
                "mode": "image",
                "confirmation": 0.72,
                "exclusion": 0.52,
                "grayZone": {"lower": 0.52, "upper": 0.72},
            },
        },
    }
    controller.tool_records = [{"tool": "matchImage", "ok": True, "skipped": False}]
    controller.tool_chain = ["getTrack", "getFrames", "getRegistry", "matchImage"]
    controller.rounds = []
    controller.display_limit = 3
    controller.display_record = None
    controller.display_groups = []
    controller.session_id = "session-test"
    controller.question = "003有没有在视频中出现？"
    controller.event_handler = None
    controller._pending_registry_items = []
    controller.tools = SimpleNamespace()

    result = controller._synthesize("sufficient", "视觉核验完成")

    assert result["conclusion"] == "确认舷号 003 在视频中出现"
    assert [item["trackId"] for item in result["confirmedTracks"]] == ["10", "13", "11", "15"]
    assert [item["trackId"] for item in result["uncertainTracks"]] == ["17"]
    assert [item["trackId"] for item in result["tracks"]] == ["10", "13", "11", "15"]
    assert result["matchCount"] == 5
    assert result["confirmedMatchCount"] == 4
    assert result["uncertainMatchCount"] == 1
    assert result["classificationKind"] == "track"
    assert result["matchThresholds"]["confirmation"] == 0.72
    assert "确认阈值为 0.720" in result["answerText"]
    assert "灰区为 0.520 到 0.720" in result["answerText"]
    assert [group["trackId"] for group in result["displayGroups"]] == ["10", "13", "11", "15", "17"]
    assert [group["scoreBand"] for group in result["displayGroups"]] == ["match", "match", "match", "match", "uncertain"]

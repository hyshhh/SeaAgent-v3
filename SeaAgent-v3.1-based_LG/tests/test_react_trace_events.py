"""ReAct 微循环实时事件（角色节点内工具往返）的回归测试。"""
from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import run_sea_agent


def _out_fields() -> dict:
    return {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "out",
        "hullNumber": None,
        "description": None,
        "questionType": "registry_out_list",
        "expectedOutcome": "返回未在库轨迹",
        "successCriteria": "完成全库对照",
        "nextAgentFocus": "getTrack → getFrames → listRegistry → matchImage",
    }


class _FakeTools:
    @staticmethod
    def execute(name, arguments):
        if name == "getTrack":
            return {"ok": True, "trackIds": [], "tracks": []}
        raise AssertionError(name)


class _FakeAgent:
    """intent 节点模拟 ReAct：先调 parseTime，再 submit_intent。"""

    def __init__(self, name):
        self.name = name

    def stream(self, *args, **kwargs):
        if self.name == "intent":
            fields = _out_fields()
            messages = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "parseTime",
                            "args": {"expression": "昨天下午"},
                            "id": "intent-parse-time",
                            "type": "tool_call",
                        },
                        {
                            "name": "submit_intent",
                            "args": {"intent": fields, "note": ""},
                            "id": "intent-handoff",
                            "type": "tool_call",
                        },
                    ],
                ),
                ToolMessage(
                    content=json.dumps(
                        {"ok": True, "timeRange": [1000, 2000], "expression": "昨天下午"},
                        ensure_ascii=False,
                    ),
                    tool_call_id="intent-parse-time",
                    name="parseTime",
                ),
                ToolMessage(
                    content=json.dumps({"ok": True, "handoff": "plan", "intent": fields, "note": ""}, ensure_ascii=False),
                    tool_call_id="intent-handoff",
                    name="submit_intent",
                ),
            ]
            yield "values", {"messages": messages}
            return
        raise RuntimeError(f"{self.name} 模拟未移交")

    def invoke(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} 模拟未移交")


def test_react_tool_roundtrip_events_are_emitted():
    events = []

    def _fake_create_agent(model, tools, system_prompt, name):
        return _FakeAgent(name)

    with patch("agent.graph.build_chat_model", return_value=object()), patch(
        "agent.graph.create_agent", side_effect=_fake_create_agent
    ):
        run_sea_agent(
            "昨天下午有哪些未在库船？",
            object(),
            _FakeTools(),
            max_rounds=1,
            query_top_k=1,
            broad_match_top_k=0,
            event_handler=events.append,
        )

    tool_events = [
        event for event in events
        if event.get("type") == "agent_tool" and event.get("role") == "intent"
    ]
    parse_events = [event for event in tool_events if event.get("tool") == "parseTime"]
    assert parse_events, f"未收到 intent 的 parseTime 工具事件: {[e.get('tool') for e in tool_events]}"
    phases = [(event.get("phase"), event.get("ok")) for event in parse_events]
    assert ("running", True) in phases, phases
    assert ("completed", True) in phases, phases
    # handoff 移交不展示为工具事件
    assert not any(event.get("tool") == "submit_intent" for event in tool_events)

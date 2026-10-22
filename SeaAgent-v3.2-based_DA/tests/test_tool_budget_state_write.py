import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import ToolMessage

from config import load_config
from harness.middleware import build_middleware
from harness.runtime import SeaVideoHarness
from harness.shared_state import SharedStateStore, SharedTaskState
from harness.state_write import StateWriteMiddleware
from harness.tool_guard import AgentCancelled, RepeatToolCallMiddleware
from harness.trace import _Trace


def _runtime(tmp_path, settings=None):
    instance = SeaVideoHarness.__new__(SeaVideoHarness)
    instance.config = {"harness": dict(settings or {})}
    instance.state_store = SharedStateStore(tmp_path / "state.sqlite")
    instance.shared_state = instance.state_store.load("t", "verify target")
    instance.shared_state.update(
        "planner", {"requirements": ["verify target"], "plan": ["scan", "verify"]}
    )
    instance.shared_state.runtime_update({"iteration": 1, "pending_agent": "executor"})
    instance._cancel = None
    instance._connection = None
    instance.agents = {}
    instance.agent_specs = {"executor": {"task_prompt": "当前问题：{question}"}}
    return instance


def _trace(instance):
    return _Trace(None, 4000, shared_state=instance.shared_state)


class _Request:
    def __init__(self, tools, messages=None):
        self.tools = tools
        self.messages = messages or []

    def override(self, **updates):
        return _Request(
            updates.get("tools", self.tools), updates.get("messages", self.messages)
        )


def _tool(name):
    return SimpleNamespace(name=name)


def test_budget_settings_have_no_summary_request():
    settings = load_config()["harness"]
    assert settings["tool_timeout_seconds"] == 60
    assert "convergence_timeout_seconds" not in settings
    assert "tool_calls_per_run" not in settings
    assert "subagent_tool_calls" not in settings


def test_new_turn_resets_shared_domain_tool_budget(tmp_path):
    state = SharedTaskState("old", {"agent_calls": {"executor": 40}})
    guard = RepeatToolCallMiddleware(state=state, scope="executor", tool_budget=40)
    assert state.limit_reason("executor", 40) == "agent_tool_limit"

    state.start_turn("next")
    request = SimpleNamespace(tool_call={"name": "get_track", "args": {}, "id": "c1"})
    response = guard.wrap_tool_call(
        request, lambda req: ToolMessage(content="ok", tool_call_id=req.tool_call["id"])
    )

    assert response.content == "ok"
    assert state.snapshot()["agent_calls"]["executor"] == 1
    assert state.limit_reason("executor", 40) == ""


def test_budget_limit_blocks_domain_tools_but_keeps_state_tools():
    state = SharedTaskState("q", {"agent_calls": {"executor": 1}})
    guard = RepeatToolCallMiddleware(
        state=state,
        scope="executor",
        tool_budget=1,
        allowed_tools={"get_track", "read_shared_state", "update_execution_state"},
    )
    request = _Request(
        [_tool("get_track"), _tool("read_shared_state"), _tool("update_execution_state")]
    )

    response = guard.wrap_model_call(request, lambda req: req)

    assert {tool.name for tool in response.tools} == {
        "read_shared_state",
        "update_execution_state",
    }


def test_domain_tool_budget_is_atomic_and_status_tools_are_free():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(
        state=state,
        scope="executor",
        tool_budget=2,
    )
    executed = []

    for name in ("get_track", "get_frames", "get_clip", "read_shared_state", "update_execution_state", "show_evidence"):
        request = SimpleNamespace(
            tool_call={"name": name, "args": {"index": len(executed)}, "id": name}
        )
        result = guard.wrap_tool_call(
            request,
            lambda req: (
                executed.append(req.tool_call["name"])
                or ToolMessage(content='{"ok":true}', tool_call_id=req.tool_call["id"])
            ),
        )
        if name == "get_clip":
            assert result.artifact["budget_exhausted"] == "agent_tool_limit"

    snapshot = state.snapshot()
    assert executed == [
        "get_track",
        "get_frames",
        "read_shared_state",
        "update_execution_state",
        "show_evidence",
    ]
    assert snapshot["agent_calls"]["executor"] == 2


def test_agent_stream_continues_after_budget_and_keeps_last_result(tmp_path):
    instance = _runtime(tmp_path)
    executed = []
    guard = RepeatToolCallMiddleware(
        state=instance.shared_state, scope="executor", tool_budget=6
    )

    class Graph:
        def stream(self, *args, **kwargs):
            for index in range(10):
                request = SimpleNamespace(
                    tool_call={"name": "get_frames", "args": {"index": index}, "id": str(index)}
                )

                def handler(value, index=index):
                    executed.append(index)
                    return ToolMessage(
                        content=json.dumps({"keyframeId": f"kf{index}"}),
                        name="get_frames",
                        tool_call_id=str(index),
                    )

                yield {"tools": {"messages": [guard.wrap_tool_call(request, handler)]}}

    instance.agents["executor"] = Graph()
    outcome = instance._consume_driver(
        instance._run_agent_steps("executor", "q", "t", _trace(instance))
    )

    assert outcome == {"state": "completed"}
    assert executed == list(range(6))
    assert "kf5" in instance.shared_state.snapshot()["evidence_ids"]
    assert instance.shared_state.snapshot()["agent_calls"]["executor"] == 6


def test_state_write_middleware_forces_one_status_only_repair():
    state = SharedTaskState("q")
    state.start_agent("planner")
    middleware = StateWriteMiddleware(
        "planner",
        state,
        6,
        {"read_shared_state", "update_planner_state", "get_track"},
    )
    middleware.before_agent({})
    forced = middleware.after_agent({})
    assert forced["jump_to"] == "model"
    assert "状态更新工具" in forced["messages"][0].content

    request = _Request(
        [_tool("read_shared_state"), _tool("update_planner_state"), _tool("get_track")]
    )
    filtered = middleware._request(request)
    assert {tool.name for tool in filtered.tools} == {
        "read_shared_state",
        "update_planner_state",
    }

    state.update("planner", {"next_action": "scan"})
    unchanged = state.update("planner", {"next_action": "scan"})
    assert unchanged["ok"] is True and unchanged["unchanged"] is True
    filtered_after_write = middleware._request(request)
    assert {tool.name for tool in filtered_after_write.tools} == {
        "read_shared_state",
        "update_planner_state",
    }
    assert middleware.after_agent({}) is None


def test_state_write_middleware_enters_status_only_mode_after_budget_exhaustion():
    state = SharedTaskState("q", {"agent_calls": {"executor": 1}})
    middleware = StateWriteMiddleware(
        "executor",
        state,
        1,
        {"read_shared_state", "update_execution_state", "get_track"},
    )
    middleware.before_agent({})
    request = _Request(
        [_tool("read_shared_state"), _tool("update_execution_state"), _tool("get_track")]
    )

    filtered = middleware._request(request)

    assert {tool.name for tool in filtered.tools} == {
        "read_shared_state",
        "update_execution_state",
    }
    assert "状态更新工具" in filtered.messages[-1].content


def test_state_write_baseline_survives_checkpointed_resume(tmp_path):
    state = SharedTaskState("q")
    state.start_agent("planner")
    state.update("planner", {"next_action": "scan"})
    store = SharedStateStore(tmp_path / "state.sqlite")
    store.save("t", state)
    restored = store.load("t")
    middleware = StateWriteMiddleware("planner", restored, 6)

    middleware.before_agent({})

    assert not middleware._write_required()
    assert "agent_start_write_counts" not in restored.view()


def test_agent_budget_does_not_trigger_an_extra_model_summary(tmp_path):
    instance = _runtime(tmp_path)
    instance.shared_state.claim_call("executor", None, 1)
    summary_calls = []
    instance.model = SimpleNamespace(invoke=lambda *args, **kwargs: summary_calls.append(args))

    class Agent:
        def stream(self, *args, **kwargs):
            yield {"model": {"messages": []}}

    instance.agents["executor"] = Agent()
    outcome = instance._consume_driver(
        instance._run_agent_steps("executor", "q", "t", _trace(instance))
    )
    assert outcome["state"] == "completed"
    assert summary_calls == []


def test_iteration_limit_returns_stalled_without_summary(tmp_path):
    instance = _runtime(tmp_path, {"max_iterations": 1})

    def steps(role, *args):
        patch = (
            {"next_action": "scan"}
            if role == "planner"
            else {"execution_status": "running"}
            if role == "executor"
            else {"decision": "replan"}
        )
        instance.shared_state.update(role, patch)
        yield None
        return {"state": "completed"}

    instance._run_agent_steps = steps
    outcome = instance._consume_driver(instance._drive_multi_steps("q", "t", _trace(instance)))

    assert outcome["state"] == "stalled"
    assert "最大迭代次数" in outcome["reason"]


def test_cancel_still_stops_model_requests():
    guard = RepeatToolCallMiddleware(cancelled=lambda: True)
    with pytest.raises(AgentCancelled, match="cancelled"):
        guard.wrap_model_call(_Request([]), lambda request: pytest.fail("cancel ignored"))


def test_model_timeout_setting_remains_independent_of_tool_budget(tmp_path):
    instance = _runtime(tmp_path)
    build_middleware(
        {"harness": {"retry_delay_seconds": 0}},
        FakeListChatModel(responses=["ok"]),
        agent_spec={
            "name": "executor",
            "tool_budget": 6,
            "tools": [],
            "skills": [],
            "skill_reminder": {"max_per_run": 0, "message": "read skills"},
        },
        shared_state=instance.shared_state,
        model_timeout=instance._model_timeout_seconds,
    )
    assert instance._model_timeout_seconds() == 60


def test_restored_pending_agent_keeps_its_call_budget(tmp_path):
    instance = _runtime(tmp_path)
    instance.shared_state.runtime_update({"task_status": "running"})
    for _ in range(4):
        instance.shared_state.claim_call("executor", None, 6)

    def restored(role, *args):
        assert role == "executor"
        assert instance.shared_state.snapshot()["agent_calls"][role] == 4
        yield None
        return {"state": "cancelled"}

    instance._run_agent_steps = restored
    outcome = instance._consume_driver(instance._drive_multi_steps("q", "t", _trace(instance)))
    assert outcome["state"] == "cancelled"

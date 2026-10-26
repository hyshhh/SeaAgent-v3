import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import ToolMessage

from config import load_config
from harness.middleware import build_middleware
from harness.shared_state import SharedTaskState
from harness.state_write import (
    AgentCloseoutComplete,
    MissingStateWrite,
    StateWriteMiddleware,
)
from harness.tool_guard import AgentCancelled, RepeatToolCallMiddleware


class _Request:
    def __init__(self, tools, messages=None):
        self.tools = tools
        self.messages = messages or []
        self.model_settings = {}

    def override(self, **updates):
        request = _Request(
            updates.get("tools", self.tools), updates.get("messages", self.messages)
        )
        request.model_settings = updates.get("model_settings", self.model_settings)
        return request


def _tool(name):
    return SimpleNamespace(name=name)


def _call(name, args=None, call_id="call"):
    return SimpleNamespace(tool_call={"name": name, "args": args or {}, "id": call_id})


def _result(request):
    return ToolMessage(
        content=json.dumps({"ok": True, "tool": request.tool_call["name"]}),
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
    )


def test_config_has_no_total_tool_budget_and_a_two_minute_timeout():
    config = load_config()
    from tools.service import ToolService

    assert ToolService.STATISTICS_TIMEOUT_SECONDS == 120
    assert ToolService.IDENTIFICATION_TIMEOUT_SECONDS == 60
    assert ToolService.TOOL_TIMEOUTS["show_evidence"] == 120
    assert ToolService.TOOL_TIMEOUTS["verify_vessel_images"] == 60
    assert "tool_calls_per_run" not in config["harness"]
    assert "subagent_tool_calls" not in config["harness"]


def test_tool_calls_are_counted_without_a_total_call_limit():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    calls = []

    for index in range(20):
        request = _call("get_track", {"page": index}, str(index))
        result = guard.wrap_tool_call(request, lambda req: calls.append(req) or _result(req))
        assert result.status == "success"

    assert len(calls) == 20
    assert state.snapshot()["tool_calls"]["executor"] == 20
    assert not state.is_closeout_required("executor")


def test_second_and_third_exact_calls_replay_and_fourth_enters_closeout():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    calls = []
    handler = lambda request: calls.append(request) or _result(request)

    first = guard.wrap_tool_call(_call("get_track", {"page": 1}, "1"), handler)
    second = guard.wrap_tool_call(_call("get_track", {"page": 1}, "2"), handler)
    third = guard.wrap_tool_call(_call("get_track", {"page": 1}, "3"), handler)
    fourth = guard.wrap_tool_call(_call("get_track", {"page": 1}, "4"), handler)

    assert first.status == "success"
    assert json.loads(second.content)["cached_result"]["ok"] is True
    assert json.loads(third.content)["message"]
    assert fourth.artifact["closeout_required"] is True
    assert fourth.artifact["repeat_count"] == 4
    assert len(calls) == 1
    assert state.is_closeout_required("executor")

    emitted_call = guard.wrap_tool_call(
        _call("get_frames", {"track_id": 1}, "5"), handler
    )
    assert emitted_call.status == "success"
    assert len(calls) == 2


def test_closeout_narrows_tools_without_intercepting_emitted_tool_calls():
    state = SharedTaskState("q")
    state.start_agent("executor")
    tools = [
        _tool("read_shared_state"),
        _tool("update_execution_state"),
        _tool("read_file"),
        _tool("get_track"),
    ]
    request = _Request(tools)
    middleware = StateWriteMiddleware(
        "executor",
        state,
        model_request_budget=6,
        closeout_request_budget=2,
        allowed_tools={tool.name for tool in tools},
    )
    guard = RepeatToolCallMiddleware(state=state, scope="executor")

    assert {tool.name for tool in middleware._request(request).tools} == {
        "read_shared_state",
        "update_execution_state",
        "read_file",
        "get_track",
    }
    for index in range(4):
        guard.wrap_tool_call(
            _call("get_track", {"track_id": 7}, str(index)), _result
        )

    calls = []
    guard.wrap_tool_call(
        _call("get_frames", {"track_id": 7}, "remaining"),
        lambda current: calls.append(current) or _result(current),
    )
    assert len(calls) == 1
    closeout = guard.wrap_model_call(request, lambda current: middleware._request(current))
    assert {tool.name for tool in closeout.tools} == {
        "read_shared_state",
        "update_execution_state",
        "read_file",
    }


def test_repeat_threshold_switches_next_model_request_to_closeout_tools():
    state = SharedTaskState("q")
    state.start_agent("executor")
    tools = [
        _tool("read_shared_state"),
        _tool("update_execution_state"),
        _tool("read_file"),
        _tool("get_track"),
    ]
    request = _Request(tools)
    middleware = StateWriteMiddleware(
        "executor", state, 6, 2, {tool.name for tool in tools}
    )
    guard = RepeatToolCallMiddleware(state=state, scope="executor")

    for index in range(4):
        guard.wrap_tool_call(_call("get_track", {"track_id": 7}, str(index)), _result)

    closeout = guard.wrap_model_call(request, lambda current: middleware._request(current))
    assert state.is_closeout_required("executor")
    assert {tool.name for tool in closeout.tools} == {
        "read_shared_state",
        "update_execution_state",
        "read_file",
    }


def test_repeat_signature_normalizes_argument_order_and_is_scoped_by_role_and_iteration():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    calls = []
    handler = lambda request: calls.append(request) or _result(request)

    guard.wrap_tool_call(_call("get_track", {"a": 1, "b": 2}, "1"), handler)
    replay = guard.wrap_tool_call(_call("get_track", {"b": 2, "a": 1}, "2"), handler)
    assert replay.artifact["replayed"] is True
    assert len(calls) == 1

    state.runtime_update({"iteration": 1})
    guard.wrap_tool_call(_call("get_track", {"b": 2, "a": 1}, "3"), handler)
    assert len(calls) == 2


def test_shared_state_read_cache_is_invalidated_by_new_tool_results():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    reads = []

    def read(request):
        reads.append(request)
        return ToolMessage(
            content=json.dumps(state.view()),
            name="read_shared_state",
            tool_call_id=request.tool_call["id"],
        )

    guard.wrap_tool_call(_call("read_shared_state", call_id="r1"), read)
    replay = guard.wrap_tool_call(_call("read_shared_state", call_id="r2"), read)
    assert json.loads(replay.content)["message"]
    assert len(reads) == 1

    guard.wrap_tool_call(_call("get_track", {"page": 1}, "t1"), _result)
    refreshed = guard.wrap_tool_call(_call("read_shared_state", call_id="r3"), read)
    assert len(reads) == 2
    assert json.loads(refreshed.content)["tool_results"][0]["tool"] == "get_track"
    read_key = next(key for key in state.snapshot()["call_history"] if key.startswith("read_shared_state:"))
    assert state.snapshot()["call_history"][read_key]["repeat_count"] == 1


def test_model_budget_transitions_to_status_only_closeout_then_stalls_if_unwritten():
    state = SharedTaskState("q")
    state.start_agent("executor")
    middleware = StateWriteMiddleware(
        "executor",
        state,
        model_request_budget=2,
        closeout_request_budget=2,
        allowed_tools={"read_shared_state", "update_execution_state", "get_track"},
    )
    request = _Request(
        [_tool("read_shared_state"), _tool("update_execution_state"), _tool("get_track")]
    )

    assert middleware._request(request).tools == request.tools
    assert middleware._request(request).tools == request.tools
    closeout = middleware._request(request)
    assert {tool.name for tool in closeout.tools} == {
        "read_shared_state",
        "update_execution_state",
    }
    assert "update_execution_state" in closeout.messages[-1].content
    middleware._request(request)
    with pytest.raises(MissingStateWrite, match="executor 未提交"):
        middleware._request(request)


def test_written_state_completes_when_closeout_requests_are_exhausted():
    state = SharedTaskState("q")
    state.start_agent("planner")
    middleware = StateWriteMiddleware("planner", state, 1, 1)
    request = _Request([_tool("read_shared_state"), _tool("update_planner_state")])

    middleware._request(request)
    state.update("planner", {"next_action": "scan"})
    assert {tool.name for tool in middleware._request(request).tools} == {
        "read_shared_state",
        "update_planner_state",
    }
    with pytest.raises(AgentCloseoutComplete):
        middleware._request(request)


def test_agent_end_gets_one_state_write_repair():
    state = SharedTaskState("q")
    state.start_agent("planner")
    middleware = StateWriteMiddleware(
        "planner", state, 4, 2, {"read_shared_state", "update_planner_state", "get_track"}
    )

    repair = middleware.after_agent({})
    assert repair["jump_to"] == "model"
    assert "update_planner_state" in repair["messages"][0].content
    assert "clarification, next_action, plan, requirements" in repair["messages"][0].content
    assert "无需重复读取" in repair["messages"][0].content
    assert "其他角色字段" in repair["messages"][0].content
    filtered = middleware._request(
        _Request([_tool("read_shared_state"), _tool("update_planner_state"), _tool("get_track")])
    )
    assert {tool.name for tool in filtered.tools} == {
        "read_shared_state",
        "update_planner_state",
    }
    state.update("planner", {"next_action": "scan"})
    assert middleware.after_agent({}) is None


def test_a_second_missing_state_repair_is_rejected():
    state = SharedTaskState("q")
    state.start_agent("planner")
    middleware = StateWriteMiddleware("planner", state, 4, 2)
    assert middleware.after_agent({})["jump_to"] == "model"
    with pytest.raises(MissingStateWrite, match="一次收尾机会"):
        middleware.after_agent({})


def test_closeout_reminder_names_the_fields_owned_by_each_role():
    expected = {
        "planner": ("clarification, next_action, plan, requirements", "update_planner_state"),
        "executor": ("evidence_ids, execution_status, step_status", "update_execution_state"),
        "reflector": ("assessment, conflicts, decision, evidence_gaps", "update_reflection_state"),
    }
    for role, (fields, tool_name) in expected.items():
        state = SharedTaskState("q")
        state.start_agent(role)
        middleware = StateWriteMiddleware(role, state, 4, 2)

        reminder = middleware.after_agent({})["messages"][0].content
        assert fields in reminder
        assert tool_name in reminder


def test_counters_survive_restore_and_reset_for_a_new_phase(tmp_path):
    from harness.shared_state import SharedStateStore

    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load("t", "q")
    state.start_agent("executor")
    state.claim_model_request("executor", 6, 2)
    state.claim_tool_call("executor", "get_track:{}", "get_track")
    store.save("t", state)
    restored = store.load("t")

    assert restored.snapshot()["model_requests"]["executor"] == 1
    assert restored.snapshot()["tool_calls"]["executor"] == 1
    restored.start_agent("executor")
    assert restored.snapshot()["model_requests"]["executor"] == 0
    assert restored.snapshot()["tool_calls"]["executor"] == 0


def test_tool_guard_does_not_apply_a_global_tool_timeout():
    guard = RepeatToolCallMiddleware(scope="executor")
    assert not hasattr(guard, "tool_timeout")


def test_cancel_and_model_timeout_are_independent_of_tool_statistics():
    guard = RepeatToolCallMiddleware(cancelled=lambda: True)
    with pytest.raises(AgentCancelled, match="cancelled"):
        guard.wrap_model_call(_Request([]), lambda _request: pytest.fail("cancel ignored"))

    config = load_config()
    spec = {
        "name": "executor",
        "model_request_budget": 6,
        "closeout_request_budget": 2,
        "tools": [],
        "skills": [],
        "skill_reminder": {"max_per_run": 0, "message": "read skills"},
    }
    stack = build_middleware(
        config, FakeListChatModel(responses=["ok"]), agent_spec=spec, shared_state=SharedTaskState("q")
    )
    assert any(isinstance(item, RepeatToolCallMiddleware) for item in stack)
    assert "tool_budget" not in spec

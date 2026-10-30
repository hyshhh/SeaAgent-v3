"""共享状态的回归测试：权限、持久化、工具幂等和确认恢复。"""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from harness.middleware import build_middleware
from harness.runtime import SeaVideoHarness, _Trace
from harness.shared_state import SharedStateStore, SharedTaskState, StateConflictError
from harness.state_tools import build_state_tools
from harness.tool_guard import RepeatToolCallMiddleware


def request(name, args=None, call_id="c1"):
    return SimpleNamespace(tool_call={"name": name, "args": args or {}, "id": call_id})


def test_plan_progress_has_separate_write_ownership():
    state = SharedTaskState("q")
    state.update("planner", {"plan": [{"id": "scan", "content": "scan tracks"}]})
    state.update("executor", {"step_status": {"scan": "completed"}})
    with pytest.raises(ValueError, match="cannot update"):
        state.update("executor", {"plan": []})
    state.update(
        "planner",
        {"plan": [{"id": "scan", "content": "scan tracks"}, {"id": "verify", "content": "verify"}]},
    )
    assert state.snapshot()["step_status"] == {"scan": "completed", "verify": "pending"}
    state.update("planner", {"plan": [{"id": "scan", "content": "scan a different time"}]})
    assert state.snapshot()["step_status"] == {"scan": "pending"}


@pytest.mark.parametrize(
    "patch",
    [
        {"plan": [{"id": "same", "content": "a"}, {"id": "same", "content": "b"}]},
        {"plan": [{"content": ""}]},
        {"requirements": "wrong"},
        {"plan": [None]},
    ],
)
def test_bad_plans_do_not_partially_modify_state(patch):
    state = SharedTaskState("q")
    before = state.snapshot()
    with pytest.raises((ValueError, TypeError)):
        state.update("planner", patch)
    assert state.snapshot() == before


def test_model_cannot_overwrite_tool_results_or_remove_evidence():
    state = SharedTaskState("q")
    state.record_tool_result("executor", "get_frames", "c1", {"keyframeId": "kf1"}, "completed")
    with pytest.raises(ValueError, match="evidence_ids"):
        state.update("executor", {"evidence_ids": []})
    assert state.snapshot()["evidence_ids"] == ["kf1"]
    with pytest.raises(ValueError):
        state.update("executor", {"tool_results": []})


def test_plan_events_and_answer_only_come_from_shared_state():
    events = []
    state = SharedTaskState("q")
    state.update("planner", {"plan": ["scan"]})
    trace = _Trace(events.append, 4000, shared_state=state)
    trace.consume(
        {
            "todos": [{"content": "wrong", "status": "completed"}],
            "messages": [AIMessage(content='{"decision":"terminate"}')],
        },
        agent_name="reflector",
    )
    assert trace.result("t", {})["answer"] == ""
    state.update("executor", {"step_status": {"step-1": "completed"}})
    trace.consume_plan()
    assert events[-1]["steps"][0]["content"] == "scan"
    assert events[-1]["completed"] == 1
    assert state.snapshot()["decision"] is None
    trace.consume_plan()
    assert len([event for event in events if event["type"] == "plan"]) == 2


def test_state_tools_can_read_again_without_a_model_version_argument():
    state = SharedTaskState("q")
    tools = {tool.name: tool for tool in build_state_tools(state)}
    guard = RepeatToolCallMiddleware(state=state, scope="planner")

    def handler(req):
        value = tools[req.tool_call["name"]].invoke(req.tool_call["args"])
        return ToolMessage(
            content=json.dumps(value), name=req.tool_call["name"], tool_call_id=req.tool_call["id"]
        )

    read = request("read_shared_state")
    first = json.loads(guard.wrap_tool_call(read, handler).content)
    updated = guard.wrap_tool_call(
        request(
            "update_planner_state",
            {"patch": {"requirements": ["r"]}},
            "c2",
        ),
        handler,
    )
    assert json.loads(updated.content)["ok"]
    second = json.loads(guard.wrap_tool_call(read, handler).content)
    assert second["requirements"] == ["r"]
    assert second["version"] > first["version"]
    assert "expected_version" not in tools[
        "update_planner_state"
    ].args_schema.model_json_schema()["properties"]


def test_executor_accepts_json_string_patch_and_ignores_runtime_collected_evidence_ids():
    state = SharedTaskState("q")
    state.update("planner", {"plan": [{"id": "step-1", "content": "count"}]})
    tool = {
        item.name: item for item in build_state_tools(state)
    }["update_execution_state"]

    result = tool.invoke(
        {
            "patch": json.dumps(
                {
                    "evidence_ids": ["track-1", "keyframe-1"],
                    "execution_status": "success",
                    "step_status": {"step-1": "completed"},
                }
            )
        }
    )

    assert result["ok"] is True
    assert result["ignored_fields"] == ["evidence_ids"]
    assert state.snapshot()["execution_status"] == "success"
    assert state.snapshot()["step_status"] == {"step-1": "completed"}
    assert state.snapshot()["evidence_ids"] == []


def test_state_tool_schema_filters_full_snapshot_without_cross_role_writes():
    state = SharedTaskState("q")
    tools = {tool.name: tool for tool in build_state_tools(state)}
    schema = tools["update_planner_state"].args_schema.model_json_schema()
    patch_ref = schema["properties"]["patch"]["$ref"].rsplit("/", 1)[-1]
    patch_schema = schema["$defs"][patch_ref]
    assert patch_schema["additionalProperties"] is False
    assert set(patch_schema["properties"]) == {
        "requirements", "plan", "next_action", "clarification"
    }
    plan_schema = next(
        item for item in patch_schema["properties"]["plan"]["anyOf"] if item.get("type") == "array"
    )
    step_ref = next(item["$ref"] for item in plan_schema["items"]["anyOf"] if "$ref" in item)
    step_schema = schema["$defs"][step_ref.rsplit("/", 1)[-1]]
    assert step_schema["properties"]["id"]["type"] == "string"
    assert "id" in step_schema["required"]
    evidence_schema = step_schema["properties"]["required_evidence"]
    assert evidence_schema["type"] == "array"
    assert evidence_schema["items"]["type"] == "string"

    snapshot = state.view()
    patch = {
        **snapshot,
        "requirements": ["verify target"],
        "plan": [
            {
                "id": "scan",
                "content": "scan tracks",
                "required_evidence": "track result",
            }
        ],
        "next_action": "execute",
        "task_status": "completed",
    }
    result = tools["update_planner_state"].invoke({"patch": patch})

    updated = state.snapshot()
    assert result["ok"] is True
    assert {"task_status", "tool_results", "version"} <= set(result["ignored_fields"])
    assert updated["requirements"] == ["verify target"]
    assert updated["plan"][0]["required_evidence"] == ["track result"]
    assert updated["task_status"] == "running"
    assert updated["tool_results"] == []

    rejected = tools["update_planner_state"].invoke(
        {"patch": {"task_status": "completed", "version": 100}}
    )
    assert rejected["ok"] is False
    assert rejected["error"] == "no_owned_fields"
    assert rejected["ignored_fields"] == ["task_status", "version"]
    assert "clarification, next_action, plan, requirements" in rejected["message"]
    assert "删除其他角色字段后重试" in rejected["message"]
    assert state.snapshot() == updated


def test_reflection_assessment_text_becomes_a_final_answer_object():
    state = SharedTaskState("q")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_reflection_state"].invoke(
        {
            "patch": {
                "decision": "terminate",
                "assessment": "舷号 320 出现在轨迹记忆中，但未登记在先验库。",
            }
        }
    )

    assert result["ok"] is True
    assert state.snapshot()["assessment"] == {
        "final_answer": "舷号 320 出现在轨迹记忆中，但未登记在先验库。"
    }


def test_reflection_final_conclusion_is_promoted_to_final_answer():
    state = SharedTaskState("q")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_reflection_state"].invoke(
        {
            "patch": {
                "decision": "continue",
                "assessment": {
                    "direct_search": "3 个候选",
                    "final_conclusion": "2 个候选不匹配，1 个仍未解决。",
                },
            }
        }
    )

    assert result["ok"] is True
    assessment = state.snapshot()["assessment"]
    assert assessment["final_answer"] == "2 个候选不匹配，1 个仍未解决。"
    assert assessment["final_conclusion"] == "2 个候选不匹配，1 个仍未解决。"


def test_repeat_cache_is_scoped_to_the_agent_and_reload(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load("t", "q")
    guard = RepeatToolCallMiddleware(
        state=state, scope="executor", persist=lambda: store.save("t", state)
    )
    calls = []

    def handler(req):
        calls.append(req)
        return ToolMessage(
            content='{"trackId":"t1"}', name="get_track", tool_call_id=req.tool_call["id"]
        )

    guard.wrap_tool_call(request("get_track"), handler)
    restored = store.load("t")
    other = RepeatToolCallMiddleware(state=restored, scope="reflector")
    result = other.wrap_tool_call(request("get_track", call_id="c2"), handler)
    assert json.loads(result.content) == {"trackId": "t1"}
    assert len(calls) == 2
    assert restored.snapshot()["tool_calls"] == {"executor": 1, "reflector": 1}


def test_uncertain_write_is_never_repeated(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load("t", "q")
    guard = RepeatToolCallMiddleware(
        state=state, scope="executor", persist=lambda: store.save("t", state)
    )
    calls = []

    def handler(req):
        calls.append(req)
        raise ConnectionError("connection dropped after write")

    first = guard.wrap_tool_call(request("add_registry_vessel"), handler)
    assert first.status == "error"
    assert json.loads(first.content)["errorType"] == "ConnectionError"
    restored = RepeatToolCallMiddleware(state=store.load("t"), scope="executor")
    result = restored.wrap_tool_call(request("add_registry_vessel"), handler)
    assert result.status == "error" and "connection dropped" in result.content
    assert len(calls) == 1


def test_tool_statistics_reset_when_a_new_role_phase_starts():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    calls = []

    def handler(req):
        calls.append(req)
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"])

    for index in range(5):
        guard.wrap_tool_call(request("get_track", {"i": index}), handler)
    assert len(calls) == 5
    assert state.snapshot()["tool_calls"]["executor"] == 5
    state.start_agent("executor")
    state.runtime_update({"iteration": 1})
    for index in range(5, 10):
        guard.wrap_tool_call(request("get_track", {"i": index}), handler)
    assert len(calls) == 10
    assert state.snapshot()["tool_calls"]["executor"] == 5


def test_multiple_store_instances_preserve_independent_sessions(tmp_path):
    path = tmp_path / "state.sqlite"
    stores = [SharedStateStore(path), SharedStateStore(path)]

    def save(index):
        store = stores[index % 2]
        state = store.load(str(index), f"q{index}")
        store.save(str(index), state)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(20)))
    assert [stores[0].load(str(index)).snapshot()["question"] for index in range(20)] == [
        f"q{index}" for index in range(20)
    ]


def test_stale_disk_snapshot_is_rejected_even_if_only_counters_changed(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load("t", "q")
    store.save("t", state)
    old = store.load("t")
    state.claim_tool_call("executor", None, "get_track")
    store.save("t", state)
    with pytest.raises(StateConflictError):
        store.save("t", old)


def test_legacy_json_import_is_once_and_keeps_all_evidence(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "t": {
                    "question": "q",
                    "plan": ["scan"],
                    "evidence_ids": [f"kf{i}" for i in range(300)],
                }
            }
        ),
        encoding="utf-8",
    )
    store = SharedStateStore(path)
    assert len(store.load("t").snapshot()["evidence_ids"]) == 300
    assert store.load("t").snapshot()["plan"][0]["content"] == "scan"
    store.delete("t")
    assert SharedStateStore(path).load("t").snapshot()["question"] == ""
    assert path.exists()


def test_confirmation_replan_consumes_resume_once_and_keeps_iteration(tmp_path):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {"harness": {"max_iterations": 3}}
    runtime.state_store = SharedStateStore(tmp_path / "state.sqlite")
    runtime.shared_state = runtime.state_store.load("t", "q")
    runtime.shared_state.runtime_update(
        {"iteration": 2, "pending_agent": "reflector", "task_status": "awaiting_confirmation"}
    )
    roles, payloads = [], []

    def role_steps(role, question, thread_id, trace, cancel, payload):
        roles.append(role)
        payloads.append(payload)
        if role == "planner":
            runtime.shared_state.update(role, {"plan": ["scan"]})
        elif role == "executor":
            runtime.shared_state.update(role, {"execution_status": "success"})
        else:
            runtime.shared_state.update(
                role,
                {
                    "decision": "replan" if len(roles) == 1 else "terminate",
                    "assessment": {"final_answer": "done"},
                },
            )
        yield None
        return {"state": "completed"}

    runtime._run_agent_steps = role_steps
    outcome = runtime._consume_driver(
        runtime._drive_multi_steps("q", "t", _Trace(None, 4000), resume_payload="approval")
    )
    assert roles == ["reflector", "planner", "executor", "reflector"]
    assert payloads == ["approval", None, None, None]
    assert runtime.shared_state.snapshot()["iteration"] == 3
    assert outcome["state"] == "completed"


def test_tool_whitelist_is_enforced_even_if_framework_registers_other_tools():
    guard = RepeatToolCallMiddleware(allowed_tools={"read_shared_state"})
    result = guard.wrap_tool_call(
        request("execute", {"command": "danger"}),
        lambda req: pytest.fail("out-of-scope tool executed"),
    )
    assert result.status == "error"


def test_failed_results_never_become_evidence():
    state = SharedTaskState("q")
    state.record_tool_result("executor", "get_track", "c1", {"trackIds": ["invented"]}, "error")
    assert state.snapshot()["evidence_ids"] == []


def test_failed_tool_is_returned_once_and_replayed_without_retry():
    from types import SimpleNamespace

    from langchain_core.language_models import FakeListChatModel

    state = SharedTaskState("q")
    tool = SimpleNamespace(name="get_track")
    spec = {
        "name": "executor",
        "model_request_budget": 4,
        "closeout_request_budget": 2,
        "tools": [tool],
        "skills": [],
        "skill_reminder": {"max_per_run": 0, "message": "read skills"},
    }
    stack = build_middleware(
        {
            "harness": {"model_retry_attempts": 0, "retry_delay_seconds": 0},
        },
        FakeListChatModel(responses=["ok"]),
        agent_spec=spec,
        shared_state=state,
    )
    guard = next(item for item in stack if isinstance(item, RepeatToolCallMiddleware))
    calls = []

    def handler(req):
        calls.append(req)
        raise ConnectionError("offline")

    first = guard.wrap_tool_call(request("get_track"), handler)
    replay = guard.wrap_tool_call(request("get_track", call_id="c2"), handler)
    assert first.status == "error"
    assert replay.status == "error"
    assert "cached_error" in replay.content
    assert len(calls) == 1
    assert state.snapshot()["tool_calls"]["executor"] == 2


def test_failed_read_replays_its_error_instead_of_retrying():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(state=state)
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise ConnectionError("temporary")
        return ToolMessage(content='{"trackId":"t1"}', tool_call_id=req.tool_call["id"])

    first = guard.wrap_tool_call(request("get_track"), handler)
    replay = guard.wrap_tool_call(request("get_track", call_id="c2"), handler)
    assert first.status == replay.status == "error"
    assert len(calls) == 1


def test_tool_results_do_not_depend_on_removed_registry_write_invalidation():
    guard = RepeatToolCallMiddleware()
    calls = []

    def handler(req):
        calls.append(req.tool_call["name"])
        return ToolMessage(content='{"ok":true}', tool_call_id=req.tool_call["id"])

    guard.wrap_tool_call(request("recognize_tracks"), handler)
    guard.wrap_tool_call(request("recognize_tracks"), handler)
    assert calls == ["recognize_tracks"]


def test_truncated_cached_results_are_explicitly_marked():
    guard = RepeatToolCallMiddleware()
    handler = lambda req: ToolMessage(
        content=json.dumps({"tracks": list(range(10_000))}), tool_call_id=req.tool_call["id"]
    )
    guard.wrap_tool_call(request("get_track"), handler)
    cached = json.loads(guard.wrap_tool_call(request("get_track"), handler).content)
    assert cached["truncated"] is True
    assert len(cached["cached_result"]["tracks"]) == 100


def test_new_reflection_cannot_reuse_an_old_answer():
    state = SharedTaskState("q")
    state.update("reflector", {"decision": "terminate", "assessment": {"final_answer": "old"}})
    state.start_agent("reflector")
    trace = _Trace(None, 4000, "", shared_state=state)
    assert trace.result("t", {})["answer"] == ""
    assert state.snapshot()["decision"] is None


def test_concurrent_csv_instances_do_not_lose_updates(tmp_path):
    from memory.csv_store import CsvTable

    path = tmp_path / "table.csv"
    tables = [CsvTable(path, ["id", "value"]), CsvTable(path, ["id", "value"])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda index: tables[index % 2].upsert({"id": index, "value": index}, "id"),
                range(25),
            )
        )
    assert {int(row["id"]) for row in tables[0].rows()} == set(range(25))


def test_disk_conflict_cannot_be_reported_as_completion(tmp_path):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {}
    runtime.state_store = SharedStateStore(tmp_path / "state.sqlite")
    runtime.shared_state = runtime.state_store.load("t", "q")
    runtime.state_store.save("t", runtime.shared_state)
    newer = runtime.state_store.load("t")
    newer.update("planner", {"requirements": ["newer"]})
    runtime.state_store.save("t", newer)
    result = runtime._finish(
        _Trace(None, 4000, shared_state=runtime.shared_state), "t", {"state": "completed"}
    )
    assert result["state"] == "error"
    assert runtime.state_store.load("t").snapshot()["requirements"] == ["newer"]

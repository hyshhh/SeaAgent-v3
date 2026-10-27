import json
from pathlib import Path

from harness.shared_state import SharedStateStore, SharedTaskState
from harness.state_tools import build_state_tools


def test_each_agent_can_only_update_owned_fields():
    state = SharedTaskState("问题")
    planner = state.update("planner", {"requirements": ["r1"], "plan": ["p1"], "next_action": "execute"})
    assert planner["ok"] is True
    assert state.snapshot()["version"] == 2
    try:
        state.update("planner", {"decision": "terminate"})
    except ValueError as error:
        assert "decision" in str(error)
    else:
        raise AssertionError("planner modified reflector state")


def test_repeating_the_same_state_patch_is_idempotent_success():
    state = SharedTaskState("问题")
    patch = {"requirements": ["找到目标轨迹"]}
    state.update("planner", patch)
    version = state.snapshot()["version"]

    result = state.update("planner", patch)

    assert result["ok"] is True
    assert result["unchanged"] is True
    assert state.snapshot()["version"] == version

def test_tool_results_collect_evidence_ids():
    state = SharedTaskState("问题")
    state.record_tool_result("executor", "get_frames", "c1", {"keyframeIds": ["kf-1"], "trackIds": ["t-1"]}, "completed")
    snapshot = state.snapshot()
    assert snapshot["evidence_ids"] == ["kf-1", "t-1"]
    assert snapshot["tool_results"][0]["callId"] == "c1"


def test_invalid_evidence_id_is_ignored_without_polluting_state():
    state = SharedTaskState("问题")
    state.update("planner", {"plan": [{"id": "scan", "content": "扫描轨迹"}]})
    state.record_tool_result("executor", "get_track", "chatcmpl-tool-call", {"trackId": "t-1"}, "completed")
    tools = {tool.name: tool for tool in build_state_tools(state)}
    update_tool = tools["update_execution_state"]
    schema = update_tool.args_schema.model_json_schema()
    patch_ref = schema["properties"]["patch"]["$ref"].rsplit("/", 1)[-1]
    evidence_description = schema["$defs"][patch_ref]["properties"]["evidence_ids"]["description"]
    assert "tool_call_id" in update_tool.description
    assert "tool_call_id" in evidence_description

    result = update_tool.invoke(
        {
            "patch": {
                "evidence_ids": ["chatcmpl-tool-call"],
                "step_status": {"scan": "completed"},
            }
        }
    )

    assert result["ok"] is True
    assert result["ignored_evidence_ids"] == ["chatcmpl-tool-call"]
    assert "tool_call_id" in result["message"]
    assert state.snapshot()["step_status"] == {"scan": "completed"}
    assert state.snapshot()["evidence_ids"] == ["t-1"]


def test_only_invalid_evidence_ids_still_returns_a_repairable_error():
    state = SharedTaskState("问题")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_execution_state"].invoke(
        {"patch": {"evidence_ids": ["chatcmpl-tool-call"]}}
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_evidence_ids"
    assert result["ignored_evidence_ids"] == ["chatcmpl-tool-call"]


def test_executor_can_update_progress_without_repassing_runtime_evidence():
    state = SharedTaskState("问题")
    state.update("planner", {"plan": [{"id": "scan", "content": "扫描轨迹"}]})
    state.record_tool_result("executor", "get_track", "call-1", {"trackId": "t-1"}, "completed")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_execution_state"].invoke(
        {"patch": {"step_status": {"scan": "completed"}, "execution_status": "success"}}
    )

    assert result["ok"] is True
    assert state.snapshot()["evidence_ids"] == ["t-1"]
    assert state.snapshot()["step_status"] == {"scan": "completed"}


def test_numeric_plan_ids_are_normalized_before_execution_status_updates():
    state = SharedTaskState("问题")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    planned = tools["update_planner_state"].invoke(
        {"patch": {"plan": [{"id": 1, "content": "扫描轨迹"}]}}
    )
    updated = tools["update_execution_state"].invoke(
        {"patch": {"step_status": {"1": "completed"}}}
    )

    assert planned["ok"] is True
    assert state.snapshot()["plan"][0]["id"] == "1"
    assert updated["ok"] is True
    assert state.snapshot()["step_status"] == {"1": "completed"}


def test_reflection_update_accepts_json_patch_and_sibling_fields():
    state = SharedTaskState("问题")
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_reflection_state"].invoke(
        {
            "patch": json.dumps({"assessment": {"final_answer": "已完成"}}),
            "decision": "terminate",
        }
    )

    assert result["ok"] is True
    assert state.snapshot()["assessment"]["final_answer"] == "已完成"
    assert state.snapshot()["decision"] == "terminate"


def test_reflection_update_accepts_apostrophe_escaped_json_patch():
    state = SharedTaskState("问题")
    tools = {tool.name: tool for tool in build_state_tools(state)}
    patch = json.dumps({
        "assessment": {"confirmed": "未发现明确的大型军舰证据"},
        "evidence_gaps": ["未获得用户关于'大型军舰'的具体定义标准"],
        "conflicts": [],
        "decision": "continue",
    }, ensure_ascii=False).replace("'大型军舰'", "\\'大型军舰\\'")

    result = tools["update_reflection_state"].invoke({"patch": patch})

    assert result["ok"] is True
    assert state.snapshot()["evidence_gaps"] == ["未获得用户关于'大型军舰'的具体定义标准"]


def test_executor_maps_unambiguous_evidence_name_to_plan_id():
    state = SharedTaskState("问题")
    state.update(
        "planner",
        {
            "plan": [
                {
                    "id": "step-1",
                    "content": "比对先验库",
                    "required_evidence": ["match_registry_tracks"],
                }
            ]
        },
    )
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_execution_state"].invoke(
        {"patch": {"step_status": {"match_registry_tracks": "completed"}}}
    )

    assert result["ok"] is True
    assert result["mapped_step_ids"] == {"match_registry_tracks": "step-1"}
    assert state.snapshot()["step_status"] == {"step-1": "completed"}


def test_unknown_step_status_reports_current_plan_ids_without_partial_write():
    state = SharedTaskState("问题")
    state.update("planner", {"plan": [{"id": "step-1", "content": "搜索轨迹"}]})
    tools = {tool.name: tool for tool in build_state_tools(state)}

    result = tools["update_execution_state"].invoke(
        {
            "patch": {
                "execution_status": "running",
                "step_status": {"unknown_tool": "completed"},
            }
        }
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_state_patch"
    assert "Current IDs: step-1" in result["message"]
    assert state.snapshot()["execution_status"] == "idle"

def test_state_store_survives_reload(tmp_path):
    store = SharedStateStore(tmp_path / "state.json")
    state = SharedTaskState("问题")
    state.update("planner", {"requirements": ["r1"]})
    store.save("thread-1", state)
    loaded = store.load("thread-1")
    assert loaded.snapshot()["question"] == "问题"
    assert loaded.snapshot()["requirements"] == ["r1"]
    assert loaded.snapshot()["version"] == 2


def test_time_anchor_is_persisted_but_hidden_from_other_agents(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load(
        "thread-1", "昨天有几艘船？", time_anchor="2026-10-27T09:00:00+08:00"
    )
    store.save("thread-1", state)

    resumed = store.load("thread-1", "昨天有几艘船？", time_anchor="2026-10-28T09:00:00+08:00")
    assert resumed.snapshot()["time_anchor"] == "2026-10-27T09:00:00+08:00"
    assert "time_anchor" not in resumed.view()

    resumed.start_turn("今天有几艘船？", time_anchor="2026-10-28T09:00:00+08:00")
    assert resumed.snapshot()["time_anchor"] == "2026-10-28T09:00:00+08:00"


def test_reflector_writes_the_answer_through_shared_state():
    state = SharedTaskState("问题")
    state.update("reflector", {"decision": "terminate", "assessment": {"final_answer": "已确认"}})
    snapshot = state.snapshot()
    assert snapshot["decision"] == "terminate"
    assert snapshot["assessment"]["final_answer"] == "已确认"


def test_delete_cleans_only_the_session_external_results(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    first = store.load("thread-1")
    second = store.load("thread-2")
    first.record_tool_result("executor", "get_frames", "c1", {"data": "a" * 14000}, "completed")
    second.record_tool_result("executor", "get_frames", "c2", {"data": "b" * 14000}, "completed")
    first_ref = first.snapshot()["tool_results"][0]["result"]["external_ref"]
    second_ref = second.snapshot()["tool_results"][0]["result"]["external_ref"]
    store.save("thread-1", first)
    store.save("thread-2", second)

    store.delete("thread-1")

    assert not Path(first_ref).exists()
    assert Path(second_ref).exists()


def test_start_turn_cleans_previous_external_results(tmp_path):
    store = SharedStateStore(tmp_path / "state.sqlite")
    state = store.load("thread-1")
    state.record_tool_result("executor", "get_frames", "c1", {"data": "a" * 14000}, "completed")
    result_path = state.snapshot()["tool_results"][0]["result"]["external_ref"]

    state.start_turn("new question")

    assert not Path(result_path).exists()
    assert state.snapshot()["tool_results"] == []

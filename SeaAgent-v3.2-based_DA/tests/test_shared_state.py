from pathlib import Path

from harness.shared_state import SharedStateStore, SharedTaskState


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

def test_tool_results_collect_evidence_ids():
    state = SharedTaskState("问题")
    state.record_tool_result("executor", "get_frames", "c1", {"keyframeIds": ["kf-1"], "trackIds": ["t-1"]}, "completed")
    snapshot = state.snapshot()
    assert snapshot["evidence_ids"] == ["kf-1", "t-1"]
    assert snapshot["tool_results"][0]["callId"] == "c1"

def test_state_store_survives_reload(tmp_path):
    store = SharedStateStore(tmp_path / "state.json")
    state = SharedTaskState("问题")
    state.update("planner", {"requirements": ["r1"]})
    store.save("thread-1", state)
    loaded = store.load("thread-1")
    assert loaded.snapshot()["question"] == "问题"
    assert loaded.snapshot()["requirements"] == ["r1"]
    assert loaded.snapshot()["version"] == 2

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

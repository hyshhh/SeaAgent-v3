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

def test_stale_version_is_rejected():
    state = SharedTaskState("问题")
    state.update("planner", {"requirements": ["r1"]}, expected_version=1)
    result = state.update("executor", {"execution_status": "running"}, expected_version=1)
    assert result["ok"] is False
    assert result["error"] == "stale_state_version"

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

def test_json_text_can_recover_agent_state():
    state = SharedTaskState("问题")
    state.apply_agent_text("reflector", "{\"decision\": \"terminate\", \"final_answer\": \"已确认\"}")
    snapshot = state.snapshot()
    assert snapshot["decision"] == "terminate"
    assert snapshot["assessment"]["final_answer"] == "已确认"

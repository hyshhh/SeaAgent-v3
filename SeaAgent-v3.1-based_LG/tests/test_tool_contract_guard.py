from services.vlm_service import AgentLLMService
from agent.graph import _build_acceptance_progress, _find_tool_contract_failures, _prepare_plan_calls
from agent.plan_executor import PlanExecutor


class _SpyTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {"ok": True}


def _description_intent():
    return {
        "operation": "existence",
        "targetScope": "track_memory",
        "targetKind": "description",
        "registryRelation": "any",
        "description": "黄色的无人艇",
        "nextAgentFocus": "getTrack → getFrames → matchText(description=黄色的无人艇)",
    }


def test_invalid_get_track_description_is_repaired_to_description_chain():
    calls, repair = _prepare_plan_calls(
        [{"id": "bad", "tool": "getTrack", "arguments": {"description": "黄色的无人艇", "limit": 1}}],
        _description_intent(),
        1,
        broad_match_top_k=0,
    )

    assert repair == "argument_not_allowed:getTrack:description"
    assert [call["tool"] for call in calls] == ["getTrack", "getFrames", "matchText"]
    assert "description" not in calls[0]["arguments"]
    assert calls[2]["arguments"]["description"] == "黄色的无人艇"
    assert calls[2]["arguments"]["topK"] == 1


def test_executor_blocks_unknown_arguments_before_tool_service_call():
    tools = _SpyTools()
    executed = PlanExecutor(tools).execute([
        {"id": "bad", "tool": "getTrack", "arguments": {"description": "黄色的无人艇", "limit": 1}},
    ])

    assert tools.calls == []
    record = executed["tool_records"][0]
    assert record["skipped"] is True
    assert record["error"] == "argument_not_allowed:getTrack:description"


def test_contract_failure_detection_only_uses_current_round():
    records = [
        {
            "round": 1,
            "tool": "getTrack",
            "ok": False,
            "error": "ToolService.getTrack() got an unexpected keyword argument 'description'",
        },
        {
            "round": 2,
            "tool": "getTrack",
            "ok": True,
            "result": {"ok": True, "tracks": []},
        },
    ]

    assert _find_tool_contract_failures(records, 1) == [
        "getTrack: ToolService.getTrack() got an unexpected keyword argument 'description'"
    ]
    assert _find_tool_contract_failures(records, 2) == []


def test_failed_tools_do_not_satisfy_description_acceptance_progress():
    progress = _build_acceptance_progress(
        _description_intent(),
        set(),
        track_count=None,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    requirements = {item["key"]: item["completed"] for item in progress["requirements"]}
    assert requirements == {"tracks": False, "frames": False, "text_match": False}
    assert progress["acceptanceSatisfied"] is False


def test_vlm_strip_thinking_removes_orphan_closing_tag_prefix():
    assert AgentLLMService._strip_thinking('内部草稿</think>最终答案') == '最终答案'
    assert AgentLLMService._strip_thinking('内部草稿</think>') == ''

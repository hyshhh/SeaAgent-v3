"""事件载荷与重复调用守卫：这两条都对应真实事故。

1. 终局事件的 result 曾经在超限时被压成字符串，前端读不到 answerText/state/toolRecords，
   界面显示「未生成回答、0 tool records」——结构必须永远保住。
2. 4B 模型会反复调用同一工具：首次执行，第二、三次复用结果并提醒，第四次进入状态收尾。
"""
import json
from types import SimpleNamespace

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import ToolMessage

from config import load_config
from harness.middleware import build_middleware
from harness.shared_state import SharedTaskState
from harness.tool_guard import REPEAT_HINT, RepeatToolCallMiddleware
from harness.trace import _bounded, _Trace


class _Request:
    def __init__(self, name, args, call_id="c1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}


def _handler(request):
    return ToolMessage(content=f"executed:{request.tool_call['name']}", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])


def test_repeat_call_is_skipped_with_a_hint():
    guard = RepeatToolCallMiddleware()
    first = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)
    second = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)

    assert first.content == "executed:read_file", "第一次照常执行"
    replay = json.loads(second.content)
    assert replay["cached_result"] == "executed:read_file"
    assert replay["message"] == REPEAT_HINT
    assert second.status == "success", "跳过不是失败，否则会误触发连续失败守卫"


def test_tool_result_is_compact_for_model_but_full_for_trace_and_state():
    guard = RepeatToolCallMiddleware(state=SharedTaskState("q"), scope="executor")
    full_result = {
        "ok": True,
        "task": "direct_recognition",
        "status": "confirmed",
        "confidence": "confirmed",
        "coverageComplete": True,
        "matches": [
            {
                "trackId": "t-1",
                "score": 0.92,
                "scoreBand": "match",
                "keyframeId": "kf-first",
                "keyframeIds": ["kf-1"],
                "description": "yellow vessel",
            }
        ],
    }
    message = guard.wrap_tool_call(
        _Request("recognize_tracks", {"description": "yellow vessel"}),
        lambda request: ToolMessage(
            content=json.dumps(full_result),
            name="recognize_tracks",
            tool_call_id=request.tool_call["id"],
        ),
    )

    model_result = json.loads(message.content)
    assert model_result["status"] == "confirmed"
    assert model_result["matches"][0]["trackId"] == "t-1"
    assert model_result["matches"][0]["scoreBand"] == "match"
    assert model_result["matches"][0]["keyframeId"] == "kf-first"
    assert "keyframeIds" not in model_result["matches"][0]
    assert "confidence" not in model_result
    assert "task" not in model_result
    assert message.artifact["full_result"] == full_result
    assert guard.state.snapshot()["tool_results"][0]["result"] == full_result

    trace = _Trace(None, event_limit=4000, shared_state=SharedTaskState("q"))
    trace.consume({"messages": [message]}, agent_name="executor")
    assert trace.records[0]["result"] == full_result


def test_recognition_model_result_reads_counts_from_nested_statistics():
    guard = RepeatToolCallMiddleware(state=SharedTaskState("q"), scope="executor")
    result = {
        "ok": True,
        "status": "unconfirmed",
        "matches": [],
        "coverageComplete": False,
        "statistics": {"searchedTrackCount": 12, "unsearchableTrackCount": 3},
    }
    message = guard.wrap_tool_call(
        _Request("recognize_tracks", {"hull_number": "0857"}),
        lambda request: ToolMessage(
            content=json.dumps(result),
            name="recognize_tracks",
            tool_call_id=request.tool_call["id"],
        ),
    )

    model_result = json.loads(message.content)
    assert model_result["searchedTrackCount"] == 12
    assert model_result["unsearchableTrackCount"] == 3


def test_registry_count_hides_match_details_from_model_and_shared_state(tmp_path):
    state = SharedTaskState("count", artifact_dir=tmp_path, artifact_root=tmp_path)
    guard = RepeatToolCallMiddleware(state=state, scope="executor")
    full_result = {
        "ok": True,
        "trackCount": 2,
        "inRegistryCount": 1,
        "unregisteredCount": 1,
        "coverageComplete": True,
        "matches": [
            {"trackId": "t-1", "score": 0.91, "scoreBand": "match", "matchedObject": {"hullNumber": "0857"}},
            {"trackId": "t-2", "score": 0.42, "scoreBand": "mismatch", "matchedObject": {"hullNumber": "003"}},
        ],
    }
    message = guard.wrap_tool_call(
        _Request("match_entities", {}),
        lambda request: ToolMessage(
            content=json.dumps(full_result),
            name="match_entities",
            tool_call_id=request.tool_call["id"],
        ),
    )

    assert json.loads(message.content) == {
        "ok": True,
        "trackCount": 2,
        "inRegistryCount": 1,
        "unregisteredCount": 1,
        "coverageComplete": True,
    }
    assert message.artifact["full_result"] == json.loads(message.content)
    assert state.snapshot()["tool_results"][0]["result"] == json.loads(message.content)
    assert state.evidence_artifact_results() == [{
        "tool": "match_entities",
        "result": {
            "matches": [
                {key: item[key] for key in ("trackId", "score", "scoreBand", "matchedObject")}
                for item in full_result["matches"]
            ],
            "trackCount": 2,
            "truncated": False,
        },
    }]
    trace = _Trace(None, event_limit=4000, shared_state=state)
    trace.consume({"messages": [message]}, agent_name="executor")
    result = trace.result("thread", {"harness": {}})
    assert result["tool_records"][0]["result"] == json.loads(message.content)
    assert result["enrichedEvidence"]["trackIds"] == ["t-1", "t-2"]
    assert result["enrichedEvidence"]["displayGroups"][0]["score"] == 0.91
    assert result["enrichedEvidence"]["trackRows"][0]["registryMatch"]["status"] == "In registry"
    assert state.evidence_artifact_results() == []


def test_tool_error_keeps_error_type_for_model_debugging():
    guard = RepeatToolCallMiddleware(state=SharedTaskState("q"), scope="executor")
    message = guard.wrap_tool_call(
        _Request("dedup_tracks", {"time_range": [1, 2]}),
        lambda _request: ToolMessage(
            content=json.dumps({"ok": False, "error": "tool_timeout", "errorType": "TimeoutError", "hint": "timed out"}),
            name="dedup_tracks",
            status="error",
            tool_call_id="c1",
        ),
    )

    error = json.loads(message.content)
    assert error == {
        "ok": False,
        "error": "tool_timeout",
        "errorType": "TimeoutError",
        "hint": "timed out",
    }


def test_different_arguments_are_not_skipped():
    guard = RepeatToolCallMiddleware()
    guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)
    other = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/b/SKILL.md"}), _handler)
    assert other.content == "executed:read_file"


def test_bounded_payload_keeps_its_structure_when_it_exceeds_the_limit():
    """超限只截字符串，字典还是字典——这是终局事件能被前端解析的前提。"""
    payload = {
        "answer": "回答" * 50,
        "state": "stalled",
        "tool_records": [{"id": f"c{i}", "tool": "read_file", "result": "技能正文" * 800, "status": "completed"} for i in range(6)],
    }
    bounded = _bounded(payload, 4000)

    assert isinstance(bounded, dict), "整体不能被压成字符串"
    assert bounded["state"] == "stalled"
    assert isinstance(bounded["tool_records"], list) and len(bounded["tool_records"]) == 6
    assert all(isinstance(record, dict) for record in bounded["tool_records"])
    assert len(bounded["tool_records"][0]["result"]) <= 4001, "字符串字段照旧截断"
    json.dumps(bounded, ensure_ascii=False)  # 仍然可序列化


def test_middleware_stack_includes_the_repeat_guard():
    stack = build_middleware(
        load_config(),
        FakeListChatModel(responses=["ok"]),
        agent_spec={
            "name": "executor",
            "model_request_budget": 12,
            "closeout_request_budget": 2,
            "tools": [SimpleNamespace(name="get_track")],
            "skills": ["/skills/execution"],
            "skill_reminder": {"max_per_run": 1, "message": "Read execution skills."},
        },
        shared_state=SharedTaskState("q"),
    )
    assert any(isinstance(item, RepeatToolCallMiddleware) for item in stack)


def test_state_and_skill_reads_are_counted_like_all_other_tools():
    state = SharedTaskState("q")
    guard = RepeatToolCallMiddleware(
        state=state,
        scope="planner",
    )
    guard.wrap_tool_call(_Request("read_shared_state", {}), _handler)
    guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)
    assert state.snapshot()["tool_calls"]["planner"] == 2


def test_tool_guard_has_no_runtime_timeout_override():
    guard = RepeatToolCallMiddleware(scope="executor")
    assert not hasattr(guard, "tool_timeout")

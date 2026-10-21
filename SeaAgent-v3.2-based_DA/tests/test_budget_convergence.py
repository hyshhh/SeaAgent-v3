"""内外层预算收敛：一次无工具总结、真实证据保留、取消与持久化。"""

import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import load_config
from harness.convergence import ConvergenceRequired, summarize_once
from harness.middleware import build_middleware
from harness.runtime import SeaVideoHarness
from harness.shared_state import SharedStateStore, SharedTaskState
from harness.tool_guard import RepeatToolCallMiddleware
from harness.trace import _Trace


class SummaryModel:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("offline")
        return AIMessage(content="已取得 kf1；尚有步骤未验证，不能确认最终目标。")


def runtime(tmp_path, settings=None):
    instance = SeaVideoHarness.__new__(SeaVideoHarness)
    instance.config = {
        "harness": {
            "subagent_tool_calls": 6,
            **(settings or {}),
        }
    }
    instance.state_store = SharedStateStore(tmp_path / "state.sqlite")
    instance.shared_state = instance.state_store.load("t", "verify target")
    instance.shared_state.update(
        "planner", {"requirements": ["verify target"], "plan": ["scan", "verify"]}
    )
    instance.shared_state.runtime_update({"iteration": 1, "pending_agent": "executor"})
    instance.model = SummaryModel()
    instance._cancel = None
    instance._connection = None
    instance.agents = {}
    return instance


def trace(instance):
    return _Trace(None, 4000, shared_state=instance.shared_state)


def test_user_limits_are_declared():
    settings = load_config()["harness"]
    assert settings["subagent_tool_calls"] == 6
    assert settings["tool_timeout_seconds"] == 60
    assert settings["convergence_timeout_seconds"] == 60


def test_new_turn_resets_run_budget(tmp_path):
    instance = runtime(tmp_path)
    instance.shared_state = SharedTaskState("q", {"tool_calls_run": 40})
    stack = build_middleware(instance.config, FakeListChatModel(responses=["ok"]),
                             shared_state=instance.shared_state, agent_name="executor")
    guard = next(item for item in stack if isinstance(item, RepeatToolCallMiddleware))
    assert instance._convergence_reason("executor") == "run_tool_limit"
    instance.shared_state.start_turn("next")
    assert instance._convergence_reason("executor") == ""
    request = SimpleNamespace(tool_call={"name": "get_track", "args": {}, "id": "c1"})
    result = guard.wrap_tool_call(request, lambda req: ToolMessage(content="ok", tool_call_id="c1"))
    assert result.content == "ok"
    assert instance.shared_state.snapshot()["tool_calls_run"] == 1
    assert instance._convergence_reason("executor") == ""


def test_agent_hits_six_and_keeps_the_last_tool_result(tmp_path):
    instance = runtime(tmp_path)
    executed = []
    guard = RepeatToolCallMiddleware(
        state=instance.shared_state, scope="executor", limits={"agent": 6}
    )

    class Graph:
        def stream(self, *args, **kwargs):
            for index in range(10):
                req = SimpleNamespace(
                    tool_call={"name": "get_frames", "args": {"index": index}, "id": str(index)}
                )

                def handler(value, index=index):
                    executed.append(index)
                    return ToolMessage(
                        content=json.dumps({"keyframeId": f"kf{index}"}),
                        name="get_frames",
                        tool_call_id=str(index),
                    )

                yield {"tools": {"messages": [guard.wrap_tool_call(req, handler)]}}

    instance.agents["executor"] = Graph()
    outcome = instance._consume_driver(
        instance._run_agent_steps("executor", "q", "t", trace(instance))
    )
    assert outcome == {"state": "converge", "reason": "agent_tool_limit"}
    assert executed == [0, 1, 2, 3, 4, 5]
    assert "kf5" in instance.shared_state.snapshot()["evidence_ids"]
    instance._converge("executor", outcome["reason"], "t", trace(instance))
    assert len(instance.model.calls) == 1
    assert instance.shared_state.snapshot()["agent_calls"]["executor"] == 6
    assert instance.shared_state.snapshot()["step_status"] == {
        "step-1": "pending",
        "step-2": "pending",
    }
    assert "kf5" in instance.shared_state.snapshot()["evidence"]["shownKeyframeIds"]


def test_tool_batch_cannot_exceed_six_and_limits_are_not_errors():
    guard = RepeatToolCallMiddleware(scope="executor", limits={"agent": 6})
    calls = []
    for index in range(11):
        req = SimpleNamespace(
            tool_call={"name": "get_track", "args": {"index": index}, "id": str(index)}
        )
        result = guard.wrap_tool_call(
            req,
            lambda value: (
                calls.append(value) or ToolMessage(content="ok", tool_call_id=value.tool_call["id"])
            ),
        )
        if index >= 6:
            assert result.status == "success"
            assert result.artifact["convergence_required"]
    assert len(calls) == 6
    with pytest.raises(ConvergenceRequired, match="agent_tool_limit"):
        guard.wrap_model_call(SimpleNamespace(), lambda request: pytest.fail("extra model call"))


@pytest.mark.parametrize("role", ["planner", "executor", "reflector", "task"])
def test_convergence_is_once_per_scope_and_does_not_execute_tools(tmp_path, role):
    instance = runtime(tmp_path)
    instance.shared_state.record_tool_result(
        "executor", "get_frames", "c1", {"keyframeId": "kf1"}, "completed"
    )
    before = instance.shared_state.snapshot()["tool_calls_run"]
    for _ in range(2):
        assert (
            instance._converge(role, "agent_tool_limit", "t", trace(instance))["state"]
            == "completed"
        )
    assert len(instance.model.calls) == 1
    assert instance.shared_state.snapshot()["tool_calls_run"] == before
    assert "工具返回空不证明目标不存在" in instance.model.calls[0][0].content
    assert len(instance.state_store.load("t").snapshot()["convergence"]) == 1


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("tool_calls_per_run", 1, "run_tool_limit"),
    ],
)
def test_global_budget_goes_directly_to_one_final_summary(tmp_path, key, value, reason):
    instance = runtime(tmp_path, {key: value})
    instance.shared_state.claim_call("executor", None, {})
    instance.agents["executor"] = SimpleNamespace(
        stream=lambda *args, **kwargs: pytest.fail("budget ignored")
    )
    outcome = instance._consume_driver(instance._drive_multi_steps("q", "t", trace(instance)))
    assert outcome == {"state": "converge", "reason": reason}
    result = instance._finish(trace(instance), "t", outcome)
    assert result["state"] == "completed" and result["partial"]
    assert result["answer"]
    assert len(instance.model.calls) == 1
    assert instance.shared_state.snapshot()["pending_agent"] is None
    assert instance.state_store.load("t").snapshot()["task_status"] == "completed"


def test_iteration_limit_summarizes_instead_of_stalling(tmp_path):
    instance = runtime(tmp_path, {"max_iterations": 1})

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
    outcome = instance._consume_driver(instance._drive_multi_steps("q", "t", trace(instance)))
    assert outcome["state"] == "converge"
    result = instance._finish(trace(instance), "t", outcome)
    assert result["state"] == "completed" and result["partial"]
    assert len(instance.model.calls) == 1


def test_stage_wall_clock_does_not_trigger_convergence(tmp_path):
    instance = runtime(tmp_path)

    class Graph:
        def stream(self, *args, **kwargs):
            instance.shared_state.record_tool_result(
                "executor", "get_frames", "c1", {"keyframeId": "kf1"}, "completed"
            )
            yield {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content='{"keyframeId":"kf1"}', name="get_frames", tool_call_id="c1"
                        )
                    ]
                }
            }

    instance.agents["executor"] = Graph()
    outcome = instance._consume_driver(
        instance._run_agent_steps("executor", "q", "t", trace(instance))
    )
    assert outcome["state"] != "converge" or outcome.get("reason") != "agent_time_limit"


def test_outer_wall_clock_does_not_trigger_convergence(tmp_path):
    instance = runtime(tmp_path)
    assert instance._convergence_reason("executor") == ""
    signal = threading.Event()
    signal.convergence_requested = "manual_request"
    instance._cancel = signal
    assert instance._convergence_reason("executor") == "manual_request"
    assert not signal.is_set()


def test_model_requests_use_model_timeout_without_stage_limit(tmp_path):
    instance = runtime(tmp_path)
    build_middleware(
        {"harness": {"retry_delay_seconds": 0}},
        FakeListChatModel(responses=["ok"]),
        skills_attached=False,
        shared_state=instance.shared_state,
        convergence_reason=lambda: instance._convergence_reason("executor"),
        model_timeout=instance._model_timeout_seconds,
    )
    assert instance._model_timeout_seconds() == 60


def test_summary_updates_actual_sdk_timeout_and_disables_hidden_retries(tmp_path):
    instance = runtime(tmp_path)
    requests = []

    def unavailable(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "offline", "type": "server_error"}})

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as client:
        model = ChatOpenAI(
            model="offline-test",
            api_key="test",
            http_client=client,
            timeout=90,
            max_retries=2,
            streaming=True,
        )
        answer, ok = summarize_once(
            model, instance.shared_state.view(), "task", "run_time_limit", 60
        )
        assert not ok and "无法确认" in answer
        assert len(requests) == 1
        assert requests[0].extensions["timeout"]["read"] == 60
        payload = json.loads(requests[0].content)
        assert payload["stream"] is False and "tools" not in payload
        assert model.root_client.max_retries == 2 and model.request_timeout == 90


def test_summary_model_failure_does_not_retry_or_report_budget_error(tmp_path):
    instance = runtime(tmp_path)
    instance.model = SummaryModel(fail=True)
    result = instance._finish(
        trace(instance), "t", {"state": "converge", "reason": "run_tool_limit"}
    )
    assert len(instance.model.calls) == 1
    assert result["state"] == "completed" and result["partial"]
    assert result["convergence"][0]["status"] == "fallback"
    assert "尚未取得可引用证据" in result["answer"]


def test_user_cancel_does_not_call_the_convergence_model(tmp_path):
    instance = runtime(tmp_path)
    instance._cancel = threading.Event()
    instance._cancel.set()
    result = instance._finish(
        trace(instance), "t", {"state": "converge", "reason": "run_tool_limit"}
    )
    assert result["state"] == "cancelled"
    assert instance.model.calls == []


def test_restored_pending_agent_keeps_its_call_budget(tmp_path):
    instance = runtime(tmp_path)
    instance.shared_state.runtime_update({"task_status": "running"})
    for _ in range(4):
        instance.shared_state.claim_call("executor", None, {})

    def restored(role, *args):
        assert role == "executor"
        assert instance.shared_state.snapshot()["agent_calls"][role] == 4
        yield None
        return {"state": "cancelled"}

    instance._run_agent_steps = restored
    outcome = instance._consume_driver(instance._drive_multi_steps("q", "t", trace(instance)))
    assert outcome["state"] == "cancelled"


def test_user_cancel_prevents_further_exploration_model_requests():
    guard = RepeatToolCallMiddleware(cancelled=lambda: True)
    with pytest.raises(ConvergenceRequired, match="cancelled"):
        guard.wrap_model_call(SimpleNamespace(), lambda request: pytest.fail("cancel ignored"))


def test_interrupted_summary_is_not_invoked_again(tmp_path):
    instance = runtime(tmp_path)
    instance.shared_state.runtime_update(
        {
            "convergence": [
                {"role": "task", "reason": "run_time_limit", "iteration": 1, "status": "pending"}
            ]
        }
    )
    instance.state_store.save("t", instance.shared_state)
    instance.shared_state = instance.state_store.load("t")
    instance._converge("task", "run_time_limit", "t", trace(instance))
    assert instance.model.calls == []
    stored = instance.state_store.load("t").snapshot()
    assert stored["convergence"][0]["status"] == "fallback"
    assert "无法确认" in stored["assessment"]["final_answer"]

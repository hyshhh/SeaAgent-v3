"""使用真实 Deep Agents 图和 SQLite 检查点，模型与领域服务离线替身。"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from config import load_config
from harness.runtime import SeaVideoHarness


class _ScriptedModel(ChatOpenAI):
    write: bool = False
    batch_tools: bool = False
    replan: bool = False
    rejected: bool = False
    skip_planner_state_write: bool = False
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        system = str(messages[0].content)
        role = (
            "planner"
            if "你是 Planner Agent" in system
            else "executor"
            if "你是 Executor Agent" in system
            else "reflector"
        )
        last = messages[-1]
        budget_was_hit = any(
            isinstance(message, ToolMessage)
            and isinstance(getattr(message, "artifact", None), dict)
            and message.artifact.get("budget_exhausted")
            for message in messages
        )
        if self.skip_planner_state_write and role == "planner" and (
            (isinstance(last, ToolMessage) and last.name == "read_shared_state")
            or (
                isinstance(last, HumanMessage)
                and "本角色必须先把真实进展写入共享状态" in str(last.content)
            )
        ):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="未提交状态"))])
        if self.batch_tools and role == "executor" and not budget_was_hit:
            calls = [
                {
                    "name": "get_track",
                    "args": {"start_time": index, "end_time": index + 1},
                    "id": uuid.uuid4().hex,
                }
                for index in range(9)
            ]
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]
            )
        if isinstance(last, ToolMessage) and "User rejected" in str(last.content):
            self.rejected = True
        name, args = "read_shared_state", {}
        if isinstance(last, ToolMessage):
            if last.name.startswith("update_"):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
            if last.name == "read_shared_state":
                state = json.loads(last.content)
                if role == "planner":
                    name = "update_planner_state"
                    args["patch"] = {
                        "requirements": ["verify target"],
                        "plan": ["scan"],
                        "next_action": f"iteration {state['iteration']}",
                    }
                elif role == "executor":
                    if not state["tool_results"] and not self.rejected:
                        name = "add_registry_vessel" if self.write else "get_track"
                        args = (
                            {"hull_number": "003", "user_intent": "add 003"}
                            if self.write
                            else {"start_time": 0, "end_time": 60}
                        )
                    else:
                        name = "update_execution_state"
                        args["patch"] = {
                            "execution_status": (
                                "failed"
                                if self.rejected
                                else "running"
                                if state["iteration"] % 2 == 0
                                else "success"
                            ),
                            "step_status": {
                                "step-1": (
                                    "failed"
                                    if self.rejected
                                    else "in_progress"
                                    if state["iteration"] % 2 == 0
                                    else "completed"
                                )
                            },
                        }
                elif not state["evidence"]:
                    name, args = "show_evidence", {"keyframe_ids": ["kf1"]}
                else:
                    name = "update_reflection_state"
                    args["patch"] = {
                        "decision": "replan" if self.replan else "terminate",
                        "assessment": {"final_answer": "verified kf1"},
                    }
        message = AIMessage(
            content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex}]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Service:
    def __init__(self):
        self.writes = 0
        self.queries = 0

    def __getattr__(self, name):
        def invoke(**kwargs):
            if name == "addRegistryVessel":
                self.writes += 1
            if name == "getTrack":
                self.queries += 1
            if name == "showEvidence":
                return {"ok": True, "shownKeyframeIds": ["kf1"]}
            return {"ok": True, "trackIds": ["t1"], "keyframeIds": ["kf1"]}

        return invoke


def _runtime(tmp_path, service, write=False, tool_limit=12, **model_settings):
    config = load_config()
    config["llm"]["model"] = "shared-state-offline-test"
    config["harness"].update(
        {
            "checkpointer": str(tmp_path / "checkpoint.sqlite"),
            "shared_state_file": str(tmp_path / "state.sqlite"),
            "heartbeat_seconds": 0,
            "retry_delay_seconds": 0,
            "reconnect_attempts": 0,
            "subagent_tool_calls": tool_limit,
            "skill_reminder_max_per_run": 0,
        }
    )
    model = _ScriptedModel(
        model="shared-state-offline-test",
        api_key="test",
        streaming=False,
        write=write,
        **model_settings,
    )
    return SeaVideoHarness(config, service, model=model)


@pytest.mark.parametrize("stream", [False, True])
def test_real_graph_completes_with_tool_budget_and_state_updates(tmp_path, stream):
    runtime = _runtime(tmp_path, _Service(), tool_limit=4)
    result = list(runtime.stream("scan", "t"))[-1]["result"] if stream else runtime.run("scan", "t")
    assert result["state"] == "completed" and not result["partial"]
    assert runtime.model.calls > 0
    assert result["shared_state"]["agent_calls"]["executor"] == 1
    assert "convergence" not in result["shared_state"]
    assert result["evidence"]["shownKeyframeIds"] == ["kf1"]
    assert runtime._connection is None


def test_real_graph_parallel_batch_cannot_exceed_budget(tmp_path):
    service = _Service()
    runtime = _runtime(tmp_path, service, tool_limit=4, batch_tools=True)
    result = runtime.run("scan", "t")
    assert result["state"] == "completed" and not result["partial"]
    assert service.queries == 4
    assert runtime.model.calls > 1
    assert result["shared_state"]["agent_calls"]["executor"] == 4
    assert (
        len(
            [item for item in result["shared_state"]["tool_results"] if item["tool"] == "get_track"]
        )
        == 4
    )
    assert result["shared_state"]["step_status"] == {"step-1": "completed"}


@pytest.mark.parametrize("limit", ["tool_calls_per_run", "max_iterations"])
def test_real_graph_outer_limits_do_not_start_a_summary_request(tmp_path, limit):
    runtime = _runtime(tmp_path, _Service(), replan=limit == "max_iterations")
    runtime.config["harness"][limit] = 1
    result = runtime.run("scan", "t")
    if limit == "tool_calls_per_run":
        assert result["state"] == "completed" and not result["partial"]
        assert result["shared_state"]["decision"] == "terminate"
    else:
        assert result["state"] == "stalled" and result["partial"]
        assert result["shared_state"]["decision"] == "replan"
    assert result["shared_state"]["pending_agent"] is None
    assert result["evidence"]["shownKeyframeIds"] == ["kf1"]
    if limit == "tool_calls_per_run":
        assert result["shared_state"]["tool_calls_run"] == 1


def test_agent_without_state_write_gets_one_forced_repair_turn(tmp_path):
    runtime = _runtime(tmp_path, _Service(), skip_planner_state_write=True)

    result = runtime.run("scan", "missing-state-write")

    assert result["state"] == "stalled"
    assert "planner 未提交负责的共享状态" in result["error"]
    assert runtime.model.calls >= 3


@pytest.mark.parametrize("decision,writes", [("approve", 1), ("reject", 0)])
def test_four_call_limit_preserves_confirmation_and_prevents_replay(tmp_path, decision, writes):
    service = _Service()
    paused = _runtime(tmp_path, service, write=True, tool_limit=4).run("add 003", "t")
    assert paused["state"] == "awaiting_confirmation" and service.writes == 0
    continued = _runtime(tmp_path, service, write=True, tool_limit=4).resume(decision, "t")
    assert continued["state"] == "completed"
    assert service.writes == writes
    assert continued["shared_state"]["agent_calls"]["executor"] <= 4
    with pytest.raises(ValueError, match="没有待确认"):
        _runtime(tmp_path, service, write=True, tool_limit=4).resume(decision, "t")
    assert service.writes == writes


def test_real_graphs_complete_using_only_shared_task_state(tmp_path):
    runtime = _runtime(tmp_path, _Service())
    result = runtime.run("scan", "t")
    assert result["state"] == "completed", result.get("error")
    assert result["answer"] == "verified kf1"
    assert result["shared_state"]["step_status"] == {"step-1": "completed"}
    assert runtime.state_store.load("t").snapshot()["task_status"] == "completed"
    for graph in runtime.agents.values():
        assert "todos" not in graph.channels
        assert not any(
            "TodoListMiddleware" in name or "ToolCallLimitMiddleware" in name
            for name in graph.nodes
        )


def test_confirmation_survives_a_new_runtime_and_cannot_be_replayed(tmp_path):
    service = _Service()
    paused = _runtime(tmp_path, service, write=True).run("add 003", "t")
    assert paused["state"] == "awaiting_confirmation"
    assert service.writes == 0
    before = paused["shared_state"]["tool_calls_run"]
    continued = _runtime(tmp_path, service, write=True).resume("approve", "t")
    assert continued["state"] == "completed"
    assert service.writes == 1
    assert continued["shared_state"]["tool_calls_run"] > before
    with pytest.raises(ValueError, match="没有待确认"):
        _runtime(tmp_path, service, write=True).resume("approve", "t")
    assert service.writes == 1


def test_same_question_after_completion_starts_fresh_task_state(tmp_path):
    runtime = _runtime(tmp_path, _Service())
    first = runtime.run("scan", "t")
    second = runtime.run("scan", "t")
    assert second["state"] == "completed"
    assert second["shared_state"]["iteration"] == 1
    assert second["shared_state"]["tool_calls_run"] > 0
    assert second["shared_state"]["version"] > first["shared_state"]["version"]


@pytest.mark.parametrize("stream", [False, True])
def test_failed_graph_assembly_saves_error_and_closes_connection(tmp_path, monkeypatch, stream):
    runtime = _runtime(tmp_path, _Service())

    def broken():
        runtime._open_checkpoint()
        raise RuntimeError("assembly failed")

    monkeypatch.setattr(runtime, "_build_agents", broken)
    if stream:
        events = list(runtime.stream("q", "t"))
        assert events[-1]["type"] == "error"
    else:
        with pytest.raises(RuntimeError, match="assembly failed"):
            runtime.run("q", "t")
    assert runtime._connection is None
    assert runtime.state_store.load("t").snapshot()["task_status"] == "error"


@pytest.mark.parametrize("stream", [False, True])
def test_driver_failure_still_saves_terminal_state(tmp_path, monkeypatch, stream):
    runtime = _runtime(tmp_path, _Service())

    def broken(*args, **kwargs):
        yield None
        raise RuntimeError("driver failed")

    monkeypatch.setattr(runtime, "_drive_multi_steps", broken)
    result = list(runtime.stream("q", "t"))[-1]["result"] if stream else runtime.run("q", "t")
    assert result["state"] == "error"
    assert runtime.state_store.load("t").snapshot()["task_status"] == "error"
    assert runtime._connection is None


def test_reconnect_retries_build_failures_without_resetting_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runtime = _runtime(tmp_path, _Service())
    runtime.config["harness"].update(reconnect_attempts=2, reconnect_backoff_seconds=0)
    attempts = []

    class FailingGraph:
        def stream(self, *args, **kwargs):
            raise ConnectionError("stream failed")

    class ReconnectedGraph:
        def get_state(self, config):
            return SimpleNamespace(next=("model",), tasks=())

        def stream(self, payload, **kwargs):
            assert payload is None
            assert runtime.shared_state.snapshot()["tool_calls_run"] == 3
            yield {}

    def rebuild(role):
        attempts.append(role)
        if len(attempts) == 1:
            raise ConnectionError("rebuild failed")
        runtime.agents[role] = ReconnectedGraph()

    runtime._initialize_state("t", "q")
    for _ in range(3):
        runtime.shared_state.claim_call("executor", None, {})
    runtime.agents["executor"] = FailingGraph()
    monkeypatch.setattr(runtime, "_reconnect_agent", rebuild)
    from harness.trace import _Trace

    try:
        result = runtime._consume_driver(
            runtime._run_agent_steps("executor", "q", "t", _Trace(None, 4000))
        )
        assert result["state"] == "completed"
        assert attempts == ["executor", "executor"]
    finally:
        runtime.close()


def test_deleting_a_session_removes_all_three_checkpoints_and_task_state(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from web.routes.agent_api import _delete_runtime_state

    runtime = _runtime(tmp_path, _Service())
    assert runtime.run("scan", "t")["state"] == "completed"
    _delete_runtime_state(runtime.config, "t")
    assert runtime.state_store.load("t").snapshot()["question"] == ""
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        for role in ("planner", "executor", "reflector"):
            assert saver.get_tuple({"configurable": {"thread_id": f"t:{role}"}}) is None

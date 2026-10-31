"""使用真实 Deep Agents 图和 SQLite 检查点，模型与领域服务离线替身。"""

import json
import threading
import time
import uuid
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from config import load_config
from harness.runtime import SeaVideoHarness


class _ScriptedModel(ChatOpenAI):
    batch_tools: bool = False
    batch_vlm: bool = False
    replan: bool = False
    skip_planner_state_write: bool = False
    calls: int = 0
    batch_sent: bool = False
    vlm_batch_sent: bool = False

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
        if self.skip_planner_state_write and role == "planner" and (
            (isinstance(last, ToolMessage) and last.name == "read_shared_state")
            or (
                isinstance(last, HumanMessage)
                and "本角色必须把已确认进展写入共享状态" in str(last.content)
            )
        ):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="未提交状态"))])
        if self.batch_tools and role == "executor" and not self.batch_sent:
            self.batch_sent = True
            calls = [
                {
                    "name": "recognize_tracks",
                    "args": {"description": f"target-{index}"},
                    "id": uuid.uuid4().hex,
                }
                for index in range(9)
            ]
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]
            )
        if self.batch_vlm and role == "executor" and not self.vlm_batch_sent:
            self.vlm_batch_sent = True
            calls = [
                {
                    "name": "verify_vessel_images",
                    "args": {"candidate_keyframe_id": f"keyframe-{index}"},
                    "id": uuid.uuid4().hex,
                }
                for index in range(9)
            ]
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]
            )
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
                    if not state["tool_results"]:
                        name = "recognize_tracks"
                        args = {"description": "target vessel"}
                    else:
                        name = "update_execution_state"
                        args["patch"] = {
                            "execution_status": "success",
                            "step_status": {
                                "step-1": "completed"
                            },
                        }
                elif not state["evidence"]:
                    name, args = "show_evidence", {"track_ids": ["t1"]}
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
        self.queries = 0
        self.active_queries = 0
        self.max_active_queries = 0
        self.verify_queries = 0
        self.active_verify = 0
        self.max_active_verify = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        def invoke(**kwargs):
            if name == "recognize_tracks":
                with self._lock:
                    self.queries += 1
                    self.active_queries += 1
                    self.max_active_queries = max(
                        self.max_active_queries, self.active_queries
                    )
                try:
                    time.sleep(0.005)
                finally:
                    with self._lock:
                        self.active_queries -= 1
            if name == "verify_vessel_images":
                with self._lock:
                    self.verify_queries += 1
                    self.active_verify += 1
                    self.max_active_verify = max(
                        self.max_active_verify, self.active_verify
                    )
                try:
                    time.sleep(0.02)
                finally:
                    with self._lock:
                        self.active_verify -= 1
            if name == "show_evidence":
                return {"ok": True, "trackIds": ["t1"], "shownKeyframeIds": ["kf1"]}
            if name == "verify_vessel_images":
                return {
                    "ok": True,
                    "decision": "match",
                    "referenceKeyframeId": "reference",
                    "keyframeId": kwargs.get("candidate_keyframe_id"),
                }
            return {"ok": True, "trackIds": ["t1"], "keyframeIds": ["kf1"]}

        return invoke


def _runtime(tmp_path, service, executor_model_budget=12, **model_settings):
    config = load_config()
    config["llm"]["model"] = "shared-state-offline-test"
    role_config = yaml.safe_load(
        (Path(__file__).parents[1] / config["harness"]["subagents_file"]).read_text(encoding="utf-8")
    )
    for spec in role_config["agents"]:
        spec["skill_reminder"]["max_per_run"] = 0
        if spec["name"] == "executor":
            spec["model_request_budget"] = executor_model_budget
    spec_path = tmp_path / "subagents.yaml"
    spec_path.write_text(yaml.safe_dump(role_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config["harness"].update(
        {
            "checkpointer": str(tmp_path / "checkpoint.sqlite"),
            "shared_state_file": str(tmp_path / "state.sqlite"),
            "subagents_file": str(spec_path),
            "heartbeat_seconds": 0,
            "retry_delay_seconds": 0,
            "reconnect_attempts": 0,
        }
    )
    model = _ScriptedModel(
        model="shared-state-offline-test",
        api_key="test",
        streaming=False,
        **model_settings,
    )
    return SeaVideoHarness(config, service, model=model)


@pytest.mark.parametrize("stream", [False, True])
def test_real_graph_completes_with_model_budget_and_state_updates(tmp_path, stream):
    runtime = _runtime(tmp_path, _Service(), executor_model_budget=12)
    result = list(runtime.stream("scan", "t"))[-1]["result"] if stream else runtime.run("scan", "t")
    assert result["state"] == "completed" and not result["partial"]
    assert runtime.model.calls > 0
    assert runtime.shared_state.snapshot()["model_requests"]["executor"] > 0
    assert runtime.shared_state.snapshot()["tool_calls"]["executor"] >= 3
    assert "convergence" not in result["shared_state"]
    assert result["evidence"]["trackIds"] == ["t1"]
    assert "shownKeyframeIds" not in result["evidence"]
    assert runtime._connection is None


def test_real_graph_runs_tool_batches_without_a_total_tool_limit(tmp_path):
    service = _Service()
    runtime = _runtime(tmp_path, service, executor_model_budget=12, batch_tools=True)
    result = runtime.run("scan", "t")
    assert result["state"] == "completed" and not result["partial"]
    assert service.queries == 9
    assert service.max_active_queries == 1
    assert runtime.model.calls > 1
    assert runtime.shared_state.snapshot()["tool_calls"]["executor"] >= 11
    assert (
        len(
            [item for item in result["shared_state"]["tool_results"] if item["tool"] == "recognize_tracks"]
        )
        == 9
    )
    assert result["shared_state"]["step_status"] == {"step-1": "completed"}


def test_real_graph_runs_vlm_batches_concurrently(tmp_path):
    service = _Service()
    runtime = _runtime(tmp_path, service, executor_model_budget=12, batch_vlm=True)

    result = runtime.run("scan", "t")

    assert result["state"] == "completed"
    assert service.verify_queries == 9
    assert 2 <= service.max_active_verify <= 12
    assert service.max_active_queries == 0
    assert runtime.tool_concurrency_limit == 15


def test_real_graph_iteration_limit_does_not_start_a_summary_request(tmp_path):
    runtime = _runtime(tmp_path, _Service(), replan=True)
    runtime.config["harness"]["max_iterations"] = 1
    result = runtime.run("scan", "t")
    assert result["state"] == "stalled" and result["partial"]
    assert result["shared_state"]["decision"] == "replan"
    assert result["shared_state"]["pending_agent"] is None
    assert result["evidence"]["trackIds"] == ["t1"]
    assert "shownKeyframeIds" not in result["evidence"]


def test_agent_without_state_write_gets_one_forced_repair_turn(tmp_path):
    runtime = _runtime(tmp_path, _Service(), skip_planner_state_write=True)

    result = runtime.run("scan", "missing-state-write")

    assert result["state"] == "stalled"
    assert "planner 未提交负责的共享状态" in result["error"]
    assert runtime.model.calls >= 3


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


def test_same_question_after_completion_starts_fresh_task_state(tmp_path):
    runtime = _runtime(tmp_path, _Service())
    first = runtime.run("scan", "t")
    second = runtime.run("scan", "t")
    assert second["state"] == "completed"
    assert second["shared_state"]["iteration"] == 1
    assert second["shared_state"]["execution_status"] == "success"
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
            assert runtime.shared_state.snapshot()["model_requests"]["executor"] == 3
            yield {}

    def rebuild(role):
        attempts.append(role)
        if len(attempts) == 1:
            raise ConnectionError("rebuild failed")
        runtime.agents[role] = ReconnectedGraph()

    runtime._initialize_state("t", "q")
    for _ in range(3):
        assert runtime.shared_state.claim_model_request("executor", 12, 2) == "normal"
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

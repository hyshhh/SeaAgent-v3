"""运行时契约：事件记账、单个角色的失控收尾、取消。

三 Agent 协同下，每个角色由 ``_run_agent_steps`` 单独起一条流，所以这里的桩只需要
一个能吐帧的假 agent，不必造整个图。
"""

import threading

from langchain_core.messages import AIMessage, ToolMessage

from config import load_config
from harness.runtime import SeaVideoHarness, _Trace
from harness.shared_state import SharedStateStore, SharedTaskState

# ---------------------------------------------------------------- 事件记账


def test_trace_correlates_tool_result_and_projects_shared_evidence():
    events = []
    state = SharedTaskState("问题")
    state.record_tool_result(
        "executor", "show_evidence", "call-1", {"ok": True, "shownKeyframeIds": ["kf-1"]}, "completed"
    )
    state.update("reflector", {"assessment": {"final_answer": "已完成"}})
    trace = _Trace(events.append, event_limit=80, shared_state=state)
    trace.consume(
        {
            "skills_metadata": [{"name": "query", "description": "查询"}],
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "show_evidence",
                            "args": {"keyframe_ids": ["kf-1"]},
                            "id": "call-1",
                        }
                    ],
                )
            ],
        },
        agent_name="executor",
    )
    trace.consume(
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content='{"ok": true, "shownKeyframeIds": ["kf-1"]}',
                        name="show_evidence",
                        tool_call_id="call-1",
                    )
                ]
            }
        },
        agent_name="executor",
    )
    trace.consume(
        {"model": {"messages": [AIMessage(content="已完成")]}}, agent_name="reflector"
    )
    assert trace.records[0]["status"] == "completed"
    assert trace.records[0]["id"] == "call-1"
    assert [event["type"] for event in events if event["type"] != "plan"] == [
        "skill",
        "tool_start",
        "model",
        "tool_result",
        "model",
    ]
    role_output = next(
        event for event in events if event["type"] == "model" and event.get("agent") == "reflector"
    )
    assert role_output["title"] == "角色输出"
    assert trace.result("thread-1", load_config())["answer"] == "已完成"


def test_planner_clarification_stops_before_executor_and_reflector(tmp_path):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {"harness": {"max_iterations": 2}}
    runtime.shared_state = SharedTaskState("目标船舶在哪里？")
    runtime.state_store = SharedStateStore(tmp_path / "state.sqlite")
    runtime._cancel = None
    called = []

    def run_role(role, *_args):
        called.append(role)
        if role == "planner":
            runtime.shared_state.update(
                "planner", {"clarification": "请补充查询时间范围。"}
            )
        yield None
        return {"state": "completed"}

    runtime._run_agent_steps = run_role
    outcome = runtime._consume_driver(
        runtime._drive_multi_steps(
            "目标船舶在哪里？", "qa-test", _Trace(None, 4000)
        )
    )

    assert outcome == {
        "state": "awaiting_input",
        "clarification": "请补充查询时间范围。",
    }
    assert called == ["planner"]
    assert runtime.shared_state.snapshot()["task_status"] == "awaiting_input"


def test_trace_accepts_root_level_updates_and_bounds_event_payloads():
    events = []
    trace = _Trace(events.append, event_limit=10, shared_state=SharedTaskState("问题"))
    trace.consume({"messages": [AIMessage(content="x" * 50)]}, agent_name="executor")
    assert all(len(str(value)) <= 11 for event in events for value in event.values())
    assert any(event["type"] == "model" and event["message"].startswith("x") for event in events)


def test_trace_marks_each_record_with_the_calling_role():
    """角色是调用方直接给的——事件与记录都要带上它，前端才画得出分工。"""
    events = []
    trace = _Trace(events.append, event_limit=4000, shared_state=SharedTaskState("问题"))
    trace.consume(
        {
            "model": {
                "messages": [
                    AIMessage(
                        content="", tool_calls=[{"name": "get_track", "args": {}, "id": "call-1"}]
                    )
                ]
            }
        },
        agent_name="executor",
    )
    assert trace.records[0]["agent"] == "executor"
    tool_event = next(event for event in events if event["type"] == "tool_start")
    assert tool_event["agent"] == "executor"


def test_builtin_file_tools_get_chinese_labels():
    """时间线要能一眼看出这一轮有没有去读技能正文。"""
    from harness.runtime import _tool_labels

    labels = _tool_labels(load_config())
    assert labels["read_file"] == "读取技能正文"
    assert labels["get_track"] == "轨迹记忆"


# ---------------------------------------------------------------- 单个角色的执行


class _FakeAgent:
    def __init__(self, updates=None, error=None):
        self.updates = updates or []
        self.error = error

    def stream(self, *_args, **_kwargs):
        yield from self.updates
        if self.error:
            raise self.error


class _CheckpointAgent(_FakeAgent):
    def __init__(self, updates=None, error=None, snapshot=None, state_error=None):
        super().__init__(updates, error)
        self.snapshot = snapshot
        self.state_error = state_error
        self.received = []

    def get_state(self, _config):
        if self.state_error:
            raise self.state_error
        return self.snapshot

    def stream(self, payload, **kwargs):
        self.received.append(payload)
        yield from super().stream(payload, **kwargs)


class _Store:
    """共享状态的落盘口：这些用例只关心返回的终态，不做真实写盘。"""

    def save(self, *_args, **_kwargs):
        pass


def _runtime_with_agent(agent):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {
        "harness": {
            "stream_mode": "updates",
            "event_payload_max_chars": 4000,
            "evidence_tool": "show_evidence",
            "output": {
                "answer_field": "answer",
                "state_field": "state",
                "evidence_field": "evidence",
            },
        },
        "tools": [{"name": "show_evidence", "label": "证据汇总"}],
    }
    runtime.agents = {"executor": agent}
    runtime.agent_specs = {"executor": {"task_prompt": "用户问题：{question}"}}
    runtime.shared_state = SharedTaskState("问题")
    runtime.state_store = _Store()
    runtime.event_handler = None
    runtime._connection = None
    runtime._stale_checkpoint_roles = set()
    return runtime


def _run_steps(runtime, trace, cancel=None, resume_payload=None):
    """把 _run_agent_steps 这个生成器跑到底，返回它的终态。"""
    driver = runtime._run_agent_steps(
        "executor", "问题", "thread-x", trace, cancel, resume_payload
    )
    while True:
        try:
            next(driver)
        except StopIteration as stopped:
            return stopped.value


def _trace():
    return _Trace(
        None, event_limit=4000, shared_state=SharedTaskState("问题")
    )


def test_role_steps_stop_between_frames_when_cancelled():
    """中断信号置位后本角色立即收尾，不再消费后面的帧。"""
    cancel = threading.Event()
    cancel.set()
    trace = _trace()
    runtime = _runtime_with_agent(
        _FakeAgent(
            [
                {
                    "model": {
                        "messages": [AIMessage(content="不该被消费的回答", id="answer-cancel")]
                    }
                },
            ]
        )
    )
    outcome = _run_steps(runtime, trace, cancel)
    assert outcome["state"] == "cancelled"
    assert trace.records == []


def test_role_steps_complete_without_a_cancel_signal():
    trace = _trace()
    runtime = _runtime_with_agent(
        _FakeAgent(
            [
                {"model": {"messages": [AIMessage(content="回答文本", id="answer-ok")]}},
            ]
        )
    )
    outcome = _run_steps(runtime, trace, threading.Event())
    assert outcome["state"] == "completed"
    assert runtime.shared_state.snapshot()["assessment"] == {}


def test_stalled_role_is_reported_as_incomplete_not_as_a_finished_answer():
    events = []
    runtime = _runtime_with_agent(_FakeAgent())
    trace = _Trace(events.append, event_limit=4000, shared_state=runtime.shared_state)

    result = runtime._finish(
        trace,
        "thread-x",
        {"state": "stalled", "reason": "planner 未更新共享状态"},
    )

    assert result["state"] == "stalled"
    assert result["answer"] == "planner 未更新共享状态"
    assert result["error"] == "planner 未更新共享状态"
    assert events[-1]["type"] == "error"
    assert events[-1]["title"] == "本轮未能完成"


def test_agent_input_uses_checkpoint_for_reconnect_without_repeating_prompt():
    snapshot = type("_Snapshot", (), {"next": ("model",), "tasks": ()})()
    agent = _CheckpointAgent(snapshot=snapshot)
    runtime = _runtime_with_agent(agent)

    payload = runtime._agent_input_for_attempt("executor", "问题", "thread-x:executor")

    assert payload is None


def test_agent_input_resumes_human_interrupt_and_only_prompts_for_fresh_state():
    resume_payload = object()
    interrupted = type(
        "_Snapshot",
        (),
        {"next": (), "tasks": (type("_Task", (), {"interrupts": (object(),)})(),)},
    )()
    runtime = _runtime_with_agent(_CheckpointAgent(snapshot=interrupted))
    assert runtime._agent_input_for_attempt(
        "executor", "问题", "thread-x:executor", resume_payload
    ) is resume_payload

    fresh = type("_Snapshot", (), {"next": (), "tasks": ()})()
    runtime.agents["executor"] = _CheckpointAgent(snapshot=fresh)
    payload = runtime._agent_input_for_attempt("executor", "问题", "thread-x:executor")
    assert payload["messages"][0]["content"].startswith("用户问题：问题")


def test_new_turn_discards_stale_checkpoint_but_keeps_completed_history():
    snapshot = type(
        "_Snapshot",
        (),
        {
            "next": ("tools",),
            "tasks": (),
            "values": {
                "messages": [
                    AIMessage(content="上一轮回答", id="previous-answer"),
                    {
                        "role": "user",
                        "content": "用户问题：本轮问题\n当前角色：executor",
                    },
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "update_execution_state",
                                "args": {"patch": {}},
                                "id": "stale-update",
                            }
                        ],
                    ),
                ]
            },
        },
    )()
    runtime = _runtime_with_agent(_CheckpointAgent(snapshot=snapshot))
    runtime._stale_checkpoint_roles = {"executor"}
    deleted = []
    runtime._open_checkpoint = lambda: type(
        "_Saver", (), {"delete_thread": lambda _self, thread: deleted.append(thread)}
    )()

    payload = runtime._agent_input_for_attempt(
        "executor", "本轮问题", "thread-x:executor"
    )

    assert deleted == ["thread-x:executor"]
    assert len(payload["messages"]) == 2
    assert payload["messages"][0].content == "上一轮回答"
    assert payload["messages"][1]["content"].startswith("用户问题：本轮问题")
    assert runtime._stale_checkpoint_roles == set()


def test_reconnect_uses_the_same_checkpoint_input_decision():
    fresh = type("_Snapshot", (), {"next": (), "tasks": ()})()
    initial = _CheckpointAgent(error=RuntimeError("connection lost"), snapshot=fresh)
    resumed = _CheckpointAgent(
        updates=[{"model": {"messages": [AIMessage(content="恢复完成", id="reconnected")]}}],
        snapshot=type("_Snapshot", (), {"next": ("model",), "tasks": ()})(),
    )
    runtime = _runtime_with_agent(initial)
    runtime.config["harness"].update(reconnect_attempts=1, reconnect_backoff_seconds=0)
    runtime._reconnect_agent = lambda _role: runtime.agents.update(executor=resumed)

    outcome = _run_steps(runtime, _trace())

    assert outcome["state"] == "completed"
    assert initial.received[0]["messages"][0]["content"].startswith("用户问题：问题")
    assert resumed.received == [None]


def test_interrupt_reconnect_reuses_resume_payload():
    resume_payload = object()
    interrupted = type(
        "_Snapshot",
        (),
        {"next": (), "tasks": (type("_Task", (), {"interrupts": (object(),)})(),)},
    )()
    initial = _CheckpointAgent(error=RuntimeError("connection lost"), snapshot=interrupted)
    resumed = _CheckpointAgent(
        updates=[{"model": {"messages": [AIMessage(content="确认后完成", id="interrupt-resumed")]}}],
        snapshot=interrupted,
    )
    runtime = _runtime_with_agent(initial)
    runtime.config["harness"].update(reconnect_attempts=1, reconnect_backoff_seconds=0)
    runtime._reconnect_agent = lambda _role: runtime.agents.update(executor=resumed)

    outcome = _run_steps(runtime, _trace(), resume_payload=resume_payload)

    assert outcome["state"] == "completed"
    assert initial.received == [resume_payload]
    assert resumed.received == [resume_payload]


def test_role_steps_do_not_converge_on_consecutive_tool_failures():
    frames = [
        {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="temporary failure",
                        name="get_track",
                        tool_call_id=f"call-{index}",
                        status="error",
                    )
                ]
            }
        }
        for index in range(4)
    ]
    frames.append({"model": {"messages": [AIMessage(content="已完成", id="after-errors")]}})
    trace = _trace()
    runtime = _runtime_with_agent(_FakeAgent(frames))
    outcome = _run_steps(runtime, trace)
    assert outcome["state"] == "completed"
    assert len(trace.records) == 4
    assert trace.records[0]["status"] == "error"


def test_role_steps_do_not_converge_on_repeated_tool_requests():
    """重复调用的阶段级收敛保护已移除；Runtime 消费完整个 Agent stream。"""
    frames = []
    for index in range(6):
        frames.append(
            {
                "model": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "read_file",
                                    "args": {"file_path": "/skills/track/query/SKILL.md"},
                                    "id": f"read-{index}",
                                }
                            ],
                        )
                    ]
                }
            }
        )
        frames.append(
            {
                "tools": {
                    "messages": [
                        ToolMessage(content="ok", name="read_file", tool_call_id=f"read-{index}")
                    ]
                }
            }
        )
    frames.append({"model": {"messages": [AIMessage(content="最终回答", id="answer-done")]}})
    trace = _trace()
    runtime = _runtime_with_agent(_FakeAgent(frames))
    outcome = _run_steps(runtime, trace)
    assert outcome["state"] == "completed"
    reads = [record for record in trace.records if record["tool"] == "read_file"]
    assert len(reads) == 6


def test_resumed_thread_re_emits_skill_events_from_the_checkpoint():
    """续接会话时框架不再发 skills_metadata，运行时要从检查点把技能回执补回事件流。"""
    events = []
    trace = _Trace(events.append, event_limit=4000, shared_state=SharedTaskState("问题"))
    seen = {}

    class _AgentWithState:
        def get_state(self, config):
            seen.update(config)
            values = {
                "skills_metadata": [
                    {"name": "query", "description": "查询"},
                    {"name": "finalize", "description": "收尾"},
                ]
            }
            return type("_Snapshot", (), {"values": values})()

    runtime = _runtime_with_agent(_FakeAgent([]))
    runtime.agents = {"executor": _AgentWithState()}

    runtime._seed_skills_from_checkpoint(trace, "thread-resume")

    assert [event["skill"] for event in events if event["type"] == "skill"] == [
        "query",
        "finalize",
    ]
    # 补发要读执行角色自己的检查点，线程名不是会话 id
    assert seen == {"configurable": {"thread_id": "thread-resume:executor"}}


def test_skill_seeding_tolerates_an_agent_without_state_snapshot():
    trace = _Trace(None, event_limit=4000, shared_state=SharedTaskState("问题"))
    _runtime_with_agent(_FakeAgent([]))._seed_skills_from_checkpoint(trace, "thread-fresh")

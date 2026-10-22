"""人工确认流：写入前中断、事件契约、批准/拒绝后的恢复。

写入工具本身另测；这里测链路——中断有没有被认出来（而不是被当成失败吞掉）、
恢复有没有把 Command 交回那个被中断的角色。中断检测在单个角色的执行里，
所以桩只需要一个能吐帧的假 agent。
"""
from langchain_core.messages import AIMessage
from langgraph.types import Command, Interrupt

from harness.runtime import (
    SeaVideoHarness,
    _confirmation_from,
    _interrupt_payload,
    _resume_command,
    _Trace,
)
from harness.shared_state import SharedTaskState


class _FakeAgent:
    """按脚本吐帧的假 agent，并把每次收到的输入记下来供断言。"""

    def __init__(self, frames):
        self.frames = list(frames)
        self.received = []

    def stream(self, payload, config=None, stream_mode=None, subgraphs=False):
        self.received.append(payload)
        return iter(self.frames)

    def get_state(self, _config):
        raise RuntimeError("no checkpoint")


class _Store:
    def save(self, *_args, **_kwargs):
        pass


def _interrupt_frame():
    return {"__interrupt__": (Interrupt(
        value={
            "action_requests": [{
                "name": "add_registry_vessel",
                "args": {"hull_number": "003", "user_intent": "把 003 加进库里"},
                "description": "写入先验库：003",
            }],
            "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
        },
        id="interrupt-1",
    ),)}


def _runtime_with_agent(agent):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {
        "harness": {
            "stream_mode": "updates",
            "event_payload_max_chars": 4000,
            "evidence_tool": "show_evidence",
            "execution_mode": "evidence-state-multi-agent",
            "output": {"answer_field": "answer", "state_field": "state", "evidence_field": "evidence"},
        },
        "tools": [{"name": "add_registry_vessel", "label": "先验库写入"}],
    }
    runtime.agents = {"executor": agent}
    runtime.agent_specs = {
        "executor": {"task_prompt": "用户问题：{question}，按 Executor 职责处理。"}
    }
    runtime.shared_state = SharedTaskState("把 003 加进库里")
    runtime.state_store = _Store()
    runtime.event_handler = None
    runtime._connection = None
    runtime._released = False
    runtime._pending_agent = "executor"
    return runtime


def _run_steps(runtime, trace, payload=None):
    driver = runtime._run_agent_steps("executor", "把 003 加进库里", "session-x", trace, None, payload)
    while True:
        try:
            next(driver)
        except StopIteration as stopped:
            return stopped.value


def _trace():
    return _Trace(None, event_limit=4000)


# ---------------------------------------------------------------- 载荷解析


def test_interrupt_payload_is_recognised_and_translated():
    interrupts = _interrupt_payload(_interrupt_frame())
    assert len(interrupts) == 1
    confirmation = _confirmation_from(interrupts)
    assert confirmation["interruptId"] == "interrupt-1"
    action = confirmation["actions"][0]
    assert action["tool"] == "add_registry_vessel"
    assert action["arguments"]["hull_number"] == "003"
    assert action["allowedDecisions"] == ["approve", "reject"]


def test_a_normal_frame_carries_no_interrupt():
    assert _interrupt_payload({"model": {"messages": []}}) == []


def test_resume_command_shapes():
    approve = _resume_command("approve", "")
    assert isinstance(approve, Command)
    assert approve.resume == {"decisions": [{"type": "approve"}]}

    reject = _resume_command("reject", "舷号写错了，应该是 0123")
    decision = reject.resume["decisions"][0]
    assert decision["type"] == "reject"
    # 拒绝理由必须原样带给模型，它要靠这个换做法
    assert "0123" in decision["message"]


# ---------------------------------------------------------------- 中断路径


def test_an_interrupt_pauses_the_role_instead_of_failing_it():
    """中断不是失败：本角色以 awaiting_confirmation 收尾，共享状态记下停在谁身上。"""
    trace = _trace()
    runtime = _runtime_with_agent(_FakeAgent([_interrupt_frame()]))
    outcome = _run_steps(runtime, trace)

    assert outcome["state"] == "awaiting_confirmation"
    assert outcome["confirmation"]["actions"][0]["tool"] == "add_registry_vessel"
    assert runtime.shared_state.snapshot()["pending_agent"] == "executor"
    assert runtime.shared_state.snapshot()["task_status"] == "awaiting_confirmation"


def test_resume_sends_the_command_to_the_interrupted_role():
    """恢复时把 Command 原样交给那个角色，而不是重新提问。"""
    agent = _FakeAgent([{"model": {"messages": [AIMessage(content="已写入 003。", id="a1")]}}])
    runtime = _runtime_with_agent(agent)
    _run_steps(runtime, _trace(), payload=_resume_command("approve", ""))
    assert agent.received[0].resume == {"decisions": [{"type": "approve"}]}


def test_rejection_carries_the_feedback_to_the_model():
    agent = _FakeAgent([{"model": {"messages": [AIMessage(content="那我改成查库。", id="a1")]}}])
    runtime = _runtime_with_agent(agent)
    _run_steps(runtime, _trace(), payload=_resume_command("reject", "这条不该入库，先查它在不在库里"))
    decision = agent.received[0].resume["decisions"][0]
    assert decision["type"] == "reject"
    assert "先查它在不在库里" in decision["message"]


def test_resume_rejects_an_unknown_decision():
    runtime = _runtime_with_agent(_FakeAgent([]))
    try:
        runtime.resume("maybe", thread_id="session-x")
    except ValueError as error:
        assert "approve" in str(error)
    else:
        raise AssertionError("非法 decision 没有被拦下")


def test_resume_rebuilds_agents_from_persisted_state(tmp_path):
    """人工确认恢复只读持久化的角色与状态，不依赖旧实例的私有标记。"""
    from harness.shared_state import SharedStateStore

    runtime = _runtime_with_agent(_FakeAgent([]))
    runtime.state_store = SharedStateStore(tmp_path / "state.sqlite")
    runtime.shared_state.runtime_update({"task_status": "awaiting_confirmation", "pending_agent": "executor"})
    runtime.state_store.save("session-x", runtime.shared_state)
    rebuilt = {"count": 0}

    def _rebuild():
        rebuilt["count"] += 1
        runtime.agents = {"executor": _FakeAgent([])}

    runtime._build_agents = _rebuild  # type: ignore[method-assign]
    runtime._initialize_state("session-x", resume=True)
    assert rebuilt["count"] == 1
    assert runtime.shared_state.snapshot()["pending_agent"] == "executor"

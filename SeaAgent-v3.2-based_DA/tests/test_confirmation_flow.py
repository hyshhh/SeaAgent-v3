"""人工确认流：写入前中断、事件契约、批准/拒绝后的恢复。

写入工具本身另测；这里测链路——中断有没有被认出来（而不是被当成失败吞掉）、
恢复有没有带上正确的命令与同一个 thread_id。
"""
import threading

from langgraph.types import Command

from harness.runtime import SeaVideoHarness, _confirmation_from, _interrupt_payload, _resume_command


class _FakeAgent:
    """按脚本吐帧的假 agent，并把每次收到的输入记下来供断言。"""

    def __init__(self, frames, raise_on_first=False):
        self.frames = list(frames)
        self.received = []
        self._first = True

    def stream(self, payload, config=None, stream_mode=None, subgraphs=False):
        self.received.append(payload)
        if self._first and getattr(self, "raise_on_first", False):
            self._first = False
            raise RuntimeError("provider unavailable")
        self._first = False
        return iter(self.frames)

    def get_state(self, _config):
        raise RuntimeError("no checkpoint")


def _interrupt_frame():
    from langgraph.types import Interrupt

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
            "execution_mode": "three-phase-subagents",
            "output": {"answer_field": "answer", "state_field": "state", "evidence_field": "evidence"},
        },
        "tools": [{"name": "add_registry_vessel", "label": "先验库写入"}],
    }
    runtime.agent = agent
    runtime.event_handler = None
    runtime._connection = None
    runtime._released = False
    runtime.subagents = []
    return runtime


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


def test_an_interrupt_pauses_the_turn_instead_of_failing_it():
    """中断不是失败：本轮以 awaiting_confirmation 收尾，并发一条 confirm 事件。"""
    events = []
    runtime = _runtime_with_agent(_FakeAgent([_interrupt_frame()]))
    runtime.event_handler = events.append
    result = runtime.run("把 003 加进库里", thread_id="session-x")

    assert result["state"] == "awaiting_confirmation"
    assert "error" not in result
    assert result["confirmation"]["actions"][0]["tool"] == "add_registry_vessel"
    confirm_events = [event for event in events if event["type"] == "confirm"]
    assert len(confirm_events) == 1
    assert confirm_events[0]["confirmation"]["interruptId"] == "interrupt-1"
    assert all(event["type"] != "error" for event in events)


def test_resume_sends_a_command_on_the_same_thread():
    """恢复用的是同一个 thread_id，且输入是 Command —— 不是重新提问。"""
    agent = _FakeAgent([{"model": {"messages": [__import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(content="已写入 003。", id="a1")]}}])
    runtime = _runtime_with_agent(agent)
    result = runtime.resume("approve", thread_id="session-x")
    assert agent.received[0].resume == {"decisions": [{"type": "approve"}]}
    assert result["state"] == "completed"
    assert result["answer"] == "已写入 003。"


def test_rejection_carries_the_feedback_back_to_the_model():
    agent = _FakeAgent([{"model": {"messages": [__import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(content="那我改成查库。", id="a1")]}}])
    runtime = _runtime_with_agent(agent)
    runtime.resume("reject", thread_id="session-x", feedback="这条不该入库，先查它在不在库里")
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


def test_resume_rebuilds_the_agent_after_the_turn_released_it():
    """上一轮结束时连接已释放，恢复必须重新装配——否则 agent 是关着的。"""
    runtime = _runtime_with_agent(_FakeAgent([{"model": {"messages": []}}]))
    rebuilt = {"count": 0}

    def _build():
        rebuilt["count"] += 1
        return _FakeAgent([{"model": {"messages": []}}])

    runtime._build_agent = _build  # type: ignore[method-assign]
    runtime._released = True
    runtime._resume_agent()
    assert rebuilt["count"] == 1
    assert runtime._released is False

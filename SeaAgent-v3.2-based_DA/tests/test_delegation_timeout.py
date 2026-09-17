"""单次委派超时：卡住的委派要能被中止，而且说得出卡在谁身上。"""
import time

from langchain_core.messages import AIMessage, ToolMessage

from harness.runtime import SeaVideoHarness


class _SlowAgent:
    """第一帧是委派，之后「卡住」：每帧之间睡够 timeout 才继续。"""

    def __init__(self, frames, gap=0.0):
        self.frames = list(frames)
        self.gap = gap
        self.received = []

    def stream(self, payload, config=None, stream_mode=None, subgraphs=False):
        self.received.append(payload)

        def gen():
            for index, frame in enumerate(self.frames):
                if index and self.gap:
                    time.sleep(self.gap)
                yield frame

        return gen()

    def get_state(self, _config):
        raise RuntimeError("no checkpoint")


def _runtime(agent, **harness_overrides):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    settings = {
        "stream_mode": "updates",
        "event_payload_max_chars": 4000,
        "evidence_tool": "show_evidence",
        "execution_mode": "three-phase-subagents",
        "output": {"answer_field": "answer", "state_field": "state", "evidence_field": "evidence"},
    }
    settings.update(harness_overrides)
    runtime.config = {"harness": settings, "tools": []}
    runtime.agent = agent
    runtime.event_handler = None
    runtime._connection = None
    runtime._released = False
    runtime.subagents = []
    return runtime


def _delegation_frame(subagent, call_id="t1"):
    return {"model": {"messages": [AIMessage(content="", tool_calls=[{
        "name": "task", "args": {"subagent_type": subagent, "description": "查轨迹"}, "id": call_id,
    }])]}}


def test_a_stalled_delegation_is_aborted_and_named():
    """卡住时：以 delegation_timeout 收尾、不是 error，并点名卡在哪个子智能体。"""
    events = []
    agent = _SlowAgent([_delegation_frame("executor"), {"ignored": True}], gap=1.2)
    runtime = _runtime(agent, delegation_timeout_seconds=1)
    runtime.event_handler = events.append
    result = runtime.run("查一下", thread_id="session-slow")

    assert result["state"] == "delegation_timeout"
    assert "error" not in result
    # 卡在哪一步要说清楚，否则用户不知道该缩哪一段范围
    assert "executor" in result["answer"]
    assert "全量扫描" in result["answer"]
    assert any(event["type"] == "status" and "超时" in event["title"] for event in events)
    assert all(event["type"] != "error" for event in events)


def test_a_slow_but_moving_delegation_is_not_aborted():
    """每一步都在动就不该被判超时：VLM 与嵌入调用本来就慢。

    间隔取 0.2 秒（阈值 1 秒），参数每步都不同 —— 同参数会被重复调用守卫拦下，
    那是另一条规则，不该混进这个用例。
    """
    frames = [_delegation_frame("executor")]
    for index in range(6):
        frames.append({"model": {"messages": [AIMessage(content="", tool_calls=[{
            "name": "get_track", "args": {"offset": index, "limit": 5}, "id": f"c{index}",
        }])]}})
        frames.append({"tools": {"messages": [ToolMessage(content='{"ok": true, "trackCount": 3}', name="get_track", tool_call_id=f"c{index}")]}})
    frames.append({"model": {"messages": [AIMessage(content="查到了 3 条轨迹。", id="a1")]}})

    agent = _SlowAgent(frames, gap=0.2)
    runtime = _runtime(agent, delegation_timeout_seconds=1)
    result = runtime.run("查一下", thread_id="session-moving")
    assert result["state"] == "completed"
    assert result["answer"] == "查到了 3 条轨迹。"


def test_timeout_zero_disables_the_watchdog():
    """配 0 表示不设单步上限：同一个慢间隔在阈值 1 秒时会中止，配 0 时不中止。"""
    frames = [_delegation_frame("planner"), {"ignored": True}]
    aborted = _runtime(_SlowAgent(frames, gap=1.2), delegation_timeout_seconds=1).run("查一下", thread_id="session-a")
    assert aborted["state"] == "delegation_timeout"

    allowed = _runtime(_SlowAgent(frames, gap=1.2), delegation_timeout_seconds=0).run("查一下", thread_id="session-b")
    assert allowed["state"] == "completed"

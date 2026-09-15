"""心跳：慢步骤期间前端不该是静止的。"""
import time

from config import load_config
from harness.runtime import SeaVideoHarness
from langchain_core.messages import AIMessage


class _SlowAgent:
    def stream(self, *_args, **_kwargs):
        time.sleep(0.5)
        yield {"model": {"messages": [AIMessage(content="回答", id="a1")]}}
    def get_state(self, _config):
        raise RuntimeError("no snapshot")


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"ok": True}


def test_heartbeat_reports_progress_while_a_step_is_running():
    config = load_config()
    config["harness"]["heartbeat_seconds"] = 0.2
    events = []
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = config
    runtime.agent = _SlowAgent()
    runtime.event_handler = events.append
    runtime._connection = None
    runtime.subagents = []
    runtime.tools = []

    result = runtime.run("问题", thread_id="t")
    assert result["answer"] == "回答"
    beats = [event for event in events if event.get("title") == "仍在执行"]
    assert beats, "慢步骤期间应有心跳"
    assert "已等待" in beats[0]["message"]


def test_heartbeat_can_be_disabled():
    config = load_config()
    config["harness"]["heartbeat_seconds"] = 0
    events = []
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = config
    runtime.agent = _SlowAgent()
    runtime.event_handler = events.append
    runtime._connection = None
    runtime.subagents = []
    runtime.tools = []

    runtime.run("问题", thread_id="t")
    assert not [event for event in events if event.get("title") == "仍在执行"]

"""超时预算与心跳：一轮问答必须能给前端一个结局，不能永远静止。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from web.models import AgentQuery
from web.routes import agent_api


class _HangingController:
    """模拟真实故障：工具调用一直不返回。"""

    def __init__(self, event_handler=None):
        self.event_handler = event_handler

    def answer(self, _question, _session_id=None, _cancel=None):
        import time
        time.sleep(30)
        return {"success": True}


async def _read_events(response):
    return [json.loads(chunk) async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_stream_gives_up_after_the_wall_clock_budget(monkeypatch):
    """没有这条预算时，worker 线程卡在工具里，整条 NDJSON 流会永久静止。"""
    monkeypatch.setattr(agent_api, "_controller", lambda _request, event_handler=None: _HangingController(event_handler))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent_runs={}, config={"harness": {"run_timeout_seconds": 0.3}})))

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), request)
    events = await _read_events(response)

    assert [event["type"] for event in events] == ["error"]
    assert events[-1]["title"] == "Harness 已超时"
    assert "秒" in events[-1]["message"]
    assert request.app.state.agent_runs == {}, "运行登记要被清掉"


@pytest.mark.asyncio
async def test_missing_config_still_produces_a_terminal_event(monkeypatch):
    """预算读取本身出错时也必须发出终止事件，否则事件队列无人出队，流会永久等待。"""
    monkeypatch.setattr(agent_api, "_controller", lambda _request, event_handler=None: _HangingController(event_handler))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent_runs={})))  # 故意不给 config

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), request)
    events = await _read_events(response)

    assert events and events[-1]["type"] == "error"

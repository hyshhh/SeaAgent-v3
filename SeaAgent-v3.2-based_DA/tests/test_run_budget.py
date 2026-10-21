"""墙钟预算触发收敛，前端等待实际结果而不是将预算耗尽当错误。"""

import json
from types import SimpleNamespace

import pytest

from web.models import AgentQuery
from web.routes import agent_api


class _HangingController:
    """模拟工具超过探索预算后返回，由运行时进行收敛。"""

    def __init__(self, event_handler=None):
        self.event_handler = event_handler

    def answer(self, _question, _session_id=None, _cancel=None):
        import time

        time.sleep(0.5)
        return {"success": True, "state": "completed", "answerText": "partial evidence"}


async def _read_events(response):
    return [json.loads(chunk) async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_stream_has_no_overall_wall_clock_budget(monkeypatch):
    """整轮问答不因墙钟时间自动请求收敛。"""
    monkeypatch.setattr(
        agent_api,
        "_controller",
        lambda _request, event_handler=None: _HangingController(event_handler),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(agent_runs={}, config={"harness": {}})
        )
    )

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), request)
    iterator = response.body_iterator
    final = json.loads(await anext(iterator))
    assert final["type"] == "complete"
    assert final["result"]["answerText"] == "partial evidence"
    await iterator.aclose()
    assert request.app.state.agent_runs == {}


@pytest.mark.asyncio
async def test_missing_config_still_produces_a_terminal_event(monkeypatch):
    """没有 Harness 配置时仍可正常完成流式问答。"""
    monkeypatch.setattr(
        agent_api,
        "_controller",
        lambda _request, event_handler=None: _HangingController(event_handler),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(agent_runs={}))
    )  # 故意不给 config

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), request)
    events = await _read_events(response)

    assert events and events[-1]["type"] == "complete"

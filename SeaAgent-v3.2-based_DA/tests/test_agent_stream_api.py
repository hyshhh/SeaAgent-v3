import json
from types import SimpleNamespace

import pytest

from web.models import AgentQuery
from web.routes import agent_api


class _FakeController:
    def __init__(self, event_handler=None, result=None):
        self.event_handler = event_handler
        self.result = result

    def answer(self, _question, _session_id=None, _cancel=None):
        self.event_handler({"type": "status", "title": "Harness 已启动"})
        if self.result.get("state") == "cancelled":
            # 被停止的一轮由接口层兜底收尾，运行期不发终局事件
            return self.result
        if self.result["success"]:
            self.event_handler({"type": "complete", "title": "Harness 完成", "result": self.result})
        else:
            self.event_handler({
                "type": "error",
                "title": "Harness 执行失败",
                "message": self.result["error"],
                "result": self.result,
            })
        return self.result


def _request():
    """最小请求桩：流式接口要往 app.state.agent_runs 登记进行中的运行。"""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent_runs={})))


async def _read_events(response):
    return [json.loads(chunk) async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_stream_does_not_duplicate_runtime_error(monkeypatch):
    result = {"success": False, "state": "error", "error": "provider unavailable"}
    monkeypatch.setattr(agent_api, "_controller", lambda _request, event_handler=None: _FakeController(event_handler, result))
    request = _request()

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), request)
    events = await _read_events(response)

    assert [event["type"] for event in events] == ["status", "error"]
    assert events[-1]["message"] == "provider unavailable"
    assert events[-1]["result"] == result
    assert request.app.state.agent_runs == {}


@pytest.mark.asyncio
async def test_stream_does_not_duplicate_runtime_complete(monkeypatch):
    result = {"success": True, "state": "completed", "answerText": "回答"}
    monkeypatch.setattr(agent_api, "_controller", lambda _request, event_handler=None: _FakeController(event_handler, result))

    response = await agent_api.stream_agent_query(AgentQuery(question="问题"), _request())
    events = await _read_events(response)

    assert [event["type"] for event in events] == ["status", "complete"]
    assert events[-1]["result"] == result


@pytest.mark.asyncio
async def test_stream_reports_a_cancelled_run_as_complete(monkeypatch):
    """被停止的一轮不是错误：前端据此收尾，而不是弹一条失败提示。"""
    result = {"success": False, "state": "cancelled", "answerText": ""}
    monkeypatch.setattr(agent_api, "_controller", lambda _request, event_handler=None: _FakeController(event_handler, result))

    response = await agent_api.stream_agent_query(AgentQuery(question="问题", sessionId="session-stop"), _request())
    events = await _read_events(response)

    assert [event["type"] for event in events] == ["status", "complete"]
    assert events[-1]["title"] == "Harness 已停止"
    assert events[-1]["result"] == result

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


def _request(config=None):
    """最小请求桩：流式接口要往 app.state.agent_runs 登记运行，也要读 harness 配置。"""
    state = SimpleNamespace(agent_runs={}, config=config or {"harness": {}})
    return SimpleNamespace(app=SimpleNamespace(state=state))


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


@pytest.mark.asyncio
async def test_confirmation_is_a_terminal_event_without_a_following_error(monkeypatch):
    class Controller:
        def __init__(self, emit):
            self.emit = emit

        def answer(self, *args):
            self.emit({"type": "confirm", "confirmation": {"actions": [{"tool": "add_registry_vessel"}]}})
            return {"success": False, "state": "awaiting_confirmation"}

    monkeypatch.setattr(agent_api, "_controller", lambda request, event_handler=None: Controller(event_handler))
    response = await agent_api.stream_agent_query(AgentQuery(question="q"), _request())
    events = await _read_events(response)
    assert [event["type"] for event in events] == ["confirm"]


@pytest.mark.asyncio
async def test_same_session_cannot_start_two_workers():
    from fastapi import HTTPException
    request = _request()
    first_cancel = agent_api._begin_run(request, "same")
    with pytest.raises(HTTPException) as error:
        await agent_api.stream_agent_query(AgentQuery(question="q", sessionId="same"), request)
    assert error.value.status_code == 409
    assert request.app.state.agent_runs["same"] is first_cancel


@pytest.mark.asyncio
async def test_resume_endpoint_passes_decision_and_feedback(monkeypatch):
    from web.models import AgentResume
    calls = []

    class Controller:
        def resume(self, *args):
            calls.append(args)
            return {"success": True, "state": "completed"}

    monkeypatch.setattr(agent_api, "_controller", lambda request: Controller())
    request = _request()
    result = await agent_api.resume_agent_query(AgentResume(sessionId="same", decision="reject", feedback="wrong number"), request)
    assert calls[0][:3] == ("reject", "same", "wrong number")
    assert result["state"] == "completed"
    assert request.app.state.agent_runs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "delete"])
async def test_cleanup_blocks_new_runs_until_it_finishes(monkeypatch, action):
    import asyncio

    from fastapi import HTTPException

    started, release = asyncio.Event(), asyncio.Event()
    request = _request()
    request.app.state.memory_manager = SimpleNamespace(clear_qa_memory=dict)
    request.app.state.repository = SimpleNamespace(
        get_session=lambda session_id: {"sessionId": session_id, "turns": []},
        delete_session=lambda session_id: True,
    )

    async def cleanup(call, *args):
        started.set()
        await release.wait()
        return call(*args)

    monkeypatch.setattr(agent_api, "run_in_threadpool", cleanup)
    monkeypatch.setattr(agent_api, "_delete_runtime_state", lambda *args: None)
    operation = agent_api.clear_agent_memory(request) if action == "clear" else agent_api.delete_agent_session("t", request)
    task = asyncio.create_task(operation)
    await started.wait()
    try:
        with pytest.raises(HTTPException) as error:
            agent_api._begin_run(request, "t")
        assert error.value.status_code == 409
    finally:
        release.set()
        await task
    assert request.app.state.agent_runs == {}
    assert not getattr(request.app.state, "agent_memory_clearing", False)

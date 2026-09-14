"""技能披露：目录永远在提示词里，正文要靠模型自己 read_file；忘了读时补一句提醒。"""
from langchain.agents.middleware import ModelRequest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from harness.disclosure import SkillDisclosureMiddleware


class _Handler:
    """记录最后一次请求，模拟下层模型调用。"""

    def __init__(self):
        self.request = None

    def __call__(self, request):
        self.request = request
        return "ok"


def _request(messages, system="SYSTEM"):
    return ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        system_message=SystemMessage(content=system),
        messages=list(messages),
        tools=[],
        state={"messages": list(messages)},
        model_settings={},
    )


def _system_text(request):
    return "".join(block.get("text", "") for block in request.system_message.content_blocks)


def test_reminder_is_added_before_the_first_model_call():
    middleware = SkillDisclosureMiddleware()
    handler = _Handler()
    middleware.wrap_model_call(_request([HumanMessage(content="有几艘船？")]), handler)
    assert "has not read any skill" in _system_text(handler.request)
    assert "SYSTEM" in _system_text(handler.request)


def test_no_reminder_once_a_skill_body_has_been_read():
    read = ToolMessage(content="1  ---\n 2  name: query", name="read_file", tool_call_id="c1")
    handler = _Handler()
    SkillDisclosureMiddleware().wrap_model_call(_request([HumanMessage(content="问"), read]), handler)
    assert "has not read any skill" not in _system_text(handler.request)


def test_failed_read_does_not_count_as_disclosure():
    failed = ToolMessage(content="Error: permission denied", name="read_file", tool_call_id="c1", status="error")
    handler = _Handler()
    SkillDisclosureMiddleware().wrap_model_call(_request([HumanMessage(content="问"), failed]), handler)
    assert "has not read any skill" in _system_text(handler.request)


def test_reminder_is_sent_at_most_once_per_run():
    middleware = SkillDisclosureMiddleware(max_reminders=1)
    first, second = _Handler(), _Handler()
    middleware.wrap_model_call(_request([HumanMessage(content="问")]), first)
    middleware.wrap_model_call(_request([HumanMessage(content="追问")]), second)
    assert "has not read any skill" in _system_text(first.request)
    assert "has not read any skill" not in _system_text(second.request)


def test_reminder_can_be_disabled():
    handler = _Handler()
    SkillDisclosureMiddleware(max_reminders=0).wrap_model_call(_request([HumanMessage(content="问")]), handler)
    assert "has not read any skill" not in _system_text(handler.request)

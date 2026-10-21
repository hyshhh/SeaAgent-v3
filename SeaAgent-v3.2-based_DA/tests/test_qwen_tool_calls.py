from types import SimpleNamespace

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from harness.qwen_tool_calls import QwenXmlToolCallMiddleware, parse_qwen_xml_tool_calls


def _read_tool():
    return StructuredTool.from_function(
        lambda: {"version": 1}, name="read_shared_state", description="read state"
    )


def _update_tool():
    def update(patch: dict, expected_version: int):
        return {"ok": True}

    return StructuredTool.from_function(update, name="update_planner_state", description="update")


def test_parse_qwen_xml_tool_call_without_arguments():
    calls = parse_qwen_xml_tool_calls(
        "<tool_call>\n<function=read_shared_state> </function>\n</tool_call>",
        {"read_shared_state"},
        [_read_tool()],
    )
    assert calls and calls[0]["name"] == "read_shared_state" and calls[0]["args"] == {}


def test_parse_qwen_xml_tool_call_decodes_parameter_values():
    calls = parse_qwen_xml_tool_calls(
        """<tool_call><function=update_planner_state>
        <parameter=patch>{\"plan\":[{\"id\":\"p1\"}]}</parameter>
        <parameter=expected_version>3</parameter>
        </function></tool_call>""",
        {"update_planner_state"},
        [_update_tool()],
    )
    assert calls and calls[0]["args"] == {
        "patch": {"plan": [{"id": "p1"}]},
        "expected_version": 3,
    }


def test_parser_rejects_unknown_or_mixed_text():
    block = "<tool_call><function=delete_everything></function></tool_call>"
    assert parse_qwen_xml_tool_calls(block, {"read_shared_state"}) is None
    assert parse_qwen_xml_tool_calls("先思考\n" + block, {"delete_everything"}) is None


def test_middleware_rewrites_xml_into_native_tool_calls():
    middleware = QwenXmlToolCallMiddleware({"read_shared_state"})
    request = SimpleNamespace(tools=[_read_tool()])
    response = middleware.wrap_model_call(
        request,
        lambda _request: ModelResponse(
            result=[
                AIMessage(
                    content="<tool_call><function=read_shared_state></function></tool_call>"
                )
            ]
        ),
    )
    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert message.content == ""
    assert message.tool_calls[0]["name"] == "read_shared_state"


def test_middleware_retries_unknown_xml_tool_with_available_names():
    class Request:
        def __init__(self, tools, messages=None):
            self.tools = tools
            self.messages = messages or []

        def override(self, **updates):
            return Request(
                updates.get("tools", self.tools), updates.get("messages", self.messages)
            )

    requests = []

    def handler(request):
        requests.append(request)
        content = (
            "<tool_call><function=query><parameter=time_range>[0, 1000000]"
            "</parameter></function></tool_call>"
            if len(requests) == 1
            else "<tool_call><function=read_shared_state></function></tool_call>"
        )
        return ModelResponse(result=[AIMessage(content=content)])

    request = Request([_read_tool()])
    response = QwenXmlToolCallMiddleware({"read_shared_state"}).wrap_model_call(
        request, handler
    )

    assert len(requests) == 2
    retry_messages = requests[1].messages
    assert isinstance(retry_messages[-2], AIMessage)
    assert isinstance(retry_messages[-1], HumanMessage)
    assert "query" in retry_messages[-1].content
    assert "read_shared_state" in retry_messages[-1].content
    assert isinstance(response.result[-2], HumanMessage)
    assert "read_shared_state" in response.result[-2].content
    assert response.result[-1].tool_calls[0]["name"] == "read_shared_state"


def test_middleware_explains_when_retry_still_requests_unknown_tool():
    class Request:
        def __init__(self):
            self.tools = [_read_tool()]
            self.messages = []

        def override(self, **updates):
            return self

    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[AIMessage(content="<tool_call><function=query></function></tool_call>")]
        )

    response = QwenXmlToolCallMiddleware({"read_shared_state"}).wrap_model_call(
        Request(), handler
    )

    assert calls == 2
    assert "工具调用未执行" in response.result[-1].content
    assert "query" in response.result[-1].content
    assert not response.result[-1].tool_calls

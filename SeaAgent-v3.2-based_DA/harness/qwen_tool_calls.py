"""Normalize Qwen XML tool-call text returned by OpenAI-compatible model servers."""

from __future__ import annotations

import html
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage

from .thinking_text import strip_thinking

_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)\s*>(.*?)</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([\w.-]+)\s*>(.*?)</parameter>", re.IGNORECASE | re.DOTALL
)


def _tool_schema(tool: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(tool, dict):
        function = tool.get("function", tool)
        return str(function.get("name") or ""), function.get("parameters") or {}

    name = str(getattr(tool, "name", "") or "")
    schema = getattr(tool, "args_schema", None)
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    return name, schema if isinstance(schema, dict) else {}


def _decode_value(value: str, schema: dict[str, Any]) -> Any:
    text = html.unescape(value).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text

    value_type = schema.get("type")
    if value_type == "string" and not isinstance(parsed, str):
        return text
    return parsed


def _parse_arguments(body: str, properties: dict[str, Any]) -> dict[str, Any] | None:
    arguments: dict[str, Any] = {}
    position = 0
    for match in _PARAMETER.finditer(body):
        if body[position : match.start()].strip():
            return None
        name = match.group(1)
        if name in arguments:
            return None
        schema = properties.get(name)
        arguments[name] = _decode_value(
            match.group(2), schema if isinstance(schema, dict) else {}
        )
        position = match.end()
    if body[position:].strip():
        return None
    return arguments


def parse_qwen_xml_tool_calls(
    content: str, allowed_tools: set[str], tools: list[Any] | None = None
) -> list[dict[str, Any]] | None:
    """Parse a response made entirely of allowed ``<tool_call>`` blocks."""
    text = strip_thinking(content).strip()
    matches = list(_TOOL_CALL.finditer(text))
    if not matches or _TOOL_CALL.sub("", text).strip():
        return None

    schemas = {
        name: schema
        for tool in tools or []
        for name, schema in [_tool_schema(tool)]
        if name
    }
    calls = []
    for match in matches:
        name = match.group(1)
        if name not in allowed_tools:
            return None
        properties = schemas.get(name, {}).get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        arguments = _parse_arguments(match.group(2), properties)
        if arguments is None:
            return None
        calls.append(
            {"name": name, "args": arguments, "id": uuid.uuid4().hex, "type": "tool_call"}
        )
    return calls


class QwenXmlToolCallMiddleware(AgentMiddleware):
    """Convert Qwen's XML tool-call text into native LangChain tool calls."""

    def __init__(self, allowed_tools: set[str] | None = None) -> None:
        self.allowed_tools = frozenset(allowed_tools or ())

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        response = handler(request)
        return self._normalize_response(response, request)

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        response = await handler(request)
        return self._normalize_response(response, request)

    def _normalize_response(self, response: Any, request: Any) -> Any:
        if not isinstance(response, ModelResponse) or not response.result:
            return response

        message = response.result[-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return response
        if not isinstance(message.content, str) or not self.allowed_tools:
            return response

        requested_tools = request.tools or []
        requested_names = {name for tool in requested_tools if (name := _tool_schema(tool)[0])}
        available_tools = set(self.allowed_tools) & requested_names
        calls = parse_qwen_xml_tool_calls(message.content, available_tools, requested_tools)
        if calls is None:
            return response

        result = list(response.result)
        result[-1] = message.model_copy(
            update={"content": "", "tool_calls": calls, "invalid_tool_calls": []}
        )
        return ModelResponse(result=result, structured_response=response.structured_response)

"""Skill reminder injected into the system message before the first domain call."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

DEFAULT_REMINDER = (
    "This session has not read any skill. Before domain tools, use read_file to read the relevant "
    "skill files; they define query scope, evidence and completion rules."
)


class SkillDisclosureMiddleware(AgentMiddleware):
    def __init__(self, reminder: str = DEFAULT_REMINDER, read_tool: str = "read_file", max_reminders: int = 1) -> None:
        self.reminder = reminder
        self.read_tool = read_tool
        self.max_reminders = max(0, int(max_reminders))
        self._reminders = 0

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if self._should_remind(request):
            self._reminders += 1
            request = request.override(system_message=self._with_reminder(getattr(request, "system_message", None)))
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        if self._should_remind(request):
            self._reminders += 1
            request = request.override(system_message=self._with_reminder(getattr(request, "system_message", None)))
        return await handler(request)

    def _should_remind(self, request: Any) -> bool:
        if self.max_reminders == 0 or self._reminders >= self.max_reminders:
            return False
        state = getattr(request, "state", None)
        messages = state.get("messages") if isinstance(state, dict) else None
        return not self._skill_read(messages or [])

    def _skill_read(self, messages: list[Any]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.read_tool
            and str(getattr(message, "status", "success")) != "error"
            for message in messages
        )

    def _with_reminder(self, system_message: Any) -> SystemMessage:
        if system_message is None:
            return SystemMessage(content=self.reminder)
        blocks = [*system_message.content_blocks, {"type": "text", "text": f"\n\n{self.reminder}"}]
        return SystemMessage(content=blocks)

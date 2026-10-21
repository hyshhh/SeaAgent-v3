"""技能披露提醒：模型还没读过技能正文时，动手前把规范再点一次（走 wrap_model_call，不进消息历史）。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage

DEFAULT_REMINDER = (
    "This session has not read any skill yet. Before running domain tools, read the skills that "
    "apply to the current question with read_file (for example /skills/query/SKILL.md, limit 1000). "
    "Their rules decide how to query, what counts as evidence, and how to finish."
)


# ------------------------------------------------------------------ #
# 一、技能披露提醒                                                           #
# ------------------------------------------------------------------ #
# 1.1 模型动手前把技能规范点一次
class SkillDisclosureMiddleware(AgentMiddleware):

    # 1.1.1 绑定提醒上限
    def __init__(self, reminder: str = DEFAULT_REMINDER, read_tool: str = "read_file", max_reminders: int = 1) -> None:
        self.reminder = reminder
        self.read_tool = read_tool
        self.max_reminders = max(0, int(max_reminders))
        self._reminders = 0

    # 1.1.2 同步注入提醒
    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if self._should_remind(request):
            self._reminders += 1
            request = request.override(system_message=self._with_reminder(getattr(request, "system_message", None)))
        return handler(request)

    # 1.1.3 异步注入提醒
    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        if self._should_remind(request):
            self._reminders += 1
            request = request.override(system_message=self._with_reminder(getattr(request, "system_message", None)))
        return await handler(request)

    # 1.1.4 判断这一轮该不该提醒
    def _should_remind(self, request: Any) -> bool:
        if self.max_reminders == 0 or self._reminders >= self.max_reminders:
            return False
        state = getattr(request, "state", None)
        messages = state.get("messages") if isinstance(state, dict) else None
        return not self._skill_read(messages or [])

    # 1.1.5 会话里是否已读过技能正文
    def _skill_read(self, messages: list[Any]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.read_tool
            and str(getattr(message, "status", "success")) != "error"
            for message in messages
        )

    # 1.1.6 把提醒并进 system 消息末尾
    def _with_reminder(self, system_message: Any) -> SystemMessage:
        if system_message is None:
            return SystemMessage(content=self.reminder)
        blocks = [*system_message.content_blocks, {"type": "text", "text": f"\n\n{self.reminder}"}]
        return SystemMessage(content=blocks)

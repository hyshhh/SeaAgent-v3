"""技能披露提醒：会话里还没读过任何技能正文时，在模型动手前把规范再点一次。

skills 的渐进式披露是「提示词给目录、模型自己 read_file 取正文」。目录一定在提示词里，
但模型可能直接就去调领域工具——那一轮就完全没有规范约束。本模块补一句话提醒：

    · 用 wrap_model_call 追加到 system 消息末尾，不进消息历史。放在 after_model 插话不行：
      那时模型已经给出 tool_calls，跳回模型会留下「有 tool_calls 却没有 tool 结果」的悬空调用，
      OpenAI 兼容端点会直接报错。
    · 判定看整段会话：只要读过一次技能正文（read_file 成功返回），提示词里已经有正文了，
      就不再提醒。
    · 提醒次数按运行实例计数，一次运行最多一句，不额外消耗模型轮次。
"""
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


class SkillDisclosureMiddleware(AgentMiddleware):
    """把「先读技能正文，再动手查」这条规范在模型第一次决策前再说一次。"""

    def __init__(self, reminder: str = DEFAULT_REMINDER, read_tool: str = "read_file", max_reminders: int = 1) -> None:
        self.reminder = reminder
        self.read_tool = read_tool
        self.max_reminders = max(0, int(max_reminders))
        self._reminders = 0

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        """同步钩子：需要时把提醒并进 system 消息，再交给下一层。"""
        if self._should_remind(request):
            self._reminders += 1
            request = request.override(system_message=self._with_reminder(getattr(request, "system_message", None)))
        return handler(request)

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        """异步钩子：与同步版本同义，agent 走异步路径时同样生效。"""
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
        """整段会话里是否已经成功读过一次技能正文。"""
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.read_tool
            and str(getattr(message, "status", "success")) != "error"
            for message in messages
        )

    def _with_reminder(self, system_message: Any) -> SystemMessage:
        """把提醒并到 system 消息末尾；没有 system 消息时它就是全部。"""
        if system_message is None:
            return SystemMessage(content=self.reminder)
        blocks = [*system_message.content_blocks, {"type": "text", "text": f"\n\n{self.reminder}"}]
        return SystemMessage(content=blocks)

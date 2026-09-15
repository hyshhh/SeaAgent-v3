"""重复工具调用守卫：同一个「工具 + 参数」第二次出现时不再执行，改回一条提示。

为什么放在中间件而不是运行时：运行时那一层只能「把整轮停掉」——用户看到的是没有回答；
中间件可以在工具边界拦下这一次调用，并给模型一句可执行的话，让它把活干完。

实测场景：4B 模型会把同一个 SKILL.md 读 8 遍，工具预算耗尽也没委派。拦截 + 提示比直接收尾有用得多。

不拦 ``task``：同一 scope 重复委派由提示词约束，不该在这里被挡（而且委派的参数每次都可能不同）。
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

REPEAT_HINT = (
    "Skipped: you already made this exact call earlier in this turn and its result is in the "
    "conversation above. Do not repeat it — use the result you already have and continue with "
    "the next step."
)


def _signature(name: str, arguments: Any) -> str:
    try:
        return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        return f"{name}:{arguments!r}"


class RepeatToolCallMiddleware(AgentMiddleware):
    """拦截重复的相同调用，返回提示而不是执行。"""

    def __init__(self, exempt: Iterable[str] = ("task",)) -> None:
        self.exempt = {str(name) for name in exempt}
        self._seen: set[str] = set()

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        blocked = self._blocked_message(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        blocked = self._blocked_message(request)
        return blocked if blocked is not None else await handler(request)

    def _blocked_message(self, request: Any) -> ToolMessage | None:
        call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
        name = str(call.get("name") or "")
        if not name or name in self.exempt:
            return None
        key = _signature(name, call.get("args") or {})
        if key not in self._seen:
            self._seen.add(key)
            return None
        # status 用 success：这不是"失败"，只是被跳过；否则会误触发连续失败守卫
        return ToolMessage(content=REPEAT_HINT, tool_call_id=str(call.get("id") or ""), name=name, status="success")

"""Keep every completed agent phase represented in the shared task state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

from .shared_state import ROLE_UPDATE_TOOLS

STATE_WRITE_REMINDER = (
    "本角色必须先把真实进展写入共享状态，再结束。请调用 read_shared_state 确认当前状态，"
    "然后调用你负责的状态更新工具；只记录已确认内容，不得把未验证步骤标为完成。"
)


class StateWriteMiddleware(AgentMiddleware):
    """Force one status-only model turn when a role has not committed its state."""

    def __init__(
        self,
        role: str,
        shared_state: Any,
        limits: dict[str, int],
        allowed_tools: set[str] | None = None,
    ) -> None:
        if role not in ROLE_UPDATE_TOOLS:
            raise ValueError(f"unknown agent role: {role}")
        self.role = role
        self.shared_state = shared_state
        self.limits = limits
        self.allowed_tools = allowed_tools
        self.forcing = False
        self.reminder_sent = False
        self.repair_requested = False

    def before_agent(self, state: dict[str, Any], runtime: Any = None) -> None:
        self.forcing = False
        self.reminder_sent = False
        self.repair_requested = False

    def _write_required(self) -> bool:
        state = self.shared_state.snapshot()
        wrote = state["role_write_counts"].get(self.role, 0) > state[
            "agent_start_write_counts"
        ].get(self.role, 0)
        if not wrote:
            return True
        return self.role == "reflector" and state["decision"] == "terminate" and not str(
            state["assessment"].get("final_answer") or ""
        ).strip()

    def _budget_exhausted(self) -> bool:
        return bool(self.shared_state.limit_reason(self.role, self.limits))

    def _request(self, request: Any) -> Any:
        missing = self._write_required()
        exhausted = self._budget_exhausted()
        if missing and exhausted:
            self.forcing = True
            self.repair_requested = True
        elif self.repair_requested:
            self.forcing = True
        elif not missing:
            self.forcing = False
        if not self.forcing and not exhausted:
            return request

        permitted = {"read_shared_state", ROLE_UPDATE_TOOLS[self.role]}
        if self.role == "reflector":
            permitted.add("show_evidence")
        if self.allowed_tools is not None:
            permitted &= self.allowed_tools
        overrides: dict[str, Any] = {}
        if missing and not self.reminder_sent:
            overrides["messages"] = [*request.messages, HumanMessage(content=STATE_WRITE_REMINDER)]
            self.reminder_sent = True
        if request.tools is not None:
            overrides["tools"] = [
                tool for tool in request.tools if getattr(tool, "name", None) in permitted
            ]
        return request.override(**overrides)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._request(request))

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        return await handler(self._request(request))

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        if not self._write_required():
            return None
        if self.repair_requested:
            raise MissingStateWrite(self.role)
        self.forcing = True
        self.repair_requested = True
        self.reminder_sent = True
        return {
            "messages": [HumanMessage(content=STATE_WRITE_REMINDER)],
            "jump_to": "model",
        }


class MissingStateWrite(RuntimeError):
    """Raised when a role still has not updated its state after one repair turn."""

    def __init__(self, role: str) -> None:
        super().__init__(f"{role} 未提交负责的共享状态；一次补写机会后仍未写入。")

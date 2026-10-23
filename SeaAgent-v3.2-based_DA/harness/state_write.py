"""Count model requests and give each Agent a bounded state-write closeout."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

from .shared_state import ROLE_UPDATE_TOOLS

STATE_WRITE_REMINDER = (
    "本角色必须把已确认进展写入共享状态。先调用 read_shared_state，再调用 {tool}；"
    "不得把未验证步骤标为完成。"
)


class AgentCloseoutComplete(RuntimeError):
    """The state is committed, so the Agent can stop without another model request."""


class MissingStateWrite(RuntimeError):
    """The Agent used its single closeout repair opportunity without committing state."""

    def __init__(self, role: str) -> None:
        super().__init__(f"{role} 未提交负责的共享状态；一次收尾机会后仍未写入。")


class StateWriteMiddleware(AgentMiddleware):
    """Enforce per-role model limits and keep the final model calls status-only."""

    def __init__(
        self,
        role: str,
        shared_state: Any,
        model_request_budget: int,
        closeout_request_budget: int,
        allowed_tools: set[str] | None = None,
        persist: Callable[[], None] | None = None,
    ) -> None:
        if role not in ROLE_UPDATE_TOOLS:
            raise ValueError(f"unknown agent role: {role}")
        self.role = role
        self.shared_state = shared_state
        self.model_request_budget = max(1, int(model_request_budget))
        self.closeout_request_budget = max(1, int(closeout_request_budget))
        self.allowed_tools = allowed_tools
        self.persist = persist

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

    def _request(self, request: Any) -> Any:
        mode = self.shared_state.claim_model_request(
            self.role, self.model_request_budget, self.closeout_request_budget
        )
        if mode == "exhausted":
            if self.persist:
                self.persist()
            if self._write_required():
                raise MissingStateWrite(self.role)
            raise AgentCloseoutComplete(self.role)
        if mode == "normal":
            if self.persist:
                self.persist()
            return request

        # This limits the next model request; the shared-state call guard separately
        # handles ordinary calls already emitted in the current model response.
        permitted = {"read_shared_state", ROLE_UPDATE_TOOLS[self.role]}
        if self.allowed_tools is not None:
            permitted &= self.allowed_tools
        overrides: dict[str, Any] = {}
        if self._write_required() and self.shared_state.mark_closeout_reminder(self.role):
            overrides["messages"] = [
                *request.messages,
                HumanMessage(content=STATE_WRITE_REMINDER.format(tool=ROLE_UPDATE_TOOLS[self.role])),
            ]
        if request.tools is not None:
            overrides["tools"] = [
                tool for tool in request.tools if getattr(tool, "name", None) in permitted
            ]
        if self.persist:
            self.persist()
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
        if not self.shared_state.claim_closeout_repair(self.role):
            raise MissingStateWrite(self.role)
        self.shared_state.require_closeout(self.role)
        self.shared_state.mark_closeout_reminder(self.role)
        if self.persist:
            self.persist()
        reminder = STATE_WRITE_REMINDER.format(tool=ROLE_UPDATE_TOOLS[self.role])
        return {"messages": [HumanMessage(content=reminder)], "jump_to": "model"}

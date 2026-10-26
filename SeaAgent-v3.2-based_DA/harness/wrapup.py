"""Ask the reflector to attach visual evidence before a terminal answer."""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

NUDGE_KEY = "evidence_wrapup_nudge"
DEFAULT_INSTRUCTION = (
    "Before finishing, follow `/skills/answer/finalize/SKILL.md`, then read shared state and save "
    "the conclusion through `update_reflection_state`."
)


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class EvidenceWrapUpMiddleware(AgentMiddleware):
    def __init__(
        self,
        evidence_tool: str,
        instruction: str = DEFAULT_INSTRUCTION,
        max_nudges: int = 1,
        *,
        shared_state: Any = None,
    ) -> None:
        self.evidence_tool = evidence_tool
        self.instruction = instruction
        self.max_nudges = max(0, int(max_nudges))
        self.shared_state = shared_state

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        if not self.evidence_tool or self.max_nudges == 0:
            return None
        if (
            self.shared_state is None
            or self.shared_state.is_closeout_required("reflector")
            or self.shared_state.snapshot().get("decision") != "terminate"
        ):
            return None
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls or not _text(last).strip():
            return None
        turn = self._current_turn(messages)
        if self._evidence_landed(turn) or self._nudges(turn) >= self.max_nudges:
            return None
        return {
            "messages": [HumanMessage(content=self.instruction, additional_kwargs={NUDGE_KEY: True})],
            "jump_to": "model",
        }

    @staticmethod
    def _current_turn(messages: list[Any]) -> list[Any]:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) and NUDGE_KEY not in (
                getattr(message, "additional_kwargs", None) or {}
            ):
                return messages[index:]
        return messages

    def _evidence_landed(self, turn: list[Any]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.evidence_tool
            and str(getattr(message, "status", "success")) != "error"
            and not (
                isinstance(getattr(message, "artifact", None), dict)
                and (
                    getattr(message, "artifact", {}).get("closeout_required")
                    or getattr(message, "artifact", {}).get("replayed")
                )
            )
            for message in turn
        )

    @staticmethod
    def _nudges(turn: list[Any]) -> int:
        return sum(
            1
            for message in turn
            if NUDGE_KEY in (getattr(message, "additional_kwargs", None) or {})
        )

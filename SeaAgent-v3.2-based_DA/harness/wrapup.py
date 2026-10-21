"""收尾守卫：仅在反思决定终止却没落地证据时提醒；连续无视就放行。"""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

NUDGE_KEY = "evidence_wrapup_nudge"  # 提醒消息的标记：既用于去重计数，也用于把它和真实用户提问区分开

DEFAULT_INSTRUCTION = (
    "Before you finalize: call the evidence tool once with every keyframe, clip and "
    "registry reference ID that this turn's tool results produced, then read_shared_state "
    "and use update_reflection_state to save assessment.final_answer with the final "
    "Chinese answer, conclusion, evidence IDs and limitations."
)


# ------------------------------------------------------------------ #
# 一、收尾守卫                                                             #
# ------------------------------------------------------------------ #
# 1.1 消息正文转纯文本
def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


# 1.2 定稿前确保证据已落地
class EvidenceWrapUpMiddleware(AgentMiddleware):

    # 1.2.1 绑定证据工具与提醒上限
    def __init__(
        self, evidence_tool: str, instruction: str = DEFAULT_INSTRUCTION,
        max_nudges: int = 1, *, shared_state: Any = None,
    ) -> None:
        self.evidence_tool = evidence_tool
        self.instruction = instruction
        self.max_nudges = max(0, int(max_nudges))
        self.shared_state = shared_state

    # 1.2.2 定稿合规判定
    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        if not self.evidence_tool or self.max_nudges == 0:
            return None
        if self.shared_state is None or self.shared_state.snapshot().get("decision") != "terminate":
            return None
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls or not _text(last).strip():  # 带 tool_calls 的输出是过程而非定稿；没有公开文本的输出也不值得打断
            return None
        turn = self._current_turn(messages)
        if self._evidence_landed(turn) or self._nudges(turn) >= self.max_nudges:
            return None
        return {  # jump_to 是每步清空的一次性通道（EphemeralValue），不会把后续轮次也钉在模型节点
            "messages": [HumanMessage(content=self.instruction, additional_kwargs={NUDGE_KEY: True})],
            "jump_to": "model",
        }

    # 1.2.3 截取本轮消息（跳过自己发的提醒）
    @staticmethod
    def _current_turn(messages: list[Any]) -> list[Any]:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) and NUDGE_KEY not in (getattr(message, "additional_kwargs", None) or {}):
                return messages[index:]
        return messages

    # 1.2.4 本轮是否已成功调过证据工具
    def _evidence_landed(self, turn: list[Any]) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.evidence_tool
            and str(getattr(message, "status", "success")) != "error"
            and not (
                isinstance(getattr(message, "artifact", None), dict)
                and (
                    getattr(message, "artifact", {}).get("budget_exhausted")
                    or getattr(message, "artifact", {}).get("replayed")
                )
            )
            for message in turn
        )

    # 1.2.5 本轮已经提醒过几次
    @staticmethod
    def _nudges(turn: list[Any]) -> int:
        return sum(1 for message in turn if NUDGE_KEY in (getattr(message, "additional_kwargs", None) or {}))

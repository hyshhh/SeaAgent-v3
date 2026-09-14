"""收尾守卫中间件：把「回答定稿」与「证据落地」绑在一起。

为什么单独一个模块：收尾约束需要两半才能闭合——``skills/finalize/SKILL.md`` 告诉模型
该怎么收尾（先汇总证据、再写结论），本模块在模型忘记时把它拉回来一次。前者是规范，
后者是兜底，单独靠任何一个都会漏。

判定发生在每次模型输出之后（``after_model`` 钩子）：

    模型这一轮输出
        ├─ 还带 tool_calls        → 还在干活，交给工具节点，不干预
        ├─ 没有公开文本           → 不算定稿，不干预
        ├─ 本轮已调用过证据工具   → 收尾合规，放行
        └─ 其余                   → 追加一条提醒消息并跳回模型节点

「本轮」从最后一条真实用户提问开始算，因此上一轮问答留下的证据不会冒充本轮证据。
提醒次数有上限：模型连续无视时放行收尾，宁可少一次证据也不把会话拖死。
"""
from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 提醒消息的标记：既用于去重计数，也用于把它和真实用户提问区分开
NUDGE_KEY = "evidence_wrapup_nudge"

DEFAULT_INSTRUCTION = (
    "Before you finalize: call the evidence tool once with every keyframe, clip and "
    "registry reference ID that this turn's tool results produced, then write the final "
    "Chinese answer with conclusion, evidence IDs and limitations."
)


def _text(message: Any) -> str:
    """消息正文转纯文本；多模态正文是内容块列表，序列化后仍可判空。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


class EvidenceWrapUpMiddleware(AgentMiddleware):
    """回答定稿前确保证据工具已落地，缺失时提醒模型，提醒无效则放行。"""

    def __init__(self, evidence_tool: str, instruction: str = DEFAULT_INSTRUCTION, max_nudges: int = 1) -> None:
        self.evidence_tool = evidence_tool
        self.instruction = instruction
        self.max_nudges = max(0, int(max_nudges))

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        """模型每次输出后判定一次收尾合规性；返回 None 即不干预。"""
        if not self.evidence_tool or self.max_nudges == 0:
            return None
        messages = list(state.get("messages") or [])
        last = messages[-1] if messages else None
        # 带 tool_calls 的输出是过程而非定稿；没有公开文本的输出也不值得打断
        if not isinstance(last, AIMessage) or last.tool_calls or not _text(last).strip():
            return None
        turn = self._current_turn(messages)
        if self._evidence_landed(turn) or self._nudges(turn) >= self.max_nudges:
            return None
        # jump_to 是每步清空的一次性通道（EphemeralValue），不会把后续轮次也钉在模型节点
        return {
            "messages": [HumanMessage(content=self.instruction, additional_kwargs={NUDGE_KEY: True})],
            "jump_to": "model",
        }

    @staticmethod
    def _current_turn(messages: list[Any]) -> list[Any]:
        """截取本轮消息：从最后一条真实用户提问开始（跳过本中间件自己发的提醒）。"""
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage) and NUDGE_KEY not in (getattr(message, "additional_kwargs", None) or {}):
                return messages[index:]
        return messages

    def _evidence_landed(self, turn: list[Any]) -> bool:
        """本轮是否已成功调用过证据工具。"""
        return any(
            isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == self.evidence_tool
            and str(getattr(message, "status", "success")) != "error"
            for message in turn
        )

    @staticmethod
    def _nudges(turn: list[Any]) -> int:
        """本轮已经补发过几次提醒。"""
        return sum(1 for message in turn if NUDGE_KEY in (getattr(message, "additional_kwargs", None) or {}))

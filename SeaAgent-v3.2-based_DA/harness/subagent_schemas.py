"""三阶段从智能体的结构化返回契约。

每个字段都给默认值：4B 模型漏字段是常态，缺字段应当仍然校验通过、由父方按"这项没说"处理，
而不是让一次委派直接抛 ValidationError 把整轮问答打断。

字段里带 ``default_factory`` 的用 Field，是因为可变默认值不能直接写 ``= []``。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 规划阶段：意图 + 验收清单
# ---------------------------------------------------------------------------


class AcceptanceItem(BaseModel):
    """验收清单的一项：这条要求要怎么才算成立。"""

    requirement: str = ""
    how_to_check: str = ""


class IntentPlan(BaseModel):
    """规划智能体的产出：目标、操作、时间范围与验收清单。"""

    target_hull_number: str = ""
    target_description: str = ""
    target_scope: str = ""  # single / multiple / unknown
    operation: str = ""  # existence / count / list / time / explain
    time_start: float | None = None
    time_end: float | None = None
    time_source: str = ""  # explicit / quoted_previous_turn / all_monitoring_time
    checklist: list[AcceptanceItem] = Field(default_factory=list)
    restated_question: str = ""
    uncertain: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# 执行阶段：带证据 ID 的发现
# ---------------------------------------------------------------------------


class ExecutionFinding(BaseModel):
    """一条发现：证据 ID 必须来自本次工具返回，不许编造。"""

    track_id: str = ""
    keyframe_ids: list[str] = Field(default_factory=list)
    registry_ids: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    band: str = ""  # confirmed / uncertain / mismatch
    reason: str = ""


class ExecutionFindings(BaseModel):
    """执行智能体的产出：这一步拿到了什么、哪部分没做成。"""

    findings: list[ExecutionFinding] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# 反思阶段：验收判定
# ---------------------------------------------------------------------------


class VerdictItem(BaseModel):
    """清单逐条对账：哪条要求、成没成立、凭什么 ID。"""

    requirement: str = ""
    status: str = ""  # met / unmet / unverifiable
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class AcceptanceVerdict(BaseModel):
    """反思智能体的产出：能否退出、下一步做什么、证据有没有落地。"""

    checklist: list[VerdictItem] = Field(default_factory=list)
    can_exit: bool = False
    next_step: str = ""
    blocking_gap: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_called: bool = False
    summary: str = ""


SUBAGENT_RESPONSE_FORMATS: dict[str, type[BaseModel]] = {
    "IntentPlan": IntentPlan,
    "ExecutionFindings": ExecutionFindings,
    "AcceptanceVerdict": AcceptanceVerdict,
}


def resolve_response_format(name: object) -> type[BaseModel] | None:
    """按名字取结构化返回契约；未声明返回 None（退回自由文本）。"""
    key = str(name or "").strip()
    if not key:
        return None
    if key not in SUBAGENT_RESPONSE_FORMATS:
        raise ValueError(f"未知的从智能体返回契约：{key}（可选：{'、'.join(SUBAGENT_RESPONSE_FORMATS)}）")
    return SUBAGENT_RESPONSE_FORMATS[key]

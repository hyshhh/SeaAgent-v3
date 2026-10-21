"""Tools that let each agent maintain its owned part of the shared state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StatePatch(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class _PlanStep(BaseModel):
    """一个可执行计划步骤；schema 明确告诉模型证据要求是字符串数组。"""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    content: str
    required_evidence: list[str] = Field(
        default_factory=list,
        description="完成该步骤所需的证据类型；必须是字符串数组。",
    )

    @field_validator("required_evidence", mode="before")
    @classmethod
    def _accept_single_evidence(cls, value: Any) -> Any:
        """兼容模型偶尔把唯一证据写成字符串，但 schema 仍声明为数组。"""
        if value is None:
            return []
        return [value] if isinstance(value, str) else value


class _PlannerPatch(_StatePatch):
    requirements: list[str] | None = None
    plan: list[_PlanStep | str] | None = None
    next_action: str | dict[str, Any] | None = None


class _ExecutionPatch(_StatePatch):
    evidence_ids: list[str] | None = None
    execution_status: Literal["idle", "running", "success", "failed"] | None = None
    step_status: dict[str, Literal["pending", "in_progress", "completed", "failed"]] | None = None


class _ReflectionPatch(_StatePatch):
    assessment: dict[str, Any] | None = None
    evidence_gaps: list[str] | None = None
    conflicts: list[str] | None = None
    decision: Literal["terminate", "replan", "continue", "failed"] | None = None


# ------------------------------------------------------------------ #
# 一、状态读写工具
# ------------------------------------------------------------------ #
# 1.1 读工具共用，写工具按角色拆开
def build_state_tools(state: Any) -> list[Any]:
    from langchain_core.tools import StructuredTool

    def read_shared_state() -> dict[str, Any]:
        return state.view()

    def planner_update(patch: _PlannerPatch) -> dict[str, Any]:
        return _update(state, "planner", patch)

    def executor_update(patch: _ExecutionPatch) -> dict[str, Any]:
        return _update(state, "executor", patch)

    def reflector_update(patch: _ReflectionPatch) -> dict[str, Any]:
        return _update(state, "reflector", patch)

    return [
        StructuredTool.from_function(
            read_shared_state,
            name="read_shared_state",
            description="读取当前共享任务状态。任何决策前先读取，避免使用过期信息。",
        ),
        StructuredTool.from_function(
            planner_update,
            name="update_planner_state",
            description="仅 Planner 使用。patch 只填 requirements、plan、next_action；不要复制整份共享状态。",
        ),
        StructuredTool.from_function(
            executor_update,
            name="update_execution_state",
            description="仅 Executor 使用。patch 只填 evidence_ids、execution_status、step_status（步骤 ID 到 pending/in_progress/completed/failed）。工具原始结果由运行时自动记录。",
        ),
        StructuredTool.from_function(
            reflector_update,
            name="update_reflection_state",
            description="仅 Reflector 使用。patch 只填 assessment、evidence_gaps、conflicts、decision；不要复制整份共享状态。",
        ),
    ]


# 1.2 落到共享状态并校验字段归属
def _update(state: Any, role: str, patch: _StatePatch) -> dict[str, Any]:
    ignored = sorted(patch.model_extra or {})
    values = patch.model_dump(exclude_unset=True, exclude=set(ignored))
    if not values:
        return {
            "ok": False,
            "error": "no_owned_fields",
            "message": f"{role} 必须至少更新一个自己负责的字段；当前可写字段：{', '.join(sorted(type(patch).model_fields))}。",
            "ignored_fields": ignored,
        }
    result = state.update(role, values)
    if ignored:
        result["ignored_fields"] = ignored
        result["message"] = "仅更新当前角色有权维护的字段；其余字段已忽略。"
    return result

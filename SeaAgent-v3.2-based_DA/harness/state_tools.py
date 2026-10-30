"""Tools that let each agent maintain its owned part of the shared state."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .shared_state import OWNER_FIELDS


class _StatePatch(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class _PlanStep(BaseModel):
    """一个可执行计划步骤；schema 明确告诉模型证据要求是字符串数组。"""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        description="唯一字符串步骤 ID，如 step-1；执行时用此 ID 更新 step_status。",
    )
    content: str
    required_evidence: list[str] = Field(
        default_factory=list,
        description="所需证据类型，字符串数组。",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _normalize_id(cls, value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            raise ValueError("plan step id must be non-empty text or an integer")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        raise ValueError("plan step id must be non-empty text or an integer")

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
    clarification: str | None = None


class _ExecutionPatch(_StatePatch):
    execution_status: Literal["idle", "running", "success", "failed"] | None = None
    step_status: dict[str, Literal["pending", "in_progress", "completed", "failed"]] | None = Field(
        default=None,
        description="键必须是当前共享状态 plan[].id 的原值，不是工具名；先读取共享状态确认步骤 ID。",
    )


class _ReflectionPatch(_StatePatch):
    assessment: dict[str, Any] | None = None
    evidence_gaps: list[str] | None = None
    conflicts: list[str] | None = None
    decision: Literal["terminate", "replan", "continue", "failed"] | None = None

    @field_validator("assessment", mode="before")
    @classmethod
    def _accept_text_assessment(cls, value: Any) -> Any:
        """兼容模型把最终回答直接写成字符串。"""
        if isinstance(value, str):
            text = value.strip()
            return {"final_answer": text} if text else {}
        if isinstance(value, dict):
            normalized = dict(value)
            if not str(normalized.get("final_answer") or "").strip():
                for key in ("final_conclusion", "conclusion", "answer"):
                    text = str(normalized.get(key) or "").strip()
                    if text:
                        normalized["final_answer"] = text
                        break
            return normalized
        return value


def _decode_patch(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_unset=True)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            try:
                value = json.loads(value.replace("\\'", "'"))
            except json.JSONDecodeError:
                raise ValueError("patch must be a JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("patch must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.
    return value


class _UpdateArguments(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patch: _StatePatch

    @model_validator(mode="before")
    @classmethod
    def _normalize_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("state update must be a JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("state update must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.

        nested = _decode_patch(value["patch"]) if "patch" in value else value
        extra_fields = {key: item for key, item in value.items() if key != "patch"}
        return {"patch": {**nested, **extra_fields}}


class _PlannerUpdateArguments(_UpdateArguments):
    patch: _PlannerPatch


class _ExecutionUpdateArguments(_UpdateArguments):
    patch: _ExecutionPatch


class _ReflectionUpdateArguments(_UpdateArguments):
    patch: _ReflectionPatch


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
            description="读取当前任务状态；决策前先读。",
        ),
        StructuredTool.from_function(
            planner_update,
            name="update_planner_state",
            args_schema=_PlannerUpdateArguments,
            description=(
                "Planner 专用；patch 传对象（运行时兼容 JSON 字符串）。传至少一个自有字段：requirements、plan（id/content/required_evidence）、"
                "next_action、clarification。不要提交 decision、step_status 或 task_status。"
            ),
        ),
        StructuredTool.from_function(
            executor_update,
            name="update_execution_state",
            args_schema=_ExecutionUpdateArguments,
            description=(
                "Executor 专用；patch 传对象（运行时兼容 JSON 字符串）。只传 execution_status、step_status（当前 plan[].id→pending/in_progress/completed/failed）。"
                "step_status 的键必须使用共享状态 plan[].id；不要使用工具名，先读取共享状态确认 ID。"
                "若误传了唯一匹配某计划步骤 required_evidence 的证据名，运行时会映射到该步骤并返回 mapped_step_ids。"
                "成功工具结果中的证据 ID 由运行时自动记录。"
            ),
        ),
        StructuredTool.from_function(
            reflector_update,
            name="update_reflection_state",
            args_schema=_ReflectionUpdateArguments,
            description=(
                "Reflector 专用；patch 传对象（运行时兼容 JSON 字符串）。传 assessment、evidence_gaps、conflicts、decision（terminate/replan/continue/failed）。"
                "assessment 必须是对象，如 {'final_answer': '结论'}。"
            ),
        ),
    ]


# 1.2 落到共享状态并校验字段归属
def _update(state: Any, role: str, patch: _StatePatch) -> dict[str, Any]:
    ignored = sorted(patch.model_extra or {})
    values = patch.model_dump(exclude_unset=True, exclude=set(ignored))
    if not values:
        owned_fields = ", ".join(sorted(OWNER_FIELDS[role]))
        ignored_message = f"；已忽略字段：{', '.join(ignored)}" if ignored else ""
        return {
            "ok": False,
            "error": "no_owned_fields",
            "message": (
                f"{role} 没有提交可写字段{ignored_message}。只传入至少一个自有字段："
                f"{owned_fields}；删除其他角色字段后重试。"
            ),
            "ignored_fields": ignored,
        }
    try:
        result = state.update(role, values)
    except (TypeError, ValueError) as error:
        return {
            "ok": False,
            "error": "invalid_state_patch",
            "message": str(error),
        }
    if ignored:
        result["ignored_fields"] = ignored
        result["message"] = "仅更新当前角色有权维护的字段；其余字段已忽略。"
    return result

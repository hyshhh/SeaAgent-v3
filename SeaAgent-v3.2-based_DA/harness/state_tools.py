"""Tools that let each agent maintain its owned part of the shared state."""
from __future__ import annotations

import json
from typing import Any


def build_state_tools(state: Any) -> list[Any]:
    from langchain_core.tools import StructuredTool

    def read_shared_state() -> dict[str, Any]:
        return state.snapshot()

    def planner_update(patch: dict[str, Any], expected_version: int | None = None) -> dict[str, Any]:
        return _update(state, "planner", patch, expected_version)

    def executor_update(patch: dict[str, Any], expected_version: int | None = None) -> dict[str, Any]:
        return _update(state, "executor", patch, expected_version)

    def reflector_update(patch: dict[str, Any], expected_version: int | None = None) -> dict[str, Any]:
        return _update(state, "reflector", patch, expected_version)

    return [
        StructuredTool.from_function(
            read_shared_state,
            name="read_shared_state",
            description="读取当前共享任务状态。任何决策前先读取，避免使用过期信息。",
        ),
        StructuredTool.from_function(
            planner_update,
            name="update_planner_state",
            description="仅 Planner 使用。更新 requirements、plan、next_action。",
        ),
        StructuredTool.from_function(
            executor_update,
            name="update_execution_state",
            description="仅 Executor 使用。更新 evidence_ids、tool_results、execution_status。",
        ),
        StructuredTool.from_function(
            reflector_update,
            name="update_reflection_state",
            description="仅 Reflector 使用。更新 assessment、evidence_gaps、conflicts、decision。",
        ),
    ]


def _update(state: Any, role: str, patch: dict[str, Any], expected_version: int | None) -> dict[str, Any]:
    if isinstance(patch, str):
        patch = json.loads(patch)
    if not isinstance(patch, dict):
        raise ValueError("patch must be a JSON object")
    return state.update(role, patch, expected_version)

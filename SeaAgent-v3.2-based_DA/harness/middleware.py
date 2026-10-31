"""中间件装配：模型额度、工具去重、状态收尾、重试与证据守卫。"""

from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------ #
# 一、中间件装配
# ------------------------------------------------------------------ #
# 1.1 按配置装配中间件栈
def build_middleware(
    config: dict[str, Any],
    model: Any,
    *,
    agent_spec: dict[str, Any],
    shared_state: Any,
    persist: Any = None,
    cancelled: Any = None,
    model_timeout: Any = None,
    tool_concurrency_gate: Any = None,
) -> list[Any]:
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        SummarizationMiddleware,
    )

    from config import project_root

    from .disclosure import SkillDisclosureMiddleware
    from .state_write import (
        AgentCloseoutComplete,
        MissingStateWrite,
        StateWriteMiddleware,
    )
    from .tool_guard import AgentCancelled, RepeatToolCallMiddleware
    from .wrapup import EvidenceWrapUpMiddleware

    settings = config.get("harness", {})
    role = str(agent_spec["name"])
    model_request_budget = int(agent_spec["model_request_budget"])
    closeout_request_budget = int(agent_spec["closeout_request_budget"])
    retry_attempts = max(0, int(settings.get("model_retry_attempts", 2)))
    selected_tools = {str(tool.name) for tool in agent_spec["tools"]}
    if agent_spec["skills"]:
        selected_tools.add("read_file")
    reminder = agent_spec["skill_reminder"]
    prompt_file = project_root() / str(
        settings.get("summary_prompt_file", "harness/summary_prompt.md")
    )  # 摘要提示词来自文件，要求逐字保留时间范围与各类 ID
    summary_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else None
    summarization_kwargs = {}
    if summary_prompt:
        summarization_kwargs["summary_prompt"] = summary_prompt
    model_retry = ModelRetryMiddleware(
        max_retries=retry_attempts,
        initial_delay=float(settings.get("retry_delay_seconds", 1.0)),
    )
    retryable = model_retry.retry_on
    model_retry.retry_on = lambda error: not isinstance(
        error, (AgentCancelled, AgentCloseoutComplete, MissingStateWrite)
    ) and retryable(error)
    stack: list[Any] = [
        # 放在请求计数外层，模型重试的每次实际请求都会计数。
        model_retry,
        RepeatToolCallMiddleware(
            state=shared_state,
            scope=role,
            persist=persist,
            allowed_tools=selected_tools,
            cancelled=cancelled,
            model_timeout=model_timeout,
            concurrency_gate=tool_concurrency_gate,
        ),
    ]
    stack.append(
        StateWriteMiddleware(
            role,
            shared_state,
            model_request_budget,
            closeout_request_budget,
            selected_tools,
            persist,
        )
    )
    if agent_spec["skills"]:
        stack.append(
            SkillDisclosureMiddleware(
                reminder=str(reminder["message"]),
                max_reminders=int(reminder["max_per_run"]),
            )
        )
    stack += [
        # 历史超阈值就摘要旧消息，只留最近若干条
        SummarizationMiddleware(
            model,
            trigger=("tokens", int(settings.get("summarization_trigger_tokens", 12000))),
            keep=("messages", int(settings.get("summarization_keep_messages", 12))),
            **summarization_kwargs,
        ),
    ]
    evidence_wrapup = agent_spec.get("evidence_wrapup")
    if evidence_wrapup:
        stack.append(
            EvidenceWrapUpMiddleware(
                evidence_tool=str(evidence_wrapup["tool"]),
                max_nudges=int(evidence_wrapup["max_nudges"]),
                shared_state=shared_state,
            )
        )
    return stack

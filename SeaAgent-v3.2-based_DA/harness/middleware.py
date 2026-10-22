"""中间件装配：上下文压缩、工具预算、状态写入、重试与证据守卫。"""

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
    skills_attached: bool = True,
    evidence_tool: str | None = None,
    shared_state: Any = None,
    persist: Any = None,
    allowed_tools: set[str] | None = None,
    cancelled: Any = None,
    model_timeout: Any = None,
    agent_name: str = "",
    tool_limit: int | None = None,
) -> list[Any]:
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        SummarizationMiddleware,
        ToolRetryMiddleware,
    )

    from config import project_root

    from .disclosure import SkillDisclosureMiddleware
    from .state_write import StateWriteMiddleware
    from .tool_guard import AgentCancelled, RepeatToolCallMiddleware
    from .wrapup import EvidenceWrapUpMiddleware

    settings = config.get("harness", {})
    tool_timeout_seconds = float(settings.get("tool_timeout_seconds", 60) or 0)
    prompt_file = project_root() / str(
        settings.get("summary_prompt_file", "harness/summary_prompt.md")
    )  # 摘要提示词来自文件，要求逐字保留时间范围与各类 ID
    summary_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else None
    summarization_kwargs = {}
    if summary_prompt:
        summarization_kwargs["summary_prompt"] = summary_prompt
    model_retry = ModelRetryMiddleware(
        max_retries=int(settings.get("retry_attempts", 2)),
        initial_delay=float(settings.get("retry_delay_seconds", 1.0)),
    )
    retryable = model_retry.retry_on
    model_retry.retry_on = lambda error: not isinstance(error, AgentCancelled) and retryable(error)
    limits = {
        "run": int(settings.get("tool_calls_per_run", 40)),
        "agent": int(
            tool_limit if tool_limit is not None else settings.get("subagent_tool_calls", 6)
        ),
        "attempts": 1 + max(0, int(settings.get("retry_attempts", 2))),
    }
    stack: list[Any] = [
        # 重试包在预算守卫外，每次实际尝试都先占用共享预算。
        ToolRetryMiddleware(  #用配置参数构造一个中间件实例
            tools=[
                str(item["name"])
                for item in config.get("tools", [])
                if str(item["name"]) != "add_registry_vessel"
            ]
            + ["read_file"],
            max_retries=int(settings.get("retry_attempts", 2)),
            initial_delay=float(settings.get("retry_delay_seconds", 1.0)),
        ),
        model_retry,  # 把实例上的方法替换成新函数（装饰器模式：在原有函数外包一层，增加行为）
        RepeatToolCallMiddleware(
            state=shared_state,
            scope=agent_name,
            persist=persist,
            allowed_tools=allowed_tools,
            cancelled=cancelled,
            model_timeout=model_timeout,
            tool_timeout=lambda: tool_timeout_seconds,
            limits=limits,
        ),
        #1. 每次工具调用前检查取消/权限，查询共享状态的额度预算，预算耗尽终止
        #2. 去重
        #3. 增加工具执行上限
        #4. 工具调用结束后把结果写入共享状态，并触发共享记忆的持久化
        #5. 模型请求预算
    ]
    if shared_state is not None and agent_name in {"planner", "executor", "reflector"}:
        stack.append(StateWriteMiddleware(agent_name, shared_state, limits, allowed_tools))
    if skills_attached:
        stack.append(
            SkillDisclosureMiddleware(
                max_reminders=int(settings.get("skill_reminder_max_per_run", 1))
            )
        )  # 还没读过技能正文就提醒一次
    stack += [
        # 历史超阈值就摘要旧消息，只留最近若干条
        SummarizationMiddleware(
            model,
            trigger=("tokens", int(settings.get("summarization_trigger_tokens", 12000))),
            keep=("messages", int(settings.get("summarization_keep_messages", 12))),
            **summarization_kwargs,
        ),
        EvidenceWrapUpMiddleware(  # 仅 decision=terminate 时提醒落证据，重规划阶段直接放行。
            evidence_tool=str(
                settings.get("evidence_tool", "") if evidence_tool is None else evidence_tool
            ),
            max_nudges=int(settings.get("evidence_wrapup_max_nudges", 1)),
            shared_state=shared_state,
        ),
    ]
    return stack

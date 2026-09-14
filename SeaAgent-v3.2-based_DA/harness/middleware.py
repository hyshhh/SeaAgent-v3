"""LangChain middleware assembly; runtime constraints live in configuration.

中间件是这套 harness 的硬约束层：上下文压缩、模型/工具调用上限、重试、待办清单。
所有阈值都来自 config/harness.yaml，本模块只做装配、不定义策略——想调行为改配置即可。
"""
from __future__ import annotations

from typing import Any


def build_middleware(config: dict[str, Any], model: Any) -> list[Any]:
    """按 harness 配置装配官方中间件。

    返回值顺序即执行顺序：``wrap_*`` 类中间件是洋葱嵌套（排在前面的在外层），
    ``before_model`` / ``after_model`` 类按同向顺序串行。
    """
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ModelRetryMiddleware,
        SummarizationMiddleware,
        TodoListMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )
    settings = config.get("harness", {})
    return [
        # 上下文压缩：历史超阈值即摘要旧消息，只留最近若干条，保证长会话不撑爆窗口
        SummarizationMiddleware(model, trigger=("tokens", int(settings.get("summarization_trigger_tokens", 12000))), keep=("messages", int(settings.get("summarization_keep_messages", 12)))),
        # 模型调用上限：单次运行最多几轮决策；exit_behavior="end" 表示超限后正常收尾而非抛错
        ModelCallLimitMiddleware(run_limit=int(settings.get("model_calls_per_run", 12)), exit_behavior="end"),
        # 工具调用上限：run 级管单次问答，thread 级管跨会话累计，防止同一会话无限试探
        ToolCallLimitMiddleware(run_limit=int(settings.get("tool_calls_per_run", 24)), thread_limit=int(settings.get("tool_calls_per_thread", 100))),
        # 模型重试：provider 抖动或限流时自动重发，避免一次网络失败毁掉整轮问答
        ModelRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        # 工具重试：工具内部异常（读文件、调服务）同样重试，与模型重试共用一组参数
        ToolRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        # 待办清单：注入 write_todos 工具，把长任务的中间计划写进 state，供后续轮次自查
        TodoListMiddleware(),
    ]

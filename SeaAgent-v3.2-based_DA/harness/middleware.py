"""LangChain middleware assembly; runtime constraints live in configuration."""
from __future__ import annotations

from typing import Any


def build_middleware(config: dict[str, Any], model: Any) -> list[Any]:
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
        SummarizationMiddleware(model, trigger=("tokens", int(settings.get("summarization_trigger_tokens", 12000))), keep=("messages", int(settings.get("summarization_keep_messages", 12)))),
        ModelCallLimitMiddleware(run_limit=int(settings.get("model_calls_per_run", 12)), exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=int(settings.get("tool_calls_per_run", 24)), thread_limit=int(settings.get("tool_calls_per_thread", 100))),
        ModelRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        ToolRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        TodoListMiddleware(),
    ]

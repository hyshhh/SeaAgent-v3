"""Configured LangChain chat model adapter."""
from __future__ import annotations

from typing import Any


def build_model(config: dict[str, Any]):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("缺少 langchain-openai，请安装生产依赖") from error
    settings = config.get("llm", {})
    return ChatOpenAI(model=settings["model"], api_key=settings.get("api_key"), base_url=settings.get("base_url"), temperature=float(settings.get("temperature", 0)), timeout=float(settings.get("timeout_seconds", 90)))

"""LangChain ChatModel：对接现有 OpenAI 兼容 LLM 配置。"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from services import AgentLLMService


# ---------------------------------------------------------------------------
# 本文件只做一件事：把 app.yaml 的 llm 配置翻译成 ChatOpenAI 实例。
# 编排逻辑不在这里 —— 四节点的模型选择（含受限的 reflect_model）在 graph.py
# 的 build_sea_agent_graph 里完成，本文件不感知角色。
# ---------------------------------------------------------------------------
def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_chat_model(llm: AgentLLMService | None = None, config: dict[str, Any] | None = None) -> ChatOpenAI:
    """用 app.yaml 的 llm 配置构造 ChatOpenAI。"""
    if llm is not None:
        settings = llm.settings
    else:
        from config import load_config
        settings = (config or load_config())["llm"]
    # SeaAgent 前端只展示结构化计划和工具结果，内部思考链系统级强制关闭。
    # 注意：这里是硬编码 False，配置里的 llm.enable_thinking 不起作用。
    thinking = False
    return ChatOpenAI(
        model=str(settings.get("model") or "gpt-4o-mini"),
        api_key=str(settings.get("api_key") or "EMPTY"),
        base_url=str(settings.get("base_url") or "").rstrip("/"),
        temperature=float(settings.get("temperature") or 0.0),
        timeout=float(settings.get("timeout_seconds") or 60),
        max_retries=1,
        # 显式传 extra_body，避免 langchain 提示应写在 model_kwargs 外
        extra_body={
            "chat_template_kwargs": {"enable_thinking": thinking},
            "enable_thinking": thinking,
        },
    )

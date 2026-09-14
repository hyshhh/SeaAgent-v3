"""Configured LangChain chat model adapter.

把 config 里的 llm 段翻译成一个 LangChain ChatModel：模型名、地址、密钥、温度、超时
全部来自配置，业务代码里不出现任何硬编码端点。embedding 模型不在此构建。
"""
from __future__ import annotations

from typing import Any


def build_model(config: dict[str, Any]):
    """按 ``config["llm"]`` 构建 ChatOpenAI（指向 OpenAI 兼容端点）。

    langchain_openai 延迟到函数内导入：本模块会被配置层与测试广泛引用，
    只有真正建模型时才要求装该依赖；缺失时给出可直接照做的中文报错。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("缺少 langchain-openai，请安装生产依赖") from error
    # 只有 model 是必填项，其余取保守默认：温度 0 保证可复现，超时 90s 防长尾挂死
    settings = config.get("llm", {})
    return ChatOpenAI(model=settings["model"], api_key=settings.get("api_key"), base_url=settings.get("base_url"), temperature=float(settings.get("temperature", 0)), timeout=float(settings.get("timeout_seconds", 90)))

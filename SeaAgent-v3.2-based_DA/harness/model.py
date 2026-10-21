"""把 config 的 llm 段翻译成 LangChain ChatModel，业务代码里不出现任何硬编码端点。"""

from __future__ import annotations

from typing import Any


# ------------------------------------------------------------------ #
# 一、模型装配                                                             #
# ------------------------------------------------------------------ #
# 1.1 按配置构建 ChatOpenAI
def build_model(config: dict[str, Any]):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("缺少 langchain-openai，请安装生产依赖") from error
    settings = config.get("llm", {})  # 模型为必填项；温度 0，请求超时默认 60s。
    max_tokens = settings.get("max_tokens")  # 思考与答案共用这个额度，不设走服务端默认
    return ChatOpenAI(
        model=settings["model"],
        api_key=settings.get("api_key"),
        base_url=settings.get("base_url"),
        temperature=float(settings.get("temperature", 0)),
        timeout=float(settings.get("timeout_seconds", 60)),
        max_retries=0,  # 重试统一由中间件管理，避免 SDK 隐式增加调用次数。
        streaming=bool(settings.get("streaming", True)),  # 边生成边回帧，长回答首字延迟更低
        max_tokens=int(max_tokens) if max_tokens else None,
    )

"""把 config 的 llm 段翻译成 LangChain ChatModel，业务代码里不出现任何硬编码端点。"""

from __future__ import annotations

from typing import Any

from config.net import is_loopback_url


# ------------------------------------------------------------------ #
# 一、模型装配                                                             #
# ------------------------------------------------------------------ #
# 1.1 按配置构建 ChatOpenAI
def build_model(config: dict[str, Any]):
    try:
        import httpx
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise RuntimeError("缺少 langchain-openai，请安装生产依赖") from error
    settings = config.get("llm", {})  # 模型为必填项；温度 0，请求超时默认 60s。
    max_tokens = settings.get("max_tokens")  # 思考与答案共用这个额度，不设走服务端默认
    timeout = float(settings.get("timeout_seconds", 60))
    base_url = settings.get("base_url")
    # 厂商私有字段（如 vLLM 的 chat_template_kwargs）放配置里，不写死在代码中。
    extra_body = settings.get("extra_body") or None
    # 本机服务显式关掉代理读取：shell 里的 ALL_PROXY 可能是 socks:// 这种
    # httpx 不认的 scheme，会在构造 ChatOpenAI 时直接抛异常（详见 config/net.py）。
    local_clients: dict[str, Any] = {}
    if is_loopback_url(base_url):
        local_clients = {
            "http_client": httpx.Client(trust_env=False, timeout=timeout),
            "http_async_client": httpx.AsyncClient(trust_env=False, timeout=timeout),
        }
    return ChatOpenAI(
        model=settings["model"],
        api_key=settings.get("api_key"),
        base_url=base_url,
        temperature=float(settings.get("temperature", 0)),
        timeout=timeout,
        max_retries=0,  # 重试统一由中间件管理，避免 SDK 隐式增加调用次数。
        streaming=bool(settings.get("streaming", True)),  # 边生成边回帧，长回答首字延迟更低
        max_tokens=int(max_tokens) if max_tokens else None,
        extra_body=extra_body,
        **local_clients,
    )

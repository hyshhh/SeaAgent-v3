"""模型端点相关的网络工具。"""

from __future__ import annotations

from urllib.parse import urlsplit

# 本机地址：这些端点上的服务（vLLM / embedding）不该经过系统代理。
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_loopback_url(url: str | None) -> bool:
    """判断 base_url 是否指向本机。

    为什么需要它：交互式 shell 里 clash/mihomo 常导出
    ``ALL_PROXY=socks://127.0.0.1:7890``。openai SDK 会把这个值原样交给
    httpx，而 httpx 只认 ``socks5://`` / ``socks5h://``，于是构造代理时直接
    抛 ``ValueError: Unknown scheme for proxy URL``；``NO_PROXY`` 也拦不住，
    因为代理对象在匹配 no_proxy 之前就已经建好。本机端点显式关闭
    ``trust_env`` 即可绕开整个代理环境。
    """
    if not url:
        return False
    host = (urlsplit(url).hostname or "").strip().lower()
    return host in _LOOPBACK_HOSTS

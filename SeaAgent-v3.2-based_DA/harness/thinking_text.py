"""剥离模型泄漏的内部推理。

Qwen3 这一系在思考模式下会输出 `...推理...`，close 标签之后才是正式答案；
provider 偶尔只回一个孤立的 `</think>`。两种形态都要处理干净，且**只用于最终答案**——
子智能体的思考流是要给人看的，不能剥。
"""
from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ORPHAN_CLOSE = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)
_UNCLOSED_OPEN = re.compile(r"<(?:think|thinking)>.*$", re.IGNORECASE | re.DOTALL)


def strip_thinking(text: str) -> str:
    """去掉思考标签，只留最终正文。"""
    content = str(text or "")
    content = _THINK_BLOCK.sub("", content)
    # 只有闭合标签时，其前方是泄漏的推理；保留标签之后可能存在的正文
    orphan = _ORPHAN_CLOSE.search(content)
    if orphan:
        content = content[orphan.end():]
    # 残留未闭合的起始标签：从那以后都是思考
    content = _UNCLOSED_OPEN.sub("", content)
    return content.strip()

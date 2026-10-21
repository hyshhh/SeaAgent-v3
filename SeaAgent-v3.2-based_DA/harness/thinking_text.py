"""剥离模型泄漏的内部推理（Qwen3 的 think 段），只用于最终答案——子智能体的思考流是要给人看的。"""
from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ORPHAN_CLOSE = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)
_UNCLOSED_OPEN = re.compile(r"<(?:think|thinking)>.*$", re.IGNORECASE | re.DOTALL)


# ------------------------------------------------------------------ #
# 一、思考文本剥离                                                           #
# ------------------------------------------------------------------ #
# 1.1 剥掉模型泄漏的思考段
def strip_thinking(text: str) -> str:
    content = str(text or "")
    content = _THINK_BLOCK.sub("", content)
    orphan = _ORPHAN_CLOSE.search(content)  # 只有闭合标签时，其前方是泄漏的推理；保留标签之后可能存在的正文
    if orphan:
        content = content[orphan.end():]
    content = _UNCLOSED_OPEN.sub("", content)  # 残留未闭合的起始标签：从那以后都是思考
    return content.strip()

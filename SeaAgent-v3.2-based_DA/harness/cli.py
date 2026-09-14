"""Small CLI for smoke-running the harness.

命令行冒烟入口：给一个问题，跑完整 harness，把最终结果 JSON 打到标准输出。
只有结果、没有过程——要看事件流请走 web 层（NDJSON）。用于本地快速验证链路是否通。
"""
from __future__ import annotations

import argparse
import json

from config import load_config
from memory import MemoryRepository
from services import AgentLLMService
from tools import ToolService

from .runtime import SeaVideoHarness


def main() -> None:
    """解析问题并跑一次问答，结果以 JSON 打印（不抛异常，失败也返回 error 结果）。"""
    parser = argparse.ArgumentParser(description="Sea-Video-Harness")
    parser.add_argument("question")
    args = parser.parse_args()
    # 与 web 层同构的装配顺序：配置 -> 记忆库 -> 工具服务 -> harness
    config = load_config(); repo = MemoryRepository(config)
    llm = AgentLLMService(config)  # 供工具服务内部调用（非主对话模型）
    tools = ToolService(config, repo, llm=llm)
    result = SeaVideoHarness(config, tools, model=None).run(args.question)  # model=None 即按配置构建主模型
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))  # default=str 兜住不可 JSON 化的值

if __name__ == "__main__": main()

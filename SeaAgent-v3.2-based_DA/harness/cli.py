"""命令行冒烟入口：跑完整 harness 并把结果 JSON 打到标准输出，只有结果、没有过程。"""
from __future__ import annotations

import argparse
import json

from config import load_config
from memory import MemoryRepository
from services import AgentLLMService
from tools import ToolService

from .runtime import SeaVideoHarness


# ------------------------------------------------------------------ #
# 一、命令行入口                                                            #
# ------------------------------------------------------------------ #
# 1.1 命令行入口
def main() -> None:
    parser = argparse.ArgumentParser(description="Sea-Video-Harness")
    parser.add_argument("question")
    args = parser.parse_args()
    config = load_config(); repo = MemoryRepository(config)  # 与 web 层同构的装配顺序：配置 -> 记忆库 -> 工具服务 -> harness
    llm = AgentLLMService(config)  # 供工具服务内部调用（非主对话模型）
    tools = ToolService(config, repo, llm=llm)
    result = SeaVideoHarness(config, tools, model=None).run(args.question)  # model=None 即按配置构建主模型
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))  # default=str 兜住不可 JSON 化的值

if __name__ == "__main__": main()

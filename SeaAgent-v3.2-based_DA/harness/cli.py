"""Small CLI for smoke-running the harness."""
from __future__ import annotations

import argparse
import json

from config import load_config
from memory import MemoryRepository
from services import AgentLLMService
from tools import ToolService

from .runtime import SeaVideoHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Sea-Video-Harness")
    parser.add_argument("question")
    args = parser.parse_args()
    config = load_config(); repo = MemoryRepository(config)
    llm = AgentLLMService(config)
    tools = ToolService(config, repo, llm=llm)
    result = SeaVideoHarness(config, tools, model=None).run(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()

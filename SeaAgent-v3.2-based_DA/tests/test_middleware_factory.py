from langchain.agents.middleware import (
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import FakeListChatModel

from config import load_config
from harness.middleware import build_middleware
from harness.wrapup import EvidenceWrapUpMiddleware


def test_official_middleware_stack():
    stack = build_middleware(load_config(), FakeListChatModel(responses=["ok"]))
    assert [type(item) for item in stack] == [SummarizationMiddleware, ToolCallLimitMiddleware, ModelRetryMiddleware, ToolRetryMiddleware, TodoListMiddleware, EvidenceWrapUpMiddleware]


def test_model_rounds_are_not_capped():
    """轮次上限已移除：收尾时机交给模型自己判断，链路里只保留工具调用安全阀。"""
    stack = build_middleware(load_config(), FakeListChatModel(responses=["ok"]))
    assert not any(type(item).__name__ == "ModelCallLimitMiddleware" for item in stack)


def test_evidence_guard_reads_its_settings_from_config():
    config = load_config()
    guard = build_middleware(config, FakeListChatModel(responses=["ok"]))[-1]
    assert guard.evidence_tool == config["harness"]["evidence_tool"]
    assert guard.max_nudges == int(config["harness"]["evidence_wrapup_max_nudges"])

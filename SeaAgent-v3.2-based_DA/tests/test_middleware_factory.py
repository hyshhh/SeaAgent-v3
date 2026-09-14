from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import FakeListChatModel

from config import load_config
from harness.middleware import build_middleware


def test_official_middleware_stack():
    stack = build_middleware(load_config(), FakeListChatModel(responses=["ok"]))
    assert [type(item) for item in stack] == [SummarizationMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelRetryMiddleware, ToolRetryMiddleware, TodoListMiddleware]

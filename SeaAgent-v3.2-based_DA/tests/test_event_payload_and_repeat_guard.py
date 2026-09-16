"""事件载荷与重复调用守卫：这两条都对应真实事故。

1. 终局事件的 result 曾经在超限时被压成字符串，前端读不到 answerText/state/toolRecords，
   界面显示「未生成回答、0 tool records」——结构必须永远保住。
2. 4B 模型会把同一个技能文件读 8 遍且从不委派：中间件要在第二次就跳过并提示，
   把模型推回正轨，而不是等运行时把整轮停掉（那样用户拿到的是没有回答）。
"""
import json

import pytest
from langchain_core.messages import ToolMessage

from config import load_config
from harness.middleware import build_middleware
from harness.runtime import _bounded
from harness.tool_guard import REPEAT_HINT, RepeatToolCallMiddleware
from langchain_core.language_models import FakeListChatModel


class _Request:
    def __init__(self, name, args, call_id="c1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}


def _handler(request):
    return ToolMessage(content=f"executed:{request.tool_call['name']}", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])


def test_repeat_call_is_skipped_with_a_hint():
    guard = RepeatToolCallMiddleware()
    first = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)
    second = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)

    assert first.content == "executed:read_file", "第一次照常执行"
    assert second.content == REPEAT_HINT, "第二次跳过并给出提示"
    assert second.status == "success", "跳过不是失败，否则会误触发连续失败守卫"


def test_different_arguments_are_not_skipped():
    guard = RepeatToolCallMiddleware()
    guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/a/SKILL.md"}), _handler)
    other = guard.wrap_tool_call(_Request("read_file", {"file_path": "/skills/b/SKILL.md"}), _handler)
    assert other.content == "executed:read_file"


def test_exempt_tool_names_are_never_skipped():
    """豁免名单里的工具即使同名同参也照常执行——守卫的豁免口子仍然有效。"""
    guard = RepeatToolCallMiddleware(exempt=("list_registry",))
    for index in range(3):
        result = guard.wrap_tool_call(_Request("list_registry", {}, f"t{index}"), _handler)
        assert result.content == "executed:list_registry"


def test_bounded_payload_keeps_its_structure_when_it_exceeds_the_limit():
    """超限只截字符串，字典还是字典——这是终局事件能被前端解析的前提。"""
    payload = {
        "answer": "回答" * 50,
        "state": "stalled",
        "tool_records": [{"id": f"c{i}", "tool": "read_file", "result": "技能正文" * 800, "status": "completed"} for i in range(6)],
    }
    bounded = _bounded(payload, 4000)

    assert isinstance(bounded, dict), "整体不能被压成字符串"
    assert bounded["state"] == "stalled"
    assert isinstance(bounded["tool_records"], list) and len(bounded["tool_records"]) == 6
    assert all(isinstance(record, dict) for record in bounded["tool_records"])
    assert len(bounded["tool_records"][0]["result"]) <= 4001, "字符串字段照旧截断"
    json.dumps(bounded, ensure_ascii=False)  # 仍然可序列化


def test_middleware_stack_includes_the_repeat_guard():
    stack = build_middleware(load_config(), FakeListChatModel(responses=["ok"]))
    assert isinstance(stack[0], RepeatToolCallMiddleware)

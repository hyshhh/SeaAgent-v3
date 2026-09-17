"""子智能体步骤外发：把委派内部的每一步接进主事件流。

问题：框架用 ``subagent.invoke()`` 把从智能体当**独立 agent** 调用（不是 LangGraph 子图），
所以它内部的步骤从来不进主流。表现就是轨迹页只有一条「委派子智能体 · 执行中」，
然后一路空转到结束 —— 用户完全不知道 planner 在里面干了什么、卡在哪。

办法：框架自己说了「父方的 callbacks 会传到子智能体」，于是挂一个回调处理器，
把委派内部的工具调用与模型调用转成同一条事件契约（带 ``agent`` 标记）。
因为每次委派都在自己的线程里同步跑完，用线程本地的委派栈就能把步骤准确归到某个子智能体头上。
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 标记：告诉回调「现在开始/结束一次委派」，以及委派给谁
DELEGATION_TOOL = "task"


def _label(serialized: Any, fallback: str = "") -> str:
    """从回调的 serialized 里取工具名；取不到就用兜底值。"""
    if isinstance(serialized, dict):
        for key in ("name", "id"):
            value = serialized.get(key)
            if isinstance(value, list):
                value = value[-1] if value else ""
            if value:
                return str(value)
    return fallback


def _preview(value: Any, limit: int = 400) -> Any:
    """回调拿到的输入输出可能是任意对象：能 JSON 化就结构化返回，否则截断成字符串。"""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + "…"
        return value
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


class SubagentTraceCallback(BaseCallbackHandler):
    """把子智能体的每一步转成对外事件。

    委派栈是**线程本地**的：``task`` 工具在哪个线程里跑，被委派的子智能体就在哪个线程里
    同步执行，所以这个线程里此期间发生的所有回调都属于那次委派 —— 并发委派也不会串。
    """

    def __init__(self, emit: Callable[[dict[str, Any]], None], event_limit: int = 4000) -> None:
        super().__init__()
        self._emit = emit
        self._limit = max(200, int(event_limit))
        self._local = threading.local()
        self._calls: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 委派栈
    @property
    def _stack(self) -> list[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @property
    def active(self) -> str:
        """当前线程正在跑的委派目标；不在委派中返回空串。"""
        return self._stack[-1] if self._stack else ""

    def event(self, payload: dict[str, Any]) -> None:
        try:
            self._emit(payload)
        except Exception:  # 回调里绝不能把整轮问答炸掉
            pass

    # ------------------------------------------------------------------ 工具
    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any = None, inputs: Any = None, **kwargs: Any) -> None:
        name = _label(serialized, "tool")
        arguments = inputs if isinstance(inputs, dict) else {"input": input_str}
        agent = self.active
        if name == DELEGATION_TOOL and not agent:
            # 主智能体发起委派：压栈，之后这个线程里的步骤都归这个子智能体。
            # 出入栈都由这一层负责——子智能体内部不会再出现 task 调用。
            target = str((arguments or {}).get("subagent_type") or "子智能体")
            self._stack.append(target)
            with self._lock:
                self._calls[str(run_id or "")] = {"tool": name, "agent": target, "delegation": True}
            self.event({
                "type": "status",
                "title": f"委派给 {target}",
                "message": "子智能体已接手，它内部的每一步都会标在它名下",
                "agent": target,
            })
            return
        if not agent:
            return  # 主智能体自己的工具调用：主流已经在记，不重复发
        call_id = str(run_id or "")
        with self._lock:
            self._calls[call_id] = {"tool": name, "agent": agent, "delegation": False}
        self.event({
            "type": "tool_start",
            "title": name,
            "message": "子智能体发起工具调用",
            "tool": name,
            "label": name,
            "arguments": _preview(arguments, self._limit),
            "callId": call_id,
            "agent": agent,
        })

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        call_id = str(run_id or "")
        with self._lock:
            record = self._calls.pop(call_id, None)
        if record:
            self.event({
                "type": "tool_result",
                "title": record["tool"],
                "message": "子智能体工具调用完成",
                "status": "completed",
                "tool": record["tool"],
                "label": record["tool"],
                "result": _preview(output, self._limit),
                "callId": call_id,
                "agent": record["agent"],
            })
        # 只有委派本身结束才出栈；子智能体内部的工具结束不动栈
        if record and record.get("delegation"):
            if self._stack:
                self._stack.pop()
            self.event({
                "type": "status",
                "title": "委派结束",
                "message": "子智能体已交回结果",
                "agent": self._stack[-1] if self._stack else "",
            })

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        call_id = str(run_id or "")
        with self._lock:
            record = self._calls.pop(call_id, None)
        if not record:
            return
        self.event({
            "type": "tool_result",
            "title": record["tool"],
            "message": "子智能体工具调用失败",
            "status": "error",
            "tool": record["tool"],
            "label": record["tool"],
            "result": _preview(str(error), self._limit),
            "callId": call_id,
            "agent": record["agent"],
        })

    # ------------------------------------------------------------------ 模型
    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        agent = self.active
        if not agent:
            return
        self.event({
            "type": "model",
            "title": "子智能体思考",
            "message": "子智能体正在决策下一步",
            "agent": agent,
        })

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        agent = self.active
        if not agent:
            return
        self.event({
            "type": "status",
            "title": "子智能体模型调用失败",
            "message": _preview(str(error), 300),
            "agent": agent,
        })

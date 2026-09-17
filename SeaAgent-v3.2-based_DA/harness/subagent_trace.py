"""子智能体调用流外发：让委派内部可见。

背景：框架用 ``subagent.invoke()`` 把从智能体当**独立 agent** 调用（不是子图），
所以它内部发生的事从来不进主事件流。轨迹页于是只有一条「委派子智能体 · 执行中」，
然后一路空转 —— 看不出它在读什么、调什么、还是纯粹卡住了。

这个模块挂在父方的 callbacks 上（框架会把 callbacks 随 config 传进子智能体），
把子智能体的**模型输出、工具调用、等待状态**全部转成对外事件：

    子智能体输出   ← 模型流式吐出的文字（它在想什么、打算调什么）
    工具调用       ← 工具名 + 参数 + 结果
    等待提示       ← 每 N 秒一条「正在等 X，已等 M 秒」，区分「慢」和「死」

归属靠**线程本地的委派栈**：委派在自己的线程里同步跑完，所以这个线程里发生的
回调都属于那次委派，并发委派也不会串。
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 标记：告诉回调「现在开始/结束一次委派」，以及委派给谁
DELEGATION_TOOL = "task"
# 计划由 planner 产出：它的模型输出里带 plan 步骤，运行时就地广播成前端清单
# 流式正文：攒够这么多字符才发一块，避免一个字一条事件把轨迹页刷爆
FLUSH_CHARS = 48
# 距上次发送超过这么久也发一块：慢速模型下也要有可见的进度
FLUSH_SECONDS = 0.8
# 单块上限（轨迹页是看进度，不是读全文）
DELTA_LIMIT = 240


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


def _text_of(chunk: Any) -> str:
    """把流式块转成纯文本：正文可能是字符串，也可能是内容块列表。"""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


def _json_block_of(text: str) -> Any:
    """从子智能体的输出里取出 JSON：优先代码块，其次整段。

    模型偶尔会漏掉围栏或前后加话，所以先找 ```json 块，再退化为第一个 {...}。
    取不到就返回 None，由调用方按散文处理（绝不猜）。
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if 0 <= start < end else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        return None


def _preview(value: Any, limit: int = 400) -> Any:
    """回调拿到的输入输出可能是任意对象：能结构化就结构化，否则截断成字符串。"""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + "…"
        return value
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


class SubagentTraceCallback(BaseCallbackHandler):
    """把子智能体的调用流转成对外事件（带 ``agent`` 归属）。"""

    def __init__(self, emit: Callable[[dict[str, Any]], None], event_limit: int = 4000, tick_seconds: float = 15.0) -> None:
        super().__init__()
        self._emit = emit
        self._limit = max(200, int(event_limit))
        self._tick = max(5.0, float(tick_seconds or 15.0))
        self._local = threading.local()
        self._lock = threading.Lock()
        # run_id -> 工具调用记录；委派本身也在里面（delegation=True）
        self._calls: dict[str, dict[str, Any]] = {}
        # run_id -> 正在等的模型/工具（有心跳的那种）
        self._waiting: dict[str, dict[str, Any]] = {}
        # 模型流式正文：run_id -> 已累积的文字（整段，on_llm_end 时给出长度）
        self._buffers: dict[str, list[str]] = {}
        # run_id -> 尚未发出的待发块与发送时间，用于攒批
        self._pending_text: dict[str, dict[str, Any]] = {}
        # run_id -> 已经流式发出去的片段，用来判断"正文是否已完整送达"，避免重复补发
        self._emitted: dict[str, list[dict[str, Any]]] = {}
        self._beater: threading.Thread | None = None
        self._stop = threading.Event()

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

    # ------------------------------------------------------------------ 心跳
    def _ensure_beater(self) -> None:
        """有委派在跑时才起心跳线程：它负责说「还在等什么」。"""
        if self._beater is not None and self._beater.is_alive():
            return
        self._stop.clear()
        self._beater = threading.Thread(target=self._beat, name="subagent-heartbeat", daemon=True)
        self._beater.start()

    def close(self) -> None:
        """本轮结束：停掉心跳线程。"""
        self._stop.set()
        beater, self._beater = self._beater, None
        if beater is not None and beater.is_alive():
            beater.join(timeout=1.0)

    def _beat(self) -> None:
        while not self._stop.wait(self._tick):
            now = time.monotonic()
            with self._lock:
                snapshot = [(dict(item), now - item["since"]) for item in self._waiting.values()]
            for item, elapsed in snapshot:
                what = str(item.get("what") or "模型响应")
                self.event({
                    "type": "status",
                    "title": f"等待{what}",
                    "message": f"{item.get('agent') or '子智能体'} 正在等{what}，已等 {int(elapsed)} 秒",
                    "agent": str(item.get("agent") or ""),
                })

    # ------------------------------------------------------------------ 工具
    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any = None, inputs: Any = None, **kwargs: Any) -> None:
        name = _label(serialized, "tool")
        arguments = inputs if isinstance(inputs, dict) else {"input": input_str}
        agent = self.active
        call_id = str(run_id or "")

        if name == DELEGATION_TOOL and not agent:
            # 主智能体发起委派：压栈，之后这个线程里的步骤都归这个子智能体
            target = str((arguments or {}).get("subagent_type") or "子智能体")
            self._stack.append(target)
            with self._lock:
                self._calls[call_id] = {"tool": name, "agent": target, "delegation": True}
            self.event({
                "type": "status",
                "title": f"委派给 {target}",
                "message": "子智能体已接手，它内部的每一步都会标在它名下",
                "agent": target,
            })
            self._ensure_beater()
            return

        if not agent:
            return  # 主智能体自己的工具调用：主流已经在记，不重复发

        with self._lock:
            self._calls[call_id] = {"tool": name, "agent": agent, "delegation": False}
            self._waiting[call_id] = {"agent": agent, "what": f"工具 {name}", "since": time.monotonic()}
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
            self._waiting.pop(call_id, None)
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
            if not self._stack:
                self.close()

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        call_id = str(run_id or "")
        with self._lock:
            record = self._calls.pop(call_id, None)
            self._waiting.pop(call_id, None)
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
        call_id = str(run_id or "")
        with self._lock:
            self._waiting[call_id] = {"agent": agent, "what": "模型响应", "since": time.monotonic()}
            self._buffers[call_id] = []
        self.event({
            "type": "model",
            "title": "子智能体思考",
            "message": "正在等待模型响应",
            "agent": agent,
        })
        self._ensure_beater()

    def on_llm_new_token(self, token: str, *, run_id: Any = None, **kwargs: Any) -> None:
        """模型流式输出：攒成块再外发。

        逐 token 发事件会把轨迹页刷成「一个字一行」，所以这里先攒：
        攒够 FLUSH_CHARS 个字符、或距上次发送超过 FLUSH_SECONDS，才发一块。
        每块带 ``append`` 与 ``streamKey``，前端把同一段输出的块合并进一行。
        """
        agent = self.active
        if not agent or not token:
            return
        call_id = str(run_id or "")
        now = time.monotonic()
        with self._lock:
            buffer = self._buffers.setdefault(call_id, [])
            buffer.append(str(token))
            # 一旦开始出字，就不再算「等模型响应」——它在动
            self._waiting.pop(call_id, None)
            pending = self._pending_text.setdefault(call_id, {"text": "", "sent_at": now})
            pending["text"] += str(token)
            should_flush = len(pending["text"]) >= FLUSH_CHARS or (now - pending["sent_at"]) >= FLUSH_SECONDS
            chunk = pending["text"] if should_flush else ""
            if should_flush:
                pending["text"] = ""
                pending["sent_at"] = now
        if chunk:
            self._emit_text(agent, call_id, chunk)

    def _emit_text(self, agent: str, call_id: str, chunk: str) -> None:
        with self._lock:
            self._emitted.setdefault(call_id, []).append({"chars": chunk})
        self.event({
            "type": "model",
            "title": "子智能体输出",
            "message": chunk if len(chunk) <= DELTA_LIMIT else chunk[:DELTA_LIMIT],
            "agent": agent,
            # 前端据此把同一段输出的多块合并到一行，而不是每块新建一行
            "append": True,
            "streamKey": call_id,
        })

    def _flush_pending(self, call_id: str, agent: str) -> None:
        """把还没发出去的尾巴发掉（模型输出结束时调用）。"""
        with self._lock:
            pending = self._pending_text.pop(call_id, None)
        if pending and pending.get("text"):
            self._emit_text(agent, call_id, pending["text"])

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        """模型调用收尾：给出正文长度与 token 用量，便于区分「慢」与「死」。"""
        agent = self.active
        call_id = str(run_id or "")
        with self._lock:
            self._waiting.pop(call_id, None)
            buffer = self._buffers.pop(call_id, None)
        if not agent:
            return
        self._flush_pending(call_id, agent)  # 尾巴也要发出去，否则最后几个字会丢
        emitted = self._emitted.pop(call_id, [])
        body = "".join(buffer) if buffer else ""
        length = len(body)
        usage: dict[str, Any] = {}
        try:
            message = response.generations[0][0].message
            usage = dict(getattr(message, "usage_metadata", None) or {})
        except Exception:
            usage = {}
        # 本体正文：工具轮里它是"我打算先做什么"的说明，最能看出它在完善哪一步。
        # 但流式通常已经把它逐块发完了，再补一条就是同一段话出现两次 —— 只在没发全时才补。
        streamed = sum(len(item.get("chars", "")) for item in emitted)
        if body.strip() and streamed < len(body.strip()):
            self.event({
                "type": "model",
                "title": "子智能体思考",
                "message": _preview(body.strip(), 600),
                "agent": agent,
            })
        detail = f"模型响应完成（正文 {length} 字"
        if usage:
            detail += f"，输入 {usage.get('input_tokens', '?')} / 输出 {usage.get('output_tokens', '?')} token"
        detail += "）"
        self.event({
            "type": "status",
            "title": "模型响应完成",
            "message": detail,
            "agent": agent,
        })

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        agent = self.active
        call_id = str(run_id or "")
        with self._lock:
            self._waiting.pop(call_id, None)
            self._buffers.pop(call_id, None)
        if not agent:
            return
        self._flush_pending(call_id, agent)  # 失败前已吐出的内容也别丢
        self.event({
            "type": "status",
            "title": "子智能体模型调用失败",
            "message": _preview(str(error), 300),
            "agent": agent,
        })

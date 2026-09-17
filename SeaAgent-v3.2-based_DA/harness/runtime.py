"""Deep Agents runtime for Sea-Video-Harness.

本模块是 v3.2 三阶段协同 harness 的运行核心，自上而下分四节：

1. 消息与载荷工具：把 LangChain 消息对象归一为裸值，并裁剪对外事件负载。
2. ``_Trace``：消费 ``agent.stream`` 的每一帧，产出对外事件契约（status / skill /
   model / tool_start / tool_result / complete / error），并汇总答案、证据与调用链。
3. ``SeaVideoHarness``：按配置装配模型、工具、skills、三阶段从智能体（规划 / 执行 / 反思）、
   中间件与 SQLite 检查点，再以 ``run``（一次性）或 ``stream``（增量）两种方式驱动同一个 agent。
4. ``run_harness``：单函数便捷入口。

约定：原始 agent state 不出模块，对外只暴露裁剪后的事件与 result 字典。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

from config import project_root

from .evidence import build_evidence_payload
from .subagent_trace import SubagentTraceCallback
from .middleware import build_middleware
from .model import build_model
from .tools import build_tools

logger = logging.getLogger(__name__)

def _error_message(error: BaseException, limit: int) -> str:
    """Return a useful, bounded public error without leaking a traceback."""
    message = str(error).strip()
    if not message:
        message = f"{type(error).__name__}: {error!r}"
    return message if len(message) <= limit else message[:limit] + "…"


# ---------------------------------------------------------------------------
# 一、消息与载荷工具
# ---------------------------------------------------------------------------


def _content(value: Any) -> Any:
    """取出消息正文；传入的若不是消息对象（已是裸值）则原样返回。"""
    return getattr(value, "content", value)


def _text(value: Any) -> str:
    """正文转纯文本。多模态正文是内容块列表，序列化后仍保留可读信息。"""
    content = _content(value)
    if isinstance(content, str):
        return content
    # default=str 兜住块里的非 JSON 类型（如 datetime、自定义对象），避免序列化直接抛错
    return json.dumps(content, ensure_ascii=False, default=str)


def _message_kind(message: Any) -> str:
    """消息种类，即 LangChain 的 message.type：ai / tool / human / system。"""
    return str(getattr(message, "type", ""))


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    """模型消息里的工具调用列表；provider 结构不保证规范，故只收 dict 形态。"""
    calls = getattr(message, "tool_calls", None)
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _parse_payload(value: Any) -> Any:
    """还原工具结果：多数工具把 JSON 字符串塞在 content 里，能解析就还原成结构。"""
    content = _content(value)
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def _bounded(value: Any, limit: int) -> Any:
    """Bound public payloads while retaining structured data whenever it fits.

    逐字段裁剪、**只截字符串**：字典还是字典、列表还是列表。
    早先这里还有一条「装不下就整体降级为截断的 JSON 字符串」的兜底，看似安全，实则把结构毁了——
    终局事件的 result 一旦超限就变成字符串，前端读不到 answerText / state / toolRecords，
    界面显示成「未生成回答、0 tool records」（技能正文一多必然触发）。
    代价是超限时载荷会变大，但结构完整、可解析，这才是对外契约该有的行为。
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {str(key): _bounded(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, limit) for item in value]
    return value



# ---------------------------------------------------------------------------
# 二、运行轨迹采集器
# ---------------------------------------------------------------------------
#
# _Trace 是 harness 与外部世界之间唯一的翻译层：输入是 agent.stream 的原始帧，
# 输出是两类东西——实时事件（推给前端）和运行时产物（交给 result() 落库）。
#
# 数据流：
#
#     agent.stream 帧
#          │  consume()
#          ├─ 展开 state → 消息指纹去重 → 按消息类型分派
#          │      ├─ 带 tool_calls 的模型消息 → 开一轮、登记调用、广播 tool_start
#          │      ├─ 带文本的模型消息         → 更新 answer、广播 model
#          │      └─ tool 结果消息            → 回填记录、广播 tool_result、捕获证据
#          └─ 账本：messages / records / by_call_id / rounds / evidence / answer
#                 │  result()
#                 └─ 对外结果字典（键名由 config.harness.output 决定）
#
# 对外事件契约（按 type 区分；除 status / complete / error 外都带 tool 或 skill 定位）：
#
#     status       启动或收尾的一次性状态，带 title、message
#     skill        某个 skill 首次被框架加载，带 skill、title、message
#     tool_start   一次工具调用已发出，带 tool、label、arguments、callId
#     tool_result  该调用已返回，带 tool、label、result、callId
#     model        模型决策或输出更新；选工具时带 tools，纯文本输出时无附加字段
#     complete     整轮结束；run 下只有文案，stream 下另带 result
#     error        运行失败，带 message；stream 下另带 result
#
# 所有事件在离开本模块前都会被 _bounded 裁剪，前端拿到的永远是有限长度的载荷。


class _Trace:
    """把 agent.stream 的原始帧翻译成对外事件，并汇总本次运行的产物。

    为什么要有这一层：agent 的 state 里既有业务数据也有框架内部结构，直接外泄会让
    前端和落库层耦合到 Deep Agents 的版本细节。所以这里只做三件事——
    去重（updates 流会重复投递同一批消息）、分派（按消息类型决定广播什么事件）、
    记账（把散落在多帧里的消息拼成一次完整的调用链）。

    三个账本对应三类产物：
        messages / answer  → 最终回答（模型没给过文本时，兜底取最后一条消息）
        records / rounds   → 工具调用链（前端画时间线，落库写 evidence 行）
        evidence           → 证据工具的返回值，单独留档供页面渲染证据卡片

    本类不依赖 agent 的返回值，只吃 stream 的帧：run 与 stream 因此可以共用同一套逻辑。
    """

    def __init__(self, emit: Callable[[dict[str, Any]], None] | None, event_limit: int, evidence_tool: str, tool_labels: dict[str, str] | None = None, subagent_tools: dict[str, str] | None = None):
        # —— 输入侧：来自配置与调用方 ——
        self.emit = emit  # 事件外发回调；为 None 时只记账、不广播（离线跑法）
        self.event_limit = event_limit  # 单条事件载荷上限（字符）
        self.evidence_tool = evidence_tool  # 哪个工具的结果算「证据」（配置项 evidence_tool）
        self.tool_labels = tool_labels or {}  # 工具名 -> 中文标签，仅用于展示
        self.subagent_tools = subagent_tools or {}  # 工具名 -> 从智能体名，用于认领命名空间

        # —— 账本一：回答 ——
        self.messages: list[Any] = []  # 去重后的原始消息，answer 兜底时取最后一条
        self.answer = ""

        # —— 账本二：工具调用链（先记请求，再按 call_id 回填结果）——
        self.records: list[dict[str, Any]] = []  # 每次调用的完整记录，顺序即发生顺序
        self.by_call_id: dict[str, dict[str, Any]] = {}  # call_id -> records 中的同一条记录
        self.rounds: list[dict[str, Any]] = []  # 每轮模型决策选中的工具链

        # —— 账本三：证据 ——
        self.evidence: dict[str, Any] = {}  # evidence_tool 的返回值，单独留档

        # —— 失控检测：连续失败 + 原地打转 ——
        # 工具预算用完后，框架会把后续调用一律驳回成 error 结果；模型若继续硬调，
        # 就会无限刷「Tool call limit exceeded」。连续失败计数是停止这种空转的依据。
        self.error_streak = 0
        # 另一种空转：同一个工具 + 同一份参数反复成功调用（实测弱模型会反复读同一个技能文件），
        # 失败计数抓不到它，所以另记一份调用签名次数。
        self.repeat_streak = 0
        self._call_signatures: dict[str, int] = {}

        # —— 去重游标：抵御 updates 流的重复投递 ——
        self._seen_messages: set[str] = set()  # 已处理过的消息指纹
        self._seen_skills: set[str] = set()  # 已上报过的 skill 名
        self._plan_signature = ""  # 上一次广播过的待办清单指纹，清单没变就不重复发

        # —— 三阶段协同：委派 → 从智能体名的映射 ——
        # subgraphs=True 时从智能体的每一帧都带命名空间 ('tools:<LangGraph 任务 id>',)，
        # 那个 id 不是模型给的 tool_call_id，所以认领顺序是：
        #   ① 该命名空间之前已认领过 → 沿用；
        #   ② 这一帧调用的工具只属于某一个从智能体（工具面按阶段切开）→ 就是它；
        #   ③ 只剩一个未结束的委派 → 归它；
        #   ④ 仍认不出（并发委派且工具面有重叠）→ 记成「从智能体」，如实标注而不是猜错人。
        self._subagent_of_namespace: dict[str, str] = {}
        self._pending_delegations: list[str] = []

    def event(self, payload: dict[str, Any]) -> None:
        """事件的唯一出口：逐字段裁剪后再外发。

        逐字段裁剪（而非整体序列化后截断）是为了保住结构：小字段原样保留，
        只有真正超限的大字段——典型是工具结果里的候选列表——才降级成截断字符串。
        """
        if not self.emit:
            return
        bounded = {
            key: _bounded(value, self.event_limit)
            for key, value in payload.items()
        }
        self.emit(bounded)

    @staticmethod
    def _message_fingerprint(message: Any) -> str:
        """消息指纹，用于跨帧去重。

        优先用框架分配的 id——它天然唯一且稳定；消息没有 id 时（部分 provider 不回填）
        退化为「类型 + 正文 + 工具调用」的 JSON 指纹。sort_keys 不能省：字典序不一致时，
        同一条消息在两帧里会算出两个指纹，去重随即失效，表现为前端重复出现同一轮工具调用。
        """
        message_id = getattr(message, "id", None)
        if message_id:
            return f"id:{message_id}"
        return json.dumps(
            {
                "type": _message_kind(message),
                "content": _content(message),
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "tool_calls": getattr(message, "tool_calls", []),
            },
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )

    def consume_skills(self, state: dict[str, Any]) -> None:
        """把框架写进 state 的 skills_metadata 转成 skill 事件。

        skills_metadata 是 Deep Agents 的回执，记录了本次注入了哪些 skill 及其 description。
        它每次都是全量回写（不是增量），所以必须按名字去重，否则前端会把同一个 skill
        反复显示成「加载中」。
        """
        metadata = state.get("skills_metadata")
        if not isinstance(metadata, list):
            return
        for item in metadata:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or name in self._seen_skills:
                continue
            self._seen_skills.add(name)
            self.event({
                "type": "skill",
                "title": name,
                "message": str(item.get("description") or "Skill 已加载"),
                "skill": name,
            })

    @property
    def pending_subagent(self) -> str:
        """当前还没结束的委派目标；没有就返回空串。

        超时信息要能点名是谁卡住了，否则用户只看到「等太久」，不知道该缩哪一步的范围。
        并发委派时可能有多个，这里如实按顺序列出，不做猜测。
        """
        return "、".join(self._pending_delegations)

    def consume_todos(self, state: dict[str, Any]) -> None:
        """把待办清单转成 plan 事件，让前端画出「计划执行到哪一步」。

        清单由框架的 ``TodoListMiddleware`` 写在 ``state["todos"]``，每次 ``write_todos``
        都是**全量回写**，所以这里按内容指纹去重：清单没变就不重复发事件，
        变了才广播一次完整快照（前端按快照重画，不做增量合并）。

        与 ``consume_skills`` 同构：续接会话时框架不重发，运行时从检查点把它补回事件流。
        """
        todos = state.get("todos")
        if not isinstance(todos, list):
            return
        items = []
        for item in todos:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            items.append({"content": content, "status": str(item.get("status") or "pending")})
        if not items:
            return
        signature = json.dumps(items, ensure_ascii=False, sort_keys=True)
        if signature == self._plan_signature:
            return
        self._plan_signature = signature
        done = sum(1 for item in items if item["status"] == "completed")
        self.event({
            "type": "plan",
            "title": "任务清单",
            "message": f"{done}/{len(items)} 步已完成",
            "todos": items,
            "total": len(items),
            "completed": done,
        })

    def consume(self, update: Any, namespace: tuple[str, ...] = ()) -> None:
        """消费一帧 stream 输出，这是本类唯一的入口。

        帧有两种形态：裸 state（顶层就有 messages），或 {节点名: state}。这里统一展开成
        state 列表走同一条路径，框架调整节点命名不会波及此处。

        ``namespace`` 是 ``subgraphs=True`` 带来的命名空间：空元组是主智能体自己，非空（形如
        ``('tools:<任务 id>',)``）是某个从智能体的内部步骤。三阶段协同时靠它把每个从智能体
        的步骤如实标到轨迹上；参数保留也兼容了 stream 的二元组帧形，
        也让将来若再引入子图时不必改动逐帧分派逻辑。

        每条新消息按类型分派，三条分支互不排斥：
            带 tool_calls 的模型消息 → 开一轮（round）、逐条登记调用、广播 tool_start
            带文本的模型消息         → 更新 answer 并广播 model
            tool 结果消息            → 按 call_id 回填记录、广播 tool_result、捕获证据
        同一条消息理论上只命中一种情况，但判定彼此独立，新增消息类型时不必改动既有分支。
        """
        if not isinstance(update, dict):
            return
        states = [update] if isinstance(update.get("messages"), list) else list(update.values())
        frames = [state for state in states if isinstance(state, dict)]
        subagent = self._resolve_subagent(namespace, frames)
        for state in frames:
            self.consume_skills(state)  # skill 回执可能出现在任一节点的 state 里
            self.consume_todos(state)  # 待办清单同理，可能出现在任一节点的 state 里
            messages = state.get("messages")
            if not isinstance(messages, list):
                continue  # 该节点这一帧只更新了别的字段
            for message in messages:
                # 同一条消息会在多帧里反复出现，指纹命中即跳过
                fingerprint = self._message_fingerprint(message)
                if fingerprint in self._seen_messages:
                    continue
                self._seen_messages.add(fingerprint)
                self.messages.append(message)
                kind = _message_kind(message)
                calls = _tool_calls(message)
                if calls:
                    # 模型这一轮选了工具：开一轮、逐条登记调用、发 tool_start
                    # 轮次按「模型决策次数」计，与工具个数无关：一次决策并行调 3 个工具仍算一轮
                    round_no = len(self.rounds) + 1
                    round_tools: list[str] = []
                    for call in calls:
                        name = str(call.get("name") or "")
                        # provider 偶尔不给调用 id，补一个随机值，保证请求与结果仍能对上
                        call_id = str(call.get("id") or uuid.uuid4().hex)
                        arguments = call.get("args") or {}
                        if name == "task":
                            # 记下这次委派给了谁：这一轮里它就一直是「进行中的委派」
                            target = str((arguments or {}).get("subagent_type") or "")
                            if target and not subagent:
                                self._pending_delegations.append(target)
                        record = {"id": call_id, "round": round_no, "tool": name, "label": self.tool_labels.get(name, name), "arguments": arguments, "status": "requested"}
                        if subagent:
                            record["agent"] = subagent
                        self.records.append(record)
                        self.by_call_id[call_id] = record
                        round_tools.append(name)
                        self._note_repeat(name, arguments)
                        self.event({"type": "tool_start", "title": record["label"], "message": "工具调用已提交", "tool": name, "label": record["label"], "arguments": _bounded(record["arguments"], self.event_limit), "callId": call_id, "agent": subagent})
                    self.rounds.append({"round": round_no, "toolChain": round_tools})
                    self.event({"type": "model", "title": "模型决策", "message": "已选择工具", "tools": round_tools, "agent": subagent})
                elif kind in {"ai", "assistant"} and _text(message).strip():
                    if subagent:
                        # 从智能体的文本是它的中间稿：进事件流供人查看，但绝不能成为最终回答
                        self.event({"type": "model", "title": "子智能体输出", "message": _text(message), "agent": subagent})
                    else:
                        # 纯文本的主智能体输出即当前答案，后续更完整的输出会覆盖它
                        self.answer = _text(message)
                        self.event({"type": "model", "title": "模型输出已更新", "message": "模型已完成一次公开输出更新"})
                if kind == "tool":
                    # 工具结果回填：按 call_id 找回请求记录，把参数、结果、状态凑成一条完整记录。
                    # 找不到对应请求时补一条 completed 记录（如会话从检查点恢复、或 provider 未回传 id），
                    # 宁可多一条孤立记录，也不让一次真实调用从前端时间线上消失。
                    call_id = str(getattr(message, "tool_call_id", ""))
                    name = str(getattr(message, "name", "") or "")
                    record = self.by_call_id.get(call_id)
                    if record is None:
                        record = {"id": call_id or uuid.uuid4().hex, "round": len(self.rounds) or 1, "tool": name, "label": self.tool_labels.get(name, name), "arguments": {}, "status": "completed"}
                        if subagent:
                            record["agent"] = subagent
                        self.records.append(record)
                    result = _parse_payload(message)
                    effective_name = name or str(record.get("tool") or "")
                    failed = str(getattr(message, "status", "") or "").lower() == "error"
                    # 连续失败计数：成功一次就清零，只有「一直失败」才算失控
                    self.error_streak = self.error_streak + 1 if failed else 0
                    record.update({"tool": effective_name, "label": self.tool_labels.get(effective_name, record.get("label", effective_name)), "result": result, "status": "error" if failed else "completed"})
                    if effective_name == "task" and record.get("arguments", {}).get("subagent_type") in self._pending_delegations:
                        # 委派结束：把它从「进行中」摘掉，后续帧不再归到它头上
                        self._pending_delegations.remove(record["arguments"]["subagent_type"])
                    if effective_name == self.evidence_tool and isinstance(result, dict) and not failed:
                        self.evidence = result  # 证据工具的结果单独留档，随 result() 一并交出
                    self.event({"type": "tool_result", "title": record.get("label") or effective_name or "tool", "message": "工具调用失败" if failed else "工具调用完成", "status": "error" if failed else "completed", "tool": effective_name, "label": record.get("label") or effective_name, "result": _bounded(result, self.event_limit), "callId": call_id, "agent": subagent})

    def _note_repeat(self, name: str, arguments: Any) -> None:
        """记一次调用签名，并在同一个「工具 + 参数」被反复调用时给出停止信号。

        为什么需要它：连续失败计数只能抓「一直报错」，抓不到「一直成功但毫无进展」。
        实测弱模型会陷在反复 read_file 同一个技能文件上，工具预算耗尽前一直在原地打转。
        """
        if not name or name == "task":
            return  # 委派不参与：同一 scope 派两次由提示词约束，不该让守卫直接掐断
        try:
            signature = f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"
        except (TypeError, ValueError):
            signature = f"{name}:{arguments!r}"
        count = self._call_signatures.get(signature, 0) + 1
        self._call_signatures[signature] = count
        self.repeat_streak = max(self.repeat_streak, count)

    def _resolve_subagent(self, namespace: tuple[str, ...], frames: list[dict[str, Any]]) -> str:
        """判断这一帧属于哪个从智能体；主智能体自己返回空串。"""
        if not namespace:
            return ""
        head = str(namespace[0])
        key = head.split(":", 1)[1] if ":" in head else head
        claimed = self._subagent_of_namespace.get(key)
        if claimed:
            return claimed
        # ② 这一帧调用的工具只可能属于一个从智能体
        called = {
            str(call.get("name") or "")
            for state in frames
            for message in (state.get("messages") or [])
            for call in _tool_calls(message)
        } if frames else set()
        owners = {self.subagent_tools[name] for name in called if name in self.subagent_tools}
        if len(owners) == 1:
            owner = owners.pop()
            self._subagent_of_namespace[key] = owner
            return owner
        # ③ 只剩一个未结束的委派
        if len(self._pending_delegations) == 1:
            owner = self._pending_delegations[0]
            self._subagent_of_namespace[key] = owner
            return owner
        # ④ 认不出就如实标注，别把步骤记到错误的从智能体头上
        return "子智能体"

    def result(self, thread_id: str, config: dict[str, Any], state: str = "completed") -> dict[str, Any]:
        """汇总本次运行的对外结果，run 与 stream 结束（含异常路径）时都会调用。

        三个动态键名——answer / state / evidence 的字段名——由 config.harness.output 决定，
        对接层要改契约只改 yaml，不必动 runtime。

        其余固定键是给下游复用的中间产物：
            session_id    会话 id，同时也是检查点的 thread_id
            tool_chain    工具名序列，顺序即调用顺序
            tool_records  每次调用的完整记录：轮次、中文标签、参数、结果、状态
            rounds        每轮模型决策选中的工具组合
            evidence      与动态 evidence 键同值的别名，兼容只认小写键的调用方

        `state` 由调用方给：正常结束为 completed，异常路径为 error（此时另有 error 字段）。
        """
        output = config.get("harness", {}).get("output", {})
        answer_field = str(output.get("answer_field", "answer"))
        state_field = str(output.get("state_field", "state"))
        evidence_field = str(output.get("evidence_field", "evidence"))
        enriched_field = str(output.get("enriched_evidence_field", "enrichedEvidence"))
        # 模型始终没吐过纯文本时，退化为最后一条**模型消息**的文本。
        # 不能退回最后一条消息本身：那可能是工具结果（例如被驳回的
        # 「Tool call limit exceeded」），会把它当成回答展示给用户。
        answer = self.answer or next(
            (_text(message) for message in reversed(self.messages) if _message_kind(message) in {"ai", "assistant"} and _text(message).strip()),
            "",
        )
        return {
            "session_id": thread_id,
            answer_field: answer,
            state_field: state,
            evidence_field: self.evidence,
            "tool_chain": [str(item.get("tool")) for item in self.records if item.get("tool")],
            "tool_records": self.records,
            "rounds": self.rounds,
            "evidence": self.evidence,
            # 证据富化：本轮散落在各次工具结果里的关键帧 / 片段 / 参考图 ID 汇总成一份载荷，
            # 前端证据面板按它一次渲染完（见 harness/evidence.py 的说明）。
            enriched_field: build_evidence_payload(self.records),
        }


# ---------------------------------------------------------------------------
# 三、Harness 装配与执行
# ---------------------------------------------------------------------------


# 框架自带工具的中文标签：工具时间线要能一眼看出「这一轮到底有没有去读技能正文」。
# read_file 之所以叫「读取技能正文」，是因为读取面已被权限规则收敛到 /skills 目录内。
_BUILTIN_TOOL_LABELS = {
    "read_file": "读取技能正文",
    "ls": "技能目录",
    "glob": "技能检索",
    "grep": "技能内容检索",
    "write_todos": "任务清单",
    "task": "委派子智能体",
}


def _tool_labels(config: dict[str, Any]) -> dict[str, str]:
    """工具名 -> 中文标签：内置文件工具打底，config/tools.yaml 的声明优先。"""
    labels = dict(_BUILTIN_TOOL_LABELS)
    for item in config.get("tools", []):
        if isinstance(item, dict) and item.get("name"):
            name = str(item["name"])
            labels[name] = str(item.get("label") or name)
    return labels


def _resume_command(decision: str, feedback: str) -> Any:
    """把人工决定翻成 LangGraph 的恢复命令。

    批准 → 中间件放行那个工具，由它自己执行（写事务在工具内部，见 harness/registry_write.py）。
    拒绝 → 工具**不执行**，拒绝理由作为工具结果交回模型，让它换个做法继续这一轮。
    """
    from langgraph.types import Command

    if decision == "approve":
        decisions = [{"type": "approve"}]
    else:
        decisions = [{"type": "reject", "message": str(feedback or "用户拒绝了这次写入，请换一种做法。")}]
    return Command(resume={"decisions": decisions})


def _interrupt_payload(update: Any) -> list[Any]:
    """一帧里若带中断，取出中断载荷列表；没有则返回空列表。

    LangGraph 的人工确认以 ``{"__interrupt__": (Interrupt(...),)}`` 这种帧出现，
    和普通 state 更新混在同一条流里。中断不是错误——它是「等人拍板」的正常状态，
    所以必须在这一层认出来，否则会被下面的通用异常处理当成运行失败吞掉。
    """
    if not isinstance(update, dict):
        return []
    pending = update.get("__interrupt__")
    if isinstance(pending, (list, tuple)):
        return list(pending)
    return [pending] if pending is not None else []


def _confirmation_from(interrupts: list[Any]) -> dict[str, Any]:
    """把中断载荷整理成前端确认卡需要的最小字段。"""
    first = interrupts[0]
    value = getattr(first, "value", first)
    value = value if isinstance(value, dict) else {}
    requests = value.get("action_requests") or []
    configs = value.get("review_configs") or []
    actions = []
    for index, request in enumerate(requests):
        request = request if isinstance(request, dict) else {}
        config = configs[index] if index < len(configs) and isinstance(configs[index], dict) else {}
        actions.append({
            "tool": str(request.get("name") or ""),
            "arguments": request.get("args") if isinstance(request.get("args"), dict) else {},
            "description": str(request.get("description") or ""),
            "allowedDecisions": list(config.get("allowed_decisions") or ["approve", "reject"]),
        })
    return {"interruptId": str(getattr(first, "id", "") or ""), "actions": actions}


def _skill_sources(harness: dict[str, Any], groups: list[str]) -> list[str]:
    """把技能组名展开成官方 skills 源路径（``/skills/<组>``，组目录的子目录才是技能）。"""
    root = "/" + str(harness.get("skills_dir", "skills")).replace("\\", "/").strip("/")
    return [f"{root}/{str(group).strip('/')}" for group in groups]


def _readonly_permissions(scope_paths: list[str], permission_type: Any, *, deny_when_empty: bool = False) -> list[Any] | None:
    """按读取白名单生成权限规则。

    ``deny_when_empty=True`` 时，白名单为空表示「什么都不许读」而不是「不做限制」。
    三阶段单智能体走 ``harness.readonly_paths``（默认 /skills），用不到这条分支；
    保留它是给「临时收紧到某个目录」的调用方一个明确的空集语义。
    """
    if not scope_paths:
        if not deny_when_empty:
            return None
        return [
            permission_type(operations=["read"], paths=["/**"], mode="deny"),
            permission_type(operations=["write"], paths=["/**"], mode="deny"),
        ]
    rules = [
        permission_type(operations=["read"], paths=scope_paths, mode="allow"),
        permission_type(operations=["read"], paths=["/**"], mode="deny"),
        permission_type(operations=["write"], paths=["/**"], mode="deny"),
    ]
    return rules


def _read_scope_paths(sources: list[str]) -> list[str]:
    """把技能源路径展开成「读取白名单」：容器目录本身 + 其下所有文件。"""
    return [path for source in sources for path in (source, f"{source}/**")]


def _load_subagents(config: dict[str, Any], tools: list[Any], permission_type: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """按 ``config/subagents.yaml`` 装配三阶段从智能体：规划 / 执行 / 反思。

    返回（从智能体规格, 主智能体工具白名单, 主智能体技能组）。

    几条硬约束都来自 Deep Agents 官方语义：
      · ``tools`` 不写就继承主智能体全部工具——所以逐个按名字显式取，名字写错就当场报错；
      · ``skills`` 不继承，不写就一条技能都看不到，需要技能的从智能体必须自己声明组名；
      · 从智能体默认隔离（只看得到 task 里那段描述），所以 system_prompt 必须自带输出契约，
        再叠一个 ``response_format`` 让父方直接收到 JSON。

    与旧版主从协同的关键区别：**按阶段切，不按数据切**。执行智能体一个人拿全部领域工具
    与四个技能组（轨迹 / 先验库 / 视觉 / 执行），不再拆成三个数据分身；规划与反思只拿
    各自阶段需要的窄工具面。

    主智能体工具白名单（``master_tools``）只留 ``show_evidence``：证据必须回到主事件流，
    否则前端证据面板永远是空的。
    """
    import yaml

    from langchain.agents.middleware import ToolCallLimitMiddleware

    from .subagent_schemas import resolve_response_format
    from .tool_guard import RepeatToolCallMiddleware

    harness = config.get("harness", {})
    spec_path = project_root() / str(harness.get("subagents_file", "config/subagents.yaml"))
    if not spec_path.is_file():
        raise RuntimeError(f"开启了三阶段协同但找不到从智能体配置：{spec_path}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    subagents: list[dict[str, Any]] = []
    for spec in raw.get("subagents") or []:
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("从智能体缺少 name")
        wanted = [str(item) for item in (spec.get("tools") or [])]
        missing = [item for item in wanted if item not in by_name]
        if missing:
            raise ValueError(f"从智能体 {name} 引用了未在 tools.yaml 声明的工具：{'、'.join(missing)}")
        built: dict[str, Any] = {
            "name": name,
            "description": str(spec.get("description") or ""),
            "system_prompt": str(spec.get("system_prompt") or ""),
            "tools": [by_name[item] for item in wanted],
            "skills": _skill_sources(harness, [str(group) for group in (spec.get("skills") or [])]),
        }
        # 读取面按组收口：从智能体只能读自己那几组技能，
        # 否则模型会顺着 ls /skills 逛到别人的规范里（实测会陷在反复读同一个文件上出不来）。
        scope = _read_scope_paths(built["skills"])
        permissions = _readonly_permissions(scope, permission_type, deny_when_empty=not scope)
        if permissions is not None:
            built["permissions"] = permissions
        # 从智能体必须自带守卫：框架只给它们 summarization + filesystem + patch_tool_calls，
        # 主智能体的限流/重复守卫**不会**继承下去。实测缺了这两样时，一个从智能体会用同一份
        # 参数把同一个工具调到天荒地老（一次委派就是一段无人管的 ReAct）。
        built["middleware"] = [
            RepeatToolCallMiddleware(),
            ToolCallLimitMiddleware(run_limit=max(1, int(harness.get("subagent_tool_calls", 12)))),
        ]
        response_format = resolve_response_format(spec.get("response_format"))
        if response_format is not None:
            built["response_format"] = response_format
        # 人工确认：把 YAML 的 interrupt_on 原样交给框架，它会在编译这个从智能体时
        # 自动加上 HumanInTheLoopMiddleware（见 deepagents graph.py 的 interrupt_on 处理）。
        # 漏搬这一个字段的表现是「配了也不中断」——写库会直接执行，没有任何确认。
        interrupt_on = spec.get("interrupt_on")
        if isinstance(interrupt_on, dict) and interrupt_on:
            built["interrupt_on"] = interrupt_on
        subagents.append(built)
    if not subagents:
        raise ValueError(f"开启了三阶段协同但 {spec_path} 里没有 subagents")
    return subagents, [str(item) for item in (raw.get("master_tools") or [])], [str(item) for item in (raw.get("master_skills") or [])]


def _filesystem_permissions(harness: dict[str, Any], permission_type: Any) -> list[Any] | None:
    """（保留给单智能体模式的配置化白名单）把 ``harness.readonly_paths`` 翻译成权限规则。"""
    return _readonly_permissions([str(path) for path in (harness.get("readonly_paths") or [])], permission_type)


class SeaVideoHarness:
    """Build and run one Deep Agents main agent with configured tools and middleware.

    一个实例对应一次运行：构造时就把 agent 装好，``run`` / ``stream`` 结束后释放
    SQLite 检查点连接（也可用 with 语句托管）。
    """

    def __init__(self, config: dict[str, Any], service: Any, model: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None):
        self.config = config
        self.service = service
        self.event_handler = event_handler
        # 只有带 bind_tools 的对象才算可用模型；注入无效时回退到配置构建，不静默接受
        self.model = model if model is not None and callable(getattr(model, "bind_tools", None)) else build_model(config)
        self.tools = build_tools(config, service)
        harness = config.get("harness", {})
        # 三阶段协同：规划 / 执行 / 反思各挂一个从智能体，主智能体只规划、委派与落证据。
        # 关掉开关即退回单智能体（不传 subagents ⇒ 根本没有 task 工具），三阶段由主智能体自己跑。
        self.subagents, master_tools, master_skills = ([], [], [])
        if harness.get("subagents_enabled"):
            from deepagents import FilesystemPermission

            self.subagents, master_tools, master_skills = _load_subagents(config, self.tools, FilesystemPermission)
        self.agent_tools = [tool for tool in self.tools if str(getattr(tool, "name", "")) in master_tools] if self.subagents else self.tools
        # 协同时主智能体改读「规划者」提示词：它只规划、委派与汇总，不亲自查数据。
        prompt_file = str(harness.get("planner_prompt_file", "harness/planner.md")) if self.subagents else str(harness.get("system_prompt_file", "harness/system.md"))
        prompt_path = project_root() / prompt_file
        self.system_prompt = prompt_path.read_text(encoding="utf-8")  # 系统提示词是文件而非配置项
        # 技能分组：单智能体读全部组；三阶段协同时只读 master_skills 指定的组，
        # 且**空列表就是"不挂技能"**——不能再 or 一个兜底，否则空配置会被当成未配置，
        # 主智能体又把所有技能组挂回去（这正是"技能读不完"屡修不止的原因）。
        if self.subagents:
            groups = master_skills
        else:
            groups = [path.name for path in sorted((project_root() / str(harness.get("skills_dir", "skills"))).iterdir()) if path.is_dir()]
        self.skill_sources = _skill_sources(harness, groups)
        self._connection: sqlite3.Connection | None = None
        self._released = False  # 跑完一轮后连接已释放，恢复时要重新装配 agent
        self.agent = self._build_agent()  # 构造即装配，后续多次调用共用同一 agent

    def _build_agent(self) -> Any:
        """装配主智能体：注册 harness profile、开检查点、挂 skills，最后交给 create_deep_agent。"""
        try:
            from deepagents import (
                FilesystemPermission,
                GeneralPurposeSubagentProfile,
                HarnessProfile,
                create_deep_agent,
                register_harness_profile,
            )
            from deepagents.backends import FilesystemBackend
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ModuleNotFoundError as error:
            if error.name == "deepagents":
                raise RuntimeError(
                    "缺少 deepagents 依赖，请使用启动服务的同一 Python 执行 "
                    "python -m pip install -e ."
                ) from error
            raise

        harness = self.config.get("harness", {})
        # 检查点连接由实例自己持有，run/stream 结束时统一 close()，避免句柄泄漏
        checkpoint = project_root() / str(harness.get("checkpointer", "data/memory/checkpoints.sqlite"))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(checkpoint), check_same_thread=False)
        saver = SqliteSaver(self._connection)
        # 「单智能体」在这里固化：禁用写入与 shell 工具、关掉通用子智能体，
        # 使 ReAct 循环只发生在主智能体内部。profile 以 openai:<模型名> 为键注册，
        # 若改了 llm.model 却没同步这个键，上述禁用会静默失效。
        disabled_tools = frozenset(str(item) for item in (harness.get("disabled_deepagent_tools") or []))
        profile = HarnessProfile(excluded_tools=disabled_tools, general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
        model_name = str(self.config.get("llm", {}).get("model", ""))
        if model_name:
            register_harness_profile(f"openai:{model_name}", profile)
        # FilesystemBackend 把项目根暴露成虚拟文件系统，skills 才能按需读到 SKILL.md
        backend = FilesystemBackend(root_dir=project_root())
        # 读取工具（read_file / ls / glob / grep）留给模型，否则框架的 skills 渐进式披露
        # 断在第二步——它能看见技能名与 description，却打不开 SKILL.md 正文。开放的同时用权限
        # 规则把读取面收敛到白名单：协同模式下主智能体没挂技能 ⇒ 一律拒绝（它没有要读的文件）；
        # 单智能体模式走 harness.readonly_paths（默认只放 /skills）。
        if self.subagents:
            permissions = _readonly_permissions(_read_scope_paths(self.skill_sources), FilesystemPermission, deny_when_empty=True)
        else:
            permissions = _filesystem_permissions(harness, FilesystemPermission)
        # 渐进式披露：技能源按组交给框架。协同时主智能体默认**不挂**技能（master_skills 为空），
        # 它的规则全在 planner.md 里——4B 模型看到技能目录会把"把技能读完"当成任务本身。
        skills_attached = bool(self.skill_sources)
        try:
            return create_deep_agent(
                model=self.model,
                tools=self.agent_tools,
                system_prompt=self.system_prompt,
                skills=self.skill_sources or None,
                backend=backend,
                permissions=permissions,
                subagents=self.subagents or None,
                middleware=build_middleware(self.config, self.model, skills_attached=skills_attached),
                checkpointer=saver,
                name="sea_video_harness",
            )
        except Exception:
            logger.exception("Harness assembly failed")
            self.close()
            raise

    def _start_heartbeat(self, trace: _Trace, question: str) -> Any:
        """跑起来之后每隔一段时间报一次「还在跑」。

        为什么需要：一轮里最慢的往往是子智能体内部的某次工具调用（全量扫描、算嵌入），
        运行时的循环只有在**收到下一帧**时才看得到东西——期间前端是彻底静止的，用户以为卡死了。
        心跳线程只是定期往外发一条 status，不碰 agent，也不影响取消与守卫。
        """
        interval = float(self.config.get("harness", {}).get("heartbeat_seconds", 15) or 0)
        if interval <= 0:
            return None
        stop = threading.Event()

        def beat() -> None:
            waited = 0.0
            while not stop.wait(interval):
                waited += interval
                trace.event({"type": "status", "title": "仍在执行", "message": f"已等待 {int(waited)} 秒（模型或工具仍在工作）"})

        thread = threading.Thread(target=beat, name="harness-heartbeat", daemon=True)
        thread.start()
        return stop

    def _subagent_tool_owners(self) -> dict[str, str]:
        """工具名 -> 从智能体名。工具面按阶段切开时，用它认领子智能体的流式帧。"""
        return {str(getattr(tool, "name", "")): str(spec["name"]) for spec in self.subagents for tool in spec["tools"]}

    def close(self) -> None:
        """Release the SQLite checkpoint connection owned by this run.

        幂等：连接取出后置为 None，重复调用、或在 __exit__ 之后再调用都安全。
        """
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()
            # 跑完这轮连接就没了；下次（例如人工确认后恢复）要先重新装配 agent
            self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, _exc_value: BaseException | None, _traceback: TracebackType | None) -> None:
        self.close()

    def _seed_skills_from_checkpoint(self, trace: _Trace, thread_id: str) -> None:
        """续接会话时补发技能事件，让每一轮都看得到实际注入的技能目录。

        框架只在**首次**运行时把 skills_metadata 写进 state；续接同一个 thread 时它已经在
        检查点里，中间件会直接跳过（skills.py 里 `if "skills_metadata" in state: return None`），
        于是一轮运行下来一条 skill 事件都没有，前端把「技能已注入」显示成 0。
        技能目录本身仍在每一轮的 system prompt 里（由 modify_request 注入），受影响的只是回执，
        这里从检查点把它补回事件流。纯展示用途，取不到就跳过。
        """
        try:
            snapshot = self.agent.get_state({"configurable": {"thread_id": thread_id}})
        except Exception:  # 新会话没有检查点，或 agent 不支持读快照
            logger.debug("跳过快照技能回执：thread_id=%s", thread_id, exc_info=True)
            return
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            trace.consume_skills(values)
            trace.consume_todos(values)

    def _drive(self, payload: Any, thread_id: str, trace: _Trace, cancel: Any = None) -> dict[str, Any]:
        """把一次 agent.stream 跑完：消费帧、发事件、按失控阈值收尾。

        ``payload`` 有两种：首次运行是 ``{"messages": [...]}``，人工确认后恢复是
        ``Command(resume=...)``。两条路径共用本方法，所以事件的产生方式完全一致。

        返回值里的 ``interrupted`` 为真时，本轮停在「等人确认」——这不是失败，
        调用方应当把确认载荷交给前端，等人拍板后再调 ``resume``。
        """
        harness = self.config.get("harness", {})
        stall_limit = max(1, int(harness.get("stall_guard_consecutive_errors", 4)))
        repeat_limit = max(1, int(harness.get("stall_guard_repeat_calls", 3)))
        # 单次委派的墙钟上限：0 表示不设（退回只靠 HTTP 层那 600 秒兜底）
        delegation_timeout = float(harness.get("delegation_timeout_seconds", 0) or 0)
        last_frame_at = time.monotonic()
        state = "completed"
        confirmation: dict[str, Any] | None = None
        try:
            for frame in self.agent.stream(
                payload,
                config={
                    "configurable": {"thread_id": thread_id},
                    # 子智能体的内部步骤不进主流（框架用 subagent.invoke 单独跑的），
                    # 靠回调把它们接出来：父方 callbacks 会随 config 传进子智能体。
                    "callbacks": [SubagentTraceCallback(
                        trace.event,
                        trace.event_limit,
                        float(harness.get("heartbeat_seconds", 15) or 15),
                    )],
                },
                stream_mode=harness.get("stream_mode", "updates"),
                subgraphs=True,
            ):
                namespace, update = frame if isinstance(frame, tuple) and len(frame) == 2 else ((), frame)
                # 心跳式看门狗：从智能体每完成一步都会回一帧，所以「距上一帧的间隔」就是
                # 单步耗时。明显超限说明它内部卡住了（长工具调用、或在自己的循环里打转），
                # 这时中止本轮并点名卡在哪一步，比让用户干等到 HTTP 层超时有用得多。
                now = time.monotonic()
                if delegation_timeout > 0 and now - last_frame_at > delegation_timeout:
                    stalled_for = int(now - last_frame_at)
                    who = trace.pending_subagent
                    logger.warning("Delegation stalled: thread_id=%s, subagent=%s, %.0fs", thread_id, who or "?", stalled_for)
                    return {
                        "state": "delegation_timeout",
                        "confirmation": None,
                        "stalled_seconds": stalled_for,
                        "subagent": who,
                    }
                last_frame_at = now
                if cancel is not None and cancel.is_set():
                    return {"state": "cancelled", "confirmation": None}
                interrupts = _interrupt_payload(update)
                if interrupts:
                    confirmation = _confirmation_from(interrupts)
                    self._awaiting_user = True
                    return {"state": "awaiting_confirmation", "confirmation": confirmation}
                trace.consume(update, namespace)
                if trace.error_streak >= stall_limit:
                    return {"state": "stalled", "confirmation": None, "reason": f"连续 {trace.error_streak} 次工具调用失败"}
                if trace.repeat_streak > repeat_limit:
                    return {"state": "stalled", "confirmation": None, "reason": f"同一个工具调用重复了 {trace.repeat_streak} 次"}
        except Exception as error:
            logger.exception("Harness drive failed: thread_id=%s", thread_id)
            return {"state": "error", "confirmation": None, "error": error}
        return {"state": state, "confirmation": confirmation}

    def run(self, question: str, thread_id: str | None = None, cancel: Any = None, **_: Any) -> dict[str, Any]:
        """跑完一轮问答后一次性返回结果；运行期异常一律降级为 error 结果，不向外抛。

        ``cancel`` 是可选的 ``threading.Event``：置位后本轮在**下一个超步边界**收尾，
        状态记为 ``cancelled``。之所以只能按超步取消，是因为 ``agent.stream`` 是阻塞生成器，
        模型调用本身无法从外部打断——能保证的是「当前这一步跑完就停」。
        """
        # thread_id 同时是检查点的会话键：复用同一 id 即续接历史，缺省则视为全新问答
        thread_id = thread_id or uuid.uuid4().hex
        harness = self.config.get("harness", {})
        trace = _Trace(
            self.event_handler,
            int(harness.get("event_payload_max_chars", 4000)),
            str(harness.get("evidence_tool", "")),
            _tool_labels(self.config),
            self._subagent_tool_owners(),
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        self._seed_skills_from_checkpoint(trace, thread_id)
        heartbeat = self._start_heartbeat(trace, question)
        try:
            outcome = self._drive({"messages": [{"role": "user", "content": question}]}, thread_id, trace, cancel)
        finally:
            if heartbeat is not None:
                heartbeat.set()
            self.close()
        return self._finish(trace, thread_id, outcome)

    def resume(self, decision: str, thread_id: str, feedback: str = "", cancel: Any = None) -> dict[str, Any]:
        """人工拍板后继续上一轮：批准则执行那个工具，拒绝则把理由交回模型换个做法。

        ``decision`` 取 ``approve`` 或 ``reject``。恢复必须用**同一个 thread_id** ——
        中断点存在检查点里，换 id 就对不上了。

        与 run 共用同一段驱动循环，所以事件契约完全一致。
        """
        normalized = str(decision or "").strip().lower()
        if normalized not in {"approve", "reject"}:
            raise ValueError(f"decision 只能是 approve 或 reject，收到：{decision!r}")
        thread_id = thread_id or uuid.uuid4().hex
        harness = self.config.get("harness", {})
        trace = _Trace(
            self.event_handler,
            int(harness.get("event_payload_max_chars", 4000)),
            str(harness.get("evidence_tool", "")),
            _tool_labels(self.config),
            self._subagent_tool_owners(),
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        self._seed_skills_from_checkpoint(trace, thread_id)
        self._resume_agent()
        heartbeat = self._start_heartbeat(trace, "resume")
        try:
            outcome = self._drive(_resume_command(normalized, feedback), thread_id, trace, cancel)
        finally:
            if heartbeat is not None:
                heartbeat.set()
            self.close()
        return self._finish(trace, thread_id, outcome)

    def _resume_agent(self) -> None:
        """恢复前重新装配 agent：上一轮结束时检查点连接已释放。"""
        if getattr(self, "_released", False):
            self.agent = self._build_agent()
            self._released = False

    def _finish(self, trace: _Trace, thread_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """run / stream / resume 三条路径共用的收尾：把驱动结果翻成对外事件与结果。"""
        harness = self.config.get("harness", {})
        state = outcome.get("state")
        if state == "awaiting_confirmation":
            result = trace.result(thread_id, self.config, state="awaiting_confirmation")
            result["confirmation"] = outcome.get("confirmation") or {}
            trace.event({
                "type": "confirm",
                "title": "等待人工确认",
                "message": "有一项写入操作需要你确认",
                "confirmation": result["confirmation"],
                "result": result,
            })
            return result
        if state == "cancelled":
            result = trace.result(thread_id, self.config, state="cancelled")
            trace.event({"type": "complete", "title": "Harness 已停止", "message": "本轮已按请求停止", "result": result})
            return result
        if state == "delegation_timeout":
            # 不是失败，是「这一步太重」：如实说明卡在哪、下一步该怎么缩
            result = trace.result(thread_id, self.config, state="delegation_timeout")
            answer_field = str(harness.get("output", {}).get("answer_field", "answer"))
            who = str(outcome.get("subagent") or "")
            waited = int(outcome.get("stalled_seconds") or 0)
            limit = int(float(harness.get("delegation_timeout_seconds", 0) or 0))
            where = f"委派给 {who} 的那一步" if who else "某次委派"
            message = (
                f"{where}已经 {waited} 秒没有任何进展，已中止本轮。"
                f"最常见的原因是全量扫描：轨迹查询没带时间范围，或把未收敛的轨迹直接喂给了关键帧/去重。"
                f"请把时间范围缩小（或指定舷号）后重试。"
            )
            if not str(result.get(answer_field) or "").strip():
                result[answer_field] = message
            trace.event({
                "type": "status",
                "title": "委派已超时",
                "message": f"{where}超过 {limit} 秒没有返回，已中止",
            })
            trace.event({"type": "complete", "title": "Harness 已收尾", "message": "单步超时，已在现有结果上收尾", "result": result})
            return result
        if state == "stalled":
            result = trace.result(thread_id, self.config, state="stalled")
            answer_field = str(harness.get("output", {}).get("answer_field", "answer"))
            reason = str(outcome.get("reason") or "")
            if not str(result.get(answer_field) or "").strip():
                result[answer_field] = f"{reason}，已按现有结果收尾。" if reason else "本轮未能继续推进，已按现有结果收尾。"
            trace.event({"type": "status", "title": "工具调用已收尾", "message": f"{reason}，已停止继续尝试"})
            trace.event({"type": "complete", "title": "Harness 已收尾", "message": "工具调用连续失败，已在现有结果上收尾", "result": result})
            return result
        if state == "error":
            error = outcome["error"]
            message = _error_message(error, trace.event_limit)
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            trace.event({"type": "error", "title": "Harness 执行失败", "message": message, "errorType": type(error).__name__, "result": result})
            return result
        result = trace.result(thread_id, self.config)
        trace.event({"type": "complete", "title": "Harness 完成", "message": "回答与证据已生成", "result": result})
        return result

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield the same public event contract as ``run`` without raw model state.

        与 run 的差别只在交付方式：事件边产生边 yield，适合 NDJSON / SSE 推送。
        """
        thread_id = thread_id or uuid.uuid4().hex
        harness = self.config.get("harness", {})
        # _Trace 的回调是同步的，而本函数是生成器：事件先入队，再在 yield 点按序吐出
        pending: list[dict[str, Any]] = []

        def emit(event: dict[str, Any]) -> None:
            pending.append(event)
            if self.event_handler:
                self.event_handler(event)

        tool_labels = _tool_labels(self.config)
        trace = _Trace(
            emit,
            int(harness.get("event_payload_max_chars", 4000)),
            str(harness.get("evidence_tool", "")),
            tool_labels,
            self._subagent_tool_owners(),
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        self._seed_skills_from_checkpoint(trace, thread_id)
        # 与 run 用同一套空转阈值：连续失败、或同一个调用原地重复，都要收尾
        stall_limit = max(1, int(harness.get("stall_guard_consecutive_errors", 4)))
        repeat_limit = max(1, int(harness.get("stall_guard_repeat_calls", 3)))
        stalled = False
        stall_reason = ""
        heartbeat = self._start_heartbeat(trace, question)
        try:
            while pending:
                yield pending.pop(0)
            for frame in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=harness.get("stream_mode", "updates"),
                subgraphs=True,
            ):
                namespace, update = frame if isinstance(frame, tuple) and len(frame) == 2 else ((), frame)
                interrupts = _interrupt_payload(update)
                if interrupts:
                    # 停在人工确认：把确认载荷推给前端，本轮以 awaiting_confirmation 收尾（不是失败）
                    result = trace.result(thread_id, self.config, state="awaiting_confirmation")
                    result["confirmation"] = _confirmation_from(interrupts)
                    trace.event({
                        "type": "confirm",
                        "title": "等待人工确认",
                        "message": "有一项写入操作需要你确认",
                        "confirmation": result["confirmation"],
                        "result": result,
                    })
                    while pending:
                        yield pending.pop(0)
                    return
                trace.consume(update, namespace)
                while pending:  # 帧内产生的多条事件在同一个 yield 点按序交付
                    yield pending.pop(0)
                if trace.error_streak >= stall_limit:
                    stalled = True
                    stall_reason = f"连续 {trace.error_streak} 次工具调用失败"
                    break
                if trace.repeat_streak > repeat_limit:
                    stalled = True
                    stall_reason = f"同一个工具调用重复了 {trace.repeat_streak} 次"
                    break
            if stalled:
                result = trace.result(thread_id, self.config, state="stalled")
                answer_field = str(harness.get("output", {}).get("answer_field", "answer"))
                if not str(result.get(answer_field) or "").strip():
                    result[answer_field] = f"{stall_reason}，已按现有结果收尾。" if stall_reason else "本轮未能继续推进，已按现有结果收尾。"
                trace.event({"type": "status", "title": "工具调用已收尾", "message": f"{stall_reason}，已停止继续尝试"})
                trace.event({"type": "complete", "title": "Harness 已收尾", "message": "工具调用连续失败，已在现有结果上收尾", "result": result})
                while pending:
                    yield pending.pop(0)
                return
            result = trace.result(thread_id, self.config)
            trace.event({"type": "complete", "title": "Harness 完成", "message": "回答与证据已生成", "result": result})
            while pending:
                yield pending.pop(0)
        except Exception as error:
            logger.exception("Harness stream failed: thread_id=%s", thread_id)
            message = _error_message(error, trace.event_limit)
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            trace.event({"type": "error", "title": "Harness 执行失败", "message": message, "errorType": type(error).__name__, "result": result})
            while pending:
                yield pending.pop(0)
        finally:
            if heartbeat is not None:
                heartbeat.set()
            self.close()


# ---------------------------------------------------------------------------
# 四、模块级便捷入口
# ---------------------------------------------------------------------------


def run_harness(config: dict[str, Any], tools: Any, llm: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None, **kwargs: Any) -> dict[str, Any]:
    """一次性问答入口：装配即运行、用完即释放；要事件流或复用实例时直接用 SeaVideoHarness。"""
    return SeaVideoHarness(config, tools, model=llm, event_handler=event_handler).run(kwargs.get("question", ""), kwargs.get("thread_id"))






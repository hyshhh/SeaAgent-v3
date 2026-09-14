"""Deep Agents runtime for Sea-Video-Harness.

本模块是 v3.2 单智能体 harness 的运行核心，自上而下分四节：

1. 消息与载荷工具：把 LangChain 消息对象归一为裸值，并裁剪对外事件负载。
2. ``_Trace``：消费 ``agent.stream`` 的每一帧，产出对外事件契约（status / skill /
   model / tool_start / tool_result / complete / error），并汇总答案、证据与调用链。
3. ``SeaVideoHarness``：按配置装配模型、工具、skills、中间件与 SQLite 检查点，
   再以 ``run``（一次性）或 ``stream``（增量）两种方式驱动同一个 agent。
4. ``run_harness``：单函数便捷入口。

约定：原始 agent state 不出模块，对外只暴露裁剪后的事件与 result 字典。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

from config import project_root

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

    递归裁剪：装得下就保留原本结构，装不下整体降级为截断的 JSON 字符串。
    目的是让任何单条事件都不会撑爆前端负载（上限见 harness.event_payload_max_chars）。
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        value = {str(key): _bounded(item, limit) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_bounded(item, limit) for item in value]
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(encoded) <= limit else encoded[:limit] + "…"



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

    def __init__(self, emit: Callable[[dict[str, Any]], None] | None, event_limit: int, evidence_tool: str, tool_labels: dict[str, str] | None = None):
        # —— 输入侧：来自配置与调用方 ——
        self.emit = emit  # 事件外发回调；为 None 时只记账、不广播（离线跑法）
        self.event_limit = event_limit  # 单条事件载荷上限（字符）
        self.evidence_tool = evidence_tool  # 哪个工具的结果算「证据」（配置项 evidence_tool）
        self.tool_labels = tool_labels or {}  # 工具名 -> 中文标签，仅用于展示

        # —— 账本一：回答 ——
        self.messages: list[Any] = []  # 去重后的原始消息，answer 兜底时取最后一条
        self.answer = ""

        # —— 账本二：工具调用链（先记请求，再按 call_id 回填结果）——
        self.records: list[dict[str, Any]] = []  # 每次调用的完整记录，顺序即发生顺序
        self.by_call_id: dict[str, dict[str, Any]] = {}  # call_id -> records 中的同一条记录
        self.rounds: list[dict[str, Any]] = []  # 每轮模型决策选中的工具链

        # —— 账本三：证据 ——
        self.evidence: dict[str, Any] = {}  # evidence_tool 的返回值，单独留档

        # —— 去重游标：抵御 updates 流的重复投递 ——
        self._seen_messages: set[str] = set()  # 已处理过的消息指纹
        self._seen_skills: set[str] = set()  # 已上报过的 skill 名

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

    def consume(self, update: Any) -> None:
        """消费一帧 stream 输出，这是本类唯一的入口。

        帧有两种形态：裸 state（顶层就有 messages），或 {节点名: state}。这里统一展开成
        state 列表走同一条路径，框架调整节点命名不会波及此处。

        每条新消息按类型分派，三条分支互不排斥：
            带 tool_calls 的模型消息 → 开一轮（round）、逐条登记调用、广播 tool_start
            带文本的模型消息         → 更新 answer 并广播 model
            tool 结果消息            → 按 call_id 回填记录、广播 tool_result、捕获证据
        同一条消息理论上只命中一种情况，但判定彼此独立，新增消息类型时不必改动既有分支。
        """
        if not isinstance(update, dict):
            return
        states = [update] if isinstance(update.get("messages"), list) else list(update.values())
        for state in states:
            if not isinstance(state, dict):
                continue
            self.consume_skills(state)  # skill 回执可能出现在任一节点的 state 里
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
                        record = {"id": call_id, "round": round_no, "tool": name, "label": self.tool_labels.get(name, name), "arguments": arguments, "status": "requested"}
                        self.records.append(record)
                        self.by_call_id[call_id] = record
                        round_tools.append(name)
                        self.event({"type": "tool_start", "title": record["label"], "message": "工具调用已提交", "tool": name, "label": record["label"], "arguments": _bounded(record["arguments"], self.event_limit), "callId": call_id})
                    self.rounds.append({"round": round_no, "toolChain": round_tools})
                    self.event({"type": "model", "title": "模型决策", "message": "已选择工具", "tools": round_tools})
                elif kind in {"ai", "assistant"} and _text(message).strip():
                    # 纯文本的模型输出即当前答案，后续更完整的输出会覆盖它
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
                        self.records.append(record)
                    result = _parse_payload(message)
                    effective_name = name or str(record.get("tool") or "")
                    record.update({"tool": effective_name, "label": self.tool_labels.get(effective_name, record.get("label", effective_name)), "result": result, "status": "completed"})
                    if effective_name == self.evidence_tool and isinstance(result, dict):
                        self.evidence = result  # 证据工具的结果单独留档，随 result() 一并交出
                    self.event({"type": "tool_result", "title": record.get("label") or effective_name or "tool", "message": "工具调用完成", "tool": effective_name, "label": record.get("label") or effective_name, "result": _bounded(result, self.event_limit), "callId": call_id})

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
        # 模型始终没吐过纯文本时，退化为最后一条消息的文本
        answer = self.answer or (_text(self.messages[-1]) if self.messages else "")
        return {
            "session_id": thread_id,
            answer_field: answer,
            state_field: state,
            evidence_field: self.evidence,
            "tool_chain": [str(item.get("tool")) for item in self.records if item.get("tool")],
            "tool_records": self.records,
            "rounds": self.rounds,
            "evidence": self.evidence,
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
}


def _tool_labels(config: dict[str, Any]) -> dict[str, str]:
    """工具名 -> 中文标签：内置文件工具打底，config/tools.yaml 的声明优先。"""
    labels = dict(_BUILTIN_TOOL_LABELS)
    for item in config.get("tools", []):
        if isinstance(item, dict) and item.get("name"):
            name = str(item["name"])
            labels[name] = str(item.get("label") or name)
    return labels


def _filesystem_permissions(harness: dict[str, Any], permission_type: Any) -> list[Any] | None:
    """把 ``harness.readonly_paths`` 翻译成文件工具权限规则。

    读取工具（read_file / ls / glob / grep）必须留给模型，否则框架的 skills 渐进式披露会断在
    第二步——模型看得见技能名与 description，却打不开 SKILL.md 正文。开放的同时把读取面收敛到
    白名单目录：规则先匹配先生效，白名单内的读取放行，其余读取与全部写入一律拒绝。

    返回 None 表示未配置白名单，此时不做任何限制（保持框架默认行为）。
    """
    readonly_paths = [str(path) for path in (harness.get("readonly_paths") or [])]
    if not readonly_paths:
        return None
    return [
        permission_type(operations=["read"], paths=readonly_paths, mode="allow"),
        permission_type(operations=["read"], paths=["/**"], mode="deny"),
        permission_type(operations=["write"], paths=["/**"], mode="deny"),
    ]


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
        prompt_file = project_root() / str(harness.get("system_prompt_file", "harness/system.md"))
        self.system_prompt = prompt_file.read_text(encoding="utf-8")  # 系统提示词是文件而非配置项
        self._connection: sqlite3.Connection | None = None
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
        skills_path = "/" + str(harness.get("skills_dir", "skills")).replace("\\", "/").strip("/")
        # 读取工具（read_file / ls / glob / grep）必须留给模型，否则框架的 skills 渐进式披露
        # 断在第二步——它能看见技能名与 description，却打不开 SKILL.md 正文。开放的同时用权限
        # 规则把读取面收敛到 skills 目录：先匹配先生效，越界读取由 FilesystemMiddleware 直接拒绝。
        permissions = _filesystem_permissions(harness, FilesystemPermission)
        # 渐进式披露：只把 skills 目录交给框架，description 随提示词注入、正文由模型自行读取
        try:
            return create_deep_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=self.system_prompt,
                skills=[skills_path],
                backend=backend,
                permissions=permissions,
                middleware=build_middleware(self.config, self.model),
                checkpointer=saver,
                name="sea_video_harness",
            )
        except Exception:
            logger.exception("Harness assembly failed")
            self.close()
            raise

    def close(self) -> None:
        """Release the SQLite checkpoint connection owned by this run.

        幂等：连接取出后置为 None，重复调用、或在 __exit__ 之后再调用都安全。
        """
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

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
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        self._seed_skills_from_checkpoint(trace, thread_id)
        cancelled = False
        try:
            # updates 模式每次产出一帧增量，交给 _Trace 去重并翻译成事件
            for update in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=harness.get("stream_mode", "updates"),
            ):
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                trace.consume(update)
        except Exception as error:
            # 对外只返回安全的短消息，完整 traceback 进入服务日志，便于定位模型/工具/中间件故障。
            logger.exception("Harness run failed: thread_id=%s", thread_id)
            message = _error_message(error, trace.event_limit)
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            trace.event({"type": "error", "title": "Harness 执行失败", "message": message, "errorType": type(error).__name__, "result": result})
            return result
        finally:
            self.close()
        if cancelled:
            result = trace.result(thread_id, self.config, state="cancelled")
            trace.event({"type": "complete", "title": "Harness 已停止", "message": "本轮已按请求停止", "result": result})
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
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        self._seed_skills_from_checkpoint(trace, thread_id)
        try:
            while pending:
                yield pending.pop(0)
            for update in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=harness.get("stream_mode", "updates"),
            ):
                trace.consume(update)
                while pending:  # 帧内产生的多条事件在同一个 yield 点按序交付
                    yield pending.pop(0)
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
            self.close()


# ---------------------------------------------------------------------------
# 四、模块级便捷入口
# ---------------------------------------------------------------------------


def run_harness(config: dict[str, Any], tools: Any, llm: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None, **kwargs: Any) -> dict[str, Any]:
    """一次性问答入口：装配即运行、用完即释放；要事件流或复用实例时直接用 SeaVideoHarness。"""
    return SeaVideoHarness(config, tools, model=llm, event_handler=event_handler).run(kwargs.get("question", ""), kwargs.get("thread_id"))


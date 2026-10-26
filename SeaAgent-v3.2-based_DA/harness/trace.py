"""将框架消息投影为界面事件和审计记录；任务状态只读写共享字典。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from .evidence import build_evidence_payload
from .shared_state import SharedTaskState
from .thinking_text import strip_thinking


# ------------------------------------------------------------------ #
# 一、消息与载荷工具
# ------------------------------------------------------------------ #
# 1.1 取出消息正文
def _content(value: Any) -> Any:
    return getattr(value, "content", value)


# 1.2 将消息正文转换为可读文本
def _text(value: Any) -> str:
    content = _content(value)
    if isinstance(content, str):
        return content
    return json.dumps(
        content, ensure_ascii=False, default=str
    )  # default=str 兜住块里的非 JSON 类型（如 datetime、自定义对象），避免序列化直接抛错


# 1.3 识别消息类型
def _message_kind(message: Any) -> str:
    return str(getattr(message, "type", ""))


# 1.4 提取模型消息中的工具调用
def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None)
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


# 1.5 将工具结果正文还原为结构化载荷
def _parse_payload(value: Any) -> Any:
    artifact = getattr(value, "artifact", None)
    if isinstance(artifact, dict) and "full_result" in artifact:
        return artifact["full_result"]
    content = _content(value)
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def _tool_error_type(result: Any, failed: bool) -> str | None:
    if not failed:
        return None
    if isinstance(result, dict):
        return str(
            result.get("errorType")
            or result.get("error_type")
            or result.get("error")
            or "ToolError"
        )
    return "ToolError"


# 1.6 逐字段裁剪字符串，保留字典与列表结构
def _bounded(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        return {str(key): _bounded(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, limit) for item in value]
    return value


# ------------------------------------------------------------------ #
# 二、运行轨迹采集器
# ------------------------------------------------------------------ #
# _Trace 把 agent.stream 的帧翻译成对外事件与运行产物，事件都经 _bounded 裁剪。


# 2.1 将框架帧翻译为对外事件与审计记录
class _Trace:
    # 2.1.1 绑定事件回调，初始化展示记录与去重游标
    def __init__(
        self,
        emit: Callable[[dict[str, Any]], None] | None,
        event_limit: int,
        tool_labels: dict[str, str] | None = None,
        shared_state: SharedTaskState | None = None,
    ):
        # —— 输入侧：来自配置与调用方 ——
        self.emit = emit  # 事件外发回调；为 None 时只记账、不广播（离线跑法）
        self.event_limit = event_limit  # 单条事件载荷上限（字符）
        self.tool_labels = tool_labels or {}  # 工具名 -> 中文标签，仅用于展示
        self.shared_state = shared_state

        # —— 账本二：工具调用链（先记请求，再按 call_id 回填结果）——
        self.records: list[dict[str, Any]] = []  # 每次调用的完整记录，顺序即发生顺序
        self.by_call_id: dict[str, dict[str, Any]] = {}  # call_id -> records 中的同一条记录
        self.rounds: list[dict[str, Any]] = []  # 每轮模型决策选中的工具链

        # —— 去重游标：抵御 updates 流的重复投递 ——
        self._seen_messages: set[str] = set()  # 已处理过的消息指纹
        self._seen_skills: set[tuple[str, str]] = set()  # 已上报过的 Agent 与 skill 组合
        self._plan_signature = ""  # 前端计划快照没有变化时不重复广播。
        self._state_signature = ""  # 前端状态页只接收发生变化的共享状态快照。

    def agent_status(self, agent: str, state: str, iteration: int, action: str) -> None:
        self.event(
            {
                "type": "agent_status",
                "agent": agent,
                "state": state,
                "iteration": iteration,
                "message": action,
            }
        )
        self.consume_state()

    # 2.1.2 裁剪并广播事件
    def event(self, payload: dict[str, Any]) -> None:
        if not self.emit:
            return
        bounded = {key: _bounded(value, self.event_limit) for key, value in payload.items()}
        self.emit(bounded)

    # 2.1.3 生成消息指纹，抵御重复帧
    @staticmethod
    def _message_fingerprint(message: Any) -> str:
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

    # 2.1.4 广播首次出现的技能回执
    def consume_skills(self, state: dict[str, Any], agent_name: str = "") -> None:
        metadata = state.get("skills_metadata")
        if not isinstance(metadata, list):
            return
        for item in metadata:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            key = (str(agent_name or ""), name)
            if not name or key in self._seen_skills:
                continue
            self._seen_skills.add(key)
            self.event(
                {
                    "type": "skill",
                    "title": name,
                    "message": str(item.get("description") or "Skill 已加载"),
                    "skill": name,
                    "agent": str(agent_name or ""),
                }
            )

    # 2.1.5 从共享计划与执行进度生成界面快照
    def consume_plan(self) -> None:
        if self.shared_state is None:
            return
        state = self.shared_state.snapshot()
        items = [
            {**step, "status": state["step_status"].get(step["id"], "pending")}
            for step in state["plan"]
        ]
        signature = json.dumps(items, ensure_ascii=False, sort_keys=True)
        if signature == self._plan_signature:
            return
        self._plan_signature = signature
        done = sum(1 for item in items if item["status"] == "completed")
        self.event(
            {
                "type": "plan",
                "title": "任务清单",
                "message": f"{done}/{len(items)} 步已完成",
                "steps": items,
                "total": len(items),
                "completed": done,
                "iteration": state.get("iteration") or 1,
                "pendingAgent": state.get("pending_agent"),
            }
        )

    # 2.1.5.1 将共享业务状态投影给状态页；缓存结果仍由共享状态内部管理。
    def consume_state(self) -> None:
        # Tiny payload limits are used by focused tests and compact clients;
        # a complete state snapshot cannot fit those event contracts.
        if self.shared_state is None or self.event_limit < 256:
            return
        state = self.shared_state.view()
        signature = json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)
        if signature == self._state_signature:
            return
        self._state_signature = signature
        self.event(
            {
                "type": "state_snapshot",
                "sharedState": state,
                "iteration": state.get("iteration") or 1,
                "agent": state.get("pending_agent") or "",
            }
        )

    # 2.1.6 消费框架帧，关联请求与结果并广播角色事件
    def consume(self, update: Any, agent_name: str = "") -> None:
        if not isinstance(update, dict):
            return
        states = (
            [update] if isinstance(update.get("messages"), list) else list(update.values())
        )  # 裸 state 直接收，{节点名: state} 取各节点的值
        frames = [
            state for state in states if isinstance(state, dict)
        ]  # 只留真正的 state，值可能是别的类型
        role_name = str(agent_name or "")
        iteration = (
            self.shared_state.snapshot().get("iteration") or 1
            if self.shared_state is not None
            else 1
        )
        for state in frames:
            self.consume_skills(state, role_name)  # skill 回执可能出现在任一节点的 state 里
            messages = state.get("messages")
            if not isinstance(messages, list):
                continue  # 该节点这一帧只更新了别的字段
            for message in messages:
                fingerprint = self._message_fingerprint(
                    message
                )  # 同一条消息会在多帧里反复出现，指纹命中即跳过
                if fingerprint in self._seen_messages:
                    continue
                self._seen_messages.add(fingerprint)
                kind = _message_kind(message)
                calls = _tool_calls(message)
                if calls:
                    round_no = (
                        len(self.rounds) + 1
                    )  # 轮次按模型决策次数计，一次决策并行调 3 个工具仍算一轮
                    round_tools: list[str] = []
                    for call in calls:
                        name = str(call.get("name") or "")
                        call_id = str(
                            call.get("id") or uuid.uuid4().hex
                        )  # provider 偶尔不给调用 id，补一个随机值，保证请求与结果仍能对上
                        arguments = call.get("args") or {}
                        record = {
                            "id": call_id,
                            "round": round_no,
                            "iteration": iteration,
                            "tool": name,
                            "label": self.tool_labels.get(name, name),
                            "arguments": arguments,
                            "status": "requested",
                        }
                        if role_name:
                            record["agent"] = role_name
                        self.records.append(record)
                        self.by_call_id[call_id] = record
                        round_tools.append(name)
                        self.event(
                            {
                                "type": "tool_start",
                                "title": record["label"],
                                "message": "工具调用已提交",
                                "tool": name,
                                "label": record["label"],
                                "arguments": _bounded(record["arguments"], self.event_limit),
                                "callId": call_id,
                                "round": round_no,
                                "iteration": iteration,
                                "agent": role_name,
                            }
                        )
                    self.rounds.append({"round": round_no, "toolChain": round_tools})
                    self.event(
                        {
                            "type": "model",
                            "title": "模型决策",
                            "message": "已选择工具",
                            "tools": round_tools,
                            "round": round_no,
                            "iteration": iteration,
                            "agent": role_name,
                        }
                    )
                elif kind in {"ai", "assistant"} and _text(message).strip():
                    if role_name:
                        self.event(
                            {
                                "type": "model",
                                "title": "角色输出",
                                "message": _text(message),
                                "agent": role_name,
                            }
                        )  # 角色中间稿只进入事件流；最终回答从共享状态读取
                if kind == "tool":
                    call_id = str(
                        getattr(message, "tool_call_id", "")
                    )  # 找不到请求也补一条孤立记录，别让真实调用从时间线上消失
                    name = str(getattr(message, "name", "") or "")
                    record = self.by_call_id.get(call_id)
                    if record is None:
                        record = {
                            "id": call_id or uuid.uuid4().hex,
                            "round": len(self.rounds) or 1,
                            "tool": name,
                            "label": self.tool_labels.get(name, name),
                            "arguments": {},
                            "status": "completed",
                        }
                        if role_name:
                            record["agent"] = role_name
                        self.records.append(record)
                    result = _parse_payload(message)
                    effective_name = name or str(record.get("tool") or "")
                    failed = str(getattr(message, "status", "") or "").lower() == "error" or (
                        isinstance(result, dict) and result.get("ok") is False
                    )
                    error_type = _tool_error_type(result, failed)
                    record.update(
                        {
                            "tool": effective_name,
                            "label": self.tool_labels.get(
                                effective_name, record.get("label", effective_name)
                            ),
                            "result": result,
                            "status": "error" if failed else "completed",
                        }
                    )
                    if error_type:
                        record["errorType"] = error_type
                    tool_event = {
                        "type": "tool_result",
                        "title": record.get("label") or effective_name or "tool",
                        "message": "工具调用失败" if failed else "工具调用完成",
                        "status": "error" if failed else "completed",
                        "tool": effective_name,
                        "label": record.get("label") or effective_name,
                        "arguments": _bounded(record.get("arguments") or {}, self.event_limit),
                        "result": _bounded(result, self.event_limit),
                        "callId": call_id,
                        "iteration": record.get("iteration") or iteration,
                        "round": record.get("round") or len(self.rounds) or 1,
                        "agent": role_name,
                    }
                    if error_type:
                        tool_event["errorType"] = error_type
                    self.event(
                        tool_event
                    )
        self.consume_state()
        self.consume_plan()

    # 2.1.7 汇总对外结果，共享模式的回答与证据来自共享状态
    def result(
        self, thread_id: str, config: dict[str, Any], state: str = "completed"
    ) -> dict[str, Any]:
        output = config.get("harness", {}).get("output", {})
        answer_field = str(output.get("answer_field", "answer"))
        state_field = str(output.get("state_field", "state"))
        evidence_field = str(output.get("evidence_field", "evidence"))
        enriched_field = str(output.get("enriched_evidence_field", "enrichedEvidence"))
        answer = ""
        records = list(self.records)
        evidence: dict[str, Any] = {}
        if self.shared_state is not None:
            shared = self.shared_state.snapshot()
            answer = strip_thinking(str(shared["assessment"].get("final_answer") or ""))
            evidence = shared["evidence"]
            seen = {record["id"] for record in records}
            restored = [
                {**item, "id": item["callId"], "round": 0}
                for item in shared["tool_results"]
                if item["callId"] not in seen
            ]
            records = restored + records
        result = {
            "session_id": thread_id,
            answer_field: answer,
            state_field: state,
            evidence_field: evidence,
            "tool_chain": [str(item.get("tool")) for item in records if item.get("tool")],
            "tool_records": records,
            "rounds": self.rounds,
            "evidence": evidence,
            enriched_field: build_evidence_payload(records),
        }
        if self.shared_state is not None:
            result["shared_state"] = self.shared_state.view()
            result["partial"] = state == "stalled"
        return result

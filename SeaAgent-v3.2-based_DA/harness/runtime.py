"""Deep Agents runtime for Sea-Video-Harness."""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

from config import project_root

from .middleware import build_middleware
from .model import build_model
from .tools import build_tools


def _content(value: Any) -> Any:
    return getattr(value, "content", value)


def _text(value: Any) -> str:
    content = _content(value)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _message_kind(message: Any) -> str:
    return str(getattr(message, "type", ""))


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None)
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _parse_payload(value: Any) -> Any:
    content = _content(value)
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def _bounded(value: Any, limit: int) -> Any:
    """Bound public payloads while retaining structured data whenever it fits."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, dict):
        value = {str(key): _bounded(item, limit) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_bounded(item, limit) for item in value]
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(encoded) <= limit else encoded[:limit] + "…"



class _Trace:
    def __init__(self, emit: Callable[[dict[str, Any]], None] | None, event_limit: int, evidence_tool: str, tool_labels: dict[str, str] | None = None):
        self.emit = emit
        self.event_limit = event_limit
        self.messages: list[Any] = []
        self.records: list[dict[str, Any]] = []
        self.by_call_id: dict[str, dict[str, Any]] = {}
        self.rounds: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}
        self.answer = ""
        self.evidence_tool = evidence_tool
        self.tool_labels = tool_labels or {}
        self._seen_messages: set[str] = set()
        self._seen_skills: set[str] = set()

    def event(self, payload: dict[str, Any]) -> None:
        if not self.emit:
            return
        bounded = {
            key: _bounded(value, self.event_limit)
            for key, value in payload.items()
        }
        self.emit(bounded)

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

    def _consume_skills(self, state: dict[str, Any]) -> None:
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
        if not isinstance(update, dict):
            return
        states = [update] if isinstance(update.get("messages"), list) else list(update.values())
        for state in states:
            if not isinstance(state, dict):
                continue
            self._consume_skills(state)
            messages = state.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                fingerprint = self._message_fingerprint(message)
                if fingerprint in self._seen_messages:
                    continue
                self._seen_messages.add(fingerprint)
                self.messages.append(message)
                kind = _message_kind(message)
                calls = _tool_calls(message)
                if calls:
                    round_no = len(self.rounds) + 1
                    round_tools: list[str] = []
                    for call in calls:
                        name = str(call.get("name") or "")
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
                    self.answer = _text(message)
                    self.event({"type": "model", "title": "模型输出已更新", "message": "模型已完成一次公开输出更新"})
                if kind == "tool":
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
                        self.evidence = result
                    self.event({"type": "tool_result", "title": record.get("label") or effective_name or "tool", "message": "工具调用完成", "tool": effective_name, "label": record.get("label") or effective_name, "result": _bounded(result, self.event_limit), "callId": call_id})

    def result(self, thread_id: str, config: dict[str, Any], state: str = "completed") -> dict[str, Any]:
        output = config.get("harness", {}).get("output", {})
        answer_field = str(output.get("answer_field", "answer"))
        state_field = str(output.get("state_field", "state"))
        evidence_field = str(output.get("evidence_field", "evidence"))
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


class SeaVideoHarness:
    """Build and run one Deep Agents main agent with configured tools and middleware."""

    def __init__(self, config: dict[str, Any], service: Any, model: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None):
        self.config = config
        self.service = service
        self.event_handler = event_handler
        self.model = model if model is not None and callable(getattr(model, "bind_tools", None)) else build_model(config)
        self.tools = build_tools(config, service)
        harness = config.get("harness", {})
        prompt_file = project_root() / str(harness.get("system_prompt_file", "harness/system.md"))
        self.system_prompt = prompt_file.read_text(encoding="utf-8")
        self._connection: sqlite3.Connection | None = None
        self.agent = self._build_agent()

    def _build_agent(self) -> Any:
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            create_deep_agent,
            register_harness_profile,
        )
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.sqlite import SqliteSaver

        harness = self.config.get("harness", {})
        checkpoint = project_root() / str(harness.get("checkpointer", "data/memory/checkpoints.sqlite"))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(checkpoint), check_same_thread=False)
        saver = SqliteSaver(self._connection)
        disabled_tools = frozenset(str(item) for item in (harness.get("disabled_deepagent_tools") or []))
        profile = HarnessProfile(excluded_tools=disabled_tools, general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False))
        model_name = str(self.config.get("llm", {}).get("model", ""))
        if model_name:
            register_harness_profile(f"openai:{model_name}", profile)
        backend = FilesystemBackend(root_dir=project_root())
        skills_path = "/" + str(harness.get("skills_dir", "skills")).replace("\\", "/").strip("/")
        return create_deep_agent(model=self.model, tools=self.tools, system_prompt=self.system_prompt, skills=[skills_path], backend=backend, middleware=build_middleware(self.config, self.model), checkpointer=saver, name="sea_video_harness")

    def close(self) -> None:
        """Release the SQLite checkpoint connection owned by this run."""
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, _exc_value: BaseException | None, _traceback: TracebackType | None) -> None:
        self.close()

    def run(self, question: str, thread_id: str | None = None, **_: Any) -> dict[str, Any]:
        thread_id = thread_id or uuid.uuid4().hex
        harness = self.config.get("harness", {})
        tool_labels = {
            str(item.get("name")): str(item.get("label") or item.get("name"))
            for item in self.config.get("tools", [])
            if isinstance(item, dict) and item.get("name")
        }
        trace = _Trace(
            self.event_handler,
            int(harness.get("event_payload_max_chars", 4000)),
            str(harness.get("evidence_tool", "")),
            tool_labels,
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        try:
            for update in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=harness.get("stream_mode", "updates"),
            ):
                trace.consume(update)
        except Exception as error:  # noqa: BLE001 - provider/tool errors are runtime-defined
            message = str(error)[: trace.event_limit]
            trace.event({"type": "error", "title": "Harness 执行失败", "message": message})
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            return result
        finally:
            self.close()
        result = trace.result(thread_id, self.config)
        trace.event({"type": "complete", "title": "Harness 完成", "message": "回答与证据已生成"})
        return result

    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield the same public event contract as ``run`` without raw model state."""
        thread_id = thread_id or uuid.uuid4().hex
        harness = self.config.get("harness", {})
        pending: list[dict[str, Any]] = []

        def emit(event: dict[str, Any]) -> None:
            pending.append(event)
            if self.event_handler:
                self.event_handler(event)

        tool_labels = {
            str(item.get("name")): str(item.get("label") or item.get("name"))
            for item in self.config.get("tools", [])
            if isinstance(item, dict) and item.get("name")
        }
        trace = _Trace(
            emit,
            int(harness.get("event_payload_max_chars", 4000)),
            str(harness.get("evidence_tool", "")),
            tool_labels,
        )
        trace.event({"type": "status", "title": "Harness 已启动", "message": "已挂载 Skills、工具和记忆检查点"})
        try:
            while pending:
                yield pending.pop(0)
            for update in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=harness.get("stream_mode", "updates"),
            ):
                trace.consume(update)
                while pending:
                    yield pending.pop(0)
            result = trace.result(thread_id, self.config)
            trace.event({"type": "complete", "title": "Harness 完成", "message": "回答与证据已生成", "result": result})
            while pending:
                yield pending.pop(0)
        except Exception as error:  # noqa: BLE001 - provider/tool errors are runtime-defined
            message = str(error)[: trace.event_limit]
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            trace.event({"type": "error", "title": "Harness 执行失败", "message": message, "result": result})
            while pending:
                yield pending.pop(0)
        finally:
            self.close()


def run_harness(config: dict[str, Any], tools: Any, llm: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None, **kwargs: Any) -> dict[str, Any]:
    return SeaVideoHarness(config, tools, model=llm, event_handler=event_handler).run(kwargs.get("question", ""), kwargs.get("thread_id"))


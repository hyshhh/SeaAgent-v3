"""Thin HTTP/application boundary for the evidence-state multi-agent harness."""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from config import load_config
from harness.runtime import SeaVideoHarness
from memory import MemoryRepository
from tools import ToolService

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 一、控制器与问答流程                                                         #
# ------------------------------------------------------------------ #
# 1.1 装配服务并驱动一轮问答
class AgentController:
    # 1.1.1 装配配置、仓库与工具服务
    def __init__(self, config: dict[str, Any] | None = None, repository: Any = None, tools: Any = None, llm: Any = None, embedder: Any = None, vectors: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.tools = tools or ToolService(self.config, self.repository, embedder, llm, vectors)
        self.llm = llm
        self.event_handler = event_handler

    # 1.1.2 跑一轮问答并写入会话记忆
    def answer(self, question: str, session_id: str | None = None, cancel: Any = None) -> dict[str, Any]:
        session_id = str(session_id or "").strip() or f"session-{uuid.uuid4().hex[:12]}"
        session = self.repository.get_session(session_id)
        if session is None:
            self.repository.create_session(session_id, question)
            turn_index = 1
        else:
            turn_index = len(session.get("turns") or []) + 1
        runtime: SeaVideoHarness | None = None
        try:
            runtime = SeaVideoHarness(self.config, self.tools, event_handler=self.event_handler)
            state = runtime.run(question, thread_id=session_id, cancel=cancel)
            result = self._project(session_id, state, turn_index, question)
        except Exception as error:
            logger.exception("Agent controller failed: session_id=%s", session_id)
            result = self._error_result(session_id, question, error, "暂时无法完成视频检索。")
        finally:
            if runtime is not None:
                runtime.close()
        # 成功与失败都留档：失败轮次同样属于会话历史，前端据此还原完整对话
        result["sessionId"] = session_id
        result["turnIndex"] = self.repository.append_turn(session_id, question, result)
        return result

    # 1.1.3 人工确认后继续上一轮
    def resume(self, decision: str, session_id: str, feedback: str = "", cancel: Any = None) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("恢复确认必须带上 session_id（中断点是按它会话存的）")
        session = self.repository.get_session(session_id)
        turn_index = len((session or {}).get("turns") or []) + 1
        question = str((session or {}).get("title") or "（人工确认后继续）")
        runtime: SeaVideoHarness | None = None
        try:
            runtime = SeaVideoHarness(self.config, self.tools, event_handler=self.event_handler)
            state = runtime.resume(decision, thread_id=session_id, feedback=feedback, cancel=cancel)
            question = str((state.get("shared_state") or {}).get("question") or question)
            result = self._project(session_id, state, turn_index, question)
        except Exception as error:
            logger.exception("Agent controller resume failed: session_id=%s", session_id)
            result = self._error_result(session_id, question, error, "确认后仍无法完成这一步。")
        finally:
            if runtime is not None:
                runtime.close()
        result["sessionId"] = session_id
        result["turnIndex"] = self.repository.append_turn(session_id, question, result)
        return result

    # 1.1.4 异常时构造对外错误结果
    def _error_result(
        self, session_id: str, question: str, error: Exception, answer_text: str
    ) -> dict[str, Any]:
        message = str(error).strip() or f"{type(error).__name__}: {error!r}"
        return {
            "success": False,
            "sessionId": session_id,
            "question": str(question or ""),
            "answerText": answer_text,
            "conclusion": "",
            "state": "error",
            "evidence": {},
            "rounds": [],
            "toolChain": [],
            "toolRecords": [],
            "executionMode": str(
                self.config.get("harness", {}).get("execution_mode", "evidence-state-multi-agent")
            ),
            "error": message,
            "errorType": type(error).__name__,
        }

    # 1.1.5 把内部状态投成对外结构并写审计表
    def _project(self, session_id: str, state: dict[str, Any], turn_index: int = 1, question: str = "") -> dict[str, Any]:
        records = state.get("tool_records") or []
        rounds = state.get("rounds") or []
        persisted_rounds = {
            int(item.get("round")): item
            for item in rounds
            if isinstance(item, dict) and str(item.get("round", "")).isdigit()
        }
        for index, record in enumerate(records):
            round_number = int(record.get("round") or 0) or index + 1
            round_id = f"{session_id}-round-{turn_index}-{round_number}"
            round_info = persisted_rounds.get(round_number, {})
            self.repository.add_round(
                round_id,
                session_id,
                {"toolChain": round_info.get("toolChain", [record.get("tool")])},
                {"status": record.get("status", "completed")},
            )
            record_id = str(record.get("id") or index + 1)
            self.repository.add_evidence(
                f"{round_id}-evidence-{record_id}",
                round_id,
                record,
                {"tool": record.get("tool"), "callId": record_id},
            )
        output = self.config.get("harness", {}).get("output", {})
        answer_key = str(output.get("answer_field", "answer"))
        state_key = str(output.get("state_field", "state"))
        evidence_key = str(output.get("evidence_field", "evidence"))
        answer = str(state.get(answer_key) or state.get("answer") or "")
        run_state = str(state.get(state_key) or state.get("state") or "completed")
        evidence = state.get(evidence_key) or state.get("evidence") or {}
        harness_settings = self.config.get("harness", {})
        result = {
            # 确认等待由前端弹卡；未完成的运行以 stalled 状态呈现。
            "success": run_state == "completed",
            "sessionId": session_id,
            # 带上问句：证据面板用它显示 "Evidence for: …"，截图进论文时这张图能自证是哪一问
            "question": str(question or ""),
            "answerText": answer,
            "conclusion": answer,
            "state": run_state,
            "evidence": evidence,
            "rounds": state.get("rounds") or [],
            "toolChain": state.get("tool_chain") or [],
            "toolRecords": records,
            "sharedState": state.get("shared_state") or {},
            "partial": bool(state.get("partial", False)),
            "executionMode": str(harness_settings.get("execution_mode", "evidence-state-multi-agent")),
        }
        if state.get("error"):
            result["error"] = str(state["error"])
        if state.get("confirmation"):
            result["confirmation"] = state["confirmation"]
        result["enrichedEvidence"] = state.get(str(output.get("enriched_evidence_field", "enrichedEvidence"))) or {}
        return result

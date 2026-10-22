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
            original_question = question
            clarification = ""
        else:
            turns = session.get("turns") or []
            if turns and turns[-1].get("state") == "awaiting_confirmation":
                raise ValueError("请先完成当前问答的确认，再开始新问题")
            original_question = question
            clarification = ""
            if turns and turns[-1].get("state") == "awaiting_input":
                original_question = str(turns[-1].get("question") or question)
                clarification = str(turns[-1].get("answer") or "")
                clarification_response = question
                question = f"{original_question}\n\n用户补充信息：{clarification_response}"
        if not clarification:
            clarification_response = ""
        runtime_id = f"qa-{uuid.uuid4().hex}"
        runtime: SeaVideoHarness | None = None
        result: dict[str, Any] = {}
        try:
            runtime = SeaVideoHarness(self.config, self.tools, event_handler=self.event_handler)
            state = runtime.run(question, thread_id=runtime_id, cancel=cancel)
            result = self._project(session_id, state, original_question)
        except Exception as error:
            logger.exception("Agent controller failed: session_id=%s", session_id)
            result = self._error_result(session_id, question, error, "暂时无法完成视频检索。")
        finally:
            if runtime is not None:
                runtime.close()
                if result.get("state") != "awaiting_confirmation":
                    self._discard_run(runtime, runtime_id)
        if clarification:
            result["clarificationQuestion"] = str(result.get("clarification") or clarification)
            result["clarificationResponse"] = clarification_response
        # 成功与失败都留档：失败轮次同样属于会话历史，前端据此还原完整对话
        result["question"] = original_question
        result["sessionId"] = session_id
        if clarification:
            result["turnIndex"] = self.repository.replace_last_turn(
                session_id, original_question, result, runtime_id=runtime_id
            )
        else:
            result["turnIndex"] = self.repository.append_turn(
                session_id, question, result, runtime_id=runtime_id
            )
        return result

    # 1.1.3 人工确认后继续上一轮
    def resume(self, decision: str, session_id: str, feedback: str = "", cancel: Any = None) -> dict[str, Any]:
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("恢复确认必须带上 session_id（中断点是按它会话存的）")
        session = self.repository.get_session(session_id)
        turns = (session or {}).get("turns") or []
        if not turns or turns[-1].get("state") != "awaiting_confirmation":
            raise ValueError("当前会话没有等待确认的问答")
        pending_turn = turns[-1]
        runtime_id = str(pending_turn.get("runtimeId") or "")
        if not runtime_id:
            raise ValueError("等待确认的问答缺少运行标识，请重新发起该问题")
        question = str(pending_turn.get("question") or "（人工确认后继续）")
        runtime: SeaVideoHarness | None = None
        result: dict[str, Any] = {}
        try:
            runtime = SeaVideoHarness(self.config, self.tools, event_handler=self.event_handler)
            state = runtime.resume(decision, thread_id=runtime_id, feedback=feedback, cancel=cancel)
            question = str((state.get("shared_state") or {}).get("question") or question)
            result = self._project(session_id, state, question)
        except Exception as error:
            logger.exception("Agent controller resume failed: session_id=%s", session_id)
            result = self._error_result(session_id, question, error, "确认后仍无法完成这一步。")
            result.update(
                state="awaiting_confirmation",
                confirmation=pending_turn.get("confirmation") or {},
            )
        finally:
            if runtime is not None:
                runtime.close()
                if result.get("state") != "awaiting_confirmation":
                    self._discard_run(runtime, runtime_id)
        result["sessionId"] = session_id
        result["turnIndex"] = self.repository.replace_last_turn(
            session_id, question, result, runtime_id=runtime_id
        )
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

    # 1.1.5 把单次 QA 的运行结果投成 API 结构
    def _project(self, session_id: str, state: dict[str, Any], question: str = "") -> dict[str, Any]:
        records = state.get("tool_records") or []
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
            "success": run_state in {"completed", "awaiting_input"},
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
        if state.get("clarification"):
            result["clarification"] = str(state["clarification"])
        return result

    @staticmethod
    def _discard_run(runtime: Any, runtime_id: str) -> None:
        discard = getattr(runtime, "discard_run", None)
        if callable(discard):
            try:
                discard(runtime_id)
            except Exception:
                logger.exception("Could not clean temporary QA state: run_id=%s", runtime_id)

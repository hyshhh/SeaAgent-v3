"""Thin HTTP/application boundary for the single-agent harness."""
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


class AgentController:
    def __init__(self, config: dict[str, Any] | None = None, repository: Any = None, tools: Any = None, llm: Any = None, embedder: Any = None, vectors: Any = None, event_handler: Callable[[dict[str, Any]], None] | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.tools = tools or ToolService(self.config, self.repository, embedder, llm, vectors)
        self.llm = llm
        self.event_handler = event_handler

    def answer(self, question: str, session_id: str | None = None, cancel: Any = None) -> dict[str, Any]:
        """跑一轮问答，并把这轮问答写进会话记忆。

        ``session_id`` 决定这一轮是「新会话」还是「追问」：不传（或会话已不存在）就新建并
        以本轮问题作标题；传入已存在的会话则沿用同一个 thread_id —— SQLite 检查点会带回
        该会话的历史消息，模型因此看得到上文，前端也在这条会话下继续追加轮次。

        ``cancel`` 是可选的中断信号（``threading.Event``）：置位后本轮以 ``cancelled`` 收尾，
        问题仍会作为一轮记录留在会话里，只是没有回答——用户停掉的那一轮不该凭空消失。
        """
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
            message = str(error).strip() or f"{type(error).__name__}: {error!r}"
            result = {
                "success": False,
                "sessionId": session_id,
                "question": str(question or ""),
                "answerText": "暂时无法完成视频检索。",
                "conclusion": "",
                "state": "error",
                "evidence": {},
                "rounds": [],
                "toolChain": [],
                "toolRecords": [],
                "executionMode": str(self.config.get("harness", {}).get("execution_mode", "single-agent-harness")),
                "error": message,
                "errorType": type(error).__name__,
            }
        finally:
            if runtime is not None:
                runtime.close()
        # 成功与失败都留档：失败轮次同样属于会话历史，前端据此还原完整对话
        result["sessionId"] = session_id
        result["turnIndex"] = self.repository.append_turn(session_id, question, result)
        return result

    def resume(self, decision: str, session_id: str, feedback: str = "", cancel: Any = None) -> dict[str, Any]:
        """人工拍板后继续上一轮。

        中断点存在 SQLite 检查点里，所以恢复必须带着**同一个 session_id**（即同一个 thread_id）。
        批准则执行那个被拦下的工具；拒绝则把反馈交回模型让它换个做法，本轮继续往下跑。

        结果同样按一轮问答留档：用户看到的是「提问 → 确认 → 回答」，中间那次人工介入
        不该让这轮记录凭空消失。
        """
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
            result = self._project(session_id, state, turn_index, question)
        except Exception as error:
            logger.exception("Agent controller resume failed: session_id=%s", session_id)
            message = str(error).strip() or f"{type(error).__name__}: {error!r}"
            result = {
                "success": False,
                "sessionId": session_id,
                "question": str(question or ""),
                "answerText": "确认后仍无法完成这一步。",
                "conclusion": "",
                "state": "error",
                "evidence": {},
                "rounds": [],
                "toolChain": [],
                "toolRecords": [],
                "executionMode": str(self.config.get("harness", {}).get("execution_mode", "three-phase-subagents")),
                "error": message,
                "errorType": type(error).__name__,
            }
        finally:
            if runtime is not None:
                runtime.close()
        result["sessionId"] = session_id
        result["turnIndex"] = self.repository.append_turn(session_id, question, result)
        return result

    def _project(self, session_id: str, state: dict[str, Any], turn_index: int = 1, question: str = "") -> dict[str, Any]:
        """把一轮的内部状态投成对外结果，并把工具调用落进 qa_rounds / qa_evidence。

        ``turn_index`` 必须进主键：runtime 的轮次编号是**单轮内**计数（1..N），只按它拼 id 的话，
        会话第二轮的 `-round-1` 会把第一轮的同一行覆盖掉——审计表就只剩最后一轮了。
        """
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
            # awaiting_confirmation 不是「成功」也不是「失败」：前端据此弹确认卡而不是渲染回答
            # awaiting_confirmation 是「等人拍板」，delegation_timeout 是「这一步太重没跑完」——
            # 两者都不算成功，前端据此弹卡片或标失败，而不是渲染成正常回答
            "success": run_state not in {"error", "cancelled", "awaiting_confirmation", "delegation_timeout"},
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
            "executionMode": str(harness_settings.get("execution_mode", "single-agent-harness")),
        }
        if state.get("error"):
            result["error"] = str(state["error"])
        return result

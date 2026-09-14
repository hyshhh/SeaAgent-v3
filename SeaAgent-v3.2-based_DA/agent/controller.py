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

    def answer(self, question: str) -> dict[str, Any]:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        self.repository.add_session(session_id, {"question": question})
        runtime: SeaVideoHarness | None = None
        try:
            runtime = SeaVideoHarness(self.config, self.tools, event_handler=self.event_handler)
            state = runtime.run(question, thread_id=session_id)
            result = self._project(session_id, state)
            self.repository.finish_session(session_id, result)
            return result
        except Exception as error:
            logger.exception("Agent controller failed: session_id=%s", session_id)
            message = str(error).strip() or f"{type(error).__name__}: {error!r}"
            result = {
                "success": False,
                "sessionId": session_id,
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
            self.repository.finish_session(session_id, result)
            return result
        finally:
            if runtime is not None:
                runtime.close()

    def _project(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        records = state.get("tool_records") or []
        rounds = state.get("rounds") or []
        persisted_rounds = {
            int(item.get("round")): item
            for item in rounds
            if isinstance(item, dict) and str(item.get("round", "")).isdigit()
        }
        for index, record in enumerate(records):
            round_number = int(record.get("round") or 0) or index + 1
            round_id = f"{session_id}-round-{round_number}"
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
            "success": run_state != "error",
            "sessionId": session_id,
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

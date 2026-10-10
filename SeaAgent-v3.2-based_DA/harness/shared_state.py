"""Shared task state for the planner, executor and reflector agents."""
from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any


STATE_FIELDS = (
    "question",
    "requirements",
    "plan",
    "next_action",
    "evidence_ids",
    "tool_results",
    "execution_status",
    "assessment",
    "evidence_gaps",
    "conflicts",
    "decision",
    "task_status",
    "iteration",
    "version",
    "pending_agent",
)

OWNER_FIELDS = {
    "planner": {"requirements", "plan", "next_action"},
    "executor": {"evidence_ids", "tool_results", "execution_status"},
    "reflector": {"assessment", "evidence_gaps", "conflicts", "decision"},
    "runtime": {"question", "task_status", "iteration", "version", "pending_agent"},
}


def _json_value(value: Any, limit: int = 6000) -> Any:
    """Keep persisted state bounded while retaining its structure."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {str(key): _json_value(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, limit) for item in value[-100:]]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


class SharedTaskState:
    """Versioned dictionary with explicit write ownership."""

    def __init__(self, question: str, values: dict[str, Any] | None = None):
        self._lock = threading.RLock()
        defaults = {
            "question": str(question or ""),
            "requirements": [],
            "plan": [],
            "next_action": None,
            "evidence_ids": [],
            "tool_results": [],
            "execution_status": "idle",
            "assessment": {},
            "evidence_gaps": [],
            "conflicts": [],
            "decision": None,
            "task_status": "running",
            "iteration": 0,
            "version": 1,
            "pending_agent": None,
        }
        if isinstance(values, dict):
            defaults.update({key: deepcopy(values[key]) for key in STATE_FIELDS if key in values})
        self._values = defaults

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._values)

    def update(self, role: str, patch: dict[str, Any], expected_version: int | None = None) -> dict[str, Any]:
        role = str(role or "").strip()
        if role not in OWNER_FIELDS:
            raise ValueError(f"unknown state owner: {role}")
        if not isinstance(patch, dict):
            raise ValueError("state patch must be an object")
        with self._lock:
            current_version = int(self._values["version"])
            if expected_version is not None and int(expected_version) != current_version:
                return {"ok": False, "error": "stale_state_version", "version": current_version}
            forbidden = sorted(set(patch) - OWNER_FIELDS[role])
            if forbidden:
                raise ValueError(f"{role} cannot update: {', '.join(forbidden)}")
            for key, value in patch.items():
                self._values[key] = deepcopy(value)
            self._values["version"] = current_version + 1
            return {"ok": True, "version": self._values["version"], "state": self.snapshot()}

    def runtime_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        return self.update("runtime", patch)

    def record_tool_result(self, agent: str, tool: str, call_id: str, result: Any, status: str) -> None:
        """Append a bounded execution record and collect evidence identifiers."""
        with self._lock:
            records = list(self._values["tool_results"])
            records.append({
                "agent": str(agent or ""),
                "tool": str(tool or ""),
                "callId": str(call_id or ""),
                "status": str(status or "completed"),
                "result": _json_value(result),
            })
            evidence = list(self._values["evidence_ids"])
            self._collect_ids(result, evidence)
            self._values["tool_results"] = records[-100:]
            self._values["evidence_ids"] = list(dict.fromkeys(evidence))[-500:]
            self._values["execution_status"] = "failed" if str(status) == "error" else "running"
            self._values["version"] = int(self._values["version"]) + 1

    @staticmethod
    def _collect_ids(value: Any, output: list[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {
                    "evidenceids", "keyframeids", "shipsegmentids",
                    "registryreferenceids", "trackids",
                } and isinstance(item, list):
                    output.extend(str(entry) for entry in item if entry)
                SharedTaskState._collect_ids(item, output)
        elif isinstance(value, list):
            for item in value:
                SharedTaskState._collect_ids(item, output)

    def apply_agent_text(self, role: str, text: str) -> dict[str, Any] | None:
        """Recover a JSON state patch when a provider skips the state tool."""
        payload = _extract_json(text)
        if not isinstance(payload, dict):
            return None
        if role == "planner":
            patch = {
                key: payload[key]
                for key in ("requirements", "plan", "next_action")
                if key in payload
            }
            if not patch and "checklist" in payload:
                patch["requirements"] = payload["checklist"]
        elif role == "reflector":
            patch = {
                key: payload[key]
                for key in ("assessment", "evidence_gaps", "conflicts", "decision")
                if key in payload
            }
            if "summary" in payload:
                patch.setdefault("assessment", {})["summary"] = payload["summary"]
            if "final_answer" in payload:
                patch.setdefault("assessment", {})["final_answer"] = payload["final_answer"]
            if "can_exit" in payload and "decision" not in patch:
                patch["decision"] = "terminate" if payload["can_exit"] else "replan"
        else:
            patch = {
                key: payload[key]
                for key in ("execution_status", "evidence_ids")
                if key in payload
            }
        return self.update(role, patch) if patch else None

    def to_json(self) -> dict[str, Any]:
        return _json_value(self.snapshot())


def _extract_json(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    candidates = [text]
    if "```" in text:
        candidates.extend(part for part in text.split("```") if "{" in part)
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(candidate[start:end + 1])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


class SharedStateStore:
    """Small atomic JSON store used to survive HTTP reconnects and resume calls."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def load(self, thread_id: str, question: str = "") -> SharedTaskState:
        with self._lock:
            data: dict[str, Any] = {}
            if self.path.is_file():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    data = raw.get(str(thread_id), {}) if isinstance(raw, dict) else {}
                except (OSError, TypeError, ValueError):
                    data = {}
            return SharedTaskState(question, data if isinstance(data, dict) else {})

    def save(self, thread_id: str, state: SharedTaskState) -> None:
        with self._lock:
            payload: dict[str, Any] = {}
            if self.path.is_file():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    payload = raw if isinstance(raw, dict) else {}
                except (OSError, TypeError, ValueError):
                    payload = {}
            payload[str(thread_id)] = state.to_json()
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)

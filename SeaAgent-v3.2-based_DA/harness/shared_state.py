"""三 Agent 的唯一任务状态：角色权限、执行进度、调用账本与 SQLite 恢复。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ #
# 一、字段归属与载荷校验
# ------------------------------------------------------------------ #
# 1.1 声明各角色可维护的字段与合法状态
OWNER_FIELDS = {
    "planner": {"requirements", "plan", "next_action", "clarification"},
    "executor": {"evidence_ids", "step_status", "execution_status"},
    "reflector": {"assessment", "evidence_gaps", "conflicts", "decision"},
    "runtime": {"question", "task_status", "iteration", "pending_agent"},
}
ROLE_UPDATE_TOOLS = {
    "planner": "update_planner_state",
    "executor": "update_execution_state",
    "reflector": "update_reflection_state",
}
STATE_TOOLS = frozenset(   #状态工具独立，不可变
    {
        "read_shared_state",
        "update_planner_state",
        "update_execution_state",
        "update_reflection_state",
    }
)
STATE_UPDATE_TOOLS = frozenset(ROLE_UPDATE_TOOLS.values())
STEP_STATUSES = {"pending", "in_progress", "completed", "failed"}   #步骤状态【没开始/正在做/做完了/失败了】
MAX_TOOL_RESULTS_BYTES = 160_000   #tool_result列表的容量上限
MAX_PLAN_BYTES = 32_000  #paln字段的字节上限
MAX_ASSESSMENT_BYTES = 64_000 #reflector的判定结论【证据和回答】
MAX_INLINE_RESULT_BYTES = 12_000 # 单条工具结果上限，过多就转移到其他地方而不会写入状态空间


def _json_bytes(value: Any) -> int:  #字节判定工具
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))  #ASCII（1960 年代）它只有英文，没有中文
#UTF-8 —— 现在通用的编码 （一个汉字三个字节）


def _truncate_text(value: str, limit: int) -> str:  #按照字节截断方法
    raw = str(value).encode("utf-8")
    if len(raw) <= limit:
        return str(value)
    suffix = "…".encode()
    return raw[: max(0, limit - len(suffix))].decode("utf-8", "ignore") + "…"


def _bounded_payload(value: Any, limit: int) -> Any:
    value = _bounded_result(value)
    if _json_bytes(value) <= limit:
        return value
    if isinstance(value, str):
        return _truncate_text(value, limit)
    if isinstance(value, list):
        result = []
        for item in value:
            candidate = result + [item]
            if _json_bytes(candidate) > limit:
                if not result:
                    result.append(_bounded_payload(item, max(1, limit - 2)))
                break
            result.append(item)
        return result
    if isinstance(value, dict):
        items = list(value.items())
        if "final_answer" in value:
            items.sort(key=lambda item: item[0] != "final_answer")
        result = {}
        for key, item in items:
            candidate = {**result, str(key): _bounded_payload(item, limit)}
            if _json_bytes(candidate) > limit:
                continue
            result[str(key)] = candidate[str(key)]
        return result
    return _truncate_text(str(value), limit)


def _bounded_plan(value: Any) -> list[dict[str, Any]]:  #plan字段丢计划的方法
    steps = _plan_steps(value)
    while _json_bytes(steps) > MAX_PLAN_BYTES and len(steps) > 1:
        steps.pop()
    if steps and _json_bytes(steps) > MAX_PLAN_BYTES:
        step = steps[0]
        step["id"] = _truncate_text(step["id"], 256)
        step["content"] = _truncate_text(step["content"], max(64, MAX_PLAN_BYTES // 2))
        step["required_evidence"] = [str(item) for item in step["required_evidence"][:20]]
    bounded = _bounded_payload(steps, MAX_PLAN_BYTES)
    return bounded if isinstance(bounded, list) else []


def _bounded_assessment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("assessment must be an object")
    bounded = _bounded_payload(value, MAX_ASSESSMENT_BYTES)
    return bounded if isinstance(bounded, dict) else {}


# 1.2 共享存储冲突：旧实例不得覆盖新版本
class StateConflictError(RuntimeError):
    """另一运行已保存了此会话的新版本；旧快照不能覆盖它。"""


# 1.3 裁剪工具结果并归一为可持久化载荷
def _bounded_result(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 6000 else value[:6000] + "..."
    if isinstance(value, dict):
        return {str(key): _bounded_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bounded_result(item) for item in value[:100]]
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


# 1.4 归一计划步骤，兼容历史字符串计划
def _plan_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("plan must be a list")
    steps = []
    for index, item in enumerate(value):
        if isinstance(item, str):  # 兼容已有存档的字符串计划。
            item = {"id": f"step-{index + 1}", "content": item}
        if not isinstance(item, dict) or not str(item.get("content") or "").strip():
            raise ValueError("each plan step needs id and non-empty content")
        evidence = item.get("required_evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(entry, str) for entry in evidence):
            raise ValueError("required_evidence must be a list of strings")
        steps.append(
            {
                "id": str(item.get("id") or f"step-{index + 1}"),
                "content": str(item["content"]).strip(),
                "required_evidence": evidence,
            }
        )
    if len({step["id"] for step in steps}) != len(steps):
        raise ValueError("plan step ids must be unique")
    return steps


def _session_artifact_dir(root: Path, thread_id: str) -> Path:
    key = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
    return root / key


# ------------------------------------------------------------------ #
# 二、共享任务状态
# ------------------------------------------------------------------ #
# 2.1 三个 Agent 的唯一业务状态与调用账本
class SharedTaskState:
    # 2.1.1 初始化字段，恢复已保存的状态
    def __init__(
        self,
        question: str,
        values: dict[str, Any] | None = None,
        artifact_dir: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ):
        self._lock = threading.RLock()
        self._saved_version: int | None = None
        default_artifact_root = Path(__file__).resolve().parents[1] / "data" / "memory" / "tool_results"
        self._artifact_root = Path(artifact_root or artifact_dir or default_artifact_root).resolve()
        self._artifact_dir = Path(artifact_dir or self._artifact_root).resolve()
        self._values = {
            "question": str(question or ""),
            "requirements": [],
            "plan": [],
            "next_action": None,
            "clarification": None,
            "step_status": {},
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
            "agent_calls": {},
            "call_history": {},
            "role_write_counts": {},
            "agent_start_write_counts": {},
            "evidence": {},
        }
        if values:
            original_result_refs = self._external_refs(values.get("tool_results", []))
            self._values.update(
                {key: deepcopy(values[key]) for key in self._values if key in values}
            )
            self._values["plan"] = _bounded_plan(self._values["plan"])
            self._values["assessment"] = _bounded_assessment(self._values["assessment"])
            self._values["tool_results"] = self._bounded_tool_results(self._values["tool_results"])
            self._cleanup_artifacts(
                original_result_refs - self._external_refs(self._values["tool_results"])
            )

    # 2.1.2 获取完整快照，供运行时与持久化使用
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._values)

    # 2.1.3 获取业务投影，供 Agent 与前端读取
    def view(self) -> dict[str, Any]:
        """Agent 和界面读取同一状态的业务投影，避免把缓存结果再次塞入上下文。"""
        with self._lock:
            return {
                key: deepcopy(value)
                for key, value in self._values.items()
                if key not in {
                    "call_history",
                    "role_write_counts", "agent_start_write_counts",
                }
            }

    # 2.1.4 推进业务版本号
    def _changed(self) -> None:
        self._values["version"] += 1

    # 2.1.5 校验字段归属并原子合并角色补丁
    def update(self, role: str, patch: dict[str, Any]) -> dict[str, Any]:
        if role not in OWNER_FIELDS or not isinstance(patch, dict):
            raise ValueError("unknown owner or invalid state patch")
        if not patch:
            raise ValueError("state patch must not be empty")
        forbidden = sorted(set(patch) - OWNER_FIELDS[role])
        if forbidden:
            raise ValueError(f"{role} cannot update: {', '.join(forbidden)}")
        with self._lock:  #python上下文管理器，进入加锁，离开解锁
            version = self._values["version"]
            patch = deepcopy(patch)
            self._validate(patch)
            if "plan" in patch:
                patch["plan"] = _bounded_plan(patch["plan"])
                old_steps = {step["id"]: step for step in self._values["plan"]}
                self._values["step_status"] = {
                    step["id"]: self._values["step_status"].get(step["id"], "pending")
                    if old_steps.get(step["id"]) == step
                    else "pending"
                    for step in patch["plan"]
                }
            if "step_status" in patch:
                patch["step_status"] = {**self._values["step_status"], **patch["step_status"]}
            if "evidence_ids" in patch:
                patch["evidence_ids"] = list(
                    dict.fromkeys(self._values["evidence_ids"] + patch["evidence_ids"])
                )
            if "assessment" in patch:
                patch["assessment"] = _bounded_assessment(
                    {**self._values["assessment"], **patch["assessment"]}
                )
            if "evidence_ids" in patch:
                known_ids: list[str] = []
                for item in self._values["tool_results"]:
                    if item.get("status") != "error":
                        self._collect_ids(item.get("result"), known_ids)
                unknown = sorted(
                    set(patch["evidence_ids"])
                    - set(known_ids)
                    - set(self._values["evidence_ids"])
                )
                if unknown:
                    raise ValueError(
                        "evidence_ids 只能包含成功工具结果中的实体证据 ID（如 trackId、keyframeId、"
                        "shipSegmentId、registryReferenceId），不能填 tool_call_id；运行时已自动收集证据，"
                        f"通常可省略此字段。无效 ID：{', '.join(unknown)}"
                    )
            changed = any(self._values.get(key) != value for key, value in patch.items())
            counts = self._values["role_write_counts"]
            counts[role] = counts.get(role, 0) + 1
            if not changed:
                return {
                    "ok": True,
                    "unchanged": True,
                    "version": version,
                    "message": "状态已是最新值，无需重复修改。",
                }
            self._values.update(patch)
            self._changed()
            return {"ok": True, "version": self._values["version"]}

    # 2.1.6 校验类型、枚举、步骤 ID 与可序列化性
    def _validate(self, patch: dict[str, Any]) -> None:
        for key in {"requirements", "evidence_ids", "evidence_gaps", "conflicts"} & patch.keys():
            if not isinstance(patch[key], list) or any(
                not isinstance(item, str) for item in patch[key]
            ):
                raise ValueError(f"{key} must be a list of strings")
        if "assessment" in patch and not isinstance(patch["assessment"], dict):
            raise ValueError("assessment must be an object")
        if (
            "assessment" in patch
            and "final_answer" in patch["assessment"]
            and not isinstance(patch["assessment"]["final_answer"], str)
        ):
            raise TypeError("assessment.final_answer must be text")
        enums = {
            "execution_status": {"idle", "running", "success", "failed"},
            "decision": {None, "terminate", "replan", "continue", "failed"},
            "task_status": {
                "running",
                "completed",
                "stalled",
                "cancelled",
                "awaiting_confirmation",
                "awaiting_input",
                "failed",
                "error",
            },
            "pending_agent": {None, "planner", "executor", "reflector"},
        }
        for key in enums.keys() & patch.keys():
            if not isinstance(patch[key], (str, type(None))) or patch[key] not in enums[key]:
                raise ValueError(f"invalid {key}: {patch[key]!r}")
        if "step_status" in patch:
            statuses = patch["step_status"]
            ids = {step["id"] for step in self._values["plan"]}
            if not isinstance(statuses, dict) or any(
                key not in ids or not isinstance(value, str) or value not in STEP_STATUSES
                for key, value in statuses.items()
            ):
                raise ValueError("step_status must refer to current plan ids and valid statuses")
        if "iteration" in patch and (type(patch["iteration"]) is not int or patch["iteration"] < 0):
            raise ValueError("iteration must be a non-negative integer")
        if (
            "next_action" in patch
            and patch["next_action"] is not None
            and not isinstance(patch["next_action"], (str, dict))
        ):
            raise ValueError("next_action must be text, an object or null")
        if "clarification" in patch:
            clarification = patch["clarification"]
            if clarification is not None and not isinstance(clarification, str):
                raise TypeError("clarification must be text or null")
            if isinstance(clarification, str) and len(clarification) > 1000:
                raise ValueError("clarification must be at most 1000 characters")
        json.dumps(patch, ensure_ascii=False, allow_nan=False)

    # 2.1.7 更新由运行时维护的调度字段
    def runtime_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        return self.update("runtime", patch)

    # 2.1.8 开始新问答，重置本轮调用数并递增版本
    def start_turn(self, question: str) -> None:
        """新问答清空任务字段，调用预算从本轮重新计算。"""
        with self._lock:
            self._clear_turn_artifacts()
            version = self._values["version"]
            self._values = SharedTaskState(
                question, artifact_dir=self._artifact_dir, artifact_root=self._artifact_root
            ).snapshot()
            self._values.update(version=version + 1)

    def _clear_turn_artifacts(self) -> None:
        self._cleanup_artifacts(self._external_refs(self._values.get("tool_results", [])))
        if self._artifact_dir != self._artifact_root:
            shutil.rmtree(self._artifact_dir, ignore_errors=True)

    # 2.1.9 开始一个角色阶段，重置阶段预算与旧反思结论
    def start_agent(self, role: str) -> None:
        with self._lock:
            self._values["pending_agent"] = role
            self._values["agent_calls"][role] = 0
            self._values["agent_start_write_counts"][role] = self._values["role_write_counts"].get(role, 0)
            if role == "reflector":
                self._values["decision"] = None
                self._values["assessment"] = {}
                self._values["evidence_gaps"] = []
                self._values["conflicts"] = []
            self._changed()

    # 2.1.9.1 查询当前角色是否达到自己的工具预算
    def limit_reason(self, role: str, tool_budget: int) -> str:
        with self._lock:
            used = self._values["agent_calls"].get(role, 0)
            return "agent_tool_limit" if tool_budget > 0 and used >= tool_budget else ""

    # 2.1.10 执行前原子占用工具预算并登记请求
    def claim_call(
        self,
        role: str,
        key: str | None,
        tool_budget: int,
        write: bool = False,
        count: bool = True,
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """原子占用当前角色预算并登记请求，重连不能重复写入。"""
        with self._lock:
            if count and tool_budget > 0 and self._values["agent_calls"].get(role, 0) >= tool_budget:
                return {"blocked": "agent_tool_limit"}
            if count:
                self._values["agent_calls"][role] = self._values["agent_calls"].get(role, 0) + 1
            previous = self._values["call_history"].get(key) if key else None
            if key and previous is None:
                self._values["call_history"][key] = {
                    "status": "pending",
                    "write": write,
                    "attempts": 1,
                }
            elif (
                previous
                and previous["status"] == "error"
                and not previous.get("write")
                and previous.get("attempts", 1) < max(1, max_attempts)
            ):
                previous.update(status="pending", attempts=previous.get("attempts", 1) + 1)
                self._changed()
                return {"execute": True}
            if key is not None or count:
                self._changed()
            return deepcopy(previous) if previous else {"execute": True}

    # 2.1.11 保存调用结果，写入后使相关查询缓存失效
    def finish_call(self, key: str | None, result: Any, failed: bool) -> None:
        if key is None:
            return
        with self._lock:
            record = self._values["call_history"][key]
            bounded = _bounded_payload(result, MAX_INLINE_RESULT_BYTES)
            truncated = json.dumps(bounded, default=str) != json.dumps(result, default=str)
            record.update(
                status="error" if failed else "completed", result=bounded, truncated=truncated
            )
            if record.get("write"):
                # 即使写入结果不确定，旧先验库查询也可能已失效。
                self._values["call_history"] = {
                    name: value
                    for name, value in self._values["call_history"].items()
                    if not name.startswith(
                        ("get_registry:", "list_registry:", "match_hull:", "match_text:")
                    )
                }
            self._changed()

    # 2.1.12 记录领域工具结果，只从成功结果收集证据
    def record_tool_result(
        self, agent: str, tool: str, call_id: str, result: Any, status: str
    ) -> None:
        with self._lock:
            records = self._values["tool_results"]
            if call_id and any(item["callId"] == call_id for item in records):
                return
            previous_refs = self._external_refs(records)
            stored_result = self._externalize_result(result)
            records.append(
                {
                    "agent": agent,
                    "tool": tool,
                    "callId": call_id,
                    "status": status,
                    "result": stored_result,
                }
            )
            bounded_records = self._bounded_tool_results(records)
            self._values["tool_results"] = bounded_records
            self._cleanup_artifacts(previous_refs - self._external_refs(bounded_records))
            if status != "error":
                evidence = list(self._values["evidence_ids"])
                self._collect_ids(result, evidence)
                self._values["evidence_ids"] = list(dict.fromkeys(evidence))
                if tool == "show_evidence" and isinstance(result, dict):
                    self._values["evidence"] = deepcopy(result)
            if agent == "executor":
                self._values["execution_status"] = "failed" if status == "error" else "running"
            self._changed()

    def _externalize_result(self, result: Any) -> Any:
        """把超大工具结果写到会话外部文件，状态只保留可控预览。"""
        try:
            raw = json.dumps(result, ensure_ascii=False, default=str, allow_nan=False)
        except (TypeError, ValueError):
            raw = json.dumps(str(result), ensure_ascii=False)
        if len(raw.encode("utf-8")) <= MAX_INLINE_RESULT_BYTES:
            return _bounded_result(result)
        try:
            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            path = self._artifact_dir / f"tool-result-{uuid.uuid4().hex}.json"
            path.write_text(raw, encoding="utf-8")
            ids: list[str] = []
            self._collect_ids(result, ids)
            return {
                "external_ref": str(path),
                "bytes": len(raw.encode("utf-8")),
                "preview": _bounded_payload(result, MAX_INLINE_RESULT_BYTES // 2),
                "evidenceIds": list(dict.fromkeys(ids)),
            }
        except OSError:
            return _bounded_payload(result, MAX_INLINE_RESULT_BYTES)

    @staticmethod
    def _bounded_tool_results(records: Any) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            return []
        bounded = [_bounded_payload(item, MAX_INLINE_RESULT_BYTES) for item in records if isinstance(item, dict)]
        while len(bounded) > 1 and _json_bytes(bounded) > MAX_TOOL_RESULTS_BYTES:
            bounded.pop(0)
        if bounded and _json_bytes(bounded) > MAX_TOOL_RESULTS_BYTES:
            bounded = [_bounded_payload(bounded[-1], MAX_TOOL_RESULTS_BYTES)]
        return bounded

    @staticmethod
    def _external_refs(records: Any) -> set[str]:
        refs: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                ref = value.get("external_ref")
                if isinstance(ref, str):
                    refs.add(ref)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(records)
        return refs

    def _cleanup_artifacts(self, refs: set[str]) -> None:
        root = self._artifact_root.resolve()
        for ref in refs:
            try:
                path = Path(ref).resolve()
                if path.is_relative_to(root) and path.is_file():
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    # 2.1.13 递归收集轨迹、关键帧、片段与参考图 ID
    @staticmethod
    def _collect_ids(value: Any, output: list[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.replace("_", "").lower()
                if normalized in {
                    "evidenceids",
                    "keyframeids",
                    "shipsegmentids",
                    "registryreferenceids",
                    "trackids",
                } and isinstance(item, list):
                    output.extend(str(entry) for entry in item if entry)
                elif (
                    normalized in {"keyframeid", "shipsegmentid", "registryreferenceid", "trackid"}
                    and isinstance(item, str)
                    and item
                ):
                    output.append(item)
                SharedTaskState._collect_ids(item, output)
        elif isinstance(value, list):
            for item in value:
                SharedTaskState._collect_ids(item, output)


# ------------------------------------------------------------------ #
# 三、共享状态持久化
# ------------------------------------------------------------------ #
# 3.1 SQLite 事务存储与修订号校验
class SharedStateStore:
    """每会话一行，以事务和版本比较防止多实例覆盖；保留原 JSON 存档。"""

    # 3.1.1 初始化表与 WAL，首次导入历史 JSON
    def __init__(self, path: str | Path):
        supplied = Path(path)
        self.path = supplied.with_suffix(".sqlite") if supplied.suffix == ".json" else supplied
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = (self.path.parent / "tool_results").resolve()
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS task_states (thread_id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS store_metadata (key TEXT PRIMARY KEY)")
            legacy = self.path.with_suffix(".json")
            imported = connection.execute(
                "SELECT 1 FROM store_metadata WHERE key = 'legacy_json_imported'"
            ).fetchone()
            if legacy.is_file() and not imported:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("legacy shared state must be an object")
                for thread_id, values in data.items():
                    state = SharedTaskState(
                        "",
                        values,
                        _session_artifact_dir(self.artifact_root, thread_id),
                        self.artifact_root,
                    ).snapshot()
                    connection.execute(
                        "INSERT OR IGNORE INTO task_states VALUES (?, ?, ?)",
                        (thread_id, state["version"], json.dumps(state, ensure_ascii=False)),
                    )
                connection.execute("INSERT INTO store_metadata VALUES ('legacy_json_imported')")

    # 3.1.2 删除单个会话的共享状态
    def delete(self, thread_id: str) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            row = connection.execute(
                "SELECT payload FROM task_states WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            connection.execute("DELETE FROM task_states WHERE thread_id = ?", (thread_id,))
        self._remove_artifacts(thread_id, row[0] if row else None)

    # 3.1.3 清空全部会话的共享状态
    def clear(self) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            rows = connection.execute("SELECT thread_id, payload FROM task_states").fetchall()
            connection.execute("DELETE FROM task_states")
        for thread_id, payload in rows:
            self._remove_artifacts(thread_id, payload)
        shutil.rmtree(self.artifact_root, ignore_errors=True)

    def _remove_artifacts(self, thread_id: str, payload: str | None) -> None:
        state = SharedTaskState(
            "",
            artifact_dir=_session_artifact_dir(self.artifact_root, thread_id),
            artifact_root=self.artifact_root,
        )
        if payload:
            try:
                values = json.loads(payload)
                state._cleanup_artifacts(state._external_refs(values.get("tool_results", [])))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        shutil.rmtree(state._artifact_dir, ignore_errors=True)

    # 3.1.4 加载状态及其存储修订号
    def load(self, thread_id: str, question: str = "") -> SharedTaskState:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT version, payload FROM task_states WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        state = SharedTaskState(
            question,
            json.loads(row[1]) if row else None,
            _session_artifact_dir(self.artifact_root, thread_id),
            self.artifact_root,
        )
        state._saved_version = row[0] if row else None
        return state

    # 3.1.5 在事务内比较修订号，拒绝旧快照覆盖
    def save(self, thread_id: str, state: SharedTaskState) -> None:
        with state._lock:
            with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT version FROM task_states WHERE thread_id = ?", (thread_id,)
                ).fetchone()
                saved = row[0] if row else None
                if saved != state._saved_version:
                    raise StateConflictError(
                        "shared task changed in another run; reload before continuing"
                    )
                payload = state.snapshot()
                revision = (
                    saved or 0
                ) + 1  # 存储修订号独立于业务版本，计数变化也不能被旧实例覆盖。
                connection.execute(
                    "INSERT INTO task_states VALUES (?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET version=excluded.version, payload=excluded.payload",
                    (thread_id, revision, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
                )
            state._saved_version = revision

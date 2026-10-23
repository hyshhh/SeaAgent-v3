"""三阶段协同的运行核心：原始 agent state 不出模块，只暴露裁剪后的事件与 result 字典。"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any, Self

from config import project_root

from .middleware import build_middleware
from .model import build_model
from .shared_state import (
    STATE_TOOLS,
    SharedStateStore,
    SharedTaskState,
    StateConflictError,
)
from .state_tools import build_state_tools
from .state_write import AgentCloseoutComplete, MissingStateWrite
from .tool_guard import AgentCancelled
from .tools import build_tools
from .trace import _Trace

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 一、运行辅助工具
# ------------------------------------------------------------------ #
# 1.1 裁剪对外错误信息
def _error_message(error: BaseException, limit: int) -> str:
    message = str(error).strip()
    if not message:
        message = f"{type(error).__name__}: {error!r}"
    return message if len(message) <= limit else message[:limit] + "…"


_BUILTIN_TOOL_LABELS = {  # 内置工具的中文标签；读技能正文那条由读取面收敛而来
    "read_file": "读取技能正文",
    "ls": "技能目录",
    "glob": "技能检索",
    "grep": "技能内容检索",
    "read_shared_state": "读取共享状态",
    "update_planner_state": "更新任务计划",
    "update_execution_state": "更新执行进度",
    "update_reflection_state": "更新证据审查",
}


# 1.2 合并内置与配置中的工具标签
def _tool_labels(config: dict[str, Any]) -> dict[str, str]:
    labels = dict(_BUILTIN_TOOL_LABELS)
    for item in config.get("tools", []):
        if isinstance(item, dict) and item.get("name"):
            name = str(item["name"])
            labels[name] = str(item.get("label") or name)
    return labels


# 1.3 把人工确认转换为检查点恢复命令
def _resume_command(decision: str, feedback: str) -> Any:
    from langgraph.types import Command

    if decision == "approve":
        decisions = [{"type": "approve"}]
    else:
        decisions = [
            {"type": "reject", "message": str(feedback or "用户拒绝了这次写入，请换一种做法。")}
        ]
    return Command(resume={"decisions": decisions})


# 1.4 提取当前帧中的人工确认中断
def _interrupt_payload(update: Any) -> list[Any]:
    if not isinstance(update, dict):
        return []
    pending = update.get("__interrupt__")
    if isinstance(pending, (list, tuple)):
        return list(pending)
    return [pending] if pending is not None else []


# 1.5 将中断信息投影为前端确认载荷
def _confirmation_from(interrupts: list[Any]) -> dict[str, Any]:
    first = interrupts[0]
    value = getattr(first, "value", first)
    value = value if isinstance(value, dict) else {}
    requests = value.get("action_requests") or []
    configs = value.get("review_configs") or []
    actions = []
    for index, request in enumerate(requests):
        request = request if isinstance(request, dict) else {}
        config = configs[index] if index < len(configs) and isinstance(configs[index], dict) else {}
        actions.append(
            {
                "tool": str(request.get("name") or ""),
                "arguments": request.get("args") if isinstance(request.get("args"), dict) else {},
                "description": str(request.get("description") or ""),
                "allowedDecisions": list(config.get("allowed_decisions") or ["approve", "reject"]),
            }
        )
    return {"interruptId": str(getattr(first, "id", "") or ""), "actions": actions}


# ------------------------------------------------------------------ #
# 二、技能与权限配置
# ------------------------------------------------------------------ #
# 2.1 按角色技能组生成挂载路径
def _skill_sources(harness: dict[str, Any], groups: list[str]) -> list[str]:  #将真实路径转换为虚拟路径，Deep_agent的filesystem约定，不会给模型看真实的路径
    root = "/" + str(harness.get("skills_dir", "skills")).replace("\\", "/").strip("/")
    return [f"{root}/{str(group).strip('/')}" for group in groups]


# 2.2 生成只读权限，拒绝越界读取与写入
def _readonly_permissions(  #把「能读哪些路径」翻译成 deepagents 认的 FilesystemPermission 列表。
    scope_paths: list[str], permission_type: Any, *, deny_when_empty: bool = False
) -> list[Any] | None:
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


# 2.3 展开技能目录及其子路径
def _read_scope_paths(sources: list[str]) -> list[str]:
    return [path for source in sources for path in (source, f"{source}/**")]


# 2.4 读取 Agent 规格并校验工具、技能和中断配置
def _load_subagents(
    config: dict[str, Any], tools: list[Any], permission_type: Any
) -> list[dict[str, Any]]:
    import yaml

    harness = config.get("harness", {})
    spec_path = project_root() / str(harness.get("subagents_file", "config/subagents.yaml"))
    if not spec_path.is_file():
        raise RuntimeError(f"找不到 Agent 配置文件：{spec_path}")
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("agents"), list):
        raise TypeError(f"{spec_path} 必须包含 agents 列表")
    by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    subagents: list[dict[str, Any]] = []
    for spec in raw["agents"]:
        if not isinstance(spec, dict):
            raise TypeError("Agent 配置项必须是对象")
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("Agent 配置缺少 name")
        if any(item["name"] == name for item in subagents):
            raise ValueError(f"重复 Agent 名称：{name}")
        description = str(spec.get("description") or "").strip()
        system_prompt = str(spec.get("system_prompt") or "").strip()
        task_prompt = str(spec.get("task_prompt") or "").strip()
        if not description or not system_prompt or task_prompt.count("{question}") != 1:
            raise ValueError(f"Agent {name} 必须显式配置 description、system_prompt 和含一个 {{question}} 的 task_prompt")
        model_request_budget = spec.get("model_request_budget")
        closeout_request_budget = spec.get("closeout_request_budget")
        if type(model_request_budget) is not int or model_request_budget < 1:
            raise ValueError(f"Agent {name} 的 model_request_budget 必须是正整数")
        if type(closeout_request_budget) is not int or closeout_request_budget < 1:
            raise ValueError(f"Agent {name} 的 closeout_request_budget 必须是正整数")
        skills = spec.get("skills")
        if not isinstance(skills, list) or any(not isinstance(group, str) or not group.strip() for group in skills):
            raise ValueError(f"Agent {name} 的 skills 必须是字符串列表")
        reminder = spec.get("skill_reminder")
        if not isinstance(reminder, dict):
            raise TypeError(f"Agent {name} 的 skill_reminder 必须是对象")
        reminder_count = reminder.get("max_per_run")
        reminder_text = str(reminder.get("message") or "").strip()
        if type(reminder_count) is not int or reminder_count < 0 or not reminder_text:
            raise ValueError(f"Agent {name} 的 skill_reminder 需要非负 max_per_run 和 message")
        wanted_tools = spec.get("tools")
        if not isinstance(wanted_tools, list):
            raise TypeError(f"Agent {name} 的 tools 必须是列表")
        wanted = [str(item) for item in wanted_tools]
        missing = [item for item in wanted if item not in by_name]
        if missing:
            raise ValueError(
                f"从智能体 {name} 引用了未在 tools.yaml 声明的工具：{'、'.join(missing)}"
            )
        built: dict[str, Any] = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "task_prompt": task_prompt,
            "model_request_budget": model_request_budget,
            "closeout_request_budget": closeout_request_budget,
            "skill_reminder": {
                "max_per_run": reminder_count,
                "message": reminder_text,
            },
            "tools": [by_name[item] for item in wanted],
            "skills": _skill_sources(harness, skills),
        }
        scope = _read_scope_paths(built["skills"])  # 读取面按组收口，免得模型逛到别人的技能规范里
        permissions = _readonly_permissions(scope, permission_type, deny_when_empty=not scope)
        if permissions is not None:
            built["permissions"] = permissions
        if spec.get("response_format"):
            raise ValueError(f"Agent {name} 不应配置 response_format；请通过状态更新工具提交结果。")
        interrupt_on = spec.get("interrupt_on")
        if isinstance(interrupt_on, dict) and interrupt_on:
            built["interrupt_on"] = interrupt_on
        evidence_wrapup = spec.get("evidence_wrapup")
        if evidence_wrapup is not None:
            if (
                not isinstance(evidence_wrapup, dict)
                or str(evidence_wrapup.get("tool") or "") not in wanted
                or type(evidence_wrapup.get("max_nudges")) is not int
                or evidence_wrapup["max_nudges"] < 0
            ):
                raise ValueError(f"Agent {name} 的 evidence_wrapup 配置无效")
            built["evidence_wrapup"] = {
                "tool": str(evidence_wrapup["tool"]),
                "max_nudges": evidence_wrapup["max_nudges"],
            }
        subagents.append(built)
    if not subagents:
        raise ValueError(f"{spec_path} 里没有配置任何 Agent")
    if {item["name"] for item in subagents} != {"planner", "executor", "reflector"}:
        raise ValueError(f"{spec_path} 必须且只能配置 planner、executor、reflector 三个 Agent")
    return subagents


# ------------------------------------------------------------------ #
# 三、Harness 装配与执行
# ------------------------------------------------------------------ #
# 3.1 三个独立 Agent 的共享状态编排器
class SeaVideoHarness:
    # 3.1.1 绑定模型、领域工具和共享状态存储
    def __init__(
        self,
        config: dict[str, Any],
        service: Any,
        model: Any = None,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.service = service
        self.event_handler = event_handler
        self.model = (
            model
            if model is not None and callable(getattr(model, "bind_tools", None))
            else build_model(config)
        )
        self.tools = build_tools(config, service)
        harness = config.get("harness", {})
        self.state_store = SharedStateStore(
            project_root()
            / str(harness.get("shared_state_file", "data/memory/shared_task_state.sqlite"))
        )
        self.shared_state: SharedTaskState | None = None
        self.subagents: list[dict[str, Any]] = []
        self.agent_specs: dict[str, dict[str, Any]] = {}
        self.agents: dict[str, Any] = {}
        self._connection: sqlite3.Connection | None = None
        self._thread_id: str | None = None
        self._cancel: Any = None
        self._stale_checkpoint_roles: set[str] = set()
        self._prepare_specs()

    # 3.1.2 注入状态工具，校验角色与字段维护权限
    def _prepare_specs(self) -> None:  # 状态工具在这里统一注入，不写进 yaml
        from deepagents import FilesystemPermission

        self.subagents = _load_subagents(
            self.config,
            self.tools + build_state_tools(SharedTaskState("")),
            FilesystemPermission,
        )
        self.agent_specs = {str(item["name"]): item for item in self.subagents}
        missing = {"planner", "executor", "reflector"} - set(self.agent_specs)
        if missing:
            raise ValueError(f"三阶段协同缺少 Agent：{', '.join(sorted(missing))}")
        for role, spec in self.agent_specs.items():
            forbidden = {str(tool.name) for tool in spec["tools"]} & (
                STATE_TOOLS
                - {
                    "read_shared_state",
                    {
                        "planner": "update_planner_state",
                        "executor": "update_execution_state",
                        "reflector": "update_reflection_state",
                    }.get(role),
                }
            )
            if forbidden:
                raise ValueError(f"{role} 配置了越权状态工具：{', '.join(sorted(forbidden))}")

    # 3.1.3 打开 SQLite 检查点连接
    def _open_checkpoint(self) -> Any:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint = project_root() / str(
            self.config.get("harness", {}).get("checkpointer", "data/memory/checkpoints.sqlite")
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if self._connection is None:
            self._connection = sqlite3.connect(str(checkpoint), check_same_thread=False)
        return SqliteSaver(self._connection)

    # 3.1.4 为单个角色装配独立图、技能、权限和中间件
    def _build_agent_graph(self, name: str, state: SharedTaskState) -> Any:
        try:
            from deepagents import (
                GeneralPurposeSubagentProfile,
                HarnessProfile,
                create_deep_agent,
                register_harness_profile,
            )
            from deepagents.backends import FilesystemBackend
        except ModuleNotFoundError as error:
            if error.name == "deepagents":
                raise RuntimeError("缺少 deepagents 依赖，请安装项目生产依赖") from error
            raise

        spec = self.agent_specs[name]  # 每个角色一个独立 graph，工具、技能、权限各不相同
        tools_by_name = {
            str(getattr(tool, "name", "")): tool for tool in self.tools + build_state_tools(state)
        }
        selected = [tools_by_name[str(getattr(item, "name", item))] for item in spec["tools"]]
        harness = self.config.get("harness", {})
        sources = list(spec.get("skills") or [])
        profile = HarnessProfile(   #工具治理配置，当这个agent当子智能体时，哪些工具不给，哪些能力关上
            excluded_tools=frozenset(
                str(item) for item in (harness.get("disabled_deepagent_tools") or [])
            )
            | {"write_todos", "task"},
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),  #不允许子智能体的派生
            #deepagents 就不会在 system prompt 里塞"你可以派子任务"的说明，也不会生成 task 工具。
        )
        model_name = str(self.config.get("llm", {}).get("model", ""))
        if model_name:
            register_harness_profile(f"openai:{model_name}", profile)  #注册上面的配置
        return create_deep_agent(
            model=self.model,
            tools=selected,
            system_prompt=str(spec["system_prompt"]),
            skills=sources or None,
            backend=FilesystemBackend(root_dir=project_root()),  #agent文件系统后端，模型调用，让read_file 时，真正去磁盘读文件的就是它。
            permissions=spec["permissions"],  #文件权限规则，（读写）
            interrupt_on=spec.get("interrupt_on") or None,  #调用哪些工具前要暂停？
            middleware=build_middleware(
                self.config,
                self.model,
                agent_spec=spec,
                shared_state=state,
                cancelled=lambda: self._cancel is not None and self._cancel.is_set(),
                model_timeout=self._model_timeout_seconds,
                persist=lambda: (
                    self.state_store.save(self._thread_id, state) if self._thread_id else None
                ),
            ),
            checkpointer=self._open_checkpoint(),
            name=f"sea_video_{name}",
        )

    # 3.1.5 将三个角色连接到同一共享状态
    def _build_agents(self) -> None:
        if self.shared_state is None:
            raise RuntimeError("initialize shared state before building agents")
        state = self.shared_state
        self.agents = {name: self._build_agent_graph(name, state) for name in self.agent_specs}

    # 3.1.6 在阻塞调用期间广播运行心跳
    def _start_heartbeat(self, trace: _Trace, question: str) -> Any:
        interval = float(self.config.get("harness", {}).get("heartbeat_seconds", 15) or 0)
        if interval <= 0:
            return None
        stop = threading.Event()

        def beat() -> None:
            waited = 0.0
            while not stop.wait(interval):
                waited += interval
                trace.event(
                    {
                        "type": "status",
                        "title": "仍在执行",
                        "message": f"已等待 {int(waited)} 秒（模型或工具仍在工作）",
                    }
                )

        thread = threading.Thread(target=beat, name="harness-heartbeat", daemon=True)
        thread.start()
        return stop

    # 3.1.7 关闭检查点连接
    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def discard_run(self, thread_id: str) -> None:
        """Remove temporary shared state, external tool results and role checkpoints."""
        self.state_store.delete(thread_id)
        saver = self._open_checkpoint()
        try:
            for role in ("planner", "executor", "reflector"):
                saver.delete_thread(f"{thread_id}:{role}")
        finally:
            self.close()

    # 3.1.8 进入运行时上下文
    def __enter__(self) -> Self:
        return self

    # 3.1.9 离开上下文时释放连接
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    # 3.1.10 从历史检查点补回技能回执
    def _seed_skills_from_checkpoint(self, trace: _Trace, thread_id: str) -> None:
        try:
            snapshot = self.agents["executor"].get_state(
                {"configurable": {"thread_id": f"{thread_id}:executor"}}
            )
        except Exception:  # 新会话没有检查点，或 agent 不支持读快照
            logger.debug("跳过快照技能回执：thread_id=%s", thread_id, exc_info=True)
            return
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            trace.consume_skills(values)
        trace.consume_plan()

    # 3.1.11 生成角色开工消息，要求先读取共享状态
    def _agent_prompt(self, role: str, question: str) -> str:
        return str(self.agent_specs[role]["task_prompt"]).replace("{question}", question)

    # 3.1.11.1 统一选择新输入、检查点续接或人工确认恢复载荷
    def _agent_input_for_attempt(
        self,
        role: str,
        question: str,
        agent_thread: str,
        resume_payload: Any = None,
    ) -> Any:
        def new_input(history: list[Any] | None = None) -> dict[str, Any]:
            return {
                "messages": [
                    *(history or []),
                    {"role": "user", "content": self._agent_prompt(role, question)},
                ]
            }

        try:
            snapshot = self.agents[role].get_state(
                {"configurable": {"thread_id": agent_thread}}
            )
        except Exception:
            logger.debug(
                "Agent checkpoint lookup failed: role=%s thread_id=%s",
                role,
                agent_thread,
                exc_info=True,
            )
            return resume_payload if resume_payload is not None else new_input()

        tasks = getattr(snapshot, "tasks", ()) or ()
        interrupted = any(getattr(task, "interrupts", ()) for task in tasks)
        recoverable = bool(
            snapshot is not None and (getattr(snapshot, "next", None) or interrupted)
        )
        if interrupted and resume_payload is not None:
            return resume_payload
        if recoverable and role in self._stale_checkpoint_roles:
            history = self._history_before_current_prompt(snapshot)
            self._open_checkpoint().delete_thread(agent_thread)
            logger.info(
                "Discarded stale agent checkpoint: role=%s thread_id=%s",
                role,
                agent_thread,
            )
            self._stale_checkpoint_roles.discard(role)
            return new_input(history)
        self._stale_checkpoint_roles.discard(role)
        if recoverable:
            return None
        return new_input()

    @staticmethod
    def _checkpoint_messages(snapshot: Any) -> list[Any]:
        values = getattr(snapshot, "values", None)
        messages = values.get("messages") if isinstance(values, dict) else None
        return list(messages) if isinstance(messages, list) else []

    @classmethod
    def _history_before_current_prompt(cls, snapshot: Any) -> list[Any]:
        """丢弃最后一条失败轮 prompt，同时保留更早的会话消息。"""
        messages = cls._checkpoint_messages(snapshot)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict):
                kind = str(message.get("type") or message.get("role") or "")
                content = str(message.get("content") or "")
            else:
                kind = str(getattr(message, "type", "") or "")
                content = str(getattr(message, "content", "") or "")
            if kind in {"human", "user"} and content.startswith("用户问题："):
                return messages[:index]
        return messages

    # 3.1.12 重建单个角色，保留共享状态和检查点
    def _reconnect_agent(self, role: str) -> None:
        if not self.shared_state:
            return
        self.agents[role] = self._build_agent_graph(role, self.shared_state)

    # 3.1.12.1 模型请求只使用模型自身超时
    def _model_timeout_seconds(self) -> float:
        return max(0.001, float(self.config.get("llm", {}).get("timeout_seconds", 60)))

    # 3.1.13 驱动单个角色，处理重连、取消与人工确认
    def _run_agent_steps(
        self,
        role: str,
        question: str,
        thread_id: str,
        trace: _Trace,
        cancel: Any = None,
        resume_payload: Any = None,
    ) -> Iterator[None]:
        harness = self.config.get("harness", {})
        retries = max(0, int(harness.get("reconnect_attempts", 2)))  # 最多重建几次
        backoff = max(0.0, float(harness.get("reconnect_backoff_seconds", 1.0)))  # 首次退避秒数
        agent_thread = f"{thread_id}:{role}"  # 每个角色一条独立历史，互相看不到对方的中间过程
        payload = self._agent_input_for_attempt(
            role, question, agent_thread, resume_payload
        )
        trace.shared_state = self.shared_state  # 绑定状态，工具结果才能自动入账
        self._cancel = cancel
        self.shared_state.runtime_update({"pending_agent": role})  # 记下现在轮到谁，断了能续
        self.state_store.save(thread_id, self.shared_state)
        attempt = 0
        while True:
            if cancel is not None and cancel.is_set():
                return {"state": "cancelled"}
            try:
                if attempt:   #重连计数
                    self._reconnect_agent(role)  # 只重建这个角色，检查点不动
                    payload = self._agent_input_for_attempt(
                        role, question, agent_thread, resume_payload
                    )
                agent = self.agents[role]
                for frame in agent.stream(
                    payload,
                    config={
                        "configurable": {"thread_id": agent_thread},
                        "max_concurrency": 1,
                    },
                    stream_mode=harness.get("stream_mode", "updates"),
                    subgraphs=False,  # 每个角色只有一层图，不需要展开子图
                ):
                    if cancel is not None and cancel.is_set():
                        self.shared_state.runtime_update(
                            {"task_status": "cancelled", "pending_agent": role}
                        )
                        self.state_store.save(thread_id, self.shared_state)
                        return {"state": "cancelled"}
                    interrupts = _interrupt_payload(frame)  # 命中写库前的人工确认
                    if interrupts:  #打断情况
                        self.shared_state.runtime_update(
                            {"task_status": "awaiting_confirmation", "pending_agent": role}
                        )
                        self.state_store.save(thread_id, self.shared_state)
                        return {
                            "state": "awaiting_confirmation",
                            "confirmation": _confirmation_from(interrupts),
                        }
                    trace.consume(frame, agent_name=role)  #给这帧产生的所有事件和记录打上角色标记。
                    self.state_store.save(
                        thread_id, self.shared_state
                    )  # 每帧落盘，换个 runtime 也能接着跑
                    yield None
                trace.consume_plan()  # 流正常结束：补发最终的计划快照
                return {"state": "completed"}
            except Exception as error:  #这一段执行中抛出的任何普通异常，都会被这里接住
                from langgraph.errors import (  # LangGraph 的一个保护异常。图有一个递归上限（默认 25 步左右），
                    GraphRecursionError,
                )
                #一次执行超过这个步数就抛它——防止图在 model ⇄ tools 循环里无限转下去。
                if isinstance(error, AgentCancelled):
                    return {"state": "cancelled"}
                if isinstance(error, AgentCloseoutComplete):
                    return {"state": "completed"}
                if isinstance(error, MissingStateWrite):
                    return {"state": "stalled", "reason": str(error)}
                if isinstance(error, GraphRecursionError):
                    return {"state": "stalled", "reason": f"{role} 超过图执行步数上限，未完成本阶段。"}
                if isinstance(error, (StateConflictError, ValueError, TypeError)):  # 硬错误，重试也没用
                    return {"state": "error", "error": error}
                if attempt >= retries:
                    logger.exception("Agent failed: role=%s thread_id=%s", role, thread_id)
                    return {"state": "error", "error": error}
                attempt += 1
                trace.event(
                    {
                        "type": "status",
                        "title": "Agent 正在重连",
                        "message": f"{role} 第 {attempt} 次重连",
                        "agent": role,
                    }
                )
                yield None
                if backoff:
                    delay = min(30.0, backoff * (2 ** (attempt - 1)))  # 指数退避，封顶 30 秒
                    if cancel is not None:
                        if cancel.wait(delay):  # 退避期间也能被取消
                            return {"state": "cancelled"}
                    else:
                        time.sleep(delay)

    # 3.1.14 按规划、执行、反思闭环调度并校验状态提交
    def _drive_multi_steps(
        self,
        question: str,
        thread_id: str,
        trace: _Trace,
        cancel: Any = None,
        resume_payload: Any = None,
    ) -> Iterator[None]:
        harness = self.config.get("harness", {})
        max_iterations = max(1, int(harness.get("max_iterations", 4)))
        if self.shared_state is None:
            return {"state": "error", "error": RuntimeError("shared state is not initialized")}
        current = self.shared_state.snapshot()
        self._cancel = cancel
        start_role = current["pending_agent"] or "planner"
        continuing = bool(current["pending_agent"] and current["task_status"] == "running")
        self.shared_state.runtime_update({"task_status": "running"})
        iteration = current["iteration"] or 1
        all_roles = ["planner", "executor", "reflector"]
        while iteration <= max_iterations:
            self.shared_state.runtime_update({"iteration": iteration, "pending_agent": start_role})
            roles = all_roles[all_roles.index(start_role) :]
            for role in roles:
                if resume_payload is None and not continuing:
                    self.shared_state.start_agent(role)
                continuing = False  # 恢复待执行角色时保留原计数；之后才开始新阶段。
                trace.event(
                    {
                        "type": "status",
                        "title": f"{role} 开始工作",
                        "message": "读取共享状态并维护负责字段",
                        "agent": role,
                    }
                )
                payload = (
                    resume_payload if role == start_role and resume_payload is not None else None
                )
                outcome = yield from self._run_agent_steps(
                    role, question, thread_id, trace, cancel, payload
                )
                resume_payload = None  # 确认只消费一次，下一轮重新规划。
                if outcome["state"] != "completed":
                    return outcome
                if role == "planner":
                    clarification = str(
                        self.shared_state.snapshot().get("clarification") or ""
                    ).strip()
                    if clarification:
                        self.shared_state.runtime_update(
                            {"task_status": "awaiting_input", "pending_agent": None}
                        )
                        self.state_store.save(thread_id, self.shared_state)
                        return {"state": "awaiting_input", "clarification": clarification}
                next_index = all_roles.index(role) + 1  #计算下一个跑哪个角色
                self.shared_state.runtime_update(
                    {"pending_agent": all_roles[next_index] if next_index < 3 else None}  #下一个没有了就取none
                )
                self.state_store.save(thread_id, self.shared_state)  #共享状态写入Sql，task_status。并不是简单的写文件，而是带冲突检测
            current = self.shared_state.snapshot()
            decision = str(current.get("decision") or "replan").lower()  # 收尾还是再来一轮，由 reflector 决定
            if decision == "terminate":  #收尾检查，检查后收尾
                if not str(current["assessment"].get("final_answer") or "").strip():
                    return {"state": "stalled", "reason": "Reflector 尚未提交最终回答"}
                self.shared_state.runtime_update(
                    {"task_status": "completed", "pending_agent": None}
                )
                return {"state": "completed"}
            if decision == "failed":  #业务推进不了，不检查，直接收尾
                self.shared_state.runtime_update({"task_status": "failed", "pending_agent": None})
                return {"state": "stalled", "reason": "反思阶段判定任务无法继续"}
            self.shared_state.runtime_update({"task_status": "running", "pending_agent": "planner"})
            start_role = "planner"
            iteration += 1
        return {
            "state": "stalled",
            "reason": f"已达到最大迭代次数 {max_iterations}，反思结果仍要求继续或重规划。",
        }

    # 3.1.15 消费驱动生成器并取得终态
    @staticmethod
    def _consume_driver(driver: Iterator[None]) -> dict[str, Any]:
        while True:
            try:
                next(driver)
            except StopIteration as stopped:
                return (
                    stopped.value
                    if isinstance(stopped.value, dict)
                    else {"state": "error", "error": RuntimeError("运行结果为空")}
                )

    # 3.1.16 普通问答入口
    def run(
        self, question: str, thread_id: str | None = None, cancel: Any = None, **_: Any
    ) -> dict[str, Any]:
        return self._execute(question, thread_id or uuid.uuid4().hex, cancel)

    # 3.1.17 人工确认恢复入口
    def resume(
        self, decision: str, thread_id: str, feedback: str = "", cancel: Any = None
    ) -> dict[str, Any]:
        normalized = str(decision or "").strip().lower()
        if normalized not in {"approve", "reject"}:
            raise ValueError(f"decision 只能是 approve 或 reject，收到：{decision!r}")
        if not thread_id:
            raise ValueError("恢复确认必须带上原 thread_id")
        return self._execute("", thread_id, cancel, _resume_command(normalized, feedback))

    # 3.1.18 问答与确认恢复共用执行及资源释放流程
    def _execute(
        self, question: str, thread_id: str, cancel: Any, resume_payload: Any = None
    ) -> dict[str, Any]:
        """普通问答与确认恢复共用收尾，异常和取消都关闭检查点连接。"""
        harness = self.config.get("harness", {})
        heartbeat = None
        self._cancel = cancel
        try:
            self._initialize_state(thread_id, question, resume=resume_payload is not None)
            question = self.shared_state.snapshot()["question"]
            trace = _Trace(
                self.event_handler,
                int(harness.get("event_payload_max_chars", 4000)),
                _tool_labels(self.config),
                self.shared_state,
            )
            trace.event(
                {"type": "status", "title": "Harness 已启动", "message": "三 Agent 已连接共享状态"}
            )
            self._seed_skills_from_checkpoint(trace, thread_id)
            heartbeat = self._start_heartbeat(trace, question)
            try:
                outcome = self._consume_driver(
                    self._drive_multi_steps(question, thread_id, trace, cancel, resume_payload)
                )
            except Exception as error:
                logger.exception("Harness failed: thread_id=%s", thread_id)
                outcome = {"state": "error", "error": error}
            return self._finish(trace, thread_id, outcome)
        finally:
            if heartbeat is not None:
                heartbeat.set()
            self.close()

    # 3.1.19 恢复会话状态或开始新问答，再装配 Agent
    def _initialize_state(self, thread_id: str, question: str = "", resume: bool = False) -> None:
        self._thread_id = thread_id
        self.shared_state = self.state_store.load(thread_id, question)
        current = self.shared_state.snapshot()
        starting_new_turn = False
        if resume:
            if current["task_status"] != "awaiting_confirmation" or not current["pending_agent"]:
                raise ValueError("此会话没有待确认的中断，不能重复批准或拒绝")
        elif current["task_status"] == "awaiting_confirmation":
            raise ValueError("此会话正在等待人工确认，请先批准或拒绝")
        elif current["question"] != question or current["task_status"] in {
            "completed",
            "failed",
            "error",
            "cancelled",
            "stalled",
            "awaiting_input",
        }:
            self.shared_state.start_turn(question)
            starting_new_turn = True
        self._stale_checkpoint_roles = (
            set(self.agent_specs) if starting_new_turn else set()
        )
        self.state_store.save(thread_id, self.shared_state)
        try:
            self._build_agents()
        except Exception:
            self.shared_state.runtime_update({"task_status": "error"})
            self.state_store.save(thread_id, self.shared_state)
            raise

    # 3.1.20 保存终态并生成结果与终局事件
    def _finish(self, trace: _Trace, thread_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        harness = self.config.get("harness", {})
        state = outcome.get("state")
        if self.shared_state is not None:
            current = self.shared_state.snapshot()
            if current["task_status"] != "failed":
                patch = {"task_status": state}
                if state in {"completed", "stalled", "error", "cancelled", "awaiting_input"}:
                    patch["pending_agent"] = None
                self.shared_state.runtime_update(patch)
            try:
                self.state_store.save(thread_id, self.shared_state)
            except StateConflictError as error:
                state = "error"
                outcome = {"state": "error", "error": error}
                self.shared_state.runtime_update({"task_status": "error"})
        if state == "awaiting_confirmation":
            result = trace.result(thread_id, self.config, state="awaiting_confirmation")
            result["confirmation"] = outcome.get("confirmation") or {}
            trace.event(
                {
                    "type": "confirm",
                    "title": "等待人工确认",
                    "message": "有一项写入操作需要你确认",
                    "confirmation": result["confirmation"],
                    "result": result,
                }
            )
            return result
        if state == "awaiting_input":
            result = trace.result(thread_id, self.config, state="awaiting_input")
            answer_field = str(harness.get("output", {}).get("answer_field", "answer"))
            clarification = str(outcome.get("clarification") or "").strip()
            result[answer_field] = clarification
            result["clarification"] = clarification
            trace.event(
                {
                    "type": "complete",
                    "title": "需要补充信息",
                    "message": clarification,
                    "result": result,
                }
            )
            return result
        if state == "cancelled":
            result = trace.result(thread_id, self.config, state="cancelled")
            trace.event(
                {
                    "type": "complete",
                    "title": "Harness 已停止",
                    "message": "本轮已按请求停止",
                    "result": result,
                }
            )
            return result
        if state == "stalled":
            result = trace.result(thread_id, self.config, state="stalled")
            answer_field = str(harness.get("output", {}).get("answer_field", "answer"))
            reason = str(outcome.get("reason") or "")
            if not str(result.get(answer_field) or "").strip():
                result[answer_field] = reason or "本轮未能继续推进。"
            result["error"] = reason or "未能生成有效结果"
            trace.event(
                {
                    "type": "error",
                    "title": "本轮未能完成",
                    "message": reason or "未能生成有效结果",
                    "errorType": "AgentStalled",
                    "result": result,
                }
            )
            return result
        if state == "error":
            error = outcome["error"]
            message = _error_message(error, trace.event_limit)
            result = trace.result(thread_id, self.config, state="error")
            result["error"] = message
            trace.event(
                {
                    "type": "error",
                    "title": "Harness 执行失败",
                    "message": message,
                    "errorType": type(error).__name__,
                    "result": result,
                }
            )
            return result
        result = trace.result(thread_id, self.config)
        trace.event(
            {
                "type": "complete",
                "title": "Harness 完成",
                "message": "回答与证据已生成",
                "result": result,
            }
        )
        return result

    # 3.1.21 将协同驱动过程投影为增量事件流
    def _stream_multi(self, question: str, thread_id: str) -> Iterator[dict[str, Any]]:
        harness = self.config.get("harness", {})
        pending: deque[dict[str, Any]] = deque()

        def emit(event: dict[str, Any]) -> None:
            pending.append(event)
            if self.event_handler:
                self.event_handler(event)

        trace = _Trace(
            emit,
            int(harness.get("event_payload_max_chars", 4000)),
            _tool_labels(self.config),
        )
        heartbeat, driver = None, None
        initialized = False
        self._cancel = None
        try:
            self._initialize_state(thread_id, question)
            initialized = True
            trace.shared_state = self.shared_state
            trace.event(
                {"type": "status", "title": "Harness 已启动", "message": "三 Agent 已连接共享状态"}
            )
            self._seed_skills_from_checkpoint(trace, thread_id)
            heartbeat = self._start_heartbeat(trace, question)
            driver = self._drive_multi_steps(question, thread_id, trace)
            while True:
                try:
                    next(driver)
                except StopIteration as stopped:
                    outcome = (
                        stopped.value
                        if isinstance(stopped.value, dict)
                        else {"state": "error", "error": RuntimeError("运行结果为空")}
                    )
                    break
                while pending:
                    yield pending.popleft()
            self._finish(trace, thread_id, outcome)
            while pending:
                yield pending.popleft()
        except Exception as error:
            logger.exception("Harness multi-agent stream failed: thread_id=%s", thread_id)
            if initialized:
                self._finish(trace, thread_id, {"state": "error", "error": error})
                while pending:
                    yield pending.popleft()
            else:
                trace.shared_state = self.shared_state
                result = trace.result(thread_id, self.config, state="error")
                result["error"] = _error_message(error, trace.event_limit)
                yield {
                    "type": "error",
                    "title": "Harness 执行失败",
                    "message": result["error"],
                    "result": result,
                }
        finally:
            if driver is not None:
                driver.close()
            if heartbeat is not None:
                heartbeat.set()
            self.close()

    # 3.1.22 流式问答入口
    def stream(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        yield from self._stream_multi(question, thread_id or uuid.uuid4().hex)


# ------------------------------------------------------------------ #
# 四、模块级便捷入口
# ------------------------------------------------------------------ #
# 4.1 创建运行时并执行一次问答
def run_harness(
    config: dict[str, Any],
    tools: Any,
    llm: Any = None,
    event_handler: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return SeaVideoHarness(config, tools, model=llm, event_handler=event_handler).run(
        kwargs.get("question", ""), kwargs.get("thread_id")
    )

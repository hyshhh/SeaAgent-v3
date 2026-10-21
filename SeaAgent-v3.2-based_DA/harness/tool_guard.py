"""执行前从共享状态占用预算，重复领域查询复用结果，写入请求只执行一次。"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .convergence import ConvergenceRequired
from .shared_state import STATE_TOOLS, SharedTaskState

REPEAT_HINT = "Skipped: this exact call was already made. Reuse its recorded result; do not repeat the same tool and arguments."


class _ToolTimeout(TimeoutError):
    pass


# ------------------------------------------------------------------ #
# 一、调用签名
# ------------------------------------------------------------------ #
# 1.1 按工具名与参数生成稳定签名
def _signature(name: str, arguments: Any) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


# ------------------------------------------------------------------ #
# 二、工具调用守卫
# ------------------------------------------------------------------ #
# 2.1 统一角色权限、调用预算、查询复用与写入防重
class RepeatToolCallMiddleware(AgentMiddleware):
    # 2.1.1 绑定共享状态、角色、限额与持久化回调
    def __init__(
        self,
        state: SharedTaskState | None = None,
        scope: str = "",
        limits: dict[str, int] | None = None,
        persist: Callable[[], None] | None = None,
        allowed_tools: set[str] | None = None,
        cancelled: Callable[[], bool] | None = None,
        convergence_reason: Callable[[], str] | None = None,
        model_timeout: Callable[[], float] | None = None,
        tool_timeout: Callable[[], float] | None = None,
    ):
        self.state = state if state is not None else SharedTaskState("")
        self.scope = scope
        self.limits = limits or {}
        self.persist = persist
        self.allowed_tools = allowed_tools
        self.cancelled = cancelled
        self.convergence_reason = convergence_reason
        self.model_timeout = model_timeout
        self.tool_timeout = tool_timeout

    # 2.1.2 请求前检查预算并限制可见工具
    def _model_request(self, request: Any) -> Any:
        if self.cancelled and self.cancelled():
            raise ConvergenceRequired("cancelled")
        reason = (
            self.convergence_reason()
            if self.convergence_reason
            else self.state.limit_reason(self.scope, self.limits)
        )
        if reason:
            raise ConvergenceRequired(reason)
        overrides = {}
        if self.allowed_tools is not None:
            overrides["tools"] = [
                tool for tool in request.tools if getattr(tool, "name", None) in self.allowed_tools
            ]
        if self.model_timeout is not None:
            # 每次请求和重试使用模型配置的超时。
            overrides["model_settings"] = {
                **request.model_settings,
                "timeout": self.model_timeout(),
            }
        return request.override(**overrides) if overrides else request

    # 2.1.3 同步模型请求的工具权限过滤
    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._model_request(request))

    # 2.1.4 异步模型请求的工具权限过滤
    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._model_request(request))

    # 2.1.5 保存共享调用账本
    def _save(self) -> None:
        if self.persist:
            self.persist()

    # 2.1.6 执行前校验权限与预算，复用缓存或阻止重复写入
    def _prepare(self, request: Any) -> tuple[str | None, ToolMessage | None]:
        call = request.tool_call
        name = str(call.get("name") or "")
        if (self.cancelled and self.cancelled()) or (
            self.allowed_tools is not None and name not in self.allowed_tools
        ):
            return None, ToolMessage(
                content="Call cancelled or outside this agent's tool permissions.",
                name=name,
                tool_call_id=str(call.get("id") or ""),
                status="error",
            )
        key = None if name in STATE_TOOLS else _signature(name, call.get("args") or {})
        if name == "read_file":
            key = f"{self.scope}:{self.state.snapshot()['iteration']}:{key}"
        reason = self.convergence_reason() if self.convergence_reason else ""
        # 状态与技能读取登记调用签名，但不消耗整轮或单 Agent 工具预算。
        budgeted = name not in STATE_TOOLS and name != "read_file"
        if reason:
            record = {"blocked": reason}
        else:
            try:
                record = self.state.claim_call(
                    self.scope,
                    key,
                    self.limits,
                    write=name == "add_registry_vessel",
                    count=budgeted,
                )
            except TypeError as error:
                if "count" not in str(error):
                    raise
                limits = self.limits if budgeted else {**self.limits, "run": 0, "thread": 0, "agent": 0}
                record = self.state.claim_call(
                    self.scope, key, limits, write=name == "add_registry_vessel"
                )
        self._save()
        if record.get("execute"):
            return key, None
        artifact = {"replayed": True}
        if record.get("blocked"):
            content, status = (
                "Execution budget reached. Stop tools and converge using existing evidence.",
                "success",
            )
            artifact["convergence_required"] = True
        elif record.get("status") == "error" and not record.get("write"):
            content = json.dumps(
                {
                    "error": record.get("result"),
                    "message": "读取重试次数已用完；请使用已有结果或调整查询。",
                },
                ensure_ascii=False,
                default=str,
            )
            status = "error"
        elif record.get("status") == "completed" and name != "read_file":
            result = record.get("result")
            if record.get("truncated"):
                result = {
                    "cached_result": result,
                    "truncated": True,
                    "message": "缓存仅保留摘要，缺失条目不能作为不存在的证据；请缩小参数范围查询。",
                }
            content = json.dumps(result, ensure_ascii=False, default=str)
            status = "success"
        else:
            content = REPEAT_HINT
            status = (
                "error" if record.get("write") or record.get("status") == "pending" else "success"
            )
            if record.get("write"):
                content += (
                    " The write outcome is uncertain; verify it before taking further action."
                )
        return key, ToolMessage(
            content=content,
            name=name,
            tool_call_id=str(call.get("id") or ""),
            status=status,
            artifact=artifact,
        )

    def _timeout_seconds(self) -> float:
        if self.tool_timeout is None:
            return 0.0
        try:
            return max(0.001, float(self.tool_timeout()))
        except (TypeError, ValueError):
            return 0.0

    def _run_sync(self, handler: Callable[[Any], Any], request: Any) -> Any:
        timeout = self._timeout_seconds()
        if timeout <= 0:
            return handler(request)
        box: dict[str, Any] = {}
        done = threading.Event()

        def invoke() -> None:
            try:
                box["response"] = handler(request)
            except Exception as error:  # noqa: BLE001 - relay worker failures to caller
                box["error"] = error
            finally:
                done.set()

        threading.Thread(target=invoke, name="sea-tool-call", daemon=True).start()
        if not done.wait(timeout):
            raise _ToolTimeout(f"tool call exceeded {timeout:.2f}s; worker continues in background")
        if "error" in box:
            raise box["error"]
        return box.get("response")

    # 2.1.7 将返回结果登记到共享状态并持久化
    def _complete(self, request: Any, key: str | None, response: Any) -> None:
        call = request.tool_call
        name = str(call.get("name") or "")
        result = getattr(response, "content", response)
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                pass
        failed = getattr(response, "status", "") == "error" or (
            isinstance(result, dict) and result.get("ok") is False
        )
        self.state.finish_call(key, result, failed)
        if name not in STATE_TOOLS and name != "read_file":
            self.state.record_tool_result(
                self.scope,
                name,
                str(call.get("id") or ""),
                result,
                "error" if failed else "completed",
            )
        self._save()

    # 2.1.8 保留失败记录，避免结果不确定的写入重放
    def _failed(self, request: Any, key: str | None, error: BaseException) -> None:
        call = request.tool_call
        name = str(call.get("name") or "")
        result = {"ok": False, "error": str(error)}
        self.state.finish_call(key, str(error), failed=True)
        if name not in STATE_TOOLS and name != "read_file":
            self.state.record_tool_result(
                self.scope,
                name,
                str(call.get("id") or ""),
                result,
                "error",
            )
        self._save()

    # 2.1.9 同步工具执行与异常登记
    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        name = str(request.tool_call.get("name") or "")
        try:
            response = (
                self._run_sync(handler, request)
                if name not in STATE_TOOLS and name != "read_file"
                else handler(request)
            )
        except _ToolTimeout as error:
            self._failed(request, key, error)
            return ToolMessage(
                content=str(error),
                name=name,
                tool_call_id=str(request.tool_call.get("id") or ""),
                status="error",
                artifact={"tool_timeout": True},
            )
        except Exception as error:
            self._failed(request, key, error)
            raise
        self._complete(request, key, response)
        return response

    # 2.1.10 异步工具执行与异常登记
    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        name = str(request.tool_call.get("name") or "")
        try:
            if name not in STATE_TOOLS and name != "read_file" and self._timeout_seconds() > 0:
                response = await asyncio.wait_for(handler(request), self._timeout_seconds())
            else:
                response = await handler(request)
        except asyncio.TimeoutError as error:
            timeout = self._timeout_seconds()
            self._failed(request, key, error)
            return ToolMessage(
                content=f"tool call exceeded {timeout:.2f}s",
                name=name,
                tool_call_id=str(request.tool_call.get("id") or ""),
                status="error",
                artifact={"tool_timeout": True},
            )
        except asyncio.CancelledError as error:
            self._failed(request, key, error)
            raise
        except Exception as error:
            self._failed(request, key, error)
            raise
        self._complete(request, key, response)
        return response

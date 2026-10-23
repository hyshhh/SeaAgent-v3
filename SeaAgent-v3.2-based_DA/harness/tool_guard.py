"""Count tool calls, replay exact repeats, and enter closeout on the fourth call."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .shared_state import ROLE_UPDATE_TOOLS, STATE_TOOLS, SharedTaskState

REPEAT_LIMIT = 4
REPEAT_HINT = (
    "该工具与参数已调用过。请使用返回结果或调整参数；不要重复相同调用。"
)
_REPEAT_CLOSEOUT_MESSAGE = "相同工具和参数已调用 4 次；已进入状态收尾，本次调用未执行。"


class AgentCancelled(RuntimeError):
    """Stop an in-flight agent before it starts another model request."""


class _ToolTimeout(TimeoutError):
    pass


def _signature(name: str, arguments: Any) -> str:
    normalized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{normalized}"


def _message(
    request: Any,
    content: str,
    *,
    status: str = "success",
    artifact: Any = None,
) -> ToolMessage:
    call = request.tool_call
    return ToolMessage(
        content=content,
        name=str(call.get("name") or ""),
        tool_call_id=str(call.get("id") or ""),
        status=status,
        artifact=artifact,
    )


class RepeatToolCallMiddleware(AgentMiddleware):
    """Apply role permissions, exact-call replay, statistics, and per-call timeouts."""

    def __init__(
        self,
        state: SharedTaskState | None = None,
        scope: str = "",
        repeat_limit: int = REPEAT_LIMIT,
        persist: Callable[[], None] | None = None,
        allowed_tools: set[str] | None = None,
        cancelled: Callable[[], bool] | None = None,
        model_timeout: Callable[[], float] | None = None,
        tool_timeout: Callable[[], float] | None = None,
    ) -> None:
        self.state = state if state is not None else SharedTaskState("")
        self.scope = scope
        self.repeat_limit = max(2, int(repeat_limit))
        self.persist = persist
        self.allowed_tools = allowed_tools
        self.cancelled = cancelled
        self.model_timeout = model_timeout
        self.tool_timeout = tool_timeout

    def _model_request(self, request: Any) -> Any:
        if self.cancelled and self.cancelled():
            raise AgentCancelled("cancelled")
        overrides: dict[str, Any] = {}
        if self.allowed_tools is not None and request.tools is not None:
            overrides["tools"] = [
                tool for tool in request.tools if getattr(tool, "name", None) in self.allowed_tools
            ]
        if self.model_timeout is not None:
            overrides["model_settings"] = {
                **request.model_settings,
                "timeout": self.model_timeout(),
            }
        return request.override(**overrides) if overrides else request

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._model_request(request))

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        return await handler(self._model_request(request))

    def _save(self) -> None:
        if self.persist:
            self.persist()

    def _prepare(self, request: Any) -> tuple[str | None, ToolMessage | None]:
        call = request.tool_call
        name = str(call.get("name") or "")
        if (self.cancelled and self.cancelled()) or (
            self.allowed_tools is not None and name not in self.allowed_tools
        ):
            return None, _message(request, "Call cancelled or outside this agent's tool permissions.", status="error")

        snapshot = self.state.snapshot()
        iteration = int(snapshot.get("iteration", 0))
        key = f"{_signature(name, call.get('args') or {})}|agent={self.scope}|iteration={iteration}"
        closeout_tools = {"read_shared_state"}
        if self.scope in ROLE_UPDATE_TOOLS:
            closeout_tools.add(ROLE_UPDATE_TOOLS[self.scope])
        record = self.state.claim_tool_call(
            self.scope,
            key,
            name,
            write=name == "add_registry_vessel",
            repeat_limit=self.repeat_limit,
            closeout_tools=closeout_tools,
        )
        self._save()

        if record.get("blocked") == "closeout_required":
            return None, _message(
                request,
                "重复调用已触发收尾；请读取共享状态并提交本角色负责的状态。",
                artifact={"closeout_required": True},
            )
        if name == "read_shared_state" and record.get("status") == "invalidated":
            return key, None
        if record.get("blocked"):
            if record.get("blocked") == "repeat_limit":
                return None, _message(
                    request,
                    _REPEAT_CLOSEOUT_MESSAGE,
                    artifact={"closeout_required": True, "repeat_count": record["repeat_count"]},
                )
            if self.state.is_closeout_required(self.scope):
                return None, _message(
                    request,
                    "收尾模式只允许读取共享状态和提交本角色状态；本次工具调用未执行。",
                    artifact={"closeout_required": True},
                )
            return None, _message(
                request,
                _REPEAT_CLOSEOUT_MESSAGE,
                artifact={"closeout_required": True, "repeat_count": record["repeat_count"]},
            )
        if record.get("execute"):
            return key, None

        count = int(record.get("repeat_count", 1))
        artifact = {"replayed": True, "repeat_count": count}
        if record.get("status") == "completed":
            result = record.get("result")
            payload = {"cached_result": result, "message": REPEAT_HINT}
            if record.get("truncated"):
                payload["truncated"] = True
            content = json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            return None, _message(request, content, artifact=artifact)

        if record.get("status") == "error":
            content = json.dumps(
                {"cached_error": record.get("result"), "message": REPEAT_HINT},
                ensure_ascii=False,
                default=str,
            )
            return None, _message(request, content, status="error", artifact=artifact)
        return None, _message(request, REPEAT_HINT, artifact=artifact)

    def _timeout_seconds(self) -> float:
        if self.tool_timeout is None:
            return 120.0
        try:
            return max(0.001, float(self.tool_timeout()))
        except (TypeError, ValueError):
            return 120.0

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

    def _failed(self, request: Any, key: str | None, error: BaseException) -> None:
        call = request.tool_call
        name = str(call.get("name") or "")
        result = {"ok": False, "error": str(error)}
        self.state.finish_call(key, result, failed=True)
        if name not in STATE_TOOLS and name != "read_file":
            self.state.record_tool_result(
                self.scope, name, str(call.get("id") or ""), result, "error"
            )
        self._save()

    def _run_sync(self, request: Any, key: str | None, handler: Callable[[Any], Any]) -> Any:
        timeout = self._timeout_seconds()
        result: dict[str, Any] = {}
        completed = threading.Event()

        def invoke() -> None:
            try:
                result["response"] = handler(request)
            except Exception as error:  # noqa: BLE001 - Convert tool failures to model-visible results.
                result["error"] = error
            finally:
                completed.set()

        threading.Thread(target=invoke, name="sea-tool-call", daemon=True).start()
        if not completed.wait(timeout):
            raise _ToolTimeout(f"tool call exceeded {timeout:.0f}s; synchronous worker may still be running")
        if "error" in result:
            raise result["error"]
        return result.get("response")

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        try:
            response = self._run_sync(request, key, handler)
        except _ToolTimeout as error:
            self._failed(request, key, error)
            return _message(
                request,
                str(error),
                status="error",
                artifact={"tool_timeout": True},
            )
        except Exception as error:  # noqa: BLE001 - Tool failures are returned to the model.
            self._failed(request, key, error)
            return _message(
                request,
                json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
                status="error",
            )
        self._complete(request, key, response)
        return response

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        try:
            response = await asyncio.wait_for(handler(request), self._timeout_seconds())
        except asyncio.TimeoutError as error:
            self._failed(request, key, error)
            return _message(
                request,
                f"tool call exceeded {self._timeout_seconds():.0f}s",
                status="error",
                artifact={"tool_timeout": True},
            )
        except asyncio.CancelledError as error:
            self._failed(request, key, error)
            raise
        except Exception as error:  # noqa: BLE001 - Tool failures are returned to the model.
            self._failed(request, key, error)
            return _message(
                request,
                json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
                status="error",
            )
        self._complete(request, key, response)
        return response

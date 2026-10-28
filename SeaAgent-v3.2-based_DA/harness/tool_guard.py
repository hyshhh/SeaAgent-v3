"""Count tool calls, replay exact repeats, and enter closeout on the fourth call."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from .shared_state import STATE_TOOLS, SharedTaskState

REPEAT_LIMIT = 4
REPEAT_HINT = (
    "该工具与参数已调用过。请使用返回结果或调整参数；不要重复相同调用。"
)
_REPEAT_CLOSEOUT_MESSAGE = "相同工具和参数已调用 4 次；已进入状态收尾，本次调用未执行。"
_MODEL_RESULT_FIELDS = {
    "recognize_tracks": (
        "ok", "status", "matches", "coverageComplete", "nextStep", "recommendation",
    ),
    "match_registry_tracks": (
        "ok", "status", "matches", "coverageComplete", "registryFound", "nextStep",
    ),
    "verify_vessel_images": (
        "ok", "decision", "facts", "formatRecovered", "registryReferenceId", "keyframeId",
    ),
    "dedup_tracks": (
        "ok", "trackCount", "highThresholdShipCount", "lowThresholdShipCount",
        "coverageComplete",
    ),
    "match_entities": (
        "ok", "trackCount", "inRegistryCount", "unregisteredCount",
        "coverageComplete",
    ),
    "show_evidence": ("ok", "trackIds"),
}
_MATCH_FIELDS = (
    "trackId", "startTimeText", "endTimeText", "keyframeId", "hullNumber", "description", "score",
    "scoreBand", "registryReferenceIds", "matchedKeyframeId", "matchedRegistryReferenceId",
)


def _payload(response: Any) -> Any:
    value = getattr(response, "content", response)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _model_result(name: str, result: Any) -> Any:
    if name not in _MODEL_RESULT_FIELDS or not isinstance(result, dict):
        return result
    if result.get("ok") is False:
        fields = ("ok", "error", "errorType", "tool", "timeoutSeconds", "hint", "decision", "registryReferenceId", "keyframeId")
        compact = {key: result[key] for key in fields if result.get(key) is not None}
        if "hint" in compact:
            compact["hint"] = str(compact["hint"])[:400]
        return compact
    compact = {key: result[key] for key in _MODEL_RESULT_FIELDS[name] if key in result}
    if name in {"recognize_tracks", "match_registry_tracks"}:
        statistics = result.get("statistics")
        statistics = statistics if isinstance(statistics, dict) else {}
        for key in ("searchedTrackCount", "candidateCount", "unsearchableTrackCount"):
            if key in statistics:
                compact[key] = statistics[key]
    matches = compact.get("matches")
    if isinstance(matches, list):
        compact["matches"] = [
            {key: match[key] for key in _MATCH_FIELDS if key in match}
            for match in matches[:6]
            if isinstance(match, dict)
        ]
    if isinstance(compact.get("facts"), list):
        compact["facts"] = [str(fact)[:160] for fact in compact["facts"][:3]]
    return compact


def _model_message(request: Any, response: Any, result: Any) -> Any:
    call = request.tool_call
    name = str(call.get("name") or "")
    if name not in _MODEL_RESULT_FIELDS or not isinstance(result, dict):
        return response
    compact = _model_result(name, result)
    if compact == result:
        return response
    previous = getattr(response, "artifact", None)
    artifact = dict(previous) if isinstance(previous, dict) else {}
    if name == "show_evidence":
        artifact["full_result"] = {**compact, "trackRows": result.get("trackRows", [])}
    else:
        artifact["full_result"] = compact if name in {"dedup_tracks", "match_entities"} else result
    return _message(
        request,
        json.dumps(compact, ensure_ascii=False, default=str),
        status=str(getattr(response, "status", "success") or "success"),
        artifact=artifact,
    )


class AgentCancelled(RuntimeError):
    """Stop an in-flight agent before it starts another model request."""


def _signature(name: str, arguments: Any) -> str:
    normalized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{normalized}"


def _error_payload(error: BaseException) -> dict[str, Any]:
    return {"ok": False, "errorType": type(error).__name__, "error": str(error)}


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
    ) -> None:
        self.state = state if state is not None else SharedTaskState("")
        self.scope = scope
        self.repeat_limit = max(2, int(repeat_limit))
        self.persist = persist
        self.allowed_tools = allowed_tools
        self.cancelled = cancelled
        self.model_timeout = model_timeout

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
        record = self.state.claim_tool_call(
            self.scope,
            key,
            name,
            repeat_limit=self.repeat_limit,
        )
        self._save()

        if record.get("blocked") == "repeat_limit":
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
            payload = {"cached_result": _model_result(name, result), "message": REPEAT_HINT}
            if record.get("truncated"):
                payload["truncated"] = True
            content = json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
            artifact["full_result"] = result
            return None, _message(request, content, artifact=artifact)

        if record.get("status") == "error":
            result = record.get("result")
            content = json.dumps(
                {"cached_error": _model_result(name, result), "message": REPEAT_HINT},
                ensure_ascii=False,
                default=str,
            )
            artifact["full_result"] = result
            return None, _message(request, content, status="error", artifact=artifact)
        return None, _message(request, REPEAT_HINT, artifact=artifact)

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
        state_result = result
        if name in {"recognize_tracks", "match_registry_tracks", "verify_vessel_images"} and not failed:
            state_result = _model_result(name, result)
        elif name == "match_entities" and not failed and isinstance(result, dict):
            try:
                rows = [
                    {
                        field: item[field]
                        for field in ("trackId", "startTime", "endTime", "startTimeText", "endTimeText", "keyframeId", "score", "scoreBand", "matchedObject")
                        if field in item
                    }
                    for item in result.get("matches", [])[:60]
                    if isinstance(item, dict)
                ]
                self.state.store_evidence_artifact({
                    "tool": name,
                    "result": {
                        "matches": rows,
                        "trackCount": result.get("trackCount", len(result.get("matches", []))),
                        "truncated": len(result.get("matches", [])) > len(rows),
                    },
                })
            except OSError:
                pass
            state_result = _model_result(name, result)
        elif name == "dedup_tracks" and not failed and isinstance(result, dict):
            try:
                self.state.store_evidence_artifact({
                    "tool": name,
                    "result": {
                        "trackRows": result.get("trackRows", [])[:60],
                        "truncated": bool(result.get("trackRowsTruncated")),
                    },
                })
            except OSError:
                pass
            state_result = _model_result(name, result)
        elif name == "show_evidence" and not failed and isinstance(result, dict):
            try:
                self.state.store_evidence_artifact({
                    "tool": name,
                    "result": {
                        "trackRows": result.get("trackRows", [])[:60],
                    },
                })
            except OSError:
                pass
            state_result = _model_result(name, result)
        self.state.finish_call(key, state_result, failed)
        if name not in STATE_TOOLS and name != "read_file":
            self.state.record_tool_result(
                self.scope,
                name,
                str(call.get("id") or ""),
                state_result,
                "error" if failed else "completed",
            )
        self._save()

    def _failed(self, request: Any, key: str | None, error: BaseException) -> None:
        call = request.tool_call
        name = str(call.get("name") or "")
        result = _error_payload(error)
        self.state.finish_call(key, result, failed=True)
        if name not in STATE_TOOLS and name != "read_file":
            self.state.record_tool_result(
                self.scope, name, str(call.get("id") or ""), result, "error"
            )
        self._save()

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        try:
            response = handler(request)
        except Exception as error:  # noqa: BLE001 - Tool failures are returned to the model.
            self._failed(request, key, error)
            result = _error_payload(error)
            return _model_message(
                request,
                _message(request, json.dumps(result, ensure_ascii=False), status="error"),
                result,
            )
        self._complete(request, key, response)
        return _model_message(request, response, _payload(response))

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        key, blocked = self._prepare(request)
        if blocked is not None:
            return blocked
        try:
            response = await handler(request)
        except Exception as error:  # noqa: BLE001 - Tool failures are returned to the model.
            self._failed(request, key, error)
            result = _error_payload(error)
            return _model_message(
                request,
                _message(request, json.dumps(result, ensure_ascii=False), status="error"),
                result,
            )
        self._complete(request, key, response)
        return _model_message(request, response, _payload(response))

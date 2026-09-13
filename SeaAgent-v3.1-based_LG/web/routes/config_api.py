"""运行参数设置接口。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from config.settings import public_settings, reset_settings, update_settings
from web.models import ApiResponse


def _sync_runtime_services(request: Request) -> None:
    config = request.app.state.config
    llm = getattr(request.app.state, "llm", None)
    if llm is not None:
        llm.prompts = config.get("prompts", {})
        llm.config = config
    tools = getattr(request.app.state, "tool_service", None)
    if tools is not None and getattr(tools, "llm", None) is not None:
        tools.llm.prompts = config.get("prompts", {})
        tools.llm.config = config

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings(request: Request):
    return public_settings(request.app.state.config)


@router.put("", response_model=ApiResponse)
async def save_settings(body: dict[str, Any], request: Request):
    try:
        data = update_settings(request.app.state.config, body)
        _sync_runtime_services(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ApiResponse(success=True, message="运行参数与提示词已保存，下一轮任务与问答生效", data=data)


@router.post("/reset", response_model=ApiResponse)
async def restore_settings(request: Request):
    data = reset_settings(request.app.state.config)
    _sync_runtime_services(request)
    return ApiResponse(success=True, message="运行参数与提示词已恢复默认值", data=data)

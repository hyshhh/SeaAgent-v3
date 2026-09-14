"""轨迹记忆管理接口。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from memory import TrackMemoryManager
from web.models import ApiResponse
from web.routes.pipeline_api import has_running_pipeline

router = APIRouter(prefix="/api/memory", tags=["memory"])


class MemorySettingsUpdate(BaseModel):
    retention_seconds: float = Field(default=0, ge=0, le=86400)


def get_manager(request: Request) -> TrackMemoryManager:
    return request.app.state.memory_manager


@router.get("/tracks")
async def list_tracks(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    return manager.snapshot()


@router.get("/settings")
async def get_settings(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    return manager.settings.read()


@router.put("/settings", response_model=ApiResponse)
async def update_settings(body: MemorySettingsUpdate, manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    settings = manager.settings.write(body.retention_seconds)
    return ApiResponse(success=True, message="记忆维护时间已更新", data=settings)


@router.delete("", response_model=ApiResponse)
async def clear_memory(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    if has_running_pipeline():
        raise HTTPException(409, "流水线正在运行，请先停止处理再清除记忆")
    result = manager.clear_all()
    return ApiResponse(success=True, message="轨迹记忆已全部清除", data=result)

"""轨迹记忆管理接口。"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from memory import TrackMemoryManager
from web.models import ApiResponse
from web.routes.pipeline_api import has_running_pipeline

router = APIRouter(prefix="/api/memory", tags=["memory"])


# ------------------------------------------------------------------ #
# 一、记忆管理                                                             #
# ------------------------------------------------------------------ #
# 1.1 记忆设置请求体
class MemorySettingsUpdate(BaseModel):
    retention_seconds: float = Field(default=0, ge=0, le=86400)


# 1.2 取记忆管理器
def get_manager(request: Request) -> TrackMemoryManager:
    return request.app.state.memory_manager


# 1.3 轨迹列表
@router.get("/tracks")
async def list_tracks(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    return manager.snapshot()


# 1.4 读取记忆设置
@router.get("/settings")
async def get_settings(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    return manager.settings.read()


# 1.5 更新记忆设置
@router.put("/settings", response_model=ApiResponse)
async def update_settings(body: MemorySettingsUpdate, manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    settings = manager.settings.write(body.retention_seconds)
    return ApiResponse(success=True, message="记忆维护时间已更新", data=settings)


# 1.6 清空记忆
@router.delete("", response_model=ApiResponse)
async def clear_memory(manager: Annotated[TrackMemoryManager, Depends(get_manager)]):
    if has_running_pipeline():
        raise HTTPException(409, "流水线正在运行，请先停止处理再清除记忆")
    result = manager.clear_track_memory()
    return ApiResponse(success=True, message="轨迹记忆已全部清除", data=result)

"""网页接口数据结构。"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class ShipCreate(BaseModel):
    hull_number: str = Field(..., min_length=1, max_length=50)
    description: str = Field(default="", max_length=2000)
    aliases: list[str] = []

class ShipUpdate(BaseModel):
    description: str | None = Field(default=None, max_length=2000)
    aliases: list[str] | None = None

class ShipBulkCreate(BaseModel):
    ships: dict[str, str]

class AgentQuery(BaseModel):
    """一轮问答请求；带上 sessionId 即在该会话里追问，不带则开新会话。"""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, alias="sessionId", max_length=80, pattern=r"^[A-Za-z0-9_-]+$")

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Any = None

class ShipItem(BaseModel):
    registry_id: str | None = None
    hull_number: str
    description: str
    aliases: list[str] = []
    references: list[dict[str, Any]] = []
    searchable: bool = False

class ShipListResponse(BaseModel):
    total: int
    ships: list[ShipItem]

class StatsResponse(BaseModel):
    total_ships: int
    total_reference_images: int = 0
    backend: str

class SearchResponse(BaseModel):
    total: int
    results: list[ShipItem]

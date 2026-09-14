"""从智能体的结构化返回契约（官方 ``response_format``）。

为什么值得多写这几个模型：隔离模式下主智能体**只拿到子智能体最后一条消息**，自由文本里少一个
trackId、多一句客套，主智能体就得靠猜。挂上 schema 后父方直接收到 JSON，字段齐全、可解析，
小模型也不用再学"怎么把表格写整齐"。

字段一律给默认值（空列表/空串），把"没查到"表达成空集合而不是缺字段——4B 模型偶尔漏填时
校验仍然能过，不会因为一个字段把整轮委派打回。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrackFinding(BaseModel):
    """一条去重后的候选轨迹。"""

    track_id: str = Field(default="", description="轨迹编号")
    time_range: str = Field(default="", description="起止时间，例如 15:30:12-15:34:40")
    hull_number: str = Field(default="", description="识别到的舷号；未识别留空")
    confidence: float = Field(default=0.0, description="匹配置信度 0-1")
    keyframe_ids: list[str] = Field(default_factory=list, description="该轨迹的关键帧编号")
    note: str = Field(default="", description="需要主智能体知道的补充，例如存疑原因")


class TrackScoutFindings(BaseModel):
    """轨迹筛查结果。"""

    tracks: list[TrackFinding] = Field(default_factory=list, description="去重后的候选轨迹")
    uncertain_track_ids: list[str] = Field(default_factory=list, description="证据不足、无法定论的轨迹")
    summary: str = Field(default="", description="一句话摘要：共几条、去重后几条、存疑几条")


class RegistryMatch(BaseModel):
    """一条先验库比对结论。"""

    track_id: str = Field(default="", description="被核对的轨迹编号")
    registry_id: str = Field(default="", description="命中的先验库编号；未命中留空")
    hull_number: str = Field(default="", description="库内舷号")
    reference_ids: list[str] = Field(default_factory=list, description="库内参考图编号")
    confidence: float = Field(default=0.0, description="匹配置信度 0-1")
    reason: str = Field(default="", description="判定依据")


class RegistryCheckFindings(BaseModel):
    """先验库核验结果，三组分开。"""

    in_registry: list[RegistryMatch] = Field(default_factory=list, description="确认在库")
    not_in_registry: list[RegistryMatch] = Field(default_factory=list, description="确认不在库")
    uncertain: list[RegistryMatch] = Field(default_factory=list, description="存疑")
    summary: str = Field(default="", description="一句话摘要")


class VisualProof(BaseModel):
    """一条视觉取证结论。"""

    verdict: str = Field(default="undetermined", description="confirmed / mismatched / undetermined 三选一")
    track_id: str = Field(default="", description="轨迹编号")
    keyframe_ids: list[str] = Field(default_factory=list, description="作为证据的关键帧编号")
    registry_reference_ids: list[str] = Field(default_factory=list, description="作为证据的库参考图编号")
    ship_segment_ids: list[str] = Field(default_factory=list, description="生成的视频片段编号")
    confidence: float = Field(default=0.0, description="置信度 0-1")
    reason: str = Field(default="", description="判定依据；无法判定时写清缺什么证据")


class VisualProofFindings(BaseModel):
    """视觉取证结果，按结论分组，证据 ID 逐条带出。"""

    confirmed: list[VisualProof] = Field(default_factory=list, description="确认匹配")
    mismatched: list[VisualProof] = Field(default_factory=list, description="确认不匹配")
    undetermined: list[VisualProof] = Field(default_factory=list, description="无法判定")
    summary: str = Field(default="", description="一句话摘要")


#: yaml 里按名字引用，避免把 schema 写进配置
SUBAGENT_RESPONSE_FORMATS: dict[str, type[BaseModel]] = {
    "TrackScoutFindings": TrackScoutFindings,
    "RegistryCheckFindings": RegistryCheckFindings,
    "VisualProofFindings": VisualProofFindings,
}


def resolve_response_format(name: Any) -> type[BaseModel] | None:
    """按名字取结构化返回契约；未声明返回 None（退回自由文本）。"""
    key = str(name or "").strip()
    if not key:
        return None
    if key not in SUBAGENT_RESPONSE_FORMATS:
        raise ValueError(f"未知的从智能体返回契约：{key}（可选：{'、'.join(SUBAGENT_RESPONSE_FORMATS)}）")
    return SUBAGENT_RESPONSE_FORMATS[key]

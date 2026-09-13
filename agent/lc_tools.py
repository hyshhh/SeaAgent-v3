"""将现有 ToolService / 解析函数封装为 LangChain tools。

定位：这是**模型直调**路径的工具封装，与 plan_executor.py 的**确定性执行**
路径并存。当前 graph 只用到本文件的 intent 解析工具与技能工具 ——
observe 的业务检索工具一律走 PlanExecutor.execute，不经过模型，故本文件不再
保留业务工具的封装（原 build_observe_tools 及其专属 Args 已删除）。

文件分区（以 L 编号为锚点检索）：
    L1  序列化辅助与参数 schema   _jsonable / _dump / intent 工具的 *Args
    L2  intent 工具                parseTime / parseTargets / extractHull
    L3  技能工具                   一技能一工具：load_<skill_id>
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.target_parser import extract_hull_number, extract_target_items
from tools.time_normalizer import normalize_time_range

from .skill_loader import list_skill_catalog, load_skill_body

logger = logging.getLogger(__name__)


# ===========================================================================
# L1 序列化辅助与参数 schema
# _jsonable / _dump 负责把工具结果转成可 JSON 化的形态；*Args 是各工具的参数
# schema —— 它们的 description 会进入模型看到的工具说明，是提示词的一部分。
# ===========================================================================
def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _dump(result: Any) -> str:
    return json.dumps(_jsonable(result), ensure_ascii=False)


class ParseTimeArgs(BaseModel):
    expression: str = Field(description="自然语言时间表达，如昨天下午")


class ParseTargetsArgs(BaseModel):
    question: str = Field(description="用户原问题，用于多目标切分")


class ExtractHullArgs(BaseModel):
    question: str = Field(description="用户原问题，用于抽取舷号")


# ===========================================================================
# L2 intent 工具
# 三个纯解析工具（时间/多目标/舷号），由 IntentAgent 调用。返回的是「待确认」
# 的建议值，节点侧还会用 infer_intent_fields 做规则回填兜底。
# ===========================================================================
def build_intent_tools(reference_time: datetime | None = None) -> list[StructuredTool]:
    now = reference_time or datetime.now().astimezone()

    def parse_time(expression: str) -> str:
        rng = normalize_time_range(expression, now=now)
        return _dump({
            "ok": True,
            "timeRange": list(rng) if rng else None,
            "expression": expression,
            "hint": "确认后写入意图 timeRange",
        })

    def parse_targets(question: str) -> str:
        return _dump({
            "ok": True,
            "targetItems": extract_target_items(question),
            "hint": "确认后写入意图 targetItems",
        })

    def extract_hull(question: str) -> str:
        return _dump({"ok": True, "hullNumber": extract_hull_number(question)})

    return [
        StructuredTool.from_function(
            name="parseTime",
            description="仅当用户原问题明确包含时间表达时调用，将该表达归一化为 Unix 秒区间 [start, end]；禁止为无时间问题生成默认范围。",
            func=parse_time,
            args_schema=ParseTimeArgs,
        ),
        StructuredTool.from_function(
            name="parseTargets",
            description="从问题中切分多个船舶目标",
            func=parse_targets,
            args_schema=ParseTargetsArgs,
        ),
        StructuredTool.from_function(
            name="extractHull",
            description="从问题中抽取疑似舷号",
            func=extract_hull,
            args_schema=ExtractHullArgs,
        ),
    ]


# ===========================================================================
# L3 技能工具
# 每个 skill 包装成一个独立工具：工具名 load_<skill_id>，description 取
# catalog.yaml 的简介 —— **模型看到的工具列表本身就是技能目录**，调用才拿到
# 正文，这就是两级披露的第二级。用哪个技能由模型看描述自行判断，代码不做预选。
# ===========================================================================
def build_skill_tools(agent_key: str) -> list[StructuredTool]:
    """把 skills/{agent_key}/ 下的每个技能包装成一个无参工具。

    正文读不出来的技能直接跳过挂载（免得模型去调一个必然失败的工具），但要记
    warning —— catalog 解析失败或目录为空会让整条技能通道无声消失，不能静默。
    """
    catalog = list_skill_catalog(agent_key)
    if not catalog:
        logger.warning("skills/%s 未登记任何技能，该 Agent 将没有技能工具", agent_key)

    def _make(skill_id: str) -> Callable[[], str]:
        def _load() -> str:
            body = load_skill_body(agent_key, skill_id)
            if not body:
                return _dump({"ok": False, "error": f"skill_unavailable:{skill_id}"})
            return _dump({"ok": True, "skillId": skill_id, "content": body})

        return _load

    tools: list[StructuredTool] = []
    for meta in catalog:
        if not load_skill_body(agent_key, meta.id):
            logger.warning(
                "技能 %s/%s 正文为空（file=%s），已跳过挂载", agent_key, meta.id, meta.file
            )
            continue
        description = f"{meta.title}：{meta.description}" if meta.description else meta.title
        tools.append(
            StructuredTool.from_function(
                name=f"load_{meta.id}",
                description=description,
                func=_make(meta.id),
            )
        )
    return tools

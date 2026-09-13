"""将现有 ToolService / 解析函数封装为 LangChain tools。

定位：这是**模型直调**路径的工具封装，与 plan_executor.py 的**确定性执行**
路径并存。当前 graph 只用到本文件的 intent 解析工具与 loadSkill ——
observe 的业务检索工具一律走 PlanExecutor.execute，不经过模型，故本文件不再
保留业务工具的封装（原 build_observe_tools 及其专属 Args 已删除）。

文件分区（以 L 编号为锚点检索）：
    L1  序列化辅助与参数 schema   _jsonable / _dump / intent 与 loadSkill 的 *Args
    L2  intent 工具                parseTime / parseTargets / extractHull
    L3  技能加载工具                loadSkill
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.target_parser import extract_hull_number, extract_target_items
from tools.time_normalizer import normalize_time_range


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


class LoadSkillArgs(BaseModel):
    skillId: str = Field(description="catalog 中的 skill id")


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
# L3 技能加载工具
# 模型在 ReAct 过程中按需拉取「只给了目录、未注入正文」的技能全文，
# 与 skill_loader 的两级披露配套。load_fn 由 graph 注入（_skill_loader）。
# ===========================================================================
def build_load_skill_tool(agent_key: str, load_fn: Callable[[str], dict[str, Any]]) -> StructuredTool:
    def _run(skillId: str) -> str:
        return _dump(load_fn(skillId))

    return StructuredTool.from_function(
        name="loadSkill",
        description="按需加载可选 skill 全文",
        func=_run,
        args_schema=LoadSkillArgs,
    )

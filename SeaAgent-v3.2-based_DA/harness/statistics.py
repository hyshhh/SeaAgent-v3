"""Deterministic routing and answers for plain vessel-count requests."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

QA_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")

_COUNT = re.compile(r"多少|几(?:艘|只|个|条|船)|船数|数量|总数|how many|count|number of", re.IGNORECASE)
_VESSEL = re.compile(r"艘|船|舰|艇|实体|轨迹|目标|vessels?|ships?|boats?|tracks?|trajectories|entities", re.IGNORECASE)
_UNREGISTERED = re.compile(
    r"不在(?:先验)?库|未在(?:先验)?库|库外|未注册|未入库|非在库|不属于(?:先验)?库|"
    r"unregistered|not (?:in (?:the )?registry|registered)|outside (?:the )?registry",
    re.IGNORECASE,
)
_REGISTERED = re.compile(r"在库|库内|注册|先验库|registered|in (?:the )?registry", re.IGNORECASE)
_FILTERED = re.compile(r"舷号|船名|颜色|黄色|黑色|白色|图片|照片|描述|特征|相似|识别|某类|某种|指定的", re.IGNORECASE)
_DATE = re.compile(
    r"(?P<iso>20\d{2}[-/]\d{1,2}[-/]\d{1,2})|"
    r"(?P<cn>20\d{2}年\d{1,2}月\d{1,2}日?)|"
    r"(?P<relative>大前天|前天|昨天|今日|今天|明天|后天)",
    re.IGNORECASE,
)
_RELATIVE_DAYS = {"大前天": -3, "前天": -2, "昨天": -1, "今日": 0, "今天": 0, "明天": 1, "后天": 2}
_ALL_TRACKS = re.compile(r"全部时间|所有时间|全部轨迹|所有轨迹|全时段|全视频|整个视频|all (?:stored )?(?:tracks|time)|entire video", re.IGNORECASE)
_UNSUPPORTED_TIME = re.compile(
    r"本周|上周|这周|下周|本月|上月|下月|今年|去年|最近|近\s*\d+\s*天|前\s*\d+\s*天|过去|"
    r"上午|下午|晚上|凌晨|早上|夜间|\b\d{1,2}:\d{2}\b|"
    r"\b(?:last|past|next)\s+\d+\s+(?:days?|weeks?|months?|years?)\b|"
    r"\b(?:from|since|during|last|this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year|quarter|morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
_AMBIGUOUS = re.compile(
    r"为什么|怎么|如何|原因|解释|是否|有没有|能否|并且|同时|分别|以及|"
    r"(?:总数|全部船数).{0,12}(?:在库|不在库)|"
    r"(?:在库|不在库).{0,12}(?:总数|全部船数)|"
    r"(?:其中|各有|各多少|这些|那些|其余|刚才|上述|这批|那批|当天)|"
    r"\b(?:why|how does|what is)\b",
    re.IGNORECASE,
)


def plan_statistics(question: str, time_anchor: str) -> dict[str, Any] | None:
    """Return a one-tool plan only for unfiltered total-count requests."""
    text = str(question or "").strip()
    if not _COUNT.search(text) or not _VESSEL.search(text) or any(
        pattern.search(text) for pattern in (_AMBIGUOUS, _FILTERED, _UNSUPPORTED_TIME)
    ):
        return None

    unregistered = bool(_UNREGISTERED.search(text))
    registered = bool(_REGISTERED.search(_UNREGISTERED.sub("", text)))
    if unregistered and registered:
        return None

    try:
        time_range = _time_range(text, time_anchor)
    except (TypeError, ValueError, OverflowError):
        return None
    if unregistered:
        kind, tool, target = "unregistered", "match_entities", "不在库船舶数量"
    elif registered:
        kind, tool, target = "registered", "match_entities", "在库船舶数量"
    else:
        kind, tool, target = "total", "dedup_tracks", "船舶总数"

    scope = (
        f"范围：{time_range[0]} 至 {time_range[1]}"
        if time_range
        else "全部已索引轨迹"
    )
    return {
        "kind": kind,
        "tool": tool,
        "time_range": time_range,
        "requirements": [f"统计已索引轨迹中的{target}"],
        "plan": [{
            "id": "count-1",
            "content": f"调用 {tool} 统计{target}；{scope}。",
            "required_evidence": [tool],
        }],
        "next_action": "execute count-1",
        "clarification": None,
    }


def _time_range(text: str, time_anchor: str) -> list[str] | None:
    tokens = list(_DATE.finditer(text))
    if not tokens:
        if _ALL_TRACKS.search(text):
            return None
        if re.search(
            r"时间|日期|时段|周|月|年|\d+\s*(?:天|周|个月|年)前|"
            r"\b(?:yesterday|today|tomorrow|date|time|week|month|year|quarter|morning|afternoon|evening|night)\b",
            text,
            re.IGNORECASE,
        ):
            raise ValueError("time expression is not a single date range")
        return None
    try:
        anchor = datetime.fromisoformat(time_anchor).date()
        dates = []
        for token in tokens:
            if token.group("iso"):
                year, month, day = map(int, re.split(r"[-/]", token.group("iso")))
                dates.append(date(year, month, day))
            elif token.group("cn"):
                year, month, day = map(
                    int,
                    re.findall(r"\d+", token.group("cn")),
                )
                dates.append(date(year, month, day))
            else:
                dates.append(anchor + timedelta(days=_RELATIVE_DAYS[token.group("relative")]))
        if len(dates) > 2 or (
            len(dates) == 2
            and not re.search(r"到|至|和|之间|\bto\b|\band\b", text[tokens[0].end():tokens[1].start()], re.IGNORECASE)
        ):
            raise ValueError("time expression is ambiguous")
        first, last = dates[0], dates[-1]
        if first > last:
            raise ValueError("time range starts after it ends")
        start = datetime.combine(first, time.min, tzinfo=QA_TIMEZONE)
        end = datetime.combine(last, time.max, tzinfo=QA_TIMEZONE)
        return [start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("time expression is invalid") from error

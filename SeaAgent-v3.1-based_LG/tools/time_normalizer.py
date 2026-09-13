"""将监控问答中的自然语言时间归一化为本地时间戳范围。

词表、时段、检测模式等均来自 skills/intent_agent/time_parsing.yaml；
本模块只加载规则并执行归一化算法。
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from agent.skill_loader import load_skill_yaml


@lru_cache(maxsize=1)
def _rules() -> dict[str, Any]:
    data = load_skill_yaml("intent_agent", "time_parsing")
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _cfg() -> dict[str, Any]:
    rules = _rules()
    cn_digits_raw = rules.get("cn_digits") or {}
    cn_digits = {str(k): int(v) for k, v in cn_digits_raw.items()} if isinstance(cn_digits_raw, dict) else {}
    cn_number = str(rules.get("cn_number_pattern") or r"[〇零一二两三四五六七八九十]{1,4}")
    short = rf"(?:\d{{1,2}}|{cn_number})"
    long = rf"(?:\d{{1,4}}|{cn_number})"
    period_names = tuple(str(x) for x in (rules.get("period_names") or []))
    period_pattern = "|".join(re.escape(name) for name in period_names) if period_names else "凌晨"
    period_alias = {str(k): str(v) for k, v in (rules.get("period_alias") or {}).items()}
    period_spans_raw = rules.get("period_spans") or {}
    period_spans = {
        str(k): (int(v[0]), int(v[1]))
        for k, v in period_spans_raw.items()
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }
    fuzzy = tuple(str(x) for x in (rules.get("fuzzy_markers") or []))
    duration_units = {
        str(k): float(v) for k, v in (rules.get("duration_unit_seconds") or {}).items()
    }
    def _inject(template: str) -> str:
        return (
            str(template)
            .replace("{short}", short)
            .replace("{long}", long)
            .replace("{period}", period_pattern)
        )

    patterns_tpl = list(rules.get("time_expression_patterns") or [])
    time_patterns = [_inject(p) for p in patterns_tpl]
    has_date = _inject(str(rules.get("has_date_phrase_pattern") or ""))
    return {
        "cn_digits": cn_digits,
        "cn_number": cn_number,
        "short": short,
        "long": long,
        "period_names": period_names,
        "period_pattern": period_pattern,
        "period_alias": period_alias,
        "period_spans": period_spans,
        "fuzzy": fuzzy,
        "duration_units": duration_units,
        "time_patterns": time_patterns,
        "has_date_phrase": has_date,
        "text_substitutions": list(rules.get("text_substitutions") or []),
        "literal_replacements": {
            str(k): str(v) for k, v in (rules.get("literal_replacements") or {}).items()
        },
        "just_now_markers": tuple(str(x) for x in (rules.get("just_now_markers") or [])),
        "just_now_minutes": int(rules.get("just_now_minutes") or 10),
        "later_markers": tuple(str(x) for x in (rules.get("later_markers") or [])),
        "later_hours": int(rules.get("later_hours") or 1),
        "daytime_marker": str(rules.get("daytime_marker") or "白天"),
        "daytime_span": tuple(rules.get("daytime_span") or [6, 18]),
        "fuzzy_clock_window_minutes": int(rules.get("fuzzy_clock_window_minutes") or 30),
        "relative_day_phrases": list(rules.get("relative_day_phrases") or []),
        "year_offsets": {str(k): int(v) for k, v in (rules.get("year_offsets") or {}).items()},
        "week_prefix_offsets": {
            (None if k in ("null", "None", "") else str(k)): int(v)
            for k, v in (rules.get("week_prefix_offsets") or {}).items()
        },
        "weekday_index": {str(k): int(v) for k, v in (rules.get("weekday_index") or {}).items()},
        "relative_month_phrases": {
            str(k): int(v) for k, v in (rules.get("relative_month_phrases") or {}).items()
        },
        "model_start_keys": list(rules.get("model_start_keys") or ["start", "startTime", "begin"]),
        "model_end_keys": list(rules.get("model_end_keys") or ["end", "endTime", "finish"]),
        "datetime_formats": list(rules.get("datetime_formats") or []),
        "next_day_bridge_markers": tuple(str(x) for x in (rules.get("next_day_bridge_markers") or [])),
        "relative_end_day_markers": {
            str(k): int(v) for k, v in (rules.get("relative_end_day_markers") or {}).items()
        },
        "clock_bridge_pattern": str(
            rules.get("clock_bridge_pattern")
            or r"(?:左右|前后)*(?:到|至|[-—~～－])(?:左右|前后)*"
        ),
        "date_range_bridge_pattern": str(
            rules.get("date_range_bridge_pattern") or r"到|至|[-—~～－]"
        ),
    }


def normalize_time_range(question: str, now: datetime | None = None) -> tuple[float, float] | None:
    """将自然语言中的日期、时段和钟点转换为具体的起止时间戳。"""
    text = _normalize_text(question)
    if not text:
        return None
    current = _local_now(now)
    recent = _relative_duration_range(text, current)
    if recent is not None:
        return _timestamp_range(*recent)

    date_tokens = _date_tokens(text, current)
    time_tokens = _time_tokens(text)
    date_range = _explicit_date_range(text, date_tokens, time_tokens)
    if date_range is not None:
        return _timestamp_range(*date_range)

    date_span = _date_span(text, current, date_tokens)
    if date_span is None:
        date_span = _day_span(current)
    start_day, end_day = date_span

    if len(date_tokens) >= 2:
        return _timestamp_range(start_day, end_day)
    if end_day - start_day > timedelta(days=1):
        return _timestamp_range(start_day, end_day)

    time_range = _clock_range(text, start_day, time_tokens, current)
    if time_range is not None:
        return _timestamp_range(*time_range)

    period_range = _period_range(text, start_day)
    if period_range is not None:
        return _timestamp_range(*period_range)

    if date_tokens or _has_date_phrase(text):
        return _timestamp_range(start_day, end_day)
    return None


def has_time_expression(question: str) -> bool:
    """判断用户是否显式给出了应当作为查询约束的时间表达。"""
    text = _normalize_text(question)
    if not text:
        return False
    for pattern in _cfg()["time_patterns"]:
        if re.search(pattern, text):
            return True
    return False



def parse_time(
    expression: str,
    now: datetime | None = None,
    *,
    fallback_question: str | None = None,
) -> dict[str, Any]:
    """Intent 工具协议：parseTime。"""
    time_range = normalize_time_range(expression, now=now)
    if time_range is None and fallback_question and fallback_question != expression:
        time_range = normalize_time_range(fallback_question, now=now)
    return {
        "ok": True,
        "timeRange": list(time_range) if time_range else None,
        "expression": expression,
        "hint": "可选参考；请你确认后写入 result.timeRange",
    }


def _normalize_text(question: str) -> str:
    cfg = _cfg()
    text = unicodedata.normalize("NFKC", str(question or "")).lower()
    for item in cfg["text_substitutions"]:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or "")
        replacement = str(item.get("replacement") or "")
        if pattern:
            text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"(?i)(\d{1,2}(?::\d{2})?)\s*p\.?m\.?", r"下午\1", text)
    text = re.sub(r"(?i)(\d{1,2}(?::\d{2})?)\s*a\.?m\.?", r"上午\1", text)
    for src, dst in cfg["literal_replacements"].items():
        text = text.replace(src, dst)
    text = re.sub(r"[\s()（）\[\]【】]", "", text)
    return text


def _local_now(now: datetime | None) -> datetime:
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return current


def _parse_integer(value: Any) -> int | None:
    cfg = _cfg()
    cn_digits = cfg["cn_digits"]
    token = str(value or "").strip()
    if not token:
        return None
    if re.fullmatch(r"\d+", token):
        return int(token)
    if all(character in cn_digits for character in token):
        return int("".join(str(cn_digits[character]) for character in token))
    if "十" not in token:
        return None
    if token.count("十") != 1:
        return None
    left, right = token.split("十", 1)
    if left and (len(left) != 1 or left not in cn_digits):
        return None
    if right and (len(right) != 1 or right not in cn_digits):
        return None
    tens = cn_digits[left] if left else 1
    ones = cn_digits[right] if right else 0
    return tens * 10 + ones


def _parse_amount(value: Any) -> float | None:
    token = str(value or "").strip()
    if token == "半":
        return 0.5
    if token == "一刻":
        return 1.0
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    parsed = _parse_integer(token)
    return float(parsed) if parsed is not None else None


def _day_span(value: datetime) -> tuple[datetime, datetime]:
    start = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _month_start(value: datetime, shift: int = 0) -> datetime:
    month_index = value.year * 12 + value.month - 1 + shift
    year, month_index = divmod(month_index, 12)
    return value.replace(year=year, month=month_index + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _month_span(value: datetime, shift: int = 0) -> tuple[datetime, datetime]:
    start = _month_start(value, shift)
    return start, _month_start(start, 1)


def _make_day(current: datetime, year: int | None, month: int, day: int) -> datetime | None:
    try:
        return current.replace(year=year or current.year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None


def _date_tokens(text: str, current: datetime) -> list[dict[str, Any]]:
    cfg = _cfg()
    long, short = cfg["long"], cfg["short"]
    records: list[dict[str, Any]] = []
    patterns = (
        re.compile(rf"(?P<year>{long})\s*(?:年|[./-])\s*(?P<month>{short})\s*(?:月|[./-])\s*(?P<day>{short})(?:日|号)?"),
        re.compile(rf"(?<![\d年])(?P<month>{short})月(?P<day>{short})(?:日|号)?"),
        re.compile(rf"(?<![\d年月])(?P<day>{short})(?:日|号)"),
    )
    for index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            if any(not (match.end() <= item["start"] or match.start() >= item["end"]) for item in records):
                continue
            month = _parse_integer(match.groupdict().get("month"))
            day = _parse_integer(match.group("day"))
            year = _parse_integer(match.groupdict().get("year"))
            if month is None:
                month = current.month
            if day is None:
                continue
            resolved = _make_day(current, year, month, day)
            if resolved is None:
                continue
            records.append({"start": match.start(), "end": match.end(), "day": resolved, "kind": ("ymd", "md", "d")[index]})
    return sorted(records, key=lambda item: item["start"])


def _relative_duration_range(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    amount_pattern = rf"(?:\d+(?:\.\d+)?|{cfg['cn_number']}|半|一刻)"
    recent = re.search(rf"(?:最近|近|过去)(?P<amount>{amount_pattern})(?:个)?(?P<unit>分钟?|小时|天|日|周|星期|月|年|刻钟)", text)
    if recent:
        amount = _parse_amount(recent.group("amount"))
        seconds = _duration_seconds(amount, recent.group("unit"))
        if seconds is not None:
            return current - timedelta(seconds=seconds), current
    point = re.search(rf"(?P<amount>{amount_pattern})(?:个)?(?P<unit>分钟?|小时|刻钟)(?P<direction>前|后)", text)
    if point:
        amount = _parse_amount(point.group("amount"))
        seconds = _duration_seconds(amount, point.group("unit"))
        if seconds is not None:
            target = current + timedelta(seconds=seconds if point.group("direction") == "后" else -seconds)
            width = 60 if point.group("unit").startswith("分") or point.group("unit") == "刻钟" else 3600
            return target, target + timedelta(seconds=width)
    if any(marker in text for marker in cfg["just_now_markers"]):
        return current - timedelta(minutes=cfg["just_now_minutes"]), current
    if any(marker in text for marker in cfg["later_markers"]):
        return current, current + timedelta(hours=cfg["later_hours"])
    return None


def _duration_seconds(amount: float | None, unit: str) -> float | None:
    if amount is None:
        return None
    seconds = _cfg()["duration_units"].get(unit)
    if seconds is None:
        return None
    return amount * seconds


def _date_span(text: str, current: datetime, date_tokens: list[dict[str, Any]]) -> tuple[datetime, datetime] | None:
    if date_tokens:
        return _day_span(date_tokens[0]["day"])
    relative = _relative_day_span(text, current)
    if relative is not None:
        return relative
    week = _week_span(text, current)
    if week is not None:
        return week
    month = _month_expression_span(text, current)
    if month is not None:
        return month
    year_offsets = _cfg()["year_offsets"]
    year_match = re.search("|".join(re.escape(k) for k in year_offsets) if year_offsets else r"(?!)", text)
    if year_match:
        year = current.year + year_offsets[year_match.group(0)]
        start = current.replace(year=year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, start.replace(year=year + 1)
    return None


def _relative_day_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    for item in cfg["relative_day_phrases"]:
        if not isinstance(item, dict):
            continue
        phrases = item.get("phrases") or []
        offset = int(item.get("offset") or 0)
        if any(str(value) in text for value in phrases):
            start, _ = _day_span(current + timedelta(days=offset))
            return start, start + timedelta(days=1)
    amount_pattern = rf"(?:\d+|{cfg['cn_number']}|半)"
    match = re.search(rf"(?P<amount>{amount_pattern})(?:个)?(?P<unit>天|日|周|星期|月|年)(?P<direction>前|后)", text)
    if not match:
        return None
    amount = _parse_amount(match.group("amount"))
    if amount is None:
        return None
    direction = -1 if match.group("direction") == "前" else 1
    unit = match.group("unit")
    if unit in {"天", "日"}:
        return _day_span(current + timedelta(days=direction * amount))
    if unit in {"周", "星期"}:
        return _day_span(current + timedelta(days=direction * amount * 7))
    if unit == "月":
        target = _month_start(current, direction * int(amount))
        return _month_span(target)
    if unit == "年":
        target = current.replace(year=current.year + direction * int(amount))
        return _day_span(target)
    return None


def _week_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    prefix_offsets = dict(cfg["week_prefix_offsets"])
    prefix_offsets.setdefault(None, 0)
    weekday_index = cfg["weekday_index"]
    match = re.search(r"(?P<prefix>上上|下下|上|下|本|这)?(?:周|星期)(?P<weekday>[一二三四五六日天末])", text)
    monday, _ = _day_span(current - timedelta(days=current.weekday()))
    if match:
        offset = prefix_offsets.get(match.group("prefix"), 0)
        week_start = monday + timedelta(days=offset * 7)
        weekday = match.group("weekday")
        if weekday == "末":
            return week_start + timedelta(days=5), week_start + timedelta(days=7)
        index = weekday_index.get(weekday)
        if index is None:
            return None
        start = week_start + timedelta(days=index)
        return start, start + timedelta(days=1)
    match = re.search(r"(?P<prefix>上上|下下|上|下|本|这)?(?:周|星期)(?![一二三四五六日天末])", text)
    if match:
        week_start = monday + timedelta(days=prefix_offsets.get(match.group("prefix"), 0) * 7)
        return week_start, week_start + timedelta(days=7)
    return None


def _month_expression_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    for phrase, offset in cfg["relative_month_phrases"].items():
        if phrase in text:
            return _month_span(current, offset)
    long, short = cfg["long"], cfg["short"]
    match = re.search(rf"(?:(?P<year>{long})年)?(?P<month>{short})月(?P<part>初|中旬|中|末|底)?", text)
    if not match:
        return None
    year = _parse_integer(match.group("year"))
    month = _parse_integer(match.group("month"))
    if month is None or not 1 <= month <= 12:
        return None
    base = current.replace(year=year or current.year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
    part = match.group("part")
    if part in {None, ""}:
        return base, _month_start(base, 1)
    days = calendar.monthrange(base.year, base.month)[1]
    if part == "初":
        return base, base + timedelta(days=min(10, days))
    if part in {"中", "中旬"}:
        start = base + timedelta(days=10)
        return start, base + timedelta(days=min(20, days))
    start = base + timedelta(days=20)
    return start, _month_start(base, 1)


def _time_tokens(text: str) -> list[dict[str, Any]]:
    cfg = _cfg()
    short, period = cfg["short"], cfg["period_pattern"]
    records: list[dict[str, Any]] = []
    colon = re.compile(rf"(?P<period>{period})?(?P<hour>{short})[:：](?P<minute>\d{{1,2}})(?:[:：](?P<second>\d{{1,2}}))?")
    point = re.compile(rf"(?P<period>{period})?(?P<hour>{short})(?:点|时)(?P<minute>半|一刻|三刻|{short})?(?:分)?(?P<second>\d{{1,2}})?(?:秒)?")
    for pattern, precision in ((colon, "minute"), (point, "hour")):
        for match in pattern.finditer(text):
            hour = _parse_integer(match.group("hour"))
            minute_value = match.group("minute")
            second_value = match.group("second")
            if minute_value == "半":
                minute = 30
            elif minute_value == "一刻":
                minute = 15
            elif minute_value == "三刻":
                minute = 45
            else:
                minute = _parse_integer(minute_value) if minute_value else 0
            second = _parse_integer(second_value) if second_value else 0
            if hour is None or minute is None or second is None or not 0 <= minute <= 59 or not 0 <= second <= 59:
                continue
            records.append({
                "start": match.start(),
                "end": match.end(),
                "hour": hour,
                "minute": minute,
                "second": second,
                "period": match.group("period"),
                "precision": "minute" if precision == "minute" or minute_value else "hour",
            })
    records.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    selected: list[dict[str, Any]] = []
    for item in records:
        if any(not (item["end"] <= previous["start"] or item["start"] >= previous["end"]) for previous in selected):
            continue
        selected.append(item)
    return selected


def _resolve_hour(hour: int, period: str | None) -> int | None:
    cfg = _cfg()
    if not 0 <= hour <= 23:
        return None
    resolved_period = cfg["period_alias"].get(period or "", period)
    if resolved_period in {"下午", "傍晚", "晚上"} and 1 <= hour < 12:
        return hour + 12
    if resolved_period == "中午" and 1 <= hour <= 5:
        return hour + 12
    if resolved_period in {"凌晨", "早上", "上午"} and hour == 12:
        return 0
    return hour


def _clock_datetime(day: datetime, token: dict[str, Any], inherited_period: str | None = None) -> datetime | None:
    hour = _resolve_hour(int(token["hour"]), token.get("period") or inherited_period)
    if hour is None:
        return None
    return day.replace(hour=hour, minute=int(token["minute"]), second=int(token["second"]), microsecond=0)


def _clock_range(text: str, day: datetime, tokens: list[dict[str, Any]], current: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    for index in range(len(tokens) - 1):
        first, second = tokens[index], tokens[index + 1]
        bridge = text[first["end"]:second["start"]]
        start = _clock_datetime(day, first)
        end_day = day
        if any(marker in bridge for marker in cfg["next_day_bridge_markers"]):
            end_day = day + timedelta(days=1)
        else:
            for marker, offset in cfg["relative_end_day_markers"].items():
                if marker in bridge:
                    end_day, _ = _day_span(current + timedelta(days=offset))
                    break
        cleaned_bridge = bridge
        for marker in list(cfg["relative_end_day_markers"]) + list(cfg["next_day_bridge_markers"]):
            cleaned_bridge = cleaned_bridge.replace(marker, "")
        if not re.fullmatch(cfg["clock_bridge_pattern"], cleaned_bridge):
            continue
        inherited_period = first.get("period")
        if (
            inherited_period in {"晚上", "晚间", "夜间"}
            and not second.get("period")
            and int(first["hour"]) >= 6
            and int(second["hour"]) <= 5
            and end_day == day
        ):
            end_day += timedelta(days=1)
            inherited_period = None
        end = _clock_datetime(end_day, second, inherited_period)
        if start is None or end is None:
            continue
        if end < start and end_day == day:
            end += timedelta(days=1)
        if end > start:
            return start, end
    if not tokens:
        return None
    token = tokens[0]
    start = _clock_datetime(day, token)
    if start is None:
        return None
    window = text[max(0, token["start"] - 6):min(len(text), token["end"] + 6)]
    if any(marker in window for marker in cfg["fuzzy"]):
        half = cfg["fuzzy_clock_window_minutes"]
        return start - timedelta(minutes=half), start + timedelta(minutes=half)
    if token["precision"] == "hour":
        return start, start + timedelta(hours=1)
    return start, start + timedelta(minutes=1)


def _period_range(text: str, day: datetime) -> tuple[datetime, datetime] | None:
    cfg = _cfg()
    for period in cfg["period_names"]:
        if period not in text:
            continue
        normalized_period = cfg["period_alias"].get(period, period)
        span = cfg["period_spans"].get(normalized_period)
        if not span:
            continue
        start_hour, end_hour = span
        start = day.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end = day + timedelta(days=1) if end_hour == 24 else day.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        return start, end
    if cfg["daytime_marker"] in text:
        start_h, end_h = cfg["daytime_span"]
        return day.replace(hour=int(start_h)), day.replace(hour=int(end_h))
    return None


def _explicit_date_range(text: str, dates: list[dict[str, Any]], times: list[dict[str, Any]]) -> tuple[datetime, datetime] | None:
    if len(dates) < 2:
        return None
    bridge_pat = _cfg()["date_range_bridge_pattern"]
    for left, right in zip(dates, dates[1:]):
        bridge = text[left["end"]:right["start"]]
        if not re.search(bridge_pat, bridge):
            continue
        left_times = [item for item in times if left["end"] <= item["start"] < right["start"]]
        right_times = [item for item in times if item["start"] >= right["end"]]
        start = _clock_datetime(left["day"], left_times[-1]) if left_times else left["day"]
        end = _clock_datetime(right["day"], right_times[0]) if right_times else right["day"] + timedelta(days=1)
        if start is not None and end is not None and end > start:
            return start, end
    return None


def _has_date_phrase(text: str) -> bool:
    pattern = _cfg()["has_date_phrase"]
    return bool(pattern and re.search(pattern, text))


def _timestamp_range(start: datetime, end: datetime) -> tuple[float, float]:
    return start.timestamp(), end.timestamp()

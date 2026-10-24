"""Readable representations for monitoring and source-video timestamps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

MONITOR_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
EPOCH_TIME_FLOOR = 946684800


def format_monitor_time(timestamp: float) -> str:
    value = float(timestamp)
    if value < EPOCH_TIME_FLOOR:
        seconds = max(0, int(value))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"Video {hours:02d}:{minutes:02d}:{seconds:02d}"
    return datetime.fromtimestamp(value, MONITOR_TIMEZONE).isoformat(timespec="seconds")

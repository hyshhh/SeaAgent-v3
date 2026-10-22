---
name: time-range
description: "Use whenever the question mentions a time, or deliberately does not."
---

# Sea-Video Time Range

1. Fill the range **only** when the user specifies a time. No time condition means all stored monitoring time; omit `time_range` and never invent a window.
2. Resolve explicit and relative times to ISO-8601 start/end timestamps with a timezone. Use `+08:00` for Asia/Shanghai; keep any timezone the user explicitly gave. Never calculate Unix epoch seconds in the prompt.
3. Resolve relative expressions against the current Asia/Shanghai time: 刚才/刚刚 ≈ last ten minutes; 最近/近/过去 N 单位 ≈ now minus that duration; N 单位前/后 ≈ that instant with a narrow window.
4. Precedence when several readings fit: relative duration → explicit date range → date span (a lone date means that whole day) → clock range → named period → whole day of the date phrase. Use inclusive endpoints; represent a full end date as 23:59:59.999999.
5. Named periods: 凌晨 0-6, 早上 6-9, 上午 9-12, 中午 11-13, 下午 12-18, 傍晚 17-19, 晚上 18-24.
6. Week/month: 周末 is Friday 00:00 to Monday 00:00; 本月/这个月 is the calendar month; 初 = days 1-10, 中/中旬 11-20, 末/底 21 to month end. 半 = 30 minutes, 一刻 = 15.
7. 刚才/上一轮 quotes the previous turn's range verbatim from the conversation. If it is genuinely gone, say so and ask — never substitute a new window silently.
8. If a time expression is ambiguous, ask for clarification instead of omitting the range, because omission means all monitoring time.

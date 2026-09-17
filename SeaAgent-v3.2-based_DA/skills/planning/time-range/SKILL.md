---
name: time-range
description: "Use whenever the question mentions a time, or deliberately does not."
---

# Sea-Video Time Range

1. Fill the range **only** from an explicit expression in the user's own question. No expression means all monitoring time — never invent a default window from the current clock.
2. Resolve relative expressions against the current time, as absolute epoch seconds: 刚才/刚刚 ≈ last ten minutes; 最近/近/过去 N 单位 ≈ now minus that duration; N 单位前/后 ≈ that instant with a narrow window.
3. Precedence when several readings fit: relative duration → explicit date range → date span (a lone date means that whole day) → clock range → named period → whole day of the date phrase.
4. Named periods: 凌晨 0-6, 早上 6-9, 上午 9-12, 中午 11-13, 下午 12-18, 傍晚 17-19, 晚上 18-24.
5. Week/month: 周末 is Friday 00:00 to Monday 00:00; 本月/这个月 is the calendar month; 初 = days 1-10, 中/中旬 11-20, 末/底 21 to month end. 半 = 30 minutes, 一刻 = 15.
6. 刚才/上一轮 quotes the previous turn's range verbatim from the conversation. If it is genuinely gone, say so and ask — never substitute a new window silently.
7. An expression that cannot be resolved is reported as a parse error with an empty range; do not guess.

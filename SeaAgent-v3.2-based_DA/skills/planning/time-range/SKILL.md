---
name: time-range
description: "Use to convert explicit or follow-up time conditions into tool ranges."
---

# Time Scope

1. No time condition: omit `time_range` and query all stored monitoring time. Never invent a window or ask for one just because it was omitted.
2. Convert explicit/relative times to ISO-8601 start/end with timezone; default to Asia/Shanghai (`+08:00`), preserving any timezone given. Do not calculate epoch seconds.
3. Resolve “刚才/刚刚” as about the last 10 minutes unless it refers to a previous QA turn; then reuse that turn's range. “最近/近/过去 N 单位” means now minus N through now; “N 单位前/后” means that instant with a narrow window. Ask only if the reference or time is genuinely ambiguous.
4. Resolve multiple expressions in this order: relative duration → explicit range → date span (a lone date is its full local day) → clock span → named period → full day. Use inclusive endpoints; a full day ends at 23:59:59.999999.
5. Named periods: 凌晨 0–6, 早上 6–9, 上午 9–12, 中午 11–13, 下午 12–18, 傍晚 17–19, 晚上 18–24. 周末是周五 00:00 至周一 00:00；本月是自然月；初/中旬/末为 1–10/11–20/21–月末。“一刻”=15 分钟，“半”=30 分钟。

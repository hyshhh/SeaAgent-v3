---
name: time-range
description: "Convert clear user time conditions into valid tool ranges without guessing."
---

# Time Scope

- No time condition (including “所有时间”): omit `time_range`; the tool searches all stored monitoring time. Time scope is not a separate plan step. Never invent a default window.
- Convert a clear explicit or relative range to a two-element array of ISO-8601 timestamps: `[start, end]`, never an object or JSON-encoded string. Resolve relative terms from the Planner's fixed system-context time anchor and its offset, never model memory; preserve the user's timezone and use `Asia/Shanghai` (`+08:00`) when none is stated. Never generate epoch seconds.
- Resolve relative calendar days arithmetically from the anchor's local date: “前天” is anchor date minus two days, “昨天” minus one day, and “今天” the anchor date. Then encode the resulting full day using the rule below; do not return the date as `{start, end}`.
- Resolve “刚才/刚刚” as the last 10 minutes unless it clearly refers to a prior QA turn; then reuse that turn's range. “最近/过去 N” means now minus N through now; “N 单位前/后” means that instant (`start=end`). Ask only when the reference is genuinely ambiguous.
- A single date means its full local day: start at `00:00:00`, end at `23:59:59.999999` (for example, `2026-10-25` becomes `2026-10-25T00:00:00+08:00` through `2026-10-25T23:59:59.999999+08:00`). Named periods use local time: 凌晨 00–06、早上 06–09、上午 09–12、中午 11–13、下午 12–18、傍晚 17–19、晚上 18–24；周末为周六 00:00 至周一 00:00；本月为自然月；上/中/下旬为 1–10、11–20、21–月末。“一刻”=15 分钟，“半”=30 分钟。
- Use inclusive endpoints; a full day ends at 23:59:59.999999. Reuse prior QA scope only when the follow-up clearly refers to it.

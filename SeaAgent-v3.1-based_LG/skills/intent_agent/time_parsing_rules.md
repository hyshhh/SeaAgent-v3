# IntentAgent · 时间解析规则

## 原则
- 有时间表达才填 `timeRange` / `timeExpression`；没有则二者为 null。
- 相对时间（昨天、上周、今天下午）必须相对 `referenceTime` 换算为 Unix 秒时间戳。
- 无法可靠解析时设置 `timeParseError`，不要瞎猜。

## 推荐工具
- `parseTime(expression)`：解析自然语言时间 → `[startTs, endTs]`
- 词表/时段/检测模式见 `time_parsing.yaml`（代码只加载执行）
- 绝对时间如 `2024-01-01 08:00` 也应走工具或等价规则。

## 输出约束
- `timeRange` 必须是长度为 2 的数字数组，或 null。
- `timeSource` 取 `model` / `rule` / null。

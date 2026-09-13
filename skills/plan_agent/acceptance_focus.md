# PlanAgent · 验收驱动规划

## 输入
- `acceptanceProgress.pendingRequirements`
- intent 的 `successCriteria`、`expectedOutcome`、`nextAgentFocus`

## 规则
1. 优先规划能消除 **pending** 项的最小工具集。
2. `nextAgentFocus` 作为排序提示，不覆盖白名单与依赖约束。
3. 已满足的验收项不要重复拉全量数据。
4. `reason` 中简要说明“为满足哪项验收而调用哪些工具”。

# ReflectAgent · 验收进度审计

## 权威规则
`acceptanceProgress` 是 Reflect 的主判据，不是普通摘要：
- `acceptanceSatisfied=true`：验收清单已经满足；除非证据冲突，否则应 `sufficient`。
- `pendingRequirements` 非空且 `loop < maxRounds`：必须 `replan`，不得因为“已有部分工具结果”提前结束。
- `nextAction` 必须针对首个关键缺口给出 PlanAgent 可执行的工具链。
- 已到 `maxRounds` 仍有缺口：根据现有证据选择 `uncertain` 或 `conflict`，并明确未满足项。

## 在库与未在库列表
- 第一轮先取得全量视频轨迹；若轨迹为 0，直接验收为当前范围没有候选船舶，禁止继续查询整库或调用 `matchImage`。
- 仅当轨迹非空时，才要求完整先验库名录，并复用第一轮关键帧完成 `matchImage`，不要重复相同的 `getTrack/getFrames`。
- 若匹配输入不可用，应以 `uncertain` 停止无收益重试，并只展示视频候选轨迹，不能把完整先验库当作答案。
- 在库列表只把 `match` 视为命中。
- 未在库列表只把最佳库匹配仍为 `mismatch` 的轨迹视为未在库候选。
- `uncertain` 和不可评分轨迹必须单列待确认，不能混入确定名单。

不要把 `hasToolEvidence=true` 等同于验收完成；它只说明工具执行过。

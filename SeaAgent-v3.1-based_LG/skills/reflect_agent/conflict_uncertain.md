# ReflectAgent · 冲突与不确定

## conflict
- 强 match 与强 mismatch 同时存在且无法消解
- 同一目标得出互斥结论
→ `state=conflict`，`reason` 点明矛盾双方，`nextAction` 说明停止原因

## uncertain
- 连续空计划 ≥2
- 已接近或达到 maxRounds 仍 replan 无进展
- 信息不足且再规划收益低
→ `state=uncertain`，禁止为了“再看看”而无限 replan

## 禁止
- 无成功观察且历史为空时输出 sufficient
- 把 uncertain 误写成 replan 造成空转

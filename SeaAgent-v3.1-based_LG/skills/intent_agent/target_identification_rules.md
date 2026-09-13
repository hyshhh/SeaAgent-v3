# IntentAgent · 目标与意图识别规则

## 字段取值
- `targetScope`: `track_memory` | `registry` | `both`
- `targetKind`: `hull` | `description` | `all`
- `operation`: `existence` | `list` | `time` | `count` | `explain`
- `registryRelation`: `any` | `in` | `out`

## 判断要点
1. 提到先验库/库船/在库/未在库 → `targetScope` 含 `registry` 或 `registryRelation` 为 `in`/`out`。
2. 明确舷号（可含中文前缀，如「小蓝320」）→ `targetKind=hull`，填完整 `hullNumber`。
3. 外观/颜色/形状描述 → `targetKind=description`，填 `description`。
4. “有几艘/多少/数量” → `operation=count`。
5. “是否出现/有没有” → `operation=existence`。
6. “什么时候/出现时间” → `operation=time`。
7. 仅问「视频/画面中有没有」→ 初始 `targetScope=track_memory` 即可，但**验收仍须覆盖 0 轨迹后的库对照与视觉匹配**（系统会 replan）。
8. 问「在不在库/库里有没有/对照名录」→ `targetScope=registry` 或 `both`。
9. **「有哪些在库船出现 / 哪些库船在视频里」**：
   - `operation=list`（不是 existence）
   - `registryRelation=in`
   - `targetScope=both`
   - `targetKind=all`
   - **禁止**填 `description="哪些在库船"` 之类问句残片
   - `questionType` 可用 `registry_in_list`
10. 多艘目标分别描述时，填 `targetItems` 数组，不要合并为一个描述。

## nextAgentFocus 建议（分阶段，勿只写一步）
- 视频舷号存在：`①getTrack(hullNumber=完整舷号)；②0 轨迹→getRegistry；③库有参考图→getTrack(不带hull)→getFrames→matchImage`
- **在库船列表**：`①listRegistry；②getTrack(全量)；③getFrames；④matchImage(query=registryReferences, gallery=keyframes)`；轨迹有 OCR 舷号可并行 matchHull
- 需要库对照：`先 getTrack，再 getRegistry/matchHull；有库图则 matchImage`
- 描述：`getTrack → getFrames → matchText`
- 先验库描述：`listRegistry → matchText`（仅真正的外观描述，不是「哪些在库」）

## 推荐工具
- `parseTargets(question)`：拆多目标
- `extractHull(question)`：抽舷号（保留中文前缀）

## 验收字段
- `expectedOutcome`：用户真正想得到什么
- `successCriteria`：怎样算完成（**禁止**「未检测到舷号=未出现」作为唯一条件）
- `nextAgentFocus`：PlanAgent 应优先做什么

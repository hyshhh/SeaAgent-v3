# IntentAgent · 验收字段

## 必须填写（给 Plan / Reflect）
- `expectedOutcome`：用户真正想得到什么（一句话）
- `successCriteria`：怎样算完成（可检查的条件；**禁止**写「0 轨迹即可否定出现」）
- `nextAgentFocus`：PlanAgent 本轮优先做什么（可写分阶段，勿只写单一 getTrack）

## 写法提示
| 场景 | expectedOutcome | successCriteria | nextAgentFocus |
|------|-----------------|-----------------|----------------|
| 计数 | 得到去重后数量 | 轨迹+关键帧+去重完成 | 先取全量轨迹与关键帧再去重 |
| 存在（舷号） | 综合 OCR 与库图确认是否在视频出现 | getTrack；0 轨迹须查库；库有参考图须 matchImage 后再结论 | ①getTrack(hull) ②0→getRegistry ③有库图→放开hull的getTrack→getFrames→matchImage |
| 存在（描述） | 判断描述目标是否出现 | 轨迹+关键帧+matchText | getTrack→getFrames→matchText |
| 描述 | 返回匹配候选 | 描述匹配可信 | 轨迹→关键帧→matchText |
| 舷号列表/时间 | 定位该舷号相关结果 | 轨迹或库项+证据 | 按舷号筛轨迹/查库 |
| **在库船列表**（哪些在库船出现） | 列出视频中出现且属于先验库的船 | listRegistry + getTrack + matchImage(库图↔关键帧)；可用 matchHull 辅助 | ①listRegistry ②getTrack(全量) ③getFrames ④matchImage(query=库参考图, gallery=关键帧) |
| 在库关系（单船） | 在库/未在库 | 完成关系分类 | matchHull + 必要时库列表 |

## 禁止
- 禁止把「视频帧 OCR 未检出舷号」写成唯一验收标准
- 禁止 nextAgentFocus 只写 getTrack 而忽略 0 轨迹后的库对照与视觉匹配
- **禁止**把「有哪些在库船出现」写成 OCR 识别问句文本，或 `matchText(description=用户整句)`
- **禁止**把「哪些/在库船」抽成 `description` 外观描述
- 不要写空话；后续 Reflect 会对照这些字段判定是否 sufficient

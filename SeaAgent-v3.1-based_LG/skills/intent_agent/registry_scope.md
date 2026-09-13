# IntentAgent · 数据库范围识别

## 触发词
“数据库”“数据库中/里/内”“数据仓库”“先验库”“库中/库里”“名录”均表示先验数据库范围。

## 范围边界
1. 用户只问“数据库中有没有/有哪些/多少……”且没有“视频、监控、轨迹、画面、视野”等视频侧限定时：
   - `targetScope=registry`
   - 只解析数据库目标，不得把“有没有、出现、找到”误当作视频检索指令。
2. 同时明确提到数据库与视频侧时，才使用 `targetScope=both`。
3. “有哪些在库船出现在视频中”属于视频与数据库关系查询，仍使用 `both`，不是纯数据库查询。

## 描述清洗
从 `description` 中删除“数据库中/数据库里/先验库中/库中”等范围词，只保留颜色、船型、外观等目标短语。

## 验收字段
- 数据库描述：`listRegistry → matchText(galleryImages=$ref registry.registryReferences)`。
- 数据库舷号：`getRegistry(hullNumber)`。
- 禁止在纯数据库问题中规划 `getTrack`、`getFrames`。

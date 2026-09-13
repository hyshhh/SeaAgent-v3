# PlanAgent · 失败修复与补洞

## 何时使用
- Reflect `evidenceGap` / `nextAction` 指出缺证据
- 上轮工具失败、空结果或大量 skip
- 上轮走了默认兜底计划

## 做法
1. 空轨迹（视频）：若问题只需视频存在结论，可交给 Reflect；若需库内外/身份 → 改规划 `getRegistry` 或 `listRegistry`+`matchHull`。
2. 有轨迹无帧：补 `getFrames`。
3. 有帧无匹配：补 `matchText`/`matchImage`/`matchHull`。
4. 计数缺去重：补 `dedupTracks`。
5. 不要重复与上轮完全相同且已空结果的调用；换参数或换数据源。
6. 仍无法形成可执行计划：`submit_observation`，写清 `evidenceGap`。

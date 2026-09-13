# PlanAgent · 工具规划规则

## 允许工具
`getTrack` `getFrames` `getClip` `getRegistry` `listRegistry`
`matchHull` `matchText` `matchImage` `verifyTarget` `showEvidence` `dedupTracks`

## 通用原则
1. 只规划本轮需要的最小工具集。
2. 依赖用 `{"$ref": "callId.field"}`。
3. 未拿到轨迹就不要 `getFrames`；未拿到关键帧/库图就不要 `matchText/matchImage`。
4. 优先完成 Reflect `nextAction`。

## 典型链路
- 视频存在/列表：`getTrack`（必要时再 `getFrames`）
- 描述检索：`getTrack` → `getFrames` → `matchText`
- 舷号（视频 OCR）：`getTrack(hullNumber)` → 有轨迹再 `getFrames`
- 舷号（先验库）：`getRegistry(hullNumber)`
- **视觉补洞（OCR 未命中）**：
  `getRegistry` → `getTrack`(**不要** hullNumber) → `getFrames` →
  `matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)`
- **在库/未在库船列表**：
  第一轮 `getTrack`(全量) → 仅在 `trackIds` 非空时 `getFrames`；
  若全量轨迹为 0，直接结束；否则第二轮复用已有 `frames`，执行 `listRegistry` →
  `matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)`
  （轨迹已有 OCR 舷号时可加 `matchHull`；**禁止** `matchText(description=用户问句)`）
- 计数：`getTrack` → `getFrames` → `dedupTracks`
- 先验库描述：`listRegistry` → `matchText(galleryImages=$ref registry.registryReferences)`（仅真正外观描述）

## 再规划时
- 指定舷号查询中，上轮按舷号过滤的轨迹为 0 且要求查库 → `getRegistry`，不要重复完全相同的 getTrack。
- 上轮已 getRegistry 且要求视觉匹配 → 必须带 **matchImage**，getTrack 放开舷号过滤。
- 上轮全量轨迹为 0 → 直接结束，不查整库、不调用 matchImage；上轮某步因依赖空被 skip → 不要再规划依赖该空结果的步骤，除非先补上游。
- 上轮误用 matchText(问句) 做在库列表 → 改成 listRegistry + matchImage。

## 参数要点
- `getTrack`: `timeRange` / `hullNumber` / `offset` / `limit`
- `getFrames`: `trackIds`（$ref）
- `getRegistry`: `hullNumber`
- `matchImage`: `queryImages` + `galleryImages` + `topK`（一侧库图、一侧关键帧）
- 广泛多库、多轨迹链在轨迹非空时复用第一轮结果执行 `listRegistry → matchImage`，使用系统提供的 `broadMatchTopK`，不要复用普通 `queryTopK`；其中 `0` 表示不截断。
- `matchText`: `description` + `galleryImages`（description 必须是外观短语，不是「哪些在库船」）
- `matchHull`: `hullNumberArray`

## 输出
必须通过 `submit_plan` 提交 calls；禁止只输出正文。

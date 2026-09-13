# ReflectAgent · 再规划指示

## 何时 replan
仍缺关键证据，且未触达冲突/轮次上限/连续空计划。

## 常见缺口 → nextAction 示例
- 视频轨迹 0、未查库 → `getRegistry(hullNumber=…)`
- 已查库有参考图、未做视觉匹配 →  
  `getRegistry → getTrack(不带hullNumber) → getFrames → matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)`
- **在库船列表未完成** →  
  `listRegistry → getTrack(不带hullNumber) → getFrames → matchImage(queryImages=$ref registry.registryReferences, galleryImages=$ref frames.keyframes)`
- 误用 matchText(用户整句) 做在库列表 → 同上，改 matchImage
- 有轨迹无关键帧 → `补 getFrames($ref trackIds)`
- 有帧无描述匹配 → `补 matchText(description, galleryImages=$ref frames.keyframes)`（description 须为外观短语）
- 在库/未在库未做精确舷号匹配 → `补 matchHull(hullNumberArray)`
- 计数未去重 → `补 dedupTracks`

## nextAction 写法
- 给 **PlanAgent** 可执行的下一步，点名工具与关键参数字段
- 视觉匹配必须写清 **query=库参考图、gallery=视频关键帧**，且 getTrack **不要**再带 hullNumber（OCR 未命中时要放开扫）
- 避免空泛的「继续查」

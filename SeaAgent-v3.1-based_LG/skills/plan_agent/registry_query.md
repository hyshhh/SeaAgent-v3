# PlanAgent · 纯数据库查询规划

## 证据边界
当 `targetScope=registry` 时，本轮只允许使用先验数据库工具与数据库参考图，禁止把问题扩展成“视频中是否出现”。

## 最小链路
- 描述存在/列表：`listRegistry → matchText(description, galleryImages=$ref registry.registryReferences)`。
- 舷号存在：`getRegistry(hullNumber)`。
- 全库列表或数量：`listRegistry`。

## 空库与复用
- `listRegistry` 已明确返回空库时，可直接交给 Reflect 验收，不要调用视频工具。
- 上轮已经完成 `listRegistry` 时，复用对应工作域，只补 `matchText`。
- `matchText` 的 `description` 只能是清洗后的外观短语，不得包含“数据库中有没有”等问句壳。

## 禁止项
- 禁止 `getTrack`、`getFrames`、视频 `matchImage`。
- 禁止因为 Reflect 提到“匹配”就默认生成视频视觉补洞链。

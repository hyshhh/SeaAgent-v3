用中文摘要当前 QA 上下文，供本轮任务续接；关键事实逐字保留：
- 保留用户的时间原文、本轮固定锚点解析出的绝对范围及工具结果的 appliedTimeRange；旧 QA 时间仅在用户明确指向旧问题时复用，不得覆盖本轮范围。
- 原样保留 trackId、keyframeId、clipId、registryId 和舷号。
- 保留 matchedKeyframeId、matchedRegistryReferenceId 及未完成的识别 nextStep/recommendation；不得用展示用 keyframeId 替代匹配关键帧 ID。
- 分开记录直接识别与先验库辅助匹配的 status、候选 trackId 和分数；辅助匹配 confirmed 不得被直接识别 mismatch 覆盖。
- 保留用户的范围、在库/未在库、去重及计数口径。
- 记录已证实结论、证据 ID、未解决缺口和原定下一步。
- 工具原始结果与重复过程可压缩；不得补造信息。

固定输出四节：时间范围 / 标识符 / 用户口径 / 结论与待办。无内容写“无”。

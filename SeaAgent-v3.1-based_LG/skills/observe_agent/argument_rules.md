# 依赖解析与必填参数（Observe）

## 必填参数
见 `agent/plan_executor.py` 的 `_REQUIRED_ARGUMENTS`（工具必填参数的唯一事实源）。

## showEvidence
`keyframeIds` / `shipSegmentIds` / `registryReferenceIds` 至少一个非空。

## $ref
- 形式：`{"$ref":"callId.field"}`
- 支持 `$map` / `$compact` / `$list` / `$default`
- 依赖根 callId 不在 scope 或 ok=false → 跳过并记录原因

## 条件
`condition: {ref, equals|in}` 不满足则 skip。

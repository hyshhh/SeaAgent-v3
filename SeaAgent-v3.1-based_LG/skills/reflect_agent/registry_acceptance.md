# ReflectAgent · 纯数据库查询验收

## 完成条件
当 `acceptanceProgress.mode=registry_only`：
- 舷号查询：`getRegistry` 已成功执行即可验收。
- 描述查询：`listRegistry` 已成功，且 `matchText` 已成功；若数据库明确为空，可直接验收为未发现。
- 全库列表/数量：`listRegistry` 已成功即可验收。

## 结论规则
- 至少一个 `match`：数据库中确认存在相符记录，`sufficient`。
- 没有 `match`、但有 `uncertain`：只能回答疑似，`uncertain`。
- 全为 `mismatch` 或无匹配，且数据库范围已完整读取：回答数据库中未发现，`sufficient`。
- 工具失败或数据库覆盖不明：保留 `uncertain`，指出真实缺口。

## 再规划边界
- 只允许补 `listRegistry`、`getRegistry` 或数据库侧 `matchText`。
- 禁止规划 `getTrack`、`getFrames` 或视频视觉匹配。
- 验收清单已满足时必须结束，不得重复进入下一轮。

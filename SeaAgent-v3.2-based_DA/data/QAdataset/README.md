# SeaQA 问答评测集

本目录用于维护 SeaQA 问答数据集、内部数据构建脚本和面向用户的评测工具，与线上服务 Harness 的运行逻辑相互隔离。

目录职责如下：

- `evaluator/`：面向用户的评测程序、使用说明和规则自测。
- `dataset/`：根据服务器端轨迹、关键帧和先验库生成的真实数据集及配套资源。
- `collector/`：内部维护脚本，只负责从服务器端数据构建数据集，不属于用户评测流程。
- `config.yaml`：数据集生成的数量、类别分布、随机种子和路径等配置。

真实的 `questions.jsonl`、查询图片、特征缓存和 `manifest.json` 只在服务器端生成，并通过 `.gitignore` 排除，不写入 Git。

用户评测入口见 [`evaluator/README.md`](evaluator/README.md)。

## 数据集范围

SeaQA 包含六类问题：

- 识别类：
  - `hull_number_lookup`：按舷号查询；
  - `description_lookup`：按外观描述查询；
  - `image_lookup`：按输入图片查询。
- 统计类：
  - `total_count`：统计全部实体；
  - `registered_count`：统计在库实体；
  - `unregistered_count`：统计不在库实体。

每题的真值以实体集合表示，并保留轨迹、关键帧、图片或先验库来源。无法可靠确定的关系会标记为不可判定，并与可判定问题分开评测。

## 当前评分口径

- 识别任务按预测实体集合与真值实体集合是否完全一致计算 Exact Match。
- 统计任务分别计算两项指标：
  - `map`：只有预测实体集合与真值实体集合完全一致时计 1，类别指标取各题平均值；
  - `count_accuracy`：只有整数计数答案与真值数量一致时计 1，类别指标取各题平均值。
- 真值为不可判定的问题，只有预测返回 `unresolved` 才算正确，并且不进入可判定问题的 EM 或 mAP 分母。

当前评分只比较实体集合和整数答案。实体顺序、重复实体 ID 和置信度不参与评分；预测引用的时间范围和关键帧证据尚未自动核验。

## 构建与评测边界

`collector/collect.py` 会读取服务器端独有轨迹、关键帧和先验库数据，并使用缓存验证结果确定性地重新生成 100 条问题。该脚本仅供数据维护，不是用户评测入口。

用户只需准备预测文件，再运行 `evaluator/evaluate.py`。具体数据格式、运行命令、指标定义和自测命令见 [`evaluator/README.md`](evaluator/README.md)。

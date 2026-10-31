# SeaQA 问答评测集

本目录用于维护 SeaQA 问答数据集、内部数据构建脚本和动态评测工具。评测程序通过 SeaAgent 的正式问答 API 逐题运行，数据集生成逻辑与问答 Harness 仍然相互独立。

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

## 动态评测方式

评测器逐题调用正在运行的 SeaAgent：

1. 不得把 `ground_truth`、`category` 或 `provenance` 发送给 Agent；
2. 文本题只输入问题本身，图片题把 `images/` 中的真实图片作为附件发送；
3. 每题使用独立会话，保存 Agent 最终答案、状态、轨迹和证据 ID；
4. 生成 CSV 结果表，每个问题连续输出两行：第一行是 `prediction`，第二行是 `truth`；
5. 人工对比两行答案、实体和证据，在 `manual_result`、`manual_notes` 中记录结论。

评测器不再读取 `predictions.jsonl`，也不再自动计算 EM、mAP 或计数准确率。

## 构建与评测边界

`collector/collect.py` 会读取服务器端独有轨迹、关键帧和先验库数据，并使用缓存验证结果确定性地重新生成 100 条问题。该脚本仅供数据维护，不是用户评测入口。

用户需要先启动 SeaAgent 服务，再运行 `evaluator/evaluate.py`。具体 API 配置、运行命令、结果表字段和自测命令见 [`evaluator/README.md`](evaluator/README.md)。

# ETA 训练工程

本目录是 SeaAgent 的隔离式 ETA 训练工程。原始轨迹、关键帧和向量索引
来自 `SeaAgent-v3.2-based_DA`；所有新增代码与运行产物只写入 `ETA/`。

## 严格输入输出约定

模型只接收视觉 embedding：

```text
candidate        [2048]       当前帧 embedding
history          [3, 2048]    当前 KEP 的三帧 embedding
history_mask     [3]          三个历史槽位均有效
        ↓
ETA
        ↓
AH 分类结果
CH 分类结果
```

模型不接收质量分、时间、轨迹编号、来源路径、舷号或文本描述。模型只有
AH 和 CH 两个输出；Stop 是外层运行时的处理策略，不是第三个模型输出。

## 数据规则

1. 每条轨迹必须已有恰好三帧 KEP，ETA 才会开始。
2. 候选帧必须晚于 KEP 中最新的关键帧。
3. 候选轨迹框按一秒间隔采样，并从源视频解码、评分、调用本地 Qwen
   embedding 服务生成 2048 维向量。
4. 每个样本的 `history` 始终包含三帧，`history_mask` 全为真。
5. 模型按时间顺序处理候选。某个候选实际写入 KEP 后，如果更新后的 KEP
   达到证据完备条件，则该样本标记 `ch_label=1`，并立即停止该轨迹，
   后续观测不再构建样本。
6. 从未达到完备条件的轨迹全部保持 `ch_label=0`。

当前清洗结果：

| 数据切分 | 样本数 | 轨迹数 | 来源视频数 |
|---|---:|---:|---:|
| 训练集 | 697 | 116 | 31 |
| 验证集 | 86 | 16 | 7 |
| 测试集 | 166 | 31 | 7 |

163 条可用轨迹中，56 条达到证据完备条件，107 条未达到条件。

## 数据集格式

每个 split 的 NPZ 文件只包含五个数组：

```text
candidate
history
history_mask
ah_label
ch_label
```

NPZ 中不保存 `sample_id`、`track_id`、`source_id`、`frame_id`、时间戳或
其他逐样本元数据。

## 目录结构

```text
ETA/
  configs/
    default.yaml
    smoke.yaml
    experiments/
      ablations.yaml
      tuning.yaml
  src/eta/
    dataset/       轨迹候选提取、embedding 与五数组数据集构建
    training/      双头训练、类别权重、早停和 checkpoint 选择
    evaluation/    指标、阈值选择、报告与复现比较
    research/      消融和调参实验编排
    model.py       ETA 交叉注意力编码器及 AH/CH 双头
    pool.py        KEP 更新与 Stop 策略
  tests/
  docs/
    data_card.md
    labeling_protocol.md
  reports/
  data/        运行时生成，Git 忽略
  artifacts/   运行时生成，Git 忽略
```

## 构建数据

```bash
cd SeaAgent-v3.2-based_DA/ETA
export PYTHONPATH="$PWD/src"
export ETA_EMBEDDING_API_KEY=abc123
python -m eta.cli build-data --config configs/default.yaml
```

候选 embedding 会缓存在 `artifacts/embedding_cache/`。缓存键包含源视频
路径、文件大小、修改时间、候选时间戳和边界框。

## 训练与评测

```bash
python -m eta.cli train --config configs/default.yaml --run-name eta
python -m eta.cli evaluate --config configs/default.yaml \
  --run-dir artifacts/eta-...
python -m eta.cli publish --config configs/default.yaml \
  --run-dir artifacts/eta-...
```

AH 和 CH 的决策阈值只在验证集选择，测试集用于固定评估。模型按验证集
双头平均准确率选择 checkpoint。

## 消融与调参

```bash
python -m eta.cli research --config configs/default.yaml
```

实验包含：

- 6 个结构消融组；
- 6 个学习率、模型规模、dropout、权重衰减和类别权重调参组；
- 统一使用同一份五数组数据和同一 source-group 切分；
- 最终按验证集双头平均准确率选择最优模型，同分时选择更低的 batch-1
  P50 时延；
- 表格报告测试集 AH/CH 准确率和单次 P50 推理时延；召回率与 F1 保存在
  实验 JSON 中。

结果表固定四列：

```text
实验设置 | 消融组/调参组 | 双头精度 | 推理速度（单次ms）
```

消融组定义在表尾注释。

发布结果包含：

- `reports/experiments.md` 和 `reports/experiments.csv`：全部消融与调参结果；
- `reports/best_run.md`、`reports/best_run.csv` 和 `reports/best_run.json`：
  最终模型的测试精度、验证精度与推理时延；
- `reports/dataset_summary.json`：最终数据切分摘要；
- `reports/latest.json`：完整机器可读实验记录；
- `reports/reproducibility.json`：相同配置独立复现后的指标比较。

最终权重统一保存在 `artifacts/best_model/best.pt`。其他实验权重和训练
缓存不属于发布结果。

## 验证

```bash
python -m pytest -q
python -m ruff check .
python -m compileall -q src tests
```

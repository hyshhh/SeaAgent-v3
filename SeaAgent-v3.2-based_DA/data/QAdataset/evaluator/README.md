# SeaQA 评测器

评测器不依赖第三方运行库。它读取服务器端真实数据集和用户提交的预测文件，按固定 SeaQA 规则输出类别汇总及逐题评分。

## 运行环境

以下命令应在 `SeaAgent-v3.2-based_DA` 项目根目录执行：

```text
/home/lab-203/SeaAgent-v3/SeaAgent-v3.2-based_DA
```

评测入口：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/evaluation.json
```

省略 `--output` 时，评测结果直接输出到标准输出。

`collector/collect.py` 仅用于服务器端数据维护，不参与用户评测。

## 数据集输入

`--dataset` 指向服务器端文件：

```text
data/QAdataset/dataset/questions.jsonl
```

每行一条问题，主要字段包括：

- `id`：稳定的问题标识；
- `category`：六类 SeaQA 问题之一；
- `question`：问题文本；
- `time_range`：查询时间范围；
- `inputs`：舷号、描述或图片路径；
- `ground_truth.entities`：目标实体集合；
- `ground_truth.unresolved_entities`：不可确定的实体关系；
- `ground_truth.answerable`：真值是否可判定；
- `ground_truth.count`：统计问题的真值数量；
- `provenance`：轨迹、关键帧、图片和先验库证据。

## 预测格式

`--predictions` 使用 JSON Lines 格式，每行一个 JSON 对象。预测对象应根据问题类别提供 `id`、`status`、`answer` 和 `entities`。

普通可判定问题示例：

```json
{
  "id": "SeaQA-0001",
  "status": "answered",
  "answer": "appears",
  "entities": [
    {
      "entity_id": "entity-000002",
      "confidence": 0.91
    }
  ]
}
```

统计问题必须在 `answer` 中返回整数，例如：

```json
{
  "id": "SeaQA-0040",
  "status": "answered",
  "answer": 12,
  "entities": [
    {
      "entity_id": "entity-000002"
    },
    {
      "entity_id": "entity-000007"
    }
  ]
}
```

无法确定的问题示例：

```json
{
  "id": "SeaQA-0012",
  "status": "unresolved",
  "entities": []
}
```

字段含义：

- `id`：必须与问题记录中的 `id` 对应；
- `status`：可取 `answered`、`unresolved` 或 `error`；
- `answer`：统计问题必须为整数；识别问题不依据该字段评分；
- `entities`：预测实体列表，每项至少包含 `entity_id`；
- `confidence`：可选，当前不参与评分。

缺少对应预测时，评测器按 `status=error` 处理。当前评测器没有单独的失败弃权机制，仍会按实体集合和答案字段参与评分，因此提交方应为每题提供完整预测。

## 评分规则

### 识别问题

适用类别：

- `hull_number_lookup`
- `description_lookup`
- `image_lookup`

评测器比较真值和预测实体 ID 的集合：

- 两个集合完全一致时，`em=1`；
- 否则 `em=0`；
- 类别 `em` 是全部可判定问题 EM 的平均值。

实体顺序、重复实体和置信度不影响结果。

### 统计问题

适用类别：

- `total_count`
- `registered_count`
- `unregistered_count`

统计问题分别计算两项指标：

- `ap`：预测实体集合与真值实体集合完全一致时为 1，否则为 0；
- `map`：全部可判定统计问题 `ap` 的平均值；
- `count_accuracy`：预测整数 `answer` 等于真值 `count` 时为 1，否则为 0；
- 类别 `count_accuracy` 是全部可判定统计问题计数正确率的平均值。

`map` 和 `count_accuracy` 相互独立。例如实体集合完全正确但数字写错时，`map=1`、`count_accuracy=0`；数字正确但实体集合不完整时，`map=0`、`count_accuracy=1`。

### 不可判定问题

当 `ground_truth.answerable=false` 时：

- 预测 `status=unresolved` 时，本题正确；
- 其他状态均视为错误；
- 该题不进入可判定问题的 EM、mAP 或计数准确率分母；
- 类别中的 `unresolved_accuracy` 单独报告不可判定问题的正确率。

## 输出内容

评测结果 JSON 包含：

- `questions`：数据集问题数；
- `predictions`：预测记录数；
- `missing_predictions`：缺失预测数量；
- `categories`：按问题类别汇总的指标；
- `rows`：逐题评分明细，便于定位错误。

每个类别的汇总包含：

- `count`：该类问题总数；
- `scoreable`：可判定问题数；
- `unresolved`：不可判定问题数；
- 识别类问题的 `em`；
- 统计类问题的 `map` 和 `count_accuracy`；
- 有不可判定问题时的 `unresolved_accuracy`。

逐题结果包含 `id`、`category`、真值和预测状态，以及该题适用的 `correct`、`em`、`ap` 或 `count_accuracy` 字段。

## 自测

在项目根目录执行：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/test_evaluate.py
```

自测覆盖以下规则：

- 识别问题使用实体集合 EM；
- 统计问题的 mAP 要求实体集合完全一致；
- 统计问题的计数准确率与 mAP 分开计算；
- 不可判定问题必须返回 `unresolved`。

## 当前评分边界

评测器目前只核对问题 `id`、实体集合、不可判定状态和统计答案。它尚未自动验证：

- 预测声明的时间范围是否与问题一致；
- 预测引用的关键帧是否真实存在；
- 证据是否确实支持所报实体。

这些内容需要在后续评测扩展中增加校验。

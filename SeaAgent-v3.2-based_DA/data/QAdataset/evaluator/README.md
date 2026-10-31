# SeaQA 动态评测

评测器逐题调用正在运行的 SeaAgent，并把 Agent 输出与数据集真值写入同一张 CSV 结果表，供人工核对。评测器不再读取预测文件，也不自动计算 EM、mAP 或其他分数。

## 运行条件

以下命令应在 `SeaAgent-v3.2-based_DA` 项目根目录执行：

```text
/home/lab-203/SeaAgent-v3/SeaAgent-v3.2-based_DA
```

运行前必须先启动 SeaAgent 服务。默认 API 地址为：

```text
http://127.0.0.1:8000
```

服务地址和单题超时可在 `data/QAdataset/config.yaml` 的 `evaluation` 段配置：

```yaml
evaluation:
  api_url: http://127.0.0.1:8000
  timeout_seconds: 900
  output_dir: results
```

## 执行评测

运行全部 100 道题：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --output data/QAdataset/results/SeaQA-100.csv \
  --overwrite
```

首次试跑一道题：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --output data/QAdataset/results/SeaQA-smoke.csv \
  --limit 1 \
  --overwrite
```

省略 `--output` 时，结果默认写入：

```text
data/QAdataset/results/SeaQA-100-<时间戳>.csv
```

结果目录已加入 `.gitignore`，不会提交到 Git。

## Agent 输入

评测器逐题读取 `questions.jsonl`，每道题使用独立 `sessionId` 调用：

```text
POST /api/agent/query/stream
```

发给 Agent 的内容只有：

- `question`：问题正文；
- `sessionId`：评测运行标识；
- 图片题额外发送真实图片的 data URL。

不会发送以下评测元数据：

- `category`
- `ground_truth`
- `provenance`

图片题的 `inputs.image_path` 相对于 `dataset/` 解析，例如：

```text
images/query-001.jpg
```

评测器会读取图片文件并作为附件发送，而不是只把路径写进问题文本。

## 结果表

每个问题连续输出两行：

```text
prediction
truth
```

因此 100 道题会生成 200 行数据。两行具有相同的 `id`、`category`、`question` 和 `time_range`，可直接上下对照。

主要字段：

- `record_type`：`prediction` 或 `truth`；
- `success`：Agent 是否成功完成；
- `status`：Agent 运行状态或真值是否可判定；
- `answer`：Agent 最终回答或真值答案；
- `count`：统计题的整数真值；
- `entity_ids`：真值实体 ID；
- `track_ids`：Agent 证据轨迹或真值轨迹；
- `evidence_ids`：Agent 关键帧、片段、先验库证据或真值关键帧；
- `details`：Agent 运行详情或完整真值来源；
- `manual_result`：留空，由人工填写 `正确`、`错误`、`部分正确` 或 `无法判定`；
- `manual_notes`：人工备注；
- `elapsed_seconds`：Agent 单题耗时；
- `error`：Agent 调用或运行错误。

`prediction` 行是 Agent 实际返回内容。`truth` 行是数据库生成的参考真值，包括实体、轨迹、关键帧、来源和不可判定关系。

## 人工核对

人工按 `id` 对照相邻的 `prediction` 与 `truth` 两行，重点核对：

1. 最终答案是否与 `truth.answer` 一致；
2. 统计数字是否与 `truth.count` 一致；
3. Agent 返回的轨迹与真值轨迹是否对应；
4. 关键帧和先验库证据是否支持结论；
5. 真值为不可判定时，Agent 是否明确说明无法确认。

核对结果填写在 `prediction` 行的 `manual_result` 和 `manual_notes`。

## 运行器行为

- 每题独立会话，避免答案或上下文跨题泄漏；
- 按文件顺序串行执行，避免并发请求干扰模型服务；
- 使用流式接口读取执行事件，直到收到 `complete`、`confirm` 或 `error`；
- 单题失败不会中断整批评测，失败信息写入对应行；
- 每完成一题立即刷新 CSV，运行中断时保留已完成结果；
- 如果 Agent 要求人工确认，当前题记为 `awaiting_confirmation`，不会自动批准或拒绝。

## 自测

自测不调用真实模型：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/test_evaluate.py
```

自测覆盖：

- 每题生成 `prediction`、`truth` 两行；
- 真值不会发送给 Agent；
- 图片会转换为 data URL；
- 可以正确读取 Agent 流式接口的终局事件。

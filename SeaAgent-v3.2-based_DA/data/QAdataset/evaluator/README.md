# SeaQA 动态评测

评测器逐题调用正在运行的 SeaAgent，并将 Agent 的回答整理成简单 QA 文件：

```text
ID,answer,question
```

每题一行，不写入类别、状态、证据、轨迹、真值或其他运行信息。

## 运行条件

以下命令应在 `SeaAgent-v3.2-based_DA` 项目根目录执行：

```text
/home/lab-203/SeaAgent-v3/SeaAgent-v3.2-based_DA
```

运行前必须先启动 SeaAgent 服务。默认 API 地址为：

```text
http://127.0.0.1:8000
```

服务地址和单题超时可在 `data/QAdataset/config.yaml` 中配置：

```yaml
evaluation:
  api_url: http://127.0.0.1:8000
  timeout_seconds: 900
  attempts: 2
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

只运行第一题：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --output data/QAdataset/results/SeaQA-smoke.csv \
  --limit 1 \
  --overwrite
```

从已有 QA 文件续跑，只重试空 answer：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --output data/QAdataset/results/SeaQA-100.csv \
  --resume
```

每题默认最多尝试两次。已有的非空 `answer` 在 `--resume` 时直接复用，
不会重复调用 Agent。

省略 `--output` 时，结果默认写入：

```text
data/QAdataset/results/SeaQA-100-<时间戳>.csv
```

结果目录已加入 `.gitignore`，不会提交到 Git。

## Agent 输入

每题使用独立 `sessionId` 调用：

```text
POST /api/agent/query/stream
```

发给 Agent 的内容只有：

- `question`：问题正文；
- `sessionId`：本次评测运行标识；
- 图片题额外发送真实图片的 data URL。

不会发送 `category`、`ground_truth` 或 `provenance`。

图片题的 `inputs.image_path` 相对于 `dataset/` 解析。评测器读取实际图片并作为附件发送。

## 输出格式

CSV 表头固定为：

```text
ID,answer,question
```

字段含义：

- `ID`：与 `questions.jsonl` 对应的稳定问题 ID；
- `answer`：Agent 返回的真实 answer 字段；
- `question`：本次提交给 Agent 的问题正文。

当前 Agent API 将最终回答投影为 `answerText`，因此运行器优先读取 `answer`，没有该字段时读取 `answerText`。其他字段不会写入 QA 文件。

Agent 执行失败、等待补充信息或停滞时，`answer` 可能为空。具体状态和错误只写入运行日志，不污染 QA 文件。

## 人工核对

人工使用相同 `ID` 在 `dataset/questions.jsonl` 中查找真值：

```text
ground_truth.answer
ground_truth.count
ground_truth.entities
ground_truth.unresolved_entities
```

然后对比 QA 文件中的 `answer` 与对应真值。运行器不再自动计算 EM、mAP 或其他分数。

## 运行行为

- 按文件顺序串行执行，避免并发请求干扰模型服务；
- 每题独立会话，避免上下文跨题影响答案；
- 使用流式接口读取执行事件，直到 `complete`、`confirm` 或 `error`；
- 每完成一题立即刷新 CSV，中断时保留已完成结果；
- 单题失败不会中断整批评测；
- 空 answer 会在同一题内有限重试，`--resume` 可继续补齐未完成题。

## 自测

自测不调用真实模型：

```bash
/home/lab-203/anaconda3/bin/conda run --no-capture-output -n seaagent \
python data/QAdataset/evaluator/test_evaluate.py
```

自测覆盖：

- CSV 表头严格为 `ID,answer,question`；
- 每题只写一行；
- 只保存 Agent answer；
- 失败题会重试并只保留最终成功答案；
- `--resume` 会复用已有答案并只补跑空行；
- 不把真值发送给 Agent；
- 图片会转换为 data URL；
- 可以读取 Agent 流式接口的终局事件。

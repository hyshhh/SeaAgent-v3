# SeaAgent 部署说明
## 环境
- Python 3.10 / 3.11
- 系统需安装 `ffmpeg`（转码与目标船片段）
- `Qwen3-VL-4B`：识别 + 四 Agent 共用同一兼容接口
- `Qwen3-VL-Embedding-2B`：独立向量接口，建议另卡部署=
## 安装
```bash
pip install -e .
ffmpeg -version
python -c "import cv2, fastapi, faiss; print('ok')"
```
## 模型服务
识别、Intent / Plan / Observe / Reflect 共用一个 LLM 服务：

```bash
vllm serve Qwen/Qwen3-VL-4B-AWQ \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --api-key abc123 \
  --port 7890 \
  --max-model-len 32768
```
向量服务：

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --served-model-name Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --api-key abc123 \
  --port 7891
```
`config/app.yaml`：

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"

embedding:
  model: "Qwen/Qwen3-VL-Embedding-2B"
  api_key: "abc123"
  base_url: "http://localhost:7891/v1"
  timeout_seconds: 60
  dimension: 2048
  normalize: true
```

应用通过 `/v1/embeddings` 取向量。关键帧与先验库只存图像特征；用户描述在查询时即时编码。

## 检测器

`config/yolo.yaml` 默认：

```yaml
yolo:
  model: yolov8n.pt
  device: ""
  confidence: 0.5
  detect_every_n_frames: 0.2
  iou: 0.5
  classes: [8]
  tracker: bytetrack
  tracker_params:
    track_high_thresh: 0.5
    track_low_thresh: 0.2
    new_track_thresh: 0.6
    track_buffer: 90
    match_thresh: 0.8
    fuse_score: true
  appearance_tracking:
    enabled: false
    appearance_thresh: 0.8
    proximity_thresh: 0.5
    model: auto
```

自训练船舶模型时改 `model` / `classes`。检测只出船舶区域，不评舷号质量。`tracker_params.track_high_thresh` 是进入主要关联的分数下限，`tracker_params.track_low_thresh` 是第二阶段补充关联的下限，`confidence` 决定正式输出并写入轨迹记忆的最低分数。低于 `confidence` 的检测仅参与关联，不会写入轨迹记忆。开启 `appearance_tracking.enabled` 后会切换到支持外观重识别的跟踪器，以提高遮挡和交叉场景下的轨迹身份稳定性。

## 存储

首次启动自动创建，无需建表：

```text
data/memory/*.csv
data/registry/*.csv
data/memory/keyframes
data/memory/trajectories
data/memory/clips
data/registry/reference_images
data/indexes
```

先验库变更会重建库索引；关键帧索引随正式池增删向量。

## 启动

网页：

```bash
python -m web.app
# 或
seaagent
```

默认 `http://127.0.0.1:8000`。

问答链路：LangGraph 四 Agent（Intent→Plan→Observe→Reflect），角色规则在 `skills/`，业务工具在 `tools/`。

视频：

```bash
python -m pipeline.cli data/videos/example.mp4 --demo --output output/result.mp4
# 或
seaagent-pipeline data/videos/example.mp4 --demo --output output/result.mp4
```

常用参数：`--conf`、`--iou`、`--detect-every`、`--target-fps`、`--max-frames`、`--device`、`--yolo-model`、`--display`、`--no-output`。

## 进程与并发

```text
进程一：Qwen3-VL-4B（识别 + 四 Agent）
进程二：Qwen3-VL-Embedding-2B
进程三：SeaAgent 网页 / 单视频流水线
```

默认 `pipeline.max_parallel_pipelines: 1`。同一任务共用一段视频、一套轨迹记忆与两类特征索引，勿并行多路写这些共享状态。

## 验证

```bash
python -m compileall -q agent config memory pipeline services tools vector_store web
python -c "from config import load_config; print(load_config()['app']['name'])"
python -c "from web.app import app; print(len(app.routes))"
python -c "from agent import AgentController; print(AgentController)"
```

向量服务就绪后再测：

```bash
python -c "from services import QwenMultimodalEmbedder; print(QwenMultimodalEmbedder().dimension)"
```

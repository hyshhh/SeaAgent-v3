# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的单主智能体 Harness。Deep Agents 负责运行时编排，Skills 负责渐进式披露领域规则，LangChain Middleware 负责上下文压缩、调用限制与重试，工具服务访问会话、轨迹和视频证据三层记忆。

## 组件

- `harness/`：Deep Agents runtime、配置驱动工具、模型和官方 middleware。
- `skills/`：查询、证据、先验库、去重、记忆和回答规范。
- `memory/`：会话、轨迹、关键帧和证据持久化。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `pipeline/`：检测、跟踪、关键帧与视频片段生成。
- `web/`：QA 页面、NDJSON 事件流和证据接口。

## 安装与启动

```powershell
py -m pip install -e ".[dev]"
py -m web.app
# 浏览器访问 http://127.0.0.1:8000
```

视频流水线：

```powershell
seaagent-pipeline data/videos/example.mp4 --demo --output output/result.mp4
```

默认依赖包含 LangChain、Deep Agents、OpenAI-compatible Chat Model 和 SQLite checkpoint。需要 FAISS 或 Ultralytics 时安装 `.[models]`。模型地址、工具映射、调用限制和路径均来自 `config/`，业务行为由 Skills 与工具结果驱动。

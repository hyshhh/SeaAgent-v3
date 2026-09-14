# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的单主智能体 Harness。Deep Agents 负责运行时编排，Skills 负责渐进式披露领域规则，LangChain Middleware 负责上下文压缩、工具调用上限、重试与回答收尾前的证据守卫，工具服务访问会话、轨迹和视频证据三层记忆。

## 组件

- `harness/`：Deep Agents runtime、配置驱动工具、模型、官方 middleware 与收尾守卫（`wrapup.py`）。
- `skills/`：查询、证据、先验库、去重、记忆、回答规范与收尾规范。模型按 description 判断是否读取正文，正文通过 `read_file` 打开 `/skills/<name>/SKILL.md`。
- `memory/`：会话、轨迹、关键帧和证据持久化。会话以 `session_id` 为键，逐轮存档问答，同一会话复用同一个 thread_id 续接检查点。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `pipeline/`：检测、跟踪、关键帧与视频片段生成。
- `web/`：QA 页面、会话记录侧栏、NDJSON 事件流和证据接口。

## 问答链路要点

- **技能渐进式披露**：技能名与 description 随系统提示词注入（每一轮都在）；`read_file`/`ls`/`glob`/`grep` 由权限规则收敛到 `/skills` 目录内，写入与 shell 工具仍然禁用。整段会话还没读过任何技能正文时，`SkillDisclosureMiddleware` 会在模型第一次决策前把「先读正文再动手」并进 system 消息提醒一次。续接会话时框架不再回写技能回执，运行时会从检查点补回，保证事件流每一轮都能看到实际注入的技能目录。
- **收尾与证据**：`skills/finalize` 规定收尾动作，`EvidenceWrapUpMiddleware` 在模型给出最终回答却没调用证据工具时提醒一次，两者合起来保证前端证据面板有内容。
- **轮次不设上限**：由模型自己判断何时答完。工具调用仍留 run/thread 两级预算，并且带失控保护——工具预算用完后框架只会驳回后续调用，模型若继续硬调，连续失败到 `stall_guard_consecutive_errors` 次即按现有结果收尾，不会一直刷同一条错误。

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

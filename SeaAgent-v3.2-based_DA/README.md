# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的单主智能体 Harness。Deep Agents 负责运行时编排，Skills 负责渐进式披露领域规则，LangChain Middleware 负责上下文压缩、工具调用上限、重试与回答收尾前的证据守卫，工具服务访问会话、轨迹和视频证据三层记忆。

## 组件

- `harness/`：Deep Agents runtime、配置驱动工具、模型、官方 middleware 与收尾守卫（`wrapup.py`）。
- `skills/`：按组存放，组目录是容器、其子目录才是技能（`skills/<组>/<技能>/SKILL.md`）——`planning`（规划与记忆规范，主智能体专用）、`track`（查询与去重）、`registry`（先验库）、`visual`（视觉证据）、`answer`（回答与收尾）。模型按 description 判断是否读取正文，正文通过 `read_file` 打开对应 SKILL.md；主从协同时每个智能体只拿到自己那几组。
- `memory/`：会话、轨迹、关键帧和证据持久化。会话以 `session_id` 为键，逐轮存档问答，同一会话复用同一个 thread_id 续接检查点。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `pipeline/`：检测、跟踪、关键帧与视频片段生成。
- `web/`：QA 页面（对话 / 轨迹两个子页，左侧会话记录、右侧运行状态与证据）、NDJSON 事件流和证据接口。轨迹子页把一轮里的助手决策、工具调用（含参数与结果）与技能加载逐行列出。

## 问答链路要点

- **主从协同（可开关）**：`harness.subagents_enabled` 打开后，主智能体改读 `harness/planner.md`，只做意图识别、时间范围解析、计划与汇总，领域工具收窄到 `config/subagents.yaml` 的 `master_tools`（默认只留 `show_evidence`）；三个从智能体按数据范围分工——`track_scout`（轨迹筛查与去重）、`registry_checker`（先验库核验）、`visual_prover`（图像与视觉取证），各自只拿自己那几个工具与技能，返回固定格式的紧凑结果。关掉开关即退回单智能体（官方语义：禁用 general-purpose 且不传 subagents ⇒ 根本没有 `task` 工具）。
- **技能渐进式披露**：技能名与 description 随系统提示词注入（每一轮都在）；`read_file`/`ls`/`glob`/`grep` 由权限规则收敛到 `/skills` 目录内，写入与 shell 工具仍然禁用。整段会话还没读过任何技能正文时，`SkillDisclosureMiddleware` 会在模型第一次决策前把「先读正文再动手」并进 system 消息提醒一次。续接会话时框架不再回写技能回执，运行时会从检查点补回，保证事件流每一轮都能看到实际注入的技能目录。
- **收尾与证据**：`skills/finalize` 规定收尾动作，`EvidenceWrapUpMiddleware` 在模型给出最终回答却没调用证据工具时提醒一次，两者合起来保证前端证据面板有内容。⚠️ `show_evidence` 必须留在主智能体：从智能体的内部调用不进主事件流，下放它会让证据面板永远为空。
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

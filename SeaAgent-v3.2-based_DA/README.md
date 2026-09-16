# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的三阶段协同 Harness：**规划 → 执行 → 反思**各挂一个从智能体，主智能体只规划、委派与落证据。Deep Agents 负责运行时编排（含委派与待办清单），Skills 负责渐进式披露各阶段的领域规则，LangChain Middleware 负责上下文压缩、工具调用上限、重试与回答收尾前的证据守卫，工具服务访问会话、轨迹和视频证据三层记忆。编排本身不做任何硬编码判定——唯一由代码承担的是**证据富化**（把散在各次工具结果里的证据 ID 收齐，供前端面板一次渲染）。

## 组件

- `harness/`：Deep Agents runtime、配置驱动工具、模型、官方 middleware 与收尾守卫（`wrapup.py`）。
- `skills/`：按组存放，组目录是容器、其子目录才是技能（`skills/<组>/<技能>/SKILL.md`）——`coordination`（协调规范：规划与记忆，主智能体专用）、`track`（查询与去重）、`registry`（先验库）、`visual`（视觉证据）、`answer`（回答与收尾）。模型按 description 判断是否读取正文，正文通过 `read_file` 打开对应 SKILL.md；主从协同时每个智能体只挂自己那几组，**读取权限也按组收口**，越界读取直接拒绝。
- `memory/`：会话、轨迹、关键帧和证据持久化。会话以 `session_id` 为键，逐轮存档问答，同一会话复用同一个 thread_id 续接检查点。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `pipeline/`：检测、跟踪、关键帧与视频片段生成。
- `web/`：QA 页面（对话 / 轨迹两个子页，左侧会话记录、右侧运行状态与证据）、NDJSON 事件流和证据接口。轨迹子页把一轮里的助手决策、工具调用（含参数与结果）与技能加载逐行列出。

## 问答链路要点

- **三阶段协同（可开关）**：`harness.subagents_enabled` 打开后，主智能体改读 `harness/planner.md`，只做规划、委派与汇总，领域工具收窄到 `config/subagents.yaml` 的 `master_tools`（默认只留 `show_evidence`）。三个从智能体**按阶段**分工，不按数据切：`planner`（意图 + 验收清单 + 时间范围，不带业务工具，只用框架自带的 `read_file` 读自己的规划技能组）、`executor`（主力执行者，拿全部 10 个领域工具与 track/registry/visual/execution 四组技能）、`reflector`（按清单验收、判定能否退出、汇总证据）。三者各自挂自己的技能组、守卫与结构化返回契约。关掉开关即退回单智能体（官方语义：禁用 general-purpose 且不传 subagents ⇒ 根本没有 `task` 工具）。
- **写入先验库要人工确认**：`executor` 手里有第 12 个工具 `add_registry_vessel`（唯一会改进知识库的操作），它配了 `interrupt_on` ——
  模型一调用，框架的 `HumanInTheLoopMiddleware` 就把整轮**中断**在写之前，前端弹确认卡；批准才真正落库（走 `ShipService` 的完整事务：
  写档案 → 存参考图 → 编码向量 → 重建 FAISS 索引，失败整体回滚），拒绝则把理由作为工具结果交回模型，让它换个做法继续这一轮。
  三道闸：工具描述写明"只在用户明确要求入库时调用"、`user_intent` 必填（要填用户原话）、以及这次人工确认。
- **计划清单上前端**：`write_todos` 由 `TodoListMiddleware` 注入主智能体，清单写在它的 state 里；运行时按内容指纹去重后广播 `plan` 事件（完整快照），前端在问答页渲染成带进度条的待办清单，展开历史轮次时从事件缓冲重建。
- **技能渐进式披露**：技能名与 description 随系统提示词注入（每一轮都在）；`read_file`/`ls`/`glob`/`grep` 由权限规则收敛到 `/skills` 目录内，写入与 shell 工具仍然禁用。整段会话还没读过任何技能正文时，`SkillDisclosureMiddleware` 会在模型第一次决策前把「先读正文再动手」并进 system 消息提醒一次。续接会话时框架不再回写技能回执，运行时会从检查点补回，保证事件流每一轮都能看到实际注入的技能目录。
- **收尾与证据**：`skills/finalize` 规定收尾动作，`EvidenceWrapUpMiddleware` 在模型给出最终回答却没调用证据工具时提醒一次，两者合起来保证前端证据面板有内容。⚠️ `show_evidence` 必须留在主智能体：从智能体的内部调用不进主事件流，下放它会让证据面板永远为空。
- **轮次不设上限**：由模型自己判断何时答完。工具调用仍留 run/thread 两级预算，并且带失控保护——工具预算用完后框架只会驳回后续调用，模型若继续硬调，连续失败到 `stall_guard_consecutive_errors` 次、或同一个「工具+参数」重复到 `stall_guard_repeat_calls` 次（弱模型会反复读同一个技能文件），即按现有结果收尾。

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


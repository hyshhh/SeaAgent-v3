# Setup

## 环境

推荐 Python 3.10–3.12。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
```

需要真实检测或向量索引时：

```powershell
py -m pip install -e ".[models]"
```

## 配置

配置位于 `config/`：`app.yaml`、`yolo.yaml`、`pipeline.yaml`、`prompts.yaml`、`harness.yaml` 和 `tools.yaml`。可通过 `SEAAGENT_CONFIG_DIR` 使用覆盖目录。

## 检查

```powershell
py -m compileall -q agent config harness memory pipeline services tools vector_store web
py -m pytest -q
py -m ruff check .
```

## 运行约定

系统默认是三阶段协同：主智能体只做规划、委派与汇总（提示词 `harness/planner.md`，领域工具只留证据汇总），三个阶段各挂一个从智能体——`planner`（意图、验收清单与时间范围，不带业务工具，只用框架自带的 `read_file` 按需读自己的技能组）、`executor`（主力执行者，拿全部领域工具与 `track`/`registry`/`visual`/`execution` 四组技能）、`reflector`（按清单验收、判定能否退出、汇总证据），各自只带自己的工具、技能组、守卫与结构化返回契约（见 `config/subagents.yaml`）；把 `harness.subagents_enabled` 设为 false 即退回单智能体。编排不做硬编码判定——能不能退出由反思阶段按 `skills/reflection` 判定，唯一由代码承担的是证据富化（`harness/evidence.py` 把散在各次工具结果里的关键帧、片段与参考图 ID 收齐成一份载荷，前端证据面板据此一次渲染）。技能按组存放（`skills/<组>/<技能>/SKILL.md`），模型需要时用 `read_file` 读正文；整段会话一次都没读过正文时，中间件会在动手前提醒一次。文件读取权限收敛在各自的技能组内，写入全拒；`harness.disabled_deepagent_tools` 那条禁用名单只管主智能体，从智能体的工具面由权限规则兜底（框架自带文件工具会出现在它们的工具列表里，但写不进去、也读不到技能组之外）。工具调用次数、重试、摘要和上下文边界由配置及官方 Middleware 管理；模型轮次不设上限，收尾时机由 `skills/answer/finalize` 与证据守卫共同约束。页面只显示公开状态、Skill、模型、工具、证据和最终消息，不显示隐藏推理；待办清单由 `TodoListMiddleware` 写在主智能体 state 里，运行时按内容指纹去重后以 `plan` 事件广播完整快照，问答页把它渲染成带进度的清单，打开历史轮次时从事件缓冲重建。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用同一个 thread_id 续接 SQLite 检查点。左侧会话栏可切换历史会话、开启新会话或删除单个会话；「Clear memory」连同检查点一并清空。

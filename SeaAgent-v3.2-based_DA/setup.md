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

系统只有一个主智能体。Skills 先提供名称和描述，模型需要时用 `read_file` 读取 `/skills/<name>/SKILL.md` 正文（文件读取权限收敛在 skills 目录内，写入与 shell 工具保持禁用）。工具调用次数、重试、摘要和上下文边界由配置及官方 Middleware 管理；模型轮次不设上限，收尾时机由 `skills/finalize` 与证据守卫共同约束。页面只显示公开状态、Skill、模型、工具、证据和最终消息，不显示隐藏推理。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用同一个 thread_id 续接 SQLite 检查点。左侧会话栏可切换历史会话、开启新会话或删除单个会话；「Clear memory」连同检查点一并清空。

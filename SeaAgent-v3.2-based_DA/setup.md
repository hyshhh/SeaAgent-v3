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

配置位于 `config/`：`app.yaml`、`yolo.yaml`、`pipeline.yaml`、`prompts.yaml`、`harness.yaml` 和 `tools.yaml`。三 Agent 的角色和工具面位于 `config/subagents.yaml`。可通过 `SEAAGENT_CONFIG_DIR` 使用覆盖目录。

关键 Harness 参数：

- `subagents_enabled`：是否启用 Planner、Executor、Reflector 三 Agent；关闭后保留单 Agent 兼容模式。
- `shared_state_file`：共享状态持久化文件。
- `subagent_tool_calls`：单 Agent 工具调用上限。
- `tool_calls_per_run`、`tool_calls_per_thread`：整轮和会话级上限。
- `max_iterations`：闭环最大迭代次数。
- `reconnect_attempts`、`reconnect_backoff_seconds`：Agent 重连次数和退避间隔。
- `stall_guard_repeat_calls`、`stall_guard_consecutive_errors`：重复调用和连续失败保护。

## 检查

```powershell
py -m compileall -q agent config harness memory pipeline services tools vector_store web
py -m pytest -q
py -m ruff check .
```

## 运行约定

一次问答按以下顺序运行：

```text
Planner
  → 写入 requirements / plan / next_action
Executor
  → 执行工具，写入 evidence_ids / tool_results / execution_status
Reflector
  → 检查充分性、缺口和冲突，写入 decision
      ├─ terminate：生成答案并结束
      └─ replan：进入下一轮 Planner
```

三个 Agent 使用不同的 SQLite `thread_id` 后缀，但共享同一个 `SharedTaskState`。每个角色只能更新自己的字段，状态版本用于防止过期写入。Executor 调用 `add_registry_vessel` 时仍然需要人工确认；恢复接口必须使用同一个 `sessionId`。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用 SQLite 检查点和共享状态。左侧会话栏可切换历史会话、开启新会话或删除单个会话；`Clear memory` 会清空会话、检查点和共享状态。

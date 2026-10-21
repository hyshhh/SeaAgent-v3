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

- `shared_state_file`：SQLite 共享任务状态路径，旧的同名 JSON 存档首次启动时自动导入。
- `subagent_tool_calls=6`：每个 Agent 阶段的领域工具调用上限；共享状态读写、技能读取和证据展示不占用预算，失败重试计入预算。
- 单 Agent 阶段和整轮问答不设墙钟时间限制；外部领域工具仍使用独立的 `tool_timeout_seconds=60`。
- `tool_calls_per_run=40`：单次问答的领域工具调用上限。
- `max_iterations`：闭环最大迭代次数。
- `reconnect_attempts`、`reconnect_backoff_seconds`：Agent 重连次数和退避间隔。
- 共享状态读写、技能读取和证据展示不占领域工具预算；预算耗尽后 Agent 仍可提交自己负责的状态。
- 内部不使用 TodoList；Planner 写计划，Executor 写 `step_status`，前端进度由共享状态生成。

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
  → 执行工具，维护 evidence_ids / step_status / execution_status；运行时记录 tool_results
Reflector
  → 检查充分性、缺口和冲突，写入 decision
      ├─ terminate：生成答案并结束
      └─ replan：进入下一轮 Planner
```

三个 Agent 使用不同的 SQLite `thread_id` 后缀，但共享同一个 `SharedTaskState`。每个角色只能更新自己的字段，状态版本用于防止过期写入。Executor 调用 `add_registry_vessel` 时仍然需要人工确认；恢复接口必须使用同一个 `sessionId`。

预算耗尽只阻止后续领域工具调用，不阻止 Agent 写入状态。状态写入中间件会在阶段结束前补充一次状态提交机会；补写仍未完成时返回 `stalled`。外部工具仍受独立的 `tool_timeout_seconds=60` 限制，阻塞的同步外部服务不能强制终止。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用 SQLite 检查点和共享状态。左侧会话栏可切换历史会话、开启新会话或删除单个会话；`Clear memory` 会清空会话、检查点和共享状态。

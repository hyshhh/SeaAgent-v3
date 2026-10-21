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
- `subagent_tool_calls=6`：每个 Agent 阶段的领域工具调用上限；共享状态读写和技能读取不占用预算，失败重试计入预算。
- 单 Agent 阶段和整轮问答不设墙钟时间限制；外部领域工具仍使用独立的 `tool_timeout_seconds=60`。
- `convergence_timeout_seconds=60`：到内外层上限后，单次无工具 LLM 总结的独立请求预算；不重试，失败时使用已有证据摘要。
- `tool_calls_per_run=40`：单次问答的领域工具调用上限。
- `max_iterations`：闭环最大迭代次数。
- `reconnect_attempts`、`reconnect_backoff_seconds`：Agent 重连次数和退避间隔。
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

到内层上限时，当前角色停止探索并将一次总结写入共享状态；到外层上限时直接生成最终回答。两者都返回正常完成状态，并用 `partial`、`convergence` 标记证据边界和收敛原因。超时在工具调用边界生效，阻塞外部服务不能强制终止，最终总结另有 60 秒请求预算；整次响应不保证在 60 秒内完成。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用 SQLite 检查点和共享状态。左侧会话栏可切换历史会话、开启新会话或删除单个会话；`Clear memory` 会清空会话、检查点和共享状态。

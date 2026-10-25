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

配置位于 `config/`：`app.yaml`、`yolo.yaml`、`pipeline.yaml`、`prompts.yaml`、`harness.yaml` 和 `tools.yaml`。七个业务工具及参数 schema 位于 `config/tools.yaml`；三 Agent 的角色、工具面和技能位于 `config/subagents.yaml`。可通过 `SEAAGENT_CONFIG_DIR` 使用覆盖目录。

关键 Harness 参数：

- `shared_state_file`：SQLite 共享任务状态路径，旧的同名 JSON 存档首次启动时自动导入。
- `config/subagents.yaml`：分别配置三个 Agent 的任务提示词、技能提醒、正常模型请求额度和收尾请求额度；正常额度为 Planner 4、Executor 12、Reflector 6，收尾额度均为 2。
- 单 Agent 阶段和整轮问答不设墙钟时间限制；识别工具在服务层设 60 秒预算，统计与证据工具设 120 秒预算。
- `max_iterations`：闭环最大迭代次数。
- `reconnect_attempts`、`reconnect_backoff_seconds`：Agent 重连次数和退避间隔。
- 不限制工具调用总次数。相同工具和参数第 2、3 次返回缓存结果，第 4 次进入状态收尾。
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

三个 Agent 使用不同的 SQLite `thread_id` 后缀，但共享同一个 `SharedTaskState`。每个角色只能更新自己的字段，状态版本用于防止过期写入；恢复接口必须使用同一个 `sessionId`。

模型请求额度耗尽或重复调用达到第 4 次后，只允许共享状态读取和本角色状态更新；一次修复机会仍未写入时返回 `stalled`。工具按类别在 60 或 120 秒后返回超时错误；Python 线程无法被强制终止，底层操作可能继续运行。

QA 会话按 `session_id` 落库（`data/memory/qa_sessions.csv`），同一会话的追问复用 SQLite 检查点和共享状态。左侧会话栏可切换历史会话、开启新会话或删除单个会话；`Clear memory` 会清空会话、检查点和共享状态。

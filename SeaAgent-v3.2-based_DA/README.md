# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的证据状态驱动多智能体系统。系统由三个独立 Agent 组成：`planner` 维护任务需求与执行计划，`executor` 执行轨迹、关键帧、视频片段、先验库和视觉核验工具，`reflector` 审查证据充分性、冲突与终止条件。三个 Agent 不通过主 Agent 的 `task` 工具传递自然语言结果，而是通过版本化的共享任务状态字典协作。

## 组件

- `harness/runtime.py`：按 `planner → executor → reflector` 调度三个独立 Agent，处理重连、超时、取消、人工确认和事件流。
- `harness/shared_state.py`：唯一任务状态、字段归属、版本校验、证据 ID 收集和 SQLite 事务持久化。
- `harness/trace.py`：将框架消息和共享计划投影为前端事件与审计记录。
- `harness/state_tools.py`：各 Agent 读取共享状态并只更新自己负责的字段。
- `harness/tool_guard.py`：统计工具调用、复用精确重复的结果，并在第 4 次重复时进入状态收尾。
- `harness/state_write.py`：按角色限制模型请求；额度耗尽后只开放共享状态读取和本角色状态写入。
- `config/subagents.yaml`：三个 Agent 的工具、技能、提示词和人工确认配置。
- `skills/`：按组存放领域规则，Agent 只读取自己挂载的技能组。
- `memory/`：会话、轨迹、关键帧和问答证据持久化。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `web/`：QA 页面、NDJSON 事件流和证据接口。

## 共享状态

共享状态包含以下字段：

```text
requirements / plan / next_action              Planner
evidence_ids / step_status / execution_status   Executor
assessment / evidence_gaps / conflicts / decision Reflector
task_status / iteration / pending_agent / version Runtime
tool_results / evidence / call_history / 模型与工具调用计数 Runtime 自动记账
```

状态更新工具只提交当前角色负责的业务字段，模型不需要维护版本号；状态版本由运行时内部推进，SQLite 保存时仍校验存储修订号，避免旧运行时覆盖新状态。状态按 `thread_id` 存入 `data/memory/shared_task_state.sqlite`；旧的同名 JSON 存档首次启动时导入，原文件保留，删除后的会话不会再次导入。

`plan` 中每步包含 `id`、`content`、`required_evidence`，其中 `required_evidence` 是字符串数组，Executor 用独立的 `step_status` 维护进度。前端 `plan` 事件完全由这两项生成，内部不装配 TodoList，也不从模型回答的 JSON 反向恢复状态。最终回答只取 `assessment.final_answer`。框架检查点仍保存消息、技能元数据和人工确认中断，这些是执行记录，不维护另一份业务计划。

## 运行保护

- `config/subagents.yaml` 为每个角色分别配置正常模型请求额度和收尾请求额度；当前正常额度为 Planner 4、Executor 12、Reflector 6，收尾额度均为 2。
- 单 Agent 阶段和整轮问答均不设墙钟时间限制；工具批次串行执行，每个工具调用独立受 `tool_timeout_seconds=120` 限制。
- `max_iterations` 将 Planner、Executor、Reflector 的闭环次数限制为 4。
- 模型请求额度耗尽后，只允许最多两次状态收尾请求；若状态仍未写入则返回 `stalled`。模型重试也计入模型请求额度。
- 同一角色、同一轮的相同工具和规范化参数只执行一次；第 2、3 次复用结果并提醒，第 4 次切换到状态收尾。共享状态读取保持实时，状态工具在收尾时仍可用。
- 不再限制工具总次数；工具计数只用于轨迹审计和重复调用判断。
- 写库工具不自动重试；结果不确定时保留请求标记并阻止再次执行。
- Agent 异常按 `reconnect_attempts` 和指数退避间隔重建，重连时保留共享状态和 SQLite 检查点。
- 模型额度或重复调用触发收尾时，中间件只开放共享状态读取和本角色状态更新，不另起总结请求。
- 每个 Agent 结束时检查其负责状态；缺失时发起一次仅限状态工具的修复请求，仍失败则返回 `stalled`，不伪造计划、步骤或证据结论。
- 人工确认保留中断与原调用计数；用户取消立即走取消路径，不触发总结。
- 同一会话同时只允许一个 HTTP 问答；用户取消后等后台线程实际退出才释放登记。

同步工具超时会在两分钟后向 Agent 返回错误；Python 无法强制终止已经运行的线程，超时工具可能在后台继续执行。SQLite 保护共享状态，CSV 记忆表的锁仍限于单进程，Web 服务应保持单 worker 运行。

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

默认依赖包含 LangChain、Deep Agents、OpenAI-compatible Chat Model 和 SQLite checkpoint。需要 FAISS 或 Ultralytics 时安装 `.[models]`。模型地址、工具映射、Agent 角色、模型额度和路径均来自 `config/`。

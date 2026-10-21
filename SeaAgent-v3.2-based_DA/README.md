# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的证据状态驱动多智能体系统。系统由三个独立 Agent 组成：`planner` 维护任务需求与执行计划，`executor` 执行轨迹、关键帧、视频片段、先验库和视觉核验工具，`reflector` 审查证据充分性、冲突与终止条件。三个 Agent 不通过主 Agent 的 `task` 工具传递自然语言结果，而是通过版本化的共享任务状态字典协作。

## 组件

- `harness/runtime.py`：按 `planner → executor → reflector` 调度三个独立 Agent，处理重连、超时、取消、人工确认和事件流。
- `harness/shared_state.py`：唯一任务状态、字段归属、版本校验、证据 ID 收集和 SQLite 事务持久化。
- `harness/trace.py`：将框架消息和共享计划投影为前端事件与审计记录。
- `harness/state_tools.py`：各 Agent 读取共享状态并只更新自己负责的字段。
- `harness/tool_guard.py`：从共享状态读取调用账本和预算，复用已完成的查询，阻止重复写入。
- `harness/convergence.py`：预算耗尽时汇集已有证据，进行一次无工具总结；模型失败则使用确定性摘要。
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
task_status / iteration / pending_agent / version / convergence Runtime
tool_results / evidence / call_history / 调用计数 Runtime 自动记账
```

状态更新工具必须提交最近读取的 `expected_version`；旧版本更新会被拒绝。状态按 `thread_id` 存入 `data/memory/shared_task_state.sqlite`，保存时另外校验存储修订号，避免多个运行时互相覆盖。旧的同名 JSON 存档首次启动时导入，原文件保留；删除后的会话不会再次导入。

`plan` 中每步包含 `id`、`content`、`required_evidence`，Executor 用独立的 `step_status` 维护进度。前端 `plan` 事件完全由这两项生成，内部不装配 TodoList，也不从模型回答的 JSON 反向恢复状态。最终回答只取 `assessment.final_answer`。框架检查点仍保存消息、技能元数据和人工确认中断，这些是执行记录，不维护另一份业务计划。

## 运行保护

- 每个 Agent 阶段的 `subagent_tool_calls` 上限为 **6**，只统计领域工具调用；共享状态读写和技能读取不占用预算，失败重试计入预算，并行批量请求也不能越过上限。
- 单 Agent 阶段和整轮问答均不设墙钟时间限制；外部领域工具仍受独立的 `tool_timeout_seconds=60` 限制。
- 整轮 `tool_calls_per_run` 上限为 40；每次新问答重新计算工具预算。
- `max_iterations` 将 Planner、Executor、Reflector 的闭环次数限制为 4。
- 阶段和整轮工具预算在执行前统一拦截，重连和人工确认不重置调用计数；新角色阶段重置阶段预算，新问答重置单轮预算。
- 相同领域查询复用成功结果；失败读取只允许有限重试，每次尝试计入预算。共享状态读取和更新可以多次调用。
- 写库工具不自动重试；结果不确定时保留请求标记并阻止再次执行。
- Agent 异常按 `reconnect_attempts` 和指数退避间隔重建，重连时保留共享状态和 SQLite 检查点。
- 内层次数耗尽时，每个角色在当前迭代只追加一次无工具 LLM 总结；Planner、Executor 的总结写入共享状态供后续角色审查，Reflector 总结直接生成最终回答。
- 外层次数或迭代耗尽时，直接汇集已有证据并追加一次最终 LLM 总结，避免重复进行内层总结。预算耗尽生成正常终态，不作为错误；结果用 `partial` 和 `convergence` 标明收敛原因与证据限制。
- 收敛请求有独立的 `convergence_timeout_seconds=60` 预算，不调用工具、不自动重试；模型失败时返回确定性证据摘要，不把未验证步骤标成完成。
- 人工确认保留中断与原调用计数；用户取消立即走取消路径，不触发总结。
- 同一会话同时只允许一个 HTTP 问答；用户取消后等后台线程实际退出才释放登记。

工具超时与取消在调用边界生效，不能强制终止阻塞中的外部服务。因此 60 秒是探索预算，最终总结另有请求预算，整次响应不保证在 60 秒内完成。SQLite 保护共享状态，CSV 记忆表的锁仍限于单进程，Web 服务应保持单 worker 运行。

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

默认依赖包含 LangChain、Deep Agents、OpenAI-compatible Chat Model 和 SQLite checkpoint。需要 FAISS 或 Ultralytics 时安装 `.[models]`。模型地址、工具映射、Agent 角色、调用限制和路径均来自 `config/`。

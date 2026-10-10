# Sea-Video-Harness

Sea-Video-Harness 是面向海域监控视频问答的证据状态驱动多智能体系统。系统由三个独立 Agent 组成：`planner` 维护任务需求与执行计划，`executor` 执行轨迹、关键帧、视频片段、先验库和视觉核验工具，`reflector` 审查证据充分性、冲突与终止条件。三个 Agent 不通过主 Agent 的 `task` 工具传递自然语言结果，而是通过版本化的共享任务状态字典协作。

## 组件

- `harness/runtime.py`：按 `planner → executor → reflector` 调度三个独立 Agent，处理重连、超时、取消、人工确认和事件流。
- `harness/shared_state.py`：共享状态、字段归属、版本校验、证据 ID 收集和原子 JSON 持久化。
- `harness/state_tools.py`：各 Agent 读取共享状态并只更新自己负责的字段。
- `harness/tool_guard.py`：跨 Agent 实例复用的重复调用账本，阻止相同工具和参数重复执行。
- `config/subagents.yaml`：三个 Agent 的工具、技能、提示词和人工确认配置。
- `skills/`：按组存放领域规则，Agent 只读取自己挂载的技能组。
- `memory/`：会话、轨迹、关键帧和问答证据持久化。
- `tools/`：视频轨迹、关键帧、片段、先验库和视觉核验工具。
- `web/`：QA 页面、NDJSON 事件流和证据接口。

## 共享状态

共享状态包含以下字段：

```text
requirements / plan / next_action       Planner
 evidence_ids / tool_results / status   Executor
 assessment / gaps / conflicts / decision Reflector
 task_status / iteration / version       Runtime
```

每次有效更新都会递增 `version`。Agent 可以提交 `expected_version`，发现旧状态时更新会被拒绝，避免重连或延迟返回覆盖新结果。状态按 `thread_id` 原子写入 `data/memory/shared_task_state.json`，因此 HTTP 重连和人工确认恢复不会丢失交接信息。

## 运行保护

- 每个 Agent 有独立 `subagent_tool_calls` 上限。
- 整轮由 `tool_calls_per_run` 再做全局上限。
- 整段会话受 `tool_calls_per_thread` 限制。
- `max_iterations` 限制 Planner、Executor、Reflector 的闭环次数。
- `RepeatToolCallMiddleware` 阻止相同工具和参数的重复调用。
- Agent 异常按 `reconnect_attempts` 和指数退避间隔重建，重连时保留共享状态和 SQLite 检查点。
- 单步超时、连续工具失败、人工确认和取消都会生成明确的终态事件。

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

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

系统只有一个主智能体。Skills 先提供名称和描述，需要时再读取完整规范；工具参数、调用次数、重试、摘要和上下文边界由配置及官方 Middleware 管理。页面只显示公开状态、Skill、模型、工具、证据和最终消息，不显示隐藏推理。

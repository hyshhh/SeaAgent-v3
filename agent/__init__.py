"""SeaAgent 四子智能体：LangChain 工具 + LangGraph 编排。

- IntentAgent / PlanAgent / ObserveAgent / ReflectAgent
- handoff 工具在 Agent 间移交
- AgentController.answer() 保持前端契约

包内文件分工：
    controller.py      对外边界：answer() 契约、终态投影、落库
    graph.py           编排主体：AgentState 状态机与四个节点
    plan_executor.py   observe 的确定性执行内核
    lc_tools.py        业务工具与技能工具 → LangChain 工具封装
    roles.py           四角色系统提示词拼装
    skill_loader.py    skills/ 的读取（目录 + 正文 + YAML）
    task_profiles.py   任务画像单一事实源（membership / 证据量级）
    llm_adapter.py     ChatOpenAI 构造
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

# 唯一对外符号：整个 agent 包的公开 API 面就是 AgentController
__all__ = ["AgentController"]

if TYPE_CHECKING:
    from .controller import AgentController


def __getattr__(name: str) -> Any:
    # 延迟导入：controller 会连带拉起 langgraph / langchain / faiss 等重依赖，
    # 放在模块顶层会让任何 `import agent`（含测试收集）都付出这个代价。
    if name == "AgentController":
        from .controller import AgentController
        return AgentController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

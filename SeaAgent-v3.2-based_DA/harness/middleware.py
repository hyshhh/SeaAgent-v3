"""LangChain middleware assembly for the single-agent harness.

中间件是这套 harness 的硬约束层：上下文压缩、工具调用上限、重试、待办清单，
以及回答定稿前的证据守卫。所有阈值都来自 config/harness.yaml，本模块只做装配与参数绑定。

模型轮次不设上限：收尾时机交给模型自己判断——规范写在 ``skills/finalize/SKILL.md``，
兜底由 ``harness/wrapup.py`` 的证据守卫完成，因此这里不再挂 ModelCallLimitMiddleware。
"""
from __future__ import annotations

from typing import Any


def build_middleware(config: dict[str, Any], model: Any, *, skills_attached: bool = True) -> list[Any]:
    """按 harness 配置装配中间件。

    返回值顺序即执行顺序：``wrap_*`` 类中间件是洋葱嵌套（排在前面的在外层），
    ``before_model`` / ``after_model`` 类按同向顺序串行。

    ``skills_attached=False``（主从协同时主智能体没挂技能）会跳过披露提醒：
    没有技能目录还提醒"先读技能"，只会把模型推向读一个不存在的文件。
    """
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        SummarizationMiddleware,
        TodoListMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )

    from config import project_root

    from .disclosure import SkillDisclosureMiddleware
    from .tool_guard import RepeatToolCallMiddleware
    from .wrapup import EvidenceWrapUpMiddleware

    settings = config.get("harness", {})
    # 摘要提示词来自文件：默认摘要会把"刚才""15:30 到 15:40"这类上下文压没，
    # 导致续接会话时模型自己都说"对话已被总结，不知道刚才指哪一段"。这里要求摘要逐字保留
    # 时间范围、各类 ID、用户口径与待办。
    prompt_file = project_root() / str(settings.get("summary_prompt_file", "harness/summary_prompt.md"))
    summary_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else None
    summarization_kwargs = {}
    if summary_prompt:
        summarization_kwargs["summary_prompt"] = summary_prompt
    stack: list[Any] = [
        # 重复调用守卫：同一个「工具 + 参数」重复出现时跳过执行并提示模型，
        # 放在最外层，这样后面的重试/限流都看不到这次调用（它压根不该被执行）
        RepeatToolCallMiddleware(),
    ]
    if skills_attached:
        # 技能披露提醒：整段会话还没读过技能正文时，在模型第一次决策前把规范并进 system 消息
        stack.append(SkillDisclosureMiddleware(max_reminders=int(settings.get("skill_reminder_max_per_run", 1))))
    stack += [
        # 上下文压缩：历史超阈值即摘要旧消息，只留最近若干条，保证长会话不撑爆窗口。
        # 摘要提示词来自文件——默认摘要会把时间范围与各类 ID 压没，续接会话就"不知道刚才指哪一段"
        SummarizationMiddleware(model, trigger=("tokens", int(settings.get("summarization_trigger_tokens", 12000))), keep=("messages", int(settings.get("summarization_keep_messages", 12))), **summarization_kwargs),
        # 工具调用上限：run 级管单次问答，thread 级管整段会话。轮次不设上限，这一层只拦失控试探；
        # exit_behavior 保持 continue——超限的工具以错误结果回给模型，让它自己收敛并给出收尾回答
        ToolCallLimitMiddleware(run_limit=int(settings.get("tool_calls_per_run", 40)), thread_limit=int(settings.get("tool_calls_per_thread", 400))),
        # 模型重试：provider 抖动或限流时自动重发，避免一次网络失败毁掉整轮问答
        ModelRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        # 工具重试：工具内部异常（读文件、调服务）同样重试，与模型重试共用一组参数
        ToolRetryMiddleware(max_retries=int(settings.get("retry_attempts", 2)), initial_delay=float(settings.get("retry_delay_seconds", 1.0))),
        # 待办清单：注入 write_todos 工具，把长任务的中间计划写进 state，供后续轮次自查
        TodoListMiddleware(),
        # 收尾守卫：模型给出最终回答却没落地证据时提醒一次，与 skills/finalize 合成完整收尾约束
        EvidenceWrapUpMiddleware(evidence_tool=str(settings.get("evidence_tool", "")), max_nudges=int(settings.get("evidence_wrapup_max_nudges", 1))),
    ]
    return stack

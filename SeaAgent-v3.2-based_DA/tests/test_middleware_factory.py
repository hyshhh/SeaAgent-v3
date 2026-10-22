from copy import deepcopy

from deepagents import FilesystemPermission
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import FakeListChatModel

from config import load_config
from harness.disclosure import SkillDisclosureMiddleware
from harness.middleware import build_middleware
from harness.runtime import _load_subagents
from harness.shared_state import SharedTaskState
from harness.state_tools import build_state_tools
from harness.state_write import StateWriteMiddleware
from harness.tool_guard import RepeatToolCallMiddleware
from harness.tools import build_tools
from harness.wrapup import EvidenceWrapUpMiddleware


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def _setup():
    config = load_config()
    tools = build_tools(config, _Service()) + build_state_tools(SharedTaskState(""))
    specs = _load_subagents(config, tools, FilesystemPermission)
    return config, {spec["name"]: spec for spec in specs}


def _stack(role="reflector", skills=True):
    config, specs = _setup()
    spec = deepcopy(specs[role])
    if not skills:
        spec["skills"] = []
    state = SharedTaskState("q")
    stack = build_middleware(
        config,
        FakeListChatModel(responses=["ok"]),
        agent_spec=spec,
        shared_state=state,
    )
    return config, spec, stack


def test_role_specific_middleware_stack():
    _, _, stack = _stack()
    assert [type(item) for item in stack] == [
        ToolRetryMiddleware,
        ModelRetryMiddleware,
        RepeatToolCallMiddleware,
        StateWriteMiddleware,
        SkillDisclosureMiddleware,
        SummarizationMiddleware,
        EvidenceWrapUpMiddleware,
    ]


def test_model_rounds_are_not_capped():
    _, _, stack = _stack()
    assert not any(type(item).__name__ == "ModelCallLimitMiddleware" for item in stack)


def test_role_budget_and_reminder_are_read_from_role_spec():
    _, spec, stack = _stack("executor")
    guard = next(item for item in stack if isinstance(item, RepeatToolCallMiddleware))
    reminder = next(item for item in stack if isinstance(item, SkillDisclosureMiddleware))
    assert guard.tool_budget == spec["tool_budget"] == 6
    assert reminder.max_reminders == spec["skill_reminder"]["max_per_run"] == 2
    assert reminder.reminder == spec["skill_reminder"]["message"]


def test_reflector_evidence_guard_uses_its_role_configuration():
    _, spec, stack = _stack("reflector")
    guard = next(item for item in stack if isinstance(item, EvidenceWrapUpMiddleware))
    assert guard.evidence_tool == spec["evidence_wrapup"]["tool"]
    assert guard.max_nudges == spec["evidence_wrapup"]["max_nudges"]


def test_disclosure_guard_is_skipped_when_role_has_no_skills():
    _, _, with_skills = _stack("planner")
    _, _, without_skills = _stack("planner", skills=False)
    assert any(isinstance(item, SkillDisclosureMiddleware) for item in with_skills)
    assert not any(isinstance(item, SkillDisclosureMiddleware) for item in without_skills)


def test_summarization_uses_the_domain_prompt_from_file():
    _, _, stack = _stack()
    summarization = next(item for item in stack if isinstance(item, SummarizationMiddleware))
    assert "逐字保留" in summarization.summary_prompt
    assert "时间范围" in summarization.summary_prompt

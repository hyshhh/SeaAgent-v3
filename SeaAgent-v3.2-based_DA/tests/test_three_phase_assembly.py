"""三阶段协同装配：plan / 执行 / reflect 三个从智能体，主智能体只规划、委派与落证据。

守的是四件事：
  · 执行阶段要"丰富"——全部领域工具 + 四个技能组都挂在执行智能体身上，不按数据拆分身；
  · 规划与反思只拿各自阶段的窄工具面，各自挂自己的技能组与返回契约；
  · 主智能体手里只有 show_evidence（证据必须回到主事件流），且不挂技能；
  · 从智能体各自带守卫与读取收口，且**没有**任何自造的判定中间件——编排靠框架本身。
"""
import pathlib
from typing import ClassVar

from config import load_config
from deepagents import FilesystemPermission
from deepagents.middleware.filesystem import _check_fs_permission
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models import FakeListChatModel

from harness.middleware import build_middleware
from harness.runtime import (
    SeaVideoHarness,
    _filesystem_permissions,
    _load_subagents,
    _read_scope_paths,
    _readonly_permissions,
    _skill_sources,
    _tool_labels,
)
from harness.tool_guard import RepeatToolCallMiddleware
from harness.tools import build_tools

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _Service:
    """任何方法都返回自己的名字，装配测试不碰真实服务。"""

    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


class _Model(FakeListChatModel):
    """可装配的假模型：摘要中间件要 with_retry 与 _llm_type，用官方假模型而不是自造对象。"""

    model_name: ClassVar[str] = "fake-model"

    def bind_tools(self, tools):
        return self


def _specs():
    config = load_config()
    subagents, master_tools, master_skills, interrupts = _load_subagents(config, build_tools(config, _Service()), FilesystemPermission)
    return subagents, master_tools, master_skills


def _by_name(specs):
    return {spec["name"]: spec for spec in specs}


def _runtime(tmp_path):
    config = load_config()
    config["harness"]["checkpointer"] = str(tmp_path / "checkpoints.sqlite")
    return SeaVideoHarness(config, _Service(), model=_Model(responses=["ok"]))


# ---------------------------------------------------------------- 三个阶段的形状


def test_two_subagents_beside_the_executing_main_agent():
    """主智能体本人就是执行者，所以只有规划与反思两个从智能体。"""
    subagents, _, _ = _specs()
    assert [spec["name"] for spec in subagents] == ["planner", "reflector"]


def test_the_main_agent_is_the_executor():
    """主智能体 = 执行者：全部领域工具 + 执行类技能都在它手里，不再多一层委派。"""
    _, master_tools, master_skills = _specs()
    configured = [str(item["name"]) for item in load_config()["tools"]]
    # 顺序无关：配置里 show_evidence 排在业务工具之前
    assert sorted(master_tools) == sorted(configured), "主智能体应当拿全部领域工具"
    assert len(configured) == 12
    assert "add_registry_vessel" in master_tools, "写库工具也在执行者手里"
    assert master_skills == ["execution", "track", "registry", "visual", "answer"]


def test_planner_plans_without_touching_data():
    """规划阶段只解析意图与验收清单，不拿任何领域工具。"""
    subagents, _, _ = _specs()
    planner = _by_name(subagents)["planner"]
    assert planner["tools"] == []
    assert planner["skills"] == ["/skills/planning"]
    assert "```json" in planner["system_prompt"], "规划阶段要约定返回的 JSON 块"


def test_reflector_audits_and_lands_the_evidence():
    """反思阶段要能落证据（show_evidence）并查库核对，返回验收判定契约。"""
    subagents, _, _ = _specs()
    reflector = _by_name(subagents)["reflector"]
    assert [str(tool.name) for tool in reflector["tools"]] == ["show_evidence", "get_registry", "list_registry"]
    assert reflector["skills"] == ["/skills/reflection", "/skills/answer"]
    assert "```json" in reflector["system_prompt"], "反思阶段要约定返回的 JSON 块"


def test_no_phase_uses_structured_response_format():
    """回归：不许再挂 response_format。

    事故经过：框架把结构化 schema 绑成一个「工具」，而这个 4B 模型在思考模式下一次回复里
    把它调用了十几次；框架判定「结构化返回只能一次」后，按 ToolStrategy 的默认
    handle_errors=True 把错误塞回对话重试，模型每次都犯同样的错，直到烧光子智能体的
    工具预算（12 次）——对外表现就是「委派卡住、什么都出不来」。
    """
    subagents, _, _ = _specs()
    offenders = [spec["name"] for spec in subagents if "response_format" in spec]
    assert not offenders, f"这些子智能体又挂上了结构化返回：{offenders}"


def test_every_phase_carries_a_prompt_and_description():
    subagents, _, _ = _specs()
    for spec in subagents:
        assert str(spec["description"]).strip()
        assert str(spec["system_prompt"]).strip()


def test_each_phase_carries_its_own_guards_and_read_scope():
    """从智能体不继承主智能体的中间件，限流与重复守卫必须各自挂；读取面按技能组收口。"""
    subagents, _, _ = _specs()
    limit = int(load_config()["harness"]["subagent_tool_calls"])
    for spec in subagents:
        kinds = [type(item) for item in spec["middleware"]]
        assert RepeatToolCallMiddleware in kinds
        assert ToolCallLimitMiddleware in kinds
        assert next(item for item in spec["middleware"] if isinstance(item, ToolCallLimitMiddleware)).run_limit == limit
    planner = _by_name(subagents)["planner"]
    allow = [rule for rule in planner["permissions"] if str(rule.mode) == "allow"]
    assert set(allow[0].paths) == set(_read_scope_paths(["/skills/planning"]))


def test_the_reflector_keeps_its_narrow_tool_set():
    """反思阶段只拿证据与查库工具；show_evidence 主智能体也持有（证据要落到主事件流）。"""
    subagents, master_tools, _ = _specs()
    reflector = _by_name(subagents)["reflector"]
    assert [str(tool.name) for tool in reflector["tools"]] == ["show_evidence", "get_registry", "list_registry"]
    assert "show_evidence" in master_tools


def test_master_assembly_carries_the_domain_tools(tmp_path):
    """装配结果：主智能体拿全领域工具与执行类技能，中断配置也在它身上。"""
    runtime = _runtime(tmp_path)
    try:
        configured = [str(item["name"]) for item in load_config()["tools"]]
        assert [str(tool.name) for tool in runtime.agent_tools] == configured
        assert runtime.skill_sources == ["/skills/execution", "/skills/track", "/skills/registry", "/skills/visual", "/skills/answer"]
        assert runtime.master_interrupts == {"add_registry_vessel": {"allowed_decisions": ["approve", "reject"]}}
        assert [spec["name"] for spec in runtime.subagents] == ["planner", "reflector"]
    finally:
        runtime.close()


def test_a_subagent_can_only_read_its_own_skill_groups():
    """读取面按组收口：planner 只读得到 planning，反思组对它是拒绝的，写入一律拒绝。"""
    planner_rules = _by_name(_specs()[0])["planner"]["permissions"]
    assert _check_fs_permission(planner_rules, "read", "/skills/planning/intent/SKILL.md") == "allow"
    assert _check_fs_permission(planner_rules, "read", "/skills/reflection/exit_rules/SKILL.md") == "deny"
    assert _check_fs_permission(planner_rules, "write", "/skills/planning/intent/SKILL.md") == "deny"


def test_unknown_tool_name_and_structured_format_fail_loudly(tmp_path):
    """配置写错要在启动时炸掉，而不是等模型真去调它。"""
    import yaml

    config = load_config()
    spec_path = tmp_path / "broken.yaml"
    config["harness"]["subagents_file"] = str(spec_path)
    tools = build_tools(config, _Service())

    spec_path.write_text(yaml.safe_dump({"subagents": [{"name": "x", "description": "d", "system_prompt": "s", "tools": ["no_such_tool"]}]}), encoding="utf-8")
    try:
        _load_subagents(config, tools, FilesystemPermission)
    except ValueError as error:
        assert "no_such_tool" in str(error)
    else:
        raise AssertionError("未声明的工具名没有被拦下")

    # 结构化返回一律拒绝：配了要当场报错，不能静默失效（见 test_no_phase_uses_structured_response_format）
    spec_path.write_text(yaml.safe_dump({"subagents": [{"name": "x", "description": "d", "system_prompt": "s", "tools": [], "response_format": "IntentPlan"}]}), encoding="utf-8")
    try:
        _load_subagents(config, tools, FilesystemPermission)
    except ValueError as error:
        assert "response_format" in str(error)
    else:
        raise AssertionError("配了 response_format 却没有被拦下")


# ---------------------------------------------------------------- 编排靠框架，不靠自造判定


def test_middleware_is_framework_guards_only():
    """中间件链里只有官方与本项目原有的通用守卫，没有任何自造的阶段判定。"""
    config = load_config()
    names = [type(item).__name__ for item in build_middleware(config, _Model(responses=["ok"]), skills_attached=True)]
    assert names == [
        "RepeatToolCallMiddleware",
        "SkillDisclosureMiddleware",
        "SummarizationMiddleware",
        "ToolCallLimitMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "TodoListMiddleware",
        "EvidenceWrapUpMiddleware",
    ]
    without_skills = [type(item).__name__ for item in build_middleware(config, _Model(responses=["ok"]), skills_attached=False)]
    assert "SkillDisclosureMiddleware" not in without_skills


def test_skill_reads_stay_confined_and_writes_are_denied():
    rules = _filesystem_permissions(load_config()["harness"], FilesystemPermission)
    assert _check_fs_permission(rules, "read", "/skills/planning/intent/SKILL.md") == "allow"
    assert _check_fs_permission(rules, "read", "/config/harness.yaml") == "deny"
    assert _check_fs_permission(rules, "write", "/skills/planning/intent/SKILL.md") == "deny"


def test_empty_read_scope_can_be_locked_down_explicitly():
    assert _readonly_permissions([], FilesystemPermission) is None
    assert [str(rule.mode) for rule in _readonly_permissions([], FilesystemPermission, deny_when_empty=True)] == ["deny", "deny"]


def test_plan_and_delegation_tools_have_display_labels():
    labels = _tool_labels(load_config())
    assert labels["write_todos"] == "任务清单"
    assert labels["task"] == "委派子智能体"
    assert labels["show_evidence"] == "证据汇总"


def test_skill_sources_expand_group_names_to_paths():
    assert _skill_sources({"skills_dir": "skills"}, ["planning", "execution"]) == ["/skills/planning", "/skills/execution"]

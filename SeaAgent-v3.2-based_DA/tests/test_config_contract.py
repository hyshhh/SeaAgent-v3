from deepagents import FilesystemPermission

from config import load_config
from harness.runtime import _load_subagents
from harness.shared_state import SharedTaskState
from harness.state_tools import build_state_tools
from harness.tools import build_tools


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def test_harness_config_contract():
    config = load_config()
    assert config["harness"]["stream_mode"] == "updates"
    assert config["tools"]
    assert "agent" not in config["pipeline"]
    assert "enable_thinking" not in config["llm"]


def test_skills_read_channel_is_open_but_scoped():
    """原生渐进式披露要求 read_file 可达；读取面收敛在 skills 目录内。"""
    harness = load_config()["harness"]
    disabled = set(harness["disabled_deepagent_tools"])
    assert "read_file" not in disabled
    assert {"write_file", "edit_file", "execute"} <= disabled
    assert {"write_todos", "task"} <= disabled


def test_directory_exploration_tools_are_disabled():
    """ls/glob/grep 会诱使模型拿它们去找数据（实测试过 /data、glob /），技能路径本来就由提示词给出。"""
    disabled = set(load_config()["harness"]["disabled_deepagent_tools"])
    assert {"ls", "glob", "grep"} <= disabled


def test_three_phase_mode_is_declared_by_config():
    """三阶段协同是配置事实：规划 / 执行 / 反思各一个 Agent，规格写在 subagents.yaml。

    钉住的是「换形态必须改配置」——角色、工具面、技能组都由 yaml 决定，不写死在代码里。
    """
    harness = load_config()["harness"]
    assert harness["execution_mode"] == "evidence-state-multi-agent"
    assert str(harness["subagents_file"]).endswith("subagents.yaml")
    assert harness["shared_state_file"]
    assert int(harness["max_iterations"]) >= 1
    assert int(harness["reconnect_attempts"]) >= 0


def test_model_round_limit_is_gone():
    harness = load_config()["harness"]
    assert "model_calls_per_run" not in harness


def test_role_prompts_skill_reminders_and_budgets_are_explicit():
    config = load_config()
    tools = build_tools(config, _Service()) + build_state_tools(SharedTaskState(""))
    agents = _load_subagents(config, tools, FilesystemPermission)
    assert [agent["name"] for agent in agents] == ["planner", "executor", "reflector"]
    assert [agent["tool_budget"] for agent in agents] == [1, 6, 2]
    for agent in agents:
        assert "{question}" in agent["task_prompt"]
        assert agent["system_prompt"].strip()
        assert agent["skill_reminder"]["message"].strip()
        assert agent["skill_reminder"]["max_per_run"] >= 0
    harness = config["harness"]
    assert "tool_calls_per_run" not in harness
    assert "subagent_tool_calls" not in harness
    assert "skill_reminder_max_per_run" not in harness

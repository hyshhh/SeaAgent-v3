from config import load_config


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
    assert "/skills/**" in harness["readonly_paths"]


def test_delegation_is_switched_by_config_not_by_disabling_task():
    """task 不再靠禁用名单关闭（关掉它主从协同就没法委派）。

    官方语义：不挂 SubAgentMiddleware 的唯一办法是「禁用 general-purpose 且不传 subagents」，
    本项目的 general_purpose_subagent 一直是关的，所以 subagents_enabled 就是那个开关。
    """
    harness = load_config()["harness"]
    assert "task" not in harness["disabled_deepagent_tools"]
    assert isinstance(harness["subagents_enabled"], bool)
    assert str(harness["subagents_file"]).endswith("subagents.yaml")
    assert str(harness["planner_prompt_file"]).endswith("planner.md")


def test_model_round_limit_is_gone():
    harness = load_config()["harness"]
    assert "model_calls_per_run" not in harness
    assert int(harness["evidence_wrapup_max_nudges"]) >= 1


def test_tool_budget_has_a_stall_guard():
    """工具预算用完只是驳回调用，还得有「连续失败就收尾」这条，否则模型会一直空转。"""
    harness = load_config()["harness"]
    assert 1 <= int(harness["stall_guard_consecutive_errors"]) <= 8
    assert int(harness["tool_calls_per_run"]) < int(harness["tool_calls_per_thread"])

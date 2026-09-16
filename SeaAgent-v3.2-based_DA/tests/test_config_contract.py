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


def test_directory_exploration_tools_are_disabled():
    """ls/glob/grep 会诱使模型拿它们去找数据（实测试过 /data、glob /），技能路径本来就由提示词给出。"""
    disabled = set(load_config()["harness"]["disabled_deepagent_tools"])
    assert {"ls", "glob", "grep"} <= disabled


def test_three_phase_mode_is_switched_by_config_not_by_editing_prompts():
    """三阶段协同是配置事实：规划 / 执行 / 反思各一个从智能体，规格写在 subagents.yaml。

    钉住的是「换形态必须改配置」——不能靠往禁用名单里塞 task 来假装没有委派。
    """
    harness = load_config()["harness"]
    assert harness["execution_mode"] == "three-phase-subagents"
    assert harness["subagents_enabled"] is True
    assert str(harness["subagents_file"]).endswith("subagents.yaml")
    assert str(harness["planner_prompt_file"]).endswith("planner.md")
    assert "task" not in harness["disabled_deepagent_tools"]


def test_model_round_limit_is_gone():
    harness = load_config()["harness"]
    assert "model_calls_per_run" not in harness
    assert int(harness["evidence_wrapup_max_nudges"]) >= 1


def test_tool_budget_has_a_stall_guard():
    """工具预算用完只是驳回调用，还得有「连续失败就收尾」这条，否则模型会一直空转。"""
    harness = load_config()["harness"]
    assert 1 <= int(harness["stall_guard_consecutive_errors"]) <= 8
    assert int(harness["tool_calls_per_run"]) < int(harness["tool_calls_per_thread"])

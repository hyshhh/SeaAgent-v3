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
    assert int(harness["evidence_wrapup_max_nudges"]) >= 1


def test_tool_budget_is_declared():
    harness = load_config()["harness"]
    assert int(harness["tool_calls_per_run"]) == 40
    assert int(harness["subagent_tool_calls"]) == 6
    assert "tool_calls_per_thread" not in harness

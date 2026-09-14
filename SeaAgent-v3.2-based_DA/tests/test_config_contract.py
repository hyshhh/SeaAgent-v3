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
    assert {"write_file", "edit_file", "execute", "task"} <= disabled
    assert "/skills/**" in harness["readonly_paths"]


def test_model_round_limit_is_gone():
    harness = load_config()["harness"]
    assert "model_calls_per_run" not in harness
    assert int(harness["evidence_wrapup_max_nudges"]) >= 1

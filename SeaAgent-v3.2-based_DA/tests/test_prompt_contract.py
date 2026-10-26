from pathlib import Path

import yaml

from config import load_config
from config.settings import PROMPT_SETTING_SPECS

ROOT = Path(__file__).resolve().parents[1]


def test_all_business_tools_are_exposed_as_compact_workflows():
    config = load_config()
    names = {item["name"] for item in config["tools"]}
    assert names == {
        "recognize_tracks",
        "match_registry_tracks",
        "verify_vessel_images",
        "count_total_entities",
        "count_registry_entities",
        "count_unregistered_entities",
        "show_evidence",
    }
    assert not names & {"get_track", "get_frames", "get_clip", "get_registry", "list_registry", "add_registry_vessel"}


def test_prompts_and_skills_match_the_new_recognition_flow():
    config = load_config()
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    executor = next(item for item in role_config["agents"] if item["name"] == "executor")
    recognition = (ROOT / "skills/recognition/SKILL.md").read_text(encoding="utf-8")
    prompt = yaml.safe_load((ROOT / "config/prompts.yaml").read_text(encoding="utf-8"))["prompts"]

    assert "recognize_tracks" in executor["system_prompt"]
    assert "match_registry_tracks" in executor["system_prompt"]
    assert "image_path" in executor["system_prompt"]
    assert "candidate_keyframe_id" in recognition
    assert "verify_images" in prompt
    verify = next(item for item in config["tools"] if item["name"] == "verify_vessel_images")
    assert "reference_registry_reference_id" in verify["description"]


def test_prompt_settings_expose_only_configured_prompts():
    config = load_config()
    template = (ROOT / "web/templates/index.html").read_text(encoding="utf-8")
    settings_keys = {path.removeprefix("prompts.") for path in PROMPT_SETTING_SPECS}

    assert settings_keys == set(config["prompts"])
    assert 'data-setting="prompts.verify_images"' in template
    assert 'data-setting="prompts.verify_description"' not in template
    assert 'data-setting="prompts.verify_registry"' not in template


def test_supplied_aliases_follow_the_registry_assistance_step():
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    executor = next(item for item in role_config["agents"] if item["name"] == "executor")
    recognition = (ROOT / "skills/recognition/SKILL.md").read_text(encoding="utf-8")

    assert "目标名称（如“小红”）" in executor["system_prompt"]
    assert "nextStep" in executor["system_prompt"]
    assert "“小红”" in recognition


def test_count_skills_map_user_intent_to_configured_tools():
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    agents = {item["name"]: item for item in role_config["agents"]}
    expected = ("count_total_entities", "count_registry_entities", "count_unregistered_entities")
    planning = (ROOT / "skills/planning/acceptance/SKILL.md").read_text(encoding="utf-8")
    statistics = (ROOT / "skills/statistics/SKILL.md").read_text(encoding="utf-8")

    assert all(name in planning for name in expected)
    assert all(name in statistics for name in expected)
    assert "统计按用户要求的口径映射" in agents["planner"]["system_prompt"]
    assert "statistics 技能" in agents["executor"]["system_prompt"]
    assert "count_ships" not in planning


def test_planner_keeps_count_only_requests_out_of_recognition_flow():
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    planner = next(item for item in role_config["agents"] if item["name"] == "planner")
    acceptance = (ROOT / "skills/planning/acceptance/SKILL.md").read_text(encoding="utf-8")
    time_scope = (ROOT / "skills/planning/time-range/SKILL.md").read_text(encoding="utf-8")

    assert "统计任务只规划对应 count_* 工具，不得加入 recognize_tracks" in planner["system_prompt"]
    assert "Count-only plans must contain only the requested count tools, never `recognize_tracks`" in acceptance
    assert "including “所有时间”" in time_scope
    assert "Time scope is not a separate plan step" in time_scope


def test_time_scope_is_optional_and_defaults_to_all_stored_tracks():
    tools = {item["name"]: item for item in load_config()["tools"]}
    for name in ("recognize_tracks", "count_total_entities", "count_registry_entities", "count_unregistered_entities"):
        assert "省略 time_range 表示全部监控时间" in tools[name]["description"]

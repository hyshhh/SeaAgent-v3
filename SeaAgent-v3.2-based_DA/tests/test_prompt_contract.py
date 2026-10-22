from pathlib import Path

import yaml

from config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_no_time_means_all_stored_monitoring_time_across_prompt_sources():
    config = load_config()
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    roles = {agent["name"]: agent for agent in role_config["agents"]}
    get_track = next(tool for tool in config["tools"] if tool["name"] == "get_track")
    time_skill = (ROOT / "skills/planning/time-range/SKILL.md").read_text(encoding="utf-8")
    query_skill = (ROOT / "skills/track/query/SKILL.md").read_text(encoding="utf-8")

    assert "省略 time_range=全部监控时间" in get_track["description"]
    assert "按全部已存轨迹规划" in roles["planner"]["system_prompt"]
    assert "query all stored monitoring time" in time_skill
    assert "all stored monitoring time" in query_skill
    query_rules = (ROOT / "skills/execution/query-rules/SKILL.md").read_text(encoding="utf-8")
    assert "ask the user for one" not in query_rules


def test_model_prompts_match_supported_tool_fields_and_gallery_records():
    config = load_config()
    descriptions = {tool["name"]: tool["description"] for tool in config["tools"]}
    prompt = yaml.safe_load((ROOT / "config/prompts.yaml").read_text(encoding="utf-8"))["prompts"]
    role_config = yaml.safe_load((ROOT / "config/subagents.yaml").read_text(encoding="utf-8"))
    roles = {agent["name"]: agent["system_prompt"] for agent in role_config["agents"]}

    assert "keyframes" in descriptions["match_text"]
    assert "registryReferences" in descriptions["match_text"]
    assert "路径字符串" in descriptions["match_text"]
    assert "match、mismatch、uncertain 之一" in prompt["verify_description"]
    assert "match、mismatch、uncertain 之一" in prompt["verify_registry"]
    assert "restated_question" not in roles["planner"]
    assert "how_to_check" not in roles["planner"]
    assert "assessment 对象的 final_answer" in roles["reflector"]

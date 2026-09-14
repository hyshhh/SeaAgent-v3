"""主从协同装配：主智能体只留白名单工具，从智能体按任务收窄工具面并自带输出契约。"""
import pytest

from config import load_config
from harness.runtime import _load_subagents
from harness.tools import build_tools


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def _tools():
    return build_tools(load_config(), _Service())


def test_subagents_are_built_from_config_with_narrow_tool_sets():
    config = load_config()
    subagents, master_tools = _load_subagents(config, _tools())

    assert [spec["name"] for spec in subagents] == ["track_scout", "registry_checker", "visual_prover"]
    by_name = {spec["name"]: spec for spec in subagents}
    assert [tool.name for tool in by_name["track_scout"]["tools"]] == ["get_track", "get_frames", "dedup_tracks"]
    assert [tool.name for tool in by_name["visual_prover"]["tools"]] == ["match_image", "verify_target", "get_clip"]
    assert master_tools == ["show_evidence"]


def test_every_subagent_carries_a_prompt_and_skills():
    subagents, _ = _load_subagents(load_config(), _tools())
    for spec in subagents:
        # 隔离模式下从智能体看不到对话历史，system_prompt 就是它唯一的规范来源
        assert spec["system_prompt"].strip(), f"{spec['name']} 缺少 system_prompt"
        assert spec["description"].strip(), f"{spec['name']} 缺少 description"
        assert spec["skills"] == ["/skills"], f"{spec['name']} 未声明技能目录（不声明就一条技能都看不到）"


def test_domain_tools_are_split_between_master_and_subagents():
    """下放必须是真的：主智能体保留的证据工具不能同时出现在从智能体工具表里。"""
    subagents, master_tools = _load_subagents(load_config(), _tools())
    handed_down = {tool.name for spec in subagents for tool in spec["tools"]}
    assert set(master_tools) & handed_down == set()
    # 证据工具必须留在主智能体：从智能体的内部调用不进主事件流，下放会让证据面板永远为空
    assert "show_evidence" in master_tools
    assert "show_evidence" not in handed_down


def test_unknown_tool_name_fails_loudly():
    config = load_config()
    config["harness"]["subagents_file"] = "config/subagents.yaml"
    subagents, _ = _load_subagents(config, _tools())
    assert subagents  # 正常路径先跑通

    broken = {"subagents": [{"name": "x", "description": "d", "system_prompt": "s", "tools": ["no_such_tool"]}]}
    import yaml, tempfile, pathlib
    path = pathlib.Path(tempfile.mkdtemp()) / "broken.yaml"
    path.write_text(yaml.safe_dump(broken, allow_unicode=True), encoding="utf-8")
    config["harness"]["subagents_file"] = str(path)
    with pytest.raises(ValueError, match="no_such_tool"):
        _load_subagents(config, _tools())

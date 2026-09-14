"""主从协同装配：主智能体只留白名单工具，从智能体按任务收窄工具面、技能组与返回契约。"""
import pathlib
import tempfile

import pytest
import yaml
from langchain_core.messages import AIMessage, ToolMessage

from config import load_config
from harness.runtime import _load_subagents, _Trace
from harness.subagent_schemas import RegistryCheckFindings, TrackScoutFindings, VisualProofFindings
from harness.tools import build_tools


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def _tools():
    return build_tools(load_config(), _Service())


def _specs():
    return _load_subagents(load_config(), _tools())


def test_subagents_are_built_from_config_with_narrow_tool_sets():
    subagents, master_tools, master_skills = _specs()

    assert [spec["name"] for spec in subagents] == ["track_scout", "registry_checker", "visual_prover"]
    by_name = {spec["name"]: spec for spec in subagents}
    assert [tool.name for tool in by_name["track_scout"]["tools"]] == ["get_track", "get_frames", "dedup_tracks"]
    assert [tool.name for tool in by_name["visual_prover"]["tools"]] == ["match_image", "verify_target", "get_clip"]
    assert master_tools == ["show_evidence"]
    assert master_skills == ["planning", "answer"]


def test_each_subagent_has_its_own_skill_group_and_return_contract():
    subagents, _, _ = _specs()
    by_name = {spec["name"]: spec for spec in subagents}

    assert by_name["track_scout"]["skills"] == ["/skills/track"]
    assert by_name["registry_checker"]["skills"] == ["/skills/registry"]
    assert by_name["visual_prover"]["skills"] == ["/skills/visual"]
    assert by_name["track_scout"]["response_format"] is TrackScoutFindings
    assert by_name["registry_checker"]["response_format"] is RegistryCheckFindings
    assert by_name["visual_prover"]["response_format"] is VisualProofFindings


def test_every_subagent_carries_a_prompt_and_description():
    subagents, _, _ = _specs()
    for spec in subagents:
        # 隔离模式下从智能体看不到对话历史，system_prompt 就是它唯一的规范来源
        assert spec["system_prompt"].strip(), f"{spec['name']} 缺少 system_prompt"
        assert spec["description"].strip(), f"{spec['name']} 缺少 description"


def test_domain_tools_are_split_between_master_and_subagents():
    """下放必须是真的：主智能体保留的证据工具不能同时出现在从智能体工具表里。"""
    subagents, master_tools, _ = _specs()
    handed_down = {tool.name for spec in subagents for tool in spec["tools"]}
    assert set(master_tools) & handed_down == set()
    # 证据工具必须留在主智能体：从智能体的内部调用不进主事件流，下放会让证据面板永远为空
    assert "show_evidence" in master_tools
    assert "show_evidence" not in handed_down


def _write_broken(config, spec):
    path = pathlib.Path(tempfile.mkdtemp()) / "broken.yaml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    config["harness"]["subagents_file"] = str(path)
    return config


def test_unknown_tool_name_fails_loudly():
    config = _write_broken(load_config(), {"subagents": [{"name": "x", "description": "d", "system_prompt": "s", "tools": ["no_such_tool"]}]})
    with pytest.raises(ValueError, match="no_such_tool"):
        _load_subagents(config, _tools())


def test_unknown_response_format_fails_loudly():
    config = _write_broken(load_config(), {"subagents": [{"name": "x", "description": "d", "system_prompt": "s", "tools": [], "response_format": "NoSuchSchema"}]})
    with pytest.raises(ValueError, match="NoSuchSchema"):
        _load_subagents(config, _tools())


def _trace():
    events = []
    trace = _Trace(
        events.append,
        event_limit=4000,
        evidence_tool="show_evidence",
        tool_labels={"task": "委派子智能体", "get_track": "轨迹记忆"},
        subagent_tools={"get_track": "track_scout"},
    )
    return trace, events


def test_subagent_frames_are_tagged_and_do_not_become_the_answer():
    """subgraphs=True 会把子智能体的帧吐给主事件流，它们必须被认领且不能污染回答。"""
    trace, events = _trace()
    trace.consume({"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "查轨迹", "subagent_type": "track_scout"}, "id": "call-task-1"}])]}})
    namespace = ("tools:some-runtime-id",)  # 命名空间里是 LangGraph 的任务 id，不是 tool_call_id
    trace.consume({"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "get_track", "args": {"time_range": "15:30-15:40"}, "id": "call-sub-1"}])]}}, namespace)
    trace.consume({"model": {"messages": [AIMessage(content="子智能体中间结论：看到两艘")]}}, namespace)
    trace.consume({"tools": {"messages": [ToolMessage(content="子智能体报告", name="task", tool_call_id="call-task-1")]}})
    trace.consume({"model": {"messages": [AIMessage(content="主智能体最终回答")]}})

    tagged = [(event["type"], event.get("agent", "")) for event in events]
    assert ("tool_start", "track_scout") in tagged, "子智能体的工具调用必须被认领"
    assert ("model", "track_scout") in tagged, "子智能体的中间文本要展示但带标记"

    result = trace.result("t1", {"harness": {"output": {"answer_field": "answer", "state_field": "state", "evidence_field": "evidence"}}})
    assert result["answer"] == "主智能体最终回答"
    assert "两艘" not in result["answer"], "子智能体的中间文本不能污染最终回答"
    assert [record.get("agent") for record in result["tool_records"] if record["tool"] == "get_track"] == ["track_scout"]


def test_unnamed_subagent_frames_are_tagged_generically():
    """并发委派且工具面重叠时认不出归属，就如实写成「子智能体」，不能记到错误的人头上。"""
    trace, events = _trace()
    trace.consume({"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "a", "subagent_type": "track_scout"}, "id": "t1"}])]}})
    trace.consume({"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "task", "args": {"description": "b", "subagent_type": "visual_prover"}, "id": "t2"}])]}})
    trace.consume({"model": {"messages": [AIMessage(content="", tool_calls=[{"name": "unknown_tool", "args": {}, "id": "u1"}])]}}, ("tools:another-id",))
    assert any(event.get("agent") == "子智能体" for event in events)

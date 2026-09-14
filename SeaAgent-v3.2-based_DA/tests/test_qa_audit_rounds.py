"""审计表主键：会话第几轮必须进 round_id，否则第二轮会覆盖第一轮。"""
import os
import tempfile
from types import SimpleNamespace

from agent import controller as controller_module
from agent.controller import AgentController
from memory import MemoryRepository


def _paths(tmp_path):
    return {name: str(tmp_path / f"{name}.csv") for name in ("tracks_csv", "keyframes_csv", "registry_csv", "registry_images_csv", "qa_sessions_csv", "qa_rounds_csv", "qa_evidence_csv")}


class _TwoRoundRuntime:
    """每轮都从 1 开始编轮次，模拟 runtime 的单轮内计数。"""

    def __init__(self, config, service, model=None, event_handler=None):
        pass

    def run(self, question, thread_id=None, cancel=None, **_kwargs):
        return {
            "answer": f"答：{question}",
            "state": "completed",
            "evidence": {},
            "tool_records": [
                {"id": f"call-{question}", "round": 1, "tool": "get_track", "status": "completed"},
                {"id": f"call-{question}-evidence", "round": 2, "tool": "show_evidence", "status": "completed"},
            ],
            "rounds": [{"round": 1, "toolChain": ["get_track"]}, {"round": 2, "toolChain": ["show_evidence"]}],
            "tool_chain": ["get_track", "show_evidence"],
        }

    def close(self):
        return None


def test_each_turn_keeps_its_own_round_and_evidence_rows(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    repository = MemoryRepository({"paths": paths})
    monkeypatch.setattr(controller_module, "SeaVideoHarness", _TwoRoundRuntime)
    agent = AgentController(config={"harness": {}, "paths": paths}, repository=repository, tools=object())

    first = agent.answer("第一轮问题")
    agent.answer("第二轮追问", first["sessionId"])

    rounds = repository.qa_rounds.rows()
    assert len(rounds) == 4, "两轮各 2 次工具决策，审计表应有 4 行"
    assert {row["round_id"] for row in rounds} == {
        f"{first['sessionId']}-round-1-1",
        f"{first['sessionId']}-round-1-2",
        f"{first['sessionId']}-round-2-1",
        f"{first['sessionId']}-round-2-2",
    }
    assert len(repository.qa_evidence.rows()) == 4
    assert {row["session_id"] for row in rounds} == {first["sessionId"]}


def test_deleting_a_session_still_removes_every_turn(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    repository = MemoryRepository({"paths": paths})
    monkeypatch.setattr(controller_module, "SeaVideoHarness", _TwoRoundRuntime)
    agent = AgentController(config={"harness": {}, "paths": paths}, repository=repository, tools=object())

    first = agent.answer("第一轮问题")
    agent.answer("第二轮追问", first["sessionId"])
    assert repository.delete_session(first["sessionId"]) is True
    assert repository.qa_rounds.rows() == []
    assert repository.qa_evidence.rows() == []

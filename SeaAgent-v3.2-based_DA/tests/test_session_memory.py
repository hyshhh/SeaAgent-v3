"""会话记忆：仓库层的轮次存档，以及控制器把「追问」接到同一个 thread 上。"""
import pytest
from fastapi import HTTPException

from agent import controller as controller_module
from agent.controller import AgentController
from memory import MemoryRepository
from web.models import AgentQuery
from web.routes import agent_api


def _paths(tmp_path):
    return {
        "tracks_csv": str(tmp_path / "tracks.csv"),
        "keyframes_csv": str(tmp_path / "track_keyframes.csv"),
        "registry_csv": str(tmp_path / "registry.csv"),
        "registry_images_csv": str(tmp_path / "registry_reference_images.csv"),
        "qa_sessions_csv": str(tmp_path / "qa_sessions.csv"),
        "qa_rounds_csv": str(tmp_path / "qa_rounds.csv"),
        "qa_evidence_csv": str(tmp_path / "qa_evidence.csv"),
    }


def _repository(tmp_path):
    return MemoryRepository({"paths": _paths(tmp_path)})


class _FakeRuntime:
    """只记录 thread_id：会话续接的全部证据就是这个 id 是否被复用。"""

    threads = []

    def __init__(self, config, service, model=None, event_handler=None):
        self.event_handler = event_handler

    def run(self, question, thread_id=None, **_kwargs):
        type(self).threads.append(thread_id)
        return {"answer": f"回答：{question}", "state": "completed", "evidence": {"shownKeyframeIds": ["kf-1"]}, "tool_records": [], "rounds": [], "tool_chain": ["show_evidence"]}

    def close(self):
        return None


@pytest.fixture
def agent(tmp_path, monkeypatch):
    repository = _repository(tmp_path)
    _FakeRuntime.threads = []
    monkeypatch.setattr(controller_module, "SeaVideoHarness", _FakeRuntime)
    return AgentController(config={"harness": {}, "paths": _paths(tmp_path)}, repository=repository, tools=object())


def test_first_question_creates_a_session(agent):
    result = agent.answer("15:30 到 15:40 有几艘船？")
    assert result["turnIndex"] == 1
    session = agent.repository.get_session(result["sessionId"])
    assert session["title"] == "15:30 到 15:40 有几艘船？"
    assert session["turnCount"] == 1
    assert session["turns"][0]["answer"] == "回答：15:30 到 15:40 有几艘船？"
    assert session["turns"][0]["evidence"]["shownKeyframeIds"] == ["kf-1"]


def test_follow_up_reuses_the_same_thread_and_appends_a_turn(agent):
    first = agent.answer("有哪些在库船？")
    second = agent.answer("那未在库的呢？", first["sessionId"])
    assert _FakeRuntime.threads == [first["sessionId"], first["sessionId"]]
    assert second["sessionId"] == first["sessionId"]
    assert second["turnIndex"] == 2
    session = agent.repository.get_session(first["sessionId"])
    assert [turn["question"] for turn in session["turns"]] == ["有哪些在库船？", "那未在库的呢？"]


def test_unknown_session_id_falls_back_to_a_new_session(agent):
    first = agent.answer("第一个问题")
    agent.repository.clear_qa_memory()
    revived = agent.answer("清空之后的问题", first["sessionId"])
    assert agent.repository.get_session(first["sessionId"])["turnCount"] == 1
    assert revived["turnIndex"] == 1


def test_sessions_are_listed_newest_first(tmp_path):
    repository = _repository(tmp_path)
    repository.create_session("session-old", "旧问题")
    repository.create_session("session-new", "新问题")
    repository.append_turn("session-old", "旧追问", {"answerText": "答", "state": "completed"})
    listed = repository.list_sessions()
    assert [item["sessionId"] for item in listed] == ["session-old", "session-new"]
    assert listed[0]["turnCount"] == 1
    assert "turns" not in listed[0]


def test_deleting_a_session_removes_its_rounds_and_evidence(tmp_path):
    repository = _repository(tmp_path)
    repository.create_session("session-a", "问题")
    repository.add_round("session-a-round-1", "session-a", {}, {})
    repository.add_evidence("session-a-round-1-evidence-1", "session-a-round-1", {}, {})
    assert repository.delete_session("session-a") is True
    assert repository.get_session("session-a") is None
    assert repository.qa_rounds.rows() == []
    assert repository.qa_evidence.rows() == []


@pytest.mark.asyncio
async def test_session_endpoints_expose_the_transcript(tmp_path):
    repository = _repository(tmp_path)
    repository.create_session("session-abc", "问题")
    request = type("R", (), {"app": type("A", (), {"state": type("S", (), {"repository": repository})()})()})()

    listed = await agent_api.list_agent_sessions(request)
    assert listed["total"] == 1 and listed["sessions"][0]["sessionId"] == "session-abc"
    assert (await agent_api.get_agent_session("session-abc", request))["title"] == "问题"
    assert (await agent_api.delete_agent_session("session-abc", request))["success"] is True
    with pytest.raises(HTTPException):
        await agent_api.get_agent_session("session-abc", request)


def test_agent_query_accepts_both_session_id_spellings():
    assert AgentQuery(question="问题", sessionId="session-abc").session_id == "session-abc"
    assert AgentQuery(question="问题", session_id="session-abc").session_id == "session-abc"
    assert AgentQuery(question="问题").session_id is None

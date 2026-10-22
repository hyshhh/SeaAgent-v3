"""对话存档仅保留 QA，运行轨迹不进入历史记忆。"""

from agent import controller as controller_module
from agent.controller import AgentController
from memory import MemoryRepository


def _paths(tmp_path):
    return {
        name: str(tmp_path / f"{name}.csv")
        for name in (
            "tracks_csv", "keyframes_csv", "registry_csv", "registry_images_csv",
            "qa_sessions_csv",
        )
    }


class _RuntimeWithTrace:
    thread_ids = []

    def __init__(self, config, service, model=None, event_handler=None):
        pass

    def run(self, question, thread_id=None, **_kwargs):
        type(self).thread_ids.append(thread_id)
        return {
            "answer": f"答：{question}",
            "state": "completed",
            "evidence": {"shownKeyframeIds": ["kf-1"]},
            "tool_records": [{"tool": "get_track", "result": {"trackIds": ["12"]}}],
            "tool_chain": ["get_track"],
            "shared_state": {"plan": [{"id": "scan"}]},
        }

    def close(self):
        pass


def test_each_question_has_a_fresh_thread_and_history_stores_only_qa(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    repository = MemoryRepository({"paths": paths})
    monkeypatch.setattr(controller_module, "SeaVideoHarness", _RuntimeWithTrace)
    agent = AgentController(
        config={"harness": {}, "paths": paths}, repository=repository, tools=object()
    )

    first = agent.answer("第一条问题")
    agent.answer("第二条问题", first["sessionId"])

    turns = repository.get_session(first["sessionId"])["turns"]
    assert len(_RuntimeWithTrace.thread_ids) == 2
    assert len(set(_RuntimeWithTrace.thread_ids)) == 2
    assert all(run_id.startswith("qa-") for run_id in _RuntimeWithTrace.thread_ids)
    assert [turn["question"] for turn in turns] == ["第一条问题", "第二条问题"]
    assert all(set(turn) == {"question", "answer", "state", "partial", "createdAt"} for turn in turns)
    assert not hasattr(repository, "qa_rounds")
    assert not hasattr(repository, "qa_evidence")


def test_deleting_a_session_removes_all_saved_qa(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    repository = MemoryRepository({"paths": paths})
    monkeypatch.setattr(controller_module, "SeaVideoHarness", _RuntimeWithTrace)
    agent = AgentController(
        config={"harness": {}, "paths": paths}, repository=repository, tools=object()
    )

    first = agent.answer("第一条问题")
    agent.answer("第二条问题", first["sessionId"])
    assert repository.delete_session(first["sessionId"]) is True
    assert repository.get_session(first["sessionId"]) is None

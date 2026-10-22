"""会话只存 QA 对话；每个问题使用独立的 Agent 运行标识。"""
import csv
import json
import threading
from types import SimpleNamespace

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


def test_error_result_keeps_context_and_distinct_user_messages():
    controller = AgentController(config={"harness": {}}, repository=object(), tools=object())

    answer_error = controller._error_result(
        "session-1", "查一艘船", RuntimeError("failed"), "暂时无法完成视频检索。"
    )
    resume_error = controller._error_result(
        "session-1", "查一艘船", RuntimeError("failed"), "确认后仍无法完成这一步。"
    )

    assert answer_error["question"] == resume_error["question"] == "查一艘船"
    assert answer_error["answerText"] == "暂时无法完成视频检索。"
    assert resume_error["answerText"] == "确认后仍无法完成这一步。"
    assert answer_error["error"] == resume_error["error"] == "failed"
    assert answer_error["errorType"] == resume_error["errorType"] == "RuntimeError"


def test_first_question_creates_a_session(agent):
    result = agent.answer("15:30 到 15:40 有几艘船？")
    assert result["turnIndex"] == 1
    session = agent.repository.get_session(result["sessionId"])
    assert session["title"] == "15:30 到 15:40 有几艘船？"
    assert session["turnCount"] == 1
    assert session["turns"][0]["answer"] == "回答：15:30 到 15:40 有几艘船？"
    assert set(session["turns"][0]) == {"question", "answer", "state", "partial", "createdAt"}


def test_follow_up_is_a_fresh_run_and_appends_only_qa(agent):
    first = agent.answer("有哪些在库船？")
    second = agent.answer("那未在库的呢？", first["sessionId"])
    assert len(set(_FakeRuntime.threads)) == 2
    assert all(run_id.startswith("qa-") for run_id in _FakeRuntime.threads)
    assert second["sessionId"] == first["sessionId"]
    assert second["turnIndex"] == 2
    session = agent.repository.get_session(first["sessionId"])
    assert [turn["question"] for turn in session["turns"]] == ["有哪些在库船？", "那未在库的呢？"]
    assert all(set(turn) == {"question", "answer", "state", "partial", "createdAt"} for turn in session["turns"])


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


def test_deleting_a_session_removes_its_qa_transcript(tmp_path):
    repository = _repository(tmp_path)
    repository.create_session("session-a", "问题")
    repository.append_turn("session-a", "问题", {"state": "completed", "answerText": "回答"})
    assert repository.delete_session("session-a") is True
    assert repository.get_session("session-a") is None


@pytest.mark.asyncio
async def test_session_endpoints_expose_the_transcript(tmp_path):
    repository = _repository(tmp_path)
    repository.create_session("session-abc", "问题")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repository=repository, agent_runs={})))

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


def test_legacy_session_rows_are_migrated_on_read(tmp_path):
    """旧格式（一问一会话）的行必须能读出标题与轮次，否则会话栏全是「未命名会话」。"""
    paths = _paths(tmp_path)
    with open(paths["qa_sessions_csv"], "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session_id", "query_info", "final_result"))
        writer.writeheader()
        writer.writerow({
            "session_id": "session-old",
            "query_info": json.dumps({"question": "15:30 有几艘船？"}, ensure_ascii=False),
            "final_result": json.dumps({"answerText": "三艘", "state": "completed"}, ensure_ascii=False),
        })
    repository = MemoryRepository({"paths": paths})

    session = repository.get_session("session-old")
    assert session["title"] == "15:30 有几艘船？"
    assert session["turnCount"] == 1
    assert session["turns"][0]["answer"] == "三艘"
    assert repository.list_sessions()[0]["turnCount"] == 1


def test_appending_to_a_legacy_session_keeps_the_migrated_turn(tmp_path):
    paths = _paths(tmp_path)
    with open(paths["qa_sessions_csv"], "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session_id", "query_info", "final_result"))
        writer.writeheader()
        writer.writerow({"session_id": "session-old", "query_info": json.dumps({"question": "旧问题"}, ensure_ascii=False), "final_result": "{}"})
    repository = MemoryRepository({"paths": paths})

    assert repository.append_turn("session-old", "追问", {"answerText": "答", "state": "completed"}) == 2
    assert [turn["question"] for turn in repository.get_session("session-old")["turns"]] == ["旧问题", "追问"]


def _write_legacy_sessions(path, entries):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session_id", "query_info", "final_result"))
        writer.writeheader()
        for session_id, question, answer in entries:
            writer.writerow({
                "session_id": session_id,
                "query_info": json.dumps({"question": question}, ensure_ascii=False),
                "final_result": json.dumps({"answerText": answer, "state": "completed"}, ensure_ascii=False),
            })


def test_legacy_sessions_survive_the_first_write(tmp_path):
    """核心回归：旧列必须在任何写入之前迁移完，否则第一次写入就把历史会话抹成空壳。"""
    paths = _paths(tmp_path)
    _write_legacy_sessions(paths["qa_sessions_csv"], [("session-old", "舷号 0857 出现过吗？", "出现过一次")])
    repository = MemoryRepository({"paths": paths})

    repository.create_session("session-new", "新问题")

    reopened = MemoryRepository({"paths": paths})
    titles = {item["sessionId"]: (item["title"], item["turnCount"]) for item in reopened.list_sessions()}
    assert titles["session-old"] == ("舷号 0857 出现过吗？", 1)
    assert titles["session-new"] == ("新问题", 0)


def test_unrecoverable_session_shells_are_dropped_on_load(tmp_path):
    """既无标题也无轮次的行只会在会话栏里制造「未命名会话」，启动时清掉。"""
    paths = _paths(tmp_path)
    with open(paths["qa_sessions_csv"], "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session_id", "title", "created_at", "updated_at", "turns"))
        writer.writeheader()
        writer.writerow({"session_id": "session-shell", "title": "", "created_at": "", "updated_at": "", "turns": "[]"})
        writer.writerow({"session_id": "session-real", "title": "真实问题", "created_at": "", "updated_at": "", "turns": "[]"})

    repository = MemoryRepository({"paths": paths})

    assert [item["sessionId"] for item in repository.list_sessions()] == ["session-real"]
    assert repository.get_session("session-shell") is None


def test_cancelled_run_is_kept_as_a_stopped_turn(agent, monkeypatch):
    """被停掉的那一轮仍留在会话里：问题不该凭空消失，回答位置写明已停止。"""

    class _CancellingRuntime(_FakeRuntime):
        def run(self, question, thread_id=None, cancel=None, **_kwargs):
            if cancel is not None:
                cancel.set()
            type(self).threads.append(thread_id)
            return {"answer": "", "state": "cancelled", "evidence": {}, "tool_records": [], "rounds": [], "tool_chain": []}

    monkeypatch.setattr(controller_module, "SeaVideoHarness", _CancellingRuntime)
    result = agent.answer("会被停掉的问题")

    assert result["state"] == "cancelled"
    assert result["success"] is False
    session = agent.repository.get_session(result["sessionId"])
    assert session["turnCount"] == 1
    assert session["turns"][0]["question"] == "会被停掉的问题"
    assert session["turns"][0]["state"] == "cancelled"
    assert session["turns"][0]["answer"] == ""


@pytest.mark.asyncio
async def test_stop_endpoint_signals_the_running_session():
    registry = {"session-abc": threading.Event()}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent_runs=registry)))

    assert (await agent_api.stop_agent_run("session-abc", request))["success"] is True
    assert registry["session-abc"].is_set()
    assert (await agent_api.stop_agent_run("session-none", request))["success"] is False


def test_history_keeps_only_qa_and_temporary_confirmation_resume_data(tmp_path):
    repository = _repository(tmp_path)
    confirmation = {"actions": [{"tool": "add_registry_vessel", "arguments": {"hull_number": "003"}}]}
    repository.append_turn("t", "add 003", {
        "state": "awaiting_confirmation", "confirmation": confirmation,
        "sharedState": {"plan": [{"id": "write", "content": "add ship"}], "step_status": {"write": "in_progress"}},
    }, runtime_id="qa-pending-1")
    turn = repository.get_session("t")["turns"][0]
    assert turn["confirmation"] == confirmation
    assert turn["runtimeId"] == "qa-pending-1"
    assert "plan" not in turn and "sharedState" not in turn


def test_history_preserves_a_converged_answer_as_partial(tmp_path):
    repository = _repository(tmp_path)
    repository.append_turn("t", "verify", {
        "state": "completed", "partial": True, "answerText": "已有 kf1，目标尚未验证",
        "evidence": {"shownKeyframeIds": ["kf1"]},
    })
    turn = _repository(tmp_path).get_session("t")["turns"][0]
    assert turn["partial"] is True
    assert turn["answer"] == "已有 kf1，目标尚未验证"
    assert "evidence" not in turn and "toolChain" not in turn


def test_resume_updates_the_same_pending_qa_without_appending_a_turn(agent, monkeypatch):
    class _ConfirmRuntime(_FakeRuntime):
        def run(self, question, thread_id=None, **_kwargs):
            type(self).threads.append(thread_id)
            return {"answer": "", "state": "awaiting_confirmation", "confirmation": {"actions": []}}

        def resume(self, decision, thread_id, **_kwargs):
            type(self).threads.append(thread_id)
            return {"answer": "已确认", "state": "completed", "shared_state": {"question": "写入船舶 003"}}

    monkeypatch.setattr(controller_module, "SeaVideoHarness", _ConfirmRuntime)
    pending = agent.answer("写入船舶 003")
    resumed = agent.resume("approve", pending["sessionId"])

    turns = agent.repository.get_session(pending["sessionId"])["turns"]
    assert len(turns) == 1
    assert turns[0]["question"] == "写入船舶 003"
    assert turns[0]["answer"] == "已确认"
    assert "runtimeId" not in turns[0]
    assert resumed["turnIndex"] == pending["turnIndex"] == 1
    assert _ConfirmRuntime.threads[0] == _ConfirmRuntime.threads[1]


def test_only_pending_confirmation_keeps_temporary_run_state(agent, monkeypatch):
    class _CleanupRuntime(_FakeRuntime):
        discarded = []
        pending = False

        def run(self, question, thread_id=None, **_kwargs):
            type(self).threads.append(thread_id)
            state = "awaiting_confirmation" if self.pending else "completed"
            return {
                "answer": "待确认" if self.pending else "完成",
                "state": state,
                "confirmation": {"actions": []} if self.pending else None,
            }

        def discard_run(self, thread_id):
            type(self).discarded.append(thread_id)

    monkeypatch.setattr(controller_module, "SeaVideoHarness", _CleanupRuntime)
    completed = agent.answer("普通问题")
    _CleanupRuntime.pending = True
    pending = agent.answer("需要确认的问题", completed["sessionId"])

    assert _CleanupRuntime.discarded == _CleanupRuntime.threads[:1]
    assert pending["state"] == "awaiting_confirmation"
    assert agent.repository.get_session(pending["sessionId"])["turns"][-1]["runtimeId"] == _CleanupRuntime.threads[-1]


def test_planner_clarification_stops_and_user_reply_continues_same_qa(agent, monkeypatch):
    class _ClarifyingRuntime(_FakeRuntime):
        def run(self, question, thread_id=None, **_kwargs):
            type(self).threads.append(thread_id)
            if "用户补充信息：" not in question:
                return {"answer": "请补充轨迹时间范围。", "state": "awaiting_input"}
            return {"answer": "已按补充范围查到 2 条轨迹。", "state": "completed"}

    monkeypatch.setattr(controller_module, "SeaVideoHarness", _ClarifyingRuntime)
    first = agent.answer("查询目标船舶")
    completed = agent.answer("2025-10-21 14:00 到 14:10", first["sessionId"])

    turns = agent.repository.get_session(first["sessionId"])["turns"]
    assert first["state"] == "awaiting_input"
    assert completed["state"] == "completed"
    assert completed["question"] == "查询目标船舶"
    assert len(turns) == 1
    assert turns[0]["answer"] == "已按补充范围查到 2 条轨迹。"
    assert turns[0]["clarificationResponse"] == "2025-10-21 14:00 到 14:10"
    assert len(set(_ClarifyingRuntime.threads)) == 2


def test_old_transcripts_drop_saved_tool_history_on_migration(tmp_path):
    paths = _paths(tmp_path)
    paths["qa_sessions_csv"] = str(tmp_path / "qa_sessions.csv")
    with open(paths["qa_sessions_csv"], "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("session_id", "title", "created_at", "updated_at", "turns"))
        writer.writeheader()
        writer.writerow({
            "session_id": "old",
            "title": "问题",
            "created_at": "",
            "updated_at": "",
            "turns": json.dumps([{
                "question": "问题", "answer": "回答", "state": "completed",
                "toolChain": ["get_track"], "evidence": {"shownKeyframeIds": ["kf1"]},
                "plan": {"steps": []},
            }]),
        })

    turn = MemoryRepository({"paths": paths}).get_session("old")["turns"][0]

    assert set(turn) == {"question", "answer", "state", "partial", "createdAt"}

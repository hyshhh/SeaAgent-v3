"""QA 记忆落库闭环与决策计量（缺陷①③）的回归测试。

① controller._persist_qa_memory 把 LangGraph 的 rounds/tool_records
   真正写入 qa_rounds/qa_evidence（此前两张表从未被生产代码写入）；
③ controller._build_decision_metrics 统计模型决策 vs 确定性守卫/兜底。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import AgentController
from memory import MemoryRepository


class _FakeLLM:
    settings: dict = {}


class _FakeEmbedder:
    dimension = 2048


class _FakeVectors:
    pass


class _FakeTools:
    def execute(self, name, arguments):
        return {"ok": True}


def _make_config(root: Path) -> dict:
    return {
        "paths": {
            "tracks_csv": str(root / "memory" / "tracks.csv"),
            "keyframes_csv": str(root / "memory" / "track_keyframes.csv"),
            "qa_sessions_csv": str(root / "memory" / "qa_sessions.csv"),
            "qa_rounds_csv": str(root / "memory" / "qa_rounds.csv"),
            "qa_evidence_csv": str(root / "memory" / "qa_evidence.csv"),
            "registry_csv": str(root / "registry" / "registry.csv"),
            "registry_images_csv": str(root / "registry" / "registry_reference_images.csv"),
            "memory_settings_json": str(root / "memory" / "settings.json"),
            "keyframe_dir": str(root / "memory" / "keyframes"),
            "trajectory_dir": str(root / "memory" / "trajectories"),
            "clip_dir": str(root / "memory" / "clips"),
        }
    }


def _make_controller(root: Path) -> AgentController:
    config = _make_config(root)
    repository = MemoryRepository(config)
    controller = AgentController(
        config,
        repository=repository,
        tools=_FakeTools(),
        llm=_FakeLLM(),
        embedder=_FakeEmbedder(),
        vectors=_FakeVectors(),
    )
    controller.session_id = "session-test"
    return controller


def _state_with_rounds_and_tools() -> dict:
    """构造一轮 replan（模型决策）与一轮 finish（守卫决策）的 LangGraph 状态。"""
    return {
        "question": "有哪些在库船？",
        "rounds": [
            {
                "round": 1,
                "planHint": "getTrack → getFrames → listRegistry → matchImage",
                "observation": "计划：tracks=12 条轨迹；结果：registry=8 个库项",
                "reflection": {
                    "handoff": "plan",
                    "replan": True,
                    "state": "replan",
                    "decisionSource": "model",
                    "reason": "需要视觉补洞",
                    "nextActionSpec": {"requiredCapabilities": ["image_matching"]},
                },
                "toolChain": ["getTrack", "getFrames"],
                "planRepair": "argument_not_allowed:getTrack:description",
                "planUsedDefault": True,
            },
            {
                "round": 2,
                "planHint": "listRegistry → matchImage",
                "observation": "结果：confirmedMatchCount=2",
                "reflection": {
                    "handoff": "finish",
                    "state": "sufficient",
                    "decisionSource": "acceptance_guard",
                    "reason": "验收清单已满足",
                },
                "toolChain": ["listRegistry", "matchImage"],
                "planRepair": "",
                "planUsedDefault": False,
            },
        ],
        "tool_records": [
            {
                "id": "tracks",
                "tool": "getTrack",
                "round": 1,
                "arguments": {"offset": 0, "limit": 60},
                "result": {"ok": True, "trackCount": 12, "tracks": [{"trackId": "1"}]},
                "summary": {"tool": "getTrack", "trackCount": 12},
                "ok": True,
                "skipped": False,
                "error": None,
            },
            {
                "id": "match",
                "tool": "matchImage",
                "round": 1,
                "arguments": {"topK": 0},
                "result": {"ok": False, "error": "argument_not_allowed:matchImage:foo"},
                "summary": {"tool": "matchImage", "error": "argument_not_allowed:matchImage:foo"},
                "ok": False,
                "skipped": False,
                "error": "argument_not_allowed:matchImage:foo",
            },
            {
                "id": "tracks",
                "tool": "getTrack",
                "round": 2,
                "arguments": {"offset": 0, "limit": 60},
                "result": {"ok": True, "trackCount": 12, "tracks": [{"trackId": "1"}]},
                "summary": {"tool": "getTrack", "trackCount": 12},
                "ok": True,
                "skipped": False,
                "error": None,
            },
            {
                "id": "frames",
                "tool": "getFrames",
                "round": 2,
                "arguments": {"trackIds": {"$ref": "tracks.trackIds"}},
                "result": {"ok": True},
                "summary": {"tool": "getFrames", "skipped": True, "skipReason": "condition_not_met"},
                "ok": False,
                "skipped": True,
                "error": "condition_not_met",
            },
        ],
        "reflection": {"decisionSource": "acceptance_guard", "state": "sufficient"},
        "final_state": "sufficient",
    }


class QaMemoryPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_persist_qa_memory_writes_rounds_and_evidence(self):
        controller = _make_controller(self.root)
        state = _state_with_rounds_and_tools()

        controller._persist_qa_memory(state)

        self.assertEqual(controller.memory_persist_error, "")
        rounds = controller.repository.qa_rounds.rows()
        self.assertEqual(len(rounds), 2)
        self.assertEqual({row["round_id"] for row in rounds}, {"session-test-r1", "session-test-r2"})
        first = json.loads(rounds[0]["plan"])
        self.assertEqual(first["planHint"], "getTrack → getFrames → listRegistry → matchImage")
        self.assertTrue(first["planUsedDefault"])
        self.assertEqual(first["planRepair"], "argument_not_allowed:getTrack:description")
        reflection = json.loads(rounds[0]["reflection"])
        self.assertEqual(reflection["decisionSource"], "model")

        evidence = controller.repository.qa_evidence.rows()
        self.assertEqual(len(evidence), 4)
        ids = {row["evidence_id"] for row in evidence}
        # 跨轮次同名 call id（tracks）不互相覆盖
        self.assertIn("session-test-r1-tracks", ids)
        self.assertIn("session-test-r2-tracks", ids)
        by_id = {row["evidence_id"]: row for row in evidence}
        tool_result = json.loads(by_id["session-test-r1-match"]["tool_result"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(tool_result["error"], "argument_not_allowed:matchImage:foo")
        # 关键帧/轨迹大列表不落库
        self.assertNotIn("tracks", tool_result["resultSummary"])
        # 依赖未满足而跳过不算失败，但仍记录
        skipped = json.loads(by_id["session-test-r2-frames"]["tool_result"])
        self.assertTrue(skipped["skipped"])

    def test_persist_qa_memory_does_not_break_when_records_malformed(self):
        controller = _make_controller(self.root)
        state = {
            "rounds": [{"round": "not-a-number"}, None, {"round": 1, "planHint": "ok"}],
            "tool_records": [None, {"tool": "getTrack", "id": "x", "round": 1}],
        }

        controller._persist_qa_memory(state)

        self.assertEqual(controller.memory_persist_error, "")
        self.assertEqual(len(controller.repository.qa_rounds.rows()), 1)
        self.assertEqual(len(controller.repository.qa_evidence.rows()), 1)

    def test_build_decision_metrics_counts_model_vs_guard(self):
        metrics = AgentController._build_decision_metrics(_state_with_rounds_and_tools())

        self.assertEqual(metrics["roundCount"], 2)
        self.assertEqual(metrics["modelDecisionCount"], 1)
        self.assertEqual(metrics["guardDecisionCount"], 1)
        self.assertEqual(metrics["decisionSourceCounts"], {"model": 1, "acceptance_guard": 1})
        self.assertEqual(metrics["replanCount"], 1)
        self.assertEqual(metrics["finishCount"], 1)
        self.assertEqual(metrics["planFallbackCount"], 1)
        self.assertEqual(metrics["planRepairCount"], 1)
        self.assertEqual(metrics["toolFailedCount"], 1)
        self.assertEqual(metrics["toolSkippedCount"], 1)
        self.assertEqual(metrics["finalDecisionSource"], "acceptance_guard")
        self.assertEqual(metrics["finalState"], "sufficient")

    def test_session_audit_includes_decision_metrics(self):
        controller = _make_controller(self.root)
        controller.decision_metrics = {"modelDecisionCount": 1, "guardDecisionCount": 1}
        controller.memory_persist_error = ""

        audit = controller._session_audit_result({"conclusion": "结论", "tracks": [], "planMode": "langgraph"})

        self.assertEqual(audit["conclusion"], "结论")
        self.assertEqual(audit["decisionMetrics"]["modelDecisionCount"], 1)
        self.assertIn("memoryPersistError", audit)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory import MemoryRepository, MemorySettingsStore, TrackMemoryManager


class FakeKeyframeIndex:
    def __init__(self):
        self.removed: list[int] = []
        self.rebuilt = None

    def remove(self, vector_ids):
        self.removed.extend(int(value) for value in vector_ids)

    def rebuild(self, entries):
        self.rebuilt = list(entries)


class FakeVectors:
    def __init__(self):
        self.keyframes = FakeKeyframeIndex()


class TrackMemoryManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.config = {
            "paths": {
                "tracks_csv": str(root / "memory" / "tracks.csv"),
                "keyframes_csv": str(root / "memory" / "track_keyframes.csv"),
                "qa_sessions_csv": str(root / "memory" / "qa_sessions.csv"),
                "qa_rounds_csv": str(root / "memory" / "qa_rounds.csv"),
                "qa_evidence_csv": str(root / "memory" / "qa_evidence.csv"),
                "memory_settings_json": str(root / "memory" / "settings.json"),
                "registry_csv": str(root / "registry" / "registry.csv"),
                "registry_images_csv": str(root / "registry" / "registry_reference_images.csv"),
                "keyframe_dir": str(root / "memory" / "keyframes"),
                "trajectory_dir": str(root / "memory" / "trajectories"),
                "clip_dir": str(root / "memory" / "clips"),
            }
        }
        for key in ("keyframe_dir", "trajectory_dir", "clip_dir"):
            Path(self.config["paths"][key]).mkdir(parents=True, exist_ok=True)
        self.repository = MemoryRepository(self.config)
        self.vectors = FakeVectors()
        self.manager = TrackMemoryManager(self.config, self.repository, self.vectors)

    def tearDown(self):
        self.temporary.cleanup()

    def add_track(self, track_id: str, end_time: float, vector_id: int, completed: bool = False):
        keyframe_path = Path(self.config["paths"]["keyframe_dir"]) / f"{track_id}.jpg"
        keyframe_path.write_bytes(b"frame")
        trajectory_path = Path(self.config["paths"]["trajectory_dir"]) / f"{track_id}.json"
        if completed:
            trajectory_path.write_text("{}", encoding="utf-8")
        self.repository.upsert_track({
            "track_id": track_id,
            "start_time": 0,
            "end_time": end_time,
            "final_hull_number": "0857" if track_id == "1" else "",
            "final_description": "灰色船体",
            "final_match_type": "confirmed" if track_id == "1" else "unknown",
            "trajectory_path": str(trajectory_path) if completed else "",
        })
        self.repository.upsert_keyframe({
            "keyframe_id": f"frame-{track_id}",
            "track_id": track_id,
            "timestamp": end_time,
            "keyframe_path": str(keyframe_path),
            "bbox": [0, 0, 10, 10],
            "quality_score": 0.8,
            "retention_score": 1.2,
            "keyframe_vector_id": vector_id,
            "is_embedded": True,
        })
        return keyframe_path, trajectory_path

    def test_snapshot_reports_tracks_and_keyframes(self):
        self.add_track("1", 10, 101, completed=True)
        payload = self.manager.snapshot()
        self.assertEqual(payload["trackCount"], 1)
        self.assertEqual(payload["keyframeCount"], 1)
        self.assertEqual(payload["embeddedKeyframeCount"], 1)
        self.assertEqual(payload["tracks"][0]["memoryState"], "已完成")

    def test_prune_expired_keeps_protected_track(self):
        old_frame, old_trajectory = self.add_track("1", 10, 101, completed=True)
        active_frame, _ = self.add_track("2", 12, 102)
        self.manager.settings.write(6)
        expired = self.manager.prune_expired(20, protected_track_ids=["2"])
        self.assertEqual(expired, ["1"])
        self.assertEqual(self.vectors.keyframes.removed, [101])
        self.assertIsNone(self.repository.get_track("1"))
        self.assertIsNotNone(self.repository.get_track("2"))
        self.assertFalse(old_frame.exists())
        self.assertFalse(old_trajectory.exists())
        self.assertTrue(active_frame.exists())

    def test_zero_retention_disables_pruning(self):
        self.add_track("1", 1, 101)
        self.manager.settings.write(0)
        self.assertEqual(self.manager.prune_expired(100), [])
        self.assertIsNotNone(self.repository.get_track("1"))

    def test_clear_all_preserves_registry_and_settings(self):
        self.add_track("1", 10, 101)
        self.repository.upsert_registry({"registry_id": "registry-1", "hull_number": "0857", "description": "测试库项"})
        self.repository.add_session("session-1", {"question": "测试"})
        Path(self.config["paths"]["clip_dir"], "clip.mp4").write_bytes(b"clip")
        self.manager.settings.write(6)
        result = self.manager.clear_all()
        self.assertEqual(result, {"deletedTracks": 1, "deletedKeyframes": 1})
        self.assertEqual(self.repository.find_tracks(), [])
        self.assertEqual(self.repository.qa_sessions.rows(), [])
        self.assertEqual(len(self.repository.list_registry()), 1)
        self.assertEqual(self.vectors.keyframes.rebuilt, [])
        self.assertEqual(self.manager.settings.read()["retentionSeconds"], 6)
        self.assertFalse(any(Path(self.config["paths"]["clip_dir"]).iterdir()))

    def test_clear_qa_memory_preserves_track_memory(self):
        self.add_track("1", 10, 101)
        self.repository.add_session("session-1", {"question": "测试"})
        self.repository.add_round("round-1", "session-1", {"calls": []}, {"state": "sufficient"})
        self.repository.add_evidence("evidence-1", "round-1", {"ok": True}, {"type": "keyframe"})
        clip = Path(self.config["paths"]["clip_dir"], "qa-clip.mp4")
        clip.write_bytes(b"clip")

        result = self.manager.clear_qa_memory()

        self.assertEqual(result, {"sessionCount": 1, "roundCount": 1, "evidenceCount": 1})
        self.assertIsNotNone(self.repository.get_track("1"))
        self.assertEqual(self.repository.qa_sessions.rows(), [])
        self.assertEqual(self.repository.qa_rounds.rows(), [])
        self.assertEqual(self.repository.qa_evidence.rows(), [])
        self.assertFalse(clip.exists())

    def test_settings_are_visible_to_another_store(self):
        self.manager.settings.write(9)
        other = MemorySettingsStore(self.config["paths"]["memory_settings_json"])
        self.assertEqual(other.read()["retentionSeconds"], 9)


if __name__ == "__main__":
    unittest.main()

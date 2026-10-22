from memory import MemoryRepository
from memory import maintenance
from memory.maintenance import TrackMemoryManager


def _paths(tmp_path):
    return {
        "tracks_csv": str(tmp_path / "tracks.csv"),
        "keyframes_csv": str(tmp_path / "keyframes.csv"),
        "registry_csv": str(tmp_path / "registry.csv"),
        "registry_images_csv": str(tmp_path / "registry_images.csv"),
        "qa_sessions_csv": str(tmp_path / "qa_sessions.csv"),
    }


def test_clear_track_memory_preserves_qa_memory(tmp_path):
    repository = MemoryRepository({"paths": _paths(tmp_path)})
    repository.upsert_track({"track_id": "1", "start_time": 1, "end_time": 2})
    repository.create_session("session-1", "问题")
    repository.append_turn("session-1", "问题", {"state": "completed", "answerText": "回答"})

    repository.clear_track_memory()

    assert repository.tracks.rows() == []
    assert repository.keyframes.rows() == []
    assert repository.get_session("session-1") is not None


class _Index:
    def rebuild(self, _entries):
        return None


class _Vectors:
    keyframes = _Index()


def test_cleanup_scopes_keep_clips_for_track_clear(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    paths.update({
        "memory_settings_json": str(tmp_path / "settings.json"),
        "keyframe_dir": str(tmp_path / "keyframes"),
        "trajectory_dir": str(tmp_path / "trajectories"),
        "clip_dir": str(tmp_path / "clips"),
    })
    config = {"paths": paths, "harness": {"shared_state_file": str(tmp_path / "state.sqlite")}}
    repository = MemoryRepository(config)
    manager = TrackMemoryManager(config, repository, _Vectors())
    clip = tmp_path / "clips" / "keep.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"clip")

    manager.clear_track_memory()
    assert clip.exists()

    artifact_root = tmp_path / "project"
    artifact = artifact_root / "data" / "memory" / "tool_results" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    configured_artifact = tmp_path / "tool_results" / "session" / "result.json"
    configured_artifact.parent.mkdir(parents=True)
    configured_artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(maintenance, "project_root", lambda: artifact_root)
    manager.clear_qa_memory()
    assert not clip.exists()
    assert not artifact.exists()
    assert not configured_artifact.exists()

from types import SimpleNamespace

import numpy as np

from agent.graph import _build_acceptance_progress
from agent.plan_executor import PlanExecutor
from tools.service import ToolService


class _CaptureTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {
            "ok": True,
            "trackCount": len(arguments.get("tracks") or []),
            "highThresholdShipCount": 2,
            "lowThresholdShipCount": 2,
            "highGroups": [["1"], ["2"]],
            "lowGroups": [["1"], ["2"]],
        }


class _FakeVectorIndex:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_many(self, ids):
        return {int(item): self.vectors[int(item)] for item in ids if int(item) in self.vectors}


def test_dedup_executor_converts_frames_result_to_keyframes_by_track():
    tools = _CaptureTools()
    executor = PlanExecutor(tools)
    scope = {
        "tracks": {
            "ok": True,
            "tracks": [
                {"trackId": "1", "startTime": 0, "endTime": 1},
                {"trackId": "2", "startTime": 2, "endTime": 3},
            ],
        },
        "frames": {
            "ok": True,
            "keyframes": [
                {"trackId": "1", "keyframeId": "frame-1", "keyframeVectorId": 1},
                {"trackId": "2", "keyframeId": "frame-2", "keyframeVectorId": 2},
            ],
            "keyframeIds": ["frame-1", "frame-2"],
        },
    }
    calls = [{
        "id": "dedup",
        "tool": "dedupTracks",
        "arguments": {
            "tracks": {"$ref": "tracks.tracks"},
            # 复现截图：规划传了 frames 整体，而不是 keyframesByTrack。
            "frames": {"$ref": "frames"},
        },
    }]

    result = executor.execute(calls, scope)

    assert result["summary"]["skippedCount"] == 0
    assert tools.calls[0][0] == "dedupTracks"
    sent = tools.calls[0][1]
    assert "frames" not in sent
    assert sent["keyframesByTrack"]["1"]["keyframes"][0]["keyframeId"] == "frame-1"
    assert result["scope"]["dedup"]["highThresholdShipCount"] == 2


def test_dedup_executor_recovers_full_tracks_when_plan_passes_track_ids():
    tools = _CaptureTools()
    executor = PlanExecutor(tools)
    track_records = [
        {"trackId": "1", "startTime": 0, "endTime": 1},
        {"trackId": "2", "startTime": 2, "endTime": 3},
    ]
    scope = {
        "tracks": {"ok": True, "trackIds": [1, 2], "tracks": track_records},
        "frames": {
            "ok": True,
            "keyframesByTrack": {
                "1": {"keyframes": [{"trackId": "1", "keyframeId": "frame-1", "keyframeVectorId": 1}]},
                "2": {"keyframes": [{"trackId": "2", "keyframeId": "frame-2", "keyframeVectorId": 2}]},
            },
        },
    }
    calls = [{
        "id": "dedup",
        "tool": "dedupTracks",
        "arguments": {
            # 复现实际后端错误：模型误把 trackIds 传给需要完整轨迹记录的 tracks。
            "tracks": {"$ref": "tracks.trackIds"},
            "keyframesByTrack": {"$ref": "frames.keyframesByTrack"},
        },
    }]

    result = executor.execute(calls, scope)

    assert result["summary"]["failedCount"] == 0
    sent_tracks = tools.calls[0][1]["tracks"]
    assert sent_tracks == track_records
    assert all(isinstance(item, dict) for item in sent_tracks)


def test_count_acceptance_requires_successful_dedup_result_not_just_tool_name():
    base_kwargs = dict(
        track_count=11,
        registry_checked=False,
        registry_listed=False,
        registry_has_items=False,
        can_try_visual=False,
        visual_attempted=False,
        match_image_attempted=False,
        match_image_usable=False,
        has_tool_evidence=True,
    )

    skipped_progress = _build_acceptance_progress(
        {"operation": "count"},
        {"getTrack", "getFrames", "dedupTracks"},
        dedup_usable=False,
        **base_kwargs,
    )
    assert skipped_progress["acceptanceSatisfied"] is False
    assert any(not item["completed"] and item["key"] == "dedup" for item in skipped_progress["requirements"])

    completed_progress = _build_acceptance_progress(
        {"operation": "count"},
        {"getTrack", "getFrames", "dedupTracks"},
        dedup_usable=True,
        **base_kwargs,
    )
    assert completed_progress["acceptanceSatisfied"] is True


def test_dedup_tracks_accepts_vector_id_when_is_embedded_flag_missing():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.9, "dedup_low": 0.7}
    service.vectors = SimpleNamespace(
        keyframes=_FakeVectorIndex({
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0], dtype=np.float32),
        })
    )
    tracks = [
        {"trackId": "1", "startTime": 0, "endTime": 1},
        {"trackId": "2", "startTime": 2, "endTime": 3},
    ]
    groups = {
        "1": [{"trackId": "1", "keyframeId": "frame-1", "keyframeVectorId": 1, "retentionScore": 0.9}],
        "2": [{"trackId": "2", "keyframeId": "frame-2", "keyframeVectorId": 2, "retentionScore": 0.8}],
    }

    result = service.dedupTracks(tracks, groups)

    assert result["ok"] is True
    assert result["highThresholdShipCount"] == 1
    assert result["unsearchableTrackIds"] == []

def test_dedup_tracks_returns_confirmed_and_pending_merge_groups():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.9, "dedup_low": 0.7}
    service.vectors = SimpleNamespace(
        keyframes=_FakeVectorIndex({
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0], dtype=np.float32),
            3: np.array([0.8, 0.6], dtype=np.float32),
            4: np.array([0.0, 1.0], dtype=np.float32),
        })
    )
    tracks = [
        {"trackId": str(index), "startTime": index * 10, "endTime": index * 10 + 1}
        for index in range(1, 5)
    ]
    groups = {
        str(index): [{
            "trackId": str(index),
            "keyframeId": f"frame-{index}",
            "keyframeVectorId": index,
            "retentionScore": 0.9,
        }]
        for index in range(1, 5)
    }

    result = service.dedupTracks(tracks, groups)

    assert result["confirmedShipCount"] == 3
    assert result["minimumShipCount"] == 2
    assert result["confirmedMergeGroups"][0]["trackIds"] == ["1", "2"]
    assert result["pendingMergeGroups"][0]["trackIds"] == ["1", "2", "3"]
    assert result["pendingMergeGroups"][0]["currentGroups"] == [["1", "2"], ["3"]]
    assert result["pendingReduction"] == 1
    assert result["countStability"] == "sensitive"

def test_dedup_tool_recovers_track_records_from_id_list():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.9, "dedup_low": 0.7}
    track_records = {
        "1": {"trackId": "1", "startTime": 0, "endTime": 1},
        "2": {"trackId": "2", "startTime": 2, "endTime": 3},
    }
    service.repository = SimpleNamespace(get_track=lambda track_id: track_records.get(str(track_id)))
    service.vectors = SimpleNamespace(
        keyframes=_FakeVectorIndex({
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0], dtype=np.float32),
        })
    )
    groups = {
        1: [{"trackId": "1", "keyframeId": "frame-1", "keyframeVectorId": "1", "retentionScore": 0.9}],
        2: [{"trackId": "2", "keyframeId": "frame-2", "keyframeVectorId": "2", "retentionScore": 0.8}],
    }

    result = service.dedupTracks([1, 2], groups)

    assert result["ok"] is True
    assert result["trackCount"] == 2
    assert result["highThresholdShipCount"] == 1
    assert result["unresolvedTrackIds"] == []


def test_dedup_tool_returns_structured_error_for_unresolved_track_ids():
    service = ToolService.__new__(ToolService)
    service.repository = SimpleNamespace(get_track=lambda track_id: None)

    result = service.dedupTracks([999], {})

    assert result["ok"] is False
    assert result["error"] == "dedup_tracks_unresolved"
    assert result["unresolvedTrackIds"] == ["999"]

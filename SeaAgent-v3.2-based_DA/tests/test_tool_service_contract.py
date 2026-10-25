import threading
import time

from tools.service import ToolService


def _track(track_id, hull=None):
    return {
        "trackId": track_id,
        "finalHullNumber": hull,
        "finalDescription": "",
        "startTime": 1.0,
        "endTime": 2.0,
    }


def test_direct_recognition_returns_scored_candidates_not_all_tracks():
    tracks = [_track(f"t-{index}") for index in range(150)]
    service = ToolService.__new__(ToolService)
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda items: ({item["trackId"]: [] for item in items}, [])
    service._track_text_scores = lambda *_args: {
        "t-7": {
            "embeddingScore": 0.91,
            "scoreBand": "match",
            "matchedKeyframeIds": [],
        }
    }
    service._track_image_scores = lambda *_args: {}
    service._band = lambda *_args: "match"

    result = service._recognize_tracks(description="yellow vessel")

    assert result["status"] == "confirmed"
    assert result["trackIds"] == ["t-7"]
    assert result["statistics"]["searchedTrackCount"] == 150
    assert result["statistics"]["unsearchableTrackCount"] == 149
    assert len(result["unsearchableTrackIds"]) <= service.MAX_RETURNED_MATCHES
    assert "tracks" not in result


def test_hull_number_in_keyframe_counts_as_a_hit():
    frame = {
        "keyframeId": "kf-0857",
        "vlmHullNumber": "0857",
        "readabilityConfidence": 0.94,
    }

    class Repository:
        def get_keyframes(self, track_ids, embedded_only=False):
            assert embedded_only is False
            return {"t-1": [frame]}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)
    service._load_search_frames = lambda _items: ({"t-1": []}, [])

    result = service._recognize_tracks(hull_number="0857")

    assert result["status"] == "confirmed"
    assert result["filterMissed"] is False
    assert result["matches"][0]["keyframeIds"] == ["kf-0857"]


def test_unmatched_supplied_alias_requests_registry_assistance():
    class Repository:
        def get_keyframes(self, _track_ids, embedded_only=False):
            return {}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([], True)
    service._load_search_frames = lambda _items: ({}, [])

    result = service._recognize_tracks(hull_number="小红")

    assert result["status"] == "mismatch"
    assert result["nextStep"] == "match_registry_tracks"


def test_vlm_can_resolve_a_registry_reference_id_and_track_keyframe(tmp_path):
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    reference_path.write_bytes(b"reference")
    candidate_path.write_bytes(b"candidate")

    class Repository:
        def references_by_ids(self, reference_ids):
            assert reference_ids == ["ref-1"]
            return [{"imagePath": str(reference_path)}]

        def get_keyframe(self, keyframe_id):
            assert keyframe_id == "kf-1"
            return {"keyframePath": str(candidate_path)}

    class VLM:
        def verify(self, reference, candidate):
            assert reference == [str(reference_path)]
            assert candidate == [str(candidate_path)]
            return {"decision": "match", "facts": ["same hull outline"]}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.llm = VLM()

    result = service.verifyVesselImages(
        referenceRegistryReferenceId="ref-1",
        candidateKeyframeId="kf-1",
    )

    assert result["ok"] is True
    assert result["decision"] == "match"
    assert result["registryReferenceId"] == "ref-1"
    assert result["keyframeId"] == "kf-1"


def test_registry_assisted_search_without_reference_images_is_unconfirmed():
    class Repository:
        def registry_by_hull(self, _hull):
            return [{"registryId": "r-1", "references": []}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)
    service._load_search_frames = lambda _tracks: ({"t-1": []}, [])

    result = service._match_registry_tracks("0857")

    assert result["status"] == "unconfirmed"
    assert result["confidence"] == "unconfirmed"
    assert result["coverageComplete"] is False


def test_registered_entity_without_comparable_reference_is_uncertain():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "hullNumber": "A01", "references": []}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)
    service._load_search_frames = lambda _tracks: ({"t-1": [{"keyframeId": "kf-1"}]}, [])
    service._deduplicate_tracks = lambda *_args: {
        "ok": True,
        "highGroups": [["t-1"]],
        "lowGroups": [["t-1"]],
        "pendingMergeCount": 0,
        "countStability": "stable",
        "unsearchableTrackIds": [],
    }

    result = service._entity_analysis(compare_registry=True)

    assert result["inRegistryCount"] == 0
    assert result["unregisteredCount"] == 0
    assert result["uncertainCount"] == 1
    assert result["countConfidence"] == "uncertain"


def test_each_business_tool_has_its_own_timeout_budget():
    assert ToolService.TOOL_TIMEOUTS == {
        "recognize_tracks": 60,
        "match_registry_tracks": 60,
        "verify_vessel_images": 60,
        "count_total_entities": 120,
        "count_registry_entities": 120,
        "count_unregistered_entities": 120,
        "show_evidence": 120,
    }


def test_tool_timeout_returns_structured_result():
    finished = threading.Event()
    service = ToolService.__new__(ToolService)
    original = ToolService.TOOL_TIMEOUTS["recognize_tracks"]
    ToolService.TOOL_TIMEOUTS["recognize_tracks"] = 0.01
    try:
        result = service._run_tool(
            "recognize_tracks",
            lambda: (time.sleep(0.03), finished.set(), {"ok": True})[-1],
        )
    finally:
        ToolService.TOOL_TIMEOUTS["recognize_tracks"] = original

    assert result["ok"] is False
    assert result["error"] == "tool_timeout"
    assert result["timeoutSeconds"] == 0.01
    assert finished.wait(0.5)

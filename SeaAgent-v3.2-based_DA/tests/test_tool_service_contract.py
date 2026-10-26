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
    assert [item["trackId"] for item in result["matches"]] == ["t-7"]
    assert result["statistics"]["searchedTrackCount"] == 150
    assert result["statistics"]["unsearchableTrackCount"] == 149
    assert result["coverageComplete"] is False
    assert "confidence" not in result
    assert "confirmedMatches" not in result
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
    assert result["status"] == "unconfirmed"
    assert "confidence" not in result
    assert result["coverageComplete"] is False


def test_total_entity_count_returns_only_deduplication_counts():
    service = ToolService.__new__(ToolService)
    service._entity_analysis = lambda *_args, **_kwargs: {
        "ok": True,
        "coverageComplete": False,
        "deduplication": {
            "highGroups": [["t-1"], ["t-2"]],
            "lowGroups": [["t-1", "t-2"]],
            "rawTrackCount": 2,
            "highThresholdCount": 2,
            "lowThresholdCount": 1,
        },
    }

    result = service.countTotalEntities()

    assert result == {
        "ok": True,
        "rawTrackCount": 2,
        "highThresholdCount": 2,
        "lowThresholdCount": 1,
        "coverageComplete": False,
    }


def test_registry_entity_analysis_reads_all_frames_once_and_preserves_coverage():
    tracks = [_track("t-1")]
    frames = {"t-1": [{"keyframeId": "kf-1"}, {"keyframeId": "kf-2"}]}
    calls = []
    service = ToolService.__new__(ToolService)
    service._validate_time_range = lambda _value: (None, None)
    service._scan_tracks = lambda _scope: (tracks, True)

    def load_search_frames(_tracks, max_frames):
        calls.append(max_frames)
        return frames, ["t-1"]

    service._load_search_frames = load_search_frames
    service._deduplicate_tracks = lambda _tracks, grouped: {
        "lowGroups": [["t-1"]],
        "receivedAllFrames": grouped == {"t-1": {"keyframes": frames["t-1"]}},
    }

    result = service._entity_analysis(all_keyframes=True)

    assert calls == [None]
    assert result["deduplication"]["receivedAllFrames"] is True
    assert result["coverageComplete"] is False


def test_track_deduplication_uses_instance_safe_overlap_check():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.9, "dedup_low": 0.8}
    service.vectors = type("Vectors", (), {"keyframes": object()})()
    service._load_indexed_vectors = lambda *_args, **_kwargs: {1: [1.0], 2: [1.0]}

    result = service._deduplicate_tracks(
        [_track("t-1"), _track("t-2")],
        {
            "t-1": {"keyframes": [{"keyframeVectorId": 1, "retentionScore": 1.0}]},
            "t-2": {"keyframes": [{"keyframeVectorId": 2, "retentionScore": 1.0}]},
        },
    )

    assert result["highGroups"] == [["t-1"], ["t-2"]]
    assert result["lowGroups"] == [["t-1"], ["t-2"]]
    assert result == {
        "highGroups": [["t-1"], ["t-2"]],
        "lowGroups": [["t-1"], ["t-2"]],
        "rawTrackCount": 2,
        "highThresholdCount": 2,
        "lowThresholdCount": 2,
    }


def test_registry_count_combines_categories_and_uses_low_threshold_groups():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "hullNumber": "A01", "references": [{"referenceId": "ref-1"}]}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72}
    service._entity_analysis = lambda *_args, **_kwargs: {
        "ok": True,
        "coverageComplete": True,
        "tracks": [_track("t-1"), _track("t-2")],
        "frames": {"t-1": [{"keyframeId": "kf-1"}], "t-2": [{"keyframeId": "kf-2"}]},
        "deduplication": {
            "lowGroups": [["t-1", "t-2"]],
            "highGroups": [["t-1"], ["t-2"]],
        },
    }
    def match_images(**kwargs):
        assert kwargs["best_by_max_score"] is True
        return {
            "matches": [{
                "matchedTrackId": "t-1",
                "matchedRegistryId": "r-1",
                "maxEmbeddingScore": 0.73,
                "embeddingScore": 0.61,
                "matchedKeyframeIds": ["kf-1"],
                "matchedRegistryReferenceIds": ["ref-1"],
                "bestMatchedKeyframeId": "kf-1",
                "bestMatchedRegistryReferenceId": "ref-1",
            }],
            "registryCoverageComplete": True,
        }

    service._match_images = match_images

    result = service.countEntitiesByRegistry()

    assert result["entityCount"] == 1
    assert result["inRegistryCount"] == 1
    assert result["unregisteredCount"] == 0
    assert result["coverageComplete"] is True
    assert result["registeredMatches"][0]["score"] == 0.73
    assert result["registeredMatches"][0]["matchedObject"] == {
        "registryId": "r-1",
        "hullNumber": "A01",
        "registryReferenceId": "ref-1",
        "keyframeId": "kf-1",
    }
    assert result["unregisteredMatches"] == []


def test_image_match_can_select_a_registry_by_its_best_single_frame_score():
    class Index:
        def __init__(self, values):
            self.values = values

        def get_many(self, ids):
            return {value: self.values[value] for value in ids if value in self.values}

    service = ToolService.__new__(ToolService)
    service.settings = {"image_match": 0.72, "image_exclude": 0.5}
    service.vectors = type("Vectors", (), {
        "keyframes": Index({10: [1.0, 0.0], 11: [0.0, 1.0]}),
        "registry": Index({
            1: [0.9, 0.4],
            2: [0.8, 0.8],
        }),
    })()
    registry_items = [
        {"registryId": "r-1", "references": [{"registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1}]},
        {"registryId": "r-2", "references": [{"registryId": "r-2", "referenceId": "ref-2", "registryVectorId": 2}]},
    ]
    keyframes = [
        {"trackId": "t-1", "keyframeId": "kf-1", "keyframeVectorId": 10},
        {"trackId": "t-1", "keyframeId": "kf-2", "keyframeVectorId": 11},
    ]

    result = service._match_images(
        queryImages=[reference for item in registry_items for reference in item["references"]],
        galleryImages=keyframes,
        registryItems=registry_items,
        best_by_max_score=True,
    )

    assert result["matches"][0]["matchedRegistryId"] == "r-1"
    assert result["matches"][0]["maxEmbeddingScore"] == 0.9


def test_image_match_does_not_encode_or_persist_missing_vectors():
    class Index:
        def __init__(self):
            self.writes = 0

        def get_many(self, _ids):
            return {}

        def add_many(self, *_args):
            self.writes += 1

    class Embedder:
        def encode_images(self, _paths):
            raise AssertionError("tool matching must not encode missing stored vectors")

    service = ToolService.__new__(ToolService)
    service.vectors = type("Vectors", (), {"keyframes": Index(), "registry": Index()})()
    service.embedder = Embedder()
    result = service._match_images(
        queryImages=[{
            "registryId": "r-1",
            "referenceId": "ref-1",
            "registryVectorId": 10,
            "imagePath": "/missing/reference.jpg",
        }],
        galleryImages=[{
            "trackId": "t-1",
            "keyframeId": "kf-1",
            "keyframeVectorId": 20,
            "keyframePath": "/missing/keyframe.jpg",
        }],
        registryItems=[{"registryId": "r-1"}],
    )

    assert result == {
        "ok": True,
        "matches": [],
        "unscoredTrackIds": ["t-1"],
        "registryCoverageComplete": False,
    }
    assert service.vectors.keyframes.writes == 0
    assert service.vectors.registry.writes == 0


def test_each_business_tool_has_its_own_timeout_budget():
    assert ToolService.TOOL_TIMEOUTS == {
        "recognize_tracks": 60,
        "match_registry_tracks": 60,
        "verify_vessel_images": 60,
        "count_total_entities": 120,
        "count_entities_by_registry": 120,
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
    assert result["errorType"] == "TimeoutError"
    assert result["timeoutSeconds"] == 0.01
    assert finished.wait(0.5)


def test_tool_exception_returns_its_type():
    service = ToolService.__new__(ToolService)

    def fail():
        raise ValueError("bad input")

    result = service._run_tool("recognize_tracks", fail)

    assert result["ok"] is False
    assert result["error"] == "tool_failed"
    assert result["errorType"] == "ValueError"
    assert result["hint"] == "bad input"

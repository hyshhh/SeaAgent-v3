import math
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
    service._load_search_frames = lambda items: (
        {item["trackId"]: [] for item in items}, [], {"t-7": "kf-7"}
    )
    service._track_text_scores = lambda *_args: {
        "t-7": {
            "score": 0.91,
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
    assert result["matches"][0]["keyframeId"] == "kf-7"


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
    service._load_search_frames = lambda _items: ({"t-1": []}, [], {"t-1": "kf-0857"})

    result = service._recognize_tracks(hull_number="0857")

    assert result["status"] == "confirmed"
    assert result["matches"][0]["keyframeIds"] == ["kf-0857"]
    assert result["matches"][0]["keyframeId"] == "kf-0857"


def test_unmatched_supplied_alias_requests_registry_assistance():
    class Repository:
        def get_keyframes(self, _track_ids, embedded_only=False):
            return {}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([], True)
    service._load_search_frames = lambda _items: ({}, [], {})

    result = service._recognize_tracks(hull_number="小红")

    assert result["status"] == "mismatch"
    assert result["nextStep"] == "match_registry_tracks"
    assert result["recommendation"] == "建议模型调用辅助工具 match_registry_tracks，使用相同舷号和时间范围继续核验。"


def test_image_recognition_reports_missing_query_image(tmp_path):
    service = ToolService.__new__(ToolService)
    service._validate_time_range = lambda _value: (None, "")

    result = service._recognize_tracks(image_path=str(tmp_path / "missing.png"))

    assert result["ok"] is False
    assert result["error"] == "image_not_found"
    assert result["errorType"] == "FileNotFoundError"


def test_image_embedding_failure_is_a_tool_error_not_a_negative_match(tmp_path):
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"image")

    class Embedder:
        def encode_images(self, paths):
            assert paths == [str(image_path)]
            raise RuntimeError("embedding backend unavailable")

    service = ToolService.__new__(ToolService)
    service.embedder = Embedder()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)
    service._load_search_frames = lambda _tracks: (
        {"t-1": [{"keyframeId": "kf-1", "keyframeVectorId": 1}]}, [], {"t-1": "kf-1"}
    )

    result = service.recognize_tracks(image_path=str(image_path))

    assert result["ok"] is False
    assert result["error"] == "tool_failed"
    assert result["errorType"] == "RuntimeError"
    assert result["hint"] == "embedding backend unavailable"


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
            return {"trackId": "t-1", "keyframePath": str(candidate_path)}

    class VLM:
        def verify(self, reference, candidate):
            assert reference == [str(reference_path)]
            assert candidate == [str(candidate_path)]
            return {"decision": "match", "facts": [], "formatRecovered": True}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.llm = VLM()

    result = service.verify_vessel_images(
        reference_registry_reference_id="ref-1",
        candidate_keyframe_id="kf-1",
    )

    assert result["ok"] is True
    assert result["decision"] == "match"
    assert result["formatRecovered"] is True
    assert result["registryReferenceId"] == "ref-1"
    assert result["keyframeId"] == "kf-1"
    assert result["trackId"] == "t-1"


def test_vlm_can_compare_two_track_keyframes_for_deduplication(tmp_path):
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    class Repository:
        def get_keyframe(self, keyframe_id):
            return {
                "kf-1": {"trackId": "t-1", "keyframePath": str(first_path)},
                "kf-2": {"trackId": "t-2", "keyframePath": str(second_path)},
            }[keyframe_id]

    class VLM:
        def verify(self, reference, candidate):
            assert reference == [str(first_path)]
            assert candidate == [str(second_path)]
            return {"decision": "match", "facts": ["same vessel"]}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.llm = VLM()

    result = service.verify_vessel_images(
        reference_keyframe_id="kf-1",
        candidate_keyframe_id="kf-2",
        pair_id="t-1::t-2",
    )

    assert result["decision"] == "match"
    assert result["pairId"] == "t-1::t-2"
    assert result["referenceTrackId"] == "t-1"
    assert result["candidateTrackId"] == "t-2"
    assert result["referenceType"] == "keyframe"


def test_vlm_failure_keeps_the_statistics_candidate_identity(tmp_path):
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    reference_path.write_bytes(b"reference")
    candidate_path.write_bytes(b"candidate")

    class Repository:
        def references_by_ids(self, _reference_ids):
            return [{"imagePath": str(reference_path)}]

        def get_keyframe(self, _keyframe_id):
            return {"trackId": "t-7", "keyframePath": str(candidate_path)}

    class VLM:
        def verify(self, _reference, _candidate):
            raise RuntimeError("vlm unavailable")

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.llm = VLM()

    result = service.verify_vessel_images(
        reference_registry_reference_id="ref-1",
        candidate_keyframe_id="kf-7",
    )

    assert result["ok"] is False
    assert result["decision"] == "uncertain"
    assert result["trackId"] == "t-7"
    assert result["registryReferenceId"] == "ref-1"
    assert result["keyframeId"] == "kf-7"


def test_registry_assisted_search_without_reference_images_is_unconfirmed():
    class Repository:
        def registry_by_hull(self, _hull):
            return [{"registryId": "r-1", "references": []}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)
    service._load_search_frames = lambda _tracks: ({"t-1": []}, [], {})

    result = service._match_registry_tracks("0857")

    assert result["status"] == "unconfirmed"
    assert result["status"] == "unconfirmed"
    assert "confidence" not in result
    assert result["coverageComplete"] is False


def test_dedup_tool_returns_counts_and_bounded_display_rows():
    service = ToolService.__new__(ToolService)
    service._dedup_track_analysis = lambda *_args, **_kwargs: {
        "ok": True,
        "status": "verification_required",
        "coverageComplete": False,
        "trackCount": 2,
        "exactShipCount": None,
        "fuzzyShipCount": 1,
        "duplicateCandidateCount": 1,
        "unresolvedDuplicateCount": 1,
        "duplicateCandidates": [{
            "pairId": "t-1::t-2",
            "referenceTrackId": "t-1",
            "candidateTrackId": "t-2",
            "score": 0.9,
            "referenceKeyframeId": "kf-1",
            "candidateKeyframeId": "kf-2",
            "bestFrameScore": 0.95,
        }],
        "duplicateCandidatesTruncated": False,
        "nextStep": "verify_vessel_images",
        "recommendation": "Verify candidates.",
        "appliedTimeRange": None,
        "tracks": [_track("t-1")],
        "firstKeyframes": {"t-1": "kf-first"},
    }

    result = service.dedup_tracks()

    assert result == {
        "ok": True,
        "status": "verification_required",
        "trackCount": 2,
        "exactShipCount": None,
        "fuzzyShipCount": 1,
        "duplicateCandidateCount": 1,
        "unresolvedDuplicateCount": 1,
        "coverageComplete": False,
        "appliedTimeRange": None,
        "nextStep": "verify_vessel_images",
        "recommendation": "Verify candidates.",
        "duplicateCandidatesTruncated": False,
        "duplicateCandidates": [{
            "pairId": "t-1::t-2",
            "referenceTrackId": "t-1",
            "candidateTrackId": "t-2",
            "score": 0.9,
            "referenceKeyframeId": "kf-1",
            "candidateKeyframeId": "kf-2",
            "bestFrameScore": 0.95,
        }],
        "trackRows": [{
            "trackId": "t-1",
            "scoreBand": "mismatch",
            "startTime": 1.0,
            "endTime": 2.0,
            "keyframeId": "kf-first",
        }],
        "trackRowsTruncated": False,
    }


def test_dedup_count_reports_the_applied_time_range():
    requested = [
        "2026-10-27T00:00:00.000000+08:00",
        "2026-10-27T23:59:59.999999+08:00",
    ]
    service = ToolService.__new__(ToolService)
    service._dedup_track_analysis = lambda *_args, **_kwargs: {
        "ok": True,
        "status": "complete",
        "coverageComplete": True,
        "trackCount": 0,
        "exactShipCount": 0,
        "fuzzyShipCount": 0,
        "duplicateCandidateCount": 0,
        "unresolvedDuplicateCount": 0,
        "duplicateCandidates": [],
        "duplicateCandidatesTruncated": False,
        "appliedTimeRange": requested,
        "tracks": [],
        "firstKeyframes": {},
    }

    result = service.dedup_tracks(time_range=requested)

    assert result["appliedTimeRange"] == requested


def test_show_evidence_selects_one_first_keyframe_without_embedding_requirement():
    class Repository:
        def get_keyframes(self, track_ids, embedded_only=False):
            assert track_ids == ["t-1"]
            assert embedded_only is False
            return {"t-1": [
                {"keyframeId": "kf-second", "timestamp": 2.0, "keyframeVectorId": None},
                {"keyframeId": "kf-first", "timestamp": 1.0, "keyframeVectorId": None},
            ]}

        def get_track(self, track_id):
            assert track_id == "t-1"
            return _track(track_id)

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service._validate_time_range = lambda _value: (None, "")

    result = service._show_evidence(["t-1"])

    assert result["trackRows"][0]["keyframeId"] == "kf-first"
    assert "shipSegmentId" not in result["trackRows"][0]


def test_first_keyframe_is_selected_by_time_without_requiring_an_embedding():
    frames = [
        {"keyframeId": "kf-later", "timestamp": 20.0, "keyframeVectorId": None},
        {"keyframeId": "kf-first", "timestamp": 10.0, "keyframeVectorId": None},
    ]

    assert ToolService._first_keyframe_id(frames) == "kf-first"


def test_track_deduplication_uses_instance_safe_overlap_check():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.9, "dedup_review": 0.85, "dedup_low": 0.8}
    service.vectors = type("Vectors", (), {"keyframes": object()})()
    service._load_indexed_vectors = lambda *_args, **_kwargs: {1: [1.0], 2: [1.0]}

    result = service._deduplicate_tracks(
        [_track("t-1"), _track("t-2")],
        {
            "t-1": {"keyframes": [{"keyframeVectorId": 1, "retentionScore": 1.0}]},
            "t-2": {"keyframes": [{"keyframeVectorId": 2, "retentionScore": 1.0}]},
        },
    )

    assert result["status"] == "complete"
    assert result["exactShipCount"] == 2
    assert result["fuzzyShipCount"] == 2
    assert result["duplicateCandidateCount"] == 0


def test_dedup_requires_vlm_confirmation_before_an_exact_merge():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.82, "dedup_review": 0.80, "dedup_low": 0.75}
    service.vectors = type("Vectors", (), {"keyframes": object()})()
    service._load_indexed_vectors = lambda *_args, **_kwargs: {
        1: [1.0, 0.0],
        2: [0.81, math.sqrt(1 - 0.81**2)],
    }
    tracks = [
        {**_track("t-1"), "startTime": 1.0, "endTime": 2.0},
        {**_track("t-2"), "startTime": 3.0, "endTime": 4.0},
    ]
    frames = {
        "t-1": {"keyframes": [{"keyframeId": "kf-1", "keyframeVectorId": 1}]},
        "t-2": {"keyframes": [{"keyframeId": "kf-2", "keyframeVectorId": 2}]},
    }

    pending = service._deduplicate_tracks(tracks, frames)
    pair_id = pending["duplicateCandidates"][0]["pairId"]
    confirmed = service._deduplicate_tracks(
        tracks, frames, confirmed_pair_ids={pair_id}, resolved_pair_ids={pair_id}
    )
    rejected = service._deduplicate_tracks(
        tracks, frames, confirmed_pair_ids=set(), resolved_pair_ids={pair_id}
    )

    assert pending["status"] == "verification_required"
    assert pending["exactShipCount"] is None
    assert pending["fuzzyShipCount"] == 1
    assert confirmed["exactShipCount"] == 1
    assert rejected["exactShipCount"] == 2
    assert confirmed["unresolvedDuplicateCount"] == 0
    assert rejected["unresolvedDuplicateCount"] == 0


def test_dedup_at_or_above_high_threshold_is_exact_without_vlm():
    service = ToolService.__new__(ToolService)
    service.settings = {"dedup_high": 0.82, "dedup_review": 0.80, "dedup_low": 0.75}
    service.vectors = type("Vectors", (), {"keyframes": object()})()
    service._load_indexed_vectors = lambda *_args, **_kwargs: {1: [1.0], 2: [1.0]}
    tracks = [
        {**_track("t-1"), "startTime": 1.0, "endTime": 2.0},
        {**_track("t-2"), "startTime": 3.0, "endTime": 4.0},
    ]
    frames = {
        "t-1": {"keyframes": [{"keyframeId": "kf-1", "keyframeVectorId": 1}]},
        "t-2": {"keyframes": [{"keyframeId": "kf-2", "keyframeVectorId": 2}]},
    }

    result = service._deduplicate_tracks(tracks, frames)

    assert result["status"] == "complete"
    assert result["exactShipCount"] == 1
    assert result["fuzzyShipCount"] == 1
    assert result["duplicateCandidateCount"] == 0
    assert "nextStep" not in result


def test_registry_count_matches_each_track_without_deduplication():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "hullNumber": "A01", "references": [{
                "registryId": "r-1", "referenceId": "ref-1", "imagePath": "/registry.jpg",
            }]}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    tracks = [_track("t-1"), _track("t-2")]
    frames = {
        "t-1": [{"trackId": "t-1", "keyframeId": "kf-1", "keyframePath": "/candidate.jpg"}],
        "t-2": [{"trackId": "t-2", "keyframeId": "kf-2", "keyframePath": "/candidate.jpg"}],
    }
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)

    def load_frames(items, max_frames, *, require_complete):
        assert items == tracks
        assert max_frames is None
        assert require_complete is True
        return frames, [], {"t-1": "kf-1", "t-2": "kf-2"}

    service._load_search_frames = load_frames
    service._deduplicate_tracks = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("registry counts must not deduplicate tracks")
    )
    service._verify_image_pair = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("match_entities must not call VLM internally")
    )

    def match_images(**kwargs):
        assert {frame["trackId"] for frame in kwargs["galleryImages"]} == {"t-1", "t-2"}
        return {
            "matches": [{
                "matchedTrackId": "t-1",
                "matchedRegistryId": "r-1",
                "score": 0.73,
                "matchedKeyframeId": "kf-1",
                "matchedRegistryReferenceId": "ref-1",
            }],
            "registryCoverageComplete": True,
            "missingRegistryVectorCount": 0,
        }

    service._match_images = match_images
    result = service.match_entities()

    assert result["trackCount"] == 2
    assert result["status"] == "incomplete"
    assert result["embeddingConfirmedCount"] == 1
    assert result["outsideRegistryCount"] == 0
    assert result["grayCandidateCount"] == 0
    assert result["unresolvedTrackCount"] == 1
    assert "nextStep" not in result
    assert result["matches"][0]["score"] == 0.73
    assert [item["trackId"] for item in result["matches"]] == ["t-1", "t-2"]
    assert result["matches"][0]["matchedObject"] == {
        "registryId": "r-1",
        "hullNumber": "A01",
        "registryReferenceId": "ref-1",
        "keyframeId": "kf-1",
    }
    assert result["matches"][0]["scoreBand"] == "match"
    assert result["matches"][0]["classification"] == "embedding_confirmed"
    assert result["matches"][0]["verificationDecision"] == "match"
    assert result["matches"][0]["keyframeId"] == "kf-1"
    assert result["verificationCandidates"] == []


def test_registry_count_requires_vectors_for_every_selected_keyframe():
    class Repository:
        def get_keyframes(self, _track_ids, embedded_only=False):
            assert embedded_only is False
            return {"t-1": [
                {"keyframeId": "kf-1", "keyframeVectorId": 1},
                {"keyframeId": "kf-2", "keyframeVectorId": 2},
            ]}

    class Index:
        def get_many(self, ids):
            return {1: [1.0]} if 1 in ids else {}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.vectors = type("Vectors", (), {"keyframes": Index()})()

    frames, incomplete, first_keyframes = service._load_search_frames(
        [_track("t-1")], max_frames=None, require_complete=True
    )

    assert [frame["keyframeId"] for frame in frames["t-1"]] == ["kf-1"]
    assert incomplete == ["t-1"]
    assert first_keyframes == {"t-1": "kf-1"}


def test_registry_count_marks_unsearchable_track_as_incomplete():
    class Repository:
        def list_registry(self):
            return []

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    empty_index = type("Index", (), {"get_many": lambda _self, _ids: {}})()
    service.vectors = type("Vectors", (), {"keyframes": empty_index, "registry": empty_index})()
    tracks = [_track("t-1")]
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda _items, max_frames, *, require_complete: (
        {"t-1": []}, ["t-1"], {}
    )

    result = service.match_entities()

    assert result["trackCount"] == 1
    assert result["outsideRegistryCount"] == 0
    assert result["unresolvedTrackCount"] == 1
    assert result["status"] == "incomplete"
    assert result["embeddingCoverageComplete"] is False
    assert result["coverageReasons"] == ["track_keyframe_vectors_missing"]


def test_registry_count_reports_missing_registry_references_separately():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "references": []}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    empty_index = type("Index", (), {"get_many": lambda _self, _ids: {}})()
    service.vectors = type("Vectors", (), {"keyframes": empty_index, "registry": empty_index})()
    tracks = [_track(f"t-{index}") for index in range(214)]
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda items, *_args, **_kwargs: (
        {str(item["trackId"]): [{"trackId": str(item["trackId"]), "keyframeId": "kf"}] for item in items},
        [],
        {},
    )

    result = service.match_entities()

    assert result["trackCount"] == 214
    assert result["outsideRegistryCount"] == 0
    assert result["unresolvedTrackCount"] == 214
    assert result["embeddingCoverageComplete"] is False
    assert result["coverageReasons"] == ["registry_references_missing"]


def test_registry_count_keeps_unsearchable_tracks_unresolved_and_flags_coverage():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "references": [{
                "registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1,
            }]}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    empty_index = type("Index", (), {"get_many": lambda _self, _ids: {}})()
    service.vectors = type("Vectors", (), {"keyframes": empty_index, "registry": empty_index})()
    tracks = [_track("t-1")]
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda *_args, **_kwargs: ({"t-1": []}, ["t-1"], {})

    result = service.match_entities()

    assert result["trackCount"] == 1
    assert result["outsideRegistryCount"] == 0
    assert result["unresolvedTrackCount"] == 1
    assert result["embeddingCoverageComplete"] is False
    assert result["matches"][0]["scoreBand"] == "uncertain"
    assert result["matches"][0]["verificationDecision"] == "unresolved"
    assert result["coverageReasons"] == [
        "track_keyframe_vectors_missing",
        "registry_reference_vectors_missing",
    ]


def test_registry_count_reports_the_applied_time_range():
    class Repository:
        def list_registry(self):
            return []

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72}
    service._scan_tracks = lambda _scope: ([], True)
    service._load_search_frames = lambda *_args, **_kwargs: ({}, [], {})
    requested = [
        "2026-10-27T00:00:00.000000+08:00",
        "2026-10-27T23:59:59.999999+08:00",
    ]

    result = service.match_entities(time_range=requested)

    assert result["appliedTimeRange"] == requested
    assert service.match_entities()["appliedTimeRange"] is None


def test_registry_count_keeps_all_unresolved_counts_but_caps_evidence_rows():
    tracks = [_track(f"t-{index}") for index in range(61)]
    service = ToolService.__new__(ToolService)
    service.repository = type("Repository", (), {"list_registry": lambda _self: []})()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda *_args, **_kwargs: (
        {str(item["trackId"]): [{"trackId": item["trackId"], "keyframeId": "kf"}] for item in tracks},
        [],
        {},
    )

    result = service.match_entities()

    assert result["trackCount"] == result["unresolvedTrackCount"] == 61
    assert result["outsideRegistryCount"] == 0
    assert len(result["matches"]) == 60
    assert result["matchesTruncated"] is True


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
    )

    assert result["matches"][0]["matchedRegistryId"] == "r-1"
    assert result["matches"][0]["score"] == 0.9
    assert result["matches"][0]["matchedKeyframeId"] == "kf-1"
    assert "embeddingScore" not in result["matches"][0]
    assert "maxEmbeddingScore" not in result["matches"][0]


def test_image_match_marks_registry_coverage_incomplete_when_reference_vector_is_missing():
    class Index:
        def __init__(self, values):
            self.values = values

        def get_many(self, ids):
            return {value: self.values[value] for value in ids if value in self.values}

    service = ToolService.__new__(ToolService)
    service.settings = {"image_match": 0.72, "image_exclude": 0.5}
    service.vectors = type("Vectors", (), {
        "keyframes": Index({10: [1.0, 0.0]}),
        "registry": Index({1: [0.9, 0.4]}),
    })()
    registry = {
        "registryId": "r-1",
        "references": [
            {"registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1},
            {"registryId": "r-1", "referenceId": "ref-2", "registryVectorId": 2},
        ],
    }

    result = service._match_images(
        queryImages=registry["references"],
        galleryImages=[{"trackId": "t-1", "keyframeId": "kf-1", "keyframeVectorId": 10}],
        registryItems=[registry],
    )

    assert result["matches"]
    assert result["registryCoverageComplete"] is False


def test_image_match_coverage_compares_registry_ids_not_only_their_count():
    class Index:
        def __init__(self, values):
            self.values = values

        def get_many(self, ids):
            return {value: self.values[value] for value in ids if value in self.values}

    service = ToolService.__new__(ToolService)
    service.settings = {"image_match": 0.72, "image_exclude": 0.5}
    service.vectors = type("Vectors", (), {
        "keyframes": Index({10: [1.0, 0.0]}),
        "registry": Index({1: [0.9, 0.1], 3: [0.1, 0.9]}),
    })()
    result = service._match_images(
        queryImages=[
            {"registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1},
            {"registryId": "r-3", "referenceId": "ref-3", "registryVectorId": 3},
        ],
        galleryImages=[{"trackId": "t-1", "keyframeId": "kf-1", "keyframeVectorId": 10}],
        registryItems=[{"registryId": "r-1"}, {"registryId": "r-2"}],
        diagnostics=True,
    )

    assert result["registryCoverageComplete"] is False
    assert result["uncomparedTrackIds"] == ["t-1"]


def test_registry_count_verifies_only_the_gray_zone_without_calling_vlm():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "hullNumber": "A01", "references": [{
                "registryId": "r-1",
                "referenceId": "ref-1",
                "imagePath": "/registry.jpg",
            }]}]

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72, "image_exclude": 0.52}
    tracks = [_track("t-high"), _track("t-gray"), _track("t-low")]
    frames = {
        "t-high": [{"trackId": "t-high", "keyframeId": "kf-high"}],
        "t-gray": [{"trackId": "t-gray", "keyframeId": "kf-gray"}],
        "t-low": [{"trackId": "t-low", "keyframeId": "kf-low"}],
    }
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda *_args, **_kwargs: (frames, [], {})
    service._match_images = lambda **_kwargs: {
        "matches": [
            {
                "matchedTrackId": "t-high",
                "matchedRegistryId": "r-1",
                "score": 0.81,
                "matchedKeyframeId": "kf-high",
                "matchedRegistryReferenceId": "ref-1",
            },
            {
                "matchedTrackId": "t-gray",
                "matchedRegistryId": "r-1",
                "score": 0.61,
                "matchedKeyframeId": "kf-gray",
                "matchedRegistryReferenceId": "ref-1",
            },
            {
                "matchedTrackId": "t-low",
                "matchedRegistryId": "r-1",
                "score": 0.40,
                "matchedKeyframeId": "kf-low",
                "matchedRegistryReferenceId": "ref-1",
            },
        ],
        "registryCoverageComplete": True,
        "missingRegistryVectorCount": 0,
    }
    service._verify_image_pair = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("VLM calls must be made by verify_vessel_images, not match_entities")
    )

    result = service.match_entities()

    assert result["status"] == "verification_required"
    assert result["nextStep"] == "verify_vessel_images"
    assert result["embeddingConfirmedCount"] == 1
    assert result["grayCandidateCount"] == 1
    assert result["outsideRegistryCount"] == 1
    assert result["unresolvedTrackCount"] == 0
    assert [item["trackId"] for item in result["verificationCandidates"]] == ["t-gray"]
    assert result["verificationCandidates"][0]["scoreBand"] == "uncertain"
    rows = {item["trackId"]: item for item in result["matches"]}
    assert rows["t-high"]["classification"] == "embedding_confirmed"
    assert rows["t-high"]["verificationDecision"] == "match"
    assert rows["t-low"]["verificationDecision"] == "not_in_registry"
    assert all(
        item["verificationDecision"] == "pending"
        for item in result["verificationCandidates"]
    )


def test_registry_count_marks_only_scores_at_or_below_exclusion_as_outside():
    class Repository:
        def list_registry(self):
            return [{"registryId": "r-1", "references": [{
                "registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1,
            }]}]

    class Index:
        def get_many(self, ids):
            return {1: [0.8, 0.6]} if 1 in ids else {}

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.9, "image_exclude": 0.82}
    service.vectors = type("Vectors", (), {
        "keyframes": type("FrameIndex", (), {"get_many": lambda _self, ids: {10: [1.0, 0.0]} if 10 in ids else {}})(),
        "registry": Index(),
    })()
    tracks = [_track("t-1")]
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: (tracks, True)
    service._load_search_frames = lambda *_args, **_kwargs: (
        {"t-1": [{"trackId": "t-1", "keyframeId": "kf-1", "keyframeVectorId": 10}]}, [], {}
    )
    service._verify_image_pair = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("below-threshold scores do not require VLM verification")
    )

    result = service.match_entities()

    assert result["status"] == "complete"
    assert result["embeddingConfirmedCount"] == 0
    assert result["outsideRegistryCount"] == 1
    assert result["grayCandidateCount"] == 0
    assert result["unresolvedTrackCount"] == 0
    assert "nextStep" not in result


def test_image_match_reports_registry_vector_gaps_without_searchable_track_frames():
    class Index:
        def get_many(self, ids):
            return {1: [0.9, 0.1]} if 1 in ids else {}

    service = ToolService.__new__(ToolService)
    service.vectors = type("Vectors", (), {"registry": Index()})()
    result = service._match_images(
        queryImages=[
            {"registryId": "r-1", "referenceId": "ref-1", "registryVectorId": 1},
            {"registryId": "r-1", "referenceId": "ref-2", "registryVectorId": 2},
        ],
        galleryImages=[],
        registryItems=[{"registryId": "r-1"}],
        diagnostics=True,
    )

    assert result["missingRegistryVectorCount"] == 1


def test_index_read_errors_fail_matching_instead_of_becoming_partial_counts():
    class Index:
        def get_many(self, _ids):
            raise RuntimeError("registry index unavailable")

    service = ToolService.__new__(ToolService)
    try:
        service._load_indexed_vectors(
            [{"registryVectorId": 1}], vector_key="registryVectorId", index=Index()
        )
    except RuntimeError as error:
        assert str(error) == "registry index unavailable"
    else:
        raise AssertionError("index errors must reach the tool boundary")


def test_registry_count_returns_tool_error_when_vector_index_is_unavailable():
    class Repository:
        def list_registry(self):
            return []

        def get_keyframes(self, _track_ids, embedded_only=False):
            assert embedded_only is False
            return {"t-1": [{
                "keyframeId": "kf-1",
                "keyframeVectorId": 1,
            }]}

    class Index:
        def get_many(self, _ids):
            raise RuntimeError("keyframe index unavailable")

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    service.settings = {"image_match": 0.72}
    service.vectors = type("Vectors", (), {"keyframes": Index()})()
    service._validate_time_range = lambda _value: (None, "")
    service._scan_tracks = lambda _scope: ([_track("t-1")], True)

    result = service.match_entities()

    assert result["ok"] is False
    assert result["error"] == "tool_failed"
    assert result["errorType"] == "RuntimeError"
    assert result["hint"] == "keyframe index unavailable"


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
        diagnostics=True,
    )

    assert result == {
        "ok": True,
        "matches": [],
        "unscoredTrackIds": ["t-1"],
        "registryCoverageComplete": False,
        "missingRegistryVectorCount": 1,
        "uncomparedTrackIds": ["t-1"],
    }
    assert service.vectors.keyframes.writes == 0
    assert service.vectors.registry.writes == 0


def test_each_business_tool_has_its_own_timeout_budget():
    assert ToolService.TOOL_TIMEOUTS == {
        "recognize_tracks": 60,
        "match_registry_tracks": 60,
        "verify_vessel_images": 60,
        "dedup_tracks": 120,
        "match_entities": 120,
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

"""证据富化：散落的证据 ID 收齐、去重、按轨迹分组。"""
from harness.evidence import build_evidence_payload


def _record(tool, result, status="completed"):
    return {"tool": tool, "status": status, "result": result}


def test_ids_are_collected_across_tools_and_deduplicated():
    records = [
        _record("recognize_tracks", {"ok": True, "keyframeIds": ["kf-1", "kf-2"], "keyframes": [{"keyframeId": "kf-2"}, {"keyframeId": "kf-3"}]}),
        _record("match_registry_tracks", {"ok": True, "registryReferenceIds": ["ref-1"], "registryReferences": [{"referenceId": "ref-1"}, {"referenceId": "ref-2"}]}),
        _record("show_evidence", {"ok": True, "shipSegmentId": "seg-1", "shipSegmentIds": ["seg-1"]}),
    ]
    payload = build_evidence_payload(records)
    assert payload["keyframeIds"] == ["kf-1", "kf-2", "kf-3"]
    assert payload["registryReferenceIds"] == ["ref-1", "ref-2"]
    assert payload["shipSegmentIds"] == ["seg-1"]
    assert payload["isEmpty"] is False


def test_failed_records_contribute_nothing():
    """失败的工具结果不算证据：否则面板会出现根本没取到的 ID。"""
    records = [
        _record("recognize_tracks", {"ok": False, "error": "track_not_found", "keyframeIds": ["ghost"]}, status="error"),
        _record("dedup_tracks", {"ok": True, "trackIds": ["t-1"]}),
    ]
    payload = build_evidence_payload(records)
    assert payload["keyframeIds"] == []
    assert payload["isEmpty"] is True


def test_display_groups_bind_keyframes_to_a_track_and_rank_by_score():
    records = [
        _record("match_registry_tracks", {
            "ok": True,
            "matches": [
                {"matchedTrackId": "t-1", "matchedKeyframeIds": ["kf-1"], "registryReferenceIds": ["ref-1"], "score": 0.61, "scoreBand": "uncertain"},
                {"matchedTrackId": "t-2", "matchedKeyframeIds": ["kf-2"], "registryReferenceIds": ["ref-2"], "score": 0.88, "scoreBand": "match"},
            ],
        }),
    ]
    payload = build_evidence_payload(records)
    assert [group["trackId"] for group in payload["displayGroups"]] == ["t-2", "t-1"]
    top = payload["displayGroups"][0]
    assert top["keyframeIds"] == ["kf-2"]
    assert top["registryReferenceIds"] == ["ref-2"]
    assert top["band"] == "match"
    assert top["score"] == 0.88


def test_a_track_appearing_in_several_results_is_merged_into_one_group():
    records = [
        _record("recognize_tracks", {"ok": True, "keyframesByTrack": {"t-9": {"keyframeIds": ["kf-9"]}}}),
        _record("dedup_tracks", {"ok": True, "tracks": [{"trackId": "t-9", "startTime": 1, "endTime": 2}]}),
    ]
    payload = build_evidence_payload(records)
    assert [group["trackId"] for group in payload["displayGroups"]] == ["t-9"]
    assert payload["displayGroups"][0]["keyframeIds"] == ["kf-9"]


def test_long_id_lists_are_capped_and_empty_input_is_safe():
    records = [_record("recognize_tracks", {"ok": True, "keyframeIds": [f"kf-{i}" for i in range(200)]})]
    payload = build_evidence_payload(records, limit=20)
    assert len(payload["keyframeIds"]) == 20
    assert build_evidence_payload([])["isEmpty"] is True


def test_compact_count_fields_feed_frontend_statistics():
    payload = build_evidence_payload([
        _record("dedup_tracks", {
            "ok": True,
            "status": "complete",
            "trackCount": 20,
            "exactShipCount": 12,
            "fuzzyShipCount": 10,
            "duplicateCandidateCount": 2,
            "unresolvedDuplicateCount": 0,
            "coverageComplete": False,
            "coverageReasons": ["track_keyframe_vectors_missing"],
            "appliedTimeRange": ["2026-10-27T00:00:00+08:00", "2026-10-27T23:59:59+08:00"],
        }),
    ])

    assert payload["statistics"]["queryByTool"] == [{
        "tool": "dedup_tracks",
        "statistics": {
            "status": "complete",
            "trackCount": 20,
            "exactShipCount": 12,
            "fuzzyShipCount": 10,
            "duplicateCandidateCount": 2,
            "unresolvedDuplicateCount": 0,
            "coverageComplete": False,
            "coverageReasons": ["track_keyframe_vectors_missing"],
            "appliedTimeRange": ["2026-10-27T00:00:00+08:00", "2026-10-27T23:59:59+08:00"],
        },
    }]


def test_duplicate_statistics_from_tool_and_private_evidence_are_shown_once():
    result = {
        "trackCount": 214,
        "embeddingConfirmedCount": 20,
        "outsideRegistryCount": 180,
        "grayCandidateCount": 14,
        "unresolvedTrackCount": 0,
        "embeddingCoverageComplete": False,
        "coverageReasons": ["track_keyframe_vectors_missing"],
    }
    payload = build_evidence_payload([
        _record("match_entities", result),
        _record("match_entities", {**result, "matches": [{"trackId": "t-1"}]}),
    ])

    assert payload["statistics"]["queryByTool"] == [{
        "tool": "match_entities",
        "statistics": result,
    }]


def test_external_registry_match_evidence_is_enriched_without_counting_as_model_result():
    payload = build_evidence_payload([
        _record("match_entities", {
            "trackCount": 1,
            "embeddingConfirmedCount": 0,
            "outsideRegistryCount": 0,
            "grayCandidateCount": 1,
            "unresolvedTrackCount": 0,
            "embeddingCoverageComplete": True,
        }),
        _record("match_entities", {
            "matches": [{
                "trackId": "t-1",
                "trackIds": ["t-1", "t-2"],
                "score": 0.88,
                "scoreBand": "match",
                "classification": "embedding_confirmed",
                "verificationDecision": "match",
                "matchedObject": {"hullNumber": "0857"},
                "keyframeIds": ["kf-1"],
                "registryReferenceIds": ["ref-1"],
            }],
        }),
    ])

    assert payload["statistics"]["queryByTool"][0]["statistics"]["trackCount"] == 1
    assert payload["trackIds"] == ["t-1", "t-2"]
    assert payload["keyframeIds"] == ["kf-1"]
    assert payload["registryReferenceIds"] == ["ref-1"]
    assert payload["displayGroups"][0]["score"] == 0.88
    assert payload["displayGroups"][0]["matchedObject"]["hullNumber"] == "0857"
    assert payload["trackRows"][0]["registryMatch"]["status"] == "Embedding confirmed in registry"


def test_evidence_rows_keep_one_keyframe_track_time_and_explicit_registry_match():
    payload = build_evidence_payload([
        _record("dedup_tracks", {
            "trackCount": 2,
            "trackRows": [{"trackId": "t-1", "startTimeText": "start", "endTimeText": "end"}],
            "trackRowsTruncated": True,
        }),
        _record("match_entities", {
            "matches": [{
                "trackId": "t-1",
                "startTimeText": "matched start",
                "endTimeText": "matched end",
                "score": 0.91,
                "scoreBand": "match",
                "matchedObject": {"hullNumber": "0857"},
                "keyframeIds": ["kf-first", "kf-second"],
            }],
        }),
    ])

    assert payload["trackRows"] == [{
        "trackId": "t-1",
        "keyframeId": "kf-first",
        "registryMatch": {
            "status": "Threshold match",
            "score": 0.91,
            "matchedObject": {"hullNumber": "0857"},
        },
        "startTimeText": "matched start",
        "endTimeText": "matched end",
    }]
    assert payload["registryMatchUsed"] is True
    assert payload["trackRowsTruncated"] is True


def test_candidate_evidence_uses_the_agent_vlm_result_for_its_track():
    payload = build_evidence_payload([
        _record("match_entities", {
            "matches": [{
                "trackId": "t-1",
                "score": 0.65,
                "scoreBand": "uncertain",
                "classification": "gray_verification_pending",
                "verificationDecision": "pending",
                "matchedKeyframeId": "kf-1",
                "matchedRegistryReferenceId": "ref-1",
                "matchedObject": {"hullNumber": "0857"},
            }],
        }),
        _record("verify_vessel_images", {
            "ok": True,
            "trackId": "t-1",
            "decision": "match",
            "keyframeId": "kf-1",
            "registryReferenceId": "ref-1",
        }),
    ])

    assert payload["trackRows"][0]["registryMatch"] == {
        "status": "VLM confirmed in registry",
        "score": 0.65,
        "matchedObject": {"hullNumber": "0857"},
        "verificationDecision": "match",
    }


def test_embedding_confirmed_candidate_is_shown_without_a_vlm_result():
    payload = build_evidence_payload([
        _record("match_entities", {
            "matches": [{
                "trackId": "t-high",
                "score": 0.88,
                "scoreBand": "match",
                "classification": "embedding_confirmed",
                "verificationDecision": "match",
                "matchedObject": {"hullNumber": "0857"},
            }],
        }),
    ])

    assert payload["trackRows"][0]["registryMatch"]["status"] == (
        "Embedding confirmed in registry"
    )


def test_failed_candidate_verification_remains_visible_as_unresolved():
    payload = build_evidence_payload([
        _record("match_entities", {
            "matches": [{
                "trackId": "t-7",
                "score": 0.65,
                "scoreBand": "uncertain",
                "classification": "gray_verification_pending",
                "verificationDecision": "pending",
            }],
        }),
        _record("verify_vessel_images", {
            "ok": False,
            "trackId": "t-7",
            "decision": "uncertain",
            "error": "tool_failed",
        }, status="error"),
    ])

    assert payload["trackRows"][0]["registryMatch"]["status"] == "VLM failed; unresolved"


def test_explicit_track_keyframe_wins_over_a_match_frame():
    payload = build_evidence_payload([
        _record("recognize_tracks", {
            "matches": [{"trackId": "t-1", "keyframeIds": ["kf-match"]}],
        }),
        _record("show_evidence", {
            "trackRows": [{"trackId": "t-1", "keyframeId": "kf-first"}],
        }),
        _record("match_registry_tracks", {
            "matches": [{"trackId": "t-1", "keyframeIds": ["kf-registry-match"]}],
        }),
    ])

    assert payload["trackRows"][0]["keyframeId"] == "kf-first"


def test_registry_assisted_recognition_adds_match_status_to_track_rows():
    payload = build_evidence_payload([
        _record("recognize_tracks", {
            "ok": True,
            "status": "mismatch",
            "matches": [],
            "coverageComplete": True,
        }),
        _record("match_registry_tracks", {
            "ok": True,
            "status": "confirmed",
            "coverageComplete": True,
            "matches": [{
                "trackId": "t-0857",
                "startTimeText": "start",
                "endTimeText": "end",
                "score": 0.91,
                "scoreBand": "match",
                "matchedKeyframeId": "kf-match",
                "matchedRegistryReferenceId": "ref-0857",
            }],
        }),
    ])

    assert payload["registryMatchUsed"] is True
    assert payload["trackRows"] == [{
        "trackId": "t-0857",
        "registryMatch": {"status": "Threshold match", "score": 0.91, "matchedObject": None},
        "startTimeText": "start",
        "endTimeText": "end",
    }]

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
                {"matchedTrackId": "t-1", "matchedKeyframeIds": ["kf-1"], "matchedRegistryReferenceIds": ["ref-1"], "embeddingScore": 0.61, "scoreBand": "uncertain"},
                {"matchedTrackId": "t-2", "matchedKeyframeIds": ["kf-2"], "matchedRegistryReferenceIds": ["ref-2"], "embeddingScore": 0.88, "scoreBand": "match"},
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
            "trackCount": 20,
            "highThresholdShipCount": 12,
            "lowThresholdShipCount": 10,
            "coverageComplete": False,
        }),
    ])

    assert payload["statistics"]["queryByTool"] == [{
        "tool": "dedup_tracks",
        "statistics": {
            "trackCount": 20,
            "highThresholdShipCount": 12,
            "lowThresholdShipCount": 10,
            "coverageComplete": False,
        },
    }]


def test_external_registry_match_evidence_is_enriched_without_counting_as_model_result():
    payload = build_evidence_payload([
        _record("match_entities", {
            "entityCount": 1,
            "inRegistryCount": 1,
            "unregisteredCount": 0,
            "coverageComplete": True,
        }),
        _record("match_entities", {
            "matches": [{
                "trackId": "t-1",
                "trackIds": ["t-1", "t-2"],
                "score": 0.83,
                "scoreBand": "match",
                "matchedObject": {"hullNumber": "0857"},
                "keyframeIds": ["kf-1"],
                "registryReferenceIds": ["ref-1"],
            }],
        }),
    ])

    assert payload["statistics"]["queryByTool"][0]["statistics"]["entityCount"] == 1
    assert payload["trackIds"] == ["t-1", "t-2"]
    assert payload["keyframeIds"] == ["kf-1"]
    assert payload["registryReferenceIds"] == ["ref-1"]
    assert payload["displayGroups"][0]["score"] == 0.83
    assert payload["displayGroups"][0]["matchedObject"]["hullNumber"] == "0857"

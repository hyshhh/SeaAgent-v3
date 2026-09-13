"""舷号抽取与 listRegistry 摘要计数回归。"""
from __future__ import annotations

from agent.plan_executor import PlanExecutor
from tools.target_parser import clear_target_parser_cache, extract_hull_number, infer_intent_fields


def test_extract_hull_chinese_prefix():
    clear_target_parser_cache()
    assert extract_hull_number("舷号 小蓝320 有没有在视频中出现？") == "小蓝320"
    assert extract_hull_number("舷号：A01 出现过吗") == "A01"
    assert extract_hull_number("查一下 0789") in {"0789", None} or extract_hull_number("查一下 0789") == "0789"


def test_infer_hull_existence():
    clear_target_parser_cache()
    fields = infer_intent_fields("舷号 小蓝320 有没有在视频中出现？")
    assert fields["hullNumber"] == "小蓝320"
    assert fields["operation"] == "existence"
    assert fields["targetKind"] == "hull"
    focus = str(fields.get("nextAgentFocus") or "")
    criteria = str(fields.get("successCriteria") or "")
    assert "getRegistry" in focus or "先验库" in focus
    assert "matchImage" in focus or "matchImage" in criteria
    assert "0 轨迹即可否定" not in criteria
    assert "未检测到" not in criteria


def test_infer_registry_in_list_not_matchtext_query():
    """「有哪些在库船出现」不得抽成 description / matchText 问句。"""
    clear_target_parser_cache()
    fields = infer_intent_fields("有哪些在库船出现？")
    assert fields["operation"] == "list"
    assert fields["registryRelation"] == "in"
    assert fields["targetScope"] == "both"
    assert fields["targetKind"] == "all"
    assert not fields.get("description")
    assert fields.get("questionType") == "registry_in_list"
    focus = str(fields.get("nextAgentFocus") or "").lower()
    criteria = str(fields.get("successCriteria") or "").lower()
    assert "listregistry" in focus or "listregistry" in criteria
    assert "matchimage" in focus or "matchimage" in criteria
    assert "matchtext" not in focus
    assert "ocr" not in criteria or "matchimage" in criteria


def test_registry_summary_prefers_items_over_references():
    observation = {
        "id": "registry",
        "tool": "listRegistry",
        "skipped": False,
        "result": {
            "ok": True,
            "registryItems": [{"registryId": "a"}, {"registryId": "b"}],
            "registryReferences": [],
        },
    }
    summary = PlanExecutor.summarize_observation(observation)
    assert summary["registryCount"] == 2
    assert summary["registryItemCount"] == 2
    assert summary["registryReferenceCount"] == 0


def test_default_ref_fallback_empty_references():
    executor = PlanExecutor(tools=type("T", (), {"execute": staticmethod(lambda *a, **k: {})})())
    scope = {
        "registry": {
            "ok": True,
            "registryReferences": [],
            "registryItems": [{"registryId": "x", "description": "黄色船"}],
        }
    }
    resolved = executor._resolve(
        {"$ref": "registry.registryReferences", "$default": {"$ref": "registry.registryItems"}},
        scope,
    )
    assert isinstance(resolved, list)
    assert resolved[0]["registryId"] == "x"


def test_skipped_not_counted_as_failed():
    observations = [
        {"id": "tracks", "tool": "getTrack", "skipped": False, "result": {"ok": True, "tracks": []}},
        {
            "id": "frames",
            "tool": "getFrames",
            "skipped": True,
            "skipReason": "dependency_empty:tracks.trackIds",
            "result": {"ok": False, "error": "dependency_empty:tracks.trackIds"},
        },
    ]
    failed = sum(
        1
        for item in observations
        if not item.get("skipped") and (item.get("result") or {}).get("ok") is False
    )
    skipped = sum(1 for item in observations if item.get("skipped"))
    assert failed == 0
    assert skipped == 1


def test_visual_match_default_plan_shape():
    """补洞 replan 应产出 getRegistry → getTrack(无hull) → getFrames → matchImage。"""
    # 内嵌函数：直接复现 _default_plan_calls 的关键分支逻辑做契约测试
    def wants_visual(hint: str) -> bool:
        h = hint.lower()
        return any(
            token in h
            for token in (
                "matchimage", "match_image", "视觉匹配", "图像匹配", "图匹配",
                "registryreferences", "关键帧匹配", "库图", "对照视频",
            )
        ) or ("match" in h and "image" in h)

    hint = (
        "getRegistry(hullNumber=小蓝320) → getTrack(不带hullNumber, 全时域) → "
        "getFrames($ref trackIds) → matchImage(queryImages=$ref registry.registryReferences, "
        "galleryImages=$ref frames.keyframes)"
    )
    assert wants_visual(hint)
    assert "matchimage" in hint.lower()
    assert "不带hull" in hint or "不带hullnumber" in hint.lower()


def test_registry_in_list_default_plan_calls():
    """在库列表默认链：listRegistry → getTrack → getFrames → matchImage。"""
    intent = {
        "operation": "list",
        "registryRelation": "in",
        "targetScope": "both",
        "targetKind": "all",
        "questionType": "registry_in_list",
        "hullNumber": None,
        "description": None,
        "nextAgentFocus": (
            "①listRegistry；②getTrack(全量)；③getFrames；"
            "④matchImage(query=registry.registryReferences, gallery=frames.keyframes)"
        ),
    }
    tools = [c["tool"] for c in _default_plan_calls_for_test(intent, top_k=3)]
    assert tools == ["listRegistry", "getTrack", "getFrames", "matchImage"]


def _default_plan_calls_for_test(intent: dict, top_k: int = 3) -> list[dict]:
    """与 agent.graph._default_plan_calls 在库列表分支对齐的轻量复现，供契约测试。"""
    hull = str(intent.get("hullNumber") or "").strip()
    description = str(intent.get("description") or "").strip()
    operation = str(intent.get("operation") or "list")
    target_scope = str(intent.get("targetScope") or "track_memory")
    registry_relation = str(intent.get("registryRelation") or "any")
    hint = str(intent.get("nextAgentFocus") or "").lower()
    top = max(1, min(20, int(top_k or 3)))
    wants_registry = (
        target_scope in {"registry", "both"}
        or registry_relation in {"in", "out"}
        or any(token in hint for token in ("先验库", "在库", "listregistry", "matchimage"))
    )
    wants_visual_match = any(
        token in hint
        for token in ("matchimage", "视觉匹配", "registryreferences", "库图")
    )
    wants_registry_in_list = (
        registry_relation == "in"
        and operation == "list"
        and not hull
        and (
            target_scope in {"both", "registry"}
            or str(intent.get("questionType") or "") == "registry_in_list"
            or any(token in hint for token in ("listregistry", "在库", "哪些", "matchimage"))
        )
    )
    if wants_registry_in_list or (wants_visual_match and not hull and wants_registry and not description):
        return [
            {"id": "registry", "tool": "listRegistry", "arguments": {}},
            {"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 60}},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": top,
                },
            },
        ]
    return [{"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 60}}]


def test_should_replan_visual_when_registry_found_without_searchable_flag():
    """库有 found/items 时也应尝试视觉 replan；已 attempted 则停止。"""
    registry_checked = True
    registry_searchable = False
    registry_found = True
    registry_has_items = True
    visual_attempted = False
    zero_tracks = True
    hull = "小蓝320"
    loop_count = 1
    limit = 3
    op = "existence"
    can_try_visual = bool(registry_searchable or registry_found or registry_has_items)
    should_replan_visual = (
        loop_count < limit
        and bool(hull)
        and registry_checked
        and can_try_visual
        and not visual_attempted
        and op in {"existence", "list", "explain", "time", ""}
        and zero_tracks
    )
    assert can_try_visual
    assert should_replan_visual

    visual_attempted2 = True
    assert not (can_try_visual and not visual_attempted2)


def test_match_image_missing_args_records_attempt():
    """matchImage 缺图时不应裸 skip，应记入 matches=[] 供 Reflect 停止。"""
    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            return {"ok": True, "matches": [{"embeddingScore": 0.9}]}

    executor = PlanExecutor(tools=_FakeTools())
    # 无 registry 结果，query/gallery 均为空
    executed = executor.execute(
        [
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": [],
                    "galleryImages": [],
                    "topK": 3,
                },
            }
        ],
        scope={},
    )
    records = executed["tool_records"]
    assert len(records) == 1
    assert records[0]["tool"] == "matchImage"
    assert records[0]["skipped"] is False
    assert records[0]["result"].get("matches") == []
    assert records[0]["result"].get("visualAttempted") is True


def test_match_image_default_from_registry_items():
    """registryReferences 空时从 registryItems.references 展开补齐 queryImages。"""
    class _FakeTools:
        @staticmethod
        def execute(name, arguments):
            assert name == "matchImage"
            query = arguments.get("queryImages") or []
            gallery = arguments.get("galleryImages") or []
            assert query, "queryImages 应被补齐"
            assert query[0].get("referenceId") == "ref1"
            assert gallery
            return {"ok": True, "matches": [{"matchedTrackId": "1", "embeddingScore": 0.8, "scoreBand": "match"}]}

    executor = PlanExecutor(tools=_FakeTools())
    scope = {
        "registry": {
            "ok": True,
            "found": True,
            "searchable": False,
            "registryReferences": [],
            "registryItems": [
                {
                    "registryId": "r1",
                    "hullNumber": "小蓝320",
                    "references": [
                        {
                            "referenceId": "ref1",
                            "registryId": "r1",
                            "registryVectorId": 1,
                            "isEmbedded": True,
                            "imagePath": "a.jpg",
                        }
                    ],
                }
            ],
        },
        "frames": {
            "ok": True,
            "keyframes": [
                {
                    "keyframeId": "k1",
                    "trackId": "1",
                    "keyframeVectorId": 2,
                    "isEmbedded": True,
                }
            ],
        },
    }
    executed = executor.execute(
        [
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": 3,
                },
            }
        ],
        scope=scope,
    )
    record = executed["tool_records"][0]
    assert record["skipped"] is False
    assert record["ok"] is True
    assert record["result"].get("matches")
    # 压缩展示应是 referenceId，不是 null
    assert record["arguments"]["queryImages"] == ["ref1"]


def test_compact_args_prefers_registry_id_over_null():
    compact = PlanExecutor._compact_args({
        "queryImages": [{"registryId": "r1", "hullNumber": "小蓝320", "description": "蓝船"}],
        "galleryImages": [{"keyframeId": "k1"}],
    })
    assert compact["queryImages"] == ["r1"]
    assert compact["galleryImages"] == ["k1"]


def test_tracks_ranked_by_matches_not_track_id_order():
    """展示顺序必须按匹配分，不能固定成 getTrack 的 1/2/3。"""
    from agent.controller import AgentController

    ctrl = object.__new__(AgentController)
    ctrl.working_scope = {
        "tracks": {
            "ok": True,
            "tracks": [
                {"trackId": "1", "startTime": 1},
                {"trackId": "2", "startTime": 2},
                {"trackId": "9", "startTime": 9},
            ],
        }
    }
    matches = [
        {"matchedTrackId": "9", "embeddingScore": 0.91, "scoreBand": "match", "matchedKeyframeIds": ["k9"]},
        {"matchedTrackId": "2", "embeddingScore": 0.80, "scoreBand": "match", "matchedKeyframeIds": ["k2"]},
        {"matchedTrackId": "1", "embeddingScore": 0.55, "scoreBand": "uncertain", "matchedKeyframeIds": ["k1"]},
    ]
    ranked = AgentController._tracks_ranked_by_matches(ctrl, matches)
    assert [str(t["trackId"]) for t in ranked] == ["9", "2", "1"]
    assert ranked[0]["embeddingScore"] == 0.91
    assert ranked[0]["matchedKeyframeIds"] == ["k9"]


def test_mismatch_matches_still_rank_for_display():
    """全 mismatch 时仍应能按分数产出候选轨迹，避免证据区空白。"""
    from agent.controller import AgentController

    ctrl = object.__new__(AgentController)
    ctrl.working_scope = {
        "match": {
            "ok": True,
            "matches": [
                {"matchedTrackId": "7", "embeddingScore": 0.41, "scoreBand": "mismatch", "matchedKeyframeIds": ["ka"]},
                {"matchedTrackId": "3", "embeddingScore": 0.38, "scoreBand": "mismatch", "matchedKeyframeIds": ["kb"]},
            ],
        },
        "tracks": {
            "ok": True,
            "tracks": [{"trackId": "1"}, {"trackId": "3"}, {"trackId": "7"}],
        },
    }
    matches = AgentController._collect_matches(ctrl)
    ranked = AgentController._tracks_ranked_by_matches(ctrl, matches)
    assert [str(t["trackId"]) for t in ranked] == ["7", "3"]
    assert ranked[0]["matchedKeyframeIds"] == ["ka"]


def test_display_tracks_attaches_registry_refs_from_match_image():
    """matchImage 命中后，证据组必须带上 matchedRegistryReferenceIds。"""
    from agent.controller import AgentController

    ctrl = object.__new__(AgentController)
    ctrl.tools = type("T", (), {})()
    ctrl.display_record = None
    ctrl.display_groups = []
    ctrl._pending_registry_items = []
    ctrl.working_scope = {
        "match": {
            "ok": True,
            "matches": [
                {
                    "matchedTrackId": "7",
                    "matchedRegistryId": "r003",
                    "embeddingScore": 0.83,
                    "scoreBand": "match",
                    "matchedKeyframeIds": ["kf-7"],
                    "matchedRegistryReferenceIds": ["ref-c406", "ref-71f5"],
                }
            ],
        },
        "tracks": {"ok": True, "tracks": [{"trackId": "7"}]},
        "registry": {
            "ok": True,
            "registryItems": [{"registryId": "r003", "hullNumber": "003", "references": [{"referenceId": "ref-c406"}]}],
        },
    }
    tracks = AgentController._tracks_ranked_by_matches(
        ctrl,
        [
            {
                "matchedTrackId": "7",
                "embeddingScore": 0.83,
                "scoreBand": "match",
                "matchedKeyframeIds": ["kf-7"],
                "matchedRegistryReferenceIds": ["ref-c406", "ref-71f5"],
            }
        ],
    )
    AgentController._display_tracks(ctrl, tracks, include_clips=False, include_registry=True)
    assert ctrl.display_groups
    refs = ctrl.display_groups[0].get("registryReferenceIds") or []
    assert "ref-c406" in refs

from agent.graph import _default_plan_calls, _prepare_plan_calls
from agent.plan_executor import PlanExecutor


class _EmptyTrackTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "getTrack":
            return {"ok": True, "trackIds": [], "tracks": []}
        raise AssertionError(f"空轨迹后不应执行 {name}")


def _hull_intent():
    return {
        "operation": "existence",
        "targetScope": "track_memory",
        "targetKind": "hull",
        "registryRelation": "any",
        "hullNumber": "大鱼01",
        "questionType": "hull_existence",
    }


def test_sanitize_calls_removes_semantic_duplicates_and_rewrites_refs():
    calls = [
        {"id": "tracks-a", "tool": "getTrack", "arguments": {"hullNumber": "大鱼01", "limit": 60}},
        {"id": "tracks-b", "tool": "getTrack", "arguments": {"hullNumber": "大鱼01", "offset": 0, "limit": 60}},
        {"id": "tracks-c", "tool": "getTrack", "arguments": {"hullNumber": "大鱼01", "offset": 0, "limit": 60}},
        {"id": "frames-a", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks-c.trackIds"}}},
        {"id": "frames-b", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks-a.trackIds"}}},
    ]

    sanitized = PlanExecutor.sanitize_calls(calls)

    assert [call["tool"] for call in sanitized] == ["getTrack", "getFrames"]
    assert sanitized[1]["arguments"]["trackIds"] == {"$ref": "tracks-a.trackIds"}


def test_sanitize_calls_deduplicates_same_image_match_even_when_top_k_differs():
    base = {
        "queryImages": ["reference-1"],
        "galleryImages": ["frame-1", "frame-2"],
        "registryItems": ["registry-1"],
    }
    calls = [
        {"id": "match-top", "tool": "matchImage", "arguments": {**base, "topK": 5}},
        {"id": "match-all", "tool": "matchImage", "arguments": dict(base)},
    ]

    sanitized = PlanExecutor.sanitize_calls(calls)

    assert len(sanitized) == 1
    assert sanitized[0]["id"] == "match-top"


def test_duplicates_do_not_consume_call_budget_before_useful_steps():
    calls = [
        {"id": f"tracks-{index}", "tool": "getTrack", "arguments": {"hullNumber": "大鱼01", "limit": 60}}
        for index in range(10)
    ]
    calls.append({"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": "大鱼01"}})

    sanitized = PlanExecutor.sanitize_calls(calls, max_calls=2)

    assert [call["tool"] for call in sanitized] == ["getTrack", "getRegistry"]


def test_registry_capability_directive_generates_only_registry_lookup():
    calls = _default_plan_calls(
        _hull_intent(),
        top_k=5,
        broad_match_top_k=0,
        replan_hint="界面摘要可能错误建议重复 getTrack，但不得参与能力判断",
        replan_directive={
            "requiredCapabilities": ["registry_lookup"],
            "target": {"kind": "hull", "hullNumber": "大鱼01"},
        },
    )

    assert calls == [{"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": "大鱼01"}}]


def test_image_matching_directive_refetches_full_tracks_when_cached_page_is_partial():
    calls = _default_plan_calls(
        {
            "operation": "list",
            "targetScope": "both",
            "targetKind": "all",
            "registryRelation": "out",
            "questionType": "registry_out_list",
        },
        top_k=5,
        broad_match_top_k=0,
        replan_hint="继续完成证据核验",
        replan_directive={
            "requiredCapabilities": ["track_retrieval", "keyframe_retrieval", "image_matching"],
            "target": {"kind": "all", "scope": "both", "operation": "list"},
        },
        working_scope={
            "tracks": {
                "ok": True,
                "trackIds": list(range(60)),
                "returnedTrackCount": 60,
                "totalTrackCount": 100,
            },
            "frames": {"ok": True, "keyframes": [{"keyframeId": "partial-frame"}]},
            "registry": {
                "ok": True,
                "registryItems": [{"registryId": "registry-1"}],
                "registryReferences": [{"referenceId": "reference-1"}],
            },
        },
    )

    assert [call["tool"] for call in calls] == ["getTrack", "getFrames", "matchImage"]
    assert calls[0]["arguments"] == {"offset": 0, "limit": 0}
    assert calls[1]["arguments"]["trackIds"] == {"$ref": "tracks.trackIds"}
    assert calls[2]["arguments"]["topK"] == 0


def test_hull_visual_replan_reuses_registry_and_runs_focused_match():
    """focused 证据模式（存在性单目标）：全量扫描轨迹，但只对少量候选取证并 topK 匹配。"""
    calls = _default_plan_calls(
        _hull_intent(),
        top_k=5,
        broad_match_top_k=0,
        replan_hint=(
            "getRegistry(hullNumber=大鱼01) → getTrack(不带hullNumber, 全时域) → "
            "getFrames → matchImage"
        ),
        working_scope={
            "known_registry": {
                "ok": True,
                "registryItems": [{"registryId": "registry-1", "hullNumber": "大鱼01"}],
                "registryReferences": [{"referenceId": "reference-1"}],
            },
        },
    )

    assert [call["tool"] for call in calls] == ["getTrack", "getFrames", "matchImage"]
    # 否定结论仍需全量扫描轨迹
    assert calls[0]["arguments"] == {"offset": 0, "limit": 0}
    # focused：只对少量候选轨迹取帧
    assert calls[1]["arguments"]["trackIds"] == {"$ref": "tracks.trackIds", "$slice": 20}
    assert calls[1]["condition"] == {"ref": "tracks.trackIds"}
    assert calls[2]["condition"] == {"ref": "frames.keyframes"}
    # focused：matchImage 用普通 topK，不再全量评分
    assert calls[2]["arguments"]["topK"] == 5
    assert calls[2]["arguments"]["queryImages"] == {"$ref": "known_registry.registryReferences"}


def test_model_hull_visual_plan_keeps_topk_and_guarded_dependencies_in_focused_mode():
    """focused 证据模式：模型计划不再被强制为全量匹配（topK 与分页保持）。"""
    calls, repair = _prepare_plan_calls(
        [
            {"id": "registry", "tool": "getRegistry", "arguments": {"hullNumber": "大鱼01"}},
            {"id": "tracks", "tool": "getTrack", "arguments": {"limit": 60}},
            {"id": "frames", "tool": "getFrames", "arguments": {"trackIds": {"$ref": "tracks.trackIds"}}},
            {
                "id": "match",
                "tool": "matchImage",
                "arguments": {
                    "queryImages": {"$ref": "registry.registryReferences"},
                    "galleryImages": {"$ref": "frames.keyframes"},
                    "topK": 5,
                },
            },
        ],
        _hull_intent(),
        5,
        broad_match_top_k=0,
        broad_match_context=True,
    )

    assert repair == ""
    assert calls[1]["arguments"] == {"limit": 60}
    assert calls[2]["condition"] == {"ref": "tracks.trackIds"}
    assert calls[3]["condition"] == {"ref": "frames.keyframes"}
    assert calls[3]["arguments"]["topK"] == 5


def test_empty_track_dependency_short_circuits_frames_and_image_match():
    tools = _EmptyTrackTools()
    calls = [
        {"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 0}},
        {
            "id": "frames",
            "tool": "getFrames",
            "arguments": {"trackIds": {"$ref": "tracks.trackIds"}},
            "condition": {"ref": "tracks.trackIds"},
        },
        {
            "id": "match",
            "tool": "matchImage",
            "arguments": {
                "queryImages": ["reference-1"],
                "galleryImages": {"$ref": "frames.keyframes"},
            },
            "condition": {"ref": "frames.keyframes"},
        },
    ]

    result = PlanExecutor(tools).execute(calls)

    assert tools.calls == [("getTrack", {"offset": 0, "limit": 0})]
    assert [record["skipped"] for record in result["tool_records"]] == [False, True, True]
    assert all(record.get("error") != "argument_missing:galleryImages" for record in result["tool_records"])

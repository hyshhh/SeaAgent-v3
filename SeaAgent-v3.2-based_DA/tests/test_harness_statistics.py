from __future__ import annotations

import pytest

from harness.runtime import SeaVideoHarness, _Trace
from harness.shared_state import SharedTaskState
from harness.statistics import find_tool_result, plan_statistics, statistics_answer

ANCHOR = "2026-10-29T10:00:00+08:00"


@pytest.mark.parametrize(
    ("question", "kind", "tool"),
    [
        ("视频中一共出现多少艘船？", "total", "dedup_tracks"),
        ("视频中有多少艘在库船？", "registered", "match_entities"),
        ("视频中有多少艘不在库船？", "unregistered", "match_entities"),
    ],
)
def test_plain_statistics_route_without_requesting_video_path_or_time(question, kind, tool):
    request = plan_statistics(question, ANCHOR)

    assert request["kind"] == kind
    assert request["tool"] == tool
    assert request["time_range"] is None
    assert tool in request["plan"][0]["content"]


def test_relative_date_is_bound_to_fixed_anchor_as_iso_day_range():
    request = plan_statistics("昨天视频中出现多少艘船？", ANCHOR)

    assert request["time_range"] == [
        "2026-10-28T00:00:00.000000+08:00",
        "2026-10-28T23:59:59.999999+08:00",
    ]


def test_single_digit_iso_date_is_normalized_to_a_valid_day_range():
    request = plan_statistics("2026-1-2 一共有多少艘船？", ANCHOR)

    assert request["time_range"] == [
        "2026-01-02T00:00:00.000000+08:00",
        "2026-01-02T23:59:59.999999+08:00",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "为什么在库统计不完整？",
        "统计黄色船只数量",
        "统计船总数和在库数量",
        "统计最近一周的船只数量",
    ],
)
def test_complex_or_filtered_statistics_remain_with_planner(question):
    assert plan_statistics(question, ANCHOR) is None


def test_incomplete_unregistered_result_never_reports_zero_as_the_total():
    answer = statistics_answer(
        "unregistered",
        {
            "ok": True,
            "trackCount": 214,
            "inRegistryCount": 24,
            "unregisteredCount": 0,
            "unclassifiedTrackCount": 190,
            "coverageComplete": False,
            "coverageReasons": ["registry_entries_without_references"],
            "appliedTimeRange": None,
        },
    )

    assert "无法确定不在库船舶的准确数量" in answer
    assert "190 条未分类" in answer
    assert "先验库实体没有参考图" in answer


def test_registered_result_labels_embedding_and_incomplete_coverage():
    answer = statistics_answer(
        "registered",
        {
            "ok": True,
            "trackCount": 214,
            "inRegistryCount": 24,
            "unregisteredCount": 0,
            "unclassifiedTrackCount": 190,
            "coverageComplete": False,
            "coverageReasons": ["registry_entries_without_references"],
            "appliedTimeRange": None,
        },
    )

    assert "24 条轨迹" in answer
    assert "不代表 VLM 已确认" in answer
    assert "190 条轨迹未分类" in answer
    assert "匹配数量可能偏低" in answer


def test_total_result_preserves_dedup_range_and_coverage_status():
    answer = statistics_answer(
        "total",
        {
            "ok": True,
            "trackCount": 214,
            "highThresholdShipCount": 30,
            "lowThresholdShipCount": 24,
            "coverageComplete": False,
            "appliedTimeRange": None,
        },
    )

    assert "共扫描 214 条轨迹" in answer
    assert "24–30 艘" in answer
    assert "覆盖不完整" in answer
    assert "不能视为全量精确总数" in answer


class _Store:
    def save(self, *_args, **_kwargs):
        pass


class _Tool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, arguments):
        self.calls.append(arguments)
        return self.result


@pytest.mark.parametrize(
    ("question", "tool", "result", "expected"),
    [
        (
            "视频中一共出现多少艘船？",
            "dedup_tracks",
            {
                "ok": True,
                "trackCount": 214,
                "highThresholdShipCount": 30,
                "lowThresholdShipCount": 24,
                "coverageComplete": True,
                "appliedTimeRange": None,
            },
            "24–30 艘",
        ),
        (
            "视频中有多少艘在库船？",
            "match_entities",
            {
                "ok": True,
                "trackCount": 214,
                "inRegistryCount": 24,
                "unregisteredCount": 0,
                "unclassifiedTrackCount": 190,
                "coverageComplete": False,
                "coverageReasons": ["registry_entries_without_references"],
                "appliedTimeRange": None,
            },
            "24 条轨迹",
        ),
        (
            "视频中有多少艘不在库船？",
            "match_entities",
            {
                "ok": True,
                "trackCount": 214,
                "inRegistryCount": 24,
                "unregisteredCount": 0,
                "unclassifiedTrackCount": 190,
                "coverageComplete": False,
                "coverageReasons": ["registry_entries_without_references"],
                "appliedTimeRange": None,
            },
            "无法确定不在库船舶的准确数量",
        ),
    ],
)
def test_harness_routes_and_finishes_plain_statistics(question, tool, result, expected):
    state = SharedTaskState(question, time_anchor=ANCHOR)
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {"harness": {"max_iterations": 2}}
    runtime.shared_state = state
    runtime.state_store = _Store()
    runtime._cancel = None
    selected_tool = _Tool(tool, result)
    runtime.tools = [selected_tool]
    assert state.snapshot()["plan"] == []
    trace = _Trace(None, 4000, shared_state=state)
    driver = runtime._drive_multi_steps(question, "thread", trace)
    while True:
        try:
            next(driver)
        except StopIteration as stopped:
            outcome = stopped.value
            break

    final = state.snapshot()
    assert selected_tool.calls == [{}]
    assert outcome == {"state": "completed"}
    assert expected in final["assessment"]["final_answer"]
    assert final["decision"] == "terminate"
    assert final["step_status"] == {"count-1": "completed"}


def test_statistics_result_must_match_planned_scope():
    request = plan_statistics("昨天视频中出现多少艘船？", ANCHOR)
    state = {
        "tool_results": [
            {
                "agent": "executor",
                "tool": request["tool"],
                "status": "completed",
                "result": {
                    "ok": True,
                    "appliedTimeRange": ["2024-12-18", "2024-12-19"],
                },
            }
        ]
    }

    assert find_tool_result(state, request) is None


def test_failed_statistic_tool_does_not_generate_an_answer():
    assert statistics_answer("unregistered", {"ok": False, "error": "timeout"}) is None


def test_failed_statistics_tool_result_remains_available_for_runtime_error():
    request = plan_statistics("视频中有多少艘不在库船？", ANCHOR)
    state = SharedTaskState("q")
    state.record_tool_result(
        "executor",
        request["tool"],
        "failed-call",
        {"ok": False, "error": "tool_timeout", "hint": "统计超时"},
        "error",
    )

    assert find_tool_result(state.snapshot(), request)["hint"] == "统计超时"

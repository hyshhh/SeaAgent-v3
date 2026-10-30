from __future__ import annotations

import pytest

from harness.runtime import SeaVideoHarness, _Trace
from harness.shared_state import SharedTaskState
from harness.statistics import find_tool_result, plan_statistics, statistics_answer

ANCHOR = "2026-10-29T10:00:00+08:00"


def test_plain_total_count_uses_deterministic_route():
    request = plan_statistics("视频中一共出现多少艘船？", ANCHOR)

    assert request["kind"] == "total"
    assert request["tool"] == "dedup_tracks"
    assert request["time_range"] is None
    assert "dedup_tracks" in request["plan"][0]["content"]


@pytest.mark.parametrize(
    "question",
    [
        "视频中有多少艘在库船？",
        "视频中有多少艘不在库船？",
    ],
)
def test_registry_counts_stay_with_skill_driven_agents(question):
    assert plan_statistics(question, ANCHOR) is None


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


def test_registry_counts_do_not_use_a_deterministic_fast_answer():
    assert statistics_answer("registered", {"ok": True, "trackCount": 1}) is None
    assert statistics_answer("unregistered", {"ok": True, "trackCount": 1}) is None


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


def test_harness_routes_and_finishes_plain_total_count():
    question = "视频中一共出现多少艘船？"
    result = {
        "ok": True,
        "trackCount": 214,
        "highThresholdShipCount": 30,
        "lowThresholdShipCount": 24,
        "coverageComplete": True,
        "appliedTimeRange": None,
    }
    state = SharedTaskState(question, time_anchor=ANCHOR)
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {"harness": {"max_iterations": 2}}
    runtime.shared_state = state
    runtime.state_store = _Store()
    runtime._cancel = None
    selected_tool = _Tool("dedup_tracks", result)
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
    assert "24–30 艘" in final["assessment"]["final_answer"]
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
    assert statistics_answer("total", {"ok": False, "error": "timeout"}) is None


def test_failed_statistics_tool_result_remains_available_for_runtime_error():
    request = plan_statistics("视频中一共出现多少艘船？", ANCHOR)
    state = SharedTaskState("q")
    state.record_tool_result(
        "executor",
        request["tool"],
        "failed-call",
        {"ok": False, "error": "tool_timeout", "hint": "统计超时"},
        "error",
    )

    assert find_tool_result(state.snapshot(), request)["hint"] == "统计超时"

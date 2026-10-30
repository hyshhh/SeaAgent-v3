from __future__ import annotations

import pytest

from harness.statistics import plan_statistics

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

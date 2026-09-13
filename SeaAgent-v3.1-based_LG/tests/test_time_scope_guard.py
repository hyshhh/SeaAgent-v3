from datetime import datetime, timezone

from agent.graph import _default_plan_calls, _ground_intent_time, _prepare_plan_calls
from tools.time_normalizer import has_time_expression, normalize_time_range


def _count_intent(**extra):
    return {
        "operation": "count",
        "targetScope": "track_memory",
        "targetKind": "all",
        "registryRelation": "any",
        **extra,
    }


def test_count_query_without_time_uses_all_monitoring_memory():
    question = "一共出现多少艘船？"
    fake_range = [1785069305, 1785069365]

    intent = _ground_intent_time(
        _count_intent(
            timeRange=fake_range,
            timeExpression="最近一分钟",
            queryScope=fake_range,
            timeSource="tool",
        ),
        question,
        reference_time=datetime(2026, 7, 26, 12, 36, 5, tzinfo=timezone.utc),
    )

    assert intent["hasExplicitTime"] is False
    assert intent["timeRange"] is None
    assert intent["timeExpression"] is None
    assert intent["queryScope"] is None
    assert intent["timeSource"] == "all_monitoring_time"

    calls = _default_plan_calls(intent, top_k=1)
    get_track = next(call for call in calls if call["tool"] == "getTrack")
    assert get_track["arguments"]["limit"] == 0
    assert "timeRange" not in get_track["arguments"]


def test_model_plan_cannot_inject_time_for_unscoped_question():
    fake_range = [1785069305, 1785069365]
    intent = _ground_intent_time(_count_intent(), "视频中一共出现多少艘船？")
    calls, repair = _prepare_plan_calls(
        [{"id": "tracks", "tool": "getTrack", "arguments": {"offset": 0, "limit": 0, "timeRange": fake_range}}],
        intent,
        1,
        broad_match_top_k=0,
    )

    assert repair == ""
    assert calls[0]["tool"] == "getTrack"
    assert "timeRange" not in calls[0]["arguments"]


def test_explicit_time_query_keeps_grounded_range():
    question = "2026年7月26日15:30到15:40一共出现多少艘船？"
    reference = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
    expected = normalize_time_range(question, now=reference)
    assert expected is not None

    intent = _ground_intent_time(
        _count_intent(timeRange=[1, 2], timeExpression="模型错误范围"),
        question,
        reference_time=reference,
    )

    assert intent["hasExplicitTime"] is True
    assert intent["timeRange"] == list(expected)
    assert intent["queryScope"] == list(expected)
    assert intent["timeSource"] == "question"

    calls = _default_plan_calls(intent, top_k=1)
    get_track = next(call for call in calls if call["tool"] == "getTrack")
    assert get_track["arguments"]["timeRange"] == list(expected)


def test_monitoring_context_words_are_not_time_constraints():
    for question in (
        "视频中一共出现多少艘船？",
        "当前监控记忆里有多少艘船？",
        "一共出现多少艘船？",
    ):
        assert has_time_expression(question) is False

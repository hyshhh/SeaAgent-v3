"""时间窗口校验：无意义的窗口必须报错，而不是回一个"成功的空结果"。

现场事故：模型连续 44 次用 [0, 1e-16] → [0, 1e-33] 查轨迹，每次都拿到
"ok: true, 0 条"，于是它以为"查询方式对、只是窗口没选好"，一直缩到烧光预算。
重复调用守卫拦不住 —— 它判的是「工具名 + 参数完全相等」，而参数一直在变。
"""
from datetime import datetime, timezone

from config import load_config
from harness.tools import build_tools
from tools.service import ToolService


def _validate(value):
    return ToolService._validate_time_range(value)


def test_a_placeholder_window_is_rejected_not_reported_as_empty():
    """[0, 1e-16] 这种窗口要报错。回 ok=true 等于奖励错误方向。"""
    for value in ((0.0, 1e-16), (0.0, 1e-33), (0.0, 1.0)):
        scope, problem = _validate(value)
        assert problem, f"{value} 应当被拒绝"
        assert scope is None


def test_a_reversed_window_is_rejected():
    _, problem = _validate((2000.0, 1000.0))
    assert problem and "起点晚于终点" in problem


def test_short_explicit_windows_are_valid():
    scope, problem = _validate((1700000000.0, 1700000010.0))
    assert problem == ""
    assert scope == (1700000000.0, 1700000010.0)


def test_a_plausible_window_passes_through_unchanged():
    scope, problem = _validate((1700000000.0, 1700003600.0))
    assert problem == ""
    assert scope == (1700000000.0, 1700003600.0)


def test_iso_timestamps_are_converted_to_epoch_seconds():
    scope, problem = _validate(("2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"))
    expected = (
        datetime(2026, 10, 22, 2, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 10, 22, 3, tzinfo=timezone.utc).timestamp(),
    )
    assert problem == ""
    assert scope == expected


def test_naive_iso_timestamps_use_shanghai_timezone():
    scope, problem = _validate(("2026-10-22T10:00:00", "2026-10-22T11:00:00"))
    assert problem == ""
    assert scope == _validate(("2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"))[0]


def test_mixed_iso_and_epoch_range_is_rejected():
    scope, problem = _validate(("2026-10-22T10:00:00+08:00", 1792638000))
    assert scope is None
    assert "相同格式" in problem


def test_no_window_means_everything_and_is_allowed():
    scope, problem = _validate(None)
    assert problem == "" and scope is None


def test_malformed_values_are_rejected_with_a_readable_reason():
    for value in ((float("nan"), 1.0), (float("inf"), 1.0), (1e100, 1e101), ("abc", "def"), (1.0,), "not a range"):
        scope, problem = _validate(value)
        assert problem, f"{value} 应当被拒绝"
        assert scope is None


def test_the_tool_surfaces_the_rejection_to_the_model():
    """经真实工具调用走一遍：模型拿到的必须是 ok=false 与原因，而不是空结果。"""

    class _Service:
        def __init__(self):
            self.service = ToolService.__new__(ToolService)

        def recognizeTracks(self, **kwargs):
            return ToolService.recognizeTracks(self.service, **kwargs)

        def __getattr__(self, name):
            return lambda **kwargs: {"ok": True}

    tool = {item.name: item for item in build_tools(load_config(), _Service())}["recognize_tracks"]
    result = tool.invoke({"time_range": [0.0, 1e-16], "description": "yellow vessel"})
    assert result["ok"] is False
    assert result["error"] == "invalid_time_range"
    assert "2017" in result["hint"]
    # 被拒绝时不该走到仓储（这个实例没有 repository，走到了就会 AttributeError）
    assert "tracks" not in result
    assert "totalTrackCount" not in result

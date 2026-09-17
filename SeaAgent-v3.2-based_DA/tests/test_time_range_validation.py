"""时间窗口校验：无意义的窗口必须报错，而不是回一个"成功的空结果"。

现场事故：模型连续 44 次用 [0, 1e-16] → [0, 1e-33] 查轨迹，每次都拿到
"ok: true, 0 条"，于是它以为"查询方式对、只是窗口没选好"，一直缩到烧光预算。
重复调用守卫拦不住 —— 它判的是「工具名 + 参数完全相等」，而参数一直在变。
"""
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
    scope, problem = _validate((2000.0, 1000.0))
    assert problem and "起点晚于终点" in problem


def test_a_window_too_narrow_to_contain_a_track_is_rejected():
    scope, problem = _validate((1700000000.0, 1700000010.0))  # 10 秒
    assert problem and "太窄" in problem


def test_a_plausible_window_passes_through_unchanged():
    scope, problem = _validate((1700000000.0, 1700003600.0))
    assert problem == ""
    assert scope == (1700000000.0, 1700003600.0)


def test_no_window_means_everything_and_is_allowed():
    scope, problem = _validate(None)
    assert problem == "" and scope is None


def test_malformed_values_are_rejected_with_a_readable_reason():
    for value in ((float("nan"), 1.0), (float("inf"), 1.0), ("abc", "def"), (1.0,),):
        scope, problem = _validate(value)
        assert problem, f"{value} 应当被拒绝"
        assert scope is None


def test_the_tool_surfaces_the_rejection_to_the_model():
    """经真实工具调用走一遍：模型拿到的必须是 ok=false 与原因，而不是空结果。"""

    class _Service:
        """只提供 getTrack 的真实实现，其余方法兜住（build_tools 启动时会校验方法存在）。

        这里必须绑**真实的** getTrack，否则测的是桩自己，校验根本没被走到。
        """

        def __init__(self):
            # 真实实现：校验逻辑在 ToolService.getTrack 里，直接绑它的方法
            self.service = ToolService.__new__(ToolService)

        def getTrack(self, **kwargs):
            return ToolService.getTrack(self.service, **kwargs)

        def __getattr__(self, name):
            return lambda **kwargs: {"ok": True}

    tool = {item.name: item for item in build_tools(load_config(), _Service())}["get_track"]
    result = tool.invoke({"time_range": [0.0, 1e-16]})
    assert result["ok"] is False
    assert result["error"] == "invalid_time_range"
    assert "太窄" in result["hint"] or "早于" in result["hint"]
    # 被拒绝时不该走到仓储（这个实例没有 repository，走到了就会 AttributeError）
    assert result["totalTrackCount"] == 0

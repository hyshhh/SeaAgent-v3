"""校验工具适配层对 ISO 时间和旧 epoch 时间的归一化。"""
from config import load_config
from harness.tools import build_tools


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def _get_track():
    tools = {tool.name: tool for tool in build_tools(load_config(), _Service())}
    return tools["get_track"]


def test_time_range_accepts_every_transport_shape():
    tool = _get_track()
    for value in ("[0, 10000]", "0,10000", [0, 10000], (0.0, 10000.0)):
        result = tool.invoke({"time_range": value, "hull_number": "320"})
        assert result["kwargs"]["timeRange"] == (0.0, 10000.0), value


def test_time_range_accepts_iso8601_without_losing_timezone():
    tool = _get_track()
    value = ["2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"]
    result = tool.invoke({"time_range": value})
    assert result["kwargs"]["timeRange"] == tuple(value)


def test_get_clip_accepts_the_same_iso8601_range_contract():
    tools = {item.name: item for item in build_tools(load_config(), _Service())}
    value = ["2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"]
    result = tools["get_clip"].invoke({"track_id": "t-1", "time_range": value})
    assert result["kwargs"]["timeRange"] == tuple(value)


def test_track_cursor_fields_map_to_service_arguments():
    result = _get_track().invoke({"after_start_time": 1792634400, "after_track_id": "track-17"})
    assert result["kwargs"]["afterStartTime"] == 1792634400
    assert result["kwargs"]["afterTrackId"] == "track-17"


def test_a_bad_time_range_is_rejected_with_a_reason():
    tool = _get_track()
    for value in ("[1,2,3]", "100", "[1]", "not a range"):
        try:
            tool.invoke({"time_range": value})
        except Exception as error:  # noqa: BLE001 - 只要报错即可，类型由 pydantic 决定
            assert "time_range" in str(error), f"{value} 的报错没说清是哪个参数"
        else:
            raise AssertionError(f"{value} 不该通过校验")

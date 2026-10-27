from config import load_config
from harness.tools import build_tools


class Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def test_time_range_accepts_native_json_and_iso8601_values():
    tools = {tool.name: tool for tool in build_tools(load_config(), Service())}
    tool = tools["recognize_tracks"]
    values = (
        ([1, 2], (1.0, 2.0)),
        ("[1, 2]", (1.0, 2.0)),
        (["2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"],
         ("2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00")),
    )
    for value, expected in values:
        result = tool.invoke({"time_range": value, "description": "yellow vessel"})
        assert result["kwargs"]["timeRange"] == expected


def test_statistics_tools_accept_optional_time_scope():
    tools = {tool.name: tool for tool in build_tools(load_config(), Service())}
    for name in ("dedup_tracks", "match_entities"):
        assert tools[name].invoke({})["kwargs"] == {}

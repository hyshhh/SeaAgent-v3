import math

from config import load_config
from harness.tools import build_tools
from tools.service import ToolService


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
        ('{"start":"2026-10-25","end":"2026-10-25"}',
         ("2026-10-25T00:00:00", "2026-10-25T23:59:59.999999")),
        ({"start": "2026-10-25", "end": "2026-10-25"},
         ("2026-10-25T00:00:00", "2026-10-25T23:59:59.999999")),
    )
    for value, expected in values:
        result = tool.invoke({"time_range": value, "description": "yellow vessel"})
        assert result["kwargs"]["time_range"] == expected


def test_encoded_date_range_covers_the_entire_local_day():
    tools = {tool.name: tool for tool in build_tools(load_config(), Service())}
    result = tools["dedup_tracks"].invoke({
        "time_range": '{"start":"2026-10-25","end":"2026-10-25"}'
    })
    scope, error = ToolService._validate_time_range(result["kwargs"]["time_range"])

    assert error == ""
    assert math.isclose(scope[1] - scope[0], 24 * 60 * 60 - 0.000001, abs_tol=1e-6)


def test_statistics_tools_accept_optional_time_scope():
    tools = {tool.name: tool for tool in build_tools(load_config(), Service())}
    for name in ("dedup_tracks", "match_entities"):
        assert tools[name].invoke({})["kwargs"] == {}

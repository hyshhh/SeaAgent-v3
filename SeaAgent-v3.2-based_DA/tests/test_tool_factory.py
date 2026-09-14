from config import load_config
from harness.tools import build_tools


class Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}

def test_tools_are_config_driven():
    tools = build_tools(load_config(), Service())
    assert [tool.name for tool in tools] == [
        "get_track", "get_frames", "get_clip", "get_registry", "list_registry",
        "match_hull", "match_text", "match_image", "verify_target", "show_evidence", "dedup_tracks",
    ]
    result = tools[0].invoke({"time_range": [1.0, 2.0], "hull_number": "0857"})
    assert result["kwargs"] == {"timeRange": (1.0, 2.0), "hullNumber": "0857", "offset": 0, "limit": 0}

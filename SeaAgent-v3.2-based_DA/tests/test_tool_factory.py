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
        # 唯一会改进知识库的工具：写前由人工确认中断拦下（见 config/subagents.yaml 的 interrupt_on）
        "add_registry_vessel",
    ]
    result = tools[0].invoke({"time_range": [1.0, 2.0], "hull_number": "0857"})
    assert result["kwargs"] == {"timeRange": (1.0, 2.0), "hullNumber": "0857", "limit": 50}


def test_match_text_requires_and_explains_its_gallery():
    tools = {tool.name: tool for tool in build_tools(load_config(), Service())}
    tool = tools["match_text"]
    schema = tool.args_schema.model_json_schema()

    assert "gallery_images" in schema["required"]
    assert "keyframes" in schema["properties"]["gallery_images"]["description"]
    assert "registryReferences" in schema["properties"]["gallery_images"]["description"]
    assert "不得只传 description" in tool.description

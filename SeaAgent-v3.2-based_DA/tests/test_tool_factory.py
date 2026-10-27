import json

from config import load_config
from harness.tools import build_tools


class Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def test_tools_are_config_driven():
    tools = build_tools(load_config(), Service())
    assert [tool.name for tool in tools] == [
        "recognize_tracks",
        "match_registry_tracks",
        "verify_vessel_images",
        "dedup_tracks",
        "match_entities",
        "show_evidence",
    ]
    result = tools[0].invoke({"time_range": [1.0, 2.0], "hull_number": "0857"})
    assert result["kwargs"] == {"timeRange": (1.0, 2.0), "hullNumber": "0857", "topK": 6}


def test_recognition_schema_keeps_query_inputs_compact():
    tool = {tool.name: tool for tool in build_tools(load_config(), Service())}["recognize_tracks"]
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) >= {"hull_number", "description", "image_path", "time_range"}
    assert "time_range" not in schema.get("required", [])


def test_vlm_schema_accepts_registry_reference_or_local_reference():
    tool = {tool.name: tool for tool in build_tools(load_config(), Service())}["verify_vessel_images"]
    schema = tool.args_schema.model_json_schema()
    assert {"reference_image", "reference_registry_reference_id", "candidate_image", "candidate_keyframe_id"} <= set(schema["properties"])
    assert "reference_image" not in schema.get("required", [])


def test_json_time_range_is_decoded_before_service_call():
    tool = {tool.name: tool for tool in build_tools(load_config(), Service())}["dedup_tracks"]
    result = tool.invoke({"time_range": json.dumps([1, 2])})
    assert result["kwargs"] == {"timeRange": (1.0, 2.0)}

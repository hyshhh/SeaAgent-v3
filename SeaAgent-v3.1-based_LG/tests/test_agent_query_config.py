import unittest

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent.llm_adapter import _as_bool, build_chat_model
from services import AgentLLMService
from web.models import AgentQuery


class AgentQueryConfigTest(unittest.TestCase):
    def test_query_accepts_optional_top_k(self) -> None:
        self.assertIsNone(AgentQuery(question="测试").top_k)
        self.assertEqual(AgentQuery(question="测试", top_k=8).top_k, 8)

    def test_query_rejects_out_of_range_top_k(self) -> None:
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=0)
        with self.assertRaises(ValidationError):
            AgentQuery(question="测试", top_k=21)


def test_string_false_never_enables_model_thinking():
    assert _as_bool("false") is False
    assert _as_bool("0") is False
    assert _as_bool(False) is False
    assert _as_bool("true") is True


def test_chat_model_forces_thinking_off_even_when_legacy_config_is_true():
    config = {
        "llm": {
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "http://localhost:1/v1",
            "temperature": 0,
            "timeout_seconds": 10,
            "enable_thinking": True,
        }
    }
    model = build_chat_model(AgentLLMService(config))
    payload = model._get_request_payload([HumanMessage(content="测试")])
    extra = payload.get("extra_body") or {}

    assert extra.get("enable_thinking") is False
    assert extra.get("chat_template_kwargs", {}).get("enable_thinking") is False


if __name__ == "__main__":
    unittest.main()

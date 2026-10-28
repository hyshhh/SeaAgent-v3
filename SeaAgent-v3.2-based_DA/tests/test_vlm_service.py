import pytest

from services.vlm_service import (
    AgentLLMService,
    LLMServiceError,
    _recover_verification_decision,
)


def test_malformed_vlm_facts_recover_only_explicit_top_level_decision():
    malformed = '{"decision":"match","facts":["same hull with "matching" markings"]}'

    assert _recover_verification_decision(malformed) == "match"
    assert _recover_verification_decision('{"result":{"decision":"match"}}') is None
    assert _recover_verification_decision(
        '{"decision":"match","facts":[],"decision":"mismatch"}'
    ) is None
    assert _recover_verification_decision('{"decision":"maybe","facts":[]}') is None


def test_complete_json_retries_with_format_correction(monkeypatch):
    request_texts = []
    responses = iter([
        '{"decision":"match","facts":["same hull with "matching" markings"]}',
        '{"decision":"match","facts":["same hull markings"]}',
    ])

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": next(responses)}}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, _url, *, headers, json):
            request_texts.append(json["messages"][0]["content"][0]["text"])
            return Response()

    monkeypatch.setattr("services.vlm_service.httpx.Client", Client)
    service = AgentLLMService({
        "llm": {
            "model": "test",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "",
            "timeout_seconds": 1,
        },
        "prompts": {},
    })

    result = service.complete_json("Return a verification JSON object.")

    assert result["decision"] == "match"
    assert len(request_texts) == 2
    assert "Formatting correction" not in request_texts[0]
    assert "Formatting correction" in request_texts[1]


def test_verify_recovers_decision_without_untrusted_malformed_facts(monkeypatch):
    service = AgentLLMService({"llm": {}, "prompts": {"verify_images": "Verify."}})
    monkeypatch.setattr(
        service,
        "complete_json",
        lambda *_args, **kwargs: (
            {"decision": "uncertain", "facts": [], "_formatRecovered": True}
            if kwargs.get("recover_verification_decision")
            else (_ for _ in ()).throw(AssertionError("recovery must be enabled"))
        ),
    )

    result = service.verify(["reference.jpg"], ["candidate.jpg"])

    assert result == {"decision": "uncertain", "facts": [], "formatRecovered": True}


def test_ambiguous_malformed_verification_stays_an_error(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"decision":"maybe","facts":["broken "json"]}'}}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr("services.vlm_service.httpx.Client", Client)
    service = AgentLLMService({
        "llm": {"model": "test", "base_url": "http://127.0.0.1:8000/v1", "timeout_seconds": 1},
        "prompts": {},
    })

    with pytest.raises(LLMServiceError, match="模型 JSON 无法解析"):
        service.complete_json("Verify.", retries=0, recover_verification_decision=True)

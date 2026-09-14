from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness.wrapup import NUDGE_KEY, EvidenceWrapUpMiddleware


def _middleware(max_nudges=1):
    return EvidenceWrapUpMiddleware(evidence_tool="show_evidence", instruction="call show_evidence", max_nudges=max_nudges)


def _evidence_message():
    return ToolMessage(content='{"ok": true}', name="show_evidence", tool_call_id="call-1")


def _state(*messages):
    return {"messages": list(messages)}


def test_final_answer_without_evidence_is_nudged_back_to_model():
    result = _middleware().after_model(_state(HumanMessage(content="有几艘船？"), AIMessage(content="共三艘。")))
    assert result["jump_to"] == "model"
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "call show_evidence"
    assert result["messages"][0].additional_kwargs[NUDGE_KEY] is True


def test_evidence_in_current_turn_lets_the_answer_pass():
    state = _state(
        HumanMessage(content="有几艘船？"),
        AIMessage(content="", tool_calls=[{"name": "show_evidence", "args": {}, "id": "call-1"}]),
        _evidence_message(),
        AIMessage(content="共三艘。"),
    )
    assert _middleware().after_model(state) is None


def test_evidence_from_a_previous_turn_does_not_count_as_this_turn():
    state = _state(
        HumanMessage(content="上一轮问题"),
        AIMessage(content="", tool_calls=[{"name": "show_evidence", "args": {}, "id": "call-1"}]),
        _evidence_message(),
        AIMessage(content="上一轮回答"),
        HumanMessage(content="这一轮问题"),
        AIMessage(content="这一轮回答"),
    )
    result = _middleware().after_model(state)
    assert result and result["jump_to"] == "model"


def test_pending_tool_calls_are_never_interrupted():
    state = _state(HumanMessage(content="问题"), AIMessage(content="", tool_calls=[{"name": "get_track", "args": {}, "id": "call-9"}]))
    assert _middleware().after_model(state) is None


def test_second_final_answer_is_released_after_the_nudge_budget():
    middleware = _middleware(max_nudges=1)
    nudged = middleware.after_model(_state(HumanMessage(content="问题"), AIMessage(content="回答")))
    state = _state(HumanMessage(content="问题"), AIMessage(content="回答"), nudged["messages"][0], AIMessage(content="回答（仍未调证据）"))
    assert middleware.after_model(state) is None


def test_empty_model_output_is_not_a_final_answer():
    assert _middleware().after_model(_state(HumanMessage(content="问题"), AIMessage(content="   "))) is None


def test_middleware_stands_down_when_the_evidence_tool_is_unset():
    middleware = EvidenceWrapUpMiddleware(evidence_tool="", max_nudges=2)
    assert middleware.after_model(_state(HumanMessage(content="问题"), AIMessage(content="回答"))) is None

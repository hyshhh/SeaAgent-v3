from langchain_core.messages import AIMessage, ToolMessage

import threading

from config import load_config
from harness.runtime import SeaVideoHarness, _Trace


def test_trace_correlates_tool_result_and_extracts_configured_evidence():
    events = []
    trace = _Trace(events.append, event_limit=80, evidence_tool='show_evidence')
    trace.consume({'skills_metadata': [{'name': 'query', 'description': '查询'}], 'messages': [AIMessage(content='', tool_calls=[{'name': 'show_evidence', 'args': {'keyframe_ids': ['kf-1']}, 'id': 'call-1'}])]} )
    trace.consume({'tools': {'messages': [ToolMessage(content='{"ok": true, "shownKeyframeIds": ["kf-1"]}', name='show_evidence', tool_call_id='call-1')]}})
    trace.consume({'model': {'messages': [AIMessage(content='已完成')]}})
    assert trace.evidence == {'ok': True, 'shownKeyframeIds': ['kf-1']}
    assert trace.records[0]['status'] == 'completed'
    assert trace.records[0]['id'] == 'call-1'
    assert [event['type'] for event in events] == ['skill', 'tool_start', 'model', 'tool_result', 'model']
    assert trace.result('thread-1', load_config())['answer'] == '已完成'


def test_trace_accepts_root_level_updates_and_bounds_event_payloads():
    events = []
    trace = _Trace(events.append, event_limit=10, evidence_tool='show_evidence')
    trace.consume({'messages': [AIMessage(content='x' * 50)]})
    assert trace.answer == 'x' * 50
    assert all(len(str(value)) <= 11 for event in events for value in event.values())


class _FakeAgent:
    def __init__(self, updates=None, error=None):
        self.updates = updates or []
        self.error = error

    def stream(self, *_args, **_kwargs):
        yield from self.updates
        if self.error:
            raise self.error


def _runtime_with_agent(agent):
    runtime = SeaVideoHarness.__new__(SeaVideoHarness)
    runtime.config = {
        'harness': {
            'stream_mode': 'updates',
            'event_payload_max_chars': 4000,
            'evidence_tool': 'show_evidence',
            'execution_mode': 'single-agent-harness',
            'output': {'answer_field': 'answer', 'state_field': 'state', 'evidence_field': 'evidence'},
        },
        'tools': [{'name': 'show_evidence', 'label': '证据汇总'}],
    }
    runtime.agent = agent
    runtime.event_handler = None
    runtime._connection = None
    return runtime


def test_stream_yields_only_public_events_and_complete_result():
    runtime = _runtime_with_agent(_FakeAgent([
        {'model': {'messages': [AIMessage(content='回答文本', id='answer-1')]}},
    ]))
    events = list(runtime.stream('问题', thread_id='thread-stream'))
    assert [event['type'] for event in events] == ['status', 'model', 'complete']
    assert events[-1]['result']['answer'] == '回答文本'
    assert events[-1]['result']['state'] == 'completed'
    assert all('messages' not in event and 'raw' not in event for event in events)


def test_stream_converts_runtime_error_to_public_error_event():
    runtime = _runtime_with_agent(_FakeAgent(error=RuntimeError('provider unavailable')))
    events = list(runtime.stream('问题', thread_id='thread-error'))
    assert [event['type'] for event in events] == ['status', 'error']
    assert events[-1]['result']['state'] == 'error'
    assert 'provider unavailable' in events[-1]['message']
    assert events[-1]['result']['error'] == 'provider unavailable'


def test_run_stops_between_frames_when_cancelled():
    """中断信号置位后本轮以 cancelled 收尾，且不再消费后续帧。"""
    cancel = threading.Event()
    cancel.set()
    runtime = _runtime_with_agent(_FakeAgent([
        {'model': {'messages': [AIMessage(content='不该被消费的回答', id='answer-cancel')]}},
    ]))
    result = runtime.run('问题', thread_id='thread-cancel', cancel=cancel)
    assert result['state'] == 'cancelled'
    assert result['answer'] == ''
    assert result['tool_records'] == []


def test_run_without_cancel_signal_still_completes():
    runtime = _runtime_with_agent(_FakeAgent([
        {'model': {'messages': [AIMessage(content='回答文本', id='answer-ok')]}},
    ]))
    result = runtime.run('问题', thread_id='thread-ok', cancel=threading.Event())
    assert result['state'] == 'completed'
    assert result['answer'] == '回答文本'


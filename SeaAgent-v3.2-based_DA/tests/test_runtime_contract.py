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
    runtime.subagents = []  # 单智能体模式：没有从智能体
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


def test_resumed_thread_re_emits_skill_events_from_the_checkpoint():
    """续接会话时框架不再发 skills_metadata，运行时要从检查点把技能回执补回事件流。"""
    events = []
    trace = _Trace(events.append, event_limit=4000, evidence_tool='show_evidence')
    runtime = _runtime_with_agent(_FakeAgent([]))
    runtime.agent = type('_AgentWithState', (), {
        'stream': lambda self, *_a, **_k: iter(()),
        'get_state': lambda self, _config: type('_Snapshot', (), {'values': {'skills_metadata': [{'name': 'query', 'description': '查询'}, {'name': 'finalize', 'description': '收尾'}]}})(),
    })()

    runtime._seed_skills_from_checkpoint(trace, 'thread-resume')

    assert [event['skill'] for event in events] == ['query', 'finalize']
    assert all(event['type'] == 'skill' for event in events)


def test_skill_seeding_tolerates_an_agent_without_state_snapshot():
    trace = _Trace(None, event_limit=4000, evidence_tool='show_evidence')
    _runtime_with_agent(_FakeAgent([]))._seed_skills_from_checkpoint(trace, 'thread-fresh')


def test_builtin_file_tools_get_chinese_labels():
    """时间线要能一眼看出这一轮有没有去读技能正文。"""
    from harness.runtime import _tool_labels

    labels = _tool_labels(load_config())
    assert labels['read_file'] == '读取技能正文'
    assert labels['get_track'] == '轨迹记忆'


def _blocked_tool_frame(index):
    """模拟工具预算用完后框架驳回的一次调用：error 状态的 ToolMessage。"""
    return {'tools': {'messages': [ToolMessage(content='Tool call limit exceeded. Do not make additional tool calls.', name='get_track', tool_call_id=f'call-{index}', status='error')]}}


def test_run_stops_after_consecutive_tool_failures():
    """模型被驳回后仍继续硬调时，运行时必须收尾，而不是无限刷同一条错误。"""
    frames = [_blocked_tool_frame(index) for index in range(20)]
    runtime = _runtime_with_agent(_FakeAgent(frames))
    result = runtime.run('问题', thread_id='thread-stall')
    assert result['state'] == 'stalled'
    assert '收尾' in result['answer']
    # 20 帧里只消费到阈值就停：不会把后面十几条同样的错误也跑完
    assert len(result['tool_records']) == 4


def test_a_single_failure_does_not_stall_the_run():
    """偶尔失败一次不算失控：成功一次即清零。"""
    frames = [
        _blocked_tool_frame(1),
        {'tools': {'messages': [ToolMessage(content='{"ok": true}', name='get_track', tool_call_id='call-2')]}},
        _blocked_tool_frame(3),
        {'model': {'messages': [AIMessage(content='最终回答', id='answer-final')]}},
    ]
    result = _runtime_with_agent(_FakeAgent(frames)).run('问题', thread_id='thread-ok')
    assert result['state'] == 'completed'
    assert result['answer'] == '最终回答'


def test_failed_tool_records_keep_their_error_status():
    runtime = _runtime_with_agent(_FakeAgent([_blocked_tool_frame(1)]))
    result = runtime.run('问题', thread_id='thread-status')
    assert result['tool_records'][0]['status'] == 'error'


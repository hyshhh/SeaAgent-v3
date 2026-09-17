"""子智能体步骤外发：委派内部要看得见，且不能和主智能体的调用混淆。"""
import threading

from harness.subagent_trace import SubagentTraceCallback

TASK = "task"


def _collect():
    events = []
    return events, SubagentTraceCallback(events.append)


def _start(handler, name, run_id, inputs=None):
    handler.on_tool_start({"name": name}, "", run_id=run_id, inputs=inputs or {})


def test_steps_inside_a_delegation_are_tagged_with_the_subagent():
    events, handler = _collect()
    _start(handler, TASK, "d1", {"subagent_type": "planner", "description": "解析意图"})
    _start(handler, "read_file", "t1", {"file_path": "/skills/planning/intent/SKILL.md"})
    handler.on_tool_end('{"ok": true}', run_id="t1")
    handler.on_chat_model_start({}, [[]], run_id="l1")
    handler.on_tool_end("done", run_id="d1")

    kinds = [(event["type"], event.get("agent", "")) for event in events]
    assert ("status", "planner") in kinds, "委派开始要有提示"
    assert ("tool_start", "planner") in kinds, "子智能体的工具调用要带归属"
    assert ("tool_result", "planner") in kinds
    assert ("model", "planner") in kinds, "子智能体思考也要可见"
    assert ("status", "") in kinds, "委派结束要有提示"

    tool = next(event for event in events if event["type"] == "tool_start")
    assert tool["tool"] == "read_file"
    assert "file_path" in tool["arguments"], "参数要带出来，才能看出它读了哪个技能"


def test_main_agent_calls_are_not_duplicated():
    """主智能体自己的工具日志已经由事件流记了，回调不该再发一遍。"""
    events, handler = _collect()
    _start(handler, "get_track", "m1", {"limit": 5})
    handler.on_tool_end("{}", run_id="m1")
    handler.on_chat_model_start({}, [[]], run_id="l0")
    assert events == []


def test_the_stack_unwinds_when_the_delegation_ends():
    events, handler = _collect()
    _start(handler, TASK, "d1", {"subagent_type": "executor"})
    assert handler.active == "executor"
    _start(handler, "get_frames", "t1", {"track_ids": ["a"]})
    handler.on_tool_end("{}", run_id="t1")
    assert handler.active == "executor", "子智能体内部工具结束时不该出栈"
    handler.on_tool_end("result", run_id="d1")
    assert handler.active == "", "委派结束后栈要清空"


def test_a_failed_subagent_tool_is_reported_as_an_error_result():
    events, handler = _collect()
    _start(handler, TASK, "d1", {"subagent_type": "executor"})
    _start(handler, "get_registry", "t9", {"hull_number": "003"})
    handler.on_tool_error(RuntimeError("registry offline"), run_id="t9")
    failure = next(event for event in events if event["type"] == "tool_result" and event["status"] == "error")
    assert failure["agent"] == "executor"
    assert "registry offline" in str(failure["result"])


def test_concurrent_delegations_do_not_mix_up_agents():
    """委派是同步跑在一个线程里的，所以归属按线程隔离。"""
    events, handler = _collect()
    errors = []

    def worker(name):
        try:
            _start(handler, TASK, f"d-{name}", {"subagent_type": name})
            for index in range(20):
                _start(handler, "get_track", f"{name}-{index}", {"offset": index})
                handler.on_tool_end("{}", run_id=f"{name}-{index}")
            handler.on_tool_end("done", run_id=f"d-{name}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("planner", "executor", "reflector")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    for event in events:
        if event["type"] != "tool_start":
            continue
        owner = event["agent"]
        assert event["callId"].startswith(owner), f"{event['callId']} 被记到了 {owner} 名下"


def test_a_callback_that_raises_does_not_break_the_turn():
    """回调炸了不能把整轮问答带崩。"""
    def boom(_event):
        raise RuntimeError("emit failed")

    handler = SubagentTraceCallback(boom)
    _start(handler, TASK, "d1", {"subagent_type": "planner"})
    _start(handler, "read_file", "t1", {})
    handler.on_tool_end("{}", run_id="t1")
    handler.on_tool_end("done", run_id="d1")  # 不抛异常即通过

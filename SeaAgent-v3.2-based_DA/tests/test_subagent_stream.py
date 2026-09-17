"""子智能体调用流：输出要看得见，等待要说得清。"""
import threading
import time

from harness.subagent_trace import SubagentTraceCallback

TASK = "task"


def _enter(handler, name="planner", run_id="d1"):
    handler.on_tool_start({"name": TASK}, "", run_id=run_id, inputs={"subagent_type": name})


def test_model_output_is_forwarded_in_batches_not_per_token():
    """流式输出要攒成块发——逐 token 发会把轨迹页刷成「一个字一行」。"""
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    # 逐字吐 60 个字符：应当明显少于 60 条事件
    for index in range(60):
        handler.on_llm_new_token("字", run_id="l1")
    handler.on_llm_end(_response(usage={"input_tokens": 120, "output_tokens": 60}), run_id="l1")

    outputs = [event for event in events if event["type"] == "model" and event["title"] == "子智能体输出"]
    assert 0 < len(outputs) < 10, f"攒块没生效：{len(outputs)} 条"
    assert all(event.get("append") and event.get("streamKey") == "l1" for event in outputs), "块要带合并标记"
    assert "".join(event["message"] for event in outputs) == "字" * 60, "攒块不能丢字"

    done = next(event for event in events if event["title"] == "模型响应完成")
    assert "输入 120" in done["message"] and "输出 60" in done["message"]


def test_each_round_leaves_a_readable_thought():
    """每轮结束要留一条完整思考：否则「planner 在完善哪一步」对外还是黑盒。"""
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    handler.on_llm_new_token("我先读 planning/intent 确认操作类型", run_id="l1")
    handler.on_llm_end(_response(), run_id="l1")

    # 正文可能走流式（子智能体输出）或补发（子智能体思考），两者有其一即可
    thoughts = [
        event for event in events
        if event.get("type") == "model"
        and event.get("title") in ("子智能体输出", "子智能体思考")
        and "planning/intent" in str(event.get("message", ""))
    ]
    assert thoughts, "轮次结束应当留下可读的思考正文"
    assert all(event["agent"] == "planner" for event in thoughts)


def test_a_final_answer_is_not_duplicated_as_a_thought():
    """没有正文的轮次（例如纯工具调用）不该凭空多出一条思考。"""
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    handler.on_llm_end(_response(), run_id="l1")
    # on_chat_model_start 会先发一条「正在等待模型响应」，那是等待提示不是思考正文；
    # 这里只要求「没有第二轮思考正文」，即正文为空时不再补发
    bodies = [
        event for event in events
        if event["title"] == "子智能体思考" and event.get("message") not in ("正在等待模型响应",)
    ]
    assert not bodies, f"空正文的轮次不该补发思考：{bodies}"


def _beat_once(handler):
    """直接跑一次心跳，不依赖真实时钟（tick 下限 5 秒，真等太慢）。"""
    now = time.monotonic()
    with handler._lock:
        snapshot = [(dict(item), now - item["since"]) for item in handler._waiting.values()]
    for item, elapsed in snapshot:
        handler.event({
            "type": "status",
            "title": f"等待{item['what']}",
            "message": f"{item['agent']} 正在等{item['what']}，已等 {int(elapsed)} 秒",
            "agent": item["agent"],
        })


def test_waiting_heartbeat_names_what_it_is_waiting_for():
    """模型不出字时要说「在等模型响应」，而不是只数秒数。"""
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    _beat_once(handler)
    handler.close()

    waiting = [event for event in events if event["title"].startswith("等待")]
    assert waiting, "挂着等待项时心跳该报出来"
    assert "模型响应" in waiting[0]["title"]
    assert waiting[0]["agent"] == "planner"


def test_a_tool_that_is_running_is_also_announced():
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler, "executor")
    handler.on_tool_start({"name": "get_track"}, "", run_id="t1", inputs={"limit": 0})
    _beat_once(handler)
    handler.close()
    assert any(event["title"] == "等待工具 get_track" for event in events)


def test_streaming_tokens_stop_the_waiting_notice():
    """一旦开始出字就不再是「等响应」——它在动，不该继续报等待。"""
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    handler.on_llm_new_token("好", run_id="l1")
    time.sleep(0.2)
    with handler._lock:
        assert not handler._waiting, "出字之后不该还挂着等待项"
    handler.close()


def test_the_heartbeat_stops_when_the_delegation_ends():
    events = []
    handler = SubagentTraceCallback(events.append)
    _enter(handler)
    handler.on_chat_model_start({}, [[]], run_id="l1")
    handler.on_tool_end("done", run_id="d1")
    assert handler._beater is None or not handler._beater.is_alive(), "委派结束要停掉心跳线程"


def test_callback_state_is_cleaned_up_after_a_delegation():
    handler = SubagentTraceCallback(lambda _e: None)
    _enter(handler)
    handler.on_tool_start({"name": "read_file"}, "", run_id="t1", inputs={})
    handler.on_llm_end(_response(), run_id="t1")
    handler.on_tool_end("{}", run_id="t1")
    handler.on_tool_end("result", run_id="d1")
    with handler._lock:
        assert handler._calls == {}
        assert handler._waiting == {}
        assert handler._buffers == {}


def _response(usage=None):
    class _Message:
        usage_metadata = usage or {}

    class _Gen:
        message = _Message()

    class _Response:
        generations = [[_Gen()]]

    return _Response()

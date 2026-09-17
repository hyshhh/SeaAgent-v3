"""剥思考：只作用于最终答案。

现场：答案以
    现在写最终答案。
    </think>
    根据数据库查询结果，……
开头，`</think>` 之前是模型泄漏的推理。系统提示词要求不暴露隐藏推理，运行时却没剥。
"""
from harness.thinking_text import strip_thinking


def test_an_orphan_closing_tag_drops_everything_before_it():
    """provider 只回闭合标签时，它之前全是推理。"""
    answer = "现在写最终答案。\n</think>\n\n根据数据库查询结果，确实有。"
    assert strip_thinking(answer) == "根据数据库查询结果，确实有。"


def test_a_full_think_block_is_removed():
    """成对标签（有 open 有 close）：整块去掉，保留标签之后的正文。"""
    tagged = "<" + "think>先想一下，再想一下" + "</" + "think>\n\n结论在此。"
    assert strip_thinking(tagged) == "结论在此。"
    # 标签前的正文不属于思考，保留
    assert strip_thinking("前言。\n<" + "think>推理</" + "think>\n正文。") == "前言。\n\n正文。"


def test_an_unclosed_think_tag_truncates_the_reasoning():
    assert strip_thinking("还没想完的推理…") == "还没想完的推理…"


def test_a_plain_answer_is_left_alone():
    assert strip_thinking("根据查询，舷号 003 在库。") == "根据查询，舷号 003 在库。"


def test_whitespace_is_trimmed():
    assert strip_thinking("\n\n  答案  \n") == "答案"
    assert strip_thinking("") == ""
    assert strip_thinking(None) == ""


def test_a_tool_result_containing_thinking_is_not_the_target():
    """这个函数只用于最终答案；子智能体的思考流走事件流，不经过它。
    这里只确认它对普通文本没有副作用，免得被误用在别处。"""
    trace_line = "我先读 planning/intent 确认操作类型"
    assert strip_thinking(trace_line) == trace_line

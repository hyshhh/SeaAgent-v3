from agent.graph import _content_parts, _stream_delta_piece


def test_stream_delta_piece_accepts_normal_increment():
    assert _stream_delta_piece("", "第一段") == "第一段"
    assert _stream_delta_piece("第一段", "第二段") == "第二段"


def test_stream_delta_piece_trims_cumulative_content():
    previous = "计划：先 getTrack，再 getFrames"
    incoming = "计划：先 getTrack，再 getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, incoming) == "，然后 dedupTracks"


def test_stream_delta_piece_trims_overlap_content():
    previous = "先 getTrack，再 getFrames"
    incoming = "getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, incoming) == "，然后 dedupTracks"


def test_stream_delta_piece_drops_exact_repeat():
    previous = "然后 getFrames，然后 dedupTracks"
    assert _stream_delta_piece(previous, "getFrames，然后 dedupTracks") == ""
    assert _stream_delta_piece(previous, previous) == ""


def test_stream_delta_piece_handles_previous_inside_new_cumulative_window():
    previous = "再 getFrames，然后 dedupTracks"
    incoming = "计划：先 getTrack，再 getFrames，然后 dedupTracks，最后统计数量"
    assert _stream_delta_piece(previous, incoming) == "，最后统计数量"


def test_content_parts_hides_orphan_closing_think_prefix():
    body, thinking = _content_parts('Draft: 内部规划与工具选择 </think>')
    assert body == ''
    assert '内部规划' in thinking


def test_content_parts_keeps_final_text_after_orphan_closing_think():
    body, thinking = _content_parts('内部推理草稿</think>最终可见结论')
    assert body == '最终可见结论'
    assert thinking == '内部推理草稿'


def test_content_parts_hides_complete_and_unclosed_thinking_blocks():
    body, thinking = _content_parts('<think>第一段推理</think>结论<thinking>未闭合推理')
    assert body == '结论'
    assert '第一段推理' in thinking
    assert '未闭合推理' in thinking

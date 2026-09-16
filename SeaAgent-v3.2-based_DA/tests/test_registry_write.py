"""写入工具的校验测试：拦住该拦的，放行该放行的。

工具本体是 add_registry_vessel，它外面还有一道人工确认中断（另测）。
这里只测工具自己的纪律：没有用户要求不写、只新增不覆盖、参考图不能是视频、路径要真实存在。
"""
import pathlib

from harness.registry_write import MAX_REFERENCES, RegistryWriteTool


class _FakeShipService:
    """记录 create_registry 的调用参数，并按需抛出与真实服务一致的异常。"""

    def __init__(self, exists=False, raises=None):
        self.calls = []
        self.exists = exists
        self.raises = raises

    def create_registry(self, hull_number, description="", aliases=None, images=None):
        self.calls.append({"hull": hull_number, "description": description, "aliases": aliases, "images": images})
        if self.raises is not None:
            raise self.raises
        if self.exists:
            raise FileExistsError(f"舷号已存在：{hull_number}")
        references = [{"referenceId": f"reference-{index}", "isEmbedded": True} for index, _ in enumerate(images or [])]
        return {"registryId": "registry-abc", "hullNumber": hull_number.upper(), "description": description, "aliases": aliases or [], "references": references}


def _png(tmp_path, name="kf-1.png"):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    return str(path)


# ---------------------------------------------------------------- 意图闸


def test_a_write_without_a_user_intent_is_refused():
    """模型自己觉得"该入库"不算理由：没有 user_intent 一律不写。"""
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(hull_number="003")
    assert result["ok"] is False
    assert result["error"] == "user_intent_required"
    assert service.calls == []


def test_an_empty_hull_number_is_refused():
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(hull_number="  ", user_intent="把这条船入库")
    assert result["ok"] is False
    assert result["error"] == "hull_number_required"
    assert service.calls == []


# ---------------------------------------------------------------- 参考图闸


def test_a_video_file_is_refused_as_a_reference_image(tmp_path):
    """视频关键帧可以，视频片段不行——参考图是「已知身份」，不能拿待证事实充数。"""
    clip = tmp_path / "segment-1.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(
        hull_number="003", user_intent="把 003 入库", image_paths=[str(clip)]
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_reference_image"
    assert "视频" in result["hint"]
    assert service.calls == []


def test_a_missing_image_path_is_refused(tmp_path):
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(
        hull_number="003", user_intent="把 003 入库", image_paths=[str(tmp_path / "nope.png")]
    )
    assert result["ok"] is False
    assert "不存在" in result["hint"]
    assert service.calls == []


def test_too_many_reference_images_are_refused(tmp_path):
    paths = [_png(tmp_path, f"kf-{index}.png") for index in range(MAX_REFERENCES + 1)]
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(
        hull_number="003", user_intent="把 003 入库", image_paths=paths
    )
    assert result["ok"] is False
    assert str(MAX_REFERENCES) in result["hint"]
    assert service.calls == []


# ---------------------------------------------------------------- 只新增不覆盖


def test_an_existing_hull_number_is_reported_not_overwritten():
    service = _FakeShipService(exists=True)
    result = RegistryWriteTool(service).addRegistryVessel(hull_number="003", user_intent="把 003 入库")
    assert result["ok"] is False
    assert result["error"] == "hull_already_exists"
    assert "只新增、不覆盖" in result["hint"]


# ---------------------------------------------------------------- 成功路径


def test_a_valid_write_returns_the_ids_and_index_state(tmp_path):
    service = _FakeShipService()
    result = RegistryWriteTool(service).addRegistryVessel(
        hull_number="003",
        description="黄色无人艇",
        aliases=["小黄"],
        user_intent="把 003 加进先验库",
        image_paths=[_png(tmp_path, "a.png"), _png(tmp_path, "b.png")],
    )
    assert result["ok"] is True
    assert result["hullNumber"] == "003"
    assert result["referenceCount"] == 2
    assert result["referenceIds"] == ["reference-0", "reference-1"]
    assert result["indexRebuilt"] is True
    assert result["userIntent"] == "把 003 加进先验库"
    assert service.calls[0]["aliases"] == ["小黄"]
    assert [name for name, _raw in service.calls[0]["images"]] == ["a.png", "b.png"]


def test_a_write_failure_is_reported_as_a_tool_result_not_an_exception():
    """写失败也要变成可读结果回给模型，别把整轮问答炸掉。"""
    service = _FakeShipService(raises=RuntimeError("先验库参考图与向量数量不一致"))
    result = RegistryWriteTool(service).addRegistryVessel(hull_number="003", user_intent="入库")
    assert result["ok"] is False
    assert result["error"] == "write_failed"
    assert "向量数量不一致" in result["hint"]

"""工具参数的传输形态：provider 会把数组序列化成字符串，不能因此废掉一次调用。

现场事故：模型传 time_range='[0, 10000]'（字符串），schema 期望 tuple[float, float]，
报 "Input should be a valid tuple"；模型不明白为什么失败，又重复调了几次，
最后被重复调用守卫拦下、整轮空转收尾。
"""
from config import load_config
from harness.tools import build_tools


class _Service:
    def __getattr__(self, name):
        return lambda **kwargs: {"method": name, "kwargs": kwargs}


def _get_track():
    tools = {tool.name: tool for tool in build_tools(load_config(), _Service())}
    return tools["get_track"]


def test_time_range_accepts_every_transport_shape():
    tool = _get_track()
    for value in ("[0, 10000]", "0,10000", [0, 10000], (0.0, 10000.0)):
        result = tool.invoke({"time_range": value, "hull_number": "320"})
        assert result["kwargs"]["timeRange"] == (0.0, 10000.0), value


def test_a_bad_time_range_is_rejected_with_a_reason():
    tool = _get_track()
    for value in ("[1,2,3]", "100", "a,b", "[1]"):
        try:
            tool.invoke({"time_range": value})
        except Exception as error:  # noqa: BLE001 - 只要报错即可，类型由 pydantic 决定
            assert "time_range" in str(error), f"{value} 的报错没说清是哪个参数"
        else:
            raise AssertionError(f"{value} 不该通过校验")

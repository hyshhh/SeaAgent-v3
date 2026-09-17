"""Configuration-driven tool adapter for Sea-Video-Harness.

把 config/tools.yaml 的工具声明翻译成 LangChain StructuredTool：yaml 里写
``name / method / description / arguments``，这里负责生成参数 schema、做参数名映射，
并把调用转交给 ToolService 的同名方法。新增工具只需改 yaml 与 ToolService，
中间层不需要任何胶水代码。
"""
from __future__ import annotations

from typing import Any, Annotated

from pydantic import BeforeValidator


def _coerce_string_list(value: Any) -> list[str]:
    """Accept provider JSON strings as well as native arrays for list arguments.

    OpenAI-compatible providers occasionally serialize an array-valued tool
    argument as a JSON string.  The public tool contract remains ``list[str]``
    while this boundary normalizes that transport variation before validation.
    """
    import json

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return []
        if candidate.startswith("["):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise ValueError("expected a JSON array of strings") from error
        elif "," in candidate:
            value = [item.strip() for item in candidate.split(",")]
        else:
            value = [candidate]
    if isinstance(value, (tuple, set)):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    if not all(isinstance(item, (str, int, float)) for item in value):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value]


def _coerce_float_pair(value: Any) -> tuple[float, float]:
    """Accept JSON strings as well as native arrays for a two-number argument.

    OpenAI-compatible providers occasionally serialize an array-valued tool argument as a JSON
    string (observed with `time_range`), which would otherwise fail validation with
    "Input should be a valid tuple" and cost the model a whole wasted tool call. The public
    contract stays `tuple[float, float]`; this only normalises that transport variation.
    """
    import json

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError("expected a pair of numbers")
        if candidate.startswith("["):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError as error:
                raise ValueError("expected a JSON array of two numbers") from error
        elif "," in candidate:
            try:
                value = [float(part) for part in candidate.split(",")]
            except ValueError as error:
                raise ValueError("expected two comma-separated numbers") from error
        else:
            raise ValueError("expected a pair of numbers, got a single string")
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("expected exactly two numbers")
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError) as error:
            raise ValueError("expected two numbers") from error
    raise ValueError("expected a pair of numbers")


def _type_for(spec: dict[str, Any]) -> Any:
    """把 yaml 的类型名映射成 Python 注解；未声明的类型退化为 Any，即不拦截参数。"""
    return {"string": str, "integer": int, "number": float, "string_list": Annotated[list[str], BeforeValidator(_coerce_string_list)], "tuple_float": Annotated[tuple[float, float], BeforeValidator(_coerce_float_pair)], "json": Any}.get(str(spec.get("type", "json")), Any)

def _build_schema(name: str, arguments: dict[str, Any]):
    """按声明动态生成 pydantic 参数模型，交给 LangChain 当工具的 args_schema。

    必填项用 ``...`` 占位；可选项一律允许 None（``annotation | None``），
    这样模型即使显式传了 null 也不会卡在校验上——是否使用由服务端自己决定。
    """
    from pydantic import Field, create_model
    fields: dict[str, tuple[Any, Any]] = {}
    for arg, raw_spec in (arguments or {}).items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        annotation = _type_for(spec)
        required = bool(spec.get("required", False))
        default = ... if required else spec.get("default", None)
        if not required: annotation = annotation | None
        fields[arg] = (annotation, Field(default, description=str(spec.get("description", ""))))
    return create_model(f"{name.title().replace('_', '')}Arguments", **fields)

def build_tools(config: dict[str, Any], service: Any) -> list[Any]:
    """把 tools.yaml 的声明逐个装配成可调用的工具，返回顺序即声明顺序。"""
    from langchain_core.tools import StructuredTool
    result = []
    for spec in config.get("tools", []):
        name, method_name = str(spec["name"]), str(spec["method"])
        method = getattr(service, method_name, None)
        # 配置写错应当在启动时就炸掉，而不是等模型真的调用到它
        if not callable(method):
            raise TypeError(f"Configured tool method does not exist: {method_name}")
        arguments = spec.get("arguments") or {}
        schema = _build_schema(name, arguments)
        # 参数名 -> 服务方法关键字：yaml 用蛇形、ToolService 用驼峰，未声明 target 时同名直传
        targets = {arg: str((value or {}).get("target", arg)) for arg, value in arguments.items()}
        # _method / _targets 走默认参数绑定当前循环的值，避免闭包共享循环变量
        def invoke(_method=method, _targets=targets, **kwargs: Any) -> Any:
            # 过滤 None：没传的可选参数不应覆盖服务端自己的默认值
            return _method(**{_targets[key]: value for key, value in kwargs.items() if value is not None})
        result.append(StructuredTool.from_function(invoke, name=name, description=str(spec.get("description", "")), args_schema=schema))
    return result

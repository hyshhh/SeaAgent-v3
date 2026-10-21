"""把 config/tools.yaml 的声明翻译成 LangChain StructuredTool，执行体转发给 ToolService 的同名方法。"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator


# ------------------------------------------------------------------ #
# 一、参数归一与 schema                                                     #
# ------------------------------------------------------------------ #
# 1.1 掰回字符串列表
def _coerce_string_list(value: Any) -> list[str]:
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
        raise ValueError("expected a list of strings")  # noqa: TRY004 - Pydantic validators require ValueError.
    if not all(isinstance(item, (str, int, float)) for item in value):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value]


# 1.2 掰回时间区间
def _coerce_float_pair(value: Any) -> tuple[float, float]:
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


# 1.3 声明类型 → Python 注解
def _type_for(spec: dict[str, Any]) -> Any:
    return {"string": str, "integer": int, "number": float, "string_list": Annotated[list[str], BeforeValidator(_coerce_string_list)], "tuple_float": Annotated[tuple[float, float], BeforeValidator(_coerce_float_pair)], "json": Any}.get(str(spec.get("type", "json")), Any)

# 1.4 由声明生成参数模型
def _build_schema(name: str, arguments: dict[str, Any]):
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

# ------------------------------------------------------------------ #
# 二、工具装配                                                             #
# ------------------------------------------------------------------ #
# 2.1 yaml 声明 → 可调用的工具
def build_tools(config: dict[str, Any], service: Any) -> list[Any]:
    from langchain_core.tools import (
        StructuredTool,  # 延迟导入：保持 import harness.tools 轻量
    )

    result = []
    for spec in config.get("tools", []):
        name, method_name = str(spec["name"]), str(spec["method"])    # 用下标而非 .get：缺字段是配置写错了，应当立刻抛错
        method = getattr(service, method_name, None)
        if not callable(method):  # 配置写错要在启动时炸掉，否则会伪装成运行期工具失败
            raise TypeError(f"Configured tool method does not exist: {method_name}")
        arguments = spec.get("arguments") or {}  # or {} 兜住 yaml 里写成空值的情况
        schema = _build_schema(name, arguments)  # 既生成发给模型的 JSON Schema，也在调用边界做校验与归一
        targets = {arg: str((value or {}).get("target", arg)) for arg, value in arguments.items()}  # yaml 蛇形 -> ToolService 驼峰，未声明 target 时同名直传
        def invoke(_method=method, _targets=targets, **kwargs: Any) -> Any:  # 默认参数绑定当前值：闭包捕获变量，循环结束后会全指到最后一次
            return _method(**{_targets[key]: value for key, value in kwargs.items() if value is not None})  # 过滤 None：可选项没传时 LangChain 会填 None
        result.append(StructuredTool.from_function(invoke, name=name, description=str(spec.get("description", "")), args_schema=schema))  # description 就是喂给模型的工具说明
    return result

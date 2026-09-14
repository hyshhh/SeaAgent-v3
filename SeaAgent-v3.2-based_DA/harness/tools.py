"""Configuration-driven tool adapter for Sea-Video-Harness."""
from __future__ import annotations

from typing import Any


def _type_for(spec: dict[str, Any]) -> Any:
    return {"string": str, "integer": int, "number": float, "string_list": list[str], "tuple_float": tuple[float, float], "json": Any}.get(str(spec.get("type", "json")), Any)

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

def build_tools(config: dict[str, Any], service: Any) -> list[Any]:
    from langchain_core.tools import StructuredTool
    result = []
    for spec in config.get("tools", []):
        name, method_name = str(spec["name"]), str(spec["method"])
        method = getattr(service, method_name, None)
        if not callable(method):
            raise TypeError(f"Configured tool method does not exist: {method_name}")
        arguments = spec.get("arguments") or {}
        schema = _build_schema(name, arguments)
        targets = {arg: str((value or {}).get("target", arg)) for arg, value in arguments.items()}
        def invoke(_method=method, _targets=targets, **kwargs: Any) -> Any:
            return _method(**{_targets[key]: value for key, value in kwargs.items() if value is not None})
        result.append(StructuredTool.from_function(invoke, name=name, description=str(spec.get("description", "")), args_schema=schema))
    return result

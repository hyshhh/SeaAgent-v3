"""Compare saved accuracy metrics from two independent runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import write_json


def _numeric_differences(
    left: Any,
    right: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) & set(right)):
            if key == "latency":
                continue
            differences.extend(
                _numeric_differences(left[key], right[key], f"{path}.{key}".strip("."))
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
            differences.extend(
                _numeric_differences(
                    left_value,
                    right_value,
                    f"{path}.{index}".strip("."),
                )
            )
        return differences
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        absolute = abs(float(left) - float(right))
        relative = absolute / max(abs(float(left)), abs(float(right)), 1e-12)
        return [
            {
                "path": path,
                "left": float(left),
                "right": float(right),
                "absolute_difference": absolute,
                "relative_difference": relative,
            }
        ]
    return []


def compare_run_metrics(
    left_run: str | Path,
    right_run: str | Path,
    tolerance: float = 0.05,
    output: str | Path | None = None,
) -> dict[str, Any]:
    left_path = Path(left_run) / "evaluation" / "metrics.json"
    right_path = Path(right_run) / "evaluation" / "metrics.json"
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    differences = _numeric_differences(left["metrics"], right["metrics"], "metrics")
    maximum = max(
        (item["relative_difference"] for item in differences),
        default=0.0,
    )
    result = {
        "left_run": str(left_path.parent.parent),
        "right_run": str(right_path.parent.parent),
        "tolerance": float(tolerance),
        "maximum_relative_difference": maximum,
        "timing_compared": False,
        "verdict": (
            "REPRODUCIBLE"
            if maximum <= float(tolerance)
            else "PARTIALLY_REPRODUCIBLE"
        ),
        "differences": [
            item for item in differences if item["absolute_difference"] > 1e-12
        ],
    }
    if output:
        write_json(output, result)
    return result

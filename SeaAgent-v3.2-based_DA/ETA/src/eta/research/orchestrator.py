"""Run structural ablations and KEP-capacity experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ..dataset import build_dataset
from ..evaluation import evaluate_run
from ..training import train_model
from ..utils import environment_manifest, utc_timestamp, write_json


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _result(
    label: str,
    config: dict[str, Any],
    data_summary: dict[str, Any],
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "variant": config["model"]["variant"],
        "capacity": int(config["data"]["history_pool_size"]),
        "run_id": training["run_id"],
        "run_dir": training["run_dir"],
        "best_epoch": training["best_epoch"],
        "data_splits": data_summary["splits"],
        "samples": data_summary["samples"],
        "metrics": evaluation["metrics"],
        "latency": evaluation["latency"],
    }


def run_research_series(
    config: dict[str, Any],
    ablation_config: str | Path,
    capacity_config: str | Path,
    include_ablations: bool = True,
    include_capacity: bool = True,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    run_label = f"research-{utc_timestamp()}"
    baseline_result: dict[str, Any] | None = None
    if include_ablations:
        data_summary = build_dataset(config)
        for item in _read_yaml(ablation_config).get("variants", []):
            variant = str(item["variant"])
            print(f"starting ablation variant={variant}", flush=True)
            variant_config = copy.deepcopy(config)
            variant_config["model"]["variant"] = variant
            training = train_model(variant_config, run_name=f"ablation-{variant}")
            evaluation = evaluate_run(variant_config, training["run_dir"])
            current = _result(
                f"ablation:{variant}",
                variant_config,
                data_summary,
                training,
                evaluation,
            )
            results.append(current)
            print(
                f"finished ablation variant={variant} "
                f"ah_accuracy={evaluation['metrics']['ah']['accuracy']:.4f}",
                flush=True,
            )
            if variant == "eta":
                baseline_result = current
    capacity_items = _read_yaml(capacity_config).get("capacities", []) if include_capacity else []
    root_artifact = Path(config["paths"]["artifact_dir"])
    for item in capacity_items:
        capacity = int(item["capacity"])
        if capacity == int(config["data"]["history_pool_size"]) and baseline_result is not None:
            results.append({**baseline_result, "label": f"capacity:{capacity}", "reused": True})
            continue
        capacity_config_value = copy.deepcopy(config)
        capacity_config_value["data"]["history_pool_size"] = capacity
        capacity_config_value["paths"]["processed_dir"] = str(
            root_artifact / "capacity" / f"M{capacity}" / "processed"
        )
        print(f"starting capacity experiment M={capacity}", flush=True)
        data_summary = build_dataset(capacity_config_value)
        training = train_model(
            capacity_config_value,
            run_name=f"capacity-M{capacity}",
        )
        evaluation = evaluate_run(capacity_config_value, training["run_dir"])
        results.append(
            _result(
                f"capacity:{capacity}",
                capacity_config_value,
                data_summary,
                training,
                evaluation,
            )
        )
        print(
            f"finished capacity experiment M={capacity} "
            f"ah_accuracy={evaluation['metrics']['ah']['accuracy']:.4f}",
            flush=True,
        )
    report = {
        "run_label": run_label,
        "timestamp_utc": utc_timestamp(),
        "environment": environment_manifest(),
        "config_hash_material": {
            "model": config["model"],
            "data": config["data"],
            "train": config["train"],
            "evaluation": config["evaluation"],
        },
        "results": results,
    }
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"{run_label}.json", report)
    from ..evaluation.reporter import render_research_report

    render_research_report(report, report_dir / f"{run_label}.md", report_dir)
    write_json(report_dir / "latest.json", report)
    return report

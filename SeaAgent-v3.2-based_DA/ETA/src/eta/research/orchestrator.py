"""Run structural ablations and hyperparameter comparisons."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import yaml

from ..dataset import build_dataset
from ..evaluation import evaluate_run
from ..evaluation.reporter import publish_run, render_research_report
from ..training import train_model
from ..utils import deep_update, environment_manifest, utc_timestamp, write_json


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _validation_accuracy(run_dir: str | Path, best_epoch: int) -> tuple[float, float | None]:
    with (Path(run_dir) / "history.csv").open("r", encoding="utf-8", newline="") as stream:
        row = next(
            row
            for row in csv.DictReader(stream)
            if int(row["epoch"]) == int(best_epoch)
        )
    ah = float(row["validation_ah_accuracy"])
    ch = float(row["validation_ch_accuracy"]) if "validation_ch_accuracy" in row else None
    return ah, ch


def _result(
    group: str,
    item: dict[str, Any],
    config: dict[str, Any],
    data_summary: dict[str, Any],
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    validation_ah, validation_ch = _validation_accuracy(
        training["run_dir"],
        int(training["best_epoch"]),
    )
    validation_mean = (
        (validation_ah + validation_ch) / 2.0
        if validation_ch is not None
        else validation_ah
    )
    return {
        "id": item["id"],
        "name": item["name"],
        "group": group,
        "variant": config["model"]["variant"],
        "run_id": training["run_id"],
        "run_dir": training["run_dir"],
        "best_epoch": training["best_epoch"],
        "data_splits": data_summary["splits"],
        "samples": data_summary["samples"],
        "validation_accuracy": {
            "ah": validation_ah,
            "ch": validation_ch,
            "mean": validation_mean,
        },
        "metrics": evaluation["metrics"],
        "latency": evaluation["latency"],
        "eligible_for_final": config["model"]["variant"] != "ca_sh",
    }


def _run_group(
    default_config: dict[str, Any],
    data_summary: dict[str, Any],
    group: str,
    item: dict[str, Any],
    run_prefix: str,
) -> dict[str, Any]:
    config = copy.deepcopy(default_config)
    deep_update(config, item.get("overrides", {}))
    if "variant" in item:
        config["model"]["variant"] = str(item["variant"])
    print(f"starting {group} id={item['id']} name={item['name']}", flush=True)
    training = train_model(config, run_name=f"{run_prefix}-{item['id']}")
    evaluation = evaluate_run(config, training["run_dir"])
    result = _result(group, item, config, data_summary, training, evaluation)
    print(
        f"finished {group} id={item['id']} "
        f"validation_mean={result['validation_accuracy']['mean']:.4f} "
        f"test_ah={result['metrics']['ah']['accuracy']:.4f} "
        f"test_ch={result['metrics']['ch']['accuracy'] if result['metrics']['ch'] else 0.0:.4f}",
        flush=True,
    )
    return result


def run_research_series(
    config: dict[str, Any],
    ablation_config: str | Path,
    tuning_config: str | Path,
    include_ablations: bool = True,
    include_tuning: bool = True,
) -> dict[str, Any]:
    data_summary = build_dataset(config)
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    if include_ablations:
        suite = _read_yaml(ablation_config)
        suite_config = deep_update(copy.deepcopy(config), suite.get("defaults", {}))
        notes.extend(str(note) for note in suite.get("notes", []))
        for item in suite.get("variants", []):
            results.append(
                _run_group(
                    suite_config,
                    data_summary,
                    "消融组",
                    item,
                    "ablation",
                )
            )
    if include_tuning:
        suite = _read_yaml(tuning_config)
        suite_config = deep_update(copy.deepcopy(config), suite.get("defaults", {}))
        for item in suite.get("groups", []):
            results.append(
                _run_group(
                    suite_config,
                    data_summary,
                    "调参组",
                    item,
                    "tuning",
                )
            )
    candidates = [result for result in results if result["eligible_for_final"]]
    best = max(
        candidates,
        key=lambda result: (
            result["validation_accuracy"]["mean"],
            -result["latency"]["p50_ms"],
        ),
    )
    report = {
        "run_label": f"research-{utc_timestamp()}",
        "timestamp_utc": utc_timestamp(),
        "environment": environment_manifest(),
        "config": config,
        "data_summary": data_summary,
        "selection_rule": (
            "按验证集平均双头准确率最大化；同分时选择更低的 batch-1 P50 时延"
        ),
        "best": {
            "id": best["id"],
            "name": best["name"],
            "group": best["group"],
            "run_id": best["run_id"],
            "run_dir": best["run_dir"],
            "validation_accuracy": best["validation_accuracy"],
        },
        "results": results,
        "ablation_notes": notes,
    }
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "latest.json", report)
    render_research_report(report, report_dir / "experiments.md", report_dir)
    publish_run(
        best["run_dir"],
        report_dir,
        artifact_dir=config["paths"]["artifact_dir"],
    )
    return report

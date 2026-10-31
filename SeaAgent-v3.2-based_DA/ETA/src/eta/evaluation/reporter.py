"""Render machine-readable experiment outputs into research reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import write_json
from .paired import paired_comparison


def _accuracy_with_interval(values: dict[str, Any]) -> str:
    interval = values.get("bootstrap_95_accuracy", [0.0, 0.0])
    return f"{values['accuracy']:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def render_research_report(
    report: dict[str, Any],
    markdown_path: Path,
    artifact_dir: Path,
) -> None:
    rows = report["results"]
    lines = [
        f"# Automated ETA research run: {report['run_label']}",
        "",
        "> Status: proxy-label research run. AH/CH labels are deterministic weak labels,",
        "> not the paper's human annotations. Results cannot be presented as reproduction",
        "> of the paper's target accuracy.",
        "",
        (
            "| Experiment | AH acc. [95% CI] | CH acc. [95% CI] | "
            "Joint acc. | Static Stop F1 | Latency p50 (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        metrics = result["metrics"]
        ch_accuracy = (
            _accuracy_with_interval(metrics["ch"])
            if metrics["ch"] is not None
            else "N/A"
        )
        joint = (
            f"{metrics['joint_accuracy']:.4f}"
            if metrics["joint_accuracy"] is not None
            else "N/A"
        )
        stop = f"{metrics['stop']['f1']:.4f}" if metrics["stop"] is not None else "N/A"
        lines.append(
            f"| {result['label']} | {_accuracy_with_interval(metrics['ah'])} | "
            f"{ch_accuracy} | {joint} | {stop} | "
            f"{result['latency']['p50_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Accuracy is measured on the held-out source-video split with proxy labels.",
            "- Intervals are 1,000-resample non-parametric bootstrap 95% intervals.",
            "- Batch-1 latency excludes image encoding and includes DPD forward only.",
            "- Static Stop F1 measures two-head threshold agreement on saved samples.",
            "  Actual insertion gating is enforced by `EvidencePool` and covered by tests;",
            "  sequence-level online replay remains future work.",
            "- Timing is environment-sensitive and is not treated as reproducible across GPUs.",
            "- All 11 statistical fallacies and the relevant integrity checks are recorded in",
            "  `docs/integrity_and_limitations.md`.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    with (artifact_dir / "research_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "label",
            "variant",
            "capacity",
            "train_samples",
            "validation_samples",
            "test_samples",
            "ah_accuracy",
            "ah_ci_lower",
            "ah_ci_upper",
            "ch_accuracy",
            "joint_accuracy",
            "static_stop_f1",
            "latency_p50_ms",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            metrics = result["metrics"]
            writer.writerow(
                {
                    "label": result["label"],
                    "variant": result["variant"],
                    "capacity": result["capacity"],
                    "train_samples": result["data_splits"]["train"]["samples"],
                    "validation_samples": result["data_splits"]["validation"]["samples"],
                    "test_samples": result["data_splits"]["test"]["samples"],
                    "ah_accuracy": metrics["ah"]["accuracy"],
                    "ah_ci_lower": metrics["ah"]["bootstrap_95_accuracy"][0],
                    "ah_ci_upper": metrics["ah"]["bootstrap_95_accuracy"][1],
                    "ch_accuracy": (
                        metrics["ch"]["accuracy"] if metrics["ch"] is not None else ""
                    ),
                    "joint_accuracy": (
                        metrics["joint_accuracy"]
                        if metrics["joint_accuracy"] is not None
                        else ""
                    ),
                    "static_stop_f1": (
                        metrics["stop"]["f1"] if metrics["stop"] is not None else ""
                    ),
                    "latency_p50_ms": result["latency"]["p50_ms"],
                }
            )
    history_rows = []
    for result in rows:
        history_path = Path(result["run_dir"]) / "history.csv"
        if not history_path.exists():
            continue
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                history_rows.append(
                    {
                        "experiment": result["label"],
                        "run_id": result["run_id"],
                        **row,
                    }
                )
    if history_rows:
        with (artifact_dir / "training_histories.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history_rows[0]))
            writer.writeheader()
            writer.writerows(history_rows)
    baseline = next(
        (result for result in rows if result["label"] == "ablation:eta"),
        None,
    )
    if baseline is not None:
        paired = [
            {
                "candidate_label": result["label"],
                **paired_comparison(
                    baseline["run_dir"],
                    result["run_dir"],
                    seed=42,
                ),
            }
            for result in rows
            if result["label"].startswith("ablation:") and result is not baseline
        ]
        write_json(
            artifact_dir / "paired_analysis.json",
            {
                "reference": "ablation:eta",
                "comparisons": paired,
                "interval_method": "track-clustered bootstrap",
                "iterations": 2000,
            },
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ablation_rows = [row for row in rows if row["label"].startswith("ablation:")]
    if ablation_rows:
        labels = [row["label"].split(":", 1)[1] for row in ablation_rows]
        figure, axis = plt.subplots(figsize=(9, 4.8))
        ah_values = np.asarray(
            [row["metrics"]["ah"]["accuracy"] for row in ablation_rows]
        )
        ah_intervals = np.asarray(
            [row["metrics"]["ah"]["bootstrap_95_accuracy"] for row in ablation_rows]
        )
        axis.bar(
            labels,
            ah_values,
            yerr=[
                ah_values - ah_intervals[:, 0],
                ah_intervals[:, 1] - ah_values,
            ],
            capsize=3,
            label="AH accuracy",
            alpha=0.8,
        )
        joint_values = [
            row["metrics"]["joint_accuracy"]
            if row["metrics"]["joint_accuracy"] is not None
            else np.nan
            for row in ablation_rows
        ]
        joint_errors = [
            row["metrics"].get("joint_bootstrap_95_accuracy")
            if row["metrics"]["joint_accuracy"] is not None
            else None
            for row in ablation_rows
        ]
        joint_lower = np.asarray(
            [
                error[0] if error is not None else np.nan
                for error in joint_errors
            ]
        )
        joint_upper = np.asarray(
            [
                error[1] if error is not None else np.nan
                for error in joint_errors
            ]
        )
        axis.errorbar(
            labels,
            joint_values,
            yerr=[
                np.asarray(joint_values) - joint_lower,
                joint_upper - np.asarray(joint_values),
            ],
            marker="o",
            capsize=3,
            label="Joint accuracy",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Accuracy")
        axis.set_title("TEA structural ablations (proxy labels)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(artifact_dir / "ablation_accuracy.png", dpi=180)
        plt.close(figure)
    capacity_rows = [row for row in rows if row["label"].startswith("capacity:")]
    if capacity_rows:
        capacities = [row["capacity"] for row in capacity_rows]
        ah_values = np.asarray(
            [row["metrics"]["ah"]["accuracy"] for row in capacity_rows]
        )
        ah_intervals = np.asarray(
            [row["metrics"]["ah"]["bootstrap_95_accuracy"] for row in capacity_rows]
        )
        figure, axis = plt.subplots(figsize=(7, 4.6))
        axis.errorbar(
            capacities,
            ah_values,
            yerr=[
                ah_values - ah_intervals[:, 0],
                ah_intervals[:, 1] - ah_values,
            ],
            marker="o",
            capsize=3,
            label="AH accuracy",
        )
        axis.set_xlabel("KEP capacity M")
        axis.set_ylabel("AH accuracy")
        axis.set_ylim(0, 1)
        axis.set_title("Capacity sensitivity (proxy labels)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(artifact_dir / "capacity_accuracy.png", dpi=180)
        plt.close(figure)

"""Publish strict ETA run reports and experiment tables."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..utils import write_json


def _dual_accuracy(metrics: dict[str, Any]) -> str:
    ch = metrics.get("ch")
    return (
        f"AH {metrics['ah']['accuracy']:.2%} / "
        f"CH {ch['accuracy']:.2%}"
        if ch is not None
        else f"AH {metrics['ah']['accuracy']:.2%} / CH —"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_accuracy(
    run_path: Path,
    best_epoch: int | None,
) -> dict[str, float | None] | None:
    history_path = run_path / "history.csv"
    if best_epoch is None or not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8", newline="") as stream:
        row = next(
            (
                item
                for item in csv.DictReader(stream)
                if int(item["epoch"]) == int(best_epoch)
            ),
            None,
        )
    if row is None:
        return None
    ah = float(row["validation_ah_accuracy"])
    ch = (
        float(row["validation_ch_accuracy"])
        if row.get("validation_ch_accuracy")
        else None
    )
    return {
        "ah": ah,
        "ch": ch,
        "mean": (ah + ch) / 2.0 if ch is not None else ah,
    }


def publish_run(
    run_dir: str | Path,
    report_dir: str | Path,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (run_path / "evaluation" / "metrics.json").read_text(encoding="utf-8")
    )
    metrics = evaluation["metrics"]
    report = {
        "run_id": run["run_id"],
        "variant": run["variant"],
        "data_summary": run["data_summary"],
        "environment": run["environment"],
        "metrics": metrics,
        "latency": evaluation["latency"],
        "validation_accuracy": _validation_accuracy(
            run_path,
            run.get("best_epoch"),
        ),
    }
    checkpoint = run_path / "best.pt"
    if artifact_dir is not None:
        destination = Path(artifact_dir) / "best_model" / "best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination)
        report["checkpoint"] = {
            "path": "artifacts/best_model/best.pt",
            "sha256": _sha256(destination),
        }
    write_json(output / "best_run.json", report)
    lines = [
        "# ETA 最优模型",
        "",
        "| 实验设置 | 消融组/调参组 | 双头精度 | 推理速度（单次ms） |",
        "|---|---|---|---:|",
        (
            f"| {run['variant']} | 最优模型 | {_dual_accuracy(metrics)} | "
            f"{evaluation['latency']['p50_ms']:.3f} |"
        ),
        "",
        f"- 运行编号：`{run['run_id']}`",
        "",
        "双头精度为测试集结果；模型与超参数仅依据验证集选择，同分时以 P50 时延决胜。",
        "准确率受类别不平衡影响，精确率、召回率、F1、校准与混淆矩阵见 JSON。",
        "",
    ]
    validation = report["validation_accuracy"]
    if validation is not None:
        lines.extend(
            [
                (
                    "- 验证集双头精度："
                    f"AH {validation['ah']:.2%} / "
                    f"CH {validation['ch']:.2%} / "
                    f"均值 {validation['mean']:.2%}"
                    if validation["ch"] is not None
                    else f"- 验证集 AH 精度：{validation['ah']:.2%}"
                ),
                "",
            ]
        )
    if "checkpoint" in report:
        lines.extend(
            [
                f"- 权重路径：`{report['checkpoint']['path']}`",
                f"- 权重 SHA-256：`{report['checkpoint']['sha256']}`",
                "",
            ]
        )
    (output / "best_run.md").write_text("\n".join(lines), encoding="utf-8")
    with (output / "best_run.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["实验设置", "消融组/调参组", "双头精度", "推理速度（单次ms）"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "实验设置": run["variant"],
                "消融组/调参组": "最优模型",
                "双头精度": _dual_accuracy(metrics),
                "推理速度（单次ms）": f"{evaluation['latency']['p50_ms']:.3f}",
            }
        )
    return report


def render_research_report(
    report: dict[str, Any],
    markdown_path: Path,
    artifact_dir: Path,
) -> None:
    rows = report["results"]
    write_json(artifact_dir / "dataset_summary.json", report["data_summary"])
    for legacy_name in (
        "latest.md",
        "training_histories.csv",
        "research_results.csv",
        "ablation_accuracy.png",
        "capacity_accuracy.png",
        "checksums.sha256",
        "paired_analysis.json",
        "stride5_summary.json",
        "research-20261030T143150Z.json",
        "research-20261030T143150Z.md",
    ):
        (artifact_dir / legacy_name).unlink(missing_ok=True)
    for legacy_path in artifact_dir.glob("research-*.json"):
        if legacy_path.name != "best_run.json":
            legacy_path.unlink()
    for legacy_path in artifact_dir.glob("research-*.md"):
        if legacy_path.name != "best_run.md":
            legacy_path.unlink()
    lines = [
        "# ETA 消融与调参结果",
        "",
        f"最优模型：`{report['best']['id']} {report['best']['name']}`",
        "",
        "| 实验设置 | 消融组/调参组 | 双头精度 | 推理速度（单次ms） |",
        "|---|---|---|---:|",
    ]
    for result in rows:
        lines.append(
            f"| {result['id']} {result['name']} | {result['group']} | "
            f"{_dual_accuracy(result['metrics'])} | "
            f"{result['latency']['p50_ms']:.3f} |"
        )
    if report.get("ablation_notes"):
        lines.extend(["", "消融组说明：", ""])
        lines.extend(f"- {note}" for note in report["ablation_notes"])
    lines.extend(
        [
            "",
            (
                "说明：双头精度为固定测试集结果；模型选择仅使用验证集平均双头准确率，"
                "同分时以 batch-1 P50 时延决胜。"
                "推理速度是 batch-1 单次前向的 P50，不包含图像编码。"
                "准确率受类别不平衡影响，精确率、召回率、F1、校准与混淆矩阵保存在"
                "latest.json；本表不据此宣称均衡分类性能。"
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    with (artifact_dir / "experiments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["实验设置", "消融组/调参组", "双头精度", "推理速度（单次ms）"],
        )
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    "实验设置": f"{result['id']} {result['name']}",
                    "消融组/调参组": result["group"],
                    "双头精度": _dual_accuracy(result["metrics"]),
                    "推理速度（单次ms）": f"{result['latency']['p50_ms']:.3f}",
                }
            )

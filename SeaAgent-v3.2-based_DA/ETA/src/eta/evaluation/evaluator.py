"""Checkpoint evaluation, calibration, confusion matrices and latency."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import seed_everything
from ..dataset import FeatureDataset
from ..metrics import binary_metrics, stop_metrics, summarize_predictions
from ..model import build_model
from ..utils import resolve_device, utc_timestamp, write_json


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, np.ndarray]:
    admission = []
    completeness = []
    ah_labels = []
    ch_labels = []
    sample_indices: list[int] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            candidate = batch["candidate"].to(device=device, non_blocking=True)
            history = batch["history"].to(device=device, non_blocking=True)
            history_mask = batch["history_mask"].to(device=device, non_blocking=True)
            ah_logits, ch_logits = model(candidate, history, history_mask)
            admission.append(torch.softmax(ah_logits, dim=1)[:, 1].cpu().numpy())
            if ch_logits is not None:
                completeness.append(torch.softmax(ch_logits, dim=1)[:, 1].cpu().numpy())
            ah_labels.append(batch["ah_label"].numpy())
            ch_labels.append(batch["ch_label"].numpy())
            start = len(sample_indices)
            sample_indices.extend(range(start, start + len(batch["ah_label"])))
    result = {
        "admission_probability": np.concatenate(admission),
        "ah_label": np.concatenate(ah_labels),
        "ch_label": np.concatenate(ch_labels),
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
    }
    result["completeness_probability"] = (
        np.concatenate(completeness) if completeness else None
    )
    return result


def _latency_ms(model: torch.nn.Module, dataset: FeatureDataset, device: str, repeats: int) -> dict[str, float]:
    sample = {
        key: value.unsqueeze(0).to(device)
        for key, value in dataset[0].items()
        if key in {"candidate", "history", "history_mask"}
    }
    warmups = min(10, max(1, repeats // 10))
    values: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(warmups):
            model(**sample)
        for _ in range(repeats):
            if device.startswith("cuda"):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                model(**sample)
                end.record()
                torch.cuda.synchronize()
                values.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter_ns()
                model(**sample)
                values.append((time.perf_counter_ns() - started) / 1_000_000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _write_predictions(path: Path, predictions: dict[str, np.ndarray]) -> None:
    fields = [
        "sample_index",
        "admission_probability",
        "ah_label",
        "completeness_probability",
        "ch_label",
    ]
    completeness = predictions["completeness_probability"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample_index in enumerate(predictions["sample_index"]):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "admission_probability": float(
                        predictions["admission_probability"][index]
                    ),
                    "ah_label": int(predictions["ah_label"][index]),
                    "completeness_probability": (
                        "" if completeness is None else float(completeness[index])
                    ),
                    "ch_label": int(predictions["ch_label"][index]),
                }
            )


def _select_thresholds(
    predictions: dict[str, np.ndarray],
    accuracy_floor: float,
) -> dict[str, float]:
    grid = np.linspace(0.05, 0.95, 91)
    admission = []
    for threshold in grid:
        values = binary_metrics(
            predictions["admission_probability"],
            predictions["ah_label"],
            threshold,
        )
        admission.append((values["accuracy"], values["f1"], float(threshold)))
    ah_threshold = max(admission, key=lambda row: (row[0], row[1]))[2]
    completeness = predictions["completeness_probability"]
    if completeness is None:
        return {"ah": ah_threshold, "ch": 0.5}
    candidates = []
    for threshold in grid:
        values = binary_metrics(
            completeness,
            predictions["ch_label"],
            threshold,
        )
        candidates.append(
            (
                values["accuracy"],
                values["f1"],
                values["recall"],
                float(threshold),
            )
        )
    viable = [row for row in candidates if row[0] >= accuracy_floor]
    chosen = (
        max(viable, key=lambda row: (row[1], row[0], row[2]))
        if viable
        else max(candidates, key=lambda row: (row[0], row[1]))
    )
    return {"ah": ah_threshold, "ch": chosen[3]}


def _report_markdown(metrics: dict[str, Any], latency: dict[str, float], run_id: str) -> str:
    lines = [
        f"# ETA 评测结果：{run_id}",
        "",
        "结果基于确定性代理标签，不代表人工标注准确率。",
        "",
        "| 输出头 | 准确率 | 精确率 | 召回率 | F1 | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.append(
        "| AH | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {ece:.4f} |".format(
            **metrics["ah"],
            ece=metrics["ah_calibration"]["expected_calibration_error"],
        )
    )
    if metrics["ch"] is not None:
        lines.append(
            "| CH | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {ece:.4f} |".format(
                **metrics["ch"],
                ece=metrics["ch_calibration"]["expected_calibration_error"],
            )
        )
        lines.extend(
            [
                "",
                f"- 联合准确率：{metrics['joint_accuracy']:.4f}",
                (
                    f"- 验证集选择的阈值：AH "
                    f"{metrics['threshold_selection']['ah']:.2f}, CH "
                    f"{metrics['threshold_selection']['ch']:.2f}"
                ),
                (
                    f"- 阈值 {metrics['stop']['threshold']:.2f} 下的静态 Stop F1："
                    f"{metrics['stop']['f1']:.4f}"
                ),
            ]
        )
    lines.extend(
        [
            "",
              (
                  f"- batch=1 时延，均值/P50/P95："
                  f"{latency['mean_ms']:.3f} / {latency['p50_ms']:.3f} / "
                  f"{latency['p95_ms']:.3f} ms"
              ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_run(
    config: dict[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    run_path = Path(run_dir)
    checkpoint_path = run_path / "best.pt"
    test_path = Path(config["paths"]["processed_dir"]) / "test.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")
    device = resolve_device(str(config["train"]["device"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(config)
    model_config["model"] = checkpoint["model_config"]
    model = build_model(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    dataset = FeatureDataset(test_path)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    predictions = _predict(model, loader, device)
    validation_path = Path(config["paths"]["processed_dir"]) / "validation.npz"
    validation_predictions = None
    if validation_path.exists():
        validation_dataset = FeatureDataset(validation_path)
        if len(validation_dataset):
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(config["evaluation"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=device.startswith("cuda"),
            )
            validation_predictions = _predict(model, validation_loader, device)
    thresholds = (
        _select_thresholds(
            validation_predictions,
            float(config["evaluation"].get("selection_accuracy_floor", 0.8)),
        )
        if validation_predictions is not None
        else {"ah": 0.5, "ch": 0.5}
    )
    metrics = summarize_predictions(
        predictions["admission_probability"],
        predictions["completeness_probability"],
        predictions["ah_label"],
        predictions["ch_label"],
        float(config["evaluation"]["stop_threshold"]),
        int(config["evaluation"]["calibration_bins"]),
        int(config["evaluation"].get("bootstrap_iterations", 1000)),
        int(config["project"]["seed"]),
        thresholds["ah"],
        thresholds["ch"],
    )
    metrics["threshold_selection"] = thresholds
    if predictions["completeness_probability"] is not None:
        metrics["stop_sensitivity"] = [
            stop_metrics(
                predictions["admission_probability"],
                predictions["completeness_probability"],
                predictions["ah_label"],
                predictions["ch_label"],
                float(threshold),
                thresholds["ah"],
            )
            for threshold in config["evaluation"].get(
                "stop_thresholds",
                [config["evaluation"]["stop_threshold"]],
            )
        ]
    latency = _latency_ms(
        model,
        dataset,
        device,
        int(config["evaluation"]["latency_repeats"]),
    )
    output_dir = run_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(output_dir / "predictions.csv", predictions)
    result = {
        "run_id": checkpoint["run_id"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": device,
        "split": "test",
        "metrics": metrics,
        "latency": latency,
        "completed_at_utc": utc_timestamp(),
    }
    write_json(output_dir / "metrics.json", result)
    (output_dir / "report.md").write_text(
        _report_markdown(metrics, latency, checkpoint["run_id"]),
        encoding="utf-8",
    )
    run_manifest_path = run_path / "run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["status"] = "evaluated"
    run_manifest["evaluation"] = result
    write_json(run_manifest_path, run_manifest)
    return result

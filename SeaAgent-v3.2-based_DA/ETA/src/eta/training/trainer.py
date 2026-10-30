"""Paper-aligned ETA/DPD training."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import ETA_ROOT, seed_everything
from ..dataset import FeatureDataset
from ..model import build_model
from ..utils import (
    environment_manifest,
    hash_value,
    resolve_device,
    tree_sha256,
    utc_timestamp,
    write_json,
    write_yaml,
)


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float64)
    weights = np.ones(2, dtype=np.float64)
    present = counts > 0
    weights[present] = len(labels) / (2.0 * counts[present])
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def _scheduler(optimizer: AdamW, total_steps: int, warmup_epochs: int, epochs: int) -> LambdaLR:
    warmup_steps = min(
        total_steps,
        max(0, round(total_steps * float(warmup_epochs) / max(1, epochs))),
    )

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, float(step + 1) / warmup_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def _loader(
    dataset: FeatureDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: str,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    losses: dict[str, nn.Module],
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    grad_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "ah_correct": 0.0,
        "ch_correct": 0.0,
        "joint_correct": 0.0,
        "count": 0.0,
    }
    for batch in loader:
        candidate = batch["candidate"].to(device=device, non_blocking=True)
        history = batch["history"].to(device=device, non_blocking=True)
        history_mask = batch["history_mask"].to(device=device, non_blocking=True)
        ah_label = batch["ah_label"].to(device=device, non_blocking=True)
        ch_label = batch["ch_label"].to(device=device, non_blocking=True)
        with torch.set_grad_enabled(training):
            admission_logits, completeness_logits = model(candidate, history, history_mask)
            loss = losses["ah"](admission_logits, ah_label)
            if completeness_logits is not None:
                loss = loss + losses["ch"](completeness_logits, ch_label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        count = int(ah_label.shape[0])
        ah_pred = admission_logits.argmax(dim=1)
        totals["loss"] += float(loss.detach()) * count
        totals["ah_correct"] += float((ah_pred == ah_label).sum())
        totals["count"] += count
        if completeness_logits is not None:
            ch_pred = completeness_logits.argmax(dim=1)
            totals["ch_correct"] += float((ch_pred == ch_label).sum())
            totals["joint_correct"] += float(
                ((ah_pred == ah_label) & (ch_pred == ch_label)).sum()
            )
    count = max(1.0, totals["count"])
    result = {
        "loss": totals["loss"] / count,
        "ah_accuracy": totals["ah_correct"] / count,
    }
    if model.heads.completeness is not None:
        result["ch_accuracy"] = totals["ch_correct"] / count
        result["joint_accuracy"] = totals["joint_correct"] / count
    return result


def train_model(
    config: dict[str, Any],
    run_name: str | None = None,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    processed_dir = Path(config["paths"]["processed_dir"])
    required = [processed_dir / f"{split}.npz" for split in ("train", "validation")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Build the dataset first; missing: {', '.join(missing)}")
    device = resolve_device(str(config["train"]["device"]))
    train_data = FeatureDataset(processed_dir / "train.npz")
    validation_data = FeatureDataset(processed_dir / "validation.npz")
    if not len(train_data) or not len(validation_data):
        raise RuntimeError("Train and validation splits must both be non-empty")
    model = build_model(config).to(device)
    train_loader = _loader(
        train_data,
        int(config["train"]["batch_size"]),
        True,
        int(config["train"]["num_workers"]),
        device,
    )
    validation_loader = _loader(
        validation_data,
        int(config["train"]["batch_size"]),
        False,
        int(config["train"]["num_workers"]),
        device,
    )
    ah_weights = _class_weights(train_data.data["ah_label"]).to(device)
    ch_weights = _class_weights(train_data.data["ch_label"]).to(device)
    losses = {
        "ah": nn.CrossEntropyLoss(weight=ah_weights),
        "ch": nn.CrossEntropyLoss(weight=ch_weights),
    }
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        betas=(
            float(config["train"]["adam_beta1"]),
            float(config["train"]["adam_beta2"]),
        ),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    scheduler = _scheduler(
        optimizer,
        total_steps=max(1, len(train_loader) * epochs),
        warmup_epochs=int(config["train"]["warmup_epochs"]),
        epochs=epochs,
    )
    run_label = run_name or str(config["model"]["variant"])
    run_id = f"{run_label}-{utc_timestamp()}"
    run_dir = Path(config["paths"]["artifact_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = dict(config)
    resolved_config["runtime"] = {
        "run_id": run_id,
        "device": device,
        "ah_class_weights": ah_weights.detach().cpu().tolist(),
        "ch_class_weights": ch_weights.detach().cpu().tolist(),
    }
    write_yaml(run_dir / "config.resolved.yaml", resolved_config)
    summary_path = processed_dir / "summary.json"
    data_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = {
        "run_id": run_id,
        "status": "running",
        "variant": config["model"]["variant"],
        "seed": seed,
        "device": device,
        "config_sha256": hash_value(config),
        "source_tree_sha256": tree_sha256(ETA_ROOT / "src" / "eta"),
        "data_summary": data_summary,
        "environment": environment_manifest(),
        "command_line": __import__("sys").argv,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(run_dir / "run.json", manifest)
    selected_metric = (
        "joint_accuracy"
        if model.heads.completeness is not None
        else "ah_accuracy"
    )
    configured_metric = str(config["train"].get("checkpoint_metric", "auto"))
    if configured_metric not in {"auto", selected_metric}:
        raise ValueError(
            f"checkpoint_metric={configured_metric} is invalid for this variant; "
            f"expected {selected_metric}"
        )
    best_score = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            losses,
            optimizer,
            scheduler,
            float(config["train"]["grad_clip_norm"]),
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            losses,
            None,
            None,
            float(config["train"]["grad_clip_norm"]),
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_ah={validation_metrics['ah_accuracy']:.4f}"
            + (
                f" val_joint={validation_metrics['joint_accuracy']:.4f}"
                if "joint_accuracy" in validation_metrics
                else ""
            ),
            flush=True,
        )
        score = float(validation_metrics[selected_metric])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "model_config": config["model"],
                    "epoch": epoch,
                    "metric": selected_metric,
                    "score": score,
                    "run_id": run_id,
                },
                run_dir / "best.pt",
            )
        with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        manifest["status"] = "training"
        manifest["best_epoch"] = best_epoch
        manifest["best_validation_score"] = best_score
        write_json(run_dir / "run.json", manifest)
    manifest["status"] = "trained"
    manifest["completed_at_utc"] = utc_timestamp()
    write_json(run_dir / "run.json", manifest)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "checkpoint_metric": selected_metric,
    }

"""Track-clustered paired comparisons between saved model predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _load_predictions(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(run_dir) / "evaluation" / "predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["sample_id"]: row for row in csv.DictReader(stream)}


def _bootstrap_delta(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    track_ids: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique_tracks = np.unique(track_ids)
    grouped = {
        track: np.flatnonzero(track_ids == track) for track in unique_tracks
    }
    rng = np.random.default_rng(seed)
    sampled_deltas = []
    for _ in range(max(1, iterations)):
        sampled_tracks = rng.choice(unique_tracks, size=len(unique_tracks), replace=True)
        indices = np.concatenate([grouped[track] for track in sampled_tracks])
        sampled_deltas.append(
            float(
                candidate_correct[indices].mean()
                - reference_correct[indices].mean()
            )
        )
    delta = float(candidate_correct.mean() - reference_correct.mean())
    return {
        "absolute_difference": delta,
        "cluster_bootstrap_95_interval": [
            float(np.percentile(sampled_deltas, 2.5)),
            float(np.percentile(sampled_deltas, 97.5)),
        ],
        "bootstrap_probability_delta_positive": float(
            np.mean(np.asarray(sampled_deltas) > 0)
        ),
        "track_clusters": len(unique_tracks),
    }


def paired_comparison(
    reference_run: str | Path,
    candidate_run: str | Path,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    reference = _load_predictions(reference_run)
    candidate = _load_predictions(candidate_run)
    sample_ids = sorted(set(reference) & set(candidate))
    if not sample_ids:
        raise ValueError("The two prediction files have no shared samples")
    reference_rows = [reference[sample_id] for sample_id in sample_ids]
    candidate_rows = [candidate[sample_id] for sample_id in sample_ids]
    track_ids = np.asarray([row["track_id"] for row in reference_rows])

    def admission_correct(row: dict[str, Any]) -> int:
        return int(float(row["admission_probability"]) >= 0.5) == int(row["ah_label"])

    result: dict[str, Any] = {
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "paired_samples": len(sample_ids),
        "ah_accuracy": _bootstrap_delta(
            np.asarray([admission_correct(row) for row in reference_rows]),
            np.asarray([admission_correct(row) for row in candidate_rows]),
            track_ids,
            iterations,
            seed,
        ),
    }
    reference_has_joint = all(row["completeness_probability"] for row in reference_rows)
    candidate_has_joint = all(row["completeness_probability"] for row in candidate_rows)
    if reference_has_joint and candidate_has_joint:
        def joint_correct(row: dict[str, Any]) -> int:
            ah_ok = admission_correct(row)
            ch_ok = int(float(row["completeness_probability"]) >= 0.5) == int(row["ch_label"])
            return ah_ok and ch_ok

        result["joint_accuracy"] = _bootstrap_delta(
            np.asarray([joint_correct(row) for row in reference_rows]),
            np.asarray([joint_correct(row) for row in candidate_rows]),
            track_ids,
            iterations,
            seed + 1,
        )
    else:
        result["joint_accuracy"] = None
    return result

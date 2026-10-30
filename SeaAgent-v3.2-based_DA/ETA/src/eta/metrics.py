"""Binary decision metrics used by TEA evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have identical shapes")
    predictions = (probabilities >= 0.5).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "count": int(labels.size),
        "accuracy": _safe_div(tp + tn, labels.size),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "brier": float(np.mean((probabilities - labels) ** 2)) if labels.size else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, max(2, int(bins) + 1))
    rows = []
    ece = 0.0
    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else 0.0
        accuracy = float(labels[selected].mean()) if count else 0.0
        gap = abs(confidence - accuracy)
        ece += gap * _safe_div(count, labels.size)
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {"expected_calibration_error": float(ece), "bins": rows}


def stop_metrics(
    admission_probabilities: np.ndarray,
    completeness_probabilities: np.ndarray,
    admission_labels: np.ndarray,
    completeness_labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predicted_stop = (
        (np.asarray(admission_probabilities) >= 0.5)
        & (np.asarray(completeness_probabilities) >= float(threshold))
    )
    target_stop = (
        (np.asarray(admission_labels) == 1)
        & (np.asarray(completeness_labels) == 1)
    )
    tp = int(np.sum(predicted_stop & target_stop))
    tn = int(np.sum(~predicted_stop & ~target_stop))
    fp = int(np.sum(predicted_stop & ~target_stop))
    fn = int(np.sum(~predicted_stop & target_stop))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, target_stop.size),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def bootstrap_accuracy_interval(
    correct: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> list[float]:
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if correct.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, correct.size, size=(max(1, iterations), correct.size))
    estimates = correct[indices].mean(axis=1)
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]


def summarize_predictions(
    admission_probabilities: np.ndarray,
    completeness_probabilities: np.ndarray | None,
    admission_labels: np.ndarray,
    completeness_labels: np.ndarray | None,
    stop_threshold: float,
    calibration_bins: int,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    admission_probabilities = np.asarray(admission_probabilities).reshape(-1)
    admission_labels = np.asarray(admission_labels).reshape(-1)
    result: dict[str, Any] = {
        "ah": binary_metrics(admission_probabilities, admission_labels),
        "ah_calibration": calibration_metrics(
            admission_probabilities,
            admission_labels,
            calibration_bins,
        ),
    }
    result["ah"]["bootstrap_95_accuracy"] = bootstrap_accuracy_interval(
        (admission_probabilities >= 0.5).astype(np.int64) == admission_labels,
        bootstrap_iterations,
        seed,
    )
    if completeness_probabilities is None or completeness_labels is None:
        result["ch"] = None
        result["joint_accuracy"] = None
        result["stop"] = None
        return result
    completeness_probabilities = np.asarray(completeness_probabilities).reshape(-1)
    completeness_labels = np.asarray(completeness_labels).reshape(-1)
    admission_pred = (admission_probabilities >= 0.5).astype(np.int64)
    completeness_pred = (completeness_probabilities >= 0.5).astype(np.int64)
    result["ch"] = binary_metrics(completeness_probabilities, completeness_labels)
    result["ch_calibration"] = calibration_metrics(
        completeness_probabilities,
        completeness_labels,
        calibration_bins,
    )
    result["joint_accuracy"] = float(
        np.mean((admission_pred == admission_labels) & (completeness_pred == completeness_labels))
    )
    joint_correct = (
        (admission_pred == admission_labels)
        & (completeness_pred == completeness_labels)
    )
    result["joint_bootstrap_95_accuracy"] = bootstrap_accuracy_interval(
        joint_correct,
        bootstrap_iterations,
        seed,
    )
    result["ch"]["bootstrap_95_accuracy"] = bootstrap_accuracy_interval(
        (completeness_probabilities >= 0.5).astype(np.int64) == completeness_labels,
        bootstrap_iterations,
        seed,
    )
    result["stop"] = stop_metrics(
        admission_probabilities,
        completeness_probabilities,
        admission_labels,
        completeness_labels,
        stop_threshold,
    )
    return result

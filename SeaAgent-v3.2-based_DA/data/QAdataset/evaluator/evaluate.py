"""Evaluate SeaQA predictions with entity-level EM and mAP."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

RECOGNITION = {"hull_number_lookup", "description_lookup", "image_lookup"}
COUNTING = {"total_count", "registered_count", "unregistered_count"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def truth_entity_ids(question: dict[str, Any]) -> list[str]:
    return [
        str(item["entity_id"])
        for item in question.get("ground_truth", {}).get("entities", [])
        if item.get("entity_id")
    ]


def predicted_entity_ids(prediction: dict[str, Any]) -> list[str]:
    entities = prediction.get("entities") or []
    ranked = sorted(
        (item for item in entities if isinstance(item, dict) and item.get("entity_id")),
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    return [str(item["entity_id"]) for item in ranked]


def average_precision(truth: list[str], prediction: list[str]) -> float:
    truth_set = set(truth)
    if not truth_set:
        return 1.0 if not prediction else 0.0
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for rank, entity_id in enumerate(prediction, 1):
        if entity_id in truth_set and entity_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(entity_id)
    return precision_sum / len(truth_set)


def score_answer(question: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    category = str(question["category"])
    truth = truth_entity_ids(question)
    predicted = predicted_entity_ids(prediction)
    truth_status = "answered" if question["ground_truth"].get("answerable", True) else "unresolved"
    predicted_status = str(prediction.get("status") or "answered")
    row = {
        "id": question["id"],
        "category": category,
        "truth_status": truth_status,
        "predicted_status": predicted_status,
    }
    if truth_status == "unresolved":
        row["correct"] = predicted_status == "unresolved"
        return row
    if predicted_status == "unresolved":
        row.update({"correct": False, "em": 0.0, "ap": 0.0})
        return row
    if category in RECOGNITION:
        correct = set(truth) == set(predicted)
        row.update({"correct": correct, "em": 1.0 if correct else 0.0})
        return row
    if category in COUNTING:
        ap = average_precision(truth, predicted)
        truth_count = int(question["ground_truth"].get("count", len(truth)))
        answer = prediction.get("answer")
        row.update({
            "correct": answer == truth_count,
            "count_accuracy": 1.0 if answer == truth_count else 0.0,
            "ap": ap,
        })
        return row
    row.update({"correct": False, "error": f"unknown category: {category}"})
    return row


def evaluate(dataset: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_by_id = {
        str(item.get("id")): item for item in predictions if item.get("id")
    }
    rows = [
        score_answer(question, prediction_by_id.get(str(question["id"]), {"status": "error"}))
        for question in dataset
    ]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    summary = {}
    for category, values in sorted(by_category.items()):
        scoreable = [row for row in values if row["truth_status"] == "answered"]
        unresolved = [row for row in values if row["truth_status"] == "unresolved"]
        item = {
            "count": len(values),
            "scoreable": len(scoreable),
            "unresolved": len(unresolved),
            "unresolved_accuracy": (
                sum(bool(row["correct"]) for row in unresolved) / len(unresolved)
                if unresolved else None
            ),
        }
        if category in RECOGNITION:
            item["em"] = (
                sum(float(row.get("em", 0.0)) for row in scoreable) / len(scoreable)
                if scoreable else None
            )
        elif category in COUNTING:
            item["map"] = (
                sum(float(row.get("ap", 0.0)) for row in scoreable) / len(scoreable)
                if scoreable else None
            )
            item["count_accuracy"] = (
                sum(float(row.get("count_accuracy", 0.0)) for row in scoreable) / len(scoreable)
                if scoreable else None
            )
        summary[category] = item
    return {
        "questions": len(dataset),
        "predictions": len(predictions),
        "missing_predictions": sum(1 for question in dataset if str(question["id"]) not in prediction_by_id),
        "categories": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SeaQA predictions")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.dataset), read_jsonl(args.predictions))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

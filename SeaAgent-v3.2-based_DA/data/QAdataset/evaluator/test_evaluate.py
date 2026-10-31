from __future__ import annotations

import unittest

from evaluate import evaluate


def question(category: str, entities: list[str], *, count: int | None = None) -> dict:
    truth = {"answerable": True, "entities": [{"entity_id": value} for value in entities]}
    if count is not None:
        truth["count"] = count
    return {"id": "q", "category": category, "ground_truth": truth}


def prediction(entities: list[str], *, count: int | None = None) -> dict:
    value = {
        "id": "q",
        "status": "answered",
        "entities": [{"entity_id": entity_id} for entity_id in entities],
    }
    if count is not None:
        value["answer"] = count
    return value


class EvaluatorTests(unittest.TestCase):
    def test_recognition_uses_entity_sets(self) -> None:
        result = evaluate(
            [question("description_lookup", ["e1", "e2"])],
            [prediction(["e2", "e1", "e1"])],
        )
        self.assertEqual(result["categories"]["description_lookup"]["em"], 1.0)

    def test_count_map_requires_an_exact_entity_set(self) -> None:
        result = evaluate(
            [question("total_count", ["e1", "e2"], count=2)],
            [prediction(["e1"], count=2)],
        )
        category = result["categories"]["total_count"]
        self.assertEqual(category["map"], 0.0)
        self.assertEqual(category["count_accuracy"], 1.0)

    def test_count_answer_is_scored_separately_from_map(self) -> None:
        result = evaluate(
            [question("registered_count", ["e1", "e2"], count=2)],
            [prediction(["e1", "e2"], count=3)],
        )
        category = result["categories"]["registered_count"]
        self.assertEqual(category["map"], 1.0)
        self.assertEqual(category["count_accuracy"], 0.0)

    def test_unresolved_question_requires_unresolved_status(self) -> None:
        item = question("hull_number_lookup", [])
        item["ground_truth"]["answerable"] = False
        result = evaluate([item], [{"id": "q", "status": "unresolved", "entities": []}])
        self.assertEqual(result["categories"]["hull_number_lookup"]["unresolved_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

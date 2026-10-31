from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from evaluate import (
    AgentRun,
    build_agent_payload,
    build_result_rows,
    image_data_url,
    query_agent,
    run_pipeline,
)


def question() -> dict:
    return {
        "id": "SeaQA-0001",
        "category": "hull_number_lookup",
        "question": "舷号 003 的船是否出现过？",
        "time_range": {
            "start": "2026-10-27T16:33:07+08:00",
            "end": "2026-10-27T16:33:18+08:00",
        },
        "inputs": {"hull_number": "003"},
        "ground_truth": {
            "answerable": True,
            "answer": "appears",
            "entities": [{
                "entity_id": "entity-000002",
                "track_ids": ["2"],
                "evidence_keyframe_ids": ["keyframe-1"],
            }],
            "unresolved_entities": [],
            "method": "fixture",
        },
        "provenance": {"source_track_id": "2"},
    }


class DynamicEvaluationTests(unittest.TestCase):
    def test_result_table_has_prediction_then_truth(self) -> None:
        run = AgentRun(
            result={
                "success": True,
                "state": "completed",
                "answerText": "出现过",
                "evidence": {
                    "trackIds": ["2"],
                    "keyframeIds": ["keyframe-1"],
                    "statistics": {"candidateCount": 1},
                },
                "sharedState": {"assessment": {"confidence": "high"}},
            },
            terminal_event="complete",
            error="",
            elapsed_seconds=1.25,
        )
        prediction, truth = build_result_rows(question(), run)
        self.assertEqual(prediction["record_type"], "prediction")
        self.assertEqual(prediction["answer"], "出现过")
        self.assertEqual(prediction["track_ids"], '["2"]')
        self.assertEqual(truth["record_type"], "truth")
        self.assertEqual(truth["answer"], "appears")
        self.assertEqual(truth["entity_ids"], '["entity-000002"]')
        self.assertEqual(truth["track_ids"], '["2"]')

    def test_agent_payload_omits_truth_and_category(self) -> None:
        payload = build_agent_payload(question(), Path("."), "qa-eval-test")
        self.assertEqual(payload["question"], question()["question"])
        self.assertEqual(payload["sessionId"], "qa-eval-test")
        self.assertNotIn("ground_truth", payload)
        self.assertNotIn("category", payload)

    def test_image_question_is_sent_as_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images/query.jpg").write_bytes(b"fake-jpeg")
            item = question()
            item["inputs"] = {"image_path": "images/query.jpg"}
            value = image_data_url(item, root)
            self.assertIsNotNone(value)
            self.assertTrue(value.startswith("data:image/jpeg;base64,"))

    def test_stream_parser_returns_terminal_result(self) -> None:
        expected = {
            "type": "complete",
            "result": {"success": True, "state": "completed", "answerText": "出现过"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/agent/query/stream")
            self.assertEqual(json.loads(request.content)["question"], "舷号 003 的船是否出现过？")
            body = json.dumps({"type": "heartbeat"}) + "\n" + json.dumps(expected) + "\n"
            return httpx.Response(200, text=body)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            event = query_agent(
                client,
                "http://agent.test",
                {"question": "舷号 003 的船是否出现过？", "sessionId": "qa-eval-test"},
            )
        self.assertEqual(event, expected)

    def test_pipeline_writes_prediction_then_truth_rows(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertNotIn("ground_truth", payload)
            event = {
                "type": "complete",
                "result": {
                    "success": True,
                    "state": "completed",
                    "answerText": "出现过",
                    "evidence": {"trackIds": ["2"]},
                },
            }
            return httpx.Response(200, text=json.dumps(event) + "\n")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                summary = run_pipeline(
                    [question()],
                    output,
                    client,
                    "http://agent.test",
                    Path(directory),
                )
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(summary, {"questions": 1, "completed": 1, "failed": 0})
        self.assertEqual([row["record_type"] for row in rows], ["prediction", "truth"])
        self.assertEqual(rows[0]["answer"], "出现过")
        self.assertEqual(rows[1]["answer"], "appears")


if __name__ == "__main__":
    unittest.main()

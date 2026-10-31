from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from evaluate import (
    AgentRun,
    agent_answer,
    build_agent_payload,
    build_qa_row,
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
            "entities": [{"entity_id": "entity-000002", "track_ids": ["2"]}],
        },
        "provenance": {"source_track_id": "2"},
    }


class DynamicEvaluationTests(unittest.TestCase):
    def test_qa_row_contains_only_requested_columns(self) -> None:
        run = AgentRun(
            result={"answer": "出现过", "state": "completed"},
            terminal_event="complete",
            error="",
            elapsed_seconds=1.25,
        )
        self.assertEqual(
            build_qa_row(question(), run),
            {
                "ID": "SeaQA-0001",
                "answer": "出现过",
                "question": "舷号 003 的船是否出现过？",
            },
        )

    def test_answer_uses_projected_answer_text_when_needed(self) -> None:
        run = AgentRun(
            result={"success": True, "state": "completed", "answerText": "出现过"},
            terminal_event="complete",
            error="",
            elapsed_seconds=1.0,
        )
        self.assertEqual(agent_answer(run), "出现过")
        self.assertEqual(
            agent_answer(
                AgentRun(
                    result={
                        "success": True,
                        "state": "completed",
                        "answer": "answer wins",
                        "answerText": "fallback",
                    },
                    terminal_event="complete",
                    error="",
                    elapsed_seconds=1.0,
                )
            ),
            "answer wins",
        )

    def test_failed_run_does_not_turn_error_text_into_answer(self) -> None:
        run = AgentRun(
            result={"success": False, "state": "stalled", "answerText": "internal failure"},
            terminal_event="error",
            error="internal failure",
            elapsed_seconds=1.0,
        )
        self.assertEqual(agent_answer(run), "")

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
            "result": {"success": True, "state": "completed", "answer": "出现过"},
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

    def test_pipeline_writes_exact_header_and_one_row_per_question(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertNotIn("ground_truth", payload)
            event = {
                "type": "complete",
                "result": {
                    "success": True,
                    "state": "completed",
                    "answerText": "出现过",
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
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames
        self.assertEqual(fieldnames, ["ID", "answer", "question"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer"], "出现过")
        self.assertEqual(summary, {"questions": 1, "completed": 1, "failed": 0})


if __name__ == "__main__":
    unittest.main()

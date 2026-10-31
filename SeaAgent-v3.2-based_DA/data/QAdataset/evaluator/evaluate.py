"""Run SeaQA questions through the live SeaAgent and write an ID-answer-question table."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

EVALUATOR_DIR = Path(__file__).resolve().parent
QADATASET_ROOT = EVALUATOR_DIR.parent
DEFAULT_DATASET = QADATASET_ROOT / "dataset/questions.jsonl"
DEFAULT_CONFIG = QADATASET_ROOT / "config.yaml"
DEFAULT_RESULTS_DIR = QADATASET_ROOT / "results"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
TERMINAL_EVENTS = {"complete", "confirm", "error"}
TABLE_FIELDS = ("ID", "answer", "question")


@dataclass(frozen=True)
class AgentRun:
    result: dict[str, Any]
    terminal_event: str
    error: str
    elapsed_seconds: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"config must contain an object: {path}")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def agent_answer(run: AgentRun) -> str:
    """Return a final answer only for a successful Agent completion."""
    result = run.result
    if (
        run.error
        or run.terminal_event != "complete"
        or str(result.get("state") or "") != "completed"
        or result.get("success") is False
    ):
        return ""
    value = result.get("answer")
    if value is None:
        value = result.get("answerText")
    return "" if value is None else str(value).strip()


def build_qa_row(question: dict[str, Any], run: AgentRun) -> dict[str, str]:
    return {
        "ID": str(question.get("id") or ""),
        "answer": agent_answer(run),
        "question": str(question.get("question") or ""),
    }


def image_data_url(question: dict[str, Any], dataset_dir: Path) -> str | None:
    inputs = _mapping(question.get("inputs"))
    image_name = str(inputs.get("image_path") or "").strip()
    if not image_name:
        return None
    root = dataset_dir.resolve()
    source = (root / image_name).resolve()
    if source != root and root not in source.parents:
        raise ValueError(f"image path leaves dataset directory: {image_name}")
    if not source.is_file():
        raise FileNotFoundError(f"question image does not exist: {source}")
    media_type = IMAGE_MEDIA_TYPES.get(source.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported question image type: {source.suffix}")
    content = source.read_bytes()
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"question image exceeds 4 MB: {source}")
    return f"data:{media_type};base64,{base64.b64encode(content).decode('ascii')}"


def build_agent_payload(
    question: dict[str, Any],
    dataset_dir: Path,
    session_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": str(question.get("question") or ""),
        "sessionId": session_id,
    }
    image = image_data_url(question, dataset_dir)
    if image:
        payload["image"] = image
    return payload


def query_agent(
    client: httpx.Client,
    api_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    endpoint = f"{api_url.rstrip('/')}/api/agent/query/stream"
    terminal: dict[str, Any] | None = None
    with client.stream("POST", endpoint, json=payload) as response:
        if response.status_code >= 400:
            body = response.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Agent API returned HTTP {response.status_code}: {body[:500]}"
            )
        for line in response.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise TypeError("Agent stream event must be an object")
            if event.get("type") in TERMINAL_EVENTS:
                terminal = event
                break
    if terminal is None:
        raise RuntimeError("Agent stream ended without a terminal event")
    return terminal


def run_agent_question(
    client: httpx.Client,
    api_url: str,
    question: dict[str, Any],
    dataset_dir: Path,
    session_id: str,
) -> AgentRun:
    started = time.perf_counter()
    try:
        payload = build_agent_payload(question, dataset_dir, session_id)
        event = query_agent(client, api_url, payload)
        result = _mapping(event.get("result"))
        error = ""
        if event.get("type") == "error":
            error = str(event.get("message") or result.get("error") or "Agent run failed")
        return AgentRun(
            result=result,
            terminal_event=str(event.get("type") or ""),
            error=error,
            elapsed_seconds=time.perf_counter() - started,
        )
    except (OSError, TypeError, ValueError, RuntimeError, httpx.HTTPError) as error:
        return AgentRun(
            result={},
            terminal_event="transport_error",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )


def _run_session_id(question_id: Any) -> str:
    safe_id = "".join(char.lower() if char.isalnum() else "-" for char in str(question_id))
    return f"qa-eval-{safe_id.strip('-')}-{uuid.uuid4().hex[:10]}"


def run_pipeline(
    questions: list[dict[str, Any]],
    output: Path,
    client: httpx.Client,
    api_url: str,
    dataset_dir: Path,
    *,
    attempts: int = 2,
    existing_rows: dict[str, dict[str, str]] | None = None,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    reused = 0
    existing_rows = existing_rows or {}
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        handle.flush()
        for index, question in enumerate(questions, 1):
            question_id = str(question.get("id") or "")
            previous = existing_rows.get(question_id)
            if previous and previous.get("answer", "").strip():
                row = {
                    "ID": question_id,
                    "answer": previous["answer"],
                    "question": str(question.get("question") or ""),
                }
                writer.writerow(row)
                handle.flush()
                completed += 1
                reused += 1
                print(
                    f"[{index}/{len(questions)}] {question_id}: reused",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            runs: list[AgentRun] = []
            for attempt in range(1, attempts + 1):
                run = run_agent_question(
                    client,
                    api_url,
                    question,
                    dataset_dir,
                    _run_session_id(question_id),
                )
                runs.append(run)
                row = build_qa_row(question, run)
                if row["answer"]:
                    break
                if attempt < attempts:
                    print(
                        f"[{index}/{len(questions)}] {question_id}: "
                        f"attempt {attempt} had no answer; retrying",
                        file=sys.stderr,
                        flush=True,
                    )
            writer.writerow(row)
            handle.flush()
            state = str(run.result.get("state") or ("error" if run.error else "unknown"))
            if row["answer"]:
                completed += 1
            else:
                failed += 1
            message = (
                f"[{index}/{len(questions)}] {question.get('id')}: "
                f"{state} ({run.elapsed_seconds:.1f}s, attempts={len(runs)})"
            )
            if run.error:
                message += f" error={run.error}"
            print(message, file=sys.stderr, flush=True)
    return {
        "questions": len(questions),
        "completed": completed,
        "failed": failed,
        "reused": reused,
    }


def read_existing_qa(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TABLE_FIELDS:
            raise ValueError(f"existing QA file has an incompatible header: {path}")
        return {
            str(row["ID"]): {
                "ID": str(row["ID"]),
                "answer": str(row.get("answer") or ""),
                "question": str(row.get("question") or ""),
            }
            for row in reader
            if row.get("ID")
        }


def _resolve_dataset_path(value: str | None) -> Path:
    if not value:
        return DEFAULT_DATASET
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _resolve_output_path(value: str | None, default_dir: Path) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    return default_dir.resolve() / f"SeaQA-100-{timestamp}.csv"


def _evaluation_settings(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("evaluation")
    return value if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SeaQA questions through the live SeaAgent and write ID, answer, question"
    )
    parser.add_argument("--dataset", help="questions.jsonl path")
    parser.add_argument("--output", help="CSV QA output path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="QAdataset config path")
    parser.add_argument("--api-url", help="SeaAgent base URL")
    parser.add_argument("--timeout-seconds", type=float, help="Per-question API timeout")
    parser.add_argument("--attempts", type=int, help="Attempts per unanswered question")
    parser.add_argument("--limit", type=int, help="Run only the first N questions")
    parser.add_argument("--resume", action="store_true", help="Retry only empty answers")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    settings = _evaluation_settings(config)
    dataset_path = _resolve_dataset_path(args.dataset)
    if not dataset_path.is_file():
        parser.error(f"dataset file does not exist: {dataset_path}")
    output_dir_value = str(settings.get("output_dir") or DEFAULT_RESULTS_DIR)
    output_dir = Path(output_dir_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = QADATASET_ROOT / output_dir
    output_path = _resolve_output_path(args.output, output_dir)
    if output_path.exists() and not args.overwrite and not args.resume:
        parser.error(
            f"output already exists; pass --resume to fill gaps or --overwrite to replace it: "
            f"{output_path}"
        )
    api_url = str(args.api_url or settings.get("api_url") or "http://127.0.0.1:8000")
    timeout_seconds = float(
        args.timeout_seconds or settings.get("timeout_seconds") or 900
    )
    attempts = int(args.attempts or settings.get("attempts") or 2)
    if timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if attempts <= 0:
        parser.error("--attempts must be positive")

    questions = read_jsonl(dataset_path)
    if args.limit:
        questions = questions[:args.limit]
    dataset_dir = dataset_path.parent.resolve()
    existing_rows = read_existing_qa(output_path) if args.resume else {}

    timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 30.0))
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            try:
                health = client.get(f"{api_url.rstrip('/')}/api/config")
                health.raise_for_status()
            except (OSError, ValueError, httpx.HTTPError) as error:
                print(
                    f"SeaAgent API is unavailable at {api_url}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
                return 2
            summary = run_pipeline(
                questions,
                output_path,
                client,
                api_url,
                dataset_dir,
                attempts=attempts,
                existing_rows=existing_rows,
            )
    except KeyboardInterrupt:
        print(f"\nInterrupted; partial QA file kept at {output_path}", file=sys.stderr)
        return 130

    print(f"qa_file={output_path}")
    print(
        f"questions={summary['questions']} completed={summary['completed']} "
        f"failed={summary['failed']} reused={summary['reused']} "
        f"data_rows={summary['questions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run SeaQA questions through the live SeaAgent API and build a review table."""

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

TABLE_FIELDS = (
    "id",
    "category",
    "question",
    "time_range",
    "inputs",
    "record_type",
    "success",
    "status",
    "answer",
    "count",
    "entity_ids",
    "track_ids",
    "evidence_ids",
    "details",
    "manual_result",
    "manual_notes",
    "elapsed_seconds",
    "error",
)


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


def _json_cell(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _truth_values(question: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    ground_truth = _mapping(question.get("ground_truth"))
    entities = [item for item in ground_truth.get("entities", []) if isinstance(item, dict)]
    entity_ids = _unique([str(item.get("entity_id") or "") for item in entities])
    track_ids: list[str] = []
    evidence_ids: list[str] = []
    for entity in entities:
        track_ids.extend(_string_list(entity.get("track_ids")))
        track_ids.extend(_string_list(entity.get("source_track_id")))
        evidence_ids.extend(_string_list(entity.get("evidence_keyframe_ids")))
        evidence_ids.extend(_string_list(entity.get("source_keyframe_id")))
    return ground_truth, entity_ids, _unique(track_ids), _unique(evidence_ids)


def build_truth_row(question: dict[str, Any]) -> dict[str, Any]:
    ground_truth, entity_ids, track_ids, evidence_ids = _truth_values(question)
    return {
        "id": question.get("id", ""),
        "category": question.get("category", ""),
        "question": question.get("question", ""),
        "time_range": _json_cell(question.get("time_range")),
        "inputs": _json_cell(question.get("inputs")),
        "record_type": "truth",
        "success": "",
        "status": "answerable" if ground_truth.get("answerable", True) else "unresolved",
        "answer": ground_truth.get("answer", ""),
        "count": ground_truth.get("count", ""),
        "entity_ids": _json_cell(entity_ids),
        "track_ids": _json_cell(track_ids),
        "evidence_ids": _json_cell(evidence_ids),
        "details": _json_cell({
            "method": ground_truth.get("method"),
            "entities": ground_truth.get("entities", []),
            "unresolved_entities": ground_truth.get("unresolved_entities", []),
            "provenance": question.get("provenance", {}),
        }),
        "manual_result": "",
        "manual_notes": "",
        "elapsed_seconds": "",
        "error": "",
    }


def build_prediction_row(question: dict[str, Any], run: AgentRun) -> dict[str, Any]:
    result = run.result
    evidence = _mapping(result.get("evidence"))
    enriched = _mapping(result.get("enrichedEvidence"))
    shared_state = _mapping(result.get("sharedState"))
    track_ids: list[str] = []
    evidence_ids: list[str] = []
    for payload in (evidence, enriched):
        track_ids.extend(_string_list(payload.get("trackIds")))
        for key in (
            "keyframeIds",
            "shownKeyframeIds",
            "shipSegmentIds",
            "shownShipSegmentIds",
            "registryReferenceIds",
            "shownRegistryReferenceIds",
        ):
            evidence_ids.extend(_string_list(payload.get(key)))
    statistics = evidence.get("statistics") or enriched.get("statistics", {})
    track_rows = evidence.get("trackRows") or enriched.get("trackRows") or []
    error = run.error or str(result.get("error") or "")
    status = str(result.get("state") or ("error" if error else run.terminal_event or "unknown"))
    success = result.get("success")
    return {
        "id": question.get("id", ""),
        "category": question.get("category", ""),
        "question": question.get("question", ""),
        "time_range": _json_cell(question.get("time_range")),
        "inputs": _json_cell(question.get("inputs")),
        "record_type": "prediction",
        "success": "" if success is None else ("true" if bool(success) else "false"),
        "status": status,
        "answer": result.get("answerText") or result.get("conclusion") or result.get("answer") or "",
        "count": "",
        "entity_ids": "",
        "track_ids": _json_cell(_unique(track_ids)),
        "evidence_ids": _json_cell(_unique(evidence_ids)),
        "details": _json_cell({
            "session_id": result.get("sessionId"),
            "terminal_event": run.terminal_event,
            "partial": result.get("partial"),
            "tool_chain": result.get("toolChain", []),
            "statistics": statistics,
            "track_rows": track_rows[:30],
            "assessment": shared_state.get("assessment", {}),
            "confirmation": result.get("confirmation"),
        }),
        "manual_result": "",
        "manual_notes": "",
        "elapsed_seconds": f"{run.elapsed_seconds:.3f}",
        "error": error,
    }


def build_result_rows(question: dict[str, Any], run: AgentRun) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_prediction_row(question, run), build_truth_row(question)


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
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for index, question in enumerate(questions, 1):
            run = run_agent_question(
                client,
                api_url,
                question,
                dataset_dir,
                _run_session_id(question.get("id")),
            )
            prediction, truth = build_result_rows(question, run)
            writer.writerow(prediction)
            writer.writerow(truth)
            handle.flush()
            state = str(run.result.get("state") or prediction["status"])
            if state == "completed" and not run.error:
                completed += 1
            else:
                failed += 1
            print(
                f"[{index}/{len(questions)}] {question.get('id')}: "
                f"{state} ({run.elapsed_seconds:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
    return {"questions": len(questions), "completed": completed, "failed": failed}


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
        description="Run SeaQA questions through the live SeaAgent and write a human-review table"
    )
    parser.add_argument("--dataset", help="questions.jsonl path")
    parser.add_argument("--output", help="CSV review table path")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="QAdataset config path")
    parser.add_argument("--api-url", help="SeaAgent base URL")
    parser.add_argument("--timeout-seconds", type=float, help="Per-question API timeout")
    parser.add_argument("--limit", type=int, help="Run only the first N questions")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output table")
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
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists; pass --overwrite to replace it: {output_path}")
    api_url = str(args.api_url or settings.get("api_url") or "http://127.0.0.1:8000")
    timeout_seconds = float(
        args.timeout_seconds or settings.get("timeout_seconds") or 900
    )
    if timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    questions = read_jsonl(dataset_path)
    if args.limit:
        questions = questions[:args.limit]
    dataset_dir = dataset_path.parent.resolve()

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
            summary = run_pipeline(questions, output_path, client, api_url, dataset_dir)
    except KeyboardInterrupt:
        print(f"\nInterrupted; partial review table kept at {output_path}", file=sys.stderr)
        return 130

    print(f"review_table={output_path}")
    print(
        f"questions={summary['questions']} completed={summary['completed']} "
        f"failed={summary['failed']} table_rows={summary['questions'] * 2}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

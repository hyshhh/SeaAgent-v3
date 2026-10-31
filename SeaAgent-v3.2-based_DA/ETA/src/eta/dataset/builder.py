"""Build strict ETA samples from three-frame KEP states and later candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..config import load_config, resolve_source_path, seed_everything
from ..utils import sha256_file
from .visual import (
    CandidateObservation,
    CandidateRequest,
    extract_candidate_observations,
    sample_candidate_boxes,
    source_timestamp_bases,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _domain(source_video: str) -> str:
    lower = source_video.replace("\\", "/").lower()
    if "/yz/" in lower:
        return "self_collected"
    if "/open/" in lower or "/gc/" in lower:
        return "public"
    return "unknown"


def _source_id(source_video: str) -> str:
    parts = PurePosixPath(source_video.replace("\\", "/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _allocate_counts(size: int, fractions: dict[str, float]) -> dict[str, int]:
    fractions = {
        split: float(fractions[split]) for split in ("train", "validation", "test")
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("Train/validation/test fractions must sum to 1")
    counts = {
        split: int(np.floor(size * fractions[split]))
        for split in ("train", "validation", "test")
    }
    remaining = size - sum(counts.values())
    priority = sorted(
        ("train", "validation", "test"),
        key=lambda split: (fractions[split] * size) % 1,
        reverse=True,
    )
    for split in priority[:remaining]:
        counts[split] += 1
    if size >= 3:
        for split in ("validation", "test"):
            if counts[split] == 0:
                donor = max(counts, key=counts.get)
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[split] = 1
    return counts


def split_sources(
    domains_by_source: dict[str, str],
    seed: int,
    settings: dict[str, Any],
) -> dict[str, str]:
    """Assign every source video to exactly one split."""

    sources = sorted(domains_by_source)
    if not sources:
        return {}
    mode = str(settings.get("mode", "source_group"))
    if mode == "domain_holdout":
        explicit = {
            "train": set(settings.get("train_domains", ["self_collected"])),
            "validation": set(settings.get("validation_domains", ["public"])),
            "test": set(settings.get("test_domains", ["public"])),
        }
        by_domain: dict[str, list[str]] = defaultdict(list)
        for source in sources:
            by_domain[domains_by_source[source]].append(source)
        assignment: dict[str, str] = {}
        for domain, domain_sources in by_domain.items():
            matches = [
                split
                for split in ("train", "validation", "test")
                if domain in explicit[split]
            ]
            if len(matches) == 1:
                for source in domain_sources:
                    assignment[source] = matches[0]
            elif len(matches) == 2:
                ordered = sorted(
                    domain_sources,
                    key=lambda source: hashlib.sha256(
                        f"{seed}:domain:{source}".encode()
                    ).hexdigest(),
                )
                half = (len(ordered) + 1) // 2
                for source in ordered[:half]:
                    assignment[source] = matches[0]
                for source in ordered[half:]:
                    assignment[source] = matches[1]
            else:
                for source in domain_sources:
                    assignment[source] = "test"
        return assignment
    if mode != "source_group":
        raise ValueError(f"Unknown split mode: {mode}")
    ordered = sorted(
        sources,
        key=lambda source: hashlib.sha256(f"{seed}:{source}".encode()).hexdigest(),
    )
    counts = _allocate_counts(len(ordered), settings)
    assignment: dict[str, str] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        for source in ordered[offset : offset + counts[split]]:
            assignment[source] = split
        offset += counts[split]
    return assignment


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _priority(frame: dict[str, Any]) -> float:
    return float(frame.get("priority", frame.get("quality_score", 0.0)))


def derive_admission_label(
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(history) != 3:
        raise ValueError("ETA admission requires an exactly full KEP")
    quality = float(candidate["quality_score"])
    history_quality = max(float(item["quality_score"]) for item in history)
    candidate["priority"] = 0.7 * quality + 0.3 * history_quality
    novelty = max(
        0.0,
        1.0 - max(_cosine(candidate["vector"], item["vector"]) for item in history),
    )
    admission = int(
        quality >= float(config["data"]["min_quality"])
        and novelty >= float(config["data"]["novelty_threshold"])
    )
    if quality < float(config["data"]["min_quality"]):
        reason = "below_quality_floor"
    elif admission:
        reason = "visual_novelty"
    else:
        reason = "redundant_or_uninformative"
    return {
        "ah_label": admission,
        "novelty": novelty,
        "candidate_priority": candidate["priority"],
        "admission_reason": reason,
    }


def update_history(
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    admitted: bool,
) -> tuple[list[dict[str, Any]], bool]:
    if len(history) != 3:
        raise ValueError("KEP must contain exactly three frames")
    updated = list(history)
    inserted = admitted and float(candidate["priority"]) > min(
        _priority(item) for item in updated
    )
    if inserted:
        lowest = min(updated, key=lambda item: (_priority(item), item["timestamp"]))
        updated.remove(lowest)
        updated.append(candidate)
    return sorted(updated, key=lambda item: float(item["timestamp"])), inserted


def _is_complete(
    history: list[dict[str, Any]],
    candidate_steps: int,
    config: dict[str, Any],
) -> bool:
    settings = config["data"]["completion"]
    if candidate_steps < int(settings["min_candidate_steps"]):
        return False
    readable_hull = any(
        str(item.get("has_readable_hull_number", "")).lower() == "yes"
        for item in history
    )
    high_quality = sum(
        float(item["quality_score"]) >= float(settings["high_quality_threshold"])
        for item in history
    )
    diverse = min(
        _cosine(history[left]["vector"], history[right]["vector"])
        for left in range(len(history))
        for right in range(left + 1, len(history))
    ) <= float(settings["max_min_pair_similarity"])
    quality_coverage = high_quality >= int(settings["min_high_quality_frames"])
    return readable_hull or (quality_coverage and diverse)


def build_track_records(
    track_id: str,
    initial_history: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(initial_history) != 3:
        return []
    history = sorted(initial_history, key=lambda item: float(item["timestamp"]))
    records = []
    for candidate in sorted(observations, key=lambda item: float(item["timestamp"])):
        history_snapshot = list(history)
        labels = derive_admission_label(candidate, history_snapshot, config)
        updated, inserted = update_history(
            history,
            candidate,
            bool(labels["ah_label"]),
        )
        completed = inserted and _is_complete(updated, len(records) + 1, config)
        records.append(
            {
                "track_id": track_id,
                "candidate_vector": candidate["vector"],
                "history_vectors": [item["vector"] for item in history_snapshot],
                "ah_label": labels["ah_label"],
                "ch_label": int(completed),
                "candidate_steps": len(records) + 1,
            }
        )
        history = updated
        if completed:
            break
    return records


def _load_vectors(index_path: Path, ids: list[int]) -> dict[int, np.ndarray]:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("faiss is required to reconstruct keyframe vectors") from error
    index = faiss.read_index(str(index_path))
    vectors: dict[int, np.ndarray] = {}
    for vector_id in dict.fromkeys(ids):
        try:
            vector = np.asarray(index.reconstruct(int(vector_id)), dtype=np.float32)
        except RuntimeError:
            continue
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vectors[int(vector_id)] = vector / norm
    return vectors


def build_dataset(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    seed_everything(int(config["project"]["seed"]))
    capacity = int(config["data"]["history_pool_size"])
    if capacity != 3:
        raise ValueError("This ETA training protocol requires history_pool_size=3")

    frames_path = resolve_source_path(config, config["data"]["keyframes_csv"])
    trajectories_path = resolve_source_path(config, config["data"]["trajectories_dir"])
    index_path = resolve_source_path(config, config["data"]["keyframe_index"])
    frame_rows = _read_csv(frames_path)
    vectors = _load_vectors(
        index_path,
        [
            int(row["keyframe_vector_id"])
            for row in frame_rows
            if row.get("keyframe_vector_id")
        ],
    )
    trajectories = {
        str(payload.get("trackId") or path.stem): payload
        for path in trajectories_path.glob("*.json")
        for payload in (_read_json(path),)
    }
    frames_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        try:
            vector = vectors[int(row["keyframe_vector_id"])]
        except (KeyError, TypeError, ValueError):
            continue
        frames_by_track[str(row["track_id"])].append(
            {
                "timestamp": float(row.get("timestamp") or 0.0),
                "quality_score": float(row.get("quality_score") or 0.0),
                "retention_score": float(row.get("retention_score") or 0.0),
                "priority": float(row.get("retention_score") or 0.0),
                "has_readable_hull_number": row.get("has_readable_hull_number") or "no",
                "vector": vector,
                "frame_id": str(row["keyframe_id"]),
            }
        )
    for frames in frames_by_track.values():
        frames.sort(key=lambda frame: (frame["timestamp"], frame["frame_id"]))

    source_bases = source_timestamp_bases(list(trajectories.values()))
    stride = float(config["data"]["candidate_stride_seconds"])
    requests: list[CandidateRequest] = []
    boxes_by_track: dict[str, list[dict[str, Any]]] = {}
    skipped = {
        "missing_trajectory": len(set(frames_by_track) - set(trajectories)),
        "incomplete_kep": 0,
        "no_post_kep_candidate": 0,
        "missing_video": 0,
    }
    for track_id, trajectory in sorted(trajectories.items()):
        history = frames_by_track.get(track_id, [])
        if len(history) != 3:
            skipped["incomplete_kep"] += 1
            continue
        source = Path(str(trajectory.get("sourceVideoPath") or ""))
        if not source.is_file() or source not in source_bases:
            skipped["missing_video"] += 1
            continue
        selected = sample_candidate_boxes(
            list(trajectory.get("boxes", [])),
            max(float(frame["timestamp"]) for frame in history),
            stride,
        )
        if not selected:
            skipped["no_post_kep_candidate"] += 1
            continue
        boxes_by_track[track_id] = selected
        requests.extend(
            CandidateRequest(
                track_id=track_id,
                source_video=source,
                timestamp=float(box["timestamp"]),
                bbox=tuple(int(value) for value in box["bbox"]),
            )
            for box in selected
        )
    observations, extraction_failures = extract_candidate_observations(
        requests,
        source_bases,
        config,
    )

    records: list[dict[str, Any]] = []
    rejected_tracks = 0
    for track_id in sorted(
        boxes_by_track,
        key=lambda value: int(value) if value.isdigit() else value,
    ):
        observed = [
            observations.get((track_id, float(box["timestamp"])))
            for box in boxes_by_track[track_id]
        ]
        if any(item is None for item in observed):
            rejected_tracks += 1
            continue
        source_video = str(trajectories[track_id].get("sourceVideoPath") or "")
        candidate_frames = [
            {
                "timestamp": item.timestamp,
                "quality_score": item.quality_score,
                "vector": item.vector,
            }
            for item in observed
            if isinstance(item, CandidateObservation)
        ]
        track_records = build_track_records(
            track_id,
            frames_by_track[track_id],
            candidate_frames,
            config,
        )
        for record in track_records:
            record["source_id"] = _source_id(source_video)
            record["domain"] = _domain(source_video)
        records.extend(track_records)
    if not records:
        raise RuntimeError(
            "ETA cleanup produced no samples: every track needs three KEP frames "
            "and at least one later decodable candidate"
        )

    domains_by_source = {
        record["source_id"]: record["domain"] for record in records
    }
    split_by_source = split_sources(
        domains_by_source,
        int(config["project"]["seed"]),
        config["data"]["split"],
    )
    for record in records:
        record["split"] = split_by_source[record["source_id"]]
        if len(record["history_vectors"]) != 3:
            raise RuntimeError(f"Track {record['track_id']} has non-3-frame history")

    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = processed_dir / "manifest.csv"
    if stale_manifest.exists():
        stale_manifest.unlink()
    input_dim = int(config["model"]["input_dim"])
    for split in ("train", "validation", "test"):
        selected = [record for record in records if record["split"] == split]
        candidate = (
            np.stack([record["candidate_vector"] for record in selected]).astype(np.float16)
            if selected
            else np.empty((0, input_dim), dtype=np.float16)
        )
        history = (
            np.stack(
                [
                    np.stack(record["history_vectors"])
                    for record in selected
                ]
            ).astype(np.float16)
            if selected
            else np.empty((0, capacity, input_dim), dtype=np.float16)
        )
        np.savez_compressed(
            processed_dir / f"{split}.npz",
            candidate=candidate,
            history=history,
            history_mask=np.ones((len(selected), capacity), dtype=np.bool_),
            ah_label=np.asarray(
                [record["ah_label"] for record in selected],
                dtype=np.int64,
            ),
            ch_label=np.asarray(
                [record["ch_label"] for record in selected],
                dtype=np.int64,
            ),
        )

    source_splits = {
        split: {
            source for source, assigned in split_by_source.items() if assigned == split
        }
        for split in ("train", "validation", "test")
    }
    if any(
        source_splits[left] & source_splits[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise RuntimeError("Source leakage detected")

    summary = {
        "source_tracks": len(frames_by_track),
        "source_keyframes": len(frame_rows),
        "eligible_tracks": len(boxes_by_track) - rejected_tracks,
        "completed_tracks": sum(
            any(record["ch_label"] == 1 for record in records if record["track_id"] == track_id)
            for track_id in boxes_by_track
        ),
        "incomplete_tracks": sum(
            not any(record["ch_label"] == 1 for record in records if record["track_id"] == track_id)
            for track_id in boxes_by_track
        ),
        "skipped": {**skipped, "candidate_extraction_tracks": rejected_tracks},
        "candidate_stride_seconds": stride,
        "history_pool_size": capacity,
        "history_always_full": True,
        "samples": len(records),
        "splits": {
            split: {
                "samples": sum(record["split"] == split for record in records),
                "tracks": len(
                    {
                        record["track_id"]
                        for record in records
                        if record["split"] == split
                    }
                ),
                "sources": len(source_splits[split]),
                "ah_positive": sum(
                    record["split"] == split and record["ah_label"] == 1
                    for record in records
                ),
                "ch_positive": sum(
                    record["split"] == split and record["ch_label"] == 1
                    for record in records
                ),
            }
            for split in ("train", "validation", "test")
        },
        "label_balance": {
            "ah_label": {
                str(value): sum(record["ah_label"] == value for record in records)
                for value in (0, 1)
            },
            "ch_label": {
                str(value): sum(record["ch_label"] == value for record in records)
                for value in (0, 1)
            },
        },
        "label_rule": {
            "ah": "质量下限，以及当前候选相对三帧 KEP 的余弦新颖度",
            "ch": "证据覆盖达标后的首次有效 KEP 写入；后续观测全部排除",
        },
        "extraction_failures": extraction_failures,
        "hashes": {
            f"{split}_sha256": sha256_file(processed_dir / f"{split}.npz")
            for split in ("train", "validation", "test")
        },
    }
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (processed_dir / "schema.json").write_text(
        json.dumps(
            {
                "candidate": "[N,2048] 当前帧归一化 embedding",
                "history": "[N,3,2048] 三帧已知 KEP embedding",
                "history_mask": "[N,3] 全部为真",
                "ah_label": "[N] 当前候选是否写入 KEP",
                "ch_label": "[N] 是否达到轨迹终止条件",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


class FeatureDataset:
    def __init__(self, npz_path: str | Path):
        self.path = Path(npz_path)
        self.data = np.load(self.path)
        expected = {
            "candidate",
            "history",
            "history_mask",
            "ah_label",
            "ch_label",
        }
        if set(self.data.files) != expected:
            raise ValueError(f"Unexpected ETA dataset keys: {self.data.files}")

    def __len__(self) -> int:
        return int(self.data["candidate"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        return {
            "candidate": torch.from_numpy(self.data["candidate"][index].astype(np.float32)),
            "history": torch.from_numpy(self.data["history"][index].astype(np.float32)),
            "history_mask": torch.from_numpy(self.data["history_mask"][index]),
            "ah_label": torch.tensor(int(self.data["ah_label"][index]), dtype=torch.long),
            "ch_label": torch.tensor(int(self.data["ch_label"][index]), dtype=torch.long),
        }

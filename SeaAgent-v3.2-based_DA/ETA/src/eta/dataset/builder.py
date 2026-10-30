"""Build source-grouped TEA samples from SeaAgent trajectory memory."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..config import load_config, resolve_source_path, seed_everything
from ..utils import sha256_file


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _domain(source_video: str) -> str:
    lower = source_video.replace("\\", "/").lower()
    if "/yz/" in lower:
        return "self_collected"
    if "/open/" in lower or "/gc/" in lower:
        return "public"
    return "unknown"


def _source_id(source_video: str, track_id: str) -> str:
    if not source_video:
        return f"track/{track_id}"
    parts = PurePosixPath(source_video.replace("\\", "/")).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _allocate_counts(size: int, fractions: dict[str, float]) -> dict[str, int]:
    fractions = {
        split: float(fractions[split]) for split in ("train", "validation", "test")
    }
    total = float(sum(fractions.values()))
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train/validation/test fractions must sum to 1")
    counts = {
        split: int(np.floor(size * float(fractions[split])))
        for split in ("train", "validation", "test")
    }
    remaining = size - sum(counts.values())
    priority = sorted(
        ("train", "validation", "test"),
        key=lambda split: (float(fractions[split]) * size) % 1,
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


def history_subsets(
    previous: list[dict[str, Any]],
    capacity: int,
) -> Iterator[list[dict[str, Any]]]:
    maximum = min(capacity, len(previous))
    # Every observation after the initial KEP must be conditioned on real,
    # non-empty history; otherwise each later candidate acquires a trivial label.
    for size in range(0 if not previous else 1, maximum + 1):
        for subset in itertools.combinations(previous, size):
            yield sorted(subset, key=lambda item: float(item["timestamp"]))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _select_pool(
    history: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
    capacity: int,
) -> list[dict[str, Any]]:
    pool = list(history)
    if candidate is not None:
        pool.append(candidate)
    return sorted(
        pool,
        key=lambda item: (
            float(item.get("priority", item.get("quality_score", 0.0))),
            -float(item["timestamp"]),
        ),
        reverse=True,
    )[:capacity]


def derive_proxy_labels(
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Derive deterministic AH/CH weak labels for the available database."""

    settings = config["data"]
    quality = float(candidate.get("quality_score") or 0.0)
    history_quality = max(
        (float(item.get("quality_score") or 0.0) for item in history),
        default=0.0,
    )
    candidate["priority"] = 0.7 * quality + 0.3 * history_quality
    if quality < float(settings["min_quality"]):
        admission = 0
        novelty = 0.0
        admission_reason = "below_quality_floor"
    elif not history:
        admission = 1
        novelty = 1.0
        admission_reason = "pool_initialization"
    else:
        similarity = max(_cosine(candidate["vector"], item["vector"]) for item in history)
        novelty = max(0.0, 1.0 - similarity)
        readable_gain = _truth(candidate.get("has_readable_hull_number")) and not any(
            _truth(item.get("has_readable_hull_number")) for item in history
        )
        admission = int(
            novelty >= float(settings["novelty_threshold"]) or readable_gain
        )
        if readable_gain:
            admission_reason = "new_readable_hull_number"
        elif admission:
            admission_reason = "visual_novelty"
        else:
            admission_reason = "redundant_or_uninformative"

    pool = _select_pool(
        history,
        candidate if admission else None,
        int(settings["history_pool_size"]),
    )
    if not pool:
        complete = 0
        completeness_reason = "empty_pool"
    else:
        hull_evidence = any(_truth(item.get("has_readable_hull_number")) for item in pool)
        similarities = [
            _cosine(pool[left]["vector"], pool[right]["vector"])
            for left in range(len(pool))
            for right in range(left + 1, len(pool))
        ]
        diverse = bool(similarities) and min(similarities) <= float(
            settings["completeness_similarity_max"]
        )
        quality_coverage = len(pool) >= 2 and max(
            float(item["quality_score"]) for item in pool
        ) >= float(settings["quality_coverage_min"])
        complete = int(hull_evidence or diverse or quality_coverage)
        if hull_evidence:
            completeness_reason = "readable_hull_number"
        elif diverse:
            completeness_reason = "visual_view_diversity"
        elif quality_coverage:
            completeness_reason = "multi_frame_quality_coverage"
        else:
            completeness_reason = "insufficient_identity_evidence"
    return {
        "ah_label": admission,
        "ch_label": complete,
        "novelty": novelty,
        "candidate_priority": candidate["priority"],
        "admission_reason": admission_reason,
        "completeness_reason": completeness_reason,
    }


def sample_candidates(
    frames: list[dict[str, Any]],
    stride_seconds: float,
) -> list[dict[str, Any]]:
    if stride_seconds <= 0 or len(frames) <= 1:
        return list(frames)
    base = float(frames[0]["timestamp"])
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        bins[int((float(frame["timestamp"]) - base) // stride_seconds)].append(frame)
    selected = []
    for bin_id in sorted(bins):
        selected.append(
            max(
                bins[bin_id],
                key=lambda item: (
                    float(item["quality_score"]),
                    float(item["retention_score"]),
                    -float(item["timestamp"]),
                ),
            )
        )
    return selected


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
            vector = vector / norm
        vectors[int(vector_id)] = vector
    return vectors


def build_dataset(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    seed_everything(int(config["project"]["seed"]))
    tracks_path = resolve_source_path(config, config["data"]["tracks_csv"])
    frames_path = resolve_source_path(config, config["data"]["keyframes_csv"])
    trajectories_path = resolve_source_path(config, config["data"]["trajectories_dir"])
    index_path = resolve_source_path(config, config["data"]["keyframe_index"])
    tracks = {str(row["track_id"]): row for row in _read_csv(tracks_path)}
    frame_rows = _read_csv(frames_path)
    vector_ids = [
        int(row["keyframe_vector_id"])
        for row in frame_rows
        if row.get("keyframe_vector_id")
    ]
    vectors = _load_vectors(index_path, vector_ids)
    trajectories = {
        str(payload.get("trackId") or path.stem): str(payload.get("sourceVideoPath") or "")
        for path in trajectories_path.glob("*.json")
        for payload in (_read_json(path),)
    }
    frames_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        try:
            vector = vectors[int(row["keyframe_vector_id"])]
        except (KeyError, TypeError, ValueError):
            continue
        track_id = str(row["track_id"])
        source_video = trajectories.get(track_id, "")
        frames_by_track[track_id].append(
            {
                "track_id": track_id,
                "source_video": source_video,
                "source_id": _source_id(source_video, track_id),
                "domain": _domain(source_video),
                "frame_id": str(row["keyframe_id"]),
                "timestamp": float(row.get("timestamp") or 0.0),
                "quality_score": float(row.get("quality_score") or 0.0),
                "retention_score": float(row.get("retention_score") or 0.0),
                "has_readable_hull_number": row.get("has_readable_hull_number") or "no",
                "vector": vector,
            }
        )
    for values in frames_by_track.values():
        values.sort(key=lambda item: (item["timestamp"], item["frame_id"]))
    domains_by_source = {
        item["source_id"]: item["domain"]
        for values in frames_by_track.values()
        for item in values
    }
    split_by_source = split_sources(
        domains_by_source,
        int(config["project"]["seed"]),
        config["data"]["split"],
    )
    capacity = int(config["data"]["history_pool_size"])
    stride_seconds = float(config["data"].get("candidate_stride_seconds", 5.0))
    records: list[dict[str, Any]] = []
    for track_id, all_frames in sorted(
        frames_by_track.items(),
        key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0],
    ):
        track = tracks.get(track_id, {})
        frames = sample_candidates(all_frames, stride_seconds)
        for candidate_index, candidate in enumerate(frames):
            previous = frames[:candidate_index]
            for history in history_subsets(previous, capacity):
                labels = derive_proxy_labels(candidate, history, config)
                records.append(
                    {
                        "sample_id": (
                            f"{track_id}:{candidate['frame_id']}:{len(records):08d}"
                        ),
                        "split": split_by_source[candidate["source_id"]],
                        "track_id": track_id,
                        "source_id": candidate["source_id"],
                        "source_video": candidate["source_video"],
                        "domain": candidate["domain"],
                        "frame_id": candidate["frame_id"],
                        "timestamp": candidate["timestamp"],
                        "quality_score": candidate["quality_score"],
                        "hull_number": str(track.get("final_hull_number") or ""),
                        "history_frame_ids": "|".join(
                            str(item["frame_id"]) for item in history
                        ),
                        "candidate_vector": candidate["vector"],
                        "history_vectors": history,
                        **labels,
                    }
                )
    if not records:
        raise RuntimeError("TEA dataset is empty")
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "sample_id",
        "split",
        "track_id",
        "source_id",
        "source_video",
        "domain",
        "frame_id",
        "timestamp",
        "quality_score",
        "novelty",
        "candidate_priority",
        "ah_label",
        "ch_label",
        "admission_reason",
        "completeness_reason",
        "hull_number",
        "history_frame_ids",
    ]
    with (processed_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in manifest_fields} for record in records)
    for split in ("train", "validation", "test"):
        selected = [record for record in records if record["split"] == split]
        candidate = (
            np.stack([record["candidate_vector"] for record in selected]).astype(np.float16)
            if selected
            else np.empty((0, int(config["model"]["input_dim"])), np.float16)
        )
        history = np.zeros(
            (len(selected), capacity, int(config["model"]["input_dim"])),
            dtype=np.float16,
        )
        history_mask = np.zeros((len(selected), capacity), dtype=np.bool_)
        for index, record in enumerate(selected):
            for slot, item in enumerate(record["history_vectors"][-capacity:]):
                history[index, slot] = item["vector"].astype(np.float16)
                history_mask[index, slot] = True
        np.savez_compressed(
            processed_dir / f"{split}.npz",
            candidate=candidate,
            history=history,
            history_mask=history_mask,
            ah_label=np.asarray([record["ah_label"] for record in selected], dtype=np.int64),
            ch_label=np.asarray([record["ch_label"] for record in selected], dtype=np.int64),
            sample_id=np.asarray([record["sample_id"] for record in selected]),
            track_id=np.asarray([record["track_id"] for record in selected]),
            source_id=np.asarray([record["source_id"] for record in selected]),
        )
    source_splits = {
        split: sorted(source for source, assigned in split_by_source.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    for split, sources in source_splits.items():
        overlap = any(source in sources for other, values in source_splits.items() if other != split for source in values)
        if overlap:
            raise RuntimeError(f"Source leakage detected in split {split}")
    summary = {
        "source_tracks": len(tracks),
        "source_frames": len(frame_rows),
        "reconstructed_vectors": len(vectors),
        "sampled_tracks": len(frames_by_track),
        "sampled_candidate_frames": sum(
            len(sample_candidates(frames, stride_seconds))
            for frames in frames_by_track.values()
        ),
        "candidate_stride_seconds": stride_seconds,
        "samples": len(records),
        "history_pool_size": capacity,
        "splits": {
            split: {
                "samples": sum(record["split"] == split for record in records),
                "tracks": len(
                    {record["track_id"] for record in records if record["split"] == split}
                ),
                "sources": len(source_splits[split]),
            }
            for split in ("train", "validation", "test")
        },
        "domains": {
            domain: sum(record["domain"] == domain for record in records)
            for domain in sorted({record["domain"] for record in records})
        },
        "label_balance": {
            key: {
                str(value): sum(record[key] == value for record in records)
                for value in (0, 1)
            }
            for key in ("ah_label", "ch_label")
        },
        "label_reasons": {
            key: {
                reason: sum(record[key] == reason for record in records)
                for reason in sorted({record[key] for record in records})
            }
            for key in ("admission_reason", "completeness_reason")
        },
        "source_isolation": source_splits,
        "hashes": {
            "manifest_sha256": sha256_file(processed_dir / "manifest.csv"),
            **{
                f"{split}_sha256": sha256_file(processed_dir / f"{split}.npz")
                for split in ("train", "validation", "test")
            },
        },
        "label_protocol": "deterministic proxy labels; not human annotations",
    }
    (processed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (processed_dir / "schema.json").write_text(
        json.dumps(
            {
                "feature_dim": int(config["model"]["input_dim"]),
                "history_pool_size": capacity,
                "candidate": "[N,D] float16 normalized pooled embedding",
                "history": "[N,M,D] float16 padded chronological KEP",
                "history_mask": "[N,M] bool",
                "ah_label": "[N] int64 admission label",
                "ch_label": "[N] int64 completeness label",
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
            "sample_id": str(self.data["sample_id"][index]),
            "track_id": str(self.data["track_id"][index]),
        }

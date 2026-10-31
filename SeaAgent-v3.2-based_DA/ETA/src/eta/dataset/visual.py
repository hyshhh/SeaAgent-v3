"""Extract candidate crops and Qwen embeddings from raw trajectory observations."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateRequest:
    track_id: str
    source_video: Path
    timestamp: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateObservation:
    track_id: str
    timestamp: float
    bbox: tuple[int, int, int, int]
    quality_score: float
    vector: np.ndarray


def source_timestamp_bases(trajectories: list[dict[str, Any]]) -> dict[Path, float]:
    timestamps: dict[Path, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        source = str(trajectory.get("sourceVideoPath") or "")
        if source:
            timestamps[Path(source)].extend(
                float(box["timestamp"]) for box in trajectory.get("boxes", [])
            )
    return {source: min(values) for source, values in timestamps.items() if values}


def sample_candidate_boxes(
    boxes: list[dict[str, Any]],
    history_end: float,
    stride_seconds: float,
) -> list[dict[str, Any]]:
    eligible = sorted(
        (
            box
            for box in boxes
            if float(box.get("timestamp", -1.0)) > float(history_end) + 1e-6
        ),
        key=lambda box: float(box["timestamp"]),
    )
    if not eligible or stride_seconds <= 0:
        return eligible
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for box in eligible:
        offset = float(box["timestamp"]) - float(history_end)
        bins[int(offset // stride_seconds)].append(box)
    selected = []
    for bin_id in sorted(bins):
        center = float(history_end) + (bin_id + 0.5) * stride_seconds
        selected.append(
            min(
                bins[bin_id],
                key=lambda box: abs(float(box["timestamp"]) - center),
            )
        )
    return selected


def _quality_score(
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    settings: dict[str, Any],
) -> float:
    import cv2

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    reference = float(settings.get("sharpness_reference", 1400.0))
    clarity = min(
        1.0,
        float(cv2.Laplacian(gray, cv2.CV_64F).var()) / max(1e-6, reference),
    )
    touches = sum((x1 <= 1, y1 <= 1, x2 >= width - 1, y2 >= height - 1))
    completeness = max(0.0, 1.0 - 0.2 * touches)
    dark = float(np.mean(gray < 25))
    bright = float(np.mean(gray > 230))
    exposure = max(0.0, 1.0 - dark - bright)
    relative_size = float(
        np.sqrt(max(0, (x2 - x1) * (y2 - y1)) / max(1, width * height))
    )
    size_min = float(settings.get("size_min", 0.04))
    size_max = float(settings.get("size_max", 0.35))
    scale = float(
        np.clip(
            (relative_size - size_min) / max(1e-6, size_max - size_min),
            0.0,
            1.0,
        )
    )
    return (
        0.40 * clarity
        + 0.25 * completeness
        + 0.20 * exposure
        + 0.15 * scale
    )


def _cache_key(request: CandidateRequest) -> str:
    stat = request.source_video.stat()
    payload = json.dumps(
        {
            "source": str(request.source_video.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "timestamp": f"{request.timestamp:.9f}",
            "bbox": request.bbox,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class _EmbeddingClient:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.dimension = int(settings.get("dimension", 2048))
        self.local = threading.local()

    def _client(self):
        import httpx

        client = getattr(self.local, "client", None)
        if client is None:
            client = httpx.Client(
                timeout=float(self.settings.get("timeout_seconds", 60)),
                trust_env=False,
            )
            self.local.client = client
        return client

    def embed(self, image_bytes: bytes) -> np.ndarray:
        import httpx

        payload = {
            "model": self.settings["model"],
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Represent the user's input.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + base64.b64encode(image_bytes).decode("ascii")
                                )
                            },
                        }
                    ],
                },
            ],
            "encoding_format": "float",
        }
        api_key = os.getenv(
            str(self.settings.get("api_key_env", "")),
            str(self.settings.get("api_key", "")),
        )
        url = f"{str(self.settings['base_url']).rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client().post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json().get("data") or []
                vector = np.asarray(data[0]["embedding"], dtype=np.float32)
                if vector.shape != (self.dimension,):
                    raise ValueError(
                        f"Embedding dimension {vector.shape}, expected {self.dimension}"
                    )
                if not np.all(np.isfinite(vector)):
                    raise ValueError("Embedding contains non-finite values")
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    raise ValueError("Embedding is zero")
                return vector / norm
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as caught:
                error = caught
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"Qwen embedding request failed: {error}") from error


def extract_candidate_observations(
    requests: list[CandidateRequest],
    source_bases: dict[Path, float],
    config: dict[str, Any],
) -> tuple[dict[tuple[str, float], CandidateObservation], list[dict[str, Any]]]:
    """Decode each crop once, then fetch or reuse its normalized embedding."""

    import cv2

    settings = config["embedding"]
    cache_dir = Path(config["paths"]["embedding_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    def load_cache(request: CandidateRequest) -> np.ndarray | None:
        path = cache_dir / f"{_cache_key(request)}.npy"
        if not path.exists():
            return None
        try:
            vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError):
            return None
        return vector if vector.shape == (int(settings["dimension"]),) else None

    pending: list[tuple[CandidateRequest, bytes, float, np.ndarray | None]] = []
    failures: list[dict[str, Any]] = []
    grouped: dict[Path, list[CandidateRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.source_video].append(request)
    for source, source_requests in grouped.items():
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            failures.extend(
                {"trackId": request.track_id, "reason": "video_open_failed"}
                for request in source_requests
            )
            continue
        base = float(source_bases[source])
        for request in sorted(source_requests, key=lambda item: item.timestamp):
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, request.timestamp - base) * 1000)
            ok, frame = capture.read()
            if not ok:
                failures.append(
                    {"trackId": request.track_id, "reason": "frame_decode_failed"}
                )
                continue
            x1, y1, x2, y2 = request.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                failures.append(
                    {"trackId": request.track_id, "reason": "empty_crop"}
                )
                continue
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                failures.append(
                    {"trackId": request.track_id, "reason": "crop_encode_failed"}
                )
                continue
            pending.append(
                (
                    request,
                    encoded.tobytes(),
                    _quality_score(
                        crop,
                        (x1, y1, x2, y2),
                        frame.shape,
                        config["data"]["candidate_quality"],
                    ),
                    load_cache(request),
                )
            )
        capture.release()

    client = _EmbeddingClient(settings)

    def resolve(item):
        request, image, quality, cached = item
        if cached is None:
            vector = client.embed(image)
            np.save(cache_dir / f"{_cache_key(request)}.npy", vector)
        else:
            vector = cached
        return CandidateObservation(
            track_id=request.track_id,
            timestamp=request.timestamp,
            bbox=request.bbox,
            quality_score=quality,
            vector=vector,
        )

    with ThreadPoolExecutor(max_workers=int(settings.get("request_workers", 4))) as pool:
        observations = list(pool.map(resolve, pending))
    return {
        (observation.track_id, observation.timestamp): observation
        for observation in observations
    }, failures

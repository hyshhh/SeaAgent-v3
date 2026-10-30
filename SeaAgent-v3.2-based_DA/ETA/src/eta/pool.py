"""Online CEP/KEP update and paper-aligned stopping policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoolEntry:
    frame_id: str
    embedding: np.ndarray
    quality_score: float
    priority: float
    timestamp: float


@dataclass(frozen=True)
class PoolDecision:
    admitted: bool
    inserted: bool
    replaced_frame_id: str | None
    stop: bool
    reason: str


class EvidencePool:
    """Capacity-limited KEP with an explicit CEP quality gate."""

    def __init__(
        self,
        capacity: int,
        quality_floor: float = 0.20,
        stop_threshold: float = 0.50,
    ):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.quality_floor = float(quality_floor)
        self.stop_threshold = float(stop_threshold)
        self.entries: list[PoolEntry] = []

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "frame_id": entry.frame_id,
                "quality_score": entry.quality_score,
                "priority": entry.priority,
                "timestamp": entry.timestamp,
            }
            for entry in self.entries
        ]

    def update(
        self,
        candidate: PoolEntry,
        admission: bool,
        completeness_probability: float,
    ) -> PoolDecision:
        if candidate.quality_score < self.quality_floor:
            return PoolDecision(
                admitted=False,
                inserted=False,
                replaced_frame_id=None,
                stop=False,
                reason="rejected_by_cep_quality_floor",
            )
        if not admission:
            return PoolDecision(
                admitted=False,
                inserted=False,
                replaced_frame_id=None,
                stop=False,
                reason="rejected_by_admission_head",
            )
        replaced_frame_id = None
        if len(self.entries) < self.capacity:
            self.entries.append(candidate)
            reason = "initialized_kep"
        else:
            worst = min(self.entries, key=lambda entry: (entry.priority, entry.timestamp))
            if candidate.priority <= worst.priority:
                return PoolDecision(
                    admitted=True,
                    inserted=False,
                    replaced_frame_id=None,
                    stop=False,
                    reason="candidate_lower_than_pool_priority",
                )
            replaced_frame_id = worst.frame_id
            self.entries.remove(worst)
            self.entries.append(candidate)
            reason = "replaced_lowest_priority"
        self.entries.sort(key=lambda entry: (entry.timestamp, entry.frame_id))
        stop = float(completeness_probability) >= self.stop_threshold
        return PoolDecision(
            admitted=True,
            inserted=True,
            replaced_frame_id=replaced_frame_id,
            stop=stop,
            reason=reason if not stop else f"{reason}_and_completeness_threshold_met",
        )

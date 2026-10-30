import numpy as np

from eta.pool import EvidencePool, PoolEntry


def _entry(frame_id: str, priority: float, quality: float = 0.8) -> PoolEntry:
    return PoolEntry(
        frame_id=frame_id,
        embedding=np.asarray([priority, 1.0 - priority]),
        quality_score=quality,
        priority=priority,
        timestamp=priority,
    )


def test_pool_rejects_below_quality_floor():
    pool = EvidencePool(capacity=2, quality_floor=0.2)
    decision = pool.update(_entry("low", 0.9, quality=0.1), True, 1.0)
    assert not decision.inserted
    assert decision.reason == "rejected_by_cep_quality_floor"


def test_pool_fills_and_replaces_lowest_priority():
    pool = EvidencePool(capacity=2)
    assert pool.update(_entry("a", 0.3), True, 0.0).inserted
    assert pool.update(_entry("b", 0.4), True, 0.0).inserted
    decision = pool.update(_entry("c", 0.8), True, 0.9)
    assert decision.inserted
    assert decision.replaced_frame_id == "a"
    assert decision.stop
    assert {entry.frame_id for entry in pool.entries} == {"b", "c"}


def test_stop_never_fires_without_actual_pool_change():
    pool = EvidencePool(capacity=1, stop_threshold=0.5)
    pool.update(_entry("best", 0.9), True, 0.0)
    rejected = pool.update(_entry("lower", 0.2), True, 1.0)
    assert rejected.admitted
    assert not rejected.inserted
    assert not rejected.stop

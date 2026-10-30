import numpy as np

from eta.config import load_config
from eta.dataset.builder import (
    derive_proxy_labels,
    history_subsets,
    sample_candidates,
    split_sources,
)


def _frame(index: int, timestamp: float, vector: list[float], quality: float = 0.8):
    return {
        "frame_id": f"f{index}",
        "timestamp": timestamp,
        "quality_score": quality,
        "retention_score": quality,
        "has_readable_hull_number": "no",
        "vector": np.asarray(vector, dtype=np.float32),
    }


def test_history_subsets_are_chronological_and_capacity_limited():
    previous = [_frame(1, 2.0, [1, 0]), _frame(2, 1.0, [0, 1]), _frame(3, 3.0, [1, 1])]
    subsets = list(history_subsets(previous, 2))
    assert len(subsets) == 6
    assert all(len(subset) <= 2 for subset in subsets)
    assert all(
        [item["timestamp"] for item in subset]
        == sorted(item["timestamp"] for item in subset)
        for subset in subsets
    )


def test_source_group_split_has_no_leakage():
    domains = {f"Open/{index}.mp4": "public" for index in range(20)}
    settings = {"mode": "source_group", "train": 0.7, "validation": 0.15, "test": 0.15}
    assignment = split_sources(domains, 42, settings)
    assert set(assignment) == set(domains)
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert len(assignment) == 20


def test_domain_holdout_splits_shared_holdout_domain_once():
    domains = {
        "YZ/private.mp4": "self_collected",
        **{f"Open/{index}.mp4": "public" for index in range(8)},
    }
    settings = {"mode": "domain_holdout"}
    assignment = split_sources(domains, 42, settings)
    assert assignment["YZ/private.mp4"] == "train"
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert len(set(assignment)) == 9


def test_five_second_sampling_keeps_one_high_quality_frame_per_bin():
    frames = [
        _frame(1, 0.5, [1, 0], 0.5),
        _frame(2, 1.0, [1, 0], 0.9),
        _frame(3, 5.5, [0, 1], 0.7),
    ]
    selected = sample_candidates(frames, 5.0)
    assert [frame["frame_id"] for frame in selected] == ["f2", "f3"]


def test_proxy_labels_follow_quality_novelty_and_pool_update():
    config = load_config()
    candidate = _frame(2, 1.0, [-1.0, 0.0], quality=0.8)
    history = [_frame(1, 0.0, [1.0, 0.0])]
    labels = derive_proxy_labels(candidate, history, config)
    assert labels["ah_label"] == 1
    assert labels["ch_label"] == 1
    low_quality = _frame(3, 2.0, [0.0, 1.0], quality=0.0)
    rejected = derive_proxy_labels(low_quality, history, config)
    assert rejected["ah_label"] == 0
    assert rejected["admission_reason"] == "below_quality_floor"

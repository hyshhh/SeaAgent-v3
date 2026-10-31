import numpy as np

from eta.config import load_config
from eta.dataset.builder import (
    FeatureDataset,
    build_track_records,
    derive_admission_label,
    split_sources,
)
from eta.dataset.visual import sample_candidate_boxes


def _frame(index: int, timestamp: float, vector: list[float], quality: float = 0.8):
    return {
        "frame_id": f"f{index}",
        "timestamp": timestamp,
        "quality_score": quality,
        "retention_score": quality,
        "priority": quality,
        "vector": np.asarray(vector, dtype=np.float32),
    }


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


def test_candidate_sampling_drops_observations_before_full_kep():
    boxes = [
        {"timestamp": 0.5},
        {"timestamp": 1.0},
        {"timestamp": 1.5},
        {"timestamp": 2.1},
        {"timestamp": 3.4},
    ]
    selected = sample_candidate_boxes(boxes, history_end=1.5, stride_seconds=1.0)
    assert [box["timestamp"] for box in selected] == [2.1, 3.4]


def test_labels_stop_at_first_complete_insertion_and_truncate_future_data():
    config = load_config()
    history = [
        _frame(1, 0.0, [1.0, 0.0]),
        _frame(2, 0.5, [0.0, 1.0]),
        _frame(3, 1.0, [-1.0, 0.0]),
    ]
    observations = [
        _frame(4, 2.0, [0.0, -1.0], quality=0.95),
        _frame(5, 3.0, [-0.7, 0.7], quality=0.95),
        _frame(6, 4.0, [1.0, 0.0], quality=0.95),
    ]
    records = build_track_records("12", history, observations, config)
    assert len(records) == 2
    assert [record["ch_label"] for record in records] == [0, 1]
    assert all(len(record["history_vectors"]) == 3 for record in records)
    low_quality = _frame(7, 5.0, [1.0, 0.0], quality=0.1)
    labels = derive_admission_label(low_quality, history, config)
    assert labels["ah_label"] == 0
    assert labels["admission_reason"] == "below_quality_floor"


def test_feature_dataset_has_only_five_training_arrays(tmp_path):
    path = tmp_path / "train.npz"
    np.savez_compressed(
        path,
        candidate=np.zeros((1, 2048), dtype=np.float16),
        history=np.zeros((1, 3, 2048), dtype=np.float16),
        history_mask=np.ones((1, 3), dtype=np.bool_),
        ah_label=np.zeros(1, dtype=np.int64),
        ch_label=np.ones(1, dtype=np.int64),
    )
    data = FeatureDataset(path)
    assert set(data.data.files) == {
        "candidate",
        "history",
        "history_mask",
        "ah_label",
        "ch_label",
    }

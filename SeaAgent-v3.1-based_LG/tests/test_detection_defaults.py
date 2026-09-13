import inspect

from config import load_config
from pipeline.cli import _merge_args_to_config, build_parser
from pipeline.detector import ShipDetector


def test_detection_threshold_defaults_are_consistent():
    config = load_config()
    signature = inspect.signature(ShipDetector.__init__)

    assert config["yolo"]["confidence"] == 0.5
    assert config["yolo"]["iou"] == 0.5
    assert "tracking_candidate_confidence" not in config["yolo"]
    assert "tracking_candidate_confidence" not in config["pipeline"]
    assert config["pipeline"]["conf_threshold"] == 0.5
    assert config["pipeline"]["iou_threshold"] == 0.5
    assert signature.parameters["conf_threshold"].default == 0.5
    assert "tracking_candidate_confidence" not in signature.parameters
    assert signature.parameters["iou_threshold"].default == 0.5


def test_tracking_defaults_are_consistent():
    config = load_config()
    tracker = config["yolo"]["tracker_params"]
    appearance = config["yolo"]["appearance_tracking"]

    assert tracker["track_high_thresh"] == 0.5
    assert tracker["track_low_thresh"] == 0.2
    assert tracker["new_track_thresh"] == 0.6
    assert tracker["track_buffer"] == 90
    assert tracker["match_thresh"] == 0.8
    assert tracker["fuse_score"] is True
    assert config["pipeline"]["max_stale_frames"] == 120
    assert config["pipeline"]["appearance_tracking"] == appearance
    assert appearance["enabled"] is False
    assert appearance["appearance_thresh"] == 0.8
    assert appearance["proximity_thresh"] == 0.5


def test_tracking_parameters_are_cli_overrides():
    args = build_parser().parse_args([
        "demo.mp4",
        "--conf", "0.1",
        "--track-high-thresh", "0.45",
        "--track-low-thresh", "0.15",
        "--new-track-thresh", "0.6",
        "--match-thresh", "0.78",
        "--track-buffer", "90",
        "--max-stale-frames", "120",
    ])
    config = _merge_args_to_config(args, {"pipeline": {"tracker_params": {}}})

    assert config["pipeline"]["conf_threshold"] == 0.1
    assert config["pipeline"]["tracker_params"] == {
        "track_high_thresh": 0.45,
        "track_low_thresh": 0.15,
        "new_track_thresh": 0.6,
        "match_thresh": 0.78,
        "track_buffer": 90,
    }
    assert config["pipeline"]["max_stale_frames"] == 120

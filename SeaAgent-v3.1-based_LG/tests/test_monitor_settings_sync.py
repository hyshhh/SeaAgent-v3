from __future__ import annotations

from config import load_config
from config import settings as settings_module


def test_monitoring_settings_specs_include_model_and_pipeline_controls():
    config = load_config()
    public = settings_module.public_settings(config)

    assert public["settings"]["yolo"]["model"] == "yolov8n.pt"
    assert "yolo11m.pt" in public["specs"]["yolo.model"]["choices"]
    assert public["settings"]["pipeline"]["pipe_scale"] == 0.25
    assert public["settings"]["pipeline"]["max_frames"] == 0
    assert public["settings"]["pipeline"]["save_output_video"] is True


def test_monitoring_settings_persist_model_device_and_stream_values(tmp_path, monkeypatch):
    runtime_file = tmp_path / "runtime.yaml"
    monkeypatch.setattr(settings_module, "RUNTIME_FILE", runtime_file)
    config = load_config()

    result = settings_module.update_settings(config, {
        "settings": {
            "yolo": {"model": "yolov8s.pt", "device": "0"},
            "pipeline": {
                "target_fps": 12,
                "pipe_scale": 0.05,
                "max_frames": 500,
                "save_output_video": False,
            },
        },
    })

    assert result["settings"]["yolo"]["model"] == "yolov8s.pt"
    assert result["settings"]["yolo"]["device"] == "0"
    assert result["settings"]["pipeline"]["pipe_scale"] == 0.05
    assert runtime_file.exists()


def test_monitoring_settings_persist_tracking_controls(tmp_path, monkeypatch):
    runtime_file = tmp_path / "runtime.yaml"
    monkeypatch.setattr(settings_module, "RUNTIME_FILE", runtime_file)
    config = load_config()

    result = settings_module.update_settings(config, {
        "settings": {
            "yolo": {
                "confidence": 0.1,
                "tracker_params": {
                    "track_high_thresh": 0.45,
                    "track_low_thresh": 0.15,
                    "new_track_thresh": 0.6,
                    "match_thresh": 0.78,
                    "track_buffer": 90,
                },
            },
            "pipeline": {"max_stale_frames": 120},
        },
    })

    assert result["settings"]["yolo"]["confidence"] == 0.1
    assert result["settings"]["yolo"]["tracker_params"]["track_low_thresh"] == 0.15
    assert result["settings"]["yolo"]["tracker_params"]["match_thresh"] == 0.78
    assert result["settings"]["pipeline"]["max_stale_frames"] == 120
    assert "tracking_candidate_confidence" not in result["specs"]

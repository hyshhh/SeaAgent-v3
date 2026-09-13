from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from pipeline.cli import build_parser
from pipeline.pipeline import ShipPipeline


def test_playlist_cli_arguments_are_parsed():
    args = build_parser().parse_args([
        "first.mp4",
        "--playlist-json",
        '["first.mp4", "second.mp4"]',
        "--segment-gap-seconds",
        "3.5",
        "--playlist-failure-policy",
        "stop",
    ])

    assert args.source == "first.mp4"
    assert args.playlist_json == '["first.mp4", "second.mp4"]'
    assert args.segment_gap_seconds == 3.5
    assert args.playlist_failure_policy == "stop"


def test_one_pipeline_instance_processes_all_playlist_segments(monkeypatch):
    opened_sources: list[str] = []

    class FakeInputSource:
        def __init__(self, source):
            self.source = str(source)
            self.source_fps = 10.0
            self.width = 32
            self.height = 24
            self.frames = 0
            opened_sources.append(self.source)

        def read(self):
            if self.frames >= 2:
                return False, None
            self.frames += 1
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            return None

    class FakeDetector:
        def __init__(self):
            self.reset_count = 0
            self.cleanup_count = 0

        def detect(self, frame, frame_id):
            return []

        def reset_tracking(self):
            self.reset_count += 1

        def cleanup(self):
            self.cleanup_count += 1

    class FakeMemory:
        def __init__(self):
            self.active = {}
            self.trace = []
            self.finalize_active_count = 0
            self.finalize_all_count = 0

        def observe(self, *args, **kwargs):
            return None

        def display_tracks(self):
            return {}

        def finalize_active(self):
            self.finalize_active_count += 1

        def finalize_all(self):
            self.finalize_all_count += 1

    class FakeFps:
        def tick(self, name):
            return None

        def get_all_fps(self):
            return {}

    class FakeLatency:
        def measure(self, name):
            return nullcontext()

    monkeypatch.setattr("pipeline.pipeline.InputSource", FakeInputSource)
    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._config = {
        "pipeline": {"stream_write_every_n_frames": 2, "stream_jpeg_quality": 65},
        "demo_video": {"output_dir": "output"},
    }
    pipeline._target_fps = 0.0
    pipeline._monitor_start_time = 1000.0
    pipeline._detect_every_n = 1
    pipeline._demo_enabled = False
    pipeline._save_output_video = False
    pipeline._pipe_output_size = None
    pipeline._stop_file = None
    pipeline._raw_stdout = False
    pipeline._no_output = True
    pipeline._detector = FakeDetector()
    pipeline._memory = FakeMemory()
    pipeline._renderer = None
    pipeline._fps = FakeFps()
    pipeline._latency = FakeLatency()
    events = []

    stats = pipeline.process_playlist(
        ["first.mp4", "second.mp4"],
        segment_gap_seconds=1.0,
        segment_callback=events.append,
    )

    assert opened_sources == ["first.mp4", "second.mp4"]
    assert stats["playlist_total"] == 2
    assert stats["playlist_completed"] == 2
    assert stats["total_frames"] == 4
    assert stats["video_duration_seconds"] == 1.4
    assert pipeline._detector.reset_count == 2
    assert pipeline._detector.cleanup_count == 1
    assert pipeline._memory.finalize_active_count == 2
    assert pipeline._memory.finalize_all_count == 1
    assert [event["status"] for event in events] == ["running", "completed", "running", "completed"]

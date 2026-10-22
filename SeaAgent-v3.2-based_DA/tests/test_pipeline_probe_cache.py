import os
from types import SimpleNamespace

from web.routes import pipeline_api


def _reset_probe_cache():
    with pipeline_api._probe_cache_lock:
        pipeline_api._probe_cache.clear()
        pipeline_api._probe_inflight.clear()


def test_probe_cache_reuses_results_for_unchanged_file(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = "h264\n" if any("codec_name" in arg for arg in command) else "1920\n1080\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    _reset_probe_cache()
    monkeypatch.setattr(pipeline_api, "_FFPROBE", "ffprobe")
    monkeypatch.setattr(pipeline_api.subprocess, "run", fake_run)
    try:
        assert pipeline_api._probe_codec(str(video)) == "h264"
        assert pipeline_api._probe_codec(str(video)) == "h264"
        assert pipeline_api._probe_video_size(str(video)) == (1920, 1080)
        assert pipeline_api._probe_video_size(str(video)) == (1920, 1080)
        assert len(calls) == 2
    finally:
        _reset_probe_cache()


def test_probe_cache_invalidates_when_file_metadata_changes(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="h264\n", stderr="")

    _reset_probe_cache()
    monkeypatch.setattr(pipeline_api, "_FFPROBE", "ffprobe")
    monkeypatch.setattr(pipeline_api.subprocess, "run", fake_run)
    try:
        assert pipeline_api._probe_codec(str(video)) == "h264"
        original_stat = video.stat()
        os.utime(video, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))

        assert pipeline_api._probe_codec(str(video)) == "h264"
        video.write_bytes(b"larger video")
        assert pipeline_api._probe_codec(str(video)) == "h264"
        assert len(calls) == 3
    finally:
        _reset_probe_cache()


def test_probe_cache_evicts_least_recently_used_entry(tmp_path, monkeypatch):
    videos = [tmp_path / f"{name}.mp4" for name in ("a", "b", "c")]
    for video in videos:
        video.write_bytes(b"video")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command[-1])
        return SimpleNamespace(returncode=0, stdout="h264\n", stderr="")

    _reset_probe_cache()
    monkeypatch.setattr(pipeline_api, "_PROBE_CACHE_CAPACITY", 2)
    monkeypatch.setattr(pipeline_api, "_FFPROBE", "ffprobe")
    monkeypatch.setattr(pipeline_api.subprocess, "run", fake_run)
    try:
        for video in videos[:2]:
            assert pipeline_api._probe_codec(str(video)) == "h264"
        assert pipeline_api._probe_codec(str(videos[0])) == "h264"  # Keep a most-recently used.
        assert pipeline_api._probe_codec(str(videos[2])) == "h264"  # Evicts b.
        assert pipeline_api._probe_codec(str(videos[1])) == "h264"
        assert len(calls) == 4
    finally:
        _reset_probe_cache()

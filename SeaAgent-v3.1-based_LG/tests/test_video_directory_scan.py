from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

from web.routes import pipeline_api


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def test_pipe_output_size_supports_point_zero_five():
    assert pipeline_api._get_pipe_output_size(1080, 1920, 0.05) == (54, 96)


def test_scan_video_files_includes_nested_videos_and_skips_cache(tmp_path: Path):
    for relative_path in (
        "root.mp4",
        "mission_a/clip.avi",
        "mission_b/deep/clip.mkv",
        "mission_b/note.txt",
        ".hidden/ignored.mov",
        "mission_a/_transcoded/cached.mp4",
    ):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"video")

    videos = pipeline_api._scan_video_files(tmp_path, VIDEO_EXTENSIONS)

    assert [video["filename"] for video in videos] == [
        "mission_a/clip.avi",
        "mission_b/deep/clip.mkv",
        "root.mp4",
    ]
    assert videos[0]["name"] == "clip.avi"
    assert videos[0]["relative_path"] == "mission_a/clip.avi"
    assert videos[0]["directory"] == "mission_a"
    assert videos[1]["directory"] == "mission_b"
    assert videos[2]["directory"] == ""

    assert pipeline_api._build_video_directories(videos) == [
        {"path": "", "name": "根目录", "count": 1},
        {"path": "mission_a", "name": "mission_a", "count": 1},
        {"path": "mission_b", "name": "mission_b", "count": 1},
    ]


def test_resolve_demo_video_path_allows_nested_relative_path_and_rejects_escape(tmp_path: Path, monkeypatch):
    nested_video = tmp_path / "mission_a" / "clip.mp4"
    nested_video.parent.mkdir(parents=True)
    nested_video.write_bytes(b"video")
    monkeypatch.setattr(pipeline_api, "_get_demo_dir", lambda: tmp_path)

    assert pipeline_api._resolve_demo_video_path("mission_a/clip.mp4") == nested_video.resolve()
    assert pipeline_api._resolve_demo_video_path(r"mission_a\clip.mp4") == nested_video.resolve()

    with pytest.raises(HTTPException) as exc_info:
        pipeline_api._resolve_demo_video_path("../outside.mp4")

    assert exc_info.value.status_code == 400

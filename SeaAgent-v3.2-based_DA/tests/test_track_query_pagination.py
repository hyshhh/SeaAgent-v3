from datetime import datetime

from memory import MemoryRepository
from tools.service import ToolService


def _repository(tmp_path):
    names = ("tracks", "keyframes", "registry", "registry_images", "qa_sessions")
    paths = {f"{name}_csv": str(tmp_path / f"{name}.csv") for name in names}
    return MemoryRepository({"paths": paths})


def _track(track_id, start, end, hull=None):
    return {
        "track_id": track_id,
        "start_time": start,
        "end_time": end,
        "final_hull_number": hull,
    }


def test_track_pages_are_stable_and_include_unrecognized_tracks(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert_track(_track("t-c", 20, 30))
    repository.upsert_track(_track("t-b", 20, 30, "320"))
    repository.upsert_track(_track("t-a", 10, 30, "320"))

    first, has_more = repository.find_tracks_page(limit=2)
    cursor = (first[-1]["startTime"], first[-1]["trackId"])
    second, second_has_more = repository.find_tracks_page(after=cursor, limit=2)

    assert [item["trackId"] for item in first] == ["t-a", "t-b"]
    assert [item["trackId"] for item in second] == ["t-c"]
    assert has_more is True
    assert second_has_more is False


def test_track_page_applies_time_overlap_and_hull_filters(tmp_path):
    repository = _repository(tmp_path)
    repository.upsert_track(_track("inside", 100, 200, "320"))
    repository.upsert_track(_track("outside", 201, 300, "320"))
    repository.upsert_track(_track("unknown", 100, 200))

    page, has_more = repository.find_tracks_page((150, 160), hull_number="320")
    all_tracks, all_has_more = repository.find_tracks_page((150, 160))

    assert [item["trackId"] for item in page] == ["inside"]
    assert not has_more
    assert {item["trackId"] for item in all_tracks} == {"inside", "unknown"}
    assert not all_has_more


def test_scan_tracks_reads_every_page_without_returning_pages_to_the_agent():
    class Repository:
        def __init__(self):
            self.cursors = []

        def find_tracks_page(self, time_range, hull_number, match_type, after, limit):
            self.cursors.append(after)
            assert time_range is None
            assert hull_number is None
            assert limit == 100
            start = 0 if after is None else int(after[1].split("-")[1]) + 1
            end = min(start + 100, 205)
            page = [{"trackId": f"t-{index}", "startTime": index} for index in range(start, end)]
            return page, end < 205

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    tracks, complete = service._scan_tracks(None)

    assert len(tracks) == 205
    assert complete is True
    assert service.repository.cursors == [None, (99.0, "t-99"), (199.0, "t-199")]


def test_track_record_formats_monitor_and_video_times_without_losing_zero():
    track = MemoryRepository._track_record({
        "track_id": "t-1",
        "start_time": "1792634400",
        "end_time": "1792634460",
        "video_start_time": "0",
        "video_end_time": "60",
    })

    assert track["startTimeText"] == "2026-10-22T10:00:00+08:00"
    assert track["endTimeText"] == "2026-10-22T10:01:00+08:00"
    assert track["videoStartTime"] == 0
    assert track["videoStartTimeText"] == "Video 00:00:00"
    assert track["videoEndTimeText"] == "Video 00:01:00"


def test_explicit_range_formats_local_iso_time():
    service = ToolService.__new__(ToolService)
    scope, error = service._validate_time_range(
        ("2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00")
    )
    assert not error
    assert scope == (
        datetime.fromisoformat("2026-10-22T10:00:00+08:00").timestamp(),
        datetime.fromisoformat("2026-10-22T11:00:00+08:00").timestamp(),
    )

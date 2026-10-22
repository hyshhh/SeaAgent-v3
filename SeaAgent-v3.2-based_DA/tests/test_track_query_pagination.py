from datetime import datetime

from memory import MemoryRepository
from tools.service import ToolService


def _repository(tmp_path):
    names = (
        "tracks", "keyframes", "registry", "registry_images",
        "qa_sessions", "qa_rounds", "qa_evidence",
    )
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
    assert has_more is False
    assert {item["trackId"] for item in all_tracks} == {"inside", "unknown"}
    assert all_has_more is False


def test_get_track_omitted_time_means_all_and_uses_bounded_pages():
    class Repository:
        def find_tracks_page(self, time_range, hull_number, match_type, after, limit):
            assert time_range is None
            assert after is None
            assert limit == 50
            return ([{"trackId": f"t-{index}", "startTime": index} for index in range(50)], True)

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    result = service.getTrack()

    assert result["queryScope"] is None
    assert result["hasMore"] is True
    assert result["nextCursor"] == {"afterStartTime": 49, "afterTrackId": "t-49"}
    assert len(result["tracks"]) == 50
    assert "trackIds" not in result


def test_zero_limit_uses_default_instead_of_returning_all():
    class Repository:
        def find_tracks_page(self, time_range, hull_number, match_type, after, limit):
            assert limit == 50
            return ([], False)

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    result = service.getTrack(limit=0)

    assert result["ok"] is True
    assert result["hasMore"] is False


def test_page_size_is_capped():
    class Repository:
        def find_tracks_page(self, time_range, hull_number, match_type, after, limit):
            assert limit == 100
            return ([], False)

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    assert service.getTrack(limit=1000)["ok"] is True


def test_explicit_range_is_returned_as_readable_local_iso_time():
    class Repository:
        def find_tracks_page(self, time_range, hull_number, match_type, after, limit):
            expected = (
                datetime.fromisoformat("2026-10-22T10:00:00+08:00").timestamp(),
                datetime.fromisoformat("2026-10-22T11:00:00+08:00").timestamp(),
            )
            assert time_range == expected
            return ([], False)

    service = ToolService.__new__(ToolService)
    service.repository = Repository()
    result = service.getTrack(timeRange=("2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"))

    assert result["queryScope"] == ["2026-10-22T10:00:00+08:00", "2026-10-22T11:00:00+08:00"]


def test_invalid_pagination_is_not_reported_as_an_empty_search():
    service = ToolService.__new__(ToolService)
    result = service.getTrack(afterStartTime=1)

    assert result["ok"] is False
    assert result["error"] == "invalid_pagination"
    assert "tracks" not in result

import asyncio
import base64

from web.routes import pipeline_api


def test_pool_status_groups_frames_by_track_and_emits_refresh_captions():
    task_id = "pool-test"
    pipeline_api._init_pool_rows(task_id)
    preview = "data:image/jpeg;base64," + base64.b64encode(b"jpeg").decode("ascii")

    pipeline_api._record_pool_event(task_id, {
        "pool": "candidate", "action": "upsert", "recordId": "c1",
        "trackId": "12", "image": preview, "status": "识别中",
    })
    pipeline_api._record_pool_event(task_id, {
        "pool": "candidate", "action": "upsert", "recordId": "c2",
        "trackId": "12", "status": "识别中",
    })
    pipeline_api._record_pool_event(task_id, {
        "pool": "keyframe", "action": "upsert", "recordId": "k1",
        "trackId": "12", "hullNumber": "0857", "description": "白色上层建筑",
        "status": "正式帧",
    })
    pipeline_api._record_pool_event(task_id, {
        "pool": "keyframe", "action": "upsert", "recordId": "k2",
        "trackId": "12", "hullNumber": "0857", "description": "灰色船体",
        "status": "正式帧",
    })
    pipeline_api._record_pool_event(task_id, {
        "pool": "track", "action": "upsert", "recordId": "track-12",
        "trackId": "12", "description": "目标船",
    })
    pipeline_api._record_pool_event(task_id, {
        "pool": "track", "action": "remove", "recordId": "track-12",
    })

    status = asyncio.run(pipeline_api.get_pool_status(task_id))

    assert len(status["candidate"]) == 1
    assert status["candidate"][0]["trackId"] == "12"
    assert len(status["candidate"][0]["frames"]) == 2
    assert any(frame.get("imageUrl", "").endswith("/12/c1") for frame in status["candidate"][0]["frames"])
    assert len(status["keyframe"]) == 1
    assert status["keyframe"][0]["trackId"] == "12"
    assert status["keyframe"][0]["hullNumber"] == "0857"
    assert status["keyframe"][0]["description"] == "灰色船体"
    assert "frames" not in status["keyframe"][0]
    assert status["track"] == []
    assert {row["status"] for row in status["captions"]} == {
        "轨迹候选池刷新", "轨迹关键帧池刷新"
    }
    assert all(row["kind"] == "caption" for row in status["captions"])

    image = asyncio.run(pipeline_api.get_pool_preview(task_id, "candidate", "12", "c1"))
    assert image.body == b"jpeg"

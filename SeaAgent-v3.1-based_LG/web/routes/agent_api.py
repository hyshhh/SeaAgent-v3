"""智能体、记忆和证据接口。"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from agent import AgentController
from web.models import AgentQuery
router = APIRouter(tags=["agent-memory"])

def _controller(request: Request, event_handler=None) -> AgentController:
    return AgentController(
        request.app.state.config,
        request.app.state.repository,
        request.app.state.tool_service,
        request.app.state.llm,
        request.app.state.embedder,
        request.app.state.vectors,
        event_handler=event_handler,
    )


async def _evidence_image_response(request: Request, image_path: Path, scale: float):
    preview_path = await run_in_threadpool(request.app.state.tool_service.getImagePreview, image_path, scale)
    media_type = "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@router.post("/api/agent/query")
async def query_agent(body: AgentQuery, request: Request):
    return await run_in_threadpool(_controller(request).answer, body.question, body.top_k)


@router.post("/api/agent/query/stream")
async def stream_agent_query(body: AgentQuery, request: Request):
    """逐行返回可审计的规划、观察、反思与最终回答事件。"""
    async def events():
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict] = asyncio.Queue()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        async def execute() -> None:
            try:
                result = await run_in_threadpool(_controller(request, emit).answer, body.question, body.top_k)
                await event_queue.put({"type": "complete", "title": "闭环推理完成", "message": "最终回答与视觉证据已生成", "result": result})
            except Exception as error:
                message = str(error)
                if "GRAPH_RECURSION_LIMIT" in message or "Recursion limit" in message:
                    message = "规划步骤过多已自动收敛，请重试或简化问题"
                elif "allowed-local-media-path" in message or "Cannot load local files" in message:
                    message = "视觉模型拒绝本地媒体路径，请确认服务端已用 data URL 传图"
                elif len(message) > 240:
                    message = message[:240] + "…"
                await event_queue.put({"type": "error", "title": "闭环推理失败", "message": message})

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await event_queue.get()
                yield json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                if event.get("type") in {"complete", "error"}:
                    break
        finally:
            await task

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.delete("/api/agent/memory")
async def clear_agent_memory(request: Request):
    result = await run_in_threadpool(request.app.state.memory_manager.clear_qa_memory)
    return {"success": True, "message": "问答记忆已清除", "data": result}

@router.get("/api/agent/memory-summary")
async def memory_summary(request: Request):
    snapshot = request.app.state.memory_manager.snapshot()
    registry = request.app.state.ship_service.list_ships()
    config = request.app.state.config
    qa_memory = request.app.state.repository.qa_memory_summary()
    max_rounds = int(config["pipeline"]["agent"]["max_rounds"])
    return {
        "trackCount": snapshot.get("trackCount", len(snapshot.get("tracks", []))),
        "keyframeCount": snapshot.get("keyframeCount", 0),
        "embeddedKeyframeCount": snapshot.get("embeddedKeyframeCount", 0),
        "registryCount": len(registry),
        "recognitionModel": config["llm"]["model"],
        "qaSessionCount": qa_memory["sessionCount"],
        "qaRoundCount": qa_memory["roundCount"],
        "qaEvidenceCount": qa_memory["evidenceCount"],
        "maxRounds": max_rounds,
        "retrievalTopK": int(config["pipeline"]["retrieval"].get("top_k", 3)),
        "broadMatchTopK": max(0, int(config["pipeline"]["retrieval"].get("broad_match_top_k", 0))),
    }

@router.get("/api/memory/tracks/{track_id}/frames")
async def track_frames(track_id: str, request: Request):
    return request.app.state.tool_service.getFrames([track_id])

@router.get("/api/evidence/keyframes/{keyframe_id}")
async def keyframe_file(keyframe_id: str, request: Request, scale: float = 1.0):
    item = request.app.state.repository.get_keyframe(keyframe_id)
    if not item or not Path(item["keyframePath"]).exists():
        raise HTTPException(404, "关键帧不存在")
    preview_path = await run_in_threadpool(request.app.state.tool_service.getImagePreview, Path(item["keyframePath"]), scale)
    media_type = "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "no-store, max-age=0"})

@router.get("/api/evidence/clips/{segment_id}")
async def clip_file(segment_id: str, request: Request):
    if Path(segment_id).name != segment_id:
        raise HTTPException(400, "片段编号非法")
    path = Path(request.app.state.config["paths"]["clip_dir"]) / f"{segment_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "目标船片段不存在")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store, max-age=0"})

@router.get("/api/evidence/clips/{segment_id}/poster")
async def clip_poster(segment_id: str, request: Request, scale: float = 1.0):
    if Path(segment_id).name != segment_id:
        raise HTTPException(400, "片段编号非法")
    clip_dir = Path(request.app.state.config["paths"]["clip_dir"])
    path = clip_dir / f"{segment_id}.jpg"
    clip_path = clip_dir / f"{segment_id}.mp4"
    if not path.exists() and clip_path.exists():
        quality = int(request.app.state.config["pipeline"].get("evidence", {}).get("poster_quality", 75))
        await run_in_threadpool(request.app.state.tool_service._ensure_clip_poster, clip_path, path, quality)
    if not path.exists():
        raise HTTPException(404, "目标船片段封面不存在")
    preview_path = await run_in_threadpool(request.app.state.tool_service.getImagePreview, path, scale)
    media_type = "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "no-store, max-age=0"})

@router.get("/api/evidence/tracks/{track_id}/clip")
async def track_clip(track_id: str, request: Request, startTime: float | None = None, endTime: float | None = None, scale: float = 1.0):
    time_range = (startTime, endTime) if startTime is not None and endTime is not None else None
    result = await run_in_threadpool(request.app.state.tool_service.getClip, track_id, time_range, scale)
    path = Path(result.get("segmentPath", ""))
    if not result.get("ok") or not result.get("found", True) or not path.is_file():
        raise HTTPException(404, result.get("error") or "目标船片段不存在")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store, max-age=0"})

@router.get("/api/evidence/registry/{reference_id}")
async def registry_file(reference_id: str, request: Request, scale: float = 1.0):
    items = request.app.state.repository.references_by_ids([reference_id])
    if not items or not Path(items[0]["imagePath"]).exists():
        raise HTTPException(404, "先验库参考图不存在")
    return await _evidence_image_response(request, Path(items[0]["imagePath"]), scale)

@router.get("/api/config")
async def public_config(request: Request):
    config = request.app.state.config
    return {"app": config.get("app", {}), "models": {"recognition": config["llm"]["model"], "embedding": config["embedding"]["model"], "embeddingDimension": config["embedding"]["dimension"]}, "pipeline": {"candidateEveryNFrames": config["pipeline"]["candidate_every_n_frames"], "keyframePoolSize": config["pipeline"]["keyframe_pool_size"], "maxRounds": config["pipeline"]["agent"]["max_rounds"], "displayLimit": config["pipeline"]["agent"]["display_limit"]}}

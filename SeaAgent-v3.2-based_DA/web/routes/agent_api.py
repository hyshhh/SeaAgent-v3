"""智能体、记忆和证据接口。"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from agent import AgentController
from config import project_root
from harness.shared_state import SharedStateStore
from web.models import AgentQuery, AgentResume, QueryImageUpload

router = APIRouter(tags=["agent-memory"])
logger = logging.getLogger(__name__)
MAX_QUERY_IMAGE_BYTES = 4 * 1024 * 1024
QUERY_IMAGE_SUFFIXES = {
    "data:image/jpeg;base64": ".jpg",
    "data:image/png;base64": ".png",
    "data:image/webp;base64": ".webp",
}

# ------------------------------------------------------------------ #
# 一、请求入口                                                             #
# ------------------------------------------------------------------ #
# 1.1 构造控制器，注入事件回调
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


def _query_image_directory() -> Path:
    return project_root() / "data/memory/query_images"


def _query_image_path(image_id: str) -> Path:
    image_id = str(image_id or "")
    stem, suffix = Path(image_id).stem, Path(image_id).suffix.lower()
    if Path(image_id).name != image_id or len(stem) != 32 or suffix not in {".jpg", ".png", ".webp"}:
        raise HTTPException(404, "Query image not found")
    try:
        if uuid.UUID(hex=stem).hex != stem:
            raise ValueError
    except ValueError as error:
        raise HTTPException(404, "Query image not found") from error
    return _query_image_directory() / image_id


def _save_query_image(data_url: str | None) -> tuple[str | None, str | None]:
    if not data_url:
        return None, None
    header, separator, encoded = data_url.partition(",")
    suffix = QUERY_IMAGE_SUFFIXES.get(header)
    if not separator or suffix is None:
        raise HTTPException(422, "Only JPEG, PNG and WebP images are supported")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise HTTPException(422, "The attached image is invalid") from error
    if not content or len(content) > MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(413, "Attached image must be smaller than 4 MB")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(422, "The attached image cannot be decoded")
    directory = _query_image_directory()
    directory.mkdir(parents=True, exist_ok=True)
    image_id = f"{uuid.uuid4().hex}{suffix}"
    path = directory / image_id
    try:
        path.write_bytes(content)
    except OSError as error:
        path.unlink(missing_ok=True)
        raise HTTPException(500, "Could not save the attached image") from error
    return str(path), image_id


def _resolve_query_image(body: AgentQuery, repository=None) -> tuple[str | None, str | None]:
    if body.image:
        path, image_id = _save_query_image(body.image)
    else:
        image_id = body.image_id
        if not image_id and body.session_id and repository is not None:
            turns = (repository.get_session(body.session_id) or {}).get("turns", [])
            if turns and turns[-1].get("state") == "awaiting_input":
                image_id = turns[-1].get("imageId")
        if not image_id:
            return None, None
        path = _query_image_path(image_id)
        if not path.is_file():
            raise HTTPException(404, "Query image not found")
    return str(path), image_id


def _remove_session_query_images(session: dict | None) -> None:
    for turn in (session or {}).get("turns", []):
        image_id = str(turn.get("imageId") or "")
        if image_id:
            try:
                _query_image_path(image_id).unlink(missing_ok=True)
            except HTTPException:
                continue


def _remove_all_query_images() -> None:
    directory = _query_image_directory()
    if directory.is_dir():
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".png", ".webp"}:
                path.unlink(missing_ok=True)


def _answer_controller(controller: AgentController, question: str, session_id: str, cancel: threading.Event, image_path: str | None, image_id: str | None):
    """Call both current and lightweight test controllers without passing an empty image argument."""
    kwargs = {}
    if image_path:
        kwargs["image_path"] = image_path
    if image_id:
        kwargs["image_id"] = image_id
    return controller.answer(question, session_id, cancel, **kwargs)


# 1.2 把证据图片路径包成响应
async def _evidence_image_response(request: Request, image_path: Path, scale: float):
    preview_path = await run_in_threadpool(request.app.state.tool_service.getImagePreview, image_path, scale)
    media_type = "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


# 1.3 同步问答入口
@router.post("/api/agent/query")
async def query_agent(body: AgentQuery, request: Request):
    session_id = body.session_id or f"session-{uuid.uuid4().hex[:12]}"
    cancel = _begin_run(request, session_id)
    try:
        image_path, image_id = await run_in_threadpool(
            _resolve_query_image, body, getattr(request.app.state, "repository", None)
        )
        return await run_in_threadpool(
            _answer_controller,
            _controller(request),
            body.question,
            session_id,
            cancel,
            image_path,
            image_id,
        )
    finally:
        request.app.state.agent_runs.pop(session_id, None)


@router.post("/api/agent/query-images")
async def upload_query_image(body: QueryImageUpload):
    _, image_id = await run_in_threadpool(_save_query_image, body.image)
    return {"imageId": image_id}


@router.get("/api/agent/query-images/{image_id}")
async def query_image(image_id: str):
    path = _query_image_path(image_id)
    if not path.is_file():
        raise HTTPException(404, "Query image not found")
    media_type = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[path.suffix.lower()]
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, no-store"})


@router.delete("/api/agent/query-images/{image_id}")
async def delete_query_image(image_id: str):
    path = _query_image_path(image_id)
    path.unlink(missing_ok=True)
    return {"success": True}


# 1.4 人工确认后恢复
@router.post("/api/agent/query/resume")
async def resume_agent_query(body: AgentResume, request: Request):
    cancel = _begin_run(request, body.session_id)
    try:
        return await run_in_threadpool(_controller(request).resume, body.decision, body.session_id, body.feedback, cancel)
    finally:
        request.app.state.agent_runs.pop(body.session_id, None)


# ------------------------------------------------------------------ #
# 二、运行控制                                                             #
# ------------------------------------------------------------------ #
# 2.1 登记进行中的问答
def _begin_run(request: Request, session_id: str) -> threading.Event:
    """登记一次进行中的问答，返回它的中断信号。

    进行中的运行按 session_id 记账：一个会话同时只跑一轮，停止接口凭会话号就能找到它，
    前端也可以在列表里标出「哪个会话在回答」。
    """
    if getattr(request.app.state, "agent_memory_clearing", False):
        raise HTTPException(409, "问答记忆正在清除，请稍后再试")
    if session_id in request.app.state.agent_runs:
        raise HTTPException(409, "此会话仍有问答正在执行，请等待当前运行结束")
    cancel = threading.Event()
    request.app.state.agent_runs[session_id] = cancel
    return cancel


# 1.5 NDJSON 事件流入口
@router.post("/api/agent/query/stream")
async def stream_agent_query(body: AgentQuery, request: Request):
    """逐行返回 Harness 状态、工具与最终回答事件。"""
    session_id = body.session_id or f"session-{uuid.uuid4().hex[:12]}"
    cancel = _begin_run(request, session_id)
    try:
        image_path, image_id = await run_in_threadpool(
            _resolve_query_image, body, getattr(request.app.state, "repository", None)
        )
    except Exception:
        request.app.state.agent_runs.pop(session_id, None)
        raise

    async def events():
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[dict] = asyncio.Queue()
        terminal_seen = threading.Event()
        event_lock = threading.Lock()
        worker: asyncio.Task | None = None
        captured_terminal: dict | None = None

        def release(_task=None) -> None:
            if request.app.state.agent_runs.get(session_id) is cancel:
                request.app.state.agent_runs.pop(session_id, None)
            if _task is not None and not _task.cancelled():
                _task.exception()  # 已断开的客户端也要消费后台异常。

        def emit(event: dict) -> None:
            nonlocal captured_terminal
            # runtime 在 worker 线程中执行；用线程安全标记保证 terminal 事件只由一个来源发送。
            with event_lock:
                if terminal_seen.is_set():
                    return
                if event.get("type") in {"complete", "error", "confirm"}:
                    captured_terminal = event  # 等控制器完成审计与结果投影，再发送终局事件。
                    return
                loop.call_soon_threadsafe(event_queue.put_nowait, event)

        async def execute() -> None:
            nonlocal worker
            # 预算读取也要在 try 内：任何异常都必须变成一条终止事件，
            # 否则事件队列永远不出队，整条 NDJSON 流会永久等待（这个坑刚踩过）
            try:
                worker = asyncio.create_task(
                    run_in_threadpool(
                        _answer_controller,
                        _controller(request, emit),
                        body.question,
                        session_id,
                        cancel,
                        image_path,
                        image_id,
                    )
                )
                worker.add_done_callback(release)
                result = await asyncio.shield(worker)
                if terminal_seen.is_set():
                    return
                if captured_terminal is not None:
                    terminal_seen.set()
                    event = {**captured_terminal, "result": result}
                    if result.get("state") == "error":
                        event.update(type="error", message=result.get("error") or event.get("message"))
                    await event_queue.put(event)
                elif result.get("state") == "awaiting_confirmation":
                    terminal_seen.set()
                    await event_queue.put({"type": "confirm", "title": "等待人工确认",
                                           "confirmation": result.get("confirmation") or {}, "result": result})
                elif result.get("success", False):
                    terminal_seen.set()
                    await event_queue.put({
                        "type": "complete",
                        "title": "Harness 完成",
                        "message": "最终回答与视觉证据已生成",
                        "result": result,
                    })
                elif result.get("state") == "cancelled":
                    terminal_seen.set()
                    await event_queue.put({
                        "type": "complete",
                        "title": "Harness 已停止",
                        "message": "本轮已按请求停止",
                        "result": result,
                    })
                else:
                    terminal_seen.set()
                    await event_queue.put({
                        "type": "error",
                        "title": "Harness 未完成",
                        "message": str(result.get("error") or "未能生成完整回答"),
                        "errorType": result.get("errorType"),
                        "result": result,
                    })
            except Exception as error:
                logger.exception("Agent stream task failed")
                if terminal_seen.is_set():
                    return
                terminal_seen.set()
                message = str(error).strip() or f"{type(error).__name__}: {error!r}"
                if len(message) > 240:
                    message = message[:240] + "…"
                await event_queue.put({
                    "type": "error",
                    "title": "Harness 执行失败",
                    "message": message,
                    "errorType": type(error).__name__,
                })

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await event_queue.get()
                yield json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
                if event.get("type") in {"complete", "error", "confirm"}:
                    break
        finally:
            cancel.set()
            terminal_seen.set()
            if worker is None or worker.done():
                release()
            # 超时后 worker 线程可能仍在跑，await 它会再次卡住这条流；只在已结束时收尾
            if task.done():
                await task

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 2.2 请求停止当前问答
@router.post("/api/agent/sessions/{session_id}/stop")
async def stop_agent_run(session_id: str, request: Request):
    """请求停止某个会话正在进行的问答；只置中断信号，不删除会话与历史。"""
    cancel = request.app.state.agent_runs.get(session_id)
    if cancel is None:
        return {"success": False, "message": "该会话当前没有进行中的问答", "data": {"sessionId": session_id}}
    cancel.set()
    return {"success": True, "message": "已请求停止，本轮会在当前步骤结束后收尾", "data": {"sessionId": session_id}}

# ------------------------------------------------------------------ #
# 三、会话管理                                                             #
# ------------------------------------------------------------------ #
# 3.1 清空会话与检查点
@router.delete("/api/agent/memory")
async def clear_agent_memory(request: Request):
    if request.app.state.agent_runs or getattr(request.app.state, "agent_memory_clearing", False):
        raise HTTPException(409, "请先停止进行中的问答，再清除记忆")
    request.app.state.agent_memory_clearing = True
    try:
        result = await run_in_threadpool(request.app.state.memory_manager.clear_qa_memory)
        _remove_all_query_images()
    finally:
        request.app.state.agent_memory_clearing = False
    return {"success": True, "message": "问答记忆已清除", "data": result}

# 3.2 会话摘要列表
@router.get("/api/agent/sessions")
async def list_agent_sessions(request: Request, limit: int = 50):
    """会话摘要列表，最近更新的在前；不含轮次正文。"""
    sessions = await run_in_threadpool(request.app.state.repository.list_sessions, limit)
    return {"sessions": sessions, "total": len(sessions)}

# 3.3 单个会话完整记录
@router.get("/api/agent/sessions/{session_id}")
async def get_agent_session(session_id: str, request: Request):
    """单个会话的完整记录（含每轮问答），供前端还原对话并继续追问。"""
    session = await run_in_threadpool(request.app.state.repository.get_session, session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    return session

# 3.4 删除会话
@router.delete("/api/agent/sessions/{session_id}")
async def delete_agent_session(session_id: str, request: Request):
    cancel = _begin_run(request, session_id)
    try:
        session = await run_in_threadpool(request.app.state.repository.get_session, session_id)
        removed = await run_in_threadpool(request.app.state.repository.delete_session, session_id)
        if not removed:
            raise HTTPException(404, "会话不存在")
        _remove_session_query_images(session)
        config = getattr(request.app.state, "config", None)
        if config is not None:
            run_ids = {
                str(turn.get("runtimeId"))
                for turn in (session or {}).get("turns", [])
                if turn.get("runtimeId")
            }
            run_ids.add(session_id)
            for run_id in run_ids:
                await run_in_threadpool(_delete_runtime_state, config, run_id)
    finally:
        if request.app.state.agent_runs.get(session_id) is cancel:
            request.app.state.agent_runs.pop(session_id, None)
    return {"success": True, "message": "会话已删除", "data": {"sessionId": session_id}}


# 3.5 删除运行时状态
def _delete_runtime_state(config: dict, thread_id: str) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver

    harness = config.get("harness", {})
    state_path = project_root() / str(harness.get("shared_state_file", "data/memory/shared_task_state.sqlite"))
    SharedStateStore(state_path).delete(thread_id)
    checkpoint = project_root() / str(harness.get("checkpointer", "data/memory/checkpoints.sqlite"))
    if checkpoint.is_file():
        with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
            for role in ("planner", "executor", "reflector"):
                saver.delete_thread(f"{thread_id}:{role}")

# 3.6 记忆概览
@router.get("/api/agent/memory-summary")
async def memory_summary(request: Request):
    snapshot = request.app.state.memory_manager.snapshot()
    registry = request.app.state.ship_service.list_ships()
    config = request.app.state.config
    qa_memory = request.app.state.repository.qa_memory_summary()
    return {
        "trackCount": snapshot.get("trackCount", len(snapshot.get("tracks", []))),
        "keyframeCount": snapshot.get("keyframeCount", 0),
        "embeddedKeyframeCount": snapshot.get("embeddedKeyframeCount", 0),
        "registryCount": len(registry),
        "recognitionModel": config["llm"]["model"],
        "qaSessionCount": qa_memory["sessionCount"],
        "retrievalTopK": int(config["pipeline"]["retrieval"].get("top_k", 3)),
        "broadMatchTopK": max(0, int(config["pipeline"]["retrieval"].get("broad_match_top_k", 0))),
    }

# ------------------------------------------------------------------ #
# 四、证据与媒体                                                            #
# ------------------------------------------------------------------ #
# 4.1 轨迹关键帧
@router.get("/api/memory/tracks/{track_id}/frames")
async def track_frames(track_id: str, request: Request):
    return request.app.state.tool_service._build_frames([track_id])

# 4.2 关键帧文件
@router.get("/api/evidence/keyframes/{keyframe_id}")
async def keyframe_file(keyframe_id: str, request: Request, scale: float = 1.0):
    item = request.app.state.repository.get_keyframe(keyframe_id)
    if not item or not Path(item["keyframePath"]).exists():
        raise HTTPException(404, "关键帧不存在")
    preview_path = await run_in_threadpool(request.app.state.tool_service.getImagePreview, Path(item["keyframePath"]), scale)
    media_type = "image/jpeg" if preview_path.suffix.lower() in {".jpg", ".jpeg"} else None
    return FileResponse(preview_path, media_type=media_type, headers={"Cache-Control": "no-store, max-age=0"})

# 4.3 视频片段文件
@router.get("/api/evidence/clips/{segment_id}")
async def clip_file(segment_id: str, request: Request):
    if Path(segment_id).name != segment_id:
        raise HTTPException(400, "片段编号非法")
    path = Path(request.app.state.config["paths"]["clip_dir"]) / f"{segment_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "目标船片段不存在")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store, max-age=0"})

# 4.4 片段封面
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

# 4.5 轨迹片段
@router.get("/api/evidence/tracks/{track_id}/clip")
async def track_clip(track_id: str, request: Request, startTime: float | None = None, endTime: float | None = None, scale: float = 1.0):
    time_range = (startTime, endTime) if startTime is not None and endTime is not None else None
    result = await run_in_threadpool(request.app.state.tool_service._build_clip, track_id, time_range, scale)
    path = Path(result.get("segmentPath", ""))
    if not result.get("ok") or not result.get("found", True) or not path.is_file():
        raise HTTPException(404, result.get("error") or "目标船片段不存在")
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store, max-age=0"})

# 4.6 先验库参考图
@router.get("/api/evidence/registry/{reference_id}")
async def registry_file(reference_id: str, request: Request, scale: float = 1.0):
    items = request.app.state.repository.references_by_ids([reference_id])
    if not items or not Path(items[0]["imagePath"]).exists():
        raise HTTPException(404, "先验库参考图不存在")
    return await _evidence_image_response(request, Path(items[0]["imagePath"]), scale)

# ------------------------------------------------------------------ #
# 五、前端配置                                                             #
# ------------------------------------------------------------------ #
# 5.1 暴露给前端的公开配置
@router.get("/api/config")
async def public_config(request: Request):
    config = request.app.state.config
    return {"app": config.get("app", {}), "models": {"recognition": config["llm"]["model"], "embedding": config["embedding"]["model"], "embeddingDimension": config["embedding"]["dimension"]}, "pipeline": {"candidateEveryNFrames": config["pipeline"]["candidate_every_n_frames"], "keyframePoolSize": config["pipeline"]["keyframe_pool_size"]}}



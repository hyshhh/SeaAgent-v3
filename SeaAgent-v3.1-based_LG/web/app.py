"""SeaAgent 网页服务入口。"""
from __future__ import annotations
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from config import load_config
from memory import MemoryRepository, TrackMemoryManager
from services import AgentLLMService, QwenMultimodalEmbedder
from tools import ToolService
from vector_store import VectorCatalog
from web.routes import agent_router, api_router, config_router, memory_router, pages_router, pipeline_router
from web.services import ShipService

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    repository = MemoryRepository(config)
    embedder = QwenMultimodalEmbedder(config)
    llm = AgentLLMService(config)
    vectors = VectorCatalog(config)
    app.state.config = config
    app.state.repository = repository
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.vectors = vectors
    app.state.memory_manager = TrackMemoryManager(config, repository, vectors)
    app.state.tool_service = ToolService(config, repository, embedder, llm, vectors)
    app.state.ship_service = ShipService(config, repository, embedder, llm, vectors)
    if not shutil.which("ffmpeg"):
        logging.getLogger(__name__).warning("未找到 ffmpeg，部分浏览器视频转码能力不可用")
    yield

app = FastAPI(title="SeaAgent", description="面向海域船舶监控的轨迹记忆闭环多智能体系统", version="3.0.0", lifespan=lifespan)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
app.include_router(pages_router)
app.include_router(api_router)
app.include_router(config_router)
app.include_router(memory_router)
app.include_router(agent_router)
app.include_router(pipeline_router)

def main():
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config.get("web", {}).get("host", "0.0.0.0"), port=int(config.get("web", {}).get("port", 8000)), log_level="info", ws_ping_interval=0)

if __name__ == "__main__":
    main()

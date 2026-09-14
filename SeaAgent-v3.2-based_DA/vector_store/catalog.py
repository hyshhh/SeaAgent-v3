"""正式关键帧与先验库参考图的双索引。"""
from __future__ import annotations
from typing import Any
from config import load_config
from vector_store.index import ExactFaissIndex

class VectorCatalog:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or load_config()
        settings, paths = config["embedding"], config["paths"]
        dimension, model_name = int(settings.get("dimension", 2048)), settings["model"]
        self.keyframes = ExactFaissIndex(paths["keyframe_index"], dimension, model_name)
        self.registry = ExactFaissIndex(paths["registry_index"], dimension, model_name)

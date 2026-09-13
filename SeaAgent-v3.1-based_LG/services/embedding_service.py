"""Qwen3-VL-Embedding-2B 的兼容接口封装。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np

from config import load_config


class EmbeddingUnavailableError(RuntimeError):
    pass


class QwenMultimodalEmbedder:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()
        self.settings = self.config["embedding"]
        self.dimension = int(self.settings.get("dimension", 2048))

    def process(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        timeout = float(self.settings.get("timeout_seconds", 60))
        try:
            with httpx.Client(timeout=timeout) as client:
                vectors = [self._request_embedding(client, item) for item in inputs]
        except EmbeddingUnavailableError:
            raise
        except Exception as error:
            raise EmbeddingUnavailableError(f"向量模型调用失败：{error}") from error
        values = np.asarray(vectors, dtype=np.float32)
        if bool(self.settings.get("normalize", True)):
            values = self._normalize(values)
        return values

    def encode_images(self, image_paths: Iterable[str | Path]) -> np.ndarray:
        return self.process([{"image": str(Path(path).resolve())} for path in image_paths])

    def encode_text(self, text: str, instruction: str) -> np.ndarray:
        return self.process([{"text": text, "instruction": instruction}])[0]

    def _request_embedding(self, client: httpx.Client, item: dict[str, Any]) -> np.ndarray:
        payload = {
            "model": self.settings["model"],
            "messages": self._build_messages(item),
            "encoding_format": "float",
        }
        url = f"{self.settings['base_url'].rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        try:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            raise EmbeddingUnavailableError(
                f"向量模型接口返回 {error.response.status_code}：{detail}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise EmbeddingUnavailableError(f"向量模型接口请求失败：{error}") from error

        data = body.get("data") if isinstance(body, dict) else None
        embedding = data[0].get("embedding") if isinstance(data, list) and data else None
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            actual = vector.shape[0] if vector.ndim == 1 else "未知"
            raise EmbeddingUnavailableError(
                f"向量维度错误：期望 {self.dimension}，实际 {actual}"
            )
        if not np.all(np.isfinite(vector)):
            raise EmbeddingUnavailableError("向量模型返回非有限数值")
        return vector

    def _build_messages(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        instruction = str(item.get("instruction") or "Represent the user's input.")
        content: list[dict[str, Any]] = []
        images = item.get("image")
        if images is not None:
            for image in images if isinstance(images, (list, tuple)) else [images]:
                content.append({"type": "image_url", "image_url": {"url": self._data_url(image)}})
        texts = item.get("text")
        if texts is not None:
            for text in texts if isinstance(texts, (list, tuple)) else [texts]:
                content.append({"type": "text", "text": str(text)})
        if not content:
            content.append({"type": "text", "text": ""})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _data_url(image: str | Path) -> str:
        value = str(image)
        if value.startswith(("http://", "https://", "data:")):
            return value
        path = Path(image)
        if not path.is_file():
            raise EmbeddingUnavailableError(f"图像不存在：{path}")
        mime = {".png": "image/png", ".webp": "image/webp"}.get(
            path.suffix.lower(), "image/jpeg"
        )
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise EmbeddingUnavailableError("向量模型返回零向量")
        return vectors / norms

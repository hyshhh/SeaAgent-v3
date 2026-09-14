"""使用 FAISS 点积索引保存归一化多模态向量。"""
from __future__ import annotations
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable
import numpy as np

class FaissUnavailableError(RuntimeError):
    pass

def stable_vector_id(owner_id: str) -> int:
    return int.from_bytes(hashlib.sha256(owner_id.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF

class ExactFaissIndex:
    def __init__(self, path: str | Path, dimension: int, model_name: str):
        self.path = Path(path)
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".json")
        self.dimension = int(dimension)
        self.model_name = model_name
        self._index = None
        self._loaded_signature = None
        self._lock = threading.RLock()

    @staticmethod
    def _faiss():
        try:
            import faiss
            return faiss
        except ImportError as error:
            raise FaissUnavailableError("未安装 faiss-cpu，请按 SETUPREADME.md 安装环境") from error

    def _empty(self):
        faiss = self._faiss()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dimension))

    def load(self):
        with self._lock:
            signature = self._storage_signature()
            if self._index is not None and signature == self._loaded_signature:
                return self._index
            self._validate_manifest()
            self._index = self._faiss().read_index(str(self.path)) if self.path.exists() else self._empty()
            if self._index.d != self.dimension:
                raise ValueError(f"索引维度错误：{self.path}")
            self._loaded_signature = self._storage_signature()
            return self._index

    @property
    def count(self) -> int:
        return int(self.load().ntotal)

    def add(self, vector_id: int, vector: np.ndarray) -> None:
        self.add_many([vector_id], np.asarray(vector, dtype=np.float32).reshape(1, -1))

    def add_many(self, vector_ids: Iterable[int], vectors: np.ndarray) -> None:
        ids = np.asarray(list(vector_ids), dtype=np.int64)
        values = self._normalize(vectors)
        if len(ids) != len(values):
            raise ValueError("向量编号数量与向量数量不一致")
        with self._lock:
            index = self.load()
            if len(ids):
                index.remove_ids(ids)
                index.add_with_ids(values, ids)
            self.save()

    def remove(self, vector_ids: Iterable[int]) -> None:
        ids = np.asarray(list(vector_ids), dtype=np.int64)
        with self._lock:
            if len(ids):
                self.load().remove_ids(ids)
                self.save()

    def rebuild(self, entries: Iterable[tuple[int, np.ndarray]]) -> None:
        pairs = list(entries)
        with self._lock:
            self._index = self._empty()
            if pairs:
                ids = np.asarray([item[0] for item in pairs], dtype=np.int64)
                vectors = self._normalize(np.vstack([item[1] for item in pairs]))
                self._index.add_with_ids(vectors, ids)
            self.save()

    def get(self, vector_id: int) -> np.ndarray:
        with self._lock:
            try:
                return np.asarray(self.load().reconstruct(int(vector_id)), dtype=np.float32)
            except RuntimeError as error:
                raise KeyError(f"向量不存在：{vector_id}") from error

    def get_many(self, vector_ids: Iterable[int]) -> dict[int, np.ndarray]:
        result = {}
        with self._lock:
            index = self.load()
            for vector_id in vector_ids:
                try:
                    result[int(vector_id)] = np.asarray(index.reconstruct(int(vector_id)), dtype=np.float32)
                except RuntimeError:
                    continue
        return result

    def search(self, query: np.ndarray, top_k: int, allowed_ids: set[int] | None = None) -> list[dict[str, float | int]]:
        if top_k <= 0:
            return []
        vector = self._normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        with self._lock:
            index = self.load()
            if index.ntotal == 0:
                return []
            search_k = int(index.ntotal) if allowed_ids is not None else min(top_k, int(index.ntotal))
            scores, ids = index.search(vector, search_k)
        results = []
        for vector_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if vector_id < 0 or allowed_ids is not None and vector_id not in allowed_ids:
                continue
            results.append({"vectorId": int(vector_id), "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        index = self._index if self._index is not None else self.load()
        self._faiss().write_index(index, str(temporary))
        os.replace(temporary, self.path)
        manifest = {"model": self.model_name, "dimension": self.dimension, "normalized": True, "index_type": "IndexIDMap2(IndexFlatIP)"}
        temp_manifest = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_manifest, self.manifest_path)
        self._loaded_signature = self._storage_signature()

    def reset_cache(self) -> None:
        with self._lock:
            self._index = None
            self._loaded_signature = None

    def _storage_signature(self):
        return self._file_signature(self.path), self._file_signature(self.manifest_path)

    @staticmethod
    def _file_signature(path: Path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0)
        except FileNotFoundError:
            return None

    def _validate_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != self.model_name or int(manifest.get("dimension", 0)) != self.dimension:
            raise ValueError(f"索引清单与当前模型不一致：{self.manifest_path}")
        if manifest.get("normalized") is not True:
            raise ValueError(f"索引必须保存归一化向量：{self.manifest_path}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"向量形状必须为 N×{self.dimension}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("禁止写入零向量")
        return np.ascontiguousarray(values / norms, dtype=np.float32)

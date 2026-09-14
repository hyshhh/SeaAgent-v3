"""先验库图片、CSV 与向量索引的一致性管理。"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import load_config
from memory import MemoryRepository, normalize_hull_number
from services import AgentLLMService, QwenMultimodalEmbedder
from vector_store import VectorCatalog, stable_vector_id


class ShipService:
    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.image_dir = Path(self.config["paths"]["registry_image_dir"])
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def list_registry(self) -> list[dict[str, Any]]:
        return self.repository.list_registry()

    def get_registry(self, hull_number: str) -> list[dict[str, Any]]:
        return self.repository.registry_by_hull(hull_number)

    def create_registry(self, hull_number: str, description: str = "", aliases: list[str] | None = None, images: list[tuple[str, bytes]] | None = None) -> dict[str, Any]:
        hull = normalize_hull_number(hull_number)
        if not hull:
            raise ValueError("舷号不能为空")
        if self.repository.registry_by_hull(hull):
            raise FileExistsError(f"舷号已存在：{hull}")
        files = images or []
        if len(files) > 6:
            raise ValueError("每个库项最多保存六张参考图")
        inferred = self._recognize(files[0][1]) if files and not description.strip() else {}
        final_description = description.strip() or inferred.get("description", "")
        result: dict[str, Any] = {}
        def mutate():
            registry_id = self.repository.upsert_registry({"hull_number": hull, "aliases": aliases or [], "description": final_description})
            self._save_references(registry_id, files)
            result.update(registryId=registry_id)
        self._transaction(mutate)
        return self.repository.registry_by_hull(hull)[0]

    def update_registry(self, hull_number: str, description: str | None = None, aliases: list[str] | None = None, images: list[tuple[str, bytes]] | None = None) -> dict[str, Any]:
        items = self.repository.registry_by_hull(hull_number)
        if not items:
            raise KeyError(f"未找到舷号：{hull_number}")
        item = items[0]
        files = images or []
        if len(item.get("references", [])) + len(files) > 6:
            raise ValueError("每个库项最多保存六张参考图")
        def mutate():
            self.repository.upsert_registry({"registry_id": item["registryId"], "hull_number": item["hullNumber"], "aliases": item["aliases"] if aliases is None else aliases, "description": item["description"] if description is None else description, "structured_attributes": item["structuredAttributes"]})
            self._save_references(item["registryId"], files)
        self._transaction(mutate)
        return self.repository.registry_by_hull(item["hullNumber"])[0]

    def delete_registry(self, hull_number: str) -> bool:
        items = self.repository.registry_by_hull(hull_number)
        if not items:
            return False
        def mutate():
            for item in items:
                _, references = self.repository.delete_registry(item["registryId"])
                for reference in references:
                    Path(reference["imagePath"]).unlink(missing_ok=True)
        self._transaction(mutate)
        return True

    def delete_reference(self, hull_number: str, reference_id: str) -> bool:
        items = self.repository.registry_by_hull(hull_number)
        allowed = {reference["referenceId"] for item in items for reference in item.get("references", [])}
        if reference_id not in allowed:
            return False
        def mutate():
            reference = self.repository.delete_registry_reference(reference_id)
            if reference:
                Path(reference["imagePath"]).unlink(missing_ok=True)
        self._transaction(mutate)
        return True

    def _rebuild_registry_index(self) -> dict[str, Any]:
        references = self.repository.registry_references()
        if not references:
            self.vectors.registry.rebuild([])
            return {"references": 0, "vectors": 0}
        paths = [Path(item["imagePath"]) for item in references]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"先验库参考图缺失：{missing}")
        vectors = self.embedder.encode_images(paths)
        if len(vectors) != len(references):
            raise RuntimeError("先验库参考图与向量数量不一致")
        entries = [(stable_vector_id(reference["referenceId"]), vector) for reference, vector in zip(references, vectors)]
        self.vectors.registry.rebuild(entries)
        for reference, (vector_id, _) in zip(references, entries):
            self.repository.upsert_registry_reference({"reference_id": reference["referenceId"], "registry_id": reference["registryId"], "image_path": reference["imagePath"], "registry_vector_id": vector_id, "is_embedded": True})
        return {"references": len(references), "vectors": len(entries)}

    def recognize_ship(self, image_bytes: bytes, filename: str = "upload.jpg") -> dict[str, Any]:
        result = self._recognize(image_bytes)
        hull = result.get("hull_number", "")
        existing = self.repository.registry_by_hull(hull) if hull else []
        return {**result, "already_exists": bool(existing), "existing_description": existing[0]["description"] if existing else None}

    def recognize_and_add(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        result = self.recognize_ship(image_bytes, filename)
        if not result.get("hull_number"):
            return {"error": "未能识别出舷号，请手动输入", "result": result}
        if result["already_exists"]:
            item = self.update_registry(result["hull_number"], result["description"], images=[(filename, image_bytes)])
            action = "updated"
        else:
            item = self.create_registry(result["hull_number"], result["description"], images=[(filename, image_bytes)])
            action = "added"
        return {**result, "action": action, "registry": item}

    def list_ships(self) -> list[dict[str, Any]]:
        return [{"registry_id": item["registryId"], "hull_number": item["hullNumber"], "description": item["description"], "aliases": item["aliases"], "references": item.get("references", []), "searchable": any(reference["isEmbedded"] for reference in item.get("references", []))} for item in self.list_registry()]

    def get_ship(self, hull_number: str) -> dict[str, Any] | None:
        items = self.get_registry(hull_number)
        return self.list_ships_by_items(items)[0] if items else None

    def create_ship(self, hull_number: str, description: str) -> bool:
        try:
            self.create_registry(hull_number, description)
            return True
        except FileExistsError:
            return False

    def bulk_create(self, ships: dict[str, str]) -> dict[str, int]:
        added = skipped = 0
        for hull, description in ships.items():
            if self.create_ship(hull, description):
                added += 1
            else:
                skipped += 1
        return {"added": added, "skipped": skipped}

    def search(self, query: str) -> list[dict[str, Any]]:
        keyword = query.strip().lower()
        return [item for item in self.list_ships() if keyword in item["hull_number"].lower() or keyword in item["description"].lower()]

    def stats(self) -> dict[str, Any]:
        items = self.list_ships()
        backend = self.config["registry"]["backend"]
        return {
            "total_ships": len(items),
            "total_reference_images": sum(len(item["references"]) for item in items),
            "backend": str(backend),
        }

    @staticmethod
    def list_ships_by_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"registry_id": item["registryId"], "hull_number": item["hullNumber"], "description": item["description"], "aliases": item["aliases"], "references": item.get("references", []), "searchable": any(reference["isEmbedded"] for reference in item.get("references", []))} for item in items]

    def _save_references(self, registry_id: str, images: list[tuple[str, bytes]]) -> None:
        target_dir = self.image_dir / registry_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, raw in images:
            image = self._decode(raw)
            reference_id = f"reference-{uuid.uuid4().hex[:12]}"
            suffix = Path(filename).suffix.lower() if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".jpg"
            path = target_dir / f"{reference_id}{suffix}"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"参考图保存失败：{path}")
            self.repository.upsert_registry_reference({"reference_id": reference_id, "registry_id": registry_id, "image_path": str(path), "registry_vector_id": None, "is_embedded": False})

    def _transaction(self, mutate: Callable[[], None]) -> None:
        registry_rows = self.repository.registry.rows()
        reference_rows = self.repository.registry_images.rows()
        index_path = Path(self.config["paths"]["registry_index"])
        manifest_path = index_path.with_suffix(index_path.suffix + ".json")
        with tempfile.TemporaryDirectory() as temp:
            backup = Path(temp)
            image_backup = backup / "images"
            if self.image_dir.exists():
                shutil.copytree(self.image_dir, image_backup)
            for source, name in ((index_path, "index.faiss"), (manifest_path, "index.json")):
                if source.exists():
                    shutil.copy2(source, backup / name)
            try:
                mutate()
                self._rebuild_registry_index()
            except Exception:
                self.repository.registry.replace_all(registry_rows)
                self.repository.registry_images.replace_all(reference_rows)
                if self.image_dir.exists():
                    shutil.rmtree(self.image_dir)
                if image_backup.exists():
                    shutil.copytree(image_backup, self.image_dir)
                else:
                    self.image_dir.mkdir(parents=True, exist_ok=True)
                for target, name in ((index_path, "index.faiss"), (manifest_path, "index.json")):
                    saved = backup / name
                    if saved.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(saved, target)
                    else:
                        target.unlink(missing_ok=True)
                self.vectors.registry.reset_cache()
                raise

    def _recognize(self, image_bytes: bytes) -> dict[str, str]:
        image = self._decode(image_bytes)
        result = self.llm.recognize(image)
        return {"hull_number": result.get("vlm_hull_number") or "", "description": result.get("description") or ""}

    @staticmethod
    def _decode(raw: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析上传图片")
        return image

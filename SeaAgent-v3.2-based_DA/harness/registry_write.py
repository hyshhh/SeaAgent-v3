"""待人工确认的先验库写入工具。

为什么单独一个模块：
    这是整条问答链路里**唯一会改进知识库**的操作，所以它与那 10 个读工具分开，
    并且配了 HumanInTheLoop 中断——写之前必须由人过一眼。

它在链路里的位置：
    executor 调 add_registry_vessel → 中间件拦下并中断 → 前端弹确认卡
      ├─ 批准 → 工具真正执行（复用 ShipService 的完整事务：写档案 → 存图 → 编码 → 重建索引）
      └─ 拒绝 + 理由 → 工具**不执行**，理由作为工具结果回给模型，让它换个做法

三条纪律写死在工具描述和校验里（参考 registry.registry 技能的"多步操作"精神）：
    1. 只有用户明确要求入库/写入时才调用；模型自己觉得"这条船像 003"不算理由。
    2. 只新增，不覆盖：舷号已存在直接报错，让人来判断是改还是不加。
    3. 参考图必须是先验库口径的图，不能把视频关键帧当"已知身份"存进去。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 允许的参考图后缀：与 web 层上传同一套口径
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 单个库项的参考图上限：与 ShipService 保持一致
MAX_REFERENCES = 6
# 视频容器后缀：关键帧/片段不是参考图，出现即拒绝
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class RegistryWriteRefused(Exception):
    """写入前的校验未通过：不执行写入，把原因作为工具结果交回模型。"""


def _image_paths(values: Any) -> list[Path]:
    """把入参收敛成存在的图片路径列表。"""
    if values in (None, ""):
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise RegistryWriteRefused("image_paths 必须是图片路径的列表")
    paths: list[Path] = []
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        path = Path(text)
        if path.suffix.lower() in VIDEO_SUFFIXES:
            raise RegistryWriteRefused(
                f"参考图不能是视频文件：{path.name}。先验库的参考图是「已知身份」，视频关键帧是「待证事实」，"
                "两者不能混用；请先用 get_frames 取到关键帧图片，或让人工确认后走界面上传入库。"
            )
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise RegistryWriteRefused(f"不支持的参考图格式：{path.name}")
        if not path.is_file():
            raise RegistryWriteRefused(f"参考图不存在：{path}")
        paths.append(path)
    if len(paths) > MAX_REFERENCES:
        raise RegistryWriteRefused(f"每个库项最多 {MAX_REFERENCES} 张参考图，收到 {len(paths)} 张")
    return paths


class RegistryWriteTool:
    """把「新增一条先验库船舶」包成一个可被中断确认的工具。"""

    def __init__(self, service: Any) -> None:
        # service 是 web.services.ShipService 的实例（惰性注入，避免 tools 层反向依赖 web）
        self._service = service

    # ------------------------------------------------------------------ 工具主体
    def addRegistryVessel(
        self,
        hull_number: str,
        description: str | None = None,
        image_paths: Any = None,
        aliases: Any = None,
        user_intent: str | None = None,
    ) -> dict[str, Any]:
        """新增一条先验库记录并重建向量索引；校验不过则拒绝写入并说明原因。"""
        intent = str(user_intent or "").strip()
        if not intent:
            return {
                "ok": False,
                "error": "user_intent_required",
                "hint": "写入前必须说明这是用户的哪句话要求的（user_intent 填用户的原始说法）。"
                        "如果你只是自己推断这条船应该入库，不要调用本工具——先问用户。",
            }
        hull = str(hull_number or "").strip()
        if not hull:
            return {"ok": False, "error": "hull_number_required", "hint": "舷号不能为空，请先确认舷号再写入。"}
        try:
            paths = _image_paths(image_paths)
            alias_list = [str(item).strip() for item in (aliases or []) if str(item).strip()] if isinstance(aliases, (list, tuple)) else []
            item = self._service.create_registry(
                hull,
                str(description or "").strip(),
                alias_list,
                [(path.name, path.read_bytes()) for path in paths],
            )
        except RegistryWriteRefused as refused:
            return {"ok": False, "error": "invalid_reference_image", "hint": str(refused), "hullNumber": hull}
        except FileExistsError as exists:
            return {
                "ok": False,
                "error": "hull_already_exists",
                "hint": f"{exists}。本工具只新增、不覆盖：要改已有记录请让人工在界面上更新，或换一个舷号。",
                "hullNumber": hull,
            }
        except (ValueError, OSError, RuntimeError) as failure:
            return {"ok": False, "error": "write_failed", "hint": str(failure), "hullNumber": hull}

        references = item.get("references") or []
        return {
            "ok": True,
            "hullNumber": item.get("hullNumber", hull),
            "registryId": item.get("registryId", ""),
            "description": item.get("description", ""),
            "aliases": item.get("aliases", []),
            "referenceIds": [reference.get("referenceId") for reference in references],
            "referenceCount": len(references),
            # 索引重建由 ShipService 的事务负责：这一步没成功，写库也不会留下半条记录
            "indexRebuilt": bool(references) and all(reference.get("isEmbedded") for reference in references),
            "userIntent": intent,
        }

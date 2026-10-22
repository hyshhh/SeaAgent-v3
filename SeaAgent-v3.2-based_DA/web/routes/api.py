"""先验库管理接口。"""
from __future__ import annotations
import json
import logging
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from web.models import ApiResponse, SearchResponse, ShipBulkCreate, ShipCreate, ShipItem, ShipListResponse, ShipUpdate, StatsResponse
from web.services import ShipService
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ships", tags=["registry"])
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024

# ------------------------------------------------------------------ #
# 一、依赖与工具                                                            #
# ------------------------------------------------------------------ #
# 1.1 从应用状态取 ShipService
def get_service(request: Request) -> ShipService:
    return request.app.state.ship_service

# 1.2 读取上传的图片
async def _read_images(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    images = []
    for file in files:
        if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, f"不支持的文件类型：{file.content_type}")
        raw = await file.read()
        if len(raw) > MAX_FILE_SIZE:
            raise HTTPException(400, "单张图片不能超过 20MB")
        images.append((file.filename or "upload.jpg", raw))
    return images

# ------------------------------------------------------------------ #
# 二、档案查询                                                             #
# ------------------------------------------------------------------ #
# 2.1 船舶列表
@router.get("", response_model=ShipListResponse)
async def list_ships(service: Annotated[ShipService, Depends(get_service)]):
    ships = service.list_ships()
    return ShipListResponse(total=len(ships), ships=[ShipItem(**item) for item in ships])

# 2.2 船舶搜索
@router.get("/search", response_model=SearchResponse)
async def search_ships(q: Annotated[str, Query()], service: Annotated[ShipService, Depends(get_service)]):
    results = service.search(q)
    return SearchResponse(total=len(results), results=[ShipItem(**item) for item in results])

# 2.3 档案统计
@router.get("/stats", response_model=StatsResponse)
async def stats(service: Annotated[ShipService, Depends(get_service)]):
    return StatsResponse(**service.stats())

# ------------------------------------------------------------------ #
# 三、档案写入                                                             #
# ------------------------------------------------------------------ #
# 3.1 新增船舶
@router.post("", response_model=ApiResponse)
async def create_ship(body: ShipCreate, service: Annotated[ShipService, Depends(get_service)]):
    try:
        item = service.create_registry(body.hull_number, body.description, body.aliases)
    except FileExistsError as error:
        raise HTTPException(409, str(error)) from error
    return ApiResponse(success=True, message=f"成功添加舷号：{body.hull_number}", data=item)

# 3.2 上传先验库
@router.post("/upload", response_model=ApiResponse)
async def upload_registry(files: list[UploadFile] = File(...), hull_number: str = Form(""), description: str = Form(""), aliases: str = Form("[]"), service: ShipService = Depends(get_service)):
    images = await _read_images(files)
    if not 1 <= len(images) <= 6:
        raise HTTPException(400, "每次需上传一至六张参考图")
    try:
        alias_values = json.loads(aliases) if aliases else []
        if not hull_number.strip():
            recognized = service.recognize_ship(images[0][1], images[0][0])
            hull_number = recognized["hull_number"]
            description = description or recognized["description"]
        item = service.create_registry(hull_number, description, alias_values, images)
    except FileExistsError as error:
        raise HTTPException(409, str(error)) from error
    except Exception as error:
        logger.exception("先验库上传失败")
        raise HTTPException(500, str(error)) from error
    return ApiResponse(success=True, message="先验库项和参考图已写入", data=item)

# 3.3 批量新增
@router.post("/bulk", response_model=ApiResponse)
async def bulk_create(body: ShipBulkCreate, service: Annotated[ShipService, Depends(get_service)]):
    result = service.bulk_create(body.ships)
    return ApiResponse(success=True, message=f"成功添加 {result['added']} 条，跳过 {result['skipped']} 条", data=result)

# ------------------------------------------------------------------ #
# 四、识别                                                               #
# ------------------------------------------------------------------ #
# 4.1 识别船舶
@router.post("/recognize", response_model=ApiResponse)
async def recognize_ship(file: UploadFile = File(...), service: ShipService = Depends(get_service)):
    images = await _read_images([file])
    try:
        result = service.recognize_ship(images[0][1], images[0][0])
    except Exception as error:
        raise HTTPException(500, f"识别失败：{error}") from error
    return ApiResponse(success=True, message="识别成功", data=result)

# 4.2 识别并入库
@router.post("/recognize-and-add", response_model=ApiResponse)
async def recognize_and_add(file: UploadFile = File(...), service: ShipService = Depends(get_service)):
    images = await _read_images([file])
    try:
        result = service.recognize_and_add(images[0][1], images[0][0])
    except Exception as error:
        raise HTTPException(500, f"识别入库失败：{error}") from error
    if "error" in result:
        raise HTTPException(400, result["error"])
    return ApiResponse(success=True, message="识别图片已写入先验库", data=result)

# 2.4 单条船舶
@router.get("/{hull_number}", response_model=ShipItem)
async def get_ship(hull_number: str, service: Annotated[ShipService, Depends(get_service)]):
    item = service.get_ship(hull_number)
    if item is None:
        raise HTTPException(404, f"未找到舷号：{hull_number}")
    return ShipItem(**item)

# 3.4 更新船舶
@router.put("/{hull_number}", response_model=ApiResponse)
async def update_ship(hull_number: str, body: ShipUpdate, service: Annotated[ShipService, Depends(get_service)]):
    try:
        item = service.update_registry(hull_number, body.description, body.aliases)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    return ApiResponse(success=True, message=f"成功更新舷号：{hull_number}", data=item)

# 3.5 添加参考图
@router.post("/{hull_number}/images", response_model=ApiResponse)
async def add_images(hull_number: str, files: list[UploadFile] = File(...), service: ShipService = Depends(get_service)):
    images = await _read_images(files)
    try:
        item = service.update_registry(hull_number, images=images)
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return ApiResponse(success=True, message="参考图已添加并重建向量库", data=item)

# 3.6 删除参考图
@router.delete("/{hull_number}/images/{reference_id}", response_model=ApiResponse)
async def delete_image(hull_number: str, reference_id: str, service: Annotated[ShipService, Depends(get_service)]):
    if not service.delete_reference(hull_number, reference_id):
        raise HTTPException(404, "未找到参考图")
    return ApiResponse(success=True, message="参考图已删除并重建向量库")

# 3.7 删除船舶
@router.delete("/{hull_number}", response_model=ApiResponse)
async def delete_ship(hull_number: str, service: Annotated[ShipService, Depends(get_service)]):
    if not service.delete_registry(hull_number):
        raise HTTPException(404, f"未找到舷号：{hull_number}")
    return ApiResponse(success=True, message=f"成功删除舷号：{hull_number}")

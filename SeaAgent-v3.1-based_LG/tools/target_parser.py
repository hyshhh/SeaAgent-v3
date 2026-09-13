"""解析问答中以分隔符列出的多个独立船舶目标。

切分词、正则与舷号形态均来自 skills/intent_agent/target_parsing.yaml，
本模块只加载规则并执行，不在代码中维护业务词表。
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from agent.skill_loader import load_skill_yaml


@lru_cache(maxsize=1)
def _rules() -> dict[str, Any]:
    data = load_skill_yaml("intent_agent", "target_parsing")
    return data if isinstance(data, dict) else {}


def clear_target_parser_cache() -> None:
    """测试或热更新规则后清空缓存。"""
    _rules.cache_clear()
    _compiled.cache_clear()


@lru_cache(maxsize=1)
def _compiled() -> dict[str, Any]:
    rules = _rules()
    flags = re.IGNORECASE
    # 默认允许中文+字母数字舷号（如 小蓝320）
    default_hull = r"[0-9A-Za-z一-鿿-]{2,16}$"
    return {
        "query_words": tuple(str(w) for w in (rules.get("query_words") or [])),
        "split": re.compile(str(rules.get("split_pattern") or r"[，,、；;]+")),
        "leading": re.compile(str(rules.get("leading_pattern") or r"^"), flags),
        "trailing": re.compile(str(rules.get("trailing_pattern") or r"$")),
        "hull": re.compile(str(rules.get("hull_pattern") or default_hull)),
        "ordinal": re.compile(str(rules.get("ordinal_prefix_pattern") or r"^")),
        "hull_label": re.compile(str(rules.get("hull_label_prefix_pattern") or r"^"), flags),
        "trim_chars": str(rules.get("trim_chars") or " \t\r\n：:，,、；;。！？?!"),
    }


def extract_target_items(question: str) -> list[dict[str, str | None]]:
    """从用户问题中提取逗号、顿号或并列词分隔的独立目标。"""
    cfg = _compiled()
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text or not any(word in text for word in cfg["query_words"]):
        return []
    if not cfg["split"].search(text):
        return []
    text = cfg["leading"].sub("", text)
    text = cfg["trailing"].sub("", text).strip()
    parts = [part.strip(cfg["trim_chars"]) for part in cfg["split"].split(text)]
    return _deduplicate(_build_item(part, index + 1) for index, part in enumerate(parts))


def normalize_target_items(value: Any) -> list[dict[str, str | None]]:
    """校验模型给出的多目标数组，禁止模型把多个目标合并为一个字符串。"""
    if not isinstance(value, list):
        return []
    raw_items = []
    cfg = _compiled()
    for index, item in enumerate(value):
        if isinstance(item, str):
            raw_items.append(_build_item(item, index + 1))
            continue
        if not isinstance(item, dict):
            continue
        label = str(
            item.get("label")
            or item.get("targetText")
            or item.get("description")
            or item.get("hullNumber")
            or ""
        ).strip()
        built = _build_item(label, index + 1)
        kind = str(item.get("targetKind") or item.get("kind") or built.get("kind") or "").strip().lower()
        candidate = str(item.get("hullNumber") or label).replace(" ", "")
        if kind == "hull" and cfg["hull"].fullmatch(candidate):
            hull = candidate.upper()
            built = {
                "targetId": f"target-{index + 1}",
                "label": hull,
                "kind": "hull",
                "hullNumber": hull,
                "description": None,
            }
        raw_items.append(built)
    return _deduplicate(raw_items)


def extract_hull_number(question: str) -> str | None:
    """从问题中抽取可能的舷号（规则辅助，供 Intent/Plan 工具调用）。

    支持「舷号 小蓝320」「舷号：A01」等中文+字母数字形态，与 memory.normalize_hull_number 对齐。
    """
    text = str(question or "")
    # 显式「舷号/弦号」后：允许中文名+编号（如 小蓝320），直到空白/标点/问句尾巴
    explicit = re.search(
        r"[舷弦]号\s*[:：]?\s*"
        r"([0-9A-Za-z一-鿿][0-9A-Za-z一-鿿-]{0,15})"
        r"(?=\s|[，,、；;。！？?!的吗呢啊呀有没是在库中出现]|$)",
        text,
        re.I,
    )
    if explicit:
        hull = re.sub(r"[^0-9A-Za-z一-鿿-]", "", explicit.group(1)).upper()
        if hull and not re.fullmatch(r"\d{1,2}", hull):
            return hull
    if not any(token in text for token in ("船", "出现", "轨迹", "编号", "时间", "有没有", "是否", "库", "舷号", "弦号")):
        return None
    # 裸舷号：纯字母数字 3–8 位；或「中文+数字」如 小蓝320
    for value in re.findall(
        r"(?<![\d:：A-Za-z一-鿿])"
        r"((?:[一-鿿]{1,6})?[0-9A-Za-z]{2,8}|[0-9A-Za-z]{3,8})"
        r"(?![\d:：A-Za-z])",
        text,
    ):
        compact = re.sub(r"[^0-9A-Za-z一-鿿-]", "", value).upper()
        if not compact or re.fullmatch(r"\d{1,2}", compact):
            continue
        # 排除常见非舷号词
        if compact in {"YOLO", "HTTP", "JSON", "API"}:
            continue
        return compact
    return None


def parse_targets(question: str) -> dict[str, Any]:
    """Intent 工具协议：parseTargets。"""
    return {
        "ok": True,
        "targetItems": extract_target_items(question),
        "hint": "可选参考；请你确认后写入 result.targetItems",
    }


def extract_hull(question: str) -> dict[str, Any]:
    """Intent 工具协议：extractHull。"""
    return {"ok": True, "hullNumber": extract_hull_number(question)}


def extract_description(question: str) -> str | None:
    """从问题中抽取外观/类别描述（非舷号查询时的规则兜底）。"""
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    if not text:
        return None
    if extract_hull_number(text):
        # 明确舷号问句时不把整句当描述
        if re.search(r"[舷弦]号", text):
            return None
    # 「哪些/有哪些 + 在库船」是列表问法，不是外观描述
    if _is_registry_list_question(text):
        return None
    # 去掉常见问句壳子，保留外观短语
    cleaned = text
    for pattern in (
        r"^[请问]+",
        r"视频中?",
        r"监控(?:画面|视频|记录)?中?",
        r"轨迹记忆中?",
        r"画面中?",
        r"(?:先验)?数据库(?:中|里|内)?(?:的)?",
        r"数据仓库(?:中|里|内)?(?:的)?",
        r"库中(?:的)?",
        r"库里(?:的)?",
        r"有没有",
        r"是否有?",
        r"有无",
        r"能不能看到",
        r"能不能找到",
        r"出现过?",
        r"存在",
        r"出现",
        r"看到",
        r"找到",
        r"查询",
        r"检索",
        r"统计",
        r"多少艘?",
        r"几艘",
        r"有哪些",
        r"哪些",
        r"哪几[艘条个]?",
        r"在库(?:船舶|船只|船)?",
        r"库内(?:船舶|船只|船)?",
        r"先验库(?:中|里)?(?:的)?(?:船舶|船只|船)?",
        r"一个",
        r"一条",
        r"一艘",
        r"这[个艘条]?",
        r"那[个艘条]?",
        r"[？?。！!]+$",
    ):
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n：:，,、；;的了吗呢啊呀")
    # 去掉残留「有/是」等单字
    cleaned = re.sub(r"^(?:有|是|为)\s*", "", cleaned).strip()
    if len(cleaned) < 2:
        return None
    # 纯数字/短舷号形态不当描述
    if re.fullmatch(r"[0-9A-Za-z-]{2,12}", cleaned):
        return None
    # 问句残留词不当描述
    if re.fullmatch(r"(?:船|船舶|船只|目标|对象|记录|结果|列表)+", cleaned):
        return None
    if any(token in cleaned for token in ("哪些", "有哪些", "在库", "未在库", "先验库")):
        return None
    return cleaned


def _is_registry_list_question(text: str) -> bool:
    """「有哪些在库船出现 / 哪些库船在视频里」类列表问法。"""
    t = str(text or "")
    has_list_word = any(token in t for token in ("哪些", "有哪些", "哪几", "列出", "名单", "列表"))
    has_registry = any(token in t for token in ("在库", "库内", "库船", "先验库", "名录"))
    has_video = any(token in t for token in ("出现", "视频", "监控", "轨迹", "画面", "看到", "找到"))
    # 仅问库列表（无视频）也算 registry list
    if has_list_word and has_registry:
        return True
    if has_registry and has_video and not re.search(r"[舷弦]号", t) and not extract_hull_number(t):
        # 「在库船出现了吗」更像 existence of relation，仍走 in-list 管线更合理
        if any(token in t for token in ("哪些", "有哪些", "哪几", "列出")):
            return True
    return False


def infer_intent_fields(question: str) -> dict[str, Any]:
    """规则推断意图字段，供 Intent 未提交或字段残缺时补全。"""
    text = str(question or "").strip()
    hull = extract_hull_number(text)
    description = None if hull and re.search(r"[舷弦]号", text) else extract_description(text)
    if hull and description and description.upper() == hull:
        description = None

    existence_tokens = ("有没有", "是否", "有无", "出现", "存在", "看到", "找到")
    count_tokens = ("多少", "几艘", "数量", "几个", "计数")
    registry_list_q = _is_registry_list_question(text)
    explicit_registry_scope = any(
        token in text
        for token in ("数据库", "数据仓库", "先验库", "库中", "库里", "库内", "名录")
    )
    explicit_video_scope = any(
        token in text
        for token in ("视频", "监控", "轨迹", "画面", "视野", "镜头", "录像")
    )

    if any(token in text for token in count_tokens):
        operation = "count"
    elif registry_list_q:
        # 「有哪些在库船出现」是列表，不是对问句本身做 existence/matchText
        operation = "list"
    elif any(token in text for token in existence_tokens):
        operation = "existence"
    else:
        operation = "list"

    if "未在库" in text or "不在库" in text or "库外" in text:
        registry_relation = "out"
    elif explicit_registry_scope or "在库" in text or "库船" in text:
        registry_relation = "in"
    else:
        registry_relation = "any"

    if hull:
        target_kind = "hull"
    elif description and not registry_list_q:
        target_kind = "description"
    else:
        target_kind = "all"
        if registry_list_q:
            description = None

    if explicit_registry_scope and not explicit_video_scope and not registry_list_q:
        # “数据库中有没有……”只查询先验数据库，不能被“出现/找到”等谓词扩展为视频检索。
        target_scope = "registry"
    elif explicit_registry_scope or "在库" in text or "未在库" in text or "库船" in text or registry_list_q:
        target_scope = "both"
    else:
        target_scope = "track_memory"

    items = extract_target_items(text)
    if not items and (hull or description):
        if hull:
            items = [{
                "targetId": "target-1",
                "label": hull,
                "kind": "hull",
                "hullNumber": hull,
                "description": None,
            }]
        else:
            items = [{
                "targetId": "target-1",
                "label": description,
                "kind": "description",
                "hullNumber": None,
                "description": description,
            }]

    if target_scope == "registry" and target_kind == "hull" and hull:
        expected = f"确认先验数据库中是否存在舷号 {hull} 的船舶记录"
        criteria = f"完成 getRegistry(hullNumber={hull}) 精确查询，并仅依据数据库返回结果作答；禁止扩展为视频轨迹检索"
        focus = f"getRegistry(hullNumber={hull})"
        question_type = f"registry_hull_{operation}"
        confidence = 0.94
    elif target_scope == "registry" and target_kind == "description" and description:
        expected = f"确认先验数据库中是否存在与「{description}」相符的船舶记录"
        criteria = (
            "完成 listRegistry 获取数据库库项及参考图，再执行 "
            f"matchText(description={description}, galleryImages=registry.registryReferences)；"
            "只以数据库匹配结果作答，禁止调用 getTrack/getFrames 或扩展为视频是否出现"
        )
        focus = f"listRegistry → matchText(description={description}, galleryImages=$ref registry.registryReferences)"
        question_type = f"registry_description_{operation}"
        confidence = 0.94
    elif target_scope == "registry" and target_kind == "all":
        expected = "返回先验数据库中的船舶记录" if operation != "count" else "统计先验数据库中的船舶记录数量"
        criteria = "完成 listRegistry，并仅依据完整数据库结果作答；禁止调用视频轨迹工具"
        focus = "listRegistry"
        question_type = f"registry_all_{operation}"
        confidence = 0.9
    elif registry_list_q and registry_relation == "out" and not hull:
        expected = "列出视频中出现但未命中任何先验库项的船舶轨迹"
        criteria = (
            "先完成 getTrack(全量) 枚举视频目标；若全量轨迹为 0，直接验收为没有未在库船出现，无需查整库；"
            "轨迹非空时再完成 getFrames 与 listRegistry，库有参考图时必须执行 matchImage(全部库图↔全部轨迹关键帧)；"
            "仅将最佳库匹配为 mismatch 的轨迹列为未在库，uncertain 必须单列为待确认，禁止只凭轨迹存在下结论"
        )
        focus = (
            "第一轮执行 getTrack(全量，不带hullNumber)，仅在轨迹非空时 getFrames；"
            "全量轨迹为 0 时结束，否则交由 ReflectAgent 决定下一轮并复用已有轨迹/关键帧补全库匹配"
        )
        question_type = "registry_out_list"
        confidence = 0.88
    elif registry_list_q and registry_relation == "in" and not hull:
        expected = "列出视频中出现且属于先验库的船舶（舷号/库项）"
        criteria = (
            "完成 listRegistry 取在库名录；getTrack 取视频轨迹；"
            "优先 matchHull(轨迹舷号↔库)；库有参考图时 getFrames→matchImage(库图↔关键帧)；"
            "汇总在库且出现的船舶列表。禁止把用户整句当 matchText 描述"
        )
        focus = (
            "①listRegistry；②getTrack(全量，不带hull)；③getFrames；"
            "④matchImage(query=registry.registryReferences, gallery=frames.keyframes)；"
            "若轨迹已有 OCR 舷号可并行 matchHull"
        )
        question_type = "registry_in_list"
        confidence = 0.85
    elif target_kind == "hull" and hull:
        if operation == "existence":
            # 存在判断≠「0 轨迹即未出现」：OCR 未命中后仍应库对照 + 视觉匹配
            expected = (
                f"综合视频轨迹 OCR 与先验库参考图，确认舷号 {hull} 是否在视频中出现"
            )
            criteria = (
                f"完成 getTrack(hullNumber={hull})；"
                "若 0 轨迹须 getRegistry 查先验库；"
                "库有可搜参考图时须 getTrack(不带hull)→getFrames→matchImage(库图↔关键帧)；"
                "再给出出现/未出现结论，禁止仅凭单次 0 轨迹否定"
            )
            focus = (
                f"①getTrack(hullNumber={hull})；"
                f"②0 轨迹→getRegistry(hullNumber={hull})；"
                "③库有参考图→getTrack(不带hullNumber)→getFrames→matchImage"
            )
        else:
            expected = f"返回舷号 {hull} 相关轨迹/库证据"
            criteria = "轨迹或库项+关键证据足以回答"
            focus = f"getTrack(hullNumber={hull})，必要时 getRegistry/getFrames/matchImage/showEvidence"
        question_type = f"{target_kind}_{operation}"
        confidence = 0.72
    elif target_kind == "description" and description:
        expected = f"确认是否存在「{description}」" if operation == "existence" else f"返回与「{description}」匹配的轨迹"
        criteria = "完成轨迹检索与描述匹配（matchText），再下结论"
        focus = f"getTrack → getFrames → matchText(description={description})"
        question_type = f"{target_kind}_{operation}"
        confidence = 0.72
    else:
        expected = "返回相关轨迹"
        criteria = "工具结果足以回答用户问题"
        focus = "先 getTrack 筛选轨迹，再按需匹配"
        question_type = f"{target_kind}_{operation}"
        confidence = 0.35

    return {
        "targetScope": target_scope,
        "targetKind": target_kind,
        "operation": operation,
        "registryRelation": registry_relation,
        "hullNumber": hull,
        "description": description,
        "targetItems": items,
        "expectedOutcome": expected,
        "successCriteria": criteria,
        "nextAgentFocus": focus,
        "questionType": question_type,
        "intentConfidence": confidence if (hull or description or registry_list_q) else 0.35,
    }


def _build_item(value: str, index: int) -> dict[str, str | None]:
    cfg = _compiled()
    label = cfg["leading"].sub("", str(value or "")).strip(cfg["trim_chars"])
    label = cfg["ordinal"].sub("", label).strip()
    explicit = cfg["hull_label"].sub("", label).strip()
    compact = explicit.replace(" ", "")
    if cfg["hull"].fullmatch(compact):
        hull = compact.upper()
        return {
            "targetId": f"target-{index}",
            "label": hull,
            "kind": "hull",
            "hullNumber": hull,
            "description": None,
        }
    return {
        "targetId": f"target-{index}",
        "label": label,
        "kind": "description",
        "hullNumber": None,
        "description": label or None,
    }


def _deduplicate(items: Any) -> list[dict[str, str | None]]:
    result: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        label = str(item.get("label") or "").strip()
        kind = str(item.get("kind") or "description")
        if not label:
            continue
        key = kind, label.upper() if kind == "hull" else label
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "targetId": f"target-{len(result) + 1}",
            "label": label,
            "kind": kind,
            "hullNumber": item.get("hullNumber") if kind == "hull" else None,
            "description": item.get("description") if kind == "description" else None,
        })
    return result

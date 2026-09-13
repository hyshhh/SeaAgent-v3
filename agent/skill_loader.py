"""从 skills/ 加载 Markdown 叙述规则与 YAML 结构化约束。

叙述性 skill 采用「目录 + 按需选用」：
- catalog.yaml 登记每个 skill 的 id / 简介 / always / 触发条件
- always=true 的 skill 始终注入 system prompt
- 其余按任务上下文关键词/字段匹配，或由模型通过 loadSkill 补载
- YAML 仍供代码侧读取（如 target/time 解析），不整包注入对话

定位：四个 agent 的 system prompt 装配线。本文件只决定「放哪些规则、以什么形式
放」，不决定「规则内容」—— 规则正文都在 skills/ 目录里。

披露层级（两级）：
  第一级  未选中的技能只给「id：标题 — 简介（适用场景）」目录，不占正文 token
  第二级  选中的技能注入全文；模型可随时用 loadSkill 拉取目录里的其他技能

注意：选择是**代码做的字面匹配**（token in text），不是模型做的语义判断。
      因此某个技能的 match_any 里若含「船」这类高频词，会实际退化为 always。

文件分区（以 K 编号为锚点检索）：
    K1  路径与元数据      skill_dir / SkillMeta
    K2  目录与正文读取    带 lru_cache 的加载入口
    K3  技能选择          select_skill_ids：always + 上下文 + 显式 三通道
    K4  提示词拼装        compose_skills 正文 + 可选目录
    K5  匹配辅助          上下文化文本、关键词匹配、点路径取值
    K6  YAML 通道与缓存   代码侧结构化约束 + 缓存失效
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"


# ============================================================================
# K1 路径与元数据
# skills/{agent_key}/ 的目录定位，以及 catalog.yaml 条目的内存模型。
# ============================================================================
def skill_dir(agent_key: str) -> Path:
    return _SKILLS_ROOT / agent_key


@dataclass(frozen=True)
class SkillMeta:
    id: str
    title: str
    description: str
    always: bool = False
    file: str = ""
    when: str = ""
    match_any: tuple[str, ...] = ()
    match_all: tuple[str, ...] = ()
    match_fields: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_catalog_item(self) -> dict[str, Any]:
        item = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "always": self.always,
        }
        if self.when:
            item["when"] = self.when
        if self.tags:
            item["tags"] = list(self.tags)
        return item


# ============================================================================
# K2 目录与正文读取
# 全部带 lru_cache：技能文件在进程生命周期内视为不可变，改动技能后需重启
# 进程才会重新读取。
# ============================================================================
@lru_cache(maxsize=64)
def load_skill_file(agent_key: str, filename: str) -> str:
    path = skill_dir(agent_key) / filename
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


@lru_cache(maxsize=32)
def list_skill_catalog(agent_key: str) -> tuple[SkillMeta, ...]:
    """读取 skills/{agent}/catalog.yaml；若无 catalog 则把全部 .md 当作 always 兼容项。

    若 catalog.yaml 存在但解析失败或 skills 为空，返回空目录（不回退整包 always）。
    """
    directory = skill_dir(agent_key)
    catalog_path = directory / "catalog.yaml"
    if catalog_path.is_file():
        try:
            data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return ()
        items = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return ()
        result: list[SkillMeta] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            skill_id = str(raw.get("id") or "").strip()
            if not skill_id:
                continue
            body_file = str(raw.get("file") or f"{skill_id}.md")
            result.append(
                SkillMeta(
                    id=skill_id,
                    title=str(raw.get("title") or skill_id),
                    description=str(raw.get("description") or ""),
                    always=bool(raw.get("always")),
                    file=body_file,
                    when=str(raw.get("when") or ""),
                    match_any=tuple(str(x) for x in (raw.get("match_any") or [])),
                    match_all=tuple(str(x) for x in (raw.get("match_all") or [])),
                    match_fields=tuple(str(x) for x in (raw.get("match_fields") or [])),
                    tags=tuple(str(x) for x in (raw.get("tags") or [])),
                )
            )
        return tuple(result)

    if not directory.is_dir():
        return ()
    fallback: list[SkillMeta] = []
    for path in sorted(directory.glob("*.md")):
        fallback.append(
            SkillMeta(
                id=path.stem,
                title=path.stem,
                description=f"规则文件 {path.name}",
                always=True,
                file=path.name,
            )
        )
    return tuple(fallback)


@lru_cache(maxsize=64)
def load_skill_body(agent_key: str, skill_id: str) -> str:
    meta = get_skill_meta(agent_key, skill_id)
    if meta is None:
        return ""
    return load_skill_file(agent_key, meta.file)


def get_skill_meta(agent_key: str, skill_id: str) -> SkillMeta | None:
    for item in list_skill_catalog(agent_key):
        if item.id == skill_id:
            return item
    return None


# ============================================================================
# K3 技能选择
# 三个通道按优先级依次叠加，去重保序：always → 显式 extra_ids → 上下文匹配
# （上限 max_optional）。若一个都没选中，强制取 catalog 首项兜底，避免空 prompt。
# ============================================================================
def select_skill_ids(
    agent_key: str,
    context: dict[str, Any] | None = None,
    *,
    extra_ids: list[str] | None = None,
    max_optional: int = 6,
) -> list[str]:
    """按 always + 上下文匹配 + 显式指定 选出 skill id（去重保序）。"""
    catalog = list_skill_catalog(agent_key)
    if not catalog:
        return []
    ctx = context if isinstance(context, dict) else {}
    text = _context_text(ctx)
    selected: list[str] = []
    seen: set[str] = set()

    def _add(skill_id: str) -> None:
        if skill_id and skill_id not in seen and get_skill_meta(agent_key, skill_id):
            seen.add(skill_id)
            selected.append(skill_id)

    for meta in catalog:
        if meta.always:
            _add(meta.id)

    for skill_id in extra_ids or []:
        _add(str(skill_id).strip())

    optional_hits: list[str] = []
    for meta in catalog:
        if meta.always or meta.id in seen:
            continue
        if _skill_matches(meta, ctx, text):
            optional_hits.append(meta.id)

    for skill_id in optional_hits[: max(0, int(max_optional))]:
        _add(skill_id)

    if not selected and catalog:
        _add(catalog[0].id)
    return selected


# ============================================================================
# K4 提示词拼装
# compose_skills 产出最终注入 system prompt 的 Markdown：选中技能给全文，
# 未选中且非 always 的技能只给「目录」（含 when 适用场景），提示模型可按需
# loadSkill 补载 —— 这就是两级披露的落点。
# ============================================================================
def compose_skills(
    agent_key: str,
    skill_ids: list[str] | None = None,
    *,
    include_catalog_index: bool = True,
) -> str:
    """拼装注入正文：选中 skill 全文 + 可选技能目录（仅简介，便于 loadSkill）。"""
    catalog = list_skill_catalog(agent_key)
    if not catalog:
        return ""
    ids = skill_ids if skill_ids is not None else [m.id for m in catalog if m.always]
    chunks: list[str] = []
    loaded: set[str] = set()
    for skill_id in ids:
        meta = get_skill_meta(agent_key, skill_id)
        body = load_skill_body(agent_key, skill_id)
        if not meta or not body:
            continue
        loaded.add(skill_id)
        flag = "【始终启用】" if meta.always else "【本轮选用】"
        chunks.append(f"### {flag} {meta.title} (`{meta.id}`)\n{body}")

    if include_catalog_index:
        remaining = [m for m in catalog if m.id not in loaded and not m.always]
        if remaining:
            lines = ["### 可选技能目录（未注入正文；需要时 toolCalls loadSkill）"]
            for meta in remaining:
                when = f"；适用：{meta.when}" if meta.when else ""
                lines.append(f"- `{meta.id}`：{meta.title} — {meta.description}{when}")
            chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def catalog_index(agent_key: str, exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
    excluded = set(exclude_ids or [])
    return [
        meta.to_catalog_item()
        for meta in list_skill_catalog(agent_key)
        if meta.id not in excluded
    ]


# ============================================================================
# K5 匹配辅助
# _context_text 把上下文摊平成一大段文本（含 question 全文与 intent 全部字段），
# _skill_matches 在其上做**字面包含**判断 —— 没有分词、没有语义相关性。
# 这是 K3 选择质量的上限所在：高频词会让技能被普遍命中。
# ============================================================================
def _context_text(ctx: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "question",
        "evidenceGap",
        "nextAction",
        "nextAgentFocus",
        "expectedOutcome",
        "successCriteria",
        "goal",
        "reason",
        "planValidationError",
        "repairHint",
        "plan_hint",
        "observation_summary",
    ):
        value = ctx.get(key)
        if value:
            parts.append(str(value))
    intent = ctx.get("intent")
    if isinstance(intent, dict):
        for key, value in intent.items():
            if value not in (None, "", [], {}):
                parts.append(f"{key}:{value}")
    for key, value in ctx.items():
        if isinstance(value, str) and value and key not in {
            "question", "evidenceGap", "nextAction", "nextAgentFocus",
            "expectedOutcome", "successCriteria", "goal", "reason",
            "planValidationError", "repairHint", "plan_hint", "observation_summary",
        }:
            parts.append(value)
    return "\n".join(parts)


def _skill_matches(meta: SkillMeta, ctx: dict[str, Any], text: str) -> bool:
    if meta.match_fields:
        for field_name in meta.match_fields:
            value = _dig(ctx, field_name)
            if value not in (None, "", [], {}, False):
                return True
    if meta.match_all:
        if all(token in text for token in meta.match_all):
            return True
        if not meta.match_any and not meta.match_fields:
            return False
    if meta.match_any:
        return any(token in text for token in meta.match_any)
    return False


def _dig(ctx: dict[str, Any], dotted: str) -> Any:
    current: Any = ctx
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


# ============================================================================
# K6 YAML 通道
# .yaml 与 .md 走两条路：YAML 只给代码读（如 tools/target_parser.py），
# 不注入对话。
# ============================================================================
@lru_cache(maxsize=32)
def load_skill_yaml(agent_key: str, filename: str) -> dict[str, Any]:
    """加载 skills/{agent_key}/{filename}.yaml 或 .yml。"""
    directory = skill_dir(agent_key)
    for suffix in (".yaml", ".yml"):
        path = directory / filename if filename.endswith((".yaml", ".yml")) else directory / f"{filename}{suffix}"
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}
    return {}

"""从 skills/ 加载 Markdown 叙述规则与 YAML 结构化约束。

叙述性 skill 采用「描述 + 按需取全文」的两级披露：
- catalog.yaml 登记每个 skill 的 id / 标题 / 简介 / 正文文件
- 本模块只负责**读取**目录与正文；技能的**挂载**在 lc_tools.build_skill_tools
  —— 每个 skill 包装成 load_<id> 工具，description 就是简介，模型调用才拿到正文。
- YAML 仍供代码侧读取（如 target/time 解析），不整包注入对话

定位：技能文件的读取层。它不决定「什么时候用哪个技能」—— 代码不做关键词预选，
由模型看着工具描述自行决定。

文件分区（以 K 编号为锚点检索）：
    K1  路径与元数据      skill_dir / SkillMeta
    K2  目录与正文读取    带 lru_cache 的加载入口
    K3  YAML 通道         代码侧结构化约束
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
    """catalog.yaml 的一条技能登记。"""

    id: str
    title: str
    description: str
    file: str = ""


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
    """读取 skills/{agent}/catalog.yaml；若无 catalog 则把全部 .md 当作技能。

    若 catalog.yaml 存在但解析失败或 skills 为空，返回空目录（不回退整包）。
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
            result.append(
                SkillMeta(
                    id=skill_id,
                    title=str(raw.get("title") or skill_id),
                    description=str(raw.get("description") or ""),
                    file=str(raw.get("file") or f"{skill_id}.md"),
                )
            )
        return tuple(result)

    if not directory.is_dir():
        return ()
    return tuple(
        SkillMeta(
            id=path.stem,
            title=path.stem,
            description=f"规则文件 {path.name}",
            file=path.name,
        )
        for path in sorted(directory.glob("*.md"))
    )


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
# K3 YAML 通道
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

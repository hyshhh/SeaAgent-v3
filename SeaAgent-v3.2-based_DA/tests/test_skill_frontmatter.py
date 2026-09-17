"""把技能的 frontmatter 问题钉进测试：YAML 必须能解析、name 必须合规范、目录名与 name 一致。

背景：框架对坏 frontmatter 只发一条 warning 就跳过该技能 —— 技能等于没挂，
而日志在一堆 INFO 里很容易漏看。这条测试把它变成硬失败。
"""
import pathlib

import pytest
import yaml

root = pathlib.Path(__file__).resolve().parents[1] / "skills"
SKILLS = sorted(root.glob("*/*/SKILL.md"))


def _frontmatter(path):
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) >= 3, f"{path} 缺少 frontmatter"
    return yaml.safe_load(parts[1]) or {}


def test_every_skill_frontmatter_parses():
    """冒号没加引号会让整个 frontmatter 解析失败，技能被静默跳过。"""
    broken = []
    for path in SKILLS:
        try:
            _frontmatter(path)
        except yaml.YAMLError as error:
            broken.append(f"{path.relative_to(root)}: {str(error).splitlines()[0]}")
    assert not broken, "frontmatter 解析失败：\n" + "\n".join(broken)


def test_skill_names_follow_the_agent_skills_spec():
    """Agent Skills 规范：小写字母数字，单词之间单个连字符。"""
    import re

    bad = []
    for path in SKILLS:
        name = str(_frontmatter(path).get("name", ""))
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
            bad.append(f"{path.relative_to(root)}: name={name!r}")
    assert not bad, "技能名不合规范（需小写+连字符）：\n" + "\n".join(bad)


def test_skill_name_matches_its_folder():
    """name 与目录名不一致时，框架按目录找、按 name 登记，两边会对不上。"""
    mismatched = []
    for path in SKILLS:
        folder = path.parent.name
        name = str(_frontmatter(path).get("name", ""))
        if name != folder:
            mismatched.append(f"{path.relative_to(root)}: name={name!r} != 目录 {folder!r}")
    assert not mismatched, "\n".join(mismatched)


def test_every_skill_has_a_usable_description():
    """description 是模型决定「要不要读这个技能」的唯一依据，必须非空。"""
    missing = [str(path.relative_to(root)) for path in SKILLS if not str(_frontmatter(path).get("description", "")).strip()]
    assert not missing, "缺 description：\n" + "\n".join(missing)

"""技能布局：技能按组存放，组目录是「容器」，其子目录才是技能。

    skills/<组>/<技能名>/SKILL.md      ← 官方 skills 源就是这个层级

路径层级必须一起断言：glob 模式一旦和实际布局脱节（例如重组后仍写 skills/*/SKILL.md），
循环就一次都不执行，测试会静默变成空跑。
"""
import re
from pathlib import Path

EXPECTED_GROUPS = {"coordination", "track", "registry", "visual", "answer"}


def _skill_files():
    return sorted(Path("skills").glob("*/*/SKILL.md"))


def test_skill_directories_exist():
    found = _skill_files()
    assert found, "没有找到任何 skills/<组>/<技能>/SKILL.md，布局与 glob 不一致"
    groups = {path.parent.parent.name for path in found}
    assert groups == EXPECTED_GROUPS, f"技能组与预期不符：{sorted(groups)}"


def test_skills_have_valid_frontmatter():
    for path in _skill_files():
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^---\s*$", text, re.MULTILINE)
        assert re.search(rf"^name: {re.escape(path.parent.name)}\s*$", text, re.MULTILINE)
        assert re.search(r"^description: .+", text, re.MULTILINE)

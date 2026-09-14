import re
from pathlib import Path


def test_skills_have_valid_frontmatter():
    for path in Path("skills").glob("*/SKILL.md"):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^---\s*$", text, re.MULTILINE)
        assert re.search(rf"^name: {re.escape(path.parent.name)}\s*$", text, re.MULTILINE)
        assert re.search(r"^description: .+", text, re.MULTILINE)

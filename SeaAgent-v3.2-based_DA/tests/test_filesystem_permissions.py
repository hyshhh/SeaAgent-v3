"""文件工具权限：原生 skills 渐进式披露要读得到正文，又不能让模型读到项目其他文件。"""
from deepagents import FilesystemPermission
from deepagents.middleware.filesystem import _check_fs_permission

from config import load_config
from harness.runtime import _filesystem_permissions


def _rules():
    return _filesystem_permissions(load_config()["harness"], FilesystemPermission)


def test_skill_files_are_readable():
    rules = _rules()
    assert _check_fs_permission(rules, "read", "/skills/query/SKILL.md") == "allow"
    assert _check_fs_permission(rules, "read", "/skills/finalize/SKILL.md") == "allow"
    assert _check_fs_permission(rules, "read", "/skills") == "allow"


def test_everything_outside_the_skills_directory_is_unreadable():
    rules = _rules()
    assert _check_fs_permission(rules, "read", "/config/app.yaml") == "deny"
    assert _check_fs_permission(rules, "read", "/harness/runtime.py") == "deny"
    assert _check_fs_permission(rules, "read", "/") == "deny"


def test_writes_are_denied_everywhere_including_inside_skills():
    rules = _rules()
    assert _check_fs_permission(rules, "write", "/skills/query/SKILL.md") == "deny"
    assert _check_fs_permission(rules, "write", "/data/memory/tracks.csv") == "deny"


def test_missing_whitelist_leaves_the_framework_default_untouched():
    assert _filesystem_permissions({}, FilesystemPermission) is None

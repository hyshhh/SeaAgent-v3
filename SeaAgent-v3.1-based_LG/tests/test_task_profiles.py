"""task_profiles 单一事实源（② 最小治理）的回归测试。"""
from __future__ import annotations

from agent.task_profiles import (
    is_membership_question_type,
    registry_membership_list_mode,
    relation_for_membership,
    resolve_evidence_mode,
)


def _intent(**overrides) -> dict:
    base = {
        "operation": "list",
        "targetScope": "both",
        "targetKind": "all",
        "registryRelation": "in",
        "hullNumber": None,
        "questionType": "registry_in_list",
    }
    base.update({key: value for key, value in overrides.items() if value is not None})
    for key, value in overrides.items():
        if value is None:
            base.pop(key, None)
    return base


def test_membership_mode_in_list():
    assert registry_membership_list_mode(_intent()) == "in"


def test_membership_mode_out_list():
    intent = _intent(registryRelation="out", questionType="registry_out_list")
    assert registry_membership_list_mode(intent) == "out"


def test_not_membership_when_hull_present():
    intent = _intent(hullNumber="0857")
    assert registry_membership_list_mode(intent) == ""


def test_not_membership_when_operation_is_existence():
    intent = _intent(operation="existence")
    assert registry_membership_list_mode(intent) == ""


def test_not_membership_when_relation_any():
    intent = _intent(registryRelation="any", questionType="")
    assert registry_membership_list_mode(intent) == ""


def test_membership_requires_both_scope_or_matching_question_type():
    # target_scope=track_memory 且 questionType 与 relation 不匹配 → 不是列表任务
    intent = _intent(targetScope="track_memory", questionType="hull_existence")
    assert registry_membership_list_mode(intent) == ""
    # questionType 匹配但 target_scope 为 registry → 仍判定（与旧行为一致：两者任一满足即可）
    intent = _intent(targetScope="registry", questionType="registry_in_list")
    assert registry_membership_list_mode(intent) == "in"


def test_non_dict_input_is_not_membership():
    assert registry_membership_list_mode(None) == ""
    assert registry_membership_list_mode([]) == ""


def test_question_type_helpers():
    assert is_membership_question_type("registry_in_list") is True
    assert is_membership_question_type("registry_out_list") is True
    assert is_membership_question_type("hull_existence") is False
    assert is_membership_question_type(None) is False
    assert relation_for_membership("registry_in_list") == "in"
    assert relation_for_membership("registry_out_list") == "out"
    assert relation_for_membership("hull_existence") == ""


def test_resolve_evidence_mode_focused_for_existence_with_target():
    assert resolve_evidence_mode({"operation": "existence", "hullNumber": "小蓝320"}) == "focused"
    assert resolve_evidence_mode({"operation": "existence", "description": "黄色无人艇"}) == "focused"
    assert resolve_evidence_mode({"operation": "explain", "hullNumber": "0857"}) == "focused"


def test_resolve_evidence_mode_broad_for_enumeration():
    assert resolve_evidence_mode({"operation": "count"}) == "broad"
    assert resolve_evidence_mode({"operation": "list"}) == "broad"
    assert resolve_evidence_mode({"operation": "time"}) == "broad"
    assert resolve_evidence_mode({"operation": "list", "questionType": "registry_in_list", "registryRelation": "in"}) == "broad"
    assert resolve_evidence_mode({"operation": "existence"}) == "broad"  # 无目标对象


def test_resolve_evidence_mode_membership_list_is_broad():
    intent = {"operation": "list", "targetScope": "both", "registryRelation": "out", "questionType": "registry_out_list"}
    assert resolve_evidence_mode(intent) == "broad"


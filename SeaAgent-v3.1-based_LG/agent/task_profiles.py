"""任务模式判定的单一事实源（② 最小治理）。

「在库/未在库船舶列表」这类问法（哪些船在库/未在库）的判定条件，此前在
`agent/graph.py` 与 `agent/controller.py` 各写一遍，改一处漏一处。
这里集中定义 membership 模式的判定与问法类型工具，graph / controller
共同调用，避免语义漂移。
"""
from __future__ import annotations

from typing import Any

_MEMBERSHIP_QUESTION_TYPES = frozenset({"registry_in_list", "registry_out_list"})


# ===========================================================================
# T1 membership 判定：识别「在库 / 未在库船舶列表」这类问法
# 消费方：graph.py（验收清单分派）与 controller.py（结论措辞）共用，
# 二者必须同源，否则会出现「验收按 membership 走、结论按别的走」的矛盾。
# ===========================================================================
def registry_membership_list_mode(intent: dict[str, Any] | None) -> str:
    """识别「在库/未在库船舶列表」任务，返回 in/out；其他任务返回空字符串。

    判定依据（与 graph 侧的验收分派同源）：
    - registryRelation 为 in/out；
    - operation 为 list；
    - 没有具体舷号（列表任务面向全体候选）；
    - targetScope 为 both，或 questionType 与 relation 匹配（registry_in_list/out）。
    """
    if not isinstance(intent, dict):
        return ""
    relation = str(intent.get("registryRelation") or "")
    operation = str(intent.get("operation") or "")
    hull = str(intent.get("hullNumber") or "").strip()
    target_scope = str(intent.get("targetScope") or "")
    question_type = str(intent.get("questionType") or "")
    expected_type = f"registry_{relation}_list" if relation in {"in", "out"} else ""
    if (
        relation in {"in", "out"}
        and operation == "list"
        and not hull
        and (target_scope == "both" or question_type == expected_type)
    ):
        return relation
    return ""


def is_membership_question_type(question_type: Any) -> bool:
    """问法类型是否为在库/未在库列表（registry_in_list / registry_out_list）。"""
    return str(question_type or "") in _MEMBERSHIP_QUESTION_TYPES


def relation_for_membership(question_type: Any) -> str:
    """从问法类型反推在库关系：registry_in_list→in，registry_out_list→out，否则空串。"""
    qt = str(question_type or "")
    if qt == "registry_in_list":
        return "in"
    if qt == "registry_out_list":
        return "out"
    return ""


# ===========================================================================
# T2 证据量级：focused（单目标，少量证据）vs broad（枚举/对照，全量证据）
# 注意：这不是配置项，而是从意图派生的值。它只影响检索体量与展示截断，
# 不进入验收清单 —— 即「要多少证据」与「够不够」是两套判定。
# ===========================================================================
# 枚举型问法：需要全量证据（列表 / 计数 / 时间定位 / 在库对照）
_BROAD_QUESTION_TYPES = frozenset({
    "registry_in_list", "registry_out_list", "track_list", "registry_list",
    "relation_description", "cross_reference", "count", "description_count",
    "registry_count", "registry_description_count", "out_of_registry", "in_registry",
})


def resolve_evidence_mode(intent: dict[str, Any] | None) -> str:
    """判断证据量级：focused（单目标判断，少量证据）vs broad（枚举/对照，全量证据）。

    规则为准（与意图时间落地一致，不信任模型自报）：
    - 在库/未在库列表、计数、列表、时间定位 → broad；
    - 存在性/解释（有具体目标）→ focused；
    - 无目标或无法判定 → 默认 broad（宁可多证据，不可漏检）。
    """
    if not isinstance(intent, dict):
        return "broad"
    operation = str(intent.get("operation") or "")
    question_type = str(intent.get("questionType") or "")
    hull = str(intent.get("hullNumber") or "").strip()
    description = str(intent.get("description") or "").strip()
    if registry_membership_list_mode(intent):
        return "broad"
    if question_type and question_type in _BROAD_QUESTION_TYPES:
        return "broad"
    if operation in {"count", "list", "time"}:
        return "broad"
    if operation in {"existence", "explain"} and (hull or description):
        return "focused"
    return "broad"

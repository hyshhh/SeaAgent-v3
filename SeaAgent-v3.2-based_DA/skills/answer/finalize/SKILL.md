---
name: finalize
description: "Display this turn's visual evidence before finalizing the answer."
---

# Finalize

1. Call `show_evidence` once when shared state contains track IDs. The UI shows each track's ID, appearance time and first keyframe. Registry match-statistics evidence is collected outside model context.
2. Reconcile direct recognition with every successful helper result. For a confirmed `match_registry_tracks` result, report the matched track IDs and scores even if direct recognition returned `mismatch`; state what each result establishes without conflating them.
3. Answer in Chinese unless requested otherwise. Lead with the supported conclusion, cite evidence IDs (never `displayId`), and state material gaps or uncertainty. Stop when the claim is established or its remaining gap is clear.
4. For total-only deduplication, state the actual tool values directly: `按照模糊匹配为 A 艘、精确匹配为 B 艘，预测总船只实体在 A–B 之间`. If any duplicate pair remains unresolved, add that qualification.
5. For registered or unregistered questions, make sure the answer clearly includes all three values: `总数：N`、`在库船：M`、`不在库船：K`. Use `N = trackCount`, `M = embeddingConfirmedCount + gray VLM match`, and `K = N - M`. Both question phrasings must produce the same three values; natural prose is acceptable.

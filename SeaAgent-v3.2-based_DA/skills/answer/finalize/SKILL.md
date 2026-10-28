---
name: finalize
description: "Display this turn's visual evidence before finalizing the answer."
---

# Finalize

1. Call `show_evidence` once when shared state contains track IDs. The UI shows each track's ID, appearance time and first keyframe. Registry match-statistics evidence is collected outside model context.
2. Reconcile direct recognition with every successful helper result. For a confirmed `match_registry_tracks` result, report the matched track IDs and scores even if direct recognition returned `mismatch`; state what each result establishes without conflating them.
3. Answer in Chinese unless requested otherwise. Lead with the supported conclusion, cite evidence IDs (never `displayId`), and state material gaps or uncertainty. Stop when the claim is established or its remaining gap is clear.

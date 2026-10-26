---
name: finalize
description: "Display this turn's visual evidence before finalizing the answer."
---

# Finalize

1. Call `show_evidence` once with current-turn track, keyframe and registry-reference IDs; omit missing ID types. The tool creates clips internally.
2. Answer in Chinese unless requested otherwise. Lead with the supported conclusion, cite evidence IDs (never `displayId`), and state material gaps or uncertainty. Stop when the claim is established or its remaining gap is clear.

---
name: finalize
description: "Use before the final answer to display this turn's visual evidence."
---

# Finalize

1. Call `show_evidence` once with returned track, keyframe and registry-reference IDs; if none, call without IDs.
2. Use only current-turn IDs. Report evidence IDs, not the panel's `displayId`.
3. Answer in Chinese: conclusion, evidence IDs, then limitations. Stop when the claim is established or the remaining gap is clear.

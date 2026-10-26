---
name: finalize
description: "Display this turn's visual evidence before finalizing the answer."
---

# Finalize

1. Call `show_evidence` once with current-turn track, keyframe and registry-reference IDs; omit missing ID types. The tool creates any clips internally.
2. Cite evidence IDs, never the panel's `displayId`.
3. Give the supported conclusion and material limits in Chinese unless requested otherwise. Stop when the claim is established or its remaining gap is clear.

---
name: finalize
description: "Display this turn's visual evidence before finalizing the answer."
---

# Finalize

1. Call `show_evidence` once when shared state contains track IDs. It returns track IDs and appearance times; the UI loads compact clips. Registry match-statistics evidence is collected outside model context.
2. Answer in Chinese unless requested otherwise. Lead with the supported conclusion, cite evidence IDs (never `displayId`), and state material gaps or uncertainty. Stop when the claim is established or its remaining gap is clear.

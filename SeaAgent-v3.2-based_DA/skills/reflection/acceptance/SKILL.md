---
name: acceptance
description: "Audit each requirement against successful tool evidence before deciding."
---

# Evidence Audit

- Mark every requirement met, unmet or unverifiable from successful tool results; a call alone is not evidence.
- Keep confirmed, possible and unconfirmed findings distinct. Incomplete coverage or unresolved items cannot prove absence.
- If a recognition result has `nextStep`, require a successful corresponding helper result before terminating. For uncertain candidates, use `verify_vessel_images.decision`; preserve `uncertain` if VLM cannot resolve it. Report only current-turn evidence IDs.
- Include successful helper findings in the final answer. For `match_registry_tracks`, report confirmed candidate track IDs and scores, and reconcile them with the direct search; a direct `mismatch` does not cancel an assisted `confirmed` match.

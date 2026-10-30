---
name: acceptance
description: "Audit each requirement against successful tool evidence before deciding."
---

# Evidence Audit

- Mark every requirement met, unmet or unverifiable from successful tool results; a call alone is not evidence.
- Keep confirmed, possible and unconfirmed findings distinct. Incomplete coverage or unresolved items cannot prove absence.
- If a recognition or `match_entities` result has `nextStep` or `recommendation`, require a successful corresponding helper result for every returned gray candidate before terminating. For uncertain candidates, use `verify_vessel_images.decision`; preserve `uncertain` or failure if VLM cannot resolve it. Report only current-turn evidence IDs.
- For count results, compare `appliedTimeRange` with the current plan before accepting them; `null` means all stored monitoring time. A mismatch leaves the requirement unmet and requires replanning. For `match_entities`, start with `embeddingConfirmedCount` and `outsideRegistryCount`, then add gray VLM `match` to the in-registry count and gray VLM `mismatch` to the outside count. Keep uncertain/failed/missing decisions unresolved.
- Include successful helper findings in the final answer. For `match_registry_tracks`, report confirmed candidate track IDs and scores, and reconcile them with the direct search; a direct `mismatch` does not cancel an assisted `confirmed` match.

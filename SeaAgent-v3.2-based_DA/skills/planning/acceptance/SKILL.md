---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- For recognition, plan `recognize_tracks`; for hull/alias searches, add a conditional `match_registry_tracks` step, executed only when `nextStep` names it. Count-only plans must contain only the requested count tools, never `recognize_tracks`.
- For uncertain image candidates, use `verify_vessel_images` with one reference and one candidate image.
- Map counts exactly: threshold-range totals use `dedup_tracks`; registry totals use `match_entities`, which also returns the low-threshold total. Add `dedup_tracks` only when the user also requests the threshold range. Never add recognition steps to a count-only request or invent tool names.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.

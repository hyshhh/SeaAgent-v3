---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- For recognition, plan `recognize_tracks`; for hull/alias searches, add a conditional `match_registry_tracks` step, executed only when `nextStep` names it. Count-only plans must contain only the requested count tools, never `recognize_tracks`.
- For uncertain image candidates, use `verify_vessel_images` with one reference and one candidate image.
- Map counts exactly: registered/unregistered intent takes priority even when the user asks “how many”; use `match_entities` to match every track without deduplication. Plain total-vessel or threshold-range counts use `dedup_tracks`; a total-only request must not include `match_entities`. Combine both only when the user explicitly asks for both total and registry classification. Never add recognition steps to a count-only request or invent tool names.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.

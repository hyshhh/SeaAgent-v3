---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- For the six presets, plan distinctly: hull number, description and image lookup use `recognize_tracks` with the matching input; total count uses `dedup_tracks`; registered and unregistered counts use `match_entities`. Handle other or mixed requests by their explicit requirements. For hull/alias searches, add a conditional `match_registry_tracks` step only when `nextStep` names it. Count-only plans must not include recognition steps.
- For uncertain image candidates, use `verify_vessel_images` with one reference and one candidate image.
- Use `match_entities` for registered or unregistered counts, matching every track without deduplication. Plain total-vessel or threshold-range counts use `dedup_tracks`; a total-only request must not include registry matching. Combine both only when the user explicitly asks for total and registry counts. Never invent tool names.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.

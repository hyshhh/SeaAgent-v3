---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- Plan `recognize_tracks` first; for hull/alias searches, add a conditional `match_registry_tracks` step, executed only when `nextStep` names it.
- For uncertain image candidates, use `verify_vessel_images` with one reference and one candidate image.
- Map counts exactly: total `count_total_entities`, registered `count_registry_entities`, unregistered `count_unregistered_entities`. Plan only requested measures; never invent tool names.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.

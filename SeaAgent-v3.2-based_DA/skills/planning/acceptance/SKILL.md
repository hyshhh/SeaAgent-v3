---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- Start recognition with `recognize_tracks`; add `match_registry_tracks` only after its result requests that next step.
- For uncertain image candidates, use `verify_vessel_images` with one reference and one candidate image.
- For counts, choose exactly the corresponding `count_*` tool; it handles scope, deduplication and registry matching.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.

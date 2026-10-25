---
name: acceptance
description: "Use to define the evidence required for each user requirement."
---

# Plan Acceptance

Put each verifiable claim in `requirements`; make `plan` steps name needed tools/evidence in `content` and `required_evidence` (string array). Do not add unsupported fields such as `how_to_check`.

- Recognition: `recognize_tracks`; for a supplied hull/name with `nextStep=match_registry_tracks` or an uncertain result, add `match_registry_tracks`.
- Uncertain image or registry candidate: compare its reference and candidate images with `verify_vessel_images`.
- Total, registered or unregistered count: use exactly the corresponding `count_*` tool; it handles scope, deduplication and matching internally.
- Visual evidence: use `show_evidence` with IDs returned by the business tools.

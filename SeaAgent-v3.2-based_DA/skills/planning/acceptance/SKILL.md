---
name: acceptance
description: "Use to define the evidence required for each user requirement."
---

# Plan Acceptance

Put each verifiable claim in `requirements`; make `plan` steps name needed tools/evidence in `content` and `required_evidence` (string array). Do not add unsupported fields such as `how_to_check`.

- Video presence/list: `get_track`; inspect frames only when visual identity matters.
- Appearance: tracks → keyframes → `match_text`.
- Hull or registry membership: query the requested source; use unfiltered tracks and image comparison when recognition/identity is uncertain.
- Count: complete candidate scope → keyframes → `dedup_tracks`.
- Registry-only: hull/catalogue lookup and registry-image text search as needed; no video query unless requested.

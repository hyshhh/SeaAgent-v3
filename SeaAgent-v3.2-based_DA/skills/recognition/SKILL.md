---
name: recognition
description: "Choose direct track search, registry-assisted matching, or a two-image vessel check."
---

# Recognition

- Search by hull number or target name/alias (for example, “小红”) with `recognize_tracks.hull_number`; use `description` for appearance and `image_path` for a user image. Omit `time_range` to search all stored monitoring time.
- Call `match_registry_tracks` once only when the direct result sets `nextStep=match_registry_tracks`; reuse the same value and range.
- `verify_vessel_images` compares exactly two images: one reference (`reference_image` or `reference_registry_reference_id`) and one candidate (`candidate_image` or `candidate_keyframe_id`). It does not search tracks.
- Read `status` and each match's `scoreBand` separately. Incomplete coverage or `unconfirmed` results cannot establish absence.
- Pass returned evidence IDs to `show_evidence` when visual display is needed. Never read raw track files or registry listings.
